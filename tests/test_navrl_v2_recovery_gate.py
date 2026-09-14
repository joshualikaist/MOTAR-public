import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import torch
from torch.utils.tensorboard import SummaryWriter


_ROOT = Path(__file__).resolve().parents[1]
_ATTEST_PATH = _ROOT / "tools/navrl_v2_recovery_attestation.py"
_SPEC = importlib.util.spec_from_file_location("navrl_v2_attestation_test", _ATTEST_PATH)
_ATTEST = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ATTEST)
_RUNNER = (
    _ROOT
    / "aerial_gym/rl_training/rl_games/train_navrl_v2_recover_safe.sh"
)
_EVALUATOR = (
    _ROOT
    / "aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh"
)
_PYTHON = "/home/fair/miniconda3/envs/aerialgym/bin/python"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NavRLV2RecoveryGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_root = Path(self.temp.name) / "run"
        self.nn = self.run_root / "nn"
        self.nn.mkdir(parents=True)
        (self.run_root / ".aerial_training_finished").write_text(
            "epoch=9600\n", encoding="utf-8"
        )

    def _state(self, stage="smoke", current_lr=5e-6):
        state = copy.deepcopy(_ATTEST.SMOKE_STATE_CONTRACT)
        state.update(
            {
                "n_bars_active": 130,
                "cfg_recovery_stage": stage,
                "cfg_recovery_source_sha256": _ATTEST.TRUSTED_SOURCE_SHA256,
                "cfg_recovery_source_epoch": 9500,
                "cfg_recovery_smoke_required_epochs": 100,
                "cfg_recovery_smoke_bars": 130,
                "current_action_learning_rate": current_lr,
            }
        )
        return state

    def _save_checkpoint(self, epoch=9600, state=None):
        path = self.nn / f"last_gen_ppo_ep_{epoch}_rew_1.0.pth"
        saved_state = copy.deepcopy(state or self._state())
        saved_state["num_task_steps"] = epoch * 32
        torch.save(
            {"epoch": epoch, "frame": epoch * 4096, "env_state": saved_state}, path
        )
        return path

    def _result_file(self):
        path = Path(self.temp.name) / "130bars.json"
        path.write_text('{"canonical":"heldout"}\n', encoding="utf-8")
        return path

    def _attestation(self, checkpoint, result_path):
        captured, crash, timeout = 1435, 600, 14
        episodes = captured + crash + timeout
        return {
            "schema_version": 1,
            "verdict": "PASS",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_epoch": 9600,
            "checkpoint_frame": 39321600,
            "source_checkpoint_sha256": _ATTEST.TRUSTED_SOURCE_SHA256,
            "source_epoch": 9500,
            "smoke_epochs": 100,
            "bars": 130,
            "seed": 42,
            "requested_episodes": 2049,
            "episodes": episodes,
            "captured": captured,
            "crash": crash,
            "timeout": timeout,
            "capture_rate": captured / episodes,
            "crash_rate": crash / episodes,
            "timeout_rate": timeout / episodes,
            "training_max_kl": 0.01,
            "training_max_task_input_oob_rate": 0.0,
            "evaluation_max_task_input_oob_rate": 0.0,
            "max_task_input_oob_rate": 0.0,
            "max_rollback_streak": 0.0,
            "final_rollback_streak": 0.0,
            "heldout_result_json": str(result_path.resolve()),
            "heldout_result_sha256": _sha256(result_path),
            "evaluation_contract": copy.deepcopy(_ATTEST.EVALUATION_CONTRACT),
        }

    def _write_training_audit(self, rollback_step=None, duplicate_step=None):
        writer = SummaryWriter(log_dir=str(self.run_root / "summaries"))
        for step in range(9501, 9601):
            writer.add_scalar("ppo/behavior_kl_audit_max", 0.01, step)
            rolled_back = 1.0 if step == rollback_step else 0.0
            writer.add_scalar("ppo/epoch_rollback", rolled_back, step)
            writer.add_scalar("ppo/epoch_rollback_total", rolled_back, step)
            writer.add_scalar(
                "ppo/epoch_rollback_streak", rolled_back, step
            )
            for axis in _ATTEST.ACTION_AXES:
                writer.add_scalar(f"policy_action/raw_oob_{axis}", 0.0, step)
        if duplicate_step is not None:
            writer.add_scalar("ppo/behavior_kl_audit_max", 0.01, duplicate_step)
        writer.close()

    def _write_bulk_result(
        self, checkpoint, *, falsify_capture_rate=False, with_receipt=True
    ):
        captured, crash, timeout = 1435, 600, 14
        episodes = captured + crash + timeout
        nonce = "a" * 64
        checkpoint_sha = _sha256(checkpoint)
        snapshot = Path(self.temp.name) / "checkpoint_snapshot.pth"
        shutil.copyfile(checkpoint, snapshot)
        result_path = Path(self.temp.name) / "bulk_result.json"
        receipt_path = result_path.with_suffix(".receipt.json")
        log_path = result_path.with_suffix(".log")
        log_path.write_text("synthetic evaluator log\n", encoding="utf-8")
        source_snapshot = Path(self.temp.name) / "source_snapshot" / "aerial_gym"
        source_snapshot.mkdir(parents=True, exist_ok=True)
        source_copy = source_snapshot / "evaluator.sh"
        shutil.copyfile(_EVALUATOR, source_copy)
        environment_path = Path(self.temp.name) / "python_environment.txt"
        environment_path.write_text("synthetic test environment\n", encoding="utf-8")
        manifest_path = Path(self.temp.name) / "source_manifest.json"
        manifest = {
            "schema_version": 2,
            "repository_root": str(_ROOT),
            "git_commit": "0" * 40,
            "git_dirty": False,
            "python_environment": environment_path.name,
            "python_environment_sha256": _sha256(environment_path),
            "runtime_file_count": 1,
            "runtime_files": [
                {
                    "path": str(_EVALUATOR.relative_to(_ROOT)),
                    "sha256": _sha256(_EVALUATOR),
                    "size_bytes": _EVALUATOR.stat().st_size,
                    "snapshot": str(source_copy.relative_to(manifest_path.parent)),
                }
            ],
        }
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        payload = {
            "schema_version": 1,
            "requested_episodes": 2049,
            "actual_episodes": episodes,
            "checkpoint": str(checkpoint.resolve()),
            "condition": {
                "seed": 42,
                "bars": 130,
                "target_pattern": "mixed",
                "target_speed_mode": "uniform",
                "target_speed_min_mps": 0.3,
                "target_speed_max_mps": 1.5,
                "pursuer_max_speed_mps": 2.5,
                "pursuer_speed_limit_semantics": "per_axis_xy",
                "pursuer_per_axis_speed_limit_mps": 2.5,
                "pursuer_max_horizontal_request_norm_mps": 2.5 * 2**0.5,
                "policy_output_dim": 4,
                "policy_z_output_overwritten_by_altitude_pi": True,
                "policy_z_persisted_in_prev_action_observation": True,
                "oob_margin_m": 1.0,
                "episode_len_steps": 600,
                "num_envs": 128,
                "goal_dist_min_m": 6.0,
                "goal_dist_max_m": 28.0,
                "full_goal_distribution": True,
                "fov_curriculum_saturated": True,
                "evaluation_nonce": nonce,
                "runtime_sim_config_class": "BaseSimConfig",
                "physics_dt_s": 0.01,
                "physics_substeps": 1,
                "physics_steps_per_rl_step": 10,
                "rl_step_dt_s": 0.1,
            },
            "outcome": {
                "captured": captured,
                "crash": crash,
                "timeout": timeout,
                "capture_rate": 0.99 if falsify_capture_rate else captured / episodes,
                "crash_rate": crash / episodes,
                "timeout_rate": timeout / episodes,
            },
            "action": {
                "policy": "squashed_gaussian",
                "samples": 25000,
                "task_input_oob_rate": [0.0, 0.0, 0.0, 0.0],
            },
            "v2_evaluation_contract": copy.deepcopy(_ATTEST.EVALUATION_CONTRACT),
            "checkpoint_sha256": checkpoint_sha,
            "evaluated_checkpoint_snapshot": str(snapshot.resolve()),
            "evaluated_checkpoint_snapshot_sha256": _sha256(snapshot),
            "evaluator_script": str(_EVALUATOR.resolve()),
            "evaluator_script_sha256": _sha256(_EVALUATOR),
            "runtime_source_manifest": str(manifest_path.resolve()),
            "runtime_source_manifest_sha256": _sha256(manifest_path),
            "evaluation_receipt": str(receipt_path.resolve()),
        }
        result_path.write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        if with_receipt:
            receipt = {
                "schema_version": 2,
                "producer": "eval_navrl_v2_density_sweep.sh",
                "started_at_utc": "2026-08-01T00:00:00+00:00",
                "completed_at_utc": "2026-08-01T00:01:00+00:00",
                "evaluation_nonce": nonce,
                "source_checkpoint": str(checkpoint.resolve()),
                "source_checkpoint_sha256": checkpoint_sha,
                "evaluated_checkpoint_snapshot": str(snapshot.resolve()),
                "evaluated_checkpoint_snapshot_sha256": _sha256(snapshot),
                "result_json": str(result_path.resolve()),
                "result_sha256": _sha256(result_path),
                "log_file": str(log_path.resolve()),
                "log_sha256": _sha256(log_path),
                "evaluator_script": str(_EVALUATOR.resolve()),
                "evaluator_script_sha256": _sha256(_EVALUATOR),
                "runtime_source_manifest": str(manifest_path.resolve()),
                "runtime_source_manifest_sha256": _sha256(manifest_path),
                "runtime_source_file_count": 1,
                "runtime_git_commit": "0" * 40,
                "runtime_git_dirty": False,
                "python_environment_manifest": str(environment_path.resolve()),
                "python_environment_manifest_sha256": _sha256(environment_path),
                "bars": 130,
                "seed": 42,
                "requested_episodes": 2049,
                "actual_episodes": episodes,
            }
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
            )
        return result_path

    def _run_preflight(self, mode, checkpoint, extra_env=None):
        env = os.environ.copy()
        env.update(
            {
                "PYTHON": _PYTHON,
                "RECOVERY_MODE": mode,
                "CKPT": str(checkpoint),
                "NAVRL_PREFLIGHT_ONLY": "1",
                "SEED": "1",
            }
        )
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [str(_RUNNER)],
            cwd=_RUNNER.parent,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def _run_eval_preflight(self, checkpoint, extra_env=None):
        env = os.environ.copy()
        for name in (
            "NAVRL_GENERAL_EVAL",
            "NAVRL_INTERACTIVE",
            "NAVRL_V2_FORCE",
        ):
            env.pop(name, None)
        env.update(
            {
                "PYTHON": _PYTHON,
                "NAVRL_PREFLIGHT_ONLY": "1",
                "NAVRL_V2_DENSITIES": "130",
            }
        )
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [str(_EVALUATOR), str(checkpoint), "2049"],
            cwd=_EVALUATOR.parent,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_curriculum_preflight_accepts_bound_artifact_and_preserves_lr(self):
        checkpoint = self._save_checkpoint()
        result_path = self._write_bulk_result(checkpoint)
        self._write_training_audit()
        _ATTEST.create_attestation(checkpoint, result_path)

        completed = self._run_preflight("curriculum", checkpoint)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("lr=5e-06", completed.stdout)
        self.assertIn("attestation VERIFIED", completed.stdout)
        self.assertIn("PREFLIGHT PASS", completed.stdout)
        self.assertIn("child handoff validated", completed.stdout)

    def test_curriculum_preflight_rejects_tampered_result_binding(self):
        checkpoint = self._save_checkpoint()
        result_path = self._write_bulk_result(checkpoint)
        self._write_training_audit()
        _ATTEST.create_attestation(checkpoint, result_path)
        result_path.write_text('{"tampered":true}\n', encoding="utf-8")

        completed = self._run_preflight("curriculum", checkpoint)
        self.assertNotEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("recovery attestation REFUSED", completed.stdout)

    def test_curriculum_preflight_rejects_self_written_pass_and_arbitrary_result(self):
        checkpoint = self._save_checkpoint()
        result_path = self._result_file()
        artifact = self._attestation(checkpoint, result_path)
        (self.run_root / ".navrl_v2_recovery_eval_pass.json").write_text(
            json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8"
        )

        completed = self._run_preflight("curriculum", checkpoint)
        self.assertNotEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("recovery attestation REFUSED", completed.stdout)

    def test_continue_preflight_validates_embedded_attestation_and_preserves_lr(self):
        smoke_checkpoint = self._save_checkpoint()
        result_path = self._write_bulk_result(smoke_checkpoint)
        self._write_training_audit()
        artifact_path = _ATTEST.create_attestation(smoke_checkpoint, result_path)
        artifact_bytes = artifact_path.read_bytes()
        state = self._state(stage="curriculum", current_lr=1.25e-6)
        state["cfg_recovery_eval_attestation_sha256"] = hashlib.sha256(
            artifact_bytes
        ).hexdigest()
        state["cfg_recovery_eval_attestation_b64"] = base64.b64encode(
            artifact_bytes
        ).decode("ascii")
        checkpoint = self._save_checkpoint(epoch=9700, state=state)

        completed = self._run_preflight("continue", checkpoint)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("lr=1.25e-06", completed.stdout)

    def test_continue_preflight_rejects_broken_smoke_anchor_clock(self):
        smoke_checkpoint = self._save_checkpoint()
        result_path = self._write_bulk_result(smoke_checkpoint)
        self._write_training_audit()
        artifact_path = _ATTEST.create_attestation(smoke_checkpoint, result_path)
        artifact_bytes = artifact_path.read_bytes()
        state = self._state(stage="curriculum", current_lr=1.25e-6)
        state["cfg_recovery_eval_attestation_sha256"] = hashlib.sha256(
            artifact_bytes
        ).hexdigest()
        state["cfg_recovery_eval_attestation_b64"] = base64.b64encode(
            artifact_bytes
        ).decode("ascii")
        checkpoint = self._save_checkpoint(epoch=9700, state=state)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload["env_state"]["num_task_steps"] -= 1
        torch.save(payload, checkpoint)

        completed = self._run_preflight("continue", checkpoint)
        self.assertNotEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("breaks the smoke-anchor lineage", completed.stdout)

    def test_recovery_training_seed_cannot_equal_heldout_seed(self):
        checkpoint = self._save_checkpoint()
        completed = self._run_preflight("curriculum", checkpoint, {"SEED": "42"})
        self.assertNotEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("training seed is fixed at 1", completed.stdout)

    def test_recovery_evaluator_preflight_uses_integer_provenance(self):
        checkpoint = self._save_checkpoint()
        completed = self._run_eval_preflight(checkpoint)
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("fov_curriculum=3000 detector_pixels=2 ", completed.stdout)
        self.assertNotIn("fov_curriculum=3000.0", completed.stdout)
        self.assertIn("action_selection=deterministic", completed.stdout)

    def test_recovery_evaluator_rejects_stochastic_attestation(self):
        checkpoint = self._save_checkpoint()
        completed = self._run_eval_preflight(
            checkpoint, {"NAVRL_V2_ACTION_MODE": "stochastic"}
        )
        self.assertNotEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("requires deterministic action selection", completed.stdout)

    def test_evaluator_rejects_unknown_action_selection(self):
        checkpoint = self._save_checkpoint()
        completed = self._run_eval_preflight(
            checkpoint, {"NAVRL_V2_ACTION_MODE": "coin-flip"}
        )
        self.assertNotEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("must be deterministic or stochastic", completed.stdout)

    def test_recovery_evaluator_pins_main_runtime_against_hostile_environment(self):
        checkpoint = self._save_checkpoint()
        completed = self._run_eval_preflight(
            checkpoint,
            {
                "AERIAL_GYM_SIM_NAME": "base_sim_4ms",
                "NUM_ENVS": "1",
                "GPU4GB": "0",
            },
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn(
            "runtime=main/base_sim envs=128 physics=base_sim_dt0.01",
            completed.stdout,
        )

    def test_recovery_evaluator_rejects_noncanonical_4gb_runtime(self):
        checkpoint = self._save_checkpoint()
        completed = self._run_eval_preflight(checkpoint, {"GPU4GB": "1"})
        self.assertNotEqual(completed.returncode, 0, completed.stdout)
        self.assertIn(
            "requires the canonical main/base_sim/128-env runtime", completed.stdout
        )

    def test_general_evaluator_4gb_profile_reports_actual_inherited_timestep(self):
        state = self._state(stage="curriculum")
        checkpoint = self._save_checkpoint(epoch=9700, state=state)
        completed = self._run_eval_preflight(checkpoint, {"GPU4GB": "1"})
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn(
            "runtime=4gb/base_sim_4gb envs=64 physics=base_sim_4gb_dt0.01_buffers",
            completed.stdout,
        )

    def test_recovery_evaluator_forbids_force_override(self):
        checkpoint = self._save_checkpoint()
        completed = self._run_eval_preflight(checkpoint, {"NAVRL_V2_FORCE": "1"})
        self.assertNotEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("NAVRL_V2_FORCE is forbidden", completed.stdout)

    def test_attestation_helper_binds_full_contract_and_training_window(self):
        checkpoint = self._save_checkpoint()
        result = self._write_bulk_result(checkpoint)
        self._write_training_audit()

        artifact_path = _ATTEST.create_attestation(checkpoint, result)
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(artifact["verdict"], "PASS")
        self.assertEqual(artifact["max_rollback_streak"], 0.0)
        self.assertEqual(artifact["max_epoch_rollback"], 0.0)
        self.assertEqual(artifact["max_rollback_total"], 0.0)
        self.assertEqual(artifact["final_rollback_total"], 0.0)
        self.assertEqual(
            artifact["evaluator_receipt_sha256"],
            _sha256(Path(artifact["evaluator_receipt_json"])),
        )
        self.assertEqual(artifact["evaluation_contract"], _ATTEST.EVALUATION_CONTRACT)

    def test_attestation_helper_rejects_any_mid_window_rollback(self):
        checkpoint = self._save_checkpoint()
        result = self._write_bulk_result(checkpoint)
        self._write_training_audit(rollback_step=9550)

        with self.assertRaisesRegex(RuntimeError, "contains a rollback"):
            _ATTEST.create_attestation(checkpoint, result)

    def test_attestation_helper_rejects_rates_that_disagree_with_counts(self):
        checkpoint = self._save_checkpoint()
        result = self._write_bulk_result(checkpoint, falsify_capture_rate=True)
        self._write_training_audit()

        with self.assertRaisesRegex(RuntimeError, "disagrees with count/actual"):
            _ATTEST.create_attestation(checkpoint, result)

    def test_attestation_helper_rejects_plausible_json_without_evaluator_receipt(self):
        checkpoint = self._save_checkpoint()
        result = self._write_bulk_result(checkpoint, with_receipt=False)
        self._write_training_audit()

        with self.assertRaisesRegex(RuntimeError, "evaluator receipt cannot be read"):
            _ATTEST.create_attestation(checkpoint, result)

    def test_attestation_helper_rejects_checkpoint_swapped_after_evaluation(self):
        checkpoint = self._save_checkpoint()
        result = self._write_bulk_result(checkpoint)
        self._write_training_audit()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload["swapped_after_evaluation"] = True
        torch.save(payload, checkpoint)

        with self.assertRaisesRegex(RuntimeError, "snapshot bytes differ"):
            _ATTEST.create_attestation(checkpoint, result)

    def test_attestation_helper_rejects_duplicate_tensorboard_step(self):
        checkpoint = self._save_checkpoint()
        result = self._write_bulk_result(checkpoint)
        self._write_training_audit(duplicate_step=9550)

        with self.assertRaisesRegex(RuntimeError, "duplicate writes"):
            _ATTEST.create_attestation(checkpoint, result)


if __name__ == "__main__":
    unittest.main()
