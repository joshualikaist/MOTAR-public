import importlib.util
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from history_helpers import require_local_evidence
from unittest import mock

import torch
from torch.utils.tensorboard import SummaryWriter


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/update_status_snapshot.py"
_SPEC = importlib.util.spec_from_file_location("update_status_snapshot", _MODULE_PATH)
_STATUS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STATUS)

_ATTESTATION_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools/navrl_v2_recovery_attestation.py"
)
_ATTESTATION_SPEC = importlib.util.spec_from_file_location(
    "navrl_v2_recovery_attestation_for_status_test", _ATTESTATION_MODULE_PATH
)
_ATTESTATION = importlib.util.module_from_spec(_ATTESTATION_SPEC)
_ATTESTATION_SPEC.loader.exec_module(_ATTESTATION)


class StatusSnapshotTest(unittest.TestCase):
    @staticmethod
    def _write_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write_training_audit(run_dir):
        writer = SummaryWriter(log_dir=str(run_dir / "summaries"))
        for step in range(9501, 9601):
            writer.add_scalar("ppo/behavior_kl_audit_max", 0.01, step)
            writer.add_scalar("ppo/epoch_rollback", 0.0, step)
            writer.add_scalar("ppo/epoch_rollback_total", 0.0, step)
            writer.add_scalar("ppo/epoch_rollback_streak", 0.0, step)
            for axis in ("x", "y", "z", "yaw"):
                writer.add_scalar(f"policy_action/raw_oob_{axis}", 0.0, step)
        writer.close()

    def _make_recovery_artifacts(self, root, *, canonical=True):
        runs_root = root / "runs"
        run_name = "ppo_260801_1200_navrl_v2-recover-smoke-130bars-s1"
        run_dir = runs_root / run_name
        checkpoint_path = run_dir / "nn/last_gen_ppo_ep_9600_rew_1.pth"
        checkpoint_path.parent.mkdir(parents=True)

        env_state = dict(_STATUS._RECOVERY_CHECKPOINT_CONTRACT)
        env_state.update(
            {
                "cfg_recovery_stage": "smoke",
                "cfg_recovery_source_sha256": _STATUS._RECOVERY_SOURCE_SHA256,
                "cfg_recovery_source_epoch": 9500,
                "cfg_recovery_smoke_required_epochs": 100,
                "cfg_recovery_smoke_bars": 130,
                "n_bars_active": 130,
            }
        )
        torch.save(
            {"epoch": 9600, "frame": 39321600, "env_state": env_state},
            checkpoint_path,
        )
        (run_dir / ".aerial_training_finished").write_text(
            "epoch=9600\n", encoding="utf-8"
        )

        actual_episodes = 2176
        captured, crash, timeout = 1500, 600, 76
        result_path = root / "eval/130bars.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_snapshot = result_path.parent / "checkpoint_snapshot.pth"
        shutil.copyfile(checkpoint_path, checkpoint_snapshot)
        receipt_path = result_path.with_suffix(".receipt.json")
        log_path = result_path.with_suffix(".log")
        log_path.write_text("synthetic evaluator log\n", encoding="utf-8")
        evaluator_path = _ATTESTATION.EVALUATOR_SCRIPT.resolve()
        source_snapshot = result_path.parent / "source_snapshot/aerial_gym"
        source_snapshot.mkdir(parents=True, exist_ok=True)
        evaluator_copy = source_snapshot / "evaluator.sh"
        shutil.copyfile(evaluator_path, evaluator_copy)
        environment_path = result_path.parent / "python_environment.txt"
        environment_path.write_text("synthetic test environment\n", encoding="utf-8")
        manifest_path = result_path.parent / "source_manifest.json"
        manifest = {
            "schema_version": 2,
            "repository_root": str(Path(__file__).resolve().parents[1]),
            "git_commit": "0" * 40,
            "git_dirty": False,
            "python_environment": environment_path.name,
            "python_environment_sha256": self._digest(environment_path),
            "runtime_file_count": 1,
            "runtime_files": [
                {
                    "path": str(
                        evaluator_path.relative_to(Path(__file__).resolve().parents[1])
                    ),
                    "sha256": self._digest(evaluator_path),
                    "size_bytes": evaluator_path.stat().st_size,
                    "snapshot": str(evaluator_copy.relative_to(manifest_path.parent)),
                }
            ],
        }
        self._write_json(manifest_path, manifest)
        nonce = "b" * 64
        result = {
            "schema_version": 1,
            "requested_episodes": 2049,
            "actual_episodes": actual_episodes,
            "checkpoint": str(checkpoint_path.resolve()),
            "condition": {
                "seed": 42,
                "bars": 130,
                "target_pattern": "mixed",
                "target_speed_mode": "uniform",
                "target_speed_mps": None,
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
                "capture_rate": captured / actual_episodes,
                "crash_rate": crash / actual_episodes,
                "timeout_rate": timeout / actual_episodes,
            },
            "action": {
                "policy": "squashed_gaussian",
                "samples": 25000,
                "task_input_oob_rate": [0.0, 0.0, 0.0, 0.0],
            },
            "v2_evaluation_contract": dict(_STATUS._RECOVERY_RESULT_CONTRACT),
            "checkpoint_sha256": self._digest(checkpoint_path),
            "evaluated_checkpoint_snapshot": str(checkpoint_snapshot.resolve()),
            "evaluated_checkpoint_snapshot_sha256": self._digest(
                checkpoint_snapshot
            ),
            "evaluator_script": str(evaluator_path),
            "evaluator_script_sha256": self._digest(evaluator_path),
            "runtime_source_manifest": str(manifest_path.resolve()),
            "runtime_source_manifest_sha256": self._digest(manifest_path),
            "evaluation_receipt": str(receipt_path.resolve()),
        }
        self._write_json(result_path, result)
        receipt = {
            "schema_version": 2,
            "producer": "eval_navrl_v2_density_sweep.sh",
            "started_at_utc": "2026-08-01T00:00:00+00:00",
            "completed_at_utc": "2026-08-01T00:01:00+00:00",
            "evaluation_nonce": nonce,
            "source_checkpoint": str(checkpoint_path.resolve()),
            "source_checkpoint_sha256": self._digest(checkpoint_path),
            "evaluated_checkpoint_snapshot": str(checkpoint_snapshot.resolve()),
            "evaluated_checkpoint_snapshot_sha256": self._digest(
                checkpoint_snapshot
            ),
            "result_json": str(result_path.resolve()),
            "result_sha256": self._digest(result_path),
            "log_file": str(log_path.resolve()),
            "log_sha256": self._digest(log_path),
            "evaluator_script": str(evaluator_path),
            "evaluator_script_sha256": self._digest(evaluator_path),
            "runtime_source_manifest": str(manifest_path.resolve()),
            "runtime_source_manifest_sha256": self._digest(manifest_path),
            "runtime_source_file_count": 1,
            "runtime_git_commit": "0" * 40,
            "runtime_git_dirty": False,
            "python_environment_manifest": str(environment_path.resolve()),
            "python_environment_manifest_sha256": self._digest(environment_path),
            "bars": 130,
            "seed": 42,
            "requested_episodes": 2049,
            "actual_episodes": actual_episodes,
        }
        self._write_json(receipt_path, receipt)

        attestation_path = run_dir / ".navrl_v2_recovery_eval_pass.json"
        if canonical:
            self._write_training_audit(run_dir)
            attestation_path = _ATTESTATION.create_attestation(
                checkpoint_path, result_path
            )
        else:
            # A plausible, hand-written PASS must not substitute for the missing 100-epoch
            # TensorBoard evidence. This mirrors the historical dashboard bypass exactly.
            attestation = {
                "schema_version": 1,
                "verdict": "PASS",
                "created_at_utc": "2026-08-01T00:00:00+00:00",
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": self._digest(checkpoint_path),
                "checkpoint_epoch": 9600,
                "checkpoint_frame": 39321600,
                "source_checkpoint_sha256": _STATUS._RECOVERY_SOURCE_SHA256,
                "source_epoch": 9500,
                "smoke_epochs": 100,
                "bars": 130,
                "seed": 42,
                "requested_episodes": 2049,
                "episodes": actual_episodes,
                "captured": captured,
                "crash": crash,
                "timeout": timeout,
                "capture_rate": captured / actual_episodes,
                "crash_rate": crash / actual_episodes,
                "timeout_rate": timeout / actual_episodes,
                "training_max_kl": 0.01,
                "training_max_task_input_oob_rate": 0.0,
                "evaluation_max_task_input_oob_rate": 0.0,
                "max_task_input_oob_rate": 0.0,
                "max_rollback_streak": 0.0,
                "final_rollback_streak": 0.0,
                "heldout_result_json": str(result_path.resolve()),
                "heldout_result_sha256": self._digest(result_path),
                "thresholds": dict(_STATUS._RECOVERY_ATTESTATION_THRESHOLDS),
                "evaluation_contract": dict(_STATUS._RECOVERY_RESULT_CONTRACT),
            }
            self._write_json(attestation_path, attestation)
        return {
            "runs_root": runs_root,
            "run_name": run_name,
            "run_dir": run_dir,
            "checkpoint_path": checkpoint_path,
            "result_path": result_path,
            "attestation_path": attestation_path,
        }

    def test_short_transaction_runs_do_not_replace_latest(self):
        self.assertTrue(
            _STATUS._is_smoke_run(
                "ppo_260801_0528_navrl_v2-full-transaction-forced-final-s1"
            )
        )
        self.assertTrue(
            _STATUS._is_smoke_run(
                "ppo_260801_0455_navrl_v2-central-transaction-integration-smoke-s1"
            )
        )

    def test_detection_range_stage1_is_current_and_blocks_stage2(self):
        result = _STATUS._detection_range_stage1_result()
        self.assertIsNotNone(result)
        self.assertEqual(result["quality_gate_count"], 17)
        self.assertAlmostEqual(
            result["delta"]["never_acquired"] * 100.0,
            -5.270863836017569,
        )
        update = _STATUS._detection_range_stage1_update()
        self.assertFalse(update["active_experiment"]["ab_gate_pass"])
        self.assertFalse(update["active_experiment"]["stage2_authorised"])
        self.assertIn("Do not run the 10k Stage 2", update["decision"])

        plan = _STATUS._sim2real_72h()
        if (
            plan["simulation_verification"]
            .get("recovery_v2_lower1p25_gate", {})
            .get("status")
            != "VERIFIED_FAIL"
        ):
            # The finalized receipts bind their original absolute worktree root. A clean clone or
            # a copied ignored result bundle must fail closed rather than silently re-root evidence.
            self.assertIn("TRACK B EVIDENCE UNAVAILABLE/MALFORMED", plan["status"])
            self.assertIn("NO TRACK B AUTHORITY", plan["status"])
            self.assertNotIn("ROUTE MECHANISM FAILED", plan["status"])
            self.assertFalse(plan["evidence"]["stage2_authorised"])
            return
        self.assertEqual(
            plan["status"],
            "TRACK A STAGE 1 RANGE INCONCLUSIVE · TRACK B RECOVERY-V2 ROUTE "
            "MECHANISM FAILED · NO FURTHER TRACK B AUTHORITY · HARDWARE NEXT",
        )
        self.assertEqual(plan["as_of"], "2026-08-26")
        self.assertEqual(
            plan["simulation_verification"]["preflight_claim_status"],
            "SYNTHETIC_ONLY",
        )
        self.assertEqual(plan["simulation_verification"]["physical_gate"], "BLOCKED")
        current = plan["simulation_verification"]["recovery_v2_lower1p25_gate"]
        self.assertEqual(current["status"], "VERIFIED_FAIL")
        self.assertEqual(current["contract_variant"], "baseline_1p25")
        self.assertEqual(
            current["canonical_1p5_contract"], "SEPARATE_UNCHANGED_NOT_PASSED"
        )
        self.assertEqual(current["integrity"], "PASS_32_CELL_INTEGRITY")
        self.assertEqual(current["route_mechanism"], "FAIL_ROUTE_MECHANISM")
        self.assertEqual(
            current["cells"],
            {
                "passed": 7,
                "total": 32,
                "route_off_passed": 7,
                "route_off_total": 16,
                "recovery_passed": 0,
                "recovery_total": 16,
                "passing_lineage": "route_off_only",
            },
        )
        self.assertEqual(
            current["plan_success_70bar_4speed"]["numerator"], 190
        )
        self.assertEqual(
            current["plan_success_70bar_4speed"]["denominator"], 203
        )
        self.assertEqual(current["plan_success_70bar_4speed"]["pct"], 93.60)
        self.assertEqual(current["fallback_70bar_4speed"]["numerator"], 18381)
        self.assertEqual(current["fallback_70bar_4speed"]["denominator"], 38400)
        self.assertEqual(current["fallback_70bar_4speed"]["pct"], 47.87)
        self.assertEqual(
            current["goals_per_env_70bar_0_6mps"],
            {"numerator": 7, "denominator": 32, "value": 0.21875, "gate": 0.5},
        )
        self.assertEqual(current["no_connector_occupancy"]["numerator"], 96854)
        self.assertEqual(current["no_connector_occupancy"]["denominator"], 153600)
        self.assertEqual(current["no_connector_occupancy"]["pct"], 63.06)
        self.assertEqual(
            current["hard_breach_no_connector_entries"],
            {"numerator": 0, "denominator": 534},
        )
        self.assertEqual(
            current["authority"], "NO_FURTHER_TRACK_B_GPU_PPO_RETUNE_RERUN"
        )

        forensics = plan["simulation_verification"][
            "recovery_v2_no_connector_forensics"
        ]
        self.assertEqual(forensics["status"], "DESCRIPTIVE_ONLY")
        self.assertEqual(
            forensics["decision_rule"],
            {
                "label": "INCONCLUSIVE",
                "primary_n": 1,
                "anchor_present": 0,
                "hard_free_soft_unsafe": 1,
                "identity_void": False,
                "passes_32_cell_mechanism": False,
                "authorizes_retune_or_ppo": False,
            },
        )
        self.assertEqual(
            forensics["no_connector_classes"],
            {
                "total": 106,
                "connect_failed_certificate_likely": 49,
                "brake_timeout": 32,
                "connect_failed_resume_likely": 23,
                "connect_timeout": 1,
                "brake_no_anchor_likely": 1,
            },
        )
        self.assertEqual(
            plan["simulation_verification"]["track_b_authority"],
            "CLOSED_NO_FURTHER_GPU_PPO_RETUNE_RERUN",
        )
        self.assertEqual(
            plan["simulation_verification"]["preflight_steps"][
                "physical_target_gate"
            ],
            current,
        )

        # Attempt 2 and the v1 RECOVERY_DOMINANT diagnosis remain visible as history.
        routed = plan["simulation_verification"]["routed_physical_target_gate_attempt2"]
        self.assertEqual(routed["integrity"], "PASS_32_CELL_INTEGRITY")
        self.assertEqual(routed["route_mechanism"], "FAIL_ROUTE_MECHANISM")
        self.assertEqual(routed["lineage_status"], "HISTORICAL_ATTEMPT2")
        self.assertEqual(
            routed["highest_passing_speed_mps_by_density"],
            {"70": None, "150": None, "205": None, "300": None},
        )
        self.assertEqual(routed["plan_success_70bar_4speed_pool_pct"], 14.5467)
        historical = plan["simulation_verification"]["historical_post_wall_brake_speed_envelope"]
        self.assertEqual(historical["route_mode"], "off_historical_lineage")
        self.assertEqual(historical["highest_passing_speed_mps_by_density"]["70"], 0.9)
        v1 = plan["simulation_verification"]["preflight_steps"][
            "route_recovery_forensics"
        ]
        self.assertEqual(v1["diagnostic_verdict"], "RECOVERY_DOMINANT")
        self.assertEqual(v1["lineage_status"], "HISTORICAL_V1_DIAGNOSTIC")
        stored_snapshot = json.loads(_STATUS.STATUS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(stored_snapshot["repo"], _STATUS.REPOSITORY_ID)
        stored = stored_snapshot["sim2real_72h"]
        self.assertEqual(stored, plan)
        self.assertFalse(plan["evidence"]["stage2_authorised"])
        self.assertEqual(len(plan["days"]), 3)

    def test_recovery_v2_readers_fail_closed_on_missing_or_malformed_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root / "missing.json"
            malformed = root / "malformed.json"
            malformed.write_text("{not json", encoding="utf-8")
            with mock.patch.object(
                _STATUS, "RECOVERY_V2_LOWER1P25_GATE_PATH", missing
            ):
                gate = _STATUS._recovery_v2_lower1p25_gate()
            with mock.patch.object(
                _STATUS, "RECOVERY_V2_NO_CONNECTOR_FORENSICS_PATH", malformed
            ):
                forensics = _STATUS._recovery_v2_no_connector_forensics()
        for block in (gate, forensics):
            self.assertEqual(block["status"], "RESULT_UNAVAILABLE_OR_MALFORMED")
            self.assertEqual(
                block["authority"], "NO_FURTHER_TRACK_B_GPU_PPO_RETUNE_RERUN"
            )
            self.assertEqual(block["physical_ppo"], "BLOCKED")
            self.assertFalse(block["hardware_claim"])

    def test_recovery_v2_canonical_receipt_verifiers_run_on_real_evidence(self):
        with mock.patch.object(
            _STATUS, "_load_status_module", wraps=_STATUS._load_status_module
        ) as loader:
            gate = _STATUS._recovery_v2_lower1p25_gate()
            forensics = _STATUS._recovery_v2_no_connector_forensics()
        if gate.get("status") != "VERIFIED_FAIL" or forensics.get("status") != "DESCRIPTIVE_ONLY":
            self.skipTest("canonical Track B receipts require their exact recorded source root")
        self.assertEqual(gate["status"], "VERIFIED_FAIL")
        self.assertEqual(forensics["status"], "DESCRIPTIVE_ONLY")
        loader.assert_any_call(
            _STATUS.RECOVERY_V2_GATE_VERIFIER_PATH,
            "_recovery_v2_gate_verifier_for_status",
        )
        loader.assert_any_call(
            _STATUS.RECOVERY_V2_NO_CONNECTOR_VERIFIER_PATH,
            "_recovery_v2_no_connector_verifier_for_status",
        )

    def test_recovery_v2_gate_rejects_malformed_and_receipt_unbound_temp_summary(self):

        require_local_evidence(
            _REPO_ROOT / "results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/summary.json",
            _REPO_ROOT / "results/navrl_physical_target_recovery_v2_no_connector_forensics_seed827/summary.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            malformed = root / "malformed.json"
            malformed.write_text("{not json", encoding="utf-8")
            with mock.patch.object(
                _STATUS, "RECOVERY_V2_LOWER1P25_GATE_PATH", malformed
            ):
                block = _STATUS._recovery_v2_lower1p25_gate()
            self.assertEqual(block["status"], "RESULT_UNAVAILABLE_OR_MALFORMED")

            unbound = root / "summary.json"
            payload = json.loads(
                _STATUS.RECOVERY_V2_LOWER1P25_GATE_PATH.read_text(encoding="utf-8")
            )
            payload["cells"][0]["pass"] = False
            self._write_json(unbound, payload)
            with mock.patch.object(
                _STATUS, "RECOVERY_V2_LOWER1P25_GATE_PATH", unbound
            ):
                block = _STATUS._recovery_v2_lower1p25_gate()
            self.assertEqual(block["status"], "RESULT_UNAVAILABLE_OR_MALFORMED")

    def test_recovery_v2_gate_rejects_duplicate_identity_and_impossible_fallback(self):

        require_local_evidence(
            _REPO_ROOT / "results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/summary.json",
            _REPO_ROOT / "results/navrl_physical_target_recovery_v2_no_connector_forensics_seed827/summary.json")
        payload = json.loads(
            _STATUS.RECOVERY_V2_LOWER1P25_GATE_PATH.read_text(encoding="utf-8")
        )
        duplicate = copy.deepcopy(payload)
        for cell in duplicate["cells"]:
            if (
                cell["route_mode"] == "global_astar_recovery_v2"
                and cell["bars"] == 70
                and cell["speed_mps"] == 1.2
            ):
                cell["speed_mps"] = 0.9
                break
        with mock.patch.object(
            _STATUS, "_canonical_recovery_v2_gate_summary", return_value=duplicate
        ):
            block = _STATUS._recovery_v2_lower1p25_gate()
        self.assertEqual(block["status"], "RESULT_UNAVAILABLE_OR_MALFORMED")

        impossible = copy.deepcopy(payload)
        for cell in impossible["cells"]:
            if cell["route_mode"] == "global_astar_recovery_v2" and cell["bars"] == 70:
                cell["route"]["counter_delta"]["fallback_intervals"] = (
                    cell["telemetry_summary"]["interval_denominator"] + 1
                )
                break
        with mock.patch.object(
            _STATUS, "_canonical_recovery_v2_gate_summary", return_value=impossible
        ):
            block = _STATUS._recovery_v2_lower1p25_gate()
        self.assertEqual(block["status"], "RESULT_UNAVAILABLE_OR_MALFORMED")

    def test_recovery_v2_no_anchor_rejects_impossible_counts(self):

        require_local_evidence(
            _REPO_ROOT / "results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/summary.json",
            _REPO_ROOT / "results/navrl_physical_target_recovery_v2_no_connector_forensics_seed827/summary.json")
        payload = json.loads(
            _STATUS.RECOVERY_V2_NO_CONNECTOR_FORENSICS_PATH.read_text(encoding="utf-8")
        )
        impossible = copy.deepcopy(payload)
        impossible["pooled"]["anchor_present"] = 999
        impossible["pooled"]["decision_rule"]["anchor_present"] = 999
        impossible["decision_rule"]["anchor_present"] = 999
        with mock.patch.object(
            _STATUS,
            "_canonical_no_connector_summary",
            return_value=impossible,
        ):
            block = _STATUS._recovery_v2_no_connector_forensics()
        self.assertEqual(block["status"], "RESULT_UNAVAILABLE_OR_MALFORMED")

        non_integer = copy.deepcopy(payload)
        non_integer["pooled"]["class_counts"]["brake_timeout"] = True
        with mock.patch.object(
            _STATUS,
            "_canonical_no_connector_summary",
            return_value=non_integer,
        ):
            block = _STATUS._recovery_v2_no_connector_forensics()
        self.assertEqual(block["status"], "RESULT_UNAVAILABLE_OR_MALFORMED")

    def test_recovery_v2_no_anchor_rejects_malformed_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.json"
            receipt = root / "receipt.json"
            marker = root / ".COMPLETE.json"
            summary.write_text("{}", encoding="utf-8")
            receipt.write_text("{}", encoding="utf-8")
            marker.write_text(
                json.dumps(
                    {
                        "schema": "wrong",
                        "receipt_sha256": self._digest(receipt),
                        "summary_sha256": self._digest(summary),
                    }
                ),
                encoding="utf-8",
            )
            verifier = mock.Mock()
            verifier.verify_receipt.return_value = 0
            with mock.patch.object(
                _STATUS, "RECOVERY_V2_NO_CONNECTOR_FORENSICS_PATH", summary
            ), mock.patch.object(
                _STATUS, "_load_status_module", return_value=verifier
            ):
                self.assertIsNone(_STATUS._canonical_no_connector_summary())
            verifier.verify_receipt.assert_called_once_with(root)

    def test_recovery_v2_top_level_status_has_no_positive_claim_when_evidence_unavailable(self):
        for gate, forensics in (
            (None, {"status": "DESCRIPTIVE_ONLY"}),
            ({"status": "VERIFIED_FAIL"}, None),
        ):
            with self.subTest(gate=gate is None), mock.patch.object(
                _STATUS, "_recovery_v2_lower1p25_gate", return_value=gate
            ), mock.patch.object(
                _STATUS, "_recovery_v2_no_connector_forensics", return_value=forensics
            ):
                plan = _STATUS._sim2real_72h()
            self.assertIn("TRACK B EVIDENCE UNAVAILABLE/MALFORMED", plan["status"])
            self.assertIn("NO TRACK B AUTHORITY", plan["status"])
            self.assertNotIn("ROUTE MECHANISM FAILED", plan["status"])

    def test_real_recovery_smoke_remains_a_research_stage(self):
        self.assertFalse(
            _STATUS._is_smoke_run("ppo_260801_1200_navrl_v2-recover-smoke-130bars-s1")
        )

    def test_ttc_run_is_rendered_as_fixed_density_ablation(self):
        update = _STATUS._v2_search_update(
            {
                "run": "ppo_260801_1200_navrl_v2-ttc-ttc-s1",
                "last_epoch": 5250,
                "last_n_bars_active": 70,
                "last_captured_rate": 0.80,
                "last_crash_rate": 0.20,
            },
            is_live=False,
        )
        experiment = update["active_experiment"]
        self.assertEqual(experiment["selector"], "ttc_sector")
        self.assertEqual(experiment["bars"], 70)
        self.assertFalse(experiment["density_curriculum"])
        self.assertIn("selector A/B", update["subtitle"])

    def test_main_ep24000_baseline_requires_heldout_before_ttc(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            _STATUS, "MAIN_TTC_RESULT_ROOT", Path(tmp)
        ):
            update = _STATUS._v2_search_update(
                {
                    "run": "ppo_260803_1819_navrl_v2-ep24000-205bars-main-baseline-s1",
                    "last_epoch": 25000,
                    "last_n_bars_active": 205,
                    "last_captured_rate": 2.0 / 3.0,
                    "last_crash_rate": 1.0 / 3.0,
                },
                is_live=False,
            )
        experiment = update["active_experiment"]
        self.assertTrue(experiment["ab_experiment"])
        self.assertEqual(experiment["ab_phase"], "BASELINE EVAL PENDING")
        self.assertEqual(experiment["adaptation_samples"], 4_096_000)
        self.assertFalse(experiment["heldout_complete"])
        self.assertIn("Do not start TTC", update["decision"])
        baseline_gate = next(
            gate for gate in update["gates"] if gate["label"] == "baseline held-out"
        )
        self.assertIn("TTC arm blocked", baseline_gate["value"])

    def test_main_ep24000_baseline_result_unlocks_ttc_with_canonical_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "navrl_v2_ep24000_ttc_main_baseline"
            result_dir.mkdir(parents=True)
            result = {
                "schema_version": 1,
                "requested_episodes": 2049,
                "actual_episodes": 2049,
                "checkpoint_sha256": _STATUS._MAIN_TTC_BASELINE_SHA256,
                "evaluated_checkpoint_snapshot_sha256": _STATUS._MAIN_TTC_BASELINE_SHA256,
                "condition": {
                    "bars": 205,
                    "seed": 42,
                    "action_selection": "deterministic",
                    "reflection_mode": "original",
                    "target_speed_mode": "uniform",
                },
                "v2_evaluation_contract": {
                    "obstacle_selector": "cluster_sector",
                    "runtime_profile": "main",
                },
                "outcome": {
                    "captured": 1424,
                    "crash": 594,
                    "timeout": 31,
                    "capture_rate": 1424 / 2049,
                    "crash_rate": 594 / 2049,
                    "timeout_rate": 31 / 2049,
                },
                "crash_causes": {"bar_contact": 570},
            }
            self._write_json(result_dir / "205bars.json", result)
            with mock.patch.object(_STATUS, "MAIN_TTC_RESULT_ROOT", root):
                update = _STATUS._v2_search_update(
                    {
                        "run": "ppo_260803_1819_navrl_v2-ep24000-205bars-main-baseline-s1",
                        "last_epoch": 25000,
                        "last_n_bars_active": 205,
                    },
                    is_live=False,
                )
        experiment = update["active_experiment"]
        self.assertEqual(experiment["ab_phase"], "TTC ARM READY")
        self.assertTrue(experiment["heldout_complete"])
        self.assertIn("may now start", update["headline"])
        floor = next(
            gate
            for gate in update["gates"]
            if gate["label"] == "canonical replacement floor"
        )
        self.assertIn("72.44%", floor["value"])

    def test_completed_main_ttc_rejects_bundle_and_discloses_fov_confound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_dir = root / "navrl_v2_ep24000_ttc_main_baseline"
            ttc_dir = root / "navrl_v2_ep24000_ttc_main_ttc"
            baseline_sha = _STATUS._MAIN_TTC_BASELINE_SHA256
            baseline = {
                "schema_version": 1,
                "requested_episodes": 2049,
                "actual_episodes": 2049,
                "checkpoint_sha256": baseline_sha,
                "evaluated_checkpoint_snapshot_sha256": baseline_sha,
                "condition": {
                    "bars": 205,
                    "seed": 42,
                    "action_selection": "deterministic",
                    "reflection_mode": "original",
                    "target_speed_mode": "uniform",
                },
                "v2_evaluation_contract": {
                    "obstacle_selector": "cluster_sector",
                    "runtime_profile": "main",
                },
                "outcome": {
                    "captured": 1424,
                    "crash": 594,
                    "timeout": 31,
                    "capture_rate": 1424 / 2049,
                    "crash_rate": 594 / 2049,
                    "timeout_rate": 31 / 2049,
                },
                "crash_causes": {"bar_contact": 570},
            }
            ttc_sha = "a" * 64
            ttc = {
                "schema_version": 1,
                "requested_episodes": 2049,
                "actual_episodes": 2051,
                "checkpoint_sha256": ttc_sha,
                "evaluated_checkpoint_snapshot_sha256": ttc_sha,
                "condition": {
                    "bars": 205,
                    "seed": 42,
                    "action_selection": "deterministic",
                    "reflection_mode": "original",
                    "target_speed_mode": "uniform",
                },
                "v2_evaluation_contract": {
                    "obstacle_selector": "ttc_sector",
                    "runtime_profile": "main",
                },
                "outcome": {
                    "captured": 1440,
                    "crash": 605,
                    "timeout": 6,
                    "capture_rate": 1440 / 2051,
                    "crash_rate": 605 / 2051,
                    "timeout_rate": 6 / 2051,
                },
                "crash_causes": {"bar_contact": 578},
            }
            self._write_json(baseline_dir / "205bars.json", baseline)
            self._write_json(ttc_dir / "205bars.json", ttc)
            with mock.patch.object(_STATUS, "MAIN_TTC_RESULT_ROOT", root):
                update = _STATUS._v2_search_update(
                    {
                        "run": "ppo_260804_0813_navrl_v2-ep24000-205bars-main-ttc-s1",
                        "last_epoch": 25000,
                        "last_n_bars_active": 205,
                    },
                    is_live=False,
                )

        experiment = update["active_experiment"]
        self.assertEqual(experiment["ab_phase"], "TTC REJECT")
        self.assertTrue(experiment["ab_gate_complete"])
        self.assertFalse(experiment["ab_gate_pass"])
        self.assertTrue(experiment["representation_bundle"])
        self.assertEqual(experiment["baseline_effective_fov_deg"], 240)
        self.assertEqual(experiment["ttc_effective_fov_deg"], 360)
        self.assertFalse(experiment["pure_ranking_isolated"])
        self.assertEqual(experiment["final_checkpoint_sha256"], ttc_sha)
        self.assertIn("failed both", update["headline"])
        self.assertIn("TTC-240", update["decision"])
        current_arm = next(
            milestone for milestone in update["milestones"]
            if milestone["label"] == "CURRENT ARM"
        )
        self.assertEqual(current_arm["state"], "warn")
        isolation = next(
            gate for gate in update["gates"]
            if gate["label"] == "experimental isolation"
        )
        self.assertIn("240→360", isolation["value"])

    def test_completed_riskcap_uses_final_heldout_not_training_tail(self):
        update = _STATUS._v2_search_update(
            {
                "run": "ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1",
                "last_epoch": 25000,
                "last_n_bars_active": 205,
                "last_captured_rate": 0.75,
                "last_crash_rate": 0.25,
            },
            is_live=False,
        )
        experiment = update["active_experiment"]
        self.assertEqual(experiment["stage_status"], "FINAL PASS")
        self.assertTrue(experiment["post_evaluation_complete"])
        self.assertTrue(experiment["generalization_pass"])
        self.assertEqual(
            experiment["winner_checkpoint_sha256"], _STATUS._RISKCAP_TRAINED_SHA256
        )
        self.assertAlmostEqual(experiment["heldout_capture"], 0.8194241093)
        self.assertAlmostEqual(experiment["heldout_crash"], 0.1566617862)
        self.assertEqual(experiment["heldout_episodes"], 2049)
        self.assertIn("+3.75 pp", update["summary"])
        self.assertIn("Do not extend", update["decision"])
        self.assertEqual(len(update["comparison"]), 6)

    def test_final_v2_snapshot_consumes_completed_causal_artifacts(self):
        update = _STATUS._v2_search_update(
            {
                "run": "ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1",
                "last_epoch": 24010,
                "last_n_bars_active": 205,
            },
            is_live=False,
        )
        experiment = update["active_experiment"]
        self.assertTrue(experiment["causal_checks_complete"])
        self.assertTrue(experiment["fixed_speed_complete"])
        self.assertTrue(experiment["forgetting_complete"])
        self.assertFalse(experiment["causal_checks_pending"])
        self.assertTrue(experiment["ttc_1650_gate_pass"])
        self.assertTrue(experiment["next_training_authorized"])
        self.assertIn("causal audit complete", update["subtitle"])
        self.assertIn("READY", next(
            gate["value"] for gate in update["gates"]
            if gate["label"] == "next training"
        ))

    def test_recovery_attestation_requires_real_matching_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self._make_recovery_artifacts(Path(tmp))
            self.assertEqual(
                _STATUS._validate_recovery_attestation(artifacts["run_dir"]), []
            )
            with mock.patch.object(_STATUS, "RUNS_ROOT", artifacts["runs_root"]):
                update = _STATUS._v2_search_update(
                    {
                        "run": artifacts["run_name"],
                        "last_epoch": 9600,
                        "last_n_bars_active": 130,
                    },
                    is_live=False,
                )
            self.assertTrue(update["active_experiment"]["recovery_attestation_valid"])
            self.assertIn("passed this snapshot's verification", update["headline"])
            gate = next(
                item
                for item in update["gates"]
                if item["label"] == "held-out 130-bar attestation"
            )
            self.assertEqual(gate["value"], "PASS")

    def test_recovery_attestation_rejects_handwritten_pass_without_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = self._make_recovery_artifacts(
                Path(tmp), canonical=False
            )
            errors = _STATUS._validate_recovery_attestation(artifacts["run_dir"])
            self.assertTrue(errors)
            self.assertTrue(
                any(error.startswith("canonical verifier:") for error in errors),
                errors,
            )
            with mock.patch.object(_STATUS, "RUNS_ROOT", artifacts["runs_root"]):
                update = _STATUS._v2_search_update(
                    {
                        "run": artifacts["run_name"],
                        "last_epoch": 9600,
                        "last_n_bars_active": 130,
                    },
                    is_live=False,
                )
            self.assertFalse(
                update["active_experiment"]["recovery_attestation_valid"]
            )
            self.assertIn("curriculum remains blocked", update["headline"])

    def test_recovery_attestation_rejects_mutated_artifact_contract_and_rollback(self):
        mutations = ("result", "checkpoint", "rollback")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                artifacts = self._make_recovery_artifacts(Path(tmp))
                attestation = json.loads(
                    artifacts["attestation_path"].read_text(encoding="utf-8")
                )
                if mutation == "result":
                    result = json.loads(
                        artifacts["result_path"].read_text(encoding="utf-8")
                    )
                    result["condition"]["seed"] = 1
                    self._write_json(artifacts["result_path"], result)
                    # Even if a tamperer updates the quoted digest, the canonical seed contract fails.
                    attestation["heldout_result_sha256"] = self._digest(
                        artifacts["result_path"]
                    )
                elif mutation == "checkpoint":
                    checkpoint = torch.load(
                        artifacts["checkpoint_path"],
                        map_location="cpu",
                        weights_only=False,
                    )
                    checkpoint["env_state"]["cfg_oob_margin"] = 0.5
                    torch.save(checkpoint, artifacts["checkpoint_path"])
                    # A rewritten attestation digest must not hide changed checkpoint semantics.
                    attestation["checkpoint_sha256"] = self._digest(
                        artifacts["checkpoint_path"]
                    )
                else:
                    attestation["max_rollback_streak"] = 1.0
                self._write_json(artifacts["attestation_path"], attestation)

                errors = _STATUS._validate_recovery_attestation(artifacts["run_dir"])
                self.assertTrue(errors)
                with mock.patch.object(_STATUS, "RUNS_ROOT", artifacts["runs_root"]):
                    update = _STATUS._v2_search_update(
                        {
                            "run": artifacts["run_name"],
                            "last_epoch": 9600,
                            "last_n_bars_active": 130,
                        },
                        is_live=False,
                    )
                self.assertFalse(
                    update["active_experiment"]["recovery_attestation_valid"]
                )
                self.assertIn("curriculum remains blocked", update["headline"])


if __name__ == "__main__":
    unittest.main()
