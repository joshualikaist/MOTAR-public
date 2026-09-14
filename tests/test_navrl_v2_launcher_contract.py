import os
from pathlib import Path
import subprocess
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_LAUNCHER = _ROOT / "aerial_gym/rl_training/rl_games/train_navrl_v2_search.sh"


class NavRLV2LauncherContractTest(unittest.TestCase):
    def _run(self, args, **updates):
        env = os.environ.copy()
        for name in ("CKPT", "NAVRL_V2_ALLOW_RESUME", "NAVRL_V2_PROFILE"):
            env.pop(name, None)
        env.update(updates)
        return subprocess.run(
            [str(_LAUNCHER), *args],
            cwd=_LAUNCHER.parent,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_fresh_mode_rejects_every_cli_argument(self):
        completed = self._run(["--file", "hostile.yaml"])
        self.assertEqual(completed.returncode, 2, completed.stdout)
        self.assertIn("fresh mode accepts no CLI arguments", completed.stdout)

    def test_continuation_rejects_extra_or_mismatched_arguments(self):
        checkpoint = "/tmp/expected-v2-checkpoint.pth"
        for args in (
            ["--checkpoint", checkpoint],
            ["--checkpoint", "/tmp/other.pth", "--branch_run"],
            ["--checkpoint", checkpoint, "--branch_run", "--seed", "7"],
        ):
            with self.subTest(args=args):
                completed = self._run(
                    args,
                    NAVRL_V2_ALLOW_RESUME="1",
                    CKPT=checkpoint,
                )
                self.assertEqual(completed.returncode, 2, completed.stdout)
                self.assertIn("continuation", completed.stdout)

    def test_exact_continuation_tuple_passes_cli_gate(self):
        checkpoint = "/tmp/expected-v2-checkpoint.pth"
        # An intentionally invalid profile stops immediately after the CLI gate.  Seeing the
        # profile error proves the canonical tuple passed without ever starting a trainer.
        completed = self._run(
            ["--checkpoint", checkpoint, "--branch_run"],
            NAVRL_V2_ALLOW_RESUME="1",
            CKPT=checkpoint,
            NAVRL_V2_PROFILE="test-invalid-profile",
        )
        self.assertEqual(completed.returncode, 2, completed.stdout)
        self.assertIn("NAVRL_V2_PROFILE must be main or 4gb", completed.stdout)
        self.assertNotIn("non-canonical continuation", completed.stdout)

    def test_exact_continuation_handoff_reaches_child_preflight(self):
        checkpoint = "/tmp/expected-v2-checkpoint.pth"
        completed = self._run(
            ["--checkpoint", checkpoint, "--branch_run"],
            NAVRL_V2_ALLOW_RESUME="1",
            NAVRL_V2_CONTRACT_PREFLIGHT_ONLY="1",
            CKPT=checkpoint,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("CONTINUATION", completed.stdout)
        self.assertIn("child handoff validated", completed.stdout)

    def test_training_entry_points_clear_evaluator_only_contract(self):
        for relative in (
            "train_navrl_v2_search.sh",
            "train_navrl_v2_recover_safe.sh",
            "train_navrl_v2_ttc_ab.sh",
        ):
            source = (_LAUNCHER.parent / relative).read_text(encoding="utf-8")
            self.assertIn(
                "unset NAVRL_EVAL_RUN_NONCE NAVRL_EVAL_PROFILE NAVRL_SIM_PHYSICS_CONTRACT",
                source,
                relative,
            )


if __name__ == "__main__":
    unittest.main()
