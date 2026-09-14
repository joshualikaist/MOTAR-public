import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _ROOT / "tools/update_status_snapshot.py"
_SPEC = importlib.util.spec_from_file_location("status_completion_test", _MODULE_PATH)
_STATUS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_STATUS)


class StatusRecoveryCompletionTest(unittest.TestCase):
    RUN_NAME = "ppo_260801_1200_navrl_v2-recover-smoke-130bars-s1"

    def _render(self, root, marker=None, epoch=9600):
        run_dir = root / self.RUN_NAME
        run_dir.mkdir(parents=True)
        if marker is not None:
            (run_dir / ".aerial_training_finished").write_text(marker, encoding="utf-8")
        with mock.patch.object(_STATUS, "RUNS_ROOT", root), mock.patch.object(
            _STATUS, "_validate_recovery_attestation", return_value=["attestation: missing"]
        ):
            return _STATUS._v2_search_update(
                {
                    "run": self.RUN_NAME,
                    "last_epoch": epoch,
                    "last_n_bars_active": 130,
                },
                is_live=False,
            )

    def test_epoch_budget_without_marker_is_abnormal_and_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            update = self._render(Path(tmp))
        experiment = update["active_experiment"]
        self.assertFalse(experiment["recovery_normal_completion_marker_valid"])
        self.assertIn("without the exact normal-completion marker", update["headline"])
        self.assertIn("Do not evaluate or resume", update["decision"])
        self.assertEqual(update["milestones"][0]["state"], "warn")

    def test_only_exact_epoch_9600_marker_completes_smoke(self):
        for marker in ("epoch=9599\n", "epoch=9600\nextra=1\n", "finished epoch=9600\n"):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as tmp:
                update = self._render(Path(tmp), marker)
                self.assertFalse(
                    update["active_experiment"]["recovery_normal_completion_marker_valid"]
                )
        with tempfile.TemporaryDirectory() as tmp:
            update = self._render(Path(tmp), "epoch=9600\n")
        self.assertTrue(update["active_experiment"]["recovery_normal_completion_marker_valid"])
        gate = next(g for g in update["gates"] if g["label"].startswith("100 recovery"))
        self.assertEqual(gate["value"], "PASS · marker epoch=9600")

    def test_epoch_past_9600_is_not_exact_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            update = self._render(Path(tmp), "epoch=9600\n", epoch=9601)
        self.assertFalse(update["active_experiment"]["recovery_normal_completion_marker_valid"])
        self.assertIn("curriculum remains blocked", update["headline"])


if __name__ == "__main__":
    unittest.main()
