from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "curriculum_analyzer",
    ROOT / "tools/analyze_navrl_corrected_nonoverlap_physical_curriculum.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch", "mean_reward", "captured_rate", "crash_rate",
                "timeout_rate", "n_bars_active",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _dashboard(epoch: int, captured: int, crash: int, timeout: int, bars: int) -> str:
    n = captured + crash + timeout
    return (
        f"  epoch          : {epoch} / 30000\n"
        f"  captured (success)   :  {100.0 * captured / n:4.1f}% ({captured}/{n})\n"
        f"  crash                :  {100.0 * crash / n:4.1f}% ({crash}/{n})\n"
        f"  timeout (no capture) :  {100.0 * timeout / n:4.1f}% ({timeout}/{n})\n"
        f"  density bars         : {bars}\n"
    )


class CurriculumAnalyzerTest(unittest.TestCase):
    def test_live_status_does_not_emit_verdict_or_routed_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "ppo_live"
            log = Path(directory) / "train.log"
            _write_csv(
                run / "aerial_run/epoch_metrics.csv",
                [
                    {
                        "epoch": str(epoch),
                        "mean_reward": str(80 + epoch),
                        "captured_rate": "0.76",
                        "crash_rate": "0.24",
                        "timeout_rate": "0.0",
                        "n_bars_active": "70",
                    }
                    for epoch in range(1, 6)
                ],
            )
            log.write_text(
                "NavRL density curriculum held | bars=70 capture=0.760 over 16384 eps "
                "(threshold=0.820)\n"
                + _dashboard(5, 24, 8, 0, 70),
                encoding="utf-8",
            )
            result = MODULE.live_status(run, log)
            self.assertEqual(result["status"], "LIVE_RUNNING")
            self.assertIsNone(result["verdict"])
            self.assertEqual(result["epoch"], 5)
            self.assertEqual(result["n_bars_active"], 70)
            self.assertFalse(result["authority"]["routed_ppo"])
            self.assertFalse(result["authority"]["held_out_eval_preregistration"])
            self.assertFalse(result["authority"]["resume_or_warm_start"])
            self.assertEqual(result["latest_hold"]["threshold"], 0.82)

    def test_live_mode_rejects_a_finished_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "ppo_done"
            run.mkdir()
            (run / ".aerial_training_finished").write_text("epoch=30000\n", encoding="utf-8")
            log = Path(directory) / "train.log"
            log.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "official analyze"):
                MODULE.live_status(run, log)

    def test_promotion_parser_and_fail_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text(
                "NavRL density curriculum promoted | bars 70 -> 85 after 16384 eps, "
                "capture=0.821 (threshold=0.820) dwell=1000 epochs\n"
                "[aerial RL] FAIL-STOP: same-density capture collapse — drop 0.30\n",
                encoding="utf-8",
            )
            events = MODULE.parse_density_events(log)
            self.assertEqual(events["promotions"][0]["dst"], 85)
            self.assertTrue(events["fail_stop"])

    def test_official_analyze_without_summary_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "ppo_empty"
            run.mkdir()
            log = Path(directory) / "train.log"
            log.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "run_summary.json missing"):
                MODULE.analyze(run, log)

    def test_authority_helper_never_opens_routed_ppo(self):
        opened = MODULE._authority(True)
        self.assertTrue(opened["held_out_eval_preregistration"])
        self.assertFalse(opened["routed_ppo"])
        self.assertFalse(opened["hardware_or_sim_to_real_claim"])
        self.assertFalse(opened["resume_or_warm_start"])
        self.assertFalse(json.loads(json.dumps(opened))["parameter_search"])


if __name__ == "__main__":
    unittest.main()
