import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/analyze_navrl_v2_postrun.py"
_SPEC = importlib.util.spec_from_file_location("navrl_v2_postrun", _MODULE_PATH)
_POST = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_POST)


class NavRLV2PostrunTest(unittest.TestCase):
    @staticmethod
    def _write_csv(root, run, epochs, bars):
        path = root / run / "aerial_run/epoch_metrics.csv"
        path.parent.mkdir(parents=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "epoch",
                    "captured_rate",
                    "crash_rate",
                    "timeout_rate",
                    "mean_reward",
                    "n_bars_active",
                ),
            )
            writer.writeheader()
            for epoch in epochs:
                writer.writerow(
                    {
                        "epoch": epoch,
                        "captured_rate": 0.7,
                        "crash_rate": 0.3,
                        "timeout_rate": 0.0,
                        "mean_reward": 1.0,
                        "n_bars_active": bars,
                    }
                )

    def test_canonical_lineage_excludes_rolled_back_overlap(self):
        segments = (("smoke", 1, 2), ("old", 3, 4), ("continue", 5, None))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_csv(root, "smoke", [1, 2], 130)
            self._write_csv(root, "old", [3, 4, 5, 6], 145)
            self._write_csv(root, "continue", [5, 6, 7], 145)
            with mock.patch.object(_POST, "RUNS_ROOT", root):
                rows, sources = _POST.canonical_rows(segments)
        self.assertEqual([row["epoch"] for row in rows], list(range(1, 8)))
        self.assertEqual(sources[1]["actual_last"], 4)

    def test_log_parser_reads_hold_and_promotion_contract(self):
        text = "\n".join(
            (
                "run demo-run",
                "epoch          : 200 / 300",
                "density bars         : 205",
                "NavRL density curriculum held | bars=205 capture=0.694 over 16385 eps (threshold=0.700)",
                "NavRL density curriculum promoted | bars 190 -> 205 after 16384 eps, capture=0.704 (threshold=0.700) dwell=1000 epochs",
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "demo.log").write_text(text, encoding="utf-8")
            with mock.patch.object(_POST, "LOG_ROOT", root), mock.patch.object(
                _POST, "RL_ROOT", root
            ):
                parsed = _POST.parse_logs(["demo-run"])
        self.assertEqual([gate["result"] for gate in parsed["gates"]], ["held", "promoted"])
        self.assertEqual(parsed["gates"][0]["capture"], 0.694)
        self.assertEqual(parsed["gates"][1]["next_bars"], 205)


if __name__ == "__main__":
    unittest.main()
