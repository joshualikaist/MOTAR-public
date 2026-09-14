import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


_ROOT = Path(__file__).parents[1]
_PATH = _ROOT / "tools/evaluate_corridor_gate.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_corridor_gate", _PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


class CorridorGateTest(unittest.TestCase):
    def _csv(self, capture, bar):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", newline="", delete=False)
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        writer = csv.writer(tmp)
        writer.writerow(
            [
                "bars",
                "target_speed",
                "pursuer_limit",
                "episodes",
                "capture_rate",
                "crash_rate",
                "timeout_rate",
                "bar_contact_rate",
                "below_rate",
                "json",
            ]
        )
        for speed in (0.0, 0.5, 1.0, 1.5):
            writer.writerow([100, speed, 2.5, 1000, capture, 1 - capture, 0, bar, 0, "x"])
        tmp.close()
        return Path(tmp.name)

    def test_all_three_preregistered_gates_are_required(self):
        result = _MOD.evaluate(self._csv(0.681, 0.30))
        self.assertEqual(result["verdict"], "PASS")
        self.assertTrue(all(result["checks"].values()))

        result = _MOD.evaluate(self._csv(0.679, 0.30))
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["checks"]["capture_at_least_68pct"])

        result = _MOD.evaluate(self._csv(0.681, 0.34))
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["checks"]["bar_contact_below_baseline"])

    def test_wrong_grid_is_rejected(self):
        path = self._csv(0.70, 0.25)
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        rows[-1][1] = "1.0"
        with path.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        with self.assertRaisesRegex(ValueError, "four-speed"):
            _MOD.evaluate(path)


if __name__ == "__main__":
    unittest.main()
