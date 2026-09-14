import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_PATH = _ROOT / "tools/navrl_sensor_profile.py"
_SPEC = importlib.util.spec_from_file_location("navrl_sensor_profile_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class SensorProfileTests(unittest.TestCase):
    def _write(self, path, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_MODULE.REQUIRED_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _row(self, seq, *, trial="t0", detected=True, range_valid=True, target_present=True):
        return {
            "trial_id": trial,
            "distance_m": "12",
            "lighting": "normal",
            "motion": "static",
            "target_present": int(target_present),
            "detected": int(detected),
            "range_valid": int(range_valid),
            "ground_truth_azimuth_deg": "2.0" if target_present else "",
            "estimated_azimuth_deg": "2.5" if detected else "",
            "ground_truth_range_m": "12.0" if target_present else "",
            "estimated_range_m": "12.2" if range_valid else "",
            "confidence": "0.9" if detected else "0.0",
            "source_stamp_ns": str(seq * 100),
            "host_receive_stamp_ns": str(seq * 100 + 10),
        }

    def test_profile_is_trial_level_and_does_not_claim_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "measurements.csv"
            output = root / "profile.json"
            self._write(source, [self._row(0), self._row(1), self._row(0, trial="t1", detected=False, range_valid=False)])
            profile = _MODULE.build_profile(source, output, source_kind="real_log", run_id="r")
            self.assertEqual(profile["claim_status"], "MEASURED_CANDIDATE")
            self.assertEqual(profile["profile"]["trial_count"], 2)
            self.assertEqual(profile["quality"]["threshold_decision"], "NOT_APPLIED")
            self.assertEqual(profile["profile"]["cells"][0]["trial_count"], 2)

    def test_detected_without_bearing_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad.csv"
            row = self._row(0)
            row["estimated_azimuth_deg"] = ""
            self._write(source, [row])
            with self.assertRaises(_MODULE.ProfileError):
                _MODULE.read_measurements(source)

    def test_timestamp_regression_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bad.csv"
            first = self._row(1)
            second = self._row(0)
            self._write(source, [first, second])
            with self.assertRaises(_MODULE.ProfileError):
                _MODULE.read_measurements(source)


if __name__ == "__main__":
    unittest.main()
