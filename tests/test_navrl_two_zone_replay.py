import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_PATH = _ROOT / "tools/navrl_two_zone_replay.py"
_SPEC = importlib.util.spec_from_file_location("navrl_two_zone_replay_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class TwoZoneReplayTests(unittest.TestCase):
    def _files(self, *, far_valid=False, invalid_payload=False):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        contract = root / "contract.json"
        contract.write_text(
            json.dumps({
                "schema_version": 1,
                "source_kind": "real_log",
                "near_boundary_m": 12.0,
                "far_range_policy": "invalid",
            }),
            encoding="utf-8",
        )
        replay = root / "replay.jsonl"
        valid = not invalid_payload
        near_valid = not invalid_payload
        event = {
            "trial_id": "t0",
            "timestamp_ns": 100,
            "zone": "far",
            "azimuth_deg": 2.0,
            "elevation_deg": 0.0,
            "bearing_rate_dps": 1.0,
            "confidence": 0.8,
            "range_valid": far_valid,
            "range_m": 10.0 if far_valid else None,
            "range_sigma_m": 0.2 if far_valid else None,
            "measurement_age_ms": 3.0,
            "track_covariance": 0.1,
        }
        replay.write_text(json.dumps(event) + "\n", encoding="utf-8")
        return replay, contract

    def test_far_range_is_forced_invalid(self):
        replay, contract = self._files()
        report = _MODULE.validate_replay(replay, contract)
        self.assertEqual(report["verdict"], "PASS")

    def test_far_valid_range_fails_closed(self):
        replay, contract = self._files(far_valid=True)
        report = _MODULE.validate_replay(replay, contract)
        self.assertEqual(report["verdict"], "FAIL")
        self.assertIn("FAR_RANGE_MUST_BE_INVALID", {item["code"] for item in report["issues"]})

    def test_invalid_range_payload_fails_closed(self):
        replay, contract = self._files(invalid_payload=True)
        report = _MODULE.validate_replay(replay, contract)
        self.assertEqual(report["verdict"], "PASS")

    def test_non_real_contract_is_rejected(self):
        replay, contract = self._files()
        payload = json.loads(contract.read_text())
        payload["source_kind"] = "synthetic_fixture"
        contract.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(_MODULE.ReplayError):
            _MODULE.validate_replay(replay, contract)

    def test_nanosecond_timestamp_is_not_round_tripped_through_float(self):
        replay, contract = self._files()
        payload = json.loads(replay.read_text().splitlines()[0])
        payload["timestamp_ns"] = "1700000000123456789"
        replay.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        report = _MODULE.validate_replay(replay, contract)
        self.assertEqual(report["verdict"], "PASS")


if __name__ == "__main__":
    unittest.main()
