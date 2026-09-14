import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "navrl_sim2real_telemetry.py"
SPEC = importlib.util.spec_from_file_location("navrl_sim2real_telemetry", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _events():
    path = Path(__file__).with_name("_telemetry_fixture.jsonl")
    MODULE.write_jsonl_fixture(path, groups=4)
    try:
        manifest, events = MODULE.read_jsonl(path)
    finally:
        path.unlink(missing_ok=True)
    return manifest, events


class TelemetryContractTests(unittest.TestCase):
    def test_synthetic_fixture_passes_but_is_not_real_evidence(self):
        manifest, events = _events()
        report = MODULE.validate_events(
            events,
            manifest=manifest,
            contract=MODULE.TelemetryContract(
                max_sync_skew_ns=20_000_000,
                max_sensor_to_host_latency_ns=10_000_000,
            ),
        )
        self.assertEqual(report.verdict, "PASS")
        self.assertEqual(report.claim_status, "SYNTHETIC_ONLY")
        self.assertAlmostEqual(report.metrics["sync_skew"]["p95_ms"], 2.0)
        self.assertAlmostEqual(report.metrics["source_to_host_latency"]["camera"]["p50_ms"], 1.0)


    def test_frame_mismatch_fails_closed(self):
        manifest, events = _events()
        bad = list(events)
        bad[0] = MODULE.TelemetryEvent(
            **{**bad[0].__dict__, "frame_id": "wrong_frame"}
        )
        report = MODULE.validate_events(bad, manifest=manifest)
        self.assertEqual(report.verdict, "FAIL")
        self.assertTrue(any(issue["code"] == "FRAME_CONTRACT_MISMATCH" for issue in report.issues))


    def test_timestamp_regression_and_sequence_gap_are_reported(self):
        manifest, events = _events()
        bad = list(events)
        camera = [index for index, event in enumerate(bad) if event.topic == "camera"]
        first, second = camera[0], camera[1]
        bad[second] = MODULE.TelemetryEvent(
            **{
                **bad[second].__dict__,
                "seq": bad[first].seq + 2,
                "source_stamp_ns": bad[first].source_stamp_ns - 1,
            }
        )
        report = MODULE.validate_events(bad, manifest=manifest)
        codes = {issue["code"] for issue in report.issues}
        self.assertIn("SOURCE_TIMESTAMP_REGRESSION", codes)
        self.assertEqual(report.metrics["sequence_gaps"]["camera"], 1)


    def test_missing_topic_is_not_treated_as_zero_measurement(self):
        manifest, events = _events()
        reduced = [event for event in events if event.topic != "lidar"]
        report = MODULE.validate_events(reduced, manifest=manifest)
        self.assertEqual(report.verdict, "FAIL")
        self.assertTrue(any(issue["code"] == "MISSING_TOPIC" and issue["details"]["topic"] == "lidar" for issue in report.issues))

    def test_unknown_topic_and_missing_manifest_fail_closed(self):
        manifest, events = _events()
        unknown = MODULE.TelemetryEvent(
            topic="raw_camera_debug",
            seq=0,
            source_stamp_ns=0,
            host_receive_stamp_ns=0,
            frame_id="camera_optical_frame",
            parent_frame_id="base_link",
            row_number=999,
        )
        report = MODULE.validate_events([*events, unknown], manifest=manifest)
        self.assertEqual(report.verdict, "FAIL")
        self.assertTrue(any(issue["code"] == "UNKNOWN_TOPIC" for issue in report.issues))
        no_manifest = MODULE.validate_events(events)
        self.assertTrue(any(issue["code"] == "MISSING_MANIFEST" for issue in no_manifest.issues))
