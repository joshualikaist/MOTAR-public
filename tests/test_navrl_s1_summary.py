import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "s1_summary_test", ROOT / "tools/summarize_s1_search_state.py"
)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class S1SummaryTest(unittest.TestCase):
    @staticmethod
    def write_cell(root, arm, bars, never, *, masked=False, median=20):
        directory = root / (arm + ("_masked" if masked else ""))
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / f"{bars}bars.json"
        receipt_path = directory / f"{bars}bars.receipt.json"
        n = 2049
        crash = never + 150
        outcome = {"captured": n - crash - 9, "crash": crash, "timeout": 9}
        crash_never = max(0, never - 2)
        first = {
            "capture": {"never_acquired": 0, "first_visible_step_median": median},
            "crash": {"never_acquired": crash_never, "first_visible_step_median": 30},
            "timeout": {"never_acquired": never - crash_never, "first_visible_step_median": 40},
        }
        payload = {
            "requested_episodes": n,
            "actual_episodes": n,
            "condition": {
                "seed": 331,
                "bars": bars,
                "action_selection": "deterministic",
                "search_state": arm,
                "search_state_masked": masked,
                "search_state_telemetry": True,
            },
            "outcome": outcome,
            "target_motion": {
                "first_acquisition": first,
                "outcome_telemetry": {
                    label: {"visible_fraction_step_weighted": 0.5}
                    for label in ("capture", "crash", "timeout")
                },
            },
            "search_state": {
                "blind_phase_mean_speed_mps": 2.0,
                "blind_phase_bar_clearance_mean_m": 1.0,
                "first_visible": {},
            },
        }
        result_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
        receipt_path.write_text(
            json.dumps(
                {
                    "result_sha256": digest,
                    "runtime_git_dirty": False,
                    "seed": 331,
                    "bars": bars,
                    "search_state": arm,
                    "search_state_force_invalid": masked,
                    "search_state_telemetry": True,
                }
            ) + "\n",
            encoding="utf-8",
        )

    def test_gate_mask_mechanism_and_output_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            never_by_arm = {"off": 175, "geofence": 165, "coverage": 100, "belief": 80}
            for arm, never in never_by_arm.items():
                for bars in (70, 145):
                    self.write_cell(root, arm, bars, never, median=20 if arm == "off" else 19)
            for arm, never in (("coverage", 150), ("belief", 150)):
                for bars in (70, 145):
                    self.write_cell(root, arm, bars, never, masked=True, median=19)

            payload = SUMMARY.build_summary(root)
            self.assertEqual(payload["decisions"]["geofence"]["outcome"], "FAIL")
            self.assertEqual(payload["decisions"]["coverage"]["outcome"], "PASS_MECHANISM")
            self.assertEqual(payload["decisions"]["belief"]["outcome"], "PASS_MECHANISM")
            SUMMARY.write_outputs(root, payload)
            self.assertTrue((root / "summary.json").is_file())
            self.assertTrue((root / "summary.md").is_file())
            digest_lines = (root / "summary.sha256").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(digest_lines), 2)


if __name__ == "__main__":
    unittest.main()
