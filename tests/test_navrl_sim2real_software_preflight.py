import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools/run_navrl_sim2real_software_preflight.py"
SPEC = importlib.util.spec_from_file_location("navrl_sim2real_software_preflight_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SoftwarePreflightTests(unittest.TestCase):
    def test_synthetic_chain_passes_but_keeps_training_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = MODULE.run(Path(directory) / "preflight")
        self.assertEqual(payload["claim_status"], "SYNTHETIC_ONLY")
        self.assertEqual(payload["steps"]["telemetry"]["verdict"], "PASS")
        self.assertEqual(payload["steps"]["ingest"]["verdict"], "PASS")
        self.assertEqual(payload["steps"]["sensor_profile"]["verdict"], "PASS")
        self.assertEqual(payload["steps"]["two_zone_replay"]["verdict"], "PASS")
        self.assertEqual(payload["steps"]["fresh_ppo"]["verdict"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
