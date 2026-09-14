"""CPU-only: host-absolute lineage paths must not fail cross-machine verify."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools/run_navrl_distractor_envelope.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("distractor_envelope_verify_paths", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DistractorEnvelopeVerifyPathTest(unittest.TestCase):
    def test_verify_payload_drops_host_absolute_artifact_path(self):
        module = _load_launcher()
        recorded = {
            "lineage_reference": {
                "source": module.LINEAGE_RESULT_REL,
                "source_sha256": module.LINEAGE_RESULT_SHA,
                "artifact_path": "/home/joshuali/MOTAR/" + module.LINEAGE_RESULT_REL,
                "outcome": {"captured": 1},
            }
        }
        expected = {
            "lineage_reference": {
                "source": module.LINEAGE_RESULT_REL,
                "source_sha256": module.LINEAGE_RESULT_SHA,
                "artifact_path": str(ROOT / module.LINEAGE_RESULT_REL),
                "outcome": {"captured": 1},
            }
        }
        self.assertEqual(module._verify_payload(recorded), module._verify_payload(expected))
        self.assertNotIn("artifact_path", module._verify_payload(recorded)["lineage_reference"])


if __name__ == "__main__":
    unittest.main()
