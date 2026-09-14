"""Every receipt that carries GPU numbers must say which stack produced them.

On 2026-09-08 a candidate cache built under torch 2.10 / cuDNN 9.10.2 could not be reproduced by a
re-run under torch 2.4 / cuDNN 9.1. The two stacks pick different convolution kernels and
confidences diverged by up to 45%, but no receipt recorded the interpreter or the accelerator
libraries, so the mismatch was chased through TF32 and determinism flags that could never explain
it. These tests keep the fingerprint in the receipts and keep the documented environment single.

Run: PYTHONNOUSERSITE=1 python tests/test_runtime_fingerprint.py
"""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("runtime_fingerprint", ROOT / "tools/runtime_fingerprint.py")
FP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FP)

# Tools whose receipts describe numbers that came off a GPU.
GPU_RECEIPT_TOOLS = (
    "tools/run_perception_candidate_producer.py",
    "tools/build_perception_motion_features.py",
    "tools/eval_detfly_zeroshot.py",
    "tools/prepare_detfly_dataset.py",
)
STREAMING_ENV = "detector_runs/venv"
RETIRED_ENV = "datasets/detenv"


class FingerprintContentTest(unittest.TestCase):
    def test_records_the_fields_that_decide_kernel_selection(self):
        record = FP.runtime_fingerprint()
        for field in ("python", "executable", "platform"):
            self.assertIn(field, record)
        if record.get("torch") is not None:
            # These are exactly what differed between the two environments in the incident.
            for field in ("torch", "cuda", "cudnn", "cudnn_allow_tf32",
                          "cudnn_benchmark", "matmul_allow_tf32"):
                self.assertIn(field, record, field)

    def test_it_is_json_serialisable(self):
        import json
        json.dumps(FP.runtime_fingerprint())

    def test_differences_names_the_incident_fields(self):
        left = {"torch": "2.10.0+cu128", "cudnn": 91002, "python": "3.10.19"}
        right = {"torch": "2.4.1+cu121", "cudnn": 90100, "python": "3.10.19"}
        diff = FP.differences(left, right)
        self.assertEqual(sorted(diff), ["cudnn", "torch"])
        self.assertEqual(diff["cudnn"], (91002, 90100))

    def test_a_missing_field_on_one_side_is_a_difference(self):
        self.assertIn("cudnn", FP.differences({"cudnn": 91002}, {}))


class ReceiptsCarryTheFingerprintTest(unittest.TestCase):
    def test_every_gpu_receipt_tool_records_runtime(self):
        for rel in GPU_RECEIPT_TOOLS:
            source = (ROOT / rel).read_text(encoding="utf-8")
            self.assertIn("runtime_fingerprint()", source, rel)
            self.assertIn('"runtime"', source, rel)

    def test_those_tools_still_parse(self):
        for rel in GPU_RECEIPT_TOOLS:
            ast.parse((ROOT / rel).read_text(encoding="utf-8"))


class StreamingEnvironmentIsSingleTest(unittest.TestCase):
    """The spec must name one RGB environment, and it must be the one that made the cache."""

    SPEC_DOC = ROOT / "docs/specs/perception_streaming_v1.md"

    def test_spec_names_the_cache_producing_environment(self):
        text = self.SPEC_DOC.read_text(encoding="utf-8")
        self.assertIn(STREAMING_ENV, text)

    def test_the_retired_environment_is_never_given_as_the_command_to_run(self):
        for line in self.SPEC_DOC.read_text(encoding="utf-8").splitlines():
            if RETIRED_ENV in line and "verify_perception_streaming" in line:
                self.fail(f"spec still tells the reader to run RGB in {RETIRED_ENV}: {line.strip()}")


if __name__ == "__main__":
    unittest.main()
