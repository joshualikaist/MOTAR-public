#!/usr/bin/env python3
"""The execution stack a result was produced on, in one place.

Receipts recorded the device name and fp16 flag but not the interpreter or the accelerator
libraries. On 2026-09-08 that gap cost a day: a candidate cache built under torch 2.10 / cuDNN
9.10.2 could not be reproduced by a re-run under torch 2.4 / cuDNN 9.1, the two stacks pick
different convolution kernels, and confidences diverged by up to 45%. Nothing in the receipt said
which stack had produced it, so the mismatch looked like non-determinism and was chased through
TF32, benchmark and deterministic flags that could never have explained it.

Call `runtime_fingerprint()` when writing any receipt whose numbers came off a GPU. It imports
torch and OpenCV lazily so a CPU-only tool that never touches them still works.
"""

from __future__ import annotations

import platform
import sys


def runtime_fingerprint(include_device=True):
    """A JSON-serialisable record of what produced these numbers.

    Every field is read from the running process, never from configuration, so it cannot drift
    away from the truth the way a hand-maintained note does.
    """
    record = {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    try:
        import torch
    except ImportError:
        record["torch"] = None
    else:
        record.update({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            # These three change kernel selection and therefore the last digits of every
            # convolution, which is exactly what a reproduction check compares.
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        })
        if include_device and torch.cuda.is_available():
            record["device_name"] = torch.cuda.get_device_name(0)
            record["device_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
            record["driver_cuda"] = getattr(torch.version, "cuda", None)
    try:
        import cv2
    except ImportError:
        record["opencv"] = None
    else:
        record["opencv"] = cv2.__version__
        # Farneback flow is CPU work whose thread count is a scheduling choice, not a result
        # change; recorded so a latency comparison is read against the right machine state.
        record["opencv_threads"] = cv2.getNumThreads()
    try:
        import numpy
    except ImportError:
        record["numpy"] = None
    else:
        record["numpy"] = numpy.__version__
    return record


def differences(left, right):
    """Fields where two fingerprints disagree, for explaining a failed reproduction."""
    keys = sorted(set(left) | set(right))
    return {k: (left.get(k), right.get(k)) for k in keys if left.get(k) != right.get(k)}


if __name__ == "__main__":
    import json
    print(json.dumps(runtime_fingerprint(), indent=2, sort_keys=True))
