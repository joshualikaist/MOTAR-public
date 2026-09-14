"""Transfer-only alternatives for static G-buffers; no compute/geometry changes."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ARMS = ("A0", "A1", "A2", "A3")


def rss_mib(path=Path("/proc/self/status")):
    # Historical RC-R5 reader is deliberately not modified. Close even on early return.
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return None


def full_host(buffers):
    return {key: value.detach().cpu().numpy().copy() for key, value in buffers.items()}


class TransferPlan:
    def __init__(self, arm, template):
        if arm not in ARMS:
            raise ValueError("Unknown transfer arm")
        self.arm = arm
        self.layout = {key: (value.shape, value.dtype, value.device) for key, value in template.items()}
        self.pinned = None
        if arm == "A3":
            if any(value.device.type != "cuda" for value in template.values()):
                raise ValueError("A3 requires a CUDA source and pinned-memory support")
            self.pinned = {key: torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)
                           for key, value in template.items()}

    def transfer(self, buffers):
        if {k: (v.shape, v.dtype, v.device) for k, v in buffers.items()} != self.layout:
            raise ValueError("Buffer contract changed")
        if self.arm == "A0":
            return full_host(buffers)
        if self.arm == "A1":
            return {"valid_pixels": int(buffers["valid"].sum().item()),
                    "rgb_sum": float(buffers["rgb"].sum(dtype=torch.float64).item())}
        if self.arm == "A2":
            groups = defaultdict(list)
            for key, value in buffers.items():
                groups[value.dtype].append((key, value))
            result = {}
            for entries in groups.values():
                packed = torch.cat([value.reshape(-1) for _, value in entries])
                host = packed.detach().cpu().numpy()
                offset = 0
                for key, value in entries:
                    size = value.numel()
                    result[key] = host[offset:offset + size].reshape(tuple(value.shape))
                    offset += size
            return result
        for key, value in buffers.items():
            self.pinned[key].copy_(value, non_blocking=True)
        # Includes completion in the measured arm. No unmeasured outstanding DMA or overlap claim.
        torch.cuda.current_stream(next(iter(buffers.values())).device).synchronize()
        return {key: value.numpy() for key, value in self.pinned.items()}


def hashes(arrays):
    import hashlib
    return {key: {"sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                  "shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()}
