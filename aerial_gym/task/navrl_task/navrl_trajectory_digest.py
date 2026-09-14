"""A running hash of the simulated trajectory, for proving that instrumentation changed nothing.

The question this answers is narrow and mechanical: does turning a logger on alter what the
simulator actually did? Comparing summary statistics cannot answer it - two different trajectories
can share a mean. Comparing a hash of every recorded step can.

The digest itself is instrumentation, so it is enabled in BOTH runs of a comparison and cancels out.
It reads detached tensors, draws no randomness and writes into no task buffer.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

FIELDS = ("position", "orientation", "command")


class TrajectoryDigest:
    """Incremental sha256 over per-step state and executed command."""

    def __init__(self, num_envs, *, label=""):
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.label = str(label)
        self.steps = 0
        self._hash = hashlib.sha256()
        self._hash.update(("navrl_trajectory_digest_v1|%d|%s|" % (self.num_envs, self.label)).encode())

    def record(self, *, position, orientation, command):
        """Hash one step. float32 is hashed exactly as the simulator holds it, with no rounding."""
        for name, tensor in (("position", position), ("orientation", orientation),
                             ("command", command)):
            if tensor is None:
                raise ValueError("trajectory digest requires %s" % name)
            array = tensor.detach().to(torch.float32).cpu().numpy()
            if array.shape[0] != self.num_envs:
                raise ValueError("%s has the wrong batch size" % name)
            self._hash.update(name.encode())
            self._hash.update(np.ascontiguousarray(array).tobytes(order="C"))
        self.steps += 1

    def hexdigest(self):
        return self._hash.hexdigest()

    def as_dict(self):
        return {"schema": "navrl_trajectory_digest_v1", "label": self.label,
                "num_envs": self.num_envs, "steps": self.steps, "fields": list(FIELDS),
                "sha256": self.hexdigest()}
