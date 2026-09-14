"""P9 validation-calibrated pixel/temporal error injector.

This module has no Isaac Gym dependency.  It deliberately receives only clean detector geometry;
simulator truth, target identity and metric range truth are outside its interface.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


STATES = ("HIT", "FALSE_LOCK", "NO_LOCK")
HIT, FALSE_LOCK, NO_LOCK = range(3)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cadence_adjusted_row(probabilities, state_index, reference_dt_s, step_dt_s):
    """Competing-risk cadence conversion fixed by the P9 preregistration."""
    if not (reference_dt_s > 0.0 and step_dt_s > 0.0):
        raise ValueError("reference_dt_s and step_dt_s must be positive")
    row = [float(value) for value in probabilities]
    if len(row) != len(STATES) or any(value < 0.0 for value in row):
        raise ValueError("invalid transition row")
    total = sum(row)
    if not math.isfinite(total) or abs(total - 1.0) > 1e-6:
        raise ValueError("transition row must sum to one")
    stay = min(1.0, max(0.0, row[state_index]))
    adjusted_stay = stay ** (step_dt_s / reference_dt_s)
    exit_source = 1.0 - stay
    adjusted = [0.0] * len(STATES)
    adjusted[state_index] = adjusted_stay
    if exit_source <= 1e-12:
        adjusted[state_index] = 1.0
    else:
        exit_target = 1.0 - adjusted_stay
        for index, value in enumerate(row):
            if index != state_index:
                adjusted[index] = exit_target * value / exit_source
    # Eliminate accumulated float error before torch.multinomial-style CDF sampling.
    adjusted[-1] += 1.0 - sum(adjusted)
    return adjusted


class EmpiricalPerceptionError:
    """Vectorized deterministic sampler for the compiled P9 model."""

    def __init__(self, model_path, expected_sha256, num_envs, device, step_dt_s, seed):
        self.path = Path(model_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"P9 empirical model is missing: {self.path}")
        self.sha256 = sha256_file(self.path)
        expected = str(expected_sha256 or "").strip().lower()
        if len(expected) != 64 or self.sha256 != expected:
            raise RuntimeError(
                f"P9 empirical model SHA mismatch: expected={expected or '<missing>'} "
                f"actual={self.sha256}"
            )
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or tuple(payload.get("states") or ()) != STATES:
            raise RuntimeError("unsupported P9 empirical model schema/states")
        if payload.get("test_used") is not False:
            raise RuntimeError("P9 calibration model must be validation-only (test_used=false)")

        self.model = payload
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.step_dt_s = float(step_dt_s)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self.state = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )

        self._edges = torch.tensor(payload["size_edges_px"], device=self.device)
        self._bin_map = torch.tensor(payload["fallback_bin_by_raw_bin"], device=self.device)
        self._initial = {}
        self._rows = {}
        self._offsets = {}
        self._latency = {}
        for entry in payload["bins"]:
            target_bin = int(entry["target_bin"])
            self._initial[target_bin] = torch.tensor(
                entry["initial_state_probabilities"], device=self.device
            )
            for state_index, row in enumerate(entry["transition_rows"]):
                adjusted = cadence_adjusted_row(
                    row["probabilities"], state_index, row["reference_dt_s"], self.step_dt_s
                )
                self._rows[(target_bin, state_index)] = torch.tensor(
                    adjusted, device=self.device
                )
            for destination_text, destination_rows in entry[
                "transition_rows_by_destination_bin"
            ].items():
                destination_bin = int(destination_text)
                for state_index, row in enumerate(destination_rows):
                    adjusted = cadence_adjusted_row(
                        row["probabilities"], state_index, row["reference_dt_s"], self.step_dt_s
                    )
                    self._rows[(target_bin, destination_bin, state_index)] = torch.tensor(
                        adjusted, device=self.device
                    )
            for state_index, state in enumerate(STATES[:2]):
                values = entry["offsets"][state]["joint_du_dv_normalized_samples"]
                if not values:
                    raise RuntimeError(f"P9 bin {target_bin}/{state} has no paired offsets")
                self._offsets[(target_bin, state_index)] = torch.tensor(
                    values, dtype=torch.float32, device=self.device
                )
            latency = entry["latency_ms_samples"]
            if not latency:
                raise RuntimeError(f"P9 bin {target_bin} has no latency samples")
            self._latency[target_bin] = torch.tensor(
                latency, dtype=torch.float32, device=self.device
            )
        max_latency_ms = max(max(v) for v in payload["latency_ms_samples_by_source_bin"].values())
        self.max_latency_steps = int(math.floor(max_latency_ms / 1000.0 / self.step_dt_s + 0.5))
        self.last_latency_ms = torch.zeros(self.num_envs, device=self.device)
        self.last_source_bin = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )

    def reset_idx(self, env_ids):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.state[env_ids] = -1
        self.last_latency_ms[env_ids] = 0.0
        self.last_source_bin[env_ids] = -1

    def _categorical(self, probabilities, count):
        draws = torch.rand(count, device=self.device, generator=self.generator)
        return torch.searchsorted(probabilities.cumsum(0), draws, right=False).clamp(max=2)

    def sample(self, size_px, eligible):
        """Return state, paired normalized offset and whole-step latency for each env."""
        if size_px.shape != (self.num_envs,) or eligible.shape != (self.num_envs,):
            raise ValueError("P9 sample expects one size/eligibility value per environment")
        raw_bin = torch.bucketize(size_px, self._edges, right=True)
        source_bin = self._bin_map[raw_bin]
        state_out = torch.full_like(self.state, NO_LOCK)

        previous_state = self.state.clone()
        previous_bin = self.last_source_bin.clone()
        for target_bin in sorted(self._initial):
            in_bin = eligible & (source_bin == target_bin)
            first = in_bin & (self.state < 0)
            n_first = int(first.sum().item())
            if n_first:
                self.state[first] = self._categorical(self._initial[target_bin], n_first)
        # P8 attributes each transition to the SOURCE frame's size bin.  Advance all initialized
        # chains from that prior bin, even when the clean bbox crosses a boundary this step.
        for transition_bin in sorted(self._initial):
            for destination_bin in sorted(self._initial):
                for state_index in range(len(STATES)):
                    advance = eligible & (previous_bin == transition_bin) & (
                        source_bin == destination_bin
                    ) & (previous_state == state_index)
                    n_advance = int(advance.sum().item())
                    if n_advance:
                        self.state[advance] = self._categorical(
                            self._rows[(transition_bin, destination_bin, state_index)], n_advance
                        )
        state_out[eligible] = self.state[eligible]
        self.last_source_bin[eligible] = source_bin[eligible]

        offsets = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        latency_ms = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for target_bin in sorted(self._initial):
            in_bin = eligible & (source_bin == target_bin)
            n_bin = int(in_bin.sum().item())
            if n_bin:
                samples = self._latency[target_bin]
                indices = torch.randint(
                    len(samples), (n_bin,), device=self.device, generator=self.generator
                )
                latency_ms[in_bin] = samples[indices]
            for state_index in (HIT, FALSE_LOCK):
                selected = in_bin & (state_out == state_index)
                n_selected = int(selected.sum().item())
                if n_selected:
                    samples = self._offsets[(target_bin, state_index)]
                    indices = torch.randint(
                        len(samples), (n_selected,), device=self.device,
                        generator=self.generator,
                    )
                    offsets[selected] = samples[indices]
        self.last_latency_ms = latency_ms
        latency_steps = torch.floor(
            latency_ms / (1000.0 * self.step_dt_s) + 0.5
        ).to(torch.long)
        return state_out, offsets, latency_steps
