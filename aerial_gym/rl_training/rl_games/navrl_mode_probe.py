"""Evaluation-only counterfactual probe for single-action mode averaging.

The probe never replaces the action executed by the player.  It sends six fixed 898-D structured
observations through the frozen policy in a side forward pass: physical corridors centred on
body-forward or shifted by +/- ``offset_deg``, each with both possible two-obstacle slot orders.
The two centre observations describe the SAME +/-12-degree geometry without averaging feature
bytes. Left/right are exact reflected pairs under each slot-order control. This makes token-order
sensitivity observable instead of silently folding it into a mode-averaging verdict.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

from aerial_gym.rl_training.rl_games.ppo_update_safety import (
    mirror_navrl_actions,
    mirror_navrl_structured_observation,
)


SCHEMA_VERSION = 1
AXES = ("x", "y", "z", "yaw")

# Predeclared before policy measurement.  These are mechanism gates, not a replacement-policy
# gate and not an outcome/capture claim.
GATES = {
    "reflection_max_abs_action": 0.15,
    "slot_permutation_max_abs_action": 0.15,
    "symmetric_horizontal_speed_max_mps": 0.25,
    "perturbed_horizontal_speed_min_mps": 0.75,
    "symmetric_abs_lateral_max": 0.10,
    "perturbed_abs_lateral_min": 0.25,
    "near_zero_horizontal_speed_mps": 0.25,
}


def _layout():
    hbeams = int(os.environ.get("NAVRL_LIDAR_HBEAMS", "72"))
    vbeams = int(os.environ.get("NAVRL_LIDAR_VBEAMS", "4"))
    max_obstacles = int(os.environ.get("NAVRL_MAX_OBSTACLES", "8"))
    static_dim = hbeams * vbeams
    obstacle_size = 5 * max_obstacles * 12
    robot_size = 5 * 10
    target_size = 5 * 16
    return hbeams, vbeams, max_obstacles, static_dim, obstacle_size, robot_size, target_size


def _physical_fixture(*, device, dtype, centre_deg, swapped=False):
    """Build one physical corridor observation with an explicit token slot order."""
    hbeams, vbeams, max_obstacles, static_dim, obstacle_size, robot_size, target_size = _layout()
    total = static_dim + obstacle_size + robot_size + target_size
    obs = torch.zeros((1, total), device=device, dtype=dtype)

    # Static scan: no-return=1, with two 4 m surfaces at centre +/-12 deg. Three beams centred on
    # the nearest bin approximate a finite-width bar while remaining exactly reflection symmetric.
    scan = obs[:, :static_dim].view(1, vbeams, hbeams)
    scan.fill_(1.0)
    bin_deg = 360.0 / hbeams
    geometry_angles = (12.0 + centre_deg, -12.0 + centre_deg)
    for angle in geometry_angles:
        idx = int(round((180.0 - angle) / bin_deg)) % hbeams
        for beam_offset in (-1, 0, 1):
            scan[:, :, (idx + beam_offset) % hbeams] = 4.0 / 12.0

    # Structured obstacle histories use the documented NavRL token schema. Equal-distance cluster
    # ties may serialize either order, so both are explicit probe arms.
    offset = static_dim
    obstacles = obs[:, offset : offset + obstacle_size].view(1, 5, max_obstacles, 12)
    token_angles = tuple(reversed(geometry_angles)) if swapped else geometry_angles
    for slot, angle in enumerate(token_angles):
        radius = math.radians(angle)
        r = 4.0
        radial_sigma = 0.04 + 0.02 * r
        lateral_sigma = 0.05 + r * math.tan(math.radians(5.0))
        feat = torch.tensor(
            [
                r * math.cos(radius) / 12.0,
                r * math.sin(radius) / 12.0,
                0.0,
                0.0, 0.0, 0.0,
                0.30 / 12.0,
                1.0 - r / 12.0,
                radial_sigma**2 / 12.0**2,
                lateral_sigma**2 / 12.0**2,
                0.12 / 12.0**2,
                1.0,
            ],
            device=device,
            dtype=dtype,
        )
        obstacles[:, :, slot] = feat

    offset += obstacle_size
    robot = obs[:, offset : offset + robot_size].view(1, 5, 10)
    robot[..., 8] = 1.0 / 3.0  # level at the nominal flight altitude
    robot[..., 9] = 1.0        # valid

    offset += robot_size
    target = obs[:, offset : offset + target_size].view(1, 5, 16)
    target[..., 0] = 8.0 / 20.0  # stationary, visible target 8 m forward
    target[..., 12] = 1.0
    target[..., 13] = 1.0
    target[..., 15] = 0.30 / 12.0
    return obs


def build_probe_observations(reference_obs, *, offset_deg=5.0):
    """Return physical centre/left/right fixtures with slot-order controls."""
    if not isinstance(reference_obs, torch.Tensor) or reference_obs.ndim != 2:
        raise ValueError("reference_obs must be a [batch, features] tensor")
    fixture_args = {"device": reference_obs.device, "dtype": reference_obs.dtype}
    fixtures = {
        "symmetric_lr": _physical_fixture(**fixture_args, centre_deg=0.0, swapped=False),
        "symmetric_rl": _physical_fixture(**fixture_args, centre_deg=0.0, swapped=True),
        "left_lr": _physical_fixture(**fixture_args, centre_deg=offset_deg, swapped=False),
        "left_rl": _physical_fixture(**fixture_args, centre_deg=offset_deg, swapped=True),
        "right_lr": _physical_fixture(**fixture_args, centre_deg=-offset_deg, swapped=False),
        "right_rl": _physical_fixture(**fixture_args, centre_deg=-offset_deg, swapped=True),
    }
    if any(value.shape[1] != reference_obs.shape[1] for value in fixtures.values()):
        raise ValueError(
            "mode probe supports the base structured schema only: fixture=%d runtime=%d"
            % (fixtures["symmetric_lr"].shape[1], reference_obs.shape[1])
        )
    # Reflection preserves token identity, so it maps LR to the mirrored scene's LR token order.
    # The centre geometry is invariant as a SET but LR and RL bytes differ by slot permutation.
    pair_errors = {
        "symmetric_lr_to_rl": float(
            (mirror_navrl_structured_observation(fixtures["symmetric_lr"])
             - fixtures["symmetric_rl"]).abs().max().item()
        ),
        "left_lr_to_right_rl": float(
            (mirror_navrl_structured_observation(fixtures["left_lr"])
             - fixtures["right_rl"]).abs().max().item()
        ),
        "left_rl_to_right_lr": float(
            (mirror_navrl_structured_observation(fixtures["left_rl"])
             - fixtures["right_lr"]).abs().max().item()
        ),
    }
    if max(pair_errors.values()) > 1e-7:
        raise RuntimeError("mode-probe fixture reflection contract failed")
    return fixtures, {"reflection_pair_max_abs": pair_errors}


def model_outputs(result):
    """Extract bounded deterministic action and available Gaussian parameters."""
    action = result.get("deterministic_actions", result.get("mus"))
    if not isinstance(action, torch.Tensor) or action.ndim != 2 or action.shape[1] != 4:
        raise RuntimeError("policy did not expose a four-axis deterministic action")
    mu = result.get("mus")
    sigma = result.get("sigmas")
    return {
        "action": action.detach(),
        "mu": mu.detach() if isinstance(mu, torch.Tensor) and mu.shape == action.shape else None,
        "sigma": (
            sigma.detach()
            if isinstance(sigma, torch.Tensor) and sigma.shape == action.shape
            else None
        ),
    }


class ModeProbeRecorder:
    def __init__(self, output_path, *, max_velocity_mps, offset_deg=5.0):
        self.output_path = Path(output_path)
        self.max_velocity = float(max_velocity_mps)
        self.offset_deg = float(offset_deg)
        if not math.isfinite(self.max_velocity) or self.max_velocity <= 0.0:
            raise ValueError("max_velocity_mps must be finite and positive")
        self.samples = 0
        self.fixture_contract = None
        self.sum = {
            arm: {key: torch.zeros(4, dtype=torch.float64) for key in ("action", "mu", "sigma")}
            for arm in (
                "symmetric_lr", "symmetric_rl", "left_lr", "left_rl", "right_lr", "right_rl"
            )
        }
        self.available = {"action": True, "mu": True, "sigma": True}
        self.near_zero = {arm: 0 for arm in self.sum}

    def record(self, outputs, fixture_contract):
        if set(outputs) != set(self.sum):
            raise ValueError("mode probe requires six physical corridor/order outputs")
        self.fixture_contract = dict(fixture_contract)
        batch = None
        for arm, values in outputs.items():
            action = values["action"]
            if not torch.isfinite(action).all():
                raise RuntimeError("mode probe observed non-finite deterministic action")
            batch = int(action.shape[0]) if batch is None else batch
            if int(action.shape[0]) != batch:
                raise RuntimeError("mode probe arm batch sizes differ")
            for key in ("action", "mu", "sigma"):
                value = values[key]
                if value is None:
                    self.available[key] = False
                    continue
                if not torch.isfinite(value).all():
                    raise RuntimeError("mode probe observed non-finite %s" % key)
                self.sum[arm][key] += value.double().sum(dim=0).cpu()
            horizontal = action[:, :2].norm(dim=1) * self.max_velocity
            self.near_zero[arm] += int(
                (horizontal <= GATES["near_zero_horizontal_speed_mps"]).sum().item()
            )
        self.samples += int(batch or 0)

    def payload(self):
        if self.samples <= 0 or self.fixture_contract is None:
            raise RuntimeError("mode probe has no samples")

        def mean(arm, key):
            if not self.available[key]:
                return None
            return [float(v / self.samples) for v in self.sum[arm][key]]

        arms = {}
        for arm in self.sum:
            action = mean(arm, "action")
            arms[arm] = {
                "deterministic_action_mean": action,
                "latent_mu_mean": mean(arm, "mu"),
                "sigma_mean": mean(arm, "sigma"),
                "command_xy_mean_mps": [
                    action[0] * self.max_velocity,
                    action[1] * self.max_velocity,
                ],
                "horizontal_speed_from_mean_mps": math.hypot(action[0], action[1])
                * self.max_velocity,
                "near_zero_horizontal_rate": self.near_zero[arm] / self.samples,
            }

        action_tensors = {
            arm: torch.tensor(payload["deterministic_action_mean"]).view(1, 4)
            for arm, payload in arms.items()
        }
        reflected_pairs = (
            ("symmetric_lr", "symmetric_rl"),
            ("left_lr", "right_rl"),
            ("left_rl", "right_lr"),
        )
        reflection_errors = {}
        for left_name, right_name in reflected_pairs:
            reflection_errors[f"{left_name}_to_{right_name}"] = (
                action_tensors[right_name] - mirror_navrl_actions(action_tensors[left_name])
            ).abs().view(-1).tolist()
        slot_pairs = (
            ("symmetric_lr", "symmetric_rl"),
            ("left_lr", "left_rl"),
            ("right_lr", "right_rl"),
        )
        slot_errors = {
            f"{a}_vs_{b}": (action_tensors[a] - action_tensors[b]).abs().view(-1).tolist()
            for a, b in slot_pairs
        }
        reflection_max = max(value for row in reflection_errors.values() for value in row)
        slot_max = max(value for row in slot_errors.values() for value in row)
        centre_names = ("symmetric_lr", "symmetric_rl")
        perturb_names = ("left_lr", "left_rl", "right_lr", "right_rl")
        centre_speed_max = max(arms[name]["horizontal_speed_from_mean_mps"] for name in centre_names)
        perturb_speed_min = min(arms[name]["horizontal_speed_from_mean_mps"] for name in perturb_names)
        centre_abs_y_max = max(abs(float(action_tensors[name][0, 1])) for name in centre_names)
        perturb_abs_y_min = min(abs(float(action_tensors[name][0, 1])) for name in perturb_names)
        lateral_opposite = all(
            float(action_tensors[left][0, 1] * action_tensors[right][0, 1]) < 0.0
            for left, right in (("left_lr", "right_rl"), ("left_rl", "right_lr"))
        )
        reflection_quality = reflection_max <= GATES["reflection_max_abs_action"]
        slot_quality = slot_max <= GATES["slot_permutation_max_abs_action"]
        checks = {
            "policy_reflection_quality": reflection_quality,
            "slot_permutation_quality": slot_quality,
            "symmetric_horizontal_stall_both_orders": centre_speed_max
            <= GATES["symmetric_horizontal_speed_max_mps"],
            "perturbations_restore_motion_all_orders": perturb_speed_min
            >= GATES["perturbed_horizontal_speed_min_mps"],
            "symmetric_lateral_near_zero_both_orders": centre_abs_y_max
            <= GATES["symmetric_abs_lateral_max"],
            "perturbed_lateral_decisive_all_orders": perturb_abs_y_min
            >= GATES["perturbed_abs_lateral_min"],
            "perturbed_lateral_signs_opposite": lateral_opposite,
        }
        mechanism = all(
            value for key, value in checks.items()
            if key not in ("policy_reflection_quality", "slot_permutation_quality")
        )
        if not slot_quality:
            verdict = "INCONCLUSIVE_SLOT_ORDER_SENSITIVITY"
        elif not reflection_quality:
            verdict = "INCONCLUSIVE_POLICY_CHIRALITY"
        elif mechanism:
            verdict = "MODE_AVERAGING_SUPPORTED_IN_SYNTHETIC_POLICY_SCREEN"
        else:
            verdict = "MODE_AVERAGING_NOT_SUPPORTED_IN_SYNTHETIC_POLICY_SCREEN"
        return {
            "schema_version": SCHEMA_VERSION,
            "probe": "frozen_policy_symmetric_corridor_synthetic_screen",
            "decision_authority": "diagnostic_only_no_training_or_replacement_authority",
            "samples": self.samples,
            "independent_fixture_count": 1,
            "offset_deg": self.offset_deg,
            "max_velocity_mps": self.max_velocity,
            "fixture_contract": self.fixture_contract,
            "preregistered_gates": dict(GATES),
            "arms": arms,
            "pair": {
                "reflection_abs_errors": reflection_errors,
                "reflection_max_abs": reflection_max,
                "slot_permutation_abs_errors": slot_errors,
                "slot_permutation_max_abs": slot_max,
                "symmetric_horizontal_speed_max_mps": centre_speed_max,
                "perturbed_horizontal_speed_min_mps": perturb_speed_min,
                "symmetric_abs_lateral_max": centre_abs_y_max,
                "perturbed_abs_lateral_min": perturb_abs_y_min,
            },
            "checks": checks,
            "verdict": verdict,
            "limitations": [
                "All six inputs encode physical two-surface geometry, but remain synthetic policy "
                "fixtures rather than sensor frames replayed from the simulator.",
                "Equal-distance token ordering is not assumed invariant: both orders are measured, "
                "and excessive action sensitivity forces an inconclusive verdict.",
                "This side forward-pass does not execute commands or measure capture/crash outcomes.",
                "The reported sample count is repeated inference of one fixed fixture per arm, not "
                "independent environment samples and not a confidence-interval basis.",
                "A positive result motivates a simulator replay or candidate-head ablation; it does "
                "not by itself establish a causal high-density failure mechanism.",
            ],
        }

    def write(self):
        payload = self.payload()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            raise RuntimeError("refusing to overwrite mode-probe output: %s" % self.output_path)
        tmp = self.output_path.with_name(self.output_path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.output_path)
        return payload
