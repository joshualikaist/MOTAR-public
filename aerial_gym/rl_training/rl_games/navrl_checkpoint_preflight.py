"""Read-only safety checks for a NavRL density-curriculum resume checkpoint."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch


class CheckpointPreflightError(RuntimeError):
    pass


_CONTRACT_ENV = {
    "cfg_lidar_max_range": "NAVRL_LIDAR_RANGE",
    "cfg_max_obstacles": "NAVRL_MAX_OBSTACLES",
    "cfg_token_fov_deg": "NAVRL_OBSTACLE_FOV_DEG",
    "cfg_obstacle_suppress_deg": "NAVRL_OBSTACLE_SUPPRESS_DEG",
    "cfg_obstacle_selector": "NAVRL_OBSTACLE_SELECTOR",
    "cfg_obstacle_cluster_gap_m": "NAVRL_OBSTACLE_CLUSTER_GAP_M",
    "cfg_obstacle_sectors": "NAVRL_OBSTACLE_SECTORS",
    "cfg_lidar_hbeams": "NAVRL_LIDAR_HBEAMS",
    "cfg_lidar_vbeams": "NAVRL_LIDAR_VBEAMS",
    "cfg_corridor_tokens": "NAVRL_CORRIDOR_TOKENS",
    "cfg_corridor_horizon_m": "NAVRL_CORRIDOR_HORIZON_M",
    "cfg_corridor_min_width_m": "NAVRL_CORRIDOR_MIN_WIDTH_M",
    "cfg_geofence_actor": "NAVRL_GEOFENCE_ACTOR",
    "cfg_geofence_noise_std_m": "NAVRL_GEOFENCE_NOISE_STD_M",
    "cfg_geofence_dropout": "NAVRL_GEOFENCE_DROPOUT",
    "cfg_search_state": "NAVRL_SEARCH_STATE",
    "cfg_search_state_force_invalid": "NAVRL_SEARCH_STATE_FORCE_INVALID",
    # Arena / task version (v2 search arena). These do NOT change the observation width, so a
    # mismatch loads without error and silently measures a different task -- exactly why they
    # belong in the preflight contract rather than only in a runtime warning.
    "cfg_arena_xy": "NAVRL_ARENA_XY",
    "cfg_arena_z": "NAVRL_ARENA_Z",
    "cfg_bar_pool": "NAVRL_BAR_POOL",
    "cfg_placement_mode": "NAVRL_PLACEMENT_MODE",
    "cfg_placement_gap_m": "NAVRL_PLACEMENT_GAP_M",
    "cfg_placement_touch_m": "NAVRL_PLACEMENT_TOUCH_M",
    "cfg_placement_surface_clearance_m": "NAVRL_PLACEMENT_SURFACE_CLEARANCE_M",
    "cfg_episode_len_steps": "NAVRL_EPISODE_LEN_STEPS",
    "cfg_bar_x_min": "NAVRL_BAR_X_MIN",
    "cfg_bar_x_max": "NAVRL_BAR_X_MAX",
}
_STRING_CONTRACT_FIELDS = {
    "cfg_obstacle_selector",
    "cfg_bar_pool",
    "cfg_placement_mode",
    "cfg_search_state",
}
# Checkpoints saved before a contract field existed are interpreted at the field's historical
# behavior. corridor_tokens=0: every pre-corridor checkpoint used the plain 898-D schema.
_LEGACY_CONTRACT_DEFAULTS = {
    "cfg_obstacle_selector": "greedy_suppress",
    # Every checkpoint saved before 2026-07-31 trained in the v1 arena; these are its exact
    # settings, so a v1 checkpoint keeps passing while a v1-vs-v2 mismatch still fails loudly.
    "cfg_arena_xy": 24.0,
    "cfg_arena_z": 3.0,
    "cfg_bar_pool": "bars",
    "cfg_placement_mode": "random",
    "cfg_placement_gap_m": 1.6,
    "cfg_placement_touch_m": 0.4,
    "cfg_placement_surface_clearance_m": 0.0,
    "cfg_episode_len_steps": 300.0,
    "cfg_bar_x_min": 0.13,
    "cfg_bar_x_max": 0.96,
    "cfg_corridor_tokens": 0,
    # The first corridor checkpoints (2026-07-31) predated provenance for these two knobs but used
    # these exact launcher defaults. Future checkpoints save both fields explicitly.
    "cfg_corridor_horizon_m": 6.0,
    "cfg_corridor_min_width_m": 0.55,
    "cfg_geofence_actor": 0,
    "cfg_geofence_noise_std_m": 0.0,
    "cfg_geofence_dropout": 0.0,
    "cfg_search_state": "off",
    "cfg_search_state_force_invalid": 0,
}


def _first_nonfinite(value: Any, path: str = "checkpoint") -> Optional[str]:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            return path
        return None
    if isinstance(value, Mapping):
        for key, child in value.items():
            found = _first_nonfinite(child, f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _first_nonfinite(child, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return path
    return None


def expected_contract_from_environment() -> Dict[str, Any]:
    expected = {}
    for saved_key, env_name in _CONTRACT_ENV.items():
        raw = os.environ.get(env_name, "").strip()
        if raw:
            expected[saved_key] = (
                raw.lower() if saved_key in _STRING_CONTRACT_FIELDS else float(raw)
            )
    return expected


def _contract_values_match(saved: Any, expected: Any) -> bool:
    if isinstance(expected, str):
        return str(saved).strip().lower() == expected.strip().lower()
    try:
        return abs(float(saved) - float(expected)) <= 1e-6
    except (TypeError, ValueError):
        return False


def _format_contract_value(value: Any) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def inspect_checkpoint(
    checkpoint_path: str,
    *,
    max_epochs: int,
    density_final: int,
    expected_contract: Optional[Mapping[str, Any]] = None,
    allowed_contract_overrides: Optional[Iterable[str]] = None,
    min_k_max: Optional[float] = None,
    min_task_steps: Optional[int] = None,
) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise CheckpointPreflightError(f"checkpoint not found: {path}")

    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointPreflightError(f"checkpoint load failed: {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise CheckpointPreflightError("checkpoint root must be a dictionary")

    for section in ("model", "optimizer", "assymetric_vf_nets", "assymetric_vf_optimizer"):
        if section not in checkpoint:
            raise CheckpointPreflightError(f"checkpoint is missing required section: {section}")

    try:
        epoch = int(checkpoint["epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointPreflightError("checkpoint epoch is missing or invalid") from exc
    if int(max_epochs) <= epoch:
        raise CheckpointPreflightError(
            f"MAX_EPOCHS must exceed checkpoint epoch ({max_epochs} <= {epoch})"
        )

    state = checkpoint.get("env_state")
    if not isinstance(state, dict):
        raise CheckpointPreflightError("checkpoint env_state is missing")
    try:
        bars = int(state["n_bars_active"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointPreflightError("checkpoint n_bars_active is missing or invalid") from exc
    if bars < 0 or bars > int(density_final):
        raise CheckpointPreflightError(
            f"checkpoint density {bars} is outside requested curriculum range 0..{density_final}"
        )
    task_steps = int(state.get("num_task_steps", 0))
    k_max = float(state.get("k_max_cur", float("-inf")))
    if min_k_max is not None and k_max + 1e-6 < float(min_k_max):
        raise CheckpointPreflightError(
            f"distance curriculum is not saturated: k_max={k_max:g} < {float(min_k_max):g}"
        )
    if min_task_steps is not None and task_steps < int(min_task_steps):
        raise CheckpointPreflightError(
            "time-based curricula are not saturated: "
            f"task_steps={task_steps} < {int(min_task_steps)}"
        )

    allowed_overrides = set(allowed_contract_overrides or ())
    contract_overrides = {}
    unknown_overrides = allowed_overrides.difference(_CONTRACT_ENV)
    if unknown_overrides:
        raise CheckpointPreflightError(
            "unknown policy-input override field(s): " + ", ".join(sorted(unknown_overrides))
        )

    for saved_key, expected in (expected_contract or {}).items():
        saved = state.get(saved_key, _LEGACY_CONTRACT_DEFAULTS.get(saved_key))
        if saved is None:
            if saved_key in allowed_overrides:
                contract_overrides[saved_key] = {
                    "checkpoint": None,
                    "requested": expected,
                }
                continue
            raise CheckpointPreflightError(
                f"checkpoint lacks policy-input provenance field: {saved_key}"
            )
        if not _contract_values_match(saved, expected):
            if saved_key not in allowed_overrides:
                raise CheckpointPreflightError(
                    f"policy-input mismatch for {saved_key}: checkpoint={saved}, "
                    f"requested={expected}"
                )
            contract_overrides[saved_key] = {
                "checkpoint": saved,
                "requested": expected,
            }

    nonfinite_path = _first_nonfinite(checkpoint)
    if nonfinite_path is not None:
        raise CheckpointPreflightError(f"non-finite checkpoint value: {nonfinite_path}")

    return {
        "path": str(path),
        "epoch": epoch,
        "bars": bars,
        "task_steps": task_steps,
        "k_max": k_max,
        "token_fov_deg": state.get("cfg_token_fov_deg"),
        "max_obstacles": state.get("cfg_max_obstacles"),
        "contract_overrides": contract_overrides,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--density-final", type=int, required=True)
    parser.add_argument("--min-k-max", type=float)
    parser.add_argument("--min-task-steps", type=int)
    parser.add_argument(
        "--allow-contract-override",
        action="append",
        choices=sorted(_CONTRACT_ENV),
        default=[],
        help="allow one intentional same-shape policy-input ablation while retaining all other checks",
    )
    args = parser.parse_args()
    try:
        info = inspect_checkpoint(
            args.checkpoint,
            max_epochs=args.max_epochs,
            density_final=args.density_final,
            expected_contract=expected_contract_from_environment(),
            allowed_contract_overrides=args.allow_contract_override,
            min_k_max=args.min_k_max,
            min_task_steps=args.min_task_steps,
        )
    except CheckpointPreflightError as exc:
        print(f"[general_repr_density] preflight FAILED | {exc}", file=os.sys.stderr)
        return 2

    print(
        "[general_repr_density] checkpoint state | "
        f"epoch={info['epoch']} bars={info['bars']} task_steps={info['task_steps']} "
        f"k_max={info['k_max']:g} tokens={info['max_obstacles']} "
        f"fov={info['token_fov_deg']}deg"
    )
    for field, override in info["contract_overrides"].items():
        print(
            "[general_repr_density] INTENTIONAL contract override | "
            f"{field}: checkpoint={_format_contract_value(override['checkpoint'])} "
            f"requested={_format_contract_value(override['requested'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
