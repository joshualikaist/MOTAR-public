"""Small, dependency-light safety helpers for PPO training."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch


def apply_density_capture_guard_overrides(config: dict, environ) -> dict:
    """Apply validated environment overrides to the same-density capture fail-stop."""
    if not isinstance(config, dict):
        raise TypeError("density capture guard config must be a dictionary")

    result = dict(config)
    fields = (
        ("NAVRL_DENSITY_GUARD_WINDOW_EPOCHS", "window_epochs", int, 2, None),
        ("NAVRL_DENSITY_GUARD_MIN_EPOCHS", "min_epochs_at_density", int, 1, None),
        ("NAVRL_DENSITY_GUARD_MIN_PEAK", "min_peak_capture", float, 0.0, 1.0),
        ("NAVRL_DENSITY_GUARD_DROP", "drop_absolute", float, 0.0, 1.0),
        ("NAVRL_DENSITY_GUARD_PATIENCE", "patience_epochs", int, 1, None),
    )
    for env_name, config_name, cast, minimum, maximum in fields:
        raw = str(environ.get(env_name, "")).strip()
        if not raw:
            continue
        try:
            value = cast(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{env_name} has invalid value {raw!r}") from exc
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{env_name} must be finite")
        if value < minimum or (maximum is not None and value > maximum):
            interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
            raise ValueError(f"{env_name} must be in {interval}, got {value}")
        result[config_name] = value
    return result


def is_finite_training_value(value) -> bool:
    """Return whether a scalar or tensor contains only finite values."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value.detach()).all().item())
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def first_nonfinite_training_value(
    metric_groups: Iterable[Tuple[str, Iterable]],
    named_parameters: Iterable[Tuple[str, torch.Tensor]],
) -> Optional[str]:
    """Return the first non-finite metric/parameter path, or ``None`` when all are finite."""
    for group_name, values in metric_groups:
        for index, value in enumerate(values):
            if not is_finite_training_value(value):
                return f"{group_name}[{index}]"

    for name, parameter in named_parameters:
        if not is_finite_training_value(parameter):
            return f"model.{name}"

    return None


def reset_optimizer_learning_rate(optimizer, learning_rate: float) -> None:
    """Override a checkpoint-restored optimizer LR with the active run configuration."""
    lr = float(learning_rate)
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"learning_rate must be finite and positive, got {learning_rate!r}")
    for group in optimizer.param_groups:
        group["lr"] = lr
