"""TRAIN: PPO epoch intercept success rate and mean distance."""

from __future__ import annotations

import os
from typing import List, Optional

_EPOCH_DONE = 0
_EPOCH_SUCC = 0
_EPOCH_ENVS_WITH_SUCC: set[int] = set()
_EPOCH_DIST_SUM = 0.0
_EPOCH_DIST_COUNT = 0
_EPOCH_CLOSEST_DIST_SUM = 0.0
_EPOCH_CLOSEST_DIST_COUNT = 0
_EPOCH_SUCCESS_CLOSEST_DIST_SUM = 0.0
_EPOCH_SUCCESS_CLOSEST_DIST_COUNT = 0
_EPOCH_FAILURE_CLOSEST_DIST_SUM = 0.0
_EPOCH_FAILURE_CLOSEST_DIST_COUNT = 0
_EPOCH_SURFACE_GAP_SUM = 0.0
_EPOCH_SURFACE_GAP_COUNT = 0
_EPOCH_FAILURE_SURFACE_GAP_SUM = 0.0
_EPOCH_FAILURE_SURFACE_GAP_COUNT = 0
_EPOCH_NEAR_MISS_COUNT = 0
_EPOCH_RADIUS_NO_CONTACT_COUNT = 0
_EPOCH_CONTACT_NO_RADIUS_COUNT = 0


def train_intercept_metrics_enabled() -> bool:
    return os.environ.get("AERIAL_RL_PLAY", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    )


def record_train_epoch_episodes(
    num_finished: int,
    num_success: int,
    success_env_ids: Optional[List[int]] = None,
    closest_dist_sum: Optional[float] = None,
    closest_dist_count: int = 0,
    success_closest_dist_sum: Optional[float] = None,
    success_closest_dist_count: int = 0,
    failure_closest_dist_sum: Optional[float] = None,
    failure_closest_dist_count: int = 0,
    surface_gap_sum: Optional[float] = None,
    surface_gap_count: int = 0,
    failure_surface_gap_sum: Optional[float] = None,
    failure_surface_gap_count: int = 0,
    near_miss_count: int = 0,
    radius_no_contact_count: int = 0,
    contact_no_radius_count: int = 0,
) -> None:
    global _EPOCH_DONE, _EPOCH_SUCC, _EPOCH_ENVS_WITH_SUCC
    global _EPOCH_CLOSEST_DIST_SUM, _EPOCH_CLOSEST_DIST_COUNT
    global _EPOCH_SUCCESS_CLOSEST_DIST_SUM, _EPOCH_SUCCESS_CLOSEST_DIST_COUNT
    global _EPOCH_FAILURE_CLOSEST_DIST_SUM, _EPOCH_FAILURE_CLOSEST_DIST_COUNT
    global _EPOCH_SURFACE_GAP_SUM, _EPOCH_SURFACE_GAP_COUNT
    global _EPOCH_FAILURE_SURFACE_GAP_SUM, _EPOCH_FAILURE_SURFACE_GAP_COUNT
    global _EPOCH_NEAR_MISS_COUNT, _EPOCH_RADIUS_NO_CONTACT_COUNT, _EPOCH_CONTACT_NO_RADIUS_COUNT
    if num_finished <= 0:
        return
    _EPOCH_DONE += int(num_finished)
    _EPOCH_SUCC += int(num_success)
    if success_env_ids:
        _EPOCH_ENVS_WITH_SUCC.update(int(i) for i in success_env_ids)
    if closest_dist_sum is not None and closest_dist_count > 0:
        _EPOCH_CLOSEST_DIST_SUM += float(closest_dist_sum)
        _EPOCH_CLOSEST_DIST_COUNT += int(closest_dist_count)
    if success_closest_dist_sum is not None and success_closest_dist_count > 0:
        _EPOCH_SUCCESS_CLOSEST_DIST_SUM += float(success_closest_dist_sum)
        _EPOCH_SUCCESS_CLOSEST_DIST_COUNT += int(success_closest_dist_count)
    if failure_closest_dist_sum is not None and failure_closest_dist_count > 0:
        _EPOCH_FAILURE_CLOSEST_DIST_SUM += float(failure_closest_dist_sum)
        _EPOCH_FAILURE_CLOSEST_DIST_COUNT += int(failure_closest_dist_count)
    if surface_gap_sum is not None and surface_gap_count > 0:
        _EPOCH_SURFACE_GAP_SUM += float(surface_gap_sum)
        _EPOCH_SURFACE_GAP_COUNT += int(surface_gap_count)
    if failure_surface_gap_sum is not None and failure_surface_gap_count > 0:
        _EPOCH_FAILURE_SURFACE_GAP_SUM += float(failure_surface_gap_sum)
        _EPOCH_FAILURE_SURFACE_GAP_COUNT += int(failure_surface_gap_count)
    _EPOCH_NEAR_MISS_COUNT += int(near_miss_count)
    _EPOCH_RADIUS_NO_CONTACT_COUNT += int(radius_no_contact_count)
    _EPOCH_CONTACT_NO_RADIUS_COUNT += int(contact_no_radius_count)


def _parallel_env_count_hint() -> int:
    raw = os.environ.get("NUM_ENVS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 128


def _rate_line(succ: int, done: int) -> str:
    if done <= 0:
        return "intercept success     : n/a (0 finished)"
    rate = 100.0 * float(succ) / float(done)
    return f"intercept success     : {rate:5.1f}% ({succ}/{done})"


def record_train_step_mean_dist(mean_dist_m: float) -> None:
    global _EPOCH_DIST_SUM, _EPOCH_DIST_COUNT
    if mean_dist_m < 0.0:
        return
    _EPOCH_DIST_SUM += float(mean_dist_m)
    _EPOCH_DIST_COUNT += 1


def consume_epoch_intercept_summary(
    num_parallel_envs: int,
) -> tuple[List[str], int, int, int, int, Optional[float], Optional[float], dict[str, Optional[float]]]:
    global _EPOCH_DONE, _EPOCH_SUCC, _EPOCH_ENVS_WITH_SUCC, _EPOCH_DIST_SUM, _EPOCH_DIST_COUNT
    global _EPOCH_CLOSEST_DIST_SUM, _EPOCH_CLOSEST_DIST_COUNT
    global _EPOCH_SUCCESS_CLOSEST_DIST_SUM, _EPOCH_SUCCESS_CLOSEST_DIST_COUNT
    global _EPOCH_FAILURE_CLOSEST_DIST_SUM, _EPOCH_FAILURE_CLOSEST_DIST_COUNT
    global _EPOCH_SURFACE_GAP_SUM, _EPOCH_SURFACE_GAP_COUNT
    global _EPOCH_FAILURE_SURFACE_GAP_SUM, _EPOCH_FAILURE_SURFACE_GAP_COUNT
    global _EPOCH_NEAR_MISS_COUNT, _EPOCH_RADIUS_NO_CONTACT_COUNT, _EPOCH_CONTACT_NO_RADIUS_COUNT
    done = _EPOCH_DONE
    succ = _EPOCH_SUCC
    envs_hit = len(_EPOCH_ENVS_WITH_SUCC)
    mean_dist = None
    if _EPOCH_DIST_COUNT > 0:
        mean_dist = _EPOCH_DIST_SUM / float(_EPOCH_DIST_COUNT)
    mean_closest_dist = None
    if _EPOCH_CLOSEST_DIST_COUNT > 0:
        mean_closest_dist = _EPOCH_CLOSEST_DIST_SUM / float(_EPOCH_CLOSEST_DIST_COUNT)
    success_closest_dist = None
    if _EPOCH_SUCCESS_CLOSEST_DIST_COUNT > 0:
        success_closest_dist = _EPOCH_SUCCESS_CLOSEST_DIST_SUM / float(_EPOCH_SUCCESS_CLOSEST_DIST_COUNT)
    failure_closest_dist = None
    if _EPOCH_FAILURE_CLOSEST_DIST_COUNT > 0:
        failure_closest_dist = _EPOCH_FAILURE_CLOSEST_DIST_SUM / float(_EPOCH_FAILURE_CLOSEST_DIST_COUNT)
    mean_surface_gap = None
    if _EPOCH_SURFACE_GAP_COUNT > 0:
        mean_surface_gap = _EPOCH_SURFACE_GAP_SUM / float(_EPOCH_SURFACE_GAP_COUNT)
    failure_surface_gap = None
    if _EPOCH_FAILURE_SURFACE_GAP_COUNT > 0:
        failure_surface_gap = _EPOCH_FAILURE_SURFACE_GAP_SUM / float(_EPOCH_FAILURE_SURFACE_GAP_COUNT)
    failure_done = max(0, done - succ)
    near_miss_count = _EPOCH_NEAR_MISS_COUNT
    radius_no_contact_count = _EPOCH_RADIUS_NO_CONTACT_COUNT
    contact_no_radius_count = _EPOCH_CONTACT_NO_RADIUS_COUNT
    near_miss_rate = float(near_miss_count) / float(failure_done) if failure_done > 0 else None
    radius_no_contact_rate = (
        float(radius_no_contact_count) / float(failure_done) if failure_done > 0 else None
    )
    contact_no_radius_rate = (
        float(contact_no_radius_count) / float(failure_done) if failure_done > 0 else None
    )
    extra_metrics = {
        "success_closest_target_dist_m": success_closest_dist,
        "failure_closest_target_dist_m": failure_closest_dist,
        "mean_surface_gap_m": mean_surface_gap,
        "failure_surface_gap_m": failure_surface_gap,
        "near_miss_count": float(near_miss_count),
        "near_miss_rate": near_miss_rate,
        "radius_no_contact_count": float(radius_no_contact_count),
        "radius_no_contact_rate": radius_no_contact_rate,
        "contact_no_radius_count": float(contact_no_radius_count),
        "contact_no_radius_rate": contact_no_radius_rate,
    }
    _EPOCH_DONE = 0
    _EPOCH_SUCC = 0
    _EPOCH_ENVS_WITH_SUCC = set()
    _EPOCH_DIST_SUM = 0.0
    _EPOCH_DIST_COUNT = 0
    _EPOCH_CLOSEST_DIST_SUM = 0.0
    _EPOCH_CLOSEST_DIST_COUNT = 0
    _EPOCH_SUCCESS_CLOSEST_DIST_SUM = 0.0
    _EPOCH_SUCCESS_CLOSEST_DIST_COUNT = 0
    _EPOCH_FAILURE_CLOSEST_DIST_SUM = 0.0
    _EPOCH_FAILURE_CLOSEST_DIST_COUNT = 0
    _EPOCH_SURFACE_GAP_SUM = 0.0
    _EPOCH_SURFACE_GAP_COUNT = 0
    _EPOCH_FAILURE_SURFACE_GAP_SUM = 0.0
    _EPOCH_FAILURE_SURFACE_GAP_COUNT = 0
    _EPOCH_NEAR_MISS_COUNT = 0
    _EPOCH_RADIUS_NO_CONTACT_COUNT = 0
    _EPOCH_CONTACT_NO_RADIUS_COUNT = 0

    envs = max(1, int(num_parallel_envs))
    if envs <= 1:
        envs = _parallel_env_count_hint()

    lines = [_rate_line(succ, done)]
    if mean_dist is not None:
        lines.append(f"mean target dist (m) : {mean_dist:.3f}")
    if mean_closest_dist is not None:
        lines.append(f"closest dist/ep (m) : {mean_closest_dist:.3f}")
    if failure_closest_dist is not None:
        lines.append(f"fail closest (m)    : {failure_closest_dist:.3f}")
    if failure_surface_gap is not None:
        lines.append(f"fail surface gap (m): {failure_surface_gap:.3f}")
    if failure_done > 0:
        lines.append(
            f"near-miss failures   : {100.0 * float(extra_metrics['near_miss_rate'] or 0.0):5.1f}% "
            f"({near_miss_count}/{failure_done})"
        )
    if failure_done > 0:
        lines.append(
            f"radius/no-contact    : {100.0 * float(extra_metrics['radius_no_contact_rate'] or 0.0):5.1f}% "
            f"({radius_no_contact_count}/{failure_done})"
        )
        lines.append(
            f"contact/out-radius   : {100.0 * float(extra_metrics['contact_no_radius_rate'] or 0.0):5.1f}% "
            f"({contact_no_radius_count}/{failure_done})"
        )

    return (lines, succ, done, envs_hit, envs, mean_dist, mean_closest_dist, extra_metrics)
