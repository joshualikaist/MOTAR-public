"""TRAIN: per-epoch NavRL navigation stats for the console dashboard + TensorBoard.

Mirrors the intercept train_dashboard pattern: the task records every finished episode
(captured / crash / timeout / closest approach) into module-level accumulators each step,
and the training agent consumes them once per PPO epoch to render dashboard lines and
TensorBoard scalars (navrl/*).

Closest-approach is aggregated over NON-CRASH episodes only: a crash dies far from the goal
and would only inflate the mean, hiding how close the drone actually gets when it is really
approaching. We also surface the single best (min) approach in the window.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

_DONE = 0
_SUCC_TIMEOUT = 0
_CRASH = 0
_TIMEOUT = 0
_CLOSEST_NC_SUM = 0.0  # sum of per-episode closest approach over NON-CRASH finished episodes
_CLOSEST_NC_COUNT = 0
_CLOSEST_MIN: Optional[float] = None  # best (min) closest approach over non-crash episodes
_GOAL_DIST_MAX: Optional[float] = None
_GOAL_DIST_MIN: Optional[float] = None
_N_BARS_ACTIVE: Optional[int] = None
_TARGET_SPEED_MAX: Optional[float] = None   # Phase 3: current target-speed curriculum ceiling
_TARGET_SPEED_MEAN: Optional[float] = None  # Phase 3: mean per-episode target speed
_TARGET_SPEED_REALIZED_MEAN: Optional[float] = None


def record_navrl_epoch_episodes(
    num_finished: int,
    num_captured: int,
    num_crash: int,
    num_timeout: int,
    closest_nocrash_sum: float,
    closest_nocrash_count: int,
    closest_min: Optional[float],
    goal_dist_max: Optional[float] = None,
    goal_dist_min: Optional[float] = None,
    n_bars_active: Optional[int] = None,
    target_speed_max: Optional[float] = None,
    target_speed_mean: Optional[float] = None,
    target_speed_realized_mean: Optional[float] = None,
) -> None:
    global _DONE, _SUCC_TIMEOUT, _CRASH, _TIMEOUT
    global _CLOSEST_NC_SUM, _CLOSEST_NC_COUNT, _CLOSEST_MIN, _GOAL_DIST_MAX, _GOAL_DIST_MIN
    global _N_BARS_ACTIVE, _TARGET_SPEED_MAX, _TARGET_SPEED_MEAN
    global _TARGET_SPEED_REALIZED_MEAN
    if num_finished <= 0:
        return
    _DONE += int(num_finished)
    _SUCC_TIMEOUT += int(num_captured)
    _CRASH += int(num_crash)
    _TIMEOUT += int(num_timeout)
    if closest_nocrash_count > 0:
        _CLOSEST_NC_SUM += float(closest_nocrash_sum)
        _CLOSEST_NC_COUNT += int(closest_nocrash_count)
    if closest_min is not None:
        _CLOSEST_MIN = closest_min if _CLOSEST_MIN is None else min(_CLOSEST_MIN, closest_min)
    if goal_dist_max is not None:
        _GOAL_DIST_MAX = float(goal_dist_max)
    if goal_dist_min is not None:
        _GOAL_DIST_MIN = float(goal_dist_min)
    if n_bars_active is not None:
        _N_BARS_ACTIVE = int(n_bars_active)
    if target_speed_max is not None:
        _TARGET_SPEED_MAX = float(target_speed_max)
    if target_speed_mean is not None:
        _TARGET_SPEED_MEAN = float(target_speed_mean)
    if target_speed_realized_mean is not None:
        _TARGET_SPEED_REALIZED_MEAN = float(target_speed_realized_mean)


def consume_navrl_epoch_summary() -> Tuple[List[str], dict, int]:
    """Return (dashboard lines, TensorBoard scalars {navrl/<name>: value}, episodes done)."""
    global _DONE, _SUCC_TIMEOUT, _CRASH, _TIMEOUT
    global _CLOSEST_NC_SUM, _CLOSEST_NC_COUNT, _CLOSEST_MIN, _GOAL_DIST_MAX, _GOAL_DIST_MIN
    global _N_BARS_ACTIVE, _TARGET_SPEED_MAX, _TARGET_SPEED_MEAN
    global _TARGET_SPEED_REALIZED_MEAN
    done = _DONE
    succ = _SUCC_TIMEOUT
    crash = _CRASH
    timeout = _TIMEOUT
    closest_nc = _CLOSEST_NC_SUM / float(_CLOSEST_NC_COUNT) if _CLOSEST_NC_COUNT > 0 else None
    closest_min = _CLOSEST_MIN
    goal_dist_max = _GOAL_DIST_MAX
    goal_dist_min = _GOAL_DIST_MIN
    n_bars_active = _N_BARS_ACTIVE
    target_speed_max = _TARGET_SPEED_MAX
    target_speed_mean = _TARGET_SPEED_MEAN
    target_speed_realized_mean = _TARGET_SPEED_REALIZED_MEAN
    _DONE = _SUCC_TIMEOUT = _CRASH = _TIMEOUT = 0
    _CLOSEST_NC_SUM = 0.0
    _CLOSEST_NC_COUNT = 0
    _CLOSEST_MIN = None

    if done <= 0:
        return ([], {}, 0)

    def pct(k: int) -> str:
        return f"{100.0 * k / done:5.1f}% ({k}/{done})"

    lines = [
        f"captured (success)   : {pct(succ)}",
        f"crash                : {pct(crash)}",
        f"timeout (no capture) : {pct(timeout)}",
    ]
    if closest_nc is not None:
        best = f"   (best {closest_min:.2f})" if closest_min is not None else ""
        lines.append(f"closest, no crash (m): {closest_nc:.2f}{best}")
    if goal_dist_max is not None:
        gmin = f"{goal_dist_min:.1f}" if goal_dist_min is not None else "?"
        lines.append(f"curriculum (m)       : k_min {gmin}, k_max {goal_dist_max:.1f}")
    if n_bars_active is not None:
        lines.append(f"density bars         : {n_bars_active}")
    if target_speed_max is not None:
        tmean = f"{target_speed_mean:.2f}" if target_speed_mean is not None else "?"
        treal = (
            f"{target_speed_realized_mean:.2f}"
            if target_speed_realized_mean is not None
            else "?"
        )
        lines.append(
            f"target speed (m/s)   : command {tmean}, realized {treal}, "
            f"max {target_speed_max:.2f}"
        )

    # Interception mode: reach_rate == captured_rate (touching the radius ends the episode),
    # so only captured_rate is logged. crash/timeout/captured partition every finished episode.
    metrics = {
        "navrl/captured_rate": succ / done,
        "navrl/crash_rate": crash / done,
        "navrl/timeout_rate": timeout / done,
    }
    if closest_nc is not None:
        metrics["navrl/closest_nocrash_m"] = closest_nc
    if closest_min is not None:
        metrics["navrl/closest_min_m"] = closest_min
    if goal_dist_max is not None:
        metrics["navrl/curriculum_goal_dist_max_m"] = goal_dist_max
    if goal_dist_min is not None:
        metrics["navrl/curriculum_goal_dist_min_m"] = goal_dist_min
    if n_bars_active is not None:
        metrics["navrl/n_bars_active"] = n_bars_active
    if target_speed_max is not None:
        metrics["navrl/target_speed_max_m_s"] = target_speed_max
    if target_speed_mean is not None:
        metrics["navrl/target_speed_mean_m_s"] = target_speed_mean
    if target_speed_realized_mean is not None:
        metrics["navrl/target_speed_realized_mean_m_s"] = target_speed_realized_mean
    return (lines, metrics, done)
