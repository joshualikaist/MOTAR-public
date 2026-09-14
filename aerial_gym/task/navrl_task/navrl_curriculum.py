"""Pure density-curriculum helpers shared by NavRLTask and CPU unit tests."""

from __future__ import annotations

from typing import Any, Mapping


def parse_density_threshold_schedule(spec):
    """Parse ``"70:0.82,85:0.77,100:0.72,115:0.70"`` into sorted (bars, threshold) knots.

    Returns an empty tuple for an unset/blank spec so callers fall back to the linear ramp.
    Raises ValueError on a malformed entry: a silently ignored schedule would train against a
    different promotion gate than the one written down, which is exactly the class of confound
    this project keeps paying for.
    """
    if spec is None:
        return ()
    text = str(spec).strip()
    if not text:
        return ()
    knots = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"density threshold schedule entry {item!r} must look like '<bars>:<threshold>'"
            )
        bars_text, thr_text = item.split(":", 1)
        try:
            bars = int(bars_text.strip())
            threshold = float(thr_text.strip())
        except ValueError as exc:
            raise ValueError(
                f"density threshold schedule entry {item!r} is not '<int>:<float>'"
            ) from exc
        if bars < 0:
            raise ValueError(f"density threshold schedule bar count must be >= 0, got {bars}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"density threshold schedule value must be a capture rate in [0,1], got {threshold}"
            )
        knots.append((bars, threshold))
    if not knots:
        return ()
    knots.sort(key=lambda kv: kv[0])
    return tuple(knots)


def density_threshold_from_schedule(n_bars_active, knots):
    """Threshold held from the highest knot at or below the active density (step schedule).

    Step, not interpolation: the density curriculum advances in fixed jumps, so every density it
    can actually occupy is expected to BE a knot. Holding the previous knot in between makes an
    off-schedule density (e.g. a resume at a hand-set NAVRL_NUM_BARS) fail safe by keeping the
    stricter gate it already earned rather than inventing an easier one.
    """
    if not knots:
        raise ValueError("empty threshold schedule")
    bars = int(n_bars_active)
    value = knots[0][1]
    for knot_bars, knot_threshold in knots:
        if bars >= knot_bars:
            value = knot_threshold
        else:
            break
    return float(value)


def density_threshold_at(
    n_bars_active: int,
    n_start: int,
    n_final: int,
    threshold_start: float,
    threshold_end: float,
    schedule=(),
) -> float:
    """Capture gate for the active density.

    An explicit ``schedule`` wins when present -- the measured ceiling is not linear in density,
    so a straight ramp between two endpoints cannot express it. Without one, fall back to the
    linear interpolation between the configured endpoints.
    """
    if schedule:
        return density_threshold_from_schedule(n_bars_active, schedule)
    if int(n_final) <= int(n_start):
        return float(threshold_start)
    fraction = (int(n_bars_active) - int(n_start)) / float(int(n_final) - int(n_start))
    fraction = min(1.0, max(0.0, fraction))
    return float(threshold_start) + fraction * (
        float(threshold_end) - float(threshold_start)
    )


def track_best_reward_by_density(state, n_bars_active, mean_reward):
    """Running best reward *within the current density*, and the record of each finished density.

    ``stability/best_reward`` is a single running max over the whole run, but a density promotion
    changes the reward scale underneath it: more bars means more collisions and a harder capture,
    so the global max is set at one density and then frozen, and every later epoch is compared
    against a number that was earned under an easier task. After the first promotion the scalar
    stops carrying information about whether learning is still progressing.

    This tracks the best per density instead, and reports each density's best at the moment it is
    left behind, which is the number that actually says "this is how well the policy ever did at
    70 bars".

    Returns ``(new_state, finished)`` where ``finished`` is ``None``, or ``(bars, best)`` for the
    density just departed. ``state`` is treated as immutable.
    """
    current = None if state is None else state.get("current_bars")
    best = None if state is None else state.get("best")
    history = () if state is None else tuple(state.get("history") or ())

    if n_bars_active is None:
        return (
            {"current_bars": current, "best": best, "history": history},
            None,
        )

    bars = int(n_bars_active)
    reward = None if mean_reward is None else float(mean_reward)
    # A non-finite reward (NaN/Inf from a diverged update) must never become a "best": it would
    # poison the comparison for the rest of the density.
    if reward is not None and not (reward == reward and abs(reward) != float("inf")):
        reward = None

    finished = None
    if current is None:
        return ({"current_bars": bars, "best": reward, "history": history}, None)

    if bars != current:
        if best is not None:
            finished = (int(current), float(best))
            history = history + (finished,)
        return ({"current_bars": bars, "best": reward, "history": history}, finished)

    if reward is not None and (best is None or reward > best):
        best = reward
    return ({"current_bars": current, "best": best, "history": history}, None)


def density_dwell_epochs(
    num_task_steps: int,
    level_start_steps: int,
    ppo_horizon: int,
) -> float:
    """Return non-negative PPO epochs spent at the current density."""
    elapsed_steps = max(0, int(num_task_steps) - int(level_start_steps))
    return elapsed_steps / float(max(1, int(ppo_horizon)))


def density_dwell_ready(
    num_task_steps: int,
    level_start_steps: int,
    ppo_horizon: int,
    min_epochs: int,
) -> bool:
    return density_dwell_epochs(
        num_task_steps,
        level_start_steps,
        ppo_horizon,
    ) >= max(0, int(min_epochs))


def density_level_start_after_promotion(
    previous_start_steps: int,
    num_task_steps: int,
    promoted: bool,
) -> int:
    """Reset the dwell clock exactly when a new density becomes active."""
    if promoted:
        return max(0, int(num_task_steps))
    return max(0, int(previous_start_steps))


def restore_density_level_start_steps(
    state: Mapping[str, Any],
    num_task_steps: int,
) -> int:
    """Restore a saved dwell clock; old checkpoints conservatively start it now."""
    current_steps = max(0, int(num_task_steps))
    saved = state.get("density_level_start_steps")
    if saved is None:
        return current_steps
    # A malformed/future clock would create a negative dwell. Clamp it into the run history.
    return min(current_steps, max(0, int(saved)))
