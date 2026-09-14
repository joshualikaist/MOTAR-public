"""Simulator-independent local steering for NavRL's virtual moving target."""

import importlib.util
import math
from pathlib import Path
import sys

import torch


def _load_target_route_geometry():
    """Load geometry without importing the aerial_gym package (Isaac Gym stays out of CPU tests)."""
    name = "_navrl_target_route_geometry_standalone"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).with_name("target_route_geometry.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_GEO = _load_target_route_geometry()
terminal_stop_certificate = _GEO.terminal_stop_certificate
torch_segments_soft_safe = _GEO.torch_segments_soft_safe


TARGET_MOTION_MODEL = "symmetric_local_steer_v2_heading_continuity90"
BOUNDED_TARGET_MOTION_MODEL = "bounded_planar_drone_v1_rollout"
PHYSICAL_TARGET_MOTION_MODEL = "physx_ref5in_6dof_motor_wrench_v2_same_substep"


def support_aware_bounds(bounds_min_xy, bounds_max_xy, base_margin, support_xy):
    """Return center bounds that keep an oriented target support inside the arena.

    ``base_margin`` is the existing wall reserve. ``support_xy`` is the current world-axis OBB
    support radius; it is deliberately passed in by the caller because a tilted rigid body can
    change its XY support during flight.  This helper is geometry-only and does not clamp or
    teleport a target.
    """
    if bounds_min_xy.shape != bounds_max_xy.shape or bounds_min_xy.shape != support_xy.shape:
        raise ValueError("bounds_min_xy, bounds_max_xy, and support_xy must have matching shapes")
    if bounds_min_xy.ndim != 2 or bounds_min_xy.shape[1] != 2:
        raise ValueError("bounds and support must have shape [N, 2]")
    margin = float(base_margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("base_margin must be finite and non-negative")
    return (
        bounds_min_xy + margin + support_xy,
        bounds_max_xy - margin - support_xy,
    )


# Symmetric candidates keep obstacle avoidance from introducing a global left/right bias. The
# per-episode turn_sign only breaks exact +/- ties and is sampled 50:50 by NavRLTask.
TURN_ANGLES_DEG = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 180.0)
BOUNDED_TURN_ANGLES_DEG = (
    0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0,
    75.0, -75.0, 90.0, -90.0, 105.0, -105.0, 120.0, -120.0,
    135.0, -135.0, 150.0, -150.0, 165.0, -165.0, 180.0,
)
HEADING_CONTINUITY_RAD = math.radians(90.0)
# Residual PhysX speed is not a heading of travel.  This is the same 0.10 m/s threshold the
# frozen braking probe uses as "stopped" (STOP_THRESHOLD_MPS).  A 1e-5 m/s epsilon treats
# millimetre-per-second hover noise as a real course and slews CONNECT away from the anchor.
HEADING_VALID_SPEED_MPS = 0.10
# --- checkpoint provenance for the threshold above -------------------------------------------
# The threshold applies to *all* bounded/physical target motion, not just the recovery path, so a
# future run that edits it silently changes what every moving-target metric means.  These keys
# put it in the same env_state contract as ``cfg_target_motion_model`` so a resume either attests
# the value or is refused.
HEADING_VALID_SPEED_KEY = "cfg_target_motion_heading_valid_speed_mps"
HEADING_VALID_SPEED_PROVENANCE_KEY = "cfg_target_motion_heading_valid_speed_provenance"
# Written by a run that never restored a checkpoint: the value is the running literal, full stop.
HEADING_VALID_SPEED_SOURCE = "source_literal"
# The checkpoint carried the key and it matched the running literal.  This is a measurement.
HEADING_VALID_SPEED_ATTESTED = "checkpoint_attested"
# The checkpoint predates the key.  The running literal is ASSUMED to be what it trained under.
# This is not a measurement and must never be reported as one.
HEADING_VALID_SPEED_ASSUMED = "assumed_pre_key_default"
# Tight enough to catch any deliberate retune (0.05, 0.11, ...) while ignoring nothing real: the
# value round-trips through pickle as an exact Python float, so equal values compare exactly.
HEADING_VALID_SPEED_TOLERANCE = 1e-9


def resolve_heading_valid_speed_contract(state, current=None):
    """Resolve the heading-validity threshold contract for a restored ``env_state``.

    Returns ``(value_in_force, provenance, message)``.  ``value_in_force`` is always the running
    literal, because that is what the simulator will actually use; ``provenance`` says whether the
    checkpoint attested to it or whether it was merely assumed, so a reader can never mistake an
    assumption for a measurement.

    Raises ``RuntimeError`` when the checkpoint carries a *different* value, matching the
    fail-closed treatment the routed-target contract fields already get.
    """
    running = float(HEADING_VALID_SPEED_MPS if current is None else current)
    saved = None if not isinstance(state, dict) else state.get(HEADING_VALID_SPEED_KEY)

    if saved is None:
        # Every checkpoint written before this key existed lands here.  Do not refuse: refusing
        # would void every existing checkpoint.  Do label it, loudly.
        return (
            running,
            HEADING_VALID_SPEED_ASSUMED,
            "NavRL TARGET HEADING CONTRACT | value_in_force=%.10g provenance=%s. "
            "The checkpoint predates this key, so %.10g is ASSUMED, not attested: it is the "
            "value the running code holds, not a value the checkpoint recorded. Note that "
            "checkpoints saved before the threshold was introduced ran an inline 1e-05 m/s "
            "epsilon, so an older lineage may not in fact have trained under %.10g."
            % (running, HEADING_VALID_SPEED_ASSUMED, running, running),
        )

    try:
        saved_value = float(saved)
    except (TypeError, ValueError):
        raise RuntimeError(
            "NavRL TARGET HEADING CONTRACT | checkpoint=%r is not a number. The heading-validity "
            "threshold provenance is unreadable, so the target-motion contract cannot be "
            "attested." % (saved,)
        )

    if not abs(saved_value - running) <= HEADING_VALID_SPEED_TOLERANCE:
        raise RuntimeError(
            "NavRL TARGET HEADING MISMATCH | checkpoint=%.10g running=%.10g. The heading-validity "
            "threshold decides when residual speed counts as a heading of travel; changing it "
            "changes what every moving-target metric means, so this is a changed environment "
            "contract and not a resumable one." % (saved_value, running)
        )

    return (
        running,
        HEADING_VALID_SPEED_ATTESTED,
        "NavRL TARGET HEADING CONTRACT | value_in_force=%.10g provenance=%s checkpoint=%.10g."
        % (running, HEADING_VALID_SPEED_ATTESTED, saved_value),
    )


CV_INITIAL_HEADING_MODES = (
    "random",
    "toward",
    "tangent_left",
    "tangent_right",
    "away",
)


def limit_planar_velocity(
    current_velocity,
    desired_velocity,
    speed_limit,
    dt,
    max_accel,
    max_turn_rate,
):
    """Apply a planar multirotor trajectory envelope for one control interval.

    The bound is deliberately imposed on the *realized velocity state*, rather than merely on a
    waypoint heading.  Acceleration is Euclidean (so diagonal commands get no free authority).
    Below ``HEADING_VALID_SPEED_MPS`` there is no heading of travel, so the first command starts
    in the requested direction; above that speed the heading slew limit applies.  This is a
    trackable-trajectory contract, not a rigid-body simulation: attitude, motor and battery
    states still belong to the actual robot actor.
    """
    if current_velocity.shape != desired_velocity.shape or current_velocity.ndim != 2:
        raise ValueError("current_velocity and desired_velocity must have matching [N, 2] shape")
    n = current_velocity.shape[0]
    for name, value in (("speed_limit", speed_limit), ("max_accel", max_accel),
                        ("max_turn_rate", max_turn_rate)):
        if value.shape != (n,):
            raise ValueError(f"{name} must have shape [N]")
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive")

    desired_speed = desired_velocity.norm(dim=1).minimum(speed_limit.clamp(min=0.0))
    desired_heading = torch.atan2(desired_velocity[:, 1], desired_velocity[:, 0])
    current_speed = current_velocity.norm(dim=1)
    current_heading = torch.atan2(current_velocity[:, 1], current_velocity[:, 0])
    # At rest there is no physical heading of travel. Starting in the requested direction is
    # valid; acceleration below still ramps the speed from zero.  The rest threshold is the
    # frozen braking stop speed, not a numeric epsilon: PhysX residuals of a few mm/s are
    # typical after a commanded stop and must not lock the slew limiter onto a noise heading.
    heading_delta = torch.atan2(
        torch.sin(desired_heading - current_heading),
        torch.cos(desired_heading - current_heading),
    )
    max_delta = max_turn_rate.clamp(min=0.0) * float(dt)
    limited_heading = current_heading + heading_delta.clamp(min=-max_delta, max=max_delta)
    limited_heading = torch.where(
        current_speed > HEADING_VALID_SPEED_MPS, limited_heading, desired_heading
    )
    heading_limited_target = torch.stack(
        (torch.cos(limited_heading), torch.sin(limited_heading)), dim=1
    ) * desired_speed.unsqueeze(1)

    delta_v = heading_limited_target - current_velocity
    delta_norm = delta_v.norm(dim=1, keepdim=True)
    max_delta_v = (max_accel.clamp(min=0.0) * float(dt)).unsqueeze(1)
    delta_v = delta_v * torch.minimum(
        torch.ones_like(delta_norm), max_delta_v / delta_norm.clamp(min=1e-9)
    )
    velocity = current_velocity + delta_v
    velocity_norm = velocity.norm(dim=1, keepdim=True)
    velocity = velocity * torch.minimum(
        torch.ones_like(velocity_norm),
        speed_limit.clamp(min=0.0).unsqueeze(1) / velocity_norm.clamp(min=1e-9),
    )
    return velocity


def bounded_drone_target_step(
    old_xy,
    current_velocity,
    desired_velocity,
    speed_limit,
    dt,
    bars_xy,
    lo,
    hi,
    clearance,
    turn_sign,
    max_accel,
    max_turn_rate,
    lookahead_s,
    bars_half_extents_xy=None,
    exact_aabb_clearance=False,
    hard_epsilon_m=0.0,
    return_certificate=False,
    certificate_row_ids=None,
):
    """Choose and execute one dynamically bounded, collision-screened target step.

    Every steering candidate is rolled forward with the same acceleration and heading-rate bounds
    that govern the returned first step.  A candidate is considered feasible only when *all*
    rollout samples remain inside the arena and outside the configured bar clearance.  No wall
    reflection, positional clamp, obstacle push-out, or instantaneous velocity rewrite occurs.

    Returns ``(new_xy, new_velocity, steered, immediate_feasible)``. The final boolean says only
    whether the selected receding-horizon candidate's *first* step is safe.  If no candidate is
    safe for the complete lookahead, selection falls back to the longest safe prefix and replans
    next RL step; a safe first step therefore remains ``True``.  Do not interpret this value as a
    full-route or full-lookahead feasibility certificate.
    """
    if old_xy.ndim != 2 or old_xy.shape[1] != 2:
        raise ValueError("old_xy must have shape [N, 2]")
    n = old_xy.shape[0]
    if current_velocity.shape != old_xy.shape or desired_velocity.shape != old_xy.shape:
        raise ValueError("velocity tensors must match old_xy")
    if bars_xy.ndim != 3 or bars_xy.shape[0] != n or bars_xy.shape[2] != 2:
        raise ValueError("bars_xy must have shape [N, B, 2]")
    if bars_half_extents_xy is not None and bars_half_extents_xy.shape != bars_xy.shape:
        raise ValueError("bars_half_extents_xy must match bars_xy")
    if lo.shape != old_xy.shape or hi.shape != old_xy.shape:
        raise ValueError("lo and hi must match old_xy")
    if float(lookahead_s) < float(dt):
        raise ValueError("lookahead_s must be at least dt")
    if not math.isfinite(float(hard_epsilon_m)) or float(hard_epsilon_m) < 0.0:
        raise ValueError("hard_epsilon_m must be finite and non-negative")
    if certificate_row_ids is not None:
        if not return_certificate:
            raise ValueError("certificate_row_ids requires return_certificate=True")
        if certificate_row_ids.shape != (n,):
            raise ValueError("certificate_row_ids must have shape [N]")

    base_norm = desired_velocity.norm(dim=1, keepdim=True)
    fallback = torch.zeros_like(desired_velocity)
    fallback[:, 0] = 1.0
    base = torch.where(base_norm > 1e-6, desired_velocity / base_norm.clamp(min=1e-6), fallback)
    angles = torch.tensor(
        [math.radians(value) for value in BOUNDED_TURN_ANGLES_DEG],
        device=old_xy.device,
        dtype=old_xy.dtype,
    )
    cosine, sine = torch.cos(angles).view(1, -1), torch.sin(angles).view(1, -1)
    bx, by = base[:, 0:1], base[:, 1:2]
    directions = torch.stack((bx * cosine - by * sine, bx * sine + by * cosine), dim=2)
    # A real vehicle must be allowed to trade speed for clearance. Direction-only full-speed
    # candidates recreate the old failure in a smoother form: when no full-speed arc fits, every
    # rollout becomes infeasible even though braking would be safe. Use full and half-speed
    # copies plus one stop command. Acceleration bounds make the stop a deceleration trajectory,
    # never an instantaneous halt.
    cruise_scales = torch.tensor((1.0, 0.5, 0.25), device=old_xy.device, dtype=old_xy.dtype)
    directions = directions.unsqueeze(1).expand(-1, len(cruise_scales), -1, -1)
    candidate_angles = angles.view(1, 1, -1).expand(n, len(cruise_scales), -1).reshape(n, -1)
    candidate_scales = cruise_scales.view(1, -1, 1).expand(n, -1, len(angles)).reshape(n, -1)
    candidates = (
        directions * cruise_scales.view(1, -1, 1, 1)
        * speed_limit.clamp(min=0.0).view(n, 1, 1, 1)
    ).reshape(n, -1, 2)
    candidates = torch.cat((candidates, torch.zeros(n, 1, 2, device=old_xy.device,
                                                     dtype=old_xy.dtype)), dim=1)
    candidate_angles = torch.cat((candidate_angles, torch.zeros(n, 1, device=old_xy.device,
                                                                 dtype=old_xy.dtype)), dim=1)
    candidate_scales = torch.cat((candidate_scales, torch.zeros(n, 1, device=old_xy.device,
                                                                 dtype=old_xy.dtype)), dim=1)
    count = candidates.shape[1]

    pos = old_xy.unsqueeze(1).expand(-1, count, -1).clone()
    vel = current_velocity.unsqueeze(1).expand(-1, count, -1).clone()
    first_pos = None
    first_vel = None
    feasible = torch.ones((n, count), dtype=torch.bool, device=old_xy.device)
    immediate_feasible = torch.zeros((n, count), dtype=torch.bool, device=old_xy.device)
    safe_prefix_steps = torch.zeros((n, count), dtype=old_xy.dtype, device=old_xy.device)
    prefix_alive = torch.ones((n, count), dtype=torch.bool, device=old_xy.device)
    min_clearance = torch.full(
        (n, count), float("inf"), device=old_xy.device, dtype=old_xy.dtype
    )
    steps = max(1, int(math.ceil(float(lookahead_s) / float(dt))))
    flat_speed = speed_limit.unsqueeze(1).expand(-1, count).reshape(-1)
    flat_accel = max_accel.unsqueeze(1).expand(-1, count).reshape(-1)
    flat_turn = max_turn_rate.unsqueeze(1).expand(-1, count).reshape(-1)
    flat_desired = candidates.reshape(-1, 2)
    for step in range(steps):
        previous_pos = pos
        vel = limit_planar_velocity(
            vel.reshape(-1, 2), flat_desired, flat_speed, dt, flat_accel, flat_turn
        ).reshape(n, count, 2)
        pos = pos + vel * float(dt)
        if step == 0:
            first_pos, first_vel = pos.clone(), vel.clone()
        inside = ((pos >= lo.unsqueeze(1)) & (pos <= hi.unsqueeze(1))).all(dim=2)
        step_safe = inside
        if bars_xy.shape[1] > 0 and (float(clearance) > 0.0 or exact_aabb_clearance):
            if bars_half_extents_xy is None:
                dist = torch.cdist(pos, bars_xy).amin(dim=2)
                step_safe &= dist >= float(clearance) + 1e-4
            else:
                delta = (
                    (pos.unsqueeze(2) - bars_xy.unsqueeze(1)).abs()
                    - bars_half_extents_xy.unsqueeze(1)
                )
                if exact_aabb_clearance:
                    # Recovery's hard envelope is a closed AABB.  The normal route path retains
                    # the historical rounded Euclidean clearance behavior; callers must opt in
                    # explicitly so legacy/bounded transitions remain byte-compatible.
                    delta = delta - float(hard_epsilon_m)
                    inside = (delta <= 0.0).all(dim=3)
                    dist = delta.clamp(min=0.0).amax(dim=3).amin(dim=2)
                    step_safe &= ~inside.any(dim=2)
                    # Closed-AABB slab test on every continuous substep segment. Endpoint
                    # checks alone can miss a diagonal corner crossing.
                    p0 = previous_pos.unsqueeze(2)
                    p1 = pos.unsqueeze(2)
                    direction = p1 - p0
                    box_lo = bars_xy.unsqueeze(1) - bars_half_extents_xy.unsqueeze(1) - float(hard_epsilon_m)
                    box_hi = bars_xy.unsqueeze(1) + bars_half_extents_xy.unsqueeze(1) + float(hard_epsilon_m)
                    parallel = direction.abs() <= 1e-9
                    safe_parallel = (~parallel) | ((p0 >= box_lo) & (p0 <= box_hi))
                    t0 = torch.where(parallel, torch.full_like(direction, float("-inf")), (box_lo - p0) / direction)
                    t1 = torch.where(parallel, torch.full_like(direction, float("inf")), (box_hi - p0) / direction)
                    t_enter = torch.maximum(torch.minimum(t0, t1)[:, :, :, 0], torch.minimum(t0, t1)[:, :, :, 1])
                    t_exit = torch.minimum(torch.maximum(t0, t1)[:, :, :, 0], torch.maximum(t0, t1)[:, :, :, 1])
                    segment_hits = safe_parallel.all(dim=3) & (t_enter <= t_exit) & (t_exit >= 0.0) & (t_enter <= 1.0)
                    step_safe &= ~segment_hits.any(dim=2)
                else:
                    dist = delta.clamp(min=0.0).norm(dim=3).amin(dim=2)
                    step_safe &= dist >= float(clearance) + 1e-4
            min_clearance = torch.minimum(min_clearance, dist)
        if step == 0:
            immediate_feasible[:] = step_safe
        prefix_alive &= step_safe
        safe_prefix_steps += prefix_alive.to(old_xy.dtype)
        feasible &= step_safe

    turn_cost = candidate_angles.abs()
    tie = turn_sign.view(n, 1) * torch.sign(candidate_angles) * 1e-3
    clear_score = 1000.0 + 10.0 * candidate_scales - turn_cost + tie
    # A trapped rollout is not made "physical" by pretending it succeeded. Select the trajectory
    # with the most clearance and expose feasible=False to telemetry/tests.
    boundary_margin = torch.minimum(pos - lo.unsqueeze(1), hi.unsqueeze(1) - pos).amin(dim=2)
    # Receding-horizon fallback: if no constant-heading candidate survives the whole horizon,
    # prefer the candidate with the longest safe prefix, then replan next RL step. The old score
    # used only final clearance and could select a candidate unsafe on its very first step merely
    # because it ended farther from another obstacle.
    trapped_score = (
        100.0 * safe_prefix_steps
        + min_clearance.clamp(max=10.0)
        + boundary_margin.clamp(max=10.0)
        - 0.01 * turn_cost
        + tie
    )
    score = torch.where(feasible, clear_score, trapped_score)
    chosen = score.argmax(dim=1)
    rows = torch.arange(n, device=old_xy.device)
    selected_pos = first_pos[rows, chosen]
    selected_vel = first_vel[rows, chosen]
    base_result = (selected_pos, selected_vel, chosen != 0, immediate_feasible[rows, chosen])
    if not return_certificate:
        return base_result
    certificate = {
        "selected_index": chosen,
        "candidate_count": count,
        "horizon_steps": steps,
        "safe_prefix_steps": safe_prefix_steps[rows, chosen].to(torch.int16),
        "full_horizon_safe": feasible[rows, chosen],
        "immediate_feasible": immediate_feasible[rows, chosen],
        # Recovery progress is a property of the certified horizon, not merely the command's
        # first sample.  Keep this opt-in field out of the legacy four-tuple/API path.
        "selected_final_position_xy": pos[rows, chosen].clone(),
        # Alias retained for the first recovery-v2 CPU fixture; both fields are certificate-only.
        "selected_final_pos": pos[rows, chosen].clone(),
        "row_ids": (
            certificate_row_ids.detach().clone()
            if certificate_row_ids is not None
            else rows.detach().clone()
        ),
    }
    return base_result + (certificate,)


def braking_aware_route_step(
    old_xy,
    current_velocity,
    desired_velocity,
    speed_limit,
    dt,
    bars_xy,
    admissible_lo,
    admissible_hi,
    inflated_bar_half_xy,
    turn_sign,
    max_accel,
    max_turn_rate,
    lookahead_s,
    brake_speed_knots,
    brake_distance_knots,
    lateral_tube_m,
):
    """Fresh-v3 route step with a recursively safe terminal-stop certificate.

    Unlike :func:`bounded_drone_target_step`, this function never executes a longest-safe-prefix
    candidate.  A progress command is eligible only when its complete rollout and a subsequent
    measured zero-command stop stay inside the same exact soft envelope.  If no progress command
    is certified, the zero-command candidate is selected as a pre-brake action when certified.
    """
    if old_xy.ndim != 2 or old_xy.shape[1] != 2:
        raise ValueError("old_xy must have shape [N,2]")
    n = old_xy.shape[0]
    if current_velocity.shape != old_xy.shape or desired_velocity.shape != old_xy.shape:
        raise ValueError("velocity tensors must match old_xy")
    if admissible_lo.shape != old_xy.shape or admissible_hi.shape != old_xy.shape:
        raise ValueError("admissible bounds must match old_xy")
    if bars_xy.ndim != 3 or bars_xy.shape[0] != n or bars_xy.shape[2] != 2:
        raise ValueError("bars_xy must have shape [N,B,2]")
    if inflated_bar_half_xy.shape != bars_xy.shape:
        raise ValueError("inflated bar half extents must match bars_xy")
    if float(lookahead_s) < float(dt):
        raise ValueError("lookahead_s must be at least dt")

    base_norm = desired_velocity.norm(dim=1, keepdim=True)
    current_norm = current_velocity.norm(dim=1, keepdim=True)
    fallback = torch.zeros_like(desired_velocity)
    fallback[:, 0] = 1.0
    base = torch.where(
        base_norm > 1e-6,
        desired_velocity / base_norm.clamp(min=1e-6),
        torch.where(current_norm > 1e-6, current_velocity / current_norm.clamp(min=1e-6), fallback),
    )
    angles = torch.tensor(
        [math.radians(value) for value in BOUNDED_TURN_ANGLES_DEG],
        device=old_xy.device,
        dtype=old_xy.dtype,
    )
    cosine, sine = torch.cos(angles).view(1, -1), torch.sin(angles).view(1, -1)
    bx, by = base[:, 0:1], base[:, 1:2]
    directions = torch.stack((bx * cosine - by * sine, bx * sine + by * cosine), dim=2)
    cruise_scales = torch.tensor((1.0, 0.5, 0.25), device=old_xy.device, dtype=old_xy.dtype)
    directions = directions.unsqueeze(1).expand(-1, len(cruise_scales), -1, -1)
    candidate_angles = angles.view(1, 1, -1).expand(n, len(cruise_scales), -1).reshape(n, -1)
    candidate_scales = cruise_scales.view(1, -1, 1).expand(n, -1, len(angles)).reshape(n, -1)
    candidates = (
        directions
        * cruise_scales.view(1, -1, 1, 1)
        * speed_limit.clamp(min=0.0).view(n, 1, 1, 1)
    ).reshape(n, -1, 2)
    candidates = torch.cat(
        (candidates, torch.zeros((n, 1, 2), device=old_xy.device, dtype=old_xy.dtype)), dim=1
    )
    candidate_angles = torch.cat(
        (candidate_angles, torch.zeros((n, 1), device=old_xy.device, dtype=old_xy.dtype)), dim=1
    )
    candidate_scales = torch.cat(
        (candidate_scales, torch.zeros((n, 1), device=old_xy.device, dtype=old_xy.dtype)), dim=1
    )
    count = candidates.shape[1]
    pos = old_xy.unsqueeze(1).expand(-1, count, -1).clone()
    vel = current_velocity.unsqueeze(1).expand(-1, count, -1).clone()
    full_horizon_safe = torch.ones((n, count), dtype=torch.bool, device=old_xy.device)
    first_pos = pos.clone()
    first_vel = vel.clone()
    flat_speed = speed_limit.unsqueeze(1).expand(-1, count).reshape(-1)
    flat_accel = max_accel.unsqueeze(1).expand(-1, count).reshape(-1)
    flat_turn = max_turn_rate.unsqueeze(1).expand(-1, count).reshape(-1)
    flat_desired = candidates.reshape(-1, 2)
    horizon_steps = max(1, int(math.ceil(float(lookahead_s) / float(dt))))
    for step in range(horizon_steps):
        previous = pos
        vel = limit_planar_velocity(
            vel.reshape(-1, 2), flat_desired, flat_speed, dt, flat_accel, flat_turn
        ).reshape(n, count, 2)
        pos = pos + vel * float(dt)
        if step == 0:
            first_pos = pos.clone()
            first_vel = vel.clone()
        full_horizon_safe &= torch_segments_soft_safe(
            previous, pos, admissible_lo, admissible_hi, bars_xy, inflated_bar_half_xy
        )

    stop_safe, stop_xy, stop_distance = terminal_stop_certificate(
        pos,
        vel,
        admissible_lo,
        admissible_hi,
        bars_xy,
        inflated_bar_half_xy,
        brake_speed_knots,
        brake_distance_knots,
        lateral_tube_m,
    )
    certified = full_horizon_safe & stop_safe
    progress = candidate_scales > 0.0
    progress_certified = certified & progress
    # Prefer forward route progress, then the highest cruise scale, then the smallest turn.
    desired_dir = torch.where(
        base_norm > 1e-6, desired_velocity / base_norm.clamp(min=1e-6), base
    )
    route_progress = ((pos - old_xy.unsqueeze(1)) * desired_dir.unsqueeze(1)).sum(dim=2)
    tie = turn_sign.view(n, 1) * torch.sign(candidate_angles) * 1e-6
    score = (
        100.0 * route_progress
        + 10.0 * candidate_scales
        - candidate_angles.abs()
        + tie
    )
    score = torch.where(progress_certified, score, torch.full_like(score, float("-inf")))
    best_progress = score.argmax(dim=1)
    has_progress = progress_certified.any(dim=1)
    stop_index = count - 1
    stop_certified = certified[:, stop_index]
    chosen = torch.where(has_progress, best_progress, torch.full_like(best_progress, stop_index))
    rows = torch.arange(n, device=old_xy.device)
    accepted = has_progress | stop_certified
    # Uncertified rows still submit the zero-command first step so PhysX brakes with declared
    # dynamics.  Instantaneous velocity rewrite is not a certificate.
    selected_pos = first_pos[rows, chosen]
    selected_vel = first_vel[rows, chosen]
    prebrake = ~has_progress & stop_certified
    certificate_failure = ~accepted
    certificate = {
        "selected_index": chosen,
        "candidate_count": count,
        "horizon_steps": horizon_steps,
        "full_horizon_safe": full_horizon_safe[rows, chosen] & accepted,
        "terminal_stop_safe": stop_safe[rows, chosen] & accepted,
        "terminal_stop_xy": stop_xy[rows, chosen],
        "terminal_stop_distance_m": stop_distance[rows, chosen],
        "accepted": accepted,
        "prebrake": prebrake,
        "progress": has_progress,
        "certificate_failure": certificate_failure,
        "safe_prefix_rejected": (~full_horizon_safe).any(dim=1) & ~accepted,
    }
    return selected_pos, selected_vel, accepted, prebrake, certificate


def initial_cv_velocity(mode, speed, target_xy, pursuer_xy, random_angle):
    """Return a CV velocity under an evaluation-only radial-heading intervention.

    ``left`` and ``right`` are defined looking from the pursuer toward the target: left is a
    +90-degree rotation of the pursuer->target radial vector.  The caller always samples and
    supplies ``random_angle`` even for controlled cells so subsequent RNG draws stay aligned.
    """
    if mode not in CV_INITIAL_HEADING_MODES:
        raise ValueError(
            "unknown CV initial heading %r (expected %s)"
            % (mode, "|".join(CV_INITIAL_HEADING_MODES))
        )
    if target_xy.ndim != 2 or target_xy.shape[1] != 2:
        raise ValueError("target_xy must have shape [N, 2]")
    if pursuer_xy.shape != target_xy.shape:
        raise ValueError("pursuer_xy must match target_xy")
    n = target_xy.shape[0]
    if speed.shape != (n,) or random_angle.shape != (n,):
        raise ValueError("speed and random_angle must have shape [N]")

    radial = target_xy - pursuer_xy
    radial_norm = radial.norm(dim=1, keepdim=True)
    fallback = torch.zeros_like(radial)
    fallback[:, 0] = 1.0
    away = torch.where(radial_norm > 1e-6, radial / radial_norm.clamp(min=1e-6), fallback)
    if mode == "random":
        direction = torch.stack((torch.cos(random_angle), torch.sin(random_angle)), dim=1)
    elif mode == "away":
        direction = away
    elif mode == "toward":
        direction = -away
    elif mode == "tangent_left":
        direction = torch.stack((-away[:, 1], away[:, 0]), dim=1)
    else:  # tangent_right
        direction = torch.stack((away[:, 1], -away[:, 0]), dim=1)
    return direction * speed.unsqueeze(1)


def steer_target_step(
    old_xy,
    desired_velocity,
    speed,
    dt,
    bars_xy,
    lo,
    hi,
    clearance,
    turn_sign,
    previous_heading=None,
):
    """Choose a collision-free full-speed step with optional heading continuity.

    Returns ``(new_xy, velocity, steered, clear)``. If no candidate is clear, the least-bad
    endpoint is returned with ``clear=False`` so the caller's projection fallback can recover it.
    When ``previous_heading`` is provided, clear candidates within 90 degrees of the last flown
    heading are preferred. This is not a veto: if that window contains no clear candidate, the
    ordinary smallest-turn clear candidate still wins.
    """
    if old_xy.ndim != 2 or old_xy.shape[1] != 2:
        raise ValueError("old_xy must have shape [N, 2]")
    n = old_xy.shape[0]
    if desired_velocity.shape != old_xy.shape:
        raise ValueError("desired_velocity must match old_xy")
    if speed.shape != (n,) or turn_sign.shape != (n,):
        raise ValueError("speed and turn_sign must have shape [N]")
    if previous_heading is not None and previous_heading.shape != (n,):
        raise ValueError("previous_heading must have shape [N]")
    if lo.shape != old_xy.shape or hi.shape != old_xy.shape:
        raise ValueError("lo and hi must match old_xy")
    if bars_xy.ndim != 3 or bars_xy.shape[0] != n or bars_xy.shape[2] != 2:
        raise ValueError("bars_xy must have shape [N, B, 2]")

    base_norm = desired_velocity.norm(dim=1, keepdim=True)
    base = desired_velocity / base_norm.clamp(min=1e-6)
    fallback = torch.zeros_like(base)
    fallback[:, 0] = 1.0
    base = torch.where(base_norm > 1e-6, base, fallback)

    angles = torch.tensor(
        [math.radians(value) for value in TURN_ANGLES_DEG],
        device=old_xy.device,
        dtype=old_xy.dtype,
    )
    cosine = torch.cos(angles).view(1, -1)
    sine = torch.sin(angles).view(1, -1)
    bx, by = base[:, 0:1], base[:, 1:2]
    directions = torch.stack(
        (bx * cosine - by * sine, bx * sine + by * cosine), dim=2
    )
    proposals = old_xy.unsqueeze(1) + directions * (
        speed.clamp(min=0.0) * float(dt)
    ).view(n, 1, 1)

    inside = ((proposals >= lo.unsqueeze(1)) & (proposals <= hi.unsqueeze(1))).all(
        dim=2
    )
    if bars_xy.shape[1] > 0 and float(clearance) > 0.0:
        min_bar = torch.cdist(proposals, bars_xy).amin(dim=2)
    else:
        min_bar = torch.full(
            inside.shape, float("inf"), device=old_xy.device, dtype=old_xy.dtype
        )
    clear = inside & (min_bar >= float(clearance) + 1e-4)

    turn_cost = angles.abs().view(1, -1)
    # A clear candidate always wins; among clear candidates prefer the smallest turn. Exact
    # +/- ties use the balanced per-episode sign instead of a fixed global side.
    tie = turn_sign.view(n, 1) * torch.sign(angles).view(1, -1) * 1e-3
    clear_score = 1000.0 - turn_cost + tie
    # If trapped, prefer staying inside the arena and maximizing bar distance. The caller then
    # applies its iterative projection as a final safety net.
    trapped_score = inside.to(old_xy.dtype) * 100.0 + min_bar.clamp(max=10.0)
    trapped_score = trapped_score - 0.01 * turn_cost + tie
    if previous_heading is None:
        score = torch.where(clear, clear_score, trapped_score)
    else:
        heading = torch.atan2(directions[:, :, 1], directions[:, :, 0])
        delta = torch.atan2(
            torch.sin(heading - previous_heading.view(n, 1)),
            torch.cos(heading - previous_heading.view(n, 1)),
        ).abs()
        in_window_clear = clear & (delta <= HEADING_CONTINUITY_RAD + 1e-9)
        has_window_clear = in_window_clear.any(dim=1, keepdim=True)
        eligible_clear = torch.where(has_window_clear, in_window_clear, clear)
        has_any_clear = clear.any(dim=1, keepdim=True)
        neg_inf = torch.full_like(clear_score, float("-inf"))
        score = torch.where(
            has_any_clear,
            torch.where(eligible_clear, clear_score, neg_inf),
            trapped_score,
        )
    chosen = score.argmax(dim=1)
    rows = torch.arange(n, device=old_xy.device)
    selected_xy = proposals[rows, chosen]
    selected_velocity = directions[rows, chosen] * speed.unsqueeze(1)
    selected_clear = clear[rows, chosen]
    return selected_xy, selected_velocity, chosen != 0, selected_clear
