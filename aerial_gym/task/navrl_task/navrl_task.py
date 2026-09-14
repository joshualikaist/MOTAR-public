import hashlib
import importlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch
from gym.spaces import Box, Dict

from aerial_gym.task.base_task import BaseTask
from aerial_gym.task.navrl_task.navrl_curriculum import (
    density_dwell_epochs,
    density_dwell_ready,
    density_level_start_after_promotion,
    density_threshold_at,
    parse_density_threshold_schedule,
    restore_density_level_start_steps,
)
from aerial_gym.task.navrl_task.train_dashboard import record_navrl_epoch_episodes
from aerial_gym.task.navrl_task.target_motion import (
    BOUNDED_TARGET_MOTION_MODEL,
    PHYSICAL_TARGET_MOTION_MODEL,
    CV_INITIAL_HEADING_MODES,
    HEADING_VALID_SPEED_MPS,
    HEADING_VALID_SPEED_KEY,
    HEADING_VALID_SPEED_PROVENANCE_KEY,
    HEADING_VALID_SPEED_SOURCE,
    TARGET_MOTION_MODEL,
    resolve_heading_valid_speed_contract,
    bounded_drone_target_step,
    braking_aware_route_step,
    initial_cv_velocity,
    support_aware_bounds,
    steer_target_step,
)
from aerial_gym.task.navrl_task.target_route_planner import (
    BatchedTargetRouteManager,
    RoutePlannerConfig,
    TARGET_ROUTE_MODE_GLOBAL_ASTAR,
    TARGET_ROUTE_MODE_RECOVERY,
    TARGET_ROUTE_MODE_BRAKING_V3,
    TARGET_ROUTE_HARD_EPSILON_M,
    TARGET_ROUTE_REACHABLE_TUBE_MARGIN_M,
    TARGET_ROUTE_MODE_OFF,
    TARGET_ROUTE_MODES,
    TARGET_ROUTE_MODEL,
    TARGET_ROUTE_RECOVERY_MODEL,
    TARGET_ROUTE_RECOVERY_SCHEMA,
    TARGET_ROUTE_BRAKING_V3_MODEL,
    TARGET_ROUTE_BRAKING_V3_SCHEMA,
    RECOVERY_NORMAL,
    RECOVERY_BRAKE,
    RECOVERY_CONNECT,
    RECOVERY_ROUTE,
    RECOVERY_NO_CONNECTOR,
    conservative_xy_support_from_box,
)
from aerial_gym.task.navrl_task.target_route_geometry import (
    SoftEnvelopeSpec,
    TARGET_ROUTE_BRAKING_GEOMETRY_SCHEMA,
    torch_segments_soft_safe,
    torch_soft_envelope,
    validate_brake_lookup,
)
from aerial_gym.task.navrl_task.speed_governor import (
    SpeedGovernorConfig,
    apply_speed_governor,
    directional_lidar_clearance,
)
from aerial_gym.task.navrl_task.joint_speed_telemetry import JointSpeedTelemetry
from aerial_gym.task.navrl_task.navrl_episode_forensics import EpisodeForensics
from aerial_gym.task.navrl_task.navrl_trajectory_digest import TrajectoryDigest
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.math import quat_rotate, quat_rotate_inverse, quat_to_rotation_matrix
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("navrl_task")

# Fresh routed-recovery contract.  These are intentionally source constants, not environment
# knobs: changing them creates a new source/checkpoint lineage and must not silently retune the
# existing 0.45 m tracking reserve.
RECOVERY_HYSTERESIS_M = 0.25  # route grid resolution, one cell
RECOVERY_STOP_SPEED_MPS = 0.10  # existing braking-probe stop threshold
RECOVERY_CONNECT_PROGRESS_TOLERANCE_M = 1e-6  # fixed numerical non-regression tolerance
RECOVERY_BRAKING_RECEIPT_SCHEMA = "navrl_target_recovery_braking_receipt_v1"
RECOVERY_BRAKING_PROBE_SCHEMA = "navrl_target_recovery_braking_probe_v1"
# SHA-256 of the exact raw-first validator sources from probe lineage 7f2d806 plus its required
# returned core handoff.  If the common verifier changes, this constant and receipt/source
# lineage must change together; a mutable manifest alone cannot authorize code.
RECOVERY_PROBE_VALIDATOR_SHA256 = "963f76e3485c78e411856fd719b1a55e664f9c61d24a2917fc61e4392605c71e"
RECOVERY_RECEIPT_VALIDATOR_SHA256 = "786da78338bc1f03d3abd0f08fd0f2dedff6d74578f56d294a277b9c6d587ae0"

# 5-inch propeller radius (127 mm diameter / 2).  The prop-tip AABB documented in the header of
# resources/robots/quad/quad_navrl_ref5in.urdf is exactly 2 * (motor arm xy + this radius), an
# identity pinned by tests/test_navrl_ref5in_platform.py.  The motor arm itself is read from the
# live allocation matrix, so this is the only propeller literal the spawn geometry needs.
PROP_RADIUS_5IN_M = 0.0635


def _spawn_footprint_clearance_accepted(
    candidate_xy, bar_center_xy, bar_half_xy, required_surface_margin_m
):
    """Per-bar surface-clearance acceptance test for a batch of spawn candidates.

    `candidate_xy` is [N, 2], `bar_center_xy` / `bar_half_xy` are [N, A, 2].  A candidate is
    accepted when it clears the surface of EVERY bar by at least `required_surface_margin_m`,
    where each bar is inflated by its own XY circumradius ||half||.  A single flat centre
    distance cannot express this: the `bars_h3` pool spans circumradius 0.3133..0.5465 m, so a
    constant centre clearance leaves a large bar with a third of the surface margin a small one
    gets.  Using each bar's circumradius (rather than the exact rectangle distance) is the
    conservative direction: it implies the exact-rectangle clearance the geometry audit measures.
    """
    if bar_center_xy.shape[1] == 0:
        return torch.ones(
            candidate_xy.shape[0], dtype=torch.bool, device=candidate_xy.device
        )
    center_distance = torch.cdist(candidate_xy.unsqueeze(1), bar_center_xy).squeeze(1)
    surface_clearance = center_distance - bar_half_xy.norm(dim=2)
    return surface_clearance.amin(dim=1) >= float(required_surface_margin_m)


def _load_recovery_receipt_validator(repo_root):
    """Load the pinned raw-first receipt verifier without changing ``sys.path``.

    The probe and verifier are tools rather than importable package modules.  The verifier's
    dependency is installed in ``sys.modules`` only for the duration of its import, and any
    pre-existing module is restored.  This keeps task imports deterministic while binding the
    validator to the repository checkout that the receipt verifier itself attests.
    """
    root = Path(repo_root).resolve()
    probe_path = (root / "tools/probe_navrl_physical_target_braking.py").resolve()
    verifier_path = (root / "tools/verify_navrl_physical_target_braking.py").resolve()
    if not probe_path.is_file() or not verifier_path.is_file():
        raise RuntimeError("recovery receipt validator sources are missing")
    if probe_path.parent != verifier_path.parent or probe_path.parent != (root / "tools").resolve():
        raise RuntimeError("recovery receipt validator source origin is invalid")
    if hashlib.sha256(probe_path.read_bytes()).hexdigest() != RECOVERY_PROBE_VALIDATOR_SHA256:
        raise RuntimeError("recovery probe validator source SHA256 is not pinned")
    if hashlib.sha256(verifier_path.read_bytes()).hexdigest() != RECOVERY_RECEIPT_VALIDATOR_SHA256:
        raise RuntimeError("recovery receipt validator source SHA256 is not pinned")
    probe_spec = importlib.util.spec_from_file_location(
        "_navrl_recovery_probe_validator", str(probe_path)
    )
    verifier_spec = importlib.util.spec_from_file_location(
        "_navrl_recovery_receipt_validator", str(verifier_path)
    )
    if probe_spec is None or probe_spec.loader is None or verifier_spec is None or verifier_spec.loader is None:
        raise RuntimeError("recovery receipt validator cannot be loaded")
    probe_module = importlib.util.module_from_spec(probe_spec)
    verifier_module = importlib.util.module_from_spec(verifier_spec)
    dependency_name = "probe_navrl_physical_target_braking"
    previous_dependency = sys.modules.get(dependency_name)
    if previous_dependency is not None:
        previous_origin = getattr(previous_dependency, "__file__", None)
        if not previous_origin or Path(previous_origin).resolve() != probe_path:
            raise RuntimeError("recovery probe validator dependency origin is not pinned")
    sys.modules[dependency_name] = probe_module
    try:
        probe_spec.loader.exec_module(probe_module)
        if Path(getattr(probe_module, "__file__", "")).resolve() != probe_path:
            raise RuntimeError("recovery probe validator origin is not pinned")
        verifier_spec.loader.exec_module(verifier_module)
    finally:
        if previous_dependency is None:
            sys.modules.pop(dependency_name, None)
        else:
            sys.modules[dependency_name] = previous_dependency
    if Path(getattr(verifier_module, "__file__", "")).resolve() != verifier_path:
        raise RuntimeError("recovery receipt validator origin is not pinned")
    return verifier_module, probe_path, verifier_path


def _verify_recovery_braking_receipt(receipt_path, declared_sha256, repo_root):
    """Return only raw-recomputed braking data accepted by the common verifier.

    No producer-provided ``VALIDATED`` bit, top-level lookup, or manifest claim is trusted here.
    ``verify_receipt`` must return both its raw-cell summary and canonical integration handoff;
    a verifier that predates that API fails closed.
    """
    receipt_file = Path(receipt_path).resolve()
    if not receipt_file.is_file() or len(str(declared_sha256)) != 64:
        raise RuntimeError("recovery braking receipt path/hash is invalid")
    actual_sha = hashlib.sha256(receipt_file.read_bytes()).hexdigest()
    if actual_sha != str(declared_sha256).lower():
        raise RuntimeError("recovery braking-probe receipt SHA256 mismatch")
    validator, probe_path, verifier_path = _load_recovery_receipt_validator(repo_root)
    try:
        result = validator.verify_receipt(receipt_file.parent, Path(repo_root).resolve())
    except Exception as exc:
        raise RuntimeError("common recovery braking receipt validation failed") from exc
    if not isinstance(result, dict) or result.get("verified") is not True:
        raise RuntimeError("common recovery braking receipt validator did not verify")
    summary = result.get("summary")
    core = result.get("core_integration")
    if not isinstance(summary, dict) or not isinstance(core, dict):
        raise RuntimeError("common validator did not return raw summary/core handoff")
    # The verifier checks the current source manifest against every recorded source byte.  Bind
    # the dynamically loaded validator itself to those same attested bytes before consuming its
    # result, so a shadowed tool cannot issue an otherwise valid-looking handoff.
    manifest_path = receipt_file.parent / "source_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = {str(row["path"]): str(row["sha256"]) for row in manifest["entries"]}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("validated recovery source manifest is unreadable") from exc
    for path in (probe_path, verifier_path):
        relative = path.relative_to(Path(repo_root).resolve()).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if entries.get(relative) != digest:
            raise RuntimeError("recovery validator source is not bound by source manifest")
    return receipt_file, actual_sha, summary, core


def _full_eval_distribution_enabled(bulk_eval_mode, env_value):
    """Return whether evaluation must ignore checkpoint curriculum clocks."""
    return bool(bulk_eval_mode) or str(env_value).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _goal_distance_bounds(goal_min, goal_max, curriculum_max, full_distribution):
    """Resolve the radial sampling contract used by general-spawn episodes."""
    goal_min = float(goal_min)
    goal_max = max(goal_min + 1.0, float(goal_max))
    effective_max = (
        goal_max if full_distribution else min(goal_max, float(curriculum_max))
    )
    max_dist = max(goal_min + 1.0, effective_max)
    min_dist = min(goal_min, max_dist - 1.0)
    return float(min_dist), float(max_dist)


def _goal_front_centered(goal_vehicle, half_angle_deg=15.0):
    """Label only the forward cone; a rear target is not a centered forward goal."""
    if not isinstance(goal_vehicle, torch.Tensor) or goal_vehicle.ndim != 2:
        raise ValueError("goal_vehicle must be a [batch, xyz] tensor")
    if goal_vehicle.shape[1] < 2:
        raise ValueError("goal_vehicle must contain x and y")
    lateral_sine = goal_vehicle[:, 1].abs() / goal_vehicle[:, :2].norm(
        dim=1
    ).clamp(min=1e-6)
    return (goal_vehicle[:, 0] > 0.0) & (
        lateral_sine <= math.sin(math.radians(float(half_angle_deg)))
    )


def _fov_curriculum_saturated(
    force_final_fov, num_task_steps, ppo_horizon, curriculum_epochs
):
    """Resolve whether goal-bearing sampling uses the final, unrestricted FOV."""
    if force_final_fov:
        return True
    curriculum_epochs = float(curriculum_epochs)
    if curriculum_epochs <= 0.0:
        return True
    horizon = max(1, int(ppo_horizon))
    return (float(num_task_steps) / horizon) >= curriculum_epochs


def _fov_curriculum_bearing_limit_rad(
    force_final_fov,
    num_task_steps,
    ppo_horizon,
    curriculum_epochs,
    detector_hfov_deg,
):
    """Return the allowed initial target bearing relative to the drone nose.

    General-spawn targets must remain direction agnostic, so the curriculum constrains the
    *spawn yaw* rather than where the target may be placed.  At epoch zero the target fits safely
    inside the camera (85% of its horizontal half-FOV); the range then expands linearly to the
    complete [-pi, pi] bearing distribution.  Held-out evaluation bypasses the checkpoint clock.
    """
    if _fov_curriculum_saturated(
        force_final_fov, num_task_steps, ppo_horizon, curriculum_epochs
    ):
        return math.pi
    horizon = max(1, int(ppo_horizon))
    progress = (float(num_task_steps) / horizon) / max(
        float(curriculum_epochs), 1.0
    )
    progress = min(1.0, max(0.0, progress))
    half_fov = math.radians(max(0.0, float(detector_hfov_deg)) * 0.5)
    initial = min(math.pi, max(math.radians(8.0), 0.85 * half_fov))
    return initial + progress * (math.pi - initial)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_limit_reached(sim_steps, episode_len_steps):
    """Return the Gymnasium truncation mask for an exact N-step horizon.

    ``sim_steps`` is incremented after every RL action.  Using ``>`` here historically made a
    configured 600-step episode last 601 actions; the stored v2 evaluation receipts expose that
    off-by-one directly (every timeout occurred at step 601).
    """
    return sim_steps >= int(episode_len_steps)


def _episode_outcome_info(successes, crashes, truncations):
    """Build task diagnostics plus rl_games' required time-limit bootstrap signal.

    The human-facing ``timeouts`` key is retained for all existing logging/evaluation consumers.
    rl_games deliberately looks for the distinct ``time_outs`` spelling when
    ``value_bootstrap=True``; omitting it makes a truncation train like a true terminal.
    """
    timeouts = (truncations > 0) & ~successes & ~crashes
    return timeouts, {
        "successes": successes,
        "timeouts": timeouts,
        "time_outs": timeouts,
        "crashes": crashes,
    }


# NAVRL_OBS_DUMP outcome codes (evaluation-only per-episode table). Written as a literal so a
# CPU-only test can pin the exact contract without importing this module.
#
# Codes 0-5 are FROZEN with the meaning they had at commit cff96c2, because the published
# seed-373 dump is already on disk carrying them; renumbering would silently re-label data that
# nobody can re-derive. In particular code 1 means DRONE-body contact ONLY -- exactly what it
# meant there, before the physical-target merge widened the termination mask `d_contact` to
# `(crashed | target_contact | target_invalid)`. The two TARGET-caused terminations that merge
# folded into it therefore get NEW codes (8, 9) instead of borrowing 1, and the two height-band
# causes the dump used to collapse into 3 get NEW codes (6, 7) instead of renumbering 3.
#   3 = "crash_other" is retained for the published dump (there it meant below-or-above,
#       collapsed) but is NO LONGER EMITTED by this code: new runs emit 6 or 7.
# There are three integer encodings of crash cause in this repo -- `self._crash_cause_code`
# (0=contact/1=below/2=above/3=oob, a DIFFERENT numbering, used by the live crash diagnostics),
# this map, and a literal copy in tools/navrl_reflection_offline_audit.py (owned elsewhere, still
# duplicated). This map is the authoritative one for the dump and is EXPORTED INTO THE NPZ
# (outcome_code_values / outcome_code_names / outcome_code_map_json) so a consumer can read the
# map from the data instead of hardcoding a stale copy.
OBS_DUMP_OUTCOME_CODES = {
    0: "capture",
    1: "crash_bar_contact",
    2: "crash_oob",
    3: "crash_other",
    4: "timeout",
    5: "unattributed",
    6: "crash_below_floor",
    7: "crash_above_ceiling",
    8: "crash_target_contact",
    9: "crash_target_invalid",
}


def _obs_dump_retain_decision(call_index, stride_eff):
    """Pure: retain this call's rows under the current streaming-decimation stride?

    ``call_index`` is 0-based -- the number of ``_record_action_diagnostics`` calls strictly
    before this one -- so the very first call is always retained regardless of stride, and after
    any amount of ``_obs_dump_thin_step`` thinning the retained set stays EXACTLY the multiples of
    the current ``stride_eff``, with no permanently-misaligned leftover sample. (A 1-based counter
    breaks that: the first-ever retained call survives every thinning pass at list position 0 but
    is never itself a multiple of the doubled stride.)
    """
    return stride_eff > 0 and call_index % stride_eff == 0


def _obs_dump_thin_step(retained_len, stride_eff, max_rows, rows_per_call):
    """Pure: after appending one retained call, does the row budget require thinning?

    Every retained call contributes exactly ``rows_per_call`` rows (one row per env per sampled
    call), so a retained-call list of length ``retained_len`` holds ``retained_len * rows_per_call``
    rows. Thinning keeps list positions 0, 2, 4, ... (drops every other retained call) and doubles
    ``stride_eff``. Net effect over the whole rollout: the final sample is exactly "every
    stride_eff-th call from the start", uniform over the entire rollout even though the total call
    count is unknown in advance, with a final row count between ``max_rows / 2`` and ``max_rows``.

    Returns ``(new_retained_len, new_stride_eff, thinned)``. The caller loops on ``thinned`` in
    case a single thinning event does not bring the row count back under budget.
    """
    if retained_len <= 1 or retained_len * rows_per_call <= max_rows:
        return retained_len, stride_eff, False
    new_len = (retained_len + 1) // 2  # == len(some_list[0::2])
    return new_len, stride_eff * 2, True


def _obs_dump_outcome_code_table():
    """Pure: `OBS_DUMP_OUTCOME_CODES` as npz-storable arrays plus a JSON string.

    Exported with every dump so a CONSUMER reads the code->name map out of the data instead of
    hardcoding a copy that can drift (tools/navrl_reflection_offline_audit.py still carries such a
    literal; it is owned by another agent and left untouched on purpose).
    """
    values = np.array(sorted(int(k) for k in OBS_DUMP_OUTCOME_CODES), dtype=np.int64)
    names = np.array(
        [OBS_DUMP_OUTCOME_CODES[int(v)] for v in values.tolist()], dtype="<U32"
    )
    payload = json.dumps(
        dict((str(int(v)), OBS_DUMP_OUTCOME_CODES[int(v)]) for v in values.tolist()),
        sort_keys=True,
    )
    return values, names, payload


def _obs_dump_assert_free_path(path, path_exists):
    """Pure: refuse to write a NAVRL_OBS_DUMP over an already-existing file.

    A multi-condition sweep (eval_navrl_v2_density_sweep.sh runs one subprocess per density and
    every child inherits the SAME NAVRL_OBS_DUMP path) would otherwise leave the last successful
    condition's bytes at that path when a later condition's flush dies -- and a downstream
    ``require(frames.is_file())`` cannot tell stale bytes from fresh ones, so a receipt ends up
    attesting one condition's provenance over another condition's data. Fail-closed: the caller
    must use a per-condition path.
    """
    if path_exists:
        raise RuntimeError(
            "obs_dump export: refusing to overwrite an existing dump at %s -- another run "
            "(e.g. the previous density in a sweep) already wrote there and NAVRL_OBS_DUMP was "
            "inherited unchanged. Use a per-condition path (.../<condition>/frames.npz), or "
            "delete the old file deliberately." % (path,)
        )


def _obs_dump_check_episode_budget(existing_rows, new_rows, max_rows):
    """Pure: bound the per-episode outcome table; raise (never truncate) when it would overflow.

    The frame table is capped by streaming decimation, but the outcome table grows one entry per
    step in which any env finishes, so leaving NAVRL_OBS_DUMP set during a TRAINING run grew it
    without limit until OOM. Silent truncation is not an option: frames join to outcomes by
    episode_uid, and a truncated outcome table would either break that join or, worse, look like a
    complete table. Fail loudly and early instead. Returns the new total.
    """
    total = int(existing_rows) + int(new_rows)
    if total > int(max_rows):
        raise RuntimeError(
            "obs_dump export: per-episode outcome table would reach %d rows, over the cap %d. "
            "NAVRL_OBS_DUMP is an EVALUATION-only hook -- unset it for training runs, or raise "
            "NAVRL_OBS_DUMP_MAX_EPISODES deliberately." % (total, int(max_rows))
        )
    return total


def _obs_dump_assign_crash_codes(
    code, captured, crashed, target_contact, target_invalid, d_contact, d_oob, d_below, d_above
):
    """Pure (duck-typed over torch bool tensors and numpy bool arrays): fill per-env outcome codes.

    Reads ONLY the priority-attributed termination masks the task already computed plus the three
    raw sources `d_contact` is the union of; it defines no new attribution and writes nothing back.
    `code` is modified in place and returned; entries the masks do not cover keep whatever the
    caller pre-filled (5 = unattributed).

    Sub-split of `d_contact` (drone body > target contact > target-invalid) exists because the
    physical-target merge widened `d_contact` to `(crashed | target_contact | target_invalid)`,
    which made a clean drone flight whose TARGET left the arena report as a drone-bar collision.
    See OBS_DUMP_OUTCOME_CODES for why the new causes get new codes instead of renumbering.
    """
    code[captured] = 0
    code[d_contact & crashed] = 1
    code[d_contact & ~crashed & target_contact] = 8
    code[d_contact & ~crashed & ~target_contact & target_invalid] = 9
    code[d_oob] = 2
    code[d_below] = 6
    code[d_above] = 7
    return code


def _obs_dump_drop_reset_orphans(frame_tables, outcome_uids, reset_orphan_uids):
    """Pure: drop frame rows whose episode was ended by a FULL ``reset()``, not by an outcome.

    A full reset bumps every env's episode counter without emitting outcome rows (rl_games'
    ``BasePlayer.run`` calls ``env_reset`` at the top of every ``n_games`` iteration, and that loop
    runs more than once whenever ``max_steps`` trips first). Those frames belong to episodes that
    are neither live nor in the outcome table, which used to make the export guard fatal and cost
    the whole multi-hour dump. They are dropped here with a counted, reported reason; the guard
    stays fatal for every OTHER finished-without-outcome episode, which is a genuine inconsistency.

    Frames of episodes that are still running are NOT touched (they are legitimately orphaned and
    the guard already tolerates them), and neither are frames of reset envs that did finish with an
    outcome row in the same step. Returns ``(frame_tables, dropped_rows, dropped_episodes)``.
    """
    orphans = set(int(v) for v in reset_orphan_uids)
    if not orphans or "episode_uid" not in frame_tables:
        return frame_tables, 0, 0
    orphans -= set(int(v) for v in np.asarray(outcome_uids).tolist())
    if not orphans:
        return frame_tables, 0, 0
    uids = np.asarray(frame_tables["episode_uid"])
    drop = np.isin(uids, np.array(sorted(orphans), dtype=np.int64))
    dropped_rows = int(drop.sum())
    if dropped_rows == 0:
        return frame_tables, 0, 0
    keep = ~drop
    dropped_episodes = len(set(int(v) for v in uids[drop].tolist()))
    kept_tables = dict((key, np.asarray(arr)[keep]) for key, arr in frame_tables.items())
    return kept_tables, dropped_rows, dropped_episodes


# Appearance-distractor lock classification radius, in metres
# (docs/prereg_2026-09-01_distractor_envelope.md section 4, frozen before any measurement).
# 0.5 m = the 0.15 m target radius plus roughly twice the detector's measured 0.178 m range MAE.
# It is a preregistered threshold, so it is a module constant that the measurement reads, never a
# number any measurement path may recompute.
DISTRACTOR_LOCK_RADIUS_M = 0.5

# Category order of the three-way classification, fixed here so the accumulator index, the export
# key and the preregistration table cannot drift apart.
DISTRACTOR_LOCK_CATEGORIES = ("target_lock", "distractor_lock", "ghost_lock")


def _validate_distractor_lock_export(
    num_distractors,
    frames_total,
    visible_frames,
    target_lock,
    distractor_lock,
    ghost_lock,
    ambiguous,
    radius_m,
):
    """Fail-closed export guard for the appearance-distractor lock telemetry.

    House style: see ``_validate_obs_dump_export`` above and the OOB / first-acquisition
    cross-checks in ``_export_bulk_eval_result``.  Pure -- plain Python integers and floats, no
    torch and no ``self`` -- so a deliberately miscounted table can be unit-tested without Isaac
    Gym.  Raises ``RuntimeError`` naming the specific mismatch; it never repairs a count and never
    silently drops a frame.

    The load-bearing check is the PARTITION one.  ``visible_frames`` is accumulated straight from
    the detector's own visibility mask, while the three category counts are accumulated from three
    separately constructed masks; the only way those two totals can agree is if every visible frame
    landed in exactly one category.  A double-counted frame (overlapping masks) or a dropped one
    (a gap between them) changes one side and not the other, so a miscount cannot ship as a
    plausible-looking False Target Lock Rate.
    """
    if int(num_distractors) < 1:
        raise RuntimeError(
            "distractor_lock export: the telemetry ran with %d distractor(s); it is defined only "
            "when the scene contains at least one" % int(num_distractors)
        )
    counts = {
        "frames_total": int(frames_total),
        "visible_frames": int(visible_frames),
        "target_lock": int(target_lock),
        "distractor_lock": int(distractor_lock),
        "ghost_lock": int(ghost_lock),
        "ambiguous": int(ambiguous),
    }
    negative = sorted(name for name, value in counts.items() if value < 0)
    if negative:
        raise RuntimeError("distractor_lock export: negative count(s) %s in %s" % (negative, counts))
    if counts["frames_total"] <= 0:
        raise RuntimeError(
            "distractor_lock export: no frames were observed at all, so no rate has a denominator"
        )
    if counts["visible_frames"] > counts["frames_total"]:
        raise RuntimeError(
            "distractor_lock export: %d visible frame(s) out of %d observed frame(s)"
            % (counts["visible_frames"], counts["frames_total"])
        )
    classified = counts["target_lock"] + counts["distractor_lock"] + counts["ghost_lock"]
    if classified != counts["visible_frames"]:
        raise RuntimeError(
            "distractor_lock export: the three categories cover %d frame(s) but the detector "
            "reported visible on %d (target=%d distractor=%d ghost=%d); the classification is not "
            "a partition of the visible frames"
            % (
                classified,
                counts["visible_frames"],
                counts["target_lock"],
                counts["distractor_lock"],
                counts["ghost_lock"],
            )
        )
    # An ambiguous frame -- measurement inside BOTH radii -- is resolved to TARGET_LOCK by the
    # preregistration's category order, so it must be a subset of the target-lock frames.  If it
    # ever exceeds them, the precedence in the classifier and the bookkeeping here have diverged
    # and the FTLR numerator is not what the preregistration defines.
    if counts["ambiguous"] > counts["target_lock"]:
        raise RuntimeError(
            "distractor_lock export: %d ambiguous frame(s) exceed the %d target-lock frame(s); "
            "ambiguous frames are resolved to TARGET_LOCK and must be a subset of them"
            % (counts["ambiguous"], counts["target_lock"])
        )
    if float(radius_m) != float(DISTRACTOR_LOCK_RADIUS_M):
        raise RuntimeError(
            "distractor_lock export: classification radius %r is not the preregistered %r"
            % (float(radius_m), float(DISTRACTOR_LOCK_RADIUS_M))
        )


def _validate_obs_dump_export(
    frame_tables,
    episode_tables,
    live_obs_width,
    max_rows,
    live_episode_uids,
    schema_obs_width=None,
):
    """Fail-closed export guard for NAVRL_OBS_DUMP (house style: see ``_record_oob_exit``).

    Pure: dict-of-numpy-array tables plus a plain ``set`` of currently-live episode_uids, no
    torch, no ``self`` -- a deliberately mismatched table can be unit-tested without Isaac Gym.
    Raises ``RuntimeError`` naming the specific mismatch; never silently drops rows.

    On the two width arguments: ``live_obs_width`` is NOT an independent regression check and must
    not be read as one. The frames are sliced from ``self.task_obs["observations"]``, which is
    allocated with ``task_config.observation_space_dim`` -- the same number the caller passes in --
    so that comparison is a value against itself. It is kept because it still pins the ONE thing a
    consumer needs: the recorded width equals the width the consumer will assume when it joins
    columns. ``schema_obs_width`` is the genuinely independent one: recomputed from the perception
    schema's COMPONENTS (histories x per-token dims + static + corridor, see
    ``NavRLTask._obs_dump_schema_obs_width``), it never passes through the allocation, so an
    898->N schema regression that forgot to update one side is caught here. It is ``None`` only
    when the run has no perception front-end to derive a width from.
    """
    if "obs" not in frame_tables:
        raise RuntimeError("obs_dump export: frame table is missing the 'obs' array")
    n = int(frame_tables["obs"].shape[0])
    for key, arr in frame_tables.items():
        if int(arr.shape[0]) != n:
            raise RuntimeError(
                "obs_dump export: frame array '%s' has %d rows, 'obs' has %d rows"
                % (key, int(arr.shape[0]), n)
            )
    obs_shape = frame_tables["obs"].shape
    if len(obs_shape) != 2 or int(obs_shape[1]) != int(live_obs_width):
        raise RuntimeError(
            "obs_dump export: obs array shape %s does not match the live structured "
            "observation width %d" % (tuple(obs_shape), int(live_obs_width))
        )
    if schema_obs_width is not None and int(obs_shape[1]) != int(schema_obs_width):
        raise RuntimeError(
            "obs_dump export: obs array width %d does not match the width derived independently "
            "from the perception schema components (%d)"
            % (int(obs_shape[1]), int(schema_obs_width))
        )
    if n > int(max_rows):
        raise RuntimeError(
            "obs_dump export: frame row count %d exceeds the configured cap %d" % (n, int(max_rows))
        )
    if "ep_uid" not in episode_tables:
        raise RuntimeError("obs_dump export: episode table is missing the 'ep_uid' array")
    m = int(episode_tables["ep_uid"].shape[0])
    for key, arr in episode_tables.items():
        if int(arr.shape[0]) != m:
            raise RuntimeError(
                "obs_dump export: episode array '%s' has %d rows, 'ep_uid' has %d rows"
                % (key, int(arr.shape[0]), m)
            )
    if n > 0:
        frame_uids = set(int(v) for v in frame_tables["episode_uid"].tolist())
        finished_uids = frame_uids - set(int(v) for v in live_episode_uids)
        outcome_uids = set(int(v) for v in episode_tables["ep_uid"].tolist())
        missing = finished_uids - outcome_uids
        if missing:
            raise RuntimeError(
                "obs_dump export: %d finished episode_uid(s) have frame rows but no outcome "
                "row, e.g. %s" % (len(missing), sorted(missing)[:5])
            )


def vec_to_goal_frame(vec, goal_direction):
    """Express world-frame vector(s) in the goal coordinate frame.

    The goal frame has its x-axis along the (horizontal) start->goal direction, y-axis in the
    horizontal plane, z-axis up. Ported from NavRL's utils.vec_to_new_frame.

    vec:            (N, 3) or (N, M, 3)
    goal_direction: (N, 3)  -- world-frame direction toward the goal (z-component may be 0)
    returns:        same leading shape as vec, last dim 3.
    """
    single = vec.dim() == 2
    if single:
        vec = vec.unsqueeze(1)  # (N, 1, 3)
    n = vec.shape[0]

    gx = goal_direction / goal_direction.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    z = torch.tensor([0.0, 0.0, 1.0], device=vec.device).expand_as(gx)
    gy = torch.cross(z, gx, dim=-1)
    gy = gy / gy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gz = torch.cross(gx, gy, dim=-1)
    gz = gz / gz.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    vx = torch.bmm(vec, gx.view(n, 3, 1))
    vy = torch.bmm(vec, gy.view(n, 3, 1))
    vz = torch.bmm(vec, gz.view(n, 3, 1))
    out = torch.cat((vx, vy, vz), dim=-1)  # (N, M, 3)
    return out.squeeze(1) if single else out


def goal_frame_to_world(vec, goal_direction):
    """Inverse of vec_to_goal_frame: express goal-frame vector(s) back in the world frame.

    Ported from NavRL's utils.vec_to_world.
    """
    world_x = torch.tensor([1.0, 0.0, 0.0], device=vec.device).expand_as(goal_direction)
    world_basis_in_goal = vec_to_goal_frame(world_x, goal_direction)
    return vec_to_goal_frame(vec, world_basis_in_goal)


class NavRLTask(BaseTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp
        super().__init__(task_config)
        self.device = self.task_config.device

        logger.info(
            "Building NavRL task | sim=%s env=%s robot=%s controller=%s"
            % (
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )
        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )
        self.num_envs = self.sim_env.num_envs
        # Freeze the exact vehicle implementation at task construction.  `NAVRL_ROBOT` is read
        # before the simulator is built and two vehicle configs are intentionally shape-compatible,
        # so a checkpoint could otherwise train on ref5in and be evaluated on the legacy body with
        # no tensor-shape error.  The file hashes also make the physical model independently
        # identifiable when the git worktree is dirty.
        self._robot_provenance = self._runtime_robot_provenance()
        self._training_source_provenance = self._load_training_source_receipt()

        # --- task buffers
        self.target_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_dir_2d = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_dir_2d[:, 0] = 1.0  # placeholder unit direction before first reset
        self.height_range = torch.zeros((self.num_envs, 2), device=self.device)  # [min, max]
        self.prev_vel_w = torch.zeros((self.num_envs, 3), device=self.device)
        # Previous drone position: anchors the ego-progress heuristic to the target's CURRENT position
        # (credits only the drone's own motion when the target moves) and, together with prev_rel,
        # provides the swept-segment capture test.
        self.prev_pos = torch.zeros((self.num_envs, 3), device=self.device)
        # Previous drone-position-minus-target-position (relative frame). The capture test sweeps
        # the segment prev_rel -> (pos - target) against the capture sphere at the origin, so a
        # fast fly-through cannot tunnel between 0.1 s samples even when BOTH agents move.
        self.prev_rel = torch.zeros((self.num_envs, 3), device=self.device)
        # per-episode diagnostics: closest approach and whether the goal was ever reached
        self.ep_min_goal_dist = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.ep_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Initial radial goal distance is stable for the lifetime of an episode even when the
        # target moves. It is the correct quantity for stratifying a density competence window.
        self._episode_goal_dist = torch.zeros(self.num_envs, device=self.device)
        # Evaluation-only initial target bearing bucket. This is actor-frame geometry captured at
        # reset, before either agent moves. It lets the mirror audit compare negative/positive-y
        # starts without pretending asynchronously reset rollouts remain episode-paired.
        self._episode_bearing_bin = torch.ones(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._eval_bearing_succ = torch.zeros(3, dtype=torch.long, device=self.device)
        self._eval_bearing_crash = torch.zeros(3, dtype=torch.long, device=self.device)
        self._eval_bearing_timeout = torch.zeros(3, dtype=torch.long, device=self.device)
        self._eval_bearing_fin = torch.zeros(3, dtype=torch.long, device=self.device)
        # Held-out outcome strata are deliberately separate from the density-curriculum counters.
        # A resumed checkpoint may contain a partially accumulated training gate window; reusing
        # it during evaluation would silently mix training episodes into the held-out denominator.
        self._eval_speed_succ = torch.zeros(4, dtype=torch.long, device=self.device)
        self._eval_speed_crash = torch.zeros(4, dtype=torch.long, device=self.device)
        self._eval_speed_timeout = torch.zeros(4, dtype=torch.long, device=self.device)
        self._eval_speed_fin = torch.zeros(4, dtype=torch.long, device=self.device)
        self._eval_speed_crash_cause = torch.zeros(
            (4, 4), dtype=torch.long, device=self.device
        )
        self._eval_dist_succ = torch.zeros(4, dtype=torch.long, device=self.device)
        self._eval_dist_crash = torch.zeros(4, dtype=torch.long, device=self.device)
        self._eval_dist_timeout = torch.zeros(4, dtype=torch.long, device=self.device)
        self._eval_dist_fin = torch.zeros(4, dtype=torch.long, device=self.device)
        self._eval_dist_crash_cause = torch.zeros(
            (4, 4), dtype=torch.long, device=self.device
        )
        self._eval_pattern_succ = torch.zeros(3, dtype=torch.long, device=self.device)
        self._eval_pattern_crash = torch.zeros(3, dtype=torch.long, device=self.device)
        self._eval_pattern_timeout = torch.zeros(3, dtype=torch.long, device=self.device)
        self._eval_pattern_fin = torch.zeros(3, dtype=torch.long, device=self.device)
        self._eval_pattern_crash_cause = torch.zeros(
            (3, 4), dtype=torch.long, device=self.device
        )
        self._eval_dist_pattern_succ = torch.zeros(
            (4, 3), dtype=torch.long, device=self.device
        )
        self._eval_dist_pattern_crash = torch.zeros(
            (4, 3), dtype=torch.long, device=self.device
        )
        self._eval_dist_pattern_timeout = torch.zeros(
            (4, 3), dtype=torch.long, device=self.device
        )
        self._eval_dist_pattern_fin = torch.zeros(
            (4, 3), dtype=torch.long, device=self.device
        )
        self._eval_dist_pattern_crash_cause = torch.zeros(
            (4, 3, 4), dtype=torch.long, device=self.device
        )
        # Per-env terminal attribution for the current step: contact, below, above, OOB.
        self._crash_cause_code = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )

        # goal-distance curriculum state
        self.cur = self.task_config.curriculum
        self.cur_goal_dist_max = float(self.cur.k_start)  # current goal-x ceiling (epoch-driven)
        self.cur_goal_dist_min = float(self.cur.k_min)    # current goal-x floor (epoch-driven)

        # --- Phase 3 moving target: a VIRTUAL point (task-side coordinates only — no actor, no
        # mesh, invisible to the LiDAR). All-zero speeds (the default) keep the task byte-
        # compatible with the static Phases 1-2.
        self.tm = self.task_config.target_motion
        self._target_dynamics = str(getattr(self.tm, "dynamics", "legacy")).strip().lower()
        if self._target_dynamics not in ("legacy", "bounded", "physical"):
            raise ValueError("NAVRL_TARGET_DYNAMICS must be legacy|bounded|physical")
        self._physical_target = self._target_dynamics == "physical"
        self._target_route_mode = str(
            getattr(self.tm, "route_mode", TARGET_ROUTE_MODE_OFF)
        ).strip().lower()
        accepted_route_modes = TARGET_ROUTE_MODES + (
            TARGET_ROUTE_MODE_RECOVERY, TARGET_ROUTE_MODE_BRAKING_V3,
        )
        if self._target_route_mode not in accepted_route_modes:
            raise ValueError(
                "NAVRL_TARGET_ROUTE_MODE must be %s" % "|".join(accepted_route_modes)
            )
        self._target_route_enabled = self._target_route_mode in (
            TARGET_ROUTE_MODE_GLOBAL_ASTAR, TARGET_ROUTE_MODE_RECOVERY,
            TARGET_ROUTE_MODE_BRAKING_V3,
        )
        self._target_route_recovery_enabled = self._target_route_mode == TARGET_ROUTE_MODE_RECOVERY
        self._target_route_braking_v3_enabled = (
            self._target_route_mode == TARGET_ROUTE_MODE_BRAKING_V3
        )
        recovery_contract_variant = str(
            getattr(self.tm, "recovery_braking_contract_variant", "canonical_1p5")
        ).strip().lower()
        if self._target_route_recovery_enabled and recovery_contract_variant not in (
            "canonical_1p5", "baseline_1p25"
        ):
            raise RuntimeError("unknown routed-recovery braking contract variant")
        if self._target_route_braking_v3_enabled and recovery_contract_variant not in (
            "canonical_1p5", "baseline_1p25"
        ):
            raise RuntimeError(
                "braking-v3 requires the canonical 1.5 m/s braking receipt or the separately "
                "preregistered baseline_1p25 lower-contract receipt"
            )
        if self._target_route_enabled and (
            not self._physical_target or str(self.tm.pattern) != "waypoint"
        ):
            raise RuntimeError(
                "NAVRL_TARGET_ROUTE_MODE=%s is a physical+waypoint-only lineage; "
                "mixed/cv/circle and virtual targets are refused" % self._target_route_mode
            )
        if self._target_route_recovery_enabled and float(
            getattr(self.tm, "recovery_brake_decel_p05", 0.0)
        ) <= 0.0:
            raise RuntimeError(
                "two-envelope recovery requires a positive target-specific zero-command "
                "PhysX braking p05; run the preregistered braking probe first"
            )
        self._recovery_probe_receipt_sha256 = ""
        self._recovery_brake_speed_samples_mps = ()
        self._recovery_brake_stop_distance_samples_m = ()
        self._recovery_brake_lateral_tube_p95_m = 0.0
        if self._target_route_recovery_enabled:
            # The environment flag is only an opt-in declaration.  It is not evidence: the raw
            # receipt verifier below must recompute every cell and return the canonical handoff.
            if not bool(getattr(self.tm, "recovery_brake_probe_validated", False)):
                raise RuntimeError(
                    "recovery mode requires the common braking-probe receipt validator; "
                    "scalar p05/p95 values cannot arm recovery"
                )
            if not str(self._training_source_provenance.get("manifest", "")):
                raise RuntimeError(
                    "recovery mode requires a verified training source manifest"
                )
            p95 = float(getattr(self.tm, "recovery_brake_stop_time_p95", 0.0))
            receipt = str(getattr(self.tm, "recovery_brake_probe_receipt", ""))
            declared_sha = str(
                getattr(self.tm, "recovery_brake_probe_receipt_sha256", "")
            ).lower()
            if not (math.isfinite(p95) and p95 > 0.0 and receipt and len(declared_sha) == 64):
                raise RuntimeError(
                    "recovery mode requires measured stop-time p95 and a hashed braking-probe receipt"
                )
            receipt_path, actual_sha, summary, core = _verify_recovery_braking_receipt(
                receipt, declared_sha, Path(__file__).resolve().parents[3]
            )
            certified = core.get("certified_monotone_speed_to_p95_lookup")
            lateral_tube = core.get("certified_lateral_tube_p95_m")
            if (
                core.get("schema") != "navrl_target_recovery_braking_probe_v1"
                or not isinstance(lateral_tube, (int, float))
                or not math.isfinite(float(lateral_tube)) or float(lateral_tube) < 0.0
                or abs(float(core.get("decel_p05_mps2", -1.0)) - float(self.tm.recovery_brake_decel_p05)) > 1e-12
                or abs(float(core.get("stop_time_p95_s", -1.0)) - p95) > 1e-12
            ):
                raise RuntimeError("recovery raw braking handoff does not match task contract")
            if not isinstance(certified, dict) or not certified:
                raise RuntimeError("recovery raw braking handoff has no certified lookup")
            lookup_speeds = np.asarray(
                getattr(self.tm, "recovery_brake_speed_samples_mps", ()), dtype=np.float64
            ).reshape(-1)
            lookup_distances = np.asarray(
                getattr(self.tm, "recovery_brake_stop_distance_samples_m", ()), dtype=np.float64
            ).reshape(-1)
            canonical_rows = [certified[key] for key in sorted(certified, key=lambda key: float(key))]
            if (
                lookup_speeds.shape != (len(canonical_rows),)
                or lookup_distances.shape != lookup_speeds.shape
                or not np.allclose(lookup_speeds, [float(row["speed_mps"]) for row in canonical_rows], rtol=0.0, atol=1e-12)
                or not np.allclose(lookup_distances, [float(row["p95_stop_distance_m"]) for row in canonical_rows], rtol=0.0, atol=1e-12)
            ):
                raise RuntimeError("recovery environment braking lookup differs from canonical receipt")
            lateral_env = float(getattr(self.tm, "recovery_brake_lateral_tube_p95_m", -1.0))
            if not math.isfinite(lateral_env) or lateral_env < 0.0 or abs(lateral_env - float(lateral_tube)) > 1e-12:
                raise RuntimeError("recovery lateral stopping tube differs from canonical receipt")
            # Keep the recomputed summary in the contract path so a future verifier cannot return
            # a handoff disconnected from the raw 32-cell population.
            if not isinstance(summary.get("measured_speed_to_p95_lookup"), dict):
                raise RuntimeError("recovery raw braking summary is incomplete")
            self._recovery_brake_speed_samples_mps = tuple(lookup_speeds.tolist())
            self._recovery_brake_stop_distance_samples_m = tuple(lookup_distances.tolist())
            self._recovery_brake_lateral_tube_p95_m = lateral_env
            self._recovery_probe_receipt_sha256 = actual_sha
        # Obstacle-tensor layout is [target?] [distractors...] [bars...] (see
        # navrl_bars_env.NavRLBarsEnvCfg.env_config: distractors are keep_in_env and declared
        # before the target so the target keeps index 0). Every bar slice in this file starts at
        # _bar_offset, so widening it by the distractor count is what keeps those slices bars-only.
        # num_distractors is 0 unless NAVRL_DISTRACTOR_COUNT is set, and on an env config that has
        # never heard of distractors, so the offset is the historical one by default.
        self._num_distractors = max(
            0, int(getattr(getattr(self.sim_env.cfg, "env_config", None), "num_distractors", 0))
        )
        self._bar_offset = (1 if self._physical_target else 0) + self._num_distractors
        # First asset row of the SOLID obstacle block. Because _bar_offset is
        # (target?) + num_distractors, the distractors are exactly the num_distractors rows
        # IMMEDIATELY BEFORE _bar_offset, so
        #     [_solid_obstacle_offset : _bar_offset + n_bars_active]
        # is one contiguous slice holding [distractors...] [active bars...] and nothing else --
        # never the target at index 0, never a parked bar past the active window. Row j of that
        # slice is the same asset in obstacle_position and in asset_collision_half_extents:
        # EnvManager builds both on the same per-asset axis in one pass over the ordered asset
        # list (env_manager.py: asset_collision_half_extents is vstacked in the same loop
        # iteration that creates the actor, then viewed as [num_envs, -1, 3], and
        # obstacle_position is env_asset_state_tensor[:, :, 0:3] on that same axis).
        # With NAVRL_DISTRACTOR_COUNT unset num_distractors == 0, this equals _bar_offset, and
        # every slice below is byte-for-byte the historical bars-only one.
        # Only the two sites that must not treat a painted distractor as free space use this --
        # the generalized drone spawn sampler and the target route planner. Bar-only logic
        # (density curriculum, bar-contact probes, goal placement) keeps _bar_offset.
        self._solid_obstacle_offset = self._bar_offset - self._num_distractors
        if self._target_route_braking_v3_enabled:
            if not bool(getattr(self.tm, "recovery_brake_probe_validated", False)):
                raise RuntimeError(
                    "braking-v3 requires the common braking-probe receipt validator; "
                    "scalar p05/p95 values cannot arm v3"
                )
            if not str(self._training_source_provenance.get("manifest", "")):
                raise RuntimeError("braking-v3 requires a verified training source manifest")
            p95 = float(getattr(self.tm, "recovery_brake_stop_time_p95", 0.0))
            receipt = str(getattr(self.tm, "recovery_brake_probe_receipt", ""))
            declared_sha = str(
                getattr(self.tm, "recovery_brake_probe_receipt_sha256", "")
            ).lower()
            if not (math.isfinite(p95) and p95 > 0.0 and receipt and len(declared_sha) == 64):
                raise RuntimeError(
                    "braking-v3 requires measured stop-time p95 and a hashed braking-probe receipt"
                )
            receipt_path, actual_sha, summary, core = _verify_recovery_braking_receipt(
                receipt, declared_sha, Path(__file__).resolve().parents[3]
            )
            certified = core.get("certified_monotone_speed_to_p95_lookup")
            lateral_tube = core.get("certified_lateral_tube_p95_m")
            if (
                core.get("schema") != "navrl_target_recovery_braking_probe_v1"
                or not isinstance(lateral_tube, (int, float))
                or not math.isfinite(float(lateral_tube)) or float(lateral_tube) < 0.0
                or abs(float(core.get("stop_time_p95_s", -1.0)) - p95) > 1e-12
            ):
                raise RuntimeError("braking-v3 raw braking handoff does not match task contract")
            if not isinstance(certified, dict) or not certified:
                raise RuntimeError("braking-v3 raw braking handoff has no certified lookup")
            lookup_speeds = np.asarray(
                getattr(self.tm, "recovery_brake_speed_samples_mps", ()), dtype=np.float64
            ).reshape(-1)
            lookup_distances = np.asarray(
                getattr(self.tm, "recovery_brake_stop_distance_samples_m", ()), dtype=np.float64
            ).reshape(-1)
            canonical_rows = [certified[key] for key in sorted(certified, key=lambda key: float(key))]
            if (
                lookup_speeds.shape != (len(canonical_rows),)
                or lookup_distances.shape != lookup_speeds.shape
                or not np.allclose(lookup_speeds, [float(row["speed_mps"]) for row in canonical_rows], rtol=0.0, atol=1e-12)
                or not np.allclose(lookup_distances, [float(row["p95_stop_distance_m"]) for row in canonical_rows], rtol=0.0, atol=1e-12)
            ):
                raise RuntimeError("braking-v3 environment braking lookup differs from canonical receipt")
            try:
                validate_brake_lookup(lookup_speeds.tolist(), lookup_distances.tolist())
            except ValueError as exc:
                raise RuntimeError("braking-v3 lookup is not a monotone ceiling table") from exc
            lateral_env = float(getattr(self.tm, "recovery_brake_lateral_tube_p95_m", -1.0))
            if not math.isfinite(lateral_env) or lateral_env < 0.0 or abs(lateral_env - float(lateral_tube)) > 1e-12:
                raise RuntimeError("braking-v3 lateral stopping tube differs from canonical receipt")
            if not isinstance(summary.get("measured_speed_to_p95_lookup"), dict):
                raise RuntimeError("braking-v3 raw braking summary is incomplete")
            self._recovery_brake_speed_samples_mps = tuple(lookup_speeds.tolist())
            self._recovery_brake_stop_distance_samples_m = tuple(lookup_distances.tolist())
            self._recovery_brake_lateral_tube_p95_m = lateral_env
            self._recovery_probe_receipt_sha256 = actual_sha
        if self._physical_target and self.task_config.robot_name not in (
            "navrl_ref5in_quad", "navrl_ref5in_v2_quad"
        ):
            raise RuntimeError(
                "NAVRL_TARGET_DYNAMICS=physical requires a ref5in platform; "
                "mixing a physical ref5in target with the legacy 0.25 kg pursuer is not a valid "
                "same-platform experiment"
            )
        _physical_geometry_version = os.environ.get(
            "NAVRL_PHYSICAL_GEOMETRY_VERSION", "v1"
        ).strip().lower()
        if self._physical_target:
            expected_robot = (
                "navrl_ref5in_v2_quad" if _physical_geometry_version == "v2"
                else "navrl_ref5in_quad"
            )
            if self.task_config.robot_name != expected_robot:
                raise RuntimeError(
                    "physical geometry/robot mismatch: version=%s requires %s"
                    % (_physical_geometry_version, expected_robot)
                )
        if self._target_dynamics in ("bounded", "physical"):
            if float(self.tm.max_accel) <= 0.0:
                raise ValueError("NAVRL_TARGET_MAX_ACCEL must be positive in bounded mode")
            if float(self.tm.max_turn_rate_deg) <= 0.0:
                raise ValueError("NAVRL_TARGET_MAX_TURN_RATE_DEG must be positive in bounded mode")
            _configured_rl_dt = float(self.sim_env.sim_config.sim.dt) * int(
                self.sim_env.cfg.env.num_physics_steps_per_env_step_mean
            )
            if float(self.tm.avoidance_lookahead_s) < _configured_rl_dt:
                raise ValueError("NAVRL_TARGET_LOOKAHEAD_S must be at least one RL step")
            if float(self.tm.obstacle_clearance) <= 0.0:
                raise ValueError("NAVRL_TARGET_OBSTACLE_CLEARANCE must be positive in bounded mode")
        self._target_motion_model = (
            TARGET_ROUTE_BRAKING_V3_MODEL
            if self._target_route_braking_v3_enabled
            else TARGET_ROUTE_RECOVERY_MODEL
            if self._target_route_recovery_enabled
            else TARGET_ROUTE_MODEL
            if self._target_route_mode == TARGET_ROUTE_MODE_GLOBAL_ASTAR
            else PHYSICAL_TARGET_MOTION_MODEL
            if self._physical_target
            else BOUNDED_TARGET_MOTION_MODEL
            if self._target_dynamics == "bounded"
            else TARGET_MOTION_MODEL
        )
        # Heading-validity threshold provenance. Until a checkpoint is restored the value in force
        # is simply the running literal; set_env_state() replaces the provenance with either an
        # attestation or an explicit assumption.
        self._heading_valid_speed_mps = float(HEADING_VALID_SPEED_MPS)
        self._heading_valid_speed_provenance = HEADING_VALID_SPEED_SOURCE
        self._target_controller = None
        self._target_route_manager = None
        self._target_route_support_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self._target_route_selector = torch.zeros(self.num_envs, device=self.device)
        if self._target_route_enabled:
            support = conservative_xy_support_from_box(self.tm.physical_box_xyz)
            self._target_route_support_xy[:] = torch.as_tensor(
                support, dtype=self._target_route_support_xy.dtype, device=self.device
            )
            route_config = RoutePlannerConfig(
                resolution_m=float(self.tm.route_resolution_m),
                tracking_margin_m=float(self.tm.physical_tracking_margin),
                boundary_margin_m=(
                    float(self.cur.wall_margin) + float(self.tm.physical_boundary_margin)
                ),
                max_expansions=int(self.tm.route_max_expansions),
                max_waypoints=int(self.tm.route_max_waypoints),
                replan_cooldown_steps=int(self.tm.route_replan_cooldown_steps),
                goal_tolerance_m=float(self.tm.route_goal_tolerance_m),
                goal_exclusion_radius_m=float(self.tm.route_goal_exclusion_radius_m),
            )
            self._target_route_manager = BatchedTargetRouteManager(
                self.num_envs, self.device, route_config,
                recovery_enabled=self._target_route_recovery_enabled,
                braking_v3_enabled=self._target_route_braking_v3_enabled,
            )
        self.target_orientation = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_orientation[:, 3] = 1.0
        self.target_vel_w = torch.zeros((self.num_envs, 3), device=self.device)  # realized vel
        self._tm_speed = torch.zeros(self.num_envs, device=self.device)          # per-episode speed
        self._tm_pattern = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 0=cv 1=wp 2=circle
        self._tm_cv_vel = torch.zeros((self.num_envs, 2), device=self.device)    # cv desired velocity
        self._tm_waypoint = torch.zeros((self.num_envs, 2), device=self.device)  # waypoint target
        self._tm_circle_center = torch.zeros((self.num_envs, 2), device=self.device)
        self._tm_circle_angvel = torch.zeros(self.num_envs, device=self.device)  # signed rad/s
        self._tm_avoid_sign = torch.ones(self.num_envs, device=self.device)
        self._tm_heading = torch.zeros(self.num_envs, device=self.device)  # last flown XY heading
        self._tm_last_step_feasible = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Evaluation-only mechanism telemetry. These counters never enter observations, rewards,
        # control, termination, or checkpoint state. Per-episode values are reset with the env;
        # aggregate tensors are exported only by bulk evaluation.
        self._tm_ep_wall_reflections = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._tm_ep_bar_reflections = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._tm_ep_visible_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._tm_ep_observation_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # RESEARCH_PLAN 8.28 first-acquisition telemetry. Same non-interference contract as above:
        # nothing here reaches observations, rewards, control, termination or checkpoint state.
        #
        # Why it is separate from the visible-fraction counters: a step-weighted visible fraction
        # cannot distinguish "acquired late" from "never acquired at all", and the seed-353 result
        # (away capture 15.02% vs timeout 0.59%) is consistent with both. The hard-distance
        # contract [22.5, 28] m sits outside the target camera's 20 m and the LiDAR's 12 m, so
        # every episode starts with the target token zeroed; whether the run ever escapes that
        # state is the quantity this measures.
        #
        # -1 = not yet acquired. Never folded into a mean as 0 -- see _record_first_acquisition.
        self._fa_ep_first_fused = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._fa_ep_first_camera = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._fa_ep_transitions = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._fa_ep_prev_visible = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Independent observation-step counter. It must agree with _tm_ep_observation_steps; the
        # validator checks that rather than sharing one counter, so a future edit to either
        # telemetry cannot silently desynchronise the two chronologies.
        self._fa_ep_obs_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # OOB exit forensics (WORKLOG 2026-08-21). The ref5in hard-distance cells fail almost
        # entirely by leaving the arena -- 158 of 160 crashes in the seed-367 control arm -- and
        # nothing recorded WHERE or WHEN. Edge order is [x_min, x_max, y_min, y_max].
        self._oob_edge_counts = torch.zeros(4, dtype=torch.long, device=self.device)
        self._oob_n = 0
        self._oob_step_sum = 0.0
        self._oob_step_hist = torch.zeros(
            int(self.task_config.episode_len_steps) + 2, dtype=torch.long, device=self.device
        )
        # Split by whether the target had EVER been acquired before the exit. If exits are
        # concentrated in never-acquired episodes, leaving the arena is a symptom of searching
        # blind rather than an independent control failure.
        self._oob_never_acquired = 0
        self._oob_speed_sum = 0.0
        self._oob_goal_dist_sum = 0.0
        # Signed progress along the goal direction at the moment of exit: negative means the drone
        # was moving AWAY from the goal as it left.
        self._oob_goal_closing_sum = 0.0
        self._oob_outward_sum = 0.0
        # Cross-tab kinematics by acquisition state. Overall means alone can hide two opposite
        # exit modes (blind search drifting out vs chasing an acquired target over the boundary).
        self._oob_acquisition_groups = {
            label: {
                key: 0.0 if key != "n" else 0
                for key in ("n", "speed_sum", "goal_dist_sum", "goal_closing_sum", "outward_sum")
            }
            for label in ("never_acquired", "acquired")
        }
        self._fa_eval_outcome_fin = torch.zeros(3, dtype=torch.long, device=self.device)
        self._fa_eval_outcome_never = torch.zeros(3, dtype=torch.long, device=self.device)
        self._fa_eval_outcome_first_sum = torch.zeros(3, dtype=torch.long, device=self.device)
        self._fa_eval_outcome_transitions = torch.zeros(3, dtype=torch.long, device=self.device)
        self._fa_eval_outcome_camera_never = torch.zeros(3, dtype=torch.long, device=self.device)
        self._fa_eval_outcome_camera_first_sum = torch.zeros(
            3, dtype=torch.long, device=self.device
        )
        # Exact median needs the distribution, not a running mean. One bin per possible
        # observation-step index (episode_len_steps + 1); bounded and cheap.
        self._fa_hist_bins = int(self.task_config.episode_len_steps) + 2
        self._fa_eval_outcome_first_hist = torch.zeros(
            3, self._fa_hist_bins, dtype=torch.long, device=self.device
        )
        # S1 explicit-search evaluation telemetry. These are read-only aggregates: actor-safe
        # speed/clearance while the target track is hidden plus the search grid's latched state at
        # first acquisition. They are never read by observation, reward, control or termination.
        self._s1_blind_samples = torch.zeros((), dtype=torch.long, device=self.device)
        self._s1_blind_speed_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        self._s1_blind_clearance_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        self._s1_first_coverage_sum = torch.zeros(3, dtype=torch.float64, device=self.device)
        self._s1_first_coverage_n = torch.zeros(3, dtype=torch.long, device=self.device)
        self._s1_first_entropy_sum = torch.zeros(3, dtype=torch.float64, device=self.device)
        self._s1_first_entropy_n = torch.zeros(3, dtype=torch.long, device=self.device)
        # outcome order: capture, crash, timeout. Reflection state: zero, at least one wall hit.
        self._tm_eval_outcome_fin = torch.zeros(3, dtype=torch.long, device=self.device)
        self._tm_eval_outcome_wall_sum = torch.zeros(3, dtype=torch.long, device=self.device)
        self._tm_eval_outcome_wall_any = torch.zeros(3, dtype=torch.long, device=self.device)
        self._tm_eval_outcome_bar_sum = torch.zeros(3, dtype=torch.long, device=self.device)
        self._tm_eval_outcome_bar_any = torch.zeros(3, dtype=torch.long, device=self.device)
        self._tm_eval_outcome_visible_steps = torch.zeros(
            3, dtype=torch.long, device=self.device
        )
        self._tm_eval_outcome_observation_steps = torch.zeros(
            3, dtype=torch.long, device=self.device
        )
        self._tm_eval_speed_wall_outcome = torch.zeros(
            (4, 2, 3), dtype=torch.long, device=self.device
        )
        self._tm_eval_speed_wall_visible_steps = torch.zeros(
            (4, 2, 3), dtype=torch.long, device=self.device
        )
        self._tm_eval_speed_wall_observation_steps = torch.zeros(
            (4, 2, 3), dtype=torch.long, device=self.device
        )

        rp = self.task_config.reward_parameters
        self.rw = {k: float(v) for k, v in rp.items()}

        # --- shared views into the environment tensors
        self.obs_dict = self.sim_env.get_obs()
        if self._physical_target:
            if int(self.obs_dict["obstacle_position"].shape[1]) < 1:
                raise RuntimeError("physical target requested but no target actor was built")
            # Actor state is the single source of truth used by rewards, rendering and contact.
            self.target_position = self.obs_dict["obstacle_position"][:, 0]
            self.target_orientation = self.obs_dict["obstacle_orientation"][:, 0]
            self.target_vel_w = self.obs_dict["obstacle_linvel"][:, 0]
            from aerial_gym.task.navrl_task.physical_target import PhysicalTargetController

            self._target_controller = PhysicalTargetController(
                self.obs_dict,
                0,
                self.tm,
                self.device,
                contact_threshold=float(self.sim_env.cfg.env.collision_force_threshold),
            )
            self.sim_env.set_physics_step_callback(self._target_controller)
        self.density = getattr(self.task_config, "density", None)

        # In vision mode the LiDAR renderer owns a per-environment analytic target center. The
        # task mirrors the moving target into that buffer before every LiDAR render. Camera
        # perception uses the same task target only inside its renderer; neither path exposes the
        # ground-truth coordinate directly to the actor.
        self._sensor_target = self.obs_dict.get("navrl_target_position", None)
        self.has_vision_target = self._sensor_target is not None

        self.max_bars_available = self._get_max_bars_available()
        initial_bars_requested = self._initial_active_bars()
        self.n_bars_active = 0
        self._density_succ_agg = 0
        self._density_fin_agg = 0
        self._density_gate_not_before_steps = 0
        # num_task_steps at which the CURRENT density became active. The dwell gate below uses it
        # to require a minimum number of epochs at each density before promotion is allowed.
        self._density_level_start_steps = 0
        self._density_speed_succ = torch.zeros(4, dtype=torch.long, device=self.device)
        self._density_speed_fin = torch.zeros(4, dtype=torch.long, device=self.device)
        self._density_dist_succ = torch.zeros(4, dtype=torch.long, device=self.device)
        self._density_dist_fin = torch.zeros(4, dtype=torch.long, device=self.device)
        self._density_pattern_succ = torch.zeros(3, dtype=torch.long, device=self.device)
        self._density_pattern_fin = torch.zeros(3, dtype=torch.long, device=self.device)
        # competence-gated goal-DISTANCE window (NAVRL_K_COMPETENCE): advances by measured capture
        # instead of by epoch. Seeded to the shallow start; persisted across --checkpoint resume.
        self._k_max_cur = float(self.task_config.curriculum.k_start)
        self._k_min_cur = float(self.task_config.curriculum.k_min)
        self._kcomp_succ = 0
        self._kcomp_fin = 0
        self._set_active_bars(initial_bars_requested)
        # Parse the explicit gate schedule ONCE, here, so a malformed spec aborts at startup
        # instead of at the first promotion check hours into a run.
        self._density_threshold_schedule = parse_density_threshold_schedule(
            getattr(self.density, "success_threshold_schedule", "")
        )
        # Report the threshold the gate ACTUALLY applies. With a schedule or the start/end ramp
        # configured these differ from the flat success_threshold, and printing the unused flat
        # value here made the startup banner disagree with every promotion decision.
        _t_start = float(getattr(self.density, "success_threshold_start", 0.0))
        _t_end = float(getattr(self.density, "success_threshold_end", 0.0))
        if self._density_threshold_schedule:
            _threshold_text = "schedule[" + ",".join(
                "%d:%.2f" % (b, t) for b, t in self._density_threshold_schedule
            ) + "]"
        else:
            _threshold_text = (
                "%.3f" % _t_start
                if abs(_t_start - _t_end) <= 1e-9
                else "%.3f->%.3f" % (_t_start, _t_end)
            )
        logger.warning(
            "NavRL density config | initial_bars=%d max_bars=%d curriculum=%s "
            "final=%d step=%d threshold=%s check_eps=%d"
            % (
                self.n_bars_active,
                self.max_bars_available,
                bool(getattr(self.density, "use_density_curriculum", False)),
                int(getattr(self.density, "n_final", self.n_bars_active)),
                int(getattr(self.density, "promote_step", 0)),
                _threshold_text,
                int(getattr(self.density, "check_after_episodes", 0)),
            )
        )
        # One RL step = num_physics_steps x physics dt (0.1 s here). obs_dict["dt"] alone is the
        # PHYSICS dt (0.01 s) — integrating the target with it would move the target at 1/10 of its
        # nominal speed (the shooting_moving_target task gets away with obs_dict["dt"] only because
        # its env runs 1 physics step per RL step).
        try:
            n_phys = int(self.sim_env.cfg.env.num_physics_steps_per_env_step_mean)
        except AttributeError:
            n_phys = 10
            logger.warning("navrl_task: env config not reachable for physics-steps; assuming 10.")
        self.step_dt = float(self.obs_dict["dt"]) * n_phys
        self.speed_governor_cfg = SpeedGovernorConfig.from_environ(os.environ)
        if self.speed_governor_cfg.mode != "off":
            logger.warning(
                "NavRL speed governor | mode=%s fixed=%.2fm/s free=%.2fm/s "
                "path_half=%.2fm margin=%.2fm slow=%.2fm release=%.2fm "
                "ttc=%.2fs brake=%.2fm/s2 reaction=%.2fs"
                % (
                    self.speed_governor_cfg.mode,
                    self.speed_governor_cfg.fixed_cap_mps,
                    self.speed_governor_cfg.free_speed_cap_mps,
                    self.speed_governor_cfg.path_half_width_m,
                    self.speed_governor_cfg.hard_margin_m,
                    self.speed_governor_cfg.slow_distance_m,
                    self.speed_governor_cfg.release_distance_m,
                    self.speed_governor_cfg.ttc_s,
                    self.speed_governor_cfg.brake_mps2,
                    self.speed_governor_cfg.reaction_s,
                )
            )
        if float(self.tm.speed_final) > 0.0 or float(self.tm.speed_fixed) >= 0.0:
            logger.warning(
                "NavRL moving target | pattern=%s dynamics=%s speed_final=%.2f speed_fixed=%.2f "
                "rl_dt=%.3fs accel=%.2fm/s2 turn=%.1fdeg/s lookahead=%.2fs clearance=%.2fm"
                % (
                    self.tm.pattern,
                    self._target_dynamics,
                    self.tm.speed_final,
                    self.tm.speed_fixed,
                    self.step_dt,
                    self.tm.max_accel,
                    self.tm.max_turn_rate_deg,
                    self.tm.avoidance_lookahead_s,
                    self.tm.obstacle_clearance,
                )
            )

        # Critic privileged-state distance normalizer. Reads the SAME env var as
        # navrl_bars_env.NAVRL_ARENA_XY so the divisor always equals the arena side length
        # (24 m in the v1 arena, 40 m in the v2 NavRL-scale search arena). Changing the
        # arena therefore changes the critic input scale -- a task-version change; checkpoints
        # are not comparable across arena sizes.
        self._arena_xy_norm = float(os.environ.get("NAVRL_ARENA_XY", "").strip() or 24.0)

        # --- Phase-3 vision pivot (NAVRL_VISION=1): sensor-only actor. See task_config.vision.
        self.vis_cfg = getattr(self.task_config, "vision", None)
        self.vision_mode = bool(self.vis_cfg is not None and self.vis_cfg.enable)
        self.perception_cfg = getattr(self.task_config, "perception", None)
        self.perception_mode = bool(
            self.vision_mode
            and self.perception_cfg is not None
            and self.perception_cfg.enable
        )
        self._detector_checkpoint_name = ""
        self._detector_checkpoint_sha256 = ""
        if self.perception_mode:
            detector_checkpoint = str(
                getattr(self.perception_cfg, "detector_checkpoint", "") or ""
            ).strip()
            if detector_checkpoint:
                detector_path = Path(detector_checkpoint).expanduser().resolve()
                if not detector_path.is_file():
                    raise FileNotFoundError(
                        f"NAVRL_DETECTOR_CHECKPOINT not found: {detector_path}"
                    )
                self._detector_checkpoint_name = detector_path.name
                self._detector_checkpoint_sha256 = _sha256_file(detector_path)
        self.detector = None
        self.perception = None
        self._d8_treatment_asset_sha256 = ""
        self.prev_action = torch.zeros((self.num_envs, 4), device=self.device)
        self._visible_now = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # 8.28 secondary channel. The preregistered PRIMARY stays the fused flag; this
        # separates "the camera never framed it" from "neither sensor ever held it",
        # which matters because the hard-distance contract exceeds both sensor ranges.
        self._camera_visible_now = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if self.vision_mode:
            if not self.has_vision_target:
                raise RuntimeError(
                    "NAVRL_VISION=1 requires the LiDAR target buffer "
                    "(navrl_target_position missing)."
                )
            from aerial_gym.task.navrl_task.navrl_detector import NavRLTargetDetector

            self.detector = NavRLTargetDetector(
                warp_env=self.sim_env.warp_env,
                num_envs=self.num_envs,
                device=self.device,
                vis_cfg=self.vis_cfg,
                step_dt=self.step_dt,
            )
            # D8 observation treatment is opt-in and evaluation-only. Keep the default/off path
            # free of the treatment module and its URDF dependencies. Unknown non-off values are
            # deliberately handed to treatment_mode(), which fails closed.
            d8_raw = os.environ.get("NAVRL_DYNAMIC_MESH_TREATMENT", "off").strip().lower()
            if d8_raw not in ("", "0", "false", "no", "off"):
                from aerial_gym.task.navrl_task.navrl_dynamic_mesh_treatment import (
                    V3_ASSET_SHA256,
                    load_pinned_v3_asset,
                    treatment_mode,
                )

                treatment_mode()  # Validate before loading any asset.
                repository_root = Path(__file__).resolve().parents[3]
                d8_asset = load_pinned_v3_asset(repository_root)
                self.detector.attach_dynamic_mesh_treatment(
                    d8_asset.mesh, d8_asset.material_rgba
                )
                self._d8_treatment_asset_sha256 = V3_ASSET_SHA256
            if self.perception_mode:
                from aerial_gym.task.navrl_task.navrl_perception import (
                    NavRLPerceptionModule,
                    STRUCTURED_OBS_DIM,
                )

                if int(self.task_config.observation_space_dim) != int(STRUCTURED_OBS_DIM):
                    raise RuntimeError(
                        "NavRL perception schema mismatch: task=%d module=%d"
                        % (self.task_config.observation_space_dim, STRUCTURED_OBS_DIM)
                    )
                self.perception = NavRLPerceptionModule(
                    num_envs=self.num_envs,
                    device=self.device,
                    cfg=self.perception_cfg,
                    step_dt=self.step_dt,
                    camera_cfg=self.vis_cfg,
                )
            if self.speed_governor_cfg.mode != "off" and not self.perception_mode:
                raise RuntimeError(
                    "NavRL speed governor requires the actor-safe perception front-end; "
                    "legacy semantic/oracle target masks are forbidden"
                )
            logger.warning(
                "NavRL %s mode | actor obs=%d, "
                "critic states=%d, body-frame actions, detector range=%.1fm hfov=%.0fdeg"
                % (
                    "PERCEPTION+TRANSFORMER" if self.perception_mode else "VISION-ORACLE-BASELINE",
                    self.task_config.observation_space_dim,
                    self.task_config.state_space_dim,
                    self.vis_cfg.detector_max_range,
                    self.vis_cfg.detector_hfov_deg,
                )
            )
            if self.perception_mode:
                # Echo the knobs that change the obstacle REPRESENTATION. Without this a run leaves
                # no trace of how its tokens were selected, and a finished run cannot be interpreted
                # after the fact -- exactly what happened to ppo_260727_0930, whose token FOV could
                # not be recovered from either the log or the checkpoint.
                from aerial_gym.task.navrl_task.navrl_perception import (
                    GEOFENCE_ACTOR,
                    GEOFENCE_DROPOUT,
                    GEOFENCE_NOISE_STD_M,
                    HBEAMS,
                    MAX_OBSTACLES,
                    OBSTACLE_CLUSTER_GAP_M,
                    OBSTACLE_EFFECTIVE_FOV_DEG,
                    OBSTACLE_FOV_DEG,
                    OBSTACLE_SECTORS,
                    OBSTACLE_SELECTOR,
                    OBSTACLE_SUPPRESS_DEG,
                    OBSTACLE_SUPPRESS_ACTIVE,
                    SEARCH_DIM,
                    SEARCH_STATE,
                    SEARCH_STATE_FORCE_INVALID,
                    VBEAMS,
                )

                logger.warning(
                    "NavRL obstacle representation | tokens=%d configured_fov=%.0fdeg "
                    "effective_fov=%.0fdeg selector=%s suppress=%s cluster_gap=%.2fm sectors=%d "
                    "scan=%dx%d lidar_range=%.1fm"
                    % (
                        MAX_OBSTACLES,
                        OBSTACLE_FOV_DEG,
                        OBSTACLE_EFFECTIVE_FOV_DEG,
                        OBSTACLE_SELECTOR,
                        (
                            "+-%.0fdeg" % OBSTACLE_SUPPRESS_DEG
                            if OBSTACLE_SUPPRESS_ACTIVE
                            else "inactive"
                        ),
                        OBSTACLE_CLUSTER_GAP_M,
                        OBSTACLE_SECTORS,
                        VBEAMS,
                        HBEAMS,
                        self.task_config.lidar_max_range,
                    )
                )
                logger.warning(
                    "NavRL active-search geofence | actor=%s rays=4 noise=%.3fm "
                    "dropout=%.3f source=VIO/GPS+known-map fresh-policy-required"
                    % (
                        "on" if GEOFENCE_ACTOR else "off",
                        GEOFENCE_NOISE_STD_M,
                        GEOFENCE_DROPOUT,
                    )
                )
                logger.warning(
                    "NavRL explicit search state | arm=%s dim=%d masked=%s "
                    "source=VIO/GPS+depth+tracker fresh-policy-required"
                    % (
                        SEARCH_STATE,
                        SEARCH_DIM,
                        "yes" if SEARCH_STATE_FORCE_INVALID else "no",
                    )
                )

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.num_envs, device=self.device)

        # --- spaces
        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(self.task_config.action_space_dim,), dtype=np.float32
        )
        self.task_obs = {
            "observations": torch.zeros(
                (self.num_envs, self.task_config.observation_space_dim), device=self.device
            )
        }
        # vision mode: privileged critic input = actor obs + GT extras (asymmetric actor-critic;
        # rl_games central_value reads obs dict key 'states', forwarded by ExtractObsWrapper)
        if self.vision_mode:
            self.task_obs["states"] = torch.zeros(
                (self.num_envs, self.task_config.state_space_dim), device=self.device
            )
        self.command = torch.zeros((self.num_envs, 4), device=self.device)  # controller input
        self._yaw_cmd = torch.zeros(self.num_envs, device=self.device)  # (b) current-step yaw action a[:,3], penalized quadratically (magnitude damping)
        self._z_err_integral = torch.zeros(self.num_envs, device=self.device)  # altitude-hold PI integral term (see transform_action_to_command)
        self.infos = {}
        self.num_task_steps = 0

        # counters for periodic success/crash/timeout logging
        self._succ_agg = 0
        self._crash_agg = 0
        self._to_agg = 0
        self._reach_agg = 0
        self._fin_agg = 0
        self._mindist_sum = 0.0  # sum of closest approach over NON-CRASH finished episodes
        self._nc_agg = 0         # count of non-crash finished episodes
        self._closest_min = None  # best (min) closest approach in the window
        # Machine-readable vectorized evaluation. rl_games' player summary contains reward/length
        # only, while the periodic task summary historically waited for 2048 episodes. As a result,
        # a perfectly valid 1000-game screen completed without preserving capture/crash data. In
        # bulk mode, make the requested player game count the summary interval and atomically write
        # one result document before the vector player exits.
        self._bulk_eval_mode = os.environ.get(
            "NAVRL_BULK_EVAL", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        self._eval_full_distribution = _full_eval_distribution_enabled(
            self._bulk_eval_mode,
            os.environ.get("NAVRL_EVAL_FULL_DISTRIBUTION", "0"),
        )
        self._s1_search_telemetry_enabled = self._bulk_eval_mode and os.environ.get(
            "NAVRL_S1_SEARCH_TELEMETRY", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        # --- appearance-distractor lock telemetry (EVALUATION ONLY).
        # docs/prereg_2026-09-01_distractor_envelope.md section 4.  Classifies the 3D position the
        # RGB-D detector REPORTED against the ground-truth target and distractor centres, so the
        # preregistration can say what the detector calls "the target" when something else in the
        # scene is painted the same colour.
        #
        # Ground truth is consumed for the METRIC only, exactly as the first-acquisition telemetry
        # and the OOB exit forensics consume it (CLAUDE.md observation contract: no GT target
        # information in the actor observation; permitted in detector supervision, reward,
        # termination, critic and evaluation metrics).  Nothing computed by this telemetry is
        # written into task_obs, a reward, a termination, or any checkpointed field.
        #
        # The enable condition is derived, not a new knob: bulk EVALUATION and a scene that
        # actually contains distractors.  With NAVRL_DISTRACTOR_COUNT unset or 0 the flag is False,
        # no accumulator is allocated, and _process_obs_perception never calls the recorder -- so
        # the historical world runs the historical code path.  Gate 0.1 of the preregistration
        # (default-off bit-identity) therefore stays a property of the code rather than a promise.
        self._distractor_metric_enabled = bool(
            self._bulk_eval_mode and self._num_distractors > 0
        )
        if self._distractor_metric_enabled:
            # Row j of [_solid_obstacle_offset : _bar_offset] is distractor j, by construction:
            # _solid_obstacle_offset is defined as _bar_offset - _num_distractors.  Asserted here
            # rather than assumed, because an off-by-one would silently classify the measurement
            # against a bar centre and produce a plausible FTLR that means nothing.
            if self._bar_offset - self._solid_obstacle_offset != self._num_distractors:
                raise RuntimeError(
                    "NavRL distractor telemetry: the [%d:%d] obstacle slice holds %d row(s), not "
                    "the %d configured distractor(s)"
                    % (
                        self._solid_obstacle_offset,
                        self._bar_offset,
                        self._bar_offset - self._solid_obstacle_offset,
                        self._num_distractors,
                    )
                )
            # Accumulated on-device as int64/float64 totals and read exactly once, at export.  No
            # per-step .item() and therefore no per-step host synchronisation: this telemetry may
            # not be allowed to change the timing of the run it is measuring.
            #   0 frames_total  1 visible  2 target_lock  3 distractor_lock  4 ghost  5 ambiguous
            self._dl_counts = torch.zeros(6, dtype=torch.long, device=self.device)
            #   0 pixel-count sum  1 confidence sum  2 |meas-target| sum  3 |meas-distractor| sum
            #   (all four summed over the DETECTOR-VISIBLE frames only)
            self._dl_sums = torch.zeros(4, dtype=torch.float64, device=self.device)

        # S1 shadow-mode counterfactual accumulators (prereg_2026-09-03_s1_structure_fix_shadow).
        # Enabled by NAVRL_S1_SHADOW=1 in bulk-eval mode only; unlike the distractor-lock block it
        # is NOT gated on num_distractors>0, because the N=0 cell's shadow ghost rate is a
        # preregistered sanity prediction.
        self._s1_shadow_enabled = (
            self._bulk_eval_mode
            and os.environ.get("NAVRL_S1_SHADOW", "0").strip() == "1"
        )
        if self._s1_shadow_enabled:
            #   0 frames_total  1 visible  2 target_lock  3 distractor_lock  4 ghost  5 ambiguous
            #   6 init_frames   7 init_target_lock  8 init_distractor_lock  9 init_ghost_lock
            self._s1_counts = torch.zeros(10, dtype=torch.long, device=self.device)

        try:
            self._bulk_eval_target = max(
                1, int(os.environ.get("PLAY_GAMES_NUM", "1000"))
            )
        except ValueError:
            self._bulk_eval_target = 1000
        self._bulk_eval_output = os.environ.get(
            "NAVRL_BULK_EVAL_JSON", ""
        ).strip()
        self._bulk_eval_exported = False
        self._joint_speed_telemetry_enabled = os.environ.get(
            "NAVRL_JOINT_SPEED_TELEMETRY", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        if self._joint_speed_telemetry_enabled and (
            not self._bulk_eval_mode
            or not self._bulk_eval_output
            or not os.environ.get("NAVRL_EVAL_CHECKPOINT", "").strip()
        ):
            raise RuntimeError(
                "NAVRL_JOINT_SPEED_TELEMETRY is evaluation-only and requires "
                "NAVRL_BULK_EVAL, NAVRL_BULK_EVAL_JSON and NAVRL_EVAL_CHECKPOINT"
            )
        # Evaluation-only episode forensics (TD-T1 acquisition, TD-T2 terminal approach) and the
        # trajectory digest that proves they change nothing. Both default to off and, like the
        # joint-speed telemetry above, the forensics are refused outside a checkpointed evaluation.
        self._episode_forensics_enabled = os.environ.get(
            "NAVRL_EPISODE_FORENSICS", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        self._episode_forensics_output = os.environ.get(
            "NAVRL_EPISODE_FORENSICS_JSON", ""
        ).strip()
        if self._episode_forensics_enabled and (
            not self._bulk_eval_mode
            or not os.environ.get("NAVRL_EVAL_CHECKPOINT", "").strip()
        ):
            raise RuntimeError(
                "NAVRL_EPISODE_FORENSICS is evaluation-only and requires NAVRL_BULK_EVAL "
                "and NAVRL_EVAL_CHECKPOINT"
            )
        self._trajectory_digest_enabled = os.environ.get(
            "NAVRL_TRAJECTORY_DIGEST", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        self._trajectory_digest_output = os.environ.get(
            "NAVRL_TRAJECTORY_DIGEST_JSON", ""
        ).strip()
        self._progress_log_interval = (
            self._bulk_eval_target if self._bulk_eval_mode else 2048
        )
        # Policy-side action diagnostics. `prev_action` stores the post-clamp observation and cannot
        # reveal how much Gaussian probability was collapsed onto +/-1, so measure the actor output
        # immediately on entry to step(). This is instrumentation only; it never changes commands,
        # observations, rewards, or terminations.
        self._action_diag_enabled = self._bulk_eval_mode or os.environ.get(
            "NAVRL_ACTION_DIAG", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        self._action_diag = self._empty_action_diag()
        self._action_diag_prev = torch.zeros(
            (self.num_envs, self.task_config.action_space_dim), device=self.device
        )
        self._action_diag_prev_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._action_front_mask = None
        # R2 control-risk screen. The governor consumes only LiDAR plus the sensor-derived target
        # association already produced by actor perception, changes horizontal command magnitude
        # (never direction), and stays observable through executed previous-action feedback. In
        # bulk evaluation, mode=off still instruments baseline risk margins with identical code.
        self._speed_governor_diag_enabled = self._bulk_eval_mode or os.environ.get(
            "NAVRL_SPEED_GOVERNOR_DIAG", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        self._speed_governor_bearings = None
        self._last_governed_action_xy = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self._speed_governor_last = {
            key: torch.zeros(self.num_envs, device=self.device)
            for key in (
                "requested_speed_mps",
                "executed_speed_mps",
                "speed_cap_mps",
                "scale",
                "clearance_m",
                "ttc_requested_s",
                "stopping_margin_requested_m",
                "stopping_margin_executed_m",
                "lateral_cap_mps",
                "lateral_clearance_m",
            )
        }
        # L5 yaw-rate cap scale (1.0 = untouched); written by the governor, read at yaw command.
        self._governor_yaw_scale = torch.ones(self.num_envs, device=self.device)
        # Contact-corridor forensics (docs/prereg_2026-09-04_contact_corridor_forensics.md).
        # The governor draws its corridor around the COMMANDED direction; the vehicle moves along
        # its CURRENT velocity, which lags. Both are stashed here so the contact recorder can
        # classify the struck bar against each. Evaluation-only; nothing reaches the actor.
        self._contact_geom_enabled = (
            self._bulk_eval_mode
            and os.environ.get("NAVRL_CONTACT_GEOMETRY", "0").strip() == "1"
        )
        self._governor_command_xy = torch.zeros((self.num_envs, 2), device=self.device)
        # Contact geometry must be evaluated at the moment the governor could still have ACTED,
        # not at the instant of contact. By then the vehicle is already alongside the bar (~0.5 m
        # centre-to-centre) and every bar classifies as abeam/behind -- that measures the aftermath.
        # Bars are static, so replaying the DRONE's past pose against current bar positions
        # reconstructs the past geometry exactly. Lookbacks frozen by the prereg.
        self._CG_LOOKBACK_STEPS = (10, 5)   # 1.0 s primary, 0.5 s secondary at 10 Hz policy rate
        # Plan I1 (docs/plans/lateral_contact_density_plan_2026-09-05.md): the memory hypothesis
        # needs the policy's whole 2.5 s history window, so the ring buffer now reaches back
        # 25 steps. Depth grows; the frozen lookbacks and every existing aggregate are unchanged.
        self._CG_MEMORY_STEPS = 25
        if self._contact_geom_enabled:
            depth = max(max(self._CG_LOOKBACK_STEPS), self._CG_MEMORY_STEPS) + 1
            self._cg_hist_pos = torch.zeros((depth, self.num_envs, 3), device=self.device)
            self._cg_hist_quat = torch.zeros((depth, self.num_envs, 4), device=self.device)
            self._cg_hist_quat[:, :, 3] = 1.0
            self._cg_hist_cmd = torch.zeros((depth, self.num_envs, 2), device=self.device)
            self._cg_hist_vel = torch.zeros((depth, self.num_envs, 3), device=self.device)
            self._cg_hist_yaw = torch.zeros((depth, self.num_envs), device=self.device)
            #   0 requested 1 executed 2 cap  (m/s) -- written after the governor runs
            self._cg_hist_gov = torch.zeros((depth, self.num_envs, 3), device=self.device)
            self._cg_hist_age = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
            self._cg_hist_cursor = 0
            # I1/I2 row stores (lists of column dicts on CPU); flushed to JSONL at export.
            self._cg_contact_rows = []
            self._cg_frame_rows = []
            self._cg_frame_every = max(1, int(os.environ.get("NAVRL_CG_FRAME_SAMPLE_EVERY", "100")))
            self._cg_step_counter = 0
        if self._contact_geom_enabled:
            #   0 contacts  1 usable(history deep enough)
            #   2 vertical_out 3 behind 4 lateral 5 no_return 6 in_corridor      (commanded)
            #   7 behind_act   8 lateral_act 9 in_corridor_act                   (actual vel)
            #  10 bars_ge2    11 target_involved
            self._cg_counts = torch.zeros(12, dtype=torch.long, device=self.device)
            #   0 cmd_vs_actual_deg  1 bars_in_corridor  2 lateral_m  3 arc_clearance_m
            self._cg_sums = torch.zeros(4, dtype=torch.float64, device=self.device)
            # A2 descriptive metrics (prereg_2026-09-05_a1_forensics_replication section 3).
            # Not gates -- they answer "did it get safer by getting timid?", "what did the detour
            # cost?", "does the filter fit a control cycle?", and they give prediction 6 the
            # non-contact baseline it needed.
            self._cg_clearance_hist = torch.zeros(25, dtype=torch.long, device=self.device)  # 0-12 m, 0.5 m bins
            self._cg_path_len = torch.zeros(self.num_envs, device=self.device)
            self._cg_path_valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self._cg_prev_pos = None
            #   0 path_len_sum 1 path_len_n 2 cmd_vs_actual_sum 3 cmd_vs_actual_n
            #   4 compute_us_sum 5 compute_us_n 6 compute_us_max
            self._cg_aux = torch.zeros(7, dtype=torch.float64, device=self.device)
            # A3 star-convex shadow (prereg_2026-09-05_a3_star_convex_shadow). Counterfactual only:
            # would an omnidirectional, unknown-bounding region have SEEN the bar the corridor
            # missed? Nothing here reaches the actor.
            self._cg_sc_enabled = os.environ.get("NAVRL_STAR_CONVEX_SHADOW", "0").strip() == "1"
            #   0 eligible(lateral|no_return)  1 reclassified  2 lateral_elig  3 lateral_recl
            #   4 noret_elig  5 noret_recl  6 sc_clearance_sum  7 corridor_clearance_sum
            self._cg_sc = torch.zeros(8, dtype=torch.float64, device=self.device)
            self._cg_sc_geom = None
        self._speed_governor_diag = self._empty_speed_governor_diag()
        self._speed_governor_outcome_steps = {
            "capture": [], "crash": [], "timeout": []
        }
        # Evaluation-only joint kinematics.  It consumes the same actor-safe LiDAR clearance and
        # action-selection state already used by the speed-governor diagnostics, but changes no
        # command, observation, reward, termination, or checkpoint field.
        self._joint_speed_telemetry = (
            JointSpeedTelemetry(
                self.num_envs,
                self.device,
                step_dt=self.step_dt,
                brake_mps2=self.speed_governor_cfg.brake_mps2,
                reaction_s=self.speed_governor_cfg.reaction_s,
                hard_margin_m=self.speed_governor_cfg.hard_margin_m,
            )
            if self._joint_speed_telemetry_enabled
            else None
        )
        # Evaluation-only forensics. It reads detached tensors, draws no randomness, writes into no
        # task buffer, and its labels never reach an observation, a reward or a termination.
        self._episode_forensics = (
            EpisodeForensics(
                self.num_envs,
                self.device,
                step_dt=self.step_dt,
                success_radius_m=float(self.task_config.success_radius),
                target_radius_m=float(getattr(self.vis_cfg, "camera_target_radius", 0.15)),
                checkpoint_sha256=os.environ.get("NAVRL_EVAL_CHECKPOINT", "").strip() or None,
            )
            if self._episode_forensics_enabled
            else None
        )
        self._trajectory_digest = (
            TrajectoryDigest(
                self.num_envs,
                label=os.environ.get("NAVRL_TRAJECTORY_DIGEST_LABEL", "").strip(),
            )
            if self._trajectory_digest_enabled
            else None
        )
        # --- crash-cause diagnosis (NAVRL_CRASH_DIAG=1): split the aggregate "crash" number into
        # its termination source (bar contact / height bound / out-of-arena side) so a stuck run
        # can be diagnosed from measured counts instead of guesses. Off by default: zero overhead.
        self._oob_probe = os.environ.get("NAVRL_OOB_PROBE", "0").strip().lower() in (
            "1", "true", "yes", "on"
        )
        self._crash_diag = self._bulk_eval_mode or self._oob_probe or os.environ.get(
            "NAVRL_CRASH_DIAG", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        self._diag = {k: 0 for k in ("contact", "below", "above", "oob", "oob_w", "oob_e", "oob_s", "oob_n")}
        # "oob" is the EXACT per-env count; W/E/S/N are informational side buckets (a rare
        # diagonal corner exit can land in two of them, so their sum may exceed "oob").
        self._diag_steps = {"contact": 0.0, "oob": 0.0, "below": 0.0}  # steps-to-death sums per cause
        self._diag_below_tilt = 0.0  # sum of tilt angle [deg] at the moment of each below-death
        self._diag_x_sum = 0.0  # death-x sum for bar contacts (bar band starts ~3.1 m)
        # NAVRL_OOB_PROBE=1 is evaluation-only instrumentation. None of these tensors enter the
        # actor/critic observation. Side-aligned values are positive toward the wall that was
        # crossed, letting N and S exits be pooled without hiding a directional policy bias.
        self._probe_ep_start_y = torch.zeros(self.num_envs, device=self.device)
        self._probe_ep_target_start_y = torch.zeros(self.num_envs, device=self.device)
        self._probe_ep_bar_mean_y = torch.zeros(self.num_envs, device=self.device)
        self._probe_ep_y_min = torch.zeros(self.num_envs, device=self.device)
        self._probe_ep_y_max = torch.zeros(self.num_envs, device=self.device)
        self._probe = {
            k: 0.0
            for k in (
                "n",
                "start_y",
                "goal_pull_side",
                "goal_now_pull_side",
                "bar_bias_side",
                "world_vy_side",
                "command_vy_side",
                "action_y_side",
                "excursion_side",
                "visible",
                "track_age",
                "track_cov_pos",
            )
        }
        # NAVRL_EPISODE_DUMP=<path>.npz: evaluation-only per-episode layout/outcome dump for the
        # 검증 4 static reachability oracle. GT bar positions go to DISK for offline analysis
        # only -- nothing recorded here reaches the actor, critic, reward, or termination.
        self._episode_dump_path = os.environ.get("NAVRL_EPISODE_DUMP", "").strip()
        if self._episode_dump_path:
            self._episode_spawn = torch.zeros(self.num_envs, 3, device=self.device)
            self._dump_records = []
            import atexit

            atexit.register(self._flush_episode_dump)

        # NAVRL_OBS_DUMP=<path>.npz: evaluation-only per-frame observation dump. One row per env
        # per SAMPLED call to `_record_action_diagnostics`, tagged with a globally unique
        # episode_uid and the context masks that method already computes (target visibility,
        # front-blocked). A separate per-episode outcome table (joined by episode_uid) is filled
        # from the EXISTING crash-cause/timeout attribution, never a new definition. No-op (no
        # allocation, no GPU sync) when unset. See _collect_obs_dump_frame / _flush_obs_dump.
        self._obs_dump_path = os.environ.get("NAVRL_OBS_DUMP", "").strip()
        self._obs_dump_enabled = bool(self._obs_dump_path)
        if self._obs_dump_enabled:
            self._obs_dump_stride_eff = max(
                1, int(os.environ.get("NAVRL_OBS_DUMP_STRIDE", "1"))
            )
            self._obs_dump_max_rows = max(
                1, int(os.environ.get("NAVRL_OBS_DUMP_MAX", "16384"))
            )
            self._obs_dump_calls = 0
            self._obs_dump_decimations = 0
            self._obs_dump_frames = []
            self._obs_dump_episode_rows = []
            # -1 == "no episode yet"; the first reset_idx() call (full reset, at env construction)
            # bumps every env to episode 0, so this never collides with a real episode index.
            self._obs_dump_ep_idx = torch.full(
                (self.num_envs,), -1, dtype=torch.int64, device=self.device
            )
            # Host mirror of the SAME counter, updated in lockstep in reset_idx(). The flush then
            # needs no GPU access at all -- an atexit-time `.cpu()` on a device tensor runs after
            # the simulator may already be torn down, which is a new failure mode for a hook that
            # is supposed to be free.
            self._obs_dump_ep_idx_host = np.full(self.num_envs, -1, dtype=np.int64)
            # The frame table is capped by streaming decimation; the outcome table needs its own
            # bound (see _obs_dump_check_episode_budget) or a training run with the var set grows
            # it to multi-GB and is discovered only at OOM.
            self._obs_dump_max_episode_rows = max(
                1, int(os.environ.get("NAVRL_OBS_DUMP_MAX_EPISODES", "1048576"))
            )
            self._obs_dump_episode_row_count = 0
            self._obs_dump_episode_overflow = ""
            # episode_uids ended by a FULL reset() with no outcome row (see _flush_obs_dump).
            self._obs_dump_reset_orphan_uids = set()
            self._obs_dump_reset_orphan_cap = 1 << 22
            self._obs_dump_flush_done = False
            self._obs_dump_flush_active = False
            # Fail fast: a sweep that inherited one NAVRL_OBS_DUMP path across conditions must not
            # spend hours of rollout before discovering it cannot write. Same rule re-checked
            # immediately before the write itself.
            _obs_dump_assert_free_path(
                self._obs_dump_path, Path(self._obs_dump_path).exists()
            )
            import atexit

            atexit.register(self._flush_obs_dump)

        # NAVRL_BAR_PROBE=1: evaluation-only bar-contact forensics (zero overhead when off).
        # Probe v2 uses both bearing and range to associate LiDAR surface tokens with GT bars. It
        # also reports collisions inside the token-selection FOV separately; comparing all 240-deg
        # hits against a geometric coverage estimate silently counted the excluded rear 120 deg.
        self._bar_probe = os.environ.get("NAVRL_BAR_PROBE", "0").strip().lower() in (
            "1", "true", "yes", "on"
        )
        if self._bar_probe:
            self._crash_diag = True
        self._bprobe = {
            k: 0.0
            for k in (
                "n",                    # bar-contact deaths sampled
                "bars_in_range",        # GT bars within the LiDAR horizon at impact
                "bars_in_token_fov",    # in-range GT bars eligible for token selection
                "occupied_bins",        # scan bearings returning an obstacle
                "hit_dist",             # center distance to the struck bar
                "hit_in_token_fov",     # struck bar is inside the configured token FOV
                "hit_in_tokens",        # v2 surface-ray/range association to the struck bar
                "hit_in_tokens_in_fov", # v2 match and the struck bar is inside the token FOV
                "valid_tokens",         # valid token slots at impact
                "associated_tokens",    # tokens associated with any GT bar
                "unique_token_bars",    # distinct GT bars represented by associated tokens
                "duplicate_tokens",     # associated slots spent on an already represented GT bar
                "hit_center_offset",    # token surface point to struck-bar center (not an error)
                "hit_cross_track",      # lateral distance from token ray to struck-bar center
                "hit_radial_gap",       # bar-center range minus token surface range
                "hit_token_rank",       # matched slot (0 = nearest)
            )
        }

        # Native 3-D application controls. They are completely disabled during ordinary train/play
        # runs and never become actor observations. The debug target overlay uses GT only for the
        # human viewer; sensor/perception tensors remain the policy's sole target input.
        self.interactive_mode = os.environ.get("NAVRL_INTERACTIVE", "0").strip().lower() in (
            "1", "true", "yes", "on"
        )
        self.general_eval_mode = os.environ.get("NAVRL_GENERAL_EVAL", "0").strip().lower() in (
            "1", "true", "yes", "on"
        )
        self.general_train_mode = os.environ.get("NAVRL_GENERAL_TRAIN", "0").strip().lower() in (
            "1", "true", "yes", "on"
        )
        self._eval_cv_initial_heading = os.environ.get(
            "NAVRL_EVAL_CV_INITIAL_HEADING", "random"
        ).strip().lower()
        if self._eval_cv_initial_heading not in CV_INITIAL_HEADING_MODES:
            raise RuntimeError(
                "NAVRL_EVAL_CV_INITIAL_HEADING must be one of %s, got %r"
                % ("|".join(CV_INITIAL_HEADING_MODES), self._eval_cv_initial_heading)
            )
        if self._eval_cv_initial_heading != "random":
            if (
                not self._bulk_eval_mode
                or not self._bulk_eval_output
                or not os.environ.get("NAVRL_EVAL_CHECKPOINT", "").strip()
            ):
                raise RuntimeError(
                    "NAVRL_EVAL_CV_INITIAL_HEADING is evaluation-only and requires a bulk-result "
                    "path plus NAVRL_EVAL_CHECKPOINT"
                )
            if str(self.tm.pattern) != "cv":
                raise RuntimeError(
                    "NAVRL_EVAL_CV_INITIAL_HEADING requires NAVRL_TARGET_PATTERN=cv"
                )
        self._eval_cv_heading_diag = {
            "samples": 0,
            "radial_cos_sum": 0.0,
            "radial_sin_sum": 0.0,
            "max_contract_error": 0.0,
        }
        self.general_spawn_mode = (
            self.general_eval_mode
            or self.general_train_mode
            or self._eval_full_distribution
        )
        self.general_density_min = int(os.environ.get("NAVRL_GENERAL_DENSITY_MIN", "25"))
        self.general_density_max = int(os.environ.get("NAVRL_GENERAL_DENSITY_MAX", "110"))
        self.general_goal_dist_min = max(
            1.0, float(os.environ.get("NAVRL_GENERAL_GOAL_DIST_MIN", "4"))
        )
        self.general_goal_dist_max = max(
            self.general_goal_dist_min + 1.0,
            float(os.environ.get("NAVRL_GENERAL_GOAL_DIST_MAX", "18")),
        )
        self.general_num_trials = max(1, int(os.environ.get("NAVRL_GENERAL_NUM_TRIALS", "10")))
        self.general_trial_index = 0
        self.general_completed_trials = 0
        self.general_successes = 0
        self.general_crashes = 0
        self.general_timeouts = 0
        self.general_trial_records = []
        self._general_results_exported = False
        self._hud = None
        self._hud_last_outcome = ""
        if self.general_eval_mode and self.num_envs != 1:
            raise RuntimeError("NAVRL_GENERAL_EVAL currently requires exactly one viewer env.")
        self._interactive_reset_requested = False
        self._interactive_show_lidar = True
        self._interactive_manual = False
        self._interactive_manual_keys = {}
        self._interactive_manual_action = torch.zeros(
            (self.num_envs, self.task_config.action_space_dim), device=self.device
        )
        self._interactive_target_trail = []
        self._runtime_target_speed = None
        if self.interactive_mode:
            self._register_interactive_viewer()

    def _get_max_bars_available(self):
        for key in ("obstacle_position", "env_asset_state_tensor"):
            tensor = self.obs_dict.get(key, None)
            if tensor is not None and len(tensor.shape) >= 2:
                return max(0, int(tensor.shape[1]) - int(self._bar_offset))
        logger.warning(
            "NavRL density | obstacle-state tensor not found in obs_dict "
            "(tried obstacle_position, env_asset_state_tensor) -> max_bars=0. Density control is "
            "INERT (0 bars active); check the env build / obs key names."
        )
        return 0

    def _initial_active_bars(self):
        # An explicit NAVRL_NUM_BARS ALWAYS wins — even with the density curriculum flag left on —
        # so density-sweep evals/resumes run at the REQUESTED density instead of silently falling
        # back to n_start (this mirrors the same "NAVRL_NUM_BARS wins" rule in set_env_state).
        if self.density is not None and os.environ.get("NAVRL_NUM_BARS", "").strip():
            requested = getattr(self.density, "num_bars_active", self.max_bars_available)
        elif self.density is None:
            requested = self.max_bars_available
        elif getattr(self.density, "use_density_curriculum", False):
            requested = getattr(self.density, "n_start", self.max_bars_available)
        else:
            requested = getattr(self.density, "num_bars_active", self.max_bars_available)
        return requested

    def _clamp_active_bars(self, n_bars):
        try:
            requested = int(n_bars)
        except (TypeError, ValueError):
            requested = 0
        return min(max(0, requested), self.max_bars_available)

    def _set_active_bars(self, n_bars, log=True):
        try:
            requested = int(n_bars)
        except (TypeError, ValueError):
            requested = 0
        clamped = self._clamp_active_bars(requested)
        if log and clamped != requested:
            logger.warning(
                "Requested %d active bars but only %d were built; using %d."
                % (requested, self.max_bars_available, clamped)
            )
        self.n_bars_active = clamped
        self.obs_dict["num_obstacles_in_env"] = clamped + int(self._bar_offset)
        return clamped

    def set_runtime_bars(self, n_bars):
        """Change density for subsequent resets and request an immediate all-env reset."""
        value = self._set_active_bars(n_bars)
        self._interactive_reset_requested = True
        return value

    def set_runtime_target_speed(self, speed_mps):
        """Force one exact target speed for interactive episodes (not a training curriculum)."""
        self._runtime_target_speed = max(0.0, float(speed_mps))
        self._interactive_reset_requested = True
        return self._runtime_target_speed

    def set_runtime_drone_speed(self, speed_mps):
        """Set the action-to-velocity scale used by the controller."""
        self.task_config.max_velocity = max(0.25, float(speed_mps))
        return float(self.task_config.max_velocity)

    def _sample_general_density(self):
        """Sample one randomized clutter level for the next single-env evaluation trial."""
        lo = self._clamp_active_bars(min(self.general_density_min, self.general_density_max))
        hi = self._clamp_active_bars(max(self.general_density_min, self.general_density_max))
        value = int(torch.randint(lo, hi + 1, (1,), device=self.device).item())
        self._set_active_bars(value, log=False)
        self.general_trial_index += 1
        logger.warning(
            "NavRL general trial %d/%d | randomized bars=%d"
            % (self.general_trial_index, self.general_num_trials, value)
        )

    def _record_general_result(self, successes, crashes, timeouts, finished):
        if not self.general_eval_mode or not bool(finished.any()):
            return
        self.general_completed_trials += int(finished.sum().item())
        self.general_successes += int(successes.sum().item())
        self.general_crashes += int((crashes > 0).sum().item())
        self.general_timeouts += int(timeouts.sum().item())
        outcome = "captured" if bool(successes.any()) else (
            "crashed" if bool((crashes > 0).any()) else "timeout"
        )
        env_ids = finished.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.ndim == 0:
            env_ids = env_ids.unsqueeze(0)
        for env_id in env_ids.tolist():
            self.general_trial_records.append(
                {
                    "trial": int(self.general_completed_trials),
                    "bars": int(self.n_bars_active),
                    "outcome": outcome,
                    "min_goal_dist_m": float(self.ep_min_goal_dist[env_id].item()),
                    "steps": int(self.sim_env.sim_steps[env_id].item()),
                    "target_speed_mps": float(self._tm_speed[env_id].item()),
                }
            )
        logger.warning(
            "NavRL general result %d/%d | %s"
            % (
                min(self.general_completed_trials, self.general_num_trials),
                self.general_num_trials,
                outcome,
            )
        )
        if self._hud is not None:
            from aerial_gym.apps.navrl_3d_hud import NavRL3DHud, build_hud_lines, build_hud_pip

            flash_text = outcome.upper()
            if outcome == "captured":
                self._hud.flash(flash_text, color=NavRL3DHud.OK)
            elif outcome == "crashed":
                self._hud.flash(flash_text, color=NavRL3DHud.BAD)
            else:
                self._hud.flash(flash_text, color=NavRL3DHud.WARN)
        if self.general_completed_trials >= self.general_num_trials:
            logger.warning(
                "NavRL general summary | captured=%d crash=%d timeout=%d / %d"
                % (
                    self.general_successes,
                    self.general_crashes,
                    self.general_timeouts,
                    self.general_num_trials,
                )
            )
            self._export_general_results_json()

    def _export_general_results_json(self):
        if self._general_results_exported or not self.general_eval_mode:
            return
        path = os.environ.get("NAVRL_GENERAL_RESULTS_JSON", "").strip()
        if not path:
            return
        payload = {
            "num_trials": int(self.general_num_trials),
            "density_min": int(self.general_density_min),
            "density_max": int(self.general_density_max),
            "target_speed_mps": float(os.environ.get("NAVRL_TARGET_SPEED", "0") or 0.0),
            "drone_max_speed_mps": float(self.task_config.max_velocity),
            "summary": {
                "captured": int(self.general_successes),
                "crash": int(self.general_crashes),
                "timeout": int(self.general_timeouts),
            },
            "trials": list(self.general_trial_records),
        }
        try:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            logger.warning("NavRL general results saved -> %s" % out)
            if self._hud is not None:
                cap = payload["summary"]["captured"]
                self._hud.set_summary(
                    [
                        "Evaluation complete",
                        "Captured %d / %d" % (cap, self.general_num_trials),
                        "Crash %d  Timeout %d" % (
                            payload["summary"]["crash"],
                            payload["summary"]["timeout"],
                        ),
                        "Saved %s" % out.name,
                    ]
                )
        except OSError as exc:
            logger.warning("NavRL general results export failed: %s" % exc)
        self._general_results_exported = True

    def _robot_spawn_inflation_radius_m(self):
        """Yaw-invariant robot circumradius (m) that spawn feasibility must inflate bars by.

        Read from the LIVE vehicle rather than copied from the audit tool, so v1/v2 geometry (and
        any future frame) cannot drift away from what the CPU geometry audit inflates free space
        by in tools/audit_navrl_v2_density_geometry.py:

          * prop-tip span = 2 * (motor arm xy + PROP_RADIUS_5IN_M).  The arm comes from the live
            allocation matrix (navrl_ref5in_quad_config.py); quad_navrl_ref5in.urdf's header
            documents the same identity and test_navrl_ref5in_platform.py pins it.
          * base_link collision box from the live URDF.  quad_navrl_ref5in_v2.urdf's 0.283 m box
            slightly exceeds the 0.2825634 m tip span, so the actual support is the larger of the
            two.

        The disk is the half-diagonal of that square span and is therefore invariant to the random
        spawn yaw.  For the legacy 0.13 m-arm body the 5-inch propeller radius over-estimates the
        tip span; spawn inflation is deliberately conservative rather than under-bounding it.
        """
        cached = getattr(self, "_spawn_inflation_radius_cache", None)
        if cached is not None:
            return cached
        robot_cfg = self.sim_env.robot_manager.cfg
        allocation = robot_cfg.control_allocator_config.allocation_matrix
        arms = {round(abs(float(v)), 9) for row in allocation[3:5] for v in row}
        arms.discard(0.0)
        if len(arms) != 1:
            raise RuntimeError(
                "spawn geometry needs a square motor-arm layout; allocation matrix gives %s"
                % sorted(arms)
            )
        prop_tip_span = 2.0 * (arms.pop() + PROP_RADIUS_5IN_M)
        asset_cfg = robot_cfg.robot_asset
        asset_path = (Path(asset_cfg.asset_folder) / asset_cfg.file).resolve()
        box = ET.parse(str(asset_path)).getroot().find(
            "link[@name='base_link']/collision/geometry/box"
        )
        if box is None:
            raise RuntimeError(
                "robot URDF has no base_link collision box for spawn geometry: %s" % asset_path
            )
        box_xy = max(float(v) for v in box.get("size").split()[0:2])
        radius = 0.5 * max(prop_tip_span, box_xy) * math.sqrt(2.0)
        self._spawn_inflation_radius_cache = radius
        return radius

    def _spawn_required_surface_margin_m(self):
        """Surface clearance a spawn point must keep from every bar's own footprint.

        Identical to the inflation the geometry audit's connectivity graph uses: the yaw-invariant
        robot disk plus the closed-loop tracking reserve (`physical_tracking_margin`, read live
        from navrl_task_config.py, not a copied literal).  Spawning inside that inflated obstacle
        set puts the episode's first step in a cell the audit calls infeasible.
        """
        return self._robot_spawn_inflation_radius_m() + float(
            self.tm.physical_tracking_margin
        )

    def _randomize_general_drone_spawn(self, env_ids):
        """Place the drone at a collision-free random XY/yaw for generalized evaluation.

        Acceptance is footprint aware: each candidate must clear EVERY solid obstacle's own
        surface by `_spawn_required_surface_margin_m()`.  The earlier flat 0.65 m bar-CENTRE test
        was blind to bar size, so a 0.5465 m-circumradius bar left only 0.10 m of real surface
        clearance and 27.45% of 205-bar spawns started inside the inflated obstacle set
        (results/navrl_v2_density_geometry_audit_2026-08-27/summary.md).

        The obstacle set is `_solid_obstacle_offset`-based, not `_bar_offset`-based: the painted
        distractors are solid collidable bodies, so a spawn INSIDE one is a physically impossible
        frame whose detection outcome means nothing.  With no distractors configured the slice is
        identical to the historical bars-only one.
        """
        n = len(env_ids)
        if n == 0:
            return
        b_min = self.obs_dict["env_bounds_min"][env_ids]
        b_max = self.obs_dict["env_bounds_max"][env_ids]
        obstacles = self.obs_dict["obstacle_position"][
            env_ids, self._solid_obstacle_offset : self._bar_offset + self.n_bars_active, 0:2
        ]
        # Same slice on the same asset axis as the runtime planner geometry
        # (`_plan_target_routes`), so row j of `obstacle_half` is the footprint of the obstacle
        # whose centre is row j of `obstacles`, across the distractor->bar boundary too.
        obstacle_half = self.obs_dict["asset_collision_half_extents"][
            env_ids, self._solid_obstacle_offset : self._bar_offset + self.n_bars_active, 0:2
        ]
        required_margin = self._spawn_required_surface_margin_m()
        lo = b_min[:, 0:2] + 1.0
        hi = b_max[:, 0:2] - 1.0
        chosen = torch.zeros((n, 2), device=self.device)
        todo = torch.ones(n, dtype=torch.bool, device=self.device)
        candidates_per_round = 64
        rounds = 128
        for _ in range(rounds):
            if not bool(todo.any()):
                break
            ids = todo.nonzero(as_tuple=False).squeeze(-1)
            candidate = lo[ids].unsqueeze(1) + (
                hi[ids] - lo[ids]
            ).unsqueeze(1) * torch.rand(
                (len(ids), candidates_per_round, 2), device=self.device
            )
            flat_count = len(ids) * candidates_per_round
            accepted = _spawn_footprint_clearance_accepted(
                candidate.reshape(flat_count, 2),
                obstacles[ids].repeat_interleave(candidates_per_round, dim=0),
                obstacle_half[ids].repeat_interleave(candidates_per_round, dim=0),
                required_margin,
            ).reshape(len(ids), candidates_per_round)
            has_candidate = accepted.any(dim=1)
            if bool(has_candidate.any()):
                first = accepted.to(dtype=torch.int64).argmax(dim=1)
                resolved = ids[has_candidate]
                chosen[resolved] = candidate[has_candidate, first[has_candidate]]
                todo[resolved] = False
        if bool(todo.any()):
            raise RuntimeError(
                "drone spawn has no footprint-clear sample after %d attempts for %d envs"
                % (candidates_per_round * rounds, int(todo.sum()))
            )

        self.obs_dict["robot_position"][env_ids, 0:2] = chosen
        self.obs_dict["robot_position"][env_ids, 2] = float(self.task_config.flight_altitude)
        yaw = -math.pi + 2.0 * math.pi * torch.rand(n, device=self.device)
        quat = torch.zeros((n, 4), device=self.device)
        quat[:, 2] = torch.sin(0.5 * yaw)
        quat[:, 3] = torch.cos(0.5 * yaw)
        self.obs_dict["robot_orientation"][env_ids] = quat
        self.obs_dict["robot_linvel"][env_ids] = 0.0
        self.obs_dict["robot_angvel"][env_ids] = 0.0

    def _general_goal_distance_bounds(self):
        full_distribution = bool(
            self.general_eval_mode or self._eval_full_distribution
        )
        curriculum_max = (
            self.general_goal_dist_max
            if full_distribution
            else self._goal_x_max()
        )
        min_dist, max_dist = _goal_distance_bounds(
            self.general_goal_dist_min,
            self.general_goal_dist_max,
            curriculum_max,
            full_distribution,
        )
        return min_dist, max_dist, full_distribution

    def _fov_curriculum_is_saturated(self):
        return _fov_curriculum_saturated(
            self.general_eval_mode or self._eval_full_distribution,
            self.num_task_steps,
            getattr(self.cur, "ppo_horizon", 1),
            getattr(self.vis_cfg, "fov_curriculum_epochs", 0.0),
        )

    def _fov_curriculum_bearing_limit_rad(self):
        return _fov_curriculum_bearing_limit_rad(
            self.general_eval_mode or self._eval_full_distribution,
            self.num_task_steps,
            getattr(self.cur, "ppo_horizon", 1),
            getattr(self.vis_cfg, "fov_curriculum_epochs", 0.0),
            getattr(self.vis_cfg, "detector_hfov_deg", 0.0),
        )

    def _align_general_spawn_yaw_to_target(self, env_ids, start_pos, goal):
        """Apply the FOV curriculum without biasing general target positions.

        `_sample_general_target` deliberately samples every world direction.  Before saturation we
        rotate only the drone's initial yaw so the target-relative bearing lies inside the current
        curriculum limit.  At saturation (and in full-distribution evaluation) the yaw produced by
        `_randomize_general_drone_spawn` remains uniformly random.
        """
        if (
            not self.general_spawn_mode
            or not self.vision_mode
            or self._fov_curriculum_is_saturated()
        ):
            return
        delta = goal[:, 0:2] - start_pos[:, 0:2]
        target_bearing = torch.atan2(delta[:, 1], delta[:, 0])
        bearing_limit = self._fov_curriculum_bearing_limit_rad()
        relative_bearing = (2.0 * torch.rand_like(target_bearing) - 1.0) * bearing_limit
        yaw = target_bearing - relative_bearing
        quat = torch.zeros((len(env_ids), 4), device=self.device)
        quat[:, 2] = torch.sin(0.5 * yaw)
        quat[:, 3] = torch.cos(0.5 * yaw)
        self.obs_dict["robot_orientation"][env_ids] = quat

    def _runtime_physics_contract(self):
        """Return measured simulator timing, never a launcher label or assumed default."""
        sim_config = self.sim_env.sim_config
        sim_config_class = getattr(sim_config, "__name__", type(sim_config).__name__)
        physics_dt = float(self.obs_dict["dt"])
        physics_substeps = int(sim_config.sim.substeps)
        physics_steps_per_rl_step = int(
            self.sim_env.cfg.env.num_physics_steps_per_env_step_mean
        )
        return {
            "runtime_sim_config_class": str(sim_config_class),
            "physics_dt_s": physics_dt,
            "physics_substeps": physics_substeps,
            "physics_steps_per_rl_step": physics_steps_per_rl_step,
            "rl_step_dt_s": float(self.step_dt),
        }

    @staticmethod
    def _sha256_file(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _runtime_robot_provenance(self):
        """Return an exact, checkpoint-safe identity for the instantiated vehicle.

        Robot choice is not encoded in the policy tensor shapes.  Record both the selected name
        and the source bytes which define its URDF/config so evaluators can restore the vehicle
        before importing the task config and can reject a silent cross-lineage replay.
        """
        robot_cfg = self.sim_env.robot_manager.cfg
        asset_cfg = robot_cfg.robot_asset
        asset_path = (Path(asset_cfg.asset_folder) / asset_cfg.file).resolve()
        module = importlib.import_module(robot_cfg.__module__)
        config_path = Path(module.__file__).resolve()
        repo_root = Path(__file__).resolve().parents[3]
        if not asset_path.is_file():
            raise RuntimeError("NavRL robot asset is missing: %s" % asset_path)
        if not config_path.is_file():
            raise RuntimeError("NavRL robot config source is missing: %s" % config_path)
        try:
            config_relative = config_path.relative_to(repo_root).as_posix()
            asset_relative = asset_path.relative_to(repo_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                "NavRL robot source must live inside the repository: %s / %s"
                % (config_path, asset_path)
            ) from exc
        return {
            "robot_name": str(self.task_config.robot_name),
            "robot_config_class": str(robot_cfg.__name__),
            "robot_config_module": str(robot_cfg.__module__),
            "robot_config_file": config_path.name,
            "robot_config_path": config_relative,
            "robot_config_sha256": self._sha256_file(config_path),
            "robot_asset_file": str(asset_cfg.file),
            "robot_asset_path": asset_relative,
            "robot_asset_sha256": self._sha256_file(asset_path),
        }

    def _load_training_source_receipt(self):
        """Validate and bind an optional immutable training-source bundle.

        Legacy/interactive tasks may omit it.  Closed ref5in launchers require it through
        NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=1.  Validation is repeated whenever a checkpoint is
        saved so a mid-run source edit cannot be hidden behind the launch-time receipt.
        """
        manifest_text = os.environ.get("NAVRL_TRAINING_SOURCE_MANIFEST", "").strip()
        expected_sha = os.environ.get(
            "NAVRL_TRAINING_SOURCE_MANIFEST_SHA256", ""
        ).strip().lower()
        def _require_flag(name):
            """Parse a gate-ARMING flag. An unknown spelling must not quietly mean "off".

            These two turn the source-attestation gate on. Membership testing against a tuple of
            accepted words meant a launcher writing =Y, =enabled or =TRUE1 silently disarmed the
            gate and the run proceeded with no attestation at all. A flag whose job is to demand
            evidence is the last place to guess.
            """
            raw = os.environ.get(name, "0").strip().lower()
            if raw in ("1", "true", "yes", "on"):
                return True
            if raw in ("", "0", "false", "no", "off"):
                return False
            raise ValueError(
                f"{name}={raw!r} is not a recognised boolean; use 1/0, true/false, yes/no or "
                "on/off. Refusing to treat an unknown value as 'not required'."
            )

        required = _require_flag("NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT")
        require_clean = _require_flag("NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE")
        if not manifest_text:
            if required:
                raise RuntimeError("NavRL training source receipt is required but unset")
            return {
                "manifest": "",
                "manifest_sha256": "",
                "git_commit": "",
                "git_dirty": None,
                "runtime_file_count": 0,
            }

        manifest_path = Path(manifest_text).expanduser().resolve()
        if not manifest_path.is_file():
            raise RuntimeError("NavRL training source manifest is missing: %s" % manifest_path)
        actual_sha = self._sha256_file(manifest_path)
        if len(expected_sha) != 64 or actual_sha != expected_sha:
            raise RuntimeError("NavRL training source manifest SHA-256 mismatch")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("NavRL training source manifest is unreadable") from exc
        if manifest.get("schema_version") != 1 or manifest.get("purpose") != (
            "navrl_training_source_receipt"
        ):
            raise RuntimeError("unsupported NavRL training source manifest contract")
        if require_clean and bool(manifest.get("git_dirty")):
            raise RuntimeError("closed NavRL training requires a clean source receipt")

        provenance = {
            "manifest": str(manifest_path),
            "manifest_sha256": actual_sha,
            "git_commit": str(manifest.get("git_commit", "")),
            "git_dirty": bool(manifest.get("git_dirty")),
            "runtime_file_count": int(manifest.get("runtime_file_count", 0)),
            "payload": manifest,
        }
        self._verify_training_source_receipt(provenance)
        return provenance

    def _verify_training_source_receipt(self, provenance=None):
        provenance = provenance or self._training_source_provenance
        manifest_path_text = str(provenance.get("manifest", ""))
        if not manifest_path_text:
            return
        manifest_path = Path(manifest_path_text)
        if self._sha256_file(manifest_path) != provenance["manifest_sha256"]:
            raise RuntimeError("NavRL training source manifest changed during the run")
        manifest = provenance.get("payload") or json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        entries = manifest.get("runtime_files") or []
        if not entries or len(entries) != int(manifest.get("runtime_file_count", -1)):
            raise RuntimeError("NavRL training source receipt has invalid file accounting")
        repo = Path(manifest["repository_root"]).resolve()
        seen = {}
        for entry in entries:
            original = (repo / entry["path"]).resolve()
            snapshot = (manifest_path.parent / entry["snapshot"]).resolve()
            expected = str(entry["sha256"])
            if not original.is_file() or self._sha256_file(original) != expected:
                raise RuntimeError("NavRL training runtime source changed: %s" % entry["path"])
            if not snapshot.is_file() or self._sha256_file(snapshot) != expected:
                raise RuntimeError("NavRL training source snapshot changed: %s" % entry["snapshot"])
            relative = Path(entry["path"]).as_posix()
            if relative in seen:
                raise RuntimeError(
                    "NavRL training receipt contains a duplicate runtime path: %s" % relative
                )
            seen[relative] = expected
        environment_path = (
            manifest_path.parent / str(manifest.get("python_environment", ""))
        ).resolve()
        if (
            not environment_path.is_file()
            or self._sha256_file(environment_path)
            != str(manifest.get("python_environment_sha256", ""))
        ):
            raise RuntimeError("NavRL training Python environment receipt changed")
        for label, filename, digest in (
            (
                "robot config",
                self._robot_provenance["robot_config_path"],
                self._robot_provenance["robot_config_sha256"],
            ),
            (
                "robot URDF",
                self._robot_provenance["robot_asset_path"],
                self._robot_provenance["robot_asset_sha256"],
            ),
        ):
            if seen.get(filename) != digest:
                raise RuntimeError(
                    "NavRL training receipt does not bind the active %s: %s"
                    % (label, filename)
                )

    def _sample_general_target(self, env_ids, start_pos, b_min, b_max, bars_xy, bar_half):
        """Sample a range-controlled, collision-free target in a random direction.

        General-spawn training is deliberately direction agnostic, unlike the legacy left-to-right
        task. Keep its real radial contract explicit instead of silently reporting the legacy
        ``k_min_cur`` while actually accepting every distance above a hard-coded four metres.
        """
        n = len(env_ids)
        spawn_wall_margin = 1.0
        if self._physical_target:
            # The legacy sampler's fixed 1 m inset can place a dynamic body inside the planner's
            # own stopping reserve (wall_margin + physical_boundary_margin = 1.25 m by default).
            # Start every physical actor inside the same admissible set used at runtime.
            spawn_wall_margin = max(
                spawn_wall_margin,
                float(self.cur.wall_margin)
                + float(getattr(self.tm, "physical_boundary_margin", 0.0)),
            )
        lo = b_min[:, 0:2] + spawn_wall_margin
        hi = b_max[:, 0:2] - spawn_wall_margin
        chosen = lo + (hi - lo) * torch.rand((n, 2), device=self.device)
        todo = torch.ones(n, dtype=torch.bool, device=self.device)
        min_dist, max_dist, _ = self._general_goal_distance_bounds()
        use_center_clearance = (
            self._target_dynamics in ("bounded", "physical") and self._target_speed_max() > 1e-6
        )
        spawn_clearance = (
            self._target_spawn_center_clearance()
            if use_center_clearance
            else float(getattr(self.task_config, "goal_min_bar_clearance", 0.65))
        )
        # The static-goal fallback (else branch above) was flat-clearance-from-centre, blind to
        # each bar's own footprint -- the same defect fixed for drone spawn on 2026-08-27
        # (_spawn_footprint_clearance_accepted). goal_min_bar_clearance exists to keep the 0.5 m
        # capture sphere actually flyable (navrl_task_config.py:670-671), which is exactly the
        # robot-footprint reachability question the geometry audit checks, so the same required
        # margin applies here.
        goal_required_margin = (
            None if use_center_clearance else self._spawn_required_surface_margin_m()
        )
        # Draw candidates in batches. The former one-candidate-per-env loop exhausted after 1024
        # samples in a valid 205-bar corrected layout during the 2026-08-31 gate. Every proposal
        # remains iid uniform and we retain the first accepted proposal, so batching increases the
        # fail-closed search budget without changing the conditional accepted distribution.
        candidates_per_round = 64
        rounds = 256 if self._target_dynamics in ("bounded", "physical") else 16
        attempts = candidates_per_round * rounds
        for _ in range(rounds):
            if not bool(todo.any()):
                break
            ids = todo.nonzero(as_tuple=False).squeeze(-1)
            # Uniform arena positions preserve the validated general-spawn task distribution.
            # The explicit radial acceptance range below makes its real contract observable.
            candidate = lo[ids].unsqueeze(1) + (
                hi[ids] - lo[ids]
            ).unsqueeze(1) * torch.rand(
                (len(ids), candidates_per_round, 2), device=self.device
            )
            drone_dist = torch.norm(
                candidate - start_pos[ids, 0:2].unsqueeze(1), dim=2
            )
            accepted = (
                (candidate[:, :, 0] >= lo[ids, 0].unsqueeze(1))
                & (candidate[:, :, 0] <= hi[ids, 0].unsqueeze(1))
                & (candidate[:, :, 1] >= lo[ids, 1].unsqueeze(1))
                & (candidate[:, :, 1] <= hi[ids, 1].unsqueeze(1))
                & (drone_dist >= min_dist)
                & (drone_dist <= max_dist)
            )
            if bars_xy.shape[1] > 0:
                if goal_required_margin is None:
                    bar_dist = (
                        torch.cdist(candidate, bars_xy[ids, : self.n_bars_active])
                        .min(2)
                        .values
                    )
                    accepted &= bar_dist >= spawn_clearance
                else:
                    flat_count = len(ids) * candidates_per_round
                    accepted &= _spawn_footprint_clearance_accepted(
                        candidate.reshape(flat_count, 2),
                        bars_xy[ids, : self.n_bars_active].repeat_interleave(
                            candidates_per_round, dim=0
                        ),
                        bar_half[ids, : self.n_bars_active].repeat_interleave(
                            candidates_per_round, dim=0
                        ),
                        goal_required_margin,
                    ).reshape(len(ids), candidates_per_round)
            has_candidate = accepted.any(dim=1)
            if bool(has_candidate.any()):
                first = accepted.to(dtype=torch.int64).argmax(dim=1)
                resolved = ids[has_candidate]
                chosen[resolved] = candidate[has_candidate, first[has_candidate]]
                todo[resolved] = False
        if self._target_dynamics in ("bounded", "physical") and bool(todo.any()):
            raise RuntimeError(
                "%s target spawn has no collision-free sample after %d attempts for %d envs"
                % (self._target_dynamics, attempts, int(todo.sum()))
            )
        goal = start_pos.clone()
        goal[:, 0:2] = chosen
        goal[:, 2] = float(self.task_config.flight_altitude)
        return goal

    def _register_interactive_viewer(self):
        """Attach NavRL controls and overlays to the already-created Isaac Gym viewer."""
        from isaacgym import gymapi

        viewer = getattr(getattr(self.sim_env, "IGE_env", None), "viewer", None)
        if viewer is None or getattr(viewer, "viewer", None) is None:
            raise RuntimeError("NAVRL_INTERACTIVE=1 requires headless=False (no viewer was created).")

        def on_press(fn):
            return lambda value: fn() if value > 0 else None

        if not self.general_eval_mode:
            viewer.subscribe_keyboard_event(
                gymapi.KEY_LEFT_BRACKET,
                "navrl_bars_down",
                on_press(lambda: self._interactive_change_bars(-5)),
            )
            viewer.subscribe_keyboard_event(
                gymapi.KEY_RIGHT_BRACKET,
                "navrl_bars_up",
                on_press(lambda: self._interactive_change_bars(5)),
            )
        viewer.subscribe_keyboard_event(
            gymapi.KEY_COMMA,
            "navrl_target_speed_down",
            on_press(lambda: self._interactive_change_target_speed(-0.25)),
        )
        viewer.subscribe_keyboard_event(
            gymapi.KEY_PERIOD,
            "navrl_target_speed_up",
            on_press(lambda: self._interactive_change_target_speed(0.25)),
        )
        viewer.subscribe_keyboard_event(
            gymapi.KEY_MINUS,
            "navrl_drone_speed_down",
            on_press(lambda: self._interactive_change_drone_speed(-0.25)),
        )
        viewer.subscribe_keyboard_event(
            gymapi.KEY_EQUAL,
            "navrl_drone_speed_up",
            on_press(lambda: self._interactive_change_drone_speed(0.25)),
        )
        viewer.subscribe_keyboard_event(
            gymapi.KEY_G,
            "navrl_toggle_lidar",
            on_press(self._interactive_toggle_lidar),
        )
        viewer.subscribe_keyboard_event(
            gymapi.KEY_M,
            "navrl_toggle_manual",
            on_press(self._interactive_toggle_manual),
        )
        viewer.subscribe_keyboard_event(
            gymapi.KEY_N,
            "navrl_reset",
            on_press(self._interactive_request_reset),
        )
        for key, action in (
            (gymapi.KEY_I, "manual_forward"),
            (gymapi.KEY_K, "manual_back"),
            (gymapi.KEY_J, "manual_left"),
            (gymapi.KEY_L, "manual_right"),
            (gymapi.KEY_U, "manual_yaw_left"),
            (gymapi.KEY_O, "manual_yaw_right"),
        ):
            viewer.subscribe_keyboard_event(
                key,
                "navrl_" + action,
                lambda value, name=action: self._interactive_manual_key(name, value),
            )
        viewer.add_render_callback(self._draw_interactive_overlay)
        if os.environ.get("NAVRL_3D_HUD", "1").strip().lower() not in ("0", "false", "no", "off"):
            try:
                from aerial_gym.apps.navrl_3d_hud import NavRL3DHud, build_hud_lines, build_hud_pip

                self._hud = NavRL3DHud()
            except Exception as exc:
                logger.warning("NavRL 3D HUD unavailable: %s" % exc)
                self._hud = None
        density_help = "" if self.general_eval_mode else "[/] bars±5  "
        logger.warning(
            "NavRL 3D controls | %s,/. target-speed±0.25  -/= drone-speed±0.25\n"
            "                     G LiDAR  N reset  M policy/manual  I/K/J/L move  U/O yaw\n"
            "                     red wireframe=debug target (never given to actor)"
            % density_help
        )

    def _interactive_request_reset(self):
        self._interactive_reset_requested = True
        logger.warning("NavRL 3D | reset requested")

    def _interactive_change_bars(self, delta):
        value = self.set_runtime_bars(self.n_bars_active + int(delta))
        logger.warning("NavRL 3D | bars=%d (reset requested)" % value)

    def _interactive_change_target_speed(self, delta):
        current = (
            float(self._runtime_target_speed)
            if self._runtime_target_speed is not None
            else float(self._tm_speed[0].item())
        )
        value = self.set_runtime_target_speed(current + float(delta))
        logger.warning("NavRL 3D | target speed=%.2f m/s (reset requested)" % value)

    def _interactive_change_drone_speed(self, delta):
        value = self.set_runtime_drone_speed(float(self.task_config.max_velocity) + float(delta))
        logger.warning("NavRL 3D | drone max speed=%.2f m/s" % value)

    def _interactive_toggle_lidar(self):
        self._interactive_show_lidar = not self._interactive_show_lidar
        logger.warning("NavRL 3D | LiDAR overlay=%s" % self._interactive_show_lidar)

    def _interactive_toggle_manual(self):
        self._interactive_manual = not self._interactive_manual
        self._interactive_manual_keys.clear()
        self._interactive_manual_action.zero_()
        logger.warning(
            "NavRL 3D | control=%s" % ("MANUAL" if self._interactive_manual else "POLICY")
        )

    def _interactive_manual_key(self, name, value):
        self._interactive_manual_keys[name] = max(0.0, float(value))
        fwd = self._interactive_manual_keys.get("manual_forward", 0.0)
        back = self._interactive_manual_keys.get("manual_back", 0.0)
        left = self._interactive_manual_keys.get("manual_left", 0.0)
        right = self._interactive_manual_keys.get("manual_right", 0.0)
        yaw_l = self._interactive_manual_keys.get("manual_yaw_left", 0.0)
        yaw_r = self._interactive_manual_keys.get("manual_yaw_right", 0.0)
        self._interactive_manual_action[:, 0] = fwd - back
        self._interactive_manual_action[:, 1] = left - right
        self._interactive_manual_action[:, 2] = 0.0
        self._interactive_manual_action[:, 3] = yaw_l - yaw_r

    def _draw_interactive_overlay(self):
        """Draw debug target/velocity and the selected environment's actual LiDAR scan."""
        viewer_ctl = self.sim_env.IGE_env.viewer
        gym = viewer_ctl.gym
        viewer = viewer_ctl.viewer
        env_id = int(viewer_ctl.current_target_env)
        env_handle = viewer_ctl.env_handles[env_id]
        gym.clear_lines(viewer)

        target = self.target_position[env_id].detach().cpu().numpy().astype(np.float32)
        # The target itself is a 0.3 m virtual drone. Draw a larger human-only marker and a short
        # trajectory trail so motion remains obvious in a 24x24 m overview camera.
        size = np.float32(0.35)
        corners = np.asarray(
            [[target[0] + sx * size, target[1] + sy * size, target[2] + sz * size]
             for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=np.float32,
        )
        edges = []
        for a in range(8):
            for bit in (1, 2, 4):
                b = a ^ bit
                if a < b:
                    edges.extend((corners[a], corners[b]))
        velocity_tip = target + self.target_vel_w[env_id].detach().cpu().numpy().astype(np.float32)
        edges.extend((target, velocity_tip))
        target_vertices = np.ascontiguousarray(np.asarray(edges, dtype=np.float32))
        target_colors = np.ascontiguousarray(
            np.vstack(
                [np.tile(np.asarray([[1.0, 0.08, 0.08]], dtype=np.float32), (12, 1)),
                 np.asarray([[0.1, 0.4, 1.0]], dtype=np.float32)]
            )
        )
        gym.add_lines(viewer, env_handle, 13, target_vertices, target_colors)

        if self._interactive_target_trail:
            jump = np.linalg.norm(target - self._interactive_target_trail[-1])
            if jump > 2.0:
                self._interactive_target_trail.clear()
        if not self._interactive_target_trail or np.linalg.norm(
            target - self._interactive_target_trail[-1]
        ) >= 0.03:
            self._interactive_target_trail.append(target.copy())
            self._interactive_target_trail = self._interactive_target_trail[-40:]
        if len(self._interactive_target_trail) >= 2:
            trail = np.asarray(self._interactive_target_trail, dtype=np.float32)
            trail_vertices = np.stack((trail[:-1], trail[1:]), axis=1).reshape(-1, 3)
            alpha = np.linspace(0.25, 1.0, len(trail) - 1, dtype=np.float32)
            trail_colors = np.stack((np.ones_like(alpha), 0.15 + 0.25 * alpha, 0.05 * alpha), axis=1)
            gym.add_lines(
                viewer,
                env_handle,
                len(trail) - 1,
                np.ascontiguousarray(trail_vertices),
                np.ascontiguousarray(trail_colors),
            )

        if self._interactive_show_lidar:
            ranges = self._lidar_distance_m()[env_id]
            # Ray directions must follow the warp generator's ordering: azimuth DECREASES with the
            # bin index (see navrl_perception.lidar_bin_bearings) and elevation DECREASES with the
            # scan line (warp_lidar.py: vfov_max at row 0). The old hard-coded 36-beam increasing
            # tables drew every ray mirrored left-right (and were stale at 72 beams).
            from aerial_gym.task.navrl_task.navrl_perception import VBEAMS, lidar_bin_bearings

            az = lidar_bin_bearings(self.device)
            el = torch.deg2rad(torch.linspace(20.0, -10.0, VBEAMS, device=self.device))
            ee, aa = torch.meshgrid(el, az, indexing="ij")
            dirs = torch.stack(
                (torch.cos(ee) * torch.cos(aa), torch.cos(ee) * torch.sin(aa), torch.sin(ee)),
                dim=-1,
            ).reshape(-1, 3)
            quat = self.obs_dict["robot_vehicle_orientation"][env_id].expand(dirs.shape[0], -1)
            dirs_w = quat_rotate(quat, dirs)
            origin = self.obs_dict["robot_position"][env_id]
            tips = origin.unsqueeze(0) + dirs_w * ranges.unsqueeze(1)
            lidar_vertices = torch.stack(
                (origin.expand_as(tips), tips), dim=1
            ).reshape(-1, 3).detach().cpu().numpy().astype(np.float32)
            hit = (ranges < float(self.task_config.lidar_max_range) * 0.99).detach().cpu().numpy()
            lidar_colors = np.zeros((len(hit), 3), dtype=np.float32)
            lidar_colors[~hit] = (0.10, 0.45, 0.10)
            lidar_colors[hit] = (1.00, 0.55, 0.05)
            gym.add_lines(
                viewer,
                env_handle,
                len(hit),
                np.ascontiguousarray(lidar_vertices),
                np.ascontiguousarray(lidar_colors),
            )

        if self._hud is not None and getattr(self._hud, "enabled", False):
            from aerial_gym.apps.navrl_3d_hud import build_hud_lines, build_hud_pip

            self._hud.update(build_hud_lines(self, env_id), build_hud_pip(self, env_id))

    def close(self):
        if self._obs_dump_enabled:
            # Deterministic, in-process flush point (F7): does not depend on interpreter shutdown,
            # and runs BEFORE the simulator is deleted below. Idempotent with the atexit trigger.
            self._flush_obs_dump()
        if self._hud is not None:
            self._hud.close()
            self._hud = None
        if self.general_eval_mode and not self._general_results_exported:
            self._export_general_results_json()
        self.sim_env.delete_env()

    # ------------------------------------------------------------------ checkpoint state
    def get_env_state(self):
        """Saved into the rl_games checkpoint ('env_state') so the epoch-proportional goal
        curriculum and optional density curriculum survive a --checkpoint resume."""
        self._verify_training_source_receipt()
        representation = self._obstacle_representation_or_zero()
        physics = self._runtime_physics_contract()
        rollback_raw = os.environ.get("NAVRL_PPO_EPOCH_ROLLBACK", "").strip().lower()
        if rollback_raw:
            rollback_enabled = rollback_raw in ("1", "true", "yes", "on")
        else:
            rollback_enabled = float(
                os.environ.get("NAVRL_PPO_KL_STOP", "0") or 0.0
            ) > 0.0
        return {
            "num_task_steps": int(self.num_task_steps),
            # Vehicle lineage.  This is intentionally separate from the observation/action
            # contract: navrl_quad and navrl_ref5in_quad have identical tensor shapes but different
            # rigid-body dynamics.  Evaluators must restore this identity before importing the task.
            "cfg_robot_contract_version": 1,
            "cfg_robot_name": self._robot_provenance["robot_name"],
            "cfg_robot_config_class": self._robot_provenance["robot_config_class"],
            "cfg_robot_config_module": self._robot_provenance["robot_config_module"],
            "cfg_robot_config_file": self._robot_provenance["robot_config_file"],
            "cfg_robot_config_path": self._robot_provenance["robot_config_path"],
            "cfg_robot_config_sha256": self._robot_provenance["robot_config_sha256"],
            "cfg_robot_asset_file": self._robot_provenance["robot_asset_file"],
            "cfg_robot_asset_path": self._robot_provenance["robot_asset_path"],
            "cfg_robot_asset_sha256": self._robot_provenance["robot_asset_sha256"],
            "cfg_training_source_manifest": self._training_source_provenance[
                "manifest"
            ],
            "cfg_training_source_manifest_sha256": self._training_source_provenance[
                "manifest_sha256"
            ],
            "cfg_training_source_git_commit": self._training_source_provenance[
                "git_commit"
            ],
            "cfg_training_source_git_dirty": self._training_source_provenance[
                "git_dirty"
            ],
            "cfg_training_source_runtime_file_count": self._training_source_provenance[
                "runtime_file_count"
            ],
            "cfg_ppo_horizon": int(getattr(self.cur, "ppo_horizon", 1)),
            "cfg_runtime_sim_config_class": physics["runtime_sim_config_class"],
            "cfg_physics_dt_s": physics["physics_dt_s"],
            "cfg_physics_substeps": physics["physics_substeps"],
            "cfg_physics_steps_per_rl_step": physics[
                "physics_steps_per_rl_step"
            ],
            "cfg_rl_step_dt_s": physics["rl_step_dt_s"],
            # Training/termination semantics.  Checkpoints created before 2026-08-10 lack these
            # fields and used a 601-action horizon for cfg_episode_len_steps=600 while silently
            # skipping rl_games' time-limit bootstrap.  New checkpoints make the correction
            # auditable instead of relying on the installed framework's naming convention.
            "cfg_episode_limit_comparator": "gte",
            "cfg_rlgames_timeout_info_key": "time_outs",
            "cfg_time_limit_bootstrap_signal": True,
            "cfg_reward_contract_version": "navrl_intercept_heuristic_v2",
            "cfg_reward_progress_kind": "ego_reanchored_heuristic",
            "cfg_reward_progress_gamma": float(self.task_config.progress_gamma),
            "cfg_reward_parameters": dict(self.rw),
            "cfg_action_command_semantics": (
                "vx_vy_axis_limited_vz_pi_yawrate_raw_z_in_prev_action"
            ),
            "n_bars_active": int(self.n_bars_active),
            "k_max_cur": float(self._k_max_cur),
            "k_min_cur": float(self._k_min_cur),
            # Scaling-critical settings, recorded so set_env_state() can warn when a checkpoint is
            # replayed under a different config. These are NOT restored (an eval may legitimately
            # override them) -- they exist purely to make a silent mismatch loud. lidar_max_range in
            # particular is BOTH the sensor horizon and the observation divisor (scan/range), so
            # evaluating an 8 m policy at 4 m rescales every scan input and the policy misreads the
            # world entirely -- a failure that looks like "the policy is broken", not like a config
            # error. Learned the hard way twice in one session.
            "cfg_lidar_max_range": float(self.task_config.lidar_max_range),
            "cfg_max_velocity": float(self.task_config.max_velocity),
            "cfg_yaw_rate_max": float(self.task_config.yaw_rate_max),
            "cfg_max_tilt_deg": float(
                os.environ.get("NAVRL_MAX_TILT_DEG", "").strip() or 45.0
            ),
            "cfg_tilt_comp": os.environ.get("NAVRL_TILT_COMP", "1").strip().lower()
            not in ("0", "false", "no", "off"),
            "cfg_max_obstacles": int(representation["max_obstacles"]),
            "cfg_token_fov_deg": float(representation["token_fov_deg"]),
            "cfg_token_effective_fov_deg": float(
                representation["token_effective_fov_deg"]
            ),
            "cfg_obstacle_suppress_deg": float(representation["suppress_deg"]),
            "cfg_obstacle_suppress_active": bool(representation["suppress_active"]),
            "cfg_obstacle_selector": str(representation["selector"]),
            "cfg_obstacle_cluster_gap_m": float(representation["cluster_gap_m"]),
            "cfg_obstacle_sectors": int(representation["sectors"]),
            "cfg_obstacle_ttc_idle_s": float(representation["ttc_idle_s"]),
            "cfg_obstacle_ttc_min_speed": float(representation["ttc_min_speed"]),
            "cfg_lidar_hbeams": int(representation["hbeams"]),
            "cfg_lidar_vbeams": int(representation["vbeams"]),
            "cfg_corridor_tokens": int(representation["corridor_tokens"]),
            "cfg_corridor_horizon_m": float(representation["corridor_horizon_m"]),
            "cfg_corridor_min_width_m": float(representation["corridor_min_width_m"]),
            "cfg_geofence_actor": bool(representation["geofence_actor"]),
            "cfg_geofence_noise_std_m": float(representation["geofence_noise_std_m"]),
            "cfg_geofence_dropout": float(representation["geofence_dropout"]),
            "cfg_search_state": str(representation["search_state"]),
            "cfg_search_state_force_invalid": bool(
                representation["search_state_force_invalid"]
            ),
            "cfg_fov_curriculum_epochs": int(
                getattr(self.vis_cfg, "fov_curriculum_epochs", 0)
            ),
            "cfg_detector_min_pixels": int(
                getattr(self.perception_cfg, "min_target_pixels", 0)
            ),
            # These three fields define the target detector's geometric support.  Observation
            # width does not change when they change, so a shape-compatible checkpoint can load
            # under the wrong sensor contract unless the checkpoint records them explicitly.
            "cfg_detector_max_range": float(
                getattr(self.vis_cfg, "detector_max_range", 0.0)
            ),
            "cfg_detect_width": int(getattr(self.vis_cfg, "detect_width", 0)),
            "cfg_detect_height": int(getattr(self.vis_cfg, "detect_height", 0)),
            "cfg_detector_threshold": float(
                getattr(self.perception_cfg, "pixel_threshold", 0.0)
            ),
            "cfg_detector_checkpoint_name": self._detector_checkpoint_name,
            "cfg_detector_checkpoint_sha256": self._detector_checkpoint_sha256,
            "cfg_perception_perturb": bool(
                getattr(self.perception_cfg, "enable_perturbations", False)
            ),
            "cfg_detection_dropout": float(
                getattr(self.perception_cfg, "detection_dropout_prob", 0.0)
            ),
            "cfg_detection_latency_s": float(
                getattr(self.perception_cfg, "detection_latency_s", 0.0)
            ),
            "cfg_range_error_m": float(
                getattr(self.perception_cfg, "range_error_m", 0.0)
            ),
            "cfg_rgb_noise_std": float(
                getattr(self.perception_cfg, "rgb_noise_std", 0.0)
            ),
            "cfg_depth_noise_std": float(
                getattr(self.perception_cfg, "depth_noise_std", 0.0)
            ),
            "cfg_p9_empirical_error_enabled": bool(
                getattr(getattr(self, "perception", None), "empirical_error", None) is not None
            ),
            "cfg_p9_empirical_error_sha256": (
                self.perception.empirical_error.sha256
                if getattr(getattr(self, "perception", None), "empirical_error", None) is not None else ""
            ),
            "cfg_p9_empirical_error_seed": int(
                getattr(self.perception_cfg, "empirical_error_seed", 1701)
            ),
            "cfg_appearance_hue_deg": float(
                getattr(self.vis_cfg, "appearance_hue_deg", 0.0)
            ),
            "cfg_appearance_light_gain": float(
                getattr(self.vis_cfg, "appearance_light_gain", 0.0)
            ),
            "cfg_appearance_albedo_jitter": float(
                getattr(self.vis_cfg, "appearance_albedo_jitter", 0.0)
            ),
            "cfg_appearance_texture_std": float(
                getattr(self.vis_cfg, "appearance_texture_std", 0.0)
            ),
            "cfg_appearance_motion_blur": float(
                getattr(self.vis_cfg, "appearance_motion_blur", 0.0)
            ),
            "cfg_camera_mount_rot_deg": float(
                getattr(self.vis_cfg, "camera_mount_rot_deg", 0.0)
            ),
            "cfg_camera_mount_trans_m": float(
                getattr(self.vis_cfg, "camera_mount_trans_m", 0.0)
            ),
            "cfg_camera_fov_scale_err": float(
                getattr(self.vis_cfg, "camera_fov_scale_err", 0.0)
            ),
            "cfg_target_motion_model": self._target_motion_model,
            # The heading-validity threshold is a target-motion contract term, not a tuning
            # knob: record the value actually in force and whether it was attested or assumed.
            HEADING_VALID_SPEED_KEY: float(self._heading_valid_speed_mps),
            HEADING_VALID_SPEED_PROVENANCE_KEY: self._heading_valid_speed_provenance,
            "cfg_target_route_recovery_schema": (
                TARGET_ROUTE_RECOVERY_SCHEMA if self._target_route_recovery_enabled else "off"
            ),
            "cfg_target_route_recovery_model": (
                TARGET_ROUTE_RECOVERY_MODEL if self._target_route_recovery_enabled else "off"
            ),
            "cfg_target_route_recovery_hard_envelope": (
                "closed_aabb_support_v1" if self._target_route_recovery_enabled else "off"
            ),
            "cfg_target_route_recovery_soft_envelope": (
                "closed_aabb_support_plus_tracking_v1" if self._target_route_recovery_enabled else "off"
            ),
            "cfg_target_route_recovery_hard_epsilon_m": 1e-4 if self._target_route_recovery_enabled else 0.0,
            "cfg_target_route_recovery_reachable_tube_margin_m": (
                TARGET_ROUTE_REACHABLE_TUBE_MARGIN_M if self._target_route_recovery_enabled else 0.0
            ),
            "cfg_target_route_recovery_hysteresis_m": RECOVERY_HYSTERESIS_M if self._target_route_recovery_enabled else 0.0,
            "cfg_target_route_recovery_stop_speed_mps": RECOVERY_STOP_SPEED_MPS if self._target_route_recovery_enabled else 0.0,
            "cfg_target_route_recovery_progress_tolerance_m": (
                RECOVERY_CONNECT_PROGRESS_TOLERANCE_M if self._target_route_recovery_enabled else 0.0
            ),
            "cfg_target_route_recovery_anchor_radius_cells": 3 if self._target_route_recovery_enabled else 0,
            "cfg_target_dynamics": self._target_dynamics,
            "cfg_target_max_accel_mps2": float(self.tm.max_accel),
            "cfg_target_max_turn_rate_degps": float(self.tm.max_turn_rate_deg),
            "cfg_target_lookahead_s": float(self.tm.avoidance_lookahead_s),
            "cfg_target_obstacle_clearance_m": float(self.tm.obstacle_clearance),
            "cfg_target_physical_mass_kg": float(self.tm.physical_mass),
            "cfg_target_physical_motor_arm_xy_m": float(self.tm.physical_motor_arm_xy),
            "cfg_target_physical_max_motor_thrust_n": float(self.tm.physical_max_motor_thrust),
            "cfg_target_physical_motor_tau_s": float(self.tm.physical_motor_tau),
            "cfg_target_physical_yaw_torque_ratio_m": float(self.tm.physical_yaw_torque_ratio),
            "cfg_target_physical_max_tilt_deg": float(self.tm.physical_max_tilt_deg),
            "cfg_target_physical_box_xyz_m": [
                float(value) for value in self.tm.physical_box_xyz
            ],
            "cfg_target_physical_tracking_margin_m": float(self.tm.physical_tracking_margin),
            "cfg_target_physical_boundary_margin_m": float(self.tm.physical_boundary_margin),
            "cfg_target_route_wall_margin_m": float(self.cur.wall_margin),
            "cfg_target_recovery_brake_decel_p05_mps2": float(
                getattr(self.tm, "recovery_brake_decel_p05", 0.0)
            ),
            "cfg_target_recovery_stop_time_p95_s": float(
                getattr(self.tm, "recovery_brake_stop_time_p95", 0.0)
            ),
            "cfg_target_recovery_probe_receipt_sha256": self._recovery_probe_receipt_sha256,
            "cfg_target_recovery_braking_contract_variant": str(
                getattr(self.tm, "recovery_braking_contract_variant", "canonical_1p5")
            ),
            "cfg_target_recovery_brake_speed_samples_mps": list(self._recovery_brake_speed_samples_mps),
            "cfg_target_recovery_brake_stop_distance_samples_m": list(self._recovery_brake_stop_distance_samples_m),
            "cfg_target_recovery_brake_lateral_tube_p95_m": self._recovery_brake_lateral_tube_p95_m,
            "cfg_target_recovery_timeout_steps": (
                max(1, int(math.ceil((float(getattr(self.tm, "recovery_brake_stop_time_p95", 0.0)) + 0.20) / self.step_dt)))
                if self._target_route_recovery_enabled else 0
            ),
            "cfg_target_route_mode": self._target_route_mode,
            "cfg_target_route_braking_v3_schema": (
                TARGET_ROUTE_BRAKING_V3_SCHEMA if self._target_route_braking_v3_enabled else "off"
            ),
            "cfg_target_route_braking_v3_model": (
                TARGET_ROUTE_BRAKING_V3_MODEL if self._target_route_braking_v3_enabled else "off"
            ),
            "cfg_target_route_braking_v3_soft_envelope": (
                TARGET_ROUTE_BRAKING_GEOMETRY_SCHEMA if self._target_route_braking_v3_enabled else "off"
            ),
            "cfg_target_route_resolution_m": float(self.tm.route_resolution_m),
            "cfg_target_route_max_expansions": int(self.tm.route_max_expansions),
            "cfg_target_route_max_waypoints": int(self.tm.route_max_waypoints),
            "cfg_target_route_replan_cooldown_steps": int(
                self.tm.route_replan_cooldown_steps
            ),
            "cfg_target_route_goal_tolerance_m": float(self.tm.route_goal_tolerance_m),
            "cfg_target_route_min_goal_distance_m": float(
                self.tm.route_min_goal_distance_m
            ),
            "cfg_target_route_goal_exclusion_radius_m": float(
                self.tm.route_goal_exclusion_radius_m
            ),
            "cfg_target_route_support_xy_m": [
                float(value) for value in self._target_route_support_xy[0].tolist()
            ],
            "cfg_target_pattern": str(self.tm.pattern),
            "cfg_target_speed_min": float(getattr(self.tm, "speed_min", 0.0)),
            "cfg_target_speed_final": float(self.tm.speed_final),
            "cfg_target_speed_fixed": float(self.tm.speed_fixed),
            "cfg_target_speed_ramp_epochs": int(self.tm.speed_ramp_epochs),
            "cfg_target_speed_ramp_start_epochs": int(
                self.tm.speed_ramp_start_epochs
            ),
            "cfg_general_train": bool(self.general_train_mode),
            "cfg_general_goal_dist_min": float(self.general_goal_dist_min),
            "cfg_general_goal_dist_max": float(self.general_goal_dist_max),
            "cfg_oob_margin": float(self.vis_cfg.oob_margin),
            "cfg_alt_hold_vmax": float(
                getattr(self.task_config, "alt_hold_vmax", self.task_config.max_velocity)
            ),
            # ARENA / task-version provenance (v2 search arena, 2026-07-31). The observation
            # width does NOT change with these, so a v2 checkpoint loads cleanly in a v1 arena
            # and would silently measure a completely different task -- the same failure class as
            # the old lidar_max_range mismatch. Recorded here so set_env_state() and the eval
            # preflight can make that loud.
            **self._arena_contract(),
            # Action-distribution provenance. Bounded and legacy models intentionally share the
            # same state_dict keys, so without this an eval can load successfully under the wrong
            # likelihood and silently measure a different policy.
            "cfg_action_policy": os.environ.get("NAVRL_ACTION_POLICY", "legacy"),
            "cfg_action_std": os.environ.get("NAVRL_ACTION_STD", ""),
            "cfg_action_mu_scale": os.environ.get("NAVRL_ACTION_MU_SCALE", "1"),
            "cfg_action_entropy_coef": float(
                os.environ.get("NAVRL_ENTROPY_COEF", "0") or 0.0
            ),
            "cfg_speed_governor_mode": self.speed_governor_cfg.mode,
            "cfg_speed_governor_fixed_mps": self.speed_governor_cfg.fixed_cap_mps,
            "cfg_speed_governor_free_mps": self.speed_governor_cfg.free_speed_cap_mps,
            "cfg_speed_governor_half_width_m": self.speed_governor_cfg.path_half_width_m,
            "cfg_speed_governor_margin_m": self.speed_governor_cfg.hard_margin_m,
            "cfg_speed_governor_slow_m": self.speed_governor_cfg.slow_distance_m,
            "cfg_speed_governor_release_m": self.speed_governor_cfg.release_distance_m,
            "cfg_speed_governor_ttc_s": self.speed_governor_cfg.ttc_s,
            "cfg_speed_governor_brake_mps2": self.speed_governor_cfg.brake_mps2,
            "cfg_speed_governor_reaction_s": self.speed_governor_cfg.reaction_s,
            "cfg_speed_governor_width_per_mps": self.speed_governor_cfg.width_per_mps,
            "cfg_speed_governor_width_per_open_m": self.speed_governor_cfg.width_per_open_m,
            "cfg_speed_governor_width_open_ref_m": self.speed_governor_cfg.width_open_ref_m,
            "cfg_speed_governor_width_max_m": self.speed_governor_cfg.width_max_m,
            "cfg_speed_governor_lateral_margin_m": self.speed_governor_cfg.lateral_margin_m,
            "cfg_speed_governor_lateral_span_m": self.speed_governor_cfg.lateral_span_m,
            "cfg_speed_governor_lateral_floor_mps": self.speed_governor_cfg.lateral_floor_mps,
            "cfg_speed_governor_yaw_cap_radps": self.speed_governor_cfg.yaw_cap_radps,
            "cfg_speed_governor_yaw_cap_margin_m": self.speed_governor_cfg.yaw_cap_margin_m,
            "cfg_speed_governor_target_exclusion": "camera_lidar_association",
            "cfg_training_seed": int(self.task_config.seed),
            "cfg_training_num_envs": int(self.num_envs),
            "cfg_training_file": os.environ.get("FILE", ""),
            "cfg_training_task": os.environ.get("TASK", ""),
            "cfg_training_sim": os.environ.get("AERIAL_GYM_SIM_NAME", ""),
            "cfg_training_profile": os.environ.get("NAVRL_V2_PROFILE", "main"),
            "cfg_action_learning_rate": float(
                os.environ.get("NAVRL_LEARNING_RATE", "0") or 0.0
            ),
            "current_action_learning_rate": float(
                os.environ.get(
                    "NAVRL_CURRENT_LEARNING_RATE",
                    os.environ.get("NAVRL_LEARNING_RATE", "0"),
                )
                or 0.0
            ),
            "cfg_recovery_stage": os.environ.get("NAVRL_RECOVERY_STAGE", ""),
            "cfg_recovery_source_epoch": int(
                os.environ.get("NAVRL_RECOVERY_SOURCE_EPOCH", "-1") or -1
            ),
            "cfg_recovery_source_sha256": os.environ.get(
                "NAVRL_RECOVERY_SOURCE_SHA256", ""
            ),
            "cfg_recovery_smoke_required_epochs": int(
                os.environ.get("NAVRL_RECOVERY_SMOKE_REQUIRED_EPOCHS", "-1") or -1
            ),
            "cfg_recovery_smoke_bars": int(
                os.environ.get("NAVRL_RECOVERY_SMOKE_BARS", "-1") or -1
            ),
            "cfg_recovery_eval_attestation_sha256": os.environ.get(
                "NAVRL_RECOVERY_EVAL_ATTESTATION_SHA256", ""
            ),
            "cfg_recovery_eval_attestation_b64": os.environ.get(
                "NAVRL_RECOVERY_EVAL_ATTESTATION_B64", ""
            ),
            "cfg_ppo_log_ratio_clamp": float(
                os.environ.get("NAVRL_PPO_LOG_RATIO_CLAMP", "0") or 0.0
            ),
            "cfg_ppo_kl_stop": float(
                os.environ.get("NAVRL_PPO_KL_STOP", "0") or 0.0
            ),
            "cfg_ppo_epoch_rollback": rollback_enabled,
            "cfg_ppo_rollback_lr_factor": float(
                os.environ.get("NAVRL_PPO_ROLLBACK_LR_FACTOR", "0.5") or 0.5
            ),
            "cfg_ppo_rollback_min_lr": float(
                os.environ.get("NAVRL_PPO_ROLLBACK_MIN_LR", "1e-6") or 1e-6
            ),
            "cfg_ppo_rollback_patience": int(
                os.environ.get("NAVRL_PPO_ROLLBACK_PATIENCE", "5") or 5
            ),
            "cfg_density_guard_window_epochs": int(
                os.environ.get("NAVRL_DENSITY_GUARD_WINDOW_EPOCHS", "50") or 50
            ),
            "cfg_density_guard_min_epochs": int(
                os.environ.get("NAVRL_DENSITY_GUARD_MIN_EPOCHS", "100") or 100
            ),
            "cfg_density_guard_min_peak": float(
                os.environ.get("NAVRL_DENSITY_GUARD_MIN_PEAK", "0.5") or 0.5
            ),
            "cfg_density_guard_drop": float(
                os.environ.get("NAVRL_DENSITY_GUARD_DROP", "0.25") or 0.25
            ),
            "cfg_density_guard_patience": int(
                os.environ.get("NAVRL_DENSITY_GUARD_PATIENCE", "25") or 25
            ),
            "cfg_lateral_latent_margin_y": float(
                os.environ.get("NAVRL_LATENT_MARGIN_Y", "0") or 0.0
            ),
            "cfg_latent_margin": os.environ.get("NAVRL_LATENT_MARGIN", ""),
            "cfg_lateral_latent_margin_coef": float(
                os.environ.get("NAVRL_LATENT_MARGIN_COEF", "0") or 0.0
            ),
            "cfg_lateral_bias_coef": float(
                os.environ.get("NAVRL_LATERAL_BIAS_COEF", "0") or 0.0
            ),
            "cfg_reflection_coef": float(
                os.environ.get("NAVRL_REFLECTION_COEF", "0") or 0.0
            ),
            "cfg_truncated_dmin": float(
                os.environ.get("NAVRL_TRUNCATED_DMIN", "0.01") or 0.01
            ),
            # Preserve the in-progress competence window. With a 16k-episode density gate, dropping
            # these counters on every periodic-checkpoint resume can discard hours of evidence and
            # indefinitely postpone the next promotion.
            "density_succ_agg": int(self._density_succ_agg),
            "density_fin_agg": int(self._density_fin_agg),
            "density_gate_not_before_steps": int(self._density_gate_not_before_steps),
            "density_level_start_steps": int(self._density_level_start_steps),
            "density_speed_succ": self._density_speed_succ.detach().cpu().tolist(),
            "density_speed_fin": self._density_speed_fin.detach().cpu().tolist(),
            "density_dist_succ": self._density_dist_succ.detach().cpu().tolist(),
            "density_dist_fin": self._density_dist_fin.detach().cpu().tolist(),
            "density_pattern_succ": self._density_pattern_succ.detach().cpu().tolist(),
            "density_pattern_fin": self._density_pattern_fin.detach().cpu().tolist(),
            "cfg_density_final": int(getattr(self.density, "n_final", self.n_bars_active)),
            "cfg_density_step": int(getattr(self.density, "promote_step", 0)),
            "cfg_density_threshold": float(
                getattr(self.density, "success_threshold", 0.0)
            ),
            "cfg_density_threshold_start": float(
                getattr(self.density, "success_threshold_start", 0.0)
            ),
            "cfg_density_threshold_end": float(
                getattr(self.density, "success_threshold_end", 0.0)
            ),
            "cfg_density_threshold_schedule": str(
                getattr(self.density, "success_threshold_schedule", "") or ""
            ),
            "cfg_density_check_eps": int(
                getattr(self.density, "check_after_episodes", 0)
            ),
            "cfg_density_min_epochs": int(
                getattr(self.density, "min_epochs_per_density", 0)
            ),
            "cfg_density_stratified_gate": bool(
                getattr(self.density, "use_stratified_gate", False)
            ),
            "cfg_density_stratified_floor": float(
                getattr(self.density, "stratified_floor", 0.0)
            ),
        }

    @staticmethod
    def _arena_contract():
        """Arena / task-version settings recorded in every checkpoint.

        These change the TASK, not the observation width, so nothing downstream would fail
        loudly on a mismatch without this record. Read from the same env vars that
        navrl_bars_env and navrl_task_config consume, so the record always equals what the
        environment was actually built with.
        """
        return {
            "cfg_arena_xy": float(os.environ.get("NAVRL_ARENA_XY", "").strip() or 24.0),
            "cfg_arena_z": float(os.environ.get("NAVRL_ARENA_Z", "").strip() or 3.0),
            "cfg_bar_pool": (os.environ.get("NAVRL_BAR_POOL", "").strip() or "bars"),
            "cfg_placement_mode": (
                os.environ.get("NAVRL_PLACEMENT_MODE", "").strip().lower() or "random"
            ),
            "cfg_placement_gap_m": float(
                os.environ.get("NAVRL_PLACEMENT_GAP_M", "").strip() or 1.6
            ),
            "cfg_placement_touch_m": float(
                os.environ.get("NAVRL_PLACEMENT_TOUCH_M", "").strip() or 0.4
            ),
            "cfg_placement_surface_clearance_m": float(
                os.environ.get("NAVRL_PLACEMENT_SURFACE_CLEARANCE_M", "").strip() or 0.0
            ),
            "cfg_episode_len_steps": float(
                os.environ.get("NAVRL_EPISODE_LEN_STEPS", "").strip() or 300
            ),
            # Obstacle placement band as a fraction of the arena. Same failure class as the arena
            # fields: changing it changes the task with no shape error to catch it.
            "cfg_bar_x_min": float(os.environ.get("NAVRL_BAR_X_MIN", "").strip() or 0.13),
            "cfg_bar_x_max": float(os.environ.get("NAVRL_BAR_X_MAX", "").strip() or 0.96),
        }

    @staticmethod
    def _obstacle_representation_or_zero():
        """Policy obstacle-representation settings, or zeros when perception is unavailable."""
        try:
            from aerial_gym.task.navrl_task.navrl_perception import (
                CORRIDOR_HORIZON_M,
                CORRIDOR_MIN_WIDTH_M,
                CORRIDOR_TOKENS,
                GEOFENCE_ACTOR,
                GEOFENCE_DROPOUT,
                GEOFENCE_NOISE_STD_M,
                HBEAMS,
                MAX_OBSTACLES,
                OBSTACLE_CLUSTER_GAP_M,
                OBSTACLE_EFFECTIVE_FOV_DEG,
                OBSTACLE_FOV_DEG,
                OBSTACLE_SECTORS,
                OBSTACLE_SELECTOR,
                OBSTACLE_SUPPRESS_DEG,
                OBSTACLE_SUPPRESS_ACTIVE,
                OBSTACLE_TTC_IDLE_S,
                OBSTACLE_TTC_MIN_SPEED,
                SEARCH_STATE,
                SEARCH_STATE_FORCE_INVALID,
                VBEAMS,
            )

            return {
                "max_obstacles": int(MAX_OBSTACLES),
                "token_fov_deg": float(OBSTACLE_FOV_DEG),
                # TTC ranks every LiDAR cluster, including the configured rear exclusion.  Keep
                # configured FOV for checkpoint compatibility, but record what selection actually
                # consumes so diagnostics do not call rear TTC tokens "out of FOV".
                "token_effective_fov_deg": float(OBSTACLE_EFFECTIVE_FOV_DEG),
                "suppress_deg": float(OBSTACLE_SUPPRESS_DEG),
                "suppress_active": bool(OBSTACLE_SUPPRESS_ACTIVE),
                "selector": str(OBSTACLE_SELECTOR),
                "cluster_gap_m": float(OBSTACLE_CLUSTER_GAP_M),
                "sectors": int(OBSTACLE_SECTORS),
                "ttc_idle_s": float(OBSTACLE_TTC_IDLE_S),
                "ttc_min_speed": float(OBSTACLE_TTC_MIN_SPEED),
                "hbeams": int(HBEAMS),
                "vbeams": int(VBEAMS),
                "corridor_tokens": int(CORRIDOR_TOKENS),
                "corridor_horizon_m": float(CORRIDOR_HORIZON_M),
                "corridor_min_width_m": float(CORRIDOR_MIN_WIDTH_M),
                "geofence_actor": bool(GEOFENCE_ACTOR),
                "geofence_noise_std_m": float(GEOFENCE_NOISE_STD_M),
                "geofence_dropout": float(GEOFENCE_DROPOUT),
                "search_state": str(SEARCH_STATE),
                "search_state_force_invalid": bool(SEARCH_STATE_FORCE_INVALID),
            }
        except Exception:
            return {
                "max_obstacles": 0,
                "token_fov_deg": 0.0,
                "token_effective_fov_deg": 0.0,
                "suppress_deg": 0.0,
                "suppress_active": False,
                "selector": "none",
                "cluster_gap_m": 0.0,
                "sectors": 0,
                "ttc_idle_s": 0.0,
                "ttc_min_speed": 0.0,
                "hbeams": 0,
                "vbeams": 0,
                "corridor_tokens": 0,
                "corridor_horizon_m": 0.0,
                "corridor_min_width_m": 0.0,
                "geofence_actor": False,
                "geofence_noise_std_m": 0.0,
                "geofence_dropout": 0.0,
                "search_state": "off",
                "search_state_force_invalid": False,
            }

    @classmethod
    def _token_fov_or_zero(cls):
        return float(cls._obstacle_representation_or_zero()["token_fov_deg"])

    def _record_episode_dump(self, pos, captured, crashed_out, crashed, below, above):
        """Append one row per terminated episode for the offline reachability oracle."""
        done = captured | crashed_out
        idx = done.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return
        cap = captured[idx]
        con = (crashed & crashed_out)[idx]
        bel = (below & ~crashed & crashed_out)[idx]
        abv = (above & ~crashed & ~below & crashed_out)[idx]
        outcome = torch.where(
            cap,
            torch.zeros_like(cap, dtype=torch.long),
            torch.where(
                con,
                torch.ones_like(cap, dtype=torch.long),
                torch.where(
                    bel,
                    torch.full_like(cap, 2, dtype=torch.long),
                    torch.where(
                        abv,
                        torch.full_like(cap, 3, dtype=torch.long),
                        torch.full_like(cap, 4, dtype=torch.long),
                    ),
                ),
            ),
        )
        bars = self.obs_dict["obstacle_position"][idx][
            :, self._bar_offset : self._bar_offset + self.n_bars_active, 0:2
        ]
        self._dump_records.append(
            {
                "outcome": outcome.to(torch.int8).cpu(),
                "spawn": self._episode_spawn[idx].to(torch.float16).cpu(),
                "end_pos": pos[idx].to(torch.float16).cpu(),
                "target_end": self.target_position[idx].to(torch.float16).cpu(),
                "bars_xy": bars.to(torch.float16).cpu(),
            }
        )

    def _flush_episode_dump(self):
        if not getattr(self, "_dump_records", None):
            return
        merged = {
            key: torch.cat([record[key] for record in self._dump_records]).numpy()
            for key in self._dump_records[0]
        }
        path = Path(self._episode_dump_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **merged)
        logger.warning(
            "NavRL episode dump | %d terminated episodes -> %s"
            % (int(merged["outcome"].shape[0]), path)
        )

    def _record_bar_contact_probe(self, hit_mask, pos):
        """Measure which current obstacle tokens geometrically represent a struck bar.

        Evaluation-only (NAVRL_BAR_PROBE=1). Uses ground-truth bar positions, which is legitimate
        here because nothing computed in this method reaches the actor, the critic, the reward, or
        any termination. Probe v2 separates FOV eligibility, range+bearing association, lateral ray
        offset, radial surface gap, and duplicate token use instead of collapsing them into the old
        ambiguous ``token_err``.
        """
        from aerial_gym.task.navrl_task.bar_probe import associate_surface_tokens_to_bars
        from aerial_gym.task.navrl_task.navrl_perception import MAX_OBSTACLES, OBSTACLE_DIM

        idx = hit_mask.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return
        rng = float(self.task_config.lidar_max_range)

        # True bars, expressed in the drone's vehicle frame (yaw-only, matching the LiDAR frame).
        bars_w = self.obs_dict["obstacle_position"][idx][
            :, self._bar_offset : self._bar_offset + self.n_bars_active, 0:3
        ]  # (K, B, 3)
        rel_w = bars_w - pos[idx].unsqueeze(1)
        quat = self.obs_dict["robot_vehicle_orientation"][idx]  # (K, 4)
        k, b = rel_w.shape[0], rel_w.shape[1]
        rel_v = quat_rotate_inverse(
            quat.unsqueeze(1).expand(k, b, 4).reshape(k * b, 4), rel_w.reshape(k * b, 3)
        ).reshape(k, b, 3)
        bar_dist = rel_v[:, :, 0:2].norm(dim=2)  # (K, B) horizontal distance to each bar
        bar_bearing = torch.atan2(rel_v[:, :, 1], rel_v[:, :, 0])

        # The bar that was struck = the closest one at the moment of contact.
        hit_d, hit_i = bar_dist.min(dim=1)
        rows = torch.arange(k, device=self.device)

        # Crowding, two independent ways: GT bars inside the range/FOV and scan bearings returning
        # something.  The FOV split is essential: a 240-deg token sector cannot represent a struck
        # bar in the excluded rear 120 deg, although the full static scan still observes it.
        bars_in_range = (bar_dist < rng).sum(dim=1).float()
        effective_fov_deg = float(
            self._obstacle_representation_or_zero()["token_effective_fov_deg"]
        )
        if effective_fov_deg < 359.9:
            in_token_fov = bar_bearing.abs() <= math.radians(effective_fov_deg * 0.5)
        else:
            in_token_fov = torch.ones_like(bar_bearing, dtype=torch.bool)
        bars_in_token_fov = ((bar_dist < rng) & in_token_fov).sum(dim=1).float()
        hit_in_token_fov = in_token_fov[rows, hit_i]
        scan = getattr(self.perception, "last_scan_nearest", None)
        occupied = (
            (scan[idx] < rng * 0.995).sum(dim=1).float()
            if scan is not None
            else torch.zeros_like(bars_in_range)
        )

        # Associate each LiDAR surface token to a plausible GT bar using BOTH bearing and range.
        # The old +/-15-deg bearing-only matcher routinely attached a nearer token to a farther bar
        # in dense scenes and then mislabeled surface-to-center distance as token position error.
        tokens = self.perception.obstacle_history[idx, -1].view(k, MAX_OBSTACLES, OBSTACLE_DIM)
        tok_pos = tokens[:, :, 0:2] * rng  # positions are stored normalized by the LiDAR range
        tok_valid = tokens[:, :, 11] > 0.5
        association = associate_surface_tokens_to_bars(tok_pos, tok_valid, rel_v[:, :, 0:2])
        token_bar = association["bar_index"]
        token_associated = association["associated"]
        hit_token = token_associated & (token_bar == hit_i.unsqueeze(1))
        hit_offsets = torch.where(
            hit_token,
            association["center_offset"],
            torch.full_like(association["center_offset"], float("inf")),
        )
        best_offset, best_slot = hit_offsets.min(dim=1)
        matched = torch.isfinite(best_offset)

        represented = (
            torch.nn.functional.one_hot(token_bar, num_classes=b).bool()
            & token_associated.unsqueeze(2)
        ).any(dim=1)
        associated_count = token_associated.sum(dim=1).float()
        unique_count = represented.sum(dim=1).float()

        self._bprobe["n"] += float(k)
        self._bprobe["bars_in_range"] += float(bars_in_range.sum().item())
        self._bprobe["bars_in_token_fov"] += float(bars_in_token_fov.sum().item())
        self._bprobe["occupied_bins"] += float(occupied.sum().item())
        self._bprobe["hit_dist"] += float(hit_d.sum().item())
        self._bprobe["hit_in_token_fov"] += float(hit_in_token_fov.sum().item())
        self._bprobe["hit_in_tokens"] += float(matched.sum().item())
        self._bprobe["hit_in_tokens_in_fov"] += float(
            (matched & hit_in_token_fov).sum().item()
        )
        self._bprobe["valid_tokens"] += float(tok_valid.sum().item())
        self._bprobe["associated_tokens"] += float(associated_count.sum().item())
        self._bprobe["unique_token_bars"] += float(unique_count.sum().item())
        self._bprobe["duplicate_tokens"] += float((associated_count - unique_count).sum().item())
        if bool(matched.any()):
            m = matched.nonzero(as_tuple=False).squeeze(1)
            slot = best_slot[m]
            self._bprobe["hit_center_offset"] += float(best_offset[m].sum().item())
            self._bprobe["hit_cross_track"] += float(
                association["cross_track"][m, slot].sum().item()
            )
            self._bprobe["hit_radial_gap"] += float(
                association["radial_gap"][m, slot].sum().item()
            )
            self._bprobe["hit_token_rank"] += float(best_slot[m].float().sum().item())

    @staticmethod
    def _max_obstacles_or_zero():
        """Obstacle-token capacity, or 0 when perception is off (import stays lazy on purpose)."""
        return int(NavRLTask._obstacle_representation_or_zero()["max_obstacles"])

    def set_env_state(self, state):
        bars_before_restore = int(self.n_bars_active)
        force_density_reset = os.environ.get(
            "NAVRL_RESET_DENSITY_WINDOW", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        density_evidence_changed = force_density_reset
        if isinstance(state, dict):
            representation = self._obstacle_representation_or_zero()
            arena = self._arena_contract()
            # Vehicle configs are shape-compatible, so loading succeeds even when the physical
            # body is wrong.  Older checkpoints predate this identity and are explicitly treated
            # as legacy navrl_quad by the v2 evaluator; present fields must match exactly.
            for key, current, name in (
                ("cfg_robot_name", self._robot_provenance["robot_name"], "robot name"),
                (
                    "cfg_robot_config_sha256",
                    self._robot_provenance["robot_config_sha256"],
                    "robot config SHA-256",
                ),
                (
                    "cfg_robot_asset_sha256",
                    self._robot_provenance["robot_asset_sha256"],
                    "robot URDF SHA-256",
                ),
            ):
                saved_value = state.get(key)
                if saved_value is None:
                    continue
                if str(saved_value).strip() != str(current).strip():
                    raise RuntimeError(
                        "NavRL ROBOT LINEAGE MISMATCH | %s: checkpoint=%s running=%s. "
                        "The policy shape is compatible but the rigid-body dynamics are not."
                        % (name, saved_value, current)
                    )
            saved_selector = str(
                state.get("cfg_obstacle_selector", "greedy_suppress")
            ).strip()
            current_selector = str(representation["selector"]).strip()
            if saved_selector != current_selector:
                density_evidence_changed = True
                logger.warning(
                    "NavRL OBSTACLE SELECTOR MISMATCH | checkpoint=%s running=%s. "
                    "This intentionally changes same-shape policy-input semantics."
                    % (saved_selector, current_selector)
                )
            for key, current, name in (
                (
                    "cfg_detector_checkpoint_name",
                    self._detector_checkpoint_name,
                    "NAVRL_DETECTOR_CHECKPOINT basename",
                ),
                (
                    "cfg_detector_checkpoint_sha256",
                    self._detector_checkpoint_sha256,
                    "NAVRL_DETECTOR_CHECKPOINT SHA-256",
                ),
            ):
                saved_str = state.get(key)
                if saved_str is not None and str(saved_str).strip() != str(current).strip():
                    density_evidence_changed = True
                    logger.warning(
                        "NavRL PERCEPTION CHECKPOINT MISMATCH | %s: checkpoint=%s running=%s. "
                        "Density evidence will be reset."
                        % (name, saved_str or "none", current or "none")
                    )
            for key, current, name in (
                ("cfg_bar_pool", arena["cfg_bar_pool"], "NAVRL_BAR_POOL"),
                ("cfg_placement_mode", arena["cfg_placement_mode"], "NAVRL_PLACEMENT_MODE"),
            ):
                saved_str = state.get(key)
                if saved_str is None:
                    continue  # checkpoint predates arena provenance
                if str(saved_str).strip() != str(current).strip():
                    density_evidence_changed = True
                    logger.warning(
                        "NavRL ARENA MISMATCH | %s: checkpoint=%s running=%s. The observation "
                        "width is unchanged, so this loads cleanly while measuring a DIFFERENT "
                        "task -- verify this is intentional."
                        % (name, saved_str, current)
                    )
            saved_schedule = state.get("cfg_density_threshold_schedule")
            current_schedule = str(
                getattr(self.density, "success_threshold_schedule", "") or ""
            )
            if (
                saved_schedule is not None
                and str(saved_schedule).strip() != current_schedule.strip()
            ):
                density_evidence_changed = True
                logger.warning(
                    "NavRL CURRICULUM CONFIG MISMATCH | NAVRL_DENSITY_THRESHOLD_SCHEDULE: "
                    "checkpoint used %r, running with %r. The promotion schedule is changing "
                    "intentionally or the runs are not directly comparable."
                    % (str(saved_schedule).strip(), current_schedule.strip())
                )
            saved_action_policy = state.get("cfg_action_policy")
            current_action_policy = os.environ.get("NAVRL_ACTION_POLICY", "legacy")
            if (
                saved_action_policy is not None
                and str(saved_action_policy).strip() != str(current_action_policy).strip()
            ):
                density_evidence_changed = True
                logger.warning(
                    "NavRL ACTION POLICY MISMATCH | checkpoint=%s running=%s. "
                    "The state_dict is shape-compatible but the action likelihood is not."
                    % (saved_action_policy, current_action_policy)
                )
            saved_training_seed = state.get("cfg_training_seed")
            if (
                saved_training_seed is not None
                and int(saved_training_seed) != int(self.task_config.seed)
            ):
                density_evidence_changed = True
                logger.warning(
                    "NavRL TRAINING SEED MISMATCH | checkpoint=%s running=%s. "
                    "Density evidence will be reset."
                    % (saved_training_seed, self.task_config.seed)
                )
            saved_action_std = str(state.get("cfg_action_std", "")).strip()
            current_action_std = os.environ.get("NAVRL_ACTION_STD", "").strip()
            if (
                saved_action_std
                and current_action_std
                and saved_action_std != current_action_std
            ):
                density_evidence_changed = True
                logger.warning(
                    "NavRL ACTION STD MISMATCH | checkpoint=%s running=%s."
                    % (saved_action_std, current_action_std)
                )
            saved_mu_scale = str(state.get("cfg_action_mu_scale", "")).strip()
            current_mu_scale = os.environ.get("NAVRL_ACTION_MU_SCALE", "1").strip()
            if saved_mu_scale and saved_mu_scale != current_mu_scale:
                density_evidence_changed = True
                logger.warning(
                    "NavRL ACTION MU-SCALE MISMATCH | checkpoint=%s running=%s."
                    % (saved_mu_scale, current_mu_scale)
                )
            moving_target_requested = (
                float(self.tm.speed_final) > 0.0
                or float(self.tm.speed_fixed) > 0.0
                or (
                    self._runtime_target_speed is not None
                    and float(self._runtime_target_speed) > 0.0
                )
            )
            saved_motion_model = str(
                state.get("cfg_target_motion_model", "")
            ).strip()
            if moving_target_requested and saved_motion_model != self._target_motion_model:
                density_evidence_changed = True
                logger.warning(
                    "NavRL TARGET MOTION MISMATCH | checkpoint=%s running=%s. "
                    "Older checkpoints saw targets stall at bars; moving-target metrics and "
                    "fine-tuning are a changed environment contract."
                    % (saved_motion_model or "legacy_bar_push", self._target_motion_model)
                )
            # Same contract, one field over: a checkpoint either attests the heading-validity
            # threshold or is refused.  Absent (every pre-key checkpoint) is assumed, never
            # silently promoted to an attestation -- the provenance string carries which it was.
            (
                self._heading_valid_speed_mps,
                self._heading_valid_speed_provenance,
                heading_contract_message,
            ) = resolve_heading_valid_speed_contract(state)
            logger.warning(heading_contract_message)
            saved_route_mode = str(state.get("cfg_target_route_mode", "")).strip().lower()
            if self._target_route_enabled and saved_route_mode != self._target_route_mode:
                raise RuntimeError(
                    "fresh-only routed target checkpoint contract: checkpoint route=%s running=%s"
                    % (saved_route_mode or "missing/off", self._target_route_mode)
                )
            if saved_route_mode not in ("", TARGET_ROUTE_MODE_OFF, self._target_route_mode):
                raise RuntimeError(
                    "target route checkpoint cannot load with route mode %s" % self._target_route_mode
                )
            if self._target_route_enabled:
                saved_support = state.get("cfg_target_route_support_xy_m")
                current_support = self._target_route_support_xy[0].detach().cpu().tolist()
                if (
                    not isinstance(saved_support, (list, tuple))
                    or len(saved_support) != 2
                    or any(
                        abs(float(saved) - float(current)) > 1e-6
                        for saved, current in zip(saved_support, current_support)
                    )
                ):
                    raise RuntimeError(
                        "fresh-only routed target support contract mismatch or missing provenance"
                    )
                if not self._target_route_recovery_enabled:
                    recovery_contract = ()
                else:
                    recovery_contract = (
                    ("cfg_target_route_recovery_schema", TARGET_ROUTE_RECOVERY_SCHEMA),
                    ("cfg_target_route_recovery_model", TARGET_ROUTE_RECOVERY_MODEL),
                    ("cfg_target_route_recovery_hard_envelope", "closed_aabb_support_v1"),
                    (
                        "cfg_target_route_recovery_soft_envelope",
                        "closed_aabb_support_plus_tracking_v1",
                    ),
                    ("cfg_target_route_recovery_hard_epsilon_m", TARGET_ROUTE_HARD_EPSILON_M),
                    (
                        "cfg_target_route_recovery_reachable_tube_margin_m",
                        TARGET_ROUTE_REACHABLE_TUBE_MARGIN_M,
                    ),
                    ("cfg_target_route_recovery_hysteresis_m", RECOVERY_HYSTERESIS_M),
                    ("cfg_target_route_recovery_stop_speed_mps", RECOVERY_STOP_SPEED_MPS),
                    (
                        "cfg_target_route_recovery_progress_tolerance_m",
                        RECOVERY_CONNECT_PROGRESS_TOLERANCE_M,
                    ),
                    ("cfg_target_route_recovery_anchor_radius_cells", 3),
                    (
                        "cfg_target_recovery_brake_decel_p05_mps2",
                        float(getattr(self.tm, "recovery_brake_decel_p05", 0.0)),
                    ),
                    (
                        "cfg_target_recovery_stop_time_p95_s",
                        float(getattr(self.tm, "recovery_brake_stop_time_p95", 0.0)),
                    ),
                    (
                        "cfg_target_recovery_probe_receipt_sha256",
                        self._recovery_probe_receipt_sha256,
                    ),
                    (
                        "cfg_target_recovery_braking_contract_variant",
                        str(getattr(self.tm, "recovery_braking_contract_variant", "canonical_1p5")),
                    ),
                    (
                        "cfg_target_recovery_brake_speed_samples_mps",
                        list(self._recovery_brake_speed_samples_mps),
                    ),
                    (
                        "cfg_target_recovery_brake_stop_distance_samples_m",
                        list(self._recovery_brake_stop_distance_samples_m),
                    ),
                    (
                        "cfg_target_recovery_brake_lateral_tube_p95_m",
                        self._recovery_brake_lateral_tube_p95_m,
                    ),
                    (
                        "cfg_target_max_accel_mps2",
                        float(self.tm.max_accel),
                    ),
                    (
                        "cfg_target_max_turn_rate_degps",
                        float(self.tm.max_turn_rate_deg),
                    ),
                    (
                        "cfg_target_lookahead_s",
                        float(self.tm.avoidance_lookahead_s),
                    ),
                    ("cfg_target_physical_tracking_margin_m", float(self.tm.physical_tracking_margin)),
                    ("cfg_target_physical_boundary_margin_m", float(self.tm.physical_boundary_margin)),
                    ("cfg_target_physical_mass_kg", float(self.tm.physical_mass)),
                    (
                        "cfg_target_physical_box_xyz_m",
                        [float(value) for value in self.tm.physical_box_xyz],
                    ),
                    ("cfg_target_route_wall_margin_m", float(self.cur.wall_margin)),
                    ("cfg_physics_dt_s", float(self._runtime_physics_contract()["physics_dt_s"])),
                    ("cfg_physics_substeps", int(self._runtime_physics_contract()["physics_substeps"])),
                    (
                        "cfg_physics_steps_per_rl_step",
                        int(self._runtime_physics_contract()["physics_steps_per_rl_step"]),
                    ),
                    ("cfg_rl_step_dt_s", float(self._runtime_physics_contract()["rl_step_dt_s"])),
                    (
                        "cfg_training_source_manifest",
                        self._training_source_provenance["manifest"],
                    ),
                    (
                        "cfg_training_source_manifest_sha256",
                        self._training_source_provenance["manifest_sha256"],
                    ),
                    (
                        "cfg_training_source_git_commit",
                        self._training_source_provenance["git_commit"],
                    ),
                    (
                        "cfg_training_source_git_dirty",
                        self._training_source_provenance["git_dirty"],
                    ),
                )
                for key, expected in recovery_contract:
                    saved = state.get(key)
                    if saved is None or (
                        isinstance(expected, float)
                        and abs(float(saved) - expected) > 1e-9
                    ) or (
                        not isinstance(expected, float) and saved != expected
                    ):
                        raise RuntimeError(
                            "fresh-only routed recovery contract mismatch or missing provenance: %s"
                            % key
                        )
                if self._target_route_braking_v3_enabled:
                    v3_contract = (
                        ("cfg_target_route_braking_v3_schema", TARGET_ROUTE_BRAKING_V3_SCHEMA),
                        ("cfg_target_route_braking_v3_model", TARGET_ROUTE_BRAKING_V3_MODEL),
                        (
                            "cfg_target_route_braking_v3_soft_envelope",
                            TARGET_ROUTE_BRAKING_GEOMETRY_SCHEMA,
                        ),
                        (
                            "cfg_target_recovery_probe_receipt_sha256",
                            self._recovery_probe_receipt_sha256,
                        ),
                        (
                            "cfg_target_recovery_brake_speed_samples_mps",
                            list(self._recovery_brake_speed_samples_mps),
                        ),
                        (
                            "cfg_target_recovery_brake_stop_distance_samples_m",
                            list(self._recovery_brake_stop_distance_samples_m),
                        ),
                        (
                            "cfg_target_recovery_brake_lateral_tube_p95_m",
                            self._recovery_brake_lateral_tube_p95_m,
                        ),
                        ("cfg_target_physical_tracking_margin_m", float(self.tm.physical_tracking_margin)),
                        ("cfg_target_physical_boundary_margin_m", float(self.tm.physical_boundary_margin)),
                        ("cfg_target_route_wall_margin_m", float(self.cur.wall_margin)),
                    )
                    for key, expected in v3_contract:
                        saved = state.get(key)
                        if saved is None or (
                            isinstance(expected, float)
                            and abs(float(saved) - expected) > 1e-9
                        ) or (
                            not isinstance(expected, float) and saved != expected
                        ):
                            raise RuntimeError(
                                "fresh-only braking-v3 contract mismatch or missing provenance: %s"
                                % key
                            )
            # Newer checkpoints record the complete moving-target/spawn/safety geometry. Missing
            # fields are tolerated for old checkpoints, but a present mismatch changes the task
            # distribution and invalidates accumulated density evidence.
            for key, current, name in (
                (
                    "cfg_target_max_accel_mps2",
                    float(self.tm.max_accel),
                    "NAVRL_TARGET_MAX_ACCEL",
                ),
                (
                    "cfg_target_max_turn_rate_degps",
                    float(self.tm.max_turn_rate_deg),
                    "NAVRL_TARGET_MAX_TURN_RATE_DEG",
                ),
                (
                    "cfg_target_lookahead_s",
                    float(self.tm.avoidance_lookahead_s),
                    "NAVRL_TARGET_LOOKAHEAD_S",
                ),
                (
                    "cfg_target_obstacle_clearance_m",
                    float(self.tm.obstacle_clearance),
                    "NAVRL_TARGET_OBSTACLE_CLEARANCE",
                ),
                ("cfg_target_pattern", str(self.tm.pattern), "NAVRL_TARGET_PATTERN"),
                (
                    "cfg_target_speed_min",
                    float(getattr(self.tm, "speed_min", 0.0)),
                    "NAVRL_TARGET_SPEED_MIN",
                ),
                (
                    "cfg_target_speed_final",
                    float(self.tm.speed_final),
                    "NAVRL_TARGET_SPEED_FINAL",
                ),
                (
                    "cfg_target_speed_fixed",
                    float(self.tm.speed_fixed),
                    "NAVRL_TARGET_SPEED",
                ),
                (
                    "cfg_target_speed_ramp_epochs",
                    float(self.tm.speed_ramp_epochs),
                    "NAVRL_TARGET_SPEED_RAMP_EPOCHS",
                ),
                (
                    "cfg_target_physical_mass_kg",
                    float(self.tm.physical_mass),
                    "NAVRL_TARGET_MASS_KG",
                ),
                (
                    "cfg_target_physical_max_motor_thrust_n",
                    float(self.tm.physical_max_motor_thrust),
                    "NAVRL_TARGET_MAX_MOTOR_THRUST_N",
                ),
                (
                    "cfg_target_physical_motor_tau_s",
                    float(self.tm.physical_motor_tau),
                    "NAVRL_TARGET_MOTOR_TAU_S",
                ),
                (
                    "cfg_target_physical_motor_arm_xy_m",
                    float(self.tm.physical_motor_arm_xy),
                    "NAVRL_TARGET_MOTOR_ARM_XY_M",
                ),
                (
                    "cfg_target_physical_yaw_torque_ratio_m",
                    float(self.tm.physical_yaw_torque_ratio),
                    "NAVRL_TARGET_YAW_TORQUE_RATIO_M",
                ),
                (
                    "cfg_target_physical_max_tilt_deg",
                    float(self.tm.physical_max_tilt_deg),
                    "NAVRL_TARGET_MAX_TILT_DEG",
                ),
                (
                    "cfg_target_physical_tracking_margin_m",
                    float(self.tm.physical_tracking_margin),
                    "NAVRL_TARGET_TRACKING_MARGIN_M",
                ),
                (
                    "cfg_target_physical_boundary_margin_m",
                    float(self.tm.physical_boundary_margin),
                    "NAVRL_TARGET_BOUNDARY_MARGIN_M",
                ),
                (
                    "cfg_target_recovery_braking_contract_variant",
                    str(getattr(self.tm, "recovery_braking_contract_variant", "canonical_1p5")),
                    "NAVRL_TARGET_BRAKING_CONTRACT_VARIANT",
                ),
                (
                    "cfg_target_route_mode",
                    self._target_route_mode,
                    "NAVRL_TARGET_ROUTE_MODE",
                ),
                (
                    "cfg_target_route_resolution_m",
                    float(self.tm.route_resolution_m),
                    "NAVRL_TARGET_ROUTE_RESOLUTION_M",
                ),
                (
                    "cfg_target_route_max_expansions",
                    float(self.tm.route_max_expansions),
                    "NAVRL_TARGET_ROUTE_MAX_EXPANSIONS",
                ),
                (
                    "cfg_target_route_max_waypoints",
                    float(self.tm.route_max_waypoints),
                    "NAVRL_TARGET_ROUTE_MAX_WAYPOINTS",
                ),
                (
                    "cfg_target_route_replan_cooldown_steps",
                    float(self.tm.route_replan_cooldown_steps),
                    "NAVRL_TARGET_ROUTE_REPLAN_COOLDOWN_STEPS",
                ),
                (
                    "cfg_target_route_goal_tolerance_m",
                    float(self.tm.route_goal_tolerance_m),
                    "NAVRL_TARGET_ROUTE_GOAL_TOLERANCE_M",
                ),
                (
                    "cfg_target_route_min_goal_distance_m",
                    float(self.tm.route_min_goal_distance_m),
                    "NAVRL_TARGET_ROUTE_MIN_GOAL_DISTANCE_M",
                ),
                (
                    "cfg_target_route_goal_exclusion_radius_m",
                    float(self.tm.route_goal_exclusion_radius_m),
                    "NAVRL_TARGET_ROUTE_GOAL_EXCLUSION_M",
                ),
                ("cfg_general_train", bool(self.general_train_mode), "NAVRL_GENERAL_TRAIN"),
                (
                    "cfg_perception_perturb",
                    bool(getattr(self.perception_cfg, "enable_perturbations", False)),
                    "NAVRL_PERCEPTION_PERTURB",
                ),
                (
                    "cfg_p9_empirical_error_enabled",
                    bool(getattr(getattr(self, "perception", None), "empirical_error", None) is not None),
                    "NAVRL_P9_ERROR_MODEL",
                ),
                (
                    "cfg_tilt_comp",
                    os.environ.get("NAVRL_TILT_COMP", "1").strip().lower()
                    not in ("0", "false", "no", "off"),
                    "NAVRL_TILT_COMP",
                ),
                ("cfg_oob_margin", float(self.vis_cfg.oob_margin), "NAVRL_OOB_MARGIN"),
                (
                    "cfg_alt_hold_vmax",
                    float(
                        getattr(
                            self.task_config,
                            "alt_hold_vmax",
                            self.task_config.max_velocity,
                        )
                    ),
                    "NAVRL_ALT_HOLD_VMAX",
                ),
            ):
                saved_value = state.get(key)
                if saved_value is None:
                    continue
                if isinstance(current, str):
                    matches = str(saved_value).strip() == current.strip()
                elif isinstance(current, bool):
                    matches = bool(saved_value) is current
                else:
                    matches = abs(float(saved_value) - float(current)) <= 1e-6
                if not matches:
                    density_evidence_changed = True
                    logger.warning(
                        "NavRL TASK CONTRACT MISMATCH | %s: checkpoint=%s running=%s. "
                        "Density evidence will be reset."
                        % (name, saved_value, current)
                    )
            # Loud config-drift guard (warn, never override: an eval may deliberately change these).
            # A mismatch here silently invalidates the run -- see get_env_state() for why.
            for key, current, name in (
                ("cfg_lidar_max_range", float(self.task_config.lidar_max_range), "NAVRL_LIDAR_RANGE"),
                ("cfg_max_velocity", float(self.task_config.max_velocity), "NAVRL_MAX_VELOCITY"),
                ("cfg_yaw_rate_max", float(self.task_config.yaw_rate_max), "NAVRL_YAW_RATE_MAX"),
                (
                    "cfg_max_tilt_deg",
                    float(os.environ.get("NAVRL_MAX_TILT_DEG", "").strip() or 45.0),
                    "NAVRL_MAX_TILT_DEG",
                ),
                (
                    "cfg_max_obstacles",
                    float(representation["max_obstacles"]),
                    "MAX_OBSTACLES (navrl_perception.py)",
                ),
                (
                    "cfg_token_fov_deg",
                    float(representation["token_fov_deg"]),
                    "NAVRL_OBSTACLE_FOV_DEG",
                ),
                (
                    "cfg_obstacle_suppress_deg",
                    float(representation["suppress_deg"]),
                    "NAVRL_OBSTACLE_SUPPRESS_DEG",
                ),
                (
                    "cfg_obstacle_cluster_gap_m",
                    float(representation["cluster_gap_m"]),
                    "NAVRL_OBSTACLE_CLUSTER_GAP_M",
                ),
                (
                    "cfg_obstacle_sectors",
                    float(representation["sectors"]),
                    "NAVRL_OBSTACLE_SECTORS",
                ),
                (
                    "cfg_obstacle_ttc_idle_s",
                    float(representation["ttc_idle_s"]),
                    "NAVRL_OBSTACLE_TTC_IDLE_S",
                ),
                (
                    "cfg_obstacle_ttc_min_speed",
                    float(representation["ttc_min_speed"]),
                    "NAVRL_OBSTACLE_TTC_MIN_SPEED",
                ),
                (
                    "cfg_lidar_hbeams",
                    float(representation["hbeams"]),
                    "NAVRL_LIDAR_HBEAMS",
                ),
                (
                    "cfg_lidar_vbeams",
                    float(representation["vbeams"]),
                    "NAVRL_LIDAR_VBEAMS",
                ),
                (
                    "cfg_corridor_tokens",
                    float(representation["corridor_tokens"]),
                    "NAVRL_CORRIDOR_TOKENS",
                ),
                (
                    "cfg_corridor_horizon_m",
                    float(representation["corridor_horizon_m"]),
                    "NAVRL_CORRIDOR_HORIZON_M",
                ),
                (
                    "cfg_corridor_min_width_m",
                    float(representation["corridor_min_width_m"]),
                    "NAVRL_CORRIDOR_MIN_WIDTH_M",
                ),
                (
                    "cfg_geofence_actor",
                    float(representation["geofence_actor"]),
                    "NAVRL_GEOFENCE_ACTOR",
                ),
                (
                    "cfg_geofence_noise_std_m",
                    float(representation["geofence_noise_std_m"]),
                    "NAVRL_GEOFENCE_NOISE_STD_M",
                ),
                (
                    "cfg_geofence_dropout",
                    float(representation["geofence_dropout"]),
                    "NAVRL_GEOFENCE_DROPOUT",
                ),
                (
                    "cfg_fov_curriculum_epochs",
                    float(getattr(self.vis_cfg, "fov_curriculum_epochs", 0)),
                    "NAVRL_FOV_CURRICULUM_EPOCHS",
                ),
                (
                    "cfg_detector_min_pixels",
                    float(getattr(self.perception_cfg, "min_target_pixels", 0)),
                    "NAVRL_DETECTOR_MIN_PIXELS",
                ),
                (
                    "cfg_detector_max_range",
                    float(getattr(self.vis_cfg, "detector_max_range", 0.0)),
                    "NAVRL_DETECTOR_MAX_RANGE",
                ),
                (
                    "cfg_detect_width",
                    float(getattr(self.vis_cfg, "detect_width", 0)),
                    "NAVRL_DETECT_WIDTH",
                ),
                (
                    "cfg_detect_height",
                    float(getattr(self.vis_cfg, "detect_height", 0)),
                    "NAVRL_DETECT_HEIGHT",
                ),
                (
                    "cfg_detector_threshold",
                    float(getattr(self.perception_cfg, "pixel_threshold", 0.0)),
                    "NAVRL_DETECTOR_THRESHOLD",
                ),
                (
                    "cfg_detection_dropout",
                    float(getattr(self.perception_cfg, "detection_dropout_prob", 0.0)),
                    "NAVRL_DETECTION_DROPOUT",
                ),
                (
                    "cfg_detection_latency_s",
                    float(getattr(self.perception_cfg, "detection_latency_s", 0.0)),
                    "NAVRL_DETECTION_LATENCY_S",
                ),
                (
                    "cfg_range_error_m",
                    float(getattr(self.perception_cfg, "range_error_m", 0.0)),
                    "NAVRL_RANGE_ERROR_M",
                ),
                (
                    "cfg_rgb_noise_std",
                    float(getattr(self.perception_cfg, "rgb_noise_std", 0.0)),
                    "NAVRL_RGB_NOISE_STD",
                ),
                (
                    "cfg_depth_noise_std",
                    float(getattr(self.perception_cfg, "depth_noise_std", 0.0)),
                    "NAVRL_DEPTH_NOISE_STD",
                ),
                # The enabled flag lives in the boolean contract above and the model SHA is an
                # identity, not a magnitude; neither belongs in a loop that computes
                # abs(float(saved) - current). Only the seed is numeric.
                (
                    "cfg_p9_empirical_error_seed",
                    float(getattr(self.perception_cfg, "empirical_error_seed", 1701)),
                    "NAVRL_P9_SEED",
                ),
                (
                    "cfg_appearance_hue_deg",
                    float(getattr(self.vis_cfg, "appearance_hue_deg", 0.0)),
                    "NAVRL_APP_HUE_DEG",
                ),
                (
                    "cfg_appearance_light_gain",
                    float(getattr(self.vis_cfg, "appearance_light_gain", 0.0)),
                    "NAVRL_APP_LIGHT_GAIN",
                ),
                (
                    "cfg_appearance_albedo_jitter",
                    float(getattr(self.vis_cfg, "appearance_albedo_jitter", 0.0)),
                    "NAVRL_APP_ALBEDO_JITTER",
                ),
                (
                    "cfg_appearance_texture_std",
                    float(getattr(self.vis_cfg, "appearance_texture_std", 0.0)),
                    "NAVRL_APP_TEXTURE_STD",
                ),
                (
                    "cfg_appearance_motion_blur",
                    float(getattr(self.vis_cfg, "appearance_motion_blur", 0.0)),
                    "NAVRL_APP_MOTION_BLUR",
                ),
                (
                    "cfg_camera_mount_rot_deg",
                    float(getattr(self.vis_cfg, "camera_mount_rot_deg", 0.0)),
                    "NAVRL_CAM_MOUNT_ROT_DEG",
                ),
                (
                    "cfg_camera_mount_trans_m",
                    float(getattr(self.vis_cfg, "camera_mount_trans_m", 0.0)),
                    "NAVRL_CAM_MOUNT_TRANS_M",
                ),
                (
                    "cfg_camera_fov_scale_err",
                    float(getattr(self.vis_cfg, "camera_fov_scale_err", 0.0)),
                    "NAVRL_CAM_FOV_SCALE_ERR",
                ),
                # Arena / task version. A mismatch here is not an "input scaling" bug but a
                # DIFFERENT TASK (v1 24 m pursuit vs v2 40 m search) that loads without error
                # because the observation width is identical.
                ("cfg_arena_xy", arena["cfg_arena_xy"], "NAVRL_ARENA_XY"),
                ("cfg_arena_z", arena["cfg_arena_z"], "NAVRL_ARENA_Z"),
                ("cfg_placement_gap_m", arena["cfg_placement_gap_m"], "NAVRL_PLACEMENT_GAP_M"),
                (
                    "cfg_placement_touch_m",
                    arena["cfg_placement_touch_m"],
                    "NAVRL_PLACEMENT_TOUCH_M",
                ),
                (
                    "cfg_placement_surface_clearance_m",
                    arena["cfg_placement_surface_clearance_m"],
                    "NAVRL_PLACEMENT_SURFACE_CLEARANCE_M",
                ),
                (
                    "cfg_episode_len_steps",
                    arena["cfg_episode_len_steps"],
                    "NAVRL_EPISODE_LEN_STEPS",
                ),
                ("cfg_bar_x_min", arena["cfg_bar_x_min"], "NAVRL_BAR_X_MIN"),
                ("cfg_bar_x_max", arena["cfg_bar_x_max"], "NAVRL_BAR_X_MAX"),
            ):
                saved = state.get(key)
                if saved is None:
                    continue  # checkpoint predates this guard
                if abs(float(saved) - current) > 1e-6:
                    density_evidence_changed = True
                    logger.warning(
                        "NavRL CONFIG MISMATCH | %s: checkpoint trained with %.3f, running with %.3f. "
                        "This changes policy inputs -- results are NOT comparable unless intentional."
                        % (name, float(saved), current)
                    )
            # The search-state contract is a MODE and a FLAG, not a magnitude. Both used to sit in
            # the float loop above, where `float("off")` raises and kills the restore -- so every
            # checkpoint trained after cfg_search_state was introduced was unloadable, while older
            # ones sailed past on the `saved is None` branch. Compare them by value instead.
            for key, current, name in (
                (
                    "cfg_search_state",
                    str(representation["search_state"]),
                    "NAVRL_SEARCH_STATE",
                ),
                (
                    "cfg_search_state_force_invalid",
                    bool(representation["search_state_force_invalid"]),
                    "NAVRL_SEARCH_STATE_FORCE_INVALID",
                ),
            ):
                saved = state.get(key)
                if saved is None:
                    continue  # checkpoint predates this guard
                if str(saved).strip() != str(current).strip():
                    density_evidence_changed = True
                    logger.warning(
                        "NavRL CONFIG MISMATCH | %s: checkpoint trained with %s, running with %s. "
                        "This changes policy inputs -- results are NOT comparable unless intentional."
                        % (name, saved, current)
                    )
            for key, current, name in (
                (
                    "cfg_general_goal_dist_min",
                    float(self.general_goal_dist_min),
                    "NAVRL_GENERAL_GOAL_DIST_MIN",
                ),
                (
                    "cfg_general_goal_dist_max",
                    float(self.general_goal_dist_max),
                    "NAVRL_GENERAL_GOAL_DIST_MAX",
                ),
            ):
                saved = state.get(key)
                if saved is not None and abs(float(saved) - current) > 1e-6:
                    density_evidence_changed = True
                    logger.warning(
                        "NavRL TASK-DISTRIBUTION MISMATCH | %s: checkpoint used %.3f, "
                        "running with %.3f. The density competence window will be reset."
                        % (name, float(saved), current)
                    )
            if bool(getattr(self.density, "use_density_curriculum", False)):
                for key, current, name in (
                    (
                        "cfg_density_final",
                        float(getattr(self.density, "n_final", self.n_bars_active)),
                        "NAVRL_DENSITY_FINAL",
                    ),
                    (
                        "cfg_density_step",
                        float(getattr(self.density, "promote_step", 0)),
                        "NAVRL_DENSITY_STEP",
                    ),
                    (
                        "cfg_density_threshold",
                        float(getattr(self.density, "success_threshold", 0.0)),
                        "NAVRL_DENSITY_THRESHOLD",
                    ),
                    (
                        "cfg_density_threshold_start",
                        float(getattr(self.density, "success_threshold_start", 0.0)),
                        "NAVRL_DENSITY_THRESHOLD_START",
                    ),
                    (
                        "cfg_density_threshold_end",
                        float(getattr(self.density, "success_threshold_end", 0.0)),
                        "NAVRL_DENSITY_THRESHOLD_END",
                    ),
                    (
                        "cfg_density_check_eps",
                        float(getattr(self.density, "check_after_episodes", 0)),
                        "NAVRL_DENSITY_CHECK_EPS",
                    ),
                    (
                        "cfg_density_min_epochs",
                        float(getattr(self.density, "min_epochs_per_density", 0)),
                        "NAVRL_DENSITY_MIN_EPOCHS",
                    ),
                ):
                    saved = state.get(key)
                    if saved is not None and abs(float(saved) - current) > 1e-6:
                        density_evidence_changed = True
                        logger.warning(
                            "NavRL CURRICULUM CONFIG MISMATCH | %s: checkpoint used %.3f, "
                            "running with %.3f. The promotion schedule is changing intentionally "
                            "or the runs are not directly comparable."
                            % (name, float(saved), current)
                        )
        if isinstance(state, dict) and state.get("num_task_steps") is not None:
            self.num_task_steps = int(state["num_task_steps"])
        if isinstance(state, dict) and state.get("k_max_cur") is not None:
            # restore the competence-gated distance window across a --checkpoint resume
            self._k_max_cur = float(state["k_max_cur"])
            self._k_min_cur = float(state.get("k_min_cur", self._k_min_cur))
        if isinstance(state, dict) and state.get("n_bars_active") is not None:
            # An explicit NAVRL_NUM_BARS wins over the checkpoint: density-sweep evals must run at
            # the REQUESTED density, not silently at whatever density the checkpoint trained on.
            if os.environ.get("NAVRL_NUM_BARS", "").strip():
                logger.warning(
                    "NAVRL_NUM_BARS set explicitly; ignoring checkpoint n_bars_active=%s (keeping %d)."
                    % (state.get("n_bars_active"), self.n_bars_active)
                )
            else:
                self._set_active_bars(state["n_bars_active"])
        if isinstance(state, dict):
            restored_fin = max(0, int(state.get("density_fin_agg", 0)))
            restored_succ = max(0, int(state.get("density_succ_agg", 0)))
            strata_keys = (
                "density_speed_succ",
                "density_speed_fin",
                "density_dist_succ",
                "density_dist_fin",
                "density_pattern_succ",
                "density_pattern_fin",
            )
            if restored_fin > 0 and any(state.get(key) is None for key in strata_keys):
                density_evidence_changed = True
                logger.warning(
                    "NavRL density evidence provenance missing | checkpoint predates stratified "
                    "counters; aggregate window cannot be completed consistently."
                )
            if density_evidence_changed:
                self._density_fin_agg = 0
                self._density_succ_agg = 0
                self._reset_density_strata()
                resume_warmup_epochs = max(
                    0, int(os.environ.get("NAVRL_DENSITY_RESUME_WARMUP", "250"))
                )
                horizon = max(1, int(getattr(self.cur, "ppo_horizon", 1)))
                self._density_gate_not_before_steps = (
                    int(self.num_task_steps) + resume_warmup_epochs * horizon
                )
                logger.warning(
                    "NavRL density evidence RESET | discarded=%d/%d eps "
                    "resume_warmup=%d epochs gate_not_before_step=%d"
                    % (
                        min(restored_succ, restored_fin),
                        restored_fin,
                        resume_warmup_epochs,
                        self._density_gate_not_before_steps,
                    )
                )
            else:
                self._density_fin_agg = restored_fin
                self._density_succ_agg = min(restored_succ, restored_fin)
                self._density_gate_not_before_steps = max(
                    0, int(state.get("density_gate_not_before_steps", 0))
                )
                self._restore_density_strata(state)
            # Dwell clock. Restore it so a resume does not reset the counter and re-serve a full
            # dwell period at a density the policy already matured on. A checkpoint that predates
            # the field falls back to "this density started now", which is the conservative choice.
            self._density_level_start_steps = restore_density_level_start_steps(
                state,
                self.num_task_steps,
            )
            logger.warning(
                "NavRL checkpoint state restored | task_steps=%d bars=%d->%d "
                "k_min=%.1f k_max=%.1f density_window=%d/%d eps"
                % (
                    int(self.num_task_steps),
                    bars_before_restore,
                    int(self.n_bars_active),
                    float(self._k_min_cur),
                    float(self._k_max_cur),
                    int(self._density_fin_agg),
                    int(getattr(self.density, "check_after_episodes", 0)),
                )
            )

    # ------------------------------------------------------------------ reset
    def reset(self):
        if self._obs_dump_enabled:
            # A full reset ends EVERY env's episode without an outcome row (rl_games'
            # BasePlayer.run calls env_reset at the top of each n_games iteration). Remember those
            # uids so the export can drop their frames with a counted reason instead of failing
            # the entire dump; envs that legitimately finished this same step are excluded at
            # flush time because they DO have an outcome row.
            self._note_obs_dump_full_reset()
        # Respawn the robots (and re-place obstacles) BEFORE sampling goals. Without this, a
        # full reset leaves the robots at their build pose (overlapping the bars near the env
        # origin), so the first step crashes every env at once — which ends rl_games play mode
        # after a single step. Mid-episode resets don't need it: the env manager has already
        # respawned those envs by the time reset_idx() is called from step().
        if self.general_eval_mode:
            self._sample_general_density()
        self.sim_env.reset()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._sync_target_to_sensor()
        # render once so the first observation carries a valid LiDAR scan
        self.sim_env.render(render_components="sensors")
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        if self._contact_geom_enabled:
            # A fresh episode has no usable pose history; the recorder drops contacts until the
            # buffer is deep enough rather than replaying another episode's poses.
            self._cg_hist_age[env_ids] = 0
            if len(env_ids) > 0:
                self._bank_contact_geometry_path(env_ids)
        if len(env_ids) == 0:
            return
        if self._obs_dump_enabled:
            # Global, monotonic, never wraps: episode_uid = env_id + num_envs * this counter.
            self._obs_dump_ep_idx[env_ids] += 1
            # Kept EXACT (same envs, same increment) so the flush reads a host array, not a GPU
            # tensor that may be gone by interpreter shutdown.
            self._obs_dump_ep_idx_host[env_ids.detach().cpu().numpy()] += 1
        if self.general_spawn_mode:
            self._randomize_general_drone_spawn(env_ids)
        # robot has already been respawned by the env manager when this is called mid-episode,
        # so robot_position holds the fresh start pose.
        start_pos = self.obs_dict["robot_position"][env_ids]
        b_min = self.obs_dict["env_bounds_min"][env_ids]
        b_max = self.obs_dict["env_bounds_max"][env_ids]

        n = len(env_ids)
        goal = start_pos.clone()
        goal[:, 2] = self.task_config.flight_altitude
        m = float(self.cur.wall_margin)
        clearance = float(getattr(self.task_config, "goal_min_bar_clearance", 0.0))
        # Only the first n_bars_active assets exist in the current density contract. Treating
        # every build-time slot as physical here makes a 25-bar episode avoid the parked/inactive
        # slots from the 150-bar capacity and silently changes the requested target dynamics.
        bars_xy = self.obs_dict["obstacle_position"][
            env_ids, self._bar_offset : self._bar_offset + self.n_bars_active, 0:2
        ]
        bar_half_extents = self.obs_dict["asset_collision_half_extents"][
            env_ids, self._bar_offset : self._bar_offset + self.n_bars_active, 0:2
        ]
        # "Cross the bar field": the drone spawns at x~0, so placing the goal at x=k on the far
        # side forces a left->right traversal of the bars. k ~ U[k_min, k_max(epoch)] (k_max
        # grows with training via _goal_x_max), y is free across the arena minus a wall margin.
        # Resample any goal within `clearance` of a bar so the 0.5 m capture sphere is flyable.
        if self.general_spawn_mode:
            goal = self._sample_general_target(env_ids, start_pos, b_min, b_max, bars_xy, bar_half_extents)
            sampled_dist = torch.norm(goal[:, 0:2] - start_pos[:, 0:2], dim=1)
            k_min = float(sampled_dist.min().item())
            k_max = float(sampled_dist.max().item())
        else:
            k_max = self._goal_x_max()
            k_min = self._goal_x_min()
        self.cur_goal_dist_max = k_max  # surfaced to the dashboard as "curriculum max"
        self.cur_goal_dist_min = k_min  # surfaced to the dashboard as "curriculum min"
        todo = torch.zeros(n, dtype=torch.bool, device=self.device) if self.general_spawn_mode else torch.ones(
            n, dtype=torch.bool, device=self.device
        )
        for _ in range(10):
            if not todo.any():
                break
            j = int(todo.sum())
            gx = k_min + (k_max - k_min) * torch.rand(j, device=self.device)
            # Density is a different source of difficulty from distance. During the density stage,
            # retain some short/medium episodes instead of training exclusively on the final far
            # window. This is enabled only by the staged launch recipe; default behavior is intact.
            mix_prob = float(getattr(self.density, "easy_goal_mix_prob", 0.0))
            horizon = max(1, int(getattr(self.cur, "ppo_horizon", 1)))
            density_start = int(getattr(self.density, "warmup_epochs", 0)) * horizon
            if mix_prob > 0.0 and self.num_task_steps >= density_start:
                easy = torch.rand(j, device=self.device) < min(1.0, max(0.0, mix_prob))
                easy_lo = float(getattr(self.density, "easy_goal_min", self.cur.k_min))
                easy_hi = min(
                    float(getattr(self.density, "easy_goal_max", k_max)), float(k_max)
                )
                if easy_hi > easy_lo and bool(easy.any()):
                    gx[easy] = easy_lo + (easy_hi - easy_lo) * torch.rand(
                        int(easy.sum()), device=self.device
                    )
            gx = gx.clamp(max=(b_max[todo, 0] - m))  # keep the capture sphere off the far wall
            gy = (b_min[todo, 1] + m) + (
                b_max[todo, 1] - b_min[todo, 1] - 2.0 * m
            ) * torch.rand(j, device=self.device)
            if self.vision_mode and not self._fov_curriculum_is_saturated():
                # Cold-start visibility: keep the goal inside the camera FOV early so the detector
                # acquires the target (the KF activates -> the actor finally gets a bearing to act
                # on), then widen the allowed bearing to the full arena. The +/- spawn-yaw headroom
                # keeps the target visible regardless of the spawn heading. Without this a sensor-
                # only from-scratch policy is goal-blind and never leaves the ~100% crash basin.
                horizon_e = max(1, int(getattr(self.cur, "ppo_horizon", 1)))
                frac = min(1.0, (self.num_task_steps / horizon_e) / float(self.vis_cfg.fov_curriculum_epochs))
                half_fov = math.radians(float(self.vis_cfg.detector_hfov_deg) * 0.5)
                yaw_head = math.radians(float(getattr(self.vis_cfg, "spawn_yaw_max_deg", 30.0)))
                bearing0 = max(math.radians(8.0), 0.85 * (half_fov - yaw_head))
                bearing_lim = bearing0 + frac * (0.5 * math.pi - bearing0)
                dy_max = (gx - start_pos[todo, 0]).clamp(min=0.5) * math.tan(bearing_lim)
                sy = start_pos[todo, 1]
                lo = torch.maximum(b_min[todo, 1] + m, sy - dy_max)
                hi = torch.maximum(torch.minimum(b_max[todo, 1] - m, sy + dy_max), lo)
                gy = lo + (hi - lo) * torch.rand(j, device=self.device)
            goal[todo, 0] = gx
            goal[todo, 1] = gy
            if clearance <= 0.0 or bars_xy.shape[1] == 0:
                break
            d_bar = (
                torch.cdist(goal[todo, 0:2].unsqueeze(1), bars_xy[todo])
                .squeeze(1)
                .min(dim=1)
                .values
            )
            still_bad = d_bar < clearance
            idx = todo.nonzero(as_tuple=False).squeeze(-1)
            todo = torch.zeros_like(todo)
            todo[idx[still_bad]] = True
        if clearance > 0.0 and bars_xy.shape[1] > 0 and bool(todo.any()):
            # At high density a few goals can survive 10 rejection rounds still inside the bar
            # clearance. Snap them radially away from the nearest bar (best effort) instead of
            # silently keeping a goal whose capture sphere is not flyable.
            gxy = goal[todo, 0:2]
            d_bar, j_bar = torch.cdist(gxy.unsqueeze(1), bars_xy[todo]).squeeze(1).min(dim=1)
            near = bars_xy[todo][torch.arange(len(j_bar), device=self.device), j_bar]
            away = gxy - near
            away = away / away.norm(dim=1, keepdim=True).clamp(min=1e-6)
            goal[todo, 0:2] = near + away * clearance
            logger.warning(
                "navrl reset: %d goals snapped out of bar clearance after 10 rejection rounds."
                % int(todo.sum())
            )
        self.target_position[env_ids] = goal
        # General-spawn target positions remain uniformly direction agnostic.  The sensor cold-
        # start curriculum acts on initial yaw only; the previous goal-sampling FOV branch is inert
        # in this mode because `todo` is intentionally all false above.
        self._align_general_spawn_yaw_to_target(env_ids, start_pos, goal)
        if self.general_spawn_mode:
            # Commit randomized XY and the optional target-relative yaw together.  One indexed
            # write avoids paying for two full simulator state updates on every episode reset.
            self.sim_env.robot_manager.robot.update_states()
            self.sim_env.IGE_env.write_to_sim()
        elif self._physical_target:
            # Unlike the historical virtual point, the target is an actor root state and must be
            # committed after task-side goal sampling.
            self.sim_env.IGE_env.write_to_sim()
        if self._oob_probe:
            self._probe_ep_start_y[env_ids] = start_pos[:, 1]
            self._probe_ep_target_start_y[env_ids] = goal[:, 1]
            if self.n_bars_active > 0:
                self._probe_ep_bar_mean_y[env_ids] = bars_xy[
                    :, : self.n_bars_active, 1
                ].mean(dim=1)
            else:
                self._probe_ep_bar_mean_y[env_ids] = 0.5 * (
                    b_min[:, 1] + b_max[:, 1]
                )
            self._probe_ep_y_min[env_ids] = start_pos[:, 1]
            self._probe_ep_y_max[env_ids] = start_pos[:, 1]

        d = self.target_position[env_ids] - start_pos
        d[:, 2] = 0.0  # horizontal goal direction defines the goal frame
        self.target_dir_2d[env_ids] = d
        self._episode_goal_dist[env_ids] = d[:, 0:2].norm(dim=1)
        rel_vehicle = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"][env_ids], d
        )
        initial_bearing = torch.atan2(rel_vehicle[:, 1], rel_vehicle[:, 0])
        centered = math.radians(5.0)
        # 0=negative vehicle-y, 1=centered (+/-5 deg), 2=positive vehicle-y.
        self._episode_bearing_bin[env_ids] = torch.where(
            initial_bearing < -centered,
            torch.zeros_like(initial_bearing, dtype=torch.long),
            torch.where(
                initial_bearing > centered,
                torch.full_like(initial_bearing, 2, dtype=torch.long),
                torch.ones_like(initial_bearing, dtype=torch.long),
            ),
        )

        self.height_range[env_ids, 0] = torch.minimum(
            start_pos[:, 2], self.target_position[env_ids, 2]
        )
        self.height_range[env_ids, 1] = torch.maximum(
            start_pos[:, 2], self.target_position[env_ids, 2]
        )
        self.prev_vel_w[env_ids] = 0.0
        self._z_err_integral[env_ids] = 0.0  # fresh episode -> no carried-over altitude-hold bias
        # Seed the ego-progress/segment-capture buffers with the spawn state. First-step progress is
        # ||start - target|| - gamma*||pos - target||, identical to the old prev_dist seeding.
        self.prev_pos[env_ids] = start_pos
        self.prev_rel[env_ids] = start_pos - self.target_position[env_ids]
        # vision mode: fresh episode -> tracker knows nothing, no previous action yet
        if self.vision_mode:
            self.detector.reset_idx(env_ids)
            if self.perception is not None:
                self.perception.reset_idx(env_ids)
            self.prev_action[env_ids] = 0.0
            self._visible_now[env_ids] = False
        if self._episode_dump_path:
            self._episode_spawn[env_ids] = self.obs_dict["robot_position"][env_ids].clone()
        if self._action_diag_enabled or self._obs_dump_enabled:
            # Must match the WRITE gate in _record_action_diagnostics: that method runs its body
            # (including the unconditional `_action_diag_prev` / `_action_diag_prev_valid` tail)
            # whenever EITHER action diagnostics or the obs dump is on. Gating the clear on
            # action-diag alone let delta_y_sum / sign_flip_y accumulate ACROSS episode boundaries
            # in a dump-only run.
            self._action_diag_prev_valid[env_ids] = False
        if self._joint_speed_telemetry is not None:
            self._joint_speed_telemetry.reset_idx(env_ids)
        if self._episode_forensics is not None:
            self._episode_forensics.reset_idx(env_ids)
        self._tm_ep_wall_reflections[env_ids] = 0
        self._tm_ep_bar_reflections[env_ids] = 0
        self._tm_ep_visible_steps[env_ids] = 0
        self._tm_ep_observation_steps[env_ids] = 0
        # 8.28: full per-env reset. -1 is "not yet acquired", NOT step 0 -- an episode that never
        # sees the target must stay distinguishable from one that saw it on its first observation.
        self._fa_ep_first_fused[env_ids] = -1
        self._fa_ep_first_camera[env_ids] = -1
        self._fa_ep_transitions[env_ids] = 0
        self._fa_ep_prev_visible[env_ids] = False
        self._fa_ep_obs_steps[env_ids] = 0
        # Phase 3: per-episode target speed + trajectory pattern (all-static when the speed
        # ceiling is 0 -> Phases 1-2 behavior).
        self._sample_target_motion(env_ids)
        if self._physical_target:
            self.target_orientation[env_ids] = 0.0
            self.target_orientation[env_ids, 3] = 1.0
            self.target_vel_w[env_ids] = 0.0
            self.obs_dict["obstacle_angvel"][env_ids, 0] = 0.0
            self._target_controller.reset_idx(env_ids)
            self.sim_env.IGE_env.write_to_sim()
        if self._target_route_enabled:
            self._target_route_manager.reset_idx(env_ids)
            self._plan_target_routes(
                env_ids, connected_goal=True, is_replan=False, plan_cause="reset"
            )
        self.ep_min_goal_dist[env_ids] = float("inf")
        self.ep_reached[env_ids] = False

    def render(self):
        return self.sim_env.render()

    # ------------------------------------------------------------------ step
    def transform_action_to_command(self, actions):
        """Action -> vehicle-frame velocity command for the controller.

        Default: NavRL 3D goal-frame velocity (the goal frame is defined by the KNOWN start->goal
        direction). Vision mode: the actor does not know the target position, so a goal frame is
        undefinable from its observations — actions are the vehicle-frame velocity directly
        (fly toward what the sensors show; the frame matches the body-frame scan and detector)."""
        if self.vision_mode:
            vel_vehicle = torch.clamp(actions[:, 0:3], -1.0, 1.0) * self.task_config.max_velocity
        else:
            vel_goal = torch.clamp(actions[:, 0:3], -1.0, 1.0) * self.task_config.max_velocity
            vel_world = goal_frame_to_world(vel_goal, self.target_dir_2d)
            vel_vehicle = quat_rotate_inverse(self.obs_dict["robot_vehicle_orientation"], vel_world)
        vel_vehicle = vel_vehicle.clone()
        vel_vehicle[:, 0:2] = self._apply_sensor_speed_governor(vel_vehicle[:, 0:2])
        self._last_governed_action_xy[:] = (
            vel_vehicle[:, 0:2] / max(1e-6, float(self.task_config.max_velocity))
        ).clamp(-1.0, 1.0)
        self.command[:, 0:3] = vel_vehicle
        # 2D flight: hold altitude. The vehicle frame is yaw-only (level), so vehicle-z == world-z.
        # A plain vz=0 command is OPEN-LOOP: the velocity controller carries no z-position feedback
        # (setpoint_position tracks the current position), so altitude bled away during aggressive
        # lateral/yaw maneuvers — measured with NAVRL_CRASH_DIAG on the first perception run, 39%
        # of all crashes were floor strikes (z < 0.1 m after ~4.5 s of flight). Close the loop with
        # a proportional altitude-hold velocity command instead (policy-independent stabilization;
        # the action space is unchanged — the actor still cannot command vertical motion).
        z_err = self.task_config.flight_altitude - self.obs_dict["robot_position"][:, 2]
        # Symmetric vertical authority. The prior +/-1 m/s hold lost to the +/-2 m/s tilt-induced
        # altitude sag during sustained lateral+yaw weaving; once the 8 m LiDAR horizon let episodes
        # survive in open space long enough (bar contacts 76% -> 14%), that latent bleed surfaced as
        # floor strikes (below 0% -> 71%). NOTE: there is NO floor mesh to hit (create_ground_plane
        # =False; the warp LiDAR raycasts only bar meshes), so this is a control-authority fix, not a
        # perception one. Match the lateral command's gain and authority so vertical recovery keeps up.
        # PI, not just P: sustained lateral+yaw weaving holds the vehicle tilted for multi-step
        # bursts, and during that tilt the attitude-tracking transient (desired vs actual body-z
        # axis) biases the achieved vertical acceleration low even though thrust magnitude has
        # plenty of headroom (T/W ~= 3.3) -- a proportional term alone settles to a nonzero
        # steady-state z_err under a persistent bias. The integral term removes that steady-state
        # sag; anti-windup clamp keeps it from overshooting once the bias clears (e.g. after a
        # crash-avoidance turn ends). Reset per-episode in reset_idx.
        # Vertical authority is alt_hold_vmax, NOT max_velocity: tying it to the horizontal speed
        # limit made every pursuer-speed sweep confound "slower pursuer crashes less" with "slower
        # pursuer has proportionally weaker altitude hold" (at 0.75 m/s it kept only ~30% of its
        # authority AND a 3x tighter anti-windup bound).
        _mv = float(getattr(self.task_config, "alt_hold_vmax", self.task_config.max_velocity))
        self._z_err_integral += z_err * self.step_dt
        _ki = 1.0
        _i_bound = _mv / _ki
        self._z_err_integral.clamp_(-_i_bound, _i_bound)
        self.command[:, 2] = torch.clamp(4.0 * z_err + _ki * self._z_err_integral, -_mv, _mv)
        # (b) learned yaw-rate: action[:, 3] in [-1, 1] -> euler yaw-rate (was held at 0).
        # yaw_rate_max matches the NavRL-scoped controller clamp; canonical v2 launchers pin 3.0
        # rad/s while the task fallback remains 2.5 for legacy/import compatibility.
        self._yaw_cmd[:] = torch.clamp(actions[:, 3], -1.0, 1.0)
        # L5 (plan I3): magnitude-only yaw cap beside close obstacles; scale is 1.0 unless enabled.
        self.command[:, 3] = self._yaw_cmd * self.task_config.yaw_rate_max * self._governor_yaw_scale
        return self.command

    def step(self, actions):
        if self.interactive_mode and self._interactive_manual:
            actions = self._interactive_manual_action
        self._record_action_diagnostics(actions)
        # Phase 3: move the virtual target FIRST — both agents move during this 0.1 s control
        # interval, and the end-of-interval reward is computed against the target's NEW position.
        # (No-op while all per-episode target speeds are 0, i.e. the static Phases 1-2 task.)
        if self._physical_target:
            self._target_controller.begin_control_interval()
        self._advance_target()
        command = self.transform_action_to_command(actions)
        if self._joint_speed_telemetry is not None:
            # Decision-time actual velocity and LiDAR clearance are sampled before physics moves.
            # This aligns the risk label with the observation/action that produced the command.
            actual_velocity_xy = self.obs_dict["robot_vehicle_linvel"][:, 0:2].detach()
            requested_command_xy = (
                torch.clamp(actions[:, 0:2], -1.0, 1.0)
                * float(self.task_config.max_velocity)
            ).detach()
            executed_command_xy = command[:, 0:2].detach()
            actual_clearance = self._diagnostic_directional_clearance(actual_velocity_xy)
            executed_clearance = self._diagnostic_directional_clearance(executed_command_xy)
            self._joint_speed_telemetry.record_step(
                actual_velocity_xy=actual_velocity_xy,
                requested_command_xy=requested_command_xy,
                executed_command_xy=executed_command_xy,
                policy_action_xy=torch.clamp(actions[:, 0:2], -1.0, 1.0).detach(),
                actual_direction_clearance_m=actual_clearance,
                requested_direction_clearance_m=self._speed_governor_last[
                    "clearance_m"
                ].detach(),
                executed_direction_clearance_m=executed_clearance,
            )
        if self.vision_mode:
            # remembered as "previous action" in the NEXT observation (ego proprioception)
            self.prev_action[:] = torch.clamp(actions[:, 0:4], -1.0, 1.0)
            if self.speed_governor_cfg.mode != "off":
                # The actor must observe what the controller actually executed; feeding the raw
                # request after a safety-layer intervention makes the transition partially hidden.
                self.prev_action[:, 0:2] = self._last_governed_action_xy
        self.sim_env.step(actions=command)

        # state-based reward + termination (LiDAR-based safety reward is added after rendering)
        self.compute_state_reward_and_terminations()

        self.truncations[:] = torch.where(
            _episode_limit_reached(
                self.sim_env.sim_steps, self.task_config.episode_len_steps
            ),
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )
        if self._interactive_reset_requested:
            self.truncations[:] = 1

        dist_to_goal = torch.norm(
            self.target_position - self.obs_dict["robot_position"], dim=1
        )
        # per-episode closest approach / ever-reached (updated before any env is reset)
        self.ep_min_goal_dist = torch.minimum(self.ep_min_goal_dist, dist_to_goal)
        self.ep_reached |= dist_to_goal < self.task_config.success_radius

        # Interception semantics (always on): capture ends the episode; timeouts are truncations
        # that never captured.
        successes = self.captured_now
        crashes = self.crashed_now
        timeouts, self.infos = _episode_outcome_info(
            successes, crashes, self.truncations
        )
        if self._obs_dump_enabled and bool(timeouts.any()):
            # 4=timeout (OBS_DUMP_OUTCOME_CODES); mutually exclusive with the crash/capture site
            # above by construction of `timeouts` (truncated & not success & not crash).
            code = torch.full((self.num_envs,), 4, dtype=torch.int8, device=self.device)
            self._record_obs_dump_outcome(timeouts, code)

        if self._bulk_eval_mode:
            steps = self.sim_env.sim_steps
            for label, mask in (
                ("capture", successes),
                ("crash", crashes > 0),
                ("timeout", timeouts),
            ):
                if bool(mask.any()):
                    self._speed_governor_outcome_steps[label].extend(
                        int(value) for value in steps[mask].detach().cpu().tolist()
                    )

        finished = (self.terminations > 0) | (self.truncations > 0)
        if self._joint_speed_telemetry is not None:
            self._joint_speed_telemetry.finish(
                finished, successes, crashes, timeouts, self._crash_cause_code
            )
        if self._episode_forensics is not None:
            scan = getattr(self.perception, "last_scan_nearest", None)
            clearance = (
                scan.amin(dim=1)
                if scan is not None
                else torch.zeros(self.num_envs, device=self.device)
            )
            # `valid` is the same finite-action mask the existing observation-step telemetry uses,
            # so both chronologies count the same steps.
            self._episode_forensics.record_step(
                valid=torch.isfinite(actions[:, 1]),
                robot_position=self.obs_dict["robot_position"],
                robot_velocity=self.obs_dict["robot_linvel"],
                robot_orientation_xyzw=self.obs_dict["robot_vehicle_orientation"],
                target_position=self.target_position,
                target_velocity=self.target_vel_w,
                requested_speed_mps=self._speed_governor_last["requested_speed_mps"],
                executed_speed_mps=self._speed_governor_last["executed_speed_mps"],
                governor_scale=self._speed_governor_last["scale"],
                action_xy=torch.clamp(actions[:, 0:2], -1.0, 1.0),
                clearance_m=clearance,
            )
            self._episode_forensics.finish(
                finished, successes, crashes, timeouts, self._crash_cause_code
            )
        if self._trajectory_digest is not None:
            self._trajectory_digest.record(
                position=self.obs_dict["robot_position"],
                orientation=self.obs_dict["robot_orientation"],
                command=command,
            )
        self._record_general_result(successes, crashes, timeouts, finished)
        self._log_progress(successes, crashes, timeouts, finished)
        self._update_curriculum(successes, finished)
        self._record_epoch_dashboard(successes, crashes, timeouts, finished)

        # The LiDAR reads this buffer inside its captured render graph. Keep it synchronized with
        # the same moving target that the camera renderer observes.
        self._sync_target_to_sensor()

        # render (raycast LiDAR from the new state) and reset finished envs
        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)
            # The env manager rendered once before the task sampled its new generalized drone and
            # target poses. Refresh both target injection and LiDAR so the first policy observation
            # of the next trial matches the newly randomized scene.
            self._sync_target_to_sensor()
            self.sim_env.render(render_components="sensors")
        self._interactive_reset_requested = False

        # LiDAR-based static-safety reward, using the freshly rendered scan
        self.add_static_safety_reward()

        self.num_task_steps += 1
        return self.get_return_tuple()

    @staticmethod
    def _empty_action_diag():
        return {
            "n": 0,
            "raw_oob": [0.0, 0.0, 0.0, 0.0],
            "exec_edge": [0.0, 0.0, 0.0, 0.0],
            "exec_edge95": [0.0, 0.0, 0.0, 0.0],
            "exec_edge99": [0.0, 0.0, 0.0, 0.0],
            "abs_sum": [0.0, 0.0, 0.0, 0.0],
            "signed_y_sum": 0.0,
            "positive_y": 0.0,
            "negative_y": 0.0,
            "high80_y": 0.0,
            "delta_y_sum": 0.0,
            "delta_y_n": 0,
            "sign_flip_y": 0.0,
            "front_clear_n": 0.0,
            "front_clear_abs_y": 0.0,
            "front_blocked_n": 0.0,
            "front_blocked_abs_y": 0.0,
            "goal_centered_n": 0.0,
            "goal_centered_abs_y": 0.0,
            "goal_offcenter_n": 0.0,
            "goal_offcenter_abs_y": 0.0,
            "clear_centered_n": 0.0,
            "clear_centered_abs_y": 0.0,
            "target_visible_n": 0.0,
            "target_visible_abs_y": 0.0,
            "target_hidden_n": 0.0,
            "target_hidden_abs_y": 0.0,
            "motion_n": 0.0,
            "motion_speed_sum": 0.0,
            "motion_command_speed_sum": 0.0,
            "motion_low_speed": 0.0,
            "motion_commanded_stall": 0.0,
        }

    def _empty_speed_governor_diag(self):
        return {
            key: torch.zeros((), device=self.device, dtype=torch.float64)
            for key in (
                "samples",
                "interventions",
                "near_stops",
                "requested_speed_sum",
                "executed_speed_sum",
                "clearance_sum",
                "scale_sum",
                "ttc_sum",
                "ttc_n",
                "negative_margin_requested",
                "negative_margin_executed",
                "contact_n",
                "contact_actual_speed_sum",
                "contact_requested_speed_sum",
                "contact_executed_speed_sum",
                "contact_clearance_sum",
                "contact_scale_sum",
                "contact_ttc_sum",
                "contact_margin_requested_sum",
                "contact_margin_executed_sum",
            )
        }

    def _apply_sensor_speed_governor(self, command_xy):
        """Apply the configured sensor-only speed layer and preserve its causal telemetry."""

        if self.speed_governor_cfg.mode == "off" and not self._speed_governor_diag_enabled:
            return command_xy
        if self._contact_geom_enabled:
            # Wall-clock cost of one governor call, so the filter can be compared against the
            # published budgets (HOCBF-QP 90.6 us mean / 384.5 us max).
            # No cuda.synchronize here: syncing would fold a full device barrier into the
            # measurement and dwarf the governor itself. This is the host-side cost of issuing
            # the batched governor for all envs, which is what a control loop actually pays.
            _cg_t0 = time.perf_counter()
        depth = self.obs_dict.get("depth_range_pixels")
        if not isinstance(depth, torch.Tensor) or depth.ndim < 4:
            raise RuntimeError("NavRL speed governor requires the LiDAR depth_range_pixels tensor")
        scan_m = torch.nan_to_num(
            depth.squeeze(1), nan=1.0, posinf=1.0, neginf=1.0
        ).clamp(0.0, 1.0) * float(self.task_config.lidar_max_range)
        hbeams = int(scan_m.shape[-1])
        if self._speed_governor_bearings is None or int(
            self._speed_governor_bearings.numel()
        ) != hbeams:
            from aerial_gym.task.navrl_task.navrl_perception import (
                HBEAMS as _HB,
                lidar_bin_bearings,
            )

            if hbeams == _HB:
                self._speed_governor_bearings = lidar_bin_bearings(self.device)
            else:
                bin_rad = 2.0 * math.pi / max(1, hbeams)
                self._speed_governor_bearings = torch.linspace(
                    math.pi,
                    -math.pi + bin_rad,
                    hbeams,
                    device=self.device,
                )
        target_return = None
        if self.perception is not None:
            candidate = self.perception.last_target_like
            if candidate.shape != scan_m.shape:
                raise RuntimeError(
                    "sensor-associated target mask does not match the current LiDAR scan"
                )
            target_return = candidate
        mode = self.speed_governor_cfg.mode
        from aerial_gym.task.navrl_task import speed_governor as SG

        # L2: per-env corridor half-width; the scalar w0 unless width_per_mps > 0.
        # L8 needs the vehicle's own clutter read BEFORE the corridor is drawn: nearest sensed
        # surface in any bearing, which is exactly the omni clearance.
        open_m = None
        if self.speed_governor_cfg.open_width_enabled:
            open_m = SG.omnidirectional_clearance(
                scan_m, max_range_m=float(self.task_config.lidar_max_range),
                target_return_mask=target_return,
            )
        half_width = SG.speed_dependent_half_width(
            self.speed_governor_cfg, command_xy.norm(dim=1), open_m=open_m
        )
        if mode == "omni":
            # A4 baseline: identical stopping law, corridor removed entirely.
            from aerial_gym.task.navrl_task.speed_governor import omnidirectional_clearance

            clearance = omnidirectional_clearance(
                scan_m,
                max_range_m=float(self.task_config.lidar_max_range),
                target_return_mask=target_return,
            )
        elif mode in ("dwa_arc", "riskcap_arc"):
            # A7: both laws measure clearance with the same sensor-only arc geometry.
            from aerial_gym.task.navrl_task.speed_governor import arc_clearance

            yaw_rate = self.obs_dict["robot_body_angvel"][:, 2].detach()
            clearance = arc_clearance(
                scan_m,
                self._speed_governor_bearings,
                command_xy,
                yaw_rate,
                max_range_m=float(self.task_config.lidar_max_range),
                path_half_width_m=half_width,
                target_return_mask=target_return,
            )
        else:
            clearance = directional_lidar_clearance(
                scan_m,
                self._speed_governor_bearings,
                command_xy,
                max_range_m=float(self.task_config.lidar_max_range),
                path_half_width_m=half_width,
                target_return_mask=target_return,
            )
        # L3 / L5 share one lateral clearance; computed only when either is enabled so every
        # earlier arm stays byte-identical in work and in numbers.
        lateral = None
        if self.speed_governor_cfg.lateral_channel_enabled or self.speed_governor_cfg.yaw_cap_enabled:
            lateral = SG.lateral_clearance(
                scan_m,
                self._speed_governor_bearings,
                command_xy,
                max_range_m=float(self.task_config.lidar_max_range),
                target_return_mask=target_return,
            )
        if self.speed_governor_cfg.yaw_cap_enabled:
            self._governor_yaw_scale[:] = SG.yaw_scale(
                lateral, float(self.task_config.yaw_rate_max), self.speed_governor_cfg
            )
        if self._contact_geom_enabled:
            # The exact vector the corridor was drawn around, before any scaling.
            self._governor_command_xy[:] = command_xy.detach()
            self._push_contact_geometry_history(command_xy.detach())
            self._record_contact_geometry_step(command_xy.detach(), clearance.detach())
        governed, telemetry = apply_speed_governor(
            command_xy, clearance, self.speed_governor_cfg, lateral_clearance_m=lateral
        )
        for key, value in telemetry.items():
            self._speed_governor_last[key][:] = value.detach()

        if self._contact_geom_enabled:
            self._push_contact_geometry_governor(telemetry)
            _cg_us = (time.perf_counter() - _cg_t0) * 1e6
            self._cg_aux[4] += _cg_us
            self._cg_aux[5] += 1.0
            self._cg_aux[6] = max(float(self._cg_aux[6]), _cg_us)

        if self._speed_governor_diag_enabled:
            diag = self._speed_governor_diag
            requested = telemetry["requested_speed_mps"].detach()
            executed = telemetry["executed_speed_mps"].detach()
            scale = telemetry["scale"].detach()
            ttc = telemetry["ttc_requested_s"].detach()
            finite_ttc = torch.isfinite(ttc)
            diag["samples"] += requested.numel()
            diag["interventions"] += (scale < 0.999).sum()
            diag["near_stops"] += ((scale < 0.05) & (requested > 0.1)).sum()
            diag["requested_speed_sum"] += requested.sum(dtype=torch.float64)
            diag["executed_speed_sum"] += executed.sum(dtype=torch.float64)
            diag["clearance_sum"] += clearance.detach().sum(dtype=torch.float64)
            diag["scale_sum"] += scale.sum(dtype=torch.float64)
            diag["ttc_sum"] += ttc[finite_ttc].sum(dtype=torch.float64)
            diag["ttc_n"] += finite_ttc.sum()
            diag["negative_margin_requested"] += (
                telemetry["stopping_margin_requested_m"] < 0.0
            ).sum()
            diag["negative_margin_executed"] += (
                telemetry["stopping_margin_executed_m"] < 0.0
            ).sum()
        return governed

    def _diagnostic_directional_clearance(self, command_xy):
        """Actor-safe LiDAR minimum in exactly ``command_xy``'s direction (diagnostic only)."""

        if self._joint_speed_telemetry is None:
            raise RuntimeError("joint directional clearance requested while telemetry is disabled")
        depth = self.obs_dict.get("depth_range_pixels")
        if not isinstance(depth, torch.Tensor) or depth.ndim < 4:
            raise RuntimeError("NavRL joint telemetry requires depth_range_pixels")
        scan_m = torch.nan_to_num(
            depth.squeeze(1), nan=1.0, posinf=1.0, neginf=1.0
        ).clamp(0.0, 1.0) * float(self.task_config.lidar_max_range)
        if self._speed_governor_bearings is None:
            raise RuntimeError("joint telemetry clearance ran before governor bearing setup")
        target_return = None
        if self.perception is not None:
            candidate = self.perception.last_target_like
            if candidate.shape != scan_m.shape:
                raise RuntimeError(
                    "sensor-associated target mask does not match the current LiDAR scan"
                )
            target_return = candidate
        return directional_lidar_clearance(
            scan_m,
            self._speed_governor_bearings,
            command_xy,
            max_range_m=float(self.task_config.lidar_max_range),
            path_half_width_m=self.speed_governor_cfg.path_half_width_m,
            target_return_mask=target_return,
        ).detach()

    def _record_action_diagnostics(self, actions):
        """Accumulate action tails plus context needed to separate avoidance from policy bias."""
        # NAVRL_OBS_DUMP reuses this method's context masks (front-blocked, target-visible,
        # valid_y_now) instead of recomputing them, so it must run the body even when action
        # diagnostics themselves are off. Still a true no-op when BOTH are disabled.
        if not self._action_diag_enabled and not self._obs_dump_enabled:
            return
        with torch.no_grad():
            action = actions[:, :4].detach()
            if action.shape[1] != 4:
                return
            finite = torch.isfinite(action)
            safe = torch.where(finite, action, torch.zeros_like(action))
            self._action_diag["n"] += int(action.shape[0])
            raw_oob = (safe.abs() > 1.0) & finite
            executed = safe.clamp(-1.0, 1.0)
            exec_edge = (executed.abs() >= 0.98) & finite
            exec_edge95 = (executed.abs() >= 0.95) & finite
            exec_edge99 = (executed.abs() >= 0.99) & finite
            abs_sum = safe.abs() * finite
            for axis in range(4):
                self._action_diag["raw_oob"][axis] += float(
                    raw_oob[:, axis].sum().item()
                )
                self._action_diag["exec_edge"][axis] += float(
                    exec_edge[:, axis].sum().item()
                )
                self._action_diag["exec_edge95"][axis] += float(
                    exec_edge95[:, axis].sum().item()
                )
                self._action_diag["exec_edge99"][axis] += float(
                    exec_edge99[:, axis].sum().item()
                )
                self._action_diag["abs_sum"][axis] += float(
                    abs_sum[:, axis].sum().item()
                )

            valid_y_now = finite[:, 1]
            ay = safe[:, 1]
            abs_y = ay.abs()
            self._action_diag["signed_y_sum"] += float(ay[valid_y_now].sum().item())
            self._action_diag["positive_y"] += float(
                ((ay > 0.1) & valid_y_now).sum().item()
            )
            self._action_diag["negative_y"] += float(
                ((ay < -0.1) & valid_y_now).sum().item()
            )
            self._action_diag["high80_y"] += float(
                ((abs_y >= 0.8) & valid_y_now).sum().item()
            )

            # Actual-vehicle motion telemetry for the dense-arena bottleneck. A low measured
            # velocity is only called a commanded stall when the target is still >1 m away and
            # the policy requests at least 20% of horizontal speed authority. This excludes
            # intentional stopping at capture and reports the real Isaac Gym vehicle, not the
            # illustrative status-site pursuer.
            robot_velocity = self.obs_dict.get("robot_linvel")
            robot_position = self.obs_dict.get("robot_position")
            if (
                isinstance(robot_velocity, torch.Tensor)
                and isinstance(robot_position, torch.Tensor)
                and robot_velocity.ndim == 2
                and robot_velocity.shape[1] >= 2
            ):
                max_velocity = max(1e-6, float(self.task_config.max_velocity))
                actual_speed = robot_velocity[:, :2].norm(dim=1)
                command_speed = executed[:, :2].norm(dim=1) * max_velocity
                goal_distance = (
                    self.target_position[:, :2] - robot_position[:, :2]
                ).norm(dim=1)
                motion_valid = finite[:, :2].all(dim=1) & (goal_distance > 1.0)
                low_speed = actual_speed < 0.2 * max_velocity
                command_active = command_speed >= 0.2 * max_velocity
                self._action_diag["motion_n"] += float(motion_valid.sum().item())
                self._action_diag["motion_speed_sum"] += float(
                    actual_speed[motion_valid].sum().item()
                )
                self._action_diag["motion_command_speed_sum"] += float(
                    command_speed[motion_valid].sum().item()
                )
                self._action_diag["motion_low_speed"] += float(
                    (motion_valid & low_speed).sum().item()
                )
                self._action_diag["motion_commanded_stall"] += float(
                    (motion_valid & low_speed & command_active).sum().item()
                )

            # Diagnostics only: classify the command using the same LiDAR frame as the actor.
            # Target returns are excluded so chasing a centered target is not mislabeled as an
            # obstacle. "Blocked" means a static return within 4 m in the forward +/-30 degree
            # sector. This never enters observations, rewards, terminations or commands.
            depth = self.obs_dict.get("depth_range_pixels")
            if isinstance(depth, torch.Tensor) and depth.ndim >= 4:
                scan = torch.nan_to_num(
                    depth.squeeze(1), nan=1.0, posinf=1.0, neginf=1.0
                ).clamp(0.0, 1.0)
                hbeams = int(scan.shape[-1])
                if (
                    self._action_front_mask is None
                    or int(self._action_front_mask.numel()) != hbeams
                ):
                    from aerial_gym.task.navrl_task.navrl_perception import (
                        HBEAMS as _HB,
                        lidar_bin_bearings,
                    )

                    if hbeams == _HB:
                        angles = torch.rad2deg(lidar_bin_bearings(self.device))
                    else:  # non-perception scan shape: derive locally, same DECREASING convention
                        bin_deg = 360.0 / max(1, hbeams)
                        angles = torch.linspace(
                            180.0, -180.0 + bin_deg, hbeams, device=self.device
                        )
                    self._action_front_mask = angles.abs() <= 30.0
                segmentation = self.obs_dict.get("segmentation_pixels")
                if isinstance(segmentation, torch.Tensor):
                    target_return = segmentation.squeeze(1) == 50
                    if target_return.shape == scan.shape:
                        scan = torch.where(target_return, torch.ones_like(scan), scan)
                front_min = scan[:, :, self._action_front_mask].amin(dim=(1, 2))
                blocked_threshold = min(
                    0.999,
                    4.0 / max(1e-6, float(self.task_config.lidar_max_range)),
                )
                front_blocked = (front_min < blocked_threshold) & valid_y_now
                front_clear = ~front_blocked & valid_y_now
                # NAVRL_OBS_DUMP context tag: this branch is the only place `front_blocked` is
                # actually defined (blocked/clear are known). Aliased, not renamed, so nothing
                # about the existing action-diagnostics tensors changes.
                _obs_dump_front_blocked = front_blocked
                for name, mask in (
                    ("front_clear", front_clear),
                    ("front_blocked", front_blocked),
                ):
                    self._action_diag[name + "_n"] += float(mask.sum().item())
                    self._action_diag[name + "_abs_y"] += float(abs_y[mask].sum().item())
            else:
                front_clear = torch.zeros_like(valid_y_now)
                # NAVRL_OBS_DUMP context tag: no depth scan this call, so blocked/clear is
                # genuinely UNKNOWN here -- do not let it silently read as "clear" downstream.
                _obs_dump_front_blocked = None

            # Ground truth is used only to label this diagnostic. The future v3 gate is derived
            # from the actor's structured target track instead; no oracle feature is introduced.
            rpos = self.target_position - self.obs_dict["robot_position"]
            goal_vehicle = quat_rotate_inverse(
                self.obs_dict["robot_vehicle_orientation"], rpos
            )
            goal_centered = _goal_front_centered(goal_vehicle) & valid_y_now
            goal_offcenter = ~goal_centered & valid_y_now
            for name, mask in (
                ("goal_centered", goal_centered),
                ("goal_offcenter", goal_offcenter),
            ):
                self._action_diag[name + "_n"] += float(mask.sum().item())
                self._action_diag[name + "_abs_y"] += float(abs_y[mask].sum().item())
            clear_centered = front_clear & goal_centered
            self._action_diag["clear_centered_n"] += float(clear_centered.sum().item())
            self._action_diag["clear_centered_abs_y"] += float(
                abs_y[clear_centered].sum().item()
            )
            visible = self._visible_now & valid_y_now
            hidden = ~self._visible_now & valid_y_now
            for name, mask in (("target_visible", visible), ("target_hidden", hidden)):
                self._action_diag[name + "_n"] += float(mask.sum().item())
                self._action_diag[name + "_abs_y"] += float(abs_y[mask].sum().item())
            if self._obs_dump_enabled:
                self._collect_obs_dump_frame(valid_y_now, visible, _obs_dump_front_blocked)
            if self._bulk_eval_mode:
                if self._s1_search_telemetry_enabled and bool(hidden.any()):
                    self._s1_blind_samples += hidden.sum().to(torch.long)
                    blind_speed = self.obs_dict["robot_linvel"][:, :2].norm(dim=1)
                    self._s1_blind_speed_sum += blind_speed[hidden].sum(dtype=torch.float64)
                    if self.perception is not None:
                        clearance = self.perception.last_scan_nearest.amin(dim=1)
                        self._s1_blind_clearance_sum += clearance[hidden].sum(
                            dtype=torch.float64
                        )
                # This labels exactly the observation consumed to select the current action.
                # `valid_y_now` excludes non-finite policy rows from both numerator and denominator.
                self._tm_ep_observation_steps += valid_y_now.to(torch.long)
                self._tm_ep_visible_steps += visible.to(torch.long)

                # RESEARCH_PLAN 8.28. Chronology is in OBSERVATION steps, not sim steps, so a
                # first-visible index is directly comparable with the visible-fraction denominator
                # the seed-353 telemetry already uses.
                self._fa_ep_obs_steps += valid_y_now.to(torch.long)
                camera_visible = self._camera_visible_now & valid_y_now
                fresh = visible & (self._fa_ep_first_fused < 0)
                if bool(fresh.any()):
                    self._fa_ep_first_fused[fresh] = self._fa_ep_obs_steps[fresh]
                fresh_cam = camera_visible & (self._fa_ep_first_camera < 0)
                if bool(fresh_cam.any()):
                    self._fa_ep_first_camera[fresh_cam] = self._fa_ep_obs_steps[fresh_cam]
                # A visible->hidden transition only counts between two VALID observations; a step
                # dropped for non-finite policy output is not evidence that the track was lost.
                lost = self._fa_ep_prev_visible & ~visible & valid_y_now
                self._fa_ep_transitions += lost.to(torch.long)
                self._fa_ep_prev_visible = torch.where(
                    valid_y_now, visible, self._fa_ep_prev_visible
                )

            valid = self._action_diag_prev_valid & finite[:, 1]
            if bool(valid.any()):
                current_y = safe[valid, 1]
                previous_y = self._action_diag_prev[valid, 1]
                self._action_diag["delta_y_sum"] += float(
                    (current_y - previous_y).abs().sum().item()
                )
                self._action_diag["delta_y_n"] += int(valid.sum().item())
                sign_flip = (
                    (current_y * previous_y < 0.0)
                    & (current_y.abs() > 0.1)
                    & (previous_y.abs() > 0.1)
                )
                self._action_diag["sign_flip_y"] += float(sign_flip.sum().item())

            self._action_diag_prev[:] = safe
            self._action_diag_prev_valid[:] = finite.all(dim=1)

    def _collect_obs_dump_frame(self, valid_y_now, visible, front_blocked):
        """NAVRL_OBS_DUMP: append one row per env for this call, if the streaming decimation
        (`_obs_dump_retain_decision` / `_obs_dump_thin_step`) retains it.

        Reads only the exact observation tensor handed to the actor and context masks this method
        already computed for action diagnostics -- nothing new is derived, and nothing here writes
        back into `self.task_obs`, rewards, or terminations.
        """
        # 0-based on purpose: the streaming decimation (see `_obs_dump_thin_step`) only stays an
        # EXACT "every stride_eff-th call from the start" grid -- with no permanently-misaligned
        # leftover sample -- if the very first call is index 0, not 1. Verified by simulation.
        this_call = self._obs_dump_calls
        self._obs_dump_calls += 1
        if not _obs_dump_retain_decision(this_call, self._obs_dump_stride_eff):
            return
        with torch.no_grad():
            obs = self.task_obs["observations"].detach().to("cpu", torch.float32).numpy()
        n = int(obs.shape[0])
        env_id = np.arange(n, dtype=np.int32)
        call_index = np.full(n, this_call, dtype=np.int64)
        # Host mirror (kept exact in reset_idx) -- identical values to the device tensor, minus
        # the per-call device->host sync.
        ep_idx_cpu = self._obs_dump_ep_idx_host
        episode_uid = env_id.astype(np.int64) + self.num_envs * ep_idx_cpu
        ep_step = self.sim_env.sim_steps.detach().to("cpu", torch.int64).numpy()
        ctx_target_visible = visible.detach().to("cpu", torch.bool).numpy()
        ctx_valid = valid_y_now.detach().to("cpu", torch.bool).numpy()
        if front_blocked is None:
            ctx_front_blocked = np.full(n, -1, dtype=np.int8)
        else:
            fb_cpu = front_blocked.detach().cpu().numpy()
            ctx_front_blocked = fb_cpu.astype(np.int8)
        self._obs_dump_frames.append(
            {
                "obs": obs,
                "env_id": env_id,
                "call_index": call_index,
                "episode_uid": episode_uid,
                "ep_step": ep_step,
                "ctx_target_visible": ctx_target_visible,
                "ctx_front_blocked": ctx_front_blocked,
                "ctx_valid": ctx_valid,
            }
        )
        while True:
            _, new_stride_eff, thinned = _obs_dump_thin_step(
                len(self._obs_dump_frames),
                self._obs_dump_stride_eff,
                self._obs_dump_max_rows,
                self.num_envs,
            )
            if not thinned:
                break
            self._obs_dump_frames = self._obs_dump_frames[0::2]
            self._obs_dump_stride_eff = new_stride_eff
            self._obs_dump_decimations += 1

    def _record_obs_dump_outcome(self, mask, code):
        """NAVRL_OBS_DUMP: append one per-episode outcome row for every env set in `mask`.

        `code` is a per-env int8 outcome-code tensor (see `OBS_DUMP_OUTCOME_CODES`); only the
        entries selected by `mask` are used. Shared by the crash/capture site (compute_state_
        reward_and_terminations) and the timeout site (step()) so both write the same row shape.
        """
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return
        try:
            self._obs_dump_episode_row_count = _obs_dump_check_episode_budget(
                self._obs_dump_episode_row_count,
                int(idx.numel()),
                self._obs_dump_max_episode_rows,
            )
        except RuntimeError as exc:
            # Remember the overflow: the flush must then refuse to write at all, because a dump
            # whose outcome table stopped growing mid-run looks complete but silently breaks the
            # frame/outcome join. The raise still aborts the run at the first overflowing step.
            self._obs_dump_episode_overflow = str(exc)
            raise
        ep_uid = idx.to(torch.int64) + self.num_envs * self._obs_dump_ep_idx[idx]
        self._obs_dump_episode_rows.append(
            {
                "ep_uid": ep_uid.detach().cpu().numpy(),
                "ep_env_id": idx.to(torch.int32).detach().cpu().numpy(),
                "outcome": code[idx].detach().cpu().numpy(),
                "ep_len": self.sim_env.sim_steps[idx].to(torch.int64).detach().cpu().numpy(),
            }
        )

    def _record_obs_dump_crash_outcomes(
        self,
        captured,
        crashed_out,
        d_contact,
        d_oob,
        d_below,
        d_above,
        crashed,
        target_contact,
        target_invalid,
    ):
        """NAVRL_OBS_DUMP: outcome rows for episodes ending in capture or a crash this step.

        Remaps the SAME priority-attributed masks the crash-cause table above just computed
        (contact > below > above > oob) onto the dump's outcome codes -- no new attribution logic,
        and the task's masks are only READ here. `crashed` / `target_contact` / `target_invalid`
        are the three raw sources the merged `d_contact` is the union of; they are passed in so
        the dump can label a TARGET-caused termination as such (codes 8/9) instead of reporting a
        clean drone flight as a drone-bar collision (code 1). `d_below` / `d_above` likewise get
        their own codes (6/7) instead of being collapsed into 3, matching the distinction
        `_crash_cause_code` already keeps.

        Timeouts are not visible here; they are recorded separately in step() once
        `_episode_outcome_info` resolves them. Any crashed_out env not covered by one of these
        masks (should not happen given how `crashed_out` is built above) gets 5=unattributed
        rather than a guess.
        """
        done_now = captured | crashed_out
        if not bool(done_now.any()):
            return
        code = torch.full((self.num_envs,), 5, dtype=torch.int8, device=self.device)
        _obs_dump_assign_crash_codes(
            code,
            captured,
            crashed,
            target_contact,
            target_invalid,
            d_contact,
            d_oob,
            d_below,
            d_above,
        )
        self._record_obs_dump_outcome(done_now, code)

    def _note_obs_dump_full_reset(self):
        """NAVRL_OBS_DUMP: remember the episode_uids a full `reset()` is about to end silently.

        Reads the host mirror only (no GPU sync, no behaviour change). Bounded: one entry per env
        per FULL reset, and full resets are rare (once per rl_games `n_games` iteration). The cap
        exists so a pathological caller cannot turn this into an unbounded set.
        """
        uids = (
            np.arange(self.num_envs, dtype=np.int64)
            + self.num_envs * self._obs_dump_ep_idx_host
        )
        self._obs_dump_reset_orphan_uids.update(int(v) for v in uids.tolist())
        if len(self._obs_dump_reset_orphan_uids) > self._obs_dump_reset_orphan_cap:
            raise RuntimeError(
                "obs_dump export: %d full-reset orphan episode_uid(s) tracked, over the cap %d -- "
                "NAVRL_OBS_DUMP is an EVALUATION-only hook and this run resets far more often "
                "than any evaluation does."
                % (len(self._obs_dump_reset_orphan_uids), self._obs_dump_reset_orphan_cap)
            )

    def _obs_dump_schema_obs_width(self):
        """Structured-observation width recomputed from the perception schema's COMPONENTS.

        Why this exists: the frames are sliced from `self.task_obs["observations"]`, which is
        allocated with `task_config.observation_space_dim`, so checking the collected array's width
        against that same number is a value compared to itself and can never catch the 898->N
        regression the check was written for. This recomputes the width from the parts instead --
        the per-history token dims from the perception module, plus the STATIC block's size taken
        from the LIVE LiDAR tensor's own shape (a different allocation, made by the sensor config,
        not by the observation buffer). Neither path runs through the observation allocation.

        What it still cannot be independent of: every one of these numbers ultimately derives from
        the same process's environment (NAVRL_LIDAR_HBEAMS / NAVRL_MAX_OBSTACLES / ...), so an
        env-var change that is consistently applied everywhere is a legitimate reconfiguration, not
        a regression, and is correctly NOT flagged. Returns None when the run has no perception
        front-end -- there is then no second source at all, and the export guard falls back to
        asserting only that the recorded width is the width the consumer will assume.
        """
        if not self.perception_mode:
            return None
        from aerial_gym.task.navrl_task.navrl_perception import (
            CORRIDOR_OBS_DIM,
            MAX_OBSTACLES,
            OBSTACLE_DIM,
            OBSTACLE_HISTORY,
            ROBOT_DIM,
            ROBOT_HISTORY,
            STATIC_DIM,
            TARGET_DIM,
            TARGET_HISTORY,
        )

        static_dim = int(STATIC_DIM)
        pixels = self.obs_dict.get("depth_range_pixels", None)
        if pixels is not None and pixels.dim() >= 3:
            # (N, [sensors,] vbeams, hbeams) -- the same layout _lidar_distance_m() relies on.
            static_dim = int(pixels.shape[-1]) * int(pixels.shape[-2])
        return int(
            ROBOT_HISTORY * ROBOT_DIM
            + TARGET_HISTORY * TARGET_DIM
            + OBSTACLE_HISTORY * MAX_OBSTACLES * OBSTACLE_DIM
            + static_dim
            + CORRIDOR_OBS_DIM
        )

    def _flush_obs_dump(self):
        """Write the NAVRL_OBS_DUMP frame + episode-outcome tables, fail-closed (see
        `_validate_obs_dump_export`).

        Two triggers, one write: `atexit` (as `_flush_episode_dump` does) plus the deterministic
        in-process call in `close()`. atexit alone loses the entire dump to a SIGKILL, an
        `os._exit`, or a teardown segfault, and it is the worst moment to touch the simulator.
        Idempotent by construction -- `_obs_dump_flush_done` makes whichever trigger fires second a
        no-op, and the write itself goes to a temp file that is `os.replace`d into position, so the
        dump path is either absent or complete, never half-written.

        Any failure inside the write leaves `<path>.FAILED` next to the intended output holding the
        traceback. atexit swallows exit codes (a raise there prints a traceback but still exits 0,
        so `set -euo pipefail` does not trip), which previously let a downstream
        `require(frames.is_file())` pass on a PREVIOUS condition's stale bytes; a marker on disk is
        checkable after the fact regardless of exit code.
        """
        if self._obs_dump_flush_done or self._obs_dump_flush_active:
            return
        if not self._obs_dump_frames:
            return
        self._obs_dump_flush_active = True
        try:
            self._write_obs_dump()
        except BaseException:
            import traceback

            detail = traceback.format_exc()
            marker = Path(str(self._obs_dump_path) + ".FAILED")
            try:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(detail, encoding="utf-8")
                print(
                    "[obs_dump] FLUSH FAILED -- wrote %s; %s does NOT contain this run's data"
                    % (marker, self._obs_dump_path)
                )
            except Exception as marker_error:  # pragma: no cover - disk-level failure
                print(
                    "[obs_dump] FLUSH FAILED and the %s marker could not be written: %s"
                    % (marker, marker_error)
                )
            print(detail)
            raise
        else:
            self._obs_dump_flush_done = True
        finally:
            self._obs_dump_flush_active = False

    def _write_obs_dump(self):
        """Body of `_flush_obs_dump`, split out so EVERY failure path is wrapped by the
        `.FAILED`-marker handler above (including the export guard's own RuntimeErrors)."""
        if self._obs_dump_episode_overflow:
            raise RuntimeError(
                "obs_dump export: refusing to write -- the per-episode outcome table overflowed "
                "earlier in this run, so the frame/outcome join is incomplete: %s"
                % self._obs_dump_episode_overflow
            )
        path = Path(self._obs_dump_path)
        # Re-checked here (also checked at construction): the file must not exist at write time
        # either, e.g. when a sibling condition of the same sweep finished in between.
        _obs_dump_assert_free_path(path, path.exists())
        frame_tables = {
            key: np.concatenate([chunk[key] for chunk in self._obs_dump_frames])
            for key in self._obs_dump_frames[0]
        }
        if self._obs_dump_episode_rows:
            episode_tables = {
                key: np.concatenate([chunk[key] for chunk in self._obs_dump_episode_rows])
                for key in self._obs_dump_episode_rows[0]
            }
        else:
            episode_tables = {
                "ep_uid": np.zeros(0, dtype=np.int64),
                "ep_env_id": np.zeros(0, dtype=np.int32),
                "outcome": np.zeros(0, dtype=np.int8),
                "ep_len": np.zeros(0, dtype=np.int64),
            }
        # Host mirror, not the device tensor: this runs at interpreter shutdown too.
        ep_idx_cpu = self._obs_dump_ep_idx_host.astype(np.int64)
        live_episode_uids = set(
            (np.arange(self.num_envs, dtype=np.int64) + self.num_envs * ep_idx_cpu).tolist()
        )
        frame_tables, dropped_rows, dropped_episodes = _obs_dump_drop_reset_orphans(
            frame_tables, episode_tables["ep_uid"], self._obs_dump_reset_orphan_uids
        )
        if dropped_rows:
            print(
                "[obs_dump] dropped %d frame row(s) from %d episode(s) ended by a full reset() "
                "with no outcome row (reason=full_reset_orphan); every other "
                "finished-without-outcome episode is still fatal"
                % (dropped_rows, dropped_episodes)
            )
        schema_obs_width = self._obs_dump_schema_obs_width()
        _validate_obs_dump_export(
            frame_tables,
            episode_tables,
            int(self.task_config.observation_space_dim),
            self._obs_dump_max_rows,
            live_episode_uids,
            schema_obs_width=schema_obs_width,
        )
        outcome_code_values, outcome_code_names, outcome_code_map_json = (
            _obs_dump_outcome_code_table()
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic publish: a partially written .npz must never appear at `path`, or a downstream
        # `require(frames.is_file())` would pass on a truncated archive.
        tmp_path = path.parent / ("%s.partial-%d" % (path.name, os.getpid()))
        try:
            with open(str(tmp_path), "wb") as stream:
                np.savez_compressed(
                    stream,
                    **frame_tables,
                    **episode_tables,
                    stride_eff=np.int64(self._obs_dump_stride_eff),
                    decimations=np.int64(self._obs_dump_decimations),
                    # Run identity: a stale file left by another condition of the same sweep is
                    # detectable after the fact even if the overwrite guard was bypassed.
                    run_bars=np.int64(int(self.n_bars_active)),
                    run_max_bars=np.int64(int(self.max_bars_available)),
                    run_seed=np.int64(int(getattr(self.task_config, "seed", -1))),
                    run_num_envs=np.int64(int(self.num_envs)),
                    run_pid=np.int64(os.getpid()),
                    run_num_bars_env=np.array(
                        os.environ.get("NAVRL_NUM_BARS", "").strip(), dtype="<U32"
                    ),
                    run_obs_dump_path=np.array(str(self._obs_dump_path), dtype="<U512"),
                    # Widths: recorded vs the allocation the consumer assumes vs the independently
                    # derived schema width (-1 = no perception front-end to derive one from).
                    obs_width_recorded=np.int64(int(frame_tables["obs"].shape[1])),
                    obs_width_live_alloc=np.int64(
                        int(self.task_config.observation_space_dim)
                    ),
                    obs_width_schema=np.int64(
                        -1 if schema_obs_width is None else int(schema_obs_width)
                    ),
                    dropped_reset_orphan_frames=np.int64(dropped_rows),
                    dropped_reset_orphan_episodes=np.int64(dropped_episodes),
                    # The outcome-code map travels WITH the data (see OBS_DUMP_OUTCOME_CODES).
                    outcome_code_values=outcome_code_values,
                    outcome_code_names=outcome_code_names,
                    outcome_code_map_json=np.array(outcome_code_map_json, dtype="<U1024"),
                )
            os.replace(str(tmp_path), str(path))
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:  # pragma: no cover - best effort cleanup
                    pass
        sha256 = _sha256_file(path)
        print(
            "[obs_dump] path=%s frames=%d episodes=%d stride_eff=%d decimations=%d "
            "dropped_reset_orphans=%d bars=%d seed=%d sha256=%s"
            % (
                path,
                int(frame_tables["obs"].shape[0]),
                int(episode_tables["ep_uid"].shape[0]),
                int(self._obs_dump_stride_eff),
                int(self._obs_dump_decimations),
                int(dropped_rows),
                int(self.n_bars_active),
                int(getattr(self.task_config, "seed", -1)),
                sha256,
            )
        )

    def _lidar_distance_m(self):
        """Per-ray distance in meters, shape (N, vbeams*hbeams). Normalized pixels * max_range."""
        pix = self.obs_dict["depth_range_pixels"].squeeze(1)  # (N, vbeams, hbeams), in [0, 1]
        pix = torch.nan_to_num(pix, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.0, 1.0)
        return (pix * self.task_config.lidar_max_range).reshape(self.num_envs, -1)

    def compute_state_reward_and_terminations(self):
        pos = self.obs_dict["robot_position"]
        vel_w = self.obs_dict["robot_linvel"]

        rpos = self.target_position - pos
        dist = rpos.norm(dim=1).clamp(min=1e-6)
        vel_dir = rpos / dist.unsqueeze(1)
        # Range-rate (closing speed): the component of the RELATIVE velocity toward the target.
        # With a static target (target_vel_w == 0, the Phases 1-2 default) the subtraction is
        # IEEE-exact zero, so this reduces to NavRL's velocity-toward-goal term bit-for-bit.
        reward_vel = ((vel_w - self.target_vel_w) * vel_dir).sum(dim=1)

        penalty_smooth = (vel_w - self.prev_vel_w).norm(dim=1)
        self.prev_vel_w[:] = vel_w

        z = pos[:, 2]
        m = self.rw["height_margin"]
        hi = self.height_range[:, 1] + m
        lo = self.height_range[:, 0] - m
        penalty_height = torch.zeros_like(z)
        penalty_height = torch.where(z > hi, (z - hi) ** 2, penalty_height)
        penalty_height = torch.where(z < lo, (lo - z) ** 2, penalty_height)

        self.rewards[:] = (
            self.rw["vel_weight"] * reward_vel
            + self.rw["alive_weight"]
            - self.rw["smooth_weight"] * penalty_smooth
            - self.rw["height_weight"] * penalty_height
        )

        # (b) Learned-yaw shaping. Penalize crabbing so the 0.28 m box leads with its 0.28 m face,
        # not its 0.40 m diagonal, through the gaps. <= 0 and speed-gated -> no standing income (the
        # loiter optimum stays closed); yaw is decoupled from the goal-frame velocity command, so this
        # shapes ONLY the yaw DOF and cannot move the nav optimum. Added BEFORE the crash/capture
        # overwrites below, so a crashed env is still overwritten to the collision penalty.
        vel_veh = self.obs_dict["robot_vehicle_linvel"]
        speed_xy = vel_veh[:, :2].norm(dim=1)
        cos_crab = vel_veh[:, 0] / speed_xy.clamp(min=1e-6)
        misalign = 0.5 * (1.0 - cos_crab)  # 0 = nose-aligned .. 1 = flying backward
        align_gate = (speed_xy / self.task_config.yaw_align_speed_ref).clamp(0.0, 1.0)
        self.rewards[:] = self.rewards - self.rw["yaw_align_weight"] * misalign * align_gate
        self.rewards[:] = self.rewards - self.rw["yaw_rate_smooth_weight"] * self._yaw_cmd.pow(2)

        # -- vision mode add-ons (before the terminal overwrites, like the other shaping terms)
        if self.vision_mode:
            # The camera is rendered once while building the observation.  Use its most recent
            # pixel-derived visibility here (one control interval old) instead of secretly
            # querying GT geometry a second time for reward shaping.
            self.rewards[:] = self.rewards + float(
                self.vis_cfg.visibility_bonus
            ) * self._visible_now.float()

        crashed = self.obs_dict["crashes"] > 0
        below = z < self.task_config.lower_height_bound
        above = z > self.task_config.upper_height_bound
        crashed_out = crashed | below | above
        target_contact = torch.zeros_like(crashed)
        target_invalid = torch.zeros_like(crashed)
        if self._physical_target:
            target_contact = self._target_controller.contact_seen.clone()
            target_z = self.target_position[:, 2]
            target_support_xyz = self._physical_target_support_xyz()
            target_invalid = (
                (target_z - target_support_xyz[:, 2] < self.task_config.lower_height_bound)
                | (target_z + target_support_xyz[:, 2] > self.task_config.upper_height_bound)
            )
            tb_min = self.obs_dict["env_bounds_min"][:, :2]
            tb_max = self.obs_dict["env_bounds_max"][:, :2]
            target_support_xy = target_support_xyz[:, :2]
            target_invalid |= (
                (self.target_position[:, :2] - target_support_xy < tb_min)
                | (self.target_position[:, :2] + target_support_xy > tb_max)
            ).any(dim=1)
            if self._target_route_recovery_enabled and hasattr(self._target_controller, "watchdog_breach"):
                target_invalid |= self._target_controller.watchdog_breach
            crashed_out |= target_contact | target_invalid
            self.obs_dict["navrl_target_contact"] = target_contact
            self.obs_dict["navrl_target_invalid"] = target_invalid
        oob = torch.zeros_like(crashed)
        if self.vision_mode:
            # out-of-arena termination: with the target unobserved there is no implicit goal
            # attraction keeping the drone in-bounds, and there are no physical walls to crash on.
            m_oob = float(self.vis_cfg.oob_margin)
            b_min = self.obs_dict["env_bounds_min"][:, 0:2]
            b_max = self.obs_dict["env_bounds_max"][:, 0:2]
            below_min = pos[:, 0:2] < b_min - m_oob
            above_max = pos[:, 0:2] > b_max + m_oob
            oob = (below_min | above_max).any(dim=1)
            crashed_out = crashed_out | oob
            if self._oob_probe:
                self._probe_ep_y_min = torch.minimum(self._probe_ep_y_min, pos[:, 1])
                self._probe_ep_y_max = torch.maximum(self._probe_ep_y_max, pos[:, 1])

        # Interception (always on): touching the capture radius ends the episode as a success
        # (terminal bonus instead of continued step reward). Capture wins over a same-step contact.
        # Swept-SEGMENT test in the target-relative frame: sweep prev_rel -> rel against the capture
        # sphere at the origin. This linearizes BOTH agents' motion over the 0.1 s step, so a fast
        # fly-through (closing speed up to 4 m/s = 0.4 m/step) cannot tunnel between samples. With a
        # static target and 0.2 m steps it is a strict superset of the old point test (adds only
        # rare grazing captures; unit-tested in tools/test_navrl_p3_math.py).
        rel = pos - self.target_position
        seg = rel - self.prev_rel
        t_close = (-(self.prev_rel * seg).sum(dim=1) / (seg * seg).sum(dim=1).clamp(min=1e-9)).clamp(0.0, 1.0)
        seg_dist = (self.prev_rel + t_close.unsqueeze(1) * seg).norm(dim=1)
        captured = seg_dist < self.task_config.success_radius
        crashed_out = crashed_out & ~captured
        self.captured_now = captured
        self.crashed_now = crashed_out
        if self._episode_forensics is not None:
            self._episode_forensics.observe_capture(seg_dist, captured)
        # Assign exactly one cause to every crash before the task logs or resets the environment.
        # Priority matches the global crash diagnostics and is exported per evaluation stratum.
        d_contact = (crashed | target_contact | target_invalid) & crashed_out
        d_below = below & ~d_contact & crashed_out
        d_above = above & ~d_contact & ~below & crashed_out
        d_oob = oob & ~d_contact & ~below & ~above & crashed_out
        self._crash_cause_code.fill_(-1)
        self._crash_cause_code[d_contact] = 0
        self._crash_cause_code[d_below] = 1
        self._crash_cause_code[d_above] = 2
        self._crash_cause_code[d_oob] = 3
        if self._episode_dump_path:
            self._record_episode_dump(pos, captured, crashed_out, crashed, below, above)
        if self._obs_dump_enabled:
            self._record_obs_dump_crash_outcomes(
                captured,
                crashed_out,
                d_contact,
                d_oob,
                d_below,
                d_above,
                crashed,
                target_contact,
                target_invalid,
            )

        if self._crash_diag:
            # Attribute each crash to ONE cause, priority contact > below > above > oob (matching
            # the sources OR-ed into crashed_out above). Same-step captures already excluded.
            self._diag["contact"] += int(d_contact.sum().item())
            self._diag["below"] += int(d_below.sum().item())
            self._diag["above"] += int(d_above.sum().item())
            steps = self.sim_env.sim_steps.float()
            if bool(d_contact.any()):
                self._diag_steps["contact"] += float(steps[d_contact].sum().item())
                self._diag_x_sum += float(pos[d_contact, 0].sum().item())
                if self._speed_governor_diag_enabled:
                    gdiag = self._speed_governor_diag
                    last = self._speed_governor_last
                    actual_speed = vel_w[:, 0:2].norm(dim=1)
                    gdiag["contact_n"] += d_contact.sum()
                    gdiag["contact_actual_speed_sum"] += actual_speed[
                        d_contact
                    ].sum(dtype=torch.float64)
                    for source, destination in (
                        ("requested_speed_mps", "contact_requested_speed_sum"),
                        ("executed_speed_mps", "contact_executed_speed_sum"),
                        ("clearance_m", "contact_clearance_sum"),
                        ("scale", "contact_scale_sum"),
                        ("ttc_requested_s", "contact_ttc_sum"),
                        (
                            "stopping_margin_requested_m",
                            "contact_margin_requested_sum",
                        ),
                        (
                            "stopping_margin_executed_m",
                            "contact_margin_executed_sum",
                        ),
                    ):
                        values = last[source][d_contact]
                        if source == "ttc_requested_s":
                            values = values[torch.isfinite(values)]
                        gdiag[destination] += values.sum(dtype=torch.float64)
                if self._bar_probe and self.perception is not None:
                    self._record_bar_contact_probe(d_contact, pos)
                if self._contact_geom_enabled and self.perception is not None:
                    self._record_contact_geometry(d_contact, pos, vel_w)
            if bool(d_below.any()):
                # below-death forensics: WHEN it dies (early sharp-turn transient vs late drift) and
                # HOW TILTED it is at death (tilt-induced thrust sag vs level sink). b3_z from the
                # quaternion directly: R[2][2] = 1 - 2*(qx^2 + qy^2), tilt = acos(b3_z).
                self._diag_steps["below"] += float(steps[d_below].sum().item())
                q = self.obs_dict["robot_orientation"][d_below]
                b3z = (1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)).clamp(-1.0, 1.0)
                self._diag_below_tilt += float(torch.rad2deg(torch.acos(b3z)).sum().item())
            if self.vision_mode:
                if bool(d_oob.any()):
                    self._diag["oob"] += int(d_oob.sum().item())
                    self._diag["oob_w"] += int((d_oob & (pos[:, 0] < b_min[:, 0] - m_oob)).sum().item())
                    self._diag["oob_e"] += int((d_oob & (pos[:, 0] > b_max[:, 0] + m_oob)).sum().item())
                    self._diag["oob_s"] += int((d_oob & (pos[:, 1] < b_min[:, 1] - m_oob)).sum().item())
                    self._diag["oob_n"] += int((d_oob & (pos[:, 1] > b_max[:, 1] + m_oob)).sum().item())
                    self._diag_steps["oob"] += float(steps[d_oob].sum().item())
                    if self._bulk_eval_mode:
                        # Same mask the crash-cause table uses, so `exits` is directly comparable
                        # with crash_causes.out_of_bounds rather than a second, larger population.
                        self._record_oob_exit(d_oob, pos, b_min, b_max, m_oob, steps)
                    if self._oob_probe:
                        north = d_oob & (pos[:, 1] > b_max[:, 1] + m_oob)
                        south = d_oob & (pos[:, 1] < b_min[:, 1] - m_oob)
                        lateral = north | south
                        if bool(lateral.any()):
                            side = torch.where(
                                north, torch.ones_like(pos[:, 1]), -torch.ones_like(pos[:, 1])
                            )
                            arena_mid_y = 0.5 * (b_min[:, 1] + b_max[:, 1])
                            arena_half_y = 0.5 * (b_max[:, 1] - b_min[:, 1]).clamp(min=1e-6)
                            command_world = quat_rotate(
                                self.obs_dict["robot_vehicle_orientation"],
                                self.command[:, 0:3],
                            )
                            excursion = torch.where(
                                north,
                                self._probe_ep_y_max - self._probe_ep_start_y,
                                self._probe_ep_start_y - self._probe_ep_y_min,
                            )
                            p = self._probe
                            p["n"] += float(lateral.sum().item())
                            p["start_y"] += float(self._probe_ep_start_y[lateral].sum().item())
                            p["goal_pull_side"] += float(
                                (
                                    (self._probe_ep_target_start_y - self._probe_ep_start_y)
                                    * side
                                )[lateral].sum().item()
                            )
                            p["goal_now_pull_side"] += float(
                                (
                                    (self.target_position[:, 1] - self._probe_ep_start_y) * side
                                )[lateral].sum().item()
                            )
                            p["bar_bias_side"] += float(
                                (
                                    (self._probe_ep_bar_mean_y - arena_mid_y)
                                    / arena_half_y
                                    * side
                                )[lateral].sum().item()
                            )
                            p["world_vy_side"] += float(
                                (self.obs_dict["robot_linvel"][:, 1] * side)[lateral].sum().item()
                            )
                            p["command_vy_side"] += float(
                                (command_world[:, 1] * side)[lateral].sum().item()
                            )
                            p["action_y_side"] += float(
                                (self.prev_action[:, 1] * side)[lateral].sum().item()
                            )
                            p["excursion_side"] += float(excursion[lateral].sum().item())
                            p["visible"] += float(self._visible_now[lateral].sum().item())
                            if self.perception is not None:
                                tracker = self.perception.tracker
                                p["track_age"] += float(tracker.age[lateral].sum().item())
                                cov_pos = torch.diagonal(
                                    tracker.cov[:, :3, :3], dim1=1, dim2=2
                                ).sum(dim=1)
                                p["track_cov_pos"] += float(cov_pos[lateral].sum().item())

        # A: ego-motion progress shaping -- dense "got closer" signal, RE-ANCHORED to the
        # target's CURRENT position:
        #   progress = ||prev_pos - target_new|| - gamma*||pos_new - target_new||
        # so only the drone's OWN motion is credited.  This future-target re-anchoring is a useful
        # heuristic but is NOT formal potential-based reward shaping for a moving target because
        # the first term is not Phi(s_t).  With a static target it equals the standard expression.
        # Zero the next-state term on TRUE terminals (capture/crash); a timeout is a truncation and
        # rl_games bootstraps it through infos["time_outs"]. Disable with progress_weight = 0.0.
        if self.rw.get("progress_weight", 0.0) != 0.0:
            gamma = self.task_config.progress_gamma
            term = self.captured_now | self.crashed_now
            prev_dist_anchored = (self.target_position - self.prev_pos).norm(dim=1)
            phi_next = torch.where(term, torch.zeros_like(dist), gamma * dist)
            self.rewards[:] = self.rewards + self.rw["progress_weight"] * (
                prev_dist_anchored - phi_next
            )

        # roll the swept-segment / ego-progress buffers forward (reset_idx re-seeds reset envs)
        self.prev_pos[:] = pos
        self.prev_rel[:] = rel

        self.rewards[:] = torch.where(
            crashed_out, torch.full_like(self.rewards, self.rw["collision_penalty"]), self.rewards
        )
        self.rewards[:] = torch.where(
            captured, self.rewards + self.rw["capture_bonus"], self.rewards
        )
        self.terminations[:] = (crashed_out | captured).to(self.terminations.dtype)

    def add_static_safety_reward(self):
        # NavRL r_ss = mean over rays of log(distance to obstacle), clamped to (0, range].
        dist_m = self._lidar_distance_m().clamp(min=1e-6, max=self.task_config.lidar_max_range)
        if self.vision_mode:
            # Target returns are valid perception but are not collision obstacles for r_ss.
            seg = self.obs_dict["segmentation_pixels"].squeeze(1).reshape(self.num_envs, -1)
            dist_m = torch.where(
                seg == 50, torch.full_like(dist_m, self.task_config.lidar_max_range), dist_m
            )
        # B1: re-baseline so OPEN SPACE (all rays at max range) scores 0 instead of +log(range).
        # This subtracts a constant log(range) per step, so the obstacle-avoidance GRADIENT is
        # byte-for-byte unchanged, but it deletes the standing "loiter income" (~+log(4)=+1.39/step
        # in the open) that made hovering short optimal (V_loiter >> V_capture -> ~7% capture).
        r_ss = (torch.log(dist_m) - math.log(self.task_config.lidar_max_range)).mean(dim=1)
        # Do not reward-shape envs that just FINISHED (crash reward is the collision penalty;
        # capture reward is the terminal bonus; TRUNCATED envs were already reset before this
        # render, so their scan belongs to the NEXT episode's spawn — shaping them would leak
        # next-episode state into this step's reward).
        alive = (self.terminations <= 0) & (self.truncations <= 0)
        self.rewards[alive] += self.rw["safety_static_weight"] * r_ss[alive]
        # (The B/C/D near-obstacle clearance penalty that used to live here was removed: three
        #  null results — crash proved geometric, not reward-driven. See CRASH_TUNING_LOG.md;
        #  code in git history @44e86a2.)

    # ------------------------------------------------------------------ obs
    def get_return_tuple(self):
        self.process_obs_for_task()
        return (self.task_obs, self.rewards, self.terminations, self.truncations, self.infos)

    def process_obs_for_task(self):
        if self.perception_mode:
            self._process_obs_perception()
            return
        if self.vision_mode:
            self._process_obs_vision()
            return
        pos = self.obs_dict["robot_position"]
        vel_w = self.obs_dict["robot_linvel"]
        rpos = self.target_position - pos
        dist = rpos.norm(dim=1, keepdim=True).clamp(min=1e-6)
        dist_2d = rpos[:, :2].norm(dim=1, keepdim=True)
        dist_z = rpos[:, 2:3]

        rpos_unit_g = vec_to_goal_frame(rpos / dist, self.target_dir_2d)
        vel_g = vec_to_goal_frame(vel_w, self.target_dir_2d)
        # (b) heading info in the vehicle (yaw-only) frame so the agent can OBSERVE its crab angle
        # (vel_body_xy: lateral slip = which way to yaw) and where the goal is relative to its NOSE
        # (goal_bearing_body, always defined even at rest), re-anchoring the body-frame LiDAR now that
        # yaw is free. robot_vehicle_orientation is yaw-only; robot_vehicle_linvel is precomputed.
        q_veh = self.obs_dict["robot_vehicle_orientation"]
        goal_veh = quat_rotate_inverse(q_veh, rpos)
        goal_bearing_body = goal_veh[:, :2] / goal_veh[:, :2].norm(dim=1, keepdim=True).clamp(min=1e-6)
        vel_body_xy = self.obs_dict["robot_vehicle_linvel"][:, :2] / self.task_config.max_velocity

        s_int = torch.cat(
            [rpos_unit_g, dist_2d, dist_z, vel_g, goal_bearing_body, vel_body_xy], dim=1
        )  # (N, 12)

        # LiDAR scan, normalized [0,1] (1 = no obstacle within range), flattened to 144
        lidar = self._lidar_distance_m() / self.task_config.lidar_max_range

        self.task_obs["observations"][:, : self.task_config.internal_state_dim] = s_int
        self.task_obs["observations"][:, self.task_config.internal_state_dim :] = lidar

    def _process_obs_vision(self):
        """Sensor-only ACTOR observation + privileged CRITIC 'states' (vision mode).

        actor (1265) = ego 9 [vel_vehicle/vmax(3), yaw_rate/max(1), prev_action(4), height(1)]
                    + detector 8 [visible, bearing sin/cos, elev, range | tracker 3]
                    + LiDAR range (144) + LiDAR target mask (144)
                    + forward camera obstacle depth (40x24=960)
        states (1273) = actor obs + GT extras [rpos_unit_veh(3), dist/24(1), tvel_veh/2(3),
                       closing/(vmax+2)(1)] — read ONLY by the central-value critic at train time.

        No ground-truth target quantity enters the actor slice: the target appears only through
        LiDAR returns and the FOV/occlusion-gated camera detector. Both LiDAR and camera also
        observe obstacle geometry."""
        pos = self.obs_dict["robot_position"]
        q_veh = self.obs_dict["robot_vehicle_orientation"]
        rpos_w = self.target_position - pos
        rpos_veh = quat_rotate_inverse(q_veh, rpos_w)

        det_vec, visible = self.detector.detect(
            pos,
            q_veh,
            self.target_position,
            self.target_orientation,
            update_tracker=True,
        )
        self._visible_now[:] = visible
        # Expose raw target-camera products for evaluation/debugging. Target mask/depth are reduced
        # to det_vec; the separate obstacle depth image is concatenated into the actor below.
        self.obs_dict["target_camera_mask"] = self.detector.target_mask
        self.obs_dict["target_camera_depth"] = self.detector.target_depth
        self.obs_dict["obstacle_camera_depth"] = self.detector.obstacle_depth

        vel_veh = self.obs_dict["robot_vehicle_linvel"] / self.task_config.max_velocity
        yaw_rate = self.obs_dict["robot_body_angvel"][:, 2:3] / self.task_config.yaw_rate_max
        height = pos[:, 2:3] / 3.0
        ego = torch.cat([vel_veh, yaw_rate, self.prev_action, height], dim=1)  # (N, 9)

        lidar = self._lidar_distance_m() / self.task_config.lidar_max_range  # (N, 144)
        seg = self.obs_dict["segmentation_pixels"].squeeze(1).reshape(self.num_envs, -1)
        lidar_target = (seg == 50).float()
        camera_obstacle = torch.nan_to_num(
            self.detector.obstacle_depth,
            nan=self.vis_cfg.camera_obstacle_max_range,
            posinf=self.vis_cfg.camera_obstacle_max_range,
            neginf=self.vis_cfg.camera_obstacle_max_range,
        ).clamp(0.0, self.vis_cfg.camera_obstacle_max_range)
        camera_obstacle = (
            camera_obstacle / self.vis_cfg.camera_obstacle_max_range
        ).reshape(self.num_envs, -1)

        obs = self.task_obs["observations"]
        d0 = self.vis_cfg.ego_dim
        d1 = d0 + self.vis_cfg.detector_dim
        d2 = d1 + lidar.shape[1]
        d3 = d2 + lidar_target.shape[1]
        obs[:, :d0] = ego
        obs[:, d0:d1] = det_vec
        obs[:, d1:d2] = lidar
        obs[:, d2:d3] = lidar_target
        if not bool(getattr(self.vis_cfg, "legacy_actor_305", False)):
            obs[:, d3:] = camera_obstacle

        # privileged critic extras (train-time only; the player ignores 'states')
        dist = rpos_w.norm(dim=1, keepdim=True).clamp(min=1e-6)
        tvel_veh = quat_rotate_inverse(q_veh, self.target_vel_w)
        closing = ((self.obs_dict["robot_linvel"] - self.target_vel_w) * (rpos_w / dist)).sum(
            dim=1, keepdim=True
        )
        states = self.task_obs["states"]
        states[:, : obs.shape[1]] = obs
        states[:, obs.shape[1] :] = torch.cat(
            [
                rpos_veh / dist,
                dist / self._arena_xy_norm,
                tvel_veh / 2.0,
                closing / (self.task_config.max_velocity + 2.0),
            ],
            dim=1,
        )

    def _process_obs_perception(self):
        """Raw RGB-D/LiDAR -> perception tracks -> actor-safe structured history.

        ``target_position`` is passed only to the simulator-side renderer, exactly as scene pose is
        used by a physical renderer. The perception module API has no target/semantic argument;
        its output is therefore structurally unable to read the oracle target state.
        """
        pos = self.obs_dict["robot_position"]
        vel_w = self.obs_dict["robot_linvel"]
        q_veh = self.obs_dict["robot_vehicle_orientation"]

        raw_rgb, raw_depth = self.detector.render_raw_rgbd(
            pos, q_veh, self.target_position, self.target_orientation
        )
        lidar_m = self._lidar_distance_m()
        structured, diagnostics = self.perception.observe(
            rgb=raw_rgb,
            depth=raw_depth,
            lidar_m=lidar_m,
            drone_pos_w=pos,
            drone_vel_w=vel_w,
            vehicle_quat=q_veh,
            yaw_rate=self.obs_dict["robot_body_angvel"][:, 2]
            / self.task_config.yaw_rate_max,
            previous_action=self.prev_action,
            max_velocity=self.task_config.max_velocity,
            flight_altitude=self.task_config.flight_altitude,
            training=bool(self.perception_cfg.enable_perturbations),
            env_bounds_min=self.obs_dict["env_bounds_min"],
            env_bounds_max=self.obs_dict["env_bounds_max"],
        )
        self.task_obs["observations"][:] = structured
        self._visible_now[:] = diagnostics["visible"]
        self._camera_visible_now[:] = diagnostics.get(
            "camera_visible", diagnostics["visible"]
        )
        if self._episode_forensics is not None:
            # Read-only: the same evaluator-facing dict the distractor and S1 recorders consume.
            self._episode_forensics.observe_perception(diagnostics)

        # Evaluation-only, and only when the scene actually contains distractors.  Placed AFTER the
        # observation has already been written above, so it cannot participate in building it.
        if self._distractor_metric_enabled:
            self._record_distractor_lock_frame(diagnostics)
        if self._s1_shadow_enabled:
            self._record_s1_shadow_frame(diagnostics)

        # Raw sensor and tracker diagnostics are available to evaluators, never concatenated into
        # actor observations. Semantic renderer buffers intentionally remain private.
        self.obs_dict["navrl_raw_rgb"] = raw_rgb
        self.obs_dict["navrl_raw_depth"] = raw_depth
        self.obs_dict["navrl_track_confidence"] = diagnostics["confidence"]
        self.obs_dict["navrl_track_age"] = diagnostics["track_age"]
        self.obs_dict["navrl_track_covariance"] = diagnostics["track_covariance"]
        for key in (
            "search_mode",
            "blind_steps",
            "blind_steps_before_first_visible",
            "coverage_fraction",
            "belief_entropy",
            "coverage_fraction_at_first_visible",
            "belief_entropy_at_first_visible",
            "search_state_masked",
        ):
            if key in diagnostics:
                self.obs_dict["navrl_" + key] = diagnostics[key]

        # Asymmetric critic: oracle quantities are appended only to the physically separate
        # states buffer. rl_games' player drops this entire tensor at deployment.
        rpos_w = self.target_position - pos
        dist = rpos_w.norm(dim=1, keepdim=True).clamp(min=1e-6)
        rpos_veh = quat_rotate_inverse(q_veh, rpos_w)
        tvel_veh = quat_rotate_inverse(q_veh, self.target_vel_w)
        closing = ((vel_w - self.target_vel_w) * (rpos_w / dist)).sum(
            dim=1, keepdim=True
        )
        obs = self.task_obs["observations"]
        states = self.task_obs["states"]
        states[:, : obs.shape[1]] = obs
        states[:, obs.shape[1] :] = torch.cat(
            [
                rpos_veh / dist,
                dist / self._arena_xy_norm,
                tvel_veh / 2.0,
                closing / (self.task_config.max_velocity + 2.0),
            ],
            dim=1,
        )

    def _record_distractor_lock_frame(self, diagnostics):
        """Classify this frame's detector measurement: TARGET / DISTRACTOR / GHOST lock.

        docs/prereg_2026-09-01_distractor_envelope.md section 4.  Called once per observation, only
        while ``_distractor_metric_enabled`` (bulk evaluation AND at least one distractor).

        WHAT IS CLASSIFIED.  ``camera_measurement_world`` is the position ``_detect_rgbd`` produced
        and handed to the tracker as its observation -- the detector's own answer, before any
        filtering.  The visibility gate is the CAMERA flag, not the fused one: a frame the camera
        lost and LiDAR is holding has no camera measurement to classify, and counting it would put
        a stale value in the numerator of a rate about the camera detector.

        WHAT GROUND TRUTH IS USED FOR.  Labelling, and nothing else.  ``self.target_position`` and
        the distractor centres are read here to decide which bucket a frame falls in; no tensor
        computed in this method reaches ``task_obs``, the reward, a termination, the curriculum, or
        get_env_state.  This is the same evaluation-only use of ground truth the first-acquisition
        telemetry and the OOB exit forensics already make.

        CATEGORY PRECEDENCE.  The preregistration lists TARGET_LOCK first, so a measurement inside
        both radii is a TARGET_LOCK.  Placement clearance makes that overlap unlikely, so instead of
        assuming it away the count is accumulated and exported (``ambiguous``): a reader can then
        see whether precedence changed any number at all.
        """
        visible = diagnostics["camera_visible"]
        measurement = diagnostics["camera_measurement_world"]
        distractors = self.obs_dict["obstacle_position"][
            :, self._solid_obstacle_offset : self._bar_offset, 0:3
        ]

        target_distance = (measurement - self.target_position).norm(dim=1)
        # (num_envs, num_distractors) -> nearest distractor centre per env.
        distractor_distance = (
            (measurement.unsqueeze(1) - distractors).norm(dim=2).min(dim=1).values
        )
        near_target = target_distance <= DISTRACTOR_LOCK_RADIUS_M
        near_distractor = distractor_distance <= DISTRACTOR_LOCK_RADIUS_M

        target_lock = visible & near_target
        distractor_lock = visible & near_distractor & ~near_target
        ghost_lock = visible & ~near_target & ~near_distractor
        ambiguous = visible & near_target & near_distractor

        self._dl_counts += torch.stack(
            [
                torch.full_like(self._dl_counts[0], int(self.num_envs)),
                visible.sum(),
                target_lock.sum(),
                distractor_lock.sum(),
                ghost_lock.sum(),
                ambiguous.sum(),
            ]
        )
        visible_f = visible.to(torch.float64)
        self._dl_sums += torch.stack(
            [
                (diagnostics["target_pixels"].to(torch.float64) * visible_f).sum(),
                (diagnostics["camera_confidence"].to(torch.float64) * visible_f).sum(),
                (target_distance.to(torch.float64) * visible_f).sum(),
                (distractor_distance.to(torch.float64) * visible_f).sum(),
            ]
        )

    def _distractor_lock_payload(self):
        """The section-4 measurement block, validated by the fail-closed export guard first."""
        counts = [int(v) for v in self._dl_counts.tolist()]
        sums = [float(v) for v in self._dl_sums.tolist()]
        frames_total, visible_frames, target_lock, distractor_lock, ghost_lock, ambiguous = counts
        _validate_distractor_lock_export(
            self._num_distractors,
            frames_total,
            visible_frames,
            target_lock,
            distractor_lock,
            ghost_lock,
            ambiguous,
            DISTRACTOR_LOCK_RADIUS_M,
        )

        def rate(count):
            # None, not 0.0, when nothing was visible: "0% of no frames" is not a measurement, and
            # a 0.0 here would read as a detector that never once misattributed.
            return (count / visible_frames) if visible_frames > 0 else None

        def mean(total):
            return (total / visible_frames) if visible_frames > 0 else None

        return {
            "schema_version": 1,
            "distractor_count": int(self._num_distractors),
            "classification_radius_m": float(DISTRACTOR_LOCK_RADIUS_M),
            "visible_source": "camera_visible",
            "measurement_source": "perception.observe -> camera_measurement_world (the tracker's "
            "own input, pre-filter)",
            "frames_total": frames_total,
            "visible_frames": visible_frames,
            "visible_frame_rate": visible_frames / frames_total,
            "target_lock": target_lock,
            "distractor_lock": distractor_lock,
            "ghost_lock": ghost_lock,
            "ambiguous_target_and_distractor": ambiguous,
            "target_lock_rate": rate(target_lock),
            "distractor_lock_rate": rate(distractor_lock),
            "ghost_lock_rate": rate(ghost_lock),
            # The preregistered primary: (DISTRACTOR_LOCK + GHOST_LOCK) / visible frames.
            "false_target_lock_rate": rate(distractor_lock + ghost_lock),
            "camera_pixel_count_mean": mean(sums[0]),
            "camera_confidence_mean": mean(sums[1]),
            "measurement_to_target_mean_m": mean(sums[2]),
            "measurement_to_nearest_distractor_mean_m": mean(sums[3]),
        }

    def _record_s1_shadow_frame(self, diagnostics):
        """S1 shadow counterfactual, classified with the SAME rule as the live lock recorder.

        Identical GT tensors and radius as _record_distractor_lock_frame; only the measurement
        differs (the shadow candidate pipeline's pick instead of the live single centroid).
        Additionally splits locks by INIT frames -- the phase where the shadow tracker had no
        active track and seeded from pixel count alone (prereg section 4, prediction 2).
        """
        visible = diagnostics["s1_shadow_visible"]
        measurement = diagnostics["s1_shadow_measurement_world"]
        initialized = diagnostics["s1_shadow_initialized"]

        target_distance = (measurement - self.target_position).norm(dim=1)
        if self._num_distractors > 0:
            distractors = self.obs_dict["obstacle_position"][
                :, self._solid_obstacle_offset : self._bar_offset, 0:3
            ]
            distractor_distance = (
                (measurement.unsqueeze(1) - distractors).norm(dim=2).min(dim=1).values
            )
        else:
            distractor_distance = torch.full_like(target_distance, float("inf"))
        near_target = target_distance <= DISTRACTOR_LOCK_RADIUS_M
        near_distractor = distractor_distance <= DISTRACTOR_LOCK_RADIUS_M

        target_lock = visible & near_target
        distractor_lock = visible & near_distractor & ~near_target
        ghost_lock = visible & ~near_target & ~near_distractor
        ambiguous = visible & near_target & near_distractor
        init = visible & initialized

        self._s1_counts += torch.stack(
            [
                torch.full_like(self._s1_counts[0], int(self.num_envs)),
                visible.sum(),
                target_lock.sum(),
                distractor_lock.sum(),
                ghost_lock.sum(),
                ambiguous.sum(),
                init.sum(),
                (target_lock & init).sum(),
                (distractor_lock & init).sum(),
                (ghost_lock & init).sum(),
            ]
        )

    def _push_contact_geometry_history(self, command_xy):
        """Ring-buffer the drone's own pose/command/velocity. Bars are static, so replaying a past
        pose against the CURRENT bar positions reconstructs the past geometry exactly."""
        i = self._cg_hist_cursor
        self._cg_hist_pos[i] = self.obs_dict["robot_position"].detach()
        self._cg_hist_quat[i] = self.obs_dict["robot_vehicle_orientation"].detach()
        self._cg_hist_cmd[i] = command_xy
        self._cg_hist_vel[i] = self.obs_dict["robot_linvel"].detach()
        self._cg_hist_yaw[i] = self.obs_dict["robot_body_angvel"][:, 2].detach()
        self._cg_hist_cursor = (i + 1) % self._cg_hist_pos.shape[0]
        self._cg_hist_age += 1

    def _push_contact_geometry_governor(self, telemetry):
        """Fill the governor columns of the slot _push_contact_geometry_history just wrote."""
        i = (self._cg_hist_cursor - 1) % self._cg_hist_pos.shape[0]
        self._cg_hist_gov[i, :, 0] = telemetry["requested_speed_mps"].detach()
        self._cg_hist_gov[i, :, 1] = telemetry["executed_speed_mps"].detach()
        self._cg_hist_gov[i, :, 2] = telemetry["speed_cap_mps"].detach()

    def _record_contact_geometry_step(self, command_xy, clearance):
        """A2 descriptive metrics, accumulated every governed step. Evaluation-only."""
        # Corridor clearance distribution over the whole trajectory, not just at contact.
        rng = float(self.task_config.lidar_max_range)
        bins = (clearance.clamp(0.0, rng) / 0.5).long().clamp(0, self._cg_clearance_hist.numel() - 1)
        self._cg_clearance_hist += torch.bincount(bins, minlength=self._cg_clearance_hist.numel())

        # Path length: integrate |dp| between governed steps, and bank it on reset.
        pos = self.obs_dict["robot_position"].detach()
        if self._cg_prev_pos is None:
            self._cg_prev_pos = pos.clone()
        else:
            # Envs reset since the last governed step have had their prev_pos invalidated by
            # _bank_contact_geometry_path; skipping them keeps the respawn teleport out of the
            # integral. Everything else accumulates normally.
            step = (pos - self._cg_prev_pos).norm(dim=1)
            self._cg_path_len += torch.where(
                self._cg_path_valid, step, torch.zeros_like(step)
            )
            self._cg_prev_pos = pos.clone()
            self._cg_path_valid[:] = True

        # Commanded-vs-actual angle over ALL frames -- prediction 6 needs this baseline.
        quat = self.obs_dict["robot_vehicle_orientation"].detach()
        vel_v = quat_rotate_inverse(quat, self.obs_dict["robot_linvel"].detach())[:, 0:2]
        moving = (command_xy.norm(dim=1) > 1e-6) & (vel_v.norm(dim=1) > 1e-6)
        if bool(moving.any()):
            cos = torch.nn.functional.cosine_similarity(
                command_xy[moving], vel_v[moving], dim=1
            ).clamp(-1.0, 1.0)
            self._cg_aux[2] += torch.rad2deg(torch.acos(cos)).sum(dtype=torch.float64)
            self._cg_aux[3] += int(moving.sum())

        # I2: a sparse sample of NON-contact frames with the same geometry columns, so a
        # per-contact distribution has a baseline to be a risk ratio against.
        self._cg_step_counter += 1
        if self._cg_step_counter % self._cg_frame_every == 0:
            self._record_contact_frame_sample(command_xy, clearance, quat, vel_v)

    def _record_contact_frame_sample(self, command_xy, clearance, quat, vel_v):
        from aerial_gym.task.navrl_task import contact_records as CR

        n = self.num_envs
        rng = float(self.task_config.lidar_max_range)
        pos = self.obs_dict["robot_position"].detach()
        bars_w = self.obs_dict["obstacle_position"][
            :, self._bar_offset : self._bar_offset + self.n_bars_active, 0:3
        ]
        half = self.obs_dict["asset_collision_half_extents"][
            :, self._bar_offset : self._bar_offset + self.n_bars_active, 0:2
        ]
        rel_w = bars_w - pos.unsqueeze(1)
        b = rel_w.shape[1]
        rel_v = quat_rotate_inverse(
            quat.unsqueeze(1).expand(n, b, 4).reshape(n * b, 4), rel_w.reshape(n * b, 3)
        ).reshape(n, b, 3)
        bar_dist = rel_v[:, :, 0:2].norm(dim=2)
        bar_bearing = torch.atan2(rel_v[:, :, 1], rel_v[:, :, 0])
        left, right, nearest = CR.side_gaps(bar_dist, bar_bearing, half.norm(dim=2), rng)
        cmd_bearing = torch.atan2(command_xy[:, 1], command_xy[:, 0]).view(-1, 1)
        dall = torch.atan2(torch.sin(bar_bearing - cmd_bearing), torch.cos(bar_bearing - cmd_bearing))
        in_corr = (bar_dist * torch.cos(dall) > 0.0) & ((bar_dist * torch.sin(dall)).abs() <= float(
            self.speed_governor_cfg.path_half_width_m)) & (bar_dist < rng)
        moving = (command_xy.norm(dim=1) > 1e-6) & (vel_v.norm(dim=1) > 1e-6)
        cos = torch.nn.functional.cosine_similarity(command_xy, vel_v, dim=1).clamp(-1.0, 1.0)
        gov = self._speed_governor_last
        self._cg_frame_rows.append(CR.tensor_columns_to_rows({
            "env": torch.arange(n, device=self.device),
            "age_steps": self._cg_hist_age.clone(),
            "speed_act": vel_v.norm(dim=1),
            "speed_cmd": command_xy.norm(dim=1),
            "cmd_vs_actual_deg": torch.where(moving, torch.rad2deg(torch.acos(cos)), torch.zeros_like(cos)),
            "yaw_rate": self.obs_dict["robot_body_angvel"][:, 2].detach(),
            "corridor_clearance": clearance,
            "gap_left": left, "gap_right": right, "nearest_surface": nearest,
            "bars_in_corridor": in_corr.sum(dim=1),
            # previous governed step's telemetry (this step's has not run yet)
            "prev_requested": gov["requested_speed_mps"].clone(),
            "prev_executed": gov["executed_speed_mps"].clone(),
            "prev_cap": gov["speed_cap_mps"].clone(),
        }))

    def _bank_contact_geometry_path(self, env_ids):
        """Episode ended: bank its path length and restart the integrator for those envs.

        Only episodes that actually ran are banked. The very first reset touches all envs with a
        zero-length integral, and counting those would drag the mean toward zero.
        """
        done = self._cg_path_len[env_ids]
        ran = done > 1e-6
        if bool(ran.any()):
            self._cg_aux[0] += done[ran].sum(dtype=torch.float64)
            self._cg_aux[1] += int(ran.sum())
        self._cg_path_len[env_ids] = 0.0
        if self._cg_prev_pos is not None:
            # The respawn teleport must not enter the next episode's integral.
            self._cg_path_valid[env_ids] = False

    def _record_contact_geometry(self, hit_mask, pos, vel_w):
        """Decompose each bar contact against the corridor AS IT WAS ~1 s before impact.

        Preregistered: docs/prereg_2026-09-04_contact_corridor_forensics.md. Categories, their
        priority and the lookback are frozen there BEFORE measurement. Ground-truth bar positions
        are used, evaluation-only, on the same basis as _record_bar_contact_probe: nothing computed
        here reaches the actor, the critic, the reward, or any termination.

        Measuring at the contact INSTANT measures the aftermath -- by then the vehicle is alongside
        the bar (~0.5 m centre-to-centre) and every bar reads as abeam. The primary lookback is the
        moment the governor could still have acted.
        """
        lookback = self._CG_LOOKBACK_STEPS[0]
        depth = self._cg_hist_pos.shape[0]
        idx = hit_mask.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return
        k = idx.numel()
        self._cg_counts[0] += k

        # Only contacts whose episode is already `lookback` steps old have a valid past pose.
        usable = self._cg_hist_age[idx] > lookback
        idx = idx[usable]
        if idx.numel() == 0:
            return
        k = idx.numel()
        self._cg_counts[1] += k

        past = (self._cg_hist_cursor - 1 - lookback) % depth
        p_pos = self._cg_hist_pos[past][idx]
        p_quat = self._cg_hist_quat[past][idx]
        p_cmd = self._cg_hist_cmd[past][idx]
        p_vel = self._cg_hist_vel[past][idx]

        half_w = float(self.speed_governor_cfg.path_half_width_m)
        rng = float(self.task_config.lidar_max_range)

        bars_w = self.obs_dict["obstacle_position"][idx][
            :, self._bar_offset : self._bar_offset + self.n_bars_active, 0:3
        ]
        rel_w = bars_w - p_pos.unsqueeze(1)
        b = rel_w.shape[1]
        rel_v = quat_rotate_inverse(
            p_quat.unsqueeze(1).expand(k, b, 4).reshape(k * b, 4), rel_w.reshape(k * b, 3)
        ).reshape(k, b, 3)
        bar_dist = rel_v[:, :, 0:2].norm(dim=2)
        bar_bearing = torch.atan2(rel_v[:, :, 1], rel_v[:, :, 0])

        # The struck bar is the one nearest AT CONTACT; find it with the contact-time pose, then
        # read its geometry from the past frame. Identity, not proximity, is what carries over.
        rel_now = self.obs_dict["obstacle_position"][idx][
            :, self._bar_offset : self._bar_offset + self.n_bars_active, 0:2
        ] - pos[idx, 0:2].unsqueeze(1)
        hit_i = rel_now.norm(dim=2).argmin(dim=1)
        rows = torch.arange(k, device=self.device)
        hit_d = bar_dist[rows, hit_i]
        hit_bearing = bar_bearing[rows, hit_i]
        hit_dz = rel_v[rows, hit_i, 2]

        def corridor_terms(ref_xy):
            ref = torch.atan2(ref_xy[:, 1], ref_xy[:, 0])
            d = torch.atan2(torch.sin(hit_bearing - ref), torch.cos(hit_bearing - ref))
            return hit_d * torch.cos(d), (hit_d * torch.sin(d)).abs()

        p_vel_v = quat_rotate_inverse(p_quat, p_vel)[:, 0:2]
        fwd_c, lat_c = corridor_terms(p_cmd)
        fwd_a, lat_a = corridor_terms(p_vel_v)

        moving = (p_cmd.norm(dim=1) > 1e-6) & (p_vel_v.norm(dim=1) > 1e-6)
        cos = torch.nn.functional.cosine_similarity(p_cmd, p_vel_v, dim=1).clamp(-1.0, 1.0)
        cmd_vs_actual = torch.where(moving, torch.rad2deg(torch.acos(cos)), torch.zeros_like(cos))

        # No-return: the live governor treats an unreturned ray as free space.
        scan = getattr(self.perception, "last_scan_nearest", None)
        bearings = self._speed_governor_bearings
        if scan is not None and bearings is not None:
            dbin = torch.atan2(
                torch.sin(bearings.view(1, -1) - hit_bearing.view(-1, 1)),
                torch.cos(bearings.view(1, -1) - hit_bearing.view(-1, 1)),
            ).abs()
            ray_returned = scan[idx][rows, dbin.argmin(dim=1)] < rng * 0.995
        else:
            ray_returned = torch.ones(k, dtype=torch.bool, device=self.device)

        top = torch.atan2(hit_dz + 1.0, hit_d.clamp(min=1e-3))
        bot = torch.atan2(hit_dz - 1.0, hit_d.clamp(min=1e-3))
        vertical_out = (bot > math.radians(20.0)) | (top < math.radians(-10.0))

        def classify(fwd, lat):
            behind = ~vertical_out & (fwd <= 0.0)
            lateral = ~vertical_out & ~behind & (lat > half_w)
            no_ret = ~vertical_out & ~behind & ~lateral & ~ray_returned
            return behind, lateral, no_ret, ~vertical_out & ~behind & ~lateral & ~no_ret

        beh_c, lat_cat_c, nor_c, in_c = classify(fwd_c, lat_c)
        beh_a, lat_cat_a, _, in_a = classify(fwd_a, lat_a)

        # Hypothesis D: a per-obstacle margin cannot certify a SET (Fraichard & Asama, Property 4).
        cmd_bearing = torch.atan2(p_cmd[:, 1], p_cmd[:, 0]).view(-1, 1)
        dall = torch.atan2(
            torch.sin(bar_bearing - cmd_bearing), torch.cos(bar_bearing - cmd_bearing)
        )
        f_all, l_all = bar_dist * torch.cos(dall), (bar_dist * torch.sin(dall)).abs()
        in_corr_all = (f_all > 0.0) & (l_all <= half_w) & (bar_dist < rng)
        bars_in_corridor = in_corr_all.sum(dim=1)
        arc_clear = torch.where(in_corr_all, bar_dist, torch.full_like(bar_dist, rng)).amin(dim=1)

        if self._cg_sc_enabled:
            self._record_star_convex_shadow(idx, hit_bearing, hit_d, lat_cat_c, nor_c, rng)

        target_involved = (self.target_position[idx] - pos[idx]).norm(dim=1) <= DISTRACTOR_LOCK_RADIUS_M

        self._cg_counts[2:] += torch.stack([
            vertical_out.sum(), beh_c.sum(), lat_cat_c.sum(), nor_c.sum(), in_c.sum(),
            beh_a.sum(), lat_cat_a.sum(), in_a.sum(),
            (bars_in_corridor >= 2).sum(), target_involved.sum(),
        ])
        self._cg_sums += torch.stack([
            cmd_vs_actual.sum(dtype=torch.float64),
            bars_in_corridor.sum(dtype=torch.float64),
            lat_c.sum(dtype=torch.float64),
            arc_clear.sum(dtype=torch.float64),
        ])

        # ---- I1: one row per contact (plan section 4, series I) ----
        from aerial_gym.task.navrl_task import contact_records as CR

        half = self.obs_dict["asset_collision_half_extents"][idx][
            :, self._bar_offset : self._bar_offset + self.n_bars_active, 0:2
        ]
        circ = half.norm(dim=2)
        gap_left, gap_right, nearest_surface = CR.side_gaps(bar_dist, bar_bearing, circ, rng)
        hit_circ = circ[rows, hit_i]

        # 0.5 s secondary lookback: speed and yaw a half second before contact.
        lb2 = self._CG_LOOKBACK_STEPS[1]
        past2 = (self._cg_hist_cursor - 1 - lb2) % depth
        vel2_v = quat_rotate_inverse(self._cg_hist_quat[past2][idx], self._cg_hist_vel[past2][idx])[:, 0:2]
        gov1 = self._cg_hist_gov[past][idx]
        yaw1 = self._cg_hist_yaw[past][idx]
        yaw2 = self._cg_hist_yaw[past2][idx]

        # H3 memory window: was the struck bar observable (in LiDAR range AND inside the
        # selector FOV) at each step of [t-2.5 s, t-1.0 s]? Static bars + past poses give this
        # exactly. Rows whose episode is younger than the window get null memory fields.
        from aerial_gym.task.navrl_task.navrl_perception import OBSTACLE_FOV_DEG

        fov_half = math.radians(0.5 * float(OBSTACLE_FOV_DEG))   # launchers pin 240 deg
        mem_ok = self._cg_hist_age[idx] > self._CG_MEMORY_STEPS
        hit_w = bars_w[rows, hit_i]                                   # world xyz of the struck bar
        seen = []
        for lb in range(lookback, self._CG_MEMORY_STEPS + 1):
            pk = (self._cg_hist_cursor - 1 - lb) % depth
            rel = quat_rotate_inverse(self._cg_hist_quat[pk][idx], hit_w - self._cg_hist_pos[pk][idx])
            d = rel[:, 0:2].norm(dim=1)
            brg = torch.atan2(rel[:, 1], rel[:, 0])
            seen.append((d < rng) & (brg.abs() <= fov_half))
        seen = torch.stack(seen, dim=1)                                # [k, W], column 0 == t-1.0 s
        in_fov_t1 = seen[:, 0]
        seen_frac, seen_then_lost = CR.memory_window(seen[:, 1:], in_fov_t1)
        nan = torch.full((k,), float("nan"), device=self.device)

        # I4 (verification follow-up, plan 2026-09-06): the gap columns above are at t-1.0 s. The
        # objection that a wide gap a second earlier says nothing about the contact instant is fair,
        # so the same side gaps are recorded at t-0.5 s and AT contact (current pose), with a
        # two-sided pinch flag: a bar surface inside the inflation radius on both sides at contact.
        def side_gaps_at(pose_pos, pose_quat):
            rel = quat_rotate_inverse(
                pose_quat.unsqueeze(1).expand(k, b, 4).reshape(k * b, 4),
                (bars_w - pose_pos.unsqueeze(1)).reshape(k * b, 3),
            ).reshape(k, b, 3)
            dist = rel[:, :, 0:2].norm(dim=2)
            return CR.side_gaps(dist, torch.atan2(rel[:, :, 1], rel[:, :, 0]), circ, rng)

        left_t0, right_t0, nearest_t0 = side_gaps_at(
            pos[idx], self.obs_dict["robot_vehicle_orientation"][idx]
        )
        left_t05, right_t05, _ = side_gaps_at(self._cg_hist_pos[past2][idx], self._cg_hist_quat[past2][idx])
        pinch_t0 = (left_t0 < CR.PINCH_M) & (right_t0 < CR.PINCH_M)

        self._cg_contact_rows.append(CR.tensor_columns_to_rows({
            "env": idx,
            "age_steps": self._cg_hist_age[idx].clone(),
            "category_cmd": CR.category_index(vertical_out, beh_c, lat_cat_c, nor_c),
            "category_act": CR.category_index(vertical_out, beh_a, lat_cat_a, torch.zeros_like(nor_c)),
            "hit_bearing_deg": torch.rad2deg(hit_bearing),
            "hit_forward_cmd": fwd_c, "hit_lateral_cmd": lat_c,
            "hit_forward_act": fwd_a, "hit_lateral_act": lat_a,
            "hit_dist": hit_d, "hit_surface": (hit_d - hit_circ).clamp(min=0.0), "hit_dz": hit_dz,
            "speed_act_t1": p_vel_v.norm(dim=1), "speed_cmd_t1": p_cmd.norm(dim=1),
            "speed_act_t05": vel2_v.norm(dim=1),
            "cmd_vs_actual_deg": cmd_vs_actual,
            "yaw_rate_t1": yaw1, "yaw_rate_t05": yaw2,
            "gov_requested_t1": gov1[:, 0], "gov_executed_t1": gov1[:, 1], "gov_cap_t1": gov1[:, 2],
            "cap_binding_t1": gov1[:, 1] < gov1[:, 0] - 1e-3,
            "gap_left": gap_left, "gap_right": gap_right, "nearest_surface": nearest_surface,
            "gap_min_t05": torch.minimum(left_t05, right_t05),
            "gap_left_t0": left_t0, "gap_right_t0": right_t0, "nearest_surface_t0": nearest_t0,
            "pinch_t0": pinch_t0,
            "bars_in_corridor": bars_in_corridor, "arc_clearance": arc_clear,
            "ray_returned": ray_returned,
            "in_fov_t1": in_fov_t1,
            "seen_frac_window": torch.where(mem_ok, seen_frac, nan),
            "seen_then_lost": seen_then_lost & mem_ok,
            "memory_window_valid": mem_ok,
            "target_involved": target_involved,
        }))

    def _record_star_convex_shadow(self, idx, hit_bearing, hit_d, lateral, no_return, rng):
        """Would a star-convex region have bounded the bar the corridor missed?

        Preregistered: docs/prereg_2026-09-05_a3_star_convex_shadow.md. Shadow only -- the live
        governor and the policy are untouched; this counts a counterfactual on the same frames.
        Eligible contacts are exactly those the corridor classified LATERAL or NO_RETURN, i.e. the
        77-78% the forensics found structurally unmonitorable.
        """
        from aerial_gym.task.navrl_task.star_convex import (
            SC_CONE_HALF_ANGLE_RAD,
            direction_clearance,
            scan_to_points,
        )

        eligible = lateral | no_return
        if not bool(eligible.any()):
            return
        # DELIBERATE: the vertically-collapsed scan, the same one the live governor consumes
        # (directional_lidar_clearance takes amin over the vertical beams). Feeding the full
        # [B,V,H] cloud here would change two things at once and the comparison would no longer
        # isolate what A3 asks. With this input the ONLY differences from the corridor are that
        # every bearing participates and that an unreturned ray bounds instead of opens.
        scan = getattr(self.perception, "last_scan_nearest", None)
        if scan is None:
            return
        scan = scan.unsqueeze(1)  # [B,H] -> [B,1,H], horizontal plane
        if self._cg_sc_geom is None:
            # Single elevation row at 0: the input is already vertically collapsed.
            self._cg_sc_geom = (
                self._speed_governor_bearings,
                torch.zeros(1, device=self.device),
            )
        bearings, elevations = self._cg_sc_geom
        if bearings is None:
            return

        pts, _ = scan_to_points(scan[idx], bearings, elevations, rng)
        # The bar's own bearing, as a unit direction in the sensor plane.
        d = torch.stack([torch.cos(hit_bearing), torch.sin(hit_bearing)], dim=1)
        sc_bound = direction_clearance(pts, d, rng, SC_CONE_HALF_ANGLE_RAD)
        seen = eligible & (hit_d <= sc_bound)

        self._cg_sc += torch.stack([
            eligible.sum().double(), seen.sum().double(),
            lateral.sum().double(), (lateral & seen).sum().double(),
            no_return.sum().double(), (no_return & seen).sum().double(),
            sc_bound[eligible].sum(dtype=torch.float64),
            hit_d[eligible].sum(dtype=torch.float64),
        ])

    def _star_convex_shadow_payload(self):
        from aerial_gym.task.navrl_task.star_convex import SC_CONE_HALF_ANGLE_RAD

        v = [float(x) for x in self._cg_sc.tolist()]
        elig, recl, lat_e, lat_r, nr_e, nr_r, sc_sum, hit_sum = v
        rate = lambda a, b: (a / b) if b > 0 else None
        r = rate(recl, elig)
        verdict = None
        if r is not None:
            verdict = ("PRESCRIPTION_SUPPORTED" if r >= 0.50
                       else "INSUFFICIENT" if r < 0.20 else "PARTIAL")
        return {
            "schema_version": 1,
            "prereg": "docs/prereg_2026-09-05_a3_star_convex_shadow.md",
            "cone_half_angle_deg": math.degrees(SC_CONE_HALF_ANGLE_RAD),
            "eligible_contacts": int(elig),
            "reclassified_seen": int(recl),
            "reclassification_rate": r,
            "lateral_eligible": int(lat_e), "lateral_reclassified": int(lat_r),
            "lateral_rate": rate(lat_r, lat_e),
            "no_return_eligible": int(nr_e), "no_return_reclassified": int(nr_r),
            "no_return_rate": rate(nr_r, nr_e),
            "mean_star_convex_bound_m": rate(sc_sum, elig),
            "mean_hit_distance_m": rate(hit_sum, elig),
            "verdict_a3": verdict,
        }

    def _contact_geometry_payload(self):
        c = [int(v) for v in self._cg_counts.tolist()]
        s = [float(v) for v in self._cg_sums.tolist()]
        (total, n, vout, beh_c, lat_c, nor_c, in_c, beh_a, lat_a, in_a, bars2, tgt) = c
        rate = lambda x: (x / n) if n > 0 else None
        mean = lambda x: (x / n) if n > 0 else None
        return {
            "schema_version": 1,
            "prereg": "docs/prereg_2026-09-04_contact_corridor_forensics.md",
            "path_half_width_m": float(self.speed_governor_cfg.path_half_width_m),
            "lookback_steps": self._CG_LOOKBACK_STEPS[0],
            "lookback_seconds": self._CG_LOOKBACK_STEPS[0] * float(self.step_dt),
            "contacts_total": total,
            "contacts_with_history": n,
            "contacts_dropped_early_episode": total - n,
            "contacts": n,
            "commanded_direction": {
                "vertical_out": vout, "behind": beh_c, "lateral": lat_c,
                "no_return": nor_c, "in_corridor": in_c,
                "vertical_out_rate": rate(vout), "behind_rate": rate(beh_c),
                "lateral_rate": rate(lat_c), "no_return_rate": rate(nor_c),
                "in_corridor_rate": rate(in_c),
            },
            "actual_velocity_direction": {
                "behind": beh_a, "lateral": lat_a, "in_corridor": in_a,
                "in_corridor_rate": rate(in_a),
            },
            "hypothesis_c_reclassified_into_corridor": in_a - in_c,
            "hypothesis_d_bars_in_corridor_ge2": bars2,
            "hypothesis_d_rate": rate(bars2),
            "target_involved": tgt,
            "mean_cmd_vs_actual_deg": mean(s[0]),
            "mean_bars_in_corridor": mean(s[1]),
            "mean_lateral_offset_m": mean(s[2]),
            "mean_arc_clearance_m": mean(s[3]),
            **self._contact_geometry_aux_payload(),
            **self._contact_records_payload(),
        }

    def _contact_records_payload(self):
        """I1/I2 sidecar files next to the result JSON, referenced by path, row count and SHA."""
        from aerial_gym.task.navrl_task import contact_records as CR

        contacts = [row for batch in self._cg_contact_rows for row in batch]
        frames = [row for batch in self._cg_frame_rows for row in batch]
        out = {
            "contact_records_schema": CR.SCHEMA_VERSION,
            "contact_records_categories": list(CR.CATEGORIES),
            "contact_records_rows": len(contacts),
            "frame_samples_rows": len(frames),
            "frame_sample_every_steps": int(self._cg_frame_every),
            "memory_window_steps": int(self._CG_MEMORY_STEPS),
            "side_sector_deg": list(CR.SIDE_SECTOR_DEG),
        }
        if not self._bulk_eval_output:
            out["contact_records_path"] = None
            return out
        base = Path(self._bulk_eval_output)
        stem = base.with_suffix("")
        extra = {"mode": str(self.speed_governor_cfg.mode), "bars": int(self.n_bars_active)}
        for key, rows, suffix in (("contact_records", contacts, ".contact_records.jsonl"),
                                  ("frame_samples", frames, ".frame_samples.jsonl")):
            path = stem.with_name(stem.name + suffix)
            try:
                n, sha = CR.rows_to_jsonl(path, rows, extra=extra)
                out[key + "_path"] = str(path)
                out[key + "_sha256"] = sha
            except OSError as exc:
                logger.warning("NavRL %s export failed: %s" % (key, exc))
                out[key + "_path"] = None
        return out

    def _contact_geometry_aux_payload(self):
        """A2 descriptive metrics. Not gates -- see prereg section 3."""
        a = [float(v) for v in self._cg_aux.tolist()]
        hist = [int(v) for v in self._cg_clearance_hist.tolist()]
        total = sum(hist)
        return {
            "corridor_clearance_hist_bin_m": 0.5,
            "corridor_clearance_hist": hist,
            "corridor_clearance_frames": total,
            "corridor_clearance_below_3m_rate": (
                (sum(hist[:6]) / total) if total else None
            ),
            "path_length_mean_m": (a[0] / a[1]) if a[1] else None,
            "path_length_episodes": int(a[1]),
            "cmd_vs_actual_deg_all_frames": (a[2] / a[3]) if a[3] else None,
            "cmd_vs_actual_frames": int(a[3]),
            "governor_compute_us_mean_batched": (a[4] / a[5]) if a[5] else None,
            "governor_compute_us_max_batched": a[6] if a[5] else None,
            "governor_compute_us_mean_per_env": (
                (a[4] / a[5] / self.num_envs) if a[5] else None
            ),
            "governor_compute_batch_envs": int(self.num_envs),
            "governor_compute_note": (
                "host-side issue cost of the batched call; no cuda.synchronize, "
                "so device execution may overlap. per_env divides by the batch."
            ),
            "governor_calls": int(a[5]),
        }

    def _s1_shadow_payload(self):
        from aerial_gym.task.navrl_task.shadow_association import (
            SHADOW_GATE_CHI2_3DOF_99,
            SHADOW_TOP_K,
        )

        counts = [int(v) for v in self._s1_counts.tolist()]
        (frames_total, visible_frames, target_lock, distractor_lock, ghost_lock,
         ambiguous, init_frames, init_target, init_distractor, init_ghost) = counts

        def rate(k):
            return (k / visible_frames) if visible_frames > 0 else None

        track_visible = visible_frames - init_frames
        track_false = (distractor_lock - init_distractor) + (ghost_lock - init_ghost)
        return {
            "schema_version": 1,
            "prereg": "docs/prereg_2026-09-03_s1_structure_fix_shadow.md",
            "classification_radius_m": DISTRACTOR_LOCK_RADIUS_M,
            "distractor_count": int(self._num_distractors),
            "gate_chi2_3dof": SHADOW_GATE_CHI2_3DOF_99,
            "top_k": SHADOW_TOP_K,
            "init_rule": "argmax_pixel_count",
            "frames_total": frames_total,
            "visible_frames": visible_frames,
            "target_lock": target_lock,
            "distractor_lock": distractor_lock,
            "ghost_lock": ghost_lock,
            "ambiguous_target_and_distractor": ambiguous,
            "target_lock_rate": rate(target_lock),
            "distractor_lock_rate": rate(distractor_lock),
            "ghost_lock_rate": rate(ghost_lock),
            "false_target_lock_rate": rate(distractor_lock + ghost_lock),
            "init_frames": init_frames,
            "init_target_lock": init_target,
            "init_distractor_lock": init_distractor,
            "init_ghost_lock": init_ghost,
            "init_false_lock_rate": (
                ((init_distractor + init_ghost) / init_frames) if init_frames > 0 else None
            ),
            "tracking_false_lock_rate": (
                (track_false / track_visible) if track_visible > 0 else None
            ),
        }

    def _goal_x_max(self):
        """Epoch-proportional goal-x ceiling: ramps k_start -> k_final over k_warmup_epochs, then
        plateaus. Uses num_task_steps as an epoch proxy (rl_games collects ppo_horizon env-steps per
        epoch, so epoch ~= num_task_steps / ppo_horizon). num_task_steps is saved/restored via
        get_env_state/set_env_state, so a --checkpoint resume (and --play) continues at the saved
        curriculum position."""
        if getattr(self.cur, "use_competence", False):
            return self._k_max_cur  # competence-gated: advanced only in _update_curriculum
        warmup_steps = max(1, int(self.cur.k_warmup_epochs) * int(self.cur.ppo_horizon))
        frac = min(1.0, self.num_task_steps / warmup_steps)
        return self.cur.k_start + (self.cur.k_final - self.cur.k_start) * frac

    def _goal_x_min(self):
        """Goal-x floor: stays at k_min early, then ramps k_min -> k_min_final over
        [k_min_ramp_start_epochs, +k_min_ramp_epochs] so late episodes drop the easy near goals and
        focus on deep crossings. The start is independent of the k_max ramp (they may overlap). Kept
        at least 1 m below k_max so the [min, max] window stays valid."""
        if getattr(self.cur, "use_competence", False):
            return min(self._k_min_cur, self._goal_x_max() - 1.0)
        h = int(self.cur.ppo_horizon)
        start_steps = int(self.cur.k_min_ramp_start_epochs) * h
        ramp_steps = max(1, int(self.cur.k_min_ramp_epochs) * h)
        frac = min(1.0, max(0.0, (self.num_task_steps - start_steps) / ramp_steps))
        k_min = self.cur.k_min + (self.cur.k_min_final - self.cur.k_min) * frac
        return min(k_min, self._goal_x_max() - 1.0)

    def _target_speed_max(self):
        """Phase 3: epoch-proportional target-speed ceiling — 0 -> speed_final over
        [speed_ramp_start_epochs, +speed_ramp_epochs], then holds. Same num_task_steps epoch proxy
        as _goal_x_max, so it survives --checkpoint resume and is restored at --play. An explicit
        NAVRL_TARGET_SPEED (speed_fixed >= 0, evaluation cells) bypasses the curriculum entirely."""
        if self._runtime_target_speed is not None:
            return float(self._runtime_target_speed)
        if float(self.tm.speed_fixed) >= 0.0:
            return float(self.tm.speed_fixed)
        final = float(self.tm.speed_final)
        minimum = max(0.0, float(getattr(self.tm, "speed_min", 0.0)))
        if final <= 0.0:
            return 0.0
        if (self.general_eval_mode or self._bulk_eval_mode) and os.environ.get(
            "NAVRL_EVAL_TARGET_SPEED_FINAL", "0"
        ).strip().lower() in ("1", "true", "yes", "on"):
            return max(minimum, final)
        h = int(self.cur.ppo_horizon)
        start_steps = int(self.tm.speed_ramp_start_epochs) * h
        ramp_steps = max(1, int(self.tm.speed_ramp_epochs) * h)
        frac = min(1.0, max(0.0, (self.num_task_steps - start_steps) / ramp_steps))
        return max(minimum, final * frac)

    def _target_planner_clearance(self):
        if not self._physical_target:
            return float(self.tm.obstacle_clearance)
        # Physical planner inflates each bar by the actor OBB's current world-XY support. The
        # remaining scalar is only closed-loop tracking reserve, not another hull radius.
        return float(getattr(self.tm, "physical_tracking_margin", 0.0))

    def _route_recovery_geometry(self, position_xy, bars_xy, bar_half_extents, support_xy):
        """Return exact closed-AABB hard/soft masks for the physical route recovery contract."""
        b_min = self.obs_dict["env_bounds_min"][:, 0:2]
        b_max = self.obs_dict["env_bounds_max"][:, 0:2]
        hard_lo, hard_hi = support_aware_bounds(
            b_min, b_max, float(self.cur.wall_margin), support_xy
        )
        hard_margin = TARGET_ROUTE_HARD_EPSILON_M + TARGET_ROUTE_REACHABLE_TUBE_MARGIN_M
        hard_lo = hard_lo + hard_margin
        hard_hi = hard_hi - hard_margin
        soft_margin = float(self.cur.wall_margin) + float(self.tm.physical_boundary_margin)
        soft_lo, soft_hi = support_aware_bounds(b_min, b_max, soft_margin, support_xy)
        geometry_valid = (
            torch.isfinite(position_xy).all(dim=1)
            & torch.isfinite(b_min).all(dim=1)
            & torch.isfinite(b_max).all(dim=1)
            & (b_max > b_min).all(dim=1)
            & torch.isfinite(support_xy).all(dim=1)
            & (support_xy >= 0.0).all(dim=1)
            & torch.isfinite(bars_xy).all(dim=(1, 2))
            & torch.isfinite(bar_half_extents).all(dim=(1, 2))
            & (bar_half_extents >= 0.0).all(dim=(1, 2))
        )
        hard_bounds_free = ((position_xy > hard_lo) & (position_xy < hard_hi)).all(dim=1) & geometry_valid
        soft_bounds_free = ((position_xy > soft_lo) & (position_xy < soft_hi)).all(dim=1) & geometry_valid
        hard_half = bar_half_extents + support_xy.unsqueeze(1) + hard_margin
        soft_half = (
            bar_half_extents
            + support_xy.unsqueeze(1)
            + float(self.tm.physical_tracking_margin)
            + hard_margin
        )
        if bars_xy.shape[1] == 0:
            hard_inside = torch.zeros(position_xy.shape[0], dtype=torch.bool, device=self.device)
            soft_inside = torch.zeros_like(hard_inside)
            soft_clearance = torch.full_like(position_xy[:, 0], float("inf"))
        else:
            hard_delta = (position_xy.unsqueeze(1) - bars_xy).abs() - hard_half
            soft_delta = (position_xy.unsqueeze(1) - bars_xy).abs() - soft_half
            hard_inside = (hard_delta <= 0.0).all(dim=2).any(dim=1)
            soft_inside_rows = (soft_delta <= 0.0).all(dim=2)
            soft_inside = soft_inside_rows.any(dim=1)
            outside_distance = soft_delta.clamp(min=0.0).norm(dim=2)
            inside_depth = soft_delta.amax(dim=2)
            signed_bar = torch.where(soft_inside_rows, inside_depth, outside_distance)
            soft_clearance = signed_bar.amin(dim=1)
        soft_boundary_clearance = torch.minimum(
            position_xy - soft_lo, soft_hi - position_xy
        ).amin(dim=1)
        soft_clearance = torch.minimum(soft_clearance, soft_boundary_clearance)
        hard_free = hard_bounds_free & ~hard_inside & geometry_valid
        soft_free = soft_bounds_free & ~soft_inside & geometry_valid
        soft_clearance = torch.where(
            geometry_valid, soft_clearance, torch.full_like(soft_clearance, float("-inf"))
        )
        if bars_xy.shape[1] == 0:
            hard_signed_bar = torch.full_like(position_xy[:, 0], float("inf"))
        else:
            hard_delta = (position_xy.unsqueeze(1) - bars_xy).abs() - hard_half
            hard_inside_rows = (hard_delta <= 0.0).all(dim=2)
            hard_outside_distance = hard_delta.clamp(min=0.0).norm(dim=2)
            hard_inside_depth = hard_delta.amax(dim=2)
            hard_signed_bar = torch.where(
                hard_inside_rows, hard_inside_depth, hard_outside_distance
            ).amin(dim=1)
        hard_boundary_clearance = torch.minimum(
            position_xy - hard_lo, hard_hi - position_xy
        ).amin(dim=1)
        hard_clearance = torch.minimum(hard_signed_bar, hard_boundary_clearance)
        hard_clearance = torch.where(
            geometry_valid, hard_clearance, torch.full_like(hard_clearance, float("-inf"))
        )
        return hard_free, soft_free, soft_clearance, hard_clearance, hard_lo, hard_hi, hard_half

    def _plan_target_routes(
        self, env_ids, *, connected_goal, is_replan=False, exclude_previous_goal=False,
        plan_cause=None,
    ):
        """CPU-plan selected environments only; ordinary route following remains on GPU.

        The planner's obstacle set is `_solid_obstacle_offset`-based, not `_bar_offset`-based:
        the painted distractors are solid collidable bodies, so a route allowed to pass straight
        through one would let the target teleport through geometry and would corrupt the
        occlusion statistics the distractor measurement reads.  `plan_idx`/`RoutePlanner.plan`
        are generic in the obstacle count (they reshape to [-1, 2] and require centres and
        half-extents to have the same shape), so widening the slice adds obstacles rather than
        changing the contract.  With no distractors configured the slice is identical to the
        historical bars-only one.
        """
        if not self._target_route_enabled or len(env_ids) == 0:
            return {}
        obstacles = self.obs_dict["obstacle_position"][
            :, self._solid_obstacle_offset : self._bar_offset + self.n_bars_active, 0:2
        ]
        obstacle_half = self.obs_dict["asset_collision_half_extents"][
            :, self._solid_obstacle_offset : self._bar_offset + self.n_bars_active, 0:2
        ]
        selector = self._target_route_selector if connected_goal else None
        status = self._target_route_manager.plan_idx(
            env_ids,
            self.target_position[:, 0:2],
            self._tm_waypoint,
            obstacles,
            obstacle_half,
            self.obs_dict["env_bounds_min"][:, 0:2],
            self.obs_dict["env_bounds_max"][:, 0:2],
            self._target_route_support_xy,
            int(self.num_task_steps),
            is_replan=is_replan,
            connected_goal_selector=selector,
            min_goal_distance_m=float(self.tm.route_min_goal_distance_m),
            excluded_goal_xy=(
                self._target_route_manager.goal if exclude_previous_goal else None
            ),
            goal_exclusion_radius_m=(
                float(self.tm.route_goal_exclusion_radius_m)
                if exclude_previous_goal else 0.0
            ),
            plan_cause=plan_cause,
        )
        # A successful connected-goal plan owns the waypoint. Failed rows retain their prior goal
        # but valid=False, so the only command they can emit is the fail-closed zero reference.
        valid = self._target_route_manager.valid[env_ids]
        if bool(valid.any()):
            valid_ids = env_ids[valid]
            self._tm_waypoint[valid_ids] = self._target_route_manager.goal[valid_ids]
        return status

    def _physical_target_support_xyz(self):
        """World-axis support radii of the current oriented physical collision box."""
        if not self._physical_target:
            return torch.zeros((self.num_envs, 3), device=self.device)
        rotation = quat_to_rotation_matrix(self.target_orientation)
        target_half = torch.tensor(
            [0.5 * float(v) for v in self.tm.physical_box_xyz],
            device=self.device,
        )
        return (rotation.abs() * target_half.view(1, 1, 3)).sum(dim=2)

    def _target_spawn_center_clearance(self):
        if not self._physical_target:
            return self._target_planner_clearance()
        # The reset sampler currently receives centers only. Add the largest active bar's XY
        # half-diagonal so its conservative center test implies the exact surface clearance used
        # by the runtime planner.
        bar_half = self.obs_dict["asset_collision_half_extents"][
            :, self._bar_offset : self._bar_offset + self.n_bars_active, 0:2
        ]
        bar_radius = float(bar_half.norm(dim=2).amax().item()) if bar_half.numel() else 0.0
        half = [0.5 * float(v) for v in self.tm.physical_box_xyz]
        target_radius = math.sqrt(sum(value * value for value in half))
        return bar_radius + target_radius + self._target_planner_clearance()

    def _sample_target_motion(self, env_ids):
        """Per-episode target speed + trajectory pattern for reset envs. Training samples
        speed ~ U[speed_min, v_max(epoch)]; the default speed_min=0 keeps static/slow episodes
        in-distribution. NAVRL_TARGET_SPEED forces the exact speed instead (evaluation cells)."""
        n = len(env_ids)
        if n == 0:
            return
        v_max = self._target_speed_max()
        if self._runtime_target_speed is not None:
            speed = torch.full((n,), float(self._runtime_target_speed), device=self.device)
        elif float(self.tm.speed_fixed) >= 0.0:
            speed = torch.full((n,), float(self.tm.speed_fixed), device=self.device)
        else:
            v_min = min(v_max, max(0.0, float(getattr(self.tm, "speed_min", 0.0))))
            speed = v_min + (v_max - v_min) * torch.rand(n, device=self.device)
        self._tm_speed[env_ids] = speed

        pat = str(self.tm.pattern)
        if pat == "mixed":
            code = torch.randint(0, 2, (n,), device=self.device)  # cv | waypoint, 50:50
        elif pat == "cv":
            code = torch.zeros(n, dtype=torch.long, device=self.device)
        elif pat == "waypoint":
            code = torch.ones(n, dtype=torch.long, device=self.device)
        elif pat == "circle":
            code = torch.full((n,), 2, dtype=torch.long, device=self.device)
        else:
            raise ValueError(
                f"unknown NAVRL_TARGET_PATTERN '{pat}' (expected cv|waypoint|circle|mixed)"
            )
        self._tm_pattern[env_ids] = code

        # cv: random persistent heading by default.  A closed evaluation may intervene on the
        # initial radial heading; random angles are still consumed to keep downstream RNG aligned.
        ang = 2.0 * math.pi * torch.rand(n, device=self.device)
        cv_velocity = initial_cv_velocity(
            self._eval_cv_initial_heading,
            speed,
            self.target_position[env_ids, 0:2],
            self.obs_dict["robot_position"][env_ids, 0:2],
            ang,
        )
        self._tm_cv_vel[env_ids] = cv_velocity
        self._tm_heading[env_ids] = torch.atan2(cv_velocity[:, 1], cv_velocity[:, 0])
        if self._eval_cv_initial_heading != "random":
            radial = self.target_position[env_ids, 0:2] - self.obs_dict[
                "robot_position"
            ][env_ids, 0:2]
            radial = radial / radial.norm(dim=1, keepdim=True).clamp(min=1e-6)
            direction = cv_velocity / speed.unsqueeze(1).clamp(min=1e-6)
            radial_cos = (radial * direction).sum(dim=1)
            radial_sin = radial[:, 0] * direction[:, 1] - radial[:, 1] * direction[:, 0]
            expected = {
                "toward": (-1.0, 0.0),
                "tangent_left": (0.0, 1.0),
                "tangent_right": (0.0, -1.0),
                "away": (1.0, 0.0),
            }[self._eval_cv_initial_heading]
            error = torch.maximum(
                (radial_cos - expected[0]).abs(), (radial_sin - expected[1]).abs()
            )
            self._eval_cv_heading_diag["samples"] += int(n)
            self._eval_cv_heading_diag["radial_cos_sum"] += float(radial_cos.sum().item())
            self._eval_cv_heading_diag["radial_sin_sum"] += float(radial_sin.sum().item())
            self._eval_cv_heading_diag["max_contract_error"] = max(
                self._eval_cv_heading_diag["max_contract_error"], float(error.max().item())
            )
        # waypoint: first waypoint anywhere inside the wall margins
        self._tm_waypoint[env_ids] = self._sample_waypoints(env_ids)
        # circle: ring around the (bar-clear) spawn goal, random direction
        self._tm_circle_center[env_ids] = self.target_position[env_ids, 0:2]
        r = max(1e-6, float(self.tm.circle_radius))
        sign = torch.where(
            torch.rand(n, device=self.device) < 0.5,
            torch.full((n,), -1.0, device=self.device),
            torch.full((n,), 1.0, device=self.device),
        )
        self._tm_circle_angvel[env_ids] = sign * speed / r
        self._tm_avoid_sign[env_ids] = torch.where(
            torch.rand(n, device=self.device) < 0.5,
            torch.full((n,), -1.0, device=self.device),
            torch.full((n,), 1.0, device=self.device),
        )
        if self._target_route_enabled:
            self._target_route_selector[env_ids] = torch.rand(n, device=self.device)
        # realized velocity starts at zero; _advance_target sets it from actual displacement
        self.target_vel_w[env_ids] = 0.0

    def _sample_waypoints(self, env_ids):
        """Uniform random XY waypoints inside the wall margins (per-env bounds)."""
        b_min = self.obs_dict["env_bounds_min"][env_ids]
        b_max = self.obs_dict["env_bounds_max"][env_ids]
        m = float(self.cur.wall_margin)
        if self._physical_target:
            m += float(getattr(self.tm, "physical_boundary_margin", 0.0))
        lo = b_min[:, 0:2] + m
        hi = b_max[:, 0:2] - m
        return lo + (hi - lo) * torch.rand(len(env_ids), 2, device=self.device)

    def _sync_target_to_sensor(self):
        """Mirror the moving target into the analytic semantic-LiDAR target buffer."""
        if self._sensor_target is not None:
            self._sensor_target[:] = self.target_position
        sensor_orientation = self.obs_dict.get("navrl_target_orientation", None)
        if sensor_orientation is not None:
            sensor_orientation[:] = self.target_orientation

    def _braking_v3_soft_envelope(self, bars_xy, bar_half, support_xy, b_min, b_max):
        spec = SoftEnvelopeSpec(
            wall_margin_m=float(self.cur.wall_margin),
            boundary_reserve_m=float(self.tm.physical_boundary_margin),
            tracking_margin_m=float(self.tm.physical_tracking_margin),
        )
        return torch_soft_envelope(b_min[:, 0:2], b_max[:, 0:2], bar_half, support_xy, spec)

    def _advance_target_braking_v3(self, moving, dt, old_xy, b_min, b_max):
        """Fresh-only GPU route step: shared envelope, full-horizon certificate, stop then replan."""
        manager = self._target_route_manager
        bars = self.obs_dict["obstacle_position"][
            :, self._bar_offset : self._bar_offset + self.n_bars_active, 0:2
        ]
        bar_half = self.obs_dict["asset_collision_half_extents"][
            :, self._bar_offset : self._bar_offset + self.n_bars_active, 0:2
        ]
        support = self._target_route_support_xy
        admissible_lo, admissible_hi, inflated_half = self._braking_v3_soft_envelope(
            bars, bar_half, support, b_min, b_max
        )
        current_safe = torch_segments_soft_safe(
            old_xy.unsqueeze(1), old_xy.unsqueeze(1),
            admissible_lo, admissible_hi, bars, inflated_half,
        )[:, 0]
        exited = moving & ~current_safe
        manager.v3_soft_envelope_exit_count += exited.to(torch.long).sum()
        speed = self.target_vel_w[:, 0:2].norm(dim=1)
        stopped = speed <= RECOVERY_STOP_SPEED_MPS
        needs = manager.needs_replan(
            self._tm_waypoint, support, int(self.num_task_steps)
        )
        may_plan = moving & current_safe & stopped & needs
        if bool(may_plan.any()):
            local_failure = may_plan & (
                (manager.status_code == manager.STATUS_CODES["local_step_infeasible"])
                | (manager.status_code == manager.STATUS_CODES["no_alternative_goal"])
                | (manager.status_code == manager.STATUS_CODES["same_goal_reselected"])
            )
            if bool(local_failure.any()):
                self._plan_target_routes(
                    local_failure.nonzero(as_tuple=False).squeeze(-1),
                    connected_goal=True, is_replan=True, exclude_previous_goal=True,
                    plan_cause="runtime_replan",
                )
            ordinary = may_plan & ~local_failure
            if bool(ordinary.any()):
                self._plan_target_routes(
                    ordinary.nonzero(as_tuple=False).squeeze(-1),
                    connected_goal=False, is_replan=True,
                    plan_cause="runtime_replan",
                )
        follow_kwargs = dict(
            reach_m=float(self.tm.waypoint_reach_m),
            admissible_lo=admissible_lo,
            admissible_hi=admissible_hi,
            bars_xy=bars,
            inflated_half=inflated_half,
            max_turn_rate=torch.full_like(
                self._tm_speed, math.radians(float(self.tm.max_turn_rate_deg))
            ),
            stop_speed_mps=RECOVERY_STOP_SPEED_MPS,
            brake_speed_knots=self._recovery_brake_speed_samples_mps,
            brake_distance_knots=self._recovery_brake_stop_distance_samples_m,
            dt=float(dt),
        )
        desired, speed_limit, route_active, route_complete, stop_turn_go, corner_prebrake, _fillet = (
            manager.braking_v3_follow_reference(
                old_xy, self.target_vel_w[:, 0:2], self._tm_speed, **follow_kwargs
            )
        )
        complete = route_complete & moving & current_safe & stopped
        if bool(complete.any()):
            complete_ids = complete.nonzero(as_tuple=False).squeeze(-1)
            self._target_route_selector[complete_ids] = torch.rand(
                len(complete_ids), device=self.device
            )
            self._plan_target_routes(
                complete_ids, connected_goal=True, is_replan=True,
                plan_cause="goal_completion",
            )
            desired, speed_limit, route_active, _, stop_turn_go, corner_prebrake, _fillet = (
                manager.braking_v3_follow_reference(
                    old_xy, self.target_vel_w[:, 0:2], self._tm_speed,
                    count_intervals=False, **follow_kwargs
                )
            )
        speed_limit = torch.where(
            moving & current_safe & route_active,
            speed_limit,
            torch.zeros_like(speed_limit),
        )
        _pos, command_xy, accepted, prebrake, certificate = braking_aware_route_step(
            old_xy,
            self.target_vel_w[:, 0:2],
            desired,
            speed_limit,
            dt,
            bars,
            admissible_lo,
            admissible_hi,
            inflated_half,
            self._tm_avoid_sign,
            torch.full_like(self._tm_speed, float(self.tm.max_accel)),
            torch.full_like(self._tm_speed, math.radians(float(self.tm.max_turn_rate_deg))),
            float(self.tm.avoidance_lookahead_s),
            self._recovery_brake_speed_samples_mps,
            self._recovery_brake_stop_distance_samples_m,
            self._recovery_brake_lateral_tube_p95_m,
        )
        command_xy = torch.where(
            (moving & current_safe).unsqueeze(1), command_xy, torch.zeros_like(command_xy)
        )
        manager.v3_accepted_command_count += (moving & accepted).to(torch.long).sum()
        manager.v3_accepted_terminal_stop_certificate_count += (
            moving & accepted & certificate["terminal_stop_safe"]
        ).to(torch.long).sum()
        manager.v3_prebrake_intervals += (moving & (prebrake | corner_prebrake)).to(torch.long).sum()
        manager.v3_certificate_failure_count += (
            moving & certificate["certificate_failure"]
        ).to(torch.long).sum()
        local_invalid = moving & current_safe & route_active & ~accepted
        self._target_route_selector = torch.where(
            local_invalid, torch.rand_like(self._target_route_selector), self._target_route_selector
        )
        manager.invalidate(local_invalid, "local_step_infeasible", int(self.num_task_steps))
        self._tm_last_step_feasible = moving & current_safe & accepted
        command = torch.zeros((self.num_envs, 3), device=self.device)
        command[:, 0:2] = torch.where(
            moving.unsqueeze(1), command_xy, torch.zeros_like(old_xy)
        )
        self._target_controller.set_hard_watchdog(
            bars, inflated_half, admissible_lo, admissible_hi, active=moving,
        )
        self._target_controller.set_command(
            command,
            torch.full_like(self._tm_speed, float(self.task_config.flight_altitude)),
        )
        self._tm_heading[moving] = torch.atan2(command_xy[moving, 1], command_xy[moving, 0])

    def _advance_target(self):
        """Phase 3: integrate the virtual target one RL step (step_dt = 0.1 s). Patterns:
        cv (heading held, reflected at the wall margins), waypoint (random waypoints), circle
        (parametric ring; the angle is re-derived from the current position each step, so bar
        push-outs simply slide the target around the ring). All patterns are then pushed out of
        bar clearance and clamped inside the wall margins; target_vel_w is set from the REALIZED
        displacement so the range-rate reward always matches the actual motion (reflections and
        push-outs included). Static episodes (speed 0 — the Phases 1-2 default) exit immediately,
        keeping the task byte-identical."""
        moving = self._tm_speed > 1e-6
        if not bool(moving.any()):
            return  # target_vel_w stays exactly zero -> range-rate == static vel term

        dt = self.step_dt
        old_xy = self.target_position[:, 0:2].clone()
        new_xy = old_xy.clone()
        b_min = self.obs_dict["env_bounds_min"]
        b_max = self.obs_dict["env_bounds_max"]
        m = float(self.cur.wall_margin)
        if self._physical_target:
            m += float(getattr(self.tm, "physical_boundary_margin", 0.0))
        lo = b_min[:, 0:2] + m
        hi = b_max[:, 0:2] - m

        if self._target_route_braking_v3_enabled and self._physical_target:
            self._advance_target_braking_v3(moving, dt, old_xy, b_min, b_max)
            return

        recovery_enabled = self._target_route_recovery_enabled and self._physical_target
        recovery_state = torch.full_like(moving, RECOVERY_NORMAL, dtype=torch.long)
        recovery_connect = torch.zeros_like(moving)
        recovery_route = torch.zeros_like(moving)
        recovery_hard_free = torch.ones_like(moving)
        recovery_soft_free = torch.ones_like(moving)
        recovery_soft_clearance = torch.full_like(self._tm_speed, float("inf"))
        recovery_hard_clearance = torch.full_like(self._tm_speed, float("inf"))
        hard_lo = hard_hi = hard_half = None
        recovery_bars = self.obs_dict["obstacle_position"][
            :, self._bar_offset : self._bar_offset + self.n_bars_active, 0:2
        ]
        recovery_bar_half = self.obs_dict["asset_collision_half_extents"][
            :, self._bar_offset : self._bar_offset + self.n_bars_active, 0:2
        ]
        recovery_support = self._target_route_support_xy
        if recovery_enabled:
            (
                recovery_hard_free,
                recovery_soft_free,
                recovery_soft_clearance,
                recovery_hard_clearance,
                hard_lo,
                hard_hi,
                hard_half,
            ) = self._route_recovery_geometry(
                old_xy, recovery_bars, recovery_bar_half, recovery_support
            )
            recovery_state = self._target_route_manager.recovery_state
            prior_local_soft_free = (
                moving
                & (self._target_route_manager.status_code == self._target_route_manager.STATUS_CODES["local_step_infeasible"])
                & recovery_hard_free
                & recovery_soft_free
                & ((recovery_state == RECOVERY_NORMAL) | (recovery_state == RECOVERY_ROUTE))
            )
            if bool(prior_local_soft_free.any()):
                self._target_route_manager.mark_local_infeasible_soft_free(prior_local_soft_free)
                recovery_state = self._target_route_manager.recovery_state
            # A hard breach is never converted into an escape/reset.  It remains a terminal
            # simulator event and the watchdog records it as a recovery failure.
            hard_breach = moving & ~recovery_hard_free & (
                recovery_state != RECOVERY_NO_CONNECTOR
            )
            if bool(hard_breach.any()):
                self._target_route_manager.mark_no_connector(hard_breach, hard_breach=True)
            soft_violation = moving & recovery_hard_free & ~recovery_soft_free
            self._target_route_manager.enter_recovery(
                soft_violation & ((recovery_state == RECOVERY_NORMAL) | (recovery_state == RECOVERY_ROUTE)),
                int(self.num_task_steps),
            )
            recovery_state = self._target_route_manager.recovery_state
            recovery_active = moving & ((recovery_state == RECOVERY_BRAKE) | (recovery_state == RECOVERY_CONNECT))
            self._target_route_manager.recovery_age_steps[recovery_active] += 1
            brake_timeout_steps = max(
                1,
                int(math.ceil((float(self.tm.recovery_brake_stop_time_p95) + 0.20) / dt)),
            )
            brake_active = moving & (recovery_state == RECOVERY_BRAKE)
            connect_active = moving & (recovery_state == RECOVERY_CONNECT)
            self._target_route_manager.recovery_brake_age_steps[brake_active] += 1
            self._target_route_manager.recovery_connect_age_steps[connect_active] += 1
            brake_timeout = brake_active & (
                self._target_route_manager.recovery_brake_age_steps > brake_timeout_steps
            )
            connect_timeout = connect_active & (
                self._target_route_manager.recovery_connect_age_steps
                > self._target_route_manager.recovery_connect_timeout_steps
            )
            recovery_timeout = brake_timeout | connect_timeout
            if bool(brake_timeout.any()):
                self._target_route_manager.mark_no_connector(brake_timeout, timeout_kind="brake")
            if bool(connect_timeout.any()):
                self._target_route_manager.mark_no_connector(connect_timeout, timeout_kind="connect")
            if bool(recovery_timeout.any()):
                recovery_state = self._target_route_manager.recovery_state
            brake_rows = moving & (recovery_state == RECOVERY_BRAKE)
            self._target_route_manager.recovery_brake_intervals += brake_rows.sum()
            brake_decel = float(getattr(self.tm, "recovery_brake_decel_p05", 0.0))
            brake_ids = brake_rows.nonzero(as_tuple=False).squeeze(-1)
            brake_safe = self._target_route_manager.brake_connector_idx(
                brake_ids,
                old_xy,
                self.target_vel_w[:, 0:2],
                recovery_bars,
                recovery_bar_half,
                b_min,
                b_max,
                recovery_support,
                float(self.cur.wall_margin),
                brake_decel,
                brake_speed_samples_mps=self._recovery_brake_speed_samples_mps,
                brake_stop_distance_samples_m=self._recovery_brake_stop_distance_samples_m,
                certified_lateral_tube_m=self._recovery_brake_lateral_tube_p95_m,
            )
            unsafe_brake = brake_rows & ~brake_safe
            anchor_rows = brake_rows & (
                self.target_vel_w[:, 0:2].norm(dim=1) <= RECOVERY_STOP_SPEED_MPS
            ) & recovery_hard_free
            anchor_rows |= unsafe_brake & recovery_hard_free
            if bool(anchor_rows.any()):
                anchor_ids = anchor_rows.nonzero(as_tuple=False).squeeze(-1)
                anchor_ok = self._target_route_manager.recovery_anchor_idx(
                    anchor_ids,
                    old_xy,
                    recovery_bars,
                    recovery_bar_half,
                    b_min,
                    b_max,
                    recovery_support,
                    float(self.cur.wall_margin),
                    float(self.cur.wall_margin) + float(self.tm.physical_boundary_margin),
                )
                no_anchor = anchor_rows & ~anchor_ok
                if bool(no_anchor.any()):
                    self._target_route_manager.mark_no_connector(no_anchor)
                good_anchor = anchor_rows & anchor_ok & ~no_anchor
                self._target_route_manager.recovery_state[good_anchor] = RECOVERY_CONNECT
                if bool(good_anchor.any()):
                    # Derived CONNECT budget: worst 7x7 radius-3 diagonal distance, acceleration
                    # ramp from the declared stop threshold, worst half-turn, and the existing
                    # 0.20 s reserve. No learned/tuned timeout is introduced.
                    # The nearest-cell projection can be 0.5 cell from the point on each axis;
                    # radius-3 opposite corner therefore bounds point->anchor by (3.5 cells).
                    max_anchor_distance = math.sqrt(2.0) * (3.0 + 0.5) * float(self.tm.route_resolution_m)
                    speed_floor = torch.full_like(self._tm_speed, RECOVERY_STOP_SPEED_MPS)
                    connect_speed = torch.maximum(self._tm_speed, speed_floor)
                    accel_time = connect_speed / max(float(self.tm.max_accel), 1e-6)
                    turn_time = math.pi / max(math.radians(float(self.tm.max_turn_rate_deg)), 1e-6)
                    connect_budget_s = (
                        max_anchor_distance / connect_speed
                        + accel_time
                        + turn_time
                        + 0.20
                    )
                    self._target_route_manager.recovery_connect_timeout_steps[good_anchor] = torch.ceil(
                        connect_budget_s[good_anchor] / dt
                    ).to(torch.long).clamp(min=1)
                    self._target_route_manager.recovery_connect_age_steps[good_anchor] = 0
            recovery_state = self._target_route_manager.recovery_state
            recovery_connect = moving & (recovery_state == RECOVERY_CONNECT)
            self._target_route_manager.recovery_connect_intervals += recovery_connect.sum()
            resume_ready = recovery_connect & recovery_soft_free & (
                recovery_soft_clearance > RECOVERY_HYSTERESIS_M
            )
            if bool(resume_ready.any()):
                resume_ids = resume_ready.nonzero(as_tuple=False).squeeze(-1)
                self._plan_target_routes(resume_ids, connected_goal=False, is_replan=True)
                resumed = resume_ready & self._target_route_manager.valid
                failed_resume = resume_ready & ~resumed
                if bool(failed_resume.any()):
                    self._target_route_manager.mark_no_connector(failed_resume)
                if bool(resumed.any()):
                    self._target_route_manager.mark_route_resume(resumed)
            recovery_state = self._target_route_manager.recovery_state
            recovery_connect = moving & (recovery_state == RECOVERY_CONNECT)
            recovery_route = moving & (recovery_state == RECOVERY_ROUTE)

        if self._target_dynamics in ("bounded", "physical"):
            desired_velocity = self._tm_cv_vel.clone()
            route_active = torch.ones_like(moving)
            if self._target_route_enabled:
                needs_replan = self._target_route_manager.needs_replan(
                    self._tm_waypoint,
                    self._target_route_support_xy,
                    int(self.num_task_steps),
                ) & moving & (
                    (self._target_route_manager.recovery_state == RECOVERY_NORMAL)
                    | (self._target_route_manager.recovery_state == RECOVERY_ROUTE)
                )
                if bool(needs_replan.any()):
                    local_failure = needs_replan & (
                        (
                            self._target_route_manager.status_code
                            == self._target_route_manager.STATUS_CODES["local_step_infeasible"]
                        )
                        | (
                            self._target_route_manager.status_code
                            == self._target_route_manager.STATUS_CODES["no_alternative_goal"]
                        )
                        | (
                            self._target_route_manager.status_code
                            == self._target_route_manager.STATUS_CODES["same_goal_reselected"]
                        )
                    )
                    if bool(local_failure.any()):
                        local_ids = local_failure.nonzero(as_tuple=False).squeeze(-1)
                        # Reusing the same deterministic route after a local dynamics failure can
                        # livelock. The selector was resampled when failure was detected; after the
                        # cooldown, choose a different reachable destination in this component.
                        self._plan_target_routes(
                            local_ids,
                            connected_goal=True,
                            is_replan=True,
                            exclude_previous_goal=True,
                        )
                    ordinary_replan = needs_replan & ~local_failure
                    if bool(ordinary_replan.any()):
                        self._plan_target_routes(
                            ordinary_replan.nonzero(as_tuple=False).squeeze(-1),
                            connected_goal=False,
                            is_replan=True,
                        )
                desired_velocity, route_active, route_complete = (
                    self._target_route_manager.velocity_reference(
                        old_xy, self._tm_speed, float(self.tm.waypoint_reach_m)
                    )
                )
                # A reached global goal is replaced only by a new, sufficiently distant goal for
                # which the same planner proves start-goal connectivity. The current interval is
                # recomputed after planning; failure remains a zero reference.
                complete = route_complete & moving
                if bool(complete.any()):
                    complete_ids = complete.nonzero(as_tuple=False).squeeze(-1)
                    self._target_route_selector[complete_ids] = torch.rand(
                        len(complete_ids), device=self.device
                    )
                    self._plan_target_routes(
                        complete_ids, connected_goal=True, is_replan=True
                    )
                    desired_velocity, route_active, _ = (
                        self._target_route_manager.velocity_reference(
                            old_xy, self._tm_speed, float(self.tm.waypoint_reach_m)
                        )
                    )
            wp = moving & (self._tm_pattern == 1)
            if not self._target_route_enabled and bool(wp.any()):
                to_wp = self._tm_waypoint[wp] - old_xy[wp]
                desired_velocity[wp] = (
                    to_wp / to_wp.norm(dim=1, keepdim=True).clamp(min=1e-6)
                    * self._tm_speed[wp].unsqueeze(1)
                )
            ci = moving & (self._tm_pattern == 2)
            if bool(ci.any()):
                rel_c = old_xy[ci] - self._tm_circle_center[ci]
                radial = rel_c / rel_c.norm(dim=1, keepdim=True).clamp(min=1e-6)
                sign = torch.sign(self._tm_circle_angvel[ci]).unsqueeze(1)
                tangent = torch.stack((-radial[:, 1], radial[:, 0]), dim=1) * sign
                desired_velocity[ci] = tangent * self._tm_speed[ci].unsqueeze(1)

            bars_all = self.obs_dict["obstacle_position"][
                :, self._bar_offset : self._bar_offset + self.n_bars_active, 0:2
            ]
            bars_half_extents = None
            if self._physical_target:
                bars_half_extents = self.obs_dict["asset_collision_half_extents"][
                    :, self._bar_offset : self._bar_offset + self.n_bars_active, 0:2
                ]
                target_support_xy = (
                    self._target_route_support_xy
                    if self._target_route_enabled
                    else self._physical_target_support_xyz()[:, :2]
                )
                bars_half_extents = bars_half_extents + target_support_xy.unsqueeze(1)
                # The planner bounds are center bounds.  Inflate the wall reserve by the current
                # world-axis OBB support so a feasible center trajectory cannot leave the actor's
                # collision box straddling the arena boundary.  The previous code checked only
                # the center, which produced rare finite-but-invalid OBB samples at 205+ bars.
                planner_lo, planner_hi = support_aware_bounds(
                    b_min[:, :2], b_max[:, :2], m, target_support_xy
                )
            else:
                planner_lo, planner_hi = lo, hi
            bounded_speed_limit = (
                torch.where(
                    route_active | recovery_connect,
                    self._tm_speed,
                    torch.zeros_like(self._tm_speed),
                )
                if self._target_route_enabled
                else self._tm_speed
            )
            bounded_xy, bounded_velocity, _, feasible = bounded_drone_target_step(
                old_xy,
                self.target_vel_w[:, 0:2],
                desired_velocity,
                bounded_speed_limit,
                dt,
                bars_all,
                planner_lo,
                planner_hi,
                self._target_planner_clearance(),
                self._tm_avoid_sign,
                torch.full_like(self._tm_speed, float(self.tm.max_accel)),
                torch.full_like(
                    self._tm_speed, math.radians(float(self.tm.max_turn_rate_deg))
                ),
                float(self.tm.avoidance_lookahead_s),
                bars_half_extents,
            )
            local_rollout_feasible = feasible.clone()
            if recovery_enabled and bool(recovery_connect.any()):
                # CONNECT is screened against the exact hard AABB, never the rounded local
                # clearance model.  The anchor itself was certified by the exact CPU connector;
                # this second rollout prevents an unmodelled turn/acceleration step from cutting
                # the hard envelope on the way there.
                connect_ids = recovery_connect.nonzero(as_tuple=False).squeeze(-1)
                to_anchor = self._target_route_manager.recovery_anchor[connect_ids] - old_xy[connect_ids]
                anchor_speed = to_anchor.norm(dim=1).clamp(min=1e-6)
                anchor_desired = to_anchor / anchor_speed.unsqueeze(1) * self._tm_speed[connect_ids].unsqueeze(1)
                connect_result = bounded_drone_target_step(
                    old_xy[connect_ids],
                    self.target_vel_w[connect_ids, 0:2],
                    anchor_desired,
                    self._tm_speed[connect_ids],
                    dt,
                    recovery_bars[connect_ids],
                    hard_lo[connect_ids],
                    hard_hi[connect_ids],
                    0.0,
                    self._tm_avoid_sign[connect_ids],
                    torch.full_like(self._tm_speed[connect_ids], float(self.tm.max_accel)),
                    torch.full_like(
                        self._tm_speed[connect_ids], math.radians(float(self.tm.max_turn_rate_deg))
                    ),
                    float(self.tm.avoidance_lookahead_s),
                    bars_half_extents[connect_ids],
                    exact_aabb_clearance=True,
                    hard_epsilon_m=TARGET_ROUTE_HARD_EPSILON_M + TARGET_ROUTE_REACHABLE_TUBE_MARGIN_M,
                    return_certificate=True,
                    certificate_row_ids=connect_ids,
                )
                connect_xy, connect_velocity, _, _, connect_certificate = connect_result
                # Recovery never accepts the ordinary longest-safe-prefix fallback.  CONNECT is
                # multi-interval, but every submitted interval must carry its own complete fixed
                # 1.0 s hard-envelope certificate.
                connect_feasible = (
                    connect_certificate["immediate_feasible"]
                    & connect_certificate["full_horizon_safe"]
                    & (connect_certificate["safe_prefix_steps"]
                       == int(connect_certificate["horizon_steps"]))
                )
                old_anchor_distance = (
                    self._target_route_manager.recovery_anchor[connect_ids] - old_xy[connect_ids]
                ).norm(dim=1)
                new_anchor_distance = (
                    self._target_route_manager.recovery_anchor[connect_ids]
                    - connect_certificate["selected_final_position_xy"]
                ).norm(dim=1)
                connect_feasible &= (
                    old_anchor_distance - new_anchor_distance
                    >= -RECOVERY_CONNECT_PROGRESS_TOLERANCE_M
                )
                bounded_xy[connect_ids] = connect_xy
                bounded_velocity[connect_ids] = connect_velocity
                local_rollout_feasible[connect_ids] = connect_feasible
                failed_connect = recovery_connect.clone()
                failed_connect[connect_ids] = ~connect_feasible
                if bool(failed_connect.any()):
                    self._target_route_manager.mark_no_connector(failed_connect)
                    recovery_connect[failed_connect] = False
            if self._physical_target:
                # The planner uses an additional tracking reserve. Crossing that soft reserve is
                # not itself a dynamically infeasible state; retain the hard hull clearance and
                # arena wall as the fail-closed feasibility contract.
                hard_lo, hard_hi = support_aware_bounds(
                    b_min[:, 0:2], b_max[:, 0:2], float(self.cur.wall_margin), target_support_xy
                )
                if recovery_enabled:
                    hard_margin = TARGET_ROUTE_HARD_EPSILON_M + TARGET_ROUTE_REACHABLE_TUBE_MARGIN_M
                    hard_lo = hard_lo + hard_margin
                    hard_hi = hard_hi - hard_margin
                feasible = ((bounded_xy >= hard_lo) & (bounded_xy <= hard_hi)).all(dim=1)
                if bars_all.shape[1] > 0:
                    delta = (
                        (bounded_xy.unsqueeze(1) - bars_all).abs() - bars_half_extents
                    )
                    if recovery_enabled:
                        # v2 recovery uses the same closed AABB hard envelope as the route
                        # certificate.  v1/legacy physical transitions retain their historical
                        # rounded local check because this branch is fresh-only.
                        feasible &= ~(
                            (delta - (TARGET_ROUTE_HARD_EPSILON_M + TARGET_ROUTE_REACHABLE_TUBE_MARGIN_M) <= 0.0)
                            .all(dim=2)
                            .any(dim=1)
                        )
                    else:
                        feasible &= delta.clamp(min=0.0).norm(dim=2).amin(dim=1) > 1e-4
                if self._target_route_enabled:
                    allowed_route = route_active | recovery_connect
                    feasible &= local_rollout_feasible & allowed_route
                    local_invalid = moving & route_active & ~local_rollout_feasible
                    if bool(local_invalid.any()):
                        self._target_route_selector[local_invalid] = torch.rand_like(
                            self._target_route_selector[local_invalid]
                        )
                    self._target_route_manager.invalidate(
                        local_invalid, "local_step_infeasible", int(self.num_task_steps)
                    )
            # A physical actor cannot be teleported or position-clamped when the rollout has no
            # safe first step.  Submit a zero planar command in that case: the real velocity
            # controller then brakes with its declared dynamics, while the strict feasibility
            # counter still records the event.  The old least-bad outward command could carry a
            # finite OBB across the arena boundary before the next task step observed it.
            self._tm_last_step_feasible = feasible
            if self._physical_target:
                command = torch.zeros((self.num_envs, 3), device=self.device)
                if recovery_enabled:
                    self._target_controller.set_hard_watchdog(
                        recovery_bars,
                        hard_half if hard_half is not None else bars_half_extents,
                        hard_lo,
                        hard_hi,
                        active=moving,
                    )
                safe_velocity = torch.where(
                    feasible.unsqueeze(1), bounded_velocity, torch.zeros_like(bounded_velocity)
                )
                command[:, 0:2] = torch.where(
                    moving.unsqueeze(1), safe_velocity, torch.zeros_like(old_xy)
                )
                self._target_controller.set_command(
                    command,
                    torch.full_like(self._tm_speed, float(self.task_config.flight_altitude)),
                )
                self._tm_heading[moving] = torch.atan2(
                    safe_velocity[moving, 1], safe_velocity[moving, 0]
                )
                cv = moving & (self._tm_pattern == 0)
                self._tm_cv_vel[cv] = bounded_velocity[cv]
                if not self._target_route_enabled and bool(wp.any()):
                    reached = (self._tm_waypoint[wp] - old_xy[wp]).norm(dim=1) < float(
                        self.tm.waypoint_reach_m
                    )
                    if bool(reached.any()):
                        wp_idx = wp.nonzero(as_tuple=False).squeeze(-1)
                        self._tm_waypoint[wp_idx[reached]] = self._sample_waypoints(
                            wp_idx[reached]
                        )
                return
            self.target_position[:, 0:2] = torch.where(
                moving.unsqueeze(1), bounded_xy, old_xy
            )
            self.target_position[:, 2] = self.task_config.flight_altitude
            self.target_vel_w[:, 0:2] = torch.where(
                moving.unsqueeze(1), bounded_velocity, torch.zeros_like(old_xy)
            )
            self.target_vel_w[:, 2] = 0.0
            self._tm_heading[moving] = torch.atan2(
                bounded_velocity[moving, 1], bounded_velocity[moving, 0]
            )
            cv = moving & (self._tm_pattern == 0)
            self._tm_cv_vel[cv] = bounded_velocity[cv]
            if not self._target_route_enabled and bool(wp.any()):
                reached = (self._tm_waypoint[wp] - self.target_position[wp, 0:2]).norm(
                    dim=1
                ) < float(self.tm.waypoint_reach_m)
                if bool(reached.any()):
                    wp_idx = wp.nonzero(as_tuple=False).squeeze(-1)
                    self._tm_waypoint[wp_idx[reached]] = self._sample_waypoints(
                        wp_idx[reached]
                    )
            return

        # -- cv: integrate the held heading, reflect position AND velocity at the wall margins
        cv = moving & (self._tm_pattern == 0)
        if bool(cv.any()):
            cv_idx = cv.nonzero(as_tuple=False).squeeze(-1)
            p = old_xy[cv] + self._tm_cv_vel[cv] * dt
            v = self._tm_cv_vel[cv]
            for ax in (0, 1):
                below = p[:, ax] < lo[cv, ax]
                above = p[:, ax] > hi[cv, ax]
                if self._bulk_eval_mode:
                    self._tm_ep_wall_reflections[cv_idx] += (below | above).to(torch.long)
                p[:, ax] = torch.where(below, 2.0 * lo[cv, ax] - p[:, ax], p[:, ax])
                p[:, ax] = torch.where(above, 2.0 * hi[cv, ax] - p[:, ax], p[:, ax])
                v[:, ax] = torch.where(below | above, -v[:, ax], v[:, ax])
            self._tm_cv_vel[cv] = v
            new_xy[cv] = p

        # -- waypoint: head toward the waypoint at the episode speed; resample on arrival
        wp = moving & (self._tm_pattern == 1)
        if bool(wp.any()):
            to_wp = self._tm_waypoint[wp] - old_xy[wp]
            d_wp = to_wp.norm(dim=1, keepdim=True).clamp(min=1e-6)
            step_len = (self._tm_speed[wp] * dt).unsqueeze(1)
            # do not overshoot the waypoint within a step
            move = to_wp / d_wp * torch.minimum(step_len, d_wp)
            new_xy[wp] = old_xy[wp] + move
            reached = (self._tm_waypoint[wp] - new_xy[wp]).norm(dim=1) < float(
                self.tm.waypoint_reach_m
            )
            if bool(reached.any()):
                wp_idx = wp.nonzero(as_tuple=False).squeeze(-1)
                self._tm_waypoint[wp_idx[reached]] = self._sample_waypoints(wp_idx[reached])

        # -- circle: orbit the center at the CURRENT radius (adaptive). Deriving both angle and
        # radius from the current position each step means a bar push-out simply enlarges the
        # orbit instead of fighting a fixed-radius snap-back (which oscillated in testing).
        ci = moving & (self._tm_pattern == 2)
        if bool(ci.any()):
            rel_c = old_xy[ci] - self._tm_circle_center[ci]
            r_cur = rel_c.norm(dim=1).clamp(min=max(0.5, 1e-6))
            theta = torch.atan2(rel_c[:, 1], rel_c[:, 0])
            # keep the TANGENTIAL speed at the episode speed regardless of the current radius
            omega = torch.sign(self._tm_circle_angvel[ci]) * self._tm_speed[ci] / r_cur
            theta = theta + omega * dt
            new_xy[ci] = self._tm_circle_center[ci] + r_cur.unsqueeze(1) * torch.stack(
                (torch.cos(theta), torch.sin(theta)), dim=1
            )

        # Local symmetric steering keeps the target moving at its commanded speed instead of
        # proposing an into-bar step and relying on positional push-out. The direct heading wins
        # whenever it is clear; otherwise +/- turn candidates are tried symmetrically with a
        # balanced per-episode tie sign. This makes mixed motion genuinely thread around bars and
        # prevents dense scenes from silently turning a nominal 1.5 m/s target into a parked one.
        flown_velocity = (new_xy - old_xy) / dt
        if self.n_bars_active > 0:
            bars_all = self.obs_dict["obstacle_position"][
                :, self._bar_offset : self._bar_offset + self.n_bars_active, 0:2
            ]
            desired_velocity = (new_xy - old_xy) / dt
            # Circle is a held-out pattern and keeps its legacy smallest-turn behavior. Passing
            # its current desired bearing as the reference makes the continuity preference a
            # no-op, while cv/waypoint use the persistent last flown heading.
            desired_heading = torch.atan2(desired_velocity[:, 1], desired_velocity[:, 0])
            continuity_heading = torch.where(
                self._tm_pattern == 2, desired_heading, self._tm_heading
            )
            steered_xy, steered_velocity, _, _ = steer_target_step(
                old_xy,
                desired_velocity,
                self._tm_speed,
                dt,
                bars_all,
                lo,
                hi,
                float(getattr(self.task_config, "goal_min_bar_clearance", 0.0)),
                self._tm_avoid_sign,
                continuity_heading,
            )
            new_xy = torch.where(moving.unsqueeze(1), steered_xy, new_xy)
            flown_velocity = steered_velocity
            self._tm_cv_vel[cv] = steered_velocity[cv]
        self._tm_heading[moving] = torch.atan2(
            flown_velocity[moving, 1], flown_velocity[moving, 0]
        )

        # -- keep the capture sphere flyable: push the target out of bar clearance. With bars only
        # 1.5 m apart and a 1.0 m clearance the exclusion discs OVERLAP: a naive radial push out of
        # the nearest bar PING-PONGS forever inside the lens between two discs (verified by probe —
        # push out of A lands in B's disc and vice versa). Instead push along the COMPOSITE of the
        # unit away-vectors of ALL violating bars: in a symmetric lens that resolves to the
        # perpendicular escape direction. Iterated with the wall clamp so a wall-adjacent bar
        # slides the target along the wall instead of fighting the clamp.
        clearance = float(getattr(self.task_config, "goal_min_bar_clearance", 0.0))
        pushed_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        push_dir = torch.zeros(self.num_envs, 2, device=self.device)
        if clearance > 0.0 and self.n_bars_active > 0:
            mov_idx = moving.nonzero(as_tuple=False).squeeze(-1)
            bars_xy = self.obs_dict["obstacle_position"][
                mov_idx, self._bar_offset : self._bar_offset + self.n_bars_active, 0:2
            ]  # (M, active B, 2)
            arangeM = torch.arange(len(mov_idx), device=self.device)
            for _ in range(6):
                new_xy = torch.maximum(torch.minimum(new_xy, hi), lo)  # walls first
                diff = new_xy[mov_idx].unsqueeze(1) - bars_xy          # (M, B, 2)
                d_all = diff.norm(dim=2)                               # (M, B)
                viol = d_all < clearance
                rows = viol.any(dim=1)
                if not bool(rows.any()):
                    break
                unit = diff / d_all.clamp(min=1e-6).unsqueeze(2)
                comp = (unit * viol.unsqueeze(2).float()).sum(dim=1)   # (M, 2)
                comp_n = comp.norm(dim=1, keepdim=True)
                # degenerate (dead-center of a symmetric lens): fall back to away-from-nearest
                jmin = d_all.min(dim=1).indices
                fallback = new_xy[mov_idx] - bars_xy[arangeM, jmin]
                fallback = fallback / fallback.norm(dim=1, keepdim=True).clamp(min=1e-6)
                dirn = torch.where(comp_n > 1e-6, comp / comp_n.clamp(min=1e-6), fallback)
                step_out = (clearance - d_all).clamp(min=0.0).max(dim=1).values + 1e-3
                sel = mov_idx[rows]
                new_xy[sel] = new_xy[sel] + dirn[rows] * step_out[rows].unsqueeze(1)
                pushed_any[sel] = True
                push_dir[sel] = dirn[rows]  # last composite escape normal, for the cv bounce
            # a pushed WAYPOINT target was heading somewhere unreachable next to a bar —
            # resample its waypoint so it does not keep fighting the push-out every step
            re_wp = pushed_any & (self._tm_pattern == 1)
            if bool(re_wp.any()):
                re_idx = re_wp.nonzero(as_tuple=False).squeeze(-1)
                self._tm_waypoint[re_idx] = self._sample_waypoints(re_idx)
            # cv targets BOUNCE off bars: reflect the held heading off the composite push normal
            # (specular, applied only to the into-bar component) plus a +-10 deg random deflection
            # so a symmetric two-bar trap cannot lock into a perfect back-and-forth loop. Without
            # this the cv heading only ever changed at the WALLS: a target aimed at a bar
            # re-entered the clearance disc every step and parked there. Measured with
            # tools/probe_target_motion.py at 75 bars / 1.5 m/s: realized speed 66% of commanded
            # with 28% of steps stalled before the bounce. Bouncing preserves the episode speed
            # exactly (reflection and rotation are both norm-preserving), so the speed-axis
            # labels of the density x speed map stay truthful.
            re_cv = pushed_any & (self._tm_pattern == 0)
            if bool(re_cv.any()):
                idx = re_cv.nonzero(as_tuple=False).squeeze(-1)
                n_hat = push_dir[idx]
                n_hat = n_hat / n_hat.norm(dim=1, keepdim=True).clamp(min=1e-6)
                v = self._tm_cv_vel[idx]
                into = (v * n_hat).sum(dim=1, keepdim=True)
                hit = (into.squeeze(1) < 0.0).float()
                if self._bulk_eval_mode:
                    self._tm_ep_bar_reflections[idx] += hit.to(torch.long)
                refl = v - 2.0 * into.clamp(max=0.0) * n_hat
                jit = (torch.rand(len(idx), device=self.device) - 0.5) * (math.pi / 9.0) * hit
                c, sn = torch.cos(jit), torch.sin(jit)
                self._tm_cv_vel[idx, 0] = refl[:, 0] * c - refl[:, 1] * sn
                self._tm_cv_vel[idx, 1] = refl[:, 0] * sn + refl[:, 1] * c
                # The next continuity window must follow the reflected heading rather than fight
                # the collision response using the pre-bounce direction.
                self._tm_heading[idx] = torch.atan2(
                    self._tm_cv_vel[idx, 1], self._tm_cv_vel[idx, 0]
                )

        # -- final wall clamp (physical bound) and write-back
        new_xy = torch.maximum(torch.minimum(new_xy, hi), lo)
        self.target_position[:, 0:2] = torch.where(
            moving.unsqueeze(1), new_xy, old_xy
        )
        self.target_position[:, 2] = self.task_config.flight_altitude
        # realized world velocity (z = 0): what the range-rate reward sees
        self.target_vel_w[:, 0:2] = torch.where(
            moving.unsqueeze(1),
            (self.target_position[:, 0:2] - old_xy) / dt,
            torch.zeros_like(old_xy),
        )
        self.target_vel_w[:, 2] = 0.0

    def _reset_density_strata(self):
        for tensor in (
            self._density_speed_succ,
            self._density_speed_fin,
            self._density_dist_succ,
            self._density_dist_fin,
            self._density_pattern_succ,
            self._density_pattern_fin,
        ):
            tensor.zero_()

    def _restore_density_strata(self, state):
        specs = (
            ("density_speed_succ", self._density_speed_succ),
            ("density_speed_fin", self._density_speed_fin),
            ("density_dist_succ", self._density_dist_succ),
            ("density_dist_fin", self._density_dist_fin),
            ("density_pattern_succ", self._density_pattern_succ),
            ("density_pattern_fin", self._density_pattern_fin),
        )
        self._reset_density_strata()
        for key, dst in specs:
            value = state.get(key)
            if not isinstance(value, (list, tuple)) or len(value) != dst.numel():
                continue
            dst.copy_(torch.as_tensor(value, dtype=dst.dtype, device=self.device))

    def _record_density_strata(self, successes, finished):
        idx = finished.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return
        captured = successes[idx].to(torch.float32)

        speed_bin, dist_bin, pattern_bin = self._episode_strata_bins(idx)

        for bins, succ_dst, fin_dst, size in (
            (speed_bin, self._density_speed_succ, self._density_speed_fin, 4),
            (dist_bin, self._density_dist_succ, self._density_dist_fin, 4),
            (pattern_bin, self._density_pattern_succ, self._density_pattern_fin, 3),
        ):
            fin_dst += torch.bincount(bins, minlength=size).to(fin_dst.dtype)
            succ_dst += torch.bincount(
                bins, weights=captured, minlength=size
            ).to(succ_dst.dtype)

    def _episode_strata_bins(self, idx):
        """Return historical curriculum-gate bins; speed intentionally starts at zero."""

        speed_max = max(1e-6, float(self._target_speed_max()))
        speed_bin = torch.floor(self._tm_speed[idx] * (4.0 / speed_max)).long().clamp(0, 3)
        dist_span = max(1e-6, self.general_goal_dist_max - self.general_goal_dist_min)
        dist_bin = torch.floor(
            (self._episode_goal_dist[idx] - self.general_goal_dist_min) * (4.0 / dist_span)
        ).long().clamp(0, 3)
        pattern_bin = self._tm_pattern[idx].long().clamp(0, 2)
        return speed_bin, dist_bin, pattern_bin

    def _episode_eval_strata_bins(self, idx):
        """Return bins over the distribution actually applied in held-out evaluation."""
        speed_min = max(0.0, float(getattr(self.tm, "speed_min", 0.0)))
        speed_max = max(speed_min, float(self._target_speed_max()))
        speed_span = max(1e-6, speed_max - speed_min)
        speed_bin = torch.floor(
            (self._tm_speed[idx] - speed_min) * (4.0 / speed_span)
        ).long().clamp(0, 3)
        dist_span = max(1e-6, self.general_goal_dist_max - self.general_goal_dist_min)
        dist_bin = torch.floor(
            (self._episode_goal_dist[idx] - self.general_goal_dist_min) * (4.0 / dist_span)
        ).long().clamp(0, 3)
        pattern_bin = self._tm_pattern[idx].long().clamp(0, 2)
        return speed_bin, dist_bin, pattern_bin

    def _record_eval_outcome_strata(self, successes, crashes, timeouts, finished):
        """Accumulate mutually exclusive held-out outcomes without touching training state."""
        idx = finished.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return
        captured = successes[idx].to(torch.float32)
        crashed = (crashes[idx] > 0).to(torch.float32)
        timed_out = timeouts[idx].to(torch.float32)
        speed_bin, dist_bin, pattern_bin = self._episode_eval_strata_bins(idx)

        for bins, succ_dst, crash_dst, timeout_dst, cause_dst, fin_dst, size in (
            (speed_bin, self._eval_speed_succ, self._eval_speed_crash,
             self._eval_speed_timeout, self._eval_speed_crash_cause,
             self._eval_speed_fin, 4),
            (dist_bin, self._eval_dist_succ, self._eval_dist_crash,
             self._eval_dist_timeout, self._eval_dist_crash_cause,
             self._eval_dist_fin, 4),
            (pattern_bin, self._eval_pattern_succ, self._eval_pattern_crash,
             self._eval_pattern_timeout, self._eval_pattern_crash_cause,
             self._eval_pattern_fin, 3),
        ):
            fin_dst += torch.bincount(bins, minlength=size).to(fin_dst.dtype)
            succ_dst += torch.bincount(
                bins, weights=captured, minlength=size
            ).to(succ_dst.dtype)
            crash_dst += torch.bincount(
                bins, weights=crashed, minlength=size
            ).to(crash_dst.dtype)
            timeout_dst += torch.bincount(
                bins, weights=timed_out, minlength=size
            ).to(timeout_dst.dtype)
            crashed_idx = crashed > 0
            if bool(crashed_idx.any()):
                cause = self._crash_cause_code[idx][crashed_idx].long()
                valid = (cause >= 0) & (cause < 4)
                if bool(valid.any()):
                    flat = bins[crashed_idx][valid] * 4 + cause[valid]
                    cause_dst += torch.bincount(
                        flat, minlength=size * 4
                    ).reshape(size, 4).to(cause_dst.dtype)

        # Joint distance×pattern cells distinguish a long-range path-length effect from one motion
        # generator dominating the marginal distance curve.  These remain descriptive strata.
        joint = dist_bin * 3 + pattern_bin
        self._eval_dist_pattern_fin += torch.bincount(
            joint, minlength=12
        ).reshape(4, 3).to(self._eval_dist_pattern_fin.dtype)
        self._eval_dist_pattern_succ += torch.bincount(
            joint, weights=captured, minlength=12
        ).reshape(4, 3).to(self._eval_dist_pattern_succ.dtype)
        self._eval_dist_pattern_crash += torch.bincount(
            joint, weights=crashed, minlength=12
        ).reshape(4, 3).to(self._eval_dist_pattern_crash.dtype)
        self._eval_dist_pattern_timeout += torch.bincount(
            joint, weights=timed_out, minlength=12
        ).reshape(4, 3).to(self._eval_dist_pattern_timeout.dtype)
        crashed_idx = crashed > 0
        if bool(crashed_idx.any()):
            cause = self._crash_cause_code[idx][crashed_idx].long()
            valid = (cause >= 0) & (cause < 4)
            if bool(valid.any()):
                flat = joint[crashed_idx][valid] * 4 + cause[valid]
                self._eval_dist_pattern_crash_cause += torch.bincount(
                    flat, minlength=48
                ).reshape(4, 3, 4).to(self._eval_dist_pattern_crash_cause.dtype)

    def _record_oob_exit(self, d_oob, pos, b_min, b_max, m_oob, steps):
        """Forensics for arena exits (WORKLOG 2026-08-21).

        Evaluation-only and read-only: it describes an exit the caller already decided, and touches
        no observation, reward, termination or checkpoint state.

        Why it exists. The ref5in hard-distance cells fail almost entirely by leaving the arena --
        158 of 160 crashes in the seed-367 control arm -- and the existing diagnostic recorded only
        the count, the compass bucket and the mean step. That cannot distinguish the two failure
        modes it lumps together: driving outward while chasing something, versus drifting out while
        searching blind. The extra fields here are chosen to separate exactly those.

        `d_oob` is the CAUSE-ATTRIBUTED mask, the same one the crash-cause table counts, so `exits`
        is comparable with crash_causes.out_of_bounds instead of being a second, larger population.
        """
        idx = d_oob.nonzero(as_tuple=False).squeeze(1)
        n = int(idx.numel())
        if n == 0:
            return
        self._oob_n += n

        # Edge attribution mirrors the existing oob_w/e/s/n buckets; a diagonal corner exit lands
        # in two of them, so these can sum above `exits`. Kept independent of _diag so the export
        # can cross-check the two counters against each other.
        for slot, mask in (
            (0, pos[idx, 0] < b_min[idx, 0] - m_oob),
            (1, pos[idx, 0] > b_max[idx, 0] + m_oob),
            (2, pos[idx, 1] < b_min[idx, 1] - m_oob),
            (3, pos[idx, 1] > b_max[idx, 1] + m_oob),
        ):
            self._oob_edge_counts[slot] += int(mask.sum().item())

        # Per-env step, i.e. the exact step each episode left on.
        ep_steps = steps[idx]
        self._oob_step_sum += float(ep_steps.sum().item())
        bounded = ep_steps.clamp(0, self._oob_step_hist.numel() - 1).to(torch.long)
        self._oob_step_hist.index_add_(0, bounded, torch.ones_like(bounded))

        vel = self.obs_dict.get("robot_linvel")
        if not hasattr(self, "_fa_ep_first_fused"):
            raise RuntimeError("NavRL OOB forensics require first-acquisition telemetry")
        if vel is None:
            raise RuntimeError("NavRL OOB forensics require robot_linvel")
        never_acquired = self._fa_ep_first_fused[idx] < 0
        self._oob_never_acquired += int(never_acquired.sum().item())

        to_goal = self.target_position[idx, 0:2] - pos[idx, 0:2]
        dist = to_goal.norm(dim=1)
        speed = vel[idx, 0:2].norm(dim=1)
        unit = to_goal / dist.clamp(min=1e-6).unsqueeze(1)
        goal_closing = (vel[idx, 0:2] * unit).sum(dim=1)
        centre = 0.5 * (b_min[idx] + b_max[idx])
        radial = pos[idx, 0:2] - centre
        radial = radial / radial.norm(dim=1, keepdim=True).clamp(min=1e-6)
        outward = (vel[idx, 0:2] * radial).sum(dim=1)

        self._oob_goal_dist_sum += float(dist.sum().item())
        self._oob_speed_sum += float(speed.sum().item())
        # >0 closing on the goal as it crossed, <0 receding from it.
        self._oob_goal_closing_sum += float(goal_closing.sum().item())
        # >0 actively driving outward rather than drifting across the line.
        self._oob_outward_sum += float(outward.sum().item())

        for label, group_mask in (
            ("never_acquired", never_acquired),
            ("acquired", ~never_acquired),
        ):
            group = self._oob_acquisition_groups[label]
            group["n"] += int(group_mask.sum().item())
            group["speed_sum"] += float(speed[group_mask].sum().item())
            group["goal_dist_sum"] += float(dist[group_mask].sum().item())
            group["goal_closing_sum"] += float(goal_closing[group_mask].sum().item())
            group["outward_sum"] += float(outward[group_mask].sum().item())

    def _record_first_acquisition(self, idx, outcome, observation_steps):
        """Attribute per-episode first-acquisition statistics to eval outcomes (RESEARCH_PLAN 8.28).

        Answers a question the visible-fraction telemetry cannot: an episode with a 0.6% visible
        fraction may have acquired the target once and lost it, or never acquired it at all, and
        the intervention differs (tracker memory vs the range contract). The hard-distance cell
        starts every episode with the target beyond both the 20 m camera and the 12 m LiDAR, so
        "never acquired" is the specific failure this is built to count.

        Never-acquired episodes contribute to `never`, and to NOTHING else -- folding their -1 (or
        a stand-in 0) into the first-visible mean would report the strongest failures as the
        fastest acquisitions.
        """
        first = self._fa_ep_first_fused[idx]
        first_cam = self._fa_ep_first_camera[idx]
        transitions = self._fa_ep_transitions[idx]
        own_steps = self._fa_ep_obs_steps[idx]

        # The two telemetries keep independent observation counters precisely so this can be
        # checked; if they ever disagree the two chronologies are not comparable.
        if not bool(torch.equal(own_steps, observation_steps)):
            raise RuntimeError("first-acquisition observation chronology diverged")
        acquired = first >= 0
        if bool((first[acquired] > own_steps[acquired]).any()) or bool((first == 0).any()):
            raise RuntimeError("first-acquisition step outside its own episode")
        cam_acq = first_cam >= 0
        # The camera is one of the two fused sources, so it cannot acquire before the fusion does.
        if bool((first_cam[cam_acq & acquired] < first[cam_acq & acquired]).any()):
            raise RuntimeError("camera acquisition precedes fused acquisition")

        self._fa_eval_outcome_fin += torch.bincount(outcome, minlength=3).to(torch.long)
        self._fa_eval_outcome_never.index_add_(0, outcome, (~acquired).to(torch.long))
        self._fa_eval_outcome_camera_never.index_add_(0, outcome, (~cam_acq).to(torch.long))
        self._fa_eval_outcome_transitions.index_add_(0, outcome, transitions)
        self._fa_eval_outcome_first_sum.index_add_(
            0, outcome, torch.where(acquired, first, torch.zeros_like(first))
        )
        self._fa_eval_outcome_camera_first_sum.index_add_(
            0, outcome, torch.where(cam_acq, first_cam, torch.zeros_like(first_cam))
        )
        if bool(acquired.any()):
            # Exact median needs the distribution; a flat (3 * bins) index_add is the cheapest way
            # to keep one histogram per outcome.
            bins = self._fa_hist_bins
            flat = outcome[acquired] * bins + first[acquired].clamp(0, bins - 1)
            self._fa_eval_outcome_first_hist.view(-1).index_add_(
                0, flat, torch.ones_like(flat)
            )

    def _record_target_motion_outcome_telemetry(
        self, successes, crashes, timeouts, finished
    ):
        """Attribute visibility and target reflections to mutually exclusive eval outcomes."""
        if not self._bulk_eval_mode:
            return
        idx = finished.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return
        captured = successes[idx].to(torch.bool)
        crashed = (crashes[idx] > 0)
        timed_out = timeouts[idx].to(torch.bool)
        exclusive = captured.to(torch.long) + crashed.to(torch.long) + timed_out.to(torch.long)
        if not bool((exclusive == 1).all()):
            raise RuntimeError("target-motion telemetry received non-exclusive outcomes")
        outcome = torch.where(
            captured,
            torch.zeros_like(idx),
            torch.where(crashed, torch.ones_like(idx), torch.full_like(idx, 2)),
        )
        wall = self._tm_ep_wall_reflections[idx]
        bar = self._tm_ep_bar_reflections[idx]
        wall_any = (wall > 0).to(torch.long)
        bar_any = (bar > 0).to(torch.long)
        visible_steps = self._tm_ep_visible_steps[idx]
        observation_steps = self._tm_ep_observation_steps[idx]
        if bool((visible_steps > observation_steps).any()) or bool(
            (observation_steps <= 0).any()
        ):
            raise RuntimeError("target-motion visibility counter contract failed")

        # RESEARCH_PLAN 8.28, bucketed here rather than in a sibling recorder so that both
        # telemetries are attributed by the SAME outcome tensor. A second copy of the
        # capture/crash/timeout derivation is exactly the kind of drift the cross-sum check below
        # is meant to catch, so there is no second copy.
        self._record_first_acquisition(idx, outcome, observation_steps)

        search = self.perception.search if self.perception is not None else None
        if self._s1_search_telemetry_enabled and search is not None:
            for values, sums, counts in (
                (
                    search.coverage_fraction_at_first_visible[idx],
                    self._s1_first_coverage_sum,
                    self._s1_first_coverage_n,
                ),
                (
                    search.belief_entropy_at_first_visible[idx],
                    self._s1_first_entropy_sum,
                    self._s1_first_entropy_n,
                ),
            ):
                finite = torch.isfinite(values)
                if bool(finite.any()):
                    sums.index_add_(0, outcome[finite], values[finite].to(sums.dtype))
                    counts.index_add_(
                        0, outcome[finite], torch.ones_like(outcome[finite], dtype=torch.long)
                    )

        self._tm_eval_outcome_fin += torch.bincount(outcome, minlength=3).to(
            self._tm_eval_outcome_fin.dtype
        )
        for source, destination in (
            (wall, self._tm_eval_outcome_wall_sum),
            (wall_any, self._tm_eval_outcome_wall_any),
            (bar, self._tm_eval_outcome_bar_sum),
            (bar_any, self._tm_eval_outcome_bar_any),
            (visible_steps, self._tm_eval_outcome_visible_steps),
            (observation_steps, self._tm_eval_outcome_observation_steps),
        ):
            destination.index_add_(0, outcome, source.to(destination.dtype))

        speed_bin, _, _ = self._episode_eval_strata_bins(idx)
        flat = speed_bin * 6 + wall_any * 3 + outcome
        self._tm_eval_speed_wall_outcome += torch.bincount(
            flat, minlength=24
        ).reshape(4, 2, 3).to(self._tm_eval_speed_wall_outcome.dtype)
        self._tm_eval_speed_wall_visible_steps.view(-1).index_add_(
            0, flat, visible_steps.to(self._tm_eval_speed_wall_visible_steps.dtype)
        )
        self._tm_eval_speed_wall_observation_steps.view(-1).index_add_(
            0, flat, observation_steps.to(self._tm_eval_speed_wall_observation_steps.dtype)
        )

    @staticmethod
    def _density_rate_text(succ, fin, labels):
        result = []
        for label, n_succ, n_fin in zip(labels, succ.tolist(), fin.tolist()):
            if n_fin > 0:
                result.append("%s=%.3f(%d)" % (label, n_succ / n_fin, n_fin))
            else:
                result.append("%s=na(0)" % label)
        return ",".join(result)

    def _density_stratified_gate(self):
        """Return (pass, reason) for the optional broad-slice competence guard."""
        enabled = bool(getattr(self.density, "use_stratified_gate", False))
        if not enabled:
            return True, "diagnostic-only"

        floor = float(getattr(self.density, "stratified_floor", 0.55))
        min_eps = max(1, int(getattr(self.density, "stratified_min_episodes", 512)))
        if self._runtime_target_speed is not None or float(self.tm.speed_fixed) >= 0.0:
            forced_speed = max(0.0, float(self._target_speed_max()))
            speed_max = max(1e-6, float(self.tm.speed_final), forced_speed)
            speed_bins = (
                min(3, max(0, int(math.floor(forced_speed * 4.0 / speed_max)))),
            )
        elif float(self.tm.speed_final) <= 0.0:
            speed_bins = (0,)
        else:
            speed_bins = range(4)
        expected = [
            ("speed", self._density_speed_succ, self._density_speed_fin, speed_bins),
            ("distance", self._density_dist_succ, self._density_dist_fin, range(4)),
        ]
        pattern = str(self.tm.pattern)
        pattern_bins = {"mixed": (0, 1), "cv": (0,), "waypoint": (1,), "circle": (2,)}.get(
            pattern, (0, 1)
        )
        expected.append(
            ("pattern", self._density_pattern_succ, self._density_pattern_fin, pattern_bins)
        )

        for name, succ, fin, bins in expected:
            for bin_index in bins:
                n_fin = int(fin[bin_index].item())
                if n_fin < min_eps:
                    return False, "%s[%d] insufficient %d<%d" % (
                        name,
                        bin_index,
                        n_fin,
                        min_eps,
                    )
                rate = int(succ[bin_index].item()) / max(1, n_fin)
                if rate < floor:
                    return False, "%s[%d] %.3f<%.3f" % (
                        name,
                        bin_index,
                        rate,
                        floor,
                    )
        return True, "all broad slices >= %.3f" % floor

    def _density_threshold_now(self):
        # An explicit per-density schedule wins when configured; otherwise a linear ramp from
        # success_threshold_start (at n_start bars) to success_threshold_end (at n_final bars),
        # evaluated at the CURRENT active bar count. Both ramp endpoints default to the flat
        # success_threshold, so with nothing set this stays a constant threshold.
        d = self.density
        n_start = int(getattr(d, "n_start", self.n_bars_active))
        n_final = int(getattr(d, "n_final", self.n_bars_active))
        t_start = float(getattr(d, "success_threshold_start", getattr(d, "success_threshold", 0.8)))
        t_end = float(getattr(d, "success_threshold_end", getattr(d, "success_threshold", 0.8)))
        return density_threshold_at(
            self.n_bars_active,
            n_start,
            n_final,
            t_start,
            t_end,
            schedule=self._density_threshold_schedule,
        )

    def _update_curriculum(self, successes, finished):
        # (A) Competence-gated goal-DISTANCE curriculum (NAVRL_K_COMPETENCE=1): deepen the goal
        # window only when measured capture clears the threshold -- the same self-pacing the density
        # curriculum uses below. When off, distance stays epoch-proportional in _goal_x_max/_min.
        if getattr(self.cur, "use_competence", False):
            n_fin_d = int(torch.sum(finished).item())
            if n_fin_d > 0:
                self._kcomp_succ += int(torch.sum(successes).item())
                self._kcomp_fin += n_fin_d
                if self._kcomp_fin >= max(1, int(self.cur.k_comp_check)):
                    rate = self._kcomp_succ / max(1, self._kcomp_fin)
                    if rate >= float(self.cur.k_comp_threshold) and self._k_max_cur < float(self.cur.k_final):
                        old = self._k_max_cur
                        self._k_max_cur = min(float(self.cur.k_final), self._k_max_cur + float(self.cur.k_comp_step))
                        self._k_min_cur = min(
                            float(self.cur.k_min_final), max(float(self.cur.k_min), self._k_max_cur - 3.0)
                        )
                        logger.warning(
                            "NavRL distance curriculum promoted | k_max %.1f -> %.1f window [%.1f, %.1f] "
                            "after %d eps, capture=%.3f"
                            % (old, self._k_max_cur, self._k_min_cur, self._k_max_cur, self._kcomp_fin, rate)
                        )
                    else:
                        logger.info(
                            "NavRL distance curriculum held | window [%.1f, %.1f] capture=%.3f over %d eps"
                            % (self._k_min_cur, self._k_max_cur, rate, self._kcomp_fin)
                        )
                    self._kcomp_succ = 0
                    self._kcomp_fin = 0
        # (B) Optional Phase-2 density curriculum (also capture-gated; orthogonal to distance above).
        if self.density is None or not getattr(self.density, "use_density_curriculum", False):
            return
        # Gate on warmup FIRST so the capture accumulators only collect POST-warmup episodes.
        # Accumulating during warmup (early low-capture training) would make the first promotion check
        # use a lifetime average dragged below threshold and stall the first promotion by one window.
        horizon = int(getattr(self.cur, "ppo_horizon", 1))
        warmup_steps = max(
            int(getattr(self.density, "warmup_epochs", 0)) * max(1, horizon),
            int(self._density_gate_not_before_steps),
        )
        if self.num_task_steps < warmup_steps:
            return
        n_fin = int(torch.sum(finished).item())
        if n_fin <= 0:
            return

        self._density_succ_agg += int(torch.sum(successes).item())
        self._density_fin_agg += n_fin
        self._record_density_strata(successes, finished)

        check_after = max(1, int(getattr(self.density, "check_after_episodes", 2048)))
        if self._density_fin_agg < check_after:
            return

        capture_rate = self._density_succ_agg / max(1, self._density_fin_agg)
        threshold = self._density_threshold_now()
        final_bars = self._clamp_active_bars(getattr(self.density, "n_final", self.n_bars_active))
        strata_pass, strata_reason = self._density_stratified_gate()
        logger.warning(
            "NavRL density strata | speed[%s] distance[%s] pattern[%s] gate=%s (%s)"
            % (
                self._density_rate_text(
                    self._density_speed_succ,
                    self._density_speed_fin,
                    ("q0", "q1", "q2", "q3"),
                ),
                self._density_rate_text(
                    self._density_dist_succ,
                    self._density_dist_fin,
                    ("q0", "q1", "q2", "q3"),
                ),
                self._density_rate_text(
                    self._density_pattern_succ,
                    self._density_pattern_fin,
                    ("cv", "waypoint", "circle"),
                ),
                "pass" if strata_pass else "hold",
                strata_reason,
            )
        )
        # Dwell gate: even a passing capture window cannot promote until the policy has spent
        # min_epochs_per_density at this density. This is what stops the curriculum from chaining
        # promotions the instant each evidence window fills and never letting a level converge.
        min_dwell_epochs = max(0, int(getattr(self.density, "min_epochs_per_density", 0)))
        dwell_epochs = density_dwell_epochs(
            self.num_task_steps,
            self._density_level_start_steps,
            horizon,
        )
        dwell_ok = density_dwell_ready(
            self.num_task_steps,
            self._density_level_start_steps,
            horizon,
            min_dwell_epochs,
        )

        if (
            capture_rate >= threshold
            and strata_pass
            and self.n_bars_active < final_bars
            and dwell_ok
        ):
            step = max(1, int(getattr(self.density, "promote_step", 15)))
            old_bars = self.n_bars_active
            self._set_active_bars(min(final_bars, old_bars + step))
            self._density_level_start_steps = density_level_start_after_promotion(
                self._density_level_start_steps,
                self.num_task_steps,
                promoted=True,
            )
            logger.warning(
                "NavRL density curriculum promoted | bars %d -> %d after %d eps, "
                "capture=%.3f (threshold=%.3f) dwell=%.0f epochs"
                % (
                    old_bars,
                    self.n_bars_active,
                    self._density_fin_agg,
                    capture_rate,
                    threshold,
                    dwell_epochs,
                )
            )
        elif capture_rate >= threshold and strata_pass and not dwell_ok:
            # Distinguish "not good enough yet" from "good enough but still maturing" -- otherwise
            # a dwell hold is indistinguishable from a failed capture gate in the log.
            logger.warning(
                "NavRL density curriculum DWELL | bars=%d capture=%.3f >= threshold=%.3f but only "
                "%.0f/%d epochs at this density; holding to let it converge"
                % (
                    self.n_bars_active,
                    capture_rate,
                    threshold,
                    dwell_epochs,
                    min_dwell_epochs,
                )
            )
        else:
            # Training runs suppress INFO logs, so an INFO-only hold made "no promotion line"
            # ambiguous: the gate may not have been evaluated yet, or it may have failed. Keep the
            # competence decision visible at the same level as a promotion.
            logger.warning(
                "NavRL density curriculum held | bars=%d capture=%.3f over %d eps (threshold=%.3f)"
                % (self.n_bars_active, capture_rate, self._density_fin_agg, threshold)
            )

        self._density_succ_agg = 0
        self._density_fin_agg = 0
        self._reset_density_strata()

    def _record_epoch_dashboard(self, successes, crashes, timeouts, finished):
        """Feed finished-episode outcomes to the per-epoch train dashboard (console + TB)."""
        n_fin = int(torch.sum(finished).item())
        if n_fin == 0:
            return
        # Closest approach EXCLUDING crashes: a crash dies far from the goal and only inflates the
        # mean, so aggregate over non-crash finished episodes and also surface the best (min).
        nocrash = finished & ~(crashes > 0)
        n_nc = int(torch.sum(nocrash).item())
        closest_nc_sum = float(torch.sum(self.ep_min_goal_dist[nocrash]).item()) if n_nc else 0.0
        closest_min = float(torch.min(self.ep_min_goal_dist[nocrash]).item()) if n_nc else None
        tm_on = float(self.tm.speed_final) > 0.0 or float(self.tm.speed_fixed) >= 0.0
        record_navrl_epoch_episodes(
            num_finished=n_fin,
            num_captured=int(torch.sum(successes).item()),
            num_crash=int(torch.sum(crashes > 0).item()),
            num_timeout=int(torch.sum(timeouts).item()),
            closest_nocrash_sum=closest_nc_sum,
            closest_nocrash_count=n_nc,
            closest_min=closest_min,
            goal_dist_max=self.cur_goal_dist_max,
            goal_dist_min=self.cur_goal_dist_min,
            n_bars_active=self.n_bars_active,
            target_speed_max=self._target_speed_max() if tm_on else None,
            target_speed_mean=float(self._tm_speed.mean().item()) if tm_on else None,
            target_speed_realized_mean=(
                float(self.target_vel_w[:, 0:2].norm(dim=1).mean().item())
                if tm_on
                else None
            ),
        )

    def _validate_eval_outcome_strata(self, total):
        """Fail closed if any held-out stratum loses or duplicates a terminal outcome."""
        expected = (
            int(total),
            int(self._succ_agg),
            int(self._crash_agg),
            int(self._to_agg),
        )
        for name, successes, crashes, timeouts, causes, episodes in (
            ("speed", self._eval_speed_succ, self._eval_speed_crash,
             self._eval_speed_timeout, self._eval_speed_crash_cause,
             self._eval_speed_fin),
            ("distance", self._eval_dist_succ, self._eval_dist_crash,
             self._eval_dist_timeout, self._eval_dist_crash_cause,
             self._eval_dist_fin),
            ("pattern", self._eval_pattern_succ, self._eval_pattern_crash,
             self._eval_pattern_timeout, self._eval_pattern_crash_cause,
             self._eval_pattern_fin),
            ("distance_pattern", self._eval_dist_pattern_succ,
             self._eval_dist_pattern_crash, self._eval_dist_pattern_timeout,
             self._eval_dist_pattern_crash_cause, self._eval_dist_pattern_fin),
        ):
            observed = (
                int(episodes.sum().item()),
                int(successes.sum().item()),
                int(crashes.sum().item()),
                int(timeouts.sum().item()),
            )
            if observed != expected:
                raise RuntimeError(
                    "NavRL bulk eval %s strata mismatch: %s != %s"
                    % (name, observed, expected)
                )
            n_causes = int(causes.sum().item())
            if n_causes != expected[2]:
                raise RuntimeError(
                    "NavRL bulk eval %s crash-cause mismatch: %d != %d"
                    % (name, n_causes, expected[2])
                )
        telemetry_outcomes = tuple(
            int(value) for value in self._tm_eval_outcome_fin.tolist()
        )
        if telemetry_outcomes != expected[1:]:
            raise RuntimeError(
                "NavRL target-motion outcome mismatch: %s != %s"
                % (telemetry_outcomes, expected[1:])
            )
        speed_wall_outcomes = tuple(
            int(value)
            for value in self._tm_eval_speed_wall_outcome.sum(dim=(0, 1)).tolist()
        )
        if speed_wall_outcomes != expected[1:]:
            raise RuntimeError(
                "NavRL target-motion speed/reflection mismatch: %s != %s"
                % (speed_wall_outcomes, expected[1:])
            )
        if int(self._tm_eval_outcome_observation_steps.sum().item()) != int(
            self._tm_eval_speed_wall_observation_steps.sum().item()
        ) or int(self._tm_eval_outcome_visible_steps.sum().item()) != int(
            self._tm_eval_speed_wall_visible_steps.sum().item()
        ):
            raise RuntimeError("NavRL target-motion visibility aggregation mismatch")

        # RESEARCH_PLAN 8.28. The first-acquisition telemetry is bucketed by the same outcome
        # tensor as the visibility telemetry, so any disagreement here means one of the two
        # accumulators dropped or double-counted an episode -- fail rather than export.
        # tuple, not list: `expected` is a tuple, and a list never compares equal to one, so a
        # list here would make this guard fire on every run regardless of the counts.
        fa_outcomes = tuple(int(v) for v in self._fa_eval_outcome_fin.tolist())
        if fa_outcomes != expected[1:]:
            raise RuntimeError(
                "NavRL first-acquisition outcome mismatch: %s != %s"
                % (fa_outcomes, expected[1:])
            )
        acquired = [
            fa_outcomes[i] - int(self._fa_eval_outcome_never[i]) for i in range(3)
        ]
        if any(a < 0 for a in acquired):
            raise RuntimeError("NavRL first-acquisition never-count exceeds episode count")
        hist_totals = [int(v) for v in self._fa_eval_outcome_first_hist.sum(dim=1).tolist()]
        if hist_totals != acquired:
            raise RuntimeError(
                "NavRL first-acquisition histogram mismatch: %s != %s"
                % (hist_totals, acquired)
            )

    def _export_episode_forensics(self):
        """Write the evaluation-only forensic rows and the trajectory digest, if either is on.

        Separate files and a separate schema from the bulk-eval payload: these are diagnostics, and
        folding them into the established result document would change what that document means.
        """
        if self._episode_forensics is not None and self._episode_forensics_output:
            try:
                path = self._episode_forensics.export(self._episode_forensics_output)
                logger.warning("NavRL episode forensics saved -> %s" % path)
            except (OSError, FileExistsError) as exc:
                logger.warning("NavRL episode forensics export failed: %s" % exc)
        if self._trajectory_digest is not None:
            record = self._trajectory_digest.as_dict()
            print("NAVRL_TRAJECTORY_DIGEST " + json.dumps(record, sort_keys=True), flush=True)
            if self._trajectory_digest_output:
                try:
                    out = Path(self._trajectory_digest_output)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
                except OSError as exc:
                    logger.warning("NavRL trajectory digest export failed: %s" % exc)

    def _export_bulk_eval_result(self, total, reach_rate, mean_nc, best):
        """Persist the exact outcome window consumed by a vectorized rl_games player."""
        if not self._bulk_eval_mode or self._bulk_eval_exported:
            return
        self._validate_eval_outcome_strata(total)

        ad = self._action_diag
        n_action = max(1, int(ad["n"]))
        n_delta = max(1, int(ad["delta_y_n"]))
        d = self._diag
        n_crash_causes = d["contact"] + d["below"] + d["above"] + d["oob"]
        n_cause_den = max(1, n_crash_causes)

        fixed_speed = float(getattr(self.tm, "speed_fixed", -1.0))
        representation = self._obstacle_representation_or_zero()
        speed_min = max(0.0, float(getattr(self.tm, "speed_min", 0.0)))
        # Record the distribution actually sampled in this process. In particular, a checkpoint
        # restored before its speed ramp finishes must not be mislabeled with speed_final.
        speed_max = max(speed_min, float(self._target_speed_max()))
        if fixed_speed >= 0.0:
            target_speed_mode = "fixed"
            target_speed_mps = fixed_speed
            speed_min = speed_max = fixed_speed
        elif speed_max > 0.0:
            target_speed_mode = "uniform"
            target_speed_mps = None
        else:
            target_speed_mode = "static"
            target_speed_mps = 0.0

        goal_dist_min, goal_dist_max, full_goal_distribution = (
            self._general_goal_distance_bounds()
        )
        fov_curriculum_saturated = self._fov_curriculum_is_saturated()
        physics = self._runtime_physics_contract()

        def scalar(name):
            return float(self._speed_governor_diag[name].item())

        def mean_scalar(total_name, count_name="samples"):
            return scalar(total_name) / max(1.0, scalar(count_name))

        def step_summary(values):
            if not values:
                return {"count": 0, "mean": None, "p10": None, "p50": None, "p90": None}
            array = np.asarray(values, dtype=np.float64)
            return {
                "count": int(array.size),
                "mean": float(array.mean()),
                "p10": float(np.quantile(array, 0.10)),
                "p50": float(np.quantile(array, 0.50)),
                "p90": float(np.quantile(array, 0.90)),
            }

        def oob_exit_payload():
            """Where and when episodes left the arena (WORKLOG 2026-08-21).

            Rates are per EXIT, not per episode, because an exit is the event being described.
            Everything is null when nothing left the arena, rather than 0 -- a zero here would read
            as "exits happen at step 0 on edge x_min", which is the opposite of no exits at all.
            """
            n = int(self._oob_n)
            # Two independent counters over the same mask. If they disagree, one of them is
            # double-counting or dropping exits, and neither number is trustworthy -- fail rather
            # than export. This is the same discipline the first-acquisition telemetry uses.
            if n != int(self._diag["oob"]):
                raise RuntimeError(
                    "NavRL OOB forensics disagree with the crash-cause counter: %d != %d"
                    % (n, int(self._diag["oob"]))
                )
            if n == 0:
                return {"exits": 0}
            group_total = sum(int(row["n"]) for row in self._oob_acquisition_groups.values())
            if group_total != n or int(self._oob_acquisition_groups["never_acquired"]["n"]) != int(
                self._oob_never_acquired
            ):
                raise RuntimeError(
                    "NavRL OOB acquisition strata disagree with the exit counter: %d/%d/%d"
                    % (group_total, n, int(self._oob_never_acquired))
                )

            def acquisition_group_payload(label):
                row = self._oob_acquisition_groups[label]
                count = int(row["n"])
                if count == 0:
                    return {
                        "exits": 0,
                        "share": 0.0,
                        "speed_mean_mps": None,
                        "goal_distance_mean_m": None,
                        "goal_closing_speed_mean_mps": None,
                        "outward_radial_speed_mean_mps": None,
                    }
                return {
                    "exits": count,
                    "share": count / n,
                    "speed_mean_mps": row["speed_sum"] / count,
                    "goal_distance_mean_m": row["goal_dist_sum"] / count,
                    "goal_closing_speed_mean_mps": row["goal_closing_sum"] / count,
                    "outward_radial_speed_mean_mps": row["outward_sum"] / count,
                }

            hist = self._oob_step_hist
            cumulative = torch.cumsum(hist, dim=0)
            half = (n + 1) // 2
            median_step = int(torch.searchsorted(
                cumulative, torch.tensor(half, device=cumulative.device, dtype=cumulative.dtype)
            ).item())
            edges = [int(v) for v in self._oob_edge_counts.tolist()]
            return {
                "exits": n,
                # A corner crossing increments two edges, so these can sum above `exits`.
                "edge_counts": {
                    "x_min": edges[0], "x_max": edges[1],
                    "y_min": edges[2], "y_max": edges[3],
                },
                "edge_shares": {
                    "x_min": edges[0] / n, "x_max": edges[1] / n,
                    "y_min": edges[2] / n, "y_max": edges[3] / n,
                },
                "step_mean": self._oob_step_sum / n,
                "step_median": min(median_step, hist.numel() - 1),
                "never_acquired": int(self._oob_never_acquired),
                "never_acquired_share": self._oob_never_acquired / n,
                "speed_mean_mps": self._oob_speed_sum / n,
                "goal_distance_mean_m": self._oob_goal_dist_sum / n,
                # Negative means the drone was receding from the goal as it crossed the boundary.
                "goal_closing_speed_mean_mps": self._oob_goal_closing_sum / n,
                # Positive means it was actively driving outward, not drifting.
                "outward_radial_speed_mean_mps": self._oob_outward_sum / n,
                "by_acquisition": {
                    label: acquisition_group_payload(label)
                    for label in ("never_acquired", "acquired")
                },
            }

        def first_acquisition_payload(index):
            """RESEARCH_PLAN 8.28 per-outcome first-acquisition statistics.

            Means and the median are over ACQUIRED episodes only, and are null when none acquired.
            Reporting 0 for a never-acquired cohort would invert the finding -- the episodes that
            never saw the target would read as the fastest to see it.
            """
            episodes = int(self._fa_eval_outcome_fin[index].item())
            never = int(self._fa_eval_outcome_never[index].item())
            acquired = episodes - never
            cam_never = int(self._fa_eval_outcome_camera_never[index].item())
            cam_acquired = episodes - cam_never

            median = None
            p10 = None
            p90 = None
            if acquired > 0:
                hist = self._fa_eval_outcome_first_hist[index]
                cumulative = torch.cumsum(hist, dim=0)
                def quantile_step(fraction):
                    # Lower empirical quantile: first step whose cumulative count reaches ceil(q*n).
                    target = max(1, int(math.ceil(fraction * acquired)))
                    pos = int(torch.searchsorted(cumulative, torch.tensor(
                        target, device=cumulative.device, dtype=cumulative.dtype)).item())
                    return int(min(pos, self._fa_hist_bins - 1))

                p10 = quantile_step(0.10)
                median = quantile_step(0.50)
                p90 = quantile_step(0.90)

            return {
                "episodes": episodes,
                "never_acquired": never,
                "never_acquired_rate": (never / episodes) if episodes else None,
                "acquired": acquired,
                "first_visible_step_mean": (
                    int(self._fa_eval_outcome_first_sum[index].item()) / acquired
                    if acquired else None
                ),
                "first_visible_step_median": median,
                "first_visible_step_p10": p10,
                "first_visible_step_p90": p90,
                "visible_hidden_transitions": int(
                    self._fa_eval_outcome_transitions[index].item()
                ),
                "visible_hidden_transitions_mean_per_episode": (
                    int(self._fa_eval_outcome_transitions[index].item()) / episodes
                    if episodes else None
                ),
                "camera_never_acquired": cam_never,
                "camera_never_acquired_rate": (cam_never / episodes) if episodes else None,
                "camera_first_visible_step_mean": (
                    int(self._fa_eval_outcome_camera_first_sum[index].item()) / cam_acquired
                    if cam_acquired else None
                ),
            }

        def target_motion_outcome_payload(index):
            episodes = int(self._tm_eval_outcome_fin[index].item())
            observations = int(
                self._tm_eval_outcome_observation_steps[index].item()
            )
            visible = int(self._tm_eval_outcome_visible_steps[index].item())
            wall_total = int(self._tm_eval_outcome_wall_sum[index].item())
            wall_any = int(self._tm_eval_outcome_wall_any[index].item())
            bar_total = int(self._tm_eval_outcome_bar_sum[index].item())
            bar_any = int(self._tm_eval_outcome_bar_any[index].item())
            return {
                "episodes": episodes,
                "observation_steps": observations,
                "visible_steps": visible,
                "visible_fraction_step_weighted": (
                    float(visible / observations) if observations > 0 else None
                ),
                "wall_reflections": wall_total,
                "wall_reflections_mean_per_episode": (
                    float(wall_total / episodes) if episodes > 0 else None
                ),
                "wall_reflection_any": wall_any,
                "wall_reflection_any_rate": (
                    float(wall_any / episodes) if episodes > 0 else None
                ),
                "bar_reflections": bar_total,
                "bar_reflections_mean_per_episode": (
                    float(bar_total / episodes) if episodes > 0 else None
                ),
                "bar_reflection_any": bar_any,
                "bar_reflection_any_rate": (
                    float(bar_any / episodes) if episodes > 0 else None
                ),
            }

        def speed_wall_payload(speed_index, reflection_index):
            counts = self._tm_eval_speed_wall_outcome[
                speed_index, reflection_index
            ]
            observations = self._tm_eval_speed_wall_observation_steps[
                speed_index, reflection_index
            ]
            visible = self._tm_eval_speed_wall_visible_steps[
                speed_index, reflection_index
            ]
            episodes = int(counts.sum().item())
            result = {
                "episodes": episodes,
                "captured": int(counts[0].item()),
                "crash": int(counts[1].item()),
                "timeout": int(counts[2].item()),
            }
            for index, label in enumerate(("capture", "crash", "timeout")):
                n_observations = int(observations[index].item())
                n_visible = int(visible[index].item())
                result[label + "_rate"] = (
                    float(counts[index].item() / episodes) if episodes > 0 else None
                )
                result[label + "_visible_fraction_step_weighted"] = (
                    float(n_visible / n_observations) if n_observations > 0 else None
                )
            return result

        def stratum_payload(
            successes, crashes, timeouts, crash_causes, episodes, labels
        ):
            result = {}
            for row, (label, n_success, n_crash, n_timeout, n_episode) in enumerate(zip(
                labels,
                successes.tolist(),
                crashes.tolist(),
                timeouts.tolist(),
                episodes.tolist(),
            )):
                n_episode = int(n_episode)
                n_success = int(n_success)
                n_crash = int(n_crash)
                n_timeout = int(n_timeout)
                result[label] = {
                    "successes": n_success,
                    "crash": n_crash,
                    "timeout": n_timeout,
                    "episodes": n_episode,
                    "capture_rate": (
                        float(n_success / n_episode) if n_episode > 0 else None
                    ),
                    "crash_rate": (
                        float(n_crash / n_episode) if n_episode > 0 else None
                    ),
                    "timeout_rate": (
                        float(n_timeout / n_episode) if n_episode > 0 else None
                    ),
                    "crash_causes": {
                        cause: int(crash_causes[row, cause_index].item())
                        for cause_index, cause in enumerate(
                            ("bar_contact", "below", "above", "out_of_bounds")
                        )
                    },
                }
            return result

        payload = {
            "schema_version": 1,
            "requested_episodes": int(self._bulk_eval_target),
            "actual_episodes": int(total),
            "checkpoint": os.environ.get("NAVRL_EVAL_CHECKPOINT", ""),
            "condition": {
                "robot_name": self._robot_provenance["robot_name"],
                "robot_config_class": self._robot_provenance["robot_config_class"],
                "robot_config_sha256": self._robot_provenance[
                    "robot_config_sha256"
                ],
                "robot_asset_file": self._robot_provenance["robot_asset_file"],
                "robot_asset_sha256": self._robot_provenance[
                    "robot_asset_sha256"
                ],
                "action_selection": os.environ.get(
                    "NAVRL_EVAL_ACTION_MODE", "configured"
                ).strip().lower(),
                # The player owns this inference-only transform. Recording it here prevents an
                # M*pi*M rollout from being mislabeled as the original controller.
                "reflection_mode": os.environ.get(
                    "NAVRL_EVAL_REFLECTION_MODE", "original"
                ).strip().lower(),
                "seed": int(self.task_config.seed),
                "bars": int(self.n_bars_active),
                "target_pattern": os.environ.get("NAVRL_TARGET_PATTERN", "static"),
                "target_motion_model": self._target_motion_model,
                "target_route_mode": self._target_route_mode,
                "target_route_recovery_schema": (
                    TARGET_ROUTE_RECOVERY_SCHEMA if self._target_route_recovery_enabled else "off"
                ),
                "target_route_recovery_hard_envelope": (
                    "closed_aabb_support_v1" if self._target_route_recovery_enabled else "off"
                ),
                "target_route_recovery_soft_envelope": (
                    "closed_aabb_support_plus_tracking_v1" if self._target_route_recovery_enabled else "off"
                ),
                "target_route_recovery_hard_epsilon_m": TARGET_ROUTE_HARD_EPSILON_M if self._target_route_recovery_enabled else 0.0,
                "target_route_recovery_reachable_tube_margin_m": TARGET_ROUTE_REACHABLE_TUBE_MARGIN_M if self._target_route_recovery_enabled else 0.0,
                "target_route_recovery_hysteresis_m": RECOVERY_HYSTERESIS_M if self._target_route_recovery_enabled else 0.0,
                "target_route_recovery_stop_speed_mps": RECOVERY_STOP_SPEED_MPS if self._target_route_recovery_enabled else 0.0,
                "target_route_recovery_brake_lateral_tube_p95_m": (
                    self._recovery_brake_lateral_tube_p95_m if self._target_route_recovery_enabled else 0.0
                ),
                "target_route_braking_v3_schema": (
                    TARGET_ROUTE_BRAKING_V3_SCHEMA if self._target_route_braking_v3_enabled else "off"
                ),
                "target_route_braking_v3_model": (
                    TARGET_ROUTE_BRAKING_V3_MODEL if self._target_route_braking_v3_enabled else "off"
                ),
                "target_route_braking_v3_soft_envelope": (
                    TARGET_ROUTE_BRAKING_GEOMETRY_SCHEMA if self._target_route_braking_v3_enabled else "off"
                ),
                "cv_initial_heading": self._eval_cv_initial_heading,
                "target_speed_mode": target_speed_mode,
                "target_speed_mps": target_speed_mps,
                "target_speed_min_mps": speed_min,
                "target_speed_max_mps": speed_max,
                # Historical field retained for readers that already consume it.  It is a
                # per-axis command limit, not a vector-norm limit.
                "pursuer_max_speed_mps": float(self.task_config.max_velocity),
                "pursuer_speed_limit_semantics": "per_axis_xy",
                "pursuer_per_axis_speed_limit_mps": float(
                    self.task_config.max_velocity
                ),
                "pursuer_max_horizontal_request_norm_mps": float(
                    math.sqrt(2.0) * self.task_config.max_velocity
                ),
                "policy_output_dim": 4,
                "policy_z_output_overwritten_by_altitude_pi": True,
                "policy_z_persisted_in_prev_action_observation": True,
                "oob_margin_m": float(self.vis_cfg.oob_margin),
                "episode_len_steps": int(self.task_config.episode_len_steps),
                "num_envs": int(self.num_envs),
                "goal_dist_min_m": goal_dist_min,
                "goal_dist_max_m": goal_dist_max,
                "full_goal_distribution": bool(full_goal_distribution),
                "fov_curriculum_saturated": bool(fov_curriculum_saturated),
                "evaluation_nonce": os.environ.get("NAVRL_EVAL_RUN_NONCE", ""),
                "speed_governor_mode": self.speed_governor_cfg.mode,
                "speed_governor_fixed_mps": self.speed_governor_cfg.fixed_cap_mps,
                "speed_governor_free_mps": self.speed_governor_cfg.free_speed_cap_mps,
                "speed_governor_half_width_m": self.speed_governor_cfg.path_half_width_m,
                "speed_governor_margin_m": self.speed_governor_cfg.hard_margin_m,
                "speed_governor_slow_m": self.speed_governor_cfg.slow_distance_m,
                "speed_governor_release_m": self.speed_governor_cfg.release_distance_m,
                "speed_governor_ttc_s": self.speed_governor_cfg.ttc_s,
                "speed_governor_brake_mps2": self.speed_governor_cfg.brake_mps2,
                "speed_governor_reaction_s": self.speed_governor_cfg.reaction_s,
                "speed_governor_width_per_mps": self.speed_governor_cfg.width_per_mps,
                "speed_governor_width_per_open_m": self.speed_governor_cfg.width_per_open_m,
                "speed_governor_width_open_ref_m": self.speed_governor_cfg.width_open_ref_m,
                "speed_governor_width_max_m": self.speed_governor_cfg.width_max_m,
                "speed_governor_lateral_margin_m": self.speed_governor_cfg.lateral_margin_m,
                "speed_governor_lateral_span_m": self.speed_governor_cfg.lateral_span_m,
                "speed_governor_lateral_floor_mps": self.speed_governor_cfg.lateral_floor_mps,
                "speed_governor_yaw_cap_radps": self.speed_governor_cfg.yaw_cap_radps,
                "speed_governor_yaw_cap_margin_m": self.speed_governor_cfg.yaw_cap_margin_m,
                "speed_governor_target_exclusion": "camera_lidar_association",
                "p9_empirical_error_enabled": bool(
                    getattr(getattr(self, "perception", None), "empirical_error", None) is not None
                ),
                "p9_empirical_error_sha256": (
                    self.perception.empirical_error.sha256
                    if getattr(getattr(self, "perception", None), "empirical_error", None) is not None else ""
                ),
                "p9_empirical_error_seed": int(
                    getattr(self.perception_cfg, "empirical_error_seed", 1701)
                ),
                "search_state": str(representation["search_state"]),
                "search_state_masked": bool(
                    representation["search_state_force_invalid"]
                ),
                "search_state_telemetry": bool(self._s1_search_telemetry_enabled),
                "target_render_mode": (
                    self.detector.target_render_mode if self.detector is not None else "none"
                ),
                "d8_dynamic_mesh_treatment": bool(
                    self.detector is not None
                    and self.detector._dynamic_mesh_treatment is not None
                ),
                "d8_dynamic_mesh_asset_sha256": self._d8_treatment_asset_sha256,
                "d8_dynamic_mesh_contract": (
                    self.detector._dynamic_mesh_treatment.contract()
                    if self.detector is not None
                    and self.detector._dynamic_mesh_treatment is not None
                    else None
                ),
                **physics,
            },
            "outcome": {
                "captured": int(self._succ_agg),
                "crash": int(self._crash_agg),
                "timeout": int(self._to_agg),
                "capture_rate": float(self._succ_agg / max(1, total)),
                "crash_rate": float(self._crash_agg / max(1, total)),
                "timeout_rate": float(self._to_agg / max(1, total)),
                "ever_reached_rate": float(reach_rate),
                "closest_nocrash_mean_m": float(mean_nc),
                "closest_nocrash_best_m": None if math.isnan(best) else float(best),
                "closest_nocrash_count": int(self._nc_agg),
            },
            "search_state": {
                "arm": str(representation["search_state"]),
                "masked": bool(representation["search_state_force_invalid"]),
                "blind_phase_samples": int(self._s1_blind_samples.item()),
                "blind_phase_mean_speed_mps": (
                    float(self._s1_blind_speed_sum.item() / self._s1_blind_samples.item())
                    if int(self._s1_blind_samples.item()) > 0 else None
                ),
                "blind_phase_bar_clearance_mean_m": (
                    float(self._s1_blind_clearance_sum.item() / self._s1_blind_samples.item())
                    if int(self._s1_blind_samples.item()) > 0 else None
                ),
                "first_visible": {
                    label: {
                        "coverage_fraction_mean": (
                            float(self._s1_first_coverage_sum[index].item() / n_cov)
                            if (n_cov := int(self._s1_first_coverage_n[index].item())) > 0
                            else None
                        ),
                        "coverage_samples": n_cov,
                        "belief_entropy_mean": (
                            float(self._s1_first_entropy_sum[index].item() / n_entropy)
                            if (n_entropy := int(self._s1_first_entropy_n[index].item())) > 0
                            else None
                        ),
                        "belief_entropy_samples": n_entropy,
                    }
                    for index, label in enumerate(("capture", "crash", "timeout"))
                },
            },
            "target_motion": {
                "cv_initial_heading": self._eval_cv_initial_heading,
                "initial_heading_samples": int(self._eval_cv_heading_diag["samples"]),
                "initial_heading_mean_radial_cos": (
                    self._eval_cv_heading_diag["radial_cos_sum"]
                    / max(1, self._eval_cv_heading_diag["samples"])
                ),
                "initial_heading_mean_radial_sin": (
                    self._eval_cv_heading_diag["radial_sin_sum"]
                    / max(1, self._eval_cv_heading_diag["samples"])
                ),
                "initial_heading_max_contract_error": float(
                    self._eval_cv_heading_diag["max_contract_error"]
                ),
                "outcome_telemetry": {
                    label: target_motion_outcome_payload(index)
                    for index, label in enumerate(("capture", "crash", "timeout"))
                },
                "oob_exit_forensics": oob_exit_payload(),
                "first_acquisition": {
                    label: first_acquisition_payload(index)
                    for index, label in enumerate(("capture", "crash", "timeout"))
                },
                "speed_by_wall_reflection": {
                    speed_label: {
                        reflection_label: speed_wall_payload(
                            speed_index, reflection_index
                        )
                        for reflection_index, reflection_label in enumerate(
                            ("zero", "one_or_more")
                        )
                    }
                    for speed_index, speed_label in enumerate(("q0", "q1", "q2", "q3"))
                },
                "route": (
                    self._target_route_manager.diagnostics()
                    if self._target_route_enabled
                    else {"mode": TARGET_ROUTE_MODE_OFF}
                ),
            },
            "action": {
                "policy": os.environ.get("NAVRL_ACTION_POLICY", "legacy"),
                "samples": int(ad["n"]),
                "task_input_oob_rate": [
                    float(value / n_action) for value in ad["raw_oob"]
                ],
                "executed_edge98_rate": [
                    float(value / n_action) for value in ad["exec_edge"]
                ],
                "executed_edge95_rate": [
                    float(value / n_action) for value in ad["exec_edge95"]
                ],
                "executed_edge99_rate": [
                    float(value / n_action) for value in ad["exec_edge99"]
                ],
                "mean_abs": [float(value / n_action) for value in ad["abs_sum"]],
                "signed_mean_y": float(ad["signed_y_sum"] / n_action),
                "positive_y_rate": float(ad["positive_y"] / n_action),
                "negative_y_rate": float(ad["negative_y"] / n_action),
                "high80_y_rate": float(ad["high80_y"] / n_action),
                "mean_abs_delta_y": float(ad["delta_y_sum"] / n_delta),
                "sign_flip_y_rate": float(ad["sign_flip_y"] / n_delta),
                "context": {
                    name: {
                        "samples": int(ad[name + "_n"]),
                        "fraction": float(ad[name + "_n"] / n_action),
                        "mean_abs_y": float(
                            ad[name + "_abs_y"] / max(1.0, ad[name + "_n"])
                        ),
                    }
                    for name in (
                        "front_clear",
                        "front_blocked",
                        "goal_centered",
                        "goal_offcenter",
                        "clear_centered",
                        "target_visible",
                        "target_hidden",
                    )
                },
                "motion": {
                    "samples": int(ad["motion_n"]),
                    "mean_speed_mps": float(
                        ad["motion_speed_sum"] / max(1.0, ad["motion_n"])
                    ),
                    "mean_command_speed_mps": float(
                        ad["motion_command_speed_sum"] / max(1.0, ad["motion_n"])
                    ),
                    "low_speed_rate": float(
                        ad["motion_low_speed"] / max(1.0, ad["motion_n"])
                    ),
                    "commanded_stall_rate": float(
                        ad["motion_commanded_stall"] / max(1.0, ad["motion_n"])
                    ),
                },
            },
            "crash_causes": {
                "count": int(n_crash_causes),
                "bar_contact": int(d["contact"]),
                "below": int(d["below"]),
                "above": int(d["above"]),
                "out_of_bounds": int(d["oob"]),
                "bar_contact_share": float(d["contact"] / n_cause_den),
                "below_share": float(d["below"] / n_cause_den),
                "above_share": float(d["above"] / n_cause_den),
                "out_of_bounds_share": float(d["oob"] / n_cause_den),
            },
            "speed_governor": {
                "mode": self.speed_governor_cfg.mode,
                "sensor_only": True,
                "direction_preserved": True,
                "target_exclusion_source": "camera_lidar_association",
                "feedback_executed_previous_action": self.speed_governor_cfg.mode != "off",
                "samples": int(scalar("samples")),
                "intervention_rate": mean_scalar("interventions"),
                "near_stop_rate": mean_scalar("near_stops"),
                "mean_requested_speed_mps": mean_scalar("requested_speed_sum"),
                "mean_executed_speed_mps": mean_scalar("executed_speed_sum"),
                "mean_clearance_m": mean_scalar("clearance_sum"),
                "mean_scale": mean_scalar("scale_sum"),
                "mean_requested_ttc_s": mean_scalar("ttc_sum", "ttc_n"),
                "negative_stopping_margin_requested_rate": mean_scalar(
                    "negative_margin_requested"
                ),
                "negative_stopping_margin_executed_rate": mean_scalar(
                    "negative_margin_executed"
                ),
                "contact": {
                    "count": int(scalar("contact_n")),
                    "mean_actual_speed_mps": mean_scalar(
                        "contact_actual_speed_sum", "contact_n"
                    ),
                    "mean_requested_speed_mps": mean_scalar(
                        "contact_requested_speed_sum", "contact_n"
                    ),
                    "mean_executed_speed_mps": mean_scalar(
                        "contact_executed_speed_sum", "contact_n"
                    ),
                    "mean_clearance_m": mean_scalar(
                        "contact_clearance_sum", "contact_n"
                    ),
                    "mean_scale": mean_scalar("contact_scale_sum", "contact_n"),
                    "mean_requested_ttc_s": mean_scalar(
                        "contact_ttc_sum", "contact_n"
                    ),
                    "mean_stopping_margin_requested_m": mean_scalar(
                        "contact_margin_requested_sum", "contact_n"
                    ),
                    "mean_stopping_margin_executed_m": mean_scalar(
                        "contact_margin_executed_sum", "contact_n"
                    ),
                    "mean_step": float(
                        self._diag_steps["contact"] / max(1, d["contact"])
                    ),
                },
                "outcome_steps": {
                    label: step_summary(self._speed_governor_outcome_steps[label])
                    for label in ("capture", "crash", "timeout")
                },
            },
            "strata": {
                # Evaluation speed bins cover the distribution actually applied.  Curriculum
                # training counters retain their historical [0,max] definition separately.
                "speed_bin_edges_mps": [
                    float(speed_min + (speed_max - speed_min) * i / 4.0)
                    for i in range(5)
                ],
                "speed": stratum_payload(
                    self._eval_speed_succ,
                    self._eval_speed_crash,
                    self._eval_speed_timeout,
                    self._eval_speed_crash_cause,
                    self._eval_speed_fin,
                    ("q0", "q1", "q2", "q3"),
                ),
                "distance_bin_edges_m": [
                    float(goal_dist_min + (goal_dist_max - goal_dist_min) * i / 4.0)
                    for i in range(5)
                ],
                "distance": stratum_payload(
                    self._eval_dist_succ,
                    self._eval_dist_crash,
                    self._eval_dist_timeout,
                    self._eval_dist_crash_cause,
                    self._eval_dist_fin,
                    ("q0", "q1", "q2", "q3"),
                ),
                "pattern": stratum_payload(
                    self._eval_pattern_succ,
                    self._eval_pattern_crash,
                    self._eval_pattern_timeout,
                    self._eval_pattern_crash_cause,
                    self._eval_pattern_fin,
                    ("cv", "waypoint", "circle"),
                ),
                "distance_by_pattern": {
                    distance_label: stratum_payload(
                        self._eval_dist_pattern_succ[distance_index],
                        self._eval_dist_pattern_crash[distance_index],
                        self._eval_dist_pattern_timeout[distance_index],
                        self._eval_dist_pattern_crash_cause[distance_index],
                        self._eval_dist_pattern_fin[distance_index],
                        ("cv", "waypoint", "circle"),
                    )
                    for distance_index, distance_label in enumerate(
                        ("q0", "q1", "q2", "q3")
                    )
                },
                "initial_target_bearing": {
                    label: {
                        "captured": int(self._eval_bearing_succ[i].item()),
                        "crash": int(self._eval_bearing_crash[i].item()),
                        "timeout": int(self._eval_bearing_timeout[i].item()),
                        "episodes": int(self._eval_bearing_fin[i].item()),
                        "capture_rate": (
                            float(
                                self._eval_bearing_succ[i].item()
                                / self._eval_bearing_fin[i].item()
                            )
                            if self._eval_bearing_fin[i].item() > 0
                            else None
                        ),
                    }
                    for i, label in enumerate(
                        ("negative_y", "centered_5deg", "positive_y")
                    )
                },
            },
        }
        if self._joint_speed_telemetry is not None:
            payload["condition"]["joint_speed_telemetry"] = True
            payload["joint_speed_allocation"] = self._joint_speed_telemetry.payload(
                (self._succ_agg, self._crash_agg, self._to_agg),
                expected_bar_contacts=d["contact"],
            )
        # The appearance-distractor axis, attested by the process that BUILT the scene rather than
        # by an echo of the request.  Unconditional and 0 in the historical world, so a cell that
        # ran with no distractors says so instead of being silent about it -- which is what lets a
        # cross-cell verifier prove the axis is the only thing that moved.
        payload["condition"]["distractor_count"] = int(self._num_distractors)
        if self._distractor_metric_enabled:
            payload["distractor_lock"] = self._distractor_lock_payload()
        if self._s1_shadow_enabled:
            payload["condition"]["s1_shadow"] = True
            payload["s1_shadow"] = self._s1_shadow_payload()
        if self._contact_geom_enabled:
            payload["condition"]["contact_geometry"] = True
            payload["contact_geometry"] = self._contact_geometry_payload()
            if self._cg_sc_enabled:
                payload["condition"]["star_convex_shadow"] = True
                payload["star_convex_shadow"] = self._star_convex_shadow_payload()
        compact = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        print("NAVRL_BULK_EVAL_RESULT " + compact, flush=True)
        self._export_episode_forensics()

        if not self._bulk_eval_output:
            logger.warning(
                "NavRL bulk eval result was printed but NAVRL_BULK_EVAL_JSON is unset."
            )
            self._bulk_eval_exported = True
            return
        try:
            out = Path(self._bulk_eval_output)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            tmp.replace(out)
            logger.warning("NavRL bulk eval results saved -> %s" % out)
            self._bulk_eval_exported = True
        except OSError as exc:
            logger.warning("NavRL bulk eval results export failed: %s" % exc)

    def _log_progress(self, successes, crashes, timeouts, finished=None):
        self._succ_agg += int(torch.sum(successes).item())
        self._crash_agg += int(torch.sum(crashes > 0).item())
        self._to_agg += int(torch.sum(timeouts).item())
        if finished is not None and finished.any():
            # Density curriculum is disabled during a held-out sweep, but the same episode labels
            # are essential for diagnosing whether a score is limited by speed, initial distance,
            # or motion pattern. Keep these counters separate from the checkpointed curriculum
            # window so held-out denominators cannot inherit training episodes.
            if self._bulk_eval_mode:
                self._record_eval_outcome_strata(successes, crashes, timeouts, finished)
                self._record_target_motion_outcome_telemetry(
                    successes, crashes, timeouts, finished
                )
                idx = finished.nonzero(as_tuple=False).squeeze(1)
                bins = self._episode_bearing_bin[idx].long().clamp(0, 2)
                self._eval_bearing_fin += torch.bincount(
                    bins, minlength=3
                ).to(self._eval_bearing_fin.dtype)
                self._eval_bearing_succ += torch.bincount(
                    bins,
                    weights=successes[idx].to(torch.float32),
                    minlength=3,
                ).to(self._eval_bearing_succ.dtype)
                self._eval_bearing_crash += torch.bincount(
                    bins,
                    weights=(crashes[idx] > 0).to(torch.float32),
                    minlength=3,
                ).to(self._eval_bearing_crash.dtype)
                self._eval_bearing_timeout += torch.bincount(
                    bins,
                    weights=timeouts[idx].to(torch.float32),
                    minlength=3,
                ).to(self._eval_bearing_timeout.dtype)
            self._reach_agg += int(torch.sum(self.ep_reached & finished).item())
            nocrash = finished & ~(crashes > 0)
            if nocrash.any():
                self._mindist_sum += float(torch.sum(self.ep_min_goal_dist[nocrash]).item())
                self._nc_agg += int(torch.sum(nocrash).item())
                m = float(torch.min(self.ep_min_goal_dist[nocrash]).item())
                self._closest_min = m if self._closest_min is None else min(self._closest_min, m)
            self._fin_agg += int(torch.sum(finished).item())
        total = self._succ_agg + self._crash_agg + self._to_agg
        if total >= self._progress_log_interval:
            reach_rate = self._reach_agg / max(1, self._fin_agg)
            mean_nc = self._mindist_sum / max(1, self._nc_agg)
            best = self._closest_min if self._closest_min is not None else float("nan")
            _run = os.environ.get("AERIAL_RUN_NAME", "").strip()
            logger.warning(
                "NavRL progress%s | captured=%.3f ever_reached=%.3f crash=%.3f timeout=%.3f "
                "closest_nocrash=%.2fm best=%.2fm (n=%d)"
                % (
                    (" [" + _run + "]") if _run else "",
                    self._succ_agg / total,
                    reach_rate,
                    self._crash_agg / total,
                    self._to_agg / total,
                    mean_nc,
                    best,
                    total,
                )
            )
            self._export_bulk_eval_result(total, reach_rate, mean_nc, best)
            if self._action_diag_enabled and self._action_diag["n"] > 0:
                ad = self._action_diag
                n_action = max(1, int(ad["n"]))
                n_delta = max(1, int(ad["delta_y_n"]))
                raw_oob = [value / n_action for value in ad["raw_oob"]]
                exec_edge = [value / n_action for value in ad["exec_edge"]]
                mean_abs = [value / n_action for value in ad["abs_sum"]]
                logger.warning(
                    "NavRL actiondiag | policy=%s std=%s "
                    "task_input_oob[x,y,z,yaw]=[%.4f,%.4f,%.4f,%.4f] "
                    "exec_edge98=[%.4f,%.4f,%.4f,%.4f] "
                    "mean_abs=[%.3f,%.3f,%.3f,%.3f] signed_y=%.3f "
                    "pos_y=%.3f neg_y=%.3f high80_y=%.3f "
                    "delta_y=%.3f sign_flip_y=%.3f (n=%d)"
                    % (
                        os.environ.get("NAVRL_ACTION_POLICY", "legacy"),
                        os.environ.get("NAVRL_ACTION_STD", "learned"),
                        *raw_oob,
                        *exec_edge,
                        *mean_abs,
                        ad["signed_y_sum"] / n_action,
                        ad["positive_y"] / n_action,
                        ad["negative_y"] / n_action,
                        ad["high80_y"] / n_action,
                        ad["delta_y_sum"] / n_delta,
                        ad["sign_flip_y"] / n_delta,
                        n_action,
                    )
                )
                logger.warning(
                    "NavRL actioncontext | clear=%.3f/|y|%.3f blocked=%.3f/|y|%.3f "
                    "centered=%.3f/|y|%.3f offcenter=%.3f/|y|%.3f "
                    "clear_centered=%.3f/|y|%.3f visible=%.3f/|y|%.3f"
                    % (
                        ad["front_clear_n"] / n_action,
                        ad["front_clear_abs_y"] / max(1.0, ad["front_clear_n"]),
                        ad["front_blocked_n"] / n_action,
                        ad["front_blocked_abs_y"] / max(1.0, ad["front_blocked_n"]),
                        ad["goal_centered_n"] / n_action,
                        ad["goal_centered_abs_y"] / max(1.0, ad["goal_centered_n"]),
                        ad["goal_offcenter_n"] / n_action,
                        ad["goal_offcenter_abs_y"] / max(1.0, ad["goal_offcenter_n"]),
                        ad["clear_centered_n"] / n_action,
                        ad["clear_centered_abs_y"] / max(1.0, ad["clear_centered_n"]),
                        ad["target_visible_n"] / n_action,
                        ad["target_visible_abs_y"] / max(1.0, ad["target_visible_n"]),
                    )
                )
                n_motion = max(1.0, ad["motion_n"])
                logger.warning(
                    "NavRL motiondiag | speed=%.3fm/s command=%.3fm/s "
                    "low_speed=%.4f commanded_stall=%.4f (n=%d)"
                    % (
                        ad["motion_speed_sum"] / n_motion,
                        ad["motion_command_speed_sum"] / n_motion,
                        ad["motion_low_speed"] / n_motion,
                        ad["motion_commanded_stall"] / n_motion,
                        int(ad["motion_n"]),
                    )
                )
                self._action_diag = self._empty_action_diag()
            if self._speed_governor_diag_enabled:
                gd = self._speed_governor_diag
                gn = max(1.0, float(gd["samples"].item()))
                gc = max(1.0, float(gd["contact_n"].item()))
                logger.warning(
                    "NavRL speedgov | mode=%s intervene=%.3f stop=%.3f "
                    "requested=%.3fm/s executed=%.3fm/s clearance=%.2fm scale=%.3f "
                    "unsafe_pre=%.3f unsafe_post=%.3f | contact_n=%d actual=%.3fm/s "
                    "executed=%.3fm/s clearance=%.2fm"
                    % (
                        self.speed_governor_cfg.mode,
                        float(gd["interventions"].item()) / gn,
                        float(gd["near_stops"].item()) / gn,
                        float(gd["requested_speed_sum"].item()) / gn,
                        float(gd["executed_speed_sum"].item()) / gn,
                        float(gd["clearance_sum"].item()) / gn,
                        float(gd["scale_sum"].item()) / gn,
                        float(gd["negative_margin_requested"].item()) / gn,
                        float(gd["negative_margin_executed"].item()) / gn,
                        int(gd["contact_n"].item()),
                        float(gd["contact_actual_speed_sum"].item()) / gc,
                        float(gd["contact_executed_speed_sum"].item()) / gc,
                        float(gd["contact_clearance_sum"].item()) / gc,
                    )
                )
                self._speed_governor_diag = self._empty_speed_governor_diag()
            if self._crash_diag:
                d = self._diag
                n_raw = d["contact"] + d["below"] + d["above"] + d["oob"]
                n_all = max(1, n_raw)  # division guard only; the printed count is the raw sum
                logger.warning(
                    "NavRL crashdiag | bar_contact=%.3f (mean_x=%.1fm steps=%.0f) below=%.3f "
                    "(steps=%.0f tilt=%.0fdeg) "
                    "above=%.3f oob=%.3f [W=%d E=%d S=%d N=%d steps=%.0f] (n_crash=%d)"
                    % (
                        d["contact"] / n_all,
                        self._diag_x_sum / max(1, d["contact"]),
                        self._diag_steps["contact"] / max(1, d["contact"]),
                        d["below"] / n_all,
                        self._diag_steps["below"] / max(1, d["below"]),
                        self._diag_below_tilt / max(1, d["below"]),
                        d["above"] / n_all,
                        d["oob"] / n_all,
                        d["oob_w"],
                        d["oob_e"],
                        d["oob_s"],
                        d["oob_n"],
                        self._diag_steps["oob"] / max(1, d["oob"]),
                        n_raw,
                    )
                )
                if self._oob_probe and self._probe["n"] > 0:
                    p = self._probe
                    n_probe = p["n"]
                    logger.warning(
                        "NavRL oobprobe | lateral_n=%d start_y=%.2fm "
                        "goal_pull_side=%.2fm goal_now_pull_side=%.2fm "
                        "bar_bias_side=%.3f outward_vy=%.2fm/s outward_cmd_vy=%.2fm/s "
                        "action_y_side=%.3f excursion=%.2fm visible=%.3f "
                        "track_age=%.2fs track_cov_pos=%.3f"
                        % (
                            int(n_probe),
                            p["start_y"] / n_probe,
                            p["goal_pull_side"] / n_probe,
                            p["goal_now_pull_side"] / n_probe,
                            p["bar_bias_side"] / n_probe,
                            p["world_vy_side"] / n_probe,
                            p["command_vy_side"] / n_probe,
                            p["action_y_side"] / n_probe,
                            p["excursion_side"] / n_probe,
                            p["visible"] / n_probe,
                            p["track_age"] / n_probe,
                            p["track_cov_pos"] / n_probe,
                        )
                    )
                if self._bar_probe and self._bprobe["n"] > 0:
                    bp = self._bprobe
                    nb = bp["n"]
                    n_match = max(1.0, bp["hit_in_tokens"])
                    n_hit_fov = max(1.0, bp["hit_in_token_fov"])
                    logger.warning(
                        "NavRL barprobe v2 | n=%d bars_range=%.1f bars_fov=%.1f occupied_bins=%.1f "
                        "hit_dist=%.2fm hit_fov=%.3f hit_token=%.3f hit_token_given_fov=%.3f "
                        "tokens=%.1f associated=%.1f unique=%.1f duplicate=%.1f "
                        "center_offset=%.2fm cross_track=%.2fm radial_gap=%.2fm rank=%.1f "
                        "(capacity=%d)"
                        % (
                            int(nb),
                            bp["bars_in_range"] / nb,
                            bp["bars_in_token_fov"] / nb,
                            bp["occupied_bins"] / nb,
                            bp["hit_dist"] / nb,
                            bp["hit_in_token_fov"] / nb,
                            bp["hit_in_tokens"] / nb,
                            bp["hit_in_tokens_in_fov"] / n_hit_fov,
                            bp["valid_tokens"] / nb,
                            bp["associated_tokens"] / nb,
                            bp["unique_token_bars"] / nb,
                            bp["duplicate_tokens"] / nb,
                            bp["hit_center_offset"] / n_match,
                            bp["hit_cross_track"] / n_match,
                            bp["hit_radial_gap"] / n_match,
                            bp["hit_token_rank"] / n_match,
                            self._max_obstacles_or_zero(),
                        )
                    )
                self._diag = {k: 0 for k in self._diag}
                self._diag_steps = {"contact": 0.0, "oob": 0.0, "below": 0.0}
                self._diag_below_tilt = 0.0
                self._bprobe = {k: 0.0 for k in self._bprobe}
                self._diag_x_sum = 0.0
                self._probe = {k: 0.0 for k in self._probe}
            self._succ_agg = self._crash_agg = self._to_agg = 0
            self._reach_agg = self._fin_agg = 0
            self._mindist_sum = 0.0
            self._nc_agg = 0
            self._closest_min = None
