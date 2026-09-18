'use strict';
/*
 * Browser GT tracking-preview logic validation (Phase 1).
 *
 * This is an ENGINEERING validation of docs/status/arena_demo_planner.js. It is
 * not a PPO experiment, not PhysX, and not research performance evidence. It
 * replays the same deterministic browser modules the public page loads, over a
 * density x target-speed x seed matrix, and records hard invariants plus
 * tracking diagnostics.
 *
 * Usage: node tools/validate_gt_browser_tracking.js [--out FILE] [--seeds N]
 *                                                   [--duration S] [--quick]
 */

const fs = require('fs');
const path = require('path');

// --planner-dir lets the SAME harness run against a snapshot of the browser
// modules (e.g. `git show HEAD:...`), which is how before/after comparisons
// are made with one measuring instrument.
function argValue(name, fallback) {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] != null ? process.argv[i + 1] : fallback;
}
const PLANNER_DIR = path.resolve(argValue('--planner-dir', path.join(__dirname, '..', 'docs', 'status')));
const Motion = require(path.join(PLANNER_DIR, 'arena_motion.js'));
const Route = require(path.join(PLANNER_DIR, 'arena_route.js'));
const Planner = require(path.join(PLANNER_DIR, 'arena_demo_planner.js'));

const STATUS_PATH = fs.existsSync(path.join(PLANNER_DIR, 'status.json'))
  ? path.join(PLANNER_DIR, 'status.json')
  : path.join(__dirname, '..', 'docs', 'status', 'status.json');
const STATUS = JSON.parse(fs.readFileSync(STATUS_PATH, 'utf8'));
const GEO = STATUS.arena_geometry;
// Termination contract with provenance: status.json carries success_radius_m
// once the snapshot is regenerated; until then the research value 0.5 m
// (navrl_task_config.success_radius) is the documented fallback.
const CAPTURE_RADIUS_M = Number(argValue('--capture-radius', GEO.success_radius_m || 0.5));
const TIMEOUT_S = Number(argValue('--timeout',
  Number(GEO.episode_len_steps || 600) * Number(GEO.rl_step_dt_s || 0.1)));
const SUPPORTS_EPISODES = typeof Planner.applyEpisodeOutcome === 'function';

// Same contract the page applies through Arena.configure/Motion.configure.
Motion.configure(GEO);

const ARENA = Number(GEO.arena_xy_m);
const X0 = 0, X1 = ARENA, Y0 = -ARENA / 2, Y1 = ARENA / 2;
const BX0 = X0 + Number(GEO.bar_x_min_ratio) * (X1 - X0);
const BX1 = X0 + Number(GEO.bar_x_max_ratio) * (X1 - X0);
const TOUCH_M = Number(GEO.placement_touch_m);
const GAP_M = Number(GEO.placement_gap_m);
const ARENA_LO = {x: X0, y: Y0};
const ARENA_HI = {x: X1, y: Y1};

const DT = 0.1;                       // fixed 10 Hz browser simulation clock
const EPS = 1e-9;

// Exact port of arena.js placeBars() for PLACEMENT === 'navrl_band'.
function placeBars(n, layoutSeed) {
  const r = Motion.seededRng(layoutSeed);
  const pts = [];
  const bw = () => 0.4 + r() * 0.4;
  let guard = 0, fails = 0;
  while (pts.length < n && guard < n * 600) {
    guard++;
    const x = BX0 + r() * (BX1 - BX0), y = Y0 + r() * (Y1 - Y0);
    let ok = true;
    for (const p of pts) {
      const d = Math.hypot(x - p.x, y - p.y);
      if (d > TOUCH_M && d < GAP_M) { ok = false; break; }
    }
    if (ok) { pts.push({x, y, w: bw()}); fails = 0; continue; }
    if (++fails >= 400 && pts.length) {
      const a = pts[Math.floor(r() * pts.length)];
      const ang = r() * Math.PI * 2, rad = 0.5 * TOUCH_M * r();
      pts.push({
        x: Math.min(BX1, Math.max(BX0, a.x + rad * Math.cos(ang))),
        y: Math.min(Y1, Math.max(Y0, a.y + rad * Math.sin(ang))),
        w: bw(),
      });
      fails = 0;
    }
  }
  return pts;
}

function hypot(x, y) { return Math.hypot(x, y); }
function finite(v) { return Number.isFinite(v); }

function agentFinite(a) {
  return finite(a.x) && finite(a.y) && finite(a.vx) && finite(a.vy)
    && finite(a.heading) && finite(a.roll) && finite(a.pitch);
}

// Raw (un-inflated) obstacle body overlap: a true penetration, independent of
// the A* tracking margin.
function penetratesRawBar(p, radius, bars) {
  for (const b of bars) {
    const hx = 0.5 * b.w + radius, hy = 0.5 * (b.h != null ? b.h : b.w) + radius;
    if (Math.abs(p.x - b.x) < hx - 1e-9 && Math.abs(p.y - b.y) < hy - 1e-9) return true;
  }
  return false;
}

function outsideWalls(p, radius) {
  return p.x - radius < X0 - 1e-9 || p.x + radius > X1 + 1e-9
    || p.y - radius < Y0 - 1e-9 || p.y + radius > Y1 + 1e-9;
}

function routeSegmentsSafe(route, cache) {
  if (!route || !route.valid || route.waypoints.length < 2) return true;
  for (let i = 1; i < route.waypoints.length; i++) {
    if (!Planner.segmentSafe(route.waypoints[i - 1], route.waypoints[i], cache)) return false;
  }
  return true;
}

function quantiles(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function newViolations() {
  return {
    nan_or_inf: 0,
    target_obstacle_penetration: 0,
    tracker_obstacle_penetration: 0,
    wall_violation: 0,
    teleport: 0,
    speed_limit_violation: 0,
    acceleration_limit_violation: 0,
    turn_rate_violation: 0,
    unsafe_route_segment: 0,
    command_without_safe_route: 0,
    blind_motion_after_route_invalidation: 0,
    // interception terminal-correctness invariants (0 in continuous mode)
    capture_without_criterion: 0,
    missed_capture: 0,
    motion_during_terminal: 0,
    reset_count_mismatch: 0,
    stale_state_after_reset: 0,
  };
}

function addViolations(into, from) {
  for (const k of Object.keys(from)) into[k] += from[k];
}

function newSession(bars, motionRng, speed, mode) {
  const episode = Motion.createEpisode(motionRng, bars, speed);
  return Planner.createSession({
    bars: bars,
    arenaLo: ARENA_LO,
    arenaHi: ARENA_HI,
    rng: motionRng,
    speed: speed,
    target: {x: episode.target.x, y: episode.target.y, heading: episode.heading},
    pursuer: {x: episode.drone.x, y: episode.drone.y, heading: 0},
    goalMode: 'auto',
    episodeMode: mode,
    captureRadiusM: CAPTURE_RADIUS_M,
    timeoutS: TIMEOUT_S,
  });
}

function sessionIsFresh(session) {
  const t = session.target, p = session.pursuer;
  return t.route == null && t.goal == null && p.route == null && p.lead == null
    && p.carrot == null && p.replanCount === 0 && (t.goalCompletions || 0) === 0
    && (!session.episode || (session.episode.terminal == null
      && session.episode.resetDue === false && session.episode.outcome === 'RUNNING'));
}

function runCell(bars, seed, speed, durationS, opts) {
  const steps = Math.round(durationS / DT);
  const mode = (opts && opts.episodeMode) || 'continuous';
  const motionRng = Motion.seededRng(seed);
  let session = newSession(bars, motionRng, speed, mode);

  const v = newViolations();
  let t = session.target, p = session.pursuer;
  const episodes = [];
  let terminals = 0, resets = 0, terminalSteps = 0;
  // goal-handoff instrumentation: target speed history and completion events
  const targetSpeedHist = [];
  const handoffEvents = [];
  let lastGoalCompletions = t.goalCompletions || 0;
  let dwellBelow005 = 0;
  const targetRadius = Planner.TARGET_SUPPORT.x;
  const trackerRadius = Planner.PURSUER_SUPPORT.x;
  const maxAccel = Motion.CONTRACT.boundedMaxAccel;
  const maxTurn = Motion.CONTRACT.boundedMaxTurnRate;
  const trackerSpeedMax = Motion.CONTRACT.pursuerSpeedMax;

  const distances = [];
  let decreasingSteps = 0, comparableSteps = 0;
  let stallSteps = 0, validRouteStallSteps = 0, noRouteSteps = 0, losSteps = 0;
  let headingSignFlips = 0, lastTurnSign = 0;
  let plannerCalls = 0;
  const digest = [];
  let prevTrackerRouteSerial = p.routeSerial;
  let prevRouteValid = false;
  let prevD = hypot(p.x - t.x, p.y - t.y);
  const initialDistance = prevD;
  let prevHeading = p.heading;

  for (let step = 0; step < steps; step++) {
    // Reset happens BETWEEN steps so no invariant is checked across a respawn.
    if (SUPPORTS_EPISODES && session.episode && session.episode.resetDue) {
      resets += 1;
      session = newSession(bars, motionRng, speed, mode);
      t = session.target; p = session.pursuer;
      if (!sessionIsFresh(session)) v.stale_state_after_reset += 1;
      prevD = hypot(p.x - t.x, p.y - t.y);
      prevHeading = p.heading;
      prevRouteValid = false;
      prevTrackerRouteSerial = p.routeSerial;
    }
    const inTerminal = Boolean(SUPPORTS_EPISODES && session.episode && session.episode.terminal);
    const outcomeBefore = inTerminal ? session.episode.terminal.outcome : 'RUNNING';
    const before = {
      t: {x: t.x, y: t.y, vx: t.vx, vy: t.vy, heading: t.heading},
      p: {x: p.x, y: p.y, vx: p.vx, vy: p.vy, heading: p.heading},
    };
    const routeWasValid = Boolean(p.route && p.route.valid);
    const replansBefore = p.replanCount;

    Planner.stepSession(session, DT);

    plannerCalls += p.replanCount - replansBefore;

    if (SUPPORTS_EPISODES && session.episode) {
      const ep = session.episode;
      const prevRel = {x: before.p.x - before.t.x, y: before.p.y - before.t.y};
      const nextRel = {x: p.x - t.x, y: p.y - t.y};
      const swept = Motion.sweptMinDistance(prevRel, nextRel);
      if (inTerminal) {
        // (B)(D) a captured or timed-out episode does not keep moving
        terminalSteps += 1;
        const moved = hypot(t.x - before.t.x, t.y - before.t.y) + hypot(p.x - before.p.x, p.y - before.p.y);
        if (moved > 1e-9 || hypot(t.vx, t.vy) > 1e-9 || hypot(p.vx, p.vy) > 1e-9) v.motion_during_terminal += 1;
      } else if (ep.terminal) {
        // this step produced the terminal state
        terminals += 1;
        episodes.push({outcome: ep.terminal.outcome, timeS: ep.terminal.timeS, closestM: ep.closestM});
        // (A) CAPTURED only when the swept criterion actually held
        if (ep.terminal.outcome === 'CAPTURED' && !(swept < CAPTURE_RADIUS_M)) v.capture_without_criterion += 1;
      } else if (mode === 'interception' && swept < CAPTURE_RADIUS_M) {
        v.missed_capture += 1;
      }
    }

    // ---- hard invariants -------------------------------------------------
    // A step that ENDS the episode (capture/timeout) zeroes both agents'
    // velocities as episode-end bookkeeping, not as vehicle dynamics. That
    // instantaneous stop is a legitimate boundary -- like the reset step -- so
    // the velocity-derivative invariants (acceleration, turn rate) are not
    // applied to it. Position, penetration, wall and speed are all still valid
    // at the pre-freeze pose and stay checked.
    const enteredTerminal = Boolean(SUPPORTS_EPISODES && session.episode
      && session.episode.terminal && !inTerminal);
    if (!agentFinite(t) || !agentFinite(p)) v.nan_or_inf += 1;
    if (penetratesRawBar(t, targetRadius, bars)) v.target_obstacle_penetration += 1;
    if (penetratesRawBar(p, trackerRadius, bars)) v.tracker_obstacle_penetration += 1;
    if (outsideWalls(t, targetRadius) || outsideWalls(p, trackerRadius)) v.wall_violation += 1;

    const tJump = hypot(t.x - before.t.x, t.y - before.t.y);
    const pJump = hypot(p.x - before.p.x, p.y - before.p.y);
    if (tJump > t.speed * DT + 1e-6) v.teleport += 1;
    if (pJump > trackerSpeedMax * DT + 1e-6) v.teleport += 1;

    if (hypot(t.vx, t.vy) > t.speed + 1e-6) v.speed_limit_violation += 1;
    if (hypot(p.vx, p.vy) > trackerSpeedMax + 1e-6) v.speed_limit_violation += 1;

    const tAccel = hypot(t.vx - before.t.vx, t.vy - before.t.vy) / DT;
    const pAccel = hypot(p.vx - before.p.vx, p.vy - before.p.vy) / DT;
    if (!enteredTerminal && tAccel > maxAccel + 1e-6) v.acceleration_limit_violation += 1;
    if (!enteredTerminal && pAccel > maxAccel + 1e-6) v.acceleration_limit_violation += 1;

    const tTurn = Math.abs(Math.atan2(
      Math.sin(t.heading - before.t.heading), Math.cos(t.heading - before.t.heading))) / DT;
    const pTurn = Math.abs(Math.atan2(
      Math.sin(p.heading - before.p.heading), Math.cos(p.heading - before.p.heading))) / DT;
    if (!enteredTerminal && tTurn > maxTurn + 1e-6) v.turn_rate_violation += 1;
    if (!enteredTerminal && pTurn > maxTurn + 1e-6) v.turn_rate_violation += 1;

    if (p.routeSerial !== prevTrackerRouteSerial) {
      if (!routeSegmentsSafe(p.route, session.caches.pursuer)) v.unsafe_route_segment += 1;
      prevTrackerRouteSerial = p.routeSerial;
    }
    if (t.route && t.route.valid && !routeSegmentsSafe(t.route, session.caches.target)) {
      v.unsafe_route_segment += 1;
    }

    const noSafeRoute = p.status === 'NO SAFE ROUTE';
    const commandMag = hypot(p.command.x, p.command.y);
    if (noSafeRoute && commandMag > EPS) v.command_without_safe_route += 1;
    // Blind motion: the route was invalidated this step and the tracker is
    // neither on a fresh valid route nor on a certified line of sight, yet it
    // still receives a non-zero command.
    const losNow = p.status === 'DIRECT LOS' || p.status === 'FOLLOWING';
    if (routeWasValid && !(p.route && p.route.valid) && !losNow && commandMag > EPS) {
      v.blind_motion_after_route_invalidation += 1;
    }

    // ---- goal-handoff instrumentation (target side) ----------------------
    const targetSpeedNow = hypot(t.vx, t.vy);
    targetSpeedHist.push(inTerminal ? NaN : targetSpeedNow);
    if (!inTerminal && targetSpeedNow < 0.05 && t.speed > 0.2) dwellBelow005 += 1;
    if ((t.goalCompletions || 0) > lastGoalCompletions) {
      lastGoalCompletions = t.goalCompletions;
      handoffEvents.push({step: step});
    }

    // ---- diagnostics (running steps only) --------------------------------
    if (inTerminal) { prevD = hypot(p.x - t.x, p.y - t.y); prevHeading = p.heading; continue; }
    const d = hypot(p.x - t.x, p.y - t.y);
    distances.push(d);
    comparableSteps += 1;
    if (d < prevD - 1e-9) decreasingSteps += 1;
    prevD = d;

    const trackerSpeed = hypot(p.vx, p.vy);
    const stalled = trackerSpeed < 0.05 && commandMag > 0.2;
    if (stalled) stallSteps += 1;
    if (stalled && (losNow || (p.route && p.route.valid))) validRouteStallSteps += 1;
    if (noSafeRoute) noRouteSteps += 1;
    if (losNow) losSteps += 1;

    const turn = Math.atan2(Math.sin(p.heading - prevHeading), Math.cos(p.heading - prevHeading));
    if (Math.abs(turn) > 5 * Math.PI / 180) {
      const sign = Math.sign(turn);
      if (lastTurnSign && sign === -lastTurnSign) headingSignFlips += 1;
      lastTurnSign = sign;
    }
    prevHeading = p.heading;
    prevRouteValid = Boolean(p.route && p.route.valid);

    if (opts && opts.digest) {
      digest.push([
        Math.round(t.x * 1e6), Math.round(t.y * 1e6),
        Math.round(p.x * 1e6), Math.round(p.y * 1e6),
        Math.round(p.heading * 1e6),
      ].join(','));
    }
  }

  // (C) exactly one reset per terminal; the last terminal may still be holding
  if (SUPPORTS_EPISODES && !(terminals - resets === 0 || terminals - resets === 1)) v.reset_count_mismatch += 1;

  // goal handoff: for each completion, the speed dip around it and the time
  // until the target is back above half its cruise speed
  const win = Math.round(1.0 / DT);
  const pauses = [], minSpeeds = [];
  for (const ev of handoffEvents) {
    let minS = Infinity;
    for (let k = Math.max(0, ev.step - win); k <= Math.min(targetSpeedHist.length - 1, ev.step + win); k++) {
      const sp = targetSpeedHist[k];
      if (Number.isFinite(sp)) minS = Math.min(minS, sp);
    }
    let recover = null;
    for (let k = ev.step; k <= Math.min(targetSpeedHist.length - 1, ev.step + 3 * win); k++) {
      const sp = targetSpeedHist[k];
      if (Number.isFinite(sp) && sp >= 0.5 * speed) { recover = (k - ev.step) * DT; break; }
    }
    pauses.push(recover == null ? 3.0 : recover);
    minSpeeds.push(Number.isFinite(minS) ? minS : 0);
  }
  const sortedPauses = pauses.slice().sort((a, b) => a - b);
  const handoff = {
    goal_transition_count: handoffEvents.length,
    goal_handoff_pause_seconds_mean: pauses.length ? pauses.reduce((a, b) => a + b, 0) / pauses.length : 0,
    goal_handoff_pause_seconds_max: pauses.length ? sortedPauses[sortedPauses.length - 1] : 0,
    goal_handoff_pause_seconds_p95: pauses.length ? quantiles(sortedPauses, 0.95) : 0,
    goal_handoff_min_speed_mps_mean: minSpeeds.length ? minSpeeds.reduce((a, b) => a + b, 0) / minSpeeds.length : 0,
    target_dwell_below_0p05_s: dwellBelow005 * DT,
  };
  const captured = episodes.filter(e => e.outcome === 'CAPTURED');
  const timedOut = episodes.filter(e => e.outcome === 'TIMEOUT');
  const tCap = captured.map(e => e.timeS).sort((a, b) => a - b);
  const interception = {
    episode_mode: mode,
    capture_radius_m: CAPTURE_RADIUS_M,
    timeout_s: TIMEOUT_S,
    episodes_terminated: episodes.length,
    resets: resets,
    terminal_steps: terminalSteps,
    capture_count: captured.length,
    timeout_count: timedOut.length,
    capture_rate: episodes.length ? captured.length / episodes.length : null,
    timeout_rate: episodes.length ? timedOut.length / episodes.length : null,
    time_to_capture_s_mean: tCap.length ? tCap.reduce((a, b) => a + b, 0) / tCap.length : null,
    time_to_capture_s_median: tCap.length ? quantiles(tCap, 0.5) : null,
    closest_distance_m_mean: episodes.length ? episodes.reduce((a, e) => a + e.closestM, 0) / episodes.length : null,
    closest_distance_m_min: episodes.length ? Math.min.apply(null, episodes.map(e => e.closestM)) : null,
  };

  const sorted = distances.slice().sort((a, b) => a - b);
  const mean = distances.reduce((a, b) => a + b, 0) / distances.length;
  const tail = distances.slice(Math.floor(distances.length * 0.75));
  const rms = Math.sqrt(distances.reduce((a, b) => a + b * b, 0) / distances.length);
  const durationActual = steps * DT;

  return {
    violations: v,
    digest: opts && opts.digest ? digest.join(';') : null,
    handoff: handoff,
    interception: interception,
    diagnostics: {
      initial_distance_m: initialDistance,
      final_distance_m: distances[distances.length - 1],
      mean_distance_m: mean,
      median_distance_m: quantiles(sorted, 0.5),
      minimum_distance_m: sorted[0],
      distance_reduction_m: initialDistance - distances[distances.length - 1],
      fraction_of_time_distance_decreasing: decreasingSteps / comparableSteps,
      tracking_error_rms_m: rms,
      tracking_error_terminal_mean_m: tail.reduce((a, b) => a + b, 0) / tail.length,
      stall_fraction: stallSteps / steps,
      valid_route_but_stalled_seconds: validRouteStallSteps * DT,
      no_safe_route_fraction: noRouteSteps / steps,
      replans_per_second: plannerCalls / durationActual,
      route_switch_count: p.routeSwitches,
      heading_oscillation_flips_per_second: headingSignFlips / durationActual,
      heading_oscillation_period2_steps: session.stats.pursuer.period2,
      direct_los_fraction: losSteps / steps,
      target_goal_completion_count: t.goalCompletions,
      tracker_emergency_holds: p.emergencyHolds || 0,
      target_emergency_holds: t.emergencyHolds || 0,
      spawn_target_relocated: session.spawn.targetRelocated ? 1 : 0,
      spawn_tracker_relocated: session.spawn.pursuerRelocated ? 1 : 0,
    },
  };
}

function aggregate(values) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  return {
    mean: values.reduce((a, b) => a + b, 0) / values.length,
    median: quantiles(sorted, 0.5),
    min: sorted[0],
    max: sorted[sorted.length - 1],
  };
}

function main() {
  const argv = process.argv.slice(2);
  const arg = (name, fallback) => {
    const i = argv.indexOf(name);
    return i >= 0 && argv[i + 1] != null ? argv[i + 1] : fallback;
  };
  const quick = argv.includes('--quick');
  const episodeMode = arg('--episode-mode', 'continuous');
  if (!['continuous', 'interception'].includes(episodeMode)) throw new Error('bad --episode-mode');
  if (episodeMode === 'interception' && !SUPPORTS_EPISODES) {
    throw new Error('this planner snapshot has no episode support; interception mode is unavailable');
  }
  const densities = [70, 115, 160, 205];
  const speeds = [0.3, 0.9, 1.5];
  const seedCount = Number(arg('--seeds', quick ? 2 : 10));
  const durationS = Number(arg('--duration', quick ? 10 : 60));
  const outFile = arg('--out', null);

  const started = Date.now();
  const cells = [];
  const totals = newViolations();
  let runs = 0;

  for (const density of densities) {
    for (const speed of speeds) {
      const cell = {
        density_bars: density,
        target_speed_m_s: speed,
        runs: [],
      };
      for (let s = 0; s < seedCount; s++) {
        const seed = 20260917 + density * 1000 + Math.round(speed * 10) * 100 + s;
        const bars = placeBars(density, seed);
        if (bars.length !== density) {
          throw new Error(`layout failed closed: ${bars.length}/${density} @seed ${seed}`);
        }
        const result = runCell(bars, seed, speed, durationS, {digest: false, episodeMode: episodeMode});
        addViolations(totals, result.violations);
        runs += 1;
        cell.runs.push(Object.assign({seed: seed, bars: bars.length}, result.diagnostics,
          result.handoff, result.interception, {violations: result.violations}));
      }
      const pick = (k) => aggregate(cell.runs.map((r) => r[k]));
      cell.summary = {
        initial_distance_m: pick('initial_distance_m'),
        final_distance_m: pick('final_distance_m'),
        mean_distance_m: pick('mean_distance_m'),
        median_distance_m: pick('median_distance_m'),
        minimum_distance_m: pick('minimum_distance_m'),
        distance_reduction_m: pick('distance_reduction_m'),
        fraction_of_time_distance_decreasing: pick('fraction_of_time_distance_decreasing'),
        tracking_error_rms_m: pick('tracking_error_rms_m'),
        tracking_error_terminal_mean_m: pick('tracking_error_terminal_mean_m'),
        stall_fraction: pick('stall_fraction'),
        valid_route_but_stalled_seconds: pick('valid_route_but_stalled_seconds'),
        no_safe_route_fraction: pick('no_safe_route_fraction'),
        replans_per_second: pick('replans_per_second'),
        route_switch_count: pick('route_switch_count'),
        heading_oscillation_flips_per_second: pick('heading_oscillation_flips_per_second'),
        direct_los_fraction: pick('direct_los_fraction'),
        target_goal_completion_count: pick('target_goal_completion_count'),
        tracker_emergency_holds: pick('tracker_emergency_holds'),
        goal_transition_count: pick('goal_transition_count'),
        goal_handoff_pause_seconds_mean: pick('goal_handoff_pause_seconds_mean'),
        goal_handoff_pause_seconds_max: pick('goal_handoff_pause_seconds_max'),
        goal_handoff_min_speed_mps_mean: pick('goal_handoff_min_speed_mps_mean'),
        target_dwell_below_0p05_s: pick('target_dwell_below_0p05_s'),
        episodes_terminated: pick('episodes_terminated'),
        capture_count: pick('capture_count'),
        timeout_count: pick('timeout_count'),
        time_to_capture_s_mean: aggregate(cell.runs.map(r => r.time_to_capture_s_mean).filter(x => x != null)),
        closest_distance_m_mean: aggregate(cell.runs.map(r => r.closest_distance_m_mean).filter(x => x != null)),
        spawn_target_relocated: pick('spawn_target_relocated'),
        spawn_tracker_relocated: pick('spawn_tracker_relocated'),
      };
      cells.push(cell);
      const s = cell.summary;
      process.stdout.write(
        `bars=${String(density).padStart(3)} v=${speed.toFixed(1)} ` +
        `d0=${s.initial_distance_m.mean.toFixed(2)} dT=${s.final_distance_m.mean.toFixed(2)} ` +
        `dmed=${s.median_distance_m.mean.toFixed(2)} dmin=${s.minimum_distance_m.mean.toFixed(2)} ` +
        `dec=${(s.fraction_of_time_distance_decreasing.mean * 100).toFixed(1)}% ` +
        `los=${(s.direct_los_fraction.mean * 100).toFixed(1)}% ` +
        `stall=${(s.stall_fraction.mean * 100).toFixed(1)}% ` +
        `noroute=${(s.no_safe_route_fraction.mean * 100).toFixed(1)}% ` +
        `replan/s=${s.replans_per_second.mean.toFixed(2)} ` +
        `hold=${(s.tracker_emergency_holds.mean / 6).toFixed(2)}% ` +
        `handoffs=${s.goal_transition_count.mean.toFixed(1)} pause=${s.goal_handoff_pause_seconds_mean.mean.toFixed(2)}s ` +
        `dwell<.05=${s.target_dwell_below_0p05_s.mean.toFixed(2)}s` +
        (episodeMode === 'interception'
          ? ` | eps=${s.episodes_terminated.mean.toFixed(1)} cap=${s.capture_count.mean.toFixed(1)} to=${s.timeout_count.mean.toFixed(1)}` +
            ` t_cap=${s.time_to_capture_s_mean ? s.time_to_capture_s_mean.mean.toFixed(1) : 'NA'}s` +
            ` closest=${s.closest_distance_m_mean ? s.closest_distance_m_mean.mean.toFixed(2) : 'NA'}m`
          : '') + `\n`
      );
    }
  }

  const report = {
    kind: 'browser_gt_tracking_logic_validation',
    episode_mode: episodeMode,
    planner_dir: PLANNER_DIR,
    termination_contract: {capture_radius_m: CAPTURE_RADIUS_M, timeout_s: TIMEOUT_S,
      capture_radius_provenance: GEO.success_radius_m != null ? 'status.json arena_geometry.success_radius_m'
        : 'fallback: navrl_task_config.success_radius = 0.5 m (status.json predates the field)'},
    boundary: [
      'BROWSER GT TRACKING PREVIEW',
      'SIMULATION ONLY',
      'NOT PPO',
      'NOT PHYSX',
      'NOT RESEARCH PERFORMANCE EVIDENCE',
    ],
    generated_utc: new Date().toISOString(),
    wall_clock_s: (Date.now() - started) / 1000,
    contract: {
      simulation_hz: 1 / DT,
      duration_s: durationS,
      seeds_per_cell: seedCount,
      densities: densities,
      target_speeds_m_s: speeds,
      total_runs: runs,
      arena_geometry: GEO,
      planner_contract: Planner.CONTRACT,
      motion_limits: {
        bounded_max_accel_m_s2: Motion.CONTRACT.boundedMaxAccel,
        bounded_max_turn_rate_rad_s: Motion.CONTRACT.boundedMaxTurnRate,
        tracker_speed_max_m_s: Motion.CONTRACT.pursuerSpeedMax,
      },
      route_contract: Route.CONTRACT,
    },
    hard_violation_totals: totals,
    cells: cells,
  };

  const clean = Object.values(totals).every((n) => n === 0);
  process.stdout.write(`\nruns=${runs} duration=${durationS}s/run ` +
    `wall=${report.wall_clock_s.toFixed(1)}s\n`);
  process.stdout.write('hard violations: ' +
    Object.entries(totals).map(([k, n]) => `${k}=${n}`).join(' ') + '\n');
  process.stdout.write(clean ? 'HARD INVARIANTS: PASS\n' : 'HARD INVARIANTS: FAIL\n');

  if (outFile) {
    fs.writeFileSync(outFile, JSON.stringify(report, null, 1) + '\n');
    process.stdout.write(`wrote ${outFile}\n`);
  }
  process.exitCode = clean ? 0 : 1;
}

if (require.main === module) main();
module.exports = {placeBars, runCell, ARENA_LO, ARENA_HI, DT};
