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
const Motion = require('../docs/status/arena_motion.js');
const Route = require('../docs/status/arena_route.js');
const Planner = require('../docs/status/arena_demo_planner.js');

const STATUS = JSON.parse(
  fs.readFileSync(path.join(__dirname, '..', 'docs', 'status', 'status.json'), 'utf8')
);
const GEO = STATUS.arena_geometry;

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
  };
}

function addViolations(into, from) {
  for (const k of Object.keys(from)) into[k] += from[k];
}

function runCell(bars, seed, speed, durationS, opts) {
  const steps = Math.round(durationS / DT);
  const motionRng = Motion.seededRng(seed);
  const episode = Motion.createEpisode(motionRng, bars, speed);
  const session = Planner.createSession({
    bars: bars,
    arenaLo: ARENA_LO,
    arenaHi: ARENA_HI,
    rng: motionRng,
    speed: speed,
    target: {x: episode.target.x, y: episode.target.y, heading: episode.heading},
    pursuer: {x: episode.drone.x, y: episode.drone.y, heading: 0},
    goalMode: 'auto',
  });

  const v = newViolations();
  const t = session.target, p = session.pursuer;
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
    const before = {
      t: {x: t.x, y: t.y, vx: t.vx, vy: t.vy, heading: t.heading},
      p: {x: p.x, y: p.y, vx: p.vx, vy: p.vy, heading: p.heading},
    };
    const routeWasValid = Boolean(p.route && p.route.valid);
    const replansBefore = p.replanCount;

    Planner.stepSession(session, DT);

    plannerCalls += p.replanCount - replansBefore;

    // ---- hard invariants -------------------------------------------------
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
    if (tAccel > maxAccel + 1e-6) v.acceleration_limit_violation += 1;
    if (pAccel > maxAccel + 1e-6) v.acceleration_limit_violation += 1;

    const tTurn = Math.abs(Math.atan2(
      Math.sin(t.heading - before.t.heading), Math.cos(t.heading - before.t.heading))) / DT;
    const pTurn = Math.abs(Math.atan2(
      Math.sin(p.heading - before.p.heading), Math.cos(p.heading - before.p.heading))) / DT;
    if (tTurn > maxTurn + 1e-6) v.turn_rate_violation += 1;
    if (pTurn > maxTurn + 1e-6) v.turn_rate_violation += 1;

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

    // ---- diagnostics -----------------------------------------------------
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

  const sorted = distances.slice().sort((a, b) => a - b);
  const mean = distances.reduce((a, b) => a + b, 0) / distances.length;
  const tail = distances.slice(Math.floor(distances.length * 0.75));
  const rms = Math.sqrt(distances.reduce((a, b) => a + b * b, 0) / distances.length);
  const durationActual = steps * DT;

  return {
    violations: v,
    digest: opts && opts.digest ? digest.join(';') : null,
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
        const result = runCell(bars, seed, speed, durationS, {digest: false});
        addViolations(totals, result.violations);
        runs += 1;
        cell.runs.push(Object.assign({seed: seed, bars: bars.length}, result.diagnostics,
          {violations: result.violations}));
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
        `osc/s=${s.heading_oscillation_flips_per_second.mean.toFixed(2)}\n`
      );
    }
  }

  const report = {
    kind: 'browser_gt_tracking_logic_validation',
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
