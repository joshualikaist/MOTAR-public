'use strict';
/*
 * Browser GT tracking preview: determinism, cadence independence, and the
 * distance-convergence sanity check.
 *
 * Engineering validation of the browser preview only. Not PPO, not PhysX, and
 * not research performance evidence.
 */

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const Motion = require('../docs/status/arena_motion.js');
const Planner = require('../docs/status/arena_demo_planner.js');
const Harness = require('../tools/validate_gt_browser_tracking.js');

const GEO = JSON.parse(
  fs.readFileSync(path.join(__dirname, '..', 'docs', 'status', 'status.json'), 'utf8')
).arena_geometry;
Motion.configure(GEO);

const DT = 0.1;

function makeSession(density, speed, seed) {
  const bars = Harness.placeBars(density, seed);
  const rng = Motion.seededRng(seed);
  const episode = Motion.createEpisode(rng, bars, speed);
  return Planner.createSession({
    bars: bars,
    arenaLo: Harness.ARENA_LO,
    arenaHi: Harness.ARENA_HI,
    rng: rng,
    speed: speed,
    target: {x: episode.target.x, y: episode.target.y, heading: episode.heading},
    pursuer: {x: episode.drone.x, y: episode.drone.y, heading: 0},
    goalMode: 'auto',
  });
}

function stateDigest(session) {
  const t = session.target, p = session.pursuer;
  return [t.x, t.y, t.vx, t.vy, t.heading, p.x, p.y, p.vx, p.vy, p.heading]
    .map(function (v) { return Math.round(v * 1e9); }).join(',');
}

/* ------------------------------------------------------------------ *
 * T1. Same seed, same trajectory. The preview must replay exactly.
 * ------------------------------------------------------------------ */
{
  const digests = [0, 1].map(function () {
    const session = makeSession(160, 0.9, 424242);
    const out = [];
    for (let i = 0; i < 300; i++) { Planner.stepSession(session, DT); out.push(stateDigest(session)); }
    return out.join(';');
  });
  assert.strictEqual(digests[0], digests[1], 'identical seeds must replay identically');
}

/* ------------------------------------------------------------------ *
 * T2. Render cadence independence.
 *
 * 30 / 60 / 120 FPS must commit the SAME simulation trajectory. The fixed
 * 10 Hz clock only decides WHEN a step is committed; the step itself always
 * receives the fixed dt, so no render-rate quantity may reach the planner.
 * ------------------------------------------------------------------ */
{
  const WALL_SECONDS = 30;
  const cadences = [30, 60, 120];
  const traces = cadences.map(function (fps) {
    const session = makeSession(115, 1.5, 987654);
    const clock = Motion.createFixedStepClock(DT, 0.25, 8);
    const committed = [];
    const frames = Math.round(WALL_SECONDS * fps);
    clock.reset(0);
    for (let f = 1; f <= frames; f++) {
      clock.advance(f / fps, true, function (dt) {
        Planner.stepSession(session, dt);
        committed.push(stateDigest(session));
      });
    }
    return committed;
  });
  const shortest = Math.min.apply(null, traces.map(function (t) { return t.length; }));
  assert(shortest >= WALL_SECONDS / DT - 2,
    `cadence run committed only ${shortest} steps`);
  for (let i = 1; i < traces.length; i++) {
    assert.strictEqual(
      traces[i].slice(0, shortest).join(';'),
      traces[0].slice(0, shortest).join(';'),
      `render cadence ${cadences[i]} FPS changed the committed trajectory`
    );
  }
  // A paused renderer must not advance the simulation at any cadence.
  const session = makeSession(115, 1.5, 987654);
  const clock = Motion.createFixedStepClock(DT, 0.25, 8);
  clock.reset(0);
  let steps = 0;
  for (let f = 1; f <= 600; f++) clock.advance(f / 60, false, function () { steps += 1; });
  assert.strictEqual(steps, 0, 'a paused renderer must commit no simulation steps');
  assert.strictEqual(session.time, 0);
}

/* ------------------------------------------------------------------ *
 * T3. Distance convergence in a deterministic obstacle-free case.
 *
 * The tracking reference is a relative position error of zero. The browser
 * preview holds a small DISPLAY-ONLY standoff so the two meshes do not overlap
 * or jitter, so convergence is asserted against that standoff band, not
 * against physical contact.
 * ------------------------------------------------------------------ */
{
  const standoff = Planner.CONTRACT.standoffM;
  assert(Motion.CONTRACT.pursuerSpeedMax > Motion.CONTRACT.targetSpeedMax,
    'the sanity check needs the tracker to have surplus speed');
  for (const targetSpeed of [0.3, 0.9, 1.5]) {
    const session = Planner.createSession({
      bars: [],
      arenaLo: Harness.ARENA_LO,
      arenaHi: Harness.ARENA_HI,
      rng: Motion.seededRng(31337),
      speed: targetSpeed,
      target: {x: 30, y: 8, heading: 0},
      pursuer: {x: 6, y: -8, heading: 0},
      goalMode: 'auto',
    });
    const initial = Math.hypot(
      session.pursuer.x - session.target.x, session.pursuer.y - session.target.y);
    assert(initial > 0, 'initial relative distance must be positive');
    const afterConvergence = [];
    let converged = false;
    let last = initial;
    for (let i = 0; i < 900; i++) {
      Planner.stepSession(session, DT);
      const d = Math.hypot(
        session.pursuer.x - session.target.x, session.pursuer.y - session.target.y);
      if (!converged && d <= standoff + 0.35) converged = true;
      if (converged) afterConvergence.push(d);
      last = d;
    }
    assert(converged,
      `open arena v=${targetSpeed}: distance never reached the standoff band (last ${last.toFixed(2)} m)`);
    // Once converged it must STAY converged: continuous following, not a
    // capture-and-reset. The bound is stated on quantiles because a hard
    // target reversal legitimately swings the trailing follow point to the far
    // side of the target; that is a bounded, recovering transient, not a lost
    // track, so a raw max would encode the transient as the contract.
    const sorted = afterConvergence.slice().sort(function (a, b) { return a - b; });
    const quantile = function (q) { return sorted[Math.floor((sorted.length - 1) * q)]; };
    assert(quantile(0.5) <= standoff + 0.6,
      `open arena v=${targetSpeed}: median follow distance ${quantile(0.5).toFixed(2)} m`);
    assert(quantile(0.9) <= standoff + 1.0,
      `open arena v=${targetSpeed}: p90 follow distance ${quantile(0.9).toFixed(2)} m`);
    assert(quantile(1) <= standoff + 2.5,
      `open arena v=${targetSpeed}: following broke after convergence (max ${quantile(1).toFixed(2)} m)`);
    assert(last <= standoff + 1.5,
      `open arena v=${targetSpeed}: final distance ${last.toFixed(2)} m above the standoff band`);
  }
}

/* ------------------------------------------------------------------ *
 * T4. Hard invariants across the density x speed matrix.
 *
 * Regression guard for the four defects the 2026-09-17 validation found:
 * the unbounded fail-closed stop, the uncertified look-ahead chord, the
 * unroutable follow point, and the sealed-pocket spawn.
 * ------------------------------------------------------------------ */
{
  const cells = [[70, 1.5], [115, 0.9], [160, 0.3], [160, 1.5], [205, 0.9], [205, 1.5]];
  for (const [density, speed] of cells) {
    for (let s = 0; s < 3; s++) {
      const seed = 20260917 + density * 1000 + Math.round(speed * 10) * 100 + s;
      const bars = Harness.placeBars(density, seed);
      const result = Harness.runCell(bars, seed, speed, 30, {});
      for (const [name, count] of Object.entries(result.violations)) {
        assert.strictEqual(count, 0,
          `${density} bars v=${speed} seed ${seed}: ${name} = ${count}`);
      }
      const d = result.diagnostics;
      assert(d.stall_fraction <= 0.15,
        `${density} bars v=${speed} seed ${seed}: stall_fraction ${d.stall_fraction.toFixed(3)}`);
      assert(d.no_safe_route_fraction <= 0.15,
        `${density} bars v=${speed} seed ${seed}: no_safe_route ${d.no_safe_route_fraction.toFixed(3)}`);
      assert(d.replans_per_second <= 5,
        `${density} bars v=${speed} seed ${seed}: replans/s ${d.replans_per_second.toFixed(2)}`);
    }
  }
}

/* ------------------------------------------------------------------ *
 * T5. The certified look-ahead never hands back a blocked chord.
 * ------------------------------------------------------------------ */
{
  const session = makeSession(205, 0.9, 5150);
  const p = session.pursuer;
  for (let i = 0; i < 400; i++) {
    // The look-ahead is certified from the pose it is computed at, which is the
    // pose BEFORE this tick's integration.
    const pose = {x: p.x, y: p.y};
    Planner.stepSession(session, DT);
    if (!p.carrot) continue;
    assert(Planner.segmentSafe(pose, p.carrot, session.caches.pursuer),
      `the follower was handed an uncertified look-ahead point at step ${i}`);
    assert(Planner.pointSafe(p.carrot, session.caches.pursuer),
      `the follower look-ahead point is inside the safety envelope at step ${i}`);
  }
  // Direct line-of-sight tracking must not publish a stale route carrot.
  const los = makeSession(70, 0.9, 5150);
  for (let i = 0; i < 400; i++) {
    Planner.stepSession(los, DT);
    if (los.pursuer.status === 'DIRECT LOS' || los.pursuer.status === 'FOLLOWING') {
      assert.strictEqual(los.pursuer.carrot, null,
        'line-of-sight tracking published a stale route look-ahead point');
    }
  }
}

console.log('MOTAR browser GT tracking determinism and convergence contracts: PASS');
