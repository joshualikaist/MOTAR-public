'use strict';
/*
 * Browser interception episode: phase machine, termination, terminal hold,
 * reset correctness, mode separation, determinism and hard invariants.
 *
 * Engineering validation of the browser preview. Not PPO, not PhysX, not
 * research performance evidence.
 */
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const Motion = require('../docs/status/arena_motion.js');
const Planner = require('../docs/status/arena_demo_planner.js');
const Harness = require('../tools/validate_gt_browser_tracking.js');

const GEO = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'docs', 'status', 'status.json'), 'utf8')).arena_geometry;
Motion.configure(GEO);
const DT = 0.1;
const RADIUS = 0.5;   // navrl_task_config.success_radius, bound by test_browser_capture_provenance.py
const TIMEOUT = 60;   // 600 steps x 0.1 s

function openSession(speed, extra) {
  return Planner.createSession(Object.assign({
    bars: [], arenaLo: Harness.ARENA_LO, arenaHi: Harness.ARENA_HI,
    rng: Motion.seededRng(31337), speed: speed,
    target: {x: 30, y: 8, heading: 0}, pursuer: {x: 6, y: -8, heading: 0},
    goalMode: 'auto', episodeMode: 'interception', captureRadiusM: RADIUS, timeoutS: TIMEOUT,
  }, extra || {}));
}
function rel(s) { return {x: s.pursuer.x - s.target.x, y: s.pursuer.y - s.target.y}; }
function digest(s) {
  const t = s.target, p = s.pursuer;
  return [t.x, t.y, p.x, p.y, p.heading, s.episode.phase].map(v => typeof v === 'number' ? Math.round(v * 1e9) : v).join(',');
}

/* T1. Open arena: CHASE -> CLOSE -> INTERCEPT -> CAPTURED at every speed, and
 * (A) CAPTURED only when the swept criterion held on the capturing step. */
for (const v of [0.3, 0.9, 1.5]) {
  const s = openSession(v);
  const phases = [];
  let capturedStepSwept = null;
  for (let i = 0; i < 900 && !s.episode.terminal; i++) {
    const prev = rel(s);
    Planner.stepSession(s, DT);
    if (phases[phases.length - 1] !== s.episode.phase) phases.push(s.episode.phase);
    if (s.episode.terminal) capturedStepSwept = Motion.sweptMinDistance(prev, rel(s));
  }
  assert.deepStrictEqual(phases, ['CHASE', 'CLOSE', 'INTERCEPT', 'CAPTURED'], `v=${v} phases ${phases}`);
  assert.strictEqual(s.episode.outcome, 'CAPTURED');
  assert(capturedStepSwept < RADIUS, `v=${v}: CAPTURED without the swept criterion (${capturedStepSwept})`);
  assert(s.episode.closestM < RADIUS);
  assert(s.episode.captureTimeS > 0 && s.episode.captureTimeS < TIMEOUT);
}

/* T2. (B) a captured episode does not keep moving; (C) resetDue exactly once,
 * exactly at the terminal hold. */
{
  const s = openSession(0.9);
  while (!s.episode.terminal) Planner.stepSession(s, DT);
  const t0 = s.time;
  let moved = 0, dueEvents = 0, dueAt = null;
  for (let i = 0; i < 60; i++) {
    const tx = s.target.x, ty = s.target.y, px = s.pursuer.x, py = s.pursuer.y;
    const wasDue = s.episode.resetDue;
    Planner.stepSession(s, DT);
    moved += Math.hypot(s.target.x - tx, s.target.y - ty) + Math.hypot(s.pursuer.x - px, s.pursuer.y - py);
    assert.strictEqual(Math.hypot(s.pursuer.vx, s.pursuer.vy), 0);
    assert.strictEqual(Math.hypot(s.target.vx, s.target.vy), 0);
    assert.strictEqual(Math.hypot(s.pursuer.command.x, s.pursuer.command.y), 0);
    if (!wasDue && s.episode.resetDue) { dueEvents += 1; dueAt = s.time - t0; }
  }
  assert.strictEqual(moved, 0, 'motion after CAPTURED');
  assert.strictEqual(dueEvents, 1, 'resetDue must fire exactly once');
  assert(Math.abs(dueAt - Planner.CONTRACT.interception.terminalHoldS) < DT + 1e-9, `resetDue at ${dueAt}`);
  assert.strictEqual(s.episode.phase, 'CAPTURED');
}

/* T3. (D) TIMEOUT is a terminal state too, and it freezes. */
{
  const s = openSession(0.3, {timeoutS: 2.0});
  while (!s.episode.terminal) Planner.stepSession(s, DT);
  assert.strictEqual(s.episode.outcome, 'TIMEOUT');
  assert(Math.abs(s.episode.terminal.timeS - 2.0) < DT + 1e-9);
  assert(s.episode.closestM > RADIUS, 'a TIMEOUT must not have satisfied the capture criterion');
  const px = s.pursuer.x, tx = s.target.x;
  for (let i = 0; i < 30; i++) Planner.stepSession(s, DT);
  assert.strictEqual(s.pursuer.x, px); assert.strictEqual(s.target.x, tx);
  assert(s.episode.resetDue);
}

/* T4. (E) a new episode carries no route, goal, lead, tracker or terminal state. */
{
  const s = openSession(0.9);
  while (!s.episode.resetDue) Planner.stepSession(s, DT);
  const fresh = openSession(0.9);   // what arena.js/harness do on reset: a NEW session
  assert.strictEqual(fresh.target.route, null); assert.strictEqual(fresh.target.goal, null);
  assert.strictEqual(fresh.pursuer.route, null); assert.strictEqual(fresh.pursuer.lead, null);
  assert.strictEqual(fresh.pursuer.replanCount, 0); assert.strictEqual(fresh.target.goalCompletions, 0);
  assert.strictEqual(fresh.episode.terminal, null); assert.strictEqual(fresh.episode.resetDue, false);
  assert.strictEqual(fresh.episode.phase, 'CHASE'); assert.strictEqual(fresh.episode.outcome, 'RUNNING');
  assert.strictEqual(fresh.episode.elapsedS, 0);
}

/* T5. (F) CONTINUOUS mode: no capture, no timeout, standoff held (GT_BROWSER_V1). */
{
  const s = openSession(1.5, {episodeMode: 'continuous', timeoutS: 5});
  let minD = Infinity;
  for (let i = 0; i < 900; i++) { Planner.stepSession(s, DT); minD = Math.min(minD, Math.hypot(rel(s).x, rel(s).y)); }
  assert.strictEqual(s.episode.terminal, null); assert.strictEqual(s.episode.outcome, 'RUNNING');
  assert.strictEqual(s.episode.phase, 'TRACK');
  assert.strictEqual(Planner.desiredStandoff(s, 0.3), Planner.CONTRACT.standoffM);
  assert(s.time > 5, 'continuous mode must ignore the timeout');
}

/* T6. The standoff ramp is continuous and monotone: 1.55 at/above CLOSE, 0 at/below INTERCEPT. */
{
  const s = openSession(0.9);
  const c = Planner.CONTRACT.interception;
  assert.strictEqual(Planner.desiredStandoff(s, c.closeEnterM + 5), Planner.CONTRACT.standoffM);
  assert.strictEqual(Planner.desiredStandoff(s, c.closeEnterM), Planner.CONTRACT.standoffM);
  assert.strictEqual(Planner.desiredStandoff(s, c.interceptEnterM), 0);
  assert.strictEqual(Planner.desiredStandoff(s, 0), 0);
  let prev = Planner.desiredStandoff(s, 0);
  for (let d = 0; d <= 8; d += 0.01) {
    const now = Planner.desiredStandoff(s, d);
    assert(now >= prev - 1e-12, 'standoff must be non-decreasing in distance');
    assert(now - prev < 0.02, `standoff jump ${now - prev} at d=${d}`);
    prev = now;
  }
  assert.strictEqual(Planner.episodePhase(s, c.closeEnterM + 1), 'CHASE');
  assert.strictEqual(Planner.episodePhase(s, (c.closeEnterM + c.interceptEnterM) / 2), 'CLOSE');
  assert.strictEqual(Planner.episodePhase(s, c.interceptEnterM - 0.1), 'INTERCEPT');
}

/* T7. Determinism in interception mode: same seed, same trajectory, and
 * 30/60/120 FPS commit the same steps. */
{
  const runs = [0, 1].map(() => {
    const bars = Harness.placeBars(115, 424242);
    const rng = Motion.seededRng(424242);
    const ep = Motion.createEpisode(rng, bars, 0.9);
    const s = Planner.createSession({bars, arenaLo: Harness.ARENA_LO, arenaHi: Harness.ARENA_HI, rng, speed: 0.9,
      target: {x: ep.target.x, y: ep.target.y, heading: ep.heading}, pursuer: {x: ep.drone.x, y: ep.drone.y, heading: 0},
      goalMode: 'auto', episodeMode: 'interception', captureRadiusM: RADIUS, timeoutS: TIMEOUT});
    const out = []; for (let i = 0; i < 400; i++) { Planner.stepSession(s, DT); out.push(digest(s)); } return out.join(';');
  });
  assert.strictEqual(runs[0], runs[1]);
  const traces = [30, 60, 120].map(fps => {
    const s = openSession(1.5); const clock = Motion.createFixedStepClock(DT, 0.25, 8); const out = [];
    clock.reset(0);
    for (let f = 1; f <= 20 * fps; f++) clock.advance(f / fps, true, dt => { Planner.stepSession(s, dt); out.push(digest(s)); });
    return out;
  });
  const n = Math.min.apply(null, traces.map(t => t.length));
  for (let i = 1; i < traces.length; i++) assert.strictEqual(traces[i].slice(0, n).join(';'), traces[0].slice(0, n).join(';'));
}

/* T8. Hard invariants and terminal correctness across a small matrix, with
 * resets actually happening inside the run. */
{
  for (const [density, speed] of [[70, 1.5], [115, 0.9], [160, 0.3], [205, 1.5]]) {
    for (let k = 0; k < 2; k++) {
      const seed = 20260918 + density * 1000 + Math.round(speed * 10) * 100 + k;
      const result = Harness.runCell(Harness.placeBars(density, seed), seed, speed, 60, {episodeMode: 'interception'});
      for (const [name, count] of Object.entries(result.violations)) {
        assert.strictEqual(count, 0, `${density} bars v=${speed} seed ${seed}: ${name} = ${count}`);
      }
      assert(result.interception.episodes_terminated >= 1, `${density}/${speed}: no episode terminated in 60 s`);
    }
  }
}

console.log('MOTAR browser interception episode contracts: PASS');
