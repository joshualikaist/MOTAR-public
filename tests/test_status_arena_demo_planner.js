'use strict';

const assert = require('assert');
const Motion = require('../docs/status/arena_motion.js');
const Route = require('../docs/status/arena_route.js');
const Planner = require('../docs/status/arena_demo_planner.js');

Motion.configure({arena_xy_m: 40, goal_dist_m: [6, 28], target_speed_m: [0.3, 1.5]});

const arenaLo = {x: 0, y: -20};
const arenaHi = {x: 40, y: 20};

function denseBars(count, seed) {
  const rng = Motion.seededRng(seed);
  const bars = [];
  let guard = 0;
  while (bars.length < count && guard < count * 2000) {
    guard++;
    const w = 0.4 + rng() * 0.4;
    const support = w / Math.sqrt(2);
    const x = support + rng() * Math.max(0, 40 - 2 * support);
    const y = -20 + support + rng() * Math.max(0, 40 - 2 * support);
    let ok = true;
    for (const bar of bars) {
      if (Math.hypot(x - bar.x, y - bar.y) < support + bar.w / Math.sqrt(2) + 0.45) {
        ok = false; break;
      }
    }
    if (ok) bars.push({x: x, y: y, w: w});
  }
  return bars;
}

function openSession() {
  return Planner.createSession({
    bars: [],
    arenaLo: arenaLo,
    arenaHi: arenaHi,
    rng: Motion.seededRng(11),
    speed: 1.5,
    target: {x: 20, y: 0, heading: 0},
    pursuer: {x: 8, y: 0, heading: 0},
  });
}

// Occupancy cache must not change historical A* waypoints.
const support = Route.conservativeXYSupportFromBox(Route.CONTRACT.physicalBoxXYZ);
const barsOne = [{x: 5, y: 0, w: 1, h: 1}];
const cached = Route.createMapCache(barsOne, {x: 0, y: -5}, {x: 10, y: 5}, support);
const plain = Route.plan(
  {x: 2, y: -2}, {x: 8, y: 2}, barsOne, {x: 0, y: -5}, {x: 10, y: 5}, support
);
const reused = Route.plan(
  {x: 2, y: -2}, {x: 8, y: 2}, barsOne, {x: 0, y: -5}, {x: 10, y: 5}, support,
  {mapCache: cached}
);
assert.strictEqual(reused.status, plain.status);
assert.deepStrictEqual(reused.waypoints, plain.waypoints);
assert.strictEqual(reused.expandedNodes, plain.expandedNodes);

const empty = openSession();
Planner.stepSession(empty, 0.1);
assert(empty.target.route && empty.target.route.valid, 'empty arena must find a roam route');
assert(['DIRECT LOS', 'FOLLOWING', 'ROUTE FOLLOW', 'REPLANNING'].includes(empty.pursuer.status),
  empty.pursuer.status);

// Fail-closed: a pursuer boxed in by a wall must stop, not teleport or clip.
const wall = [{x: 20, y: 0, w: 1.6, h: 36}];
const trapped = Planner.createSession({
  bars: wall,
  arenaLo: arenaLo,
  arenaHi: arenaHi,
  rng: Motion.seededRng(3),
  speed: 1.5,
  target: {x: 32, y: 0, heading: 0},
  pursuer: {x: 6, y: 0, heading: 0},
});
const start = {x: trapped.pursuer.x, y: trapped.pursuer.y};
for (let i = 0; i < 8; i++) Planner.stepPursuerGt(trapped, 0.1);
assert(Number.isFinite(trapped.pursuer.x) && Number.isFinite(trapped.pursuer.y));
assert(Math.hypot(trapped.pursuer.x - start.x, trapped.pursuer.y - start.y) < 3.5);
assert(Planner.pointSafe({x: trapped.pursuer.x, y: trapped.pursuer.y}, trapped.caches.pursuer)
  || trapped.pursuer.status === 'NO SAFE ROUTE');

// Click-goal rejects bars / out of bounds and accepts a free cell.
const roam = openSession();
assert.strictEqual(Planner.classifyClickGoal({x: -4, y: 0}, roam.caches.target), 'out_of_bounds');
const blockedClick = Planner.createSession({
  bars: [{x: 20, y: 0, w: 4, h: 4}],
  arenaLo: arenaLo, arenaHi: arenaHi,
  rng: Motion.seededRng(4), speed: 1.5,
  target: {x: 8, y: 8, heading: 0},
  pursuer: {x: 6, y: 8, heading: 0},
});
assert.notStrictEqual(Planner.classifyClickGoal({x: 20, y: 0}, blockedClick.caches.target), 'ok');
const accepted = Planner.setClickGoal(roam, {x: 28, y: 8});
assert.strictEqual(accepted.ok, true);

// No GT field names leak into this browser-only module's public contract as research APIs.
assert.strictEqual(Planner.CONTRACT.standoffM > 0.5, true, 'display standoff is not capture radius');

const fps30 = openSession();
const fps60 = openSession();
const clock30 = Motion.createFixedStepClock(0.1, 0.25, 8);
const clock60 = Motion.createFixedStepClock(0.1, 0.25, 8);
clock30.advance(0, true, function () {});
clock60.advance(0, true, function () {});
for (let frame = 1; frame <= 30 * 4; frame++) {
  clock30.advance(frame / 30, true, function (dt) { Planner.stepSession(fps30, dt); });
}
for (let frame = 1; frame <= 60 * 4; frame++) {
  clock60.advance(frame / 60, true, function (dt) { Planner.stepSession(fps60, dt); });
}
assert(Math.abs(fps30.target.x - fps60.target.x) < 1e-8);
assert(Math.abs(fps30.pursuer.x - fps60.pursuer.x) < 1e-8);

function runChase(count) {
  const bars = denseBars(count, 20260917 + count);
  const session = Planner.createSession({
    bars: bars,
    arenaLo: arenaLo,
    arenaHi: arenaHi,
    rng: Motion.seededRng(count),
    speed: 1.2,
    target: {x: 6, y: 0, heading: 0},
    pursuer: {x: 4, y: 2, heading: 0},
  });
  for (let step = 0; step < 120; step++) Planner.stepSession(session, 0.1);
  return session;
}

for (const count of [70, 115, 160, 205]) {
  const session = runChase(count);
  const t = session.target, p = session.pursuer;
  assert(Number.isFinite(t.x) && Number.isFinite(t.y) && Number.isFinite(t.heading));
  assert(Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.heading));
  assert(Planner.pointSafe({x: t.x, y: t.y}, session.caches.target)
    || session.labels.follow === 'NO SAFE ROUTE'
    || t.status === 'NO SAFE ROUTE');
  assert(Planner.pointSafe({x: p.x, y: p.y}, session.caches.pursuer)
    || p.status === 'NO SAFE ROUTE');
  assert.strictEqual(session.stats.pursuer.collisions, 0, 'GT pursuer penetrated at ' + count);
  assert.strictEqual(session.stats.target.collisions, 0, 'GT target penetrated at ' + count);
  assert(session.stats.pursuer.maxJump < 0.45, 'pursuer teleported at ' + count);
  assert(session.stats.target.maxJump < 0.45, 'target teleported at ' + count);
  assert(session.stats.pursuer.maxHeadingDelta <= 150 * Math.PI / 180 + 1e-6);
  assert(session.stats.pursuer.headingFlips === 0, 'GT pursuer heading flips at ' + count);
  assert(session.stats.pursuer.period2 < 8, 'GT pursuer period-2 heading oscillation at ' + count);
  console.log(
    'chase ' + count + ' bars: collisions=' + session.stats.pursuer.collisions
    + ' fail=' + session.stats.pursuer.plannerFailures
    + ' stall=' + session.stats.pursuer.stallSteps
    + ' replans=' + session.stats.pursuer.replans
    + ' maxJump=' + session.stats.pursuer.maxJump.toFixed(3)
    + ' maxHead=' + (session.stats.pursuer.maxHeadingDelta * 180 / Math.PI).toFixed(1)
    + ' period2=' + session.stats.pursuer.period2
  );
}

console.log('MOTAR browser GT preview contracts: PASS');
