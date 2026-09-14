'use strict';

const assert = require('assert');
const crypto = require('crypto');
const {execFileSync} = require('child_process');
const fs = require('fs');
const path = require('path');
const {performance} = require('perf_hooks');
const Route = require('../docs/status/arena_route.js');
const Motion = require('../docs/status/arena_motion.js');

// The browser preview mirrors the frozen global_astar_v1 geometry from a373202.  The current
// Python file additionally contains the failed recovery-v2 state machine, which this explanatory
// viewer must not silently present as implemented or successful.
const repo = path.resolve(__dirname, '..');
const pythonPlanner = execFileSync('git', [
  'show', 'a373202:aerial_gym/task/navrl_task/target_route_planner.py',
], {cwd: repo});
assert.strictEqual(
  crypto.createHash('sha256').update(pythonPlanner).digest('hex'),
  '7fec3015e5dee667b8cd64d145d29b9244c18eb0c79b4af12020be44b503cb83'
);

const support = Route.conservativeXYSupportFromBox([0.28, 0.28, 0.12]);
const expectedSupport = 0.5 * Math.hypot(0.28, 0.28, 0.12);
assert(Math.abs(support.x - expectedSupport) < 1e-15);
assert.strictEqual(Route.CONTRACT.resolutionM, 0.25);
assert.strictEqual(Route.CONTRACT.trackingMarginM, 0.45);
assert.strictEqual(Route.CONTRACT.boundaryMarginM, 1.25);
assert.strictEqual(Route.CONTRACT.minGoalDistanceM, 6.0);
assert.strictEqual(Route.CONTRACT.goalExclusionRadiusM, 1.0);

// Exact closed AABB semantics: grazing a corner is a collision, a segment below it is safe.
assert.strictEqual(Route.segmentIsSafe(
  {x: 0.2, y: 0.2}, {x: 1.8, y: 1.8}, {x: 0, y: 0}, {x: 3, y: 3},
  [{x: 1, y: 1}], [{x: 0.2, y: 0.2}]
), false);
assert.strictEqual(Route.segmentIsSafe(
  {x: 0.2, y: 0.2}, {x: 1.8, y: 0.5}, {x: 0, y: 0}, {x: 3, y: 3},
  [{x: 1, y: 1}], [{x: 0.2, y: 0.2}]
), true);

// Golden receipts from topology branch fcec3d2, including deterministic node counts.
const empty = Route.planToConnectedGoal(
  {x: 2, y: 0}, [], {x: 0, y: -20}, {x: 40, y: 20}, support, 0.37
);
assert.strictEqual(empty.status, 'ok');
assert.strictEqual(empty.expandedNodes, 21904);
assert.deepStrictEqual(empty.waypoints, [{x: 2, y: 0}, {x: 16.375, y: -18.125}]);
assert(Math.abs(empty.pathLengthM - 23.133444404152183) < 1e-12);

const corner = Route.plan(
  {x: 2, y: -2}, {x: 8, y: 2}, [{x: 5, y: 0, w: 1, h: 1}],
  {x: 0, y: -5}, {x: 10, y: 5}, support
);
assert.strictEqual(corner.status, 'ok');
assert.strictEqual(corner.expandedNodes, 280);
assert.strictEqual(corner.rawGridNodes, 34);
assert.deepStrictEqual(corner.waypoints, [
  {x: 2, y: -2}, {x: 3.875, y: 1.375}, {x: 8, y: 2},
]);

// Python main a373202 exact-corner regression: 0.5 m reach alone would accept this shortcut, but the
// outgoing segment has only ~0.018 m exact inflated-AABB clearance. The cached certificate must
// reject the unsafe point and still permit a point inside the certified ball.
const tightCorner = Route.plan(
  {x: 1, y: 5}, {x: 9, y: 5}, [{x: 5, y: 5, w: 1.6, h: 4.8}],
  {x: 0, y: 0}, {x: 10, y: 10}, {x: 0.2, y: 0.2},
  {resolutionM: 0.2, trackingMarginM: 0.1, boundaryMarginM: 0}
);
assert.strictEqual(tightCorner.status, 'ok');
assert(Math.abs(tightCorner.handoffClearanceM[1] - 0.018007149208503482) < 1e-12);
const unsafeEarly = {x: 3.65, y: 2.30};
assert(Math.hypot(
  unsafeEarly.x - tightCorner.waypoints[1].x,
  unsafeEarly.y - tightCorner.waypoints[1].y
) < 0.5);
assert.strictEqual(Route.segmentIsSafe(
  unsafeEarly, tightCorner.waypoints[2], {x: 0.2, y: 0.2}, {x: 9.8, y: 9.8},
  [{x: 5, y: 5}], [{x: 1.1, y: 2.7}]
), false);
const cornerEpisode = Motion.createEpisode(Motion.seededRng(101), [], 1);
cornerEpisode.target = Object.assign({}, unsafeEarly);
cornerEpisode.route = {
  valid: true, status: 'ok', waypoints: tightCorner.waypoints.slice(1),
  handoffClearanceM: tightCorner.handoffClearanceM.slice(1), cursor: 0,
  segmentStart: tightCorner.waypoints[0], waypointReachM: 0.5,
  goal: tightCorner.waypoints[tightCorner.waypoints.length - 1],
};
Motion.routeVelocityReference(cornerEpisode, false);
assert.strictEqual(cornerEpisode.route.cursor, 0, 'unsafe corner shortcut was accepted');
cornerEpisode.target = {
  x: tightCorner.waypoints[1].x + 0.25 * tightCorner.handoffClearanceM[1],
  y: tightCorner.waypoints[1].y,
};
Motion.routeVelocityReference(cornerEpisode, false);
assert.strictEqual(cornerEpisode.route.cursor, 1, 'certified corner handoff was rejected');

const openOvershoot = Motion.createEpisode(Motion.seededRng(102), [], 1);
openOvershoot.target = {x: 1.2, y: 0};
openOvershoot.route = {
  valid: true, status: 'ok', waypoints: [{x: 1, y: 0}, {x: 2, y: 0}],
  handoffClearanceM: [10, 0], cursor: 0, segmentStart: {x: 0, y: 0},
  waypointReachM: 0.5, goal: {x: 2, y: 0},
};
Motion.routeVelocityReference(openOvershoot, false);
assert.strictEqual(openOvershoot.route.cursor, 1, 'safe open-space overshoot regressed');

// The continuous start is safe but its nearest raster centre (5.875, 5.0) is occupied. Python's
// deterministic 7x7 anchor search connects it to (6.125, 5.0) instead of rejecting the route.
const anchored = Route.plan(
  {x: 5.91, y: 5}, {x: 1, y: 5}, [{x: 5, y: 5, w: 1, h: 1}],
  {x: 0, y: 0}, {x: 10, y: 10}, {x: 0.2, y: 0.2},
  {trackingMarginM: 0.2, boundaryMarginM: 0}
);
assert.strictEqual(anchored.status, 'ok');
assert.strictEqual(anchored.expandedNodes, 91);
assert.strictEqual(anchored.rawGridNodes, 28);
assert.deepStrictEqual(anchored.waypoints, [
  {x: 5.91, y: 5}, {x: 6.125, y: 3.875}, {x: 2.375, y: 4.125}, {x: 1, y: 5},
]);

// A full inflated wall has no route and must return no optimistic waypoint fallback.
const noPath = Route.plan(
  {x: 1, y: 5}, {x: 9, y: 5}, [{x: 5, y: 5, w: 0.6, h: 10}],
  {x: 0, y: 0}, {x: 10, y: 10}, {x: 0.2, y: 0.2},
  {trackingMarginM: 0.1, boundaryMarginM: 0}
);
assert.strictEqual(noPath.status, 'no_path');
assert.strictEqual(noPath.valid, false);
assert.deepStrictEqual(noPath.waypoints, []);

const firstConnected = Route.planToConnectedGoal(
  {x: 2, y: 0}, [], {x: 0, y: -20}, {x: 40, y: 20}, support, 0.37
);
const previousGoal = firstConnected.waypoints[firstConnected.waypoints.length - 1];
const alternative = Route.planToConnectedGoal(
  {x: 2, y: 0}, [], {x: 0, y: -20}, {x: 40, y: 20}, support, 0.37,
  {excludedGoal: previousGoal, goalExclusionRadiusM: 1.0}
);
assert.strictEqual(alternative.status, 'ok');
const alternativeGoal = alternative.waypoints[alternative.waypoints.length - 1];
assert.deepStrictEqual(alternativeGoal, {x: 16.375, y: -16.375});
assert.strictEqual(alternative.expandedNodes, 21904);
assert.strictEqual(alternative.rawGridNodes, 67);
assert(Math.abs(alternative.handoffClearanceM[0] - 0.5430183913442279) < 1e-12);
assert(Math.hypot(
  alternativeGoal.x - previousGoal.x, alternativeGoal.y - previousGoal.y
) > 1.0);
const noAlternative = Route.planToConnectedGoal(
  {x: 0.5, y: 0.5}, [], {x: 0, y: 0}, {x: 7.2, y: 1},
  {x: 0.1, y: 0.1}, 0.5,
  {boundaryMarginM: 0, excludedGoal: {x: 6.875, y: 0.5}, goalExclusionRadiusM: 1.0}
);
assert.strictEqual(noAlternative.status, 'no_alternative_goal');
assert.strictEqual(noAlternative.valid, false);
const noAlternativeEpisode = Motion.createEpisode(Motion.seededRng(8), [], 1.5);
noAlternativeEpisode.target = {x: 0.5, y: 0.5};
noAlternativeEpisode.physicalStyle.velocity = {x: 1, y: 0};
noAlternativeEpisode.route = noAlternative;
Motion.advanceTarget(
  noAlternativeEpisode, 0.1, [], Motion.seededRng(9), 'routed-preview'
);
assert.deepStrictEqual(noAlternativeEpisode.target, {x: 0.5, y: 0.5});
assert.deepStrictEqual(noAlternativeEpisode.realizedVelocity, {x: 0, y: 0});

// Routed no-path preview is explicitly zero/fail-closed even if stale velocity exists.
Motion.configure({arena_xy_m: 40, goal_dist_m: [6, 28], target_speed_m: [0.3, 1.5]});
const failedEpisode = Motion.createEpisode(Motion.seededRng(9), [], 1.5);
failedEpisode.target = {x: 10, y: 0};
failedEpisode.physicalStyle.velocity = {x: 1, y: 0};
failedEpisode.route = {valid: false, status: 'no_path', waypoints: []};
Motion.advanceTarget(failedEpisode, 0.1, [], Motion.seededRng(10), 'routed-preview');
assert.deepStrictEqual(failedEpisode.target, {x: 10, y: 0});
assert.deepStrictEqual(failedEpisode.realizedVelocity, {x: 0, y: 0});
assert.strictEqual(failedEpisode.plannerFeasible, false);

function routedEpisode() {
  const episode = Motion.createEpisode(Motion.seededRng(77), [], 1.5);
  episode.speed = 1.5;
  episode.target = {x: 10, y: 0};
  episode.heading = 0;
  episode.physicalStyle.velocity = {x: 0, y: 0};
  episode.route = {
    valid: true, status: 'ok', waypoints: [{x: 20, y: 0}],
    handoffClearanceM: [0], goal: {x: 20, y: 0},
    cursor: 0, segmentStart: {x: 10, y: 0}, waypointReachM: 0.5,
    goalToleranceM: 0.05,
  };
  return episode;
}

function runAtFps(fps) {
  const episode = routedEpisode();
  const clock = Motion.createFixedStepClock(0.1, 0.25, 8);
  let steps = 0;
  clock.advance(0, true, function () {});
  for (let frame = 1; frame <= fps * 5; frame++) {
    clock.advance(frame / fps, true, function (dt) {
      Motion.advanceTarget(episode, dt, [], Motion.seededRng(1), 'routed-preview');
      steps++;
    });
  }
  clock.advance(5 + 1e-9, true, function (dt) {
    Motion.advanceTarget(episode, dt, [], Motion.seededRng(1), 'routed-preview'); steps++;
  });
  return {episode, steps};
}

const at30 = runAtFps(30);
assert.strictEqual(at30.steps, 50);
for (const run of [runAtFps(60), runAtFps(144)]) {
  assert.strictEqual(run.steps, 50);
  assert(Math.abs(run.episode.target.x - at30.episode.target.x) < 1e-12);
  assert(Math.abs(run.episode.target.y - at30.episode.target.y) < 1e-12);
  assert(Math.abs(run.episode.realizedVelocity.x - at30.episode.realizedVelocity.x) < 1e-12);
}

// Python passes waypoint_reach_m=0.5 (not the route goal tolerance 0.05) and immediately plans a
// new connected goal after completion. Exercise the same same-interval replacement contract.
const continuous = routedEpisode();
continuous.route.waypoints = [{x: 16.5, y: 0}];
continuous.route.handoffClearanceM = [0];
continuous.route.goal = {x: 16.5, y: 0};
let replacements = 0, nearZeroRun = 0, longestNearZeroRun = 0;
function replacement(start, previousGoal) {
  replacements++;
  const goalX = start.x > 13 ? 6.5 : 16.5;
  if (previousGoal) assert(Math.hypot(goalX - previousGoal.x, -previousGoal.y) > 1.0);
  return {
    valid: true, status: 'ok', waypoints: [{x: goalX, y: 0}],
    handoffClearanceM: [0], goal: {x: goalX, y: 0},
    cursor: 0, segmentStart: {x: start.x, y: start.y}, waypointReachM: 0.5,
    goalToleranceM: 0.05, replan: replacement,
  };
}
continuous.route.replan = replacement;
for (let step = 0; step < 300; step++) {
  Motion.advanceTarget(continuous, 0.1, [], Motion.seededRng(1), 'routed-preview');
  const speed = Math.hypot(
    continuous.realizedVelocity.x, continuous.realizedVelocity.y
  );
  nearZeroRun = speed < 0.05 ? nearZeroRun + 1 : 0;
  longestNearZeroRun = Math.max(longestNearZeroRun, nearZeroRun);
}
assert(replacements >= 1, '30 s routed preview must replace a completed goal');
assert.strictEqual(continuous.routeGoalReplacements, replacements);
assert.strictEqual(longestNearZeroRun, 0, 'valid continuous routes must not create a long stall');

function denseFixture(count) {
  const bars = [];
  // Deterministic compound clusters occupy the central band while x=2 remains a valid start.
  for (let i = 0; i < count; i++) {
    const cluster = i % 25, layer = Math.floor(i / 25);
    bars.push({
      x: 8 + (cluster % 5) * 5.5 + (layer % 3) * 0.08,
      y: -13 + Math.floor(cluster / 5) * 6.5 + (layer % 4) * 0.07,
      w: 0.4 + (i % 5) * 0.1,
    });
  }
  return bars;
}

for (const count of [205, 300]) {
  const started = performance.now();
  const result = Route.planToConnectedGoal(
    {x: 2, y: 0}, denseFixture(count),
    {x: 0, y: -20}, {x: 40, y: 20}, support, 0.51
  );
  const elapsed = performance.now() - started;
  assert(['ok', 'no_connected_goal'].includes(result.status), result.status);
  assert(elapsed < 3000, `${count}-bar routed preview took ${elapsed.toFixed(1)} ms`);
  console.log(`route runtime ${count} bars: ${elapsed.toFixed(1)} ms (${result.status})`);
}

console.log('MOTAR routed-preview geometry and determinism contracts: PASS');
