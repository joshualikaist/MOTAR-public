/*
 * Browser-only GT tracking / free-roam presentation planner.
 *
 * This is not the PPO policy, not PhysX, not TM-E3, and not a performance
 * measurement. Historical arena_route.js geometry remains the source of the
 * A* contract; this module adds occupancy reuse, predicted follow points,
 * event-driven replans, LOS shortcuts, and fail-closed kinematics.
 */
(function (root, factory) {
  const Route = (typeof module === 'object' && module.exports)
    ? require('./arena_route.js') : root.NavRLArenaRoute;
  const Motion = (typeof module === 'object' && module.exports)
    ? require('./arena_motion.js') : root.NavRLArenaMotion;
  if (!Route || !Motion) throw new Error('arena_demo_planner requires arena_route and arena_motion');
  const api = factory(Route, Motion);
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.NavRLArenaDemoPlanner = api;
})(typeof window !== 'undefined' ? window : this, function (Route, Motion) {
  'use strict';

  const CONTRACT = Object.freeze({
    predictionHorizonMin: 0.2,
    predictionHorizonMax: 0.9,
    standoffM: 1.55,
    replanPeriodS: 0.5,
    noRouteRetryS: 0.2,
    goalMoveReplanM: 1.4,
    deviationReplanM: 1.2,
    lookAheadM: 1.8,
    goalReachM: 0.9,
    // The roam-goal minimum distance is NOT redeclared here: Route.CONTRACT
    // .minGoalDistanceM owns it, and a second copy silently drifts.
    minSpawnComponentCells: 400,
    minTurnSpeedFraction: 0.25,
    clickFeedback: Object.freeze({
      ok: 'ok',
      out_of_bounds: 'out_of_bounds',
      inside_bar: 'inside_bar',
      unsafe: 'unsafe',
    }),
  });

  const TARGET_SUPPORT = Route.conservativeXYSupportFromBox(Route.CONTRACT.physicalBoxXYZ);
  const PURSUER_SUPPORT = Object.freeze({
    x: Motion.CONTRACT.pursuerRadius,
    y: Motion.CONTRACT.pursuerRadius,
  });

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function hypot(x, y) {
    return Math.hypot(x, y);
  }

  function copyPoint(p) {
    return {x: p.x, y: p.y};
  }

  function finitePoint(p) {
    return p && Number.isFinite(p.x) && Number.isFinite(p.y);
  }

  function visualAttitude(velocity, accel, heading) {
    const speed = hypot(velocity.x, velocity.y);
    const yaw = speed > 1e-4 ? Math.atan2(velocity.y, velocity.x) : heading;
    const ch = Math.cos(yaw), sh = Math.sin(yaw);
    const forward = accel.x * ch + accel.y * sh;
    const lateral = -accel.x * sh + accel.y * ch;
    const limit = Math.tan(Motion.CONTRACT.physicalStyleMaxTilt);
    return {
      heading: yaw,
      roll: clamp(-lateral / 9.81, -limit, limit),
      pitch: clamp(forward / 9.81, -limit, limit),
    };
  }

  function createCaches(bars, arenaLo, arenaHi) {
    return {
      target: Route.createMapCache(bars, arenaLo, arenaHi, TARGET_SUPPORT),
      pursuer: Route.createMapCache(bars, arenaLo, arenaHi, PURSUER_SUPPORT),
    };
  }

  function preparedFrom(cache) {
    if (!cache) return null;
    return {
      geometry: cache.geometry,
      admissibleLo: cache.admissibleLo,
      admissibleHi: cache.admissibleHi,
    };
  }

  function pointSafe(point, cache) {
    if (!cache || !finitePoint(point)) return false;
    return Route.segmentIsSafe(
      point, point, cache.admissibleLo, cache.admissibleHi,
      cache.geometry.centers, cache.geometry.half
    );
  }

  function segmentSafe(a, b, cache) {
    if (!cache || !finitePoint(a) || !finitePoint(b)) return false;
    return Route.segmentIsSafe(
      a, b, cache.admissibleLo, cache.admissibleHi,
      cache.geometry.centers, cache.geometry.half
    );
  }

  function classifyClickGoal(point, cache) {
    if (!finitePoint(point) || !cache) return CONTRACT.clickFeedback.out_of_bounds;
    const lo = cache.admissibleLo, hi = cache.admissibleHi;
    if (point.x <= lo.x || point.x >= hi.x || point.y <= lo.y || point.y >= hi.y) {
      return CONTRACT.clickFeedback.out_of_bounds;
    }
    for (let i = 0; i < cache.geometry.centers.length; i++) {
      const c = cache.geometry.centers[i], half = cache.geometry.half[i];
      if (Math.abs(point.x - c.x) <= half.x && Math.abs(point.y - c.y) <= half.y) {
        return CONTRACT.clickFeedback.inside_bar;
      }
    }
    if (!pointSafe(point, cache)) return CONTRACT.clickFeedback.unsafe;
    return CONTRACT.clickFeedback.ok;
  }

  function planAgentRoute(args) {
    const start = args.start, goal = args.goal, bars = args.bars;
    const support = args.support || TARGET_SUPPORT;
    const cache = args.cache;
    return Route.plan(
      start, goal, bars, args.arenaLo, args.arenaHi, support,
      Object.assign({}, args.overrides || {}, cache ? {mapCache: cache} : {})
    );
  }

  function sampleReachableGoal(start, bars, arenaLo, arenaHi, support, selector, cache, extra) {
    return Route.planToConnectedGoal(
      start, bars, arenaLo, arenaHi, support, selector,
      Object.assign({}, extra || {}, cache ? {mapCache: cache} : {})
    );
  }

  function predictedFollowPoint(target, pursuer) {
    const speed = hypot(target.vx, target.vy);
    const dist = hypot(target.x - pursuer.x, target.y - pursuer.y);
    const horizon = clamp(
      dist / Math.max(speed, 0.45),
      CONTRACT.predictionHorizonMin,
      CONTRACT.predictionHorizonMax
    );
    let x = target.x + target.vx * horizon;
    let y = target.y + target.vy * horizon;
    if (dist < CONTRACT.standoffM + 0.15) {
      const hx = speed > 1e-3 ? target.vx / speed : Math.cos(target.heading || 0);
      const hy = speed > 1e-3 ? target.vy / speed : Math.sin(target.heading || 0);
      x = target.x - hx * CONTRACT.standoffM;
      y = target.y - hy * CONTRACT.standoffM;
    }
    return {x: x, y: y, horizon: horizon};
  }

  function snapToSafe(point, cache) {
    if (pointSafe(point, cache)) return copyPoint(point);
    for (let radius = 0.25; radius <= 3.0; radius += 0.25) {
      for (let k = 0; k < 16; k++) {
        const angle = k * Math.PI / 8;
        const candidate = {
          x: point.x + radius * Math.cos(angle),
          y: point.y + radius * Math.sin(angle),
        };
        if (pointSafe(candidate, cache)) return candidate;
      }
    }
    return copyPoint(point);
  }

  /* A follow point that is unsafe for the TRACKER is common in dense fields:
   * the tracker's support disc plus tracking margin is larger than the target's,
   * so a gap the target slips through is closed for the tracker. Declaring the
   * whole approach unroutable there stalls the follow; snapping the reference
   * onto the nearest certified-safe cell keeps a legal approach goal. */
  function nearestSafe(point, cache, fallback) {
    if (pointSafe(point, cache)) return copyPoint(point);
    if (fallback && pointSafe(fallback, cache)) return copyPoint(fallback);
    const snapped = snapToSafe(point, cache);
    if (pointSafe(snapped, cache)) return snapped;
    if (fallback) {
      const around = snapToSafe(fallback, cache);
      if (pointSafe(around, cache)) return around;
    }
    return null;
  }

  function makeRouteFollow(result, start) {
    const valid = Boolean(result && result.valid);
    return {
      valid: valid,
      status: result ? result.status : 'invalid_input',
      waypoints: valid ? result.waypoints.map(copyPoint) : [],
      cursor: 0,
      pathLengthM: valid ? result.pathLengthM : 0,
      goal: valid ? copyPoint(result.waypoints[result.waypoints.length - 1]) : null,
      plannedGoal: valid ? copyPoint(result.waypoints[result.waypoints.length - 1]) : null,
    };
  }

  /* Certified look-ahead.
   *
   * A pure arc-length carrot is a point lookAheadM along the polyline, but the
   * follower drives the STRAIGHT line from the current pose to it. On a route
   * that doubles back around a bar that chord leaves the certified corridor and
   * points into the obstacle, which is what wedged the tracker against the
   * safety envelope with a valid route and a full-speed command. Candidates are
   * therefore collected along the route and the FARTHEST one joined to the pose
   * by a certified-safe segment is returned; null means every candidate is
   * blocked and the caller must replan rather than drive blind.
   */
  function carrotCandidates(pos, route, lookAheadM) {
    const out = [];
    let remaining = lookAheadM;
    let cursor = Math.max(0, route.cursor || 0);
    let ax = pos.x, ay = pos.y;
    while (cursor < route.waypoints.length && remaining > 1e-6) {
      const wp = route.waypoints[cursor];
      const dx = wp.x - ax, dy = wp.y - ay;
      const seg = hypot(dx, dy);
      if (seg <= 1e-6) { cursor += 1; continue; }
      if (seg >= remaining) {
        out.push({x: ax + dx / seg * remaining, y: ay + dy / seg * remaining, cursor: cursor});
        return out;
      }
      remaining -= seg;
      ax = wp.x; ay = wp.y;
      out.push({x: wp.x, y: wp.y, cursor: cursor});
      cursor += 1;
    }
    if (!out.length && route.waypoints.length) {
      const last = route.waypoints[route.waypoints.length - 1];
      out.push({x: last.x, y: last.y, cursor: route.waypoints.length - 1});
    }
    return out;
  }

  function carrotPoint(pos, route, lookAheadM, cache) {
    if (!route || !route.valid || !route.waypoints.length) return null;
    const candidates = carrotCandidates(pos, route, lookAheadM);
    if (!cache) return candidates.length ? candidates[candidates.length - 1] : null;
    for (let i = candidates.length - 1; i >= 0; i--) {
      if (segmentSafe(pos, candidates[i], cache)) return candidates[i];
    }
    return null;
  }

  /* Advance on arrival OR on overshoot. A radius-only test leaves the cursor
   * pinned to a waypoint the agent blew past at speed, and the follower then
   * aims backwards along the route. */
  function advanceAlongRoute(pos, route, reachM) {
    if (!route || !route.valid) return;
    while (route.cursor < route.waypoints.length) {
      const wp = route.waypoints[route.cursor];
      if (hypot(wp.x - pos.x, wp.y - pos.y) <= reachM) { route.cursor += 1; continue; }
      if (route.cursor + 1 >= route.waypoints.length) break;
      const from = route.cursor > 0 ? route.waypoints[route.cursor - 1] : null;
      if (!from) break;
      const dx = wp.x - from.x, dy = wp.y - from.y;
      const denom = dx * dx + dy * dy;
      if (denom <= 1e-12) { route.cursor += 1; continue; }
      const t = ((pos.x - from.x) * dx + (pos.y - from.y) * dy) / denom;
      if (t > 1) { route.cursor += 1; continue; }
      break;
    }
  }

  function routeStillValid(pos, route, cache) {
    if (!route || !route.valid || !route.waypoints.length) return false;
    const next = route.waypoints[Math.min(route.cursor, route.waypoints.length - 1)];
    return segmentSafe(pos, next, cache);
  }

  function deviationFromRoute(pos, route) {
    if (!route || !route.valid || !route.waypoints.length) return Infinity;
    const prev = route.cursor > 0 ? route.waypoints[route.cursor - 1] : pos;
    const next = route.waypoints[Math.min(route.cursor, route.waypoints.length - 1)];
    const dx = next.x - prev.x, dy = next.y - prev.y;
    const denom = dx * dx + dy * dy;
    const t = denom > 1e-12
      ? clamp(((pos.x - prev.x) * dx + (pos.y - prev.y) * dy) / denom, 0, 1) : 0;
    return hypot(pos.x - (prev.x + t * dx), pos.y - (prev.y + t * dy));
  }

  /* Bounded fail-closed step selection.
   *
   * The historical implementation zeroed the velocity the instant the proposed
   * swept segment was unsafe. That is fail-closed, but 2.5 m/s -> 0 inside one
   * 0.1 s tick is a 25 m/s^2 deceleration against a 4.0 m/s^2 contract, and
   * because the rejected command was re-issued unchanged on the next tick the
   * agent latched into a stop/stutter loop instead of slowing into the turn.
   * Every candidate below is passed through the SAME limitPlanarVelocity
   * bound, so acceleration and turn rate stay inside the contract whichever
   * candidate is accepted; scaling the request down is what shortens the swept
   * segment until it is certified safe. CANDIDATE_SCALES ends at 0, which
   * limitPlanarVelocity turns into the maximum-rate bounded brake along the
   * current heading.
   */
  const CANDIDATE_SCALES = Object.freeze([1, 0.7, 0.45, 0.25, 0.1, 0]);

  function boundedCandidate(previous, desired, scale, speedLimit, dt) {
    return Motion.limitPlanarVelocity(
      previous, {x: desired.x * scale, y: desired.y * scale}, speedLimit, dt,
      Motion.CONTRACT.boundedMaxAccel, Motion.CONTRACT.boundedMaxTurnRate
    );
  }

  /* Command speed shaped by the turn the agent still owes.
   *
   * limitPlanarVelocity rotates the command by at most maxTurnRate*dt away
   * from the CURRENT velocity heading and keeps the requested magnitude, so a
   * full-speed request across a sharp corner is executed as "keep barrelling
   * forward, 15 degrees at a time". Slowing into the turn is what lets the
   * heading catch up before the swept segment leaves the corridor.
   */
  function approachVelocity(pos, goal, agent, speedLimit) {
    const dx = goal.x - pos.x, dy = goal.y - pos.y;
    const n = hypot(dx, dy);
    if (n <= 1e-6) return {x: 0, y: 0};
    const want = Math.atan2(dy, dx);
    const speed = hypot(agent.vx, agent.vy);
    const current = speed > 1e-4 ? Math.atan2(agent.vy, agent.vx) : want;
    const error = Math.abs(Math.atan2(Math.sin(want - current), Math.cos(want - current)));
    const capped = speedLimit * clamp(Math.cos(error), CONTRACT.minTurnSpeedFraction, 1);
    return {x: dx / n * capped, y: dy / n * capped};
  }

  function integrateBounded(agent, desired, speedLimit, dt, cache) {
    const previous = {x: agent.vx, y: agent.vy};
    const here = {x: agent.x, y: agent.y};
    let limited = null;
    let next = null;
    for (let i = 0; i < CANDIDATE_SCALES.length; i++) {
      const candidate = boundedCandidate(previous, desired, CANDIDATE_SCALES[i], speedLimit, dt);
      const step = {x: agent.x + candidate.x * dt, y: agent.y + candidate.y * dt};
      if (pointSafe(step, cache) && segmentSafe(here, step, cache)) {
        limited = candidate;
        next = step;
        break;
      }
    }
    if (!limited) {
      // Even the bounded brake would sweep through the safety envelope. Hold
      // the certified-safe pose and keep bleeding speed at the contract rate;
      // this is an emergency hold, not an unbounded stop or a teleport.
      const brake = boundedCandidate(previous, {x: 0, y: 0}, 0, speedLimit, dt);
      agent.vx = brake.x;
      agent.vy = brake.y;
      agent.accel = {
        x: (brake.x - previous.x) / Math.max(dt, 1e-6),
        y: (brake.y - previous.y) / Math.max(dt, 1e-6),
      };
      const attitude = visualAttitude(brake, agent.accel, agent.heading);
      agent.roll = attitude.roll;
      agent.pitch = attitude.pitch;
      agent.emergencyHolds = (agent.emergencyHolds || 0) + 1;
      return {moved: false, jump: 0, headingDelta: 0, held: true};
    }
    const moved = hypot(next.x - agent.x, next.y - agent.y);
    const oldHeading = agent.heading;
    agent.x = next.x;
    agent.y = next.y;
    agent.vx = limited.x;
    agent.vy = limited.y;
    agent.accel = {
      x: (limited.x - previous.x) / Math.max(dt, 1e-6),
      y: (limited.y - previous.y) / Math.max(dt, 1e-6),
    };
    const speed = hypot(limited.x, limited.y);
    const desiredHeading = speed > 1e-4 ? Math.atan2(limited.y, limited.x) : oldHeading;
    const turn = clamp(
      Math.atan2(Math.sin(desiredHeading - oldHeading), Math.cos(desiredHeading - oldHeading)),
      -Motion.CONTRACT.boundedMaxTurnRate * dt,
      Motion.CONTRACT.boundedMaxTurnRate * dt
    );
    agent.heading = oldHeading + turn;
    const attitude = visualAttitude(limited, agent.accel, agent.heading);
    agent.roll = attitude.roll;
    agent.pitch = attitude.pitch;
    const headingDelta = Math.abs(turn);
    return {moved: true, jump: moved, headingDelta: headingDelta};
  }

  function emptyStats() {
    return {
      collisions: 0,
      plannerFailures: 0,
      stallSteps: 0,
      replans: 0,
      maxHeadingDelta: 0,
      maxJump: 0,
      headingFlips: 0,
      steps: 0,
      lastHeading: null,
      prevHeadingDelta: 0,
      period2: 0,
    };
  }

  function noteStats(stats, agent, stepInfo, stalled, replan, failed, headingFlip) {
    stats.steps += 1;
    if (stalled) stats.stallSteps += 1;
    if (replan) stats.replans += 1;
    if (failed) stats.plannerFailures += 1;
    stats.maxJump = Math.max(stats.maxJump, stepInfo.jump || 0);
    stats.maxHeadingDelta = Math.max(stats.maxHeadingDelta, stepInfo.headingDelta || 0);
    if (headingFlip) stats.headingFlips += 1;
    if (stats.lastHeading != null && stepInfo.headingDelta > 25 * Math.PI / 180) {
      const sign = Math.sign(Math.atan2(
        Math.sin(agent.heading - stats.lastHeading),
        Math.cos(agent.heading - stats.lastHeading)
      ));
      if (sign && sign === -stats.prevHeadingDelta) stats.period2 += 1;
      stats.prevHeadingDelta = sign;
    }
    stats.lastHeading = agent.heading;
    if (!Number.isFinite(agent.x) || !Number.isFinite(agent.y)
        || !Number.isFinite(agent.vx) || !Number.isFinite(agent.heading)) {
      stats.collisions += 1;
    }
  }

  /* Free-space connectivity over the occupancy grid the A* cache already built.
   *
   * At the upper densities the navrl_band layout merges touching bars into
   * compound walls, which seals off pockets. A spawn inside one is genuinely
   * unroutable: the follower correctly fail-closes to a zero command and the
   * preview then shows two frozen aircraft for the whole session. Connectivity
   * is a SPAWN precondition, so it is resolved once at session creation rather
   * than papered over in the controller. This reads the cached occupancy; it
   * adds no obstacle geometry of its own.
   */
  function freeComponents(cache) {
    if (cache.components) return cache.components;
    const mesh = cache.mesh, free = cache.free;
    const nx = mesh.shapeX, ny = mesh.shapeY;
    const label = new Int32Array(nx * ny).fill(-1);
    const sizes = [];
    const queue = new Int32Array(nx * ny);
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const root = i * ny + j;
        if (!free[root] || label[root] !== -1) continue;
        const id = sizes.length;
        let head = 0, tail = 0, count = 0;
        queue[tail++] = root;
        label[root] = id;
        while (head < tail) {
          const node = queue[head++];
          count += 1;
          const ci = (node / ny) | 0, cj = node % ny;
          for (let k = 0; k < 4; k++) {
            const ni = ci + (k === 0 ? -1 : k === 1 ? 1 : 0);
            const nj = cj + (k === 2 ? -1 : k === 3 ? 1 : 0);
            if (ni < 0 || nj < 0 || ni >= nx || nj >= ny) continue;
            const next = ni * ny + nj;
            if (!free[next] || label[next] !== -1) continue;
            label[next] = id;
            queue[tail++] = next;
          }
        }
        sizes.push(count);
      }
    }
    cache.components = {label: label, sizes: sizes, nx: nx, ny: ny};
    return cache.components;
  }

  function cellIndexOf(cache, point) {
    const mesh = cache.mesh;
    const i = clamp(Math.round((point.x - mesh.axisX[0]) / cache.resolutionM), 0, mesh.shapeX - 1);
    const j = clamp(Math.round((point.y - mesh.axisY[0]) / cache.resolutionM), 0, mesh.shapeY - 1);
    return {i: i, j: j, index: i * mesh.shapeY + j};
  }

  function componentAt(cache, point) {
    const comp = freeComponents(cache);
    return comp.label[cellIndexOf(cache, point).index];
  }

  /* Nearest certified-safe cell to `anchor` inside an accepted component. */
  function nearestCellIn(cache, anchor, accept) {
    const comp = freeComponents(cache);
    const mesh = cache.mesh;
    const origin = cellIndexOf(cache, anchor);
    let best = null, bestCost = Infinity;
    for (let i = 0; i < mesh.shapeX; i++) {
      for (let j = 0; j < mesh.shapeY; j++) {
        const index = i * mesh.shapeY + j;
        const id = comp.label[index];
        if (id < 0 || !accept(id)) continue;
        const cost = (i - origin.i) * (i - origin.i) + (j - origin.j) * (j - origin.j);
        if (cost >= bestCost) continue;
        const point = {x: mesh.axisX[i], y: mesh.axisY[j]};
        if (!pointSafe(point, cache)) continue;
        best = point; bestCost = cost;
      }
    }
    return best;
  }

  function largestComponentId(cache) {
    const comp = freeComponents(cache);
    let best = -1, bestSize = 0;
    for (let id = 0; id < comp.sizes.length; id++) {
      if (comp.sizes[id] > bestSize) { bestSize = comp.sizes[id]; best = id; }
    }
    return best;
  }

  /* Both agents must start in free space that is large enough to roam and
   * mutually reachable, otherwise the preview is a pair of frozen aircraft.
   * The relocation is deterministic (nearest accepted cell), so a given seed
   * still replays identically. */
  function resolveConnectedSpawn(caches, targetStart, pursuerStart) {
    const out = {target: copyPoint(targetStart), pursuer: copyPoint(pursuerStart),
      targetRelocated: false, pursuerRelocated: false};
    const tCache = caches.target, pCache = caches.pursuer;
    if (!tCache || !pCache || !tCache.mesh || !pCache.mesh) return out;
    const minCells = CONTRACT.minSpawnComponentCells;

    const tComp = freeComponents(tCache);
    let tId = componentAt(tCache, out.target);
    if (tId < 0 || tComp.sizes[tId] < minCells) {
      const moved = nearestCellIn(tCache, out.target,
        function (id) { return tComp.sizes[id] >= minCells; });
      if (moved) { out.target = moved; out.targetRelocated = true; }
      tId = componentAt(tCache, out.target);
    }

    // The tracker's support disc is larger, so it has its own occupancy grid.
    // Require it to share a component with the cell nearest the target.
    const pComp = freeComponents(pCache);
    const targetSideId = componentAt(pCache, out.target);
    const wanted = targetSideId >= 0 && pComp.sizes[targetSideId] >= minCells
      ? targetSideId : largestComponentId(pCache);
    if (componentAt(pCache, out.pursuer) !== wanted && wanted >= 0) {
      const moved = nearestCellIn(pCache, out.pursuer,
        function (id) { return id === wanted; });
      if (moved) { out.pursuer = moved; out.pursuerRelocated = true; }
    }
    return out;
  }

  function createSession(options) {
    const bars = options.bars || [];
    const arenaLo = options.arenaLo;
    const arenaHi = options.arenaHi;
    const caches = createCaches(bars, arenaLo, arenaHi);
    let targetStart = snapToSafe(options.target, caches.target);
    let pursuerStart = snapToSafe(options.pursuer, caches.pursuer);
    const spawn = resolveConnectedSpawn(caches, targetStart, pursuerStart);
    targetStart = spawn.target;
    pursuerStart = spawn.pursuer;
    const target = {
      x: targetStart.x, y: targetStart.y,
      vx: 0, vy: 0, heading: options.target.heading || 0,
      speed: options.speed || 1.5,
      roll: 0, pitch: 0, accel: {x: 0, y: 0},
      route: null, goal: null, status: 'IDLE',
      command: {x: 0, y: 0}, routeSwitches: 0, goalCompletions: 0,
    };
    const pursuer = {
      x: pursuerStart.x, y: pursuerStart.y,
      vx: 0, vy: 0, heading: options.pursuer.heading || 0,
      roll: 0, pitch: 0, accel: {x: 0, y: 0},
      route: null, lead: null, carrot: null,
      status: 'IDLE', replanCount: 0, lastPlanAt: -Infinity,
      plannedGoal: null,
      command: {x: 0, y: 0}, routeSwitches: 0, routeSerial: 0,
    };
    return {
      bars: bars,
      arenaLo: arenaLo,
      arenaHi: arenaHi,
      caches: caches,
      rng: options.rng,
      time: 0,
      goalMode: options.goalMode || 'auto',
      clickGoal: null,
      target: target,
      pursuer: pursuer,
      spawn: {targetRelocated: spawn.targetRelocated, pursuerRelocated: spawn.pursuerRelocated},
      stats: {target: emptyStats(), pursuer: emptyStats()},
      labels: {
        target: 'AUTO ROAM',
        pursuer: 'GT TRACK',
        follow: 'IDLE',
        cameraIndependent: true,
      },
    };
  }

  function maybeReplanTarget(session, force) {
    const t = session.target;
    const cache = session.caches.target;
    const due = force || session.time - (t.lastPlanAt || -Infinity) >= CONTRACT.replanPeriodS;
    const noRoute = !t.route || !t.route.valid;
    const complete = t.goal && hypot(t.x - t.goal.x, t.y - t.goal.y) <= CONTRACT.goalReachM;
    if (complete && session.goalMode === 'click' && !force) {
      t.status = 'AT GOAL';
      return false;
    }
    const invalid = t.route && !routeStillValid({x: t.x, y: t.y}, t.route, cache);
    if (!(due && (noRoute || complete || invalid || force))) return false;
    t.lastPlanAt = session.time;
    let result = null;
    if (session.goalMode === 'click' && session.clickGoal) {
      const kind = classifyClickGoal(session.clickGoal, cache);
      if (kind === CONTRACT.clickFeedback.ok) {
        result = planAgentRoute({
          start: {x: t.x, y: t.y}, goal: session.clickGoal,
          bars: session.bars, arenaLo: session.arenaLo, arenaHi: session.arenaHi,
          support: TARGET_SUPPORT, cache: cache,
        });
      }
    } else {
      for (let attempt = 0; attempt < 8 && (!result || !result.valid); attempt++) {
        result = sampleReachableGoal(
          {x: t.x, y: t.y}, session.bars, session.arenaLo, session.arenaHi,
          TARGET_SUPPORT, session.rng(), cache,
          t.goal ? {excludedGoal: t.goal, goalExclusionRadiusM: Route.CONTRACT.goalExclusionRadiusM} : {}
        );
      }
    }
    if (complete && t.route && t.route.valid) t.goalCompletions += 1;
    t.route = makeRouteFollow(result, {x: t.x, y: t.y});
    if (t.route.valid) t.routeSwitches += 1;
    t.goal = t.route.goal;
    if (!t.route.valid) {
      t.status = 'NO SAFE ROUTE';
      return true;
    }
    t.status = 'ROAMING';
    return true;
  }

  function stepTargetFreeRoam(session, dt) {
    const t = session.target;
    const cache = session.caches.target;
    let replanned = maybeReplanTarget(session, !t.route || !t.route.valid);
    let desired = {x: 0, y: 0};
    if (t.route && t.route.valid) {
      advanceAlongRoute({x: t.x, y: t.y}, t.route, CONTRACT.goalReachM * 0.45);
      let carrot = carrotPoint({x: t.x, y: t.y}, t.route, CONTRACT.lookAheadM, cache);
      if (!carrot) {
        // The certified corridor no longer reaches forward from this pose.
        // Replan instead of driving the blocked chord.
        replanned = maybeReplanTarget(session, true) || replanned;
        carrot = carrotPoint({x: t.x, y: t.y}, t.route, CONTRACT.lookAheadM, cache);
      }
      if (carrot) {
        desired = approachVelocity({x: t.x, y: t.y}, carrot, t, t.speed);
        t.status = 'ROAMING';
      } else {
        t.status = 'NO SAFE ROUTE';
      }
    } else {
      t.status = 'NO SAFE ROUTE';
    }
    t.command = {x: desired.x, y: desired.y};
    const info = integrateBounded(t, desired, t.speed, dt, cache);
    const stalled = hypot(t.vx, t.vy) < 0.05 && t.speed > 0.2;
    if (info.moved && !pointSafe({x: t.x, y: t.y}, cache)) session.stats.target.collisions += 1;
    noteStats(
      session.stats.target, t, info, stalled, replanned,
      t.status === 'NO SAFE ROUTE',
      info.headingDelta > 120 * Math.PI / 180
    );
    session.labels.target = session.goalMode === 'click' ? 'CLICK GOAL' : 'AUTO ROAM';
    return t;
  }

  function maybeReplanPursuer(session, predicted, force) {
    const p = session.pursuer;
    const cache = session.caches.pursuer;
    const due = session.time - p.lastPlanAt >= CONTRACT.replanPeriodS;
    const noRoute = !p.route || !p.route.valid;
    // A genuinely unroutable pose must not cost one A* expansion per tick.
    const retryDue = session.time - p.lastPlanAt >= CONTRACT.noRouteRetryS;
    if (noRoute && !retryDue && !force) return false;
    const goalMoved = p.plannedGoal
      ? hypot(predicted.x - p.plannedGoal.x, predicted.y - p.plannedGoal.y) >= CONTRACT.goalMoveReplanM
      : true;
    const invalid = p.route && !routeStillValid({x: p.x, y: p.y}, p.route, cache);
    const offPath = deviationFromRoute({x: p.x, y: p.y}, p.route) > CONTRACT.deviationReplanM;
    const exhausted = p.route && p.route.cursor >= (p.route.waypoints.length || 0);
    if (!(force || noRoute || (due && (goalMoved || invalid || offPath || exhausted)))) {
      return false;
    }
    p.lastPlanAt = session.time;
    const safeGoal = nearestSafe(predicted, cache, {x: session.target.x, y: session.target.y});
    if (!safeGoal) {
      p.route = makeRouteFollow({valid: false, status: 'unsafe_goal'}, p);
      p.status = 'NO SAFE ROUTE';
      p.replanCount += 1;
      return true;
    }
    const result = planAgentRoute({
      start: {x: p.x, y: p.y}, goal: safeGoal,
      bars: session.bars, arenaLo: session.arenaLo, arenaHi: session.arenaHi,
      support: PURSUER_SUPPORT, cache: cache,
    });
    p.route = makeRouteFollow(result, p);
    if (p.route.valid) { p.routeSwitches += 1; p.routeSerial += 1; }
    p.plannedGoal = copyPoint(safeGoal);
    p.replanCount += 1;
    if (!p.route.valid) p.status = 'NO SAFE ROUTE';
    return true;
  }

  function stepPursuerGt(session, dt) {
    const p = session.pursuer;
    const t = session.target;
    const cache = session.caches.pursuer;
    const predicted = predictedFollowPoint(t, p);
    p.lead = predicted;
    const los = segmentSafe({x: p.x, y: p.y}, predicted, cache) && pointSafe(predicted, cache);
    let desired = {x: 0, y: 0};
    let replanned = false;
    if (los) {
      // The look-ahead belongs to the route follower. Leaving the previous
      // tick's carrot in place while tracking by line of sight publishes a
      // stale, no-longer-certified point to the diagnostics and HUD.
      p.carrot = null;
      p.status = 'DIRECT LOS';
      const range = hypot(t.x - p.x, t.y - p.y);
      let speed = Motion.CONTRACT.pursuerSpeedMax;
      if (range < CONTRACT.standoffM + 0.8) {
        speed = Math.min(speed, hypot(t.vx, t.vy) + 0.35);
        p.status = 'FOLLOWING';
      }
      desired = approachVelocity({x: p.x, y: p.y}, predicted, p, speed);
    } else {
      replanned = maybeReplanPursuer(session, predicted, !p.route || !p.route.valid);
      if (p.route && p.route.valid) {
        p.status = 'ROUTE FOLLOW';
        advanceAlongRoute({x: p.x, y: p.y}, p.route, 0.45);
        let carrot = carrotPoint({x: p.x, y: p.y}, p.route, CONTRACT.lookAheadM, cache);
        if (!carrot) {
          // Certified corridor blocked from this pose: replan now rather than
          // hold a full-speed command against the safety envelope.
          replanned = maybeReplanPursuer(session, predicted, true) || replanned;
          carrot = carrotPoint({x: p.x, y: p.y}, p.route, CONTRACT.lookAheadM, cache);
        }
        p.carrot = carrot;
        if (carrot) {
          desired = approachVelocity(
            {x: p.x, y: p.y}, carrot, p, Motion.CONTRACT.pursuerSpeedMax
          );
        } else {
          p.status = 'NO SAFE ROUTE';
        }
      } else {
        p.status = 'NO SAFE ROUTE';
        p.carrot = null;
      }
    }
    if (!los && p.route && p.route.valid && session.time - p.lastPlanAt >= CONTRACT.replanPeriodS) {
      p.status = p.status === 'NO SAFE ROUTE' ? p.status : 'REPLANNING';
    }
    p.command = {x: desired.x, y: desired.y};
    const info = integrateBounded(p, desired, Motion.CONTRACT.pursuerSpeedMax, dt, cache);
    const stalled = hypot(p.vx, p.vy) < 0.05 && hypot(desired.x, desired.y) > 0.2;
    if (info.moved && !pointSafe({x: p.x, y: p.y}, cache)) session.stats.pursuer.collisions += 1;
    noteStats(
      session.stats.pursuer, p, info, stalled, replanned,
      p.status === 'NO SAFE ROUTE',
      info.headingDelta > 120 * Math.PI / 180
    );
    session.labels.pursuer = 'GT TRACK';
    session.labels.follow = p.status;
    return p;
  }

  function stepSession(session, dt) {
    session.time += dt;
    stepTargetFreeRoam(session, dt);
    stepPursuerGt(session, dt);
    return session;
  }

  function setClickGoal(session, point) {
    const kind = classifyClickGoal(point, session.caches.target);
    if (kind !== CONTRACT.clickFeedback.ok) {
      return {ok: false, reason: kind};
    }
    session.goalMode = 'click';
    session.clickGoal = copyPoint(point);
    maybeReplanTarget(session, true);
    return {ok: true, reason: kind, goal: copyPoint(point)};
  }

  function setAutoRoam(session) {
    session.goalMode = 'auto';
    session.clickGoal = null;
    maybeReplanTarget(session, true);
  }

  return {
    CONTRACT: CONTRACT,
    TARGET_SUPPORT: TARGET_SUPPORT,
    PURSUER_SUPPORT: PURSUER_SUPPORT,
    visualAttitude: visualAttitude,
    createCaches: createCaches,
    classifyClickGoal: classifyClickGoal,
    planAgentRoute: planAgentRoute,
    sampleReachableGoal: sampleReachableGoal,
    predictedFollowPoint: predictedFollowPoint,
    createSession: createSession,
    stepSession: stepSession,
    stepTargetFreeRoam: stepTargetFreeRoam,
    stepPursuerGt: stepPursuerGt,
    setClickGoal: setClickGoal,
    setAutoRoam: setAutoRoam,
    approachVelocity: approachVelocity,
    freeComponents: freeComponents,
    resolveConnectedSpawn: resolveConnectedSpawn,
    carrotPoint: carrotPoint,
    pointSafe: pointSafe,
    segmentSafe: segmentSafe,
  };
});
