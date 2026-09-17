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
    goalMoveReplanM: 1.4,
    deviationReplanM: 1.2,
    lookAheadM: 1.8,
    goalReachM: 0.9,
    minRoamGoalM: 6.0,
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

  function nearestSafe(point, cache, fallback) {
    if (pointSafe(point, cache)) return copyPoint(point);
    if (fallback && pointSafe(fallback, cache)) return copyPoint(fallback);
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

  function carrotPoint(pos, route, lookAheadM) {
    if (!route || !route.valid || !route.waypoints.length) return null;
    let remaining = lookAheadM;
    let cursor = Math.max(0, route.cursor || 0);
    let ax = pos.x, ay = pos.y;
    while (cursor < route.waypoints.length && remaining > 1e-6) {
      const wp = route.waypoints[cursor];
      const dx = wp.x - ax, dy = wp.y - ay;
      const seg = hypot(dx, dy);
      if (seg <= 1e-6) { cursor += 1; continue; }
      if (seg >= remaining) {
        return {
          x: ax + dx / seg * remaining,
          y: ay + dy / seg * remaining,
          cursor: cursor,
        };
      }
      remaining -= seg;
      ax = wp.x; ay = wp.y;
      cursor += 1;
    }
    const last = route.waypoints[route.waypoints.length - 1];
    return {x: last.x, y: last.y, cursor: route.waypoints.length - 1};
  }

  function advanceAlongRoute(pos, route, reachM) {
    if (!route || !route.valid) return;
    while (route.cursor < route.waypoints.length) {
      const wp = route.waypoints[route.cursor];
      if (hypot(wp.x - pos.x, wp.y - pos.y) > reachM) break;
      route.cursor += 1;
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

  function integrateBounded(agent, desired, speedLimit, dt, cache) {
    const previous = {x: agent.vx, y: agent.vy};
    const limited = Motion.limitPlanarVelocity(
      previous, desired, speedLimit, dt,
      Motion.CONTRACT.boundedMaxAccel, Motion.CONTRACT.boundedMaxTurnRate
    );
    const next = {x: agent.x + limited.x * dt, y: agent.y + limited.y * dt};
    const moved = hypot(next.x - agent.x, next.y - agent.y);
    if (!pointSafe(next, cache) || !segmentSafe({x: agent.x, y: agent.y}, next, cache)) {
      agent.vx = 0;
      agent.vy = 0;
      agent.accel = {x: -previous.x / Math.max(dt, 1e-6), y: -previous.y / Math.max(dt, 1e-6)};
      const attitude = visualAttitude({x: 0, y: 0}, agent.accel, agent.heading);
      agent.roll = attitude.roll;
      agent.pitch = attitude.pitch;
      return {moved: false, jump: 0, headingDelta: 0};
    }
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

  function createSession(options) {
    const bars = options.bars || [];
    const arenaLo = options.arenaLo;
    const arenaHi = options.arenaHi;
    const caches = createCaches(bars, arenaLo, arenaHi);
    const targetStart = snapToSafe(options.target, caches.target);
    const pursuerStart = snapToSafe(options.pursuer, caches.pursuer);
    const target = {
      x: targetStart.x, y: targetStart.y,
      vx: 0, vy: 0, heading: options.target.heading || 0,
      speed: options.speed || 1.5,
      roll: 0, pitch: 0, accel: {x: 0, y: 0},
      route: null, goal: null, status: 'IDLE',
    };
    const pursuer = {
      x: pursuerStart.x, y: pursuerStart.y,
      vx: 0, vy: 0, heading: options.pursuer.heading || 0,
      roll: 0, pitch: 0, accel: {x: 0, y: 0},
      route: null, lead: null, carrot: null,
      status: 'IDLE', replanCount: 0, lastPlanAt: -Infinity,
      plannedGoal: null,
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
    t.route = makeRouteFollow(result, {x: t.x, y: t.y});
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
    const replanned = maybeReplanTarget(session, !t.route || !t.route.valid);
    let desired = {x: 0, y: 0};
    if (t.route && t.route.valid) {
      advanceAlongRoute({x: t.x, y: t.y}, t.route, CONTRACT.goalReachM * 0.45);
      const carrot = carrotPoint({x: t.x, y: t.y}, t.route, CONTRACT.lookAheadM);
      if (carrot) {
        const dx = carrot.x - t.x, dy = carrot.y - t.y;
        const n = Math.max(hypot(dx, dy), 1e-6);
        desired = {x: dx / n * t.speed, y: dy / n * t.speed};
        t.status = 'ROAMING';
      }
    } else {
      t.status = 'NO SAFE ROUTE';
    }
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
    const goalMoved = p.plannedGoal
      ? hypot(predicted.x - p.plannedGoal.x, predicted.y - p.plannedGoal.y) >= CONTRACT.goalMoveReplanM
      : true;
    const invalid = p.route && !routeStillValid({x: p.x, y: p.y}, p.route, cache);
    const offPath = deviationFromRoute({x: p.x, y: p.y}, p.route) > CONTRACT.deviationReplanM;
    const exhausted = p.route && p.route.cursor >= (p.route.waypoints.length || 0);
    if (!(force || noRoute || ((due) && (goalMoved || invalid || offPath || exhausted)))) {
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
      p.status = 'DIRECT LOS';
      const dx = predicted.x - p.x, dy = predicted.y - p.y;
      const dist = hypot(dx, dy);
      const n = Math.max(dist, 1e-6);
      const range = hypot(t.x - p.x, t.y - p.y);
      let speed = Motion.CONTRACT.pursuerSpeedMax;
      if (range < CONTRACT.standoffM + 0.8) {
        speed = Math.min(speed, hypot(t.vx, t.vy) + 0.35);
        p.status = 'FOLLOWING';
      }
      desired = {x: dx / n * speed, y: dy / n * speed};
    } else {
      replanned = maybeReplanPursuer(session, predicted, !p.route || !p.route.valid);
      if (p.route && p.route.valid) {
        p.status = 'ROUTE FOLLOW';
        advanceAlongRoute({x: p.x, y: p.y}, p.route, 0.45);
        const carrot = carrotPoint({x: p.x, y: p.y}, p.route, CONTRACT.lookAheadM);
        p.carrot = carrot;
        if (carrot) {
          const dx = carrot.x - p.x, dy = carrot.y - p.y;
          const n = Math.max(hypot(dx, dy), 1e-6);
          desired = {
            x: dx / n * Motion.CONTRACT.pursuerSpeedMax,
            y: dy / n * Motion.CONTRACT.pursuerSpeedMax,
          };
        }
      } else {
        p.status = 'NO SAFE ROUTE';
        p.carrot = null;
      }
    }
    if (!los && p.route && p.route.valid && session.time - p.lastPlanAt >= CONTRACT.replanPeriodS) {
      p.status = p.status === 'NO SAFE ROUTE' ? p.status : 'REPLANNING';
    }
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
    pointSafe: pointSafe,
    segmentSafe: segmentSafe,
  };
});
