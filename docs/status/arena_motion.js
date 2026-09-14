/*
 * Browser-side reproduction of the NavRL general-training target distribution.
 *
 * Source of truth:
 *   aerial_gym/task/navrl_task/navrl_task.py
 *     _randomize_general_drone_spawn, _sample_general_target,
 *     _sample_target_motion, _sample_waypoints, _advance_target
 *   aerial_gym/rl_training/rl_games/train_navrl_general_repr_density.sh
 *
 * Target steering parity: both this browser model and target_motion.steer_target_step use
 * the same 90° heading-continuity preference with a large-turn escape fallback.
 *
 * Coordinates use the status arena convention. Defaults preserve task v1; configure()
 * replaces bounds, goal range and target-speed support from status.json before the
 * scene starts so task v2 is not merely drawn on a larger floor with v1 motion.
 * This module is deliberately independent of THREE/DOM so its parity contracts can
 * be tested with Node.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.NavRLArenaMotion = api;
})(typeof window !== 'undefined' ? window : this, function () {
  'use strict';

  const CONTRACT = {
    bounds: { x0: 0, x1: 24, y0: -12, y1: 12 },
    wallMargin: 0.5,
    spawnMargin: 1.0,
    spawnBarClearance: 0.65,
    targetBarClearance: 1.0,
    boundedObstacleClearance: 0.77,
    boundedMaxAccel: 4.0,
    boundedMaxTurnRate: 150 * Math.PI / 180,
    boundedLookaheadSeconds: 1.0,
    physicalStyleTimeConstant: 0.24,
    physicalStyleMaxTilt: 45 * Math.PI / 180,
    targetDistanceMin: 4.0,
    targetDistanceMax: 16.0,
    waypointReach: 0.5,
    targetSpeedMin: 0.0,
    targetSpeedMax: 1.5,
    pursuerSpeedMax: 2.5,
    pursuerRadius: 0.25,
    pursuerLookaheadSeconds: 0.9,
  };
  const TURN_ANGLES = Object.freeze(
    [0, 30, -30, 60, -60, 90, -90, 120, -120, 180].map(
      function (degrees) { return degrees * Math.PI / 180; }
    )
  );
  const BOUNDED_TURN_ANGLES = Object.freeze(
    [0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90,
      105, -105, 120, -120, 135, -135, 150, -150, 165, -165, 180].map(
      function (degrees) { return degrees * Math.PI / 180; }
    )
  );

  function seededRng(seed) {
    return function () {
      seed |= 0;
      seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function angleDelta(to, from) {
    return Math.atan2(Math.sin(to - from), Math.cos(to - from));
  }

  /* Fixed 10 Hz simulation clock. Rendering may run at any refresh rate and receives alpha for
   * interpolation between the two latest simulation states. Long background-tab gaps are dropped
   * deliberately instead of creating an unbounded catch-up burst. */
  function createFixedStepClock(stepSeconds, maxFrameSeconds, maxStepsPerFrame) {
    const step = stepSeconds == null ? 0.1 : Number(stepSeconds);
    const maxFrame = maxFrameSeconds == null ? 0.25 : Number(maxFrameSeconds);
    const maxSteps = maxStepsPerFrame == null ? 8 : Math.max(1, Math.floor(maxStepsPerFrame));
    if (!(step > 0) || !(maxFrame >= step) || !Number.isFinite(step + maxFrame)) {
      throw new Error('invalid fixed-step clock contract');
    }
    let last = null, accumulator = 0, simulationTime = 0;
    return {
      stepSeconds: step,
      reset(nowSeconds) { last = nowSeconds == null ? null : Number(nowSeconds); accumulator = 0; },
      advance(nowSeconds, running, stepFn) {
        const now = Number(nowSeconds);
        if (!Number.isFinite(now)) throw new Error('clock time must be finite');
        if (last == null) { last = now; return { steps: 0, alpha: 0, simulationTime }; }
        let elapsed = clamp(now - last, 0, maxFrame); last = now;
        // A paused renderer shows the latest committed state, not the previous interpolation
        // endpoint. The arena also collapses prev/current on pause so resume cannot jump back.
        if (!running) { accumulator = 0; return { steps: 0, alpha: 1, simulationTime }; }
        accumulator += elapsed;
        let steps = 0;
        while (accumulator + 1e-12 >= step && steps < maxSteps) {
          stepFn(step);
          accumulator -= step;
          simulationTime += step;
          steps++;
        }
        if (steps === maxSteps && accumulator >= step) accumulator %= step;
        return { steps, alpha: clamp(accumulator / step, 0, 1), simulationTime };
      },
    };
  }

  function configure(cfg) {
    if (!cfg) return CONTRACT;
    if (cfg.arena_xy_m != null) {
      const size = Number(cfg.arena_xy_m);
      if (Number.isFinite(size) && size > 2 * CONTRACT.spawnMargin) {
        CONTRACT.bounds.x0 = 0;
        CONTRACT.bounds.x1 = size;
        CONTRACT.bounds.y0 = -size / 2;
        CONTRACT.bounds.y1 = size / 2;
      }
    }
    if (Array.isArray(cfg.goal_dist_m) && cfg.goal_dist_m.length >= 2) {
      const lo = Number(cfg.goal_dist_m[0]), hi = Number(cfg.goal_dist_m[1]);
      if (Number.isFinite(lo) && Number.isFinite(hi) && 0 <= lo && lo < hi) {
        CONTRACT.targetDistanceMin = lo;
        CONTRACT.targetDistanceMax = hi;
      }
    }
    if (Array.isArray(cfg.target_speed_m) && cfg.target_speed_m.length >= 2) {
      const lo = Number(cfg.target_speed_m[0]), hi = Number(cfg.target_speed_m[1]);
      if (Number.isFinite(lo) && Number.isFinite(hi) && 0 <= lo && lo <= hi) {
        CONTRACT.targetSpeedMin = lo;
        CONTRACT.targetSpeedMax = hi;
      }
    }
    return CONTRACT;
  }

  function barHalfExtent(bar) {
    return 0.5 * (bar && bar.w != null ? bar.w : 0.6);
  }

  function distanceToBars(x, y, bars) {
    let best = Infinity;
    for (const bar of bars || []) best = Math.min(best, Math.hypot(x - bar.x, y - bar.y));
    return best;
  }

  /*
   * Bars are rendered as axis-aligned boxes of side w. A circle of radius w/2 does
   * NOT cover the box corners (half-diagonal is w/2 * √2), which made the browser
   * pursuer clip through pillars even while "clearance" stayed non-negative.
   * Signed distance from a point to the box, minus the pursuer radius.
   */
  function pursuerClearance(x, y, bars) {
    let best = Infinity;
    const r = CONTRACT.pursuerRadius;
    for (const bar of bars || []) {
      const half = barHalfExtent(bar);
      const ox = Math.abs(x - bar.x) - half;
      const oy = Math.abs(y - bar.y) - half;
      let surface;
      if (ox > 0 && oy > 0) surface = Math.hypot(ox, oy);
      else if (ox > 0) surface = ox;
      else if (oy > 0) surface = oy;
      else surface = -Math.min(-ox, -oy); // inside AABB: negative depth to nearest face
      best = Math.min(best, surface - r);
    }
    return best;
  }

  function segmentPursuerClearance(x0, y0, x1, y1, bars) {
    const length = Math.hypot(x1 - x0, y1 - y0);
    const steps = Math.max(2, Math.ceil(length / 0.08));
    let best = Infinity;
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      best = Math.min(
        best,
        pursuerClearance(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, bars)
      );
    }
    return best;
  }

  function pushPursuerOut(point, bars) {
    const b = CONTRACT.bounds;
    let x = point.x;
    let y = point.y;
    let pushed = false;
    const r = CONTRACT.pursuerRadius;
    for (let iteration = 0; iteration < 8; iteration++) {
      x = clamp(x, b.x0 + CONTRACT.wallMargin, b.x1 - CONTRACT.wallMargin);
      y = clamp(y, b.y0 + CONTRACT.wallMargin, b.y1 - CONTRACT.wallMargin);
      let worst = null;
      for (const bar of bars || []) {
        const half = barHalfExtent(bar);
        const cx = clamp(x, bar.x - half, bar.x + half);
        const cy = clamp(y, bar.y - half, bar.y + half);
        let nx = x - cx, ny = y - cy;
        let d = Math.hypot(nx, ny);
        let pen;
        if (d < 1e-8) {
          // Centered inside the box: escape through the nearest face.
          const left = x - (bar.x - half);
          const right = (bar.x + half) - x;
          const bottom = y - (bar.y - half);
          const top = (bar.y + half) - y;
          const m = Math.min(left, right, bottom, top);
          if (m === left) { nx = -1; ny = 0; pen = left + r; }
          else if (m === right) { nx = 1; ny = 0; pen = right + r; }
          else if (m === bottom) { nx = 0; ny = -1; pen = bottom + r; }
          else { nx = 0; ny = 1; pen = top + r; }
        } else {
          pen = r - d;
          nx /= d;
          ny /= d;
        }
        if (pen > 1e-6 && (!worst || pen > worst.pen)) {
          worst = { nx: nx, ny: ny, pen: pen };
        }
      }
      if (!worst) break;
      pushed = true;
      x += worst.nx * (worst.pen + 1e-3);
      y += worst.ny * (worst.pen + 1e-3);
    }
    return {
      x: clamp(x, b.x0 + CONTRACT.wallMargin, b.x1 - CONTRACT.wallMargin),
      y: clamp(y, b.y0 + CONTRACT.wallMargin, b.y1 - CONTRACT.wallMargin),
      pushed: pushed,
      clear: pursuerClearance(x, y, bars) >= -1e-6,
    };
  }

  function samplePoint(rng, margin) {
    const b = CONTRACT.bounds;
    return {
      x: b.x0 + margin + rng() * (b.x1 - b.x0 - 2 * margin),
      y: b.y0 + margin + rng() * (b.y1 - b.y0 - 2 * margin),
    };
  }

  function sampleClearPoint(rng, bars, clearance, margin, accept) {
    let candidate = samplePoint(rng, margin);
    for (let i = 0; i < 96; i++) {
      const next = samplePoint(rng, margin);
      if (distanceToBars(next.x, next.y, bars) >= clearance && (!accept || accept(next))) {
        return next;
      }
      candidate = next;
    }
    return candidate;
  }

  function sampleWaypoint(rng) {
    return samplePoint(rng, CONTRACT.wallMargin);
  }

  function createEpisode(rng, bars, speedCeiling) {
    const drone = sampleClearPoint(
      rng,
      bars,
      CONTRACT.spawnBarClearance,
      CONTRACT.spawnMargin,
      function (p) {
        // Also keep the pursuer disk outside the rendered box (not just the center-distance rule).
        return pursuerClearance(p.x, p.y, bars) >= 0.05;
      }
    );
    const maxDistance = CONTRACT.targetDistanceMax;
    const target = sampleClearPoint(
      rng,
      bars,
      CONTRACT.targetBarClearance,
      CONTRACT.spawnMargin,
      function (p) {
        const d = Math.hypot(p.x - drone.x, p.y - drone.y);
        return d >= CONTRACT.targetDistanceMin && d <= maxDistance;
      }
    );
    const mode = rng() < 0.5 ? 'cv' : 'waypoint';
    const ceiling = Math.max(
      0,
      speedCeiling == null ? CONTRACT.targetSpeedMax : Number(speedCeiling)
    );
    const floor = ceiling > 0 ? Math.min(ceiling, CONTRACT.targetSpeedMin) : 0;
    const speed = floor + rng() * (ceiling - floor);
    const angle = rng() * Math.PI * 2;
    return {
      drone: drone,
      target: target,
      mode: mode,
      speed: speed,
      cvVelocity: { x: speed * Math.cos(angle), y: speed * Math.sin(angle) },
      heading: angle, // last FLOWN heading, persisted for ALL patterns (see steerTargetStep)
      waypoint: sampleWaypoint(rng),
      avoidSign: rng() < 0.5 ? -1 : 1,
      realizedVelocity: { x: 0, y: 0 },
      boundedVelocity: { x: 0, y: 0 },
      physicalStyle: { velocity: { x: 0, y: 0 }, roll: 0, pitch: 0 },
      plannerFeasible: true,
      route: null,
      age: 0,
    };
  }

  function pushOutOfBars(point, bars, clearance) {
    const b = CONTRACT.bounds;
    let x = point.x;
    let y = point.y;
    let pushed = false;
    let normalX = 0, normalY = 0;
    for (let iteration = 0; iteration < 6; iteration++) {
      x = clamp(x, b.x0 + CONTRACT.wallMargin, b.x1 - CONTRACT.wallMargin);
      y = clamp(y, b.y0 + CONTRACT.wallMargin, b.y1 - CONTRACT.wallMargin);
      let sumX = 0, sumY = 0, maxPenetration = 0, nearest = null, nearestD = Infinity;
      for (const bar of bars || []) {
        const dx = x - bar.x, dy = y - bar.y;
        const d = Math.hypot(dx, dy);
        if (d < nearestD) { nearestD = d; nearest = { dx: dx, dy: dy }; }
        if (d < clearance) {
          const denom = Math.max(d, 1e-6);
          sumX += dx / denom;
          sumY += dy / denom;
          maxPenetration = Math.max(maxPenetration, clearance - d);
        }
      }
      if (maxPenetration <= 0) break;
      pushed = true;
      let norm = Math.hypot(sumX, sumY);
      if (norm <= 1e-6 && nearest) {
        sumX = nearest.dx;
        sumY = nearest.dy;
        norm = Math.max(Math.hypot(sumX, sumY), 1e-6);
      }
      normalX = sumX / norm;
      normalY = sumY / norm;
      x += normalX * (maxPenetration + 1e-3);
      y += normalY * (maxPenetration + 1e-3);
    }
    return {
      x: x, y: y, pushed: pushed, normalX: normalX, normalY: normalY
    };
  }

  /*
   * Heading-continuity window for the target. Without it the argmax below is memoryless:
   * in waypoint mode the desired bearing is re-derived from the same static waypoint every
   * frame, so a blocking bar makes the choice flip between "aim at waypoint" and a ±120-180°
   * escape turn — a period-2 oscillation that renders as frantic spinning (measured 55°/step
   * mean heading change at 85 bars) or, at high density, as standing still at full commanded
   * speed. The window is a PREFERENCE among clear candidates, never a veto: the
   * inWindow || anyClear || trapped fallback chain must stay intact so a large escape turn
   * (and cv wall reflections) are still taken when nothing inside the window is clear.
   */
  const CONTINUITY_WINDOW = 90 * Math.PI / 180;

  function steerTargetStep(oldX, oldY, desiredVX, desiredVY, speed, dt, bars, avoidSign, prevHeading) {
    const b = CONTRACT.bounds;
    const loX = b.x0 + CONTRACT.wallMargin, hiX = b.x1 - CONTRACT.wallMargin;
    const loY = b.y0 + CONTRACT.wallMargin, hiY = b.y1 - CONTRACT.wallMargin;
    const norm = Math.max(Math.hypot(desiredVX, desiredVY), 1e-6);
    const bx = desiredVX / norm, by = desiredVY / norm;
    let inWindow = null, anyClear = null, trapped = null;
    TURN_ANGLES.forEach(function (angle, index) {
      const c = Math.cos(angle), s = Math.sin(angle);
      const dx = bx * c - by * s, dy = bx * s + by * c;
      const x = oldX + dx * speed * dt, y = oldY + dy * speed * dt;
      const inside = x >= loX && x <= hiX && y >= loY && y <= hiY;
      const minBar = distanceToBars(x, y, bars);
      const clear = inside && minBar >= CONTRACT.targetBarClearance + 1e-4;
      const tie = (avoidSign || 1) * Math.sign(angle) * 1e-3;
      const heading = Math.atan2(dy, dx);
      const candidate = {
        x: x, y: y, vx: dx * speed, vy: dy * speed,
        clear: clear, steered: index !== 0, heading: heading,
        score: clear
          ? 1000 - Math.abs(angle) + tie
          : (inside ? 100 : 0) + Math.min(minBar, 10) - 0.01 * Math.abs(angle) + tie,
      };
      if (!clear) {
        if (!trapped || candidate.score > trapped.score) trapped = candidate;
        return;
      }
      if (!anyClear || candidate.score > anyClear.score) anyClear = candidate;
      if (prevHeading == null) return;
      const turn = Math.abs(Math.atan2(
        Math.sin(heading - prevHeading), Math.cos(heading - prevHeading)
      ));
      if (turn <= CONTINUITY_WINDOW + 1e-9
          && (!inWindow || candidate.score > inWindow.score)) inWindow = candidate;
    });
    // Clear ALWAYS beats blocked (min clear score ≈ 996.86 > max trapped score ≈ 110.001);
    // with prevHeading == null, inWindow stays null and the result is exactly the legacy
    // single-pass argmax, so the 9th argument is strictly opt-in.
    return inWindow || anyClear || trapped;
  }

  /*
   * Collision-aware illustrative pursuer for the browser scene.
   *
   * This is intentionally not presented as the learned PyTorch policy.  The old
   * attractive-plus-repulsive potential field had local minima in dense layouts:
   * symmetric bar forces could exactly cancel the target direction.  Here we
   * score persistent left/right heading candidates over a short swept path.
   */
  function steerPursuerStep(oldX, oldY, targetX, targetY, speed, dt, bars, oldHeading, avoidSign) {
    const b = CONTRACT.bounds;
    const loX = b.x0 + CONTRACT.wallMargin, hiX = b.x1 - CONTRACT.wallMargin;
    const loY = b.y0 + CONTRACT.wallMargin, hiY = b.y1 - CONTRACT.wallMargin;
    const tx = targetX - oldX, ty = targetY - oldY;
    const targetDistance = Math.hypot(tx, ty);
    if (dt <= 0 || speed <= 1e-6 || targetDistance <= 1e-6) {
      return { x: oldX, y: oldY, vx: 0, vy: 0, heading: oldHeading || 0, blocked: false, hit: false };
    }

    // Already inside a pillar: freeze and signal a crash so the demo can reset.
    if (pursuerClearance(oldX, oldY, bars) < -1e-4) {
      return {
        x: oldX, y: oldY, vx: 0, vy: 0,
        heading: oldHeading == null ? Math.atan2(ty, tx) : oldHeading,
        blocked: true, hit: true,
      };
    }

    const directHeading = Math.atan2(ty, tx);
    const offsets = [0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90, 120, -120, 150, -150, 180];
    const stepDistance = Math.min(speed * dt, targetDistance);
    const lookDistance = Math.min(
      speed * CONTRACT.pursuerLookaheadSeconds,
      Math.max(stepDistance, targetDistance)
    );
    const samples = Math.max(4, Math.ceil(lookDistance / 0.1));
    let best = null;

    offsets.forEach(function (degrees) {
      const angle = directHeading + degrees * Math.PI / 180;
      const ux = Math.cos(angle), uy = Math.sin(angle);
      let minClearance = Infinity;
      let immediateClear = true;
      let pathClear = true;
      for (let i = 1; i <= samples; i++) {
        const distance = lookDistance * i / samples;
        const x = oldX + ux * distance, y = oldY + uy * distance;
        const inside = x >= loX && x <= hiX && y >= loY && y <= hiY;
        const clearance = pursuerClearance(x, y, bars);
        minClearance = Math.min(minClearance, clearance);
        if (!inside || clearance < 0) {
          pathClear = false;
          if (distance <= stepDistance + 1e-6) immediateClear = false;
        }
      }
      const stepX = oldX + ux * stepDistance, stepY = oldY + uy * stepDistance;
      if (segmentPursuerClearance(oldX, oldY, stepX, stepY, bars) < 0) immediateClear = false;
      if (!immediateClear) return;

      const endX = oldX + ux * lookDistance, endY = oldY + uy * lookDistance;
      const progress = targetDistance - Math.hypot(targetX - endX, targetY - endY);
      const turn = Math.abs(Math.atan2(
        Math.sin(angle - (oldHeading == null ? directHeading : oldHeading)),
        Math.cos(angle - (oldHeading == null ? directHeading : oldHeading))
      ));
      const sideBias = (avoidSign || 1) * Math.sign(degrees) * 0.03;
      const score = (pathClear ? 1000 : 0)
        + progress * 8
        + Math.min(minClearance, 2) * 0.8
        - turn * 0.35
        - Math.abs(degrees) * 0.002
        + sideBias;
      if (!best || score > best.score) {
        best = { angle: angle, ux: ux, uy: uy, pathClear: pathClear, score: score };
      }
    });

    if (!best) {
      return {
        x: oldX, y: oldY, vx: 0, vy: 0,
        heading: oldHeading == null ? directHeading : oldHeading,
        blocked: true, hit: false,
      };
    }

    const rawX = clamp(oldX + best.ux * stepDistance, loX, hiX);
    const rawY = clamp(oldY + best.uy * stepDistance, loY, hiY);
    const cleared = pushPursuerOut({ x: rawX, y: rawY }, bars);
    const moved = Math.hypot(cleared.x - oldX, cleared.y - oldY);
    // A proposed step that collapses to near-zero after push-out is a soft collision.
    const hit = !cleared.clear || (moved < stepDistance * 0.15 && stepDistance > 1e-3);
    if (hit) {
      return {
        x: oldX, y: oldY, vx: 0, vy: 0,
        heading: best.angle, blocked: true, hit: true,
      };
    }
    return {
      x: cleared.x,
      y: cleared.y,
      vx: best.ux * speed,
      vy: best.uy * speed,
      heading: best.angle,
      blocked: !best.pathClear,
      hit: false,
    };
  }

  function limitPlanarVelocity(current, desired, speedLimit, dt, maxAccel, maxTurnRate) {
    const desiredNorm = Math.hypot(desired.x, desired.y);
    const desiredSpeed = Math.min(Math.max(0, speedLimit), desiredNorm);
    const desiredHeading = desiredNorm > 1e-9 ? Math.atan2(desired.y, desired.x) : 0;
    const currentSpeed = Math.hypot(current.x, current.y);
    const currentHeading = currentSpeed > 1e-5 ? Math.atan2(current.y, current.x) : desiredHeading;
    const turn = clamp(angleDelta(desiredHeading, currentHeading), -maxTurnRate * dt, maxTurnRate * dt);
    const limitedHeading = currentSpeed > 1e-5 ? currentHeading + turn : desiredHeading;
    const target = {
      x: Math.cos(limitedHeading) * desiredSpeed,
      y: Math.sin(limitedHeading) * desiredSpeed,
    };
    let dvx = target.x - current.x, dvy = target.y - current.y;
    const delta = Math.hypot(dvx, dvy), maxDelta = Math.max(0, maxAccel) * dt;
    if (delta > maxDelta && delta > 1e-9) { dvx *= maxDelta / delta; dvy *= maxDelta / delta; }
    let x = current.x + dvx, y = current.y + dvy;
    const speed = Math.hypot(x, y);
    if (speed > speedLimit && speed > 1e-9) { x *= speedLimit / speed; y *= speedLimit / speed; }
    return { x, y };
  }

  // Python bounded_drone_target_step uses centre-to-centre clearance for the non-physical lineage.
  // 0.77 m already conservatively contains max bar half-diagonal + target half-width + margin.
  function boundedTargetStep(old, currentVelocity, desiredVelocity, speedLimit, dt, bars, turnSign) {
    const b = CONTRACT.bounds;
    const loX = b.x0 + CONTRACT.wallMargin, hiX = b.x1 - CONTRACT.wallMargin;
    const loY = b.y0 + CONTRACT.wallMargin, hiY = b.y1 - CONTRACT.wallMargin;
    const desiredNorm = Math.hypot(desiredVelocity.x, desiredVelocity.y);
    const baseHeading = desiredNorm > 1e-6 ? Math.atan2(desiredVelocity.y, desiredVelocity.x) : 0;
    const scales = [1, 0.5, 0.25];
    const candidates = [];
    scales.forEach(function (scale) {
      BOUNDED_TURN_ANGLES.forEach(function (angle) {
        candidates.push({
          desired: {
            x: Math.cos(baseHeading + angle) * speedLimit * scale,
            y: Math.sin(baseHeading + angle) * speedLimit * scale,
          },
          angle, scale,
        });
      });
    });
    candidates.push({ desired: { x: 0, y: 0 }, angle: 0, scale: 0 });
    const rolloutSteps = Math.max(1, Math.ceil(CONTRACT.boundedLookaheadSeconds / dt));
    let best = null;
    candidates.forEach(function (candidate, index) {
      let pos = { x: old.x, y: old.y };
      let velocity = { x: currentVelocity.x, y: currentVelocity.y };
      let firstPos, firstVelocity, feasible = true, immediateFeasible = false;
      let safePrefix = 0, prefixAlive = true, minClearance = Infinity;
      for (let step = 0; step < rolloutSteps; step++) {
        velocity = limitPlanarVelocity(
          velocity, candidate.desired, speedLimit, dt,
          CONTRACT.boundedMaxAccel, CONTRACT.boundedMaxTurnRate
        );
        pos = { x: pos.x + velocity.x * dt, y: pos.y + velocity.y * dt };
        if (step === 0) { firstPos = { x: pos.x, y: pos.y }; firstVelocity = { x: velocity.x, y: velocity.y }; }
        const inside = pos.x >= loX && pos.x <= hiX && pos.y >= loY && pos.y <= hiY;
        const barDistance = distanceToBars(pos.x, pos.y, bars);
        minClearance = Math.min(minClearance, barDistance);
        const safe = inside && barDistance >= CONTRACT.boundedObstacleClearance + 1e-4;
        if (step === 0) immediateFeasible = safe;
        prefixAlive = prefixAlive && safe;
        if (prefixAlive) safePrefix += 1;
        feasible = feasible && safe;
      }
      const tie = (turnSign || 1) * Math.sign(candidate.angle) * 1e-3;
      const boundaryMargin = Math.min(pos.x - loX, hiX - pos.x, pos.y - loY, hiY - pos.y);
      const score = feasible
        ? 1000 + 10 * candidate.scale - Math.abs(candidate.angle) + tie
        : 100 * safePrefix + Math.min(minClearance, 10) + Math.min(boundaryMargin, 10)
          - 0.01 * Math.abs(candidate.angle) + tie;
      if (!best || score > best.score) {
        best = { index, score, firstPos, firstVelocity, feasible: immediateFeasible };
      }
    });
    return best;
  }

  function desiredTargetVelocity(episode) {
    if (episode.mode === 'waypoint') {
      const dx = episode.waypoint.x - episode.target.x;
      const dy = episode.waypoint.y - episode.target.y;
      const norm = Math.max(Math.hypot(dx, dy), 1e-6);
      return { x: dx / norm * episode.speed, y: dy / norm * episode.speed };
    }
    return { x: episode.cvVelocity.x, y: episode.cvVelocity.y };
  }

  function advanceBoundedTarget(episode, dt, bars, rng, physicalStyle) {
    episode.age += dt;
    if (episode.speed <= 1e-6) return episode;
    const old = { x: episode.target.x, y: episode.target.y };
    const current = physicalStyle ? episode.physicalStyle.velocity : episode.boundedVelocity;
    const planned = boundedTargetStep(
      old, current, desiredTargetVelocity(episode), episode.speed, dt, bars, episode.avoidSign
    );
    episode.plannerFeasible = planned.feasible;
    let velocity = planned.firstVelocity;
    if (physicalStyle) {
      // Illustrative only: this is not PhysX. A first-order velocity response stands in for
      // rigid-body/controller lag and its acceleration implies a bounded display attitude.
      const alpha = 1 - Math.exp(-dt / CONTRACT.physicalStyleTimeConstant);
      const desired = velocity;
      let next = {
        x: current.x + (desired.x - current.x) * alpha,
        y: current.y + (desired.y - current.y) * alpha,
      };
      const dvx = next.x - current.x, dvy = next.y - current.y;
      const dv = Math.hypot(dvx, dvy), maxDv = CONTRACT.boundedMaxAccel * dt;
      if (dv > maxDv && dv > 1e-9) {
        next = { x: current.x + dvx * maxDv / dv, y: current.y + dvy * maxDv / dv };
      }
      velocity = next;
      const ax = (velocity.x - current.x) / dt, ay = (velocity.y - current.y) / dt;
      // Convert world-planar acceleration into the displayed target body frame. Using world ay
      // as roll and world ax as pitch only happened to look right at yaw=0; after a 90° turn it
      // swapped forward pitch and lateral roll.
      const motionHeading = Math.hypot(velocity.x, velocity.y) > 1e-6
        ? Math.atan2(velocity.y, velocity.x) : episode.heading;
      const ch = Math.cos(motionHeading), sh = Math.sin(motionHeading);
      const forwardAccel = ax * ch + ay * sh;
      const lateralAccel = -ax * sh + ay * ch;
      episode.physicalStyle.roll = clamp(-lateralAccel / 9.81, -Math.tan(CONTRACT.physicalStyleMaxTilt), Math.tan(CONTRACT.physicalStyleMaxTilt));
      episode.physicalStyle.pitch = clamp(forwardAccel / 9.81, -Math.tan(CONTRACT.physicalStyleMaxTilt), Math.tan(CONTRACT.physicalStyleMaxTilt));
      episode.physicalStyle.velocity = { x: velocity.x, y: velocity.y };
      episode.target.x = old.x + velocity.x * dt;
      episode.target.y = old.y + velocity.y * dt;
    } else {
      episode.target.x = planned.firstPos.x;
      episode.target.y = planned.firstPos.y;
      episode.boundedVelocity = { x: velocity.x, y: velocity.y };
    }
    episode.realizedVelocity = {
      x: (episode.target.x - old.x) / dt,
      y: (episode.target.y - old.y) / dt,
    };
    episode.heading = Math.atan2(velocity.y, velocity.x);
    if (episode.mode === 'cv') episode.cvVelocity = { x: velocity.x, y: velocity.y };
    // Python bounded mode checks the virtual point AFTER committing firstPos. Physical mode can
    // only check the actor's pre-physics position here because its rigid body moves afterwards.
    const waypointPosition = physicalStyle ? old : episode.target;
    if (episode.mode === 'waypoint'
        && Math.hypot(
          episode.waypoint.x - waypointPosition.x,
          episode.waypoint.y - waypointPosition.y
        ) < CONTRACT.waypointReach) {
      episode.waypoint = sampleWaypoint(rng);
    }
    return episode;
  }

  function routeVelocityReference(episode, allowReplan) {
    const route = episode.route;
    if (!route || !route.valid || !Array.isArray(route.waypoints) || !route.waypoints.length) {
      return {x: 0, y: 0, active: false, complete: false};
    }
    route.cursor = Math.max(0, Math.min(route.waypoints.length - 1, route.cursor || 0));
    route.segmentStart = route.segmentStart || {x: episode.target.x, y: episode.target.y};
    // Mirrors BatchedTargetRouteManager.velocity_reference: advance already-reached or overshot
    // smoothed waypoints repeatedly, capped at four transitions per 10 Hz control interval. A
    // transition additionally needs the planning-time exact-AABB handoff certificate: being
    // merely inside the task's 0.5 m reach radius is not safe at a tight corner.
    for (let iteration = 0; iteration < 4; iteration++) {
      const waypoint = route.waypoints[route.cursor];
      const sx = waypoint.x - route.segmentStart.x, sy = waypoint.y - route.segmentStart.y;
      const ox = episode.target.x - waypoint.x, oy = episode.target.y - waypoint.y;
      const segmentNorm = Math.max(Math.hypot(sx, sy), 1e-6);
      const crossTrack = Math.abs(ox * sy - oy * sx) / segmentNorm;
      const passed = ox * sx + oy * sy >= 0 && crossTrack <= route.waypointReachM;
      const waypointDistance = Math.hypot(waypoint.x - episode.target.x,
        waypoint.y - episode.target.y);
      const certificate = Array.isArray(route.handoffClearanceM)
        ? Number(route.handoffClearanceM[route.cursor]) : NaN;
      const connectorCertified = Number.isFinite(certificate)
        && certificate >= 0 && waypointDistance <= certificate;
      const reached = (waypointDistance <= route.waypointReachM || passed)
        && connectorCertified;
      if (!(reached && route.cursor + 1 < route.waypoints.length)) break;
      route.segmentStart = {x: waypoint.x, y: waypoint.y};
      route.cursor += 1;
    }
    const waypoint = route.waypoints[route.cursor];
    const sx = waypoint.x - route.segmentStart.x, sy = waypoint.y - route.segmentStart.y;
    const dx = waypoint.x - episode.target.x, dy = waypoint.y - episode.target.y;
    const passedFinal = (episode.target.x - waypoint.x) * sx
      + (episode.target.y - waypoint.y) * sy >= 0;
    const complete = route.cursor + 1 >= route.waypoints.length
      && (Math.hypot(dx, dy) <= route.waypointReachM || passedFinal);
    route.complete = complete;
    if (complete && allowReplan !== false && typeof route.replan === 'function') {
      const previousGoal = route.goal || route.waypoints[route.waypoints.length - 1];
      const replacement = route.replan(
        {x: episode.target.x, y: episode.target.y},
        {x: previousGoal.x, y: previousGoal.y}
      );
      if (replacement) {
        episode.route = replacement;
        episode.routeGoalReplacements = (episode.routeGoalReplacements || 0) + 1;
        return routeVelocityReference(episode, false);
      }
    }
    const norm = Math.max(Math.hypot(dx, dy), 1e-6);
    return complete
      ? {x: 0, y: 0, active: false, complete: true}
      : {x: dx / norm * episode.speed, y: dy / norm * episode.speed,
        active: true, complete: false};
  }

  function advanceRoutedPreview(episode, dt) {
    episode.age += dt;
    if (!episode.route || !episode.route.valid) {
      // Explicit fail-closed preview contract: no route means no target motion. Do not silently
      // fall back to a random waypoint or least-bad local command.
      episode.plannerFeasible = false;
      episode.boundedVelocity = {x: 0, y: 0};
      episode.physicalStyle.velocity = {x: 0, y: 0};
      episode.physicalStyle.roll = 0; episode.physicalStyle.pitch = 0;
      episode.realizedVelocity = {x: 0, y: 0};
      return episode;
    }
    const old = {x: episode.target.x, y: episode.target.y};
    const current = episode.physicalStyle.velocity;
    const reference = routeVelocityReference(episode);
    if (!episode.route || !episode.route.valid) {
      episode.plannerFeasible = false;
      episode.boundedVelocity = {x: 0, y: 0};
      episode.physicalStyle.velocity = {x: 0, y: 0};
      episode.physicalStyle.roll = 0; episode.physicalStyle.pitch = 0;
      episode.realizedVelocity = {x: 0, y: 0};
      return episode;
    }
    episode.plannerFeasible = true;
    const bounded = limitPlanarVelocity(
      current, reference, episode.speed, dt,
      CONTRACT.boundedMaxAccel, CONTRACT.boundedMaxTurnRate
    );
    // Browser-only rigid-body-like lag. The global route and bounded command mirror code
    // contracts; this response is explicitly NOT a PhysX or PPO rollout.
    const alpha = 1 - Math.exp(-dt / CONTRACT.physicalStyleTimeConstant);
    let velocity = {
      x: current.x + (bounded.x - current.x) * alpha,
      y: current.y + (bounded.y - current.y) * alpha,
    };
    const dvx = velocity.x - current.x, dvy = velocity.y - current.y;
    const delta = Math.hypot(dvx, dvy), maxDelta = CONTRACT.boundedMaxAccel * dt;
    if (delta > maxDelta && delta > 1e-9) {
      velocity = {x: current.x + dvx * maxDelta / delta,
        y: current.y + dvy * maxDelta / delta};
    }
    episode.target.x = old.x + velocity.x * dt;
    episode.target.y = old.y + velocity.y * dt;
    episode.realizedVelocity = {x: velocity.x, y: velocity.y};
    episode.boundedVelocity = {x: bounded.x, y: bounded.y};
    episode.physicalStyle.velocity = {x: velocity.x, y: velocity.y};
    const ax = (velocity.x - current.x) / dt, ay = (velocity.y - current.y) / dt;
    const motionHeading = Math.hypot(velocity.x, velocity.y) > 1e-6
      ? Math.atan2(velocity.y, velocity.x) : episode.heading;
    const ch = Math.cos(motionHeading), sh = Math.sin(motionHeading);
    const forwardAccel = ax * ch + ay * sh;
    const lateralAccel = -ax * sh + ay * ch;
    episode.physicalStyle.roll = clamp(
      -lateralAccel / 9.81,
      -Math.tan(CONTRACT.physicalStyleMaxTilt), Math.tan(CONTRACT.physicalStyleMaxTilt)
    );
    episode.physicalStyle.pitch = clamp(
      forwardAccel / 9.81,
      -Math.tan(CONTRACT.physicalStyleMaxTilt), Math.tan(CONTRACT.physicalStyleMaxTilt)
    );
    if (Math.hypot(velocity.x, velocity.y) > 1e-6) episode.heading = motionHeading;
    return episode;
  }

  // Match NavRLTask's target-relative swept-segment capture test. Endpoint-only range misses a
  // grazing crossing whose two endpoints both remain just outside the 0.5 m capture sphere.
  function sweptCapture(prevRelative, nextRelative, radius) {
    const sx = nextRelative.x - prevRelative.x;
    const sy = nextRelative.y - prevRelative.y;
    const denom = sx * sx + sy * sy;
    const t = denom > 1e-12
      ? clamp(-(prevRelative.x * sx + prevRelative.y * sy) / denom, 0, 1)
      : 0;
    return Math.hypot(prevRelative.x + t * sx, prevRelative.y + t * sy) < radius;
  }

  function advanceTarget(episode, dt, bars, rng, mode) {
    if (!episode || dt <= 0) return episode;
    const dynamics = mode || 'bounded';
    if (dynamics === 'bounded') return advanceBoundedTarget(episode, dt, bars, rng, false);
    if (dynamics === 'physical-style') return advanceBoundedTarget(episode, dt, bars, rng, true);
    if (dynamics === 'routed-preview') return advanceRoutedPreview(episode, dt);
    if (dynamics !== 'legacy') throw new Error('unknown illustrative target mode: ' + dynamics);
    // episode.age drives arena.js's 30 s watchdog: it must measure WALL CLOCK, not
    // target-motion time, or a speed-0 episode (speed slider at its 0 notch) never resets.
    episode.age += dt;
    if (episode.speed <= 1e-6) return episode;
    const b = CONTRACT.bounds;
    const loX = b.x0 + CONTRACT.wallMargin, hiX = b.x1 - CONTRACT.wallMargin;
    const loY = b.y0 + CONTRACT.wallMargin, hiY = b.y1 - CONTRACT.wallMargin;
    const oldX = episode.target.x, oldY = episode.target.y;
    let x = oldX, y = oldY;

    if (episode.mode === 'cv') {
      x += episode.cvVelocity.x * dt;
      y += episode.cvVelocity.y * dt;
      if (x < loX) { x = 2 * loX - x; episode.cvVelocity.x *= -1; }
      if (x > hiX) { x = 2 * hiX - x; episode.cvVelocity.x *= -1; }
      if (y < loY) { y = 2 * loY - y; episode.cvVelocity.y *= -1; }
      if (y > hiY) { y = 2 * hiY - y; episode.cvVelocity.y *= -1; }
    } else {
      const wx = episode.waypoint.x - oldX, wy = episode.waypoint.y - oldY;
      const d = Math.max(Math.hypot(wx, wy), 1e-6);
      const step = Math.min(episode.speed * dt, d);
      x += wx / d * step;
      y += wy / d * step;
      if (Math.hypot(episode.waypoint.x - x, episode.waypoint.y - y) < CONTRACT.waypointReach) {
        episode.waypoint = sampleWaypoint(rng);
      }
    }

    // Match target_motion.steer_target_step: keep the direct full-speed heading when clear,
    // otherwise take the smallest symmetric turn that clears the bars and arena wall.
    const steered = steerTargetStep(
      oldX, oldY, (x - oldX) / dt, (y - oldY) / dt,
      episode.speed, dt, bars, episode.avoidSign, episode.heading
    );
    x = steered.x;
    y = steered.y;
    episode.heading = steered.heading; // persist for ALL patterns, not just cv
    if (episode.mode === 'cv') {
      episode.cvVelocity.x = steered.vx;
      episode.cvVelocity.y = steered.vy;
    }

    const clear = pushOutOfBars({ x: x, y: y }, bars, CONTRACT.targetBarClearance);
    episode.target.x = clamp(clear.x, loX, hiX);
    episode.target.y = clamp(clear.y, loY, hiY);
    episode.realizedVelocity.x = (episode.target.x - oldX) / dt;
    episode.realizedVelocity.y = (episode.target.y - oldY) / dt;
    if (clear.pushed && episode.mode === 'waypoint') episode.waypoint = sampleWaypoint(rng);
    if (clear.pushed && episode.mode === 'cv') {
      const into = episode.cvVelocity.x * clear.normalX
        + episode.cvVelocity.y * clear.normalY;
      if (into < 0) {
        const vx = episode.cvVelocity.x - 2 * into * clear.normalX;
        const vy = episode.cvVelocity.y - 2 * into * clear.normalY;
        const jitter = (rng() - 0.5) * Math.PI / 9;
        const c = Math.cos(jitter), s = Math.sin(jitter);
        episode.cvVelocity.x = vx * c - vy * s;
        episode.cvVelocity.y = vx * s + vy * c;
        // Re-sync the persisted heading, or the next step's continuity window is measured
        // against the pre-bounce heading and the guard fights the bounce.
        episode.heading = Math.atan2(episode.cvVelocity.y, episode.cvVelocity.x);
      }
    }
    return episode;
  }

  return {
    CONTRACT: CONTRACT,
    configure: configure,
    seededRng: seededRng,
    createFixedStepClock: createFixedStepClock,
    limitPlanarVelocity: limitPlanarVelocity,
    boundedTargetStep: boundedTargetStep,
    routeVelocityReference: routeVelocityReference,
    distanceToBars: distanceToBars,
    pursuerClearance: pursuerClearance,
    segmentPursuerClearance: segmentPursuerClearance,
    sampleWaypoint: sampleWaypoint,
    createEpisode: createEpisode,
    pushOutOfBars: pushOutOfBars,
    pushPursuerOut: pushPursuerOut,
    steerTargetStep: steerTargetStep,
    steerPursuerStep: steerPursuerStep,
    sweptCapture: sweptCapture,
    advanceTarget: advanceTarget,
  };
});
