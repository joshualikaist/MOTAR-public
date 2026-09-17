/* MOTAR — bright daylight 3D perception arena (three.js r128).
   Dedicated stage panel (not a buried hero background).
   Bounds default to task v1 (x 0..24, y -12..12, height 0..3 = three y up); Arena.configure()
   overrides them from status.json so a v2 run is drawn at its real 40 x 40 scale with the
   full-width obstacle band and 3 m bars instead of silently rendering the old arena. */
window.Arena = (() => {
  let X0 = 0, X1 = 24, Y0 = -12, Y1 = 12, BX0 = 3.1, BX1 = 23;
  let BAR_HEIGHT = 2, PLACEMENT = 'random', TOUCH_M = 0.4, GAP_M = 1.6;
  let SURFACE_CLEARANCE_M = 0.45;
  const CAMERA_RANGE = 20, CAMERA_HALF_FOV = THREE.MathUtils.degToRad(43.5), LIDAR_RANGE = 12;
  const LIDAR_HBEAMS = 72, LIDAR_VBEAMS = 4;
  const LIDAR_ELEVATION_MIN = THREE.MathUtils.degToRad(-10);
  const LIDAR_ELEVATION_MAX = THREE.MathUtils.degToRad(20);
  const Motion = window.NavRLArenaMotion;
  if (!Motion) throw new Error('NavRLArenaMotion missing (arena_motion.js)');
  const Route = window.NavRLArenaRoute;
  if (!Route) throw new Error('NavRLArenaRoute missing (arena_route.js)');
  const Planner = window.NavRLArenaDemoPlanner;
  if (!Planner) throw new Error('NavRLArenaDemoPlanner missing (arena_demo_planner.js)');

  let scene, cam, renderer, controls, root, barMesh, drone, target, cameraFov, lidarLines;
  let pursuerTrail, targetTrail, routeLine, pursuerRouteLine, leadLine, followMark, targetHalo, resizeObserver;
  let groundMat, gridHelper, rimLight, groundMesh, sunLight;
  let bars = [], playing = true, speedCeiling = 1.5, viewMode = 0;
  let targetMotionMode = 'gt-free-roam';
  let pursuerDisplayMode = 'gt-route-track';
  let targetGoalMode = 'auto';
  let currentBars = 25, layoutSeed = 20260728, episode, gtSession;
  let showTrails = true, frame = 0, visible = true, lidarNeedsDraw = true;
  let host, lastDrone = { x: 1, y: 0 };
  let lastT = 0, vel = { x: 0, y: 0 }, heading = 0, simPrev, simCurr, simTime = 0;
  let clickFeedbackUntil = 0, clickFeedbackText = '';
  let a11yUntil = 0, a11yLast = '';
  const simClock = Motion.createFixedStepClock(0.1, 0.25, 8);
  const motionRng = Motion.seededRng(8675309);
  const routeSupport = Route.conservativeXYSupportFromBox(Route.CONTRACT.physicalBoxXYZ);
  const TRAIL_MAX = 420;
  const hud = {};
  const tmpV3a = new THREE.Vector3();
  const tmpV3b = new THREE.Vector3();
  const tmpQuat = new THREE.Quaternion();
  const tmpRay = new THREE.Raycaster();
  const tmpPtr = new THREE.Vector2();
  const pointerDown = {x: 0, y: 0, active: false};

  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const coarsePointer = matchMedia('(pointer: coarse)').matches;

  function cacheHud() {
    [
      'hud-pattern', 'hud-target-speed', 'hud-motion-lineage', 'hud-route-state',
      'hud-camera', 'hud-range', 'hud-gt-badge', 'hud-follow-state',
      'hud-pursuer-state', 'hud-click-feedback', 'hud-a11y', 'btn-view',
    ].forEach(function (id) { hud[id] = document.getElementById(id); });
  }

  function gtPreviewActive() {
    return pursuerDisplayMode === 'gt-route-track' || targetMotionMode === 'gt-free-roam';
  }

  function makeDynamicLine(color, opacity, maxPoints) {
    const positions = new Float32Array(maxPoints * 3);
    const geom = new THREE.BufferGeometry();
    const attr = new THREE.BufferAttribute(positions, 3);
    if (attr.setUsage && THREE.DynamicDrawUsage) attr.setUsage(THREE.DynamicDrawUsage);
    geom.setAttribute('position', attr);
    geom.setDrawRange(0, 0);
    const mesh = new THREE.Line(geom, new THREE.LineBasicMaterial({
      color: color, transparent: opacity < 1, opacity: opacity,
    }));
    return {mesh: mesh, positions: positions, count: 0, max: maxPoints, last: null};
  }

  function resetTrail(trail) {
    trail.count = 0;
    trail.last = null;
    trail.mesh.geometry.setDrawRange(0, 0);
    trail.mesh.geometry.attributes.position.needsUpdate = true;
  }

  function pushTrail(trail, x, y, z) {
    if (!showTrails) { trail.mesh.visible = false; return; }
    trail.mesh.visible = true;
    const last = trail.last;
    if (last && (x - last.x) * (x - last.x) + (y - last.y) * (y - last.y)
        + (z - last.z) * (z - last.z) < 0.0144) return;
    if (trail.count >= trail.max) {
      trail.positions.copyWithin(0, 3);
      trail.count = trail.max - 1;
    }
    const i = trail.count * 3;
    trail.positions[i] = x;
    trail.positions[i + 1] = y;
    trail.positions[i + 2] = z;
    trail.count += 1;
    trail.last = {x: x, y: y, z: z};
    trail.mesh.geometry.setDrawRange(0, trail.count);
    trail.mesh.geometry.attributes.position.needsUpdate = true;
  }

  function writeLinePoints(lineObj, points) {
    const geom = lineObj.geometry;
    let attr = geom.getAttribute('position');
    const need = Math.max(2, points.length);
    if (!attr || attr.count < need) {
      geom.dispose();
      const positions = new Float32Array(Math.max(need, 16) * 3);
      attr = new THREE.BufferAttribute(positions, 3);
      if (attr.setUsage && THREE.DynamicDrawUsage) attr.setUsage(THREE.DynamicDrawUsage);
      geom.setAttribute('position', attr);
    }
    const arr = attr.array;
    for (let i = 0; i < points.length; i++) {
      arr[i * 3] = points[i].x;
      arr[i * 3 + 1] = points[i].y;
      arr[i * 3 + 2] = points[i].z;
    }
    geom.setDrawRange(0, points.length);
    attr.needsUpdate = true;
    if (lineObj.material && lineObj.material.isLineDashedMaterial && geom.computeLineDistances) {
      geom.computeLineDistances();
    }
  }

  function arenaSpan() {
    return Math.max(X1 - X0, Y1 - Y0);
  }

  function setOverviewCamera() {
    if (!cam) return;
    const span = arenaSpan();
    cam.position.set(span * .58, Math.max(11, span * .30), span * .75);
    if (controls) {
      controls.target.set(0, 1.1, 0);
      controls.update();
    }
  }

  function isLight() {
    const t = document.documentElement.getAttribute('data-theme');
    if (t) return t === 'light';
    return !matchMedia('(prefers-color-scheme: dark)').matches;
  }

  function placeBars(n) {
    const r = Motion.seededRng(layoutSeed++); const pts = [];
    const bw = () => 0.4 + r() * 0.4;
    let guard = 0;

    if (PLACEMENT === 'footprint_clearance') {
      // Mirror the physical fresh-lineage guarantee. The browser bars are squares, so w/sqrt(2)
      // is their yaw-invariant circumcircle radius. There is no overlap/merge fallback.
      while (pts.length < n && guard < n * 2000) {
        guard++;
        const w = bw(), support = w / Math.sqrt(2);
        const x = BX0 + support + r() * Math.max(0, BX1 - BX0 - 2 * support);
        const y = Y0 + support + r() * Math.max(0, Y1 - Y0 - 2 * support);
        let ok = true;
        for (const p of pts) {
          const required = support + p.support + SURFACE_CLEARANCE_M;
          if (Math.hypot(x - p.x, y - p.y) < required) { ok = false; break; }
        }
        if (ok) pts.push({ x, y, w, support });
      }
      if (pts.length !== n) throw new Error(`non-overlap layout failed closed at ${pts.length}/${n}`);
      return pts;
    }

    if (PLACEMENT === 'navrl_band') {
      // Mirrors AssetManager._navrl_band_xy_spacing: a candidate is accepted only if EVERY
      // placed bar is either touching it (<= TOUCH_M, merging into a compound wall) or at
      // least GAP_M away (a passable corridor). Distances inside the band would be an
      // impassable slit and are never accepted; on saturation the candidate snaps onto a
      // placed bar (guaranteed merge) rather than relaxing the gap guarantee.
      let fails = 0;
      while (pts.length < n && guard < n * 600) {
        guard++;
        const x = BX0 + r() * (BX1 - BX0), y = Y0 + r() * (Y1 - Y0);
        let ok = true;
        for (const p of pts) {
          const d = Math.hypot(x - p.x, y - p.y);
          if (d > TOUCH_M && d < GAP_M) { ok = false; break; }
        }
        if (ok) { pts.push({ x, y, w: bw() }); fails = 0; continue; }
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

    // legacy v1 rule: min spacing with a *0.8 relaxation once saturated
    let spacing = 1.5, fails = 0;
    while (pts.length < n && guard < n * 400) {
      guard++;
      const x = BX0 + r() * (BX1 - BX0), y = Y0 + r() * (Y1 - Y0);
      let ok = true;
      for (const p of pts) { const dx = x - p.x, dy = y - p.y; if (dx * dx + dy * dy < spacing * spacing) { ok = false; break; } }
      if (ok) { pts.push({ x, y, w: bw() }); fails = 0; }
      else { fails++; if (fails >= 128) { spacing *= 0.8; fails = 0; } }
    }
    return pts;
  }

  function makeBars(n) {
    if (barMesh) { root.remove(barMesh); barMesh.geometry.dispose(); barMesh.material.dispose(); }
    bars = placeBars(n);
    const geometry = new THREE.BoxGeometry(1, BAR_HEIGHT, 1);
    geometry.translate(0, BAR_HEIGHT / 2, 0);
    const material = new THREE.MeshStandardMaterial({ color: 0xc06a2e, roughness: .62, metalness: .08 });
    barMesh = new THREE.InstancedMesh(geometry, material, bars.length);
    barMesh.castShadow = true; barMesh.receiveShadow = true;
    const m = new THREE.Matrix4();
    bars.forEach((p, i) => {
      m.compose(new THREE.Vector3(p.x, 0, p.y), new THREE.Quaternion(), new THREE.Vector3(p.w, 1, p.w));
      barMesh.setMatrixAt(i, m);
    });
    barMesh.instanceMatrix.needsUpdate = true; root.add(barMesh);
  }

  function resetEpisode(regenerateBars = true) {
    if (!drone || !target) return;
    if (regenerateBars) makeBars(currentBars);
    episode = Motion.createEpisode(motionRng, bars, speedCeiling);
    if (targetMotionMode === 'routed-preview') planRoutedEpisode();
    if (gtPreviewActive()) rebuildGtSession(false);
    drone.position.set(episode.drone.x, 1, episode.drone.y);
    target.position.set(episode.target.x, 1, episode.target.y);
    heading = motionRng() * Math.PI * 2 - Math.PI;
    vel = { x: 0, y: 0 };
    lastDrone = { x: episode.drone.x, y: episode.drone.y };
    simPrev = simCurr = snapshotSimulation();
    if (pursuerTrail) resetTrail(pursuerTrail);
    if (targetTrail) resetTrail(targetTrail);
    updateMotionHud();
    updateRouteLine();
    lidarNeedsDraw = true;
  }

  function rebuildGtSession(keepPose) {
    const startTarget = keepPose && gtSession
      ? {x: gtSession.target.x, y: gtSession.target.y, heading: gtSession.target.heading}
      : {x: episode.target.x, y: episode.target.y, heading: episode.heading};
    const startPursuer = keepPose && gtSession
      ? {x: gtSession.pursuer.x, y: gtSession.pursuer.y, heading: gtSession.pursuer.heading}
      : {x: episode.drone.x, y: episode.drone.y, heading: heading};
    gtSession = Planner.createSession({
      bars: bars,
      arenaLo: {x: X0, y: Y0},
      arenaHi: {x: X1, y: Y1},
      rng: motionRng,
      speed: speedCeiling,
      target: startTarget,
      pursuer: startPursuer,
      goalMode: targetGoalMode,
    });
    if (keepPose && episode) {
      episode.target.x = gtSession.target.x;
      episode.target.y = gtSession.target.y;
    }
  }

  function planRoutedEpisode() {
    let result = null;
    // Start sampling is still the browser's explanatory episode sampler. Retry only unsafe starts;
    // the route planner itself chooses a proven connected goal at least 6 m away. A genuine
    // no-connected-goal result is retained and displayed fail-closed rather than hidden.
    for (let attempt = 0; attempt < 12; attempt++) {
      result = Route.planToConnectedGoal(
        episode.target, bars, {x: X0, y: Y0}, {x: X1, y: Y1},
        routeSupport, motionRng()
      );
      if (result.valid || !['unsafe_start', 'unsafe_start_cell'].includes(result.status)) break;
      episode = Motion.createEpisode(motionRng, bars, speedCeiling);
    }
    episode.mode = 'waypoint';
    episode.route = makeRouteState(result, episode.target);
    episode.routeGoalReplacements = 0;
    episode.sameGoalReselectionCount = 0;
    episode.plannerFeasible = episode.route.valid;
    if (episode.route.valid) {
      const goal = episode.route.waypoints[episode.route.waypoints.length - 1];
      episode.waypoint = {x: goal.x, y: goal.y};
    }
  }

  function makeRouteState(result, start) {
    const state = {
      valid: Boolean(result && result.valid),
      status: result ? result.status : 'invalid_input',
      // Python manager excludes the continuous planning start from its GPU waypoint cache.
      waypoints: result && result.valid
        ? result.waypoints.slice(1).map(p => ({x: p.x, y: p.y})) : [],
      handoffClearanceM: result && result.valid
        ? result.handoffClearanceM.slice(1).map(value => Number(value)) : [],
      cursor: 0,
      segmentStart: {x: start.x, y: start.y},
      // Python passes task waypoint_reach_m=0.5 to velocity_reference. The route planner's
      // 0.05 m goal tolerance is a goal-change/replan contract, not waypoint following reach.
      waypointReachM: Motion.CONTRACT.waypointReach,
      goalToleranceM: Route.CONTRACT.goalToleranceM,
      complete: false,
      expandedNodes: result ? result.expandedNodes : 0,
      pathLengthM: result ? result.pathLengthM : 0,
      goal: result && result.valid
        ? Object.assign({}, result.waypoints[result.waypoints.length - 1]) : null,
      replan: replanRoutedGoal,
    };
    return state;
  }

  function replanRoutedGoal(start, previousGoal) {
    const result = Route.planToConnectedGoal(
      start, bars, {x: X0, y: Y0}, {x: X1, y: Y1}, routeSupport, motionRng(),
      {excludedGoal: previousGoal, goalExclusionRadiusM: Route.CONTRACT.goalExclusionRadiusM}
    );
    if (result.status === 'same_goal_reselected') {
      episode.sameGoalReselectionCount = (episode.sameGoalReselectionCount || 0) + 1;
    }
    return makeRouteState(result, start);
  }

  function updateRouteLine() {
    const targetRoute = targetMotionMode === 'gt-free-roam'
      ? (gtSession && gtSession.target.route)
      : (episode && episode.route);
    const showTarget = Boolean(
      (targetMotionMode === 'routed-preview' || targetMotionMode === 'gt-free-roam')
      && targetRoute && targetRoute.valid
    );
    if (routeLine) {
      routeLine.visible = showTarget;
      const points = [];
      if (showTarget) {
        const origin = targetMotionMode === 'gt-free-roam'
          ? gtSession.target : episode.target;
        points.push(new THREE.Vector3(origin.x, .10, origin.y));
        (targetRoute.waypoints || []).slice(targetRoute.cursor || 0).forEach(function (p) {
          const previous = points[points.length - 1];
          if (!previous || Math.hypot(previous.x - p.x, previous.z - p.y) > 1e-6) {
            points.push(new THREE.Vector3(p.x, .10, p.y));
          }
        });
      }
      writeLinePoints(routeLine, points);
    }
    const pursuerRoute = gtSession && gtSession.pursuer.route;
    const showPursuer = pursuerDisplayMode === 'gt-route-track'
      && pursuerRoute && pursuerRoute.valid;
    if (pursuerRouteLine) {
      pursuerRouteLine.visible = Boolean(showPursuer);
      const points = [];
      if (showPursuer) {
        points.push(new THREE.Vector3(gtSession.pursuer.x, .12, gtSession.pursuer.y));
        pursuerRoute.waypoints.slice(pursuerRoute.cursor || 0).forEach(function (p) {
          const previous = points[points.length - 1];
          if (!previous || Math.hypot(previous.x - p.x, previous.z - p.y) > 1e-6) {
            points.push(new THREE.Vector3(p.x, .12, p.y));
          }
        });
      }
      writeLinePoints(pursuerRouteLine, points);
    }
    const lead = gtSession && gtSession.pursuer.lead;
    if (leadLine) {
      const showLead = pursuerDisplayMode === 'gt-route-track' && lead;
      leadLine.visible = Boolean(showLead);
      writeLinePoints(leadLine, showLead ? [
        new THREE.Vector3(episode.target.x, .16, episode.target.y),
        new THREE.Vector3(lead.x, .16, lead.y),
      ] : []);
    }
    if (followMark) {
      const show = pursuerDisplayMode === 'gt-route-track' && lead;
      followMark.visible = Boolean(show);
      if (show) followMark.position.set(lead.x, .02, lead.y);
    }
  }

  function snapshotSimulation() {
    const fromGtTarget = targetMotionMode === 'gt-free-roam' && gtSession;
    const fromGtPursuer = pursuerDisplayMode === 'gt-route-track' && gtSession;
    const targetHeading = fromGtTarget
      ? gtSession.target.heading
      : (episode && Math.hypot(
        episode.realizedVelocity.x, episode.realizedVelocity.y
      ) > .02 ? Math.atan2(episode.realizedVelocity.y, episode.realizedVelocity.x)
        : (episode ? episode.heading : 0));
    const attitudeMode = ['physical-style', 'routed-preview', 'gt-free-roam'].includes(targetMotionMode);
    const pursuerAttitude = fromGtPursuer
      ? Planner.visualAttitude(
        {x: gtSession.pursuer.vx, y: gtSession.pursuer.vy},
        gtSession.pursuer.accel, gtSession.pursuer.heading
      )
      : {heading: heading, roll: 0, pitch: 0};
    return {
      droneX: fromGtPursuer ? gtSession.pursuer.x : (drone ? drone.position.x : lastDrone.x),
      droneY: fromGtPursuer ? gtSession.pursuer.y : (drone ? drone.position.z : lastDrone.y),
      droneHeading: fromGtPursuer ? gtSession.pursuer.heading : heading,
      droneRoll: fromGtPursuer ? Math.atan(pursuerAttitude.roll) : 0,
      dronePitch: fromGtPursuer ? Math.atan(pursuerAttitude.pitch) : 0,
      targetX: fromGtTarget ? gtSession.target.x : (episode ? episode.target.x : 0),
      targetY: fromGtTarget ? gtSession.target.y : (episode ? episode.target.y : 0),
      targetHeading: targetHeading,
      targetRoll: fromGtTarget ? Math.atan(gtSession.target.roll)
        : (episode && attitudeMode ? Math.atan(episode.physicalStyle.roll) : 0),
      targetPitch: fromGtTarget ? Math.atan(gtSession.target.pitch)
        : (episode && attitudeMode ? Math.atan(episode.physicalStyle.pitch) : 0),
    };
  }

  function updateMotionHud() {
    const mode = hud['hud-pattern'];
    const sampled = hud['hud-target-speed'];
    if (mode && episode) mode.textContent = `mixed → ${episode.mode}`;
    if (sampled && episode) sampled.textContent = `${episode.speed.toFixed(2)} m/s sampled`;
    const lineage = hud['hud-motion-lineage'];
    if (lineage) lineage.textContent = {
      legacy: 'historical legacy · checkpointed virtual point',
      bounded: 'TM-E2 local obstacle-aware · NOT policy-compared',
      'physical-style': 'NOT TESTED physical-style · illustrative, not PhysX',
      'routed-preview': 'TM-E2 global route + bounded/lagged browser preview · NOT PhysX/PPO',
      'gt-free-roam': 'BROWSER GT FREE ROAM · NOT PhysX/PPO · NOT TM-E3',
    }[targetMotionMode];
    const routeState = hud['hud-route-state'];
    if (routeState) {
      if (targetMotionMode === 'gt-free-roam' && gtSession) {
        const route = gtSession.target.route;
        const label = gtSession.labels.target;
        if (route && route.valid) {
          routeState.textContent = `${label} · ${route.waypoints.length} pts · ${(route.pathLengthM || 0).toFixed(1)} m · ${gtSession.target.status}`;
          routeState.classList.remove('route-warning');
        } else {
          routeState.textContent = `${label} · NO SAFE ROUTE · STOP + REPLAN`;
          routeState.classList.add('route-warning');
        }
      } else {
        const route = episode && episode.route;
        if (targetMotionMode !== 'routed-preview') {
          routeState.textContent = ''; routeState.classList.remove('route-warning');
        } else if (route && route.valid) {
          routeState.textContent = `ROUTE OK · ${route.waypoints.length} points · ${route.pathLengthM.toFixed(1)} m · goals ${episode.routeGoalReplacements || 0} · same-goal blocks ${episode.sameGoalReselectionCount || 0}`;
          routeState.classList.remove('route-warning');
        } else {
          routeState.textContent = `NO ROUTE · ZERO COMMAND · ${(route && route.status) || 'unplanned'} · same-goal blocks ${episode && episode.sameGoalReselectionCount || 0}`;
          routeState.classList.add('route-warning');
        }
      }
    }
    const badge = hud['hud-gt-badge'];
    if (badge) {
      badge.hidden = !gtPreviewActive();
      badge.textContent = 'BROWSER GT PREVIEW\nNOT PPO · NOT PHYSX · NOT RESEARCH EVIDENCE';
    }
    const follow = hud['hud-follow-state'];
    if (follow) {
      follow.textContent = pursuerDisplayMode === 'gt-route-track' && gtSession
        ? gtSession.pursuer.status : 'LOCAL HEURISTIC';
      follow.classList.toggle('route-warning', gtSession && gtSession.pursuer.status === 'NO SAFE ROUTE');
    }
    const pursuerHud = hud['hud-pursuer-state'];
    if (pursuerHud && gtSession && pursuerDisplayMode === 'gt-route-track') {
      const range = Math.hypot(gtSession.pursuer.x - gtSession.target.x, gtSession.pursuer.y - gtSession.target.y);
      const speed = Math.hypot(gtSession.pursuer.vx, gtSession.pursuer.vy);
      const path = gtSession.pursuer.route && gtSession.pursuer.route.valid
        ? gtSession.pursuer.route.pathLengthM.toFixed(1) : '—';
      pursuerHud.textContent = `GT TRACK · ${speed.toFixed(2)} m/s · ${range.toFixed(1)} m · route ${path} m · replans ${gtSession.pursuer.replanCount}`;
    } else if (pursuerHud) {
      pursuerHud.textContent = pursuerDisplayMode === 'local-heuristic'
        ? 'LOCAL HEURISTIC · historical browser' : '';
    }
    const clickEl = hud['hud-click-feedback'];
    if (clickEl) {
      if (performance.now() < clickFeedbackUntil) clickEl.textContent = clickFeedbackText;
      else clickEl.textContent = '';
    }
  }

  function makeDrone(color, accent, scale = 1) {
    const g = new THREE.Group();
    const bodyMat = new THREE.MeshStandardMaterial({ color, roughness: .28, metalness: .45 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x243240, roughness: .45, metalness: .5 });
    const glow = new THREE.MeshBasicMaterial({ color: accent });
    const body = new THREE.Mesh(new THREE.BoxGeometry(.36 * scale, .11 * scale, .24 * scale), bodyMat);
    body.castShadow = true; g.add(body);
    const nose = new THREE.Mesh(new THREE.ConeGeometry(.09 * scale, .20 * scale, 10), glow);
    nose.rotation.z = -Math.PI / 2; nose.position.x = .26 * scale; g.add(nose);
    for (const z of [-1, 1]) {
      const arm = new THREE.Mesh(new THREE.BoxGeometry(.46 * scale, .035 * scale, .045 * scale), dark);
      arm.rotation.y = z * .58; g.add(arm);
    }
    for (const x of [-1, 1]) for (const z of [-1, 1]) {
      const rotor = new THREE.Mesh(new THREE.RingGeometry(.095 * scale, .12 * scale, 28),
        new THREE.MeshBasicMaterial({ color: 0x5a7180, transparent: true, opacity: .7, side: THREE.DoubleSide }));
      rotor.rotation.x = -Math.PI / 2; rotor.position.set(x * .19 * scale, .055 * scale, z * .13 * scale); g.add(rotor);
      const motor = new THREE.Mesh(new THREE.CylinderGeometry(.025 * scale, .03 * scale, .055 * scale, 10), dark);
      motor.position.copy(rotor.position); g.add(motor);
    }
    return g;
  }

  function makeLabel(textValue, color) {
    const canvas = document.createElement('canvas'); canvas.width = 256; canvas.height = 64;
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, 256, 64);
    ctx.fillStyle = 'rgba(255,255,255,.92)'; ctx.strokeStyle = color; ctx.lineWidth = 2.5;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(4, 4, 248, 56, 10); else ctx.rect(4, 4, 248, 56);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#12202b'; ctx.font = '700 23px ui-monospace, monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(textValue, 128, 33);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
      map: new THREE.CanvasTexture(canvas), transparent: true, depthTest: false,
    }));
    sprite.scale.set(1.9, .48, 1); sprite.position.set(0, 1.05, 0); sprite.renderOrder = 5;
    return sprite;
  }

  function line(points, color, opacity = 1) {
    return new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(points),
      new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }));
  }

  function makeCameraFov() {
    const range = CAMERA_RANGE, half = Math.tan(CAMERA_HALF_FOV) * range;
    const verts = new Float32Array([0, .02, 0, range, .02, -half, range, .02, half,
      0, .02, 0, range, .02, half, range, .02, -half]);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    const mesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({
      color: 0x0d8f82, transparent: true, opacity: .16, side: THREE.DoubleSide, depthWrite: false }));
    const outline = line([new THREE.Vector3(0, .03, 0), new THREE.Vector3(range, .03, -half),
      new THREE.Vector3(range, .03, half), new THREE.Vector3(0, .03, 0)], 0x0d8f82, .7);
    const group = new THREE.Group(); group.position.y = -.05; group.add(mesh, outline); return group;
  }

  function setHex(color, hex) {
    if (color && typeof color.setHex === 'function') color.setHex(hex);
  }

  function applyTheme() {
    if (!scene) return;
    const light = isLight();
    const bg = light ? 0xd5e7f2 : 0x0a1018;
    const ground = light ? 0xe8eef3 : 0x0d1a23;
    const fogDensity = (light ? 0.012 : 0.019) * 24 / arenaSpan();
    // Always replace — never assume scene.background is a Color (it starts as null).
    scene.background = new THREE.Color(bg);
    if (!scene.fog) scene.fog = new THREE.FogExp2(bg, fogDensity);
    else {
      setHex(scene.fog.color, bg);
      scene.fog.density = fogDensity;
    }
    if (groundMat) setHex(groundMat.color, ground);
    if (gridHelper) {
      const mats = Array.isArray(gridHelper.material) ? gridHelper.material : [gridHelper.material];
      setHex(mats[0] && mats[0].color, light ? 0x9bb4c4 : 0x2b4b5e);
      setHex(mats[1] && mats[1].color, light ? 0xc5d5e0 : 0x182e3c);
    }
    if (rimLight) rimLight.intensity = light ? 0.85 : 1.25;
    if (renderer) renderer.toneMappingExposure = light ? 1.15 : 1.1;
    if (lidarLines && lidarLines.material) {
      setHex(lidarLines.material.color, light ? 0x0a7f88 : 0x47d9e3);
      lidarLines.material.opacity = light ? 0.55 : 0.46;
    }
  }

  function applyQuality() {
    if (!renderer || !host || !sunLight) return;
    const reduced = coarsePointer || host.clientWidth < 700;
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, reduced ? 1.25 : 2));
    renderer.shadowMap.enabled = !reduced;
    sunLight.castShadow = !reduced;
    const map = reduced ? 1024 : 2048;
    sunLight.shadow.mapSize.set(map, map);
    if (barMesh) barMesh.castShadow = !reduced;
  }

  function init(el) {
    host = el;
    cacheHud();
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xd5e7f2);
    scene.fog = new THREE.FogExp2(0xd5e7f2, 0.012 * 24 / arenaSpan());

    const w = Math.max(el.clientWidth, 320);
    const h = Math.max(el.clientHeight, 320);
    cam = new THREE.PerspectiveCamera(42, w / h, .08, 180);
    setOverviewCamera();

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(devicePixelRatio, coarsePointer ? 1.25 : 2));
    renderer.setSize(w, h, false);
    renderer.shadowMap.enabled = !coarsePointer; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputEncoding = THREE.sRGBEncoding;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    el.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(cam, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = 0.06;
    controls.target.set(0, 1.1, 0);
    controls.maxPolarAngle = Math.PI / 2.05;
    controls.minDistance = 8; controls.maxDistance = Math.max(48, arenaSpan() * 2);
    controls.enableZoom = true;
    controls.enablePan = false;
    controls.enableRotate = true;
    controls.autoRotate = !reduceMotion;
    controls.autoRotateSpeed = 0.9;
    controls.addEventListener('start', () => { controls.autoRotate = false; });

    scene.add(new THREE.HemisphereLight(0xf2f8fc, 0xb7c6d0, 0.95));
    const dl = new THREE.DirectionalLight(0xffffff, 1.55);
    sunLight = dl;
    const span = arenaSpan();
    dl.position.set(-span * .42, Math.max(28, span * .7), span * .5); dl.castShadow = true;
    dl.shadow.mapSize.set(2048, 2048);
    dl.shadow.camera.left = -span * .7; dl.shadow.camera.right = span * .7;
    dl.shadow.camera.top = span * .7; dl.shadow.camera.bottom = -span * .7;
    scene.add(dl);
    rimLight = new THREE.PointLight(0x2aa8a0, 0.85, span * 2.1);
    rimLight.position.set(-span * .5, Math.max(8, span * .2), -span * .62); scene.add(rimLight);

    root = new THREE.Group(); scene.add(root);
    const centerX = (X0 + X1) / 2, centerY = (Y0 + Y1) / 2;
    root.position.set(-centerX, 0, -centerY);

    groundMat = new THREE.MeshStandardMaterial({ color: 0xe8eef3, roughness: .95, metalness: .02 });
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(X1 - X0, Y1 - Y0), groundMat);
    ground.rotation.x = -Math.PI / 2; ground.position.set(centerX, -.02, centerY);
    ground.receiveShadow = true; root.add(ground);
    groundMesh = ground;

    gridHelper = new THREE.GridHelper(span, Math.max(2, Math.round(span)), 0x9bb4c4, 0xc5d5e0);
    gridHelper.position.set(centerX, 0.01, centerY); root.add(gridHelper);

    root.add(line([new THREE.Vector3(X0, .04, Y0), new THREE.Vector3(X1, .04, Y0),
      new THREE.Vector3(X1, .04, Y1), new THREE.Vector3(X0, .04, Y1),
      new THREE.Vector3(X0, .04, Y0)], 0x5f879c, .9));

    drone = makeDrone(0x1aa86a, 0x7dffc8, 1.2); drone.add(makeLabel('PURSUER', '#1aa86a'));
    drone.position.set(X0 + 1, 1, centerY); root.add(drone);
    target = makeDrone(0xe04545, 0xffb0a0, 1.4); target.add(makeLabel('TARGET', '#e04545'));
    target.position.set(X1 - 2, 1, centerY); root.add(target);
    targetHalo = new THREE.Mesh(new THREE.RingGeometry(.52, .64, 40), new THREE.MeshBasicMaterial({
      color: 0xe04545, transparent: true, opacity: .95, side: THREE.DoubleSide, depthTest: false }));
    targetHalo.rotation.x = -Math.PI / 2; targetHalo.position.y = .02; target.add(targetHalo);
    cameraFov = makeCameraFov(); drone.add(cameraFov);

    const lp = new Float32Array(LIDAR_HBEAMS * LIDAR_VBEAMS * 2 * 3);
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.BufferAttribute(lp, 3));
    lidarLines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({
      color: 0x0a7f88, transparent: true, opacity: .55 }));
    root.add(lidarLines);
    pursuerTrail = makeDynamicLine(0x178a52, .85, TRAIL_MAX);
    targetTrail = makeDynamicLine(0xe04545, .7, TRAIL_MAX);
    routeLine = line([], 0xf3a536, .95);
    pursuerRouteLine = line([], 0x1aa86a, .9);
    leadLine = line([], 0x5b6b75, .85);
    followMark = new THREE.Mesh(new THREE.RingGeometry(.22, .32, 24), new THREE.MeshBasicMaterial({
      color: 0x1aa86a, transparent: true, opacity: .9, side: THREE.DoubleSide, depthTest: false,
    }));
    followMark.rotation.x = -Math.PI / 2;
    followMark.visible = false;
    routeLine.material.depthTest = false; routeLine.renderOrder = 4;
    pursuerRouteLine.material.depthTest = false; pursuerRouteLine.renderOrder = 4;
    leadLine.material.depthTest = false; leadLine.renderOrder = 5;
    root.add(pursuerTrail.mesh, targetTrail.mesh, routeLine, pursuerRouteLine, leadLine, followMark);

    makeBars(currentBars);
    resetEpisode(false);
    applyTheme();
    applyQuality();

    resizeObserver = new ResizeObserver(() => { onResize(); applyQuality(); });
    resizeObserver.observe(el);
    // layout can settle a frame later — force a second resize
    requestAnimationFrame(() => { onResize(); requestAnimationFrame(onResize); });

    addEventListener('keydown', e => { if (e.key.toLowerCase() === 'v') cycleView(); });
    bindGoalPointer(renderer.domElement);
    new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      // Drop off-screen elapsed time immediately. Depending on the browser, RAF may be suspended
      // before animate() gets one last hidden frame, so resetting only inside animate is too late.
      lastT = 0;
      simClock.reset();
    }, { threshold: 0.05 }).observe(el);
    document.addEventListener('visibilitychange', () => {
      // IntersectionObserver does not report a background tab. Without this, restoring a tab
      // advances two synthetic steps from the clamped 250 ms gap before rendering resumes.
      lastT = 0;
      simClock.reset();
    });

    animate();
  }

  function onResize() {
    if (!host || !renderer) return;
    const w = Math.max(host.clientWidth, 1);
    const h = Math.max(host.clientHeight, 1);
    cam.aspect = w / h;
    cam.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }

  function bindGoalPointer(canvas) {
    canvas.addEventListener('pointerdown', function (event) {
      pointerDown.active = true;
      pointerDown.x = event.clientX;
      pointerDown.y = event.clientY;
    });
    canvas.addEventListener('pointerup', function (event) {
      if (!pointerDown.active) return;
      pointerDown.active = false;
      if (Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y) > 8) return;
      if (viewMode !== 0 || targetGoalMode !== 'click' || targetMotionMode !== 'gt-free-roam') return;
      setGoalFromPointer(event);
    });
  }

  function setGoalFromPointer(event) {
    if (!groundMesh || !cam || !gtSession) return;
    const rect = renderer.domElement.getBoundingClientRect();
    tmpPtr.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    tmpPtr.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    tmpRay.setFromCamera(tmpPtr, cam);
    const hits = tmpRay.intersectObject(groundMesh);
    if (!hits.length) {
      showClickFeedback('Click the ground to set a target goal.');
      return;
    }
    const local = root.worldToLocal(hits[0].point.clone());
    const result = Planner.setClickGoal(gtSession, {x: local.x, y: local.z});
    if (!result.ok) {
      showClickFeedback({
        out_of_bounds: 'Goal rejected: outside arena bounds',
        inside_bar: 'Goal rejected: inside a bar',
        unsafe: 'Goal rejected: inflated unsafe region',
      }[result.reason] || 'Goal rejected');
      return;
    }
    targetGoalMode = 'click';
    showClickFeedback('Target goal set');
    updateRouteLine();
    updateMotionHud();
  }

  function showClickFeedback(text) {
    clickFeedbackText = text;
    clickFeedbackUntil = performance.now() + 1800;
    updateMotionHud();
  }

  function announceFollow(now) {
    const node = hud['hud-a11y'];
    if (!node) return;
    const text = (gtSession && gtSession.pursuer.status) || '';
    if (!text || text === a11yLast || now < a11yUntil) return;
    a11yLast = text;
    a11yUntil = now + 2500;
    node.textContent = 'Pursuer ' + text;
  }

  function rayHit(x, y, z, yaw, elevation, maxRange) {
    let best = maxRange;
    const ce = Math.cos(elevation);
    const direction = [Math.cos(yaw) * ce, Math.sin(yaw) * ce, Math.sin(elevation)];
    const origin = [x, y, z];
    for (const p of bars) {
      const half = p.w * 0.5;
      const lo = [p.x - half, p.y - half, 0];
      const hi = [p.x + half, p.y + half, BAR_HEIGHT];
      let enter = 0, exit = best, hit = true;
      for (let axis = 0; axis < 3; axis++) {
        const d = direction[axis];
        if (Math.abs(d) < 1e-9) {
          if (origin[axis] < lo[axis] || origin[axis] > hi[axis]) { hit = false; break; }
          continue;
        }
        let near = (lo[axis] - origin[axis]) / d;
        let far = (hi[axis] - origin[axis]) / d;
        if (near > far) { const swap = near; near = far; far = swap; }
        enter = Math.max(enter, near);
        exit = Math.min(exit, far);
        if (enter > exit) { hit = false; break; }
      }
      if (hit && enter > 0 && enter < best) best = enter;
    }
    return best;
  }

  function drawLidar(x, y) {
    if (!lidarLines.visible) return;
    const a = lidarLines.geometry.attributes.position.array; let k = 0;
    const hStep = coarsePointer ? 2 : 1;
    for (let layer = 0; layer < LIDAR_VBEAMS; layer++) for (let i = 0; i < LIDAR_HBEAMS; i++) {
      const ang = i / LIDAR_HBEAMS * Math.PI * 2;
      const elev = LIDAR_ELEVATION_MIN
        + layer / Math.max(1, LIDAR_VBEAMS - 1) * (LIDAR_ELEVATION_MAX - LIDAR_ELEVATION_MIN);
      const h = 1;
      const hit = (i % hStep === 0) ? rayHit(x, y, h, ang, elev, LIDAR_RANGE) : LIDAR_RANGE;
      const planar = Math.cos(elev) * hit;
      a[k++] = x; a[k++] = h; a[k++] = y;
      a[k++] = x + Math.cos(ang) * planar;
      a[k++] = h + Math.sin(elev) * hit;
      a[k++] = y + Math.sin(ang) * planar;
    }
    lidarLines.geometry.attributes.position.needsUpdate = true;
  }

  function visibility(dx, dy, tx, ty, hdg) {
    const vx = tx - dx, vy = ty - dy, range = Math.hypot(vx, vy), bearing = Math.atan2(vy, vx);
    const rel = Math.atan2(Math.sin(bearing - hdg), Math.cos(bearing - hdg));
    const inFov = Math.abs(rel) <= CAMERA_HALF_FOV && range <= CAMERA_RANGE;
    const hit = rayHit(dx, dy, 1, bearing, 0, Math.min(range, CAMERA_RANGE));
    return { range, visible: inFov && hit >= range - .28, occluded: inFov && hit < range - .28, inFov };
  }

  function updateCamera() {
    if (viewMode === 0) { controls.enabled = true; return; }
    controls.enabled = false;
    drone.getWorldPosition(tmpV3a);
    tmpV3b.set(1, 0, 0).applyQuaternion(drone.getWorldQuaternion(tmpQuat));
    if (viewMode === 1) {
      cam.position.copy(tmpV3a).addScaledVector(tmpV3b, -3.4);
      cam.position.y += 2.1;
      cam.lookAt(tmpV3a.x + tmpV3b.x * 4, tmpV3a.y, tmpV3a.z + tmpV3b.z * 4);
    } else {
      cam.position.copy(tmpV3a).addScaledVector(tmpV3b, .24);
      cam.position.y += .12;
      cam.lookAt(tmpV3a.x + tmpV3b.x * 8, tmpV3a.y, tmpV3a.z + tmpV3b.z * 8);
    }
  }

  function lerpAngle(a, b, t) {
    return a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * t;
  }

  function simulationStep(dt) {
    if (!episode) resetEpisode(false);
    simPrev = simCurr || snapshotSimulation();
    if (gtSession) gtSession.time += dt;
    if (targetMotionMode === 'gt-free-roam') {
      if (!gtSession) rebuildGtSession(false);
      Planner.stepTargetFreeRoam(gtSession, dt);
      episode.target.x = gtSession.target.x;
      episode.target.y = gtSession.target.y;
      episode.realizedVelocity.x = gtSession.target.vx;
      episode.realizedVelocity.y = gtSession.target.vy;
      episode.heading = gtSession.target.heading;
      episode.physicalStyle.velocity = {x: gtSession.target.vx, y: gtSession.target.vy};
      episode.physicalStyle.roll = gtSession.target.roll;
      episode.physicalStyle.pitch = gtSession.target.pitch;
      episode.age += dt;
    } else {
      Motion.advanceTarget(episode, dt, bars, motionRng, targetMotionMode);
      if (gtSession) {
        gtSession.target.x = episode.target.x;
        gtSession.target.y = episode.target.y;
        gtSession.target.vx = episode.realizedVelocity.x;
        gtSession.target.vy = episode.realizedVelocity.y;
        gtSession.target.heading = episode.heading;
      }
    }
    if (targetMotionMode === 'routed-preview' || gtPreviewActive()) {
      updateRouteLine();
      updateMotionHud();
    }
    if (pursuerDisplayMode === 'gt-route-track') {
      if (!gtSession) rebuildGtSession(false);
      Planner.stepPursuerGt(gtSession, dt);
      lastDrone = { x: gtSession.pursuer.x, y: gtSession.pursuer.y };
      vel = { x: gtSession.pursuer.vx, y: gtSession.pursuer.vy };
      heading = gtSession.pursuer.heading;
      drone.position.x = lastDrone.x;
      drone.position.z = lastDrone.y;
      simCurr = snapshotSimulation();
      lidarNeedsDraw = true;
      return;
    }
    const proposed = Motion.steerPursuerStep(
      simPrev.droneX, simPrev.droneY, episode.target.x, episode.target.y,
      Motion.CONTRACT.pursuerSpeedMax, dt, bars, heading, episode.avoidSign
    );
    if (proposed.hit) {
      resetEpisode(true);
      return;
    }
    lastDrone = { x: proposed.x, y: proposed.y };
    vel = { x: proposed.vx, y: proposed.vy };
    if (proposed.heading != null && Math.hypot(proposed.vx, proposed.vy) > .05) {
      heading = proposed.heading;
    }
    drone.position.x = proposed.x;
    drone.position.z = proposed.y;
    simCurr = snapshotSimulation();
    lidarNeedsDraw = true;
    const captured = Motion.sweptCapture(
      { x: simPrev.droneX - simPrev.targetX, y: simPrev.droneY - simPrev.targetY },
      { x: proposed.x - episode.target.x, y: proposed.y - episode.target.y },
      0.5
    );
    if (episode.age >= 30 || captured) resetEpisode(true);
  }

  function animate() {
    requestAnimationFrame(animate);
    if (!visible || !renderer) { lastT = 0; simClock.reset(); return; }
    controls.update();

    const now = performance.now();
    const renderDt = Math.min((now - (lastT || now)) / 1000, 0.05); lastT = now;
    if (!episode) resetEpisode(false);
    const tick = simClock.advance(now / 1000, playing, simulationStep);
    simTime = tick.simulationTime;
    const previous = simPrev || snapshotSimulation();
    const current = simCurr || previous;
    const alpha = tick.alpha;
    const dx = previous.droneX + (current.droneX - previous.droneX) * alpha;
    const dy = previous.droneY + (current.droneY - previous.droneY) * alpha;
    const renderHeading = lerpAngle(previous.droneHeading, current.droneHeading, alpha);
    const hover = reduceMotion ? 0 : 0.008 * Math.sin(simTime * 2.1);
    drone.position.set(dx, 1 + hover, dy);
    drone.rotation.y = -renderHeading;
    const droneRoll = previous.droneRoll + (current.droneRoll - previous.droneRoll) * alpha;
    const dronePitch = previous.dronePitch + (current.dronePitch - previous.dronePitch) * alpha;
    const bank = pursuerDisplayMode === 'gt-route-track'
      ? droneRoll
      : THREE.MathUtils.clamp(-vel.y * 0.12, -.28, .28);
    // Model forward is +X in this y-up scene: roll/bank is rotation.x, pitch is rotation.z.
    drone.rotation.x += (bank - drone.rotation.x) * (1 - Math.exp(-renderDt * 8));
    if (pursuerDisplayMode === 'gt-route-track') {
      drone.rotation.z += (dronePitch - drone.rotation.z) * (1 - Math.exp(-renderDt * 8));
    }
    drone.children.forEach(ch => {
      if (ch.geometry && ch.geometry.type === 'RingGeometry') ch.rotation.z += renderDt * 18;
    });

    const tx = previous.targetX + (current.targetX - previous.targetX) * alpha;
    const ty = previous.targetY + (current.targetY - previous.targetY) * alpha;
    target.position.set(tx, 1 + (reduceMotion ? 0 : 0.01 * Math.sin(simTime * 1.7)), ty);
    target.rotation.y = -lerpAngle(previous.targetHeading, current.targetHeading, alpha);
    target.rotation.x = previous.targetRoll + (current.targetRoll - previous.targetRoll) * alpha;
    target.rotation.z = previous.targetPitch + (current.targetPitch - previous.targetPitch) * alpha;

    const vis = visibility(dx, dy, tx, ty, renderHeading);
    const state = hud['hud-camera'];
    if (state) {
      state.textContent = vis.visible ? 'DETECTED' : vis.occluded ? 'OCCLUDED' : 'OUT OF FOV';
      state.className = vis.visible ? 'seen' : 'lost';
    }
    const rangeEl = hud['hud-range'];
    if (rangeEl) rangeEl.textContent = vis.range.toFixed(1) + ' m';
    if (targetHalo && targetHalo.material) {
      setHex(targetHalo.material.color, vis.visible ? 0x1aa86a : 0xe04545);
      targetHalo.material.opacity = vis.visible ? .95 : .55;
      targetHalo.scale.setScalar(1 + (reduceMotion ? 0 : .06 * Math.sin(simTime * 1.8)));
    }
    if (cameraFov) {
      cameraFov.children.forEach(o => {
        const mats = o.material == null ? [] : (Array.isArray(o.material) ? o.material : [o.material]);
        mats.forEach(m => setHex(m && m.color, vis.visible ? 0x0d8f82 : 0xe04545));
      });
    }
    if (lidarNeedsDraw) {
      drawLidar(current.droneX, current.droneY);
      lidarNeedsDraw = false;
    }
    pushTrail(pursuerTrail, dx, 1, dy);
    pushTrail(targetTrail, tx, 1, ty);
    announceFollow(now);
    updateCamera(); frame++;
    renderer.render(scene, cam);
  }

  function cycleView() {
    viewMode = (viewMode + 1) % 3;
    if (viewMode === 0) {
      setOverviewCamera();
      controls.enabled = true;
    }
    const btn = hud['btn-view'] || document.getElementById('btn-view');
    if (btn) btn.textContent = ['시점 · overview', '시점 · chase', '시점 · sensor'][viewMode];
    return viewMode;
  }

  return {
    init,
    // Apply the running task's real geometry (status.json arena_geometry). Call BEFORE init().
    configure(cfg) {
      if (!cfg) return;
      if (cfg.arena_xy_m) {
        const s = Number(cfg.arena_xy_m);
        X0 = 0; X1 = s; Y0 = -s / 2; Y1 = s / 2;
      }
      if (Motion.configure) Motion.configure(cfg);
      // bar band given as arena fractions, same NAVRL_BAR_X_MIN/MAX the simulator uses
      const fx0 = cfg.bar_x_min_ratio, fx1 = cfg.bar_x_max_ratio;
      BX0 = fx0 == null ? BX0 : X0 + Number(fx0) * (X1 - X0);
      BX1 = fx1 == null ? BX1 : X0 + Number(fx1) * (X1 - X0);
      if (cfg.bar_height_m) BAR_HEIGHT = Number(cfg.bar_height_m);
      if (cfg.placement_mode) PLACEMENT = String(cfg.placement_mode);
      if (cfg.placement_touch_m != null) TOUCH_M = Number(cfg.placement_touch_m);
      if (cfg.placement_gap_m != null) GAP_M = Number(cfg.placement_gap_m);
      if (cfg.placement_surface_clearance_m != null) {
        SURFACE_CLEARANCE_M = Number(cfg.placement_surface_clearance_m);
      }
    },
    setBars(n) {
      currentBars = Math.max(1, Math.round(n));
      makeBars(currentBars);
      resetEpisode(false);
      simClock.reset();
    },
    setSpeed(s) {
      speedCeiling = Math.max(0, Number(s) || 0);
      resetEpisode(false);
      simClock.reset();
    },
    setPlaying(p) {
      playing = p;
      if (!playing && simCurr) simPrev = simCurr;
      simClock.reset();
    },
    setTargetMotionMode(mode) {
      if (!['legacy', 'bounded', 'physical-style', 'routed-preview', 'gt-free-roam'].includes(mode)) {
        throw new Error('unknown target display mode: ' + mode);
      }
      targetMotionMode = mode;
      resetEpisode(false);
      simClock.reset();
      return targetMotionMode;
    },
    setPursuerDisplayMode(mode) {
      if (!['local-heuristic', 'gt-route-track'].includes(mode)) {
        throw new Error('unknown pursuer display mode: ' + mode);
      }
      pursuerDisplayMode = mode;
      resetEpisode(false);
      simClock.reset();
      return pursuerDisplayMode;
    },
    setTargetGoalMode(mode) {
      targetGoalMode = mode === 'click' ? 'click' : 'auto';
      if (gtSession) {
        if (targetGoalMode === 'auto') Planner.setAutoRoam(gtSession);
      }
      updateMotionHud();
      return targetGoalMode;
    },
    setLidar(v) { lidarLines.visible = v; },
    setCamera(v) { cameraFov.visible = v; },
    setTrails(v) {
      showTrails = v;
      if (pursuerTrail) pursuerTrail.mesh.visible = v;
      if (targetTrail) targetTrail.mesh.visible = v;
    },
    cycleView,
    recolor: applyTheme,
    debugState() {
      return {
        targetMotionMode: targetMotionMode,
        pursuerDisplayMode: pursuerDisplayMode,
        targetGoalMode: targetGoalMode,
        gtPreview: gtPreviewActive(),
        follow: gtSession ? gtSession.pursuer.status : null,
        badge: gtPreviewActive(),
      };
    },
  };
})();
