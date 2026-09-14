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

  let scene, cam, renderer, controls, root, barMesh, drone, target, cameraFov, lidarLines;
  let pursuerTrail, targetTrail, routeLine, targetHalo, resizeObserver;
  let groundMat, gridHelper, rimLight;
  let bars = [], playing = true, speedCeiling = 1.5, viewMode = 0;
  let targetMotionMode = 'routed-preview';
  let currentBars = 25, layoutSeed = 20260728, episode;
  let showTrails = true, frame = 0, visible = true;
  let host, lastDrone = { x: 1, y: 0 }, trailA = [], trailB = [];
  let lastT = 0, vel = { x: 0, y: 0 }, heading = 0, simPrev, simCurr;
  const simClock = Motion.createFixedStepClock(0.1, 0.25, 8);
  const motionRng = Motion.seededRng(8675309);
  const routeSupport = Route.conservativeXYSupportFromBox(Route.CONTRACT.physicalBoxXYZ);

  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const coarsePointer = matchMedia('(pointer: coarse)').matches;

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
    drone.position.set(episode.drone.x, 1, episode.drone.y);
    target.position.set(episode.target.x, 1, episode.target.y);
    heading = motionRng() * Math.PI * 2 - Math.PI;
    vel = { x: 0, y: 0 };
    lastDrone = { x: episode.drone.x, y: episode.drone.y };
    simPrev = simCurr = snapshotSimulation();
    trailA.length = 0;
    trailB.length = 0;
    if (pursuerTrail) pursuerTrail.geometry.setFromPoints([]);
    if (targetTrail) targetTrail.geometry.setFromPoints([]);
    updateMotionHud();
    updateRouteLine();
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
    if (!routeLine) return;
    const route = episode && episode.route;
    const visible = targetMotionMode === 'routed-preview' && route && route.valid;
    routeLine.visible = Boolean(visible);
    const points = visible ? [new THREE.Vector3(episode.target.x, .10, episode.target.y)] : [];
    if (visible) {
      route.waypoints.slice(route.cursor).forEach(p => {
        const previous = points[points.length - 1];
        if (!previous || Math.hypot(previous.x - p.x, previous.z - p.y) > 1e-6) {
          points.push(new THREE.Vector3(p.x, .10, p.y));
        }
      });
    }
    routeLine.geometry.setFromPoints(points);
  }

  function snapshotSimulation() {
    const targetHeading = episode && Math.hypot(
      episode.realizedVelocity.x, episode.realizedVelocity.y
    ) > .02 ? Math.atan2(episode.realizedVelocity.y, episode.realizedVelocity.x)
      : (episode ? episode.heading : 0);
    return {
      droneX: drone ? drone.position.x : lastDrone.x,
      droneY: drone ? drone.position.z : lastDrone.y,
      droneHeading: heading,
      targetX: episode ? episode.target.x : 0,
      targetY: episode ? episode.target.y : 0,
      targetHeading: targetHeading,
      targetRoll: episode && ['physical-style', 'routed-preview'].includes(targetMotionMode)
        ? Math.atan(episode.physicalStyle.roll) : 0,
      targetPitch: episode && ['physical-style', 'routed-preview'].includes(targetMotionMode)
        ? Math.atan(episode.physicalStyle.pitch) : 0,
    };
  }

  function updateMotionHud() {
    const mode = document.getElementById('hud-pattern');
    const sampled = document.getElementById('hud-target-speed');
    if (mode && episode) mode.textContent = `mixed → ${episode.mode}`;
    if (sampled && episode) sampled.textContent = `${episode.speed.toFixed(2)} m/s sampled`;
    const lineage = document.getElementById('hud-motion-lineage');
    if (lineage) lineage.textContent = {
      legacy: 'legacy · checkpointed virtual point',
      bounded: 'bounded · new trajectory lineage',
      'physical-style': 'physical-style · illustrative, not PhysX',
      'routed-preview': 'global route + bounded/lagged browser preview · NOT PhysX/PPO',
    }[targetMotionMode];
    const routeState = document.getElementById('hud-route-state');
    if (routeState) {
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

  function init(el) {
    host = el;
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xd5e7f2);
    scene.fog = new THREE.FogExp2(0xd5e7f2, 0.012 * 24 / arenaSpan());

    const w = Math.max(el.clientWidth, 320);
    const h = Math.max(el.clientHeight, 320);
    cam = new THREE.PerspectiveCamera(42, w / h, .08, 180);
    setOverviewCamera();

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setSize(w, h, false);
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
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
    pursuerTrail = line([], 0x178a52, .85); targetTrail = line([], 0xe04545, .7);
    routeLine = line([], 0xf3a536, .95);
    routeLine.material.depthTest = false; routeLine.renderOrder = 4;
    root.add(pursuerTrail, targetTrail, routeLine);

    makeBars(currentBars);
    resetEpisode(false);
    applyTheme();

    resizeObserver = new ResizeObserver(() => onResize());
    resizeObserver.observe(el);
    // layout can settle a frame later — force a second resize
    requestAnimationFrame(() => { onResize(); requestAnimationFrame(onResize); });

    addEventListener('keydown', e => { if (e.key.toLowerCase() === 'v') cycleView(); });
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
    for (let layer = 0; layer < LIDAR_VBEAMS; layer++) for (let i = 0; i < LIDAR_HBEAMS; i++) {
      const ang = i / LIDAR_HBEAMS * Math.PI * 2;
      const elev = LIDAR_ELEVATION_MIN
        + layer / Math.max(1, LIDAR_VBEAMS - 1) * (LIDAR_ELEVATION_MAX - LIDAR_ELEVATION_MIN);
      const h = 1;
      const hit = rayHit(x, y, h, ang, elev, LIDAR_RANGE);
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

  function updateTrail(obj, arr, lineObj) {
    const p = obj.position;
    const last = arr.length ? arr[arr.length - 1] : null;
    if (last && last.distanceToSquared(p) < 0.0144) return;
    arr.push(p.clone()); if (arr.length > 420) arr.shift();
    lineObj.geometry.dispose();
    lineObj.geometry = new THREE.BufferGeometry().setFromPoints(arr);
    lineObj.visible = showTrails;
  }

  function updateCamera() {
    if (viewMode === 0) { controls.enabled = true; return; }
    controls.enabled = false;
    const wp = new THREE.Vector3(); drone.getWorldPosition(wp);
    const dir = new THREE.Vector3(1, 0, 0).applyQuaternion(drone.getWorldQuaternion(new THREE.Quaternion()));
    if (viewMode === 1) {
      cam.position.copy(wp).addScaledVector(dir, -3.4).add(new THREE.Vector3(0, 2.1, 0));
      cam.lookAt(wp.clone().addScaledVector(dir, 4));
    } else {
      cam.position.copy(wp).add(new THREE.Vector3(0, .12, 0)).addScaledVector(dir, .24);
      cam.lookAt(wp.clone().addScaledVector(dir, 8));
    }
  }

  function lerpAngle(a, b, t) {
    return a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * t;
  }

  function simulationStep(dt) {
    if (!episode) resetEpisode(false);
    simPrev = simCurr || snapshotSimulation();
    Motion.advanceTarget(episode, dt, bars, motionRng, targetMotionMode);
    if (targetMotionMode === 'routed-preview') {
      updateRouteLine();
      updateMotionHud();
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
    // Keep the Three objects' authoritative simulation positions at the latest 10 Hz state.
    // Rendering below interpolates without feeding interpolated values back into simulation.
    drone.position.x = proposed.x;
    drone.position.z = proposed.y;
    simCurr = snapshotSimulation();
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
    const previous = simPrev || snapshotSimulation();
    const current = simCurr || previous;
    const alpha = tick.alpha;
    const dx = previous.droneX + (current.droneX - previous.droneX) * alpha;
    const dy = previous.droneY + (current.droneY - previous.droneY) * alpha;
    const renderHeading = lerpAngle(previous.droneHeading, current.droneHeading, alpha);
    drone.position.set(dx, 1 + .03 * Math.sin(frame * .12), dy);
    drone.rotation.y = -renderHeading;
    const bank = THREE.MathUtils.clamp(-vel.y * 0.12, -.28, .28);
    // Model forward is +X in this y-up scene: roll/bank is rotation.x, pitch is rotation.z.
    drone.rotation.x += (bank - drone.rotation.x) * (1 - Math.exp(-renderDt * 8));
    // Spin rotors for visible motion even when path is slow.
    drone.children.forEach(ch => {
      if (ch.geometry && ch.geometry.type === 'RingGeometry') ch.rotation.z += renderDt * 18;
    });

    const tx = previous.targetX + (current.targetX - previous.targetX) * alpha;
    const ty = previous.targetY + (current.targetY - previous.targetY) * alpha;
    target.position.set(tx, 1 + .04 * Math.sin(frame * .1), ty);
    target.rotation.y = -lerpAngle(previous.targetHeading, current.targetHeading, alpha);
    target.rotation.x = previous.targetRoll + (current.targetRoll - previous.targetRoll) * alpha;
    target.rotation.z = previous.targetPitch + (current.targetPitch - previous.targetPitch) * alpha;

    const vis = visibility(dx, dy, tx, ty, renderHeading);
    const state = document.getElementById('hud-camera');
    if (state) {
      state.textContent = vis.visible ? 'DETECTED' : vis.occluded ? 'OCCLUDED' : 'OUT OF FOV';
      state.className = vis.visible ? 'seen' : 'lost';
    }
    const rangeEl = document.getElementById('hud-range');
    if (rangeEl) rangeEl.textContent = vis.range.toFixed(1) + ' m';
    if (targetHalo && targetHalo.material) {
      setHex(targetHalo.material.color, vis.visible ? 0x1aa86a : 0xe04545);
      targetHalo.material.opacity = vis.visible ? .95 : .55;
      targetHalo.scale.setScalar(1 + .14 * Math.sin(frame * .1));
    }
    if (cameraFov) {
      cameraFov.children.forEach(o => {
        const mats = o.material == null ? [] : (Array.isArray(o.material) ? o.material : [o.material]);
        mats.forEach(m => setHex(m && m.color, vis.visible ? 0x0d8f82 : 0xe04545));
      });
    }
    drawLidar(dx, dy);
    updateTrail(drone, trailA, pursuerTrail);
    updateTrail(target, trailB, targetTrail);
    updateCamera(); frame++;
    renderer.render(scene, cam);
  }

  function cycleView() {
    viewMode = (viewMode + 1) % 3;
    if (viewMode === 0) {
      setOverviewCamera();
      controls.enabled = true;
    }
    const btn = document.getElementById('btn-view');
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
      if (!['legacy', 'bounded', 'physical-style', 'routed-preview'].includes(mode)) {
        throw new Error('unknown target display mode: ' + mode);
      }
      targetMotionMode = mode;
      resetEpisode(false);
      simClock.reset();
      return targetMotionMode;
    },
    setLidar(v) { lidarLines.visible = v; },
    setCamera(v) { cameraFov.visible = v; },
    setTrails(v) { showTrails = v; pursuerTrail.visible = v; targetTrail.visible = v; },
    cycleView,
    recolor: applyTheme,
  };
})();
