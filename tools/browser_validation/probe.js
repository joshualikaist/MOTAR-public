/*
 * In-page browser validation probe for the MOTAR GT tracking preview.
 *
 * Injected ONLY when the local validation server is asked for
 * docs/status/index.html?probe=1. It is never part of the published page.
 * It drives the real controls, samples real frame timings, and POSTs a JSON
 * report back to the validation server.
 *
 * Engineering validation of the browser preview. Not PPO, not PhysX, not
 * research performance evidence.
 */
(function () {
  'use strict';

  const params = new URLSearchParams(location.search);
  const RUN_SECONDS = Number(params.get('seconds') || 65);
  const LABEL = params.get('label') || 'unknown';

  const report = {
    label: LABEL,
    viewport: {w: window.innerWidth, h: window.innerHeight, dpr: window.devicePixelRatio},
    ua: navigator.userAgent,
    errors: [],
    console_errors: [],
    started: new Date().toISOString(),
  };

  window.addEventListener('error', function (e) {
    report.errors.push(String(e.message || e));
  });
  window.addEventListener('unhandledrejection', function (e) {
    report.errors.push('unhandledrejection: ' + String(e.reason));
  });
  const nativeError = console.error;
  console.error = function () {
    report.console_errors.push(Array.prototype.join.call(arguments, ' ').slice(0, 400));
    return nativeError.apply(console, arguments);
  };

  function byId(id) { return document.getElementById(id); }
  function text(id) { const el = byId(id); return el ? (el.textContent || '').trim() : null; }

  function finish(status) {
    report.status = status;
    report.finished = new Date().toISOString();
    try {
      navigator.sendBeacon('/__probe', new Blob([JSON.stringify(report)],
        {type: 'application/json'}));
    } catch (e) { /* fall through to fetch */ }
    try {
      fetch('/__probe', {method: 'POST', body: JSON.stringify(report),
        headers: {'content-type': 'application/json'}, keepalive: true});
    } catch (e) { /* nothing else to try */ }
    const sink = document.createElement('div');
    sink.id = 'probe-result';
    sink.textContent = JSON.stringify(report);
    document.body.appendChild(sink);
  }

  function waitForArena(done, tries) {
    const left = tries == null ? 200 : tries;
    if (window.Arena && window.Arena.debugState && document.querySelector('#stage canvas')) {
      return done(true);
    }
    if (left <= 0) return done(false);
    setTimeout(function () { waitForArena(done, left - 1); }, 100);
  }

  waitForArena(function (ok) {
    if (!ok) {
      report.errors.push('arena never booted (no canvas or no Arena.debugState)');
      report.boot = false;
      return finish('BOOT_FAILED');
    }
    report.boot = true;
    // The arena deliberately runs only while it is on screen (IntersectionObserver).
    // A validation run has to put it on screen exactly as a reader scrolling to
    // Figure 2 does, otherwise it measures a paused scene.
    const figure = byId('arena') || byId('stage');
    if (figure && figure.scrollIntoView) figure.scrollIntoView({block: 'center'});
    setTimeout(run, 1200);
  });

  function three() {
    return window.THREE || null;
  }

  function sceneCounts() {
    const state = window.Arena && window.Arena.debugState ? window.Arena.debugState() : null;
    return state ? state.renderer : 'NOT_MEASURED';
  }

  function run() {
    const t0 = performance.now();
    const frames = [];
    const samples = [];
    let lastFrame = t0;
    let rafCount = 0;
    let stopped = false;

    const heapStart = (performance.memory && performance.memory.usedJSHeapSize) || null;
    report.heap_start_bytes = heapStart;

    // Geometry / texture counts come from the WebGL context's own accounting if
    // the page exposes it; otherwise they are reported NOT_MEASURED.
    const rendererInfo = sceneCounts();
    report.renderer_info_start = rendererInfo;

    function tick() {
      if (stopped) return;
      requestAnimationFrame(tick);
      const now = performance.now();
      frames.push(now - lastFrame);
      lastFrame = now;
      rafCount += 1;
    }
    requestAnimationFrame(tick);

    // Simulation-state sampling at 10 Hz, independent of rAF.
    let prevDrone = null, prevTarget = null, prevSim = null, prevResets = null;
    let sceneRebuilds = 0;
    let teleports = 0, hudStates = {}, routeStates = {}, followStates = {};
    let rangeSeen = [], rangeParsed = 0;
    const sampler = setInterval(function () {
      const state = window.Arena.debugState ? window.Arena.debugState() : {};
      const range = text('hud-range');
      const cam = text('hud-camera');
      const follow = text('hud-follow-state');
      const route = text('hud-route-state');
      if (cam) hudStates[cam] = (hudStates[cam] || 0) + 1;
      if (route) routeStates[route] = (routeStates[route] || 0) + 1;
      if (follow) followStates[follow] = (followStates[follow] || 0) + 1;
      const m = range && range.match(/(-?\d+(?:\.\d+)?)/);
      if (m) { rangeSeen.push(Number(m[1])); rangeParsed += 1; }
      const here = state.tracker, there = state.target;
      const simSteps = state.counters ? state.counters.simSteps : null;
      // Bound the expected motion by the number of simulation steps actually
      // COMMITTED between two samples, not by wall clock: a 10 Hz setInterval
      // drifts under render load and would report legal motion as a jump.
      const resets = state.counters ? state.counters.resets : null;
      // A scene rebuild respawns both aircraft on purpose; only compare poses
      // across samples that belong to the SAME scene.
      const sameScene = resets != null && prevResets != null && resets === prevResets;
      if (sameScene && simSteps != null && prevSim != null) {
        const budget = 2.5 * 0.1 * Math.max(0, simSteps - prevSim) + 0.05;
        if (here && prevDrone
            && Math.hypot(here.x - prevDrone.x, here.y - prevDrone.y) > budget) teleports += 1;
        if (there && prevTarget
            && Math.hypot(there.x - prevTarget.x, there.y - prevTarget.y) > budget) teleports += 1;
      }
      if (resets != null && prevResets != null && resets !== prevResets) sceneRebuilds += 1;
      prevResets = resets;
      prevSim = simSteps;
      prevDrone = here ? {x: here.x, y: here.y} : null;
      prevTarget = there ? {x: there.x, y: there.y} : null;
      samples.push({t: performance.now() - t0, follow: follow, route: route, cam: cam,
        gt: state.gtPreview === true, d: state.distance, bars: state.bars,
        sim: state.counters ? state.counters.simSteps : null,
        planner: state.counters ? state.counters.plannerCalls : null,
        lidar: state.counters ? state.counters.lidarDraws : null});
    }, 100);

    // --- scripted interaction timeline (real seconds) --------------------
    const steps = [
      [2, 'cycle camera view', function () { byId('btn-view').click(); }],
      [4, 'cycle camera view', function () { byId('btn-view').click(); }],
      [6, 'cycle camera view back to overview', function () { byId('btn-view').click(); }],
      [10, 'pause', function () { byId('btn-play').click(); }],
      [13, 'resume', function () { byId('btn-play').click(); }],
      [16, 'click goal mode', function () {
        const sel = byId('sel-target-goal');
        sel.value = 'click';
        sel.dispatchEvent(new Event('change', {bubbles: true}));
      }],
      [18, 'tap ground goal', function () {
        const stage = byId('stage');
        const rect = stage.getBoundingClientRect();
        const x = rect.left + rect.width * 0.45, y = rect.top + rect.height * 0.55;
        for (const type of ['pointerdown', 'pointerup']) {
          stage.dispatchEvent(new PointerEvent(type, {
            bubbles: true, cancelable: true, clientX: x, clientY: y,
            pointerId: 1, pointerType: 'touch', isPrimary: true,
          }));
        }
        report.click_feedback = text('hud-click-feedback');
      }],
      [24, 'auto roam', function () {
        const sel = byId('sel-target-goal');
        sel.value = 'auto';
        sel.dispatchEvent(new Event('change', {bubbles: true}));
      }],
      [28, 'bars 115', function () {
        const sl = byId('sl-bars');
        sl.value = '115';
        sl.dispatchEvent(new Event('input', {bubbles: true}));
      }],
      [34, 'bars 205 preset', function () { byId('btn-preset').click(); }],
      [40, 'target speed 0.3', function () {
        const sl = byId('sl-speed');
        sl.value = '3';
        sl.dispatchEvent(new Event('input', {bubbles: true}));
      }],
      [46, 'target speed 1.5', function () {
        const sl = byId('sl-speed');
        sl.value = '15';
        sl.dispatchEvent(new Event('input', {bubbles: true}));
      }],
      [52, 'toggle lidar off/on', function () {
        const cb = byId('cb-lidar');
        cb.click();
        setTimeout(function () { cb.click(); }, 500);
      }],
      [56, 'chase view', function () { byId('btn-view').click(); }],
    ];
    report.timeline = [];
    for (const [at, name, fn] of steps) {
      setTimeout(function () {
        let error = null;
        try { fn(); } catch (e) { error = String(e && e.message || e); }
        report.timeline.push({t: at, step: name, error: error,
          follow: text('hud-follow-state'), route: text('hud-route-state')});
      }, at * 1000);
    }

    setTimeout(function () {
      stopped = true;
      clearInterval(sampler);
      const elapsed = (performance.now() - t0) / 1000;
      const sorted = frames.slice(1).sort(function (a, b) { return a - b; });
      const q = function (p) {
        return sorted.length ? sorted[Math.min(sorted.length - 1,
          Math.floor((sorted.length - 1) * p))] : null;
      };
      report.elapsed_s = elapsed;
      report.frames = frames.length;
      report.fps_mean = frames.length / elapsed;
      report.frame_time_ms = sorted.length
        ? {median: q(0.5), p95: q(0.95), p99: q(0.99), max: q(1)} : 'NOT_MEASURED';
      report.heap_end_bytes = (performance.memory && performance.memory.usedJSHeapSize) || null;
      report.heap_delta_bytes = (report.heap_start_bytes != null && report.heap_end_bytes != null)
        ? report.heap_end_bytes - report.heap_start_bytes : null;
      report.hud_camera_states = hudStates;
      report.hud_route_states = routeStates;
      report.hud_follow_states = followStates;
      report.range_samples = rangeSeen.length;
      report.range_min = rangeSeen.length ? Math.min.apply(null, rangeSeen) : null;
      report.range_max = rangeSeen.length ? Math.max.apply(null, rangeSeen) : null;
      report.samples = samples.length;
      report.teleports = teleports;
      report.scene_rebuilds = sceneRebuilds;
      const distances = samples.map(function (x) { return x.d; })
        .filter(function (d) { return typeof d === 'number' && isFinite(d); });
      report.distance_note = 'spans all scenes in the scripted timeline, rebuilds included';
      const ds = distances.slice().sort(function (a, b) { return a - b; });
      report.distance = ds.length ? {
        first: distances[0], last: distances[distances.length - 1],
        min: ds[0], median: ds[Math.floor((ds.length - 1) * 0.5)], max: ds[ds.length - 1],
        samples: ds.length,
        decreasing_fraction: (function () {
          let dec = 0;
          for (let i = 1; i < distances.length; i++) if (distances[i] < distances[i - 1]) dec += 1;
          return distances.length > 1 ? dec / (distances.length - 1) : null;
        })(),
      } : 'NOT_MEASURED';
      const last = samples[samples.length - 1] || {};
      report.rates = {
        sim_steps_per_s: last.sim != null ? last.sim / elapsed : 'NOT_MEASURED',
        planner_calls_per_s: last.planner != null ? last.planner / elapsed : 'NOT_MEASURED',
        lidar_updates_per_s: last.lidar != null ? last.lidar / elapsed : 'NOT_MEASURED',
      };
      report.gt_badge = text('hud-gt-badge');
      report.sim_time_advanced = samples.length > 1 && samples[samples.length - 1].sim != null
        && samples[0].sim != null
        ? samples[samples.length - 1].sim - samples[0].sim : null;
      report.debug_state = window.Arena.debugState();
      report.renderer_info_end = sceneCounts();
      report.canvas = (function () {
        const c = document.querySelector('#stage canvas');
        return c ? {w: c.width, h: c.height, client_w: c.clientWidth, client_h: c.clientHeight} : null;
      })();
      report.document_scroll_width = document.documentElement.scrollWidth;
      report.window_inner_width = window.innerWidth;
      report.horizontal_overflow = document.documentElement.scrollWidth > window.innerWidth + 1;
      finish('OK');
    }, RUN_SECONDS * 1000);
  }
})();
