(function () {
  'use strict';

  const geometry = {
    arena_xy_m: 40,
    goal_dist_m: [6, 28],
    target_speed_m: [0.3, 1.5],
    bar_x_min_ratio: 0,
    bar_x_max_ratio: 1,
    bar_height_m: 3,
    placement_mode: 'footprint_clearance',
    placement_surface_clearance_m: 0.45,
  };

  function byId(id) { return document.getElementById(id); }

  function boot() {
    const stage = byId('stage');
    if (!stage) return;
    try {
      if (typeof THREE === 'undefined') throw new Error('THREE is unavailable');
      if (!THREE.OrbitControls) throw new Error('OrbitControls is unavailable');
      if (!window.Arena) throw new Error('Arena module is unavailable');

      stage.textContent = '';
      window.Arena.configure(geometry);
      window.Arena.init(stage);

      const bars = byId('sl-bars');
      const speed = byId('sl-speed');
      const targetMotion = byId('sel-target-motion');
      const setBars = function () {
        const value = Number(bars.value);
        byId('lbl-bars').textContent = String(value);
        byId('hud-bars').textContent = String(value);
        byId('hud-density').textContent = (value / 16).toFixed(1);
        window.Arena.setBars(value);
      };
      const setSpeed = function () {
        const value = Number(speed.value) / 10;
        byId('lbl-speed').textContent = value.toFixed(1);
        window.Arena.setSpeed(value);
      };

      bars.addEventListener('input', setBars);
      speed.addEventListener('input', setSpeed);
      const motionNotes = {
        legacy: '<strong>Legacy:</strong> checkpointed virtual-point lineage; instantaneous steering and push-out corrections.',
        bounded: '<strong>Bounded:</strong> 4.0 m/s², 150°/s, 1.0 s lookahead, 0.77 m centre-clearance. 새 trajectory lineage입니다.',
        'physical-style': '<strong>Physical-style illustration:</strong> bounded command에 rigid-body-like low-pass와 attitude limit을 그립니다. 실제 PhysX나 정책 평가는 아닙니다.',
        'routed-preview': '<strong>Routed preview:</strong> 0.25 m deterministic global route, exact bar AABB, 0.45 m tracking reserve, target 3-D box half-diagonal support(0.2069 m), boundary 1.25 m + support를 적용합니다. 0.5 m waypoint 전환은 exact clearance certificate로 제한하고 실패 후 직전 goal 1.0 m 이내를 제외합니다. 대안 경로가 없으면 zero command입니다. Global route + bounded/lagged browser preview · NOT PhysX/PPO.',
      };
      const setTargetMotion = function () {
        const mode = targetMotion.value;
        window.Arena.setTargetMotionMode(mode);
        const note = byId('motion-mode-note');
        if (note) note.innerHTML = motionNotes[mode];
      };
      targetMotion.addEventListener('change', setTargetMotion);

      let playing = true;
      byId('btn-play').addEventListener('click', function () {
        playing = !playing;
        window.Arena.setPlaying(playing);
        this.textContent = playing ? 'Pause' : 'Play';
        this.setAttribute('aria-pressed', String(playing));
      });
      byId('btn-view').addEventListener('click', function () { window.Arena.cycleView(); });
      byId('btn-preset').addEventListener('click', function () {
        bars.value = '205';
        setBars();
      });
      byId('cb-camera').addEventListener('change', function () { window.Arena.setCamera(this.checked); });
      byId('cb-lidar').addEventListener('change', function () { window.Arena.setLidar(this.checked); });
      byId('cb-trails').addEventListener('change', function () { window.Arena.setTrails(this.checked); });

      setBars();
      setSpeed();
      setTargetMotion();
    } catch (error) {
      stage.innerHTML = '<p class="viewer-error">3D viewer could not start: '
        + String(error && error.message ? error.message : error) + '</p>';
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
}());
