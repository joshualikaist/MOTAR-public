(function () {
  'use strict';

  /* Presentation-only geometry. The arena the browser DRAWS: bar band, bar
   * height and the placement style used for the visual field. Termination
   * semantics are NOT here -- they come from the static research task contract
   * below, so there is exactly one source for them. */
  const presentation = {
    goal_dist_m: [6, 28],
    target_speed_m: [0.3, 1.5],
    bar_height_m: 3,
    placement_mode: 'footprint_clearance',
    placement_surface_clearance_m: 0.45,
  };

  /* Static research task contract: success_radius, episode budget, RL dt and
   * the stable arena geometry, generated from research source by
   * tools/build_research_task_contract.py. Fail closed: if it is missing or
   * malformed the arena does not boot with silently different termination
   * semantics. There is no offline literal fallback on the production page. */
  const TASK_CONTRACT_URL = 'research_task_contract.json?v=20260918r2';
  const REQUIRED_CONTRACT_FIELDS = [
    'success_radius_m', 'episode_len_steps', 'rl_step_dt_s',
    'arena_xy_m', 'arena_z_m', 'placement_touch_m', 'placement_gap_m',
    'bar_x_min_ratio', 'bar_x_max_ratio',
  ];

  function validateContract(contract) {
    if (!contract || contract.schema_version !== 1 || contract.task !== 'navrl') {
      throw new Error('research task contract: unsupported schema');
    }
    for (const key of REQUIRED_CONTRACT_FIELDS) {
      if (!Number.isFinite(Number(contract[key]))) {
        throw new Error('research task contract: missing or non-numeric ' + key);
      }
    }
    if (Number(contract.success_radius_m) <= 0 || Number(contract.episode_len_steps) <= 0
        || Number(contract.rl_step_dt_s) <= 0) {
      throw new Error('research task contract: non-positive termination value');
    }
    return contract;
  }

  function geometryFrom(contract) {
    return Object.assign({}, presentation, {
      arena_xy_m: Number(contract.arena_xy_m),
      arena_z_m: Number(contract.arena_z_m),
      bar_x_min_ratio: Number(contract.bar_x_min_ratio),
      bar_x_max_ratio: Number(contract.bar_x_max_ratio),
      placement_touch_m: Number(contract.placement_touch_m),
      placement_gap_m: Number(contract.placement_gap_m),
      success_radius_m: Number(contract.success_radius_m),
      episode_len_steps: Number(contract.episode_len_steps),
      rl_step_dt_s: Number(contract.rl_step_dt_s),
    });
  }

  function byId(id) { return document.getElementById(id); }

  function boot(contract) {
    const stage = byId('stage');
    if (!stage) return;
    try {
      if (typeof THREE === 'undefined') throw new Error('THREE is unavailable');
      if (!THREE.OrbitControls) throw new Error('OrbitControls is unavailable');
      if (!window.Arena) throw new Error('Arena module is unavailable');

      stage.textContent = '';
      window.Arena.configure(geometryFrom(contract));
      window.Arena.init(stage);

      const bars = byId('sl-bars');
      const speed = byId('sl-speed');
      const targetMotion = byId('sel-target-motion');
      const pursuerDisplay = byId('sel-pursuer-display');
      const targetGoal = byId('sel-target-goal');
      const episodeModeSel = byId('sel-episode-mode');
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
        'gt-free-roam': '<strong>Browser GT free roam:</strong> reachable random (or click/tap) goals with global A*, bounded acceleration/turn, and STOP+REPLAN on failure. Teleport and bar push-out are not used. This is not TM-E3 and not a research result.',
        'gt-route-track': '<strong>Browser GT route tracking:</strong> exact browser target position/velocity/heading and obstacle AABBs drive a predicted follow point, LOS shortcut, and global route. In CLOSE-APPROACH EPISODE the follow standoff ramps to zero and the episode ends at the task success radius or timeout; in CONTINUOUS TRACKING the 1.55 m display standoff is held. Not the PPO policy.',
        interception: '<strong>Close-approach episode:</strong> tracking → close approach (standoff 1.55 m → 0) → final approach, ending at the research task\'s success radius (swept between steps; internally <code>CAPTURED</code>) or at its episode budget. Both values are generated from research source into <code>research_task_contract.json</code>. Terminal state is held ~1.5 s, then a new episode starts. Mirrors task termination semantics; not PPO or PhysX evidence.',
        continuous: '<strong>Continuous tracking:</strong> the GT_BROWSER_V1 demo. The target roams and the tracker holds a 1.55 m display standoff; no terminal outcome, no timeout, no reset.',
        'local-heuristic': '<strong>Historical local heuristic:</strong> heading candidates over a short swept path. Kept for comparison; not deleted.',
      };
      const setTargetMotion = function () {
        const mode = targetMotion.value;
        window.Arena.setTargetMotionMode(mode);
        const note = byId('motion-mode-note');
        if (note) note.innerHTML = motionNotes[mode] || motionNotes['gt-free-roam'];
      };
      const setPursuerDisplay = function () {
        if (!pursuerDisplay || !window.Arena.setPursuerDisplayMode) return;
        window.Arena.setPursuerDisplayMode(pursuerDisplay.value);
        const note = byId('motion-mode-note');
        if (note) note.innerHTML = motionNotes[pursuerDisplay.value] + ' ' + (motionNotes[targetMotion.value] || '');
      };
      const setTargetGoal = function () {
        if (!targetGoal || !window.Arena.setTargetGoalMode) return;
        window.Arena.setTargetGoalMode(targetGoal.value);
      };
      const setEpisodeMode = function () {
        if (!episodeModeSel || !window.Arena.setEpisodeMode) return;
        window.Arena.setEpisodeMode(episodeModeSel.value);
        const note = byId('motion-mode-note');
        if (note) note.innerHTML = motionNotes[episodeModeSel.value] + ' ' + (motionNotes[pursuerDisplay ? pursuerDisplay.value : 'gt-route-track'] || '');
      };
      targetMotion.addEventListener('change', setTargetMotion);
      if (pursuerDisplay) pursuerDisplay.addEventListener('change', setPursuerDisplay);
      if (targetGoal) targetGoal.addEventListener('change', setTargetGoal);
      if (episodeModeSel) episodeModeSel.addEventListener('change', setEpisodeMode);

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
      setPursuerDisplay();
      setTargetGoal();
    } catch (error) {
      stage.innerHTML = '<p class="viewer-error">3D viewer could not start: '
        + String(error && error.message ? error.message : error) + '</p>';
    }
  }

  /* Load the static task contract, then boot. Fail closed and say why: an arena
   * drawn with the wrong termination semantics is worse than no arena. */
  function start() {
    const stage = byId('stage');
    fetch(TASK_CONTRACT_URL)
      .then(function (response) {
        if (!response.ok) throw new Error('HTTP ' + response.status);
        return response.json();
      })
      .then(validateContract)
      .then(boot)
      .catch(function (error) {
        if (stage) {
          stage.textContent = '';
          const message = document.createElement('p');
          message.className = 'viewer-error';
          message.textContent = '3D viewer could not start: research task contract '
            + 'unavailable or malformed (' + error.message + ').';
          stage.appendChild(message);
        }
      });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
}());
