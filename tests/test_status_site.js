'use strict';

const assert = require('assert');
const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const repo = path.resolve(__dirname, '..');
const site = path.join(repo, 'docs/status');
// Historical evidence/figure assertions remain enforced on the preserved detail page.
// Current overview and live viewer contracts are tested separately.
const html = fs.readFileSync(path.join(site, 'archive-2026-09-13.html'), 'utf8');
const css = fs.readFileSync(path.join(site, 'style.css'), 'utf8');
// The landing page links the evidence overview; full historical figure contracts stay there.
const landing = fs.readFileSync(path.join(repo, 'README.md'), 'utf8');
assert(landing.includes('docs/results_overview_2026-09-12.md'));
const readme = fs.readFileSync(path.join(repo, 'docs/results_overview_2026-09-12.md'), 'utf8');
const platform = JSON.parse(fs.readFileSync(path.join(site, 'data/platform.json'), 'utf8'));
const status = JSON.parse(fs.readFileSync(path.join(site, 'status.json'), 'utf8'));
const experiments = JSON.parse(fs.readFileSync(path.join(site, 'data/experiments.json'), 'utf8')).experiments;
const Motion = require('../docs/status/arena_motion.js');
const arenaSource = fs.readFileSync(path.join(site, 'arena.js'), 'utf8');
const trackedFiles = new Set(
  execFileSync('git', ['ls-files', '--cached', '-z'], {
    cwd: repo,
    encoding: 'utf8',
  }).split('\0').filter(Boolean),
);
const pendingDocs = new Set(
  execFileSync('git', ['ls-files', '--others', '--exclude-standard', '-z'], {
    cwd: repo,
    encoding: 'utf8',
  }).split('\0').filter((file) => file.startsWith('docs/')),
);

// The public site is one presentation page. JavaScript is limited to the self-contained 3-D
// arena; evidence and claims remain static HTML and never depend on a dashboard renderer.
for (const id of ['arena', 'method', 'perception', 'safety-filter', 'evidence', 'platform', 'next']) {
  assert(html.includes(`id="${id}"`), `missing section #${id}`);
}
assert(html.includes('id="panel-runs"'), 'legacy #panel-runs deep link must remain valid');
assert(!css.includes('@import'));
assert(!css.includes('fonts.googleapis.com'));
assert(html.includes('style.css?v=20260827r6'), 'compact site CSS cache-bust must advance with the layout');
assert(css.includes('height: clamp(300px, 44vh, 500px)'), 'desktop viewer must stay within a viewport-friendly clamp');
assert(css.includes('height: clamp(260px, 64vw, 340px)'), 'mobile viewer must retain a compact height clamp');
assert(css.includes('font-size: 11px'), 'viewer HUD must remain compact');
assert(css.includes('body { margin: 0; overflow-x: hidden; color: var(--ink); background: var(--paper); font: 15px'), 'body text must remain compact and readable');
assert(css.includes('word-break: break-word'), 'mobile authority status must not clip long slash-delimited tokens');
assert(html.includes('../research_authority_2026-08-26.json'), 'frozen authority receipt must be linked');
for (const retired of [
  'system.html', 'setup.html', 'results.html', 'experiments.html', 'parameters.html', 'drone.html',
  'status.fallback.js', 'js',
]) {
  assert(!fs.existsSync(path.join(site, retired)), `retired dashboard asset remains: ${retired}`);
}
for (const viewerAsset of [
  'arena.js', 'arena_route.js', 'arena_motion.js', 'viewer.js',
  'vendor/three.min.js', 'vendor/OrbitControls.js',
]) assert(fs.existsSync(path.join(site, viewerAsset)), `missing viewer asset: ${viewerAsset}`);
assert(html.indexOf('three.min.js') < html.indexOf('OrbitControls.js'));
assert(html.indexOf('OrbitControls.js') < html.indexOf('arena_route.js'));
assert(html.indexOf('arena_route.js') < html.indexOf('arena_motion.js'));
assert(html.indexOf('arena_motion.js') < html.indexOf('arena.js'));
assert(html.indexOf('arena.js') < html.indexOf('viewer.js'));
for (const script of [
  'vendor/three.min.js', 'vendor/OrbitControls.js', 'arena_route.js',
  'arena_motion.js', 'arena.js', 'viewer.js',
]) assert(html.includes(`${script}?v=20260827r6`), `stale cache-bust for ${script}`);
assert(html.includes('이 화면은 PPO 실행 영상'));
assert(html.includes('10 Hz 고정 simulation clock'));
assert(html.includes('PhysX 재생'));
assert(html.includes('value="routed-preview" selected'));
assert(html.includes('historical <code>global_astar_v1</code>'));
assert(html.includes('recovery-v2 state machine'));
assert(html.includes('Physical-style illustration · NOT PhysX'));
assert(html.includes('Global route + bounded/lagged browser preview · NOT PhysX/PPO'));
assert(html.includes('half-diagonal support(0.2069 m)'));
assert(html.includes('직전 goal 1.0 m exclusion'));
assert(html.includes('id="hud-route-state"'));
assert(html.includes('id="hud-motion-lineage"'));
assert(arenaSource.includes('routeLine.geometry.setFromPoints'));
assert(arenaSource.includes('route.waypoints.slice(route.cursor)'));
assert(arenaSource.includes('waypointReachM: Motion.CONTRACT.waypointReach'));
assert(arenaSource.includes("document.addEventListener('visibilitychange'"));
assert(arenaSource.includes('drone.rotation.x += (bank - drone.rotation.x)'));
assert(arenaSource.includes('target.rotation.x = previous.targetRoll'));
assert(arenaSource.includes('target.rotation.z = previous.targetPitch'));

Motion.configure({arena_xy_m: 40, goal_dist_m: [6, 28], target_speed_m: [0.3, 1.5]});
assert.deepStrictEqual(Motion.CONTRACT.bounds, {x0: 0, x1: 40, y0: -20, y1: 20});
assert.strictEqual(Motion.CONTRACT.targetDistanceMin, 6);
assert.strictEqual(Motion.CONTRACT.targetDistanceMax, 28);
const episode = Motion.createEpisode(Motion.seededRng(20260824), [], 1.5);
assert(episode.speed >= 0.3 && episode.speed <= 1.5);
episode.speed = 0;
const age = episode.age;
Motion.advanceTarget(episode, 0.1, [], Motion.seededRng(1));
assert.strictEqual(episode.age, age + 0.1, 'speed-zero viewer episodes must still age/reset');

// GitHub README and the site must render the same canonical architecture assets directly.
for (const asset of [
  'motar-system-overview.svg', 'motar-control-stack.svg',
  'motar-perception-final.svg',
  'motar-eth-e3-evidence.svg',
  'motar-perception-detection.svg', 'motar-safety-filter.svg',
  'motar-perception-candidate.svg',
]) {
  const assetPath = path.join(repo, 'docs/assets', asset);
  const svg = fs.readFileSync(assetPath, 'utf8');
  assert(svg.includes('<title'));
  assert(svg.includes('<desc'));
  assert(readme.includes(`assets/${asset}`));
  assert(html.includes(`../assets/${asset}`));
}
const currentPerception = fs.readFileSync(
  path.join(repo, 'docs/assets/motar-perception-detection.svg'),
  'utf8',
);
assert(currentPerception.includes('단일 중심점'), 'baseline perception diagram must keep the union-centroid failure');
assert(currentPerception.includes('CURRENT BASELINE — KNOWN FAILURE'), 'baseline diagram must expose its failed status');
const candidatePerception = fs.readFileSync(
  path.join(repo, 'docs/assets/motar-perception-candidate.svg'),
  'utf8',
);
assert(candidatePerception.includes('not in the control loop'), 'candidate diagram must stay labelled as not in the control loop');
assert(candidatePerception.includes('CANDIDATE'), 'candidate diagram must stay labelled CANDIDATE');
assert(candidatePerception.includes('ARCHIVED — SAM / in-sim shape detector'), 'candidate diagram must expose its archived status');
assert(candidatePerception.includes('IMPLEMENTED / CPU'), 'candidate diagram must distinguish implemented work');
assert(candidatePerception.includes('PLANNED'), 'candidate diagram must distinguish planned work');
assert(candidatePerception.includes('현재 CC stub · SAM worker 없음'), 'candidate diagram must not imply that a SAM worker exists');
assert(candidatePerception.includes('pose(t_capture)'), 'candidate diagram must preserve capture-time pose semantics');
assert(candidatePerception.includes('다중가설'), 'candidate diagram must show the planned track-bank boundary');
assert(candidatePerception.includes('Semantic 변화 → safety output 동일 gate'), 'candidate diagram must show the safety-independence gate');
const finalPerception = fs.readFileSync(
  path.join(repo, 'docs/assets/motar-perception-final.svg'),
  'utf8',
);
for (const contract of [
  'FINAL PERCEPTION PATH', 'NPS-Drones', 'Det-Fly', 'K = 5', 'CNN + KF',
  'GRU · T=8', 'Transformer · T=16', 'Inject measured errors', 'PPO retraining',
]) assert(finalPerception.includes(contract), `final perception diagram missing: ${contract}`);
assert(html.includes('perception_final_implementation_plan_2026-09-07.md'));
assert(readme.includes('perception_final_implementation_plan_2026-09-07.md'));
const currentOverview = fs.readFileSync(
  path.join(repo, 'docs/assets/motar-system-overview.svg'),
  'utf8',
);
assert(currentOverview.includes('EVIDENCE MAP'), 'overview must distinguish the three evidence tracks');
assert(currentOverview.includes('TRACK C / PUBLIC REAL FOOTAGE'), 'overview must include the measured range study');
assert(currentOverview.includes('알려진 실패 기준선'), 'legacy single-KF path must not look like the new main path');
for (const page of [readme, html]) {
  assert(page.includes('presentation/motar-presentation-2026-09-10.zip'), 'presentation download must be linked');
}
assert(html.includes('설계 후보'), 'site must say the SAM diagram is a candidate, not adopted');
assert(readme.includes('설계 후보'), 'README must say the SAM diagram is a candidate, not adopted');
for (const page of [html, readme]) {
  assert(page.includes('SAM3_PERCEPTION_VERIFICATION_PLAN_2026-09-03.md'), 'SAM verification plan must be linked');
  assert(page.includes('offline CPU'), 'implemented candidate scope must stay explicit');
}

// Presentation claims remain explicitly bounded by the current evidence and hardware status.
assert(html.includes('Non-overlap route-off PPO · held-out complete to 145 bars'));
assert(html.includes('seed-911 stopped at epoch 21,973'));
assert(html.includes('seed-313 capture 83.70% @70 → 65.54% @145'));
assert(html.includes('routed mechanism still FAIL'));
assert(html.includes('<strong>32/32</strong><span>corrected route-gate integrity</span>'));
assert(html.includes('<strong>17.78%</strong><span>corrected 70-bar plan success</span>'));
assert(html.includes('<strong>30.02%</strong><span>corrected 70-bar fallback</span>'));
assert(html.includes('99.167%'));
assert(html.includes('300 bars FAIL 94.661%'));
assert(html.includes('route-off measured 70→145 · 205 not reached'));
assert(html.includes('preregistration_navrl_v2_corrected_density_geometry_2026-08-27.md'));
assert(html.includes('corrected_nonoverlap_route_gate_r2_result_2026-08-31.md'));
assert(html.includes('preregistration_corrected_nonoverlap_route_gate_r2_2026-08-31.md'));
assert(html.includes('navrl_corrected_nonoverlap_physical_off_heldout_seed313/summary.md'));
assert(html.includes('preregistration_corrected_nonoverlap_physical_off_heldout_eval_2026-09-02.md'));
assert(html.includes('SYNTHETIC_ONLY'));
assert(html.includes('0.21875'));
assert(html.includes('PASS_LEARNING_VIABILITY'));
assert(html.includes('160/205 미도달'));
assert(html.includes('Wilson 95% [82.04, 85.24]'));
assert(html.includes('Wilson 95% [63.46, 67.57]'));
assert(html.includes('90.27%'));
assert(html.includes('0.826→0.896→0.892'));
assert(html.includes('frozen ep25000 stopcap screen의 결과일 뿐'));
assert(html.includes('baseline_1p25'));
assert(html.includes('canonical 1.5 m/s'));
assert(html.includes('RANGE_INCONCLUSIVE'));
assert(html.includes('NOT PhysX/PPO'));
assert(html.includes('실제 기체가 미조립'));
assert(!html.includes('PPO 0 epochs'));
assert(!html.includes('corrected fresh PPO epochs'));
assert(fs.readFileSync(path.join(repo, 'docs/assets/motar-control-stack.svg'), 'utf8').includes('K_v e_v'));
assert(html.includes('0.04 s 1차 지연'));

const recoveryV2Gate = experiments.find((entry) => entry.id === '2026-08-26-physical-target-recovery-v2-lower1p25-gate-seed827');
assert(recoveryV2Gate, 'canonical recovery-v2 lower-1.25 gate entry missing');
assert.strictEqual(recoveryV2Gate.verdict, 'FAIL');
assert.strictEqual(recoveryV2Gate.validity, 'canonical');
assert.strictEqual(recoveryV2Gate.env.contract_variant, 'baseline_1p25');
assert.strictEqual(recoveryV2Gate.arms[0].extra.cells_passed, '7/32, all route-off');
assert(recoveryV2Gate.arms[1].extra.plan_success.includes('190/203'));
assert(recoveryV2Gate.arms[1].extra.fallback.includes('18381/38400'));
assert(recoveryV2Gate.verdict_note.includes('canonical 1.5'));
assert(recoveryV2Gate.verdict_note.includes('hardware'));

const noAnchor = experiments.find((entry) => entry.id === '2026-08-26-recovery-v2-no-connector-forensics-seed827');
assert(noAnchor, 'canonical recovery-v2 no-anchor entry missing');
assert.strictEqual(noAnchor.verdict, 'INCONCLUSIVE');
assert.strictEqual(noAnchor.validity, 'canonical');
assert.strictEqual(noAnchor.arms[0].extra.primary_n, 1);
assert.strictEqual(noAnchor.arms[0].extra.identity_void, false);
assert.strictEqual(noAnchor.arms[1].extra.failed_certificate, 49);
assert.strictEqual(noAnchor.arms[1].extra.brake_timeout, 32);
assert.strictEqual(noAnchor.arms[1].extra.failed_resume, 23);
assert.strictEqual(noAnchor.arms[1].extra.connect_timeout, 1);
assert.strictEqual(noAnchor.arms[1].extra.brake_no_anchor, 1);

// Attempt 2 and RECOVERY_DOMINANT remain inspectable historical lineage.
const routedGate = experiments.find((entry) => entry.id === '2026-08-25-physical-target-routed-simulator-gate-seed827-attempt2');
assert(routedGate, 'canonical attempt-2 routed gate entry missing');
assert.strictEqual(routedGate.verdict, 'FAIL');
assert.strictEqual(routedGate.validity, 'canonical');
assert.strictEqual(routedGate.lineage_status, 'historical_attempt2');
assert.strictEqual(routedGate.arms[0].extra.cells_passed, '32/32');
assert(routedGate.verdict_note.includes('unsafe_start'));
const recovery = status.sim2real_72h.simulation_verification.preflight_steps.route_recovery_forensics;
assert(recovery, 'route recovery forensics status missing');
assert.strictEqual(recovery.diagnostic_verdict, 'RECOVERY_DOMINANT');
assert.strictEqual(recovery.lineage_status, 'HISTORICAL_V1_DIAGNOSTIC');
assert.strictEqual(recovery.cells_verified, '8/8');
assert.strictEqual(recovery.local_invalidations, 358);
assert.strictEqual(recovery.local_fallback_intervals, 35666);
assert.strictEqual(recovery.unique_local_origins, 200);
assert.strictEqual(recovery.rounded_vs_square_disagreements, 1832);
assert.strictEqual(recovery.margin_tuning_allowed, false);
const recoveryExperiment = experiments.find((entry) => entry.id === '2026-08-25-physical-target-route-recovery-forensics-seed827');
assert(recoveryExperiment, 'recovery forensics experiment entry missing');
assert.strictEqual(recoveryExperiment.diagnostic_verdict, 'RECOVERY_DOMINANT');
assert.strictEqual(recoveryExperiment.lineage_status, 'historical_v1_diagnostic');
assert(recoveryExperiment.results_paths.includes('results/navrl_physical_target_route_recovery_forensics_seed827/receipt.json'));

assert.strictEqual(status.sim2real_72h.as_of, '2026-08-26');
const currentGate = status.sim2real_72h.simulation_verification.recovery_v2_lower1p25_gate;
// Missing/invalid source evidence is a valid fail-closed public state, never a PASS.
// The Python snapshot tests exercise the canonical reader and malformed-receipt rejection.
function assertUnavailableEvidence(block, source) {
  assert.deepStrictEqual(block, {
    source,
    status: 'RESULT_UNAVAILABLE_OR_MALFORMED',
    authority: 'NO_FURTHER_TRACK_B_GPU_PPO_RETUNE_RERUN',
    physical_ppo: 'BLOCKED',
    hardware_claim: false,
  });
}
const unavailableSource = 'results/example/summary.json';
const unavailableFixture = {
  source: unavailableSource,
  status: 'RESULT_UNAVAILABLE_OR_MALFORMED',
  authority: 'NO_FURTHER_TRACK_B_GPU_PPO_RETUNE_RERUN',
  physical_ppo: 'BLOCKED',
  hardware_claim: false,
};
assertUnavailableEvidence(unavailableFixture, unavailableSource);
for (const patch of [{integrity: 'PASS_32_CELL_INTEGRITY'}, {hardware_claim: true},
                     {physical_ppo: 'ENABLED'}, {status: 'PASS'}, {authority: undefined}]) {
  assert.throws(() => assertUnavailableEvidence({...unavailableFixture, ...patch}, unavailableSource));
}
if (currentGate.status === 'RESULT_UNAVAILABLE_OR_MALFORMED') {
  assertUnavailableEvidence(currentGate,
    'results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/summary.json');
} else {
  assert.strictEqual(currentGate.status, 'VERIFIED_FAIL');
  assert.strictEqual(currentGate.integrity, 'PASS_32_CELL_INTEGRITY');
  assert.strictEqual(currentGate.route_mechanism, 'FAIL_ROUTE_MECHANISM');
  assert.deepStrictEqual(currentGate.cells, {
    passed: 7,
    total: 32,
    route_off_passed: 7,
    route_off_total: 16,
    recovery_passed: 0,
    recovery_total: 16,
    passing_lineage: 'route_off_only',
  });
  assert.strictEqual(currentGate.plan_success_70bar_4speed.numerator, 190);
  assert.strictEqual(currentGate.plan_success_70bar_4speed.denominator, 203);
  assert.strictEqual(currentGate.fallback_70bar_4speed.numerator, 18381);
  assert.strictEqual(currentGate.fallback_70bar_4speed.denominator, 38400);
  assert.strictEqual(currentGate.goals_per_env_70bar_0_6mps.value, 0.21875);
  assert.strictEqual(currentGate.no_connector_occupancy.numerator, 96854);
  assert.strictEqual(currentGate.no_connector_occupancy.denominator, 153600);
  assert.strictEqual(currentGate.hard_breach_no_connector_entries.numerator, 0);
  assert.strictEqual(currentGate.hard_breach_no_connector_entries.denominator, 534);
  assert.strictEqual(currentGate.hardware_claim, false);
  assert.strictEqual(currentGate.canonical_1p5_contract, 'SEPARATE_UNCHANGED_NOT_PASSED');
}
assert.deepStrictEqual(
  status.sim2real_72h.simulation_verification.preflight_steps.physical_target_gate,
  currentGate,
);
const currentForensics = status.sim2real_72h.simulation_verification.recovery_v2_no_connector_forensics;
if (currentForensics.status === 'RESULT_UNAVAILABLE_OR_MALFORMED') {
  assertUnavailableEvidence(currentForensics,
    'results/navrl_physical_target_recovery_v2_no_connector_forensics_seed827/summary.json');
} else {
  assert.strictEqual(currentForensics.status, 'DESCRIPTIVE_ONLY');
  assert.strictEqual(currentForensics.decision_rule.label, 'INCONCLUSIVE');
  assert.strictEqual(currentForensics.decision_rule.primary_n, 1);
  assert.strictEqual(currentForensics.decision_rule.anchor_present, 0);
  assert.strictEqual(currentForensics.decision_rule.hard_free_soft_unsafe, 1);
  assert.strictEqual(currentForensics.decision_rule.identity_void, false);
  assert.strictEqual(currentForensics.no_connector_classes.total, 106);
}
assert.strictEqual(
  status.sim2real_72h.simulation_verification.track_b_authority,
  'CLOSED_NO_FURTHER_GPU_PPO_RETUNE_RERUN',
);
assert.deepStrictEqual(
  status.sim2real_72h.simulation_verification.routed_physical_target_gate_attempt2.highest_passing_speed_mps_by_density,
  {'70': null, '150': null, '205': null, '300': null},
);
assert.strictEqual(
  status.sim2real_72h.simulation_verification.historical_post_wall_brake_speed_envelope.route_mode,
  'off_historical_lineage',
);
if (currentGate.status === 'RESULT_UNAVAILABLE_OR_MALFORMED'
    || currentForensics.status === 'RESULT_UNAVAILABLE_OR_MALFORMED') {
  assert(status.sim2real_72h.status.includes('TRACK B EVIDENCE UNAVAILABLE/MALFORMED'));
  assert(status.sim2real_72h.status.includes('NO TRACK B AUTHORITY'));
} else {
  assert(status.sim2real_72h.status.includes('NO FURTHER TRACK B AUTHORITY'));
}
assert(status.sim2real_72h.status.includes('HARDWARE NEXT'));

// The concise platform card must remain tied to the generated source-of-truth values.
const ref = platform.robots.find((robot) => robot.key === 'navrl_ref5in_quad');
assert(ref);
assert.strictEqual(ref.mass_kg, 1.2);
assert.strictEqual(ref.derived.motor_diagonal_m, 0.22);
assert.deepStrictEqual(ref.collision_box_m, [0.28, 0.28, 0.12]);
const refV2 = platform.robots.find((robot) => robot.key === 'navrl_ref5in_v2_quad');
assert(refV2);
assert.deepStrictEqual(refV2.collision_box_m, [0.283, 0.283, 0.12]);
assert(html.includes('1.20 kg'));
assert(html.includes('220 mm'));
assert(html.includes('0.283 × 0.283 × 0.12 m'));

// Every local href and image source resolves from the static page.
const refs = [];
const refPattern = /(?:href|src)="([^"]+)"/g;
let match;
while ((match = refPattern.exec(html)) !== null) refs.push(match[1]);
for (const refPath of refs) {
  if (refPath.startsWith('#') || /^https?:/.test(refPath)) continue;
  const clean = refPath.split('#')[0].split('?')[0];
  let target = path.resolve(site, clean);
  assert(fs.existsSync(target), `broken local reference: ${refPath}`);
  if (fs.statSync(target).isDirectory()) {
    // Git tracks files, not directories. A public gallery link requires a tracked index.
    target = path.join(target, 'index.html');
    assert(fs.existsSync(target) && fs.statSync(target).isFile(),
      `local directory reference has no index.html: ${refPath}`);
  }
  const relative = path.relative(repo, target);
  assert(
    !relative.startsWith('..') &&
      !path.isAbsolute(relative) &&
      (trackedFiles.has(relative) || pendingDocs.has(relative)),
    `local reference is not tracked or a pending docs file: ${refPath}`,
  );
}

console.log('MOTAR static site contract: PASS');
