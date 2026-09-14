#!/usr/bin/env bash
# Corrective follow-up for verification 3: position/yaw noise use an isolated, fixed RNG stream.
# Clock-offset cells from verification 3 remain valid because they never drew random numbers.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON
POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
RESULT_ROOT="../../../results/navrl_v2_pose_noise_rng_audit_seed181"
EPISODES="${EPISODES:-2049}"
PREFLIGHT="${PREFLIGHT:-0}"
POSE_NOISE_SEED=9181

if [[ ! -f "${POLICY}" ]]; then echo "[pose-rng-audit] policy missing" >&2; exit 2; fi
if [[ -e "${RESULT_ROOT}" && "${PREFLIGHT}" != "1" ]]; then
    echo "[pose-rng-audit] refusing to overwrite ${RESULT_ROOT}" >&2; exit 2
fi
"${PYTHON}" - "${POLICY}" "${POLICY_SHA}" <<'PY'
import hashlib, sys
from pathlib import Path
actual = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
if actual != sys.argv[2]: raise SystemExit(f"[pose-rng-audit] policy SHA mismatch: {actual}")
PY

export NAVRL_V2_DENSITIES=205 NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original NAVRL_SPEED_GOVERNOR=riskcap NAVRL_SEED=181
export NAVRL_SPEED_GOVERNOR_FIXED_MPS=2.0 NAVRL_SPEED_GOVERNOR_FREE_MPS=3.53553390593
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M=0.45 NAVRL_SPEED_GOVERNOR_MARGIN_M=0.45
export NAVRL_SPEED_GOVERNOR_SLOW_M=3.0 NAVRL_SPEED_GOVERNOR_RELEASE_M=5.0
export NAVRL_SPEED_GOVERNOR_TTC_S=1.2 NAVRL_SPEED_GOVERNOR_BRAKE_MPS2=2.9608856678
export NAVRL_SPEED_GOVERNOR_REACTION_S=0.1 NAVRL_POSE_NOISE_SEED="${POSE_NOISE_SEED}"
export NAVRL_V2_SHARED_SOURCE_BUNDLE="${RESULT_ROOT}/source_bundle"
unset NAVRL_DETECTOR_CHECKPOINT

run_cell() {
    local tag="$1" pos="$2" yaw="$3"
    echo "[pose-rng-audit] cell=${tag} pos=${pos}m yaw=${yaw}deg noise_seed=${POSE_NOISE_SEED}"
    if [[ "${PREFLIGHT}" == "1" ]]; then return 0; fi
    NAVRL_PERCEPTION_PERTURB=1 NAVRL_DETECTION_DROPOUT=0 NAVRL_RANGE_ERROR_M=0 \
    NAVRL_DETECTION_LATENCY_S=0.1 NAVRL_POSE_CLOCK_OFFSET_S=0 \
    NAVRL_POSE_NOISE_POS_M="${pos}" NAVRL_POSE_NOISE_YAW_DEG="${yaw}" \
    NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
        ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
}
run_cell exact 0 0
run_cell posn_0p01 0.01 0
run_cell posn_0p03 0.03 0
run_cell posn_0p10 0.10 0
run_cell yaw_0p5 0 0.5
run_cell yaw_2 0 2
run_cell yaw_5 0 5

if [[ "${PREFLIGHT}" == "1" ]]; then
    echo "[pose-rng-audit] PREFLIGHT PASS | 7 cells seed=181 noise_seed=9181"; exit 0
fi

"${PYTHON}" - "${RESULT_ROOT}" "${POLICY_SHA}" <<'PY'
import json, math, sys
from pathlib import Path
root, policy_sha = Path(sys.argv[1]), sys.argv[2]
order = ("exact", "posn_0p01", "posn_0p03", "posn_0p10", "yaw_0p5", "yaw_2", "yaw_5")
cells = {}
for tag in order:
    p = json.loads((root/tag/"205bars.json").read_text()); o=p["outcome"]
    cells[tag] = dict(episodes=p["actual_episodes"], captured=o["captured"],
        capture=o["capture_rate"], crash=o["crash_rate"], timeout=o["timeout_rate"])
anchor=cells["exact"]
def diff(c):
    d=c["capture"]-anchor["capture"]
    se=math.sqrt(c["capture"]*(1-c["capture"])/c["episodes"]
                 +anchor["capture"]*(1-anchor["capture"])/anchor["episodes"])
    return [100*d, 100*(d-1.96*se), 100*(d+1.96*se)]
summary=dict(campaign="pose_noise_rng_audit", policy_sha256=policy_sha, environment_seed=181,
    pose_noise_seed=9181, cells=cells,
    delta_vs_exact_pp={tag:diff(cells[tag]) for tag in order[1:]},
    contract="dedicated pose RNG; global simulator RNG is identical across arms")
(root/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
lines=["# Pose-noise isolated-RNG audit (seed 181)","",
       "| cell | capture | crash | delta vs exact (95% CI) |","|---|---:|---:|---:|"]
for tag in order:
    c=cells[tag]
    delta="—" if tag=="exact" else f"{diff(c)[0]:+.2f} pp [{diff(c)[1]:+.2f}, {diff(c)[2]:+.2f}]"
    lines.append(f"| {tag} | {100*c['capture']:.2f}% | {100*c['crash']:.2f}% | {delta} |")
(root/"summary.md").write_text("\n".join(lines)+"\n"); print("\n".join(lines))
PY
echo "[pose-rng-audit] done -> ${RESULT_ROOT}/summary.{md,json}"
