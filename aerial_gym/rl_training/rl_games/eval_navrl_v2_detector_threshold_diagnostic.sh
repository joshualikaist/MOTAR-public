#!/usr/bin/env bash
# Independent audit follow-up for verification 2.
#
# The original nominal comparison changed both detector implementation and threshold (analytic
# at 0.55 versus learned-v7 at 0.70). This fresh-seed three-arm diagnostic separates them:
# analytic@0.55, learned-v7@0.55, learned-v7@0.70. It is diagnostic only; no threshold is selected
# from these results and no arm may be retried.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON
POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
DETECTOR="../../../artifacts/navrl_target_detector_v7_confirmatory.pth"
DETECTOR_SHA="85c7974bcd85c627170c5bd63030144d1c5dc2a11e5d64829cad38f615c5d5d7"
RESULT_ROOT="../../../results/navrl_v2_detector_threshold_diagnostic_seed191_193"
EPISODES="${EPISODES:-2049}"
PREFLIGHT="${PREFLIGHT:-0}"
SEEDS=(191 193)
RESUME=0
if [[ "${1:-}" == "--resume" ]]; then
    RESUME=1
    shift
fi
if (( $# != 0 )); then
    echo "usage: $0 [--resume]" >&2
    exit 2
fi

if [[ ! -f "${POLICY}" || ! -f "${DETECTOR}" ]]; then
    echo "[threshold-diagnostic] pinned policy or detector is missing" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" && "${PREFLIGHT}" != "1" && "${RESUME}" != "1" ]]; then
    echo "[threshold-diagnostic] refusing to overwrite ${RESULT_ROOT}" >&2
    exit 2
fi

"${PYTHON}" - "${POLICY}" "${POLICY_SHA}" "${DETECTOR}" "${DETECTOR_SHA}" <<'PY'
import hashlib
from pathlib import Path
import sys
for raw, expected in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    actual = hashlib.sha256(Path(raw).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"[threshold-diagnostic] SHA mismatch for {raw}: {actual}")
PY

export NAVRL_V2_DENSITIES=205
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_SPEED_GOVERNOR=riskcap
export NAVRL_SPEED_GOVERNOR_FIXED_MPS=2.0
export NAVRL_SPEED_GOVERNOR_FREE_MPS=3.53553390593
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M=0.45
export NAVRL_SPEED_GOVERNOR_MARGIN_M=0.45
export NAVRL_SPEED_GOVERNOR_SLOW_M=3.0
export NAVRL_SPEED_GOVERNOR_RELEASE_M=5.0
export NAVRL_SPEED_GOVERNOR_TTC_S=1.2
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2=2.9608856678
export NAVRL_SPEED_GOVERNOR_REACTION_S=0.1
export NAVRL_V2_SHARED_SOURCE_BUNDLE="${RESULT_ROOT}/source_bundle"

run_cell() {
    local seed="$1" arm="$2" threshold="$3"
    local out="${RESULT_ROOT}/seed${seed}_${arm}"
    echo "[threshold-diagnostic] seed=${seed} arm=${arm} threshold=${threshold}"
    if [[ "${PREFLIGHT}" == "1" ]]; then return 0; fi
    if [[ -f "${out}/205bars.json" && -f "${out}/205bars.receipt.json" ]]; then
        echo "[threshold-diagnostic] SKIP complete ${out}"
        return 0
    fi
    if [[ -e "${out}" ]]; then
        echo "[threshold-diagnostic] refusing partial cell ${out}; inspect/move it manually" >&2
        exit 2
    fi
    if [[ "${arm}" == "analytic_t055" ]]; then
        env -u NAVRL_DETECTOR_CHECKPOINT -u NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH \
            NAVRL_DETECTOR_THRESHOLD=0.55 NAVRL_SEED="${seed}" NAVRL_V2_RESULT_DIR="${out}" \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
    elif [[ "${arm}" == "learned_t055" ]]; then
        env -u NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH \
            NAVRL_DETECTOR_CHECKPOINT="${DETECTOR}" NAVRL_DETECTOR_THRESHOLD=0.55 \
            NAVRL_SEED="${seed}" NAVRL_V2_RESULT_DIR="${out}" \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
    else
        NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH=1 \
        NAVRL_DETECTOR_CHECKPOINT="${DETECTOR}" NAVRL_DETECTOR_THRESHOLD="${threshold}" \
        NAVRL_SEED="${seed}" NAVRL_V2_RESULT_DIR="${out}" \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
    fi
}

for seed in "${SEEDS[@]}"; do
    run_cell "${seed}" analytic_t055 0.55
    run_cell "${seed}" learned_t055 0.55
    run_cell "${seed}" learned_t070 0.70
done

if [[ "${PREFLIGHT}" == "1" ]]; then
    echo "[threshold-diagnostic] PREFLIGHT PASS | 6 cells, seeds=191,193"
    exit 0
fi

"${PYTHON}" - "${RESULT_ROOT}" "${POLICY_SHA}" "${DETECTOR_SHA}" <<'PY'
import json, math, sys
from pathlib import Path
root, policy_sha, detector_sha = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
seeds = (191, 193)
arms = ("analytic_t055", "learned_t055", "learned_t070")
cells = {}
for seed in seeds:
    for arm in arms:
        p = json.loads((root / f"seed{seed}_{arm}" / "205bars.json").read_text())
        o = p["outcome"]
        cells[f"seed{seed}_{arm}"] = dict(episodes=p["actual_episodes"], captured=o["captured"],
            capture=o["capture_rate"], crash=o["crash_rate"], timeout=o["timeout_rate"])
def pool(arm):
    cs = [cells[f"seed{s}_{arm}"] for s in seeds]
    return sum(c["captured"] for c in cs), sum(c["episodes"] for c in cs)
def diff(a, b):
    ka, na = pool(a); kb, nb = pool(b); pa, pb = ka/na, kb/nb
    d = pa-pb; se = math.sqrt(pa*(1-pa)/na + pb*(1-pb)/nb)
    return dict(diff_pp=100*d, ci95_pp=[100*(d-1.96*se), 100*(d+1.96*se)])
pooled = {a: pool(a)[0]/pool(a)[1] for a in arms}
contrasts = {
    "D1_model_at_matched_055": diff("learned_t055", "analytic_t055"),
    "D2_original_combined_070": diff("learned_t070", "analytic_t055"),
    "D3_threshold_within_v7_070_minus_055": diff("learned_t070", "learned_t055"),
}
summary = dict(campaign="detector_threshold_diagnostic", policy_sha256=policy_sha,
    detector_sha256=detector_sha, seeds=list(seeds), cells=cells, pooled=pooled,
    contrasts=contrasts, interpretation_contract=(
        "diagnostic only: D1 isolates detector statistics at matched threshold; D3 isolates the "
        "runtime threshold within v7; no threshold selection or adoption from these cells"))
(root/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
lines = ["# Detector threshold diagnostic (fresh seeds 191/193)", "",
         "| arm | pooled capture |", "|---|---:|"]
for arm in arms: lines.append(f"| {arm} | {100*pooled[arm]:.2f}% |")
lines.append("")
for name, c in contrasts.items():
    lines.append(f"- {name}: {c['diff_pp']:+.3f} pp "
                 f"[{c['ci95_pp'][0]:+.3f}, {c['ci95_pp'][1]:+.3f}]")
lines += ["", summary["interpretation_contract"]]
(root/"summary.md").write_text("\n".join(lines)+"\n")
print("\n".join(lines))
PY

echo "[threshold-diagnostic] done -> ${RESULT_ROOT}/summary.{md,json}"
