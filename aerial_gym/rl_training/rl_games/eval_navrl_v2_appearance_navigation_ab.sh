#!/usr/bin/env bash
# 검증 2 final stage: does navigation survive the appearance envelope?
#
# Preregistered in WORKLOG 2026-08-12 BEFORE any of this campaign's data existed. 2x2 factorial
# x 2 fresh seeds = 8 cells on the frozen ep25000+riskcap policy, 205 bars, deterministic,
# exact-600, ~2049 episodes/cell:
#
#   appearance {nominal, envelope(hue60/light0.5/albedo0.3/texture0.2/blur0.3)}
#   detector   {analytic_bootstrap(thr 0.55), learned_v7(SHA-pinned, thr 0.70)}
#   seeds      {151, 157}   (unused anywhere else in the project)
#
# Endpoints (fixed before data):
#   E1 primary  NI gate: pooled capture(nominal,learned_v7) - capture(nominal,analytic);
#               PASS iff independent-binomial 95% CI lower bound > -2.0 pp (Gate 3 protocol).
#   E2 headline descriptive: capture(envelope,learned_v7) - capture(nominal,analytic) -- the
#               navigation cost of the WHOLE appearance shift with the robust detector. No gate.
#   E3 counterfactual descriptive: capture(envelope,analytic) -- how far the pure-red bootstrap
#               falls when the world stops being pure red.
# No retry against this campaign's cells.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON

POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
DETECTOR="../../../artifacts/navrl_target_detector_v7_confirmatory.pth"
RESULT_ROOT="../../../results/navrl_v2_appearance_navigation_ab_seed151_157"
PREFLIGHT="${PREFLIGHT:-0}"
EPISODES="${EPISODES:-2049}"
SEEDS=(151 157)
LEARNED_THRESHOLD="0.70"

if [[ ! -f "${POLICY}" ]]; then
    echo "[app-ab] policy checkpoint missing: ${POLICY}" >&2
    exit 2
fi
if [[ ! -f "${DETECTOR}" ]]; then
    echo "[app-ab] detector artifact missing: ${DETECTOR}" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" && "${PREFLIGHT}" != "1" ]]; then
    echo "[app-ab] refusing to overwrite ${RESULT_ROOT}" >&2
    exit 2
fi

read -r ACTUAL_POLICY_SHA ACTUAL_DETECTOR_SHA <<< "$(
    "${PYTHON}" - "${POLICY}" "${POLICY_SHA}" "${DETECTOR}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

policy, expected, detector = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
actual = hashlib.sha256(policy.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"[app-ab] policy SHA mismatch: {actual}")
det_sha = hashlib.sha256(detector.read_bytes()).hexdigest()
# The artifact must be the confirmatory-PASS lineage: receipt-recorded gate_passed and the
# validation-selected threshold this launcher pins at runtime.
summary = json.loads(
    (Path("../../../results/navrl_detector_offline_gate_v7_confirmatory/summary.json"))
    .read_text(encoding="utf-8")
)
if summary["artifact_sha256"] != det_sha:
    raise SystemExit("[app-ab] detector SHA does not match the v7 confirmatory summary")
if not summary["gate_passed"]:
    raise SystemExit("[app-ab] refusing: v7 offline gate did not pass")
if abs(float(summary["selected_threshold"]) - 0.70) > 1e-9:
    raise SystemExit("[app-ab] selected threshold drifted from the pinned 0.70")
print(actual, det_sha)
PY
)"

# Frozen evaluation contract -- byte-identical governor block to every prior 205-bar campaign.
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

ENVELOPE=(NAVRL_APP_HUE_DEG=60 NAVRL_APP_LIGHT_GAIN=0.5 NAVRL_APP_ALBEDO_JITTER=0.3
          NAVRL_APP_TEXTURE_STD=0.2 NAVRL_APP_MOTION_BLUR=0.3)

run_cell() {
    local seed="$1"
    local appearance="$2"   # nominal | envelope
    local detector_arm="$3" # analytic | learned_v7
    local tag="seed${seed}_${appearance}_${detector_arm}"
    echo "[app-ab] cell=${tag}"
    if [[ "${PREFLIGHT}" == "1" ]]; then
        return 0
    fi
    local env_extra=()
    if [[ "${appearance}" == "envelope" ]]; then
        env_extra+=("${ENVELOPE[@]}")
    fi
    if [[ "${detector_arm}" == "learned_v7" ]]; then
        # The PPO checkpoint records the threshold it TRAINED with (0.55); running the learned
        # detector at its own validation-selected 0.70 is the point of this arm, so the
        # trained-vs-running threshold guard must be overridden HERE AND ONLY HERE. The narrow
        # override still refuses every other environmental contract mismatch.
        env_extra+=(NAVRL_DETECTOR_CHECKPOINT="${DETECTOR}"
                    NAVRL_DETECTOR_THRESHOLD="${LEARNED_THRESHOLD}"
                    NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH=1)
        env "${env_extra[@]}" \
            NAVRL_SEED="${seed}" \
            NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
    else
        env -u NAVRL_DETECTOR_CHECKPOINT "${env_extra[@]}" \
            NAVRL_SEED="${seed}" \
            NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
            ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
    fi
}

for seed in "${SEEDS[@]}"; do
    run_cell "${seed}" nominal analytic
    run_cell "${seed}" nominal learned_v7
    run_cell "${seed}" envelope analytic
    run_cell "${seed}" envelope learned_v7
done

if [[ "${PREFLIGHT}" == "1" ]]; then
    echo "[app-ab] PREFLIGHT PASS | 8 cells seeds=151,157 margin(E1)=-2pp"
    exit 0
fi

"${PYTHON}" - "${RESULT_ROOT}" "${ACTUAL_POLICY_SHA}" "${ACTUAL_DETECTOR_SHA}" <<'PY'
import json
import math
from pathlib import Path
import sys

root, policy_sha, detector_sha = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
SEEDS = (151, 157)
cells = {}
for seed in SEEDS:
    for appearance in ("nominal", "envelope"):
        for det in ("analytic", "learned_v7"):
            cell = root / f"seed{seed}_{appearance}_{det}"
            payload = json.loads((cell / "205bars.json").read_text(encoding="utf-8"))
            out = payload["outcome"]
            cells[(seed, appearance, det)] = {
                "episodes": payload["actual_episodes"],
                "captured": out["captured"],
                "capture": out["capture_rate"],
                "crash": out["crash_rate"],
                "timeout": out["timeout_rate"],
            }


def pooled(appearance, det):
    n = sum(cells[(s, appearance, det)]["episodes"] for s in SEEDS)
    k = sum(cells[(s, appearance, det)]["captured"] for s in SEEDS)
    return k, n


def diff_ci(a, b):
    ka, na = a
    kb, nb = b
    pa, pb = ka / na, kb / nb
    d = pa - pb
    se = math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    return d, (d - 1.96 * se, d + 1.96 * se)


nom_analytic = pooled("nominal", "analytic")
nom_learned = pooled("nominal", "learned_v7")
env_analytic = pooled("envelope", "analytic")
env_learned = pooled("envelope", "learned_v7")

e1, e1_ci = diff_ci(nom_learned, nom_analytic)
e2, e2_ci = diff_ci(env_learned, nom_analytic)
e3, e3_ci = diff_ci(env_analytic, nom_analytic)
e1_pass = e1_ci[0] * 100 > -2.0

summary = {
    "campaign": "appearance_navigation_ab",
    "policy_sha256": policy_sha,
    "detector_sha256": detector_sha,
    "seeds": list(SEEDS),
    "cells": {f"seed{s}_{a}_{d}": c for (s, a, d), c in sorted(cells.items())},
    "pooled": {
        "nominal_analytic": nom_analytic[0] / nom_analytic[1],
        "nominal_learned_v7": nom_learned[0] / nom_learned[1],
        "envelope_analytic": env_analytic[0] / env_analytic[1],
        "envelope_learned_v7": env_learned[0] / env_learned[1],
    },
    "E1_nominal_NI": {"diff_pp": e1 * 100, "ci95_pp": [e1_ci[0] * 100, e1_ci[1] * 100],
                       "margin_pp": -2.0, "pass": e1_pass},
    "E2_envelope_cost": {"diff_pp": e2 * 100, "ci95_pp": [e2_ci[0] * 100, e2_ci[1] * 100]},
    "E3_bootstrap_collapse": {"diff_pp": e3 * 100, "ci95_pp": [e3_ci[0] * 100, e3_ci[1] * 100]},
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
lines = ["# appearance-envelope navigation A/B (ep25000+riskcap, 205 bars, seeds 151/157)", "",
         "| cell | capture | crash | timeout | n |", "|---|---:|---:|---:|---:|"]
for (s, a, d), c in sorted(cells.items()):
    lines.append(f"| seed{s} {a} {d} | {c['capture']*100:.2f}% | {c['crash']*100:.2f}% | "
                 f"{c['timeout']*100:.2f}% | {c['episodes']} |")
lines += ["",
          f"- **E1 (NI, nominal)**: learned−analytic {e1*100:+.3f} pp, "
          f"CI [{e1_ci[0]*100:+.3f}, {e1_ci[1]*100:+.3f}] → "
          f"{'**PASS**' if e1_pass else '**FAIL**'} (margin −2.0 pp)",
          f"- **E2 (envelope cost)**: envelope+learned vs nominal+analytic {e2*100:+.3f} pp, "
          f"CI [{e2_ci[0]*100:+.3f}, {e2_ci[1]*100:+.3f}]",
          f"- **E3 (bootstrap collapse)**: envelope+analytic vs nominal+analytic {e3*100:+.3f} pp, "
          f"CI [{e3_ci[0]*100:+.3f}, {e3_ci[1]*100:+.3f}]"]
(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
echo "[app-ab] done -> ${RESULT_ROOT}/summary.{md,json}"
