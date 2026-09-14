#!/usr/bin/env bash
# Does a constant association gate let the honest covariance pay off?
#
# H3 made the LiDAR correction stop shrinking covariance it never observed, and gained only
# 1.46 pp under dropout (11.5% of the loss, 1.1 sigma) against H2's +3.42 pp in the same run.
# Part of the reason is measurable self-cancellation: the association gate is
# (0.35 + 2*pos_sigma) clamped at 1.0 m, so it widens exactly as the covariance becomes honest.
#
#   blind steps   frozen covariance -> gate   honest covariance -> gate
#       2             0.164 m -> 0.679 m          0.229 m -> 0.808 m
#       5             0.169 m -> 0.688 m          0.492 m -> 1.000 m (cap)
#      20             0.158 m -> 0.666 m          2.050 m -> 1.000 m (cap)
#
# Telling the truth about uncertainty bought a 50% wider mis-association window. A gate should
# express measurement precision, not how lost the track is, so this run pins it to a constant.
#
#   arm                              dropout  range-only  gate
#   analytic_clean                      0        off      auto   (80.54 / 17.17 재현)
#   dropout_0p3_raw                   0.3        off      auto   (67.84 / 29.33 재현)
#   dropout_0p3_ro_gate065            0.3         ON     0.65 m  ← 기존 실효 게이트 재현
#   dropout_0p3_ro_gate035            0.3         ON     0.35 m  ← 좁힌 게이트
#   dropout_0p3_no_assoc              0.3        off      auto   (H2 기준선, 71.25% 재현)
#   clean_ro_gate035                    0         ON     0.35 m  (무해성)
#
# Two gate values, because a gate tight enough stops firing at all and degenerates into H2 --
# a monotone trend across 0.65 / 0.35 / off distinguishes "tighter association is better" from
# "any association is bad".
#
# Usage:
#   ./eval_navrl_v2_lidar_assoc_gate.sh
#   PREFLIGHT=1 ./eval_navrl_v2_lidar_assoc_gate.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON

POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
# Seed 47 is the frozen contract; a second seed decides adoption. The backfill arm was rejected
# because its clean regression replicated across seeds, so this one faces the same bar.
SEED="${NAVRL_GATE_SEED:-47}"
RESULT_ROOT="../../../results/navrl_v2_lidar_assoc_gate"
if [[ "${SEED}" != "47" ]]; then
    RESULT_ROOT="${RESULT_ROOT}_seed${SEED}"
fi
PREFLIGHT="${PREFLIGHT:-0}"
EPISODES="${EPISODES:-2049}"

if [[ ! -f "${POLICY}" ]]; then
    echo "[gate] policy checkpoint missing: ${POLICY}" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" && "${PREFLIGHT}" != "1" ]]; then
    echo "[gate] refusing to overwrite ${RESULT_ROOT}" >&2
    exit 2
fi

ACTUAL_POLICY_SHA="$(
    "${PYTHON}" - "${POLICY}" "${POLICY_SHA}" <<'PY'
import hashlib
from pathlib import Path
import re
import sys
import torch

trained, expected = Path(sys.argv[1]), sys.argv[2]
actual = hashlib.sha256(trained.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"[gate] policy SHA mismatch: {actual}")
payload = torch.load(trained, map_location="cpu", weights_only=False)
state = payload.get("env_state") or {}
checks = {
    "filename": re.fullmatch(r"last_gen_ppo_ep_25000_rew_-?[0-9]+(?:\.[0-9]+)?\.pth", trained.name) is not None,
    "epoch": int(payload.get("epoch", -1)) == 25000,
    "bars": int(state.get("n_bars_active", -1)) == 205,
    "selector": state.get("cfg_obstacle_selector") == "cluster_sector",
    "governor": state.get("cfg_speed_governor_mode") == "riskcap",
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("[gate] invalid policy checkpoint: " + ", ".join(failed))
print(actual)
PY
)"

# Frozen evaluation contract -- byte-identical to the R3 and latency harnesses.
export NAVRL_V2_DENSITIES=205
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_SPEED_GOVERNOR=riskcap
export NAVRL_SEED="${SEED}"
export NAVRL_SPEED_GOVERNOR_FIXED_MPS=2.0
export NAVRL_SPEED_GOVERNOR_FREE_MPS=3.53553390593
export NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M=0.45
export NAVRL_SPEED_GOVERNOR_MARGIN_M=0.45
export NAVRL_SPEED_GOVERNOR_SLOW_M=3.0
export NAVRL_SPEED_GOVERNOR_RELEASE_M=5.0
export NAVRL_SPEED_GOVERNOR_TTC_S=1.2
export NAVRL_SPEED_GOVERNOR_BRAKE_MPS2=2.9608856678
export NAVRL_SPEED_GOVERNOR_REACTION_S=0.1
unset NAVRL_DETECTOR_CHECKPOINT  # bootstrap segmenter, same as the R3 analytic arms

run_cell() {
    local tag="$1"
    local perturb="$2"
    local dropout="$3"
    local assoc="$4"
    local range_only="$5"
    local gate="$6"
    echo "[gate] cell=${tag} perturb=${perturb} dropout=${dropout} assoc=${assoc} range_only=${range_only} gate=${gate}"
    if [[ "${PREFLIGHT}" == "1" ]]; then
        return 0
    fi
    NAVRL_PERCEPTION_PERTURB="${perturb}" \
    NAVRL_DETECTION_DROPOUT="${dropout}" \
    NAVRL_DETECTION_LATENCY_S=0 \
    NAVRL_RANGE_ERROR_M=0 \
    NAVRL_TARGET_MASK_BACKFILL=0 \
    NAVRL_LIDAR_TARGET_ASSOC="${assoc}" \
    NAVRL_LIDAR_RANGE_ONLY_UPDATE="${range_only}" \
    NAVRL_LIDAR_ASSOC_GATE_M="${gate}" \
    NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
        ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
}

run_cell analytic_clean         0 0   1 0 0
run_cell dropout_0p3_raw        1 0.3 1 0 0
run_cell dropout_0p3_ro_gate065 1 0.3 1 1 0.65
run_cell dropout_0p3_ro_gate035 1 0.3 1 1 0.35
run_cell dropout_0p3_no_assoc   1 0.3 0 0 0
run_cell clean_ro_gate035       0 0   1 1 0.35

if [[ "${PREFLIGHT}" == "1" ]]; then
    echo "[gate] PREFLIGHT PASS (no results written)"
    exit 0
fi

"${PYTHON}" - "${RESULT_ROOT}" "${ACTUAL_POLICY_SHA}" <<'PY'
import json
import os
from pathlib import Path
import sys

root, policy_sha = Path(sys.argv[1]), sys.argv[2]


def load(cell_dir):
    payload = json.loads((cell_dir / "205bars.json").read_text(encoding="utf-8"))
    out, causes = payload["outcome"], payload["crash_causes"]
    return {
        "episodes": payload["actual_episodes"],
        "capture": out["capture_rate"],
        "crash": out["crash_rate"],
        "timeout": out["timeout_rate"],
        "bar_contact": causes["bar_contact"],
        "out_of_bounds": causes["out_of_bounds"],
        "closest_nocrash_mean_m": out["closest_nocrash_mean_m"],
        "target_visible": payload["action"]["context"]["target_visible"]["fraction"],
    }


cells = {d.name: load(d) for d in sorted(p for p in root.iterdir() if p.is_dir())}
clean, raw, fixed = cells["analytic_clean"], cells["dropout_0p3_raw"], cells["dropout_0p3_ro_gate035"]
span = clean["capture"] - raw["capture"]
excess_bars = raw["bar_contact"] - clean["bar_contact"]
verdict = {
    "capture_delta_vs_raw_pp": (fixed["capture"] - raw["capture"]) * 100.0,
    "recovered_fraction": (fixed["capture"] - raw["capture"]) / span if span else None,
    "bar_contact_delta_vs_raw": fixed["bar_contact"] - raw["bar_contact"],
    "bar_contact_recovered_fraction": (
        (raw["bar_contact"] - fixed["bar_contact"]) / excess_bars if excess_bars else None
    ),
    # The fix must not cost anything when the camera never misses a frame.
    "clean_regression_pp": (cells["clean_ro_gate035"]["capture"] - clean["capture"]) * 100.0,
}
# ~2050 episodes puts the inter-arm standard error near 1.3 pp, so anything smaller is noise.
verdict["hypothesis"] = (
    "SUPPORTED" if verdict["capture_delta_vs_raw_pp"] >= 4.0
    else ("REJECTED" if verdict["capture_delta_vs_raw_pp"] <= 1.3 else "INCONCLUSIVE")
)

(root / "summary.json").write_text(
    json.dumps({"policy_sha256": policy_sha,
                "contract": f"seed{os.environ['NAVRL_SEED']}/205bars/deterministic/riskcap",
                "cells": cells, "verdict": verdict}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
lines = ["# LiDAR association gate A/B under detection dropout (ep25000+riskcap, seed %s, 205 bars)" % os.environ["NAVRL_SEED"],
         "",
         "| cell | episodes | capture | crash | bar contacts | OOB | fused visible | closest mean (m) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
for tag, c in cells.items():
    lines.append(f"| {tag} | {c['episodes']} | {c['capture']*100:.2f}% | {c['crash']*100:.2f}% | "
                 f"{c['bar_contact']} | {c['out_of_bounds']} | {c['target_visible']*100:.2f}% | "
                 f"{c['closest_nocrash_mean_m']:.3f} |")
recovered = ("n/a" if verdict["recovered_fraction"] is None
             else f"{verdict['recovered_fraction']*100:.1f}%")
bars = ("n/a" if verdict["bar_contact_recovered_fraction"] is None
        else f"{verdict['bar_contact_recovered_fraction']*100:.1f}%")
lines += ["", f"## verdict: {verdict['hypothesis']}", "",
          f"- capture vs dropout_raw: {verdict['capture_delta_vs_raw_pp']:+.2f} pp "
          f"= {recovered} of the dropout loss",
          f"- bar contacts vs dropout_raw: {verdict['bar_contact_delta_vs_raw']:+d} "
          f"= {bars} of the excess removed",
          f"- clean regression from the fix: {verdict['clean_regression_pp']:+.2f} pp",
          "",
          "Inter-arm SE is about 1.3 pp at this episode count, so SUPPORTED needs >= 4 pp."]
(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
echo "[gate] done -> ${RESULT_ROOT}/summary.{md,json}"
