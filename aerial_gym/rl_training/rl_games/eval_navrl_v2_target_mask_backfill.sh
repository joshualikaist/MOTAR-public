#!/usr/bin/env bash
# Does the phantom-target obstacle explain the detection-dropout loss?
#
# The obstacle map is edited by two paths with DIFFERENT gates: the LiDAR target_like carve-out
# is gated on fused visibility (camera OR LiDAR), the depth blanking on the camera-only pixel
# mask. On any frame the camera missed while the LiDAR track survived, the LiDAR half deletes the
# target and the camera half reinstates it, so the target becomes a solid obstacle dead ahead and
# consumes one of the 8 obstacle tokens. Demonstrated in isolation by
# tests/test_navrl_latency_compensate.py::TargetMaskBackfill (map range ahead 4.00 m with the
# camera mask, 3.00 m without it, at identical visibility).
#
# This matters because dropout costs -12.70 pp of capture while leaving fused visibility
# unchanged (21.38% vs clean 21.21%) -- the loss is NOT "the drone sees the target less often",
# so a channel that fires exactly on camera-missed frames is the leading explanation.
#
#   arm                       dropout  backfill
#   analytic_clean               0        0      (80.54 / 17.17 재현)
#   dropout_0p3_raw            0.3        0      (67.84 / 29.33 재현)
#   dropout_0p3_backfill       0.3        1      ← 가설 검정
#   clean_backfill               0        1      (수정이 clean을 해치지 않는지)
#
# Read: if the phantom obstacle is the channel, dropout_0p3_backfill recovers a large share of
# the 12.70 pp and bar contacts fall from 559 toward clean's 337. If it moves neither, the
# channel is elsewhere and the hypothesis is rejected -- which is just as useful.
#
# Usage:
#   ./eval_navrl_v2_target_mask_backfill.sh
#   PREFLIGHT=1 ./eval_navrl_v2_target_mask_backfill.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON

POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
# Seed 47 is the frozen R3/latency contract. A second seed re-measures the two marginal effects
# this arm produced (+2.54 pp under dropout, -1.42 pp on clean) against fresh episodes; the
# result root is suffixed so the two never overwrite each other.
SEED="${NAVRL_MASKBF_SEED:-47}"
RESULT_ROOT="../../../results/navrl_v2_target_mask_backfill"
if [[ "${SEED}" != "47" ]]; then
    RESULT_ROOT="${RESULT_ROOT}_seed${SEED}"
fi
PREFLIGHT="${PREFLIGHT:-0}"
EPISODES="${EPISODES:-2049}"

if [[ ! -f "${POLICY}" ]]; then
    echo "[mask-bf] policy checkpoint missing: ${POLICY}" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" && "${PREFLIGHT}" != "1" ]]; then
    echo "[mask-bf] refusing to overwrite ${RESULT_ROOT}" >&2
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
    raise SystemExit(f"[mask-bf] policy SHA mismatch: {actual}")
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
    raise SystemExit("[mask-bf] invalid policy checkpoint: " + ", ".join(failed))
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
    local backfill="$4"
    echo "[mask-bf] cell=${tag} perturb=${perturb} dropout=${dropout} backfill=${backfill}"
    if [[ "${PREFLIGHT}" == "1" ]]; then
        return 0
    fi
    NAVRL_PERCEPTION_PERTURB="${perturb}" \
    NAVRL_DETECTION_DROPOUT="${dropout}" \
    NAVRL_DETECTION_LATENCY_S=0 \
    NAVRL_RANGE_ERROR_M=0 \
    NAVRL_TARGET_MASK_BACKFILL="${backfill}" \
    NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
        ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
}

run_cell analytic_clean       0 0   0
run_cell dropout_0p3_raw      1 0.3 0
run_cell dropout_0p3_backfill 1 0.3 1
run_cell clean_backfill       0 0   1

if [[ "${PREFLIGHT}" == "1" ]]; then
    echo "[mask-bf] PREFLIGHT PASS (no results written)"
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
clean, raw, fixed = cells["analytic_clean"], cells["dropout_0p3_raw"], cells["dropout_0p3_backfill"]
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
    "clean_regression_pp": (cells["clean_backfill"]["capture"] - clean["capture"]) * 100.0,
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
lines = ["# phantom-target obstacle under detection dropout (ep25000+riskcap, seed %s, 205 bars)" % os.environ["NAVRL_SEED"],
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
echo "[mask-bf] done -> ${RESULT_ROOT}/summary.{md,json}"
