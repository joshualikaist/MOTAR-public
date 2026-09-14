#!/usr/bin/env bash
# Latency COMPENSATION arms for the frozen ep25000 + riskcap policy (WORKLOG 2026-08-05 plan).
#
# R3 measured detection latency as the dominant perception bottleneck (0.1 s: capture
# 80.54% -> 37.82%). These arms hold the latency PERTURBATION fixed and toggle the perception
# FIXES to measure recovery. No PPO. Frozen contract identical to R3: seed47, 205 bars,
# deterministic, riskcap governor, 2049 episodes/cell.
#
#   arm                perturb  latency  P0 predict  P1 lidar-backup
#   analytic_clean        0       0          off          off        (80.54 / 17.17 재현)
#   latency_0p1s_raw      1       0.1        off          off        (37.82 / 58.22 재현)
#   latency_0p1s_p0       1       0.1        ON           off
#   latency_0p1s_p0p1     1       0.1        ON           ON
#   latency_0p2s_p0p1     1       0.2        ON           ON         (NAVRL_LAT_INCLUDE_0P2=1)
#
# GO (1차, WORKLOG): latency 0.1 + P0(+P1)에서 capture >= 65% AND crash가 raw arm 대비
# >= 10 pp 감소. 완전 회복(>= 78%)은 2차 목표.
#
# Usage:
#   ./eval_navrl_v2_latency_compensate.sh            # 4 arms
#   NAVRL_LAT_INCLUDE_0P2=1 ./eval_navrl_v2_latency_compensate.sh
#   PREFLIGHT=1 ./eval_navrl_v2_latency_compensate.sh  # dry contract check
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON

POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
RESULT_ROOT="../../../results/navrl_v2_latency_compensate"
PREFLIGHT="${PREFLIGHT:-0}"
EPISODES="${EPISODES:-2049}"

if [[ ! -f "${POLICY}" ]]; then
    echo "[lat-comp] policy checkpoint missing: ${POLICY}" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" && "${PREFLIGHT}" != "1" ]]; then
    echo "[lat-comp] refusing to overwrite ${RESULT_ROOT}" >&2
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
    raise SystemExit(f"[lat-comp] policy SHA mismatch: {actual}")
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
    raise SystemExit("[lat-comp] invalid policy checkpoint: " + ", ".join(failed))
print(actual)
PY
)"

# Frozen evaluation contract -- byte-identical to eval_navrl_v2_detector_robustness.sh.
export NAVRL_V2_DENSITIES=205
export NAVRL_V2_ACTION_MODE=deterministic
export NAVRL_EVAL_REFLECTION_MODE=original
export NAVRL_SPEED_GOVERNOR=riskcap
export NAVRL_SEED=47
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
    local latency="$3"
    local compensate="$4"
    local lidar_backup="$5"
    echo "[lat-comp] cell=${tag} perturb=${perturb} latency=${latency}s P0=${compensate} P1=${lidar_backup}"
    if [[ "${PREFLIGHT}" == "1" ]]; then
        return 0
    fi
    NAVRL_PERCEPTION_PERTURB="${perturb}" \
    NAVRL_DETECTION_DROPOUT=0 \
    NAVRL_RANGE_ERROR_M=0 \
    NAVRL_DETECTION_LATENCY_S="${latency}" \
    NAVRL_LATENCY_COMPENSATE="${compensate}" \
    NAVRL_LATENCY_LIDAR_BACKUP="${lidar_backup}" \
    NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
        ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
}

run_cell analytic_clean     0 0   0 0
run_cell latency_0p1s_raw   1 0.1 0 0
run_cell latency_0p1s_p0    1 0.1 1 0
run_cell latency_0p1s_p0p1  1 0.1 1 1
if [[ "${NAVRL_LAT_INCLUDE_0P2:-0}" == "1" ]]; then
    run_cell latency_0p2s_p0p1 1 0.2 1 1
fi

if [[ "${PREFLIGHT}" == "1" ]]; then
    echo "[lat-comp] PREFLIGHT PASS (no results written)"
    exit 0
fi

"${PYTHON}" - "${RESULT_ROOT}" "${ACTUAL_POLICY_SHA}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
policy_sha = sys.argv[2]
cells = {}
for cell_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    payload = json.loads((cell_dir / "205bars.json").read_text(encoding="utf-8"))
    out, causes = payload["outcome"], payload["crash_causes"]
    cells[cell_dir.name] = {
        "episodes": payload["actual_episodes"],
        "capture": out["capture_rate"],
        "crash": out["crash_rate"],
        "timeout": out["timeout_rate"],
        "bar_contact_share": causes["bar_contact_share"],
    }

base = cells["analytic_clean"]
raw = cells["latency_0p1s_raw"]
gate = {}
for tag in ("latency_0p1s_p0", "latency_0p1s_p0p1"):
    c = cells[tag]
    gate[tag] = {
        "capture_ge_65": c["capture"] >= 0.65,
        "crash_down_10pp_vs_raw": (raw["crash"] - c["crash"]) >= 0.10,
        "GO": c["capture"] >= 0.65 and (raw["crash"] - c["crash"]) >= 0.10,
        "capture_delta_vs_clean_pp": (c["capture"] - base["capture"]) * 100.0,
        "capture_delta_vs_raw_pp": (c["capture"] - raw["capture"]) * 100.0,
    }

summary = {"policy_sha256": policy_sha, "contract": "seed47/205bars/deterministic/riskcap",
           "cells": cells, "gate": gate}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
lines = ["# latency compensation arms (ep25000+riskcap, seed47, 205 bars)", "",
         "| cell | episodes | capture | crash | timeout | bar-contact share |",
         "|---|---:|---:|---:|---:|---:|"]
for tag, c in cells.items():
    lines.append(f"| {tag} | {c['episodes']} | {c['capture']*100:.2f}% | "
                 f"{c['crash']*100:.2f}% | {c['timeout']*100:.2f}% | "
                 f"{c['bar_contact_share']*100:.1f}% |")
lines += ["", "## GO gate (capture >= 65% AND crash >= 10 pp down vs latency_0p1s_raw)", ""]
for tag, g in gate.items():
    lines.append(f"- {tag}: {'GO' if g['GO'] else 'NO-GO'} "
                 f"(vs clean {g['capture_delta_vs_clean_pp']:+.2f} pp, "
                 f"vs raw {g['capture_delta_vs_raw_pp']:+.2f} pp)")
(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
echo "[lat-comp] done -> ${RESULT_ROOT}/summary.{md,json}"
