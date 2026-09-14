#!/usr/bin/env bash
# Latency P2 arms: fix WHERE the obstacle map is edited under detection latency.
#
# P0/P1 were both NO-GO (WORKLOG 2026-08-05): +P0 moved capture -0.10 pp, +P0+P1 -8.85 pp.
# P0 provably removed the v*tau tracker lag (closest_nocrash_mean 1.611 -> 1.454 m) without
# recovering capture, which rejects the "policy sees a stale target position" hypothesis.
# The surviving channel is the obstacle map: observe() feeds the DELAYED bearing/range/pixel
# mask into _fuse_static_and_extract_obstacles, whose target_like carve-out and depth blanking
# then erase LiDAR/depth returns at the target's OLD bearing -- deleting real bars. That is why
# delaying only the camera path tripled bar contacts (337 -> 931) while LiDAR stayed undelayed.
#
#   arm                     latency  P0   obstacle fix
#   latency_0p1s_raw          0.1    off  off        (37.82 / 58.22 재현, 앵커)
#   latency_0p1s_map_skip     0.1    off  skip       stale 동안 map 편집 중단 (bar 삭제 0)
#   latency_0p1s_map_predict  0.1    off  predict    tracker 예측 위치로 carve-out 재배치
#   latency_0p1s_p0_predict   0.1    ON   predict    P0 + P2 (정책 출력 + map 동시 보정)
#
# analytic_clean is NOT re-run: the clean cell is byte-identical to
# results/navrl_v2_latency_compensate/analytic_clean (80.54 / 17.17) under the same contract and
# the compensation knobs are arithmetic no-ops at zero latency (tests/test_navrl_latency_compensate.py::
# test_no_latency_makes_every_mode_bit_identical). Set NAVRL_LAT_RERUN_CLEAN=1 to re-measure it.
#
# GO gate (unchanged, WORKLOG): capture >= 65% AND crash >= 10 pp below the raw arm.
# Secondary read regardless of GO: bar_contact absolute count vs raw -- that is the quantity the
# hypothesis predicts, so it discriminates "wrong mechanism" from "right mechanism, too small".
#
# Usage:
#   ./eval_navrl_v2_latency_obstacle_fix.sh
#   PREFLIGHT=1 ./eval_navrl_v2_latency_obstacle_fix.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON

POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
RESULT_ROOT="../../../results/navrl_v2_latency_obstacle_fix"
BASELINE_ROOT="../../../results/navrl_v2_latency_compensate"
PREFLIGHT="${PREFLIGHT:-0}"
EPISODES="${EPISODES:-2049}"

if [[ ! -f "${POLICY}" ]]; then
    echo "[lat-map] policy checkpoint missing: ${POLICY}" >&2
    exit 2
fi
if [[ -e "${RESULT_ROOT}" && "${PREFLIGHT}" != "1" ]]; then
    echo "[lat-map] refusing to overwrite ${RESULT_ROOT}" >&2
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
    raise SystemExit(f"[lat-map] policy SHA mismatch: {actual}")
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
    raise SystemExit("[lat-map] invalid policy checkpoint: " + ", ".join(failed))
print(actual)
PY
)"

# Frozen evaluation contract -- byte-identical to the P0/P1 and R3 harnesses.
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
    local obstacle_fix="$5"
    echo "[lat-map] cell=${tag} perturb=${perturb} latency=${latency}s P0=${compensate} P2=${obstacle_fix}"
    if [[ "${PREFLIGHT}" == "1" ]]; then
        return 0
    fi
    NAVRL_PERCEPTION_PERTURB="${perturb}" \
    NAVRL_DETECTION_DROPOUT=0 \
    NAVRL_RANGE_ERROR_M=0 \
    NAVRL_DETECTION_LATENCY_S="${latency}" \
    NAVRL_LATENCY_COMPENSATE="${compensate}" \
    NAVRL_LATENCY_LIDAR_BACKUP=0 \
    NAVRL_LATENCY_OBSTACLE_FIX="${obstacle_fix}" \
    NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
        ./eval_navrl_v2_density_sweep.sh "${POLICY}" "${EPISODES}"
}

if [[ "${NAVRL_LAT_RERUN_CLEAN:-0}" == "1" ]]; then
    run_cell analytic_clean 0 0 0 off
fi
run_cell latency_0p1s_raw         1 0.1 0 off
run_cell latency_0p1s_map_skip    1 0.1 0 skip
run_cell latency_0p1s_map_predict 1 0.1 0 predict
run_cell latency_0p1s_p0_predict  1 0.1 1 predict

if [[ "${PREFLIGHT}" == "1" ]]; then
    echo "[lat-map] PREFLIGHT PASS (no results written)"
    exit 0
fi

"${PYTHON}" - "${RESULT_ROOT}" "${BASELINE_ROOT}" "${ACTUAL_POLICY_SHA}" <<'PY'
import json
from pathlib import Path
import sys

root, baseline_root, policy_sha = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]


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
    }


cells = {d.name: load(d) for d in sorted(p for p in root.iterdir() if p.is_dir())}
# The clean anchor comes from the P0/P1 sweep unless this run re-measured it.
if "analytic_clean" not in cells:
    cells["analytic_clean"] = load(baseline_root / "analytic_clean")
    cells["analytic_clean"]["source"] = "navrl_v2_latency_compensate (re-used)"

raw = cells["latency_0p1s_raw"]
arms = [t for t in cells if t.startswith("latency_") and t != "latency_0p1s_raw"]
gate = {}
for tag in sorted(arms):
    c = cells[tag]
    gate[tag] = {
        "GO": c["capture"] >= 0.65 and (raw["crash"] - c["crash"]) >= 0.10,
        "capture_delta_vs_raw_pp": (c["capture"] - raw["capture"]) * 100.0,
        "capture_delta_vs_clean_pp": (c["capture"] - cells["analytic_clean"]["capture"]) * 100.0,
        # The mechanism-level read: P2 claims stale map edits delete real bars, so a working
        # fix must cut bar contacts even if capture stays short of the GO threshold.
        "bar_contact_delta_vs_raw": c["bar_contact"] - raw["bar_contact"],
    }

(root / "summary.json").write_text(
    json.dumps({"policy_sha256": policy_sha,
                "contract": "seed47/205bars/deterministic/riskcap",
                "cells": cells, "gate": gate}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
lines = ["# latency P2 obstacle-map arms (ep25000+riskcap, seed47, 205 bars)", "",
         "| cell | episodes | capture | crash | timeout | bar contacts | OOB | closest mean (m) |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
for tag, c in cells.items():
    lines.append(f"| {tag} | {c['episodes']} | {c['capture']*100:.2f}% | {c['crash']*100:.2f}% | "
                 f"{c['timeout']*100:.2f}% | {c['bar_contact']} | {c['out_of_bounds']} | "
                 f"{c['closest_nocrash_mean_m']:.3f} |")
lines += ["", "## GO gate (capture >= 65% AND crash >= 10 pp below latency_0p1s_raw)", ""]
for tag, g in gate.items():
    lines.append(f"- {tag}: {'GO' if g['GO'] else 'NO-GO'} "
                 f"(capture vs raw {g['capture_delta_vs_raw_pp']:+.2f} pp, "
                 f"vs clean {g['capture_delta_vs_clean_pp']:+.2f} pp, "
                 f"bar contacts vs raw {g['bar_contact_delta_vs_raw']:+d})")
(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
echo "[lat-map] done -> ${RESULT_ROOT}/summary.{md,json}"
