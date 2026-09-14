#!/usr/bin/env bash
# v2 density x target-speed map for the frozen candidate (ep25000 + riskcap) -- the paper's
# headline figure, re-measured on the task actually being reported.
#
# The published map is v1 data: 24 m arena, 478 m2 placement band, densities 25-150 bars, and a
# policy from before the v2 task bump. It says density costs 78 pp of capture across its grid
# while target speed costs 4.2 pp. That claim has never been tested on v2, where the goal can sit
# beyond the 20 m sensor horizon and the target moves before the drone arrives -- exactly the
# regime where the speed axis might start to matter.
#
# Grid: 5 densities x 4 fixed target speeds = 20 cells, ~2050 episodes each.
#   densities  130 / 160 / 190 / 205 / 220   (trained range, reached density, generalisation)
#   speeds     0.3 / 0.7 / 1.1 / 1.5 m/s     (inside the trained U[0.3,1.5] support; the sweep
#                                             refuses anything outside it)
# v1's 0 m/s column has no v2 counterpart: a stationary target is outside the training support.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PYTHON
POLICY="runs/ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/nn/last_gen_ppo_ep_25000_rew_39.742134.pth"
POLICY_SHA="f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
RESULT_ROOT="../../../results/navrl_v2_density_speed_map"
PREFLIGHT="${PREFLIGHT:-0}"
SPEEDS="${NAVRL_MAP_SPEEDS:-0.3 0.7 1.1 1.5}"

[[ -f "${POLICY}" ]] || { echo "[map] policy missing" >&2; exit 2; }
if [[ -e "${RESULT_ROOT}" && "${PREFLIGHT}" != "1" ]]; then
    echo "[map] refusing to overwrite ${RESULT_ROOT}" >&2; exit 2
fi
ACTUAL_SHA="$("${PYTHON}" - "${POLICY}" "${POLICY_SHA}" <<'PY'
import hashlib, sys
from pathlib import Path
p, expected = Path(sys.argv[1]), sys.argv[2]
actual = hashlib.sha256(p.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"[map] policy SHA mismatch: {actual}")
print(actual)
PY
)"

export NAVRL_V2_DENSITIES="130 160 190 205 220"
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
unset NAVRL_DETECTOR_CHECKPOINT

for speed in ${SPEEDS}; do
    tag="speed_$(echo "${speed}" | tr '.' 'p')"
    echo "[map] column ${tag}: densities=${NAVRL_V2_DENSITIES} target=${speed} m/s"
    [[ "${PREFLIGHT}" == "1" ]] && continue
    NAVRL_V2_FIXED_TARGET_SPEED="${speed}" \
    NAVRL_V2_RESULT_DIR="${RESULT_ROOT}/${tag}" \
        ./eval_navrl_v2_density_sweep.sh "${POLICY}" 2049
done
if [[ "${PREFLIGHT}" == "1" ]]; then echo "[map] PREFLIGHT PASS"; exit 0; fi

"${PYTHON}" - "${RESULT_ROOT}" "${ACTUAL_SHA}" <<'PY'
import json
from pathlib import Path
import sys

root, sha = Path(sys.argv[1]), sys.argv[2]
rows = []
for col in sorted(root.iterdir()):
    if not col.is_dir():
        continue
    speed = float(col.name.replace("speed_", "").replace("p", "."))
    for cell in sorted(col.glob("*bars.json"), key=lambda p: int(p.stem.replace("bars", ""))):
        payload = json.loads(cell.read_text(encoding="utf-8"))
        out = payload["outcome"]
        bars = int(cell.stem.replace("bars", ""))
        rows.append({
            "bars": bars,
            "density_per_100m2": round(bars / 1600.0 * 100.0, 2),
            "target_speed_ms": speed,
            "capture": out["capture_rate"],
            "crash": out["crash_rate"],
            "timeout": out["timeout_rate"],
            "bar_contact_share": payload["crash_causes"]["bar_contact_share"],
            "episodes": payload["actual_episodes"],
        })

bars_axis = sorted({r["bars"] for r in rows})
speed_axis = sorted({r["target_speed_ms"] for r in rows})
at = {(r["bars"], r["target_speed_ms"]): r for r in rows}
mean = lambda xs: sum(xs) / len(xs) if xs else None
by_bars = [mean([at[(b, s)]["capture"] for s in speed_axis if (b, s) in at]) for b in bars_axis]
by_speed = [mean([at[(b, s)]["capture"] for b in bars_axis if (b, s) in at]) for s in speed_axis]
density_cost_pp = (by_bars[0] - by_bars[-1]) * 100.0
speed_cost_pp = (by_speed[0] - by_speed[-1]) * 100.0

summary = {
    "policy_sha256": sha,
    "policy": "ep25000 + riskcap",
    "task_version": "v2",
    "arena_xy_m": 40.0,
    "placement_area_m2": 1600.0,
    "contract": "seed47/deterministic/riskcap",
    "trained_max_bars": 205,
    "density_cost_pp": density_cost_pp,
    "speed_cost_pp": speed_cost_pp,
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
lines = ["# v2 density x target-speed map (ep25000+riskcap, seed47, 40 m arena)", "",
         "capture %, ~2050 episodes per cell", "",
         "| bars | /100m² | " + " | ".join(f"{s:.1f} m/s" for s in speed_axis) + " |",
         "|---:|---:|" + "---:|" * len(speed_axis)]
for b in bars_axis:
    cells = []
    for s in speed_axis:
        r = at.get((b, s))
        cells.append(f"{r['capture']*100:.2f}" if r else "—")
    lines.append(f"| {b} | {b/1600*100:.2f} | " + " | ".join(cells) + " |")
lines += ["", f"- density axis costs **{density_cost_pp:.1f} pp** of capture over this grid",
          f"- target-speed axis costs **{speed_cost_pp:.1f} pp**",
          "",
          "v1 (24 m arena, 478 m² band, 25-150 bars) measured -78 pp vs -4.2 pp. The two grids "
          "cover different densities and a different task, so compare the SHAPE of the asymmetry, "
          "not the numbers."]
(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY
echo "[map] done -> ${RESULT_ROOT}/summary.{md,json}"
