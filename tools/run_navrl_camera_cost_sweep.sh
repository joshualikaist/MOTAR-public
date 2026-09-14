#!/usr/bin/env bash
# S0-a driver: render-cost curve over camera resolution x num_envs, plus a vision-off baseline
# per env count so the camera share can be isolated (and the foveated second-stream cost
# extrapolated). One process per cell; OOM/failed cells are recorded, not fatal.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PATH="$(dirname "${PYTHON}"):${PATH}"   # ninja for torch cpp extensions
OUT="results/navrl_camera_cost/cells.jsonl"
mkdir -p results/navrl_camera_cost

if [[ -s "${OUT}" ]]; then
    echo "[camera-cost] refusing to append to existing ${OUT}; move it first" >&2
    exit 2
fi

export NAVRL_PERCEPTION=1
export NAVRL_NUM_BARS=205        # eval-realistic scene complexity
export BENCH_STEPS=300
export BENCH_WARMUP=50
export BENCH_OUT="${OUT}"
export PYTHONNOUSERSITE=1

run_cell() {
    local vision="$1" w="$2" h="$3" envs="$4"
    echo "[camera-cost] vision=${vision} ${w}x${h} envs=${envs}"
    if NAVRL_VISION="${vision}" NAVRL_CAMERA_WIDTH="${w}" NAVRL_CAMERA_HEIGHT="${h}" \
        BENCH_NUM_ENVS="${envs}" "${PYTHON}" tools/benchmark_navrl_camera_cost.py; then
        return 0
    fi
    echo "{\"camera_width\": ${w}, \"camera_height\": ${h}, \"num_envs\": ${envs}, \"vision\": \"${vision}\", \"failed\": true}" >> "${OUT}"
}

for envs in 32 64 128; do
    run_cell 0 160 90 "${envs}"          # baseline: no vision stack
    run_cell 1 160 90 "${envs}"          # current contract
    run_cell 1 320 180 "${envs}"
    run_cell 1 640 360 "${envs}"
done

echo "[camera-cost] DONE -> ${OUT}"
"${PYTHON}" - "${OUT}" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"{'vision':>6} {'res':>9} {'envs':>5} {'ms/step':>8} {'env-steps/s':>12} {'peak MiB':>9}")
for r in rows:
    if r.get("failed"):
        print(f"{r.get('vision','?'):>6} {r['camera_width']}x{r['camera_height']:>4} {r['num_envs']:>5}  FAILED (likely OOM)")
        continue
    res = f"{r['camera_width']}x{r['camera_height']}"
    print(f"{r.get('vision','?'):>6} {res:>9} {r['num_envs']:>5} "
          f"{r['ms_per_step']:>8.1f} {r['env_steps_per_s']:>12.0f} {r['torch_peak_reserved_mib']:>9.0f}")
PY
