#!/usr/bin/env bash
# Copy last_gen checkpoints into checkpoints/density_<N>/ for a finished or running run.
# Usage: ./snapshot_density_ckpts.sh runs/ppo_YYMMDD_HHMM_navrl
set -euo pipefail
RUN="${1:?usage: $0 runs/ppo_XXXX_navrl}"
cd "$(dirname "${BASH_SOURCE[0]}")"
NN="${RUN}/nn"
[[ -d "${NN}" ]] || { echo "no nn/ in ${RUN}"; exit 1; }
mkdir -p checkpoints
for f in "${NN}"/last_gen_ppo_ep_*.pth; do
  [[ -f "$f" ]] || continue
  ep=$(basename "$f" | sed 's/last_gen_ppo_ep_//;s/.pth//')
  dest="checkpoints/seq_density_ep${ep}"
  mkdir -p "$dest"
  cp -a "$f" "$dest/last_gen.pth"
  echo "snapshotted ep${ep} -> ${dest}/last_gen.pth"
done
echo "done. $(ls -1 checkpoints/seq_density_ep* 2>/dev/null | wc -l) snapshots under checkpoints/"
