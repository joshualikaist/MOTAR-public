#!/usr/bin/env bash
# Task-v2 CEILING PROBE: how high can capture actually go at ONE fixed density?
#
# Why this run exists
# -------------------
# The v2 density curriculum promotes when measured capture clears a threshold that ramps down with
# bar count. BOTH endpoints of that ramp were guesses, and only one of them has since been measured:
#
#   70 bars  -> 0.85 assumed.  MEASURED 2026-07-31 (run ppo_260731_1722, 2300 epochs): two full
#               16,384-episode gate windows scored 0.816 and 0.837 while crash decayed to an
#               extrapolated floor of 13.1% (fit tau~650 epochs, converged). With the ~2.6% timeout
#               base that is a ceiling of ~0.843 -- BELOW the threshold. Retuned to 0.80.
#   300 bars -> 0.70 assumed.  NEVER MEASURED. v2 has never been run at 300 bars. The number was
#               inherited from v1's flat threshold. Extrapolating v1's density curve says it is
#               very likely unreachable: 300 bars is 18.8/100m^2, where a FINISHED v1 policy scored
#               0.67-0.75 with SHORT paths and a static target, while v2 already caps at 0.84 in a
#               regime where v1 reached 0.96. So the curriculum is expected to stall again near the
#               dense end, for the same reason it stalled at the sparse end.
#
# The main run cannot answer this about itself: its dwell gate blocks promotion for 1000 epochs per
# level regardless, so reaching 300 bars takes >=16,000 epochs before the question is even asked.
# This probe answers it directly by freezing density and letting capture run to its plateau, which
# also yields the per-density ceiling table needed to set every threshold from evidence.
#
#   PROBE_BARS=300 ./train_navrl_v2_ceiling_probe.sh   # the value that matters most right now
#
# Why the 4 GB card is a VALID place to measure it
# ------------------------------------------------
# Training results from the 1650 Ti (64 envs) are NOT comparable to the 3070 (128 envs) -- half the
# PPO batch learns measurably worse, and per project rule the two are never pooled. That rule kills
# a same-config parallel run, but it does NOT kill this measurement, because the inference here is
# ONE-SIDED: the 64-env policy is a lower bound on the 128-env policy.
#   64-env plateau >= 0.85  =>  0.85 is reachable, the main run's threshold is safe.  (conclusive)
#   64-env plateau <  0.85  =>  warning only; the stronger 128-env run may still clear it.
# So a PASS here is decisive and a FAIL is a flag to watch, which is exactly the asymmetry we want
# from a de-risking probe.
#
# Everything else is pinned to the main run's contract (train_navrl_v2_search.sh) so the only
# difference is the frozen density.
#
# Usage:
#   ./train_navrl_v2_ceiling_probe.sh                    # 70 bars, 2000 epochs
#   PROBE_BARS=100 ./train_navrl_v2_ceiling_probe.sh     # probe a different density
#   MAX_EPOCHS=3000 ./train_navrl_v2_ceiling_probe.sh
# Read the result from the epoch dashboard's capture tail, or:
#   python3 -c "import csv;r=list(csv.DictReader(open('runs/<run>/aerial_run/epoch_metrics.csv')));\
#               v=[float(x['captured_rate']) for x in r[-200:]];print(sum(v)/len(v))"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PROBE_BARS="${PROBE_BARS:-70}"

# Freeze the density curriculum. NAVRL_NUM_BARS sets the active count; with the curriculum off it
# never changes, so capture over epochs is a clean skill curve with no difficulty confound -- the
# one thing the main run's curve cannot give us.
export NAVRL_DENSITY_CURRICULUM=0
export NAVRL_NUM_BARS="${PROBE_BARS}"
# Keep the build-time ceiling at the main run's value so the PhysX actor count (and therefore the
# VRAM profile validated for this card) is unchanged: inactive bars are parked, not freed.
export NAVRL_MAX_BARS=300

# The dwell gate and the promotion threshold are inert with the curriculum off; pin them anyway so
# a stray shell export cannot reintroduce promotion mid-probe and silently invalidate the plateau.
export NAVRL_DENSITY_MIN_EPOCHS=0
unset NAVRL_DENSITY_START NAVRL_DENSITY_FINAL NAVRL_FIXED_BARS NAVRL_CONTROLLED_ABLATION

export MAX_EPOCHS="${MAX_EPOCHS:-2000}"
export SEED="${SEED:-1}"
export AERIAL_RUN_TAG="${AERIAL_RUN_TAG:-v2-ceiling${PROBE_BARS}-s${SEED}}"
export TRAIN_SESSION_LOG="${TRAIN_SESSION_LOG:-train_session_logs/v2_ceiling${PROBE_BARS}_$(date +%y%m%d_%H%M%S).log}"

echo "[ceiling] fixed density ${PROBE_BARS} bars ($(python3 -c "print(f'{${PROBE_BARS}/1600*100:.1f}')")/100m2), curriculum OFF"
echo "[ceiling] measuring the achievable capture plateau at this density (no curriculum confound)"
echo "[ceiling] 64-env plateau is a LOWER BOUND on the 128-env plateau (do not pool the two)"

# Delegate to the 4 GB launcher, which delegates to the main v2 launcher -- so the arena, sensors,
# action policy, reward and target-motion contract are inherited, not restated.
exec ./train_navrl_v2_search_4gb.sh "$@"
