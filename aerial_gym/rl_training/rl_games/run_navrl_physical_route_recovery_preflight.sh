#!/usr/bin/env bash
set -euo pipefail

# Safety-lineage preflight only.  This script intentionally does not invoke PPO, Isaac Gym, or a
# simulator.  A measured target-specific braking probe and the exact fresh-only route contract are
# prerequisites for a later, separately authorized smoke.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
: "${NAVRL_TARGET_RECOVERY_BRAKE_P05:?set the measured zero-command target braking p05 (m/s^2)}"
: "${NAVRL_TARGET_RECOVERY_STOP_TIME_P95_S:?set measured zero-command stop-time p95 (s)}"
: "${NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT:?set the hashed braking-probe receipt JSON path}"
: "${NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256:?set the braking-probe receipt SHA256}"

python3 - "${REPO_ROOT}" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
value = float(__import__("os").environ["NAVRL_TARGET_RECOVERY_BRAKE_P05"])
stop_p95 = float(__import__("os").environ["NAVRL_TARGET_RECOVERY_STOP_TIME_P95_S"])
receipt = pathlib.Path(__import__("os").environ["NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT"])
declared_sha = __import__("os").environ["NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256"].lower()
if not value > 0.0 or not stop_p95 > 0.0 or len(declared_sha) != 64:
    raise SystemExit("recovery braking p05/stop-time p95/SHA must be finite and positive")
if not receipt.is_file() or hashlib.sha256(receipt.read_bytes()).hexdigest() != declared_sha:
    raise SystemExit("recovery braking-probe receipt missing or SHA256 mismatch")
probe = __import__("json").loads(receipt.read_text(encoding="utf-8"))
if probe.get("schema") != "navrl_target_recovery_braking_probe_v1":
    raise SystemExit("invalid braking-probe receipt schema")
if abs(float(probe["decel_p05_mps2"]) - value) > 1e-9 or abs(float(probe["stop_time_p95_s"]) - stop_p95) > 1e-9:
    raise SystemExit("braking-probe receipt does not match measured environment values")
required = {
    "NAVRL_TARGET_DYNAMICS": "physical",
    "NAVRL_TARGET_PATTERN": "waypoint",
    "NAVRL_TARGET_ROUTE_MODE": "global_astar_recovery_v2",
}
import os
for key, expected in required.items():
    os.environ[key] = expected
files = [
    "aerial_gym/task/navrl_task/target_route_planner.py",
    "aerial_gym/task/navrl_task/target_motion.py",
    "aerial_gym/task/navrl_task/physical_target.py",
    "aerial_gym/task/navrl_task/navrl_task.py",
    "aerial_gym/rl_training/rl_games/run_navrl_physical_route_recovery_preflight.sh",
    "docs/preregistration_physical_target_two_envelope_recovery_2026-08-25.md",
]
digests = {}
for relative in files:
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"missing recovery source: {relative}")
    digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
print("two-envelope recovery preflight PASS (no PPO/simulator execution)")
print(f"model=physx_ref5in_6dof_global_astar_aabb_v2_two_envelope_recovery mode=global_astar_recovery_v2 brake_p05={value:.9g} stop_time_p95={stop_p95:.9g}")
for relative in files:
    print(f"sha256[{relative}]={digests[relative]}")
PY
