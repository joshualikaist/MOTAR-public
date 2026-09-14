#!/usr/bin/env python3
"""Fixed-grid launcher for pilot and confirmatory braking-route-v3 evaluation.

The supplied cell runner is a repository-tracked simulator adapter.  It receives the frozen cell
contract through environment variables and must atomically write the requested cell JSON.  This
keeps the gate independent of a particular Isaac Gym entry point while binding the adapter bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import verify_navrl_braking_route_v3_gate as verify


ROOT = verify.ROOT
BRAKE_VERIFY_PATH = ROOT / "tools/verify_navrl_physical_target_braking.py"


def git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    verify.require(completed.returncode == 0, completed.stderr.strip())
    return completed.stdout.strip()


def tracked_hashes(cell_runner: Path) -> dict[str, str]:
    dirty = git("status", "--porcelain", "--untracked-files=no")
    verify.require(not dirty, f"runtime-clean guard failed: {dirty}")
    required = [
        verify.PREREG,
        Path(__file__).resolve(),
        Path(verify.__file__).resolve(),
        cell_runner,
        ROOT / "aerial_gym/__init__.py",
        ROOT / "aerial_gym/task/navrl_task/navrl_task.py",
        ROOT / "aerial_gym/task/navrl_task/target_motion.py",
        ROOT / "aerial_gym/task/navrl_task/target_route_geometry.py",
        ROOT / "aerial_gym/task/navrl_task/target_route_planner.py",
        ROOT / "aerial_gym/config/task_config/navrl_task_config.py",
        ROOT / "aerial_gym/config/sim_config/base_sim_config.py",
        ROOT / "aerial_gym/config/robot_config/navrl_ref5in_v2_quad_config.py",
        ROOT / "resources/robots/quad/quad_navrl_ref5in_v2.urdf",
    ]
    if verify.EXECUTION_AUTHORITY is not None:
        required.append(verify.EXECUTION_AUTHORITY)
    tracked = set(git("ls-files").splitlines())
    for path in required:
        verify.require(path.is_file(), f"required runtime source missing: {path}")
        verify.require(str(path.relative_to(ROOT)) in tracked, f"runtime source is not tracked: {path}")
    return {str(path.relative_to(ROOT)): verify.sha256_file(path) for path in required}


def verify_braking_receipt(path: Path, expected_sha: str) -> dict:
    verify.require(path.is_file() and verify.sha256_file(path) == expected_sha, "braking receipt hash mismatch")
    spec = importlib.util.spec_from_file_location("navrl_v3_raw_brake_verify", BRAKE_VERIFY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.verify_receipt(path.parent, ROOT)
    verify.require(isinstance(result, dict) and result.get("summary") and result.get("core_integration"), "raw braking receipt did not verify")
    verify.require(result.get("verified") is True, "raw braking receipt is not verified")
    return result


def source_manifest(hashes: dict[str, str]) -> tuple[dict, str]:
    payload = {"schema": "navrl_braking_route_v3_source_manifest_v1", "git_commit": git("rev-parse", "HEAD"), "files": hashes}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return payload, hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=tuple(verify.STAGES))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cell-runner", type=Path, required=True)
    parser.add_argument("--braking-receipt", type=Path, required=True)
    parser.add_argument("--braking-receipt-sha256", required=True)
    parser.add_argument("--pilot-summary", type=Path)
    args = parser.parse_args()
    try:
        output = args.output_root.resolve()
        verify.require(not output.exists(), "unique output root already exists")
        verify.require(output != ROOT and ROOT not in output.parents, "output root cannot contain repository root")
        runner = args.cell_runner.resolve()
        receipt = args.braking_receipt.resolve()
        verify.require(runner.is_file() and os.access(runner, os.X_OK), "cell runner is missing/not executable")
        hashes = tracked_hashes(runner)
        verify.require(verify.sha256_file(verify.PREREG) == verify.PREREG_SHA256, "frozen prereg SHA mismatch")
        brake_result = verify_braking_receipt(receipt, args.braking_receipt_sha256.lower())
        core = brake_result["core_integration"]
        certified = core["certified_monotone_speed_to_p95_lookup"]
        canonical_rows = [certified[key] for key in sorted(certified, key=lambda key: float(key))]
        speed_csv = ",".join(str(row["speed_mps"]) for row in canonical_rows)
        distance_csv = ",".join(str(row["p95_stop_distance_m"]) for row in canonical_rows)
        manifest, manifest_sha = source_manifest(hashes)

        pilot_authorization = None
        if args.stage == "confirmatory":
            verify.require(args.pilot_summary is not None, "pilot PASS receipt is required")
            pilot = args.pilot_summary.resolve()
            pilot_payload = json.loads(pilot.read_text(encoding="utf-8"))
            verify.validate_payload(pilot_payload)
            pilot_verdict = verify.derive_verdict(pilot_payload)
            verify.require(pilot_verdict["gate"] == "PASS_PILOT_AUTHORIZES_CONFIRMATORY", "pilot FAIL blocks confirmatory")
            pilot_authorization = {"summary_sha256": verify.sha256_file(pilot), "verdict": pilot_verdict["gate"]}
        else:
            verify.require(args.pilot_summary is None, "pilot stage cannot consume --pilot-summary")

        output.mkdir(parents=True, exist_ok=False)
        (output / "cells").mkdir()
        (output / "logs").mkdir()
        (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        identity = {
            "runtime_clean": True,
            "physical_geometry_version": "v2",
            "placement_mode": "footprint_clearance",
            "preregistration_sha256": verify.PREREG_SHA256,
            "robot_name": "navrl_ref5in_v2_quad",
            "target_dynamics": "physical",
            "target_pattern": "waypoint",
            "source_manifest_sha256": manifest_sha,
            "import_origin_sha256": hashes["aerial_gym/__init__.py"],
            "robot_config_sha256": hashes["aerial_gym/config/robot_config/navrl_ref5in_v2_quad_config.py"],
            "robot_urdf_sha256": hashes["resources/robots/quad/quad_navrl_ref5in_v2.urdf"],
            "sim_config_sha256": hashes["aerial_gym/config/sim_config/base_sim_config.py"],
            "task_config_sha256": hashes["aerial_gym/config/task_config/navrl_task_config.py"],
            "cell_runner_sha256": verify.sha256_file(runner),
            "braking_receipt_sha256": args.braking_receipt_sha256.lower(),
        }
        if verify.EXECUTION_AUTHORITY is not None:
            identity["execution_authority_sha256"] = verify.EXECUTION_AUTHORITY_SHA256
        config = verify.STAGES[args.stage]
        cells = []
        for route in verify.ROUTE_ARMS:
            for speed in verify.SPEEDS:
                for bars in config["densities"]:
                    rid = verify.record_id(route, speed, bars)
                    cell_path = output / "cells" / f"{rid.replace('.', 'p')}.json"
                    log_path = output / "logs" / f"{rid.replace('.', 'p')}.log"
                    env = {key: value for key, value in os.environ.items() if not key.startswith("NAVRL_")}
                    env.update({
                        "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT),
                        "NAVRL_V3_CELL_OUTPUT": str(cell_path), "NAVRL_V3_RECORD_ID": rid,
                        "NAVRL_V3_STAGE": args.stage, "NAVRL_V3_SEED": str(config["seed"]),
                        "NAVRL_V3_ROUTE_MODE": route, "NAVRL_V3_SPEED_MPS": str(speed),
                        "NAVRL_V3_BARS": str(bars), "NAVRL_V3_ENVS": str(verify.ENVS),
                        "NAVRL_V3_STEPS": str(verify.STEPS), "NAVRL_V3_WARMUP_STEPS": str(verify.WARMUP_STEPS),
                        "NAVRL_PHYSICAL_GEOMETRY_VERSION": "v2", "NAVRL_PLACEMENT_MODE": "footprint_clearance",
                        "NAVRL_TARGET_DYNAMICS": "physical", "NAVRL_TARGET_PATTERN": "waypoint",
                        "NAVRL_TARGET_ROUTE_MODE": route,
                        "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT": str(receipt),
                        "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256": args.braking_receipt_sha256.lower(),
                        "NAVRL_TARGET_RECOVERY_PROBE_VALIDATED": "1",
                        "NAVRL_TARGET_BRAKING_CONTRACT_VARIANT": verify.CONTRACT_VARIANT,
                        "NAVRL_TARGET_RECOVERY_BRAKE_P05": str(core["decel_p05_mps2"]),
                        "NAVRL_TARGET_RECOVERY_STOP_TIME_P95_S": str(core["stop_time_p95_s"]),
                        "NAVRL_TARGET_RECOVERY_BRAKE_SPEEDS_MPS": speed_csv,
                        "NAVRL_TARGET_RECOVERY_BRAKE_STOP_DISTANCES_M": distance_csv,
                        "NAVRL_TARGET_RECOVERY_BRAKE_LATERAL_TUBE_P95_M": str(core["certified_lateral_tube_p95_m"]),
                        "NAVRL_V3_IDENTITY_JSON": json.dumps(identity, sort_keys=True, separators=(",", ":")),
                    })
                    with log_path.open("x", encoding="utf-8") as log:
                        completed = subprocess.run(
                            [sys.executable, str(runner)],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        )
                    verify.require(completed.returncode == 0, f"cell runner failed: {rid}")
                    verify.require(cell_path.is_file(), f"cell runner produced no payload: {rid}")
                    row = json.loads(cell_path.read_text(encoding="utf-8"))
                    verify.validate_cell(row, args.stage, identity)
                    cells.append(row)

        payload = {
            "schema": verify.SCHEMA, "stage": args.stage, "seed": config["seed"],
            "route_arms": list(verify.ROUTE_ARMS), "speeds_mps": list(verify.SPEEDS),
            "densities": list(config["densities"]), "envs": verify.ENVS, "steps": verify.STEPS,
            "warmup_steps": verify.WARMUP_STEPS, "physical_gates_preregistered": verify.PHYSICAL_GATES,
            "identity": identity, "pilot_authorization": pilot_authorization, "cells": cells,
        }
        payload["verdict"] = verify.derive_verdict(payload)
        summary = output / "summary.json"
        summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return verify.verify_summary(summary, args.pilot_summary.resolve() if args.pilot_summary else None)
    except (verify.IntegrityError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"VOID_EXECUTION: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
