#!/usr/bin/env python3
"""Run the four-cell physical-target braking probe with fresh-process isolation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from typing import Any, Dict, Optional, Sequence

import probe_navrl_physical_target_braking as probe
import verify_navrl_physical_target_braking as verifier


TOOL_PATHS = probe.TOOL_SOURCE_PATHS


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="new, nonexistent receipt directory")
    parser.add_argument("--preflight", action="store_true")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        raise SystemExit("fresh-only launcher rejects unrecognized continuation options: %s" % " ".join(unknown))
    if not args.preflight and not args.output:
        parser.error("--output is required outside --preflight")
    return args


def _write(path: Path, payload: Any) -> None:
    path.write_bytes(probe.canonical_json_bytes(payload))


def _safe_output(path: Path, root: Path) -> None:
    if path.exists():
        raise SystemExit("refusing to overwrite existing output: %s" % path)
    lowered = "/".join(path.parts).lower()
    if "attempt2" in lowered or "two-envelope" in lowered or "recovery_forensics" in lowered:
        raise SystemExit("refusing non-braking/attempt2 output path")
    if path == root or path.parent == path:
        raise SystemExit("output must be a new child receipt directory")


def _make_receipt(stage: Path, root: Path, cells: list) -> Dict[str, Any]:
    manifest = probe.recovery_source_manifest(root)
    _write(stage / "source_manifest.json", manifest)
    summary = verifier.summarize_cells([json.loads((stage / cell["path"]).read_text(encoding="utf-8")) for cell in cells])
    core_handoff = verifier.core_integration_object(summary)
    _write(stage / "summary.json", summary)
    receipt = {
        "schema": probe.RECEIPT_SCHEMA,
        "probe_schema": probe.SCHEMA,
        "subject": "physical_target_ref5in_actor",
        "contract": probe.FROZEN_CONTRACT,
        "git_head": probe.git_head(root),
        "core_base_commit": probe.REQUIRED_CORE_BASE_COMMIT,
        "source_clean": True,
        "source_manifest": "source_manifest.json",
        "source_manifest_sha256": probe.sha256_file(stage / "source_manifest.json"),
        "cells": [dict(cell) for cell in cells],
        "summary_path": "summary.json",
        "summary_sha256": probe.sha256_file(stage / "summary.json"),
        "summary": summary,
        "speed_cells": summary["speed_cells"],
        # Core recovery consumes these two scalar, conservative inputs.  They are derived from
        # the measured lookup; no value is selected after inspecting a gate outcome.
        "decel_p05_mps2": core_handoff["decel_p05_mps2"],
        "stop_time_p95_s": core_handoff["stop_time_p95_s"],
        "measured_speed_to_p95_lookup": core_handoff["measured_speed_to_p95_lookup"],
        "certified_monotone_speed_to_p95_lookup": core_handoff["certified_monotone_speed_to_p95_lookup"],
        "core_integration": core_handoff,
        "fresh_only": True,
        "process_isolation": "one fresh Isaac Gym child per registered speed",
        "completion": "atomic directory rename followed by standalone verification",
    }
    _write(stage / "receipt.json", receipt)
    (stage / "complete.marker").write_text(probe.COMPLETE_MARKER + "\n", encoding="utf-8")
    verifier.verify_receipt(stage, root)
    return receipt


def run(output: Path) -> Dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    python_bin = Path(os.environ.get("NAVRL_BRAKING_PYTHON", sys.executable)).resolve()
    if not python_bin.is_file() or not os.access(str(python_bin), os.X_OK):
        raise SystemExit("selected aerialgym Python is not executable: %s" % python_bin)
    if Path(sys.executable).resolve() != python_bin:
        raise SystemExit("launcher Python does not match NAVRL_BRAKING_PYTHON")
    ninja = os.environ.get("NAVRL_NINJA", "")
    if not ninja or not Path(ninja).is_file():
        raise SystemExit("NAVRL_NINJA must name the pinned ninja executable")
    _safe_output(output, root)
    probe.require_clean_source(root)
    output = output.resolve()
    stage = output.parent / (output.name + ".partial-%d" % os.getpid())
    if stage.exists():
        raise SystemExit("refusing to reuse partial output: %s" % stage)
    stage.mkdir(parents=True)
    try:
        cells_dir = stage / "cells"
        cells_dir.mkdir()
        token = secrets.token_hex(16)
        cells = []
        for speed in probe.REGISTERED_SPEEDS:
            speed_text = format(speed, ".12g")
            name = "speed_%s.json" % speed_text.replace(".", "p")
            cell_path = cells_dir / name
            record_id = secrets.token_hex(16)
            auth_path = stage / ("child_auth_%s.json" % record_id)
            auth = {"schema": probe.CHILD_AUTH_SCHEMA, "token": token, "speed_mps": speed, "record_id": record_id}
            auth_path.write_bytes(probe.canonical_json_bytes(auth))
            os.chmod(str(auth_path), 0o600)
            env = os.environ.copy()
            env.update({
                "AERIAL_GYM_SIM_NAME": "base_sim",
                "NAVRL_ROBOT": "navrl_ref5in_quad",
                "NAVRL_TARGET_DYNAMICS": "physical",
                "NAVRL_TARGET_PATTERN": "waypoint",
                "NAVRL_TARGET_ROUTE_MODE": "off",
                "NAVRL_TARGET_SPEED": speed_text,
                "NAVRL_NUM_BARS": "0",
                "NAVRL_MAX_BARS": "300",
                "NAVRL_BRAKING_CHILD_TOKEN": token,
                "NAVRL_BRAKING_OUTPUT_ROOT": str(stage.resolve()),
                "NAVRL_REQUIRE_SOURCE_ROOT": str(root),
                "PYTHONPATH": str(root) + os.pathsep + str(root / "tools"),
                "NAVRL_BRAKING_PYTHON": str(python_bin),
                "NAVRL_NINJA": str(Path(ninja).resolve()),
            })
            # PyTorch's extension loader discovers ninja through PATH, not NAVRL_NINJA. Pin the
            # selected executable's directory ahead of any inherited toolchain to prevent a stale
            # or missing ninja from changing the Isaac Gym runtime build.
            env["PATH"] = str(Path(ninja).resolve().parent) + os.pathsep + env.get("PATH", "")
            command = [str(python_bin), str(root / "tools/probe_navrl_physical_target_braking.py"), "--output", str(cell_path), "--speed", speed_text, "--envs", str(probe.REGISTERED_ENVS), "--_single-speed", "--_auth-file", str(auth_path)]
            completed = subprocess.run(command, cwd=str(root), env=env, check=False)
            if completed.returncode != 0:
                raise SystemExit("braking speed cell failed: %s" % speed)
            if not cell_path.is_file():
                raise SystemExit("braking speed child produced no receipt: %s" % speed)
            payload = json.loads(cell_path.read_text(encoding="utf-8"))
            verifier.validate_cell(payload)
            cells.append({
                "path": str(cell_path.relative_to(stage)),
                "speed_mps": speed,
                "sha256": probe.sha256_file(cell_path),
                "provenance_sha256": probe.sha256_bytes(probe.canonical_json_bytes(payload["provenance"])),
            })
            auth_path.unlink()
        receipt = _make_receipt(stage, root, cells)
        if output.exists():
            raise SystemExit("refusing to overwrite output created during run: %s" % output)
        os.replace(str(stage), str(output))
        verifier.verify_receipt(output, root)
        return receipt
    except BaseException:
        # A failed or interrupted run is never made visible as a final receipt.
        if stage.exists():
            import shutil
            shutil.rmtree(str(stage))
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.preflight:
        payload = {"schema": probe.SCHEMA, "fresh_only": True, "contract": probe.FROZEN_CONTRACT, "source_paths": list(TOOL_PATHS)}
        print(json.dumps(payload, sort_keys=True))
        return 0
    expected_python = Path(os.environ.get("NAVRL_BRAKING_PYTHON", "")).resolve()
    if not expected_python.is_file() or Path(sys.executable).resolve() != expected_python:
        raise SystemExit("fresh launcher requires pinned NAVRL_BRAKING_PYTHON")
    result = run(Path(args.output))
    print(json.dumps({"verified": True, "schema": result["schema"], "output": str(Path(args.output).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
