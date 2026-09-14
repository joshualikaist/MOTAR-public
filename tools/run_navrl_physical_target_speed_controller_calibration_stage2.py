#!/usr/bin/env python3
"""Run and verify the preregistered stage-2 speed/damping calibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import run_navrl_physical_target_speed_controller_calibration as cal  # noqa: E402


SCHEMA = "navrl_physical_target_speed_controller_calibration_stage2_v1"
PREREG = ROOT / "docs/preregistration_physical_target_speed_controller_calibration_stage2_2026-08-26.md"
THIS = Path(__file__).resolve()


def key(kp: float, speed: float, rate: float) -> str:
    return "kp%s_speed%s_rate%s" % tuple(
        format(v, ".2f").replace(".", "p") for v in (kp, speed, rate)
    )


def original_gate(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload["horizon"]["5"]["all_within_gate"]
        and payload["sustained_4_to_5s"]
        and payload["contact_count"] == 0 and payload["invalid_count"] == 0
        and payload["braking"]["all_stopped"]
        and payload["warmup"]["saturation_max"] <= cal.SAT_MAX
        and payload["braking"]["saturation_max"] <= cal.SAT_MAX
        and payload["warmup"]["tilt_max_deg"] <= cal.TILT_MAX
        and payload["braking"]["tilt_max_deg"] <= cal.TILT_MAX
    )


def validate_cell(payload: Mapping[str, Any], condition) -> None:
    kp, speed, rate = condition
    cal._validate_cell(payload, kp, speed)
    observed = payload.get("condition", {})
    if float(observed.get("rate_kp_scale", -1)) != rate:
        raise RuntimeError("stage2 rate condition drift")


def summarize(payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = {}
    for payload, condition in zip(payloads, cal.STAGE2_CELLS):
        validate_cell(payload, condition)
        kp, speed, rate = condition
        braking = payload["braking"]
        rows[key(kp, speed, rate)] = {
            "velocity_kp": kp, "requested_speed_mps": speed, "rate_kp_scale": rate,
            "original_gate_pass": original_gate(payload),
            "overshoot_max_mps": float(payload["overshoot_max_mps"]),
            "p95_stop_distance_m": cal.quantile(braking["stop_distance_m"], 0.95),
            "p95_stop_time_s": cal.quantile(braking["stop_time_s"], 0.95),
            "p95_lateral_deviation_m": cal.quantile(braking["lateral_deviation_m"], 0.95),
            "horizon": payload["horizon"],
            "max_tilt_deg": max(float(payload["warmup"]["tilt_max_deg"]),
                                float(braking["tilt_max_deg"])),
            "max_saturation_fraction": max(float(payload["warmup"]["saturation_max"]),
                                             float(braking["saturation_max"])),
        }
    lower = [row for row in rows.values() if row["velocity_kp"] == 2.5
             and row["rate_kp_scale"] == 1.0 and row["requested_speed_mps"] <= 1.35
             and row["original_gate_pass"]]
    ceiling = max((row["requested_speed_mps"] for row in lower), default=None)
    reference = rows[key(2.5, 1.5, 1.0)]
    selected = None
    for condition in ((2.5, 1.5, 1.5), (3.0, 1.5, 1.5), (3.0, 1.5, 2.0)):
        row = rows[key(*condition)]
        if (row["original_gate_pass"] and row["overshoot_max_mps"] <= 0.15 * 1.5
                and row["overshoot_max_mps"] <= 0.5 * reference["overshoot_max_mps"]
                and row["p95_stop_distance_m"] <= 1.10 * reference["p95_stop_distance_m"]
                and row["p95_lateral_deviation_m"] <= reference["p95_lateral_deviation_m"] + 0.05):
            selected = {"velocity_kp": condition[0], "rate_kp_scale": condition[2]}
            break
    return {"schema": SCHEMA + "_summary", "cells": rows,
            "baseline_attainable_speed_mps": ceiling, "selected_damped_controller": selected,
            "lower_contract_eligible": ceiling is not None,
            "controller_contract_eligible": selected is not None,
            "claim_boundary": "calibration_only_no_ppo_no_hardware_claim"}


def verify(directory: Path) -> Dict[str, Any]:
    directory = directory.resolve()
    if (directory / "complete.marker").read_text(encoding="utf-8").strip() != "COMPLETE":
        raise RuntimeError("incomplete stage2 artifact")
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("schema") != SCHEMA + "_receipt":
        raise RuntimeError("stage2 receipt schema drift")
    payloads = []
    for record, condition in zip(receipt["cells"], cal.STAGE2_CELLS):
        path = directory / record["path"]
        if cal.sha256_file(path) != record["sha256"]:
            raise RuntimeError("stage2 cell hash drift")
        payload = json.loads(path.read_text(encoding="utf-8")); validate_cell(payload, condition)
        payloads.append(payload)
    if summarize(payloads) != receipt.get("summary"):
        raise RuntimeError("stage2 semantic summary drift")
    if cal.sha256_file(directory / "source_manifest.json") != receipt["source_manifest_sha256"]:
        raise RuntimeError("stage2 source manifest drift")
    return receipt


def manifest() -> Dict[str, Any]:
    required = list(cal.brake.RECOVERY_SOURCE_PATHS) + [
        str(Path(cal.__file__).resolve().relative_to(ROOT)), str(THIS.relative_to(ROOT)),
        str(PREREG.relative_to(ROOT)),
        "docs/preregistration_physical_target_speed_controller_calibration_2026-08-26.md",
    ]
    tracked = sorted(set(cal.git("ls-files", *required).splitlines()))
    if set(required) - set(tracked):
        raise RuntimeError("stage2 source is not fully tracked")
    return {"git_commit": cal.require_clean(), "files": [
        {"path": rel, "sha256": cal.sha256_file(ROOT / rel), "size": (ROOT / rel).stat().st_size}
        for rel in tracked
    ]}


def run(output: Path) -> Dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise RuntimeError("refusing existing stage2 output")
    cal.require_clean()
    stage = output.parent / (".%s.partial-%d" % (output.name, os.getpid()))
    stage.mkdir(parents=True); (stage / "cells").mkdir()
    try:
        source = manifest(); (stage / "source_manifest.json").write_bytes(cal.canonical(source))
        python = Path(sys.executable).resolve()
        ninja = Path(os.environ.get("NAVRL_NINJA", str(python.parent / "ninja"))).resolve()
        if not ninja.is_file():
            raise RuntimeError("pinned ninja unavailable")
        records, payloads = [], []
        for kp, speed, rate in cal.STAGE2_CELLS:
            name = key(kp, speed, rate); cell = stage / "cells" / (name + ".json")
            auth = stage / ("auth_%s.json" % name)
            auth.write_bytes(cal.canonical({"schema": cal.SCHEMA + "_child_auth",
                "token": secrets.token_hex(16), "kp": kp, "speed": speed, "rate_scale": rate}))
            os.chmod(str(auth), 0o600)
            env = os.environ.copy(); env.update({"PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(ROOT) + os.pathsep + str(TOOLS),
                "PATH": str(ninja.parent) + os.pathsep + env.get("PATH", ""),
                "NAVRL_NINJA": str(ninja), "NAVRL_REQUIRE_SOURCE_ROOT": str(ROOT)})
            cmd = [str(python), str(Path(cal.__file__).resolve()), "--_child", "--output", str(cell),
                   "--kp", str(kp), "--speed", str(speed), "--rate-scale", str(rate), "--auth", str(auth)]
            completed = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
            if completed.returncode:
                raise RuntimeError("stage2 child failed: %s" % name)
            payload = json.loads(cell.read_text(encoding="utf-8")); validate_cell(payload, (kp, speed, rate))
            payloads.append(payload); records.append({"path": str(cell.relative_to(stage)),
                "sha256": cal.sha256_file(cell), "condition": [kp, speed, rate]})
            auth.unlink()
        summary = summarize(payloads)
        (stage / "summary.json").write_bytes(cal.canonical(summary))
        receipt = {"schema": SCHEMA + "_receipt", "git_commit": cal.git("rev-parse", "HEAD"),
            "preregistration": str(PREREG.relative_to(ROOT)), "cells": records, "summary": summary,
            "summary_sha256": cal.sha256_file(stage / "summary.json"),
            "source_manifest_sha256": cal.sha256_file(stage / "source_manifest.json")}
        (stage / "receipt.json").write_bytes(cal.canonical(receipt))
        (stage / "complete.marker").write_text("COMPLETE\n", encoding="utf-8")
        verify(stage); os.replace(str(stage), str(output)); return verify(output)
    except BaseException:
        if stage.exists(): shutil.rmtree(str(stage))
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output"); parser.add_argument("--verify"); parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        print(json.dumps({"schema": SCHEMA, "cells": cal.STAGE2_CELLS,
                          "clean": not bool(cal.git("status", "--porcelain", "--untracked-files=no"))}, sort_keys=True)); return 0
    if args.verify:
        print(json.dumps(verify(Path(args.verify))["summary"], indent=2, sort_keys=True)); return 0
    if not args.output: parser.error("--output required")
    print(json.dumps(run(Path(args.output))["summary"], indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
