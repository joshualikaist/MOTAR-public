#!/usr/bin/env python3
"""Verify and summarize a receipt-bound NavRL joint-speed evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "aerial_gym/task/navrl_task/joint_speed_telemetry.py"
)
SPEC = importlib.util.spec_from_file_location("navrl_joint_speed_metrics", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
assess_preregistered_speed_gate = MODULE.assess_preregistered_speed_gate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    parser.add_argument("receipt_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result_path = args.result_json.resolve()
    receipt_path = args.receipt_json.resolve()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if Path(receipt.get("result_json", "")).resolve() != result_path:
        raise SystemExit("joint-speed receipt points to a different result")
    if receipt.get("result_sha256") != sha256(result_path):
        raise SystemExit("joint-speed result SHA does not match its receipt")
    if int(receipt.get("schema_version", -1)) != 2:
        raise SystemExit("unsupported evaluation receipt schema")
    manifest_path = Path(receipt.get("runtime_source_manifest", "")).resolve()
    if sha256(manifest_path) != receipt.get("runtime_source_manifest_sha256"):
        raise SystemExit("runtime source manifest SHA does not match the receipt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    module_relative = "aerial_gym/task/navrl_task/joint_speed_telemetry.py"
    module_entry = next(
        (row for row in manifest.get("runtime_files", []) if row.get("path") == module_relative),
        None,
    )
    if module_entry is None:
        raise SystemExit("joint telemetry source is absent from the runtime source receipt")
    module_snapshot = (manifest_path.parent / module_entry["snapshot"]).resolve()
    if sha256(module_snapshot) != module_entry.get("sha256") or sha256(MODULE_PATH) != module_entry.get(
        "sha256"
    ):
        raise SystemExit("joint telemetry source changed after the receipt-bound evaluation")
    if int(payload.get("actual_episodes", -1)) != int(receipt.get("actual_episodes", -2)):
        raise SystemExit("receipt/result episode count mismatch")
    if receipt.get("joint_speed_telemetry") is not True:
        raise SystemExit("evaluation receipt does not attest joint-speed telemetry")
    if int((payload.get("condition") or {}).get("bars", -1)) != 205:
        raise SystemExit("joint-speed gate is preregistered for exactly 205 bars")
    if (payload.get("condition") or {}).get("speed_governor_mode") != "riskcap":
        raise SystemExit("joint-speed gate requires the frozen riskcap condition")
    if (payload.get("condition") or {}).get("joint_speed_telemetry") is not True:
        raise SystemExit("bulk condition does not attest joint-speed telemetry")

    joint = payload.get("joint_speed_allocation")
    if not isinstance(joint, dict):
        raise SystemExit("bulk result lacks joint_speed_allocation telemetry")
    assessment = assess_preregistered_speed_gate(joint)
    output = {
        "schema_version": 1,
        "source_result": str(result_path),
        "source_result_sha256": sha256(result_path),
        "source_receipt": str(receipt_path),
        "source_receipt_sha256": sha256(receipt_path),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "runtime_source_manifest_sha256": receipt.get("runtime_source_manifest_sha256"),
        "checkpoint_sha256": receipt.get("source_checkpoint_sha256"),
        "condition": {
            "bars": 205,
            "seed": receipt.get("seed"),
            "episodes": receipt.get("actual_episodes"),
            "action_selection": receipt.get("action_selection"),
            "speed_governor_mode": receipt.get("speed_governor_mode"),
        },
        "assessment": assessment,
        "joint_speed_allocation": joint,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(assessment, sort_keys=True))


if __name__ == "__main__":
    main()
