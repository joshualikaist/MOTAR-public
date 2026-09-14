#!/usr/bin/env python3
"""Run the complete software-only sim-to-real readiness chain.

This command deliberately uses synthetic fixtures.  It validates that the future real-data path
is executable and fail-closed, but it never promotes synthetic numbers to hardware evidence and it
never starts PPO.  The output is a small, hash-bound receipt that can be reviewed before replacing
the fixtures with real rosbag/CSV data.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/navrl_sim2real_software_preflight_2026-08-24"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_transport_csv(path: Path) -> None:
    fields = [
        "topic", "seq", "source_stamp_ns", "host_receive_stamp_ns", "frame_id",
        "parent_frame_id", "sync_group", "policy_input_stamp_ns", "command_publish_stamp_ns",
        "condition",
    ]
    rows = [
        ["camera", 0, 1_000_000_000, 1_001_000_000, "camera_optical_frame", "base_link", 0, "", "", "fixture"],
        ["lidar", 0, 1_002_000_000, 1_004_000_000, "lidar_frame", "base_link", 0, "", "", "fixture"],
        ["ego_state", 0, 1_001_000_000, 1_002_500_000, "base_link", "odom", 0, "", "", "fixture"],
        ["policy_input", 0, 1_002_000_000, 1_006_000_000, "base_link", "odom", "", 1_002_000_000, "", "fixture"],
        ["command", 0, 1_006_000_000, 1_007_000_000, "base_link", "odom", "", 1_002_000_000, 1_006_000_000, "fixture"],
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def _write_measurement_csv(path: Path) -> None:
    fields = [
        "trial_id", "distance_m", "lighting", "motion", "target_present", "detected",
        "range_valid", "ground_truth_azimuth_deg", "estimated_azimuth_deg",
        "ground_truth_range_m", "estimated_range_m", "confidence", "source_stamp_ns",
        "host_receive_stamp_ns",
    ]
    rows = [
        ["fixture_t0", 8, "normal", "static", 1, 1, 1, 2.0, 2.4, 8.0, 8.2, 0.9, 1_000_000_000, 1_002_000_000],
        ["fixture_t0", 8, "normal", "static", 1, 1, 1, 2.0, 2.2, 8.0, 7.9, 0.8, 1_010_000_000, 1_012_000_000],
        ["fixture_t1", 20, "low", "lateral", 1, 1, 0, -4.0, -4.8, 20.0, "", 0.7, 2_000_000_000, 2_004_000_000],
        ["fixture_t1", 20, "low", "lateral", 1, 0, 0, -4.0, "", 20.0, "", 0.0, 2_010_000_000, 2_014_000_000],
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def _write_two_zone_fixture(path: Path, contract_path: Path) -> None:
    contract_path.write_text(
        json.dumps({
            "schema_version": 1,
            "source_kind": "real_log",
            "near_boundary_m": 12.0,
            "far_range_policy": "invalid",
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    events = [
        {
            "trial_id": "fixture_t0", "timestamp_ns": 1_000_000_000, "zone": "far",
            "azimuth_deg": 2.0, "elevation_deg": 0.0, "bearing_rate_dps": 0.5,
            "confidence": 0.8, "range_valid": False, "range_m": None, "range_sigma_m": None,
            "measurement_age_ms": 3.0, "track_covariance": 0.1,
        },
        {
            "trial_id": "fixture_t0", "timestamp_ns": 1_010_000_000, "zone": "near",
            "azimuth_deg": 2.2, "elevation_deg": 0.0, "bearing_rate_dps": 0.6,
            "confidence": 0.9, "range_valid": True, "range_m": 8.2, "range_sigma_m": 0.2,
            "measurement_age_ms": 4.0, "track_covariance": 0.1,
        },
    ]
    with path.open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, sort_keys=True) + "\n")


def run(output: Path) -> Dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    telemetry = _load("navrl_telemetry_preflight", ROOT / "tools/navrl_sim2real_telemetry.py")
    ingest = _load("navrl_ingest_preflight", ROOT / "tools/navrl_sim2real_ingest.py")
    profile = _load("navrl_profile_preflight", ROOT / "tools/navrl_sensor_profile.py")
    replay = _load("navrl_replay_preflight", ROOT / "tools/navrl_two_zone_replay.py")

    transport_csv = output / "synthetic_transport.csv"
    transport_jsonl = output / "synthetic_transport.jsonl"
    telemetry_json = output / "telemetry_report.json"
    _write_transport_csv(transport_csv)
    ingest.convert_csv(transport_csv, transport_jsonl, run_id="synthetic_transport", source_kind="synthetic_fixture")
    manifest, events = telemetry.read_jsonl(transport_jsonl)
    telemetry_report = telemetry.validate_events(
        events,
        manifest=manifest,
        contract=telemetry.TelemetryContract(max_sync_skew_ns=20_000_000, max_sensor_to_host_latency_ns=500_000_000),
        source_kind="synthetic_fixture",
    ).as_dict()
    telemetry_json.write_text(json.dumps(telemetry_report, indent=2) + "\n", encoding="utf-8")

    measurements = output / "synthetic_measurements.csv"
    profile_json = output / "synthetic_sensor_profile.json"
    _write_measurement_csv(measurements)
    sensor_profile = profile.build_profile(measurements, profile_json, source_kind="synthetic_fixture", run_id="synthetic_sensor_profile")

    contract_json = output / "two_zone_contract.json"
    replay_jsonl = output / "two_zone_replay.jsonl"
    replay_report_json = output / "two_zone_report.json"
    _write_two_zone_fixture(replay_jsonl, contract_json)
    replay_report = replay.validate_replay(replay_jsonl, contract_json)
    replay_report_json.write_text(json.dumps(replay_report, indent=2) + "\n", encoding="utf-8")

    mode_summary = ROOT / "results/navrl_ref5in_symmetric_corridor_mode_probe_seed431/summary.json"
    mode_payload = json.loads(mode_summary.read_text(encoding="utf-8")) if mode_summary.is_file() else {}
    physical_summary = ROOT / "results/navrl_physical_target_speed_envelope_post_wall_brake_seed509/summary.json"
    physical_payload = json.loads(physical_summary.read_text(encoding="utf-8")) if physical_summary.is_file() else {}

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_status": "SYNTHETIC_ONLY",
        "hardware": {"status": "BLOCKED_NO_ASSEMBLED_HARDWARE_OR_SENSOR_LOG"},
        "steps": {
            "telemetry": {"verdict": telemetry_report["verdict"], "claim_status": telemetry_report["claim_status"]},
            "ingest": {"verdict": "PASS", "claim_status": "SYNTHETIC_ONLY", "events": manifest["event_count"]},
            "sensor_profile": {"verdict": "PASS", "claim_status": sensor_profile["claim_status"], "trials": sensor_profile["profile"]["trial_count"]},
            "two_zone_replay": {"verdict": replay_report["verdict"], "claim_status": "SYNTHETIC_ONLY", "events": replay_report["event_count"]},
            "mode_probe": {"verdict": mode_payload.get("interpretation", "MISSING"), "authority": "DIAGNOSTIC_ONLY"},
            "physical_target_gate": {"all_cells_pass": physical_payload.get("all_cells_pass"), "authority": "NO_PPO_PERMISSION"},
            "fresh_ppo": {"verdict": "BLOCKED", "reason": "physical_gate_and_real_sensor_contract_not_passed"},
        },
        "next_real_inputs": ["hardware_manifest", "calibration_yaml_and_sha256", "210_trial_sensor_log", "timestamp_contract"],
        "artifacts": {},
    }
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "summary.json":
            payload["artifacts"][path.name] = _sha256(path)
    summary = output / "summary.json"
    summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt = output / "summary.receipt.json"
    receipt.write_text(json.dumps({
        "schema_version": 1,
        "summary_sha256": _sha256(summary),
        "tool_sha256": _sha256(Path(__file__).resolve()),
        "claim_status": "SYNTHETIC_ONLY",
    }, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT
    try:
        payload = run(output)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"software preflight error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(output), "claim_status": payload["claim_status"], "fresh_ppo": payload["steps"]["fresh_ppo"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
