#!/usr/bin/env python3
"""Standalone, raw-data-first verifier for the physical-target braking receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from probe_navrl_physical_target_braking import (
    COMPLETE_MARKER,
    FROZEN_CONTRACT,
    REGISTERED_ENVS,
    REGISTERED_SPEEDS,
    RECEIPT_SCHEMA,
    RECOVERY_SOURCE_PATHS,
    EXPECTED_REPO_IMPORTS,
    INITIAL_SPEED_ABS_TOLERANCE_MPS,
    INITIAL_SPEED_REL_TOLERANCE,
    WARMUP_STEPS,
    BRAKE_STEPS_BUDGET,
    PHYSICS_SUBSTEPS,
    PHYSICS_DT_S,
    RL_DT_S,
    SATURATION_MAX,
    STOP_THRESHOLD_MPS,
    SCHEMA,
    TILT_MAX_DEG,
    canonical_json_bytes,
    git_head,
    quantile_stats,
    require_finite,
    sha256_bytes,
    sha256_file,
    source_manifest,
    TOOL_SOURCE_PATHS,
)


def speed_key(value: float) -> str:
    """Canonical decimal key without collapsing the 1.25 lower-contract arm to 1.2."""
    return format(float(value), ".12g")


def _same_contract(observed: Any, expected: Any, path: str = "contract") -> None:
    if isinstance(expected, float):
        if not isinstance(observed, (int, float)) or abs(float(observed) - expected) > 1e-9:
            raise ValueError("%s mismatch: %r != %r" % (path, observed, expected))
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValueError("%s length mismatch" % path)
        for index, (got, want) in enumerate(zip(observed, expected)):
            _same_contract(got, want, "%s[%d]" % (path, index))
        return
    if observed != expected:
        raise ValueError("%s mismatch: %r != %r" % (path, observed, expected))


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ValueError("missing JSON: %s" % path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    require_finite(payload, str(path))
    if canonical_json_bytes(payload) != path.read_bytes():
        raise ValueError("non-canonical JSON: %s" % path)
    if not isinstance(payload, dict):
        raise ValueError("JSON root is not an object: %s" % path)
    return payload


def _validate_provenance(payload: Mapping[str, Any]) -> None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("cell runtime provenance is missing")
    required = (
        "python_executable", "python_executable_sha256", "python_version", "torch_version", "torch_cuda_version",
        "cuda_device", "nvidia_smi_path", "nvidia_smi_sha256", "nvidia_smi_identity",
        "gpu_driver_version", "ninja_path", "ninja_sha256", "ninja_version",
        "selected_python_contract",
    )
    if any(not isinstance(provenance.get(key), str) or not provenance.get(key) for key in required):
        raise ValueError("cell runtime provenance is incomplete")
    if len(provenance["python_executable_sha256"]) != 64:
        raise ValueError("Python executable hash provenance is malformed")
    imported = provenance.get("imported_modules")
    if not isinstance(imported, dict) or not imported:
        raise ValueError("cell import-origin provenance is missing")
    expected_modules = set(EXPECTED_REPO_IMPORTS) | {"isaacgym"}
    if set(imported) != expected_modules:
        raise ValueError("cell import-origin module set drift")
    for module, record in imported.items():
        if not isinstance(module, str) or not isinstance(record, dict) or not record.get("path") or not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise ValueError("cell import-origin provenance is malformed")
        if module in EXPECTED_REPO_IMPORTS and (record.get("path") != EXPECTED_REPO_IMPORTS[module] or record.get("root_bound") is not True):
            raise ValueError("cell repository import origin drift: %s" % module)
        if module == "isaacgym" and record.get("root_bound") is not False:
            raise ValueError("Isaac Gym origin must be external and explicit")
    tools = provenance.get("tool_hashes")
    if not isinstance(tools, dict) or set(tools) != set(TOOL_SOURCE_PATHS):
        raise ValueError("cell tool hash provenance is incomplete")
    if any(not isinstance(value, str) or len(value) != 64 for value in tools.values()):
        raise ValueError("cell tool hash provenance is malformed")


def _close(observed: Any, expected: Any, label: str, tolerance: float = 1e-6) -> None:
    if abs(float(observed) - float(expected)) > tolerance:
        raise ValueError("%s does not match trace-derived value" % label)


def _trace_arrays(trace: Mapping[str, Any], envs: int) -> None:
    for field in ("speed_mps", "position_xy_m", "step_distance_m", "path_distance_m", "contact", "invalid_obb", "motor_saturation_fraction", "max_tilt_deg"):
        values = trace.get(field)
        if not isinstance(values, list) or len(values) != envs:
            raise ValueError("trace field %s has the wrong env count" % field)
    for position in trace["position_xy_m"]:
        if not isinstance(position, list) or len(position) != 2:
            raise ValueError("trace position shape mismatch")


def _validate_trace_and_rows(payload: Mapping[str, Any], speed: float, rows: List[Mapping[str, Any]]) -> None:
    setup = payload.get("setup")
    if not isinstance(setup, dict) or setup.get("mode") != "obstacle_free_center" or int(setup.get("active_bars", -1)) != 0:
        raise ValueError("certified obstacle-free setup is missing")
    if int(setup.get("warmup_steps", -1)) != WARMUP_STEPS or int(setup.get("brake_steps_budget", -1)) != BRAKE_STEPS_BUDGET:
        raise ValueError("warmup/brake budget drift")
    centers = setup.get("center_xy_m")
    clearances = setup.get("center_clearance_to_arena_m")
    if not isinstance(centers, list) or len(centers) != REGISTERED_ENVS or not isinstance(clearances, list) or len(clearances) != REGISTERED_ENVS:
        raise ValueError("setup population is incomplete")
    if any(float(value) <= 0.5 for value in clearances):
        raise ValueError("setup arena clearance gate failed")
    instantiated = payload.get("instantiated")
    if not isinstance(instantiated, dict) or instantiated.get("sim_name") != "base_sim" or int(instantiated.get("envs", -1)) != REGISTERED_ENVS:
        raise ValueError("instantiated runtime attestation is incomplete")
    _close(instantiated.get("controller_dt_s"), FROZEN_CONTRACT["physics_dt_s"], "controller dt", 1e-9)
    if int(instantiated.get("controller_substeps_per_rl_step", -1)) != PHYSICS_SUBSTEPS:
        raise ValueError("controller substep attestation drift")
    _close(instantiated.get("physical_support_xy_m"), FROZEN_CONTRACT["physical_support_xy_m"], "physical support", 1e-6)
    traces = payload.get("physics_samples")
    if not isinstance(traces, list):
        raise ValueError("physics trace is missing")
    warmup = [trace for trace in traces if isinstance(trace, dict) and trace.get("phase") == "warmup"]
    brake = [trace for trace in traces if isinstance(trace, dict) and trace.get("phase") == "brake"]
    if len(warmup) != WARMUP_STEPS * PHYSICS_SUBSTEPS or not (PHYSICS_SUBSTEPS <= len(brake) <= BRAKE_STEPS_BUDGET * PHYSICS_SUBSTEPS):
        raise ValueError("physics trace sample counts are invalid")
    for index, trace in enumerate(warmup, 1):
        if int(trace.get("sample_index", -1)) != index:
            raise ValueError("warmup trace order mismatch")
        _close(trace.get("elapsed_s"), index * PHYSICS_DT_S, "warmup elapsed", 1e-9)
        _trace_arrays(trace, REGISTERED_ENVS)
    for index, trace in enumerate(brake, 1):
        if int(trace.get("sample_index", -1)) != index:
            raise ValueError("brake trace order mismatch")
        _close(trace.get("elapsed_s"), index * PHYSICS_DT_S, "brake elapsed", 1e-9)
        _trace_arrays(trace, REGISTERED_ENVS)
    previous = centers
    previous_path = [0.0] * REGISTERED_ENVS
    for trace in warmup:
        for env_id in range(REGISTERED_ENVS):
            px, py = [float(value) for value in previous[env_id]]
            x, y = [float(value) for value in trace["position_xy_m"][env_id]]
            step = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            _close(trace["step_distance_m"][env_id], step, "warmup path step", 1e-4)
            _close(trace["path_distance_m"][env_id], previous_path[env_id] + step, "warmup path", 1e-4)
        previous = trace["position_xy_m"]
        previous_path = [float(value) for value in trace["path_distance_m"]]
    previous = warmup[-1]["position_xy_m"]
    previous_path = [0.0] * REGISTERED_ENVS
    for trace in brake:
        for env_id in range(REGISTERED_ENVS):
            px, py = [float(value) for value in previous[env_id]]
            x, y = [float(value) for value in trace["position_xy_m"][env_id]]
            _close(trace["step_distance_m"][env_id], ((x - px) ** 2 + (y - py) ** 2) ** 0.5, "brake path step", 1e-4)
            _close(trace["path_distance_m"][env_id], previous_path[env_id] + float(trace["step_distance_m"][env_id]), "brake path", 1e-4)
        previous = trace["position_xy_m"]
        previous_path = [float(value) for value in trace["path_distance_m"]]
    for env_id, value in enumerate(setup.get("warmup_path_distance_m", [])):
        _close(value, warmup[-1]["path_distance_m"][env_id], "setup warmup path", 1e-4)
    for env_id, row in enumerate(sorted(rows, key=lambda item: int(item["env_id"]))):
        if int(row["env_id"]) != env_id:
            raise ValueError("raw env ids are not a unique 0..31 population")
        if abs(float(row["requested_speed_mps"]) - speed) > 1e-12:
            raise ValueError("raw requested speed differs from cell speed")
        warmup_speeds = [float(trace["speed_mps"][env_id]) for trace in warmup]
        brake_speeds = [float(trace["speed_mps"][env_id]) for trace in brake]
        first_stop = next((index for index, value in enumerate(brake_speeds) if value <= STOP_THRESHOLD_MPS), None)
        if first_stop is None or any(value <= STOP_THRESHOLD_MPS for value in brake_speeds[:first_stop]):
            raise ValueError("stop threshold first-crossing is invalid")
        initial = warmup_speeds[-1]
        warmup_error = abs(initial - speed)
        if warmup_error > INITIAL_SPEED_ABS_TOLERANCE_MPS or warmup_error / max(speed, 1e-12) > INITIAL_SPEED_REL_TOLERANCE:
            raise ValueError("initial speed convergence gate failed")
        stop_trace = brake[first_stop]
        path = float(stop_trace["path_distance_m"][env_id])
        if path <= 0.0:
            raise ValueError("stop path distance must be positive")
        derived_decel = initial * initial / (2.0 * path)
        contact = any(bool(trace["contact"][env_id]) for trace in warmup + brake)
        invalid = any(bool(trace["invalid_obb"][env_id]) for trace in warmup + brake)
        brake_sat = float(brake[-1]["motor_saturation_fraction"][env_id])
        brake_tilt = max(float(trace["max_tilt_deg"][env_id]) for trace in brake)
        warmup_sat = float(warmup[-1]["motor_saturation_fraction"][env_id])
        warmup_tilt = max(float(trace["max_tilt_deg"][env_id]) for trace in warmup)
        for field, expected in (
            ("measured_initial_speed_mps", initial),
            ("warmup_final_speed_mps", initial),
            ("warmup_speed_error_mps", warmup_error),
            ("stop_time_s", float(stop_trace["elapsed_s"])),
            ("stop_distance_m", path),
            ("endpoint_displacement_m", ((float(stop_trace["position_xy_m"][env_id][0]) - float(warmup[-1]["position_xy_m"][env_id][0])) ** 2 + (float(stop_trace["position_xy_m"][env_id][1]) - float(warmup[-1]["position_xy_m"][env_id][1])) ** 2) ** 0.5),
            ("max_lateral_deviation_m", max(abs(float(trace["position_xy_m"][env_id][1]) - float(warmup[-1]["position_xy_m"][env_id][1])) for trace in brake)),
            ("effective_deceleration_mps2", derived_decel),
            ("warmup_motor_saturation_fraction", warmup_sat),
            ("warmup_max_tilt_deg", warmup_tilt),
            ("motor_saturation_fraction", brake_sat),
            ("max_tilt_deg", brake_tilt),
        ):
            _close(row.get(field), expected, field)
        if row.get("warmup_converged") is not True or bool(row.get("warmup_contact")) != any(bool(trace["contact"][env_id]) for trace in warmup) or bool(row.get("warmup_invalid_obb")) != any(bool(trace["invalid_obb"][env_id]) for trace in warmup):
            raise ValueError("warmup row attribution is forged")
        if bool(row.get("contact")) != contact or bool(row.get("invalid_obb")) != invalid:
            raise ValueError("safety row attribution is forged")
        if contact or invalid or brake_sat > SATURATION_MAX or brake_tilt > TILT_MAX_DEG:
            raise ValueError("physical safety gate failed")


def validate_cell(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if payload.get("schema") != SCHEMA:
        raise ValueError("cell schema mismatch")
    _same_contract(payload.get("contract"), FROZEN_CONTRACT)
    _validate_provenance(payload)
    source_attestation = payload.get("source_attestation")
    if not isinstance(source_attestation, dict) or source_attestation.get("clean") is not True or not source_attestation.get("git_head") or source_attestation.get("required_core_base_commit") != "dac38227cc3ad2130c84755ef9aab2e75becb9f0":
        raise ValueError("cell source-clean attestation is missing")
    cell = payload.get("cell")
    if not isinstance(cell, dict):
        raise ValueError("cell metadata missing")
    speed = float(cell.get("speed_mps"))
    if speed not in REGISTERED_SPEEDS or int(cell.get("envs", -1)) != REGISTERED_ENVS or int(cell.get("seed", -1)) != int(FROZEN_CONTRACT["seed"]):
        raise ValueError("cell is outside the frozen speed/env/seed grid")
    child_auth = cell.get("child_auth")
    if not isinstance(child_auth, dict) or not child_auth.get("record_id") or not isinstance(child_auth.get("sha256"), str) or len(child_auth["sha256"]) != 64:
        raise ValueError("parent child authorization receipt is missing")
    rows = payload.get("raw_samples")
    if not isinstance(rows, list) or len(rows) != REGISTERED_ENVS or any(not isinstance(row, dict) for row in rows):
        raise ValueError("raw_samples must contain exactly 32 env rows")
    _validate_trace_and_rows(payload, speed, rows)
    rows = sorted(rows, key=lambda item: int(item["env_id"]))
    summary = {
        "speed_mps": speed,
        "envs": len(rows),
        "stop_time_s": quantile_stats([float(row["stop_time_s"]) for row in rows]),
        "stop_distance_m": quantile_stats([float(row["stop_distance_m"]) for row in rows]),
        "max_lateral_deviation_m": quantile_stats([float(row["max_lateral_deviation_m"]) for row in rows]),
        "effective_deceleration_mps2": quantile_stats([float(row["effective_deceleration_mps2"]) for row in rows]),
        "max_motor_saturation_fraction": max(float(row["motor_saturation_fraction"]) for row in rows),
        "max_tilt_deg": max(float(row["max_tilt_deg"]) for row in rows),
        "contact_count": sum(bool(row["contact"]) for row in rows),
        "invalid_obb_count": sum(bool(row["invalid_obb"]) for row in rows),
        "warmup_max_motor_saturation_fraction": max(float(row["warmup_motor_saturation_fraction"]) for row in rows),
        "warmup_max_tilt_deg": max(float(row["warmup_max_tilt_deg"]) for row in rows),
        "warmup_contact_count": sum(bool(row["warmup_contact"]) for row in rows),
        "warmup_invalid_obb_count": sum(bool(row["warmup_invalid_obb"]) for row in rows),
        "gates": {
            "contact": True,
            "invalid_obb": True,
            "motor_saturation": True,
            "tilt": True,
        },
    }
    return summary


def summarize_cells(cells: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(cells) != len(REGISTERED_SPEEDS):
        raise ValueError("the receipt must contain one cell for every registered speed")
    summaries = []
    seen = set()
    for payload in cells:
        summary = validate_cell(payload)
        speed = float(summary["speed_mps"])
        if speed in seen:
            raise ValueError("duplicate speed cell")
        seen.add(speed)
        summaries.append(summary)
    if seen != set(REGISTERED_SPEEDS):
        raise ValueError("speed grid is incomplete")
    summaries.sort(key=lambda item: item["speed_mps"])
    lookup = {}
    for row in summaries:
        key = speed_key(float(row["speed_mps"]))
        lookup[key] = {
            "speed_mps": row["speed_mps"],
            "p95_stop_time_s": row["stop_time_s"]["p95"],
            "p95_stop_distance_m": row["stop_distance_m"]["p95"],
            "p05_effective_deceleration_mps2": row["effective_deceleration_mps2"]["p05"],
        }
    speed_cells = [
        {
            "speed_mps": row["speed_mps"],
            "samples": row["envs"],
            "stop_distance_p95_m": row["stop_distance_m"]["p95"],
            "max_lateral_deviation_p95_m": row["max_lateral_deviation_m"]["p95"],
            "stop_time_p95_s": row["stop_time_s"]["p95"],
            "contact": row["contact_count"],
            "invalid_obb": row["invalid_obb_count"],
            "motor_saturation_fraction": row["max_motor_saturation_fraction"],
            "max_tilt_deg": row["max_tilt_deg"],
        }
        for row in summaries
    ]
    certified_lookup = {}
    running_distance = 0.0
    for key in sorted(lookup, key=lambda value: float(value)):
        running_distance = max(running_distance, float(lookup[key]["p95_stop_distance_m"]))
        certified_lookup[key] = dict(lookup[key], p95_stop_distance_m=running_distance)
    return {
        "schema": SCHEMA,
        "subject": "physical_target_ref5in_actor",
        "speeds_mps": list(REGISTERED_SPEEDS),
        "envs_per_speed": REGISTERED_ENVS,
        "cells": summaries,
        "speed_cells": speed_cells,
        "measured_speed_to_p95_lookup": lookup,
        "certified_monotone_speed_to_p95_lookup": certified_lookup,
        "raw_first": True,
        "quantile_method": "linear_sorted_n_minus_1_probability",
        "gates": {
            "contact_count": 0,
            "invalid_obb_count": 0,
            "motor_saturation_fraction_max": SATURATION_MAX,
            "max_tilt_deg_max": TILT_MAX_DEG,
        },
    }


def core_integration_object(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Canonical handoff object for the recovery task's fail-closed receipt hook."""
    lookup_values = list(summary["measured_speed_to_p95_lookup"].values())
    return {
        "schema": SCHEMA,
        "decel_p05_mps2": min(float(row["p05_effective_deceleration_mps2"]) for row in lookup_values),
        "stop_time_p95_s": max(float(row["p95_stop_time_s"]) for row in lookup_values),
        "measured_speed_to_p95_lookup": summary["measured_speed_to_p95_lookup"],
        "certified_monotone_speed_to_p95_lookup": summary["certified_monotone_speed_to_p95_lookup"],
        "certified_lateral_tube_p95_m": max(float(row["max_lateral_deviation_p95_m"]) for row in summary["speed_cells"]),
        "task_integration": {
            "NAVRL_TARGET_RECOVERY_BRAKE_P05": "decel_p05_mps2",
            "NAVRL_TARGET_RECOVERY_STOP_TIME_P95_S": "stop_time_p95_s",
            "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT": "final receipt JSON path",
            "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256": "SHA256(final receipt JSON bytes)",
            "NAVRL_TARGET_RECOVERY_PROBE_VALIDATED": "1 only after standalone verify_receipt passes",
            "core_distance_lookup": "certified_monotone_speed_to_p95_lookup",
            "certified_lookup_api": "certified_lookup_for_speed(summary, speed_mps)",
        },
    }


def certified_lookup_for_speed(summary: Mapping[str, Any], speed_mps: float) -> Dict[str, Any]:
    """Return the exact canonical certified handoff cell for one registered speed."""
    key = speed_key(float(speed_mps))
    lookup = summary["certified_monotone_speed_to_p95_lookup"]
    if key not in lookup or abs(float(lookup[key]["speed_mps"]) - float(speed_mps)) > 1e-12:
        raise ValueError("speed is not present in the certified braking lookup")
    return dict(lookup[key])


def _verify_git_object(repo_root: Path, recorded_head: str) -> None:
    if len(recorded_head) != 40:
        raise ValueError("receipt git_head is not a full commit id")
    check = subprocess.run(["git", "-C", str(repo_root), "cat-file", "-e", recorded_head + "^{commit}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if check.returncode != 0:
        raise ValueError("receipt git_head is not an available commit object")


def verify_receipt(output: Path, repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Verify an immutable finalized receipt and recompute all statistics from raw cells.

    Recorded git HEAD is required to be a valid commit object, but is deliberately not compared
    to the current HEAD: committing the receipt must not make its provenance unverifiable.
    """
    output = output.resolve()
    root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    if not output.is_dir() or (output / ".partial").exists():
        raise ValueError("output is not a finalized receipt directory")
    marker = output / "complete.marker"
    if not marker.is_file() or marker.read_text(encoding="utf-8") != COMPLETE_MARKER + "\n":
        raise ValueError("completion marker missing or invalid")
    receipt = _read_json(output / "receipt.json")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("probe_schema") != SCHEMA or receipt.get("subject") != "physical_target_ref5in_actor":
        raise ValueError("receipt schema mismatch")
    if receipt.get("core_base_commit") != "dac38227cc3ad2130c84755ef9aab2e75becb9f0" or receipt.get("source_clean") is not True:
        raise ValueError("receipt does not bind the clean integrated core lineage")
    _same_contract(receipt.get("contract"), FROZEN_CONTRACT)
    _verify_git_object(root, str(receipt.get("git_head", "")))
    base_check = subprocess.run(["git", "-C", str(root), "cat-file", "-e", "dac38227cc3ad2130c84755ef9aab2e75becb9f0^{commit}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if base_check.returncode != 0:
        raise ValueError("required integrated core commit is unavailable")
    manifest_rel = str(receipt.get("source_manifest", ""))
    if Path(manifest_rel).is_absolute() or manifest_rel != "source_manifest.json":
        raise ValueError("source manifest must be receipt-relative")
    manifest_path = output / manifest_rel
    manifest = _read_json(manifest_path)
    if sha256_file(manifest_path) != receipt.get("source_manifest_sha256"):
        raise ValueError("source manifest hash mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("source manifest entries are missing")
    recorded_paths = [entry.get("path") for entry in entries if isinstance(entry, dict)]
    if any(not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts for path in recorded_paths):
        raise ValueError("source manifest contains an unsafe path")
    if set(recorded_paths) != set(RECOVERY_SOURCE_PATHS):
        raise ValueError("source manifest path set is not the recovery contract")
    current_manifest = source_manifest(root, tuple(recorded_paths))
    if current_manifest != manifest:
        raise ValueError("runtime source bytes differ from receipt manifest")
    cell_payloads = []
    for cell in receipt.get("cells", []):
        path = Path(str(cell.get("path", "")))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("cell path must be receipt-relative")
        cell_path = output / path
        if sha256_file(cell_path) != cell.get("sha256"):
            raise ValueError("cell hash mismatch: %s" % path)
        payload = _read_json(cell_path)
        summary = validate_cell(payload)
        if abs(float(cell.get("speed_mps")) - float(summary["speed_mps"])) > 1e-12:
            raise ValueError("cell metadata speed mismatch")
        expected_provenance_hash = sha256_bytes(canonical_json_bytes(payload["provenance"]))
        if cell.get("provenance_sha256") != expected_provenance_hash:
            raise ValueError("cell runtime provenance hash mismatch")
        cell_payloads.append(payload)
    if len(cell_payloads) != len(REGISTERED_SPEEDS):
        raise ValueError("receipt must contain exactly four speed cells")
    provenance_records = [payload["provenance"] for payload in cell_payloads]
    if any(record != provenance_records[0] for record in provenance_records[1:]):
        raise ValueError("runtime provenance differs across fresh speed cells")
    source_attestations = [payload["source_attestation"] for payload in cell_payloads]
    if any(record != source_attestations[0] for record in source_attestations[1:]) or any(record.get("git_head") != receipt.get("git_head") for record in source_attestations):
        raise ValueError("source-clean attestation differs across fresh speed cells")
    manifest_hashes = {entry["path"]: entry["sha256"] for entry in manifest["entries"]}
    provenance = provenance_records[0]
    for module, expected_path in EXPECTED_REPO_IMPORTS.items():
        if provenance["imported_modules"][module]["sha256"] != manifest_hashes.get(expected_path):
            raise ValueError("runtime import hash differs from source manifest: %s" % module)
    for relative, digest in provenance["tool_hashes"].items():
        if digest != manifest_hashes.get(relative):
            raise ValueError("runtime tool hash differs from source manifest: %s" % relative)
    summary_path = output / str(receipt.get("summary_path", "summary.json"))
    summary = _read_json(summary_path)
    if sha256_file(summary_path) != receipt.get("summary_sha256"):
        raise ValueError("summary hash mismatch")
    recomputed = summarize_cells(cell_payloads)
    if summary != recomputed or receipt.get("summary") != recomputed:
        raise ValueError("summary is not a raw-data recomputation")
    if receipt.get("speed_cells") != recomputed["speed_cells"]:
        raise ValueError("receipt speed_cells mismatch")
    lookup_values = list(recomputed["measured_speed_to_p95_lookup"].values())
    expected_decel = min(float(row["p05_effective_deceleration_mps2"]) for row in lookup_values)
    expected_time = max(float(row["p95_stop_time_s"]) for row in lookup_values)
    if abs(float(receipt.get("decel_p05_mps2", -1.0)) - expected_decel) > 1e-12:
        raise ValueError("receipt decel_p05_mps2 is not derived from raw cells")
    if abs(float(receipt.get("stop_time_p95_s", -1.0)) - expected_time) > 1e-12:
        raise ValueError("receipt stop_time_p95_s is not derived from raw cells")
    if receipt.get("measured_speed_to_p95_lookup") != recomputed["measured_speed_to_p95_lookup"]:
        raise ValueError("receipt speed lookup mismatch")
    if receipt.get("certified_monotone_speed_to_p95_lookup") != recomputed["certified_monotone_speed_to_p95_lookup"]:
        raise ValueError("receipt certified monotone lookup mismatch")
    if receipt.get("core_integration") != core_integration_object(recomputed):
        raise ValueError("core integration handoff is not canonical")
    return {"verified": True, "schema": receipt["schema"], "summary": recomputed, "core_integration": core_integration_object(recomputed), "git_head": receipt["git_head"]}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", required=True, metavar="RECEIPT_DIR")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = verify_receipt(Path(args.verify))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
