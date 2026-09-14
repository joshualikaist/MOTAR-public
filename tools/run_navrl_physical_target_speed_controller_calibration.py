#!/usr/bin/env python3
"""Fresh-process physical-target speed/gain calibration fixed on 2026-08-26."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import probe_navrl_physical_target_braking as brake  # noqa: E402


SCHEMA = "navrl_physical_target_speed_controller_calibration_v1"
RECEIPT_SCHEMA = "navrl_physical_target_speed_controller_calibration_receipt_v1"
SEED = 829
ENVS = 32
WARMUP_STEPS = 100
BRAKE_STEPS = 100
HORIZONS_S = (5, 6, 8, 10)
CELLS: Tuple[Tuple[float, float], ...] = (
    (2.5, 1.35), (2.5, 1.40), (2.5, 1.45), (2.5, 1.50),
    (3.0, 1.50), (3.5, 1.50),
)
# Stage 2 was frozen only after stage 1 established that the baseline ceiling is below 1.35 m/s
# and velocity gain alone increases oscillation.  It brackets the lower ceiling and tests explicit
# rate-damping interventions.  Stage-1 summaries never consume this grid.
STAGE2_CELLS: Tuple[Tuple[float, float, float], ...] = (
    (2.5, 1.20, 1.0), (2.5, 1.25, 1.0), (2.5, 1.30, 1.0), (2.5, 1.35, 1.0),
    (2.5, 1.50, 1.0), (2.5, 1.50, 1.5), (3.0, 1.50, 1.5), (3.0, 1.50, 2.0),
)
ABS_TOL = 0.05
REL_TOL = 0.10
SAT_MAX = 0.15
TILT_MAX = 60.0
STOP_THRESHOLD = 0.10
PREREG = ROOT / "docs/preregistration_physical_target_speed_controller_calibration_2026-08-26.md"
THIS = Path(__file__).resolve()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=str(ROOT), text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), completed.stderr.strip()))
    return completed.stdout.strip()


def require_clean() -> str:
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError("calibration requires clean committed tracked source: %s" % dirty)
    head = git("rev-parse", "HEAD")
    if git("cat-file", "-t", head) != "commit":
        raise RuntimeError("HEAD is not a commit object")
    return head


def source_manifest() -> Dict[str, Any]:
    required = list(brake.RECOVERY_SOURCE_PATHS) + [str(THIS.relative_to(ROOT)), str(PREREG.relative_to(ROOT))]
    tracked = sorted(set(git("ls-files", *required).splitlines()))
    missing = sorted(set(required) - set(tracked))
    if missing:
        raise RuntimeError("untracked calibration source: %s" % missing)
    entries = []
    for relative in tracked:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError("missing source: %s" % relative)
        entries.append({"path": relative, "sha256": sha256_file(path), "size": path.stat().st_size})
    return {"git_commit": require_clean(), "files": entries, "file_count": len(entries)}


def quantile(values, probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = probability * (len(ordered) - 1)
    lo, hi = int(math.floor(index)), int(math.ceil(index))
    if lo == hi:
        return ordered[lo]
    alpha = index - lo
    return ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha


def cell_key(kp: float, speed: float) -> str:
    return "kp%s_speed%s" % (
        format(kp, ".1f").replace(".", "p"),
        format(speed, ".2f").replace(".", "p"),
    )


def _authorize(path: Path, kp: float, speed: float, rate_scale: float) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA + "_child_auth" or float(payload.get("kp", -1)) != kp \
            or float(payload.get("speed", -1)) != speed \
            or float(payload.get("rate_scale", 1.0)) != rate_scale:
        raise RuntimeError("child authorization mismatch")
    return payload


def _run_child(output: Path, kp: float, speed: float, auth_path: Path,
               rate_scale: float = 1.0) -> None:
    allowed = {(a, b, 1.0) for a, b in CELLS} | set(STAGE2_CELLS)
    if (kp, speed, rate_scale) not in allowed or output.exists():
        raise RuntimeError("child cell/output violates fixed grid")
    auth = _authorize(auth_path, kp, speed, rate_scale)
    source = require_clean()
    os.environ.update({
        "AERIAL_GYM_SIM_NAME": "base_sim",
        "NAVRL_ROBOT": "navrl_ref5in_quad",
        "NAVRL_TARGET_DYNAMICS": "physical",
        "NAVRL_TARGET_PATTERN": "waypoint",
        "NAVRL_TARGET_ROUTE_MODE": "off",
        "NAVRL_TARGET_SPEED": format(speed, ".2f"),
        "NAVRL_TARGET_VEL_KP": format(kp, ".1f"),
        "NAVRL_NUM_BARS": "0", "NAVRL_MAX_BARS": "300",
        "NAVRL_ARENA_XY": "40", "NAVRL_ARENA_Z": "3",
        "NAVRL_BAR_POOL": "bars_h3", "NAVRL_PLACEMENT_MODE": "navrl_band",
        "NAVRL_PLACEMENT_TOUCH_M": "0.4", "NAVRL_PLACEMENT_GAP_M": "1.6",
        "NAVRL_BAR_X_MIN": "0", "NAVRL_BAR_X_MAX": "1",
        "NAVRL_TARGET_MAX_ACCEL": "4.0", "NAVRL_TARGET_MAX_TURN_RATE_DEG": "150",
        "NAVRL_TARGET_LOOKAHEAD_S": "1.0",
    })
    sys.argv[:] = [sys.argv[0]]
    import isaacgym  # noqa: F401
    import torch
    from aerial_gym.registry.task_registry import task_registry

    task = task_registry.make_task("navrl_task", seed=SEED, num_envs=ENVS, headless=True, use_warp=True)
    if hasattr(task, "seed"):
        task.seed(SEED)
    if hasattr(task, "_set_active_bars"):
        task._set_active_bars(0)
    task.reset()
    original_kp = brake.FROZEN_CONTRACT["physical_velocity_kp"]
    try:
        brake.FROZEN_CONTRACT["physical_velocity_kp"] = kp
        instantiated = brake.attest_instantiated_task(task, speed)
    finally:
        brake.FROZEN_CONTRACT["physical_velocity_kp"] = original_kp
    ctrl = task._target_controller
    if abs(float(ctrl.velocity_kp) - kp) > 1e-6:
        raise RuntimeError("instantiated velocity gain drift")
    base_rate_kp = ctrl.rate_kp.detach().clone()
    ctrl.rate_kp.mul_(float(rate_scale))
    expected_rate_kp = [float(v) * float(rate_scale) for v in brake.FROZEN_CONTRACT["physical_rate_kp"]]
    actual_rate_kp = [float(v) for v in ctrl.rate_kp.detach().cpu().reshape(-1).tolist()]
    if any(abs(a - b) > 1e-6 for a, b in zip(actual_rate_kp, expected_rate_kp)):
        raise RuntimeError("diagnostic rate gain scale drift")
    bmin, bmax = task.obs_dict["env_bounds_min"], task.obs_dict["env_bounds_max"]
    center = ((bmin + bmax) * 0.5).clone()
    center[:, 2] = float(task.task_config.flight_altitude)
    center[:, 2] = torch.maximum(center[:, 2], bmin[:, 2] + 0.5)
    center[:, 2] = torch.minimum(center[:, 2], bmax[:, 2] - 0.5)
    ctrl.reset_idx(torch.arange(ENVS, device=task.device))
    ctrl.position[:] = center
    ctrl.linvel.zero_(); ctrl.angvel_world.zero_()
    task.sim_env.IGE_env.write_to_sim(); task.sim_env.IGE_env.refresh_tensors()
    support = task._physical_target_support_xyz()
    setup_margin = torch.minimum(center[:, :2] - bmin[:, :2] - support[:, :2],
                                 bmax[:, :2] - center[:, :2] - support[:, :2]).amin(dim=1)
    if bool((setup_margin <= 0.5).any()):
        raise RuntimeError("center setup clearance failed")
    altitude = torch.full((ENVS,), float(task.task_config.flight_altitude), device=task.device)
    command = torch.zeros((ENVS, 3), device=task.device); command[:, 0] = speed
    recorder = brake._PhysicsSubstepRecorder(ctrl, bmin, bmax, support, torch, initial_xy=center)
    task.sim_env.set_physics_step_callback(recorder)
    recorder.begin_phase("warmup")
    for _ in range(WARMUP_STEPS):
        recorder.begin_interval(); ctrl.begin_control_interval(); ctrl.set_command(command, altitude)
        task.sim_env.step(actions=torch.zeros((ENVS, 4), device=task.device))
    warmup = recorder.export()
    warmup_contact = recorder.phase_contact_any.detach().clone()
    warmup_invalid = recorder.phase_invalid_any.detach().clone()
    warmup_diag = ctrl.diagnostics()
    warmup_sat = warmup_diag["motor_saturation_fraction"].detach().cpu().tolist()
    warmup_tilt = warmup_diag["max_tilt_deg"].detach().cpu().tolist()
    speed_series = [[float(v) for v in sample["speed_mps"]] for sample in warmup]
    horizon = {}
    for seconds in HORIZONS_S:
        idx = seconds * 100 - 1
        values = speed_series[idx]
        errors = [abs(value - speed) for value in values]
        horizon[str(seconds)] = {
            "speed_min_mps": min(values), "speed_mean_mps": sum(values) / len(values),
            "speed_max_mps": max(values), "error_max_mps": max(errors),
            "all_within_gate": all(error <= ABS_TOL and error / speed <= REL_TOL for error in errors),
        }
    sustained = True
    for values in speed_series[400:500]:
        if any(abs(value - speed) > ABS_TOL or abs(value - speed) / speed > REL_TOL for value in values):
            sustained = False; break
    overshoot = max(max(values) - speed for values in speed_series)
    ctrl.substeps.zero_(); ctrl.saturation_substeps.zero_(); ctrl.max_tilt_seen_rad.zero_()
    ctrl.velocity_error_integral.zero_()
    start_pos = ctrl.position[:, :2].detach().clone()
    start_speed = ctrl.linvel[:, :2].norm(dim=1).detach().clone()
    recorder.begin_phase("brake")
    for _ in range(BRAKE_STEPS):
        recorder.begin_interval(); ctrl.begin_control_interval(); ctrl.set_command(torch.zeros_like(command), altitude)
        task.sim_env.step(actions=torch.zeros((ENVS, 4), device=task.device))
        if bool(recorder.stopped.all()):
            break
    stopped = bool(recorder.stopped.all())
    brake_samples = recorder.export()
    brake_diag = ctrl.diagnostics()
    brake_sat = brake_diag["motor_saturation_fraction"].detach().cpu().tolist()
    brake_tilt = brake_diag["max_tilt_deg"].detach().cpu().tolist()
    stop_distance = [float(v) for v in recorder.stop_distance.detach().cpu().tolist()]
    stop_time = [float(v) for v in recorder.stop_time.detach().cpu().tolist()]
    brake_positions = [sample["position_xy_m"] for sample in brake_samples]
    lateral = [
        max(abs(float(sample[env][1]) - float(start_pos[env, 1].item())) for sample in brake_positions)
        for env in range(ENVS)
    ]
    contact = warmup_contact | recorder.phase_contact_any
    invalid = warmup_invalid | recorder.phase_invalid_any
    five_pass = bool(horizon["5"]["all_within_gate"] and sustained)
    safety_pass = bool(
        not contact.any().item() and not invalid.any().item() and stopped
        and max(warmup_sat) <= SAT_MAX and max(brake_sat) <= SAT_MAX
        and max(warmup_tilt) <= TILT_MAX and max(brake_tilt) <= TILT_MAX
        and overshoot <= ABS_TOL
    )
    payload = {
        "schema": SCHEMA + "_cell", "git_commit": source, "auth": auth,
        "condition": {"velocity_kp": kp, "requested_speed_mps": speed,
                      "rate_kp_scale": rate_scale, "seed": SEED, "envs": ENVS},
        "instantiated": dict(instantiated, physical_velocity_kp=kp,
                             base_rate_kp=[float(v) for v in base_rate_kp.cpu().reshape(-1).tolist()],
                             diagnostic_rate_kp=actual_rate_kp),
        "horizon": horizon, "sustained_4_to_5s": sustained, "overshoot_max_mps": overshoot,
        "warmup": {"contact_count": int(warmup_contact.sum().item()),
                   "invalid_count": int(warmup_invalid.sum().item()),
                   "saturation_max": max(warmup_sat), "tilt_max_deg": max(warmup_tilt)},
        "braking": {"all_stopped": stopped, "stop_time_s": stop_time,
                    "stop_distance_m": stop_distance, "lateral_deviation_m": lateral,
                    "saturation_max": max(brake_sat), "tilt_max_deg": max(brake_tilt)},
        "contact_count": int(contact.sum().item()), "invalid_count": int(invalid.sum().item()),
        "five_second_tracking_pass": five_pass, "safety_pass": safety_pass,
        "cell_pass": bool(five_pass and safety_pass),
        "warmup_speed_trace_mps": speed_series,
        "provenance": brake.runtime_provenance(ROOT),
    }
    brake.require_finite(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical(payload))


def _validate_cell(payload: Mapping[str, Any], kp: float, speed: float) -> None:
    if payload.get("schema") != SCHEMA + "_cell":
        raise RuntimeError("cell schema drift")
    condition = payload.get("condition", {})
    if float(condition.get("velocity_kp", -1)) != kp or float(condition.get("requested_speed_mps", -1)) != speed:
        raise RuntimeError("cell condition drift")
    trace = payload.get("warmup_speed_trace_mps")
    if not isinstance(trace, list) or len(trace) != WARMUP_STEPS * brake.PHYSICS_SUBSTEPS \
            or any(not isinstance(row, list) or len(row) != ENVS for row in trace):
        raise RuntimeError("cell trace shape drift")
    for seconds in HORIZONS_S:
        values = [float(v) for v in trace[seconds * 100 - 1]]
        errors = [abs(v - speed) for v in values]
        observed = payload["horizon"][str(seconds)]
        expected_pass = all(error <= ABS_TOL and error / speed <= REL_TOL for error in errors)
        if bool(observed["all_within_gate"]) != expected_pass or abs(float(observed["error_max_mps"]) - max(errors)) > 1e-7:
            raise RuntimeError("horizon summary drift")


def summarize(cells: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_key = {}
    for payload, (kp, speed) in zip(cells, CELLS):
        _validate_cell(payload, kp, speed)
        braking = payload["braking"]
        by_key[cell_key(kp, speed)] = {
            "velocity_kp": kp, "requested_speed_mps": speed,
            "cell_pass": bool(payload["cell_pass"]),
            "horizon": payload["horizon"],
            "overshoot_max_mps": float(payload["overshoot_max_mps"]),
            "p95_stop_distance_m": quantile(braking["stop_distance_m"], 0.95),
            "p95_stop_time_s": quantile(braking["stop_time_s"], 0.95),
            "p95_lateral_deviation_m": quantile(braking["lateral_deviation_m"], 0.95),
            "warmup_saturation_max": float(payload["warmup"]["saturation_max"]),
            "braking_saturation_max": float(braking["saturation_max"]),
            "max_tilt_deg": max(float(payload["warmup"]["tilt_max_deg"]), float(braking["tilt_max_deg"])),
        }
    baseline = [row for row in by_key.values() if row["velocity_kp"] == 2.5 and row["cell_pass"]]
    baseline_ceiling = max((row["requested_speed_mps"] for row in baseline), default=None)
    reference = by_key[cell_key(2.5, 1.5)]
    selected = None
    for kp in (3.0, 3.5):
        row = by_key[cell_key(kp, 1.5)]
        if (row["cell_pass"] and row["p95_stop_distance_m"] <= 1.10 * reference["p95_stop_distance_m"]
                and row["p95_lateral_deviation_m"] <= reference["p95_lateral_deviation_m"] + 0.05):
            selected = kp; break
    return {
        "schema": SCHEMA + "_summary", "cells": by_key,
        "baseline_attainable_speed_mps": baseline_ceiling,
        "selected_controller_velocity_kp": selected,
        "decision": (
            "BOTH_FOLLOWUPS_ELIGIBLE" if baseline_ceiling is not None and selected is not None
            else "LOWER_CONTRACT_ONLY" if baseline_ceiling is not None
            else "CONTROLLER_ONLY" if selected is not None else "NO_FOLLOWUP_ELIGIBLE"
        ),
        "claim_boundary": "calibration_only_no_ppo_no_hardware_claim",
    }


def verify(directory: Path) -> Dict[str, Any]:
    directory = directory.resolve()
    marker = directory / "complete.marker"
    receipt_path = directory / "receipt.json"
    if marker.read_text(encoding="utf-8").strip() != "COMPLETE" or not receipt_path.is_file():
        raise RuntimeError("incomplete calibration artifact")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise RuntimeError("receipt schema drift")
    cells = []
    for record, condition in zip(receipt.get("cells", []), CELLS):
        path = directory / record["path"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError("cell hash drift")
        cells.append(json.loads(path.read_text(encoding="utf-8")))
    recomputed = summarize(cells)
    if recomputed != receipt.get("summary"):
        raise RuntimeError("summary semantic drift")
    manifest_path = directory / "source_manifest.json"
    if sha256_file(manifest_path) != receipt.get("source_manifest_sha256"):
        raise RuntimeError("manifest hash drift")
    return receipt


def run(output: Path) -> Dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise RuntimeError("refusing existing output")
    require_clean()
    stage = output.parent / (".%s.partial-%d" % (output.name, os.getpid()))
    if stage.exists():
        raise RuntimeError("stale partial output")
    stage.mkdir(parents=True)
    try:
        manifest = source_manifest()
        (stage / "source_manifest.json").write_bytes(canonical(manifest))
        cells_dir = stage / "cells"; cells_dir.mkdir()
        records, payloads = [], []
        python = Path(sys.executable).resolve()
        ninja = Path(os.environ.get("NAVRL_NINJA", str(python.parent / "ninja"))).resolve()
        if not ninja.is_file() or not os.access(str(ninja), os.X_OK):
            raise RuntimeError("pinned ninja is unavailable")
        for kp, speed in CELLS:
            key = cell_key(kp, speed)
            auth_path = stage / ("auth_%s.json" % key)
            auth = {"schema": SCHEMA + "_child_auth", "token": secrets.token_hex(16),
                    "kp": kp, "speed": speed}
            auth_path.write_bytes(canonical(auth)); os.chmod(str(auth_path), 0o600)
            cell_path = cells_dir / (key + ".json")
            env = os.environ.copy()
            env.update({"PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT) + os.pathsep + str(TOOLS),
                        "PATH": str(ninja.parent) + os.pathsep + env.get("PATH", ""),
                        "NAVRL_NINJA": str(ninja), "NAVRL_REQUIRE_SOURCE_ROOT": str(ROOT)})
            command = [str(python), str(THIS), "--_child", "--output", str(cell_path),
                       "--kp", str(kp), "--speed", str(speed), "--auth", str(auth_path)]
            completed = subprocess.run(command, cwd=str(ROOT), env=env, check=False)
            if completed.returncode:
                raise RuntimeError("calibration child failed: %s" % key)
            payload = json.loads(cell_path.read_text(encoding="utf-8"))
            _validate_cell(payload, kp, speed)
            payloads.append(payload)
            records.append({"path": str(cell_path.relative_to(stage)), "sha256": sha256_file(cell_path),
                            "velocity_kp": kp, "requested_speed_mps": speed})
            auth_path.unlink()
        summary = summarize(payloads)
        receipt = {"schema": RECEIPT_SCHEMA, "git_commit": git("rev-parse", "HEAD"),
                   "preregistration": str(PREREG.relative_to(ROOT)), "cells": records,
                   "summary": summary, "source_manifest": "source_manifest.json",
                   "source_manifest_sha256": sha256_file(stage / "source_manifest.json")}
        (stage / "summary.json").write_bytes(canonical(summary))
        receipt["summary_sha256"] = sha256_file(stage / "summary.json")
        (stage / "receipt.json").write_bytes(canonical(receipt))
        (stage / "complete.marker").write_text("COMPLETE\n", encoding="utf-8")
        verify(stage)
        os.replace(str(stage), str(output))
        return verify(output)
    except BaseException:
        if stage.exists():
            shutil.rmtree(str(stage))
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--verify")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kp", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--speed", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--auth", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--rate-scale", type=float, default=1.0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.preflight:
        print(json.dumps({"schema": SCHEMA, "cells": CELLS, "clean": not bool(git("status", "--porcelain", "--untracked-files=no"))}, sort_keys=True))
        return 0
    if args._child:
        _run_child(Path(args.output), float(args.kp), float(args.speed), args.auth,
                   float(args.rate_scale))
        return 0
    if args.verify:
        print(json.dumps(verify(Path(args.verify))["summary"], indent=2, sort_keys=True))
        return 0
    if not args.output:
        parser.error("--output is required")
    print(json.dumps(run(Path(args.output))["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
