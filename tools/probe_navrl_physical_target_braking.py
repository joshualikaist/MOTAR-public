#!/usr/bin/env python3
"""Fresh-only zero-command braking probe for the physical NavRL target.

This module is deliberately independent of the pursuer braking calibration.  It measures the
already-instantiated :class:`PhysicalTargetController` by commanding a physical target at one
registered speed and then submitting a zero *target velocity* command.  It never changes the
target task implementation, planner, observations, reward, or termination logic.

The simulator path is intentionally lazy: importing this module is CPU-safe and is sufficient
for the receipt validator and contract tests.  A real probe must be run in a fresh Isaac Gym
process for each speed; the launcher enforces that rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA = "navrl_target_recovery_braking_probe_v1"
RECEIPT_SCHEMA = "navrl_target_recovery_braking_receipt_v1"
COMPLETE_MARKER = "COMPLETE"
CONTRACT_VARIANT = os.environ.get(
    "NAVRL_TARGET_BRAKING_CONTRACT_VARIANT", "canonical_1p5"
).strip().lower()
if CONTRACT_VARIANT == "canonical_1p5":
    REGISTERED_SPEEDS = (0.6, 0.9, 1.2, 1.5)
elif CONTRACT_VARIANT == "baseline_1p25":
    REGISTERED_SPEEDS = (0.6, 0.9, 1.2, 1.25)
else:
    raise RuntimeError("unknown NAVRL_TARGET_BRAKING_CONTRACT_VARIANT=%r" % CONTRACT_VARIANT)
REGISTERED_ENVS = 32
PHYSICS_DT_S = 0.01
PHYSICS_SUBSTEPS = 10
RL_DT_S = 0.1
STOP_THRESHOLD_MPS = 0.10
SATURATION_MAX = 0.15
TILT_MAX_DEG = 60.0
FINITE_EPS = 1e-12
WARMUP_STEPS = 50
BRAKE_STEPS_BUDGET = 100
INITIAL_SPEED_ABS_TOLERANCE_MPS = 0.05
INITIAL_SPEED_REL_TOLERANCE = 0.10
REQUIRED_CORE_BASE_COMMIT = "dac38227cc3ad2130c84755ef9aab2e75becb9f0"
CHILD_AUTH_SCHEMA = "navrl_target_recovery_braking_child_auth_v1"

# This is an attestation tuple, not a tuning surface.  Values are repeated here so a result is
# refused when a future task/config silently changes the physical experiment.
FROZEN_CONTRACT: Dict[str, Any] = {
    "contract_variant": CONTRACT_VARIANT,
    "sim_name": "base_sim",
    "robot": "navrl_ref5in_quad",
    "target_dynamics": "physical",
    "target_pattern": "waypoint",
    "route_mode": "off",
    "seed": 827,
    "envs": REGISTERED_ENVS,
    "num_bars": 0,
    "max_bars": 300,
    "arena_xy_m": 40.0,
    "arena_z_m": 3.0,
    "bar_pool": "bars_h3",
    "placement_mode": "navrl_band",
    "placement_touch_m": 0.4,
    "placement_gap_m": 1.6,
    "bar_x_min": 0.0,
    "bar_x_max": 1.0,
    "target_max_accel_mps2": 4.0,
    "target_max_turn_rate_deg_s": 150.0,
    "target_lookahead_s": 1.0,
    "target_obstacle_clearance_m": 0.77,
    "physical_box_xyz_m": [0.28, 0.28, 0.12],
    "physical_support_xy_m": 0.2068816087,
    "physical_mass_kg": 1.20,
    "physical_motor_arm_xy_m": 0.0777817,
    "physical_max_motor_thrust_n": 9.60,
    "physical_motor_tau_s": 0.04,
    "physical_yaw_torque_ratio": 0.01,
    "physical_max_tilt_deg": 45.0,
    "physical_velocity_kp": 2.5,
    "physical_altitude_kp": 4.0,
    "physical_attitude_kp": [0.08, 0.08, 0.04],
    "physical_rate_kp": [0.04, 0.04, 0.03],
    "tracking_margin_m": 0.45,
    "boundary_margin_m": 0.75,
    "route_resolution_m": 0.25,
    "route_goal_tol_m": 0.05,
    "route_max_expansions": 50000,
    "route_max_waypoints": 128,
    "route_replan_cooldown_steps": 10,
    "route_min_goal_distance_m": 6.0,
    "route_goal_exclusion_m": 1.0,
    "physics_dt_s": PHYSICS_DT_S,
    "physics_substeps": PHYSICS_SUBSTEPS,
    "rl_step_dt_s": RL_DT_S,
    "setup_mode": "obstacle_free_center",
    "warmup_steps": WARMUP_STEPS,
    "brake_steps_budget": BRAKE_STEPS_BUDGET,
    "initial_speed_abs_tolerance_mps": INITIAL_SPEED_ABS_TOLERANCE_MPS,
    "initial_speed_rel_tolerance": INITIAL_SPEED_REL_TOLERANCE,
}

CORE_PATHS = (
    "aerial_gym/__init__.py",
    "aerial_gym/task/navrl_task/navrl_task.py",
    "aerial_gym/task/navrl_task/target_motion.py",
    "aerial_gym/task/navrl_task/target_route_planner.py",
    "aerial_gym/task/navrl_task/physical_target.py",
    "aerial_gym/config/task_config/navrl_task_config.py",
    "aerial_gym/config/robot_config/navrl_ref5in_quad_config.py",
    "aerial_gym/config/sim_config/base_sim_config.py",
    "resources/robots/quad/quad_navrl_ref5in.urdf",
)
TOOL_SOURCE_PATHS = (
    "tools/probe_navrl_physical_target_braking.py",
    "tools/verify_navrl_physical_target_braking.py",
    "tools/run_navrl_physical_target_braking_v2_fresh.py",
    "tools/run_navrl_physical_target_braking_v2_fresh.sh",
    "docs/preregistration_navrl_physical_target_braking_2026-08-25.md",
) + ((
    "tools/run_navrl_physical_target_braking_lower1p25_fresh.sh",
    "docs/preregistration_navrl_physical_target_braking_lower1p25_2026-08-26.md",
) if CONTRACT_VARIANT == "baseline_1p25" else ())
RECOVERY_SOURCE_PATHS = CORE_PATHS + TOOL_SOURCE_PATHS
EXPECTED_REPO_IMPORTS = {
    "aerial_gym": "aerial_gym/__init__.py",
    "aerial_gym.task.navrl_task.navrl_task": "aerial_gym/task/navrl_task/navrl_task.py",
    "aerial_gym.task.navrl_task.target_motion": "aerial_gym/task/navrl_task/target_motion.py",
    "aerial_gym.task.navrl_task.target_route_planner": "aerial_gym/task/navrl_task/target_route_planner.py",
    "aerial_gym.task.navrl_task.physical_target": "aerial_gym/task/navrl_task/physical_target.py",
    "aerial_gym.config.task_config.navrl_task_config": "aerial_gym/config/task_config/navrl_task_config.py",
    "aerial_gym.config.sim_config.base_sim_config": "aerial_gym/config/sim_config/base_sim_config.py",
    "aerial_gym.config.robot_config.navrl_ref5in_quad_config": "aerial_gym/config/robot_config/navrl_ref5in_quad_config.py",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return the only JSON representation accepted by the receipt contract."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(isinstance(k, str) and _is_finite(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(_is_finite(v) for v in value)
    return False


def require_finite(value: Any, label: str = "payload") -> None:
    if not _is_finite(value):
        raise ValueError("non-finite or unsupported value in %s" % label)


def quantile(values: Sequence[float], probability: float) -> float:
    """NumPy-compatible linear quantile without importing NumPy in CPU verification."""
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(probability)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def quantile_stats(values: Sequence[float]) -> Dict[str, float]:
    require_finite(list(values), "quantile values")
    return {
        "p05": quantile(values, 0.05),
        "p50": quantile(values, 0.50),
        "p95": quantile(values, 0.95),
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_manifest(repo_root: Optional[Path] = None, extra_paths: Iterable[str] = ()) -> Dict[str, Any]:
    root = (repo_root or _repo_root()).resolve()
    paths = list(CORE_PATHS) + list(extra_paths)
    entries = []
    for relative in sorted(set(paths)):
        path = root / relative
        if not path.is_file():
            raise ValueError("required source is missing: %s" % relative)
        entries.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {
        "schema": "navrl_target_recovery_braking_source_manifest_v1",
        "root_policy": "repository-relative exact paths",
        "entries": entries,
    }


def recovery_source_manifest(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    return source_manifest(repo_root, TOOL_SOURCE_PATHS)


def git_head(repo_root: Optional[Path] = None) -> str:
    root = str((repo_root or _repo_root()).resolve())
    completed = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise ValueError("cannot attest git HEAD: %s" % completed.stderr.strip())
    head = completed.stdout.strip()
    if len(head) != 40 or any(ch not in "0123456789abcdef" for ch in head):
        raise ValueError("invalid git HEAD")
    return head


def require_clean_source(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = (repo_root or _repo_root()).resolve()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if status.returncode != 0:
        raise ValueError("cannot attest source cleanliness: %s" % status.stderr.strip())
    allowed_output = os.environ.get("NAVRL_BRAKING_OUTPUT_ROOT", "")
    allowed_path = Path(allowed_output).resolve() if allowed_output else None
    dirty_lines = []
    for line in status.stdout.splitlines():
        recorded = line[3:].split(" -> ")[-1].strip() if len(line) >= 4 else line.strip()
        candidate = (root / recorded).resolve()
        if allowed_path is not None:
            try:
                candidate.relative_to(allowed_path)
                continue
            except ValueError:
                pass
        dirty_lines.append(line)
    if dirty_lines:
        raise ValueError("source tree is dirty before GPU execution: %s" % " | ".join(dirty_lines))
    ancestry = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", REQUIRED_CORE_BASE_COMMIT, "HEAD"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if ancestry.returncode != 0:
        raise ValueError("required core base %s is not an ancestor" % REQUIRED_CORE_BASE_COMMIT)
    return {
        "clean": True,
        "git_head": git_head(root),
        "required_core_base_commit": REQUIRED_CORE_BASE_COMMIT,
    }


def verify_child_auth(auth_file: Path, speed: float) -> Dict[str, str]:
    token = os.environ.get("NAVRL_BRAKING_CHILD_TOKEN", "")
    if not token or not auth_file.is_file():
        raise ValueError("missing parent-only child authorization")
    try:
        auth = json.loads(auth_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("invalid child authorization") from exc
    if auth.get("schema") != CHILD_AUTH_SCHEMA or auth.get("token") != token:
        raise ValueError("child authorization token mismatch")
    if abs(float(auth.get("speed_mps", -1.0)) - speed) > 1e-12 or not auth.get("record_id"):
        raise ValueError("child authorization speed/record mismatch")
    if canonical_json_bytes(auth) != auth_file.read_bytes():
        raise ValueError("child authorization is not canonical")
    return {"record_id": str(auth["record_id"]), "sha256": sha256_file(auth_file)}


def _exact_float(actual: Any, expected: float, label: str, tolerance: float = 0.0) -> None:
    try:
        observed = float(actual)
    except (TypeError, ValueError):
        raise ValueError("%s is not numeric" % label)
    if not math.isfinite(observed) or abs(observed - expected) > tolerance:
        raise ValueError("%s drift: observed=%r expected=%r tolerance=%r" % (label, observed, expected, tolerance))


def attest_instantiated_task(task: Any, expected_speed: Optional[float] = None) -> Dict[str, Any]:
    """Fail-closed attestation of the instantiated task/controller, used only in GPU child."""
    cfg = task.task_config
    if str(getattr(cfg, "robot_name", "")) != FROZEN_CONTRACT["robot"]:
        raise ValueError("instantiated robot drift")
    tm = getattr(task, "tm", None)
    if tm is None:
        raise ValueError("target-motion config was not instantiated")
    for attr, key in (
        ("max_accel", "target_max_accel_mps2"),
        ("max_turn_rate_deg", "target_max_turn_rate_deg_s"),
        ("avoidance_lookahead_s", "target_lookahead_s"),
        ("obstacle_clearance", "target_obstacle_clearance_m"),
        ("physical_tracking_margin", "tracking_margin_m"),
        ("physical_boundary_margin", "boundary_margin_m"),
        ("route_resolution_m", "route_resolution_m"),
        ("route_goal_tolerance_m", "route_goal_tol_m"),
        ("route_min_goal_distance_m", "route_min_goal_distance_m"),
        ("route_goal_exclusion_radius_m", "route_goal_exclusion_m"),
    ):
        _exact_float(getattr(tm, attr, None), FROZEN_CONTRACT[key], "tm." + attr)
    for attr, key in (("route_max_expansions", "route_max_expansions"), ("route_max_waypoints", "route_max_waypoints"), ("route_replan_cooldown_steps", "route_replan_cooldown_steps")):
        if int(getattr(tm, attr, -1)) != int(FROZEN_CONTRACT[key]):
            raise ValueError("tm.%s drift" % attr)
    if str(getattr(tm, "dynamics", "")) != FROZEN_CONTRACT["target_dynamics"] or str(getattr(tm, "pattern", "")) != FROZEN_CONTRACT["target_pattern"]:
        raise ValueError("target dynamics/pattern drift")
    if str(getattr(task, "_target_route_mode", "off")) != FROZEN_CONTRACT["route_mode"]:
        raise ValueError("route mode drift")
    box = list(getattr(tm, "physical_box_xyz", ()))
    if len(box) != 3:
        raise ValueError("physical box shape drift")
    for index, (observed, expected) in enumerate(zip(box, FROZEN_CONTRACT["physical_box_xyz_m"])):
        _exact_float(observed, expected, "tm.physical_box_xyz[%d]" % index)
    if expected_speed is not None:
        _exact_float(getattr(tm, "speed_fixed", None), expected_speed, "tm.speed_fixed")
    sim_name = getattr(cfg, "sim_name", None)
    if sim_name != FROZEN_CONTRACT["sim_name"]:
        raise ValueError("instantiated sim_name drift")
    for attr, key in (
        ("physical_mass", "physical_mass_kg"),
        ("physical_motor_arm_xy", "physical_motor_arm_xy_m"),
        ("physical_max_motor_thrust", "physical_max_motor_thrust_n"),
        ("physical_motor_tau", "physical_motor_tau_s"),
        ("physical_yaw_torque_ratio", "physical_yaw_torque_ratio"),
        ("physical_max_tilt_deg", "physical_max_tilt_deg"),
        ("physical_velocity_kp", "physical_velocity_kp"),
        ("physical_altitude_kp", "physical_altitude_kp"),
    ):
        _exact_float(getattr(tm, attr, None), FROZEN_CONTRACT[key], "tm." + attr)
    controller = getattr(task, "_target_controller", None)
    if controller is None:
        raise ValueError("physical target controller was not instantiated")
    for attr, key in (("attitude_kp", "physical_attitude_kp"), ("rate_kp", "physical_rate_kp")):
        tensor = getattr(controller, attr, None)
        try:
            actual = tensor.detach().cpu().reshape(-1).tolist()
        except (AttributeError, RuntimeError):
            raise ValueError("controller.%s is not a finite gain vector" % attr)
        expected = FROZEN_CONTRACT[key]
        if len(actual) != len(expected):
            raise ValueError("%s length drift" % attr)
        for index, (observed, want) in enumerate(zip(actual, expected)):
            _exact_float(observed, want, "%s[%d]" % (attr, index), 1e-6)
    if int(getattr(task, "num_envs", -1)) != REGISTERED_ENVS:
        raise ValueError("instantiated env count drift")
    _exact_float(getattr(controller, "dt", None), PHYSICS_DT_S, "controller.dt")
    support = getattr(task, "_target_route_support_xy", None)
    if support is not None:
        try:
            support_values = support[0].detach().cpu().tolist()
        except (AttributeError, IndexError, TypeError):
            raise ValueError("instantiated support tensor is malformed")
        if len(support_values) != 2:
            raise ValueError("instantiated support shape drift")
        for index, observed in enumerate(support_values):
            # The braking probe intentionally runs route-off with zero bars, so the route manager
            # cache is zero by construction. The certified support remains the geometry-derived
            # contract; route-on children must populate and attest the nonzero cache separately.
            if abs(float(observed)) <= 1e-8 and str(getattr(task, "_target_route_mode", "off")) == "off":
                continue
            _exact_float(observed, FROZEN_CONTRACT["physical_support_xy_m"], "support_xy[%d]" % index, 1e-6)
    return {
        "sim_name": sim_name,
        "envs": int(task.num_envs),
        "controller_dt_s": float(controller.dt),
        "controller_substeps_per_rl_step": PHYSICS_SUBSTEPS,
        "physical_box_xyz_m": list(FROZEN_CONTRACT["physical_box_xyz_m"]),
        "physical_support_xy_m": FROZEN_CONTRACT["physical_support_xy_m"],
    }


def runtime_provenance(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Collect required software/GPU provenance; missing GPU identity is a hard error."""
    root = (repo_root or _repo_root()).resolve()
    import importlib
    import platform

    try:
        import torch
    except Exception as exc:
        raise ValueError("torch import failed: %s" % exc)
    if not bool(torch.cuda.is_available()):
        raise ValueError("CUDA is unavailable; physical probe is GPU-only")
    expected_python = os.environ.get("NAVRL_BRAKING_PYTHON", "")
    if expected_python and Path(sys.executable).resolve() != Path(expected_python).resolve():
        raise ValueError("selected Python executable drift")
    smi = shutil.which("nvidia-smi")
    if not smi:
        raise ValueError("nvidia-smi is required for GPU provenance")
    query = subprocess.run([smi, "--query-gpu=driver_version,name,uuid", "--format=csv,noheader"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if query.returncode != 0 or not query.stdout.strip():
        raise ValueError("nvidia-smi GPU identity query failed")
    gpu_rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    driver_version = gpu_rows[0].split(",", 1)[0].strip() if gpu_rows else ""
    if not driver_version:
        raise ValueError("nvidia-smi driver identity is empty")
    imported = {}
    module_names = (
        "isaacgym",
        "aerial_gym",
        "aerial_gym.task.navrl_task.navrl_task",
        "aerial_gym.task.navrl_task.target_motion",
        "aerial_gym.task.navrl_task.target_route_planner",
        "aerial_gym.task.navrl_task.physical_target",
        "aerial_gym.config.task_config.navrl_task_config",
        "aerial_gym.config.sim_config.base_sim_config",
        "aerial_gym.config.robot_config.navrl_ref5in_quad_config",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        origin = Path(str(getattr(module, "__file__", ""))).resolve()
        if not origin.is_file():
            raise ValueError("import origin is not a file: %s" % origin)
        try:
            relative_origin = origin.relative_to(root)
            origin_record = str(relative_origin)
            root_bound = True
        except ValueError:
            if module_name != "isaacgym":
                raise ValueError("import origin is outside repository: %s" % origin)
            origin_record = str(origin)
            root_bound = False
        imported[module_name] = {"path": origin_record, "sha256": sha256_file(origin), "root_bound": root_bound}
    ninja = os.environ.get("NAVRL_NINJA", "") or shutil.which("ninja")
    if not ninja:
        raise ValueError("ninja is required for runtime provenance")
    ninja_version = subprocess.run([ninja, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if ninja_version.returncode != 0 or not ninja_version.stdout.strip():
        raise ValueError("ninja version query failed")
    tool_hashes = {}
    for relative in TOOL_SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise ValueError("missing bound tool: %s" % relative)
        tool_hashes[relative] = sha256_file(path)
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": sha256_file(Path(sys.executable)),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(getattr(torch.version, "cuda", None)),
        "cuda_device": str(torch.cuda.get_device_name(torch.cuda.current_device())),
        "nvidia_smi_path": str(Path(smi).resolve()),
        "nvidia_smi_sha256": sha256_file(Path(smi)),
        "nvidia_smi_identity": query.stdout.strip(),
        "gpu_driver_version": driver_version,
        "ninja_path": str(Path(ninja).resolve()),
        "ninja_sha256": sha256_file(Path(ninja)),
        "ninja_version": ninja_version.stdout.strip(),
        "tool_hashes": tool_hashes,
        "imported_modules": imported,
        "selected_python_contract": str(Path(expected_python).resolve()) if expected_python else str(Path(sys.executable).resolve()),
    }


def _invalid_obb(task: Any, torch: Any) -> Any:
    """Return a finite/arena/physical-support invalid mask for raw safety telemetry."""
    position = task._target_controller.position
    support = task._physical_target_support_xyz()
    bmin = task.obs_dict["env_bounds_min"]
    bmax = task.obs_dict["env_bounds_max"]
    invalid = ~torch.isfinite(position).all(dim=1)
    invalid |= ((position - support < bmin) | (position + support > bmax)).any(dim=1)
    return invalid


class _PhysicsSubstepRecorder:
    """Transparent callback proxy: controller force semantics stay byte-for-byte delegated."""

    def __init__(self, controller: Any, bmin: Any, bmax: Any, support: Any, torch: Any, initial_xy: Any = None) -> None:
        self.controller = controller
        self.bmin = bmin
        self.bmax = bmax
        self.support = support
        self.torch = torch
        self.initial_xy = initial_xy
        self.phase = ""
        self.interval_index = 0
        self.substep_index = 0
        self.previous_position = None
        self.path_distance = None
        self.last = None
        self.samples = []
        self.phase_contact_any = None
        self.phase_invalid_any = None
        self.stopped = None
        self.stop_time = None
        self.stop_distance = None
        self.stop_position = None

    def begin_phase(self, phase: str) -> None:
        self.phase = phase
        self.interval_index = 0
        self.substep_index = 0
        if phase == "warmup" and self.initial_xy is not None:
            self.previous_position = self.initial_xy[:, :2].detach().clone()
        else:
            self.previous_position = self.controller.position[:, :2].detach().clone()
        self.path_distance = self.torch.zeros(
            self.controller.position.shape[0], device=self.controller.position.device
        )
        self.last = None
        self.samples = []
        n = self.controller.position.shape[0]
        self.phase_contact_any = self.torch.zeros(n, dtype=self.torch.bool, device=self.controller.position.device)
        self.phase_invalid_any = self.torch.zeros(n, dtype=self.torch.bool, device=self.controller.position.device)
        self.stopped = self.torch.zeros(n, dtype=self.torch.bool, device=self.controller.position.device)
        self.stop_time = self.torch.full((n,), float("nan"), device=self.controller.position.device)
        self.stop_distance = self.torch.zeros(n, device=self.controller.position.device)
        self.stop_position = self.torch.zeros((n, 2), device=self.controller.position.device)

    def begin_interval(self) -> None:
        self.interval_index += 1

    def __call__(self) -> None:
        self.controller()

    def post_physics_step(self) -> None:
        self.controller.post_physics_step()
        position = self.controller.position[:, :2].detach().clone()
        step_distance = (position - self.previous_position).norm(dim=1)
        self.path_distance = self.path_distance + step_distance
        self.previous_position = position
        self.substep_index += 1
        invalid = ~torch_isfinite(self.controller.position, self.torch).all(dim=1)
        invalid |= ((self.controller.position - self.support < self.bmin) | (self.controller.position + self.support > self.bmax)).any(dim=1)
        diag = self.controller.diagnostics()
        contact = self.controller.contact_seen.detach().clone()
        invalid = invalid.detach()
        self.phase_contact_any |= contact
        self.phase_invalid_any |= invalid
        if self.phase == "brake":
            newly = (~self.stopped) & (self.controller.linvel[:, :2].norm(dim=1) <= STOP_THRESHOLD_MPS)
            self.stop_time = self.torch.where(newly, self.torch.full_like(self.stop_time, self.substep_index * PHYSICS_DT_S), self.stop_time)
            self.stop_distance = self.torch.where(newly, self.path_distance, self.stop_distance)
            self.stop_position[newly] = position[newly]
            self.stopped |= newly
        self.last = {
            "speed_mps": self.controller.linvel[:, :2].norm(dim=1).detach(),
            "position_xy_m": position,
            "step_distance_m": step_distance.detach(),
            "path_distance_m": self.path_distance.detach().clone(),
            "contact": contact,
            "invalid_obb": invalid.detach(),
            "motor_saturation_fraction": diag["motor_saturation_fraction"].detach().clone(),
            "max_tilt_deg": diag["max_tilt_deg"].detach().clone(),
            "sample_index": self.substep_index,
            "elapsed_s": (self.interval_index * PHYSICS_SUBSTEPS - (PHYSICS_SUBSTEPS - self.substep_index)) * PHYSICS_DT_S,
        }
        self.samples.append(self.last)

    def export(self) -> List[Dict[str, Any]]:
        if not self.samples:
            return []
        fields = ("speed_mps", "position_xy_m", "step_distance_m", "path_distance_m", "contact", "invalid_obb", "motor_saturation_fraction", "max_tilt_deg")
        stacked = {field: self.torch.stack([sample[field] for sample in self.samples], dim=0).detach().cpu() for field in fields}
        result = []
        for index in range(len(self.samples)):
            result.append({
                "phase": self.phase,
                "sample_index": index + 1,
                "elapsed_s": float((index + 1) * PHYSICS_DT_S),
                "speed_mps": [float(value) for value in stacked["speed_mps"][index].tolist()],
                "position_xy_m": [[float(x), float(y)] for x, y in stacked["position_xy_m"][index].tolist()],
                "step_distance_m": [float(value) for value in stacked["step_distance_m"][index].tolist()],
                "path_distance_m": [float(value) for value in stacked["path_distance_m"][index].tolist()],
                "contact": [bool(value) for value in stacked["contact"][index].tolist()],
                "invalid_obb": [bool(value) for value in stacked["invalid_obb"][index].tolist()],
                "motor_saturation_fraction": [float(value) for value in stacked["motor_saturation_fraction"][index].tolist()],
                "max_tilt_deg": [float(value) for value in stacked["max_tilt_deg"][index].tolist()],
            })
        return result


def torch_isfinite(value: Any, torch: Any) -> Any:
    return torch.isfinite(value)


def run_speed_cell(speed: float, output: Path, envs: int = REGISTERED_ENVS, warmup_steps: int = WARMUP_STEPS, brake_steps: int = BRAKE_STEPS_BUDGET, auth_file: Optional[Path] = None) -> None:
    """Run exactly one speed arm in a fresh Isaac Gym process.

    The caller has already isolated this process.  No PPO action is used to produce target
    motion: the target controller receives a world-frame velocity command, then zero velocity.
    """
    if envs != REGISTERED_ENVS or speed not in REGISTERED_SPEEDS:
        raise ValueError("speed cell does not match the frozen grid")
    if output.exists():
        raise ValueError("refusing to overwrite cell: %s" % output)
    if auth_file is None:
        raise ValueError("parent child authorization is required")
    child_auth = verify_child_auth(auth_file, speed)
    source_attestation = require_clean_source(_repo_root())
    os.environ.update({
        "AERIAL_GYM_SIM_NAME": "base_sim",
        "NAVRL_ROBOT": "navrl_ref5in_quad",
        "NAVRL_TARGET_DYNAMICS": "physical",
        "NAVRL_TARGET_PATTERN": "waypoint",
        "NAVRL_TARGET_ROUTE_MODE": "off",
        "NAVRL_TARGET_SPEED": str(speed),
        "NAVRL_NUM_BARS": "0",
        "NAVRL_MAX_BARS": "300",
        "NAVRL_ARENA_XY": "40",
        "NAVRL_ARENA_Z": "3",
        "NAVRL_BAR_POOL": "bars_h3",
        "NAVRL_PLACEMENT_MODE": "navrl_band",
        "NAVRL_PLACEMENT_TOUCH_M": "0.4",
        "NAVRL_PLACEMENT_GAP_M": "1.6",
        "NAVRL_BAR_X_MIN": "0",
        "NAVRL_BAR_X_MAX": "1",
        "NAVRL_TARGET_MAX_ACCEL": "4.0",
        "NAVRL_TARGET_MAX_TURN_RATE_DEG": "150",
        "NAVRL_TARGET_LOOKAHEAD_S": "1.0",
    })
    # Isaac Gym consumes command-line flags while importing; hide probe flags first.
    sys.argv[:] = [sys.argv[0]]
    import isaacgym  # noqa: F401
    import torch
    from aerial_gym.registry.task_registry import task_registry

    task = task_registry.make_task("navrl_task", seed=827, num_envs=envs, headless=True, use_warp=True)
    if hasattr(task, "seed"):
        task.seed(827)
    if hasattr(task, "_set_active_bars"):
        task._set_active_bars(0)
    task.reset()
    instantiated = attest_instantiated_task(task, speed)
    ctrl = task._target_controller
    bmin = task.obs_dict["env_bounds_min"]
    bmax = task.obs_dict["env_bounds_max"]
    center = ((bmin + bmax) * 0.5).clone()
    center[:, 2] = float(task.task_config.flight_altitude)
    center[:, 2] = torch.maximum(center[:, 2], bmin[:, 2] + 0.5)
    center[:, 2] = torch.minimum(center[:, 2], bmax[:, 2] - 0.5)
    ctrl.reset_idx(torch.arange(envs, device=task.device))
    # reset_idx restores the task's sampled target state; apply the declared obstacle-free center
    # pose after reset so the recorder baseline, setup receipt, and first PhysX substep agree.
    ctrl.position[:] = center
    ctrl.linvel.zero_()
    ctrl.angvel_world.zero_()
    task.sim_env.IGE_env.write_to_sim()
    task.sim_env.IGE_env.refresh_tensors()
    support = task._physical_target_support_xyz()
    setup_margin = torch.minimum(
        center[:, :2] - bmin[:, :2] - support[:, :2],
        bmax[:, :2] - center[:, :2] - support[:, :2],
    ).amin(dim=1)
    if bool((setup_margin <= 0.5).any()):
        raise RuntimeError("obstacle-free center setup lacks certified arena clearance")
    altitude = torch.full((envs,), float(task.task_config.flight_altitude), device=task.device)
    direction = torch.zeros((envs, 3), device=task.device)
    direction[:, 0] = float(speed)
    recorder = _PhysicsSubstepRecorder(ctrl, bmin, bmax, support, torch, initial_xy=center)
    task.sim_env.set_physics_step_callback(recorder)
    recorder.begin_phase("warmup")
    for sample_index in range(1, warmup_steps + 1):
        recorder.begin_interval()
        ctrl.begin_control_interval()
        ctrl.set_command(direction, altitude)
        task.sim_env.step(actions=torch.zeros((envs, 4), device=task.device))
    warmup_traces = recorder.export()
    warmup_contact_any = recorder.phase_contact_any.detach().clone()
    warmup_invalid_any = recorder.phase_invalid_any.detach().clone()
    warmup_path = recorder.path_distance.detach().clone()
    warmup_final_speed = ctrl.linvel[:, :2].norm(dim=1).detach().clone()
    warmup_error = (warmup_final_speed - float(speed)).abs()
    warmup_converged = (warmup_error <= INITIAL_SPEED_ABS_TOLERANCE_MPS) & (
        warmup_error / max(float(speed), FINITE_EPS) <= INITIAL_SPEED_REL_TOLERANCE
    )
    if not bool(warmup_converged.all()):
        final_cpu = warmup_final_speed.detach().cpu()
        error_cpu = warmup_error.detach().cpu()
        raise RuntimeError(
            "warmup did not converge to the requested target speed: "
            "final_speed_mps[min=%.6f mean=%.6f max=%.6f] "
            "abs_error_mps[min=%.6f mean=%.6f max=%.6f]"
            % (
                float(final_cpu.min().item()), float(final_cpu.mean().item()), float(final_cpu.max().item()),
                float(error_cpu.min().item()), float(error_cpu.mean().item()), float(error_cpu.max().item()),
            )
        )
    warmup_diag = ctrl.diagnostics()
    warmup_saturation = warmup_diag["motor_saturation_fraction"].detach().clone()
    warmup_tilt = warmup_diag["max_tilt_deg"].detach().clone()
    warmup_substeps = ctrl.substeps.detach().clone()
    # Diagnostic-only accumulators; force/command/PhysX state is unchanged.
    ctrl.substeps.zero_()
    ctrl.saturation_substeps.zero_()
    ctrl.max_tilt_seen_rad.zero_()
    ctrl.velocity_error_integral.zero_()
    start_pos = ctrl.position[:, :2].detach().clone()
    start_speed = ctrl.linvel[:, :2].norm(dim=1).detach().clone()
    if not bool(torch.allclose(start_speed, warmup_final_speed, atol=1e-7, rtol=0.0)):
        raise RuntimeError("braking start speed changed during diagnostic snapshot")
    recorder.begin_phase("brake")
    traces = list(warmup_traces)
    for sample_index in range(1, brake_steps + 1):
        recorder.begin_interval()
        ctrl.begin_control_interval()
        ctrl.set_command(torch.zeros_like(direction), altitude)
        task.sim_env.step(actions=torch.zeros((envs, 4), device=task.device))
        if bool(recorder.stopped.all()):
            break
    if not bool(recorder.stopped.all()):
        raise RuntimeError("not every physical target stopped within the frozen budget")
    traces.extend(recorder.export())
    stop_time = recorder.stop_time
    stop_distance = recorder.stop_distance
    stop_position = recorder.stop_position
    contact_any = warmup_contact_any | recorder.phase_contact_any
    invalid_any = warmup_invalid_any | recorder.phase_invalid_any
    sat = ctrl.diagnostics()["motor_saturation_fraction"].detach().cpu().tolist()
    tilt = ctrl.diagnostics()["max_tilt_deg"].detach().cpu().tolist()
    brake_trace = [trace for trace in traces if trace["phase"] == "brake"]
    max_lateral = [
        max(abs(float(trace["position_xy_m"][env_id][1]) - float(start_pos[env_id, 1].item())) for trace in brake_trace)
        for env_id in range(envs)
    ]
    rows = []
    for env_id in range(envs):
        distance = float(stop_distance[env_id].item())
        initial = float(start_speed[env_id].item())
        rows.append({
            "env_id": env_id,
            "requested_speed_mps": float(speed),
            "measured_initial_speed_mps": initial,
            "warmup_final_speed_mps": float(warmup_final_speed[env_id].item()),
            "warmup_speed_error_mps": float(warmup_error[env_id].item()),
            "warmup_converged": bool(warmup_converged[env_id].item()),
            "stop_time_s": float(stop_time[env_id].item()),
            "stop_distance_m": distance,
            "endpoint_displacement_m": float((stop_position[env_id] - start_pos[env_id]).norm().item()),
            "max_lateral_deviation_m": float(max_lateral[env_id]),
            "effective_deceleration_mps2": initial * initial / max(2.0 * distance, FINITE_EPS),
            "warmup_contact": bool(warmup_contact_any[env_id].item()),
            "warmup_invalid_obb": bool(warmup_invalid_any[env_id].item()),
            "contact": bool(contact_any[env_id].item()),
            "invalid_obb": bool(invalid_any[env_id].item()),
            "warmup_motor_saturation_fraction": float(warmup_saturation[env_id].item()),
            "warmup_max_tilt_deg": float(warmup_tilt[env_id].item()),
            "motor_saturation_fraction": float(sat[env_id]),
            "max_tilt_deg": float(tilt[env_id]),
        })
    payload = {
        "schema": SCHEMA,
        "cell": {"speed_mps": float(speed), "envs": envs, "seed": 827, "child_auth": child_auth},
        "contract": FROZEN_CONTRACT,
        "source_attestation": source_attestation,
        "setup": {
            "mode": "obstacle_free_center",
            "active_bars": 0,
            "center_xy_m": [[float(x), float(y)] for x, y in center[:, :2].detach().cpu().tolist()],
            "center_clearance_to_arena_m": [float(v) for v in setup_margin.detach().cpu().tolist()],
            "warmup_steps": warmup_steps,
            "brake_steps_budget": brake_steps,
            "warmup_substeps": [int(v) for v in warmup_substeps.detach().cpu().tolist()],
            "warmup_path_distance_m": [float(v) for v in warmup_path.detach().cpu().tolist()],
        },
        "instantiated": instantiated,
        "raw_samples": rows,
        "physics_samples": traces,
        "provenance": runtime_provenance(_repo_root()),
    }
    require_finite(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(payload))


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--speed", type=float, choices=REGISTERED_SPEEDS)
    parser.add_argument("--envs", type=int, default=REGISTERED_ENVS)
    parser.add_argument("--_single-speed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_auth-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.preflight and args.speed is None:
        parser.error("--speed is required for a cell")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    root = _repo_root()
    if args.preflight:
        manifest = recovery_source_manifest(root)
        require_finite({"contract": FROZEN_CONTRACT, "manifest": manifest})
        print(json.dumps({"schema": SCHEMA, "contract": FROZEN_CONTRACT, "manifest": manifest}, sort_keys=True))
        return 0
    run_speed_cell(float(args.speed), Path(args.output), int(args.envs), auth_file=args._auth_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
