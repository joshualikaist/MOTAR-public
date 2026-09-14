"""Flight-envelope verification for the NavRL ref5in candidate (WORKLOG 2026-08-13).

`tests/test_navrl_ref5in_platform.py` checks that the URDF, the allocation matrix and the config
agree on paper. This script checks that Isaac Gym and the Lee velocity controller agree with them
in flight -- the parameters could be internally consistent and still produce a vehicle that cannot
hover, saturates its motors, or tracks velocity commands worse than the lineage it replaces.

Four open-arena manoeuvres (NAVRL_NUM_BARS=0 and speed governor off) exercise the complete
vehicle-plus-controller stack:

  hover     zero velocity command; altitude must hold at flight_altitude.
  forward   +v_max on vehicle x; reports the achieved steady-state speed and the 90% rise time.
  reversal  +v_max for 4 s then -v_max; the swing has to be flown as a pitch reversal, so this is
            where a heavier airframe's lower angular acceleration shows up if anywhere.
  yaw       +yaw_rate_max; same idea about the z axis, where izz grew 3.4x.

The candidate and legacy body deliberately use the SAME inherited Lee-controller gains.  Results
therefore measure a closed-loop body+controller condition, not intrinsic airframe dynamics and not
the performance of a controller retuned for the heavier candidate.  A separate 100 Hz roll/pitch
probe exposes that fixed-gain attitude response without pretending that it is a hardware benchmark.

The candidate collision box is 0.12 m tall versus 0.08 m for legacy.  Its level XY literals match,
but tilted bar-contact geometry does not; this tool is a flight-envelope gate, not proof that task
difficulty is unchanged.

Run (GPU, ~2 min):
  PYTHONNOUSERSITE=1 python tools/verify_navrl_ref_platform.py --output /tmp/ref_platform.json
"""

import argparse
import json
import math
import os
import subprocess
import sys

# Direct invocations often use an absolute conda Python without activating that environment.  The
# Isaac Gym torch extension then imports from the right interpreter but cannot find the matching
# `ninja` executable.  Bind PATH to the interpreter in use before either worker imports Isaac Gym.
_python_bin = os.path.dirname(os.path.realpath(sys.executable))
os.environ["PATH"] = _python_bin + os.pathsep + os.environ.get("PATH", "")

CONTROL_HZ = 10.0
DT = 1.0 / CONTROL_HZ
G = 9.81
SETTLE_STEPS = 10
PHYSICS_TILT_DURATION_S = 0.60
ATTITUDE_THRESHOLD_DEG = 20.0
ACTUATOR_NEAR_LIMIT_FRACTION = 0.02
SCHEMA_VERSION = 2

# One robot per PROCESS, for two independent reasons:
#   * task_config reads NAVRL_ROBOT at class-definition time, so a later os.environ write is a no-op;
#   * two Isaac Gym simulations in one process segfault on the second create.
# The default (no --robot) run therefore re-executes this file once per platform and merges.
_WORKER = "--robot" in sys.argv

# The v2 flight contract, minus everything that is not vehicle dynamics. NAVRL_VISION=1 matters
# most: it makes actions vehicle-frame velocity commands (the deployed contract), whereas the
# default goal-frame mode rotates them through the start->goal direction and makes an open-loop
# step response unreadable. Zero bars and no speed governor leave nothing else under test.
_V2_FLIGHT_CONTRACT = {
    "AERIAL_GYM_SIM_NAME": "base_sim",
    "NAVRL_VISION": "1",
    "NAVRL_PERCEPTION": "1",
    "NAVRL_PERCEPTION_PERTURB": "0",
    "NAVRL_GENERAL_TRAIN": "1",
    "NAVRL_MAX_VELOCITY": "2.5",
    "NAVRL_ALT_HOLD_VMAX": "2.5",
    "NAVRL_MAX_TILT_DEG": "45.0",
    # Canonical v2 training explicitly raises yaw authority from the task default 2.5 to 3.0.
    # Leaving this unset made the first envelope report test a different command contract.
    "NAVRL_YAW_RATE_MAX": "3.0",
    "NAVRL_ARENA_XY": "40",
    "NAVRL_ARENA_Z": "3",
    "NAVRL_BAR_POOL": "bars_h3",
    "NAVRL_PLACEMENT_MODE": "navrl_band",
    "NAVRL_EPISODE_LEN_STEPS": "600",
    "NAVRL_NUM_BARS": "0",
    "NAVRL_DENSITY_CURRICULUM": "0",
    "NAVRL_TARGET_SPEED": "0",
    "NAVRL_TARGET_SPEED_MIN": "0",
    "NAVRL_TARGET_SPEED_FINAL": "0",
    "NAVRL_TARGET_SPEED_RAMP_EPOCHS": "1",
    "NAVRL_TARGET_PATTERN": "cv",
    "NAVRL_TILT_COMP": "1",
    "NAVRL_SPEED_GOVERNOR": "off",
}

if _WORKER:
    for _k, _v in _V2_FLIGHT_CONTRACT.items():
        # Closed measurement contract: an interactive shell must not turn a nominal 0-bar/off
        # envelope run into a different experiment.  The parent uses one worker process per robot,
        # so overwriting here cannot contaminate the other arm.
        os.environ[_k] = _v
    os.environ["NAVRL_ROBOT"] = sys.argv[sys.argv.index("--robot") + 1]
    if "--seed" in sys.argv:
        os.environ["NAVRL_SEED"] = sys.argv[sys.argv.index("--seed") + 1]
    # isaacgym must be imported before torch (it inspects sys.modules and refuses otherwise).
    from aerial_gym.registry.task_registry import task_registry  # noqa: E402

    import torch  # noqa: E402


def _make(robot_name, num_envs, _seed):
    return task_registry.make_task("navrl_task", headless=True, num_envs=num_envs)


def _live_mask(task, alive):
    """Envs that have not reset yet. A reset teleports the drone, so its trace stops being a
    response to our command; every statistic below is computed over live envs only."""
    done = (task.terminations > 0) | (task.truncations > 0)
    return alive & ~done.bool()


VEH_VX, VEH_VY, POS_Z, YAW_RATE, SPEED_XY, PROJ, ROLL, PITCH = range(8)


def _robot(task):
    return task.sim_env.robot_manager.robot


def _new_run_diagnostics():
    return {
        "all_finite": True,
        "nonfinite_values": 0,
        "actuator_samples": 0,
        "requested_high_limit_samples": 0,
        "requested_low_limit_samples": 0,
        "actual_near_high_limit_samples": 0,
        "actual_near_low_limit_samples": 0,
        "peak_requested_fraction_of_max": float("-inf"),
        "minimum_requested_fraction_of_max": float("inf"),
        "peak_actual_fraction_of_max": float("-inf"),
    }


def _actuator_snapshot(task):
    """Return raw allocator requests and actual motor state.

    `MotorModel.update_motor_thrusts` clamps the request before exposing the current thrust, so
    looking only at current thrust hides allocator saturation.  Reapply the allocator pseudoinverse
    to the controller's last wrench to recover the pre-clamp request without modifying training code.
    """
    robot = _robot(task)
    allocator = robot.control_allocator
    raw_request = torch.bmm(
        allocator.inv_force_torque_allocation_matrix,
        robot.controller.wrench_command.unsqueeze(-1),
    ).squeeze(-1)
    motor = allocator.motor_model
    return raw_request, motor.current_motor_thrust, motor.min_thrust, motor.max_thrust


def _accumulate_run_diagnostics(diag, state_tensors, actuator):
    raw, actual, minimum, maximum = actuator
    nonfinite = 0
    for tensor in tuple(state_tensors) + (raw, actual):
        nonfinite += int((~torch.isfinite(tensor)).sum().item())
    diag["nonfinite_values"] += nonfinite
    diag["all_finite"] = bool(diag["all_finite"] and nonfinite == 0)

    span = (maximum - minimum).clamp_min(1e-6)
    eps = 1e-6
    diag["actuator_samples"] += int(raw.numel())
    diag["requested_high_limit_samples"] += int((raw > maximum + eps).sum().item())
    diag["requested_low_limit_samples"] += int((raw < minimum - eps).sum().item())
    diag["actual_near_high_limit_samples"] += int(
        (actual >= maximum - ACTUATOR_NEAR_LIMIT_FRACTION * span).sum().item()
    )
    diag["actual_near_low_limit_samples"] += int(
        (actual <= minimum + ACTUATOR_NEAR_LIMIT_FRACTION * span).sum().item()
    )

    finite_raw = raw[torch.isfinite(raw)]
    finite_actual = actual[torch.isfinite(actual)]
    if finite_raw.numel():
        raw_scale = maximum.expand_as(raw)[torch.isfinite(raw)].clamp_min(1e-6)
        fraction = finite_raw / raw_scale
        diag["peak_requested_fraction_of_max"] = max(
            diag["peak_requested_fraction_of_max"], float(fraction.max().item())
        )
        diag["minimum_requested_fraction_of_max"] = min(
            diag["minimum_requested_fraction_of_max"], float(fraction.min().item())
        )
    if finite_actual.numel():
        actual_scale = maximum.expand_as(actual)[torch.isfinite(actual)].clamp_min(1e-6)
        diag["peak_actual_fraction_of_max"] = max(
            diag["peak_actual_fraction_of_max"],
            float((finite_actual / actual_scale).max().item()),
        )


def _finalize_run_diagnostics(diag):
    out = dict(diag)
    n = max(1, int(out.pop("actuator_samples")))
    for count_name, fraction_name in (
        ("requested_high_limit_samples", "requested_high_limit_fraction"),
        ("requested_low_limit_samples", "requested_low_limit_fraction"),
        ("actual_near_high_limit_samples", "actual_near_high_limit_fraction"),
        ("actual_near_low_limit_samples", "actual_near_low_limit_fraction"),
    ):
        count = int(out.pop(count_name))
        out[fraction_name] = count / n
    out["requested_any_limit_fraction"] = (
        out["requested_high_limit_fraction"] + out["requested_low_limit_fraction"]
    )
    for name in (
        "peak_requested_fraction_of_max",
        "minimum_requested_fraction_of_max",
        "peak_actual_fraction_of_max",
    ):
        if not math.isfinite(out[name]):
            out[name] = None
    return out


def _run(task, action, steps, alive=None, ref_dir=None):
    """Hold `action` for `steps` control steps.

    Returns (trace, alive, world_vxy, diagnostics). `trace` is [steps, num_envs, 8]; PROJ is the
    world velocity projected onto `ref_dir` (a per-env unit vector), which is how the reversal
    manoeuvre is read -- the command flips in the vehicle frame, so only a fixed world axis can
    tell "flew backwards" from "yawed around and flew forwards"."""
    n, device = task.num_envs, task.device
    if alive is None:
        alive = torch.ones(n, dtype=torch.bool, device=device)
    cmd = torch.tensor(action, dtype=torch.float32, device=device).repeat(n, 1)
    trace = torch.zeros(steps, n, 8, device=device)
    world_vxy = torch.zeros(n, 2, device=device)
    diag = _new_run_diagnostics()
    for t in range(steps):
        task.step(cmd)
        # The final physics substep refreshed the raw state; refresh derived vehicle/body tensors
        # once more so this sample corresponds to the same instant.
        _robot(task).update_states()
        vveh = task.obs_dict["robot_vehicle_linvel"]
        world_vxy = task.obs_dict["robot_linvel"][:, 0:2]
        trace[t, :, VEH_VX] = vveh[:, 0]
        trace[t, :, VEH_VY] = vveh[:, 1]
        trace[t, :, POS_Z] = task.obs_dict["robot_position"][:, 2]
        trace[t, :, YAW_RATE] = task.obs_dict["robot_body_angvel"][:, 2]
        trace[t, :, SPEED_XY] = torch.norm(world_vxy, dim=1)
        trace[t, :, ROLL] = task.obs_dict["robot_euler_angles"][:, 0]
        trace[t, :, PITCH] = task.obs_dict["robot_euler_angles"][:, 1]
        if ref_dir is not None:
            trace[t, :, PROJ] = (world_vxy * ref_dir).sum(dim=1)
        _accumulate_run_diagnostics(
            diag,
            (
                task.obs_dict["robot_position"],
                task.obs_dict["robot_orientation"],
                task.obs_dict["robot_linvel"],
                task.obs_dict["robot_angvel"],
                vveh,
                task.obs_dict["robot_body_angvel"],
            ),
            _actuator_snapshot(task),
        )
        alive = _live_mask(task, alive)
    return trace, alive, world_vxy, _finalize_run_diagnostics(diag)


def _assert_open_arena_contract(task, robot_name, num_envs):
    """Fail closed if the worker did not instantiate the advertised measurement condition."""
    assert task.task_config.robot_name == robot_name
    assert int(task.n_bars_active) == 0, task.n_bars_active
    assert int(task.num_envs) == int(num_envs), task.num_envs
    assert task.speed_governor_cfg.mode == "off", task.speed_governor_cfg.mode
    assert abs(float(task.task_config.max_velocity) - 2.5) <= 1e-9
    assert abs(float(task.task_config.yaw_rate_max) - 3.0) <= 1e-9
    assert abs(float(task.task_config.alt_hold_vmax) - 2.5) <= 1e-9
    assert abs(float(task.tm.speed_fixed)) <= 1e-9
    assert abs(float(task.tm.speed_min)) <= 1e-9
    assert abs(float(task.tm.speed_final)) <= 1e-9
    assert bool(task.vision_mode), "actions must use the deployed vehicle-frame contract"
    assert task.task_config.controller_name == "lee_velocity_control_navrl"
    assert abs(float(_robot(task).controller.cfg.max_tilt_angle_deg) - 45.0) <= 1e-9
    arena_extent = task.obs_dict["env_bounds_max"] - task.obs_dict["env_bounds_min"]
    expected_extent = torch.tensor([40.0, 40.0, 3.0], device=task.device)
    assert bool(torch.allclose(arena_extent, expected_extent.expand_as(arena_extent), atol=1e-6))
    physics = task._runtime_physics_contract()
    assert physics["runtime_sim_config_class"] == "BaseSimConfig", physics
    assert abs(float(physics["rl_step_dt_s"]) - 0.1) <= 1e-9, physics


def _runtime_vehicle_contract(task, robot_name):
    """Record the actor properties Isaac Gym actually instantiated, not only source literals."""
    manager = task.sim_env.robot_manager
    masses = manager.robot_masses.detach()
    inertias = manager.robot_inertias.detach()
    if not bool(torch.isfinite(masses).all() and torch.isfinite(inertias).all()):
        raise RuntimeError(f"{robot_name}: non-finite runtime mass/inertia")
    expected_mass = {
        "navrl_quad": 0.250,
        "navrl_ref5in_quad": 1.200,
        "navrl_ref5in_v2_quad": 1.200,
    }[robot_name]
    if not bool(torch.allclose(masses, torch.full_like(masses, expected_mass), atol=1e-5)):
        raise RuntimeError(
            f"{robot_name}: runtime mass {masses.min().item():.6f}..{masses.max().item():.6f} "
            f"does not match {expected_mass:.6f} kg"
        )
    expected_inertia = {
        "navrl_quad": torch.diag(
            torch.tensor([8.45e-4, 8.45e-4, 1.69e-3], device=task.device)
        ),
        "navrl_ref5in_quad": torch.diag(
            torch.tensor([4.1422e-3, 4.1422e-3, 5.7692e-3], device=task.device)
        ),
        "navrl_ref5in_v2_quad": torch.diag(
            torch.tensor([4.1422e-3, 4.1422e-3, 5.7692e-3], device=task.device)
        ),
    }[robot_name]
    if not bool(
        torch.allclose(
            inertias,
            expected_inertia.expand_as(inertias),
            atol=2e-6,
            rtol=1e-4,
        )
    ):
        raise RuntimeError(
            f"{robot_name}: runtime inertia does not match the declared assembled tensor"
        )
    robot = _robot(task)
    mask = [int(v) for v in robot.application_mask.detach().cpu().tolist()]
    if mask != [5, 6, 7, 8]:
        raise RuntimeError(f"{robot_name}: thrust application mask {mask} does not select motor links")
    body_names = list(
        manager.gym.get_actor_rigid_body_names(
            task.sim_env.IGE_env.env_handles[0], manager.robot_handles[0]
        )
    )
    selected_body_names = [body_names[index] for index in mask]
    expected_motors = [f"motor_{index}" for index in range(4)]
    if selected_body_names != expected_motors:
        raise RuntimeError(
            f"{robot_name}: thrust mask selects {selected_body_names}, expected {expected_motors}"
        )
    return {
        "mass_kg": float(masses[0].item()),
        "inertia_kg_m2": [
            [float(value) for value in row]
            for row in inertias[0].detach().cpu().tolist()
        ],
        "force_application_indices": mask,
        "rigid_body_order": body_names,
        "force_application_body_names": selected_body_names,
        "max_thrust_per_motor_n": float(
            robot.control_allocator.motor_model.max_thrust[0, 0].item()
        ),
    }


def _controller_contract(task):
    controller = _robot(task).controller

    def first_row(name):
        return [float(v) for v in getattr(controller, name)[0].detach().cpu().tolist()]

    return {
        "controller_name": str(task.task_config.controller_name),
        "gain_treatment": "same_inherited_fixed_gains_no_candidate_retuning",
        "retuned_for_candidate": False,
        "K_velocity": first_row("K_linvel_tensor_current"),
        "K_attitude": first_row("K_rot_tensor_current"),
        "K_angular_velocity": first_row("K_angvel_tensor_current"),
        "interpretation": (
            "closed-loop vehicle-plus-controller comparison; not intrinsic airframe-only "
            "performance and not a claim about a retuned hardware controller"
        ),
    }


def _pin_controller_and_motor_midpoints(task):
    """Remove reset-time controller/actuator randomization from the measurement condition."""
    robot = _robot(task)
    controller = robot.controller
    for current, minimum, maximum in (
        ("K_pos_tensor_current", "K_pos_tensor_min", "K_pos_tensor_max"),
        ("K_linvel_tensor_current", "K_linvel_tensor_min", "K_linvel_tensor_max"),
        ("K_rot_tensor_current", "K_rot_tensor_min", "K_rot_tensor_max"),
        ("K_angvel_tensor_current", "K_angvel_tensor_min", "K_angvel_tensor_max"),
    ):
        getattr(controller, current).copy_(
            0.5 * (getattr(controller, minimum) + getattr(controller, maximum))
        )

    motor = robot.control_allocator.motor_model
    motor.motor_time_constants_increasing.copy_(
        0.5
        * (
            motor.motor_time_constant_increasing_min
            + motor.motor_time_constant_increasing_max
        )
    )
    motor.motor_time_constants_decreasing.copy_(
        0.5
        * (
            motor.motor_time_constant_decreasing_min
            + motor.motor_time_constant_decreasing_max
        )
    )
    if hasattr(motor, "motor_thrust_constant"):
        motor.motor_thrust_constant.copy_(
            0.5 * (motor.motor_thrust_constant_min + motor.motor_thrust_constant_max)
        )


def _peak_abs(trace, column):
    return float(trace[:, :, column].abs().max().item())


def _altitude_excursion(trace, altitude):
    values = trace[:, :, POS_Z]
    return {
        "minimum_m": float(values.min().item()),
        "maximum_m": float(values.max().item()),
        "peak_error_abs_m": float((values - altitude).abs().max().item()),
    }


def _first_threshold_time(trace, threshold, dt):
    for index, value in enumerate(trace):
        if value >= threshold:
            return round((index + 1) * dt, 4)
    return None


def _run_physics_rate_tilt(task, axis, prepare):
    """Probe roll/pitch at the 100 Hz physics rate under the inherited, unretuned controller.

    This bypasses only the 10 Hz task wrapper; it still runs the real Lee velocity controller,
    allocator, motor model and Isaac dynamics every physics tick.  It is intentionally labelled
    fixed-gain so a slower result cannot be misreported as an intrinsic airframe limit.
    """
    alive, settle_diag = prepare(f"physics-rate-{axis}")
    robot = _robot(task)
    physics_dt = float(robot.dt)
    steps = max(1, int(round(PHYSICS_TILT_DURATION_S / physics_dt)))
    command = torch.zeros((task.num_envs, 4), device=task.device)
    command[:, 0 if axis == "pitch" else 1] = float(task.task_config.max_velocity)
    angle_index = 1 if axis == "pitch" else 0
    rate_index = angle_index
    angle_abs_mean_deg = []
    peak_angle_deg = 0.0
    peak_rate_radps = 0.0
    min_altitude_m = float("inf")
    max_altitude_m = float("-inf")
    diag = _new_run_diagnostics()

    for _ in range(steps):
        task.sim_env.reset_tensors()
        task.sim_env.simulate(command, None)
        task.sim_env.compute_observations()
        robot.update_states()
        task.command.copy_(command)
        task.compute_state_reward_and_terminations()
        alive = _live_mask(task, alive)

        angle = task.obs_dict["robot_euler_angles"][:, angle_index]
        rate = task.obs_dict["robot_body_angvel"][:, rate_index]
        altitude_now = task.obs_dict["robot_position"][:, 2]
        angle_deg = torch.rad2deg(angle.abs())
        angle_abs_mean_deg.append(float(angle_deg.mean().item()))
        peak_angle_deg = max(peak_angle_deg, float(angle_deg.max().item()))
        peak_rate_radps = max(peak_rate_radps, float(rate.abs().max().item()))
        min_altitude_m = min(min_altitude_m, float(altitude_now.min().item()))
        max_altitude_m = max(max_altitude_m, float(altitude_now.max().item()))
        _accumulate_run_diagnostics(
            diag,
            (
                task.obs_dict["robot_position"],
                task.obs_dict["robot_orientation"],
                task.obs_dict["robot_linvel"],
                task.obs_dict["robot_angvel"],
                task.obs_dict["robot_euler_angles"],
                task.obs_dict["robot_body_angvel"],
            ),
            _actuator_snapshot(task),
        )

    return {
        "axis": axis,
        "sample_hz": round(1.0 / physics_dt, 3),
        "duration_s": round(steps * physics_dt, 4),
        "task_wrapper_bypassed": True,
        "task_altitude_hold_bypassed": True,
        "command_velocity_mps": float(task.task_config.max_velocity),
        "threshold_angle_deg": ATTITUDE_THRESHOLD_DEG,
        "time_to_threshold_s": _first_threshold_time(
            angle_abs_mean_deg, ATTITUDE_THRESHOLD_DEG, physics_dt
        ),
        "peak_angle_deg": peak_angle_deg,
        "peak_body_rate_radps": peak_rate_radps,
        "altitude": {
            "minimum_m": min_altitude_m,
            "maximum_m": max_altitude_m,
            "peak_error_abs_m": max(
                abs(min_altitude_m - float(task.task_config.flight_altitude)),
                abs(max_altitude_m - float(task.task_config.flight_altitude)),
            ),
        },
        "n_live": int(alive.sum().item()),
        "all_envs_survived": bool(alive.all().item()),
        "actuator": _finalize_run_diagnostics(diag),
        "settle_actuator": settle_diag,
    }


def _stats(trace, alive, column):
    """Mean over live envs of the last 10 steps (steady state), plus the full mean trajectory."""
    if alive.sum() == 0:
        return {"n_live": 0}
    series = trace[:, alive, column]
    return {
        "n_live": int(alive.sum().item()),
        "steady_mean": float(series[-10:].mean().item()),
        "steady_std": float(series[-10:].std().item()),
        "trace_mean": [round(float(v), 4) for v in series.mean(dim=1).tolist()],
    }


def _rise_time_s(trace_mean, target, frac=0.9):
    """First control step at which the mean response reaches `frac` of target. Resolution is one
    control step (0.1 s) -- enough to separate platforms, not enough to resolve the 0.04 s motor
    lag, which is unchanged between them anyway."""
    threshold = frac * target
    for i, v in enumerate(trace_mean):
        if (v >= threshold) if target > 0 else (v <= threshold):
            return round((i + 1) * DT, 2)
    return None


def _reversal_time_s(trace_mean, v_max):
    """Steps from the command flip until the mean velocity crosses -0.9 * v_max."""
    return _rise_time_s(trace_mean, -v_max)


def verify(robot_name, num_envs, seed):
    task = _make(robot_name, num_envs, seed)
    _assert_open_arena_contract(task, robot_name, num_envs)
    # Capture is a task outcome, not a vehicle-dynamics failure. Disable only that terminal for
    # this open-loop envelope harness; collision/OOB/altitude terminals stay active and are fatal.
    task.task_config.success_radius = 0.0
    v_max = float(task.task_config.max_velocity)
    yaw_max = float(task.task_config.yaw_rate_max)
    altitude = float(task.task_config.flight_altitude)
    robot = _robot(task)
    motor = robot.control_allocator.motor_model
    _pin_controller_and_motor_midpoints(task)
    out = {
        "robot": robot_name,
        "robot_provenance": dict(task._robot_provenance),
        "physics": task._runtime_physics_contract(),
        "runtime_vehicle": _runtime_vehicle_contract(task, robot_name),
        "controller_condition": _controller_contract(task),
        "measurement_scope": (
            "whole-platform candidate under the same inherited, unretuned Lee controller"
        ),
        "collision_scope": (
            "level XY literals match legacy, but candidate height is 0.12 m versus 0.08 m; "
            "tilted bar-contact geometry and therefore task difficulty are not equivalent"
        ),
        "v_max": v_max,
        "yaw_rate_max": yaw_max,
        "flight_altitude": altitude,
        "settle_steps_10hz": SETTLE_STEPS,
    }

    def require_integrity(manoeuvre, alive, diag):
        if not bool(diag.get("all_finite", False)):
            raise RuntimeError(
                f"{robot_name} {manoeuvre}: {diag.get('nonfinite_values')} non-finite values"
            )
        n_live = int(alive.sum().item())
        if n_live != int(num_envs):
            raise RuntimeError(
                f"{robot_name} {manoeuvre}: {num_envs - n_live}/{num_envs} environments "
                "terminated during an obstacle-free envelope measurement"
            )

    def place_centered_kinematics():
        """Place every body at the exact arena centre with one deterministic level state."""
        task.reset()
        # `task.reset()` reaches BaseMultirotor.reset_idx(), which is allowed to randomize both
        # controller gains and motor parameters.  Re-pin every manoeuvre, even when today's config
        # happens to use zero-width ranges, so a future domain-randomization edit cannot silently
        # contaminate this fixed-condition comparison.
        _pin_controller_and_motor_midpoints(task)
        bounds_min = task.obs_dict["env_bounds_min"]
        bounds_max = task.obs_dict["env_bounds_max"]
        # Exact centre, not a random inner box: the +v_max setup can travel about 10 m in four
        # seconds.  A former 6 m random margin still killed 1/16 legacy forward trials and made
        # the mean survivor-biased.  The centre supplies >=20 m to every wall in the 40 m arena.
        task.obs_dict["robot_position"][:, :2] = 0.5 * (
            bounds_min[:, :2] + bounds_max[:, :2]
        )
        task.obs_dict["robot_position"][:, 2] = altitude
        task.obs_dict["robot_linvel"].zero_()
        task.obs_dict["robot_angvel"].zero_()
        quat = torch.zeros((task.num_envs, 4), device=task.device)
        quat[:, 3] = 1.0
        task.obs_dict["robot_orientation"].copy_(quat)
        task.height_range[:, 0] = altitude
        task.height_range[:, 1] = altitude
        task._z_err_integral.zero_()
        task.prev_vel_w.zero_()
        task.terminations.zero_()
        task.truncations.zero_()
        task.sim_env.reset_tensors()

        # `BaseMultirotor.reset_idx` normally samples current thrust uniformly from [0, max].  That
        # makes the first response depend on a large, platform-scaled impulse.  Use the same physical
        # condition in every env: nominal k and exactly mg/4 current thrust before settling.
        hover_thrust = (
            task.sim_env.robot_manager.robot_masses.unsqueeze(1) * abs(float(robot.gravity[0, 2])) / 4.0
        ).expand_as(motor.current_motor_thrust)
        if bool((hover_thrust > motor.max_thrust).any()):
            raise RuntimeError(f"{robot_name}: deterministic hover thrust exceeds motor limit")
        motor.current_motor_thrust.copy_(hover_thrust)
        robot.output_forces.zero_()
        robot.output_torques.zero_()
        robot.robot_force_tensors.zero_()
        robot.robot_torque_tensors.zero_()
        task.sim_env.IGE_env.write_to_sim()
        robot.update_states()
        task.prev_pos.copy_(task.obs_dict["robot_position"])
        task.prev_rel.copy_(task.obs_dict["robot_position"] - task.target_position)
        return hover_thrust

    def recenter_after_settle():
        """Remove the tiny settle displacement without resetting the now-settled motor state."""
        bounds_min = task.obs_dict["env_bounds_min"]
        bounds_max = task.obs_dict["env_bounds_max"]
        task.obs_dict["robot_position"][:, :2] = 0.5 * (
            bounds_min[:, :2] + bounds_max[:, :2]
        )
        task.obs_dict["robot_position"][:, 2] = altitude
        task.obs_dict["robot_linvel"].zero_()
        task.obs_dict["robot_angvel"].zero_()
        task.obs_dict["robot_orientation"].zero_()
        task.obs_dict["robot_orientation"][:, 3] = 1.0
        task.height_range[:, 0] = altitude
        task.height_range[:, 1] = altitude
        task._z_err_integral.zero_()
        task.prev_vel_w.zero_()
        task.terminations.zero_()
        task.truncations.zero_()
        task.sim_env.reset_tensors()
        task.sim_env.IGE_env.write_to_sim()
        robot.update_states()
        task.prev_pos.copy_(task.obs_dict["robot_position"])
        task.prev_rel.copy_(task.obs_dict["robot_position"] - task.target_position)

    def prepare(manoeuvre):
        hover_thrust = place_centered_kinematics()
        alive = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
        _settle, alive, _world_vxy, settle_diag = _run(
            task, [0.0, 0.0, 0.0, 0.0], SETTLE_STEPS, alive=alive
        )
        require_integrity(f"{manoeuvre}-settle", alive, settle_diag)
        recenter_after_settle()
        out.setdefault("initialization", {
            "position": "exact_arena_center",
            "orientation": "identity",
            "linear_and_angular_velocity": "zero",
            "initial_motor_thrust": "runtime_mass_times_g_div_4",
            "initial_motor_thrust_per_motor_n": float(hover_thrust[0, 0].item()),
            "controller_gains": "per-platform config midpoint_no_randomization",
            "motor_time_constants": "per-platform config midpoint_no_randomization",
            "motor_thrust_constant": "per-platform config midpoint_no_randomization",
            "settle_duration_s": SETTLE_STEPS * DT,
        })
        return torch.ones(task.num_envs, dtype=torch.bool, device=task.device), settle_diag

    alive, settle_diag = prepare("hover")
    hover, alive, _, actuator = _run(task, [0.0, 0.0, 0.0, 0.0], 40, alive=alive)
    require_integrity("hover", alive, actuator)
    z = _stats(hover, alive, POS_Z)
    out["hover"] = {
        **z,
        "altitude_error_m": round(z.get("steady_mean", float("nan")) - altitude, 4),
        "speed_mean_mps": _stats(hover, alive, SPEED_XY).get("steady_mean"),
        "altitude": _altitude_excursion(hover, altitude),
        "all_envs_survived": bool(alive.all().item()),
        "actuator": actuator,
        "settle_actuator": settle_diag,
    }

    alive, settle_diag = prepare("forward")
    fwd, alive, _, actuator = _run(task, [1.0, 0.0, 0.0, 0.0], 40, alive=alive)
    require_integrity("forward", alive, actuator)
    f = _stats(fwd, alive, VEH_VX)
    out["forward"] = {
        **f,
        "tracking_error_mps": round(f.get("steady_mean", float("nan")) - v_max, 4),
        "rise_time_90pct_s": _rise_time_s(f.get("trace_mean", []), v_max),
        "world_speed": _stats(fwd, alive, SPEED_XY),
        "lateral_slip_mps": _stats(fwd, alive, VEH_VY).get("steady_mean"),
        "lateral_slip_peak_abs_mps": _peak_abs(fwd, VEH_VY),
        "altitude_error_m": round(
            _stats(fwd, alive, POS_Z).get("steady_mean", float("nan")) - altitude, 4
        ),
        "altitude": _altitude_excursion(fwd, altitude),
        "peak_roll_deg": math.degrees(_peak_abs(fwd, ROLL)),
        "peak_pitch_deg": math.degrees(_peak_abs(fwd, PITCH)),
        "all_envs_survived": bool(alive.all().item()),
        "actuator": actuator,
        "settle_actuator": settle_diag,
    }

    # Reversal: build up along +vehicle-x, freeze that world heading as the reference axis, then
    # command the exact opposite and watch the projection swing through zero to -v_max.
    alive, settle_diag = prepare("reversal")
    _, alive, world_vxy, setup_actuator = _run(
        task, [1.0, 0.0, 0.0, 0.0], 40, alive=alive
    )
    require_integrity("reversal-setup", alive, setup_actuator)
    ref_dir = world_vxy / world_vxy.norm(dim=1, keepdim=True).clamp_min(1e-6)
    rev, alive, _, actuator = _run(
        task, [-1.0, 0.0, 0.0, 0.0], 40, alive=alive, ref_dir=ref_dir
    )
    require_integrity("reversal", alive, actuator)
    r = _stats(rev, alive, PROJ)
    out["reversal"] = {
        **r,
        "reversal_time_90pct_s": _reversal_time_s(r.get("trace_mean", []), v_max),
        "zero_crossing_s": _rise_time_s([-v for v in r.get("trace_mean", [])], 1e-6),
        "tracking_error_mps": round(r.get("steady_mean", float("nan")) + v_max, 4),
        "lateral_slip_peak_abs_mps": _peak_abs(rev, VEH_VY),
        "altitude": _altitude_excursion(rev, altitude),
        "peak_roll_deg": math.degrees(_peak_abs(rev, ROLL)),
        "peak_pitch_deg": math.degrees(_peak_abs(rev, PITCH)),
        "all_envs_survived": bool(alive.all().item()),
        "actuator": actuator,
        "setup_actuator": setup_actuator,
        "settle_actuator": settle_diag,
    }

    alive, settle_diag = prepare("yaw")
    yaw, alive, _, actuator = _run(task, [0.0, 0.0, 0.0, 1.0], 40, alive=alive)
    require_integrity("yaw", alive, actuator)
    y = _stats(yaw, alive, YAW_RATE)
    out["yaw"] = {
        **y,
        "tracking_error_radps": round(y.get("steady_mean", float("nan")) - yaw_max, 4),
        "rise_time_90pct_s": _rise_time_s(y.get("trace_mean", []), yaw_max),
        "altitude": _altitude_excursion(yaw, altitude),
        "all_envs_survived": bool(alive.all().item()),
        "actuator": actuator,
        "settle_actuator": settle_diag,
    }

    pitch = _run_physics_rate_tilt(task, "pitch", prepare)
    roll = _run_physics_rate_tilt(task, "roll", prepare)
    for name, result in (("pitch", pitch), ("roll", roll)):
        if not result["all_envs_survived"]:
            raise RuntimeError(f"{robot_name} physics-rate-{name}: not all envs survived")
        if not result["actuator"]["all_finite"]:
            raise RuntimeError(f"{robot_name} physics-rate-{name}: non-finite state")
    out["physics_rate_fixed_gain_response"] = {
        "gain_treatment": "same_inherited_fixed_gains_no_candidate_retuning",
        "interpretation": (
            "low-level closed-loop response; differences include controller/body interaction and "
            "must not be labelled intrinsic airframe angular acceleration"
        ),
        "pitch": pitch,
        "roll": roll,
    }
    return out


def _verdict(
    legacy,
    ref,
    tol_mps,
    tol_alt,
    tol_slip,
    tol_reversal_s,
    max_motor_saturation_fraction,
    max_attitude_response_s,
    max_attitude_overshoot_deg,
):
    """Fail-closed gates for the fixed-controller whole-platform candidate condition."""
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    def actuator_diags(arm):
        return [
            arm["hover"]["actuator"],
            arm["forward"]["actuator"],
            arm["reversal"]["setup_actuator"],
            arm["reversal"]["actuator"],
            arm["yaw"]["actuator"],
            arm["physics_rate_fixed_gain_response"]["pitch"]["actuator"],
            arm["physics_rate_fixed_gain_response"]["roll"]["actuator"],
        ]

    def all_survived(arm):
        return all(
            bool(item["all_envs_survived"])
            for item in (
                arm["hover"],
                arm["forward"],
                arm["reversal"],
                arm["yaw"],
                arm["physics_rate_fixed_gain_response"]["pitch"],
                arm["physics_rate_fixed_gain_response"]["roll"],
            )
        )

    same_gains = all(
        legacy["controller_condition"][key] == ref["controller_condition"][key]
        for key in ("K_velocity", "K_attitude", "K_angular_velocity")
    )
    check(
        "comparison_uses_same_unretuned_controller_gains",
        same_gains
        and not legacy["controller_condition"]["retuned_for_candidate"]
        and not ref["controller_condition"]["retuned_for_candidate"],
        "same inherited K_v/K_R/K_omega; schema explicitly marks candidate retuning=false",
    )
    finite = all(diag["all_finite"] for arm in (legacy, ref) for diag in actuator_diags(arm))
    check(
        "all_measured_states_and_actuators_are_finite",
        finite,
        "both arms, all manoeuvres including 100 Hz roll/pitch",
    )
    survived = all_survived(legacy) and all_survived(ref)
    check(
        "all_environments_survive_every_open_arena_manoeuvre",
        survived,
        "no survivor-conditioned means are permitted",
    )

    check(
        "ref_holds_altitude",
        abs(ref["hover"]["altitude_error_m"]) <= tol_alt,
        f"|{ref['hover']['altitude_error_m']:+.3f}| m <= {tol_alt} m",
    )
    check(
        "ref_hovers_without_drifting",
        ref["hover"]["speed_mean_mps"] <= 0.15,
        f"{ref['hover']['speed_mean_mps']:.3f} m/s <= 0.15 m/s",
    )
    check(
        "ref_tracks_forward_command",
        abs(ref["forward"]["tracking_error_mps"]) <= tol_mps,
        f"|{ref['forward']['tracking_error_mps']:+.3f}| m/s <= {tol_mps} m/s",
    )
    check(
        "same_controller_forward_response_not_slower_than_legacy",
        (ref["forward"]["rise_time_90pct_s"] is not None
         and legacy["forward"]["rise_time_90pct_s"] is not None
         and ref["forward"]["rise_time_90pct_s"] <= legacy["forward"]["rise_time_90pct_s"] + 0.1),
        f"ref {ref['forward']['rise_time_90pct_s']} s vs legacy "
        f"{legacy['forward']['rise_time_90pct_s']} s (+0.1 s slack = one control step)",
    )
    check(
        "ref_forward_altitude_excursion_is_bounded",
        ref["forward"]["altitude"]["peak_error_abs_m"] <= tol_alt,
        f"{ref['forward']['altitude']['peak_error_abs_m']:.3f} m <= {tol_alt} m",
    )
    check(
        "ref_forward_lateral_slip_is_bounded",
        ref["forward"]["lateral_slip_peak_abs_mps"] <= tol_slip,
        f"{ref['forward']['lateral_slip_peak_abs_mps']:.3f} m/s <= {tol_slip} m/s",
    )
    check(
        "ref_tracks_reversal_command",
        abs(ref["reversal"]["tracking_error_mps"]) <= tol_mps,
        f"|{ref['reversal']['tracking_error_mps']:+.3f}| m/s <= {tol_mps} m/s",
    )
    ref_reversal = ref["reversal"]["reversal_time_90pct_s"]
    legacy_reversal = legacy["reversal"]["reversal_time_90pct_s"]
    check(
        "same_controller_reversal_response_within_declared_slack",
        ref_reversal is not None
        and legacy_reversal is not None
        and ref_reversal <= legacy_reversal + tol_reversal_s,
        f"ref {ref_reversal} s vs legacy {legacy_reversal} s (+{tol_reversal_s} s slack)",
    )
    check(
        "ref_reversal_altitude_excursion_is_bounded",
        ref["reversal"]["altitude"]["peak_error_abs_m"] <= tol_alt,
        f"{ref['reversal']['altitude']['peak_error_abs_m']:.3f} m <= {tol_alt} m",
    )
    check(
        "ref_reversal_lateral_slip_is_bounded",
        ref["reversal"]["lateral_slip_peak_abs_mps"] <= tol_slip,
        f"{ref['reversal']['lateral_slip_peak_abs_mps']:.3f} m/s <= {tol_slip} m/s",
    )
    check(
        "ref_tracks_yaw_command",
        abs(ref["yaw"]["tracking_error_radps"]) <= 0.5,
        f"|{ref['yaw']['tracking_error_radps']:+.3f}| rad/s <= 0.5 rad/s",
    )
    worst_saturation = max(
        diag["requested_any_limit_fraction"] for diag in actuator_diags(ref)
    )
    check(
        "ref_requested_motor_saturation_is_bounded",
        worst_saturation <= max_motor_saturation_fraction,
        f"worst pre-clamp request fraction {worst_saturation:.3f} "
        f"<= {max_motor_saturation_fraction:.3f}",
    )

    response = ref["physics_rate_fixed_gain_response"]
    for axis in ("pitch", "roll"):
        measured = response[axis]
        t_threshold = measured["time_to_threshold_s"]
        check(
            f"same_unretuned_controller_{axis}_reaches_{int(ATTITUDE_THRESHOLD_DEG)}deg",
            t_threshold is not None and t_threshold <= max_attitude_response_s,
            f"t={t_threshold} s <= {max_attitude_response_s} s at {measured['sample_hz']} Hz",
        )
        max_angle = 45.0 + max_attitude_overshoot_deg
        check(
            f"same_unretuned_controller_{axis}_overshoot_is_bounded",
            measured["peak_angle_deg"] <= max_angle,
            f"peak {measured['peak_angle_deg']:.2f} deg <= {max_angle:.2f} deg",
        )
        check(
            f"same_unretuned_controller_{axis}_altitude_excursion_is_bounded",
            measured["altitude"]["peak_error_abs_m"] <= tol_alt,
            f"{measured['altitude']['peak_error_abs_m']:.3f} m <= {tol_alt} m",
        )
    return checks


def _run_worker(robot, args):
    """Re-exec this file for one platform and read its JSON back."""
    out = os.path.join(os.path.dirname(os.path.abspath(args.output or "./")), f".{robot}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [sys.executable, os.path.abspath(__file__), "--robot", robot,
           "--num-envs", str(args.num_envs), "--seed", str(args.seed), "--worker-output", out]
    print(f"[{robot}] launching worker ...", flush=True)
    res = subprocess.run(cmd, env={**os.environ, "PYTHONNOUSERSITE": "1"},
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0 or not os.path.exists(out):
        sys.stderr.write(res.stdout[-4000:])
        raise SystemExit(f"[{robot}] worker failed (rc={res.returncode})")
    with open(out) as fh:
        data = json.load(fh)
    os.remove(out)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=911, help="unused-in-any-campaign seed")
    p.add_argument("--tolerance-mps", type=float, default=0.25)
    p.add_argument("--tolerance-altitude-m", type=float, default=0.15)
    p.add_argument("--tolerance-slip-mps", type=float, default=0.25)
    p.add_argument("--tolerance-reversal-s", type=float, default=0.20)
    p.add_argument("--max-motor-saturation-fraction", type=float, default=0.25)
    p.add_argument("--max-attitude-response-s", type=float, default=0.25)
    p.add_argument("--max-attitude-overshoot-deg", type=float, default=5.0)
    p.add_argument("--output", default=None)
    p.add_argument(
        "--candidate-robot",
        default="navrl_ref5in_quad",
        choices=("navrl_ref5in_quad", "navrl_ref5in_v2_quad"),
        help="candidate used by parent comparison; default preserves the historical verifier",
    )
    p.add_argument("--robot", default=None, help="worker mode: measure this one platform")
    p.add_argument("--worker-output", default=None)
    args = p.parse_args()

    if args.robot:
        with open(args.worker_output, "w") as fh:
            json.dump(verify(args.robot, args.num_envs, args.seed), fh, indent=2)
        return 0

    # Legacy first: it is the reference the gates compare against, and running it first means a
    # failure in the new platform cannot be blamed on the old one.
    report = {
        "schema_version": SCHEMA_VERSION,
        "control_hz": CONTROL_HZ,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "bars": 0,
        "speed_governor": "off",
        "closed_contract": dict(_V2_FLIGHT_CONTRACT),
        "comparison_condition": {
            "platform_scope": "whole_platform_candidate",
            "controller_gain_treatment": "same_inherited_fixed_gains_no_candidate_retuning",
            "collision_equivalence": False,
            "collision_note": (
                "candidate height 0.12 m changes tilted contact geometry versus legacy 0.08 m"
            ),
        },
    }
    report["legacy"] = _run_worker("navrl_quad", args)
    report["ref5in"] = _run_worker(args.candidate_robot, args)
    report["checks"] = _verdict(
        report["legacy"],
        report["ref5in"],
        args.tolerance_mps,
        args.tolerance_altitude_m,
        args.tolerance_slip_mps,
        args.tolerance_reversal_s,
        args.max_motor_saturation_fraction,
        args.max_attitude_response_s,
        args.max_attitude_overshoot_deg,
    )
    report["verdict"] = "PASS" if all(c["pass"] for c in report["checks"]) else "FAIL"

    print("\n" + "=" * 78)
    for arm in ("legacy", "ref5in"):
        r = report[arm]
        print(f"[{arm:7s}] {r['robot']}")
        print(f"   hover    z err {r['hover']['altitude_error_m']:+.3f} m   "
              f"drift {r['hover']['speed_mean_mps']:.3f} m/s   live {r['hover']['n_live']}")
        print(f"   forward  vx {r['forward']['steady_mean']:+.3f} / {r['v_max']:.2f} m/s   "
              f"|v| {r['forward']['world_speed']['steady_mean']:.3f}   "
              f"t90 {r['forward']['rise_time_90pct_s']} s   "
              f"z_peak {r['forward']['altitude']['peak_error_abs_m']:.3f} m   "
              f"live {r['forward']['n_live']}")
        print(f"   reversal t90 {r['reversal']['reversal_time_90pct_s']} s   "
              f"0-cross {r['reversal']['zero_crossing_s']} s   "
              f"proj {r['reversal']['steady_mean']:+.3f} m/s   "
              f"z_peak {r['reversal']['altitude']['peak_error_abs_m']:.3f} m   "
              f"live {r['reversal']['n_live']}")
        print(f"   yaw      {r['yaw']['steady_mean']:+.3f} / {r['yaw_rate_max']:.2f} rad/s   "
              f"t90 {r['yaw']['rise_time_90pct_s']} s   live {r['yaw']['n_live']}")
        fixed = r["physics_rate_fixed_gain_response"]
        print(
            "   fixed-K  pitch 20deg %s s peak %.1fdeg | roll 20deg %s s peak %.1fdeg"
            % (
                fixed["pitch"]["time_to_threshold_s"],
                fixed["pitch"]["peak_angle_deg"],
                fixed["roll"]["time_to_threshold_s"],
                fixed["roll"]["peak_angle_deg"],
            )
        )
    print("-" * 78)
    for c in report["checks"]:
        print(f"   [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: {c['detail']}")
    print(f"   VERDICT: {report['verdict']}")
    print("=" * 78)

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {args.output}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
