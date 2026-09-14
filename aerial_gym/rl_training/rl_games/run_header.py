"""Compact run banner: title, CUDA/devices, and key training settings."""

from __future__ import annotations

import importlib
import os
from typing import Any, Mapping, Optional

import torch

from aerial_gym.rl_training.rl_games.quiet_rl_io import real_print

_BAR = "\u2550" * 54

_TASK_TITLES: dict[str, str] = {
    "position_setpoint_task": "Quadrotor Intercept \u00b7 PPO",
    "position_setpoint_task_sim2real": "Quadrotor Setpoint (sim2real) \u00b7 PPO",
    "position_setpoint_task_sim2real_px4": "Quadrotor Setpoint PX4 \u00b7 PPO",
    "position_setpoint_task_acceleration_sim2real": "Quadrotor Accel Setpoint \u00b7 PPO",
    "position_setpoint_task_reconfigurable": "Quadrotor Reconfigurable \u00b7 PPO",
    "position_setpoint_task_morphy": "Quadrotor Morphy \u00b7 PPO",
    "position_setpoint_task_sim2real_end_to_end": "Quadrotor E2E sim2real \u00b7 PPO",
    "navigation_task": "Quadrotor Navigation \u00b7 PPO",
    "shooting_moving_target_task": "Moving Target Intercept \u00b7 PPO",
    "navrl_task": "NavRL Navigation \u00b7 PPO",
}

_ENV_LABELS: dict[str, str] = {
    "empty_env": "empty_env (stationary intercept target)",
}


def _kv(key: str, value: object) -> str:
    return f"  {key:<14}: {value}"


def _cuda_summary(sim_device: str, rl_device: str) -> tuple[str, str]:
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        try:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = "GPU (name unavailable)"
        line = f"yes  \u00b7  {name}"
    else:
        line = "no (CPU only)"
    return line, f"sim {sim_device}  \u00b7  rl {rl_device}"


def _task_env_label(task: str) -> tuple[str, Optional[str]]:
    title = _TASK_TITLES.get(task, f"{task} \u00b7 PPO")
    env_name: Optional[str] = None
    cfg_mod = f"aerial_gym.config.task_config.{task}_config"
    try:
        mod = importlib.import_module(cfg_mod)
        tc = getattr(mod, "task_config", None)
        if tc is not None:
            env_name = getattr(tc, "env_name", None)
    except Exception:
        pass
    if env_name == "moving_intercept_env":
        env_label = _moving_intercept_env_label()
    else:
        env_label = _ENV_LABELS.get(env_name, env_name) if env_name else None
    return title, env_label


def _moving_intercept_env_label() -> str:
    try:
        from aerial_gym.config.env_config.moving_intercept_env import MovingInterceptEnvCfg
        from aerial_gym.config.task_config.shooting_moving_target_task_config import task_config

        half_width = float(MovingInterceptEnvCfg.env.env_spacing)
        width = 2.0 * half_width
        lo = float(task_config.target_speed_min)
        hi = float(task_config.target_speed_max)
        return (
            f"moving_intercept_env ({width:g} m cube, "
            f"kinematic target {lo:.2f}–{hi:.2f} m/s)"
        )
    except Exception:
        return "moving_intercept_env"


def _intercept_radius_m(task: str) -> Optional[float]:
    if task not in ("position_setpoint_task", "shooting_moving_target_task"):
        return None
    mod_name = (
        "position_setpoint_task_config"
        if task == "position_setpoint_task"
        else "shooting_moving_target_task_config"
    )
    try:
        mod = importlib.import_module(f"aerial_gym.config.task_config.{mod_name}")
        return float(mod.task_config.intercept_success_radius)
    except Exception:
        return None


def _moving_target_speed_line(task: str) -> Optional[str]:
    if task != "shooting_moving_target_task":
        return None
    try:
        from aerial_gym.config.task_config.shooting_moving_target_task_config import task_config

        lo = float(task_config.target_speed_min)
        hi = float(task_config.target_speed_max)
        if bool(getattr(task_config, "target_speed_curriculum_enable", False)):
            slo = float(getattr(task_config, "target_speed_start_min", lo))
            shi = float(getattr(task_config, "target_speed_start_max", hi))
            end = int(getattr(task_config, "target_speed_curriculum_end_epoch", 0))
            return (
                f"{slo:.2f}–{shi:.2f} → {lo:.2f}–{hi:.2f} m/s "
                f"(curriculum to epoch {end})"
            )
        return f"{lo:.2f}–{hi:.2f} m/s horizontal (resampled each episode)"
    except Exception:
        return None


def _early_stop_line(cfg: Mapping[str, Any]) -> str:
    parts = ["NaN/Inf fail-fast"]
    stable = cfg.get("early_stop_stable") or {}
    if stable.get("enable"):
        parts.append(
            f"plateau after {stable.get('min_epoch', '?')} "
            f"({stable.get('consecutive_epochs', '?')} epochs, "
            f"band ≤ {stable.get('reward_band', '?')})"
        )
    collapse = cfg.get("early_stop_collapse") or {}
    if collapse.get("enable"):
        parts.append(
            f"collapse after {collapse.get('min_epoch', '?')} "
            f"(drop {collapse.get('drop_from_peak', '?')}, "
            f"patience {collapse.get('patience_epochs', '?')})"
        )
    elif collapse.get("disabled_by"):
        parts.append(f"reward-collapse guard disabled by {collapse['disabled_by']}")
    density_capture = cfg.get("early_stop_density_capture") or {}
    if density_capture.get("enable"):
        parts.append(
            "same-density capture guard "
            f"({density_capture.get('window_epochs', '?')} epoch window, "
            f"drop {density_capture.get('drop_absolute', '?')}, "
            f"patience {density_capture.get('patience_epochs', '?')})"
        )
    return "on (" + "; ".join(parts) + ")"


def _checkpoint_line(args: Mapping[str, Any]) -> str:
    ck = args.get("checkpoint")
    if ck is None or str(ck).strip() == "":
        return "fresh weights"
    return f"resume  \u00b7  {ck}"


def _tensorboard_logdir(cfg: Mapping[str, Any]) -> str:
    train_dir = str(cfg.get("train_dir") or "runs")
    run_name = cfg.get("full_experiment_name") or cfg.get("name") or "run"
    return os.path.abspath(os.path.join(os.getcwd(), train_dir, str(run_name), "summaries"))


def print_run_header(
    args: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    mode: str = "train",
) -> None:
    """Print once after config load; uses real_print so quiet filter does not hide it."""
    params = config.get("params") or {}
    cfg = params.get("config") or {}
    task = str(args.get("task") or cfg.get("env_name") or "unknown")
    title, env_label = _task_env_label(task)

    if mode == "play":
        title = title.replace(" \u00b7 PPO", " \u00b7 PLAY (eval)")

    sim_device = str(args.get("sim_device") or "cuda:0")
    rl_device = str(args.get("rl_device") or "cuda:0")
    cuda_line, devices_line = _cuda_summary(sim_device, rl_device)

    run_name = cfg.get("full_experiment_name") or cfg.get("name") or "\u2014"
    yaml_file = args.get("file") or "\u2014"
    tb_dir = _tensorboard_logdir(cfg)

    lines = [
        _BAR,
        f"  {title}",
        _BAR,
        _kv("task", task),
    ]
    if env_label:
        lines.append(_kv("sim env", env_label))
    lines.extend(
        [
            _kv("run folder", run_name),
            _kv("config", yaml_file),
            "",
            _kv("CUDA", cuda_line),
            _kv("devices", devices_line),
            _kv("num envs", args.get("num_envs", cfg.get("num_actors", "?"))),
            _kv("headless", args.get("headless", cfg.get("env_config", {}).get("headless", "?"))),
            _kv("use_warp", args.get("use_warp", cfg.get("env_config", {}).get("use_warp", "?"))),
            # NavRL play uses NAVRL_SEED because rl-games discards --seed in the player path.
            # Prefer the runtime override so a seed-43 held-out is not mislabeled as YAML seed 42.
            _kv(
                "seed",
                os.environ.get("NAVRL_SEED", params.get("seed", args.get("seed", "?"))),
            ),
        ]
    )

    if mode == "train":
        lines.extend(
            [
                "",
                _kv("checkpoint", _checkpoint_line(args)),
                _kv("max epochs", cfg.get("max_epochs", "?")),
                _kv("early stop", _early_stop_line(cfg)),
                _kv("TensorBoard", tb_dir),
                _kv("TB scalars", "aerial/*, intercept/*, ppo/* (not losses/*)"),
            ]
        )
        r = _intercept_radius_m(task)
        if r is not None:
            lines.append(_kv("intercept r", f"{r:.2f} m (contact @ center dist ≤ r)"))
        spd = _moving_target_speed_line(task)
        if spd is not None:
            lines.append(_kv("target speed", spd))
        try:
            cfg_mod = (
                "position_setpoint_task_config"
                if task == "position_setpoint_task"
                else "shooting_moving_target_task_config"
                if task == "shooting_moving_target_task"
                else None
            )
            if cfg_mod:
                mod = importlib.import_module(f"aerial_gym.config.task_config.{cfg_mod}")
                dm = getattr(mod.task_config, "reward_episode_design_max", None)
                if dm is not None:
                    lines.append(_kv("reward budget", f"ep max ~{float(dm):.0f} (design target)"))
        except Exception:
            pass
    else:
        try:
            games = int(cfg.get("player", {}).get("games_num", 64))
        except (TypeError, ValueError):
            games = 64
        lines.extend(
            [
                "",
                _kv("checkpoint", _checkpoint_line(args)),
                _kv("episodes", games),
            ]
        )
        r = _intercept_radius_m(task)
        if r is not None:
            lines.append(_kv("intercept r", f"{r:.2f} m (contact @ center dist ≤ r)"))

    lines.append(_BAR)
    real_print("\n" + "\n".join(lines) + "\n", flush=True)
