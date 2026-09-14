"""
Replace rl-games long one-line FPS print with a short boxed dashboard (Linux-friendly Unicode).

Disable boxed output:  AERIAL_RL_PLAIN_STATS=1

Continuous PPO (EarlyStopA2CAgent) prints the same box after mean episode rewards exist.
"""
from __future__ import annotations

import os
import importlib
from typing import List, Optional

_PATCHED_ATTR = "_aerial_boxed_statistics_patch"

_rlgames_print_statistics_original = None


def _dashboard_title() -> str:
    """Task-aware banner title; the runner sets AERIAL_RUN_TITLE per task."""
    return os.environ.get("AERIAL_RUN_TITLE", "").strip() or "Aerial RL  ·  PPO"


def _reward_design_suffix() -> str:
    """Optional '(design ~0-N)' hint; only shown when the task declares a design range."""
    raw = os.environ.get("AERIAL_REWARD_DESIGN_MAX", "").strip()
    if not raw:
        return ""
    try:
        return f"  (design ~0–{float(raw):.0f})"
    except ValueError:
        return ""


def _episode_len_max() -> int:
    raw = os.environ.get("AERIAL_TRAIN_EP_LEN_MAX", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    cfg_mod = os.environ.get("AERIAL_TASK_CONFIG_MODULE", "").strip()
    if cfg_mod:
        try:
            mod = importlib.import_module(cfg_mod)
            return int(mod.task_config.episode_len_steps)
        except Exception:
            pass
    try:
        from aerial_gym.config.task_config.position_setpoint_task_config import task_config

        return int(task_config.episode_len_steps)
    except Exception:
        return 1000


def _format_dashboard_rows(
    epoch_num: float,
    max_epochs: float,
    step_time: float,
    *,
    mean_reward: Optional[float] = None,
    mean_episode_length: Optional[float] = None,
    intercept_lines: Optional[list[str]] = None,
) -> List[str]:
    if max_epochs == -1:
        ep_s = f"{epoch_num:.0f}"
    else:
        ep_s = f"{epoch_num:.0f} / {max_epochs:.0f}"

    def _cell(v: Optional[float], fmt: str) -> str:
        if v is None:
            return "\u2014"
        return format(v, fmt)

    ep_max = _episode_len_max()
    ep_len_s = _cell(mean_episode_length, ",.1f")
    if mean_episode_length is not None:
        ep_len_s = f"{ep_len_s} / {ep_max}"

    bar = "\u2500" * 52
    rows: List[str] = [
        bar,
        f"  {_dashboard_title()}",
    ]
    _run = os.environ.get("AERIAL_RUN_NAME", "").strip()
    if _run:
        rows.append(f"  run            : {_run}")
    rows += [
        "",
        f"  epoch          : {ep_s}",
        f"  step time (s)  : {float(step_time):.4f}",
        f"  mean reward    : {_cell(mean_reward, ',.4f')}{_reward_design_suffix()}",
    ]
    if intercept_lines:
        for ln in intercept_lines:
            s = ln.rstrip()
            rows.append(s if s.startswith("  ") else f"  {s}")
    rows.append(f"  mean ep length : {ep_len_s}")
    rows.append(bar)
    return rows


def print_training_dashboard(
    print_stats,
    curr_frames,
    step_time,
    step_inference_time,
    total_time,
    epoch_num,
    max_epochs,
    frame,
    max_frames,
    *,
    mean_reward: Optional[float] = None,
    mean_episode_length: Optional[float] = None,
    intercept_lines: Optional[list[str]] = None,
) -> None:
    """One epoch summary: reward, play-style intercept success/fail, episode length."""
    if not print_stats:
        return
    plain = os.environ.get("AERIAL_RL_PLAIN_STATS", "").strip().lower() in ("1", "true", "yes", "on")
    ep_max = _episode_len_max()
    if plain:
        orig = _rlgames_print_statistics_original
        if orig is None:
            from rl_games.common import a2c_common as _ac

            orig = _ac.print_statistics
        orig(
            print_stats,
            curr_frames,
            step_time,
            step_inference_time,
            total_time,
            epoch_num,
            max_epochs,
            frame,
            max_frames,
        )
        if mean_reward is not None:
            print(f"  mean reward    : {mean_reward:,.4f}", flush=True)
        if intercept_lines:
            for ln in intercept_lines:
                print(ln if ln.startswith("  ") else f"  {ln.strip()}", flush=True)
        if mean_episode_length is not None:
            print(f"  mean ep length : {mean_episode_length:,.1f} / {ep_max}", flush=True)
        return

    msg = "\n".join(
        _format_dashboard_rows(
            epoch_num,
            max_epochs,
            step_time,
            mean_reward=mean_reward,
            mean_episode_length=mean_episode_length,
            intercept_lines=intercept_lines,
        )
    )
    print("\n" + msg + "\n", flush=True)


def install_boxed_statistics() -> None:
    global _rlgames_print_statistics_original
    if os.environ.get("AERIAL_RL_PLAIN_STATS", "").strip().lower() in ("1", "true", "yes", "on"):
        return
    try:
        from rl_games.common import a2c_common
    except ImportError:
        return

    if _rlgames_print_statistics_original is None:
        _rlgames_print_statistics_original = a2c_common.print_statistics

    if getattr(a2c_common.print_statistics, _PATCHED_ATTR, False):
        return

    def print_statistics(
        print_stats,
        curr_frames,
        step_time,
        step_inference_time,
        total_time,
        epoch_num,
        max_epochs,
        frame,
        max_frames,
    ):
        if not print_stats:
            return
        rows = _format_dashboard_rows(
            epoch_num,
            max_epochs,
            step_time,
            mean_reward=None,
            mean_episode_length=None,
            intercept_lines=None,
        )
        print("\n" + "\n".join(rows) + "\n", flush=True)

    setattr(print_statistics, _PATCHED_ATTR, True)
    a2c_common.print_statistics = print_statistics
