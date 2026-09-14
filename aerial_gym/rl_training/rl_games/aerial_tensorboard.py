"""Curated TensorBoard scalars for intercept PPO (epoch axis only)."""

from __future__ import annotations

import os
from typing import Any, Optional

from rl_games.algos_torch import torch_ext


def _reward_warmup_epochs() -> int:
    raw = os.environ.get("NAVRL_TB_REWARD_WARMUP_EPOCHS", "").strip()
    if not raw:
        return 20
    try:
        return max(0, int(raw))
    except ValueError:
        return 20


_REWARD_WARMUP_EPOCHS = _reward_warmup_epochs()


def _f(x: Any) -> float:
    if hasattr(x, "detach"):
        return float(x.detach().cpu().item())
    return float(x)


def write_aerial_epoch_scalars(
    writer,
    epoch_num: int,
    *,
    mean_reward: Optional[float] = None,
    mean_episode_length: Optional[float] = None,
    a_losses,
    c_losses,
    entropies,
    kls,
    last_lr: float,
    lr_mul: float,
    step_time: float,
    curr_frames: float,
    intercept_succ: int = 0,
    intercept_done: int = 0,
    intercept_envs_hit: int = 0,
    num_parallel_envs: int = 1,
    mean_target_dist_m: Optional[float] = None,
    mean_closest_target_dist_m: Optional[float] = None,
    explained_variance: Optional[Any] = None,
    extra_intercept_metrics: Optional[dict[str, Any]] = None,
) -> None:
    """Log only metrics useful for intercept training; x-axis = PPO epoch."""
    ep = int(epoch_num)
    w = writer

    # The first epochs sit at a large negative reward (an untrained policy crashes immediately and
    # eats the collision penalty), which stretches the TensorBoard y-axis so far that the entire
    # rest of the curve is squashed into a flat line. Skip those points: they carry no information
    # that the console dashboard does not already print, and dropping them makes the chart legible
    # from the first epoch that means anything. NAVRL_TB_REWARD_WARMUP_EPOCHS=0 logs from epoch 0.
    if mean_reward is not None and ep >= _REWARD_WARMUP_EPOCHS:
        w.add_scalar("aerial/mean_reward", mean_reward, ep)
    if mean_episode_length is not None:
        w.add_scalar("aerial/mean_episode_length", mean_episode_length, ep)
    if mean_target_dist_m is not None:
        w.add_scalar("aerial/mean_target_dist_m", mean_target_dist_m, ep)
    if mean_closest_target_dist_m is not None:
        w.add_scalar("aerial/mean_closest_target_dist_m", mean_closest_target_dist_m, ep)
    if extra_intercept_metrics:
        # intercept/* and the aerial/*_target_dist scalars only apply to the interception task;
        # skip them entirely when no intercept episodes were recorded (e.g. navrl_task) so they
        # don't leak as constant-zero curves. The task-namespaced pass-through below is unaffected.
        if intercept_done > 0:
            scalar_map = {
                "success_closest_target_dist_m": "aerial/success_closest_target_dist_m",
                "failure_closest_target_dist_m": "aerial/failure_closest_target_dist_m",
                "mean_surface_gap_m": "aerial/mean_surface_gap_m",
                "failure_surface_gap_m": "aerial/failure_surface_gap_m",
                "near_miss_count": "intercept/near_miss_count",
                "near_miss_rate": "intercept/near_miss_rate",
                "radius_no_contact_count": "intercept/radius_no_contact_count",
                "radius_no_contact_rate": "intercept/radius_no_contact_rate",
                "contact_no_radius_count": "intercept/contact_no_radius_count",
                "contact_no_radius_rate": "intercept/contact_no_radius_rate",
            }
            for key, tag in scalar_map.items():
                value = extra_intercept_metrics.get(key)
                if value is not None:
                    w.add_scalar(tag, float(value), ep)
        # task-namespaced metrics (e.g. navrl/*) are written under their own tag as-is
        for key, value in extra_intercept_metrics.items():
            if "/" in key and value is not None:
                w.add_scalar(key, float(value), ep)

    # intercept/* scalars belong to the interception task only. Skip them entirely when no
    # intercept episodes were recorded this epoch (e.g. navrl_task never feeds that summarizer),
    # so navrl runs don't get constant-zero intercept/* clutter in TensorBoard.
    if intercept_done > 0:
        w.add_scalar("intercept/ep_finished", int(intercept_done), ep)
        w.add_scalar("intercept/success_rate", float(intercept_succ) / float(intercept_done), ep)
        if num_parallel_envs > 0:
            w.add_scalar("intercept/env_hit_rate", float(intercept_envs_hit) / float(num_parallel_envs), ep)

    w.add_scalar("ppo/a_loss", torch_ext.mean_list(a_losses).item(), ep)
    w.add_scalar("ppo/c_loss", torch_ext.mean_list(c_losses).item(), ep)
    w.add_scalar("ppo/entropy", torch_ext.mean_list(entropies).item(), ep)
    w.add_scalar("ppo/kl", torch_ext.mean_list(kls).item(), ep)
    w.add_scalar("ppo/learning_rate", float(last_lr) * float(lr_mul), ep)

    if explained_variance is not None:
        w.add_scalar("ppo/explained_variance", _f(explained_variance), ep)

    if step_time and step_time > 0:
        w.add_scalar("perf/step_fps", float(curr_frames) / float(step_time), ep)
