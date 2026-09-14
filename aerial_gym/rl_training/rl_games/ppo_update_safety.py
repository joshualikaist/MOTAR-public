"""Small, simulator-independent PPO safety primitives used by NavRL training."""

import copy
from collections.abc import Iterable, Mapping as MappingABC, MutableMapping
from dataclasses import dataclass
import math
from numbers import Real
import operator
import os
from typing import Any, Mapping, Optional, Tuple

import torch


@dataclass(frozen=True)
class PPOEpochTransaction:
    """In-memory actor-update snapshot used to reject an unsafe PPO epoch.

    ``state_dict()`` values are reference-backed, so callers must not build this object directly.
    :func:`capture_epoch_transaction` deep-copies parameters, persistent model buffers, optimizer
    moments and the optional AMP scaler.  Module train/eval flags are kept separately because they
    are not part of a module state dict.
    """

    model_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any]
    scaler_state: Optional[Mapping[str, Any]]
    module_training: Tuple[Tuple[str, bool], ...]


PPO_ROLLBACK_TOTAL_KEY = "aerial_ppo_rollback_total"
PPO_ROLLBACK_STREAK_KEY = "aerial_ppo_rollback_streak"


def _nonnegative_counter(value, label: str) -> int:
    """Parse a checkpoint counter without accepting booleans or fractional values."""

    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(parsed)


def add_ppo_rollback_checkpoint_state(state, *, total, streak):
    """Add durable PPO rollback counters to an rl-games full-state checkpoint.

    The function mutates and returns ``state`` so callers can add the counters after the base
    agent has assembled model, optimizer, frame, epoch and environment state.  Strict integer
    validation prevents a damaged checkpoint from silently resetting its livelock history.
    """

    if not isinstance(state, MutableMapping):
        raise TypeError("checkpoint state must be a mutable mapping")
    parsed_total = _nonnegative_counter(total, PPO_ROLLBACK_TOTAL_KEY)
    parsed_streak = _nonnegative_counter(streak, PPO_ROLLBACK_STREAK_KEY)
    if parsed_streak > parsed_total:
        raise ValueError("PPO rollback streak cannot exceed rollback total")
    state[PPO_ROLLBACK_TOTAL_KEY] = parsed_total
    state[PPO_ROLLBACK_STREAK_KEY] = parsed_streak
    return state


def read_ppo_rollback_checkpoint_state(weights):
    """Return ``(total, streak)`` from a checkpoint, defaulting legacy files to zero."""

    if not isinstance(weights, MappingABC):
        raise TypeError("checkpoint weights must be a mapping")
    total = _nonnegative_counter(
        weights.get(PPO_ROLLBACK_TOTAL_KEY, 0), PPO_ROLLBACK_TOTAL_KEY
    )
    streak = _nonnegative_counter(
        weights.get(PPO_ROLLBACK_STREAK_KEY, 0), PPO_ROLLBACK_STREAK_KEY
    )
    if streak > total:
        raise ValueError("checkpoint PPO rollback streak exceeds rollback total")
    return total, streak


def completed_rollout_frames(curr_frames, world_size=1, *, multi_gpu=False) -> int:
    """Return the global frame increment for a rollout that has already been consumed."""

    frames = _nonnegative_counter(curr_frames, "curr_frames")
    ranks = _nonnegative_counter(world_size, "world_size")
    if frames < 1:
        raise ValueError("curr_frames must be >= 1")
    if ranks < 1:
        raise ValueError("world_size must be >= 1")
    return frames * ranks if multi_gpu else frames


def resolve_action_learning_rate(
    configured,
    *,
    explicit_override=None,
    saved_current=None,
    resume_training=False,
) -> float:
    """Choose the optimizer LR without undoing a checkpoint's safety backoff.

    An explicit launch setting wins.  Otherwise a training resume inherits the checkpoint's
    *current* LR, not its original configured LR; fresh training and evaluation keep the YAML
    value.  Invalid present values fail closed instead of quietly selecting a larger default.
    """

    def positive_finite(value, label):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be numeric; got {value!r}") from exc
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{label} must be finite and > 0; got {value!r}")
        return number

    configured_lr = positive_finite(configured, "configured learning rate")
    if explicit_override is not None and str(explicit_override).strip():
        return positive_finite(explicit_override, "NAVRL_LEARNING_RATE")
    if resume_training and saved_current is not None:
        return positive_finite(saved_current, "checkpoint current_action_learning_rate")
    return configured_lr


def capture_epoch_transaction(model, optimizer, scaler=None) -> PPOEpochTransaction:
    """Capture an exact, reusable snapshot of actor-side PPO update state.

    The snapshot intentionally excludes environment state and epoch/frame counters.  Rejecting a
    policy update must not replay an already-consumed rollout or undo curriculum progression.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    if scaler is not None and not (
        hasattr(scaler, "state_dict") and hasattr(scaler, "load_state_dict")
    ):
        raise TypeError("scaler must provide state_dict/load_state_dict")

    return PPOEpochTransaction(
        model_state=copy.deepcopy(model.state_dict()),
        optimizer_state=copy.deepcopy(optimizer.state_dict()),
        scaler_state=(copy.deepcopy(scaler.state_dict()) if scaler is not None else None),
        module_training=tuple(
            (name, bool(module.training)) for name, module in model.named_modules()
        ),
    )


def restore_epoch_transaction(snapshot, model, optimizer, scaler=None) -> None:
    """Restore a snapshot and discard gradients left by the rejected update.

    Optimizer state is deep-copied again on load.  Some PyTorch optimizers otherwise retain tensor
    aliases to the supplied state dict, allowing a later optimizer step to corrupt a reusable
    last-known-good snapshot.
    """

    if not isinstance(snapshot, PPOEpochTransaction):
        raise TypeError("snapshot must be a PPOEpochTransaction")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")
    if snapshot.scaler_state is not None and scaler is None:
        raise ValueError("snapshot contains scaler state but no scaler was provided")

    model.load_state_dict(snapshot.model_state, strict=True)
    optimizer.load_state_dict(copy.deepcopy(snapshot.optimizer_state))
    if snapshot.scaler_state is not None:
        if not (hasattr(scaler, "state_dict") and hasattr(scaler, "load_state_dict")):
            raise TypeError("scaler must provide state_dict/load_state_dict")
        scaler.load_state_dict(copy.deepcopy(snapshot.scaler_state))

    modules = dict(model.named_modules())
    expected_names = {name for name, _training in snapshot.module_training}
    if set(modules) != expected_names:
        raise ValueError("model module structure changed after the epoch snapshot")
    # Assign the flags directly. Calling ``train(mode)`` recursively would overwrite intentionally
    # mixed states such as a training model with frozen RunningMeanStd submodules.
    for name, training in snapshot.module_training:
        modules[name].training = training

    optimizer.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.grad = None


def exact_normal_kl(
    current_mu,
    current_sigma,
    reference_mu,
    reference_sigma,
    *,
    reduce=True,
):
    """Return ``KL(current Normal || reference Normal)`` summed over action axes.

    Unlike rl_games' helper, this formula does not add epsilon inside otherwise valid variances, so
    identical distributions return exactly zero.  Invalid/non-positive scales intentionally
    produce non-finite output for :func:`should_reject_ppo_update` to reject.
    """

    values = (current_mu, current_sigma, reference_mu, reference_sigma)
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("Normal means and scales must be torch.Tensor instances")
    try:
        current_mu, current_sigma, reference_mu, reference_sigma = torch.broadcast_tensors(
            *values
        )
    except RuntimeError as exc:
        raise ValueError("Normal means and scales must be broadcast-compatible") from exc
    if current_mu.ndim == 0:
        raise ValueError("Normal parameters must include an action dimension")

    per_axis = (
        torch.log(reference_sigma / current_sigma)
        + (
            current_sigma.square()
            + (reference_mu - current_mu).square()
        )
        / (2.0 * reference_sigma.square())
        - 0.5
    )
    valid_scale = (current_sigma > 0.0) & (reference_sigma > 0.0)
    per_axis = torch.where(
        valid_scale,
        per_axis,
        torch.full_like(per_axis, float("nan")),
    )
    per_sample = per_axis.sum(dim=-1)
    return per_sample.mean() if reduce else per_sample


def _all_finite(value) -> bool:
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        return value.numel() > 0 and bool(torch.isfinite(value).all().item())
    if isinstance(value, Real):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return all(_all_finite(item) for item in value)
    raise TypeError("finite values must be tensors, real numbers, mappings, or sequences")


def all_finite_ppo_state(*values) -> bool:
    """Return whether every tensor/number in nested PPO state is finite.

    This is intentionally public so an epoch commit can audit model buffers, optimizer moments,
    AMP scaler state and accumulated losses with the same recursive rule.  A boolean status flag
    must not be used as a proxy for the values themselves: ``False`` is a finite Python number.
    """

    return all(_all_finite(value) for value in values)


def should_reject_ppo_update(kl, kl_limit, *finite_values) -> bool:
    """Return whether a PPO update must be rejected for KL or non-finite state.

    A zero ``kl_limit`` disables only the distance threshold, matching the launcher's existing
    opt-out contract.  Non-finite KL, losses, gradients or parameters are always rejected.
    Tensor-valued KL is treated conservatively: any element over the limit rejects the update.
    """

    try:
        limit = float(kl_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("kl_limit must be a finite number >= 0") from exc
    if not math.isfinite(limit) or limit < 0.0:
        raise ValueError("kl_limit must be a finite number >= 0")
    if not _all_finite(kl) or not all_finite_ppo_state(*finite_values):
        return True

    if limit == 0.0:
        return False
    if isinstance(kl, torch.Tensor):
        return bool((kl > limit).any().item())
    if isinstance(kl, Real):
        return float(kl) > limit
    raise TypeError("kl must be a tensor or real number")


def stable_ppo_actor_loss(
    old_action_neglog_probs_batch,
    action_neglog_probs,
    advantage,
    is_ppo,
    curr_e_clip,
    max_abs_log_ratio,
):
    """PPO surrogate with a finite log-ratio before ``exp``.

    rl-games exponentiates the raw log-ratio. Near a tanh boundary, a very unlikely replayed action
    can make that difference large enough to overflow before PPO's ordinary ratio clip is applied.
    The wide default clamp is only a numerical guard; the agent's separate KL gate rejects updates
    that have moved materially too far.
    """
    if not is_ppo:
        return action_neglog_probs * advantage
    log_ratio = old_action_neglog_probs_batch - action_neglog_probs
    ratio = torch.exp(torch.clamp(log_ratio, -max_abs_log_ratio, max_abs_log_ratio))
    surr1 = advantage * ratio
    surr2 = advantage * torch.clamp(
        ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip
    )
    return torch.maximum(-surr1, -surr2)


def lateral_latent_margin_loss(mu, margin, axis=1):
    """Mean squared excess of the selected latent mean beyond a soft symmetric margin."""
    if not isinstance(mu, torch.Tensor) or mu.ndim != 2 or mu.shape[1] <= axis:
        raise ValueError("policy mean does not contain the requested lateral action axis")
    return torch.relu(mu[:, axis].abs() - margin).square().mean()


def latent_margin_loss(mu, margins):
    """Penalize latent policy means outside per-axis soft margins.

    A lateral-only guard is insufficient for a squashed Gaussian: any action axis can saturate
    tanh and make bounded-action replay numerically non-invertible.  ``margins`` therefore has one
    positive value per action axis (or a scalar which is broadcast to every axis).
    """
    if not isinstance(mu, torch.Tensor) or mu.ndim != 2:
        raise ValueError("policy mean must have shape [batch, actions]")
    margin_tensor = torch.as_tensor(margins, dtype=mu.dtype, device=mu.device)
    if margin_tensor.ndim == 0:
        margin_tensor = margin_tensor.expand(mu.shape[1])
    if margin_tensor.ndim != 1 or margin_tensor.numel() != mu.shape[1]:
        raise ValueError("latent margins must be a scalar or contain one value per action axis")
    if not bool(torch.isfinite(margin_tensor).all()) or bool((margin_tensor <= 0).any()):
        raise ValueError("latent margins must be finite and > 0")
    return torch.relu(mu.abs() - margin_tensor).square().mean()


def lateral_batch_bias_loss(mu, axis=1):
    """Penalize a population-wide signed lateral preference, not avoidance magnitude."""
    if not isinstance(mu, torch.Tensor) or mu.ndim != 2 or mu.shape[1] <= axis:
        raise ValueError("policy mean does not contain the requested lateral action axis")
    return mu[:, axis].mean().square()


def mirror_navrl_actions(actions):
    """Reflect body-frame actions across the vehicle x-z plane."""
    if not isinstance(actions, torch.Tensor) or actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError("NavRL actions must have shape [batch, 4]")
    mirrored = actions.clone()
    mirrored[:, 1] = -mirrored[:, 1]
    mirrored[:, 3] = -mirrored[:, 3]
    return mirrored


def mirror_navrl_structured_observation(obs):
    """Reflect every signed lateral field in the 898-D structured actor observation."""
    # Keep this helper simulator-independent: importing aerial_gym after torch violates Isaac Gym's
    # import-order requirement in CPU unit tests. These are the explicit structured-schema knobs
    # from navrl_perception.py; the dimension check below catches any drift.
    hbeams = int(os.environ.get("NAVRL_LIDAR_HBEAMS", "").strip() or 36)
    vbeams = int(os.environ.get("NAVRL_LIDAR_VBEAMS", "").strip() or 4)
    max_obstacles = int(os.environ.get("NAVRL_MAX_OBSTACLES", "").strip() or 5)
    obstacle_history, obstacle_dim = 5, 12
    robot_history, robot_dim = 5, 10
    target_history, target_dim = 5, 16
    static_dim = vbeams * hbeams
    structured_obs_dim = (
        static_dim
        + obstacle_history * max_obstacles * obstacle_dim
        + robot_history * robot_dim
        + target_history * target_dim
    )

    if (
        not isinstance(obs, torch.Tensor)
        or obs.ndim != 2
        or obs.shape[1] != structured_obs_dim
    ):
        raise ValueError(
            "NavRL structured observation must have shape [batch, %d]"
            % structured_obs_dim
        )
    mirrored = obs.clone()
    offset = 0

    # Physical bearings DECREASE with index: theta_i = pi - i*2pi/H. Reflection maps theta to
    # -theta, therefore index i maps to -i mod H. The former H-2-i formula belonged to the old
    # increasing convention and injected a two-bin (10 deg at H=72) skew into every reflection.
    scan = obs[:, offset : offset + static_dim].view(-1, vbeams, hbeams)
    reflect_index = (-torch.arange(hbeams, device=obs.device)) % hbeams
    mirrored[:, offset : offset + static_dim] = scan.index_select(
        2, reflect_index
    ).reshape(obs.shape[0], -1)
    offset += static_dim

    obstacle_size = obstacle_history * max_obstacles * obstacle_dim
    obstacles = mirrored[:, offset : offset + obstacle_size].view(
        -1, obstacle_history, max_obstacles, obstacle_dim
    )
    obstacles[..., 1] = -obstacles[..., 1]  # relative position y
    obstacles[..., 4] = -obstacles[..., 4]  # relative velocity y
    offset += obstacle_size

    robot_size = robot_history * robot_dim
    robot = mirrored[:, offset : offset + robot_size].view(
        -1, robot_history, robot_dim
    )
    # velocity y, yaw rate, previous action y, previous yaw action
    robot[..., (1, 3, 5, 7)] = -robot[..., (1, 3, 5, 7)]
    offset += robot_size

    target = mirrored[:, offset:].view(-1, target_history, target_dim)
    target[..., 1] = -target[..., 1]  # tracked relative position y
    target[..., 4] = -target[..., 4]  # tracked relative velocity y
    return mirrored


def reflection_equivariance_loss(mu, mirrored_mu):
    """Require a reflected observation to produce the reflected latent policy mean."""
    if (
        not isinstance(mu, torch.Tensor)
        or not isinstance(mirrored_mu, torch.Tensor)
        or mu.shape != mirrored_mu.shape
    ):
        raise ValueError("original and mirrored policy means must have matching shapes")
    return (mirrored_mu - mirror_navrl_actions(mu)).square().mean()
