"""
A2C PPO continuous agent identical to rl_games but with reward-stability early stop.

Copied from rl_games ``ContinuousA2CBase.train`` (near line 1360 in a2c_common.py) with one
inject block marked ``aerial_early_stop``. If rl_games is upgraded and training breaks here,
refresh this ``train()`` from the upstream method.
"""

import math
import os
import re
import time

import numpy as np
import torch
import torch.distributed as dist

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent

from aerial_gym.rl_training.rl_games.reward_stable_early_stop import (
    collapse_from_peak_should_stop,
    density_capture_collapse_should_stop,
    parse_density_capture_guard_config,
    parse_early_stop_collapse_config,
    parse_early_stop_stable_config,
    window_band_stable_should_stop,
)
from aerial_gym.rl_training.rl_games.aerial_tensorboard import (
    _REWARD_WARMUP_EPOCHS as _TB_REWARD_WARMUP_EPOCHS,
    write_aerial_epoch_scalars,
)
from aerial_gym.rl_training.rl_games.pretty_train_stats import print_training_dashboard
from aerial_gym.rl_training.rl_games.ppo_update_safety import (
    add_ppo_rollback_checkpoint_state,
    all_finite_ppo_state,
    capture_epoch_transaction,
    completed_rollout_frames,
    exact_normal_kl,
    latent_margin_loss,
    lateral_batch_bias_loss,
    lateral_latent_margin_loss,
    mirror_navrl_actions,
    mirror_navrl_structured_observation,
    reflection_equivariance_loss,
    read_ppo_rollback_checkpoint_state,
    restore_epoch_transaction,
    should_reject_ppo_update,
    stable_ppo_actor_loss,
)
from aerial_gym.rl_training.rl_games.train_run_recorder import (
    finalize_agent_run,
    record_agent_epoch,
)
from aerial_gym.rl_training.rl_games.training_safety import (
    first_nonfinite_training_value,
    is_finite_training_value,
    reset_optimizer_learning_rate,
)
from aerial_gym.task.position_setpoint_task.train_dashboard import consume_epoch_intercept_summary


_AERIAL_FINISHED_MARKER = ".aerial_training_finished"
_CKPT_REWARD_RE = re.compile(r"_rew_([-+]?\d+(?:\.\d+)?)")
_FAILED_EXIT_REASONS = {
    "early_stop_nan",
    "nonfinite_ppo",
    "early_stop_density_capture_collapse",
    "ppo_rollback_livelock",
}


def _scalar_mean_reward(mean_r0):
    """rl-games may return a CUDA tensor, CPU tensor, or NumPy scalar from game_rewards averages."""
    if isinstance(mean_r0, torch.Tensor):
        return float(mean_r0.detach().cpu().item())
    if isinstance(mean_r0, np.generic):
        return float(mean_r0)
    return float(mean_r0)


def _resolve_num_parallel_envs(agent) -> int:
    import os

    cfg = getattr(agent, "config", None) or {}
    env_cfg = cfg.get("env_config") if isinstance(cfg, dict) else None
    candidates = (
        cfg.get("num_actors") if isinstance(cfg, dict) else None,
        env_cfg.get("num_envs") if isinstance(env_cfg, dict) else None,
        os.environ.get("NUM_ENVS"),
        getattr(agent, "num_agents", None),
    )
    for raw in candidates:
        if raw is None or str(raw).strip() == "":
            continue
        try:
            v = int(raw)
            if v > 1:
                return v
        except (TypeError, ValueError):
            continue
    return 128


def _read_existing_best_reward(nn_dir: str) -> float:
    """Prevent resume runs from overwriting a better gen_ppo*.pth with a lower current policy."""
    run_root = os.path.dirname(os.path.abspath(nn_dir))
    best = -1000000000.0

    csv_path = os.path.join(run_root, "aerial_run", "epoch_metrics.csv")
    try:
        import csv

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                raw = row.get("mean_reward")
                if raw in (None, ""):
                    continue
                best = max(best, float(raw))
    except (OSError, ValueError):
        pass

    try:
        for name in os.listdir(nn_dir):
            if not name.endswith(".pth"):
                continue
            match = _CKPT_REWARD_RE.search(name)
            if match:
                best = max(best, float(match.group(1)))
    except (OSError, ValueError):
        pass

    return best


class EarlyStopA2CAgent(A2CAgent):
    def get_full_state_weights(self):
        """Extend rl-games checkpoints with durable rollback-livelock history."""

        state = super().get_full_state_weights()
        return add_ppo_rollback_checkpoint_state(
            state,
            total=getattr(self, "_aerial_ppo_rollback_total", 0),
            streak=getattr(self, "_aerial_ppo_rollback_streak", 0),
        )

    def set_full_state_weights(self, weights, set_epoch=True):
        """Restore rollback counters as part of the same full-state transaction."""

        # Validate before mutating model/optimizer state. A damaged safety counter must not leave a
        # half-restored agent that could then be trained as if the checkpoint were trustworthy.
        rollback_total, rollback_streak = read_ppo_rollback_checkpoint_state(weights)
        super().set_full_state_weights(weights, set_epoch=set_epoch)
        self._aerial_ppo_rollback_total = rollback_total
        self._aerial_ppo_rollback_streak = rollback_streak

    def _save_ppo_rollback_livelock_checkpoint(self):
        """Account for the consumed rollout and durably save the restored PPO state."""

        # Epoch rollback is deliberately single-GPU, but keep the frame rule explicit so a future
        # synchronized implementation cannot accidentally under-count completed rollouts.
        rollout_frames = completed_rollout_frames(
            self.curr_frames,
            self.world_size,
            multi_gpu=self.multi_gpu,
        )
        self.frame += rollout_frames
        checkpoint_stem = os.path.join(
            self.nn_dir,
            "last_%s_ep_%d_rew_rollback_livelock"
            % (self.config["name"], int(self.epoch_num)),
        )
        self.save(checkpoint_stem)
        checkpoint_path = checkpoint_stem + ".pth"
        self._aerial_rollback_livelock_checkpoint = checkpoint_path
        print(
            "[aerial RL] PPO rollback livelock checkpoint saved | "
            "epoch=%d frame=%d lr=%.3g path=%s"
            % (self.epoch_num, self.frame, self.last_lr, checkpoint_path),
            flush=True,
        )
        return checkpoint_path

    @staticmethod
    def _ppo_kl_gate():
        raw = os.environ.get("NAVRL_PPO_KL_STOP", "").strip()
        gate = float(raw) if raw else 0.0
        if not math.isfinite(gate) or gate < 0.0:
            raise ValueError("NAVRL_PPO_KL_STOP must be finite and >= 0")
        return gate

    @staticmethod
    def _ppo_epoch_rollback_enabled(kl_gate):
        raw = os.environ.get("NAVRL_PPO_EPOCH_ROLLBACK", "").strip().lower()
        if not raw:
            # Existing launchers already use NAVRL_PPO_KL_STOP as an opt-in safety contract.
            # Make the repaired transaction the default semantics of that contract.
            return kl_gate > 0.0
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        raise ValueError("NAVRL_PPO_EPOCH_ROLLBACK must be a boolean")

    @staticmethod
    def _slice_rollout_value(value, start, end):
        if isinstance(value, dict):
            return {
                key: EarlyStopA2CAgent._slice_rollout_value(item, start, end)
                for key, item in value.items()
            }
        return value[start:end]

    def prepare_dataset(self, batch_dict):
        """Keep immutable rollout-policy parameters beside rl-games' moving PPO references."""
        super().prepare_dataset(batch_dict)
        # PPODataset.update_mu_sigma() intentionally rewrites ``mu``/``sigma`` after every
        # minibatch. The old KL gate used those rewritten values and therefore rebased itself onto
        # the policy it had just rejected. These fields are never passed to update_mu_sigma.
        self.dataset.values_dict["behavior_mu"] = batch_dict["mus"].detach().clone()
        self.dataset.values_dict["behavior_sigma"] = batch_dict["sigmas"].detach().clone()

    def calc_gradients(self, input_dict):
        """rl-games PPO update with opt-in ratio, KL and lateral-mean safety controls."""
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        behavior_mu_batch = input_dict.get("behavior_mu", old_mu_batch)
        behavior_sigma_batch = input_dict.get("behavior_sigma", old_sigma_batch)
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = self._preproc_obs(input_dict["obs"])

        lr_mul = 1.0
        curr_e_clip = self.e_clip
        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_length
            if self.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]

        with torch.amp.autocast(
            "cuda", enabled=self.mixed_precision, dtype=torch.bfloat16
        ):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            with torch.no_grad():
                reduce_kl = rnn_masks is None
                kl_dist = exact_normal_kl(
                    mu.detach(),
                    sigma.detach(),
                    behavior_mu_batch,
                    behavior_sigma_batch,
                    reduce=reduce_kl,
                )
                if rnn_masks is not None:
                    kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

            kl_gate = self._ppo_kl_gate()
            rollback_enabled = self._ppo_epoch_rollback_enabled(kl_gate)
            # This is an early latch only. The epoch transaction below is the authority: it audits
            # the final policy over the entire rollout, then restores model/RMS/Adam/scaler if any
            # slice crossed the immutable behavior-policy boundary. NaN/Inf is a rejection too.
            skip_for_kl = rollback_enabled and should_reject_ppo_update(
                kl_dist,
                kl_gate,
                action_log_probs,
                values,
                entropy,
                mu,
                sigma,
            )
            if skip_for_kl:
                zero = action_log_probs.detach().new_zeros(())
                self.aux_loss_dict = {}
                self._aerial_epoch_kl_rejected = True
                self._aerial_epoch_kl_reject_nonfinite = not bool(
                    torch.isfinite(kl_dist).all()
                )
                if bool(torch.isfinite(kl_dist).all()):
                    self._aerial_epoch_max_pre_kl = max(
                        float(getattr(self, "_aerial_epoch_max_pre_kl", 0.0)),
                        float(kl_dist.detach().cpu()),
                    )
                self._aerial_kl_skipped_minibatches = (
                    int(getattr(self, "_aerial_kl_skipped_minibatches", 0)) + 1
                )
                skipped = self._aerial_kl_skipped_minibatches
                if skipped == 1 or skipped % 50 == 0:
                    print(
                        "[aerial RL] PPO epoch rejection latched before minibatch step | "
                        f"kl={float(kl_dist):.5f} gate={kl_gate:.5f} "
                        f"(total skipped={skipped}).",
                        flush=True,
                    )
                self.diagnostics.mini_batch(
                    self,
                    {
                        "values": value_preds_batch,
                        "returns": return_batch,
                        "new_neglogp": action_log_probs,
                        "old_neglogp": old_action_log_probs_batch,
                        "masks": rnn_masks,
                    },
                    curr_e_clip,
                    0,
                )
                self.train_result = (
                    zero,
                    zero,
                    entropy.detach().mean(),
                    kl_dist,
                    self.last_lr,
                    lr_mul,
                    mu.detach(),
                    sigma.detach(),
                    zero,
                )
                return

            raw_log_ratio_clamp = os.environ.get(
                "NAVRL_PPO_LOG_RATIO_CLAMP", ""
            ).strip()
            log_ratio_clamp = (
                float(raw_log_ratio_clamp) if raw_log_ratio_clamp else 0.0
            )
            if not math.isfinite(log_ratio_clamp) or log_ratio_clamp < 0.0:
                raise ValueError(
                    "NAVRL_PPO_LOG_RATIO_CLAMP must be finite and >= 0"
                )
            actor_loss_func = self.actor_loss_func
            if log_ratio_clamp > 0.0:
                actor_loss_func = lambda old_nlp, new_nlp, adv, is_ppo, e_clip: (
                    stable_ppo_actor_loss(
                        old_nlp,
                        new_nlp,
                        adv,
                        is_ppo,
                        e_clip,
                        log_ratio_clamp,
                    )
                )

            loss, a_loss, c_loss, entropy, b_loss, sum_mask = self.calc_losses(
                actor_loss_func,
                old_action_log_probs_batch,
                action_log_probs,
                advantage,
                curr_e_clip,
                value_preds_batch,
                values,
                return_batch,
                mu,
                entropy,
                rnn_masks,
            )

            aux_loss = self.model.get_aux_loss()
            self.aux_loss_dict = {}
            if aux_loss is not None:
                for key, value in aux_loss.items():
                    loss += value
                    if key in self.aux_loss_dict:
                        self.aux_loss_dict[key] = value.detach()
                    else:
                        self.aux_loss_dict[key] = [value.detach()]

            raw_axis_margins = os.environ.get("NAVRL_LATENT_MARGIN", "").strip()
            raw_margin = os.environ.get("NAVRL_LATENT_MARGIN_Y", "").strip()
            raw_margin_coef = os.environ.get(
                "NAVRL_LATENT_MARGIN_COEF", ""
            ).strip()
            margin_coef = float(raw_margin_coef) if raw_margin_coef else 0.0
            if not math.isfinite(margin_coef) or margin_coef < 0.0:
                raise ValueError("NAVRL_LATENT_MARGIN_COEF must be finite and >= 0")
            if raw_axis_margins and margin_coef > 0.0:
                try:
                    margins = [
                        float(value.strip())
                        for value in raw_axis_margins.split(",")
                        if value.strip()
                    ]
                except ValueError as exc:
                    raise ValueError(
                        "NAVRL_LATENT_MARGIN must be a scalar or comma-separated floats"
                    ) from exc
                margin_penalty = latent_margin_loss(mu, margins)
                weighted_margin = margin_coef * margin_penalty
                loss += weighted_margin
                self.aux_loss_dict["all_axis_latent_margin"] = [
                    weighted_margin.detach()
                ]
            elif raw_margin and margin_coef > 0.0:
                margin = float(raw_margin)
                if not math.isfinite(margin) or margin <= 0.0:
                    raise ValueError("NAVRL_LATENT_MARGIN_Y must be finite and > 0")
                weighted_margin = margin_coef * lateral_latent_margin_loss(mu, margin)
                loss += weighted_margin
                self.aux_loss_dict["lateral_latent_margin"] = [weighted_margin.detach()]

            raw_bias_coef = os.environ.get("NAVRL_LATERAL_BIAS_COEF", "").strip()
            bias_coef = float(raw_bias_coef) if raw_bias_coef else 0.0
            if not math.isfinite(bias_coef) or bias_coef < 0.0:
                raise ValueError("NAVRL_LATERAL_BIAS_COEF must be finite and >= 0")
            if bias_coef > 0.0:
                bias_penalty = lateral_batch_bias_loss(mu)
                weighted_bias = bias_coef * bias_penalty
                loss += weighted_bias
                self.aux_loss_dict["lateral_batch_bias"] = [
                    weighted_bias.detach()
                ]

            raw_reflection_coef = os.environ.get(
                "NAVRL_REFLECTION_COEF", ""
            ).strip()
            reflection_coef = (
                float(raw_reflection_coef) if raw_reflection_coef else 0.0
            )
            if not math.isfinite(reflection_coef) or reflection_coef < 0.0:
                raise ValueError("NAVRL_REFLECTION_COEF must be finite and >= 0")
            if reflection_coef > 0.0:
                reflected_batch = dict(batch_dict)
                reflected_batch["obs"] = mirror_navrl_structured_observation(
                    input_dict["obs"]
                )
                reflected_batch["prev_actions"] = mirror_navrl_actions(actions_batch)

                # The original forward already updated observation RMS once. Reuse the frozen
                # current statistics for the auxiliary reflected forward so this loss cannot
                # perturb PPO merely by double-counting normalization samples.
                rms_owner = self.model
                while hasattr(rms_owner, "_orig_mod"):
                    rms_owner = rms_owner._orig_mod
                running_rms = getattr(rms_owner, "running_mean_std", None)
                rms_was_training = (
                    bool(running_rms.training)
                    if isinstance(running_rms, torch.nn.Module)
                    else False
                )
                if isinstance(running_rms, torch.nn.Module):
                    running_rms.eval()
                try:
                    reflected_mu = self.model(reflected_batch)["mus"]
                finally:
                    if isinstance(running_rms, torch.nn.Module) and rms_was_training:
                        running_rms.train()
                symmetry_penalty = reflection_equivariance_loss(mu, reflected_mu)
                weighted_symmetry = reflection_coef * symmetry_penalty
                loss += weighted_symmetry
                self.aux_loss_dict["reflection_equivariance"] = [
                    weighted_symmetry.detach()
                ]

            # The forward/KL can be finite while a critic square, PPO ratio or auxiliary term
            # overflows. Do not send such a loss through backward/Adam; latch rejection so the
            # epoch transaction restores any earlier minibatches as well.
            if rollback_enabled and not all_finite_ppo_state(
                loss,
                a_loss,
                c_loss,
                entropy,
                b_loss,
                self.aux_loss_dict,
            ):
                zero = action_log_probs.detach().new_zeros(())
                self._aerial_epoch_kl_rejected = True
                self._aerial_epoch_kl_reject_nonfinite = True
                self._aerial_kl_skipped_minibatches = (
                    int(getattr(self, "_aerial_kl_skipped_minibatches", 0)) + 1
                )
                print(
                    "[aerial RL] PPO epoch rejection latched before backward | "
                    "non-finite PPO/critic/auxiliary loss.",
                    flush=True,
                )
                self.diagnostics.mini_batch(
                    self,
                    {
                        "values": value_preds_batch,
                        "returns": return_batch,
                        "new_neglogp": action_log_probs,
                        "old_neglogp": old_action_log_probs_batch,
                        "masks": rnn_masks,
                    },
                    curr_e_clip,
                    0,
                )
                self.train_result = (
                    zero,
                    zero,
                    entropy.detach().mean(),
                    kl_dist,
                    self.last_lr,
                    lr_mul,
                    mu.detach(),
                    sigma.detach(),
                    zero,
                )
                return

            if self.multi_gpu:
                self.optimizer.zero_grad(set_to_none=True)
            else:
                for parameter in self.model.parameters():
                    parameter.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        self.diagnostics.mini_batch(
            self,
            {
                "values": value_preds_batch,
                "returns": return_batch,
                "new_neglogp": action_log_probs,
                "old_neglogp": old_action_log_probs_batch,
                "masks": rnn_masks,
            },
            curr_e_clip,
            0,
        )
        self.train_result = (
            a_loss,
            c_loss,
            entropy,
            kl_dist,
            self.last_lr,
            lr_mul,
            mu.detach(),
            sigma.detach(),
            b_loss,
        )

    def _audit_behavior_policy(
        self,
        rollout_obs,
        rollout_actions,
        behavior_mu,
        behavior_sigma,
    ):
        """Measure final-policy KL on every rollout slice with normalization frozen."""
        module_modes = {
            name: bool(module.training) for name, module in self.model.named_modules()
        }
        max_minibatch_kl = 0.0
        max_sample_kl = 0.0
        all_finite = True
        batch_size = int(behavior_mu.shape[0])
        chunk_size = max(1, int(getattr(self, "minibatch_size", batch_size)))
        self.model.eval()
        try:
            with torch.no_grad():
                for start in range(0, batch_size, chunk_size):
                    end = min(batch_size, start + chunk_size)
                    obs = self._slice_rollout_value(rollout_obs, start, end)
                    actions = self._slice_rollout_value(rollout_actions, start, end)
                    result = self.model(
                        {
                            "is_train": True,
                            "prev_actions": actions,
                            "obs": self._preproc_obs(obs),
                        }
                    )
                    current_mu = result["mus"]
                    current_sigma = result["sigmas"]
                    per_sample = exact_normal_kl(
                        current_mu,
                        current_sigma,
                        behavior_mu[start:end],
                        behavior_sigma[start:end],
                        reduce=False,
                    )
                    required_outputs = (
                        current_mu,
                        current_sigma,
                        result.get("values"),
                        result.get("prev_neglogp"),
                        result.get("entropy"),
                    )
                    finite = all(
                        isinstance(value, torch.Tensor)
                        and bool(torch.isfinite(value).all())
                        for value in required_outputs
                    ) and bool(torch.isfinite(per_sample).all())
                    if not bool(finite):
                        all_finite = False
                        continue
                    max_minibatch_kl = max(
                        max_minibatch_kl, float(per_sample.mean().detach().cpu())
                    )
                    max_sample_kl = max(
                        max_sample_kl, float(per_sample.max().detach().cpu())
                    )
        finally:
            modules = dict(self.model.named_modules())
            for name, training in module_modes.items():
                modules[name].training = training
        return max_minibatch_kl, max_sample_kl, all_finite

    def _rollback_learning_rate(self, original_lr):
        raw_factor = os.environ.get("NAVRL_PPO_ROLLBACK_LR_FACTOR", "0.5").strip()
        raw_floor = os.environ.get("NAVRL_PPO_ROLLBACK_MIN_LR", "1e-6").strip()
        factor = float(raw_factor)
        floor = float(raw_floor)
        if not math.isfinite(factor) or not 0.0 < factor < 1.0:
            raise ValueError("NAVRL_PPO_ROLLBACK_LR_FACTOR must be finite and in (0, 1)")
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError("NAVRL_PPO_ROLLBACK_MIN_LR must be finite and > 0")
        backed_off = max(floor, float(original_lr) * factor)
        self.last_lr = backed_off
        reset_optimizer_learning_rate(self.optimizer, backed_off)
        os.environ["NAVRL_CURRENT_LEARNING_RATE"] = str(backed_off)
        return backed_off

    def train_epoch(self):
        """rl-games PPO epoch with an atomic actor-side KL transaction.

        rl-games rewrites each dataset slice's policy mean after every minibatch. The previous
        gate consequently compared against a moving reference, noticed a bad optimizer step only
        after it had happened, and never restored Adam/RMS. This copy of the upstream loop keeps
        immutable rollout parameters, audits the *post-update* policy over the full rollout, and
        commits or restores the complete actor-side epoch as one unit.
        """
        self.vec_env.set_train_info(self.frame, self)

        kl_gate = self._ppo_kl_gate()
        rollback_enabled = self._ppo_epoch_rollback_enabled(kl_gate)
        if rollback_enabled and self.multi_gpu:
            raise RuntimeError(
                "NavRL PPO epoch rollback is single-GPU only; DDP requires an all-rank "
                "reject/restore barrier. Set NAVRL_PPO_EPOCH_ROLLBACK=0 only for an "
                "intentional unsupported experiment."
            )
        if rollback_enabled and self.is_rnn:
            raise RuntimeError(
                "NavRL PPO epoch rollback does not yet support recurrent sequence-state audits."
            )

        self.set_eval()
        play_time_start = time.perf_counter()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()
        play_time_end = time.perf_counter()
        update_time_start = time.perf_counter()

        # These four values are the immutable behavior-policy audit set. PPODataset is free to
        # mutate its ordinary mu/sigma fields during optimization without changing this reference.
        rollout_obs = batch_dict["obses"]
        rollout_actions = batch_dict["actions"]
        behavior_mu = batch_dict["mus"].detach().clone()
        behavior_sigma = batch_dict["sigmas"].detach().clone()

        transaction = None
        central_transaction = None
        central_scalar_state = None
        original_lr = float(self.last_lr)
        original_entropy_coef = float(self.entropy_coef)
        if rollback_enabled:
            transaction = capture_epoch_transaction(
                self.model, self.optimizer, self.scaler
            )
            if self.has_central_value:
                central = self.central_value_net
                central_transaction = capture_epoch_transaction(
                    central.model, central.optimizer, central.scaler
                )
                central_scalar_state = (
                    float(central.lr),
                    int(central.epoch_num),
                    int(central.frame),
                    bool(central.training),
                )

        self._aerial_epoch_kl_rejected = False
        self._aerial_epoch_kl_reject_nonfinite = False
        self._aerial_epoch_max_pre_kl = 0.0
        self._aerial_epoch_rolled_back = False
        self._aerial_epoch_audit_kl = 0.0
        self._aerial_epoch_audit_sample_kl = 0.0

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []
        last_lr = self.last_lr
        lr_mul = 1.0
        central_loss = None

        try:
            # Dataset preparation updates value-normalization buffers. It is part of the PPO
            # transaction too, so snapshots above must precede it and exceptions here must restore.
            self.set_train()
            self.curr_frames = batch_dict.pop("played_frames")
            self.prepare_dataset(batch_dict)
            self.algo_observer.after_steps()
            if self.has_central_value:
                central_loss = self.train_central_value()
            stop_updates = False
            for mini_ep in range(0, self.mini_epochs_num):
                ep_kls = []
                for i in range(len(self.dataset)):
                    (
                        a_loss,
                        c_loss,
                        entropy,
                        kl,
                        last_lr,
                        lr_mul,
                        cmu,
                        csigma,
                        b_loss,
                    ) = self.train_actor_critic(self.dataset[i])
                    a_losses.append(a_loss)
                    c_losses.append(c_loss)
                    ep_kls.append(kl)
                    entropies.append(entropy)
                    if self.bounds_loss_coef is not None:
                        b_losses.append(b_loss)

                    # A pre-step breach has latched rejection of the entire epoch. Do not rebase
                    # the mutable PPO reference onto the rejected policy and do no further work.
                    if self._aerial_epoch_kl_rejected:
                        stop_updates = True
                        break

                    self.dataset.update_mu_sigma(cmu, csigma)
                    if self.schedule_type == "legacy":
                        av_kls = kl
                        if self.multi_gpu:
                            dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                            av_kls /= self.world_size
                        self.last_lr, self.entropy_coef = self.scheduler.update(
                            self.last_lr,
                            self.entropy_coef,
                            self.epoch_num,
                            0,
                            av_kls.item(),
                        )
                        self.update_lr(self.last_lr)
                        os.environ["NAVRL_CURRENT_LEARNING_RATE"] = str(self.last_lr)

                if ep_kls:
                    av_kls = torch_ext.mean_list(ep_kls)
                    if self.multi_gpu:
                        dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                        av_kls /= self.world_size
                    if self.schedule_type == "standard" and not stop_updates:
                        self.last_lr, self.entropy_coef = self.scheduler.update(
                            self.last_lr,
                            self.entropy_coef,
                            self.epoch_num,
                            0,
                            av_kls.item(),
                        )
                        self.update_lr(self.last_lr)
                        os.environ["NAVRL_CURRENT_LEARNING_RATE"] = str(self.last_lr)
                    kls.append(av_kls)
                self.diagnostics.mini_epoch(self, mini_ep)
                if self.normalize_input:
                    self.model.running_mean_std.eval()
                if stop_updates:
                    break

            if rollback_enabled:
                audit_kl, audit_sample_kl, audit_finite = self._audit_behavior_policy(
                    rollout_obs,
                    rollout_actions,
                    behavior_mu,
                    behavior_sigma,
                )
                self._aerial_epoch_audit_kl = audit_kl
                self._aerial_epoch_audit_sample_kl = audit_sample_kl
                central_commit_state = None
                if self.has_central_value:
                    central = self.central_value_net
                    central_commit_state = {
                        "model": central.model.state_dict(),
                        "optimizer": central.optimizer.state_dict(),
                        "scaler": central.scaler.state_dict(),
                        "loss": central_loss,
                        "lr": central.lr,
                    }
                commit_state_finite = all_finite_ppo_state(
                    self.model.state_dict(),
                    self.optimizer.state_dict(),
                    self.scaler.state_dict() if self.scaler is not None else None,
                    central_commit_state,
                    a_losses,
                    c_losses,
                    b_losses,
                    entropies,
                    kls,
                    self.last_lr,
                    self.entropy_coef,
                )
                reject_epoch = (
                    bool(self._aerial_epoch_kl_rejected)
                    or not audit_finite
                    or not commit_state_finite
                    or should_reject_ppo_update(audit_kl, kl_gate)
                )
                if reject_epoch:
                    restore_epoch_transaction(
                        transaction, self.model, self.optimizer, self.scaler
                    )
                    if central_transaction is not None:
                        central = self.central_value_net
                        restore_epoch_transaction(
                            central_transaction,
                            central.model,
                            central.optimizer,
                            central.scaler,
                        )
                        (
                            central.lr,
                            central.epoch_num,
                            central.frame,
                            central.training,
                        ) = central_scalar_state
                        central.update_lr(central.lr)
                    self.entropy_coef = original_entropy_coef
                    backed_off_lr = self._rollback_learning_rate(original_lr)
                    self._aerial_epoch_rolled_back = True
                    self._aerial_ppo_rollback_total = int(
                        getattr(self, "_aerial_ppo_rollback_total", 0)
                    ) + 1
                    self._aerial_ppo_rollback_streak = int(
                        getattr(self, "_aerial_ppo_rollback_streak", 0)
                    ) + 1
                    last_lr = backed_off_lr
                    reason = (
                        "nonfinite"
                        if (
                            self._aerial_epoch_kl_reject_nonfinite
                            or not audit_finite
                            or not commit_state_finite
                        )
                        else "KL"
                    )
                    print(
                        "[aerial RL] PPO EPOCH ROLLBACK | reason=%s "
                        "pre_kl_max=%.6f audit_kl_max=%.6f sample_kl_max=%.6f "
                        "gate=%.6f lr=%.3g->%.3g streak=%d"
                        % (
                            reason,
                            self._aerial_epoch_max_pre_kl,
                            audit_kl,
                            audit_sample_kl,
                            kl_gate,
                            original_lr,
                            backed_off_lr,
                            self._aerial_ppo_rollback_streak,
                        ),
                        flush=True,
                    )
                    patience = int(
                        os.environ.get("NAVRL_PPO_ROLLBACK_PATIENCE", "5")
                    )
                    if patience < 1:
                        raise ValueError(
                            "NAVRL_PPO_ROLLBACK_PATIENCE must be an integer >= 1"
                        )
                    if self._aerial_ppo_rollback_streak >= patience:
                        self._aerial_failure_reason = "ppo_rollback_livelock"
                        # play_steps() has already advanced the simulator/task clocks. The normal
                        # train() path increments frame only after train_epoch() returns, which a
                        # livelock fail-stop never does. Persist the matching global frame now,
                        # after exact model/Adam/RMS/scaler restore and LR backoff, then fail.
                        self._save_ppo_rollback_livelock_checkpoint()
                        raise FloatingPointError(
                            "[aerial RL] PPO rollback livelock: %d consecutive unsafe epochs; "
                            "last-known-good PPO actor/critic state was restored."
                            % self._aerial_ppo_rollback_streak
                        )
                else:
                    self._aerial_ppo_rollback_streak = 0
        except Exception:
            # An optimizer/runtime failure can happen after a partial actor or central-critic step.
            # Ensure the whole in-memory PPO epoch is last-known-good before train() exits.
            if transaction is not None and not self._aerial_epoch_rolled_back:
                restore_epoch_transaction(
                    transaction, self.model, self.optimizer, self.scaler
                )
                if central_transaction is not None:
                    central = self.central_value_net
                    restore_epoch_transaction(
                        central_transaction,
                        central.model,
                        central.optimizer,
                        central.scaler,
                    )
                    (
                        central.lr,
                        central.epoch_num,
                        central.frame,
                        central.training,
                    ) = central_scalar_state
                    central.update_lr(central.lr)
                self.last_lr = original_lr
                self.entropy_coef = original_entropy_coef
                os.environ["NAVRL_CURRENT_LEARNING_RATE"] = str(original_lr)
            raise

        update_time_end = time.perf_counter()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start
        return (
            batch_dict["step_time"],
            play_time,
            update_time,
            total_time,
            a_losses,
            c_losses,
            b_losses,
            entropies,
            kls,
            last_lr,
            lr_mul,
        )

    def get_action_values(self, obs):
        """Collect policy-output diagnostics before rl_games clips actions for the environment."""
        result = super().get_action_values(obs)
        if os.environ.get("NAVRL_ACTION_DIAG", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            self._accumulate_policy_action_diag(result)
        return result

    def _accumulate_policy_action_diag(self, result):
        actions = result.get("actions")
        if not isinstance(actions, torch.Tensor) or actions.ndim != 2:
            return
        with torch.no_grad():
            actions = actions.detach()
            mus = result.get("mus")
            sigmas = result.get("sigmas")
            diag = getattr(self, "_policy_action_diag", None)
            if not isinstance(diag, dict):
                diag = {
                    "n": 0,
                    "raw_oob": torch.zeros(actions.shape[1], device=actions.device),
                    "edge95": torch.zeros(actions.shape[1], device=actions.device),
                    "edge99": torch.zeros(actions.shape[1], device=actions.device),
                    "abs_sum": torch.zeros(actions.shape[1], device=actions.device),
                    "mu_abs_sum": torch.zeros(actions.shape[1], device=actions.device),
                    "signed_y_sum": torch.zeros((), device=actions.device),
                    "mu_signed_y_sum": torch.zeros((), device=actions.device),
                    "positive_y": torch.zeros((), device=actions.device),
                    "negative_y": torch.zeros((), device=actions.device),
                    "sigma_sum": torch.zeros(actions.shape[1], device=actions.device),
                    "delta_y_sum": 0.0,
                    "delta_y_n": 0,
                }
            finite = torch.isfinite(actions)
            safe = torch.where(finite, actions, torch.zeros_like(actions))
            diag["n"] += int(actions.shape[0])
            diag["raw_oob"] += ((safe.abs() > 1.0) & finite).sum(dim=0)
            diag["edge95"] += ((safe.abs() >= 0.95) & finite).sum(dim=0)
            diag["edge99"] += ((safe.abs() >= 0.99) & finite).sum(dim=0)
            diag["abs_sum"] += (safe.abs() * finite).sum(dim=0)
            diag["signed_y_sum"] += (safe[:, 1] * finite[:, 1]).sum()
            diag["positive_y"] += ((safe[:, 1] > 0.1) & finite[:, 1]).sum()
            diag["negative_y"] += ((safe[:, 1] < -0.1) & finite[:, 1]).sum()
            if isinstance(mus, torch.Tensor) and mus.shape == actions.shape:
                diag["mu_abs_sum"] += mus.detach().abs().sum(dim=0)
                diag["mu_signed_y_sum"] += mus.detach()[:, 1].sum()
            if isinstance(sigmas, torch.Tensor) and sigmas.shape == actions.shape:
                diag["sigma_sum"] += sigmas.detach().sum(dim=0)

            previous = getattr(self, "_policy_action_diag_prev", None)
            if isinstance(previous, torch.Tensor) and previous.shape == actions.shape:
                valid = finite[:, 1]
                dones = getattr(self, "dones", None)
                if isinstance(dones, torch.Tensor) and dones.shape[0] == actions.shape[0]:
                    valid &= ~dones.bool()
                if bool(valid.any()):
                    diag["delta_y_sum"] += float(
                        (safe[valid, 1] - previous[valid, 1]).abs().sum().item()
                    )
                    diag["delta_y_n"] += int(valid.sum().item())
            self._policy_action_diag_prev = safe.clone()
            self._policy_action_diag = diag

    def _consume_policy_action_diag(self):
        diag = getattr(self, "_policy_action_diag", None)
        self._policy_action_diag = None
        if not isinstance(diag, dict) or int(diag.get("n", 0)) <= 0:
            return None
        n = max(1, int(diag["n"]))
        delta_n = max(1, int(diag["delta_y_n"]))

        def values(key):
            return [float(v) / n for v in diag[key].detach().cpu().tolist()]

        return {
            "raw_oob": values("raw_oob"),
            "edge95": values("edge95"),
            "edge99": values("edge99"),
            "mean_abs": values("abs_sum"),
            "mean_mu_abs": values("mu_abs_sum"),
            "mean_sigma": values("sigma_sum"),
            "signed_y": float(diag["signed_y_sum"].detach().cpu()) / n,
            "mu_signed_y": float(diag["mu_signed_y_sum"].detach().cpu()) / n,
            "positive_y": float(diag["positive_y"].detach().cpu()) / n,
            "negative_y": float(diag["negative_y"].detach().cpu()) / n,
            "delta_y": float(diag["delta_y_sum"]) / delta_n,
            "n": n,
        }

    def restore(self, fn, set_epoch=True):
        """Restore weights/state, then apply the LR resolved by runner checkpoint provenance."""
        super().restore(fn, set_epoch=set_epoch)
        reset_actor_optimizer = os.environ.get(
            "NAVRL_RESET_ACTOR_OPTIMIZER", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        if reset_actor_optimizer:
            # A bounded-distribution branch reuses the competent representation/mean/value weights
            # but changes the actor likelihood geometry.  Old Adam moments point along the legacy
            # clipped-Gaussian objective and caused a large first update in dry-run checks.
            # Keep the separately owned asymmetric critic optimizer/state; reset only actor PPO.
            self.optimizer.state.clear()
            print(
                "[aerial RL] Actor optimizer moments reset for action-distribution branch; "
                "central critic optimizer retained.",
                flush=True,
            )
        configured_lr = float(self.config["learning_rate"])
        self.last_lr = configured_lr
        reset_optimizer_learning_rate(self.optimizer, configured_lr)
        os.environ["NAVRL_CURRENT_LEARNING_RATE"] = str(configured_lr)
        print(
            "[aerial RL] Resume optimizer LR reset to active config: "
            f"{configured_lr:.3g}.",
            flush=True,
        )

    def write_stats(
        self,
        total_time,
        epoch_num,
        step_time,
        play_time,
        update_time,
        a_losses,
        c_losses,
        entropies,
        kls,
        last_lr,
        lr_mul,
        frame,
        scaled_time,
        scaled_play_time,
        curr_frames,
    ):
        """Disable rl-games verbose TB (/step, /time, performance/*, diagnostics spam)."""
        _ = (
            total_time,
            play_time,
            update_time,
            frame,
            scaled_time,
            scaled_play_time,
        )
        self._aerial_tb_pending = {
            "epoch_num": epoch_num,
            "step_time": step_time,
            "curr_frames": curr_frames,
            "a_losses": a_losses,
            "c_losses": c_losses,
            "entropies": entropies,
            "kls": kls,
            "last_lr": last_lr,
            "lr_mul": lr_mul,
        }

    def _flush_aerial_tensorboard(
        self,
        *,
        mean_reward,
        mean_episode_length,
        intercept_succ,
        intercept_done,
        intercept_envs_hit,
        num_parallel_envs,
        mean_target_dist_m=None,
        mean_closest_target_dist_m=None,
        intercept_extra_metrics=None,
        explained_variance=None,
    ):
        pending = getattr(self, "_aerial_tb_pending", None)
        if pending is None or self.writer is None:
            return
        epoch_num = int(pending["epoch_num"])
        write_aerial_epoch_scalars(
            self.writer,
            epoch_num,
            mean_reward=mean_reward,
            mean_episode_length=mean_episode_length,
            mean_target_dist_m=mean_target_dist_m,
            a_losses=pending["a_losses"],
            c_losses=pending["c_losses"],
            entropies=pending["entropies"],
            kls=pending["kls"],
            last_lr=pending["last_lr"],
            lr_mul=pending["lr_mul"],
            step_time=pending["step_time"],
            curr_frames=pending["curr_frames"],
            intercept_succ=intercept_succ,
            intercept_done=intercept_done,
            intercept_envs_hit=intercept_envs_hit,
            num_parallel_envs=num_parallel_envs,
            mean_closest_target_dist_m=mean_closest_target_dist_m,
            explained_variance=explained_variance,
            extra_intercept_metrics=intercept_extra_metrics,
        )
        action_diag = self._consume_policy_action_diag()
        skipped_total = int(getattr(self, "_aerial_kl_skipped_minibatches", 0))
        skipped_previous = int(getattr(self, "_aerial_kl_skipped_last_epoch", 0))
        self.writer.add_scalar(
            "ppo/kl_skipped_minibatches",
            max(0, skipped_total - skipped_previous),
            epoch_num,
        )
        self._aerial_kl_skipped_last_epoch = skipped_total
        self.writer.add_scalar(
            "ppo/behavior_kl_audit_max",
            float(getattr(self, "_aerial_epoch_audit_kl", 0.0)),
            epoch_num,
        )
        self.writer.add_scalar(
            "ppo/behavior_kl_sample_max",
            float(getattr(self, "_aerial_epoch_audit_sample_kl", 0.0)),
            epoch_num,
        )
        self.writer.add_scalar(
            "ppo/epoch_rollback",
            1.0 if getattr(self, "_aerial_epoch_rolled_back", False) else 0.0,
            epoch_num,
        )
        self.writer.add_scalar(
            "ppo/epoch_rollback_total",
            int(getattr(self, "_aerial_ppo_rollback_total", 0)),
            epoch_num,
        )
        self.writer.add_scalar(
            "ppo/epoch_rollback_streak",
            int(getattr(self, "_aerial_ppo_rollback_streak", 0)),
            epoch_num,
        )
        if action_diag is not None:
            axis_names = ("x", "y", "z", "yaw")
            for axis, name in enumerate(axis_names):
                self.writer.add_scalar(
                    f"policy_action/raw_oob_{name}",
                    action_diag["raw_oob"][axis],
                    epoch_num,
                )
                self.writer.add_scalar(
                    f"policy_action/edge95_{name}",
                    action_diag["edge95"][axis],
                    epoch_num,
                )
                self.writer.add_scalar(
                    f"policy_action/edge99_{name}",
                    action_diag["edge99"][axis],
                    epoch_num,
                )
                self.writer.add_scalar(
                    f"policy_action/mean_abs_{name}",
                    action_diag["mean_abs"][axis],
                    epoch_num,
                )
                self.writer.add_scalar(
                    f"policy_action/mean_sigma_{name}",
                    action_diag["mean_sigma"][axis],
                    epoch_num,
                )
                self.writer.add_scalar(
                    f"policy_action/mean_mu_abs_{name}",
                    action_diag["mean_mu_abs"][axis],
                    epoch_num,
                )
            self.writer.add_scalar(
                "policy_action/delta_y", action_diag["delta_y"], epoch_num
            )
            for name in ("signed_y", "mu_signed_y", "positive_y", "negative_y"):
                self.writer.add_scalar(
                    f"policy_action/{name}", action_diag[name], epoch_num
                )
            if epoch_num == 1 or epoch_num % 25 == 0:
                print(
                    "[aerial RL] policy-actiondiag | mode=%s raw_oob_y=%.4f "
                    "edge95_y=%.4f edge99_y=%.4f |mu_y|=%.3f mu_y=%.3f "
                    "pos_y=%.3f neg_y=%.3f sigma_y=%.3f "
                    "delta_y=%.3f (n=%d)"
                    % (
                        os.environ.get("NAVRL_ACTION_POLICY", "legacy"),
                        action_diag["raw_oob"][1],
                        action_diag["edge95"][1],
                        action_diag["edge99"][1],
                        action_diag["mean_mu_abs"][1],
                        action_diag["mu_signed_y"],
                        action_diag["positive_y"],
                        action_diag["negative_y"],
                        action_diag["mean_sigma"][1],
                        action_diag["delta_y"],
                        action_diag["n"],
                    ),
                    flush=True,
                )
        for name, values in getattr(self, "aux_loss_dict", {}).items():
            if not isinstance(values, (list, tuple)):
                values = [values]
            finite_values = [
                float(value.detach().mean().cpu())
                for value in values
                if isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
            ]
            if finite_values:
                self.writer.add_scalar(
                    f"aux_loss/{name}",
                    sum(finite_values) / len(finite_values),
                    epoch_num,
                )
        # Skip the first epochs: an untrained policy crashes instantly and posts a large negative
        # reward that stretches the TensorBoard y-axis until the rest of the run is a flat line.
        # Same warmup the aerial/mean_reward scalar uses (NAVRL_TB_REWARD_WARMUP_EPOCHS).
        if (
            getattr(self, "last_mean_rewards", None) is not None
            and int(epoch_num) >= _TB_REWARD_WARMUP_EPOCHS
        ):
            try:
                self.writer.add_scalar("stability/best_reward", float(self.last_mean_rewards), epoch_num)
            except (TypeError, ValueError):
                pass
        try:
            self.writer.flush()
        except Exception:
            pass

    def _write_aerial_training_finished_marker(self, epoch_num):
        """Mark this run folder so train.sh skips auto-resume after a normal exit (crash has no marker)."""
        try:
            run_root = os.path.dirname(os.path.abspath(self.nn_dir))
            marker = os.path.join(run_root, _AERIAL_FINISHED_MARKER)
            with open(marker, "w", encoding="utf-8") as f:
                f.write(f"epoch={epoch_num}\n")
        except OSError:
            pass

    def train(self):
        self.init_tensors()
        save_frequency_raw = os.environ.get("NAVRL_SAVE_FREQUENCY", "").strip()
        if save_frequency_raw:
            try:
                save_frequency = int(save_frequency_raw)
            except ValueError as exc:
                raise ValueError("NAVRL_SAVE_FREQUENCY must be a positive integer") from exc
            if save_frequency <= 0:
                raise ValueError("NAVRL_SAVE_FREQUENCY must be a positive integer")
            self.save_freq = save_frequency
            print(
                "[aerial RL] checkpoint cadence | every %d epochs "
                "(NAVRL_SAVE_FREQUENCY)" % self.save_freq,
                flush=True,
            )
        self.last_mean_rewards = _read_existing_best_reward(self.nn_dir)
        if getattr(self, "global_rank", 0) == 0 and self.last_mean_rewards > -999999999:
            print(
                "[aerial RL] Best-checkpoint guard: existing best reward "
                f"{self.last_mean_rewards:.3f}; lower current policies will not overwrite "
                f"{self.config['name']}.pth.",
                flush=True,
            )
        start_time = time.perf_counter()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        early_cfg = parse_early_stop_stable_config(self.config.get("early_stop_stable"))
        collapse_cfg = parse_early_stop_collapse_config(self.config.get("early_stop_collapse"))
        density_capture_cfg = parse_density_capture_guard_config(
            self.config.get("early_stop_density_capture")
        )

        if self.multi_gpu:
            torch.cuda.set_device(self.local_rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            if self.has_central_value:
                model_params.append(self.central_value_net.state_dict())
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])
            if self.has_central_value:
                self.central_value_net.load_state_dict(model_params[1])
            print("====================broadcast done")

        exit_reason = "unknown"
        try:
            while True:
                epoch_num = self.update_epoch()
                step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = (
                    self.train_epoch()
                )
                nonfinite_path = first_nonfinite_training_value(
                    (
                        ("ppo/a_loss", a_losses),
                        ("ppo/c_loss", c_losses),
                        ("ppo/b_loss", b_losses),
                        ("ppo/entropy", entropies),
                        ("ppo/kl", kls),
                    ),
                    self.model.named_parameters(),
                )
                if nonfinite_path is not None:
                    exit_reason = "nonfinite_ppo"
                    raise FloatingPointError(
                        "[aerial RL] Non-finite PPO state detected at "
                        f"epoch {epoch_num}: {nonfinite_path}. "
                        "Optimizer output is discarded; use the last finite checkpoint."
                    )
                total_time += sum_time
                frame = self.frame // self.num_agents

                # cleaning memory to optimize space
                self.dataset.update_values_dict(None)
                should_exit = False
                pending_exit_reason = None

                if self.global_rank == 0:
                    self.diagnostics.epoch(self, current_epoch=epoch_num)
                    scaled_time = self.num_agents * sum_time
                    scaled_play_time = self.num_agents * play_time
                    curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
                    self.frame += curr_frames

                    dash_mr = dash_el = None
                    dash_dist = None
                    dash_closest_dist = None
                    intercept_extra_metrics = None
                    n_agents = _resolve_num_parallel_envs(self)
                    intercept_lines = None
                    intercept_succ = intercept_done = intercept_envs_hit = 0

                    self.write_stats(
                        total_time,
                        epoch_num,
                        step_time,
                        play_time,
                        update_time,
                        a_losses,
                        c_losses,
                        entropies,
                        kls,
                        last_lr,
                        lr_mul,
                        frame,
                        scaled_time,
                        scaled_play_time,
                        curr_frames,
                    )

                    exp_var = None
                    if getattr(self.diagnostics, "diag_dict", None):
                        exp_var = self.diagnostics.diag_dict.get("diagnostics/exp_var")

                    if self.game_rewards.current_size > 0:
                        mean_rewards = self.game_rewards.get_mean()
                        mean_lengths = self.game_lengths.get_mean()
                        self.mean_rewards = mean_rewards[0]
                        dash_mr = _scalar_mean_reward(mean_rewards[0])
                        if not is_finite_training_value(dash_mr):
                            print(
                                "[aerial RL] Early stop: NaN/Inf mean reward — "
                                f"epoch {epoch_num}. No corrupted checkpoint will be saved."
                            )
                            should_exit = True
                            pending_exit_reason = "early_stop_nan"
                        try:
                            dash_el = _scalar_mean_reward(mean_lengths[0])
                        except (TypeError, IndexError, KeyError):
                            dash_el = _scalar_mean_reward(mean_lengths)

                        if self.has_self_play_config:
                            self.self_play_manager.update(self)

                        checkpoint_name = self.config["name"] + "_ep_" + str(epoch_num) + "_rew_" + str(mean_rewards[0])

                        if not should_exit and early_cfg is not None:
                            mr0 = _scalar_mean_reward(mean_rewards[0])
                            es_state = getattr(self, "_aerial_early_stop_state", None)
                            stop_stable, es_state_next = window_band_stable_should_stop(
                                early_cfg,
                                epoch_num,
                                mr0,
                                es_state,
                            )
                            self._aerial_early_stop_state = es_state_next
                            if stop_stable:
                                tail = "_rew_" + str(mean_rewards).replace("[", "_").replace("]", "_")
                                self.save(
                                    os.path.join(
                                        self.nn_dir,
                                        "last_" + self.config["name"] + "_ep_" + str(epoch_num) + tail,
                                    )
                                )
                                print(
                                    "[aerial RL] Early stop: mean reward converged — "
                                    f"last {early_cfg['consecutive_epochs']} epochs "
                                    f"max−min ≤ {early_cfg['reward_band']} "
                                    f"(actual span {es_state_next.get('last_spread', float('nan')):.2f}, "
                                    f"epoch {epoch_num})."
                                )
                                should_exit = True
                                pending_exit_reason = "early_stop_stable"

                        if not should_exit and collapse_cfg is not None and self.game_rewards.current_size > 0:
                            mr0 = _scalar_mean_reward(mean_rewards[0])
                            cs_state = getattr(self, "_aerial_collapse_stop_state", None)
                            stop_collapse, cs_state_next = collapse_from_peak_should_stop(
                                collapse_cfg,
                                epoch_num,
                                mr0,
                                cs_state,
                            )
                            self._aerial_collapse_stop_state = cs_state_next
                            if stop_collapse:
                                if cs_state_next.get("nan_stop"):
                                    print(
                                        "[aerial RL] Early stop: NaN/Inf mean reward — "
                                        f"epoch {epoch_num}. No corrupted checkpoint will be saved."
                                    )
                                    pending_exit_reason = "early_stop_nan"
                                else:
                                    print(
                                        "[aerial RL] Early stop: reward collapse from peak — "
                                        f"peak {cs_state_next.get('collapse_peak', float('nan')):.1f} → "
                                        f"now {cs_state_next.get('collapse_mr', float('nan')):.1f} "
                                        f"for {collapse_cfg['patience_epochs']} epochs (epoch {epoch_num})."
                                    )
                                    pending_exit_reason = "early_stop_collapse"
                                    tail = "_rew_" + str(mean_rewards).replace("[", "_").replace("]", "_")
                                    self.save(
                                        os.path.join(
                                            self.nn_dir,
                                            "last_"
                                            + self.config["name"]
                                            + "_ep_"
                                            + str(epoch_num)
                                            + tail,
                                        )
                                    )
                                should_exit = True

                        if not should_exit and self.save_freq > 0:
                            if epoch_num % self.save_freq == 0:
                                self.save(os.path.join(self.nn_dir, "last_" + checkpoint_name))

                        current_mean_reward = _scalar_mean_reward(mean_rewards[0])
                        # Stashed for the navrl block further down, which needs this epoch's reward
                        # alongside the active density. It is assigned inside a conditional branch
                        # there, so read it via getattr rather than relying on it being bound.
                        self._aerial_current_mean_reward = current_mean_reward
                        if (
                            not should_exit
                            and current_mean_reward > self.last_mean_rewards
                            and epoch_num >= self.save_best_after
                        ):
                            print("saving next best rewards: ", mean_rewards)
                            self.last_mean_rewards = current_mean_reward
                            self.save(os.path.join(self.nn_dir, self.config["name"]))

                            if "score_to_win" in self.config:
                                if self.last_mean_rewards > self.config["score_to_win"]:
                                    print("Maximum reward achieved. Network won!")
                                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                                    should_exit = True
                                    pending_exit_reason = "score_to_win"

                    try:
                        (
                            intercept_lines,
                            intercept_succ,
                            intercept_done,
                            intercept_envs_hit,
                            n_agents,
                            dash_dist,
                            dash_closest_dist,
                            intercept_extra_metrics,
                        ) = (
                            consume_epoch_intercept_summary(n_agents)
                        )
                    except Exception:
                        pass

                    # NavRL navigation tasks feed their own per-epoch stats; when present they
                    # replace the intercept lines in the dashboard and add navrl/* TB scalars.
                    try:
                        from aerial_gym.task.navrl_task.train_dashboard import (
                            consume_navrl_epoch_summary,
                        )

                        nav_lines, nav_metrics, nav_done = consume_navrl_epoch_summary()
                        if nav_done > 0:
                            intercept_lines = nav_lines
                            if intercept_extra_metrics:
                                intercept_extra_metrics.update(nav_metrics)
                            else:
                                intercept_extra_metrics = nav_metrics

                            # Per-density best reward. The global stability/best_reward is a running
                            # max over the whole run, but each promotion makes the task harder and
                            # lowers the attainable reward, so after the first promotion that scalar
                            # is frozen at a number earned under an easier density and says nothing
                            # about current progress. Additive only -- self.last_mean_rewards and
                            # the best-checkpoint rule are deliberately left untouched.
                            try:
                                from aerial_gym.task.navrl_task.navrl_curriculum import (
                                    track_best_reward_by_density,
                                )

                                bd_state, bd_finished = track_best_reward_by_density(
                                    getattr(self, "_aerial_best_reward_density_state", None),
                                    nav_metrics.get("navrl/n_bars_active"),
                                    getattr(self, "_aerial_current_mean_reward", None),
                                )
                                self._aerial_best_reward_density_state = bd_state
                                if bd_state.get("best") is not None:
                                    self.writer.add_scalar(
                                        "stability/best_reward_at_density",
                                        float(bd_state["best"]),
                                        epoch_num,
                                    )
                                if bd_finished is not None:
                                    done_bars, done_best = bd_finished
                                    print(
                                        f"[aerial RL] density {done_bars} bars done: "
                                        f"best mean_reward {done_best:.3f} "
                                        f"-> now {bd_state['current_bars']} bars (epoch {epoch_num})"
                                    )
                                    # Step is the BAR COUNT, not the epoch: this series is meant to
                                    # be read as best-reward-versus-density (one point per level
                                    # the curriculum has left behind), which is the shape of the
                                    # ceiling the density schedule is trying to characterise.
                                    self.writer.add_scalar(
                                        "stability/best_reward_of_finished_density",
                                        float(done_best),
                                        int(done_bars),
                                    )
                            except Exception:
                                pass

                            if not should_exit and density_capture_cfg is not None:
                                dc_state = getattr(
                                    self, "_aerial_density_capture_stop_state", None
                                )
                                stop_density, dc_state_next = (
                                    density_capture_collapse_should_stop(
                                        density_capture_cfg,
                                        epoch_num,
                                        nav_metrics.get("navrl/captured_rate"),
                                        nav_metrics.get("navrl/n_bars_active"),
                                        dc_state,
                                    )
                                )
                                self._aerial_density_capture_stop_state = dc_state_next
                                if stop_density:
                                    print(
                                        "[aerial RL] FAIL-STOP: same-density capture collapse — "
                                        f"bars {dc_state_next.get('bars')} rolling "
                                        f"{dc_state_next.get('collapse_peak', float('nan')):.3f} → "
                                        f"{dc_state_next.get('collapse_capture', float('nan')):.3f} "
                                        f"(epoch {epoch_num}). Use the last periodic checkpoint."
                                    )
                                    should_exit = True
                                    pending_exit_reason = (
                                        "early_stop_density_capture_collapse"
                                    )
                    except Exception as dashboard_error:
                        # This block holds the density-capture-collapse FAIL-STOP. Swallowing an
                        # exception here meant a collapse could go unreported for days with no
                        # output at all, and the guard's own config validator raises on every bad
                        # value precisely so this cannot happen. Keep training alive, because a
                        # dashboard fault is not a reason to lose a run, but say so every time.
                        print(
                            "[aerial RL] WARNING: dashboard/fail-stop block raised "
                            f"{type(dashboard_error).__name__}: {dashboard_error}. The "
                            "density-capture-collapse FAIL-STOP did NOT run this epoch.",
                            flush=True,
                        )

                    self._flush_aerial_tensorboard(
                        mean_reward=dash_mr,
                        mean_episode_length=dash_el,
                        mean_target_dist_m=dash_dist,
                        mean_closest_target_dist_m=dash_closest_dist,
                        intercept_extra_metrics=intercept_extra_metrics,
                        intercept_succ=intercept_succ,
                        intercept_done=intercept_done,
                        intercept_envs_hit=intercept_envs_hit,
                        num_parallel_envs=n_agents,
                        explained_variance=exp_var,
                    )

                    print_training_dashboard(
                        self.print_stats,
                        curr_frames,
                        step_time,
                        scaled_play_time,
                        scaled_time,
                        epoch_num,
                        self.max_epochs,
                        frame,
                        self.max_frames,
                        mean_reward=dash_mr,
                        mean_episode_length=dash_el,
                        intercept_lines=intercept_lines,
                    )

                    record_agent_epoch(
                        self,
                        epoch_num=epoch_num,
                        mean_reward=dash_mr,
                        mean_episode_length=dash_el,
                        mean_target_dist_m=dash_dist,
                        mean_closest_target_dist_m=dash_closest_dist,
                        intercept_extra_metrics=intercept_extra_metrics,
                        intercept_succ=intercept_succ,
                        intercept_done=intercept_done,
                        intercept_envs_hit=intercept_envs_hit,
                        num_parallel_envs=n_agents,
                    )

                    if (
                        not should_exit
                        and epoch_num >= self.max_epochs
                        and self.max_epochs != -1
                    ):
                        if self.game_rewards.current_size == 0:
                            print("WARNING: Max epochs reached before any env terminated at least once")
                            mean_rewards = -np.inf

                        # The periodic checkpoint above has already saved this exact agent state
                        # when max_epochs is a save-frequency boundary.  Saving it again produced
                        # two semantically identical epoch-N files with different spellings
                        # (``rew_1.23`` and ``rew__1.23_``), making automatic LKG selection
                        # ambiguous.  Reuse the canonical scalar-reward name and only write when
                        # the periodic path did not already do so.
                        periodic_checkpoint_saved = (
                            self.save_freq > 0 and epoch_num % self.save_freq == 0
                        )
                        if not periodic_checkpoint_saved:
                            self.save(os.path.join(self.nn_dir, "last_" + checkpoint_name))
                        print("MAX EPOCHS NUM!")
                        should_exit = True
                        pending_exit_reason = "max_epochs"

                    if (
                        not should_exit
                        and self.frame >= self.max_frames
                        and self.max_frames != -1
                    ):
                        if self.game_rewards.current_size == 0:
                            print("WARNING: Max frames reached before any env terminated at least once")
                            mean_rewards = -np.inf

                        self.save(
                            os.path.join(
                                self.nn_dir,
                                "last_" + self.config["name"] + "_frame_" + str(self.frame)
                                + "_rew_" + str(mean_rewards).replace("[", "_").replace("]", "_"),
                            )
                        )
                        print("MAX FRAMES NUM!")
                        should_exit = True
                        pending_exit_reason = "max_frames"

                if self.multi_gpu:
                    should_exit_t = torch.tensor(should_exit, device=self.device).float()
                    dist.broadcast(should_exit_t, 0)
                    should_exit = bool(should_exit_t.item())

                if should_exit:
                    exit_reason = pending_exit_reason or "completed"
                    if (
                        exit_reason not in _FAILED_EXIT_REASONS
                        and getattr(self, "global_rank", 0) == 0
                    ):
                        self._write_aerial_training_finished_marker(epoch_num)
                    if exit_reason in _FAILED_EXIT_REASONS:
                        raise FloatingPointError(
                            f"[aerial RL] Training failed with {exit_reason} at epoch {epoch_num}."
                        )
                    break

            return self.last_mean_rewards, epoch_num
        except FloatingPointError:
            if exit_reason == "unknown":
                exit_reason = getattr(
                    self, "_aerial_failure_reason", "nonfinite_ppo"
                )
            raise
        except KeyboardInterrupt:
            exit_reason = "interrupted"
            print("\n[aerial RL] 학습 중단 (KeyboardInterrupt)", flush=True)
            raise
        finally:
            try:
                finalize_agent_run(self, exit_reason)
            except Exception as exc:
                print(f"[aerial RL] run summary skipped: {exc}", flush=True)
