"""NavRL continuous-action models with an explicit, checkpoint-compatible noise contract.

The legacy policy samples an unbounded Gaussian and lets the task hard-clamp the sample.  Once the
lateral standard deviation grows, a large fraction of distinct policy samples therefore become the
same +/-1 command.  These models keep the Transformer and checkpoint parameter names unchanged but
replace that ambiguous action-distribution layer:

* ``NavRLSquashedGaussianModel`` applies an invertible tanh transform and the corresponding
  change-of-variables correction to PPO log probabilities.
* ``NavRLTruncatedGaussianModel`` samples directly from ``[-1, 1]`` and applies the scale
  adjustment proposed for truncated-Gaussian PPO (AAAI 2025).
* ``NavRLFixedGaussianModel`` is the minimal-change control: it retains the legacy Gaussian +
  environment clamp, but uses the same small, fixed per-axis standard deviations.

The fixed noise vector is intentional for this A/B.  The checkpoint's lateral and unused-z raw
log-std parameters are already just above the old clamp ceiling, where ``torch.clamp`` has zero
gradient.  Ignoring (rather than re-clamping) that dead parameter makes the intervention explicit,
keeps old optimizer parameter groups loadable, and prevents the entropy bonus from inflating it
again.
"""

import math
import os
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from rl_games.algos_torch import models


_DEFAULT_ACTION_STD = (0.35, 0.35, 0.05, 0.08)
_DEFAULT_MU_SCALE = (1.0, 1.0, 1.0, 1.0)
_ACTION_EPS = 1.0e-6


def parse_action_std(raw=None) -> Tuple[float, ...]:
    """Parse ``NAVRL_ACTION_STD`` as one scalar or a comma-separated positive vector."""
    text = str(raw if raw is not None else os.environ.get("NAVRL_ACTION_STD", "")).strip()
    if not text:
        values = _DEFAULT_ACTION_STD
    else:
        try:
            values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
        except ValueError as exc:
            raise ValueError(
                "NAVRL_ACTION_STD must be a positive scalar or comma-separated vector, got %r"
                % text
            ) from exc
        if not values:
            raise ValueError("NAVRL_ACTION_STD cannot be empty")
    for value in values:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("NAVRL_ACTION_STD values must be finite and > 0, got %r" % (values,))
    return tuple(values)


def format_action_std(values: Iterable[float]) -> str:
    return ",".join("%.6g" % float(value) for value in values)


class _FixedStdMixin:
    def _init_action_std(self):
        self._action_std_values = parse_action_std()
        raw_scale = os.environ.get("NAVRL_ACTION_MU_SCALE", "").strip()
        if raw_scale:
            try:
                values = tuple(
                    float(item.strip()) for item in raw_scale.split(",") if item.strip()
                )
            except ValueError as exc:
                raise ValueError(
                    "NAVRL_ACTION_MU_SCALE must be a positive scalar or vector, got %r"
                    % raw_scale
                ) from exc
            if not values or any(
                not math.isfinite(value) or value <= 0.0 for value in values
            ):
                raise ValueError(
                    "NAVRL_ACTION_MU_SCALE values must be finite and > 0, got %r"
                    % (values,)
                )
            self._action_mu_scale_values = values
        else:
            self._action_mu_scale_values = _DEFAULT_MU_SCALE

    def _std_like(self, reference):
        values = self._action_std_values
        if len(values) == 1:
            values = values * int(reference.shape[-1])
        if len(values) != int(reference.shape[-1]):
            raise ValueError(
                "NAVRL_ACTION_STD has %d values but the policy has %d actions"
                % (len(values), int(reference.shape[-1]))
            )
        return reference.new_tensor(values).expand_as(reference)

    def _scaled_mu(self, raw_mu):
        values = self._action_mu_scale_values
        if len(values) == 1:
            values = values * int(raw_mu.shape[-1])
        if len(values) != int(raw_mu.shape[-1]):
            raise ValueError(
                "NAVRL_ACTION_MU_SCALE has %d values but the policy has %d actions"
                % (len(values), int(raw_mu.shape[-1]))
            )
        return raw_mu * raw_mu.new_tensor(values)

    @property
    def action_std_values(self):
        return self._action_std_values


class NavRLFixedGaussianModel(models.ModelA2CContinuousLogStd):
    """Legacy unbounded Gaussian with a fixed, per-axis standard deviation."""

    class Network(_FixedStdMixin, models.ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            self._init_action_std()

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions")
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            raw_mu, _unused_logstd, value, states = self.a2c_network(input_dict)
            mu = self._scaled_mu(raw_mu)
            sigma = self._std_like(mu)
            distr = Normal(mu, sigma, validate_args=False)

            if is_train:
                if prev_actions is None:
                    raise ValueError("prev_actions is required for a PPO training update")
                result = {
                    "prev_neglogp": -distr.log_prob(prev_actions).sum(dim=-1),
                    "values": value,
                    "entropy": distr.entropy().sum(dim=-1),
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }
            else:
                selected_action = distr.sample()
                result = {
                    "neglogpacs": -distr.log_prob(selected_action).sum(dim=-1),
                    "values": self.denorm_value(value),
                    "actions": selected_action,
                    "deterministic_actions": mu,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }
            return result


class NavRLSquashedGaussianModel(models.ModelA2CContinuousLogStd):
    """Gaussian in latent space followed by tanh, with PPO likelihood correction."""

    class Network(_FixedStdMixin, models.ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            self._init_action_std()

        @staticmethod
        def _log_abs_det_tanh(latent):
            # Stable log(1 - tanh(latent)^2), as used by squashed-Gaussian policies.
            return 2.0 * (math.log(2.0) - latent - F.softplus(-2.0 * latent))

        @staticmethod
        def _atanh(action):
            bounded = action.clamp(-1.0 + _ACTION_EPS, 1.0 - _ACTION_EPS)
            return 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))

        def _neglogp(self, distr, bounded_action):
            latent = self._atanh(bounded_action)
            log_prob = distr.log_prob(latent) - self._log_abs_det_tanh(latent)
            return -log_prob.sum(dim=-1)

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions")
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            raw_mu, _unused_logstd, value, states = self.a2c_network(input_dict)
            latent_mu = self._scaled_mu(raw_mu)
            sigma = self._std_like(latent_mu)
            distr = Normal(latent_mu, sigma, validate_args=False)
            deterministic_action = torch.tanh(latent_mu).clamp(
                -1.0 + _ACTION_EPS, 1.0 - _ACTION_EPS
            )

            if is_train:
                if prev_actions is None:
                    raise ValueError("prev_actions is required for a PPO training update")
                entropy_sample = distr.rsample()
                # H[tanh(Z)] = H[Z] + E[log |d tanh(Z) / dZ|].
                entropy = (
                    distr.entropy() + self._log_abs_det_tanh(entropy_sample)
                ).sum(dim=-1)
                result = {
                    "prev_neglogp": self._neglogp(distr, prev_actions),
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    # PPO's analytic Gaussian KL is exact in this latent space because tanh is the
                    # same bijection for both policies.
                    "mus": latent_mu,
                    "sigmas": sigma,
                }
            else:
                latent_action = distr.sample()
                selected_action = torch.tanh(latent_action).clamp(
                    -1.0 + _ACTION_EPS, 1.0 - _ACTION_EPS
                )
                result = {
                    "neglogpacs": self._neglogp(distr, selected_action),
                    "values": self.denorm_value(value),
                    "actions": selected_action,
                    # rl_games' stock continuous player returns raw `mus` in deterministic mode.
                    # The local player consumes this explicit post-tanh value instead.
                    "deterministic_actions": deterministic_action,
                    "rnn_states": states,
                    "mus": latent_mu,
                    "sigmas": sigma,
                }
            return result


class NavRLTruncatedGaussianModel(models.ModelA2CContinuousLogStd):
    """Scale-adjusted truncated Gaussian with direct support on ``[-1, 1]``.

    The network output is mapped to a valid location with tanh.  The base standard deviation is
    discounted near an action boundary using the k=2 semi-ellipse from Kim et al. (AAAI 2025):

        adjusted_std = base_std * (sqrt(1 - location**2) * (1 - d_min) + d_min)

    This avoids both clipped-Gaussian boundary pile-up and the plain truncated distribution's
    tendency to under-use a genuinely necessary boundary action.
    """

    class Network(_FixedStdMixin, models.ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            self._init_action_std()
            raw_dmin = os.environ.get("NAVRL_TRUNCATED_DMIN", "0.01").strip()
            try:
                self._d_min = float(raw_dmin)
            except ValueError as exc:
                raise ValueError("NAVRL_TRUNCATED_DMIN must be a float, got %r" % raw_dmin) from exc
            if not math.isfinite(self._d_min) or not 0.0 < self._d_min <= 1.0:
                raise ValueError("NAVRL_TRUNCATED_DMIN must be in (0, 1], got %r" % self._d_min)

        @staticmethod
        def _standard_normal_pdf(value):
            return torch.exp(-0.5 * value.square()) / math.sqrt(2.0 * math.pi)

        def _distribution_parameters(self, raw_mu):
            location = torch.tanh(self._scaled_mu(raw_mu)).clamp(
                -1.0 + _ACTION_EPS, 1.0 - _ACTION_EPS
            )
            base_std = self._std_like(raw_mu)
            discount = (
                torch.sqrt(torch.clamp(1.0 - location.square(), min=0.0))
                * (1.0 - self._d_min)
                + self._d_min
            )
            scale = (base_std * discount).clamp_min(1.0e-5)
            alpha = (-1.0 - location) / scale
            beta = (1.0 - location) / scale
            cdf_low = torch.special.ndtr(alpha)
            cdf_high = torch.special.ndtr(beta)
            normalizer = (cdf_high - cdf_low).clamp_min(1.0e-7)
            return location, scale, alpha, beta, cdf_low, normalizer

        @staticmethod
        def _sample(location, scale, cdf_low, normalizer):
            uniform = torch.rand_like(location)
            probability = (cdf_low + uniform * normalizer).clamp(
                _ACTION_EPS, 1.0 - _ACTION_EPS
            )
            sample = location + scale * torch.special.ndtri(probability)
            return sample.clamp(-1.0 + _ACTION_EPS, 1.0 - _ACTION_EPS)

        @staticmethod
        def _neglogp(action, location, scale, normalizer):
            bounded = action.clamp(-1.0 + _ACTION_EPS, 1.0 - _ACTION_EPS)
            base = Normal(location, scale, validate_args=False)
            return -(base.log_prob(bounded) - torch.log(normalizer)).sum(dim=-1)

        def _entropy(self, scale, alpha, beta, normalizer):
            # Exact differential entropy of an independently truncated Normal.
            correction = (
                alpha * self._standard_normal_pdf(alpha)
                - beta * self._standard_normal_pdf(beta)
            ) / (2.0 * normalizer)
            per_axis = (
                torch.log(scale * normalizer * math.sqrt(2.0 * math.pi * math.e))
                + correction
            )
            return per_axis.sum(dim=-1)

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions")
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            raw_mu, _unused_logstd, value, states = self.a2c_network(input_dict)
            location, scale, alpha, beta, cdf_low, normalizer = (
                self._distribution_parameters(raw_mu)
            )

            if is_train:
                if prev_actions is None:
                    raise ValueError("prev_actions is required for a PPO training update")
                result = {
                    "prev_neglogp": self._neglogp(
                        prev_actions, location, scale, normalizer
                    ),
                    "values": value,
                    "entropy": self._entropy(scale, alpha, beta, normalizer),
                    "rnn_states": states,
                    # rl_games uses these only for its KL diagnostic/scheduler.  The experiment
                    # keeps lr_schedule=None; PPO ratios themselves use the exact truncated PDF.
                    "mus": location,
                    "sigmas": scale,
                }
            else:
                selected_action = self._sample(location, scale, cdf_low, normalizer)
                result = {
                    "neglogpacs": self._neglogp(
                        selected_action, location, scale, normalizer
                    ),
                    "values": self.denorm_value(value),
                    "actions": selected_action,
                    "deterministic_actions": location,
                    "rnn_states": states,
                    "mus": location,
                    "sigmas": scale,
                }
            return result
