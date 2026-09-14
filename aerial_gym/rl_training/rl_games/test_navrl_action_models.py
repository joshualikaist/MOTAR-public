"""CPU tests for bounded NavRL PPO action distributions."""

import math
import os
import unittest

import torch
import torch.nn as nn
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from rl_games.algos_torch import models

from navrl_action_models import (
    NavRLFixedGaussianModel,
    NavRLSquashedGaussianModel,
    NavRLTruncatedGaussianModel,
)
from ppo_update_safety import (
    lateral_batch_bias_loss,
    lateral_latent_margin_loss,
    mirror_navrl_actions,
    mirror_navrl_structured_observation,
    reflection_equivariance_loss,
    stable_ppo_actor_loss,
)


class _DummyA2CNetwork(nn.Module):
    def __init__(self, action_dim=4):
        super().__init__()
        self.mu_bias = nn.Parameter(torch.zeros(action_dim))
        # Preserve the same parameter/state_dict surface as the real custom Transformer.
        self.sigma = nn.Parameter(torch.zeros(action_dim))

    def forward(self, input_dict):
        batch = input_dict["obs"].shape[0]
        mu = self.mu_bias.expand(batch, -1)
        logstd = self.sigma.expand(batch, -1)
        value = mu[:, :1] * 0.0
        return mu, logstd, value, None

    def get_aux_loss(self):
        return None

    def is_rnn(self):
        return False

    def get_value_layer(self):
        return None

    def get_default_rnn_state(self):
        return None


def _make_network(model_cls):
    return model_cls.Network(
        _DummyA2CNetwork(),
        obs_shape=(3,),
        normalize_value=False,
        normalize_input=False,
        value_size=1,
    )


class ActionModelTests(unittest.TestCase):
    def setUp(self):
        self._old_std = os.environ.get("NAVRL_ACTION_STD")
        self._old_mu_scale = os.environ.get("NAVRL_ACTION_MU_SCALE")
        os.environ["NAVRL_ACTION_STD"] = "0.35,0.35,0.05,0.08"
        os.environ["NAVRL_ACTION_MU_SCALE"] = "1,1,1,1"

    def tearDown(self):
        if self._old_std is None:
            os.environ.pop("NAVRL_ACTION_STD", None)
        else:
            os.environ["NAVRL_ACTION_STD"] = self._old_std
        if self._old_mu_scale is None:
            os.environ.pop("NAVRL_ACTION_MU_SCALE", None)
        else:
            os.environ["NAVRL_ACTION_MU_SCALE"] = self._old_mu_scale

    def test_bounded_models_are_finite_and_strictly_bounded(self):
        # This is a Monte-Carlo tail assertion; pin the RNG so a 1-count fluctuation around the
        # 1e-3 boundary cannot make the suite intermittently fail.
        torch.manual_seed(0)
        obs = torch.zeros(100_000, 3)

        for model_cls in (NavRLSquashedGaussianModel, NavRLTruncatedGaussianModel):
            network = _make_network(model_cls)
            rollout = network({"obs": obs, "is_train": False})
            actions = rollout["actions"]
            self.assertTrue(bool(torch.isfinite(actions).all()))
            self.assertTrue(bool((actions > -1.0).all()))
            self.assertTrue(bool((actions < 1.0).all()))
            # At zero mean and lateral std=.35, neither bounded model should manufacture a
            # boundary atom. The generous limit is deterministic across Torch RNG versions.
            self.assertLess(
                float((actions[:, 1].abs() >= 0.98).float().mean()), 1.0e-3
            )

            update = network(
                {"obs": obs[:512], "is_train": True, "prev_actions": actions[:512]}
            )
            loss = update["prev_neglogp"].mean() - 1.0e-4 * update["entropy"].mean()
            loss.backward()
            self.assertTrue(bool(torch.isfinite(network.a2c_network.mu_bias.grad).all()))

    def test_fixed_gaussian_tail_matches_small_std(self):
        torch.manual_seed(7)
        network = _make_network(NavRLFixedGaussianModel)
        rollout = network({"obs": torch.zeros(300_000, 3), "is_train": False})
        lateral_oob = float((rollout["actions"][:, 1].abs() > 1.0).float().mean())
        # Theoretical two-sided tail for N(0, .35): about 0.00427.
        self.assertGreater(lateral_oob, 0.0035)
        self.assertLess(lateral_oob, 0.0051)

    def test_custom_wrappers_preserve_legacy_state_dict_keys(self):
        legacy = models.ModelA2CContinuousLogStd.Network(
            _DummyA2CNetwork(),
            obs_shape=(3,),
            normalize_value=False,
            normalize_input=False,
            value_size=1,
        )
        expected = set(legacy.state_dict())
        for model_cls in (
            NavRLFixedGaussianModel,
            NavRLSquashedGaussianModel,
            NavRLTruncatedGaussianModel,
        ):
            self.assertEqual(set(_make_network(model_cls).state_dict()), expected)

    def test_squashed_deterministic_action_is_post_tanh(self):
        network = _make_network(NavRLSquashedGaussianModel)
        with torch.no_grad():
            network.a2c_network.mu_bias.copy_(torch.tensor([0.0, 0.5, -1.0, 2.0]))
        result = network({"obs": torch.zeros(2, 3), "is_train": False})
        expected = torch.tanh(network.a2c_network.mu_bias)
        self.assertTrue(
            torch.allclose(result["deterministic_actions"][0], expected, atol=2.0e-6)
        )

    def test_squashed_logprob_matches_torch_transform(self):
        network = _make_network(NavRLSquashedGaussianModel)
        latent_mu = torch.tensor([[0.1, -0.4, 0.2, 1.0]])
        sigma = network._std_like(latent_mu)
        base = Normal(latent_mu, sigma)
        action = torch.tanh(torch.tensor([[0.2, -0.7, 0.19, 0.8]]))
        actual = network._neglogp(base, action)
        transformed = TransformedDistribution(base, [TanhTransform(cache_size=1)])
        expected = -transformed.log_prob(action).sum(dim=-1)
        self.assertTrue(torch.allclose(actual, expected, atol=2.0e-5, rtol=2.0e-5))

    def test_lateral_mu_warmstart_scaling(self):
        os.environ["NAVRL_ACTION_MU_SCALE"] = "1,0.4,1,1"
        network = _make_network(NavRLSquashedGaussianModel)
        with torch.no_grad():
            network.a2c_network.mu_bias.copy_(torch.tensor([2.0, 2.0, 2.0, 2.0]))
        result = network({"obs": torch.zeros(1, 3), "is_train": False})
        expected = torch.tanh(torch.tensor([2.0, 0.8, 2.0, 2.0]))
        self.assertTrue(
            torch.allclose(result["deterministic_actions"][0], expected, atol=2.0e-6)
        )

    def test_stable_ppo_ratio_stays_finite_for_extreme_logprob_gap(self):
        old_neglogp = torch.tensor([0.0, 0.0])
        new_neglogp = torch.tensor([-1.0e4, 1.0e4])
        advantage = torch.tensor([-1.0, 1.0])
        loss = stable_ppo_actor_loss(
            old_neglogp,
            new_neglogp,
            advantage,
            True,
            0.2,
            10.0,
        )
        self.assertTrue(bool(torch.isfinite(loss).all()))
        self.assertLessEqual(float(loss.abs().max()), math.exp(10.0))

    def test_lateral_latent_margin_is_soft_and_differentiable(self):
        mu = torch.tensor(
            [[0.0, 0.5, 0.0, 0.0], [0.0, 2.25, 0.0, 0.0]],
            requires_grad=True,
        )
        penalty = lateral_latent_margin_loss(mu, margin=1.25)
        self.assertAlmostEqual(float(penalty.detach()), 0.5, places=6)
        penalty.backward()
        self.assertEqual(float(mu.grad[0, 1]), 0.0)
        self.assertGreater(float(mu.grad[1, 1]), 0.0)

    def test_lateral_batch_bias_penalizes_direction_not_magnitude(self):
        balanced = torch.tensor(
            [[0.0, -2.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
            requires_grad=True,
        )
        one_sided = torch.tensor(
            [[0.0, 2.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
            requires_grad=True,
        )
        self.assertEqual(float(lateral_batch_bias_loss(balanced).detach()), 0.0)
        penalty = lateral_batch_bias_loss(one_sided)
        self.assertEqual(float(penalty.detach()), 4.0)
        penalty.backward()
        self.assertTrue(bool((one_sided.grad[:, 1] > 0.0).all()))

    def test_navrl_reflection_is_an_involution(self):
        hbeams = int(os.environ.get("NAVRL_LIDAR_HBEAMS", "") or 36)
        vbeams = int(os.environ.get("NAVRL_LIDAR_VBEAMS", "") or 4)
        obstacles = int(os.environ.get("NAVRL_MAX_OBSTACLES", "") or 5)
        structured_obs_dim = vbeams * hbeams + 5 * obstacles * 12 + 5 * 10 + 5 * 16
        obs = torch.randn(3, structured_obs_dim)
        actions = torch.randn(3, 4)
        self.assertTrue(
            torch.equal(
                mirror_navrl_structured_observation(
                    mirror_navrl_structured_observation(obs)
                ),
                obs,
            )
        )
        self.assertTrue(
            torch.equal(mirror_navrl_actions(mirror_navrl_actions(actions)), actions)
        )

    def test_navrl_reflection_maps_the_structured_schema(self):
        hbeams = int(os.environ.get("NAVRL_LIDAR_HBEAMS", "") or 36)
        vbeams = int(os.environ.get("NAVRL_LIDAR_VBEAMS", "") or 4)
        obstacles = int(os.environ.get("NAVRL_MAX_OBSTACLES", "") or 5)
        static_dim = vbeams * hbeams
        obstacle_size = 5 * obstacles * 12
        robot_size = 5 * 10
        structured_obs_dim = static_dim + obstacle_size + robot_size + 5 * 16
        obs = torch.zeros(1, structured_obs_dim)
        obs[:, :static_dim] = torch.arange(static_dim).reshape(1, -1)
        obstacle = obs[:, static_dim : static_dim + obstacle_size].view(
            1, 5, obstacles, 12
        )
        obstacle[..., 1], obstacle[..., 4], obstacle[..., 8] = 2.0, -3.0, 4.0
        robot_offset = static_dim + obstacle_size
        robot = obs[:, robot_offset : robot_offset + robot_size].view(1, 5, 10)
        robot[..., (1, 3, 5, 7)] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = obs[:, robot_offset + robot_size :].view(1, 5, 16)
        target[..., 1], target[..., 4], target[..., 7] = 5.0, -6.0, 7.0

        mirrored = mirror_navrl_structured_observation(obs)
        expected_index = (-torch.arange(hbeams)) % hbeams
        expected_scan = obs[:, :static_dim].view(1, vbeams, hbeams).index_select(
            2, expected_index
        )
        self.assertTrue(
            torch.equal(mirrored[:, :static_dim].view_as(expected_scan), expected_scan)
        )
        # +180 is self-reflecting; the next positive bearing maps to the final negative bearing.
        self.assertEqual(int(expected_index[0]), 0)
        self.assertEqual(int(expected_index[1]), hbeams - 1)
        mirrored_obstacle = mirrored[
            :, static_dim : static_dim + obstacle_size
        ].view(1, 5, obstacles, 12)
        self.assertTrue(bool((mirrored_obstacle[..., 1] == -2.0).all()))
        self.assertTrue(bool((mirrored_obstacle[..., 4] == 3.0).all()))
        self.assertTrue(bool((mirrored_obstacle[..., 8] == 4.0).all()))
        mirrored_robot = mirrored[
            :, robot_offset : robot_offset + robot_size
        ].view(1, 5, 10)
        self.assertTrue(
            torch.equal(
                mirrored_robot[0, 0, (1, 3, 5, 7)],
                torch.tensor([-1.0, -2.0, -3.0, -4.0]),
            )
        )
        mirrored_target = mirrored[:, robot_offset + robot_size :].view(1, 5, 16)
        self.assertTrue(bool((mirrored_target[..., 1] == -5.0).all()))
        self.assertTrue(bool((mirrored_target[..., 4] == 6.0).all()))
        self.assertTrue(bool((mirrored_target[..., 7] == 7.0).all()))

    def test_reflection_loss_accepts_balanced_policy_means(self):
        mu = torch.tensor([[0.5, 1.2, -0.3, 0.4]])
        mirrored = mirror_navrl_actions(mu)
        self.assertEqual(float(reflection_equivariance_loss(mu, mirrored)), 0.0)
        self.assertGreater(float(reflection_equivariance_loss(mu, mu)), 0.0)

    def test_truncated_pdf_is_normalized(self):
        network = _make_network(NavRLTruncatedGaussianModel)
        raw_mu = torch.tensor([[-2.0, -0.3, 0.4, 2.0]])
        location, scale, _alpha, _beta, _cdf_low, normalizer = (
            network._distribution_parameters(raw_mu)
        )
        grid = torch.linspace(-1.0 + 1.0e-6, 1.0 - 1.0e-6, 20_001)
        action = grid[:, None].expand(-1, 4)
        log_pdf = Normal(location, scale).log_prob(action) - torch.log(normalizer)
        integral = torch.trapz(torch.exp(log_pdf), grid, dim=0)
        self.assertTrue(torch.allclose(integral, torch.ones_like(integral), atol=2.0e-3))


if __name__ == "__main__":
    unittest.main()
