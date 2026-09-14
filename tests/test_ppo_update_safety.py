import copy
import importlib.util
from pathlib import Path
import unittest

import torch


_ROOT = Path(__file__).parents[1]
_MODULE_PATH = _ROOT / "aerial_gym/rl_training/rl_games/ppo_update_safety.py"
_SPEC = importlib.util.spec_from_file_location("ppo_update_safety_standalone", _MODULE_PATH)
_SAFETY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SAFETY)


def _assert_nested_equal(test_case, actual, expected):
    test_case.assertEqual(type(actual), type(expected))
    if isinstance(actual, torch.Tensor):
        test_case.assertTrue(torch.equal(actual, expected))
    elif isinstance(actual, dict):
        test_case.assertEqual(set(actual), set(expected))
        for key in actual:
            _assert_nested_equal(test_case, actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        test_case.assertEqual(len(actual), len(expected))
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(test_case, actual_item, expected_item)
    else:
        test_case.assertEqual(actual, expected)


class _BufferedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.normalizer = torch.nn.BatchNorm1d(2)
        self.frozen_child = torch.nn.Dropout()
        self.register_buffer("rollout_count", torch.tensor(0, dtype=torch.long))

    def forward(self, inputs):
        self.rollout_count.add_(inputs.shape[0])
        return self.normalizer(self.linear(inputs))


class PPOUpdateSafetyTest(unittest.TestCase):
    def _scaled_step(self, model, optimizer, scaler, inputs, target):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs) - target).square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    def test_epoch_transaction_restores_model_buffers_optimizer_and_scaler(self):
        torch.manual_seed(7)
        model = _BufferedPolicy()
        optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-3)
        scaler = torch.amp.GradScaler(
            "cpu", init_scale=8.0, growth_factor=2.0, growth_interval=1
        )
        inputs = torch.randn(8, 3)
        target = torch.randn(8, 2)

        # Populate Adam moments and scaler growth state before the last-known-good snapshot.
        self._scaled_step(model, optimizer, scaler, inputs, target)
        model.train()
        model.frozen_child.eval()
        snapshot = _SAFETY.capture_epoch_transaction(model, optimizer, scaler)
        expected_model = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_scaler = copy.deepcopy(scaler.state_dict())
        expected_modes = {
            name: module.training for name, module in model.named_modules()
        }

        self._scaled_step(model, optimizer, scaler, inputs * 4.0, target * -3.0)
        model.rollout_count.add_(1000)
        model.eval()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.param_groups[0]["lr"] = 0.5

        _SAFETY.restore_epoch_transaction(snapshot, model, optimizer, scaler)

        _assert_nested_equal(self, model.state_dict(), expected_model)
        _assert_nested_equal(self, optimizer.state_dict(), expected_optimizer)
        _assert_nested_equal(self, scaler.state_dict(), expected_scaler)
        self.assertEqual(
            {name: module.training for name, module in model.named_modules()},
            expected_modes,
        )
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

        # A restored optimizer must not alias and mutate the reusable snapshot.
        self._scaled_step(model, optimizer, scaler, inputs * 2.0, target)
        _SAFETY.restore_epoch_transaction(snapshot, model, optimizer, scaler)
        _assert_nested_equal(self, model.state_dict(), expected_model)
        _assert_nested_equal(self, optimizer.state_dict(), expected_optimizer)
        _assert_nested_equal(self, scaler.state_dict(), expected_scaler)

    def test_resume_learning_rate_preserves_backoff_unless_explicitly_overridden(self):
        resolve = _SAFETY.resolve_action_learning_rate
        self.assertEqual(
            resolve(3e-5, saved_current=2.5e-6, resume_training=True),
            2.5e-6,
        )
        self.assertEqual(
            resolve(
                3e-5,
                explicit_override="5e-6",
                saved_current=2.5e-6,
                resume_training=True,
            ),
            5e-6,
        )
        self.assertEqual(
            resolve(3e-5, saved_current=2.5e-6, resume_training=False),
            3e-5,
        )
        for invalid in (0, -1, float("nan"), "bad"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    resolve(3e-5, saved_current=invalid, resume_training=True)

    def test_rollback_counters_round_trip_and_legacy_defaults(self):
        state = {"epoch": 12, "frame": 4096}
        returned = _SAFETY.add_ppo_rollback_checkpoint_state(
            state, total=7, streak=3
        )
        self.assertIs(returned, state)
        self.assertEqual(
            _SAFETY.read_ppo_rollback_checkpoint_state(state),
            (7, 3),
        )
        self.assertEqual(
            _SAFETY.read_ppo_rollback_checkpoint_state(
                {"epoch": 11, "frame": 2048}
            ),
            (0, 0),
        )

    def test_rollback_counter_checkpoint_rejects_damaged_values(self):
        invalid_values = (-1, 1.5, True, "2", None)
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _SAFETY.add_ppo_rollback_checkpoint_state(
                        {}, total=value, streak=0
                    )
                with self.assertRaises(ValueError):
                    _SAFETY.read_ppo_rollback_checkpoint_state(
                        {_SAFETY.PPO_ROLLBACK_STREAK_KEY: value}
                    )
        with self.assertRaises(ValueError):
            _SAFETY.add_ppo_rollback_checkpoint_state({}, total=2, streak=3)
        with self.assertRaises(ValueError):
            _SAFETY.read_ppo_rollback_checkpoint_state(
                {
                    _SAFETY.PPO_ROLLBACK_TOTAL_KEY: 2,
                    _SAFETY.PPO_ROLLBACK_STREAK_KEY: 3,
                }
            )

    def test_completed_rollout_frame_accounting(self):
        self.assertEqual(_SAFETY.completed_rollout_frames(4096), 4096)
        self.assertEqual(
            _SAFETY.completed_rollout_frames(4096, 3, multi_gpu=True),
            12288,
        )
        # A single-GPU agent never scales by a stale world_size value.
        self.assertEqual(
            _SAFETY.completed_rollout_frames(4096, 3, multi_gpu=False),
            4096,
        )
        for frames, ranks in ((0, 1), (-1, 1), (1, 0), (1, 1.5)):
            with self.subTest(frames=frames, ranks=ranks):
                with self.assertRaises(ValueError):
                    _SAFETY.completed_rollout_frames(
                        frames, ranks, multi_gpu=True
                    )

    def test_exact_normal_kl_identical_distribution_is_exactly_zero(self):
        mu = torch.tensor([[0.5, -1.25], [3.0, 0.0]], dtype=torch.float64)
        sigma = torch.tensor([[0.35, 0.05]], dtype=torch.float64)
        kl = _SAFETY.exact_normal_kl(mu, sigma, mu.clone(), sigma.clone(), reduce=False)
        self.assertTrue(torch.equal(kl, torch.zeros_like(kl)))
        self.assertEqual(float(_SAFETY.exact_normal_kl(mu, sigma, mu, sigma)), 0.0)

    def test_exact_normal_kl_matches_known_mean_shift(self):
        current_mu = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
        reference_mu = torch.zeros_like(current_mu)
        sigma = torch.tensor([[2.0, 0.5]], dtype=torch.float64)
        # 1^2 / (2 * 2^2) + (-2)^2 / (2 * 0.5^2) = 0.125 + 8.
        kl = _SAFETY.exact_normal_kl(
            current_mu,
            sigma,
            reference_mu,
            sigma,
        )
        self.assertAlmostEqual(float(kl), 8.125, places=12)

    def test_exact_normal_kl_marks_nonpositive_scale_nonfinite(self):
        mu = torch.zeros(1, 2)
        invalid_sigma = torch.tensor([[-1.0, 0.0]])
        kl = _SAFETY.exact_normal_kl(
            mu,
            invalid_sigma,
            mu,
            invalid_sigma,
            reduce=False,
        )
        self.assertFalse(bool(torch.isfinite(kl).all()))
        self.assertTrue(_SAFETY.should_reject_ppo_update(kl, 0.04))

    def test_rejects_threshold_breach_and_nonfinite_values(self):
        self.assertFalse(
            _SAFETY.should_reject_ppo_update(
                torch.tensor(0.02),
                0.04,
                torch.tensor([1.0, -3.0]),
            )
        )
        self.assertTrue(
            _SAFETY.should_reject_ppo_update(torch.tensor(0.041), 0.04)
        )
        self.assertTrue(
            _SAFETY.should_reject_ppo_update(
                torch.tensor([0.01, 0.05]),
                0.04,
            )
        )
        self.assertTrue(
            _SAFETY.should_reject_ppo_update(torch.tensor(float("nan")), 0.04)
        )
        self.assertTrue(
            _SAFETY.should_reject_ppo_update(
                torch.tensor(0.01),
                0.04,
                {"loss": torch.tensor(float("inf"))},
            )
        )

    def test_recursive_commit_state_guard_rejects_nonfinite_optimizer_state(self):
        finite = {
            "model": {"weight": torch.ones(2)},
            "optimizer": {"state": {0: {"exp_avg": torch.zeros(2)}}},
            "scaler": {"scale": 8.0},
        }
        self.assertTrue(_SAFETY.all_finite_ppo_state(finite))
        finite["optimizer"]["state"][0]["exp_avg"][1] = float("nan")
        self.assertFalse(_SAFETY.all_finite_ppo_state(finite))

    def test_boolean_status_is_not_a_finiteness_sentinel(self):
        # Python bool is a Real; callers must audit the underlying state, not pass a status flag.
        self.assertTrue(_SAFETY.all_finite_ppo_state(False))

    def test_zero_limit_disables_distance_but_not_finite_guard(self):
        self.assertFalse(
            _SAFETY.should_reject_ppo_update(torch.tensor(1000.0), 0.0)
        )
        self.assertTrue(
            _SAFETY.should_reject_ppo_update(
                torch.tensor(0.0),
                0.0,
                torch.tensor(float("nan")),
            )
        )

    def test_per_axis_latent_margin_penalizes_every_saturated_axis(self):
        mu = torch.tensor([[2.5, -1.5, 0.5, -3.0]])
        loss = _SAFETY.latent_margin_loss(mu, [2.0, 1.25, 1.0, 2.0])
        expected = (0.5**2 + 0.25**2 + 0.0 + 1.0**2) / 4.0
        self.assertAlmostEqual(float(loss), expected, places=6)
        with self.assertRaises(ValueError):
            _SAFETY.latent_margin_loss(mu, [1.0, 1.0])

    def test_rejects_invalid_kl_limit(self):
        for limit in (-1.0, float("nan"), float("inf")):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    _SAFETY.should_reject_ppo_update(torch.tensor(0.0), limit)


if __name__ == "__main__":
    unittest.main()
