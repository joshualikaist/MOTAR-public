import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch


_MODULE_PATH = (
    Path(__file__).parents[1]
    / "aerial_gym"
    / "task"
    / "navrl_task"
    / "target_motion.py"
)
_SPEC = importlib.util.spec_from_file_location("navrl_target_motion_standalone", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
steer_target_step = _MODULE.steer_target_step
initial_cv_velocity = _MODULE.initial_cv_velocity
limit_planar_velocity = _MODULE.limit_planar_velocity
bounded_drone_target_step = _MODULE.bounded_drone_target_step
support_aware_bounds = _MODULE.support_aware_bounds


def _bounds(n):
    lo = torch.full((n, 2), -10.0)
    hi = torch.full((n, 2), 10.0)
    return lo, hi


class SupportAwareBoundsTest(unittest.TestCase):
    def test_inflate_center_reserve(self):
        lo, hi = support_aware_bounds(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[40.0, 40.0]]),
            1.0,
            torch.tensor([[0.16, 0.14]]),
        )
        self.assertTrue(torch.allclose(lo, torch.tensor([[1.16, 1.14]])))
        self.assertTrue(torch.allclose(hi, torch.tensor([[38.84, 38.86]])))

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            support_aware_bounds(
                torch.zeros(1, 2), torch.zeros(2, 2), 1.0, torch.zeros(1, 2)
            )


def test_controlled_initial_cv_headings_follow_radial_contract():
    target = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    pursuer = torch.zeros(2, 2)
    speed = torch.tensor([1.0, 2.0])
    random_angle = torch.tensor([0.4, -0.7])
    expected = {
        "away": torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        "toward": torch.tensor([[-1.0, 0.0], [0.0, -2.0]]),
        "tangent_left": torch.tensor([[0.0, 1.0], [-2.0, 0.0]]),
        "tangent_right": torch.tensor([[0.0, -1.0], [2.0, 0.0]]),
    }
    for mode, want in expected.items():
        got = initial_cv_velocity(mode, speed, target, pursuer, random_angle)
        assert torch.allclose(got, want, atol=1e-6)


def test_random_initial_cv_heading_and_degenerate_radial_are_finite():
    speed = torch.tensor([1.5])
    xy = torch.zeros(1, 2)
    angle = torch.tensor([torch.pi / 2])
    random_velocity = initial_cv_velocity("random", speed, xy, xy, angle)
    away_velocity = initial_cv_velocity("away", speed, xy, xy, angle)
    assert torch.allclose(random_velocity, torch.tensor([[0.0, 1.5]]), atol=1e-6)
    assert torch.allclose(away_velocity, torch.tensor([[1.5, 0.0]]), atol=1e-6)
    assert torch.isfinite(away_velocity).all()


def test_unobstructed_step_preserves_heading_and_speed():
    old = torch.tensor([[0.0, 0.0]])
    desired = torch.tensor([[1.2, 1.6]])
    speed = torch.tensor([2.0])
    lo, hi = _bounds(1)
    new, velocity, steered, clear = steer_target_step(
        old, desired, speed, 0.1, torch.empty(1, 0, 2), lo, hi, 1.0, torch.ones(1)
    )
    assert not bool(steered.item())
    assert bool(clear.item())
    assert torch.allclose(velocity, desired)
    assert torch.allclose((new - old).norm(dim=1), torch.tensor([0.2]))


def test_bar_ahead_selects_a_full_speed_side_step():
    old = torch.tensor([[0.0, 0.0]])
    desired = torch.tensor([[1.0, 0.0]])
    speed = torch.tensor([1.0])
    # Direct 0.1 m motion violates the 1 m disc; +/-90 candidates are clear.
    bars = torch.tensor([[[1.05, 0.0]]])
    lo, hi = _bounds(1)
    new, velocity, steered, clear = steer_target_step(
        old, desired, speed, 0.1, bars, lo, hi, 1.0, torch.ones(1)
    )
    assert bool(steered.item())
    assert bool(clear.item())
    assert torch.allclose((new - old).norm(dim=1), torch.tensor([0.1]), atol=1e-6)
    assert torch.allclose(velocity.norm(dim=1), speed)
    assert torch.cdist(new.unsqueeze(1), bars).item() >= 1.0


def test_tie_break_is_mirror_symmetric():
    old = torch.zeros(2, 2)
    desired = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    speed = torch.ones(2)
    bars = torch.tensor([[[1.05, 0.0]], [[1.05, 0.0]]])
    lo, hi = _bounds(2)
    new, velocity, steered, clear = steer_target_step(
        old,
        desired,
        speed,
        0.1,
        bars,
        lo,
        hi,
        1.0,
        torch.tensor([1.0, -1.0]),
    )
    assert bool(steered.all())
    assert bool(clear.all())
    assert torch.allclose(new[0, 0], new[1, 0])
    assert torch.allclose(new[0, 1], -new[1, 1])
    assert torch.allclose(velocity[0, 1], -velocity[1, 1])


def test_heading_continuity_turns_toward_previous_flight_direction():
    old = torch.tensor([[0.0, 0.0]])
    desired = torch.tensor([[1.0, 0.0]])
    speed = torch.tensor([1.0])
    lo, hi = _bounds(1)
    new, velocity, steered, clear = steer_target_step(
        old,
        desired,
        speed,
        0.1,
        torch.empty(1, 0, 2),
        lo,
        hi,
        1.0,
        torch.ones(1),
        torch.tensor([torch.pi]),
    )
    heading = torch.atan2(velocity[:, 1], velocity[:, 0]).abs()
    assert bool(steered.item())
    assert bool(clear.item())
    assert torch.allclose(heading, torch.tensor([torch.pi / 2]), atol=1e-6)
    assert torch.allclose((new - old).norm(dim=1), torch.tensor([0.1]), atol=1e-6)


def test_heading_window_is_preference_not_escape_veto():
    old = torch.tensor([[0.0, 0.0]])
    desired = torch.tensor([[-1.0, 0.0]])
    speed = torch.tensor([1.0])
    lo, hi = _bounds(1)
    angles = torch.tensor(
        [torch.deg2rad(torch.tensor(value)) for value in _MODULE.TURN_ANGLES_DEG]
    )
    base = desired[0]
    directions = torch.stack(
        (
            base[0] * torch.cos(angles) - base[1] * torch.sin(angles),
            base[0] * torch.sin(angles) + base[1] * torch.cos(angles),
        ),
        dim=1,
    )
    # Block every candidate except the direct west escape. Its heading differs from the previous
    # eastward flight by 180 degrees, so selecting it proves that the 90-degree window is no veto.
    bars = (directions[1:] * 0.1).unsqueeze(0)
    new, velocity, steered, clear = steer_target_step(
        old,
        desired,
        speed,
        0.1,
        bars,
        lo,
        hi,
        0.02,
        torch.ones(1),
        torch.tensor([0.0]),
    )
    assert not bool(steered.item())
    assert bool(clear.item())
    assert torch.allclose(velocity, desired)
    assert torch.allclose(new, torch.tensor([[-0.1, 0.0]]), atol=1e-6)


def test_bounded_velocity_ramps_from_rest_without_exceeding_acceleration():
    current = torch.zeros(1, 2)
    desired = torch.tensor([[1.5, 0.0]])
    velocity = limit_planar_velocity(
        current,
        desired,
        torch.tensor([1.5]),
        0.1,
        torch.tensor([3.0]),
        torch.tensor([torch.deg2rad(torch.tensor(120.0))]),
    )
    assert torch.allclose(velocity, torch.tensor([[0.3, 0.0]]), atol=1e-6)
    assert float(((velocity - current) / 0.1).norm()) <= 3.0 + 1e-6


def test_bounded_velocity_limits_heading_slew_and_vector_acceleration():
    current = torch.tensor([[1.5, 0.0]])
    desired = torch.tensor([[0.0, 1.5]])
    max_turn = torch.deg2rad(torch.tensor([120.0]))
    velocity = limit_planar_velocity(
        current, desired, torch.tensor([1.5]), 0.1, torch.tensor([3.0]), max_turn
    )
    heading = torch.atan2(velocity[:, 1], velocity[:, 0])
    assert float(heading.abs()) <= float(max_turn[0] * 0.1) + 1e-6
    assert float(((velocity - current) / 0.1).norm()) <= 3.0 + 1e-6
    assert float(velocity.norm()) <= 1.5 + 1e-6


def test_bounded_rollout_avoids_bar_without_teleport_or_instant_turn():
    old = torch.tensor([[0.0, 0.0]])
    current = torch.tensor([[1.0, 0.0]])
    desired = torch.tensor([[1.5, 0.0]])
    speed = torch.tensor([1.5])
    bars = torch.tensor([[[1.7, 0.0]]])
    lo, hi = _bounds(1)
    max_accel = torch.tensor([3.0])
    max_turn = torch.deg2rad(torch.tensor([120.0]))
    new, velocity, steered, feasible = bounded_drone_target_step(
        old,
        current,
        desired,
        speed,
        0.1,
        bars,
        lo,
        hi,
        0.5,
        torch.ones(1),
        max_accel,
        max_turn,
        1.5,
    )
    assert bool(steered.item())
    assert bool(feasible.item())
    assert float(((velocity - current) / 0.1).norm()) <= 3.0 + 1e-6
    heading = torch.atan2(velocity[:, 1], velocity[:, 0])
    assert float(heading.abs()) <= float(max_turn[0] * 0.1) + 1e-6
    assert torch.allclose(new, old + velocity * 0.1, atol=1e-6)


def test_bounded_rollout_uses_bar_surface_not_center_when_half_extents_are_given():
    old = torch.tensor([[0.0, 0.0]])
    current = torch.zeros(1, 2)
    desired = torch.tensor([[1.0, 0.0]])
    bars = torch.tensor([[[1.5, 0.4]]])
    # Huge X half-extent brings the rectangle surface near the direct path even though its center
    # is sqrt(2) metres away. A center-distance planner would incorrectly choose straight ahead.
    half = torch.tensor([[[1.30, 0.20]]])
    lo, hi = _bounds(1)
    new, velocity, steered, feasible = bounded_drone_target_step(
        old, current, desired, torch.ones(1), 0.1, bars, lo, hi, 0.25,
        torch.ones(1), torch.tensor([4.0]), torch.tensor([3.0]), 1.0, half,
    )
    assert bool(steered.item())
    assert bool(feasible.item())
    assert torch.isfinite(new).all() and torch.isfinite(velocity).all()


def test_heading_valid_speed_matches_braking_stop_threshold():
    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import probe_navrl_physical_target_braking as probe

    assert _MODULE.HEADING_VALID_SPEED_MPS == probe.STOP_THRESHOLD_MPS
    assert _MODULE.HEADING_VALID_SPEED_MPS == 0.10


def test_physx_residual_speed_is_not_a_heading_of_travel():
    # VOID.3656376 env 3, CONNECT interval 1: 3 mm/s residual after a rest start.  The old
    # 1e-5 m/s epsilon treated that noise heading as a real course and slewed 15 deg/step
    # away from the anchor.  Below the braking stop threshold the command must start toward
    # the requested bearing; acceleration still ramps from the residual velocity.
    residual_heading = math.radians(-143.6)
    desired_heading = math.radians(-176.1)
    current = torch.tensor([[
        0.003 * math.cos(residual_heading),
        0.003 * math.sin(residual_heading),
    ]])
    desired = torch.tensor([[
        0.6 * math.cos(desired_heading),
        0.6 * math.sin(desired_heading),
    ]])
    max_turn = torch.deg2rad(torch.tensor([150.0]))
    velocity = limit_planar_velocity(
        current, desired, torch.tensor([0.6]), 0.1, torch.tensor([4.0]), max_turn
    )
    heading = float(torch.atan2(velocity[0, 1], velocity[0, 0]))
    slewed = residual_heading - math.radians(15.0)
    assert abs(heading - desired_heading) <= math.radians(1.0)
    assert abs(heading - slewed) > math.radians(10.0)
    assert float(((velocity - current) / 0.1).norm()) <= 4.0 + 1e-6


def test_cruise_speed_heading_remains_slew_limited():
    current = torch.tensor([[0.6, 0.0]])
    desired = torch.tensor([[0.0, 0.6]])
    max_turn = torch.deg2rad(torch.tensor([150.0]))
    velocity = limit_planar_velocity(
        current, desired, torch.tensor([0.6]), 0.1, torch.tensor([4.0]), max_turn
    )
    heading = float(torch.atan2(velocity[0, 1], velocity[0, 0]))
    assert heading <= float(max_turn[0] * 0.1) + 1e-6
    assert heading >= 0.0
    assert float(velocity.norm()) <= 0.6 + 1e-6


class HeadingValidSpeedContractTestCase(unittest.TestCase):
    """The heading-validity threshold is a checkpoint contract term, not a free knob.

    ``HEADING_VALID_SPEED_MPS`` gates *all* bounded/physical target motion, so a future run that
    edits it would silently change what every moving-target metric means.  These cases pin the
    provenance behaviour: a checkpoint either attests the value or is refused, and a value that
    was merely assumed can never be mistaken for one that was measured.
    """

    def test_absent_key_is_assumed_at_the_historical_value_not_refused(self):
        # Every checkpoint predating the key lands here (verified: frozen ref5in D1 ep1900 has no
        # heading key at all).  Refusing would void all of them, so this must pass -- but it must
        # come back labelled as an assumption.
        value, provenance, message = _MODULE.resolve_heading_valid_speed_contract({})
        self.assertEqual(value, 0.10)
        self.assertEqual(provenance, _MODULE.HEADING_VALID_SPEED_ASSUMED)
        self.assertIn("ASSUMED", message)
        # The attestation token must NOT appear: an assumption may not print like a measurement.
        self.assertNotIn(_MODULE.HEADING_VALID_SPEED_ATTESTED, message)

    def test_absent_key_records_the_pre_key_epsilon_caveat(self):
        # Before the constant existed the code used an inline 1e-5 epsilon, so an older lineage
        # may not in fact have trained at 0.10.  The assumed path has to say so rather than
        # asserting 0.10 as this checkpoint's history.
        _, _, message = _MODULE.resolve_heading_valid_speed_contract({})
        self.assertIn("1e-05", message)

    def test_missing_state_entirely_is_also_assumed(self):
        for state in (None, "not-a-dict", 17):
            value, provenance, _ = _MODULE.resolve_heading_valid_speed_contract(state)
            self.assertEqual(value, 0.10)
            self.assertEqual(provenance, _MODULE.HEADING_VALID_SPEED_ASSUMED)

    def test_present_and_equal_key_is_attested(self):
        state = {_MODULE.HEADING_VALID_SPEED_KEY: 0.10}
        value, provenance, message = _MODULE.resolve_heading_valid_speed_contract(state)
        self.assertEqual(value, 0.10)
        self.assertEqual(provenance, _MODULE.HEADING_VALID_SPEED_ATTESTED)
        # Symmetrically: a measurement may not print like an assumption.
        self.assertNotIn(_MODULE.HEADING_VALID_SPEED_ASSUMED, message)

    def test_attested_and_assumed_provenance_tokens_are_distinguishable(self):
        # The whole point of the field: a reader can tell the two paths apart.
        tokens = {
            _MODULE.HEADING_VALID_SPEED_SOURCE,
            _MODULE.HEADING_VALID_SPEED_ATTESTED,
            _MODULE.HEADING_VALID_SPEED_ASSUMED,
        }
        self.assertEqual(len(tokens), 3)
        _, absent_provenance, _ = _MODULE.resolve_heading_valid_speed_contract({})
        _, present_provenance, _ = _MODULE.resolve_heading_valid_speed_contract(
            {_MODULE.HEADING_VALID_SPEED_KEY: 0.10}
        )
        self.assertNotEqual(absent_provenance, present_provenance)

    def test_present_and_different_key_is_refused(self):
        for other in (0.05, 0.11, 1e-5, 0.0, 1.0):
            state = {_MODULE.HEADING_VALID_SPEED_KEY: other}
            with self.assertRaises(RuntimeError) as caught:
                _MODULE.resolve_heading_valid_speed_contract(state)
            self.assertIn("HEADING", str(caught.exception))

    def test_unreadable_key_is_refused_rather_than_assumed(self):
        # Fail closed: unreadable provenance is not the same as absent provenance.
        for junk in ("fast", [0.10], {}):
            with self.assertRaises(RuntimeError):
                _MODULE.resolve_heading_valid_speed_contract(
                    {_MODULE.HEADING_VALID_SPEED_KEY: junk}
                )

    def test_recorded_value_is_the_threshold_actually_in_force(self):
        # Bind the recorded number to real behaviour: it is not enough that the resolver returns
        # 0.10, it has to be the same 0.10 that limit_planar_velocity actually gates on.
        recorded, _, _ = _MODULE.resolve_heading_valid_speed_contract({})
        self.assertEqual(recorded, _MODULE.HEADING_VALID_SPEED_MPS)

        desired_heading = math.radians(180.0)
        max_turn = torch.deg2rad(torch.tensor([1.0]))  # 0.1 deg per 0.1 s step: nearly frozen

        def _heading_after(speed):
            current = torch.tensor([[speed, 0.0]])  # travelling along +X
            desired = torch.tensor([[
                0.6 * math.cos(desired_heading),
                0.6 * math.sin(desired_heading),
            ]])
            out = limit_planar_velocity(
                current, desired, torch.tensor([0.6]), 0.1, torch.tensor([50.0]), max_turn
            )
            return math.degrees(math.atan2(float(out[0, 1]), float(out[0, 0])))

        # Just below the recorded threshold there is no heading of travel: the command starts in
        # the requested direction (180 deg) despite the near-zero slew limit.
        below = _heading_after(recorded * 0.99)
        self.assertAlmostEqual(abs(below), 180.0, delta=1.0)

        # Just above it the slew limit binds and the heading stays essentially along +X.
        above = _heading_after(recorded * 1.01)
        self.assertLess(abs(above), 1.0)

    def test_navrl_task_writes_and_compares_the_contract_key(self):
        # navrl_task imports isaacgym, so the wiring is checked statically rather than by import.
        task_src = (
            Path(__file__).resolve().parents[1]
            / "aerial_gym" / "task" / "navrl_task" / "navrl_task.py"
        ).read_text(encoding="utf-8")
        self.assertIn("HEADING_VALID_SPEED_KEY: float(self._heading_valid_speed_mps)", task_src)
        self.assertIn(
            "HEADING_VALID_SPEED_PROVENANCE_KEY: self._heading_valid_speed_provenance", task_src
        )
        self.assertIn("resolve_heading_valid_speed_contract(state)", task_src)


class TargetMotionFunctionTestCase(unittest.TestCase):
    """Adapter so ``unittest discover`` collects the pytest-style functions above.

    Each ``test_*`` function defined at module level is pytest-collectible but invisible
    to ``unittest discover`` (this repo's only available runner -- pytest is not installed).
    Without this class, `python -m unittest discover` silently reports "Ran 0 tests / OK"
    for this file even if every assertion above is broken. Each wrapper method below
    delegates, unmodified, to the corresponding module-level function so unittest reports
    a separate named pass/fail per check -- the assertions themselves are never touched here.
    """

    def test_controlled_initial_cv_headings_follow_radial_contract(self):
        test_controlled_initial_cv_headings_follow_radial_contract()

    def test_random_initial_cv_heading_and_degenerate_radial_are_finite(self):
        test_random_initial_cv_heading_and_degenerate_radial_are_finite()

    def test_unobstructed_step_preserves_heading_and_speed(self):
        test_unobstructed_step_preserves_heading_and_speed()

    def test_bar_ahead_selects_a_full_speed_side_step(self):
        test_bar_ahead_selects_a_full_speed_side_step()

    def test_tie_break_is_mirror_symmetric(self):
        test_tie_break_is_mirror_symmetric()

    def test_heading_continuity_turns_toward_previous_flight_direction(self):
        test_heading_continuity_turns_toward_previous_flight_direction()

    def test_heading_window_is_preference_not_escape_veto(self):
        test_heading_window_is_preference_not_escape_veto()

    def test_bounded_velocity_ramps_from_rest_without_exceeding_acceleration(self):
        test_bounded_velocity_ramps_from_rest_without_exceeding_acceleration()

    def test_bounded_velocity_limits_heading_slew_and_vector_acceleration(self):
        test_bounded_velocity_limits_heading_slew_and_vector_acceleration()

    def test_bounded_rollout_avoids_bar_without_teleport_or_instant_turn(self):
        test_bounded_rollout_avoids_bar_without_teleport_or_instant_turn()

    def test_bounded_rollout_uses_bar_surface_not_center_when_half_extents_are_given(self):
        test_bounded_rollout_uses_bar_surface_not_center_when_half_extents_are_given()

    def test_heading_valid_speed_matches_braking_stop_threshold(self):
        test_heading_valid_speed_matches_braking_stop_threshold()

    def test_physx_residual_speed_is_not_a_heading_of_travel(self):
        test_physx_residual_speed_is_not_a_heading_of_travel()

    def test_cruise_speed_heading_remains_slew_limited(self):
        test_cruise_speed_heading_remains_slew_limited()


if __name__ == "__main__":
    unittest.main()
