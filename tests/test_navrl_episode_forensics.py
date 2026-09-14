"""TD-T1/TD-T2 instrumentation: the state machine, the taxonomy, and the records they produce.

The state machine and the classifier are pure functions, so they are tested directly rather than
through a simulator. The accumulator is driven with scripted tensors whose answers are worked out by
hand, and the classifier is checked in both directions: every class must be reachable, and the
preregistered priority order must actually decide when two rules both match.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    """Load by file path: importing the package would pull in Isaac Gym, which refuses to be
    imported after torch. Neither module imports aerial_gym, so a direct load is the whole file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_navrl_" + name, ROOT / ("aerial_gym/task/navrl_task/%s.py" % name))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_forensics = _load("navrl_episode_forensics")
_digest = _load("navrl_trajectory_digest")
EPISODE_LABEL_NEVER_SEEN = _forensics.EPISODE_LABEL_NEVER_SEEN
FAILURE_CLASSES = _forensics.FAILURE_CLASSES
LABELS = _forensics.LABELS
LOSS_RUN_SHORT_STEPS = _forensics.LOSS_RUN_SHORT_STEPS
TERMINAL_WINDOW_STEPS = _forensics.TERMINAL_WINDOW_STEPS
EpisodeForensics = _forensics.EpisodeForensics
classify_terminal_failure = _forensics.classify_terminal_failure
step_label = _forensics.step_label
yaw_from_xyzw = _forensics.yaw_from_xyzw
TrajectoryDigest = _digest.TrajectoryDigest


class StateMachineTest(unittest.TestCase):
    def test_every_label_is_reachable_by_its_own_rule(self):
        self.assertEqual(step_label(False, False, False, 0, False), "SEARCH")
        self.assertEqual(step_label(True, False, False, 0, False), "ACQUIRED")
        self.assertEqual(step_label(True, True, True, 0, False), "TRACKING")
        self.assertEqual(step_label(False, True, True, 1, False), "LOST_SHORT")
        self.assertEqual(step_label(False, True, False, LOSS_RUN_SHORT_STEPS, False), "LOST_SHORT")
        self.assertEqual(step_label(False, True, False, LOSS_RUN_SHORT_STEPS + 1, False), "LOST_LONG")
        self.assertEqual(step_label(True, True, False, 3, False), "REACQUIRED")
        self.assertEqual(step_label(True, True, True, 0, True), "TERMINATED")
        self.assertEqual(set(LABELS), {"SEARCH", "ACQUIRED", "TRACKING", "LOST_SHORT", "LOST_LONG",
                                       "REACQUIRED", "TERMINATED"})

    def test_termination_wins_over_every_other_label(self):
        for visible in (True, False):
            for acquired in (True, False):
                self.assertEqual(step_label(visible, acquired, False, 99, True), "TERMINATED")

    def test_the_short_long_boundary_is_the_preregistered_one(self):
        self.assertEqual(LOSS_RUN_SHORT_STEPS, 10)


class TaxonomyTest(unittest.TestCase):
    def base(self, **overrides):
        record = {"legacy_capture_0p5m": False, "ever_acquired": True, "outcome": "timeout",
                  "crash_cause": None, "visible_at_end": True, "final_loss_run_steps": 0,
                  "minimum_center_distance_m": 5.0, "final_range_m": 5.0,
                  "radial_speed_at_closest_approach_mps": 0.1,
                  "tangential_speed_at_closest_approach_mps": 0.0,
                  "governor_intervened_share_final_window": 0.0,
                  "mean_abs_command_error_final_window_mps": 0.0,
                  "mean_closing_speed_final_window_mps": 1.0, "success_radius_m": 0.5}
        record.update(overrides)
        return record

    def test_capture_is_never_given_a_failure_class(self):
        self.assertEqual(classify_terminal_failure(self.base(legacy_capture_0p5m=True)), "CAPTURED")

    def test_every_failure_class_is_reachable(self):
        cases = {
            "NEVER_ACQUIRED": self.base(ever_acquired=False),
            "OBSTACLE_CONTACT": self.base(outcome="crash", crash_cause="contact"),
            "LOST_TRACK": self.base(visible_at_end=False, final_loss_run_steps=15),
            "TERMINAL_OVERSHOOT": self.base(minimum_center_distance_m=0.6, final_range_m=2.0,
                                            radial_speed_at_closest_approach_mps=-2.0),
            "FLY_BY": self.base(minimum_center_distance_m=0.6, final_range_m=2.0,
                                radial_speed_at_closest_approach_mps=0.2,
                                tangential_speed_at_closest_approach_mps=1.2),
            "FILTER_LIMITED": self.base(governor_intervened_share_final_window=0.7),
            "CONTROL_TRACKING_ERROR": self.base(mean_abs_command_error_final_window_mps=0.9),
            "SLOW_APPROACH": self.base(mean_closing_speed_final_window_mps=0.05, final_range_m=3.0),
            "TIMEOUT_WITH_TRACK": self.base(),
            "UNKNOWN": self.base(outcome="crash", crash_cause="below", visible_at_end=True),
        }
        for expected, record in cases.items():
            self.assertEqual(classify_terminal_failure(record), expected, expected)
        self.assertEqual(set(cases), set(FAILURE_CLASSES))

    def test_priority_order_decides_when_two_rules_match(self):
        # Never acquired beats a crash cause that would otherwise be obstacle contact.
        self.assertEqual(classify_terminal_failure(
            self.base(ever_acquired=False, outcome="crash", crash_cause="contact")),
            "NEVER_ACQUIRED")
        # Obstacle contact beats a lost track.
        self.assertEqual(classify_terminal_failure(
            self.base(outcome="crash", crash_cause="contact", visible_at_end=False,
                      final_loss_run_steps=20)), "OBSTACLE_CONTACT")
        # A lost track beats the terminal geometry classes.
        self.assertEqual(classify_terminal_failure(
            self.base(visible_at_end=False, final_loss_run_steps=20,
                      minimum_center_distance_m=0.5, final_range_m=3.0,
                      radial_speed_at_closest_approach_mps=-2.0)), "LOST_TRACK")
        # Overshoot beats fly-by when both geometric conditions hold.
        self.assertEqual(classify_terminal_failure(
            self.base(minimum_center_distance_m=0.5, final_range_m=3.0,
                      radial_speed_at_closest_approach_mps=-2.0,
                      tangential_speed_at_closest_approach_mps=5.0)), "TERMINAL_OVERSHOOT")
        # A governed final window beats a command-tracking error.
        self.assertEqual(classify_terminal_failure(
            self.base(governor_intervened_share_final_window=0.6,
                      mean_abs_command_error_final_window_mps=2.0)), "FILTER_LIMITED")

    def test_an_ambiguous_episode_is_allowed_to_stay_unknown(self):
        self.assertEqual(classify_terminal_failure(
            self.base(outcome="crash", crash_cause=None)), "UNKNOWN")

    def test_overshoot_requires_the_range_to_increase_afterwards(self):
        close_and_fast = self.base(minimum_center_distance_m=0.5, final_range_m=0.5,
                                   radial_speed_at_closest_approach_mps=-3.0,
                                   tangential_speed_at_closest_approach_mps=0.0)
        self.assertNotEqual(classify_terminal_failure(close_and_fast), "TERMINAL_OVERSHOOT")


class AccumulatorTest(unittest.TestCase):
    """One scripted episode whose every recorded number can be worked out by hand."""

    VISIBILITY = (False, True, True, False, False, False, True, True)

    def forensics(self, envs=1):
        return EpisodeForensics(envs, torch.device("cpu"), step_dt=0.1, success_radius_m=0.5,
                                target_radius_m=0.15, checkpoint_sha256="test")

    def drive(self, recorder, visibility, ranges=None, envs=1):
        ranges = ranges or [5.0 - 0.5 * index for index in range(len(visibility))]
        for step, visible in enumerate(visibility):
            recorder.observe_perception({
                "visible": torch.tensor([visible] * envs),
                "camera_visible": torch.tensor([visible] * envs),
                "track_age": torch.full((envs,), 0.1 * step),
                "track_covariance": torch.eye(6).unsqueeze(0).repeat(envs, 1, 1) * 0.04,
            })
            distance = ranges[step]
            recorder.observe_capture(torch.full((envs,), distance),
                                     torch.zeros(envs, dtype=torch.bool))
            recorder.record_step(
                valid=torch.ones(envs, dtype=torch.bool),
                robot_position=torch.zeros(envs, 3),
                robot_velocity=torch.tensor([[1.0, 0.0, 0.0]] * envs),
                robot_orientation_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * envs),
                target_position=torch.tensor([[distance, 0.0, 0.0]] * envs),
                target_velocity=torch.zeros(envs, 3),
                requested_speed_mps=torch.full((envs,), 1.0),
                executed_speed_mps=torch.full((envs,), 1.0),
                governor_scale=torch.ones(envs),
                action_xy=torch.zeros(envs, 2),
                clearance_m=torch.full((envs,), 2.0),
            )

    def finish_one(self, recorder, outcome="timeout", envs=1):
        successes = torch.tensor([outcome == "capture"] * envs)
        crashes = torch.tensor([outcome == "crash"] * envs)
        timeouts = torch.tensor([outcome == "timeout"] * envs)
        recorder.finish(torch.ones(envs, dtype=torch.bool), successes, crashes, timeouts,
                        torch.full((envs,), 0 if outcome == "crash" else -1))
        return recorder.records[-1]

    def test_a_scripted_episode_records_the_hand_computed_values(self):
        recorder = self.forensics()
        self.drive(recorder, self.VISIBILITY)
        row = self.finish_one(recorder)
        self.assertEqual(row["observation_steps"], 8)
        self.assertEqual(row["visible_steps"], 4)
        self.assertAlmostEqual(row["visibility_fraction"], 0.5)
        self.assertTrue(row["ever_acquired"])
        self.assertEqual(row["first_acquisition_step"], 2)
        self.assertEqual(row["loss_count"], 1)
        self.assertEqual(row["total_lost_steps"], 3)
        self.assertEqual(row["longest_lost_steps"], 3)
        self.assertEqual(row["longest_invisible_steps"], 3)
        self.assertTrue(row["reacquired"])
        self.assertEqual(row["reacquisition_count"], 1)
        self.assertEqual(row["first_reacquisition_latency_steps"], 3)
        self.assertEqual(row["label_step_counts"],
                         {"SEARCH": 1, "ACQUIRED": 1, "TRACKING": 2, "LOST_SHORT": 3,
                          "LOST_LONG": 0, "REACQUIRED": 1, "TERMINATED": 0})
        self.assertEqual(sum(row["label_step_counts"].values()), row["observation_steps"])
        self.assertEqual(row["episode_label"], "TERMINATED")

    def test_a_never_acquired_episode_contributes_to_nothing_else(self):
        recorder = self.forensics()
        self.drive(recorder, (False,) * 6)
        row = self.finish_one(recorder)
        self.assertFalse(row["ever_acquired"])
        self.assertEqual(row["episode_label"], EPISODE_LABEL_NEVER_SEEN)
        for field in ("first_acquisition_step", "distance_at_acquisition_m",
                      "track_age_at_first_acquisition_s", "first_reacquisition_latency_steps",
                      "mean_reacquisition_latency_steps"):
            self.assertIsNone(row[field], field)
        self.assertEqual(row["failure_class"], "NEVER_ACQUIRED")

    def test_a_long_loss_is_labelled_long_and_survives_to_the_record(self):
        recorder = self.forensics()
        self.drive(recorder, (True,) + (False,) * (LOSS_RUN_SHORT_STEPS + 3))
        row = self.finish_one(recorder)
        self.assertEqual(row["label_step_counts"]["LOST_SHORT"], LOSS_RUN_SHORT_STEPS)
        self.assertEqual(row["label_step_counts"]["LOST_LONG"], 3)
        self.assertEqual(row["final_loss_run_steps"], LOSS_RUN_SHORT_STEPS + 3)
        self.assertFalse(row["visible_at_end"])
        self.assertEqual(row["failure_class"], "LOST_TRACK")

    def test_distance_metrics_use_the_swept_capture_distance(self):
        recorder = self.forensics()
        ranges = [4.0, 3.0, 0.8, 2.0, 3.0]
        self.drive(recorder, (True,) * 5, ranges=ranges)
        row = self.finish_one(recorder)
        self.assertAlmostEqual(row["minimum_center_distance_m"], 0.8, places=6)
        self.assertAlmostEqual(row["minimum_surface_distance_m"], 0.8 - 0.15, places=6)
        self.assertAlmostEqual(row["final_range_m"], 3.0, places=6)
        self.assertEqual(row["time_inside_1m_steps"], 1)
        self.assertEqual(row["time_inside_0p5m_steps"], 0)
        self.assertEqual(row["success_radius_m"], 0.5)
        self.assertFalse(row["legacy_capture_0p5m"])

    def test_invalid_steps_are_not_counted_as_observations(self):
        recorder = self.forensics()
        recorder.observe_perception({"visible": torch.tensor([True]),
                                     "camera_visible": torch.tensor([True]),
                                     "track_age": torch.zeros(1),
                                     "track_covariance": torch.eye(6).unsqueeze(0)})
        recorder.record_step(
            valid=torch.zeros(1, dtype=torch.bool), robot_position=torch.zeros(1, 3),
            robot_velocity=torch.zeros(1, 3),
            robot_orientation_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            target_position=torch.tensor([[2.0, 0.0, 0.0]]), target_velocity=torch.zeros(1, 3),
            requested_speed_mps=torch.ones(1), executed_speed_mps=torch.ones(1),
            governor_scale=torch.ones(1), action_xy=torch.zeros(1, 2),
            clearance_m=torch.ones(1))
        row = self.finish_one(recorder)
        self.assertEqual(row["observation_steps"], 0)
        self.assertFalse(row["ever_acquired"])
        self.assertIsNone(row["visibility_fraction"])

    def test_reset_clears_every_per_episode_field(self):
        recorder = self.forensics()
        self.drive(recorder, self.VISIBILITY)
        self.finish_one(recorder)
        recorder.reset_idx(torch.tensor([0]))
        self.drive(recorder, (False, False))
        row = self.finish_one(recorder)
        self.assertEqual(row["observation_steps"], 2)
        self.assertEqual(row["visible_steps"], 0)
        self.assertFalse(row["ever_acquired"])
        self.assertEqual(row["loss_count"], 0)
        self.assertEqual(row["episode_index"], 1)

    def test_the_final_window_is_bounded_and_ordered(self):
        recorder = self.forensics()
        self.drive(recorder, (True,) * (TERMINAL_WINDOW_STEPS + 7))
        row = self.finish_one(recorder)
        self.assertEqual(row["final_window_steps"], TERMINAL_WINDOW_STEPS)
        self.assertIsNotNone(row["mean_closing_speed_final_window_mps"])

    def test_export_writes_json_without_nan_and_refuses_to_overwrite(self):
        recorder = self.forensics()
        self.drive(recorder, self.VISIBILITY)
        self.finish_one(recorder)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forensics.json"
            recorder.export(path)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["schema"], "navrl_episode_forensics_v1")
            self.assertEqual(payload["summary"]["episodes"], 1)
            self.assertEqual(payload["summary"]["causality_vs_d8b"], "NOT_TESTED")
            self.assertNotIn("NaN", path.read_text())
            with self.assertRaises(FileExistsError):
                recorder.export(path)

    def test_several_environments_stay_independent(self):
        recorder = self.forensics(envs=3)
        for step, visible in enumerate((False, True, True)):
            recorder.observe_perception({
                "visible": torch.tensor([visible, False, True]),
                "camera_visible": torch.tensor([visible, False, True]),
                "track_age": torch.zeros(3), "track_covariance": torch.eye(6).unsqueeze(0).repeat(3, 1, 1)})
            recorder.observe_capture(torch.full((3,), 3.0 - step), torch.zeros(3, dtype=torch.bool))
            recorder.record_step(
                valid=torch.ones(3, dtype=torch.bool), robot_position=torch.zeros(3, 3),
                robot_velocity=torch.zeros(3, 3),
                robot_orientation_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3),
                target_position=torch.tensor([[3.0 - step, 0.0, 0.0]] * 3),
                target_velocity=torch.zeros(3, 3), requested_speed_mps=torch.ones(3),
                executed_speed_mps=torch.ones(3), governor_scale=torch.ones(3),
                action_xy=torch.zeros(3, 2), clearance_m=torch.ones(3))
        recorder.finish(torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool),
                        torch.zeros(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool),
                        torch.full((3,), -1))
        rows = {row["env_index"]: row for row in recorder.records}
        self.assertEqual(rows[0]["visible_steps"], 2)
        self.assertEqual(rows[1]["visible_steps"], 0)
        self.assertEqual(rows[2]["visible_steps"], 3)
        self.assertFalse(rows[1]["ever_acquired"])
        self.assertEqual(rows[2]["first_acquisition_step"], 1)


class YawTest(unittest.TestCase):
    def test_identity_and_quarter_turn(self):
        identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        self.assertAlmostEqual(float(yaw_from_xyzw(identity)[0]), 0.0, places=6)
        quarter = torch.tensor([[0.0, 0.0, 0.7071067811865476, 0.7071067811865476]])
        self.assertAlmostEqual(float(yaw_from_xyzw(quarter)[0]), 3.141592653589793 / 2, places=6)


class TrajectoryDigestTest(unittest.TestCase):
    def sample(self, shift=0.0):
        return {"position": torch.tensor([[1.0 + shift, 2.0, 3.0]]),
                "orientation": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                "command": torch.tensor([[0.1, 0.2, 0.3, 0.4]])}

    def test_identical_inputs_give_an_identical_digest(self):
        first, second = TrajectoryDigest(1), TrajectoryDigest(1)
        for _ in range(4):
            first.record(**self.sample())
            second.record(**self.sample())
        self.assertEqual(first.hexdigest(), second.hexdigest())
        self.assertEqual(first.steps, 4)

    def test_one_changed_value_changes_the_digest(self):
        first, second = TrajectoryDigest(1), TrajectoryDigest(1)
        first.record(**self.sample())
        second.record(**self.sample(shift=1e-6))
        self.assertNotEqual(first.hexdigest(), second.hexdigest())

    def test_step_order_matters(self):
        first, second = TrajectoryDigest(1), TrajectoryDigest(1)
        first.record(**self.sample(0.0)); first.record(**self.sample(1.0))
        second.record(**self.sample(1.0)); second.record(**self.sample(0.0))
        self.assertNotEqual(first.hexdigest(), second.hexdigest())

    def test_batch_size_and_missing_fields_are_refused(self):
        digest = TrajectoryDigest(2)
        with self.assertRaises(ValueError):
            digest.record(**self.sample())
        with self.assertRaises(ValueError):
            TrajectoryDigest(1).record(position=None, orientation=torch.zeros(1, 4),
                                       command=torch.zeros(1, 4))
        with self.assertRaises(ValueError):
            TrajectoryDigest(0)

    def test_label_separates_two_otherwise_identical_runs(self):
        first, second = TrajectoryDigest(1, label="off"), TrajectoryDigest(1, label="on")
        first.record(**self.sample()); second.record(**self.sample())
        self.assertNotEqual(first.hexdigest(), second.hexdigest())
        self.assertEqual(first.as_dict()["schema"], "navrl_trajectory_digest_v1")


if __name__ == "__main__":
    unittest.main()
