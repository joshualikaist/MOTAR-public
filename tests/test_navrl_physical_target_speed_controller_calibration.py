"""CPU-only contract tests for the speed/controller calibration."""

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "navrl_speed_controller_calibration",
    TOOLS / "run_navrl_physical_target_speed_controller_calibration.py",
)
CAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAL)


def fixture(kp, speed, *, passed=True, stop_distance=0.6, lateral=0.02):
    trace = [[speed for _ in range(CAL.ENVS)] for _ in range(CAL.WARMUP_STEPS * 10)]
    horizon = {}
    for seconds in CAL.HORIZONS_S:
        horizon[str(seconds)] = {
            "speed_min_mps": speed,
            "speed_mean_mps": speed,
            "speed_max_mps": speed,
            "error_max_mps": 0.0,
            "all_within_gate": True,
        }
    return {
        "schema": CAL.SCHEMA + "_cell",
        "condition": {"velocity_kp": kp, "requested_speed_mps": speed},
        "horizon": horizon,
        "sustained_4_to_5s": passed,
        "overshoot_max_mps": 0.0,
        "warmup": {"saturation_max": 0.0, "tilt_max_deg": 1.0},
        "braking": {
            "all_stopped": True,
            "stop_time_s": [0.5] * CAL.ENVS,
            "stop_distance_m": [stop_distance] * CAL.ENVS,
            "lateral_deviation_m": [lateral] * CAL.ENVS,
            "saturation_max": 0.0,
            "tilt_max_deg": 1.0,
        },
        "cell_pass": passed,
        "warmup_speed_trace_mps": trace,
    }


class CalibrationContractTest(unittest.TestCase):
    def test_grid_is_fixed_and_has_one_shared_baseline(self):
        self.assertEqual(
            CAL.CELLS,
            ((2.5, 1.35), (2.5, 1.40), (2.5, 1.45), (2.5, 1.50),
             (3.0, 1.50), (3.5, 1.50)),
        )
        self.assertEqual(len(set(CAL.CELLS)), 6)

    def test_lowest_safe_controller_and_baseline_ceiling_are_selected(self):
        cells = [fixture(kp, speed) for kp, speed in CAL.CELLS]
        summary = CAL.summarize(cells)
        self.assertEqual(summary["baseline_attainable_speed_mps"], 1.5)
        self.assertEqual(summary["selected_controller_velocity_kp"], 3.0)
        self.assertEqual(summary["decision"], "BOTH_FOLLOWUPS_ELIGIBLE")

    def test_controller_stopping_regression_is_rejected(self):
        cells = [fixture(kp, speed) for kp, speed in CAL.CELLS]
        cells[4] = fixture(3.0, 1.5, stop_distance=0.67)
        summary = CAL.summarize(cells)
        self.assertEqual(summary["selected_controller_velocity_kp"], 3.5)

    def test_six_second_recovery_does_not_change_five_second_failure(self):
        cells = [fixture(kp, speed) for kp, speed in CAL.CELLS]
        cells[3]["cell_pass"] = False
        cells[3]["horizon"]["5"]["all_within_gate"] = False
        cells[3]["horizon"]["5"]["error_max_mps"] = 0.06
        cells[3]["warmup_speed_trace_mps"][499] = [1.44] * CAL.ENVS
        summary = CAL.summarize(cells)
        self.assertEqual(summary["baseline_attainable_speed_mps"], 1.45)


if __name__ == "__main__":
    unittest.main()
