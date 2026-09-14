"""CPU-only decision tests for stage-2 lower-speed/damping calibration."""

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "navrl_speed_controller_calibration_stage2",
    ROOT / "tools/run_navrl_physical_target_speed_controller_calibration_stage2.py",
)
S2 = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(S2)


def fixture(kp, speed, rate, *, passed=True, overshoot=0.1, distance=0.6):
    trace = [[speed] * S2.cal.ENVS for _ in range(S2.cal.WARMUP_STEPS * 10)]
    horizon = {str(t): {"speed_min_mps": speed, "speed_mean_mps": speed,
        "speed_max_mps": speed, "error_max_mps": 0.0, "all_within_gate": passed}
        for t in S2.cal.HORIZONS_S}
    return {"schema": S2.cal.SCHEMA + "_cell",
        "condition": {"velocity_kp": kp, "requested_speed_mps": speed, "rate_kp_scale": rate},
        "horizon": horizon, "sustained_4_to_5s": passed, "overshoot_max_mps": overshoot,
        "warmup": {"saturation_max": 0.0, "tilt_max_deg": 5.0},
        "braking": {"all_stopped": True, "stop_time_s": [0.5] * S2.cal.ENVS,
            "stop_distance_m": [distance] * S2.cal.ENVS,
            "lateral_deviation_m": [0.0] * S2.cal.ENVS,
            "saturation_max": 0.0, "tilt_max_deg": 5.0},
        "contact_count": 0, "invalid_count": 0, "cell_pass": passed,
        "warmup_speed_trace_mps": trace}


class Stage2DecisionTest(unittest.TestCase):
    def test_grid_and_selection_order_are_frozen(self):
        payloads = [fixture(*condition) for condition in S2.cal.STAGE2_CELLS]
        # Shared reference overshoot must be large enough for the 50% reduction rule.
        payloads[4]["overshoot_max_mps"] = 0.4
        summary = S2.summarize(payloads)
        self.assertEqual(summary["baseline_attainable_speed_mps"], 1.35)
        self.assertEqual(summary["selected_damped_controller"],
                         {"velocity_kp": 2.5, "rate_kp_scale": 1.5})

    def test_six_second_value_cannot_rescue_five_second_failure(self):
        payloads = [fixture(*condition) for condition in S2.cal.STAGE2_CELLS]
        payloads[3] = fixture(*S2.cal.STAGE2_CELLS[3], passed=False)
        speed = S2.cal.STAGE2_CELLS[3][1]
        payloads[3]["warmup_speed_trace_mps"][499] = [speed - 0.06] * S2.cal.ENVS
        payloads[3]["horizon"]["5"]["error_max_mps"] = 0.06
        for horizon in (6, 8, 10):
            payloads[3]["horizon"][str(horizon)]["all_within_gate"] = True
        payloads[4]["overshoot_max_mps"] = 0.4
        summary = S2.summarize(payloads)
        self.assertEqual(summary["baseline_attainable_speed_mps"], 1.3)

    def test_stopping_regression_blocks_candidate(self):
        payloads = [fixture(*condition) for condition in S2.cal.STAGE2_CELLS]
        payloads[4]["overshoot_max_mps"] = 0.4
        payloads[5] = fixture(*S2.cal.STAGE2_CELLS[5], distance=0.7)
        summary = S2.summarize(payloads)
        self.assertEqual(summary["selected_damped_controller"],
                         {"velocity_kp": 3.0, "rate_kp_scale": 1.5})


if __name__ == "__main__": unittest.main()
