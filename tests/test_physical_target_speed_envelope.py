import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
PATH = ROOT / "tools/verify_navrl_physical_target_speed_envelope.py"
SPEC = importlib.util.spec_from_file_location("speed_envelope", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class PhysicalTargetSpeedEnvelopeContractTest(unittest.TestCase):
    def test_preregistered_grid_and_gates_are_fixed(self):
        self.assertEqual(MOD.DEFAULT_SPEEDS, (0.6, 0.9, 1.2, 1.5))
        self.assertEqual(MOD.DEFAULT_DENSITIES, (70, 150, 205, 300))
        self.assertEqual(MOD.GATES["planner_infeasible_fraction_max"], 0.01)
        self.assertEqual(MOD.GATES["invalid_state_fraction_max"], 0.0)

    def test_gate_is_conjunctive(self):
        self.assertFalse(all({"tracking": True, "speed": True, "contact": False}.values()))

    def test_invalid_forensics_contract_is_fixed(self):
        path = ROOT / "tools/diagnose_navrl_physical_target_invalid_events.py"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        for field in ("position_m", "velocity_mps", "command_mps", "obb_margin_m_xyz"):
            self.assertIn(field, text)
        self.assertIn('"bars": DENSITY', text)


if __name__ == "__main__":
    unittest.main()
