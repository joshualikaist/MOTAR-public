"""CPU contract tests for the opt-in lower-1.25 braking/recovery lineage."""

import json
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/fair/miniconda3/envs/aerialgym/bin/python"


def inspect(variant):
    env = os.environ.copy()
    env.update({"PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "tools")})
    if variant is not None:
        env["NAVRL_TARGET_BRAKING_CONTRACT_VARIANT"] = variant
    else:
        env.pop("NAVRL_TARGET_BRAKING_CONTRACT_VARIANT", None)
    code = """
import json
import probe_navrl_physical_target_braking as p
import verify_navrl_physical_target_braking as v
import verify_navrl_physical_target_recovery_v2_gate as g
print(json.dumps({'variant':p.CONTRACT_VARIANT,'speeds':p.REGISTERED_SPEEDS,
 'keys':[v.speed_key(x) for x in p.REGISTERED_SPEEDS],
 'gate_speeds':g.SPEEDS,'ids':[g.record_id('off',x,70) for x in g.SPEEDS],
 'prereg':g.PREREG.name,'default':g.DEFAULT_DIR.name,
 'child_variant':g.build_child_environment().get('NAVRL_TARGET_BRAKING_CONTRACT_VARIANT')}))
"""
    completed = subprocess.run([PYTHON, "-c", code], cwd=str(ROOT), env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        raise AssertionError(completed.stderr)
    return json.loads(completed.stdout)


class LowerContractTest(unittest.TestCase):
    def test_default_lineage_remains_canonical(self):
        payload = inspect(None)
        self.assertEqual(payload["variant"], "canonical_1p5")
        self.assertEqual(payload["speeds"], [0.6, 0.9, 1.2, 1.5])

    def test_lower_lineage_preserves_exact_1p25_identity(self):
        payload = inspect("baseline_1p25")
        self.assertEqual(payload["speeds"], [0.6, 0.9, 1.2, 1.25])
        self.assertEqual(payload["gate_speeds"], payload["speeds"])
        self.assertEqual(payload["keys"], ["0.6", "0.9", "1.2", "1.25"])
        self.assertEqual(len(set(payload["ids"])), 4)
        self.assertIn("lower1p25", payload["prereg"])
        self.assertIn("lower1p25", payload["default"])
        self.assertEqual(payload["child_variant"], "baseline_1p25")

    def test_unknown_variant_fails_closed(self):
        env = os.environ.copy(); env.update({"NAVRL_TARGET_BRAKING_CONTRACT_VARIANT": "unknown",
            "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "tools")})
        completed = subprocess.run([PYTHON, "-c", "import probe_navrl_physical_target_braking"],
                                   cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, check=False)
        self.assertNotEqual(completed.returncode, 0)

    def test_fresh_child_imports_isaacgym_before_packed_torch(self):
        env = os.environ.copy()
        env.update({"NAVRL_TARGET_BRAKING_CONTRACT_VARIANT": "baseline_1p25",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "tools"),
                    "PATH": str(Path(PYTHON).parent) + os.pathsep + env.get("PATH", "")})
        code = """
import sys
sys.argv = ['recovery-v2-test', '--_child']
import verify_navrl_physical_target_recovery_v2_gate
import aerial_gym
print(aerial_gym.__file__)
"""
        completed = subprocess.run([PYTHON, "-c", code], cwd=str(ROOT), env=env, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(str(ROOT / "aerial_gym"), completed.stdout)


if __name__ == "__main__": unittest.main()
