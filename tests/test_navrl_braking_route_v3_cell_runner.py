"""CPU contract tests for the braking-route-v3 simulator cell adapter."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "tools/run_navrl_braking_route_v3_cell.py"

os.environ.pop("NAVRL_TARGET_BRAKING_CONTRACT_VARIANT", None)


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CELL = _load("braking_v3_cell_runner_standalone", "tools/run_navrl_braking_route_v3_cell.py")
GATE = CELL.GATE
R2 = CELL.R2


def _write_manifest(directory):
    manifest = Path(directory) / "source_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    return manifest, GATE.sha256_file(manifest)


def _environment(directory, **overrides):
    manifest, manifest_sha = _write_manifest(directory)
    receipt_sha = "f" * 64
    identity = {
        "preregistration_sha256": GATE.PREREG_SHA256,
        "braking_receipt_sha256": receipt_sha,
        "cell_runner_sha256": GATE.sha256_file(RUNNER_PATH),
        "import_origin_sha256": "a" * 64,
    }
    values = {
        "NAVRL_V3_CELL_OUTPUT": str(Path(directory) / "cell.json"),
        "NAVRL_V3_RECORD_ID": GATE.record_id("global_astar_braking_v3", 0.6, 70),
        "NAVRL_V3_STAGE": "pilot",
        "NAVRL_V3_SEED": "829",
        "NAVRL_V3_ROUTE_MODE": "global_astar_braking_v3",
        "NAVRL_V3_SPEED_MPS": "0.6",
        "NAVRL_V3_BARS": "70",
        "NAVRL_V3_ENVS": "32",
        "NAVRL_V3_STEPS": "300",
        "NAVRL_V3_WARMUP_STEPS": "20",
        "NAVRL_V3_IDENTITY_JSON": json.dumps(identity, sort_keys=True),
        "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT": "/tmp/receipt.json",
        "NAVRL_TARGET_RECOVERY_BRAKE_PROBE_RECEIPT_SHA256": receipt_sha,
        "NAVRL_TARGET_RECOVERY_PROBE_VALIDATED": "1",
        "NAVRL_TARGET_RECOVERY_BRAKE_P05": "0.5",
        "NAVRL_TARGET_RECOVERY_STOP_TIME_P95_S": "1.0",
        "NAVRL_TARGET_RECOVERY_BRAKE_SPEEDS_MPS": "0.6,0.9,1.2,1.5",
        "NAVRL_TARGET_RECOVERY_BRAKE_STOP_DISTANCES_M": "0.3,0.5,0.7,0.9",
        "NAVRL_TARGET_RECOVERY_BRAKE_LATERAL_TUBE_P95_M": "0.001",
        "MOTAR_V3_TRAINING_SOURCE_MANIFEST": str(manifest),
        "MOTAR_V3_TRAINING_SOURCE_MANIFEST_SHA256": manifest_sha,
    }
    values.update(overrides)
    return values


class CellRunnerContractTest(unittest.TestCase):
    def test_missing_contract_variable_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            values = _environment(directory)
            values.pop("NAVRL_V3_IDENTITY_JSON")
            with self.assertRaisesRegex(CELL.CellContractError, "NAVRL_V3_IDENTITY_JSON"):
                CELL.read_cell_contract(values)

    def test_training_source_passthrough_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            values = _environment(directory)
            values.pop("MOTAR_V3_TRAINING_SOURCE_MANIFEST")
            with self.assertRaisesRegex(CELL.CellContractError, "MOTAR_V3_TRAINING_SOURCE_MANIFEST"):
                CELL.read_cell_contract(values)

    def test_record_id_grid_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            values = _environment(
                directory, NAVRL_V3_RECORD_ID=GATE.record_id("off", 0.6, 70)
            )
            with self.assertRaisesRegex(CELL.CellContractError, "record id"):
                CELL.read_cell_contract(values)

    def test_seed_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            values = _environment(directory, NAVRL_V3_SEED="839")
            with self.assertRaisesRegex(CELL.CellContractError, "seed"):
                CELL.read_cell_contract(values)

    def test_runner_byte_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            values = _environment(directory)
            identity = json.loads(values["NAVRL_V3_IDENTITY_JSON"])
            identity["cell_runner_sha256"] = "0" * 64
            values["NAVRL_V3_IDENTITY_JSON"] = json.dumps(identity, sort_keys=True)
            with self.assertRaisesRegex(CELL.CellContractError, "cell-runner hash"):
                CELL.read_cell_contract(values)

    def test_happy_path_contract_parses(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = CELL.read_cell_contract(_environment(directory))
            self.assertEqual(contract["stage"], "pilot")
            self.assertEqual(contract["seed"], 829)
            self.assertEqual(contract["route_mode"], "global_astar_braking_v3")
            self.assertEqual(contract["speed_mps"], 0.6)
            self.assertEqual(contract["bars"], 70)
            self.assertEqual(contract["envs"], 32)
            self.assertEqual(contract["steps"], 300)
            self.assertEqual(contract["warmup_steps"], 20)

    def test_frozen_environment_binds_r2_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            contract = CELL.read_cell_contract(_environment(directory))
            values = CELL.frozen_cell_environment(contract)
            expected = R2.frozen_environment("global_astar_braking_v3", 0.6)
            for key, value in expected.items():
                self.assertEqual(values[key], value, key)
            self.assertEqual(values["NAVRL_TARGET_ROUTE_MODE"], "global_astar_braking_v3")
            self.assertEqual(values["NAVRL_ROBOT"], "navrl_ref5in_v2_quad")
            self.assertEqual(values["NAVRL_PLACEMENT_MODE"], "footprint_clearance")
            self.assertEqual(values["NAVRL_NUM_BARS"], "70")
            self.assertEqual(values["NAVRL_MAX_BARS"], "300")
            self.assertIn("NAVRL_TRAINING_SOURCE_MANIFEST", values)
            self.assertIn("NAVRL_TRAINING_SOURCE_MANIFEST_SHA256", values)
            # The braking lookup injected by the gate launcher must not be overridden here.
            for key in values:
                self.assertFalse(key.startswith("NAVRL_TARGET_RECOVERY_BRAKE"), key)

    def test_executable_fails_closed_without_contract(self):
        environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("NAVRL_", "MOTAR_V3_"))
        }
        completed = subprocess.run(
            [sys.executable, str(RUNNER_PATH)], cwd=ROOT, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.assertEqual(completed.returncode, 3)
        self.assertIn("CELL_VOID", completed.stderr)

    def test_import_is_cpu_safe(self):
        completed = subprocess.run(
            [
                sys.executable, "-c",
                "import importlib.util, sys;"
                "spec = importlib.util.spec_from_file_location('m', %r);"
                "m = importlib.util.module_from_spec(spec);"
                "spec.loader.exec_module(m);"
                "assert 'torch' not in sys.modules, 'torch imported at module load';"
                "assert 'isaacgym' not in sys.modules, 'isaacgym imported at module load';"
                "print('CPU_SAFE')" % str(RUNNER_PATH),
            ],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("CPU_SAFE", completed.stdout)


if __name__ == "__main__":
    unittest.main()
