import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from history_helpers import require_research_history


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
import sys
sys.path.insert(0, str(TOOLS))

import probe_navrl_physical_target_braking as probe
import run_navrl_physical_target_braking_v2_fresh as launcher
import verify_navrl_physical_target_braking as verifier


def fake_cell(speed):
    center = [[20.0, 20.0] for _ in range(probe.REGISTERED_ENVS)]
    traces = []
    for index in range(1, probe.WARMUP_STEPS * probe.PHYSICS_SUBSTEPS + 1):
        position = [[20.0 + 0.006 * index, 20.0] for _ in range(probe.REGISTERED_ENVS)]
        traces.append({
            "phase": "warmup", "sample_index": index, "elapsed_s": index * probe.PHYSICS_DT_S,
            "speed_mps": [speed] * probe.REGISTERED_ENVS, "position_xy_m": position,
            "step_distance_m": [0.006] * probe.REGISTERED_ENVS,
            "path_distance_m": [0.006 * index] * probe.REGISTERED_ENVS,
            "contact": [False] * probe.REGISTERED_ENVS, "invalid_obb": [False] * probe.REGISTERED_ENVS,
            "motor_saturation_fraction": [0.01] * probe.REGISTERED_ENVS, "max_tilt_deg": [10.0] * probe.REGISTERED_ENVS,
        })
    for index in range(1, probe.PHYSICS_SUBSTEPS * 2 + 1):
        current_speed = speed * 0.5 if index <= probe.PHYSICS_SUBSTEPS else 0.05
        position = [[20.0 + 3.0 + 0.005 * index, 20.0] for _ in range(probe.REGISTERED_ENVS)]
        traces.append({
            "phase": "brake", "sample_index": index, "elapsed_s": index * probe.PHYSICS_DT_S,
            "speed_mps": [current_speed] * probe.REGISTERED_ENVS, "position_xy_m": position,
            "step_distance_m": [0.005] * probe.REGISTERED_ENVS,
            "path_distance_m": [0.005 * index] * probe.REGISTERED_ENVS,
            "contact": [False] * probe.REGISTERED_ENVS, "invalid_obb": [False] * probe.REGISTERED_ENVS,
            "motor_saturation_fraction": [0.02] * probe.REGISTERED_ENVS, "max_tilt_deg": [12.0] * probe.REGISTERED_ENVS,
        })
    provenance = {
        "python_executable": "synthetic", "python_executable_sha256": "7" * 64, "python_version": "3.8", "torch_version": "synthetic",
        "torch_cuda_version": "synthetic", "cuda_device": "synthetic", "nvidia_smi_path": "synthetic",
        "nvidia_smi_sha256": "0" * 64, "nvidia_smi_identity": "synthetic", "gpu_driver_version": "synthetic",
        "ninja_path": "synthetic", "ninja_sha256": "1" * 64, "ninja_version": "synthetic",
        "selected_python_contract": "synthetic",
        "imported_modules": {"isaacgym": {"path": "/synthetic/isaacgym.py", "sha256": "4" * 64, "root_bound": False}},
        "tool_hashes": {path: "3" * 64 for path in probe.TOOL_SOURCE_PATHS},
    }
    for module, path in probe.EXPECTED_REPO_IMPORTS.items():
        provenance["imported_modules"][module] = {"path": path, "sha256": probe.sha256_file(ROOT / path), "root_bound": True}
    provenance["tool_hashes"] = {path: probe.sha256_file(ROOT / path) for path in probe.TOOL_SOURCE_PATHS}
    rows = []
    for env_id in range(probe.REGISTERED_ENVS):
        rows.append({
            "env_id": env_id,
            "requested_speed_mps": speed,
            "measured_initial_speed_mps": speed,
            "warmup_final_speed_mps": speed, "warmup_speed_error_mps": 0.0, "warmup_converged": True,
            "stop_time_s": 0.11, "stop_distance_m": 0.055, "endpoint_displacement_m": 0.055, "max_lateral_deviation_m": 0.0,
            "effective_deceleration_mps2": speed * speed / 0.11,
            "warmup_contact": False, "warmup_invalid_obb": False,
            "contact": False,
            "invalid_obb": False,
            "warmup_motor_saturation_fraction": 0.01, "warmup_max_tilt_deg": 10.0,
            "motor_saturation_fraction": 0.02, "max_tilt_deg": 12.0,
        })
    return {
        "schema": probe.SCHEMA,
        "cell": {"speed_mps": speed, "envs": probe.REGISTERED_ENVS, "seed": 827, "child_auth": {"record_id": "synthetic", "sha256": "6" * 64}},
        "contract": probe.FROZEN_CONTRACT,
        "source_attestation": {"clean": True, "git_head": probe.git_head(ROOT), "required_core_base_commit": "dac38227cc3ad2130c84755ef9aab2e75becb9f0"},
        "setup": {"mode": "obstacle_free_center", "active_bars": 0, "center_xy_m": center,
                   "center_clearance_to_arena_m": [19.0] * probe.REGISTERED_ENVS,
                   "warmup_steps": probe.WARMUP_STEPS, "brake_steps_budget": probe.BRAKE_STEPS_BUDGET,
                   "warmup_substeps": [probe.WARMUP_STEPS * probe.PHYSICS_SUBSTEPS] * probe.REGISTERED_ENVS,
                   "warmup_path_distance_m": [3.0] * probe.REGISTERED_ENVS},
        "instantiated": {"sim_name": "base_sim", "envs": probe.REGISTERED_ENVS,
                          "controller_dt_s": probe.PHYSICS_DT_S,
                          "controller_substeps_per_rl_step": probe.PHYSICS_SUBSTEPS,
                          "physical_box_xyz_m": probe.FROZEN_CONTRACT["physical_box_xyz_m"],
                          "physical_support_xy_m": probe.FROZEN_CONTRACT["physical_support_xy_m"]},
        "raw_samples": rows,
        "physics_samples": traces,
        "provenance": provenance,
    }


class PhysicalTargetBrakingContractTest(unittest.TestCase):
    def test_preflight_is_cpu_safe_and_has_exact_grid(self):
        completed = subprocess.run(
            [sys.executable, str(TOOLS / "probe_navrl_physical_target_braking.py"), "--preflight", "--output", "/tmp/unused"],
            cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["contract"]["envs"], 32)
        self.assertEqual(payload["contract"]["max_bars"], 300)
        self.assertEqual(payload["contract"]["physics_substeps"], 10)
        self.assertEqual(payload["contract"]["rl_step_dt_s"], 0.1)

    def test_raw_validator_recomputes_speed_lookup(self):
        cells = [fake_cell(speed) for speed in probe.REGISTERED_SPEEDS]
        summary = verifier.summarize_cells(cells)
        self.assertEqual(summary["speeds_mps"], list(probe.REGISTERED_SPEEDS))
        lookup = summary["measured_speed_to_p95_lookup"]
        self.assertEqual(sorted(lookup), ["0.6", "0.9", "1.2", "1.5"])
        expected = verifier.quantile_stats([row["stop_distance_m"] for row in cells[0]["raw_samples"]])["p95"]
        self.assertEqual(lookup["0.6"]["p95_stop_distance_m"], expected)
        self.assertEqual(verifier.certified_lookup_for_speed(summary, 1.2)["speed_mps"], 1.2)
        cells[0]["raw_samples"][0]["contact"] = True
        with self.assertRaises(ValueError):
            verifier.validate_cell(cells[0])

    def test_summary_and_receipt_are_immutable_and_relative(self):

        require_research_history(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "braking"
            output.mkdir()
            (output / "cells").mkdir()
            cells = []
            for speed in probe.REGISTERED_SPEEDS:
                path = output / "cells" / ("speed_%s.json" % format(speed, ".1f").replace(".", "p"))
                path.write_bytes(probe.canonical_json_bytes(fake_cell(speed)))
                cell_payload = json.loads(path.read_text(encoding="utf-8"))
                cells.append({
                    "path": str(path.relative_to(output)),
                    "speed_mps": speed,
                    "sha256": probe.sha256_file(path),
                    "provenance_sha256": probe.sha256_bytes(probe.canonical_json_bytes(cell_payload["provenance"])),
                })
            manifest = probe.source_manifest(ROOT, launcher.TOOL_PATHS)
            (output / "source_manifest.json").write_bytes(probe.canonical_json_bytes(manifest))
            summary = verifier.summarize_cells([fake_cell(speed) for speed in probe.REGISTERED_SPEEDS])
            core_handoff = verifier.core_integration_object(summary)
            (output / "summary.json").write_bytes(probe.canonical_json_bytes(summary))
            receipt = {
                "schema": probe.RECEIPT_SCHEMA,
                "probe_schema": probe.SCHEMA,
                "subject": "physical_target_ref5in_actor",
                "contract": probe.FROZEN_CONTRACT,
                "git_head": probe.git_head(ROOT),
                "core_base_commit": "dac38227cc3ad2130c84755ef9aab2e75becb9f0",
                "source_clean": True,
                "source_manifest": "source_manifest.json",
                "source_manifest_sha256": probe.sha256_file(output / "source_manifest.json"),
                "cells": cells,
                "summary_path": "summary.json",
                "summary_sha256": probe.sha256_file(output / "summary.json"),
                "summary": summary,
                "speed_cells": summary["speed_cells"],
                "decel_p05_mps2": min(row["p05_effective_deceleration_mps2"] for row in summary["measured_speed_to_p95_lookup"].values()),
                "stop_time_p95_s": max(row["p95_stop_time_s"] for row in summary["measured_speed_to_p95_lookup"].values()),
                "measured_speed_to_p95_lookup": summary["measured_speed_to_p95_lookup"],
                "certified_monotone_speed_to_p95_lookup": summary["certified_monotone_speed_to_p95_lookup"],
                "core_integration": core_handoff,
                "fresh_only": True,
                "process_isolation": "one fresh Isaac Gym child per registered speed",
            }
            (output / "receipt.json").write_bytes(probe.canonical_json_bytes(receipt))
            (output / "complete.marker").write_text("COMPLETE\n", encoding="utf-8")
            verified = verifier.verify_receipt(output, ROOT)
            self.assertTrue(verified["verified"])
            self.assertEqual(receipt["cells"][0]["path"], "cells/speed_0p6.json")
            # A producer-side summary cannot hide raw tampering.
            mutated = json.loads((output / "cells/speed_0p6.json").read_text(encoding="utf-8"))
            mutated["raw_samples"][0]["stop_distance_m"] = 99.0
            (output / "cells/speed_0p6.json").write_bytes(probe.canonical_json_bytes(mutated))
            with self.assertRaises(ValueError):
                verifier.verify_receipt(output, ROOT)

    def test_fresh_launcher_rejects_continuation_arguments(self):
        completed = subprocess.run(
            [sys.executable, str(TOOLS / "run_navrl_physical_target_braking_v2_fresh.py"), "--output", "/tmp/not-created", "--checkpoint", "/tmp/x"],
            cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertIn("fresh-only", completed.stdout)

    def test_adversarial_trace_row_and_warmup_gates_fail_closed(self):
        forged = fake_cell(0.6)
        forged["physics_samples"][-1]["path_distance_m"][0] = 999.0
        with self.assertRaises(ValueError):
            verifier.validate_cell(forged)
        forged = fake_cell(0.6)
        forged["physics_samples"][500]["speed_mps"][0] = 0.05
        with self.assertRaises(ValueError):
            verifier.validate_cell(forged)
        forged = fake_cell(0.6)
        forged["physics_samples"][0]["contact"][0] = True
        with self.assertRaises(ValueError):
            verifier.validate_cell(forged)
        forged = fake_cell(0.6)
        forged["provenance"]["imported_modules"]["aerial_gym"]["path"] = "hostile.py"
        with self.assertRaises(ValueError):
            verifier.validate_cell(forged)

    def test_nonoverwrite_output_gate_is_hard(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "existing"
            existing.mkdir()
            with self.assertRaises(SystemExit):
                launcher._safe_output(existing, Path(temporary))

    def test_substep_curve_exceeds_endpoint_chord_and_records_lateral_peak(self):
        points = [(0.0, 0.0), (0.05, 0.05), (0.10, 0.0), (0.15, -0.05), (0.20, 0.0)]
        path = sum(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 for (x0, y0), (x1, y1) in zip(points, points[1:]))
        chord = ((points[-1][0] - points[0][0]) ** 2 + (points[-1][1] - points[0][1]) ** 2) ** 0.5
        lateral_peak = max(abs(y) for _, y in points)
        self.assertGreater(path, chord)
        self.assertEqual(lateral_peak, 0.05)

    def test_core_files_are_not_modified_by_this_lineage(self):

        require_research_history(ROOT)
        changed = subprocess.run(
            ["git", "diff", "--name-only", "393d1a2^", "c82035f", "--", *probe.CORE_PATHS],
            cwd=str(ROOT), text=True, stdout=subprocess.PIPE, check=True,
        ).stdout.splitlines()
        # The probe lineage itself must not modify runtime core behavior. Integrated hash pins are
        # applied in a separate commit after this probe range.
        self.assertEqual(changed, [])

if __name__ == "__main__":
    unittest.main()
