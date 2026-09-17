"""The defect impact matrix and the 48-cell preflight must stay bound to source.

The matrix's whole value is one claim: D1 is confined to arm H because the
bounded/physical branch of _advance_target returns before the legacy push-out
block. If that stops being true, every arm-scoping decision below it is wrong and
the GPU evaluation must be re-gated.
"""
import ast
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
AUDITS = ROOT / "docs/audits"
MATRIX_JSON = AUDITS / "target_motion_defect_impact_matrix_2026-09-17.json"
MATRIX_DOC = AUDITS / "target_motion_defect_impact_matrix_2026-09-17.md"
PREFLIGHT = ROOT / "results/target_motion_e0_e2_preflight_2026-09-17"
NAVRL_TASK = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"
TARGET_MOTION = ROOT / "aerial_gym/task/navrl_task/target_motion.py"


def read(path):
    return path.read_text(encoding="utf-8")


class TheStructuralFactThatScopesD1(unittest.TestCase):
    longMessage = False

    def setUp(self):
        self.src = read(NAVRL_TASK)
        tree = ast.parse(self.src)
        self.fn = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == "_advance_target")

    def _dynamics_branch(self):
        for stmt in self.fn.body:
            if isinstance(stmt, ast.If):
                test = ast.get_source_segment(self.src, stmt.test) or ""
                if "_target_dynamics" in test and "bounded" in test:
                    return stmt
        return None

    def test_bounded_branch_ends_in_an_unconditional_return(self):
        branch = self._dynamics_branch()
        self.assertIsNotNone(branch, "the bounded/physical dynamics branch disappeared")
        self.assertIsInstance(
            branch.body[-1], ast.Return,
            "the bounded/physical branch no longer ends in a return: E0/E1/E2 may now "
            "reach the legacy push-out block and D1's arm scoping is invalid")
        self.assertFalse(branch.orelse,
                         "the bounded branch grew an else; re-derive the control flow")

    def test_legacy_block_is_after_that_return(self):
        branch = self._dynamics_branch()
        boundary = branch.body[-1].lineno
        lines = self.src.splitlines()
        markers = {
            "push_out": "push_dir[sel] = dirn[rows]",
            "bounce_jitter": "jit = (torch.rand(",
            "velocity_rewrite": "realized world velocity",
        }
        for name, needle in markers.items():
            found = [i for i, line in enumerate(lines, 1) if needle in line]
            self.assertTrue(found, f"legacy marker {name!r} vanished from source")
            self.assertGreater(
                found[0], boundary,
                f"{name} moved ABOVE the bounded branch's return; it is now reachable "
                "for E0/E1/E2 and the matrix must be rebuilt")

    def test_bounded_executor_consumes_no_random_draws(self):
        # If this ever fails, the RNG audit's cross-arm conclusion changes.
        self.assertIsNone(
            re.search(r"torch\.(rand|randn|randint)|\.normal_\(|\.uniform_\(",
                      read(TARGET_MOTION)),
            "target_motion.py now consumes RNG; redo the RNG audit before evaluating")


class MatrixIsConsistent(unittest.TestCase):
    longMessage = False

    def setUp(self):
        self.matrix = json.loads(read(MATRIX_JSON))

    def test_e2_is_not_affected_by_d1(self):
        d1 = next(d for d in self.matrix["defects"] if d["id"] == "D1")
        for arm in ("E0", "E1", "E2"):
            self.assertFalse(d1["arms"][arm], f"D1 is marked as affecting {arm}")
        self.assertTrue(d1["arms"]["H"], "D1 must be marked as affecting H")
        self.assertTrue(self.matrix["summary"]["e2_unaffected_by_d1"])

    def test_d1_is_type_a_and_preserved_not_fixed(self):
        d1 = next(d for d in self.matrix["defects"] if d["id"] == "D1")
        self.assertEqual(d1["type"], "TYPE-A")
        self.assertEqual(d1["action"], "preserve",
                         "D1 must be preserved for H's reproduction, not repaired")

    def test_no_type_b_defect_is_recorded_without_blocking(self):
        type_b = [d["id"] for d in self.matrix["defects"] if d["type"] == "TYPE-B"]
        self.assertEqual(type_b, self.matrix["summary"]["type_b_evaluation_invalidating"])
        if type_b:
            go = json.loads(read(PREFLIGHT / "go_no_go.json"))
            self.assertEqual(go["verdict"], "GPU_EVALUATION_NO_GO",
                             "a TYPE-B defect exists but the preflight says GO")

    def test_every_defect_has_a_complete_classification(self):
        required = {"id", "location", "description", "arms", "evidence", "training",
                    "evaluation", "physics", "observation", "receipt_only", "historical",
                    "new_eval", "severity", "type", "action", "rerun", "note"}
        self.assertEqual(len(self.matrix["defects"]), 9)
        for defect in self.matrix["defects"]:
            missing = required - set(defect)
            self.assertFalse(missing, f"{defect['id']} missing fields: {sorted(missing)}")
            self.assertIn(defect["type"], {"TYPE-A", "TYPE-B", "TYPE-C", "TYPE-D"})
            self.assertIn(defect["action"],
                          {"preserve", "erratum", "fix-new-lineage", "gate"})
            self.assertIn(defect["severity"], {"BLOCKER", "MAJOR", "MINOR", "DEBT"})
            self.assertIn(defect["rerun"], {"yes", "no", "unknown"})
            self.assertEqual(set(defect["arms"]), {"H", "E0", "E1", "E2"})

    def test_document_and_json_agree_on_arm_impact(self):
        # The document prints an arm-impact block; parse it back rather than
        # matching a formatting-sensitive literal.
        doc = read(MATRIX_DOC)
        block = re.search(r"Arm impact summary:\s*```text\n(.*?)```", doc, re.S)
        self.assertIsNotNone(block, "the document lost its arm-impact block")
        parsed = {}
        for line in block.group(1).strip().splitlines():
            arm, _, ids = line.partition(":")
            parsed[arm.strip()] = sorted(i.strip() for i in ids.split(",") if i.strip())
        self.assertEqual(parsed, self.matrix["summary"]["arm_impact"],
                         "the document's arm-impact block drifted from the JSON")

    def test_no_defect_is_marked_fixed_anywhere(self):
        doc = read(MATRIX_DOC)
        self.assertIn("No code is changed", doc)
        for defect in self.matrix["defects"]:
            self.assertNotEqual(defect["action"], "fix-now",
                                f"{defect['id']} claims an immediate fix")


class PreflightIsDryAndComplete(unittest.TestCase):
    longMessage = False

    def setUp(self):
        self.manifest = json.loads(read(PREFLIGHT / "cell_manifest.json"))
        self.comparison = json.loads(read(PREFLIGHT / "contract_comparison.json"))

    def test_matrix_is_exactly_48_cells(self):
        self.assertEqual(self.manifest["total_cells"], 48)
        self.assertEqual(len(self.manifest["cells"]), 48)
        self.assertEqual(self.manifest["episodes_per_cell"], 2048)
        self.assertEqual(self.manifest["total_episodes"], 48 * 2048)
        ids = {c["cell_id"] for c in self.manifest["cells"]}
        self.assertEqual(len(ids), 48, "duplicate cell ids")

    def test_every_cell_matches_the_checkpoint_contract(self):
        self.assertEqual(self.comparison["cells_confounded"], 0,
                         "a cell is confounded against the checkpoint contract")
        self.assertEqual(self.comparison["cells_matched"], 48)
        for cell in self.manifest["cells"]:
            self.assertEqual(cell["gate_verdict"], "MATCHED", cell["cell_id"])

    def test_held_fixed_env_replays_the_training_contract(self):
        held = self.manifest["held_fixed_env"]
        # These are exactly the knobs whose HEAD defaults differ from training.
        for key, value in (("NAVRL_MAX_VELOCITY", "2.5"), ("NAVRL_YAW_RATE_MAX", "3.0"),
                           ("NAVRL_LIDAR_HBEAMS", "72"), ("NAVRL_LIDAR_RANGE", "12"),
                           ("NAVRL_EPISODE_LEN_STEPS", "600"), ("NAVRL_OOB_MARGIN", "1.0")):
            self.assertEqual(held.get(key), value,
                             f"{key} no longer replays the training contract")

    def test_arm_env_differs_only_on_the_target_axis(self):
        arm_env = self.manifest["arm_env"]
        keys = {k for env in arm_env.values() for k in env}
        for key in keys:
            self.assertTrue(key.startswith("NAVRL_TARGET_"),
                            f"{key} varies by arm but is not a target-behaviour knob")
        self.assertEqual(arm_env["H_historical"]["NAVRL_TARGET_DYNAMICS"], "legacy")
        for arm in ("E0_static", "E1_cv", "E2_obstacle_aware"):
            self.assertEqual(arm_env[arm]["NAVRL_TARGET_DYNAMICS"], "bounded",
                             f"{arm} is not on bounded dynamics; D1 scoping would change")

    def test_preflight_produced_no_outcome(self):
        for name in ("capture_rate", "outcome", "episodes_completed", "results"):
            for path in PREFLIGHT.glob("*.json"):
                self.assertNotIn(f'"{name}"', read(path),
                                 f"{path.name} contains outcome-shaped key {name!r}; "
                                 "the preflight must be dry")


class GoNoGoIsDerivedNotAsserted(unittest.TestCase):
    longMessage = False

    def test_go_no_go_requires_authorization_regardless_of_verdict(self):
        path = PREFLIGHT / "go_no_go.json"
        if not path.is_file():
            self.skipTest("go_no_go.json not generated yet")
        go = json.loads(read(path))
        self.assertEqual(go["execution_status"], "AUTHORIZATION_REQUIRED",
                         "a GO verdict must still require operator authorisation")
        self.assertIn(go["verdict"], {"GPU_EVALUATION_GO", "GPU_EVALUATION_NO_GO"})
        self.assertEqual(go["failed_criteria"],
                         sorted(k for k, v in go["criteria"].items() if not v))
        if go["verdict"] == "GPU_EVALUATION_GO":
            self.assertEqual(go["failed_criteria"], [])

    def test_interpretation_contract_forbids_the_easy_misreading(self):
        path = PREFLIGHT / "go_no_go.json"
        if not path.is_file():
            self.skipTest("go_no_go.json not generated yet")
        contract = json.loads(read(path))["interpretation_contract"]
        self.assertIn("E2 is harder", contract["H_vs_E2"])
        self.assertIn("OUTSIDE training distribution", contract["E0_E1_E2"])
        self.assertIn("distributional", contract["matching_kind"])


if __name__ == "__main__":
    unittest.main()
