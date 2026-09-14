"""CPU contracts for the preregistered D8-B evaluation-only campaign."""

import ast
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_dynamic_mesh_policy_sensitivity_d8b.py"
PREREG = ROOT / "docs/preregistration_dynamic_mesh_policy_sensitivity_d8b_2026-09-13.md"
TASK = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"
TREATMENT = ROOT / "aerial_gym/task/navrl_task/navrl_dynamic_mesh_treatment.py"
EVALUATOR = ROOT / "aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh"


def load_runner():
    spec = importlib.util.spec_from_file_location("d8b_runner_cpu_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreregistrationContract(unittest.TestCase):
    def test_grid_lineage_metric_and_margin_are_frozen(self):
        module = load_runner()
        self.assertEqual(module.SEEDS, (593, 599, 601))
        self.assertEqual(module.ARMS, ("analytic_flat", "mesh_flat", "mesh_shaded"))
        self.assertEqual(len(module.CELLS), 9)
        self.assertEqual(len({(seed, arm) for _, seed, arm in module.CELLS}), 9)
        self.assertEqual(module.EPISODES, 2049)
        self.assertEqual(module.MARGIN_PP, -3.0)
        self.assertEqual(module.CHECKPOINT_SHA,
                         "197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e")
        text = PREREG.read_text()
        for phrase in (
            "593", "599", "601", "2,049", "Student-t interval", "−3.0",
            "MATERIAL_LOSS", "NO_MATERIAL_LOSS_WITHIN_MARGIN",
            "INCONCLUSIVE_POLICY_SENSITIVITY", "does not authorize PPO",
        ):
            self.assertIn(phrase, text)

    def test_primary_classification_boundaries_are_literal(self):
        module = load_runner()
        integrity = {"all": True}
        self.assertEqual(
            module.classify_primary({"ci95": [-6.0, -3.1]}, integrity), "MATERIAL_LOSS"
        )
        self.assertEqual(
            module.classify_primary({"ci95": [-2.9, 1.0]}, integrity),
            "NO_MATERIAL_LOSS_WITHIN_MARGIN",
        )
        self.assertEqual(
            module.classify_primary({"ci95": [-3.1, 0.0]}, integrity),
            "INCONCLUSIVE_POLICY_SENSITIVITY",
        )
        self.assertEqual(
            module.classify_primary({"ci95": [-6.0, -3.1]}, {"all": False}),
            "INVALID_D8B_EXECUTION",
        )

    def test_seed_level_interval_not_episode_level(self):
        module = load_runner()
        result = module.paired_t([-1.0, 0.0, 1.0])
        self.assertEqual(result["unit"], "paired evaluation seed")
        self.assertEqual(result["df"], 2)
        self.assertEqual(result["mean"], 0.0)
        self.assertAlmostEqual(result["ci95"][0], -result["ci95"][1], places=12)
        with self.assertRaises(module.ContractError):
            module.paired_t([0.0, 1.0])


class RuntimeBoundaryContract(unittest.TestCase):
    def test_auto_attach_is_opt_in_and_result_attests_effective_treatment(self):
        source = TASK.read_text()
        self.assertIn('os.environ.get("NAVRL_DYNAMIC_MESH_TREATMENT", "off")', source)
        self.assertIn("if d8_raw not in", source)
        self.assertIn("load_pinned_v3_asset", source)
        self.assertIn("attach_dynamic_mesh_treatment", source)
        for field in (
            '"target_render_mode"', '"d8_dynamic_mesh_treatment"',
            '"d8_dynamic_mesh_asset_sha256"', '"d8_dynamic_mesh_contract"',
        ):
            self.assertIn(field, source)

    def test_runtime_loader_pins_asset_and_stays_behind_treatment_module(self):
        source = TREATMENT.read_text()
        self.assertIn("V3_ASSET_SHA256", source)
        self.assertIn("asset.source_sha256 != V3_ASSET_SHA256", source)
        self.assertIn("from tools.renderer_validation.urdf_asset import load_urdf_asset", source)
        task_tree = ast.parse(TASK.read_text())
        top_imports = [node for node in task_tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(
            "renderer_validation" in (getattr(node, "module", "") or "") for node in top_imports
        ))

    def test_extra_runtime_roots_are_allowlisted_not_ambient(self):
        source = EVALUATOR.read_text()
        self.assertIn("NAVRL_EVAL_EXTRA_RUNTIME_ROOTS", source)
        self.assertIn("allowed_extra_roots", source)
        self.assertIn("unsupported extra runtime roots", source)
        for root in (
            "resources/models/environment_assets/objects", "tools/renderer_validation"
        ):
            self.assertIn(root, source)

    def test_launcher_is_evaluation_only_and_fail_closed(self):
        source = RUNNER.read_text()
        self.assertIn('command = ["bash", str(EVALUATOR)', source)
        self.assertNotIn("train.py", source)
        self.assertNotIn("runner.run_train", source)
        for phrase in (
            '"NAVRL_V2_FORCE" not in env',
            '"training_started": False',
            '"no_training": True',
            "all_nine_cells",
            "D8-B preregistration must precede implementation",
            "D8-B runtime source is dirty",
        ):
            self.assertIn(phrase, source)

    def test_factor_environment_changes_only_declared_values(self):
        module = load_runner()
        left = module.evaluation_env("s593_analytic_flat", preflight=True)
        right = module.evaluation_env("s593_mesh_shaded", preflight=True)
        differences = {key for key in set(left) | set(right) if left.get(key) != right.get(key)}
        self.assertEqual(
            differences,
            {"NAVRL_V2_RESULT_DIR", "NAVRL_DYNAMIC_MESH_TREATMENT"},
        )
        other_seed = module.evaluation_env("s599_analytic_flat", preflight=True)
        differences = {
            key for key in set(left) | set(other_seed) if left.get(key) != other_seed.get(key)
        }
        self.assertEqual(differences, {"NAVRL_V2_RESULT_DIR", "NAVRL_SEED"})


if __name__ == "__main__":
    unittest.main()
