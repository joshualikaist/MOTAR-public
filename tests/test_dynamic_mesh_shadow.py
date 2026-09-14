"""The D7 shadow path must be off by default and must not reach the detector's output.

CPU-only source checks: the GPU invariance evidence lives in
results/dynamic_mesh_integrated_cost_2026-09-12/. These assert the properties that make that
evidence meaningful, so a later edit cannot quietly wire the shadow into an observation.
"""
import ast
from pathlib import Path
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SHADOW = ROOT / "aerial_gym/task/navrl_task/navrl_dynamic_mesh_shadow.py"
DETECTOR = ROOT / "aerial_gym/task/navrl_task/navrl_detector.py"


_ABSENT = object()


def shadow_module():
    """Load the module against a minimal warp stub, and restore sys.modules afterwards.

    Only shadow_enabled() is exercised here and it is pure os.environ logic; warp is imported by
    the module solely so its @wp.kernel decorator can evaluate. Importing the real package would
    make these CPU-only tests depend on CUDA, and on a full-suite run the name can already hold
    another test's stub or a half-initialised real package. A stub of our own avoids both, and
    the entry is put back exactly as it was found.
    """
    import importlib.util
    import types

    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            return _passthrough

    def _passthrough(fn=None, **_kwargs):
        return fn if callable(fn) else _passthrough

    saved = sys.modules.get("warp", _ABSENT)
    stub = _Stub("warp")
    stub.kernel = _passthrough
    sys.modules["warp"] = stub
    try:
        spec = importlib.util.spec_from_file_location("navrl_dynamic_mesh_shadow_test", SHADOW)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if saved is _ABSENT:
            sys.modules.pop("warp", None)
        else:
            sys.modules["warp"] = saved


class FlagTest(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.get("NAVRL_DYNAMIC_MESH_SHADOW")

    def tearDown(self):
        if self.saved is None:
            os.environ.pop("NAVRL_DYNAMIC_MESH_SHADOW", None)
        else:
            os.environ["NAVRL_DYNAMIC_MESH_SHADOW"] = self.saved

    def test_default_is_off(self):
        module = shadow_module()
        os.environ.pop("NAVRL_DYNAMIC_MESH_SHADOW", None)
        self.assertFalse(module.shadow_enabled())

    def test_recognised_spellings_both_ways(self):
        module = shadow_module()
        for value in ("1", "true", "yes", "on", "ON", "True"):
            os.environ["NAVRL_DYNAMIC_MESH_SHADOW"] = value
            self.assertTrue(module.shadow_enabled(), value)
        for value in ("0", "false", "no", "off", ""):
            os.environ["NAVRL_DYNAMIC_MESH_SHADOW"] = value
            self.assertFalse(module.shadow_enabled(), value)

    def test_an_unknown_value_raises_rather_than_reading_as_off(self):
        module = shadow_module()
        for value in ("Y", "enabled", "TRUE1", "2"):
            os.environ["NAVRL_DYNAMIC_MESH_SHADOW"] = value
            with self.assertRaises(ValueError, msg=value):
                module.shadow_enabled()


class NonInterferenceTest(unittest.TestCase):
    """Static guarantees that the shadow's buffers cannot become an observation."""

    def test_the_detector_never_reads_the_shadow_buffers(self):
        source = DETECTOR.read_text(encoding="utf-8")
        for name in ("shadow_depth", "shadow_hit"):
            self.assertNotIn(name, source,
                             f"navrl_detector references {name}; the shadow must be write-only "
                             f"from the detector's side")

    def test_the_detector_only_launches_the_shadow_and_stores_none_by_default(self):
        tree = ast.parse(DETECTOR.read_text(encoding="utf-8"))
        uses = [node for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and node.attr == "_dynamic_mesh_shadow"]
        self.assertGreaterEqual(len(uses), 3)
        source = DETECTOR.read_text(encoding="utf-8")
        self.assertIn("self._dynamic_mesh_shadow = None", source)
        # The only call made on it is run(); anything returning data would be a leak.
        calls = {node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and isinstance(node.func.value, ast.Attribute)
                 and node.func.value.attr == "_dynamic_mesh_shadow"}
        self.assertEqual(calls, {"run"}, f"detector calls {calls} on the shadow; expected only run")

    def test_attaching_requires_the_flag(self):
        source = DETECTOR.read_text(encoding="utf-8")
        self.assertIn("shadow_enabled()", source)
        self.assertIn("refusing to attach shadow instrumentation", source)

    def test_the_shadow_module_is_imported_lazily(self):
        """With the flag off the module must never be imported, so the default path is untouched."""
        tree = ast.parse(DETECTOR.read_text(encoding="utf-8"))
        top_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = [getattr(node, "module", "") or "" for node in top_level]
        self.assertFalse(any("dynamic_mesh_shadow" in name for name in names),
                         "the shadow module is imported at detector module level")

    def test_the_shadow_consumes_no_randomness(self):
        source = SHADOW.read_text(encoding="utf-8")
        for forbidden in ("random", "rand(", "randn", "seed("):
            self.assertNotIn(forbidden, source,
                             f"the shadow path references {forbidden}; it must be deterministic "
                             f"geometry only so it cannot move the scene or trajectory")


class EvidenceTest(unittest.TestCase):
    def test_the_recorded_run_shows_output_invariance_in_every_cell(self):
        import json
        path = ROOT / "results/dynamic_mesh_integrated_cost_2026-09-12/benchmark/benchmark.json"
        if not path.exists():
            self.skipTest("D7 benchmark receipt is not present in this checkout")
        payload = json.loads(path.read_text())
        self.assertTrue(payload["rows"], "receipt has no cells")
        for row in payload["rows"]:
            with self.subTest(cell=(row["envs"], row["width"], row["height"])):
                self.assertTrue(row["all_hashes_equal"])
                # The shadow must also have actually run, or invariance is trivially true.
                self.assertGreater(row["shadow"]["shadow_hit_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
