"""CPU-only tests for the aerial_gym import-origin guard.

The module under test is loaded BY FILE PATH on purpose: importing it as
``aerial_gym.rl_training.rl_games.navrl_import_origin`` would execute the aerial_gym package
chain, and these tests must run on a machine with no Isaac Gym and no torch.
"""

import hashlib
import importlib.util
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "aerial_gym" / "rl_training" / "rl_games" / "navrl_import_origin.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("navrl_import_origin_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


origin_guard = _load_module()


class EnvIsolation(unittest.TestCase):
    """Restore the two pieces of global state these tests read.

    NAVRL_REQUIRE_SOURCE_ROOT is obvious. The second one is not, and it cost a real bug: other
    files in this suite install a bare stand-in for ``aerial_gym`` in ``sys.modules`` so they can
    import task code without pulling in Isaac Gym. ``resolve_origin`` deliberately trusts the live
    module object over ``find_spec`` -- that is the whole point of the guard -- so under a full
    ``unittest discover`` it would read that stub, find no ``__file__``, and report
    ``package_dir=None``. These tests passed per-file and failed in the full run, which is the
    worst way for a test to be wrong. Drop a stub for the duration so the assertions are about the
    real package, and put back exactly what was there.
    """

    VAR = "NAVRL_REQUIRE_SOURCE_ROOT"

    @staticmethod
    def _is_stub(module):
        return module is not None and getattr(module, "__file__", None) is None

    def setUp(self):
        self._saved = os.environ.get(self.VAR, None)
        self._saved_module = sys.modules.get("aerial_gym", None)
        self._dropped_module = self._is_stub(self._saved_module)
        if self._dropped_module:
            del sys.modules["aerial_gym"]

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(self.VAR, None)
        else:
            os.environ[self.VAR] = self._saved
        if self._dropped_module:
            sys.modules["aerial_gym"] = self._saved_module


class TestInert(EnvIsolation):
    def test_unset_is_inert_and_never_raises(self):
        os.environ.pop(self.VAR, None)
        info = origin_guard.assert_runtime_origin()
        self.assertFalse(info["enforced"])
        self.assertNotIn("origin_sha256", info)

    def test_whitespace_only_is_also_inert(self):
        os.environ[self.VAR] = "   "
        info = origin_guard.assert_runtime_origin()
        self.assertFalse(info["enforced"])


class TestEnforcement(EnvIsolation):
    def test_bogus_root_raises_and_names_both_paths(self):
        bogus = "/nonexistent/definitely/not/a/worktree"
        os.environ[self.VAR] = bogus
        with self.assertRaises(RuntimeError) as ctx:
            origin_guard.assert_runtime_origin()
        message = str(ctx.exception)
        self.assertIn(bogus, message)
        actual = origin_guard.resolve_origin()["package_dir"]
        self.assertIsNotNone(actual)
        self.assertIn(actual, message)
        # The message must be actionable, not just a complaint.
        self.assertIn("PYTHONPATH", message)
        self.assertIn("editable", message)

    def test_actual_root_passes_and_reports_sha256(self):
        package_dir = origin_guard.resolve_origin()["package_dir"]
        self.assertIsNotNone(package_dir, "aerial_gym is not importable in this environment")
        actual_root = str(Path(package_dir).parent)
        os.environ[self.VAR] = actual_root
        info = origin_guard.assert_runtime_origin()
        self.assertTrue(info["enforced"])
        self.assertRegex(info["origin_sha256"], r"^[0-9a-f]{64}$")
        expected = hashlib.sha256(Path(info["origin"]).read_bytes()).hexdigest()
        self.assertEqual(info["origin_sha256"], expected)

    def test_explicit_argument_overrides_environment(self):
        os.environ[self.VAR] = "/nonexistent/ignored"
        package_dir = origin_guard.resolve_origin()["package_dir"]
        actual_root = str(Path(package_dir).parent)
        info = origin_guard.assert_runtime_origin(actual_root)
        self.assertTrue(info["enforced"])

    def test_parent_of_the_real_root_is_rejected(self):
        # The guard must require an EXACT package-dir match. A parent directory happens to
        # contain the real tree, so a substring/prefix style check would wrongly accept it.
        package_dir = origin_guard.resolve_origin()["package_dir"]
        parent_of_root = str(Path(package_dir).parent.parent)
        with self.assertRaises(RuntimeError):
            origin_guard.assert_runtime_origin(parent_of_root)


class TestResolution(EnvIsolation):
    def test_reports_the_live_module_when_already_imported(self):
        # find_spec answers "where would it resolve now", which is not the same question as
        # "which file is executing". When a module object exists, its __file__ wins.
        info = origin_guard.resolve_origin()
        if "aerial_gym" in sys.modules:
            live = Path(sys.modules["aerial_gym"].__file__).resolve()
            self.assertEqual(Path(info["origin"]), live)
        else:
            self.assertIsNotNone(info["origin"])

    def test_finder_label_is_one_of_the_documented_values(self):
        self.assertIn(origin_guard.resolve_origin()["finder"], ("editable", "path", "unknown"))


class TestPython38Compatibility(unittest.TestCase):
    def test_does_not_use_is_relative_to(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("is_relative_to", source)

    def test_module_imports_no_heavy_dependencies(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        for banned in ("import torch", "import isaacgym", "import numpy"):
            self.assertIsNone(
                re.search(r"^\s*%s" % re.escape(banned), source, re.MULTILINE), banned
            )


if __name__ == "__main__":
    unittest.main()
