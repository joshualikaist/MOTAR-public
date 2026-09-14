"""Meta-test: make "unittest discover collects nothing but still reports OK" impossible.

Hazard this guards against
---------------------------
`pytest` is NOT installed in this environment. `python -m unittest discover` is the only test
runner available here. `unittest discover` only ever collects `unittest.TestCase` test
methods -- it does not run bare, pytest-style module-level `def test_*(): ...` functions. A
test file that defines such functions but never wires them into a `TestCase`, and never runs
its checks any other way, imports cleanly, executes nothing, and reports:

    Ran 0 tests in 0.000s
    OK

That is a false-positive pass: the checks exist and *read* as coverage, but nothing actually
enforces them. `tests/test_navrl_target_motion.py` had exactly this hole (11 pytest-style
`test_*` functions, zero `TestCase` classes, zero self-run harness) until it was fixed by
adding a `unittest.TestCase` adapter that wraps each function as its own test method. If this
meta-test had existed before that fix, it would have failed on that file; see the
non-vacuity check this file also carries in mind when editing it.

What "safe" means
------------------
This test scans every `tests/test_*.py` file with `ast` -- it does NOT import them, because
several of these files pull in heavy/GPU-only dependencies (e.g. `isaacgym`) that must not be
a prerequisite for running this meta-test. A file is considered safe from the collection hole
if either of the following holds:

  1. Every module-level `test_*` function is mirrored by a same-named `test_*` method on some
     `unittest.TestCase` subclass declared in the file -- the adapter pattern used by
     `test_navrl_target_motion.py`. (A file with no module-level `test_*` functions at all,
     and at least one real `TestCase` test method, trivially satisfies this.)

  2. The file "self-executes": its module-level code (outside any def/class body) directly
     calls a function defined elsewhere in the same module -- i.e. it actually runs its checks
     as a side effect of being imported -- AND that module-level code is capable of raising an
     exception or calling `sys.exit`/`os.exit`/`exit()` on failure, so a broken check surfaces
     as an import-time error instead of silence. `tests/test_navrl_corridor.py` follows this
     pattern: it appends failures to a list via module-level calls to a local `check()` helper
     and, at the bottom of the module, `sys.exit(1)`s if that list is non-empty.

A file failing both of the above is a hole: it looks like a test file, `unittest discover`
will report it as contributing zero (or fewer) tests than it should, and nothing else in this
environment will ever notice a regression in it.

If this test fails, the fix is the same one applied to `test_navrl_target_motion.py`: either
wire the module-level functions into a `unittest.TestCase`, or make the module genuinely
self-executing (run its checks at import time and raise/exit on failure).

This file also separately asserts that no `tests/test_*.py` file imports `pytest` at module
level in a way that would break import in this pytest-less environment (a guarded
`try: import pytest / except ImportError` would be fine; an unconditional top-level
`import pytest` would not be, and is checked for).
"""

import ast
import os
import unittest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _iter_test_files(directory):
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("test_") and fname.endswith(".py"):
            yield fname


def _is_testcase_base(base):
    """True if an ast base-class expression plausibly refers to unittest.TestCase.

    Matches both `class Foo(TestCase):` (name imported directly) and
    `class Foo(unittest.TestCase):` (attribute access) -- structural, not name-based on the
    class itself, so it works for any file using either import style.
    """
    if isinstance(base, ast.Name):
        return base.id == "TestCase"
    if isinstance(base, ast.Attribute):
        return base.attr == "TestCase"
    return False


def _module_level_statements(body):
    """Yield top-level statements that are not def/class/import (used for structural scans of
    "real" module-level side-effecting code, as opposed to declarations)."""
    for node in body:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom),
        ):
            continue
        yield node


def _calls_local_function(body, local_names):
    """True if some module-level statement (possibly nested in an if/for/while/try at module
    level) calls a function defined elsewhere at module level in the same file."""
    for node in _module_level_statements(body):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                if sub.func.id in local_names:
                    return True
    return False


def _can_raise_or_exit(body):
    """True if some module-level statement can raise or terminate the process on failure."""
    for node in _module_level_statements(body):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Raise):
                return True
            if isinstance(sub, ast.Call):
                fn = sub.func
                if isinstance(fn, ast.Name) and fn.id == "exit":
                    return True
                if (
                    isinstance(fn, ast.Attribute)
                    and fn.attr == "exit"
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id in ("sys", "os")
                ):
                    return True
    return False


def analyze_collection_gap(source, filename="<test file>"):
    """Parse a test module's source and report whether it has an unittest-collection gap.

    Returns a dict with:
      has_gap        -- True if this file has pytest-style test_* functions that unittest
                         discover would not collect, or has no unittest-collectible surface at
                         all (neither TestCase methods nor covered module-level functions).
      self_executes  -- True if the module runs its own checks at import time and can signal
                         failure via raise/exit.
      module_funcs   -- module-level `test_*` function names.
      testcase_methods -- `test_*` method names found on TestCase subclasses in the file.
      uncovered      -- module_funcs not mirrored by a same-named TestCase method.
      is_safe        -- True if unittest discover (or self-execution) actually enforces this
                         file's checks, i.e. not (has_gap and not self_executes).
    """
    tree = ast.parse(source, filename=filename)

    module_funcs = set()
    local_defs = set()
    testcase_methods = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local_defs.add(node.name)
            if node.name.startswith("test_"):
                module_funcs.add(node.name)
        elif isinstance(node, ast.ClassDef):
            if any(_is_testcase_base(base) for base in node.bases):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith(
                        "test_"
                    ):
                        testcase_methods.add(item.name)

    uncovered = module_funcs - testcase_methods
    has_no_unittest_surface = not module_funcs and not testcase_methods
    has_gap = bool(uncovered) or has_no_unittest_surface

    self_executes = _calls_local_function(tree.body, local_defs) and _can_raise_or_exit(tree.body)

    return {
        "has_gap": has_gap,
        "self_executes": self_executes,
        "module_funcs": module_funcs,
        "testcase_methods": testcase_methods,
        "uncovered": uncovered,
        "is_safe": not (has_gap and not self_executes),
    }


def _imports_pytest_unconditionally(source, filename="<test file>"):
    """True if the module has an unconditional, module-level `import pytest` /
    `from pytest import ...` (a guarded try/except ImportError would not be flagged, since it
    would not actually break import when pytest is absent)."""
    tree = ast.parse(source, filename=filename)
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name == "pytest" or alias.name.startswith("pytest.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "pytest" or node.module.startswith("pytest.")):
                return True
    return False


class TestSuiteCollectionTestCase(unittest.TestCase):
    """Guards the whole tests/ tree against the unittest-discovery hole described above."""

    def test_every_test_file_is_actually_collected_or_self_verifying(self):
        checked = 0
        for fname in _iter_test_files(_TESTS_DIR):
            path = os.path.join(_TESTS_DIR, fname)
            with open(path, "r") as handle:
                source = handle.read()
            with self.subTest(file=fname):
                checked += 1
                result = analyze_collection_gap(source, filename=fname)
                if not result["is_safe"]:
                    self.fail(
                        "{fname} declares module-level test_* function(s) {missing} that "
                        "`python -m unittest discover` would NOT collect (TestCase test "
                        "methods found in the file: {found}), and the file does not "
                        "self-execute its checks at import time either. This is the "
                        "'Ran 0 tests / OK' false-positive hole -- wire the function(s) into "
                        "a unittest.TestCase (see test_navrl_target_motion.py) or make the "
                        "module self-run its checks and raise/exit on failure (see "
                        "test_navrl_corridor.py).".format(
                            fname=fname,
                            missing=sorted(result["uncovered"]) or sorted(result["module_funcs"]),
                            found=sorted(result["testcase_methods"]),
                        )
                    )
        self.assertGreater(checked, 0, "no tests/test_*.py files were found under {}".format(_TESTS_DIR))

    def test_no_test_file_imports_pytest_unconditionally_at_module_level(self):
        offenders = []
        for fname in _iter_test_files(_TESTS_DIR):
            path = os.path.join(_TESTS_DIR, fname)
            with open(path, "r") as handle:
                source = handle.read()
            if _imports_pytest_unconditionally(source, filename=fname):
                offenders.append(fname)
        self.assertEqual(
            offenders,
            [],
            "these tests/test_*.py files import pytest unconditionally at module level, "
            "which would make them fail to import in this environment (pytest is not "
            "installed here): {}".format(offenders),
        )


if __name__ == "__main__":
    unittest.main()
