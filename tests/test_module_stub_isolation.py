"""No test module may leave a stubbed third-party package behind for the rest of the suite.

This is a regression for a defect that actually shipped. Three CPU-only modules install a fake
`warp` in sys.modules so their subjects' @wp.kernel decorators can evaluate at import, using
sys.modules.setdefault, i.e. only when the name is free. Two of them never removed it, so in a
full-suite run every module imported afterwards got a stub whose Mesh returns None. The
dynamic-mesh raycast gates then errored with AttributeError on a None mesh, and would have been
reported as a broken feature rather than as a broken test fixture.

The rule enforced here is narrow and mechanical: after a module has been IMPORTED, sys.modules
must hold either the genuine package or nothing at all under that name.
"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
STUBBED_PACKAGES = ("warp",)

PROBE = """
import importlib.util, json, sys
sys.path.insert(0, {tests!r})
sys.path.insert(0, {root!r})
before = {{name: (name in sys.modules) for name in {packages!r}}}
spec = importlib.util.spec_from_file_location({module!r}, {path!r})
module = importlib.util.module_from_spec(spec)
sys.modules[{module!r}] = module
spec.loader.exec_module(module)
import os, types
report = {{}}
for name in {packages!r}:
    installed = sys.modules.get(name)
    # Attribute probes cannot tell these apart. The stub overrides __getattr__ to answer EVERY
    # name with a callable, so even __file__ comes back truthy. __spec__ is different: a plain
    # ModuleType has it set to None, and only a module imported from disk carries a spec whose
    # origin is a real path.
    spec = getattr(installed, "__spec__", None) if installed is not None else None
    origin = getattr(spec, "origin", None) if spec is not None else None
    report[name] = {{
        "present": installed is not None,
        "module_type_exact": installed is not None and type(installed) is types.ModuleType,
        "spec_origin_on_disk": bool(origin) and os.path.exists(origin),
        "mesh_is_a_class": isinstance(getattr(installed, "Mesh", None), type),
        "was_present_before": before[name],
    }}
print("PROBE" + json.dumps(report))
"""


def modules_that_stub():
    """Every test module that installs a stub package at import time."""
    found = []
    for path in sorted(Path(ROOT / "tests").glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        if any(f'setdefault("{name}"' in text for name in STUBBED_PACKAGES):
            found.append(path)
    return found


class StubIsolationTest(unittest.TestCase):
    def test_at_least_one_module_still_stubs_so_this_test_is_not_vacuous(self):
        self.assertTrue(modules_that_stub(),
                        "no module stubs any package; delete this regression or update the list")

    def test_importing_a_stubbing_module_leaves_no_stub_behind(self):
        """Import each offender in a fresh interpreter and look at what it left."""
        for path in modules_that_stub():
            with self.subTest(module=path.name):
                script = PROBE.format(tests=str(ROOT / "tests"), root=str(ROOT),
                                      packages=list(STUBBED_PACKAGES),
                                      module=path.stem, path=str(path))
                result = subprocess.run([sys.executable, "-c", script],
                                        capture_output=True, text=True, cwd=str(ROOT))
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                line = [l for l in result.stdout.splitlines() if l.startswith("PROBE")]
                self.assertTrue(line, result.stdout[-2000:])
                import json
                report = json.loads(line[-1][len("PROBE"):])
                for name, state in report.items():
                    if not state["present"]:
                        continue            # removed entirely, which is the cleanest outcome
                    self.assertTrue(
                        state["spec_origin_on_disk"],
                        f"{path.name} left a stub `{name}` in sys.modules: it has no import spec "
                        f"pointing at a file. Every module imported after it in a full-suite run "
                        f"would get the stub instead of the real package. Remove the name once "
                        f"the subject has been executed.")

    def test_the_real_package_is_importable_here(self):
        """If the genuine package were absent the check above could not tell stub from nothing."""
        for name in STUBBED_PACKAGES:
            self.assertIsNotNone(importlib.util.find_spec(name), f"{name} is not installed")


class GateModuleRestoresWarpTest(unittest.TestCase):
    """The dynamic-mesh gates swap the real warp in for themselves; they must put it back."""

    def test_gate_module_restores_whatever_it_found(self):
        script = (
            "import sys, types, unittest\n"
            f"sys.path.insert(0, {str(ROOT / 'tests')!r})\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "stub = types.ModuleType('warp')\n"
            "sys.modules['warp'] = stub\n"
            "import test_dynamic_mesh_raycast as gate\n"
            "runner = unittest.TextTestRunner(stream=open('/dev/null', 'w'))\n"
            "runner.run(unittest.defaultTestLoader.loadTestsFromModule(gate))\n"
            "print('RESTORED' if sys.modules.get('warp') is stub else 'NOT_RESTORED')\n")
        result = subprocess.run([sys.executable, "-c", script],
                                capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("RESTORED", result.stdout,
                      "the gate module did not put back the warp it found: " + result.stdout[-500:])


if __name__ == "__main__":
    unittest.main()
