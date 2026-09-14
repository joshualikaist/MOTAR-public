"""The config-drift guard in set_env_state must not feed non-numeric values to float().

`cfg_search_state` is a mode string ("off"), and it was added to the loop whose body computes
`abs(float(saved) - current)`. Old checkpoints skipped it on the `saved is None` branch, so nothing
failed until the first policy trained after that key existed was evaluated -- and then EVERY such
checkpoint was unloadable with `ValueError: could not convert string to float: 'off'`.

This test states the invariant structurally rather than re-testing one key: every entry of the
float-comparing loop must supply a numeric current value. A `str(...)` call or a string literal
there is the bug, whatever the key is called.
"""

import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "aerial_gym/task/navrl_task/navrl_task.py"


def _uses_float_comparison(node):
    """True if this loop body compares with abs(float(saved) - current)."""
    for inner in ast.walk(node):
        if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                and inner.func.id == "float" and inner.args
                and isinstance(inner.args[0], ast.Name) and inner.args[0].id == "saved"):
            return True
    return False


def _entry_keys_and_values(node):
    """(key, value_node) for each literal tuple in the loop's iterable."""
    out = []
    if not isinstance(node.iter, ast.Tuple):
        return out
    for element in node.iter.elts:
        if isinstance(element, ast.Tuple) and len(element.elts) >= 2:
            key = element.elts[0]
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                out.append((key.value, element.elts[1]))
    return out


class EnvStateGuardTypes(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(SOURCE.read_text())
        self.float_loops = [n for n in ast.walk(self.tree)
                            if isinstance(n, ast.For) and _uses_float_comparison(n)]

    def test_the_float_loop_is_still_there(self):
        self.assertTrue(self.float_loops, "no float-comparing config-drift loop found")

    def test_no_float_compared_entry_supplies_a_string(self):
        """Only a provably numeric expression may sit in the float-comparing loop.

        The first version of this test rejected string literals and str() calls, which is what the
        cfg_search_state incident looked like. On 2026-09-09 the P9 model SHA entered the same loop
        as an attribute access (`self.perception.empirical_error.sha256`) and slipped past, and
        every checkpoint trained after that key existed became unloadable -- the identical failure,
        one expression form later. Allow-listing the numeric forms instead of denying string forms
        removes the whole class rather than one more instance of it.
        """
        numeric_calls = {"float", "int", "len", "abs", "round", "sum", "min", "max"}
        offenders = []
        for loop in self.float_loops:
            for key, value in _entry_keys_and_values(loop):
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                    # bool(...) is the trap here: callable, and not a magnitude.
                    numeric = value.func.id in numeric_calls
                elif isinstance(value, ast.Constant):
                    numeric = isinstance(value.value, (int, float)) and not isinstance(value.value, bool)
                elif isinstance(value, (ast.Attribute, ast.IfExp, ast.JoinedStr)):
                    # An attribute or a conditional carries no type guarantee. The P9 SHA arrived
                    # as `self.perception.empirical_error.sha256` inside an IfExp and made every
                    # checkpoint trained after it unloadable. Wrap it in float() if it truly is one.
                    numeric = False
                else:
                    # Subscripts into the numeric arena dict, arithmetic and plain names are fine.
                    numeric = True
                if not numeric:
                    offenders.append(f"{key} ({type(value).__name__})")
        self.assertEqual(offenders, [],
                         "float() has no guarantee on these entries: " + ", ".join(offenders))

    def test_the_p9_identity_fields_are_not_float_compared(self):
        """The 2026-09-09 regression, pinned by name so it cannot come back quietly."""
        keys = {key for loop in self.float_loops for key, _ in _entry_keys_and_values(loop)}
        self.assertNotIn("cfg_p9_empirical_error_sha256", keys)
        self.assertNotIn("cfg_p9_empirical_error_enabled", keys)
        self.assertIn("cfg_p9_empirical_error_seed", keys, "the seed IS numeric and should stay")

    def test_search_state_is_compared_by_value(self):
        keys = {key for loop in self.float_loops for key, _ in _entry_keys_and_values(loop)}
        self.assertNotIn("cfg_search_state", keys)
        string_loops = [n for n in ast.walk(self.tree)
                        if isinstance(n, ast.For) and not _uses_float_comparison(n)
                        and "cfg_search_state" in {k for k, _ in _entry_keys_and_values(n)}]
        self.assertTrue(string_loops, "cfg_search_state must be compared in a non-float loop")


if __name__ == "__main__":
    unittest.main()
