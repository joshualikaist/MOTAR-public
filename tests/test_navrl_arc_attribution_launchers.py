"""CPU-only A7 launcher contracts; shell evaluations use an isolated stub evaluator."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CONTACT_LAUNCHER = ROOT / "tools/run_navrl_contact_geometry.py"
P3_LAUNCHER = ROOT / "aerial_gym/rl_training/rl_games/eval_navrl_v2_ep25000_arc_attribution.sh"
CHECKPOINT_SHA = "f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"


def load_contact(environ=None):
    with patch.dict(os.environ, environ or {}, clear=True):
        spec = importlib.util.spec_from_file_location("contact_a7_test", CONTACT_LAUNCHER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class ContactSelectionTest(unittest.TestCase):
    def test_historical_defaults_preserved(self):
        self.assertEqual(tuple(load_contact().ARMS), ("off", "riskcap"))
        self.assertEqual(tuple(load_contact({"NAVRL_CG_ALL_ARMS": "1"}).ARMS),
                         ("off", "fixed2p0", "riskcap", "stopcap", "omni", "dwa_arc"))

    def test_explicit_subset_has_priority_and_preserves_order(self):
        mod = load_contact({"NAVRL_CG_ALL_ARMS": "1",
                            "NAVRL_CG_ARMS": "riskcap_arc, riskcap"})
        self.assertEqual(tuple(mod.ARMS), ("riskcap_arc", "riskcap"))
        self.assertEqual(mod.ARMS["riskcap_arc"]["NAVRL_SPEED_GOVERNOR"], "riskcap_arc")

    def test_invalid_subsets_refused(self):
        for value in ("", "riskcap,", "unknown", "riskcap,riskcap", "../off"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                load_contact({"NAVRL_CG_ARMS": value})

    def test_result_root_relative_to_repo_or_absolute(self):
        self.assertEqual(load_contact({"NAVRL_CG_RESULT_ROOT": "results/a7"}).RESULT_ROOT,
                         ROOT / "results/a7")
        self.assertEqual(load_contact({"NAVRL_CG_RESULT_ROOT": "/tmp/a7"}).RESULT_ROOT,
                         Path("/tmp/a7"))

    def test_a7_pins_governor_and_disables_ambient_shadow(self):
        mod = load_contact({"NAVRL_CG_ARMS": "riskcap,riskcap_arc", "NAVRL_CG_SEED": "509"})
        envelope = SimpleNamespace(evaluation_env=lambda *a, **kw: {})
        with patch.dict(os.environ, {"NAVRL_STAR_CONVEX_SHADOW": "1"}):
            first = mod._arm_env(envelope, "riskcap", Path("/tmp/a7/riskcap"))
            second = mod._arm_env(envelope, "riskcap_arc", Path("/tmp/a7/riskcap_arc"))
        self.assertEqual(first["NAVRL_STAR_CONVEX_SHADOW"], "0")
        self.assertEqual(first["NAVRL_CONTACT_GEOMETRY"], "1")
        self.assertEqual(first["NAVRL_SEED"], "509")
        self.assertEqual(first["NAVRL_SPEED_GOVERNOR_BRAKE_MPS2"], "2.0")
        self.assertEqual(first["NAVRL_SPEED_GOVERNOR_TTC_S"], "1.0")
        changed = {k for k in first if first[k] != second[k]}
        self.assertEqual(changed, {"NAVRL_SPEED_GOVERNOR", "NAVRL_V2_RESULT_DIR"})

    def test_unselected_arm_and_partial_a7_root_refused_before_prerequisites(self):
        mod = load_contact({"NAVRL_CG_ARMS": "riskcap,riskcap_arc"})
        with self.assertRaisesRegex(SystemExit, "not selected"):
            mod.run_evaluate(None, only="stopcap")
        with tempfile.TemporaryDirectory() as directory, patch.object(mod, "RESULT_ROOT", Path(directory)):
            with self.assertRaisesRegex(SystemExit, "no partial resume"):
                mod.run_evaluate(None)

    def test_unregistered_a1_arms_do_not_crash_or_pass_vacuously(self):
        mod = load_contact({"NAVRL_CG_ARMS": "riskcap_arc", "NAVRL_CG_SEED": "509"})
        cm = dict(vertical_out_rate=0.0, behind_rate=0.0, lateral_rate=0.4,
                  no_return_rate=0.4, in_corridor_rate=0.2)
        row = {"arm": "riskcap_arc", "outcome": {"crash_rate": 0.2},
               "contact_geometry": {"contacts": 10, "commanded_direction": cm,
                   "actual_velocity_direction": {"in_corridor_rate": 0.2},
                   "hypothesis_c_reclassified_into_corridor": 0, "mean_cmd_vs_actual_deg": 5.0,
                   "hypothesis_d_rate": 0.1, "mean_bars_in_corridor": 1.0}}
        with tempfile.TemporaryDirectory() as directory, patch.object(mod, "RESULT_ROOT", Path(directory)), \
                patch.object(mod, "_load_arm", return_value=row), patch("builtins.print"):
            mod.run_summarize(SimpleNamespace(CHECKPOINT_SHA="test"))
            summary = json.loads((Path(directory) / "summary.json").read_text())
        self.assertIsNone(summary["verdict_replication"])


class P3LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.rl = self.repo / "aerial_gym/rl_training/rl_games"
        self.rl.mkdir(parents=True)
        self.capture = self.repo / "captured.jsonl"
        self.source = self.repo / "fixture.pth"
        self.source.write_bytes(b"A7 CPU launcher checkpoint fixture")
        self.launcher = self.rl / P3_LAUNCHER.name
        original = P3_LAUNCHER.read_text()
        self.assertIn(CHECKPOINT_SHA, original)
        # Only the fixture's digest changes. The complete production shell flow is exercised.
        self.launcher.write_text(original.replace(CHECKPOINT_SHA,
                                hashlib.sha256(self.source.read_bytes()).hexdigest()))
        self.evaluator = self.rl / "eval_navrl_v2_density_sweep.sh"
        self.evaluator.write_text('''#!/usr/bin/env bash
set -euo pipefail
"${PYTHON}" - "$@" <<'PY'
import json, os, sys
from pathlib import Path
with Path(os.environ["A7_TEST_CAPTURE_PATH"]).open("a") as stream:
    stream.write(json.dumps({"env": dict(os.environ), "args": sys.argv[1:]}) + "\\n")
PY
''')
        self.git("init", "-q")
        self.git("add", "aerial_gym")
        self.git("-c", "user.name=A7 test", "-c", "user.email=a7@example.invalid",
                 "commit", "-qm", "fixture sources")

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def run_launcher(self, **overrides):
        env = dict(os.environ)
        for key in tuple(env):
            if key.startswith("NAVRL_"):
                del env[key]
        env.update(PYTHON=sys.executable, A7_TEST_CAPTURE_PATH=str(self.capture))
        env.update(overrides)
        return subprocess.run(["bash", str(self.launcher), str(self.source)], env=env,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_four_arms_share_pinned_contract_despite_contaminated_environment(self):
        result = self.run_launcher(NAVRL_V2_FORCE="1", NAVRL_CG_ARMS="off", GPU4GB="1",
                                  NAVRL_DISTRACTOR_COUNT="5", NAVRL_STAR_CONVEX_SHADOW="1",
                                  NAVRL_V2_FIXED_TARGET_SPEED="0.9", NAVRL_DETECTOR_CHECKPOINT="bad",
                                  NAVRL_V2_EVAL_CONTRACT="corrected_nonoverlap_physical_off",
                                  NAVRL_SPEED_GOVERNOR_BRAKE_MPS2="9", NAVRL_V2_GOAL_DIST_MIN="22.5")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in self.capture.read_text().splitlines()]
        self.assertEqual([r["env"]["NAVRL_SPEED_GOVERNOR"] for r in rows],
                         ["riskcap", "stopcap", "dwa_arc", "riskcap_arc"])
        pinned = {"NAVRL_V2_FORCE": "0", "GPU4GB": "0", "NUM_ENVS": "128",
                  "NAVRL_SEED": "49", "NAVRL_V2_DENSITIES": "205", "NAVRL_DISTRACTOR_COUNT": "0",
                  "NAVRL_CONTACT_GEOMETRY": "1", "NAVRL_STAR_CONVEX_SHADOW": "0",
                  "NAVRL_SPEED_GOVERNOR_BRAKE_MPS2": "2.0", "NAVRL_SPEED_GOVERNOR_TTC_S": "1.0",
                  "NAVRL_V2_GOAL_DIST_MIN": "6", "NAVRL_V2_GOAL_DIST_MAX": "28",
                  "NAVRL_V2_ACTION_MODE": "deterministic", "NAVRL_EVAL_REFLECTION_MODE": "original"}
        for row in rows:
            self.assertEqual(row["args"], [str(self.source), "2049"])
            self.assertEqual({k: row["env"][k] for k in pinned}, pinned)
            for key in ("NAVRL_V2_FIXED_TARGET_SPEED", "NAVRL_DETECTOR_CHECKPOINT",
                        "NAVRL_V2_EVAL_CONTRACT", "NAVRL_CG_ARMS"):
                self.assertNotIn(key, row["env"])
        self.assertEqual(len({r["env"]["NAVRL_V2_SHARED_SOURCE_BUNDLE"] for r in rows}), 1)
        self.assertEqual(len({r["env"]["NAVRL_V2_RESULT_DIR"] for r in rows}), 4)

    def test_dirty_source_is_refused_before_any_evaluation(self):
        self.evaluator.write_text(self.evaluator.read_text() + "# uncommitted change\n")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be committed", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_commit_change_after_first_arm_voids_root_before_second_arm(self):
        self.evaluator.write_text(self.evaluator.read_text() + '''
git -C "${PYTHONPATH}" -c user.name="A7 test" -c user.email=a7@example.invalid \\
    commit --allow-empty -qm "changed HEAD during evaluation"
''')
        self.git("add", "aerial_gym")
        self.git("-c", "user.name=A7 test", "-c", "user.email=a7@example.invalid",
                 "commit", "-qm", "fixture that changes HEAD")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("git HEAD changed", result.stderr)
        self.assertEqual(len(self.capture.read_text().splitlines()), 1)

    def test_existing_root_is_refused(self):
        (self.repo / "results/navrl_arc_attribution_205bars_seed49").mkdir(parents=True)
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to overwrite", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_checkpoint_sha_mismatch_is_refused(self):
        self.source.write_bytes(b"wrong checkpoint")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source SHA mismatch", result.stderr)
        self.assertFalse(self.capture.exists())


class BuilderRulesTest(unittest.TestCase):
    def _builder(self):
        tools = str(ROOT / "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)  # the builder's script-style fallback import of build_a5
        spec = importlib.util.spec_from_file_location(
            "build_a7_test", ROOT / "tools/build_a7_arc_attribution_table.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _effect(self, delta, excludes):
        return dict(status="AVAILABLE", delta_pp=delta, excludes_zero=excludes)

    def test_arc_hurts_takes_precedence_over_interaction(self):
        b = self._builder()
        effects = {"G_stop": self._effect(-1.0, False), "G_risk": self._effect(4.0, True),
                   "L_line": self._effect(-6.0, True), "L_arc": self._effect(-11.0, True)}
        decision = b.classify(effects)
        self.assertEqual(decision["label"], "ARC_HURTS_UNDER_RISKCAP")
        self.assertEqual(set(decision["matched_labels"]), {"ARC_HURTS_UNDER_RISKCAP", "INTERACTION"})

    def test_law_carries_safety_and_inconclusive(self):
        b = self._builder()
        effects = {"G_stop": self._effect(-1.2, False), "G_risk": self._effect(0.5, False),
                   "L_line": self._effect(-6.5, True), "L_arc": self._effect(-7.0, True)}
        self.assertEqual(b.classify(effects)["label"], "LAW_CARRIES_SAFETY")
        effects["L_arc"] = self._effect(-2.0, False)
        self.assertEqual(b.classify(effects)["label"], "INCONCLUSIVE")

    def test_m5_exact_inexact_and_fail(self):
        b = self._builder()
        def rec(crash, n):
            return {"outcome": {"crash_rate": crash}, "actual_episodes": n}
        ref = rec(385 / 2051, 2051)
        self.assertEqual(b._m5(rec(385 / 2051, 2051), ref, 18.77, 2051)["status"], "PASS")
        self.assertEqual(b._m5(rec(390 / 2051, 2051), ref, 18.77, 2051)["status"], "PASS_INEXACT")
        self.assertEqual(b._m5(rec(420 / 2051, 2051), ref, 18.77, 2051)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
