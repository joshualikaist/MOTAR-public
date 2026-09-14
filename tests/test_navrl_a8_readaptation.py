"""CPU-only A8 contracts: training launcher, evaluation runner cell layout, report rules."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "aerial_gym/rl_training/rl_games/train_navrl_v2_ref5in_a8_readapt.sh"
RUNNER = ROOT / "tools/run_navrl_a8_readaptation.py"
BUILDER = ROOT / "tools/build_a8_readaptation_table.py"
A7_PARAMS = {
    "NAVRL_SPEED_GOVERNOR_FIXED_MPS": "2.0", "NAVRL_SPEED_GOVERNOR_FREE_MPS": "3.53553390593",
    "NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M": "0.45", "NAVRL_SPEED_GOVERNOR_MARGIN_M": "0.45",
    "NAVRL_SPEED_GOVERNOR_SLOW_M": "3.0", "NAVRL_SPEED_GOVERNOR_RELEASE_M": "5.0",
    "NAVRL_SPEED_GOVERNOR_TTC_S": "1.0", "NAVRL_SPEED_GOVERNOR_BRAKE_MPS2": "2.0",
    "NAVRL_SPEED_GOVERNOR_REACTION_S": "0.1",
}


def _load(path, name, environ=None):
    with patch.dict(os.environ, environ or {}, clear=False):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


class LauncherContract(unittest.TestCase):
    def test_mode_whitelist_and_pins(self):
        text = LAUNCHER.read_text()
        self.assertIn("off|riskcap|dwa_arc) ;;", text)
        self.assertIn("export MAX_EPOCHS=2900", text)
        self.assertIn("197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e", text)
        for key, value in A7_PARAMS.items():
            self.assertIn(f"export {key}={value}", text)
        self.assertIn("export NAVRL_LEARNING_RATE=1.5e-5", text)
        self.assertIn("export NAVRL_NUM_BARS=70", text)
        self.assertIn('export SEED="${A8_SEED:-197}"', text)
        self.assertIn('-s${SEED}"', text)
        self.assertNotIn("[ref5in-d1]", text)

    def test_bad_mode_refused_before_anything(self):
        for mode in ("", "stopcap", "omni", "riskcap_arc"):
            env = {k: v for k, v in os.environ.items() if not k.startswith("NAVRL_")}
            env["A8_MODE"] = mode
            r = subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, mode)
            self.assertIn("A8_MODE must be", r.stderr)


class RunnerContract(unittest.TestCase):
    def test_cells_are_the_preregistered_ten_in_order(self):
        mod = _load(RUNNER, "a8_runner_test")
        names = [mod.cell_name(p, f) for p, f in mod.CELLS]
        self.assertEqual(names, ["S_off", "S_riskcap", "S_dwa_arc", "T0_off", "T0_riskcap", "T0_dwa_arc",
                                 "T1_riskcap", "T1_off", "T2_dwa_arc", "T2_off"])
        self.assertEqual(mod.SEED, 521)
        self.assertEqual(mod.GOVERNOR_ENV, A7_PARAMS)

    def test_arm_checkpoint_requires_unique_run_and_terminal_epoch(self):
        mod = _load(RUNNER, "a8_runner_test2")
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            with self.assertRaisesRegex(SystemExit, "exactly one run dir"):
                mod.arm_checkpoint("T2", runs)
            nn = runs / "ppo_x_navrl_v2-ref5in-a8-readapt-dwa_arc-s197" / "nn"
            nn.mkdir(parents=True)
            (nn / "last_gen_ppo_ep_2750_rew_1.pth").write_bytes(b"x")
            with self.assertRaisesRegex(SystemExit, "ep2900"):
                mod.arm_checkpoint("T2", runs)
            (nn / "last_gen_ppo_ep_2900_rew_1.pth").write_bytes(b"y")
            self.assertEqual(mod.arm_checkpoint("T2", runs).name, "last_gen_ppo_ep_2900_rew_1.pth")
            (runs / "ppo_y_navrl_v2-ref5in-a8-readapt-dwa_arc-s197").mkdir()
            with self.assertRaisesRegex(SystemExit, "exactly one run dir"):
                mod.arm_checkpoint("T2", runs)

    def test_cell_env_pins_filter_and_parameters(self):
        mod = _load(RUNNER, "a8_runner_test3")
        class Env:
            def evaluation_env(self, *a, **k):
                return {"NAVRL_STAR_CONVEX_SHADOW": "1", "NAVRL_SPEED_GOVERNOR": "off"}
        env = mod.cell_env(Env(), "T2", "dwa_arc", Path("/tmp/a8/T2_dwa_arc"))
        self.assertEqual(env["NAVRL_SPEED_GOVERNOR"], "dwa_arc")
        self.assertEqual(env["NAVRL_STAR_CONVEX_SHADOW"], "0")
        self.assertEqual(env["NAVRL_SEED"], "521")
        self.assertEqual({k: env[k] for k in A7_PARAMS}, A7_PARAMS)


def _rec(crash, capture=0.7, timeout=0.1, mode="off", n=2049, interv=0.05):
    return {"outcome": {"crash_rate": crash, "capture_rate": capture, "timeout_rate": timeout},
            "speed_governor": {"intervention_rate": interv, "samples": 100000},
            "contact_geometry": {"corridor_clearance_below_3m_rate": 0.07, "corridor_clearance_frames": 100000},
            "actual_episodes": n, "condition": {"speed_governor_mode": mode, "bars": 70, "seed": 521},
            "runtime_git_dirty": False, "runtime_source_manifest_sha256": "m" * 64,
            "runtime_git_commit": "c" * 40, "checkpoint_sha256": "s"}


class BuilderRules(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(ROOT / "tools"))
        self.b = _load(BUILDER, "a8_builder_test")

    def _effects(self, **crash):
        e = {m: {} for m in self.b.METRICS}
        for k in self.b.CONTRASTS:
            for m in self.b.METRICS:
                e[m][k] = {"status": "AVAILABLE", "delta_pp": 0.0, "ci95_pp": [-1, 1], "excludes_zero": False}
        for key, (m, d, ex) in crash.items():
            e[m][key] = {"status": "AVAILABLE", "delta_pp": d, "ci95_pp": [d - 1, d + 1], "excludes_zero": ex}
        return e

    def test_q1_rules(self):
        b = self.b
        self.assertEqual(b.q1_label(self._effects(C_arc=("timeout", -3.0, True))), "ADAPTATION_RECOVERS_COST")
        e = self._effects(C_arc=("timeout", -3.0, True)); e["crash"]["C_arc"] = {"status": "AVAILABLE", "delta_pp": 4.0, "ci95_pp": [2, 6], "excludes_zero": True}
        self.assertEqual(b.q1_label(e), "ADAPTATION_COSTS_SAFETY")
        self.assertEqual(b.q1_label(self._effects()), "NO_ADAPTATION_EFFECT")

    def test_q2_q3_rules(self):
        b = self.b
        av = lambda d, ex: {"status": "AVAILABLE", "delta_pp": d, "ci95_pp": [d - 1, d + 1], "excludes_zero": ex}
        self.assertEqual(b.q2_label(av(-6.0, True)), "LAW_GAIN_PERSISTS")
        self.assertEqual(b.q2_label(av(-1.0, False)), "LAW_GAIN_ERASED")
        self.assertEqual(b.q2_label(av(4.0, True)), "LAW_GAIN_REVERSED")
        self.assertEqual(b.q2_label(av(-2.0, True)), "INCONCLUSIVE")
        self.assertEqual(b.q3_label(av(5.0, True)), "FILTER_DEPENDENT")
        self.assertEqual(b.q3_label(av(-5.0, True)), "INTERNALIZED")
        self.assertEqual(b.q3_label(av(2.0, True)), "NEUTRAL")

    def test_end_to_end_root_and_training_checks(self):
        b = self.b
        with tempfile.TemporaryDirectory() as d:
            root, runs = Path(d) / "root", Path(d) / "runs"
            vals = {"S_off": .21, "S_riskcap": .18, "S_dwa_arc": .105, "T0_off": .20, "T0_riskcap": .18,
                    "T0_dwa_arc": .10, "T1_riskcap": .175, "T1_off": .20, "T2_dwa_arc": .10, "T2_off": .26}
            for name, c in vals.items():
                (root / name).mkdir(parents=True)
                (root / name / "70bars.json").write_text(json.dumps(_rec(c, mode=name.split("_", 1)[1])))
            (root / "cells.json").write_text(json.dumps({"checkpoints": {p: {"sha256": "s"} for p in ("S", "T0", "T1", "T2")}}))
            def run(tag, last, n, crash):
                dd = runs / f"ppo_{tag}" / "aerial_run"; dd.mkdir(parents=True, exist_ok=True)
                with (dd / "epoch_metrics.csv").open("w") as f:
                    f.write("epoch,crash_rate\n")
                    for i in range(n):
                        f.write(f"{last - n + 1 + i},{crash}\n")
            run("src", 1900, 1000, .18)
            for mode in ("off", "riskcap", "dwa_arc"):
                run(f"navrl_v2-ref5in-a8-readapt-{mode}-s197", 2900, 1000, .17)
            r = b.build_summary(root, runs, runs / "ppo_src")
            self.assertEqual(r["blockers"], [])
            self.assertEqual(r["labels"]["Q2_law_gain"], "LAW_GAIN_PERSISTS")
            self.assertEqual(r["labels"]["Q3_filter_dependence"]["T2"], "FILTER_DEPENDENT")
            self.assertEqual(r["labels"]["control"], "CONTROL_STABLE")
            self.assertIn("LAW_GAIN_PERSISTS", b.render_markdown(r))
            # a collapsed arm blocks
            run("navrl_v2-ref5in-a8-readapt-dwa_arc-s197", 2900, 1000, .40)
            r = b.build_summary(root, runs, runs / "ppo_src")
            self.assertTrue(any("source + 10 pp" in x for x in r["blockers"]))


if __name__ == "__main__":
    unittest.main()
