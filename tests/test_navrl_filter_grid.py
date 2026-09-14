"""The grid runner must refuse anything but governor knobs, keep names unique, and resolve policies."""
import importlib.util, json, os, subprocess, tempfile, unittest
from pathlib import Path
import unittest.mock
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
_S = importlib.util.spec_from_file_location("grid_test", ROOT / "tools/run_navrl_filter_grid.py")
G = importlib.util.module_from_spec(_S); _S.loader.exec_module(G)


class SpecValidation(unittest.TestCase):
    def good(self):
        return {"seed": 523, "cells": [{"name": "a", "policy": "T0", "bars": 70, "env": {"NAVRL_SPEED_GOVERNOR": "stopcap"}}]}

    def test_good_spec_passes(self):
        G.validate_spec(self.good())

    def test_non_governor_env_refused(self):
        s = self.good(); s["cells"][0]["env"]["NAVRL_V2_GOAL_DIST_MIN"] = "6"
        with self.assertRaisesRegex(SystemExit, "only the governor may vary"):
            G.validate_spec(s)

    def test_duplicate_and_bad_names_and_bars(self):
        s = self.good(); s["cells"].append(dict(s["cells"][0]))
        with self.assertRaisesRegex(SystemExit, "duplicate"):
            G.validate_spec(s)
        s = self.good(); s["cells"][0]["name"] = "bad name/with slash"
        with self.assertRaisesRegex(SystemExit, "bad cell name"):
            G.validate_spec(s)
        s = self.good(); s["cells"][0]["bars"] = 5
        with self.assertRaisesRegex(SystemExit, "bars out of range"):
            G.validate_spec(s)

    def test_contract_must_be_known_and_defaults_to_ref5in(self):
        s = self.good(); s["contract"] = "bogus"
        with self.assertRaisesRegex(SystemExit, "spec.contract"):
            G.validate_spec(s)
        G.validate_spec(self.good())          # absent -> ref5in

    def test_v2_contract_is_closed_and_pins_the_ep25000_goal_band(self):
        import os
        with unittest.mock.patch.dict(os.environ, {"NAVRL_SPEED_GOVERNOR": "leaked", "NAVRL_V2_GOAL_DIST_MIN": "99"}):
            env = G._v2_env()
        self.assertNotIn("NAVRL_SPEED_GOVERNOR", env)
        self.assertEqual(env["NAVRL_V2_GOAL_DIST_MIN"], "6")
        self.assertEqual(env["NAVRL_V2_GOAL_DIST_MAX"], "28")
        self.assertEqual(env["NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH"], "0")
        self.assertNotIn("NAVRL_DETECTOR_CHECKPOINT", env)

    def test_v2_cell_env_keeps_the_contract_and_takes_the_knob(self):
        spec = {"seed": 523, "contract": "v2"}
        cell = {"name": "x", "policy": "ep25000", "bars": 130, "env": {"NAVRL_SPEED_GOVERNOR": "stopcap"}}
        env = G.cell_env(None, spec, cell, Path("/tmp/r"), Path("/tmp/r/x"))
        self.assertEqual(env["NAVRL_V2_GOAL_DIST_MIN"], "6")
        self.assertEqual(env["NAVRL_V2_DENSITIES"], "130")
        self.assertEqual(env["NAVRL_SPEED_GOVERNOR"], "stopcap")
        self.assertEqual(env["NAVRL_CONTACT_GEOMETRY"], "1")

    def test_shipped_specs_validate(self):
        for name in ("grid_d1_density_filter_T0.json", "grid_l1_halfwidth_T0.json",
                     "grid_d3_lineage_ep25000.json", "grid_d1p_density_filter_ep25000.json"):
            spec = G.validate_spec(json.loads((ROOT / "docs/specs" / name).read_text()))
            self.assertIn(spec.get("contract", "ref5in"), G.CONTRACTS, name)
            self.assertGreaterEqual(len(spec["cells"]), 4, name)
        d1 = json.loads((ROOT / "docs/specs/grid_d1_density_filter_T0.json").read_text())
        self.assertEqual(len(d1["cells"]), 20)
        self.assertEqual(sorted({c["bars"] for c in d1["cells"]}), [70, 100, 130, 160, 205])
        d1p = json.loads((ROOT / "docs/specs/grid_d1p_density_filter_ep25000.json").read_text())
        self.assertEqual(d1p["contract"], "v2")
        self.assertEqual(len(d1p["cells"]), 20)
        self.assertEqual({c["policy"] for c in d1p["cells"]}, {"ep25000"})
        d3 = json.loads((ROOT / "docs/specs/grid_d3_lineage_ep25000.json").read_text())
        self.assertEqual(d3["contract"], "v2")


class CellEnv(unittest.TestCase):
    def test_cell_env_pins_density_records_and_overrides_only_the_knob(self):
        class Env:
            def evaluation_env(self, *a, **k):
                return {"NAVRL_V2_DENSITIES": "70", "NAVRL_SPEED_GOVERNOR": "off", "NAVRL_STAR_CONVEX_SHADOW": "1"}
        spec = {"seed": 523, "frame_sample_every": 50}
        cell = {"name": "x", "policy": "T0", "bars": 205, "env": {"NAVRL_SPEED_GOVERNOR": "stopcap", "NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M": "0.8"}}
        env = G.cell_env(Env(), spec, cell, Path("/tmp/r"), Path("/tmp/r/x"))
        self.assertEqual(env["NAVRL_V2_DENSITIES"], "205")
        self.assertEqual(env["NAVRL_SPEED_GOVERNOR"], "stopcap")
        self.assertEqual(env["NAVRL_SPEED_GOVERNOR_HALF_WIDTH_M"], "0.8")
        self.assertEqual(env["NAVRL_SPEED_GOVERNOR_BRAKE_MPS2"], "2.0")
        self.assertEqual(env["NAVRL_CG_FRAME_SAMPLE_EVERY"], "50")
        self.assertEqual(env["NAVRL_CONTACT_GEOMETRY"], "1")
        self.assertEqual(env["NAVRL_STAR_CONVEX_SHADOW"], "0")


class SourceFreeze(unittest.TestCase):
    """The freeze must track the EVALUATED source, not the repository as a whole.

    An earlier version compared HEAD and VOIDed a 20-cell grid because a paper draft was
    committed while cells were running -- a commit that cannot reach the simulator.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        for path in ("aerial_gym", "tools", "resources/robots", "docs"):
            (self.repo / path).mkdir(parents=True)
            (self.repo / path / "f.txt").write_text("v1\n")
        self.git("init", "-q")
        self.commit("initial")

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              text=True, stdout=subprocess.PIPE).stdout

    def commit(self, message):
        self.git("add", "-A")
        self.git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", message)

    def test_unrelated_commit_does_not_move_the_fingerprint(self):
        with patch.object(G, "REPO", self.repo):
            before = G._source_fingerprint()
            (self.repo / "docs/f.txt").write_text("a draft revision\n")
            self.commit("docs only")
            after = G._source_fingerprint()
            self.assertEqual(before, after)
            G._frozen(before)          # must not raise

    def test_evaluated_source_commit_moves_it(self):
        with patch.object(G, "REPO", self.repo):
            before = G._source_fingerprint()
            (self.repo / "aerial_gym/f.txt").write_text("v2\n")
            self.commit("runtime change")
            self.assertNotEqual(G._source_fingerprint(), before)
            with self.assertRaisesRegex(SystemExit, "evaluated source changed"):
                G._frozen(before)

    def test_dirty_evaluated_source_is_refused(self):
        with patch.object(G, "REPO", self.repo):
            (self.repo / "tools/f.txt").write_text("uncommitted\n")
            with self.assertRaisesRegex(SystemExit, "must be committed"):
                G._frozen()

    def test_dirty_docs_is_allowed(self):
        with patch.object(G, "REPO", self.repo):
            (self.repo / "docs/f.txt").write_text("uncommitted draft\n")
            G._frozen()                # must not raise


if __name__ == "__main__":
    unittest.main()
