"""CPU-only A7 contract tests; synthetic JSON never invokes an evaluator."""

import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from tools import build_a7_arc_attribution_table as a7


def record(arm, crash=0.1877, seed=509, bars=70, episodes=2051):
    return {
        "actual_episodes": episodes,
        "checkpoint_sha256": ("a" if bars == 70 else "b") * 64,
        "runtime_git_commit": "c" * 40,
        "runtime_git_dirty": False,
        "runtime_source_manifest_sha256": "d" * 64,
        "condition": dict(a7.PARAMETERS, speed_governor_mode=arm, bars=bars, seed=seed,
                          num_envs=128, action_selection="deterministic", reflection_mode="original",
                          distractor_count=0, robot_name="ref5in" if bars == 70 else "quad",
                          goal_dist_min_m=22.5 if bars == 70 else 6.0),
        "outcome": {"crash_rate": crash, "capture_rate": 0.8 - crash, "timeout_rate": 0.2},
        "speed_governor": {"intervention_rate": 0.05, "samples": 100000},
        "contact_geometry": {"corridor_clearance_below_3m_rate": 0.07, "corridor_clearance_frames": 90000},
    }


def effect(delta, significant=False):
    half = abs(delta) / 2 if significant else abs(delta) + 1
    return dict(status="AVAILABLE", delta_pp=delta, ci95_pp=[delta - half, delta + half],
                excludes_zero=significant)


class A7DecisionTests(unittest.TestCase):
    def test_wald_matches_a5_and_uses_actual_episode_denominators(self):
        first, second = record("riskcap", .2, episodes=2051), record("stopcap", .1, episodes=2049)
        got = a7.contrast(first, second)
        expected = a7._delta_ci(first, second, "crash_rate")
        self.assertEqual((got["delta_pp"], *got["ci95_pp"]), expected)
        self.assertEqual((got["n_a"], got["n_b"]), (2051, 2049))

    def test_frame_metrics_use_recorded_sample_counts(self):
        first, second = record("dwa_arc"), record("stopcap")
        first["speed_governor"]["intervention_rate"] = .04
        second["speed_governor"]["intervention_rate"] = .19
        got = a7.contrast(first, second, "intervention")
        self.assertEqual(got["n_a"], 100000)
        expected_half = 196 * math.sqrt(.04 * .96 / 100000 + .19 * .81 / 100000)
        self.assertAlmostEqual(got["ci95_pp"][1] - got["delta_pp"], expected_half)
        self.assertEqual(a7.contrast(first, second, "below_3m")["n_a"], 90000)

    def test_missing_contact_geometry_is_unavailable(self):
        old = record("riskcap")
        del old["contact_geometry"]
        self.assertIsNone(a7.metric(old, "below_3m"))
        self.assertEqual(a7.contrast(old, record("stopcap"), "below_3m")["status"], "UNAVAILABLE")

    def test_law_safety_label(self):
        effects = dict(G_stop=effect(-1), G_risk=effect(-.2), L_line=effect(-6, True), L_arc=effect(-7, True))
        self.assertEqual(a7.classify(effects)["label"], "LAW_CARRIES_SAFETY")

    def test_geometry_safety_requires_both_three_pp_and_significance(self):
        effects = dict(G_stop=effect(-3, True), G_risk=effect(-4, True), L_line=effect(0), L_arc=effect(0))
        self.assertEqual(a7.classify(effects)["label"], "ARC_CARRIES_SAFETY")
        effects["G_risk"] = effect(-2.9, True)
        self.assertEqual(a7.classify(effects)["label"], "INCONCLUSIVE")

    def test_overlap_is_resolved_by_the_frozen_precedence(self):
        # Prereg §4 revision 1: the only co-satisfiable pair, the specific label wins, both recorded.
        effects = dict(G_stop=effect(-1), G_risk=effect(4, True), L_line=effect(-6, True), L_arc=effect(-10, True))
        got = a7.classify(effects)
        self.assertEqual(got["label"], "ARC_HURTS_UNDER_RISKCAP")
        self.assertEqual(got["status"], "CLASSIFIED")
        self.assertEqual(got["matched_labels"], ["INTERACTION", "ARC_HURTS_UNDER_RISKCAP"])

    def test_zero_in_ci_is_not_significant(self):
        first, second = record("riskcap", .1), record("stopcap", .1)
        self.assertFalse(a7.contrast(first, second)["excludes_zero"])

    def test_brake_significant_small_negative_is_unclassified(self):
        self.assertEqual(a7.brake_label(effect(-2, True)), "INCONCLUSIVE")
        self.assertEqual(a7.brake_label(effect(-3, True)), "FLIP_IS_BRAKE")
        self.assertEqual(a7.brake_label(effect(-2)), "FLIP_IS_DENSITY")
        self.assertEqual(a7.brake_label(effect(0)), "FLIP_IS_DENSITY")

    def test_m5_pass_gap_and_failure(self):
        old = record("riskcap", .1877)
        self.assertEqual(a7._m5(record("riskcap", .18771), old, 18.77, 2049)["status"], "PASS")
        for rate in (.188, .1927):
            self.assertEqual(a7._m5(record("riskcap", rate), old, 18.77, 2049)["status"], "PASS_INEXACT")
        self.assertEqual(a7._m5(record("riskcap", .193), old, 18.77, 2049)["status"], "FAIL")
        self.assertFalse(a7._m5(old, old, 18.77, 2049)["historical_episode_count_matches_prereg"])

    def test_replication_obeys_label_and_sign_rules(self):
        def part(label, line, geometry):
            return {"classification": {"label": label},
                    "effects": {"crash": {"L_line": effect(line), "G_stop": effect(geometry)}}}
        first = part("LAW_CARRIES_SAFETY", -6, -1)
        self.assertEqual(a7.replication(first, part("LAW_CARRIES_SAFETY", -5, 1)), "REPLICATED")
        self.assertEqual(a7.replication(first, part("INCONCLUSIVE", -2, -1)), "PARTIAL")
        self.assertEqual(a7.replication(first, part("INCONCLUSIVE", -2, 1)), "NOT_REPLICATED")
        self.assertEqual(a7.replication(first, part(None, -2, 1)), "UNAVAILABLE")


class A7ResultContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = {name: self.root / name for name in a7.DEFAULT_ROOTS}
        for name, seed, bars, arms in (
            ("A4", 509, 70, ("riskcap", "stopcap", "dwa_arc")),
            ("P1", 509, 70, ("riskcap", "riskcap_arc")),
            ("P2", 491, 70, a7.ARMS), ("P3", 49, 205, a7.ARMS),
            ("SCREEN", 49, 205, ("riskcap", "stopcap")),
        ):
            for arm in arms:
                crash = {"riskcap": .1877, "riskcap_arc": .18, "stopcap": .123, "dwa_arc": .11}[arm]
                if bars == 205:
                    crash = {"riskcap": .1595, "riskcap_arc": .16, "stopcap": .18, "dwa_arc": .17}[arm]
                data = record(arm, crash, seed, bars)
                if name == "SCREEN":
                    data["runtime_git_dirty"] = True
                    data["runtime_git_commit"] = "e" * 40
                    data["condition"]["speed_governor_brake_mps2"] = 2.9608856678
                    data["condition"]["speed_governor_ttc_s"] = 1.2
                    del data["contact_geometry"]
                if name == "A4":
                    data["runtime_git_commit"] = "f" * 40
                if arm == "stopcap":
                    data["outcome"]["timeout_rate"] = .3
                    data["speed_governor"]["intervention_rate"] = .19
                if arm == "dwa_arc":
                    data["speed_governor"]["intervention_rate"] = .04
                self.write(name, arm, data)

    def write(self, name, arm, data):
        path = self.paths[name] / arm / (str(data["condition"]["bars"]) + "bars.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    def mutate(self, name, arm, update):
        path = next((self.paths[name] / arm).glob("*bars.json"))
        data = json.loads(path.read_text())
        update(data)
        path.write_text(json.dumps(data))

    def test_complete_report_and_historical_limitations(self):
        report = a7.build_summary(self.paths)
        self.assertTrue(report["result_checks_passed"], report["blockers"])
        self.assertEqual(report["parts"]["P1"]["classification"]["label"], "LAW_CARRIES_SAFETY")
        self.assertEqual(report["replication"], "REPLICATED")
        self.assertTrue(report["density_limited"])
        self.assertTrue(report["parts"]["P1"]["cost_labels"]["ARC_REMOVES_COST"])
        self.assertEqual(report["integrity"]["historical_SCREEN"]["status"], "FAIL")
        self.assertEqual(report["integrity"]["M1"]["status"], "EXTERNAL_CPU_TEST_REQUIRED")
        self.assertEqual(report["integrity"]["M2"]["status"], "EXTERNAL_CPU_TEST_REQUIRED")
        self.assertIn("robot_name", report["condition_differences_P1_P3"])
        self.assertIsNone(report["sources"]["SCREEN"]["riskcap"]["metrics"]["below_3m"])
        self.assertIn("temporally dependent", " ".join(report["caveats"]))
        markdown = a7.render_markdown(report)
        self.assertIn("LAW_CARRIES_SAFETY", markdown)
        self.assertIn("unavailable", markdown)

    def test_mixed_p1_line_law_uses_historical_riskcap(self):
        self.mutate("P1", "riskcap", lambda d: d["outcome"].update(crash_rate=.18771))
        report = a7.build_summary(self.paths)
        effect = report["parts"]["P1"]["effects"]["crash"]["L_line"]
        self.assertAlmostEqual(effect["delta_pp"], (.123 - .1877) * 100)
        self.assertIn("A4", effect["b"])

    def test_m3_dirty_voids_root_and_suppresses_labels(self):
        self.mutate("P2", "dwa_arc", lambda d: d.update(runtime_git_dirty=True))
        report = a7.build_summary(self.paths)
        self.assertFalse(report["result_checks_passed"])
        self.assertEqual(report["parts"]["P2"]["root_disposition"], "VOID")
        self.assertIsNone(report["parts"]["P2"]["classification"]["label"])
        self.assertIsNone(report["parts"]["P2"]["cost_labels"]["ARC_REMOVES_COST"])

    def test_m3_manifest_mismatch_voids_root(self):
        self.mutate("P1", "riskcap_arc", lambda d: d.update(runtime_source_manifest_sha256="9" * 64))
        report = a7.build_summary(self.paths)
        self.assertEqual(report["parts"]["P1"]["root_disposition"], "VOID")

    def test_m4_cross_root_commit_change_voids_all_a7(self):
        for arm in a7.ARMS:
            self.mutate("P3", arm, lambda d: d.update(runtime_git_commit="8" * 40))
        report = a7.build_summary(self.paths)
        self.assertEqual(report["integrity"]["M3"]["P3"]["status"], "PASS")
        self.assertEqual(report["integrity"]["M4"]["status"], "FAIL")
        self.assertTrue(all(p["root_disposition"] == "VOID" for p in report["parts"].values()))

    def test_missing_provenance_cannot_pass(self):
        self.mutate("P2", "riskcap", lambda d: d.update(runtime_git_dirty=None, runtime_git_commit=None))
        report = a7.build_summary(self.paths)
        self.assertFalse(report["result_checks_passed"])

    def test_m6_requires_exact_values_and_numeric_literals(self):
        path = next((self.paths["P2"] / "riskcap").glob("*bars.json"))
        path.write_text(path.read_text().replace('"speed_governor_brake_mps2": 2.0', '"speed_governor_brake_mps2": 2.00'))
        report = a7.build_summary(self.paths)
        self.assertEqual(report["integrity"]["M6"]["status"], "FAIL")
        self.assertFalse(report["result_checks_passed"])

    def test_m5_small_gap_is_inexact_but_admissible_and_large_gap_blocks(self):
        self.mutate("P1", "riskcap", lambda d: d["outcome"].update(crash_rate=.19))
        report = a7.build_summary(self.paths, check_p1=True)
        self.assertEqual(report["integrity"]["M5"]["P1"]["status"], "PASS_INEXACT")
        self.assertTrue(report["result_checks_passed"], report["blockers"])
        self.mutate("P1", "riskcap", lambda d: d["outcome"].update(crash_rate=.21))
        report = a7.build_summary(self.paths, check_p1=True)
        self.assertEqual(report["integrity"]["M5"]["P1"]["status"], "FAIL")
        self.assertFalse(report["result_checks_passed"])

    def test_full_p1_fallback_does_not_import_a4_controls(self):
        self.write("P1", "stopcap", record("stopcap", .13))
        self.write("P1", "dwa_arc", record("dwa_arc", .12))
        self.mutate("P1", "riskcap", lambda d: d["outcome"].update(crash_rate=.20))
        report = a7.build_summary(self.paths, check_p1=True)
        self.assertTrue(report["result_checks_passed"], report["blockers"])
        self.assertEqual(report["integrity"]["M5"]["P1"]["status"], "FAIL")
        self.assertFalse(report["integrity"]["M5"]["P1"]["required_for_P1_factorial"])
        self.assertEqual(report["parts"]["P1"]["design"], "within_root_four_arm")
        self.assertIn("P1", report["parts"]["P1"]["effects"]["crash"]["L_line"]["b"])

    def test_partial_fallback_is_rejected(self):
        self.write("P1", "stopcap", record("stopcap", .13))
        report = a7.build_summary(self.paths, check_p1=True)
        self.assertFalse(report["result_checks_passed"])

    def test_wrong_checkpoint_or_condition_blocks_inference(self):
        self.mutate("P3", "riskcap", lambda d: d["condition"].update(goal_dist_min_m=22.5))
        report = a7.build_summary(self.paths)
        self.assertEqual(report["integrity"]["conditions"]["P3"]["status"], "FAIL")
        self.assertEqual(report["brake_comparison"]["label"], "BLOCKED")

    def test_corrupt_json_and_invalid_rates_are_reported(self):
        path = next((self.paths["P2"] / "dwa_arc").glob("*bars.json"))
        path.write_text("not json")
        self.mutate("P3", "riskcap", lambda d: d["outcome"].update(crash_rate=1.2))
        report = a7.build_summary(self.paths)
        self.assertFalse(report["result_checks_passed"])
        self.assertEqual(report["integrity"]["M3"]["P2"]["status"], "FAIL")
        self.assertEqual(report["integrity"]["M3"]["P3"]["status"], "FAIL")

    def test_check_p1_ignores_other_parts_and_writes_nothing(self):
        missing = dict(self.paths, P2=self.root / "absent2", P3=self.root / "absent3", SCREEN=self.root / "absent_screen")
        args = ["--check-p1"]
        for name, path in missing.items():
            args += ["--" + name.lower() + "-root", str(path)]
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            code = a7.main(args)
        report = json.loads(captured.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(set(report["parts"]), {"P1"})
        self.assertFalse((self.root / "absent2").exists())

    def test_cli_emits_reviewable_blocked_artifacts_and_nonzero_status(self):
        self.mutate("P1", "riskcap", lambda d: d["outcome"].update(crash_rate=.21))
        json_path, md_path = self.root / "summary.json", self.root / "summary.md"
        args = ["--json-output", str(json_path), "--markdown-output", str(md_path)]
        for name, path in self.paths.items():
            args += ["--" + name.lower() + "-root", str(path)]
        with contextlib.redirect_stdout(io.StringIO()):
            code = a7.main(args)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(json_path.read_text())["status"], "BLOCKED")
        self.assertIn("BLOCKED", md_path.read_text())


if __name__ == "__main__":
    unittest.main()
