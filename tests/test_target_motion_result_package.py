"""The canonical target-motion result package must stay honest about its own limits.

Three failure modes are guarded here, each of which actually occurred in the
preliminary reporting of this result and had to be withdrawn:

1. A post-hoc decision threshold being described as preregistered.
2. A conditional diagnostic being promoted into the slot of an unavailable
   preregistered primary metric.
3. An n=3 interval that excludes zero being narrated as statistical significance.

The package also has to keep saying that raw artifacts are immutable and that the
episode overshoot is bounded rather than ignored.
"""
import json
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "results/target_motion_e0_e2_2026-09-18"
RAW = ROOT / "results/target_motion_e0_e2_evaluation_2026-09-18"
SUMMARY = PKG / "canonical_summary.json"
COUNTS = PKG / "episode_count_audit.json"
PRIMARY = PKG / "primary_2048_manifest.json"
DOC = ROOT / "docs/results/target_motion_generalization_2026-09-19.md"
DEVIATIONS = ROOT / "docs/audits/target_motion_protocol_deviations_2026-09-19.md"

PREREG_PER_CELL = 2048
PREREG_CELLS = 48


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


class PackageExists(unittest.TestCase):
    def test_every_canonical_artifact_is_present(self):
        for p in (SUMMARY, COUNTS, PRIMARY, PKG / "seed_effects.csv",
                  PKG / "statistical_sensitivity.json", PKG / "RAW_MANIFEST.json",
                  DOC, DEVIATIONS):
            self.assertTrue(p.exists(), f"missing {p.relative_to(ROOT)}")


class RawStaysImmutable(unittest.TestCase):
    def test_raw_manifest_copy_matches_the_frozen_original(self):
        a = (PKG / "RAW_MANIFEST.json").read_bytes()
        b = (RAW / "RAW_MANIFEST.json").read_bytes()
        self.assertEqual(a, b, "the packaged manifest diverged from the frozen raw manifest")

    def test_all_48_cells_are_recorded(self):
        counts = load(COUNTS)
        self.assertEqual(PREREG_CELLS, len(counts["cells"]))
        self.assertEqual(PREREG_CELLS * PREREG_PER_CELL, counts["preregistered_total"])

    def test_no_cell_is_short_of_the_preregistered_count(self):
        for row in load(COUNTS)["cells"]:
            self.assertGreaterEqual(
                row["raw_episode_records"], PREREG_PER_CELL,
                f"{row['cell_id']} has fewer than {PREREG_PER_CELL} records")

    def test_every_cell_is_internally_consistent(self):
        """Outcome triple must equal the record count, or records are double counted."""
        for row in load(COUNTS)["cells"]:
            self.assertTrue(row["internally_consistent"],
                            f"{row['cell_id']}: outcome triple != actual_episodes")


class ExcessEpisodesAreBoundedNotIgnored(unittest.TestCase):
    def test_the_overshoot_is_explained_with_a_named_cause(self):
        counts = load(COUNTS)
        self.assertEqual(counts["raw_total"] - counts["preregistered_total"],
                         counts["total_excess"])
        self.assertTrue(counts["cause"], "the excess has no recorded cause")
        self.assertIn("source", counts["cause_evidence"])
        for ruled_out in ("canary_contamination", "duplicate_terminal_record",
                          "resume_retry_duplication"):
            self.assertIn(ruled_out, counts["causes_ruled_out"])

    def test_worst_case_influence_is_quantified(self):
        sens = load(PKG / "statistical_sensitivity.json")
        per_arm = sens["excess_episode_bounds"]["per_arm"]
        self.assertEqual(4, len(per_arm))
        for arm, b in per_arm.items():
            self.assertIn("max_abs_shift_pp", b, f"{arm}: no worst-case bound")
            self.assertGreaterEqual(b["max_abs_shift_pp"], 0.0)

    def test_the_bound_stays_far_below_the_smallest_reported_effect(self):
        """If this ever fails, the overshoot can no longer be dismissed."""
        sens = load(PKG / "statistical_sensitivity.json")
        worst = max(b["max_abs_shift_pp"]
                    for b in sens["excess_episode_bounds"]["per_arm"].values())
        summary = load(SUMMARY)
        effects = [abs(c["mean_diff"]) * 100
                   for c in summary["contrasts_vs_H"] + summary["contrasts_within_E"]
                   if c["metric"] == "capture_rate"]
        self.assertLess(worst, min(effects) / 10,
                        "the excess-episode bound is no longer negligible against the effects")


class PrimaryMetricLabelling(unittest.TestCase):
    def test_min_relative_distance_is_declared_not_recorded(self):
        pm = load(SUMMARY)["primary_metrics_preregistered"]["min_relative_distance_m"]
        self.assertEqual("NOT_RECORDED", pm["status"])
        self.assertTrue(pm["evidence"].strip())

    def test_closest_nocrash_is_never_promoted_to_primary(self):
        summary = load(SUMMARY)
        diag = summary["post_hoc_conditional_diagnostics"]["closest_nocrash_mean_m"]
        self.assertEqual("POST_HOC_CONDITIONAL_DIAGNOSTIC", diag["label"])
        self.assertNotIn("closest_nocrash_mean_m",
                         summary["primary_metrics_preregistered"])

    def test_the_document_does_not_call_closest_nocrash_a_primary(self):
        text = DOC.read_text(encoding="utf-8")
        for m in re.finditer(r"closest_nocrash_mean_m", text):
            window = text[max(0, m.start() - 200):m.start() + 200].lower()
            self.assertIn("post_hoc_conditional_diagnostic", window,
                          "closest_nocrash_mean_m appears without its post-hoc label")

    def test_exact_2048_view_status_is_recorded_rather_than_faked(self):
        pm = load(PRIMARY)
        self.assertEqual("NOT_CONSTRUCTIBLE", pm["status"])
        self.assertFalse(pm["per_episode_records_present"])
        self.assertTrue(pm["reason"].strip())


class ThresholdProvenance(unittest.TestCase):
    def test_the_five_pp_threshold_is_marked_not_preregistered(self):
        tp = load(SUMMARY)["retraining_verdict"]["threshold_provenance"]
        self.assertEqual("NOT_PREREGISTERED", tp["five_pp_materiality"])
        self.assertEqual("POST_HOC DECISION CONTEXT", tp["status_of_the_5pp_rule"])

    def test_no_document_calls_the_threshold_preregistered(self):
        pattern = re.compile(r"preregistered[^.\n]{0,60}(5\s*pp|materiality)", re.IGNORECASE)
        for path in (DOC, DEVIATIONS):
            text = path.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), 1):
                self.assertIsNone(
                    pattern.search(line),
                    f"{path.name}:{line_no} describes a materiality threshold as preregistered")

    # The preregistration and its GO attestation. Anything reachable from here
    # predates measurement; anything after it cannot be a preregistered rule.
    PREREG_COMMIT = "096eee5"

    def test_no_pre_measurement_commit_ever_introduced_the_threshold(self):
        """The provenance claim is mechanically checkable.

        The threshold may legitimately appear in LATER commits -- this result
        package documents it at length in order to withdraw it. What must stay
        empty is the history up to and including the preregistration, because a
        rule that appears only afterwards cannot have been preregistered.
        """
        out = subprocess.run(
            ["git", "log", self.PREREG_COMMIT, "--oneline", "-S", "material_threshold"],
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(
            "", out.stdout.strip(),
            "a pre-measurement commit introduces material_threshold; re-audit the provenance")

    def test_the_threshold_is_only_ever_mentioned_in_order_to_withdraw_it(self):
        for path in (DOC, DEVIATIONS):
            text = path.read_text(encoding="utf-8")
            if "material_threshold" not in text and "5 pp" not in text:
                continue
            self.assertRegex(text, r"(?i)not[_ ]preregistered",
                             f"{path.name} mentions the threshold without withdrawing it")

    def test_verdict_is_the_narrow_scoped_phrase(self):
        v = load(SUMMARY)["retraining_verdict"]
        self.assertEqual("NO_RETRAINING_JUSTIFIED_FOR_E2_TARGET_MOTION_SHIFT", v["verdict"])
        self.assertFalse(v["ppo_training_started"])
        self.assertTrue(v["scope"].strip())


class SignificanceWordingIsRestricted(unittest.TestCase):
    BANNED = re.compile(r"statistically significant|\bp\s*<\s*0?\.05|significant difference",
                        re.IGNORECASE)

    def test_no_significance_claim_appears_in_the_result_documents(self):
        for path in (DOC, DEVIATIONS):
            text = path.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), 1):
                m = self.BANNED.search(line)
                if not m:
                    continue
                window = line[max(0, m.start() - 120):m.start()].lower()
                self.assertRegex(
                    window, r"\b(not|cannot|never|no|insufficient|without)\b[^.]*$",
                    f"{path.name}:{line_no} makes a significance claim at n=3")

    def test_the_permutation_resolution_limit_is_stated(self):
        sens = load(PKG / "statistical_sensitivity.json")["permutation_resolution"]
        self.assertEqual(3, sens["n_seeds"])
        self.assertAlmostEqual(0.25, sens["min_attainable_two_sided_p"])

    def test_every_capture_contrast_reports_all_three_seed_effects(self):
        summary = load(SUMMARY)
        for c in summary["contrasts_vs_H"] + summary["contrasts_within_E"]:
            if c["metric"] != "capture_rate":
                continue
            self.assertEqual({"4101", "4102", "4103"}, set(c["seed_diffs"]),
                             f"{c['arm']} vs {c['baseline']}: seed effects not fully reported")


class DensityClaimsStayExploratory(unittest.TestCase):
    def test_density_block_is_labelled_exploratory(self):
        self.assertEqual("EXPLORATORY", load(SUMMARY)["density_descriptive"]["status"])

    def test_document_does_not_claim_a_formal_interaction(self):
        text = DOC.read_text(encoding="utf-8").lower()
        if "interaction" in text:
            self.assertIn("exploratory", text)


class MechanismClaimStaysBounded(unittest.TestCase):
    def test_visibility_is_not_claimed_as_a_demonstrated_cause(self):
        text = DOC.read_text(encoding="utf-8")
        self.assertRegex(text, r"(?i)consistent with a visibility/reacquisition")
        self.assertRegex(text, r"(?i)no mediation or counterfactual experiment was run")


if __name__ == "__main__":
    unittest.main()
