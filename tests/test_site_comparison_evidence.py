"""Bind public comparison rows to the original result records; no experiments run."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SITE = (ROOT / "docs/status/index.html").read_text()


class SiteComparisonEvidenceTest(unittest.TestCase):
    def assert_site_and_source(self, site_phrases, source, source_phrases):
        for phrase in site_phrases:
            self.assertIn(phrase, SITE)
        text = (ROOT / source).read_text()
        for phrase in source_phrases:
            self.assertIn(phrase, text)

    def test_arc_vs_riskcap(self):
        self.assert_site_and_source(
            ("−1.4903 pp", "[−1.8981, −1.0826]", "15/15 cells"),
            "results/independent_verification_2026-09-07/README.md",
            ("−1.4903 percentage points", "[−1.8981, −1.0826]"),
        )

    def test_arc_width(self):
        self.assert_site_and_source(
            ("−5.60 pp; capture +4.44 pp",),
            "results/independent_verification_2026-09-07/tables.md",
            ("| 205 | dwa_arc | -5.60", "| +4.44"),
        )

    def test_riskcap_adaptation(self):
        self.assert_site_and_source(
            ("+3.75 pp, 95% CI [+1.30, +6.19]", "CI [−4.42, +0.15]"),
            "results/navrl_v2_riskcap_postadapt/summary.md",
            ("capture: +3.75 pp (95% CI +1.30..+6.19)",
             "crash: -2.14 pp (95% CI -4.42..+0.15)"),
        )

    def test_latency_compensation(self):
        self.assert_site_and_source(
            ("37.82% → 78.04% (+40.21 pp)",),
            "results/navrl_v2_latency_ego_motion/summary.md",
            ("latency_0p1s_p3", "78.04%", "latency_0p1s_raw", "37.82%",
             "capture vs raw +40.21 pp"),
        )

    def test_detector_noninferiority(self):
        self.assert_site_and_source(
            ("−0.015 pp; 95% CI [−1.752, +1.723]",),
            "results/navrl_v2_detector_navigation_ab_replication_seed97_101_schema2/summary.md",
            ("**-0.015 pp**", "[-1.752, +1.723]", "margin −2.0 pp"),
        )

    def test_temporal_selector(self):
        self.assert_site_and_source(
            ("0.6655", "0.6725", "+0.00697 utility", "0.4701"),
            "results/perception_temporal_p6_p7_2026-09-08/README.md",
            ("0.66551", "0.67247", "`+0.00697`", "test superiority"),
        )
        self.assertIn(
            "0.4701",
            (ROOT / "results/perception_s4_2026-09-09/README.md").read_text(),
        )

    def test_p10_replication(self):
        self.assert_site_and_source(
            ("Mean +0.73 pp; 95% CI [−1.04, +2.50]", "INCONCLUSIVE"),
            "results/perception_p10_seed_replication_2026-09-10/README.md",
            ("mean **+0.73 pp**", "CI **[−1.04, +2.50]**", "INCONCLUSIVE"),
        )

    def test_d8b_material_loss(self):
        self.assert_site_and_source(
            ("−48.967 pp", "[−50.113, −47.821]", "MATERIAL_LOSS"),
            "results/dynamic_mesh_policy_sensitivity_d8b_2026-09-13/README.md",
            ("**-48.967 pp**", "[-50.113, -47.821]", "MATERIAL_LOSS"),
        )

    def test_external_rows_are_relations_not_head_to_head_results(self):
        section = SITE.split("<h3>6.3 Relation to published systems</h3>", 1)[1]
        section = section.split("<h3>6.4 Current evidence boundaries</h3>", 1)[0]
        self.assertIn("No selected work is currently Class A", section)
        self.assertNotIn("outperforms MOTAR", section)
        self.assertNotIn("MOTAR outperforms", section)


if __name__ == "__main__":
    unittest.main()
