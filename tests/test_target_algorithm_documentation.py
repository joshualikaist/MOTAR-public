"""The target-motion documentation must agree with the target-motion source.

These are consistency tests, not behaviour tests. They read the actual motion
implementations and fail when a document, the site, or the FAQ claims something
the code does not do. The failure mode they exist to prevent is a plausible,
confident explanation that is wrong -- above all describing the browser preview
and the research TM-E2 executor as one algorithm.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]

CANONICAL = ROOT / "docs/target_motion_algorithm_2026-09-17.md"
ANSWERS = ROOT / "docs/presentation_target_algorithm_answers_2026-09-17.md"
LADDER = ROOT / "docs/target_behavior_ladder_2026-09-16.md"
SITE = ROOT / "docs/status/index.html"

TARGET_MOTION = ROOT / "aerial_gym/task/navrl_task/target_motion.py"
ROUTE_PLANNER = ROOT / "aerial_gym/task/navrl_task/target_route_planner.py"
NAVRL_TASK = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"
TASK_CONFIG = ROOT / "aerial_gym/config/task_config/navrl_task_config.py"

BROWSER_ROUTE = ROOT / "docs/status/arena_route.js"
BROWSER_PLANNER = ROOT / "docs/status/arena_demo_planner.js"
BROWSER_MOTION = ROOT / "docs/status/arena_motion.js"


def read(path):
    return path.read_text(encoding="utf-8")


class TargetAlgorithmSourceFacts(unittest.TestCase):
    """The facts the documentation rests on, re-derived from source."""

    def test_tm_e2_executor_contains_no_global_route_planner(self):
        # The single most important claim in the documentation: TM-E2 is local.
        # If A* or a stored route ever appears in the bounded executor's module,
        # every "TM-E2 does not use A*" sentence becomes a lie and must be
        # rewritten before this test is relaxed.
        source = read(TARGET_MOTION)
        for token in ("heapq", "astar", "a_star", "plan_to_connected_goal"):
            self.assertNotIn(
                token, source.lower(),
                f"{token!r} appeared in target_motion.py; TM-E2 may no longer be "
                "purely local and docs/target_motion_algorithm_2026-09-17.md "
                "must be re-audited",
            )

    def test_tm_e2_is_a_receding_horizon_candidate_executor(self):
        source = read(TARGET_MOTION)
        self.assertIn("def bounded_drone_target_step", source)
        self.assertIn("BOUNDED_TURN_ANGLES_DEG", source)
        self.assertIn("cruise_scales", source)
        self.assertIn("safe_prefix_steps", source)

    def test_research_a_star_is_eight_connected_with_euclidean_heuristic(self):
        source = read(ROUTE_PLANNER)
        self.assertIn("import heapq", source)
        neighbours = re.search(r"_NEIGHBORS = \((.*?)\n\)", source, re.S)
        self.assertIsNotNone(neighbours, "_NEIGHBORS table not found")
        self.assertEqual(
            len(re.findall(r"\(\s*-?\d+,\s*-?\d+,", neighbours.group(1))), 8,
            "research planner is no longer 8-connected; §5.3 must be updated",
        )
        self.assertIn("math.hypot(i - goal_cell[0], j - goal_cell[1])", source)

    def test_research_route_mode_defaults_off_and_is_physical_only(self):
        self.assertIn('"NAVRL_TARGET_ROUTE_MODE", "off"', read(TASK_CONFIG))
        self.assertIn("physical+waypoint-only lineage", read(NAVRL_TASK))

    def test_reactive_and_learned_levels_fail_closed(self):
        source = read(TARGET_MOTION)
        self.assertIn("e3_reactive", source)
        self.assertIn("e4_learned", source)
        self.assertIn("NotImplementedError", source)
        self.assertIn("is PLANNED, not implemented", source)

    def test_no_target_level_claims_privileged_pursuer_gt(self):
        self.assertIn('"uses_privileged_pursuer_gt": False', read(TARGET_MOTION))

    def test_browser_planner_is_eight_connected_and_rejects_corner_cutting(self):
        source = read(BROWSER_ROUTE)
        self.assertEqual(
            len(re.findall(r"\[-?\d+,\s*-?\d+,\s*(?:1|Math\.SQRT2)\]", source)), 8,
            "browser planner is no longer 8-connected; §5.3 must be updated",
        )
        self.assertIn("di && dj && (!free", source,
                      "browser planner lost its diagonal corner-cut rejection")

    def test_browser_route_simplification_is_a_shortcut_not_a_spline(self):
        for path in (BROWSER_ROUTE, ROUTE_PLANNER):
            source = read(path).lower()
            for token in ("spline", "bezier", "catmull", "polyfit"):
                self.assertNotIn(
                    token, source,
                    f"{token!r} appeared in {path.name}; the documentation says "
                    "the route is simplified by shortcut, not curve fitting",
                )

    def test_browser_bounded_follower_enforces_the_documented_bounds(self):
        motion = read(BROWSER_MOTION)
        for token in ("boundedMaxAccel", "boundedMaxTurnRate", "limitPlanarVelocity"):
            self.assertIn(token, motion)
        self.assertIn("integrateBounded", read(BROWSER_PLANNER))


class DocumentationAgreesWithSource(unittest.TestCase):
    """The prose must not contradict the facts above."""

    # A failed assertIn on a whole document prints the whole document. Keep the
    # messages, drop the dump.
    longMessage = False
    maxDiff = 0

    def setUp(self):
        self.canonical = read(CANONICAL)
        self.answers = read(ANSWERS)

    def test_canonical_document_separates_browser_from_tm_e2(self):
        for needle in ("TM-E2 does not use A",
                       "Browser GT free-roam vs research TM-E2",
                       "They do not share a planner"):
            self.assertTrue(needle in self.canonical,
                            f"canonical document lost: {needle!r}")

    def test_canonical_document_never_generalises_a_star_to_the_whole_project(self):
        # Catch the exact sentence shape people reach for.
        for bad in (
            "MOTAR target uses A*",
            "the MOTAR target uses A*",
            "MOTAR uses A* for the target",
        ):
            self.assertFalse(bad.lower() in self.canonical.lower(),
                             f"over-generalised A* claim: {bad!r}")

    def test_documents_state_the_target_is_not_pursuer_reactive(self):
        for name, text in (("canonical", self.canonical), ("answers", self.answers)):
            lowered = text.lower()
            self.assertTrue(
                "pursuer-independent" in lowered or "pursuer에는 반응하지" in text,
                f"{name}: the pursuer-independence statement is missing",
            )
        self.assertIn("adversarial", self.canonical.lower())

    def test_documents_state_the_fail_closed_no_route_behaviour(self):
        self.assertTrue("zero command" in self.canonical.lower(),
                        "canonical document lost the zero-command statement")
        for name, text in (("canonical", self.canonical), ("answers", self.answers)):
            self.assertTrue("replan" in text.lower(), f"{name}: replan statement missing")

    def test_documents_keep_tm_e3_and_tm_e4_planned(self):
        for text in (self.canonical, self.answers):
            lowered = text.lower()
            self.assertTrue("tm-e3" in lowered, "a document lost its TM-E3 mention")
            self.assertTrue("planned", "a document lost its PLANNED status")
            self.assertTrue("planned" in lowered, "a document lost its PLANNED status")
            for claimed in ("tm-e3 is implemented", "tm-e4 is implemented"):
                self.assertFalse(claimed in lowered, f"{claimed!r} appeared in a document")

    def test_documents_keep_the_gt_boundary(self):
        for needle in ("uses_privileged_pursuer_gt", "never passed to the PPO actor"):
            self.assertTrue(needle in self.canonical,
                            f"canonical document lost the GT boundary text: {needle!r}")
        # And must never claim the pursuer policy receives target GT.
        for bad in (
            "the pursuer receives gt",
            "policy receives the gt target",
            "gt target position is in the observation",
        ):
            self.assertNotIn(bad, self.canonical.lower())

    def test_documents_keep_the_physical_route_verdict(self):
        self.assertTrue("FAIL_ROUTE_MECHANISM" in self.canonical,
                        "canonical document lost the physical route verdict")

    def test_browser_preview_is_labelled_browser_only_everywhere_it_is_explained(self):
        for name, text in (("canonical", self.canonical), ("answers", self.answers)):
            self.assertTrue("not ppo" in text.lower(), f"{name}: missing NOT PPO label")
        self.assertTrue("not physx" in self.canonical.lower(),
                        "canonical document lost the NOT PhysX label")
        self.assertTrue("not a performance" in self.canonical.lower(),
                        "canonical document lost the not-a-performance-result label")

    def test_canonical_values_match_the_browser_contract(self):
        route = read(BROWSER_ROUTE)
        motion = read(BROWSER_MOTION)
        planner = read(BROWSER_PLANNER)
        expected = {
            "0.25 m": "resolutionM: 0.25" in route,
            "0.45 m": "trackingMarginM: 0.45" in route,
            "1.25 m": "boundaryMarginM: 1.25" in route,
            "6.0 m": "minGoalDistanceM: 6.0" in route,
            "1.0 m": "goalExclusionRadiusM: 1.0" in route,
            "4.0 m/s²": "boundedMaxAccel: 4.0" in motion,
            "2.5 m/s": "pursuerSpeedMax: 2.5" in motion,
            "1.8 m": "lookAheadM: 1.8" in planner,
        }
        for value, present_in_source in expected.items():
            self.assertTrue(
                present_in_source,
                f"the source no longer defines {value}; the value table in "
                "docs/target_motion_algorithm_2026-09-17.md §9 is now stale",
            )
            self.assertTrue(
                value in self.canonical,
                f"{value} is in the source but missing from the §9 value table",
            )


class SiteAgreesWithTheCanonicalDocument(unittest.TestCase):
    longMessage = False
    maxDiff = 0

    def setUp(self):
        self.site = read(SITE)

    def test_site_explains_target_motion_and_links_the_canonical_document(self):
        self.assertTrue("Target motion" in self.site,
                        "site lost its Target motion heading")
        self.assertTrue("target_motion_algorithm_2026-09-17.md" in self.site,
                        "site no longer links the canonical target-motion document")

    def test_site_marks_the_target_explanation_as_browser_only(self):
        block = self.site.split('id="target-motion"', 1)
        self.assertEqual(len(block), 2, "site lost its target-motion section")
        section = block[1].split("</section>", 1)[0]
        for needle in ("Browser visualization only",
                       "separate historical target-motion lineages"):
            self.assertTrue(needle in section,
                            f"site target-motion section lost: {needle!r}")

    def test_site_does_not_claim_the_research_target_uses_a_star(self):
        section = self.site.split('id="target-motion"', 1)[1].split("</section>", 1)[0]
        lowered = section.lower()
        if "a*" in lowered or "a&ast;" in lowered:
            self.assertTrue(
                "browser" in lowered,
                "the site mentions A* in the target-motion section without "
                "scoping it to the browser preview",
            )

    def test_site_states_the_target_is_not_pursuer_reactive(self):
        section = self.site.split('id="target-motion"', 1)[1].split("</section>", 1)[0]
        self.assertTrue("pursuer" in section.lower(),
                        "site target-motion section never mentions the pursuer")
        self.assertTrue("adversarial" in section.lower(),
                        "site does not distinguish obstacle-aware from adversarial")


class LadderStaysConsistent(unittest.TestCase):
    def test_ladder_still_records_tm_e2_as_not_policy_compared(self):
        ladder = read(LADDER)
        self.assertIn("NOT TESTED", ladder.upper())
        self.assertIn("TM-E3", ladder)

    def test_ladder_and_canonical_document_cross_reference(self):
        self.assertIn("target_behavior_ladder_2026-09-16.md", read(CANONICAL))


if __name__ == "__main__":
    unittest.main()
