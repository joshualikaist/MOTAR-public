"""The browser interception preview must mirror the research termination contract.

Capture radius and episode budget are not browser parameters: they belong to the
research task. The page's geometry literal carries them with provenance, and this
test binds the literal to the research source so the two cannot drift. It also
guards the structural fix: no pursuit path may return before the common outcome
check, and no path may carry its own capture literal.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "docs/status/viewer.js"
ARENA = ROOT / "docs/status/arena.js"
MOTION = ROOT / "docs/status/arena_motion.js"
PLANNER = ROOT / "docs/status/arena_demo_planner.js"
TASK_CONFIG = ROOT / "aerial_gym/config/task_config/navrl_task_config.py"
NAVRL_TASK = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"
V2_LAUNCHER = ROOT / "aerial_gym/rl_training/rl_games/train_navrl_v2_search.sh"
SNAPSHOT_TOOL = ROOT / "tools/update_status_snapshot.py"
INDEX = ROOT / "docs/status/index.html"


def read(path):
    return path.read_text(encoding="utf-8")


def literal(text, key):
    match = re.search(rf"^\s*{key}:\s*([0-9.]+)\s*,", text, re.M)
    assert match, f"{key} missing from the viewer geometry literal"
    return float(match.group(1))


class TerminationContractProvenance(unittest.TestCase):
    longMessage = False

    def setUp(self):
        self.viewer = read(VIEWER)

    def test_capture_radius_equals_research_success_radius(self):
        research = re.search(r"^\s*success_radius\s*=\s*([0-9.]+)", read(TASK_CONFIG), re.M)
        self.assertIsNotNone(research, "navrl_task_config.success_radius not found")
        self.assertEqual(literal(self.viewer, "success_radius_m"), float(research.group(1)),
                         "browser capture radius drifted from navrl_task_config.success_radius")

    def test_episode_budget_equals_the_v2_training_contract(self):
        launcher = re.search(r"^export NAVRL_EPISODE_LEN_STEPS=(\d+)", read(V2_LAUNCHER), re.M)
        self.assertIsNotNone(launcher, "v2 launcher no longer exports NAVRL_EPISODE_LEN_STEPS")
        self.assertEqual(literal(self.viewer, "episode_len_steps"), float(launcher.group(1)),
                         "browser episode budget drifted from the v2 launcher contract")
        self.assertEqual(literal(self.viewer, "rl_step_dt_s"), 0.1)

    def test_research_task_still_terminates_on_capture(self):
        self.assertIn("capture ends the episode", read(NAVRL_TASK),
                      "the research task no longer states that capture ends the episode; "
                      "the browser mirror claim must be re-audited")

    def test_status_generator_carries_the_same_provenance(self):
        tool = read(SNAPSHOT_TOOL)
        self.assertIn('"success_radius_m": 0.5', tool)
        self.assertIn('"rl_step_dt_s": 0.1', tool)


class CommonTerminationLayer(unittest.TestCase):
    longMessage = False

    def setUp(self):
        self.arena = read(ARENA)
        start = self.arena.index("  function simulationStep(dt) {")
        end = self.arena.index("  function animate() {")
        self.step = self.arena[start:end]

    def test_simulation_step_reaches_the_common_outcome_check(self):
        self.assertIn("evaluateEpisodeOutcome()", self.step)
        target = self.step.index("stepTarget(dt)")
        outcome = self.step.index("evaluateEpisodeOutcome()")
        between = self.step[target:outcome]
        self.assertNotIn("return;", between,
                         "a return sits between the motion steps and the common outcome "
                         "check: some pursuit path can bypass termination again")

    def test_gt_branch_no_longer_returns_past_termination(self):
        # The old bug: the gt-route-track branch ended in `return;` before the
        # capture / watchdog code. The pursuer step must live in its own
        # function and hand control back.
        pursuer_start = self.arena.index("  function stepPursuer(dt) {")
        pursuer_end = self.arena.index("  function evaluateEpisodeOutcome() {")
        body = self.arena[pursuer_start:pursuer_end]
        self.assertIn("Planner.stepPursuerGt", body)
        self.assertNotIn("sweptCapture", body)

    def test_no_pursuit_path_carries_its_own_capture_literal(self):
        self.assertNotRegex(self.arena, r"sweptCapture\([^)]*0\.5\s*\)",
                            "a hard-coded 0.5 m capture literal is back in arena.js")
        self.assertNotIn("episode.age >= 30", self.arena,
                         "the unprovenanced 30 s browser watchdog is back")
        self.assertIn("Motion.episodeOutcome(", self.arena)
        self.assertIn("Motion.episodeOutcome(", read(PLANNER))
        self.assertIn("function episodeOutcome(", read(MOTION))

    def test_terminal_state_resets_exactly_once(self):
        self.assertIn("function finishTerminalState()", self.arena)
        self.assertIn("terminalState = null;", self.arena)
        finish = self.arena[self.arena.index("function finishTerminalState()"):]
        finish = finish[:finish.index("  function simulationStep")]
        # count CALL sites only; the explanatory comment also mentions the name
        calls = [line for line in finish.splitlines()
                 if "resetEpisode(true)" in line and not line.strip().startswith("//")]
        self.assertEqual(len(calls), 1, "finishTerminalState must reset exactly once")


class ModesAreSeparate(unittest.TestCase):
    longMessage = False

    def test_site_defaults_to_interception_and_keeps_continuous(self):
        index = read(INDEX)
        self.assertIn('<option value="interception" selected>', index)
        self.assertIn('<option value="continuous">', index)
        self.assertIn('id="hud-episode-mode"', index)
        self.assertIn('id="hud-phase"', index)
        self.assertIn('id="hud-outcome"', index)
        self.assertIn("mirrors the task termination semantics, but is not PPO or PhysX performance evidence",
                      index)

    def test_planner_default_stays_continuous(self):
        # Planner-level default must stay 'continuous' so the tracking
        # contracts (GT_BROWSER_V1) keep their meaning; the SITE picks interception.
        planner = read(PLANNER)
        self.assertIn("options.episodeMode === 'interception' ? 'interception' : 'continuous'", planner)
        arena = read(ARENA)
        self.assertIn("let episodeMode = 'interception';", arena)

    def test_standoff_is_not_a_capture_radius(self):
        planner = read(PLANNER)
        self.assertIn("standoffM: 1.55", planner)
        self.assertNotIn("captureRadiusM: 0.5", planner,
                         "the planner must receive the capture radius from the page, not declare it")


if __name__ == "__main__":
    unittest.main()
