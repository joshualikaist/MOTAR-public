"""The browser interception preview must mirror the research termination contract.

Capture radius and episode budget are not browser parameters: they belong to the
research task. The page's geometry literal carries them with provenance, and this
test binds the literal to the research source so the two cannot drift. It also
guards the structural fix: no pursuit path may return before the common outcome
check, and no path may carry its own capture literal.
"""
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "docs/status/viewer.js"
CONTRACT = ROOT / "docs/status/research_task_contract.json"
CONTRACT_TOOL = ROOT / "tools/build_research_task_contract.py"
HARNESS = ROOT / "tools/validate_gt_browser_tracking.js"
SIM_CONFIG = ROOT / "aerial_gym/config/sim_config/base_sim_config.py"
BARS_ENV = ROOT / "aerial_gym/config/env_config/navrl_bars_env.py"
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


class StaticTaskContractIsBoundToSource(unittest.TestCase):
    """The static contract is the single source; it must match research source."""

    longMessage = False

    def setUp(self):
        self.contract = json.loads(read(CONTRACT))

    def test_contract_schema(self):
        self.assertEqual(self.contract["schema_version"], 1)
        self.assertEqual(self.contract["task"], "navrl")
        self.assertTrue(self.contract["capture_ends_episode"])

    def test_capture_radius_equals_research_success_radius(self):
        research = re.search(r"^\s*success_radius\s*=\s*([0-9.]+)", read(TASK_CONFIG), re.M)
        self.assertIsNotNone(research, "navrl_task_config.success_radius not found")
        self.assertEqual(self.contract["success_radius_m"], float(research.group(1)),
                         "contract capture radius drifted from navrl_task_config.success_radius")

    def test_episode_budget_equals_the_v2_training_contract(self):
        launcher = re.search(r"^export NAVRL_EPISODE_LEN_STEPS=(\d+)", read(V2_LAUNCHER), re.M)
        self.assertIsNotNone(launcher, "v2 launcher no longer exports NAVRL_EPISODE_LEN_STEPS")
        self.assertEqual(self.contract["episode_len_steps"], int(launcher.group(1)),
                         "contract episode budget drifted from the v2 launcher")

    def test_rl_step_dt_is_derived_not_asserted(self):
        dt = re.search(r"^\s*dt\s*=\s*([0-9.]+)", read(SIM_CONFIG), re.M)
        steps = re.search(r"^\s*num_physics_steps_per_env_step_mean\s*=\s*(\d+)",
                          read(BARS_ENV), re.M)
        self.assertIsNotNone(dt); self.assertIsNotNone(steps)
        self.assertAlmostEqual(self.contract["rl_step_dt_s"],
                               float(dt.group(1)) * int(steps.group(1)), places=12)
        self.assertAlmostEqual(self.contract["episode_timeout_s"],
                               self.contract["episode_len_steps"] * self.contract["rl_step_dt_s"],
                               places=9)

    def test_research_task_still_terminates_on_capture(self):
        self.assertIn("capture ends the episode", read(NAVRL_TASK),
                      "the research task no longer states that capture ends the episode; "
                      "the browser mirror claim must be re-audited")

    def test_contract_is_generated_and_carries_provenance(self):
        prov = self.contract["provenance"]
        self.assertIn("bindings", prov)
        self.assertIn("source_sha256", prov)
        for path in prov["source_sha256"]:
            self.assertTrue((ROOT / path).is_file(), f"contract cites missing source {path}")
        self.assertTrue(CONTRACT_TOOL.is_file(), "the contract generator is missing")

    def test_generator_check_mode_agrees_with_the_committed_contract(self):
        result = subprocess.run(
            [sys.executable, str(CONTRACT_TOOL), "--check"],
            cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0,
                         f"committed contract drifted from source:\n{result.stderr}")


class ConsumersBindToTheStaticContract(unittest.TestCase):
    longMessage = False

    def test_viewer_loads_the_contract_and_fails_closed(self):
        viewer = read(VIEWER)
        self.assertIn("research_task_contract.json", viewer)
        self.assertIn("validateContract", viewer)
        self.assertIn("could not start: research task contract", viewer,
                      "viewer must fail closed when the contract is missing/malformed")

    def test_viewer_no_longer_declares_termination_literals(self):
        viewer = read(VIEWER)
        presentation = viewer[viewer.index("const presentation = {"):]
        presentation = presentation[:presentation.index("};")]
        for key in ("success_radius_m", "episode_len_steps", "rl_step_dt_s"):
            self.assertNotIn(key, presentation,
                             f"{key} is back as a presentation literal; it belongs to the contract")

    def test_harness_binds_to_the_contract_and_fails_closed(self):
        harness = read(HARNESS)
        self.assertIn("research_task_contract.json", harness)
        self.assertIn("research task contract unreadable", harness)
        self.assertNotIn("GEO.success_radius_m || 0.5", harness,
                         "the harness still has a silent status.json fallback")

    def test_arena_fallbacks_are_explicitly_labelled(self):
        arena = read(ARENA)
        self.assertIn("dev fallback; see research_task_contract.json", arena)
        self.assertIn("terminationFromContract", arena,
                      "arena must report whether termination came from the contract")

    def test_status_json_stays_a_dynamic_snapshot(self):
        # The static contract must NOT be merged into the dynamic dashboard file.
        status = json.loads(read(ROOT / "docs/status/status.json"))
        self.assertNotIn("success_radius_m", json.dumps(status.get("arena_geometry", {})),
                         "termination fields leaked into the dynamic status snapshot")


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


class PublicTerminologyMapsWithoutRenamingInternals(unittest.TestCase):
    """UI labels follow the current public framing; internals keep their names.

    Historical research records and internal state names (CHASE, INTERCEPT,
    CAPTURED, success_radius) are provenance and must NOT be renamed. Only what
    a reader sees on the HUD changes.
    """

    longMessage = False

    def test_ui_label_map_exists_and_covers_every_state(self):
        arena = read(ARENA)
        self.assertIn("const PHASE_LABEL = {", arena)
        for internal, shown in (("CHASE", "TRACKING"), ("CLOSE", "CLOSE APPROACH"),
                                ("INTERCEPT", "FINAL APPROACH"),
                                ("CAPTURED", "APPROACH COMPLETE"),
                                ("TIMEOUT", "TIMEOUT"), ("ABORT", "RESET / ABORT")):
            self.assertRegex(arena, rf"{internal}:\s*'{re.escape(shown)}'",
                             f"UI label for internal state {internal} is missing or changed")

    def test_internal_state_names_are_unchanged(self):
        # Ownership: the planner owns the PHASE names; the shared outcome
        # function owns the terminal OUTCOME names. Neither is renamed.
        planner = read(PLANNER)
        for name in ("'CHASE'", "'CLOSE'", "'INTERCEPT'", "'CAPTURED'"):
            self.assertIn(name, planner, f"internal state {name} was renamed in the planner")
        motion = read(MOTION)
        for name in ("'CAPTURED'", "'TIMEOUT'"):
            self.assertIn(name, motion, f"internal outcome {name} was renamed in episodeOutcome")

    def test_research_records_keep_capture_vocabulary(self):
        # success_radius / capture are historical semantics: never renamed.
        self.assertIn("success_radius", read(TASK_CONFIG))
        self.assertIn("capture ends the episode", read(NAVRL_TASK))

    def test_hud_shows_public_framing(self):
        index = read(INDEX)
        self.assertIn("CLOSE-APPROACH EPISODE", index)
        self.assertIn("TRACKING", index)
        self.assertNotIn("INTERCEPTION MODE", index,
                         "the HUD still shows the old interception wording")


if __name__ == "__main__":
    unittest.main()
