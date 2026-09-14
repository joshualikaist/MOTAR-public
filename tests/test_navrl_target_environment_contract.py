"""CPU-only guards for target-motion training-environment provenance.

These tests intentionally inspect source rather than importing NavRLTask, whose module import
requires Isaac Gym.  They protect fail-loud checkpoint metadata and fresh-lineage launcher rules;
they do not claim dynamic feasibility or policy performance.
"""

from pathlib import Path
import os
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
TASK_SOURCE = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text(encoding="utf-8")
TASK_CONFIG_SOURCE = (
    ROOT / "aerial_gym/config/task_config/navrl_task_config.py"
).read_text(encoding="utf-8")
PHYSICAL_LAUNCHER = (
    ROOT / "aerial_gym/rl_training/rl_games/train_navrl_physical_fresh.sh"
).read_text(encoding="utf-8")
V2_LAUNCHER = (
    ROOT / "aerial_gym/rl_training/rl_games/train_navrl_v2_search.sh"
).read_text(encoding="utf-8")
ROUTED_LAUNCHER_PATH = (
    ROOT / "aerial_gym/rl_training/rl_games/train_navrl_physical_routed_fresh.sh"
)
ROUTED_LAUNCHER = ROUTED_LAUNCHER_PATH.read_text(encoding="utf-8")
RL = ROOT / "aerial_gym/rl_training/rl_games"

ROUTED_FROZEN = {
    "NAVRL_ROBOT": "navrl_ref5in_v2_quad",
    "NAVRL_PHYSICAL_GEOMETRY_VERSION": "v2",
    "NAVRL_TARGET_BOX_XY_M": "0.283",
    "NAVRL_TARGET_DYNAMICS": "physical",
    "NAVRL_TARGET_ROUTE_MODE": "global_astar_v1",
    "NAVRL_TARGET_PATTERN": "waypoint",
    "NAVRL_ARENA_XY": "40",
    "NAVRL_ARENA_Z": "3",
    "NAVRL_BAR_POOL": "bars_h3",
    "NAVRL_BAR_X_MIN": "0",
    "NAVRL_BAR_X_MAX": "1",
    "NAVRL_PLACEMENT_MODE": "footprint_clearance",
    "NAVRL_PLACEMENT_SURFACE_CLEARANCE_M": "0.45",
    "NAVRL_MAX_BARS": "300",
    "NAVRL_TARGET_ROUTE_RESOLUTION_M": "0.25",
    "NAVRL_TARGET_ROUTE_MAX_EXPANSIONS": "50000",
    "NAVRL_TARGET_ROUTE_MAX_WAYPOINTS": "128",
    "NAVRL_TARGET_ROUTE_REPLAN_COOLDOWN_STEPS": "10",
    "NAVRL_TARGET_ROUTE_GOAL_TOLERANCE_M": "0.05",
    "NAVRL_TARGET_ROUTE_MIN_GOAL_DISTANCE_M": "6.0",
    "NAVRL_TARGET_ROUTE_GOAL_EXCLUSION_M": "1.0",
    "NAVRL_TARGET_MAX_ACCEL": "4.0",
    "NAVRL_TARGET_MAX_TURN_RATE_DEG": "150.0",
    "NAVRL_TARGET_LOOKAHEAD_S": "1.0",
    "NAVRL_TARGET_OBSTACLE_CLEARANCE": "0.77",
    "NAVRL_TARGET_MASS_KG": "1.20",
    "NAVRL_TARGET_MOTOR_ARM_XY_M": "0.0777817",
    "NAVRL_TARGET_MAX_MOTOR_THRUST_N": "9.60",
    "NAVRL_TARGET_MOTOR_TAU_S": "0.04",
    "NAVRL_TARGET_YAW_TORQUE_RATIO_M": "0.01",
    "NAVRL_TARGET_MAX_TILT_DEG": "45.0",
    "NAVRL_TARGET_VEL_KP": "2.5",
    "NAVRL_TARGET_ALT_KP": "4.0",
    "NAVRL_TARGET_TRACKING_MARGIN_M": "0.45",
    "NAVRL_TARGET_BOUNDARY_MARGIN_M": "0.75",
    "NAVRL_V2_ALLOW_RESUME": "0",
    "NAVRL_V2_PROFILE": "main",
    "NUM_ENVS": "128",
    "NAVRL_VISION": "1",
    "NAVRL_PERCEPTION": "1",
    "NAVRL_GENERAL_TRAIN": "1",
    "NAVRL_GENERAL_GOAL_DIST_MIN": "6",
    "NAVRL_GENERAL_GOAL_DIST_MAX": "28",
    "NAVRL_DENSITY_CURRICULUM": "1",
    "NAVRL_DENSITY_START": "70",
    "NAVRL_DENSITY_FINAL": "205",
    "NAVRL_DENSITY_STEP": "15",
    "NAVRL_DENSITY_THRESHOLD_START": "0.80",
    "NAVRL_DENSITY_THRESHOLD_END": "0.70",
    "NAVRL_DENSITY_THRESHOLD_SCHEDULE": "70:0.82,85:0.77,100:0.72,115:0.70",
    "NAVRL_DENSITY_WARMUP": "1000",
    "NAVRL_DENSITY_CHECK_EPS": "16384",
    "NAVRL_DENSITY_STRATIFIED_GATE": "0",
    "NAVRL_DENSITY_STRATIFIED_FLOOR": "0.55",
    "NAVRL_DENSITY_STRATIFIED_MIN_EPS": "512",
    "NAVRL_DENSITY_MIN_EPOCHS": "1000",
    "NAVRL_OBSTACLE_SELECTOR": "cluster_sector",
    "NAVRL_GEOFENCE_ACTOR": "0",
    "NAVRL_GEOFENCE_NOISE_STD_M": "0",
    "NAVRL_GEOFENCE_DROPOUT": "0",
    "NAVRL_DETECTOR_MIN_PIXELS": "2",
    "NAVRL_TARGET_SPEED_MIN": "0.3",
    "NAVRL_TARGET_SPEED_FINAL": "1.5",
    "NAVRL_TARGET_SPEED_RAMP_EPOCHS": "1",
    "NAVRL_LEARNING_RATE": "3e-5",
    "NAVRL_SPEED_GOVERNOR": "off",
}


class TargetCheckpointContractTest(unittest.TestCase):
    def test_all_motion_authority_fields_are_saved_and_checked_on_restore(self):
        keys = (
            "cfg_target_max_accel_mps2",
            "cfg_target_max_turn_rate_degps",
            "cfg_target_lookahead_s",
            "cfg_target_obstacle_clearance_m",
            "cfg_target_physical_motor_arm_xy_m",
            "cfg_target_physical_yaw_torque_ratio_m",
            "cfg_target_physical_max_tilt_deg",
            "cfg_target_physical_tracking_margin_m",
            "cfg_target_physical_boundary_margin_m",
            "cfg_target_route_mode",
            "cfg_target_route_resolution_m",
            "cfg_target_route_max_expansions",
            "cfg_target_route_max_waypoints",
            "cfg_target_route_replan_cooldown_steps",
            "cfg_target_route_goal_tolerance_m",
            "cfg_target_route_min_goal_distance_m",
            "cfg_target_route_goal_exclusion_radius_m",
        )
        # One occurrence serializes the field and a second occurrence compares it during restore.
        for key in keys:
            self.assertGreaterEqual(TASK_SOURCE.count(f'"{key}"'), 2, key)

    def test_old_checkpoints_remain_compatible(self):
        # Missing provenance is tolerated; present mismatches are loud and discard curriculum
        # evidence. This is the compatibility rule that keeps legacy checkpoints loadable.
        self.assertRegex(TASK_SOURCE, r"saved_value = state\.get\(key\)\s+if saved_value is None:\s+continue")

    def test_local_route_failure_replaces_goal_after_cooldown(self):
        self.assertIn('STATUS_CODES["local_step_infeasible"]', TASK_SOURCE)
        local_branch = TASK_SOURCE.index("if bool(local_failure.any()):")
        ordinary_branch = TASK_SOURCE.index("ordinary_replan =", local_branch)
        self.assertIn(
            "connected_goal=True",
            TASK_SOURCE[local_branch:ordinary_branch],
        )
        self.assertIn(
            "exclude_previous_goal=True",
            TASK_SOURCE[local_branch:ordinary_branch],
        )
        invalidation = TASK_SOURCE.index("local_invalid =")
        self.assertIn(
            "torch.rand_like",
            TASK_SOURCE[invalidation:invalidation + 700],
        )


class TargetLauncherContractTest(unittest.TestCase):
    def test_physical_lineage_is_fresh_only_and_forces_matching_airframe(self):
        self.assertIn("refusing CKPT/CHECKPOINT", PHYSICAL_LAUNCHER)
        self.assertIn("export NAVRL_ROBOT=navrl_ref5in_v2_quad", PHYSICAL_LAUNCHER)
        self.assertIn("export NAVRL_PHYSICAL_GEOMETRY_VERSION=v2", PHYSICAL_LAUNCHER)
        self.assertIn("export NAVRL_TARGET_BOX_XY_M=0.283", PHYSICAL_LAUNCHER)
        self.assertIn('export NAVRL_PLACEMENT_MODE="${NAVRL_PLACEMENT_MODE:-footprint_clearance}"', PHYSICAL_LAUNCHER)
        self.assertIn('export NAVRL_PLACEMENT_SURFACE_CLEARANCE_M="${NAVRL_PLACEMENT_SURFACE_CLEARANCE_M:-0.45}"', PHYSICAL_LAUNCHER)
        self.assertIn("export NAVRL_TARGET_DYNAMICS=physical", PHYSICAL_LAUNCHER)
        self.assertIn("export NAVRL_V2_PHYSICAL_FRESH_CHILD=1", PHYSICAL_LAUNCHER)

    def test_physical_child_marker_survives_until_placement_selection(self):
        self.assertIn(
            '_PHYSICAL_FRESH_CHILD="${NAVRL_V2_PHYSICAL_FRESH_CHILD:-0}"',
            V2_LAUNCHER,
        )
        self.assertIn('if [[ "${_PHYSICAL_FRESH_CHILD}" == "1" ]]; then', V2_LAUNCHER)

    def test_base_v2_refuses_stale_nonlegacy_target_dynamics(self):
        self.assertIn("refusing inherited target dynamics", V2_LAUNCHER)
        self.assertIn('export NAVRL_TARGET_DYNAMICS="${_TARGET_DYNAMICS_REQUESTED}"', V2_LAUNCHER)
        self.assertIn(
            '_PHYSICAL_FRESH_CHILD="${NAVRL_V2_PHYSICAL_FRESH_CHILD:-0}"',
            V2_LAUNCHER,
        )

    def test_route_lineage_has_distinct_marker_and_fresh_guard(self):
        self.assertIn("refusing CKPT/CHECKPOINT", ROUTED_LAUNCHER)
        self.assertIn("export NAVRL_PHYSICAL_ROUTED_CHILD=1", ROUTED_LAUNCHER)
        self.assertIn("export NAVRL_TARGET_ROUTE_MODE=global_astar_v1", ROUTED_LAUNCHER)
        self.assertIn("export NAVRL_TARGET_PATTERN=waypoint", ROUTED_LAUNCHER)
        self.assertIn("export NAVRL_TARGET_ROUTE_GOAL_EXCLUSION_M=1.0", ROUTED_LAUNCHER)
        self.assertIn("canonical physical lineage refuses target route", PHYSICAL_LAUNCHER)

    def _preflight(self, launcher, cwd=RL, **updates):
        env = {
            **os.environ,
            "NAVRL_TARGET_CONTRACT_PREFLIGHT_ONLY": "1",
            "PYTHONNOUSERSITE": "1",
            **updates,
        }
        return subprocess.run(
            [str(launcher)], cwd=cwd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )

    def test_base_physical_and_routed_preflight_contracts(self):
        base = self._preflight(
            RL / "train_navrl_v2_search.sh",
            NAVRL_TARGET_DYNAMICS="legacy", NAVRL_TARGET_ROUTE_MODE="off",
        )
        self.assertEqual(base.returncode, 0, base.stdout)
        self.assertIn("route=off pattern=mixed", base.stdout)

        stale = self._preflight(
            RL / "train_navrl_physical_fresh.sh",
            NAVRL_TARGET_ROUTE_MODE="global_astar_v1", NAVRL_TARGET_PATTERN="waypoint",
        )
        self.assertNotEqual(stale.returncode, 0, stale.stdout)
        self.assertIn("canonical physical lineage refuses target route", stale.stdout)

        physical = self._preflight(RL / "train_navrl_physical_fresh.sh")
        self.assertEqual(physical.returncode, 0, physical.stdout)
        self.assertIn("placement=footprint_clearance", physical.stdout)
        self.assertIn("surface_clearance=0.45", physical.stdout)

        routed = self._preflight(ROUTED_LAUNCHER_PATH)
        self.assertEqual(routed.returncode, 0, routed.stdout)
        self.assertIn("route=global_astar_v1 pattern=waypoint", routed.stdout)

    def test_routed_preflight_from_repo_root_overwrites_hostile_environment(self):
        hostile = {name: "hostile" for name in ROUTED_FROZEN}
        hostile.update({
            "NAVRL_PHYSICAL_ROUTED_CHILD": "1",
            "NAVRL_V2_PHYSICAL_FRESH_CHILD": "1",
            "NAVRL_V2_ROUTED_CHILD": "1",
            "NAVRL_ROUTED_CONTRACT_TOKEN": "forged",
            "NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT": "1",
            "NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE": "1",
            "NAVRL_TRAINING_SOURCE_MANIFEST": "/tmp/forged.json",
            "NAVRL_TRAINING_SOURCE_MANIFEST_SHA256": "f" * 64,
        })
        dirty_probe = ROOT / "aerial_gym/.routed_preflight_dirty_test"
        dirty_probe.write_text("preflight must not require a clean worktree\n", encoding="utf-8")
        try:
            result = self._preflight(ROUTED_LAUNCHER_PATH, cwd=ROOT, **hostile)
        finally:
            dirty_probe.unlink(missing_ok=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("arena=40x40x3 pool=bars_h3", result.stdout)
        self.assertIn("tracking=0.45 boundary=0.75", result.stdout)
        self.assertIn("envs=128 density=70:15:205 max_pool=300", result.stdout)
        self.assertIn("speed=0.3:1.5@1 selector=cluster_sector lr=3e-5 governor=off", result.stdout)
        self.assertIn("receipt_required=0 clean_required=0", result.stdout)

    def test_spoofed_child_markers_cannot_change_frozen_tuple(self):
        common = {
            **ROUTED_FROZEN,
            "NAVRL_PHYSICAL_ROUTED_CHILD": "1",
            "NAVRL_ROUTED_CONTRACT_TOKEN": (
                "physx_ref5in_6dof_global_astar_aabb_v1_frozen_v1"
            ),
            "NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT": "0",
            "NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE": "0",
        }
        physical = self._preflight(
            RL / "train_navrl_physical_fresh.sh", cwd=ROOT,
            **{**common, "NAVRL_TARGET_TRACKING_MARGIN_M": "0"},
        )
        self.assertNotEqual(physical.returncode, 0, physical.stdout)
        self.assertIn("routed contract mismatch", physical.stdout)

        base = self._preflight(
            RL / "train_navrl_v2_search.sh", cwd=ROOT,
            **{
                **common,
                "NAVRL_V2_PHYSICAL_FRESH_CHILD": "1",
                "NAVRL_V2_ROUTED_CHILD": "1",
                "NAVRL_PHYSICAL_ROUTED_CHILD": "0",
                "NAVRL_TARGET_ROUTE_RESOLUTION_M": "1.75",
            },
        )
        self.assertNotEqual(base.returncode, 0, base.stdout)
        self.assertIn("routed contract mismatch", base.stdout)

    def test_real_routed_training_requires_clean_source_receipt(self):
        self.assertIn("create_navrl_source_bundle.py", ROUTED_LAUNCHER)
        self.assertIn("--require-clean", ROUTED_LAUNCHER)
        self.assertIn("export NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT=1", ROUTED_LAUNCHER)
        self.assertIn("export NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE=1", ROUTED_LAUNCHER)
        self.assertIn("_validate_routed_source_contract", PHYSICAL_LAUNCHER)
        self.assertIn("_validate_routed_source_contract", V2_LAUNCHER)

    def test_fixed_box_support_and_non_env_gains_are_source_bound(self):
        self.assertIn("physical_attitude_kp = [0.08, 0.08, 0.04]", TASK_CONFIG_SOURCE)
        self.assertIn("physical_rate_kp = [0.04, 0.04, 0.03]", TASK_CONFIG_SOURCE)
        self.assertIn("physical_box_xyz = [physical_box_xy, physical_box_xy, 0.12]", TASK_CONFIG_SOURCE)
        self.assertIn("conservative_xy_support_from_box(self.tm.physical_box_xyz)", TASK_SOURCE)

    def test_all_launcher_handoffs_use_script_dir(self):
        for source in (ROUTED_LAUNCHER, PHYSICAL_LAUNCHER, V2_LAUNCHER):
            self.assertIn('SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"', source)
        self.assertIn('exec "${SCRIPT_DIR}/train_navrl_physical_fresh.sh"', ROUTED_LAUNCHER)
        self.assertIn('exec "${SCRIPT_DIR}/train_navrl_v2_search.sh"', PHYSICAL_LAUNCHER)
        self.assertIn('exec "${SCRIPT_DIR}/train_navrl.sh"', V2_LAUNCHER)

    def test_v2_pins_timing_geometry_and_motion_distribution(self):
        required = {
            "NAVRL_ARENA_XY": "40",
            "NAVRL_ARENA_Z": "3",
            "NAVRL_BAR_POOL": "bars_h3",
            "NAVRL_PLACEMENT_MODE": "navrl_band",
            "NAVRL_EPISODE_LEN_STEPS": "600",
            "NAVRL_MAX_BARS": "300",
            "NAVRL_TARGET_SPEED_MIN": "0.3",
            "NAVRL_TARGET_SPEED_FINAL": "1.5",
            "NAVRL_TARGET_PATTERN": "mixed",
        }
        for name, value in required.items():
            self.assertRegex(V2_LAUNCHER, rf"export {name}=(?:\"\$\{{[^\n]+:-)?{re.escape(value)}")


if __name__ == "__main__":
    unittest.main()
