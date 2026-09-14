import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


_ROOT = Path(__file__).parents[1]
_MODULE_PATH = (
    _ROOT
    / "aerial_gym/rl_training/rl_games/navrl_checkpoint_preflight.py"
)
_SPEC = importlib.util.spec_from_file_location("navrl_checkpoint_preflight_standalone", _MODULE_PATH)
_PREFLIGHT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PREFLIGHT)


def _checkpoint():
    return {
        "epoch": 15000,
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {0: {"exp_avg": torch.tensor([0.1])}}},
        "assymetric_vf_nets": {"weight": torch.tensor([2.0])},
        "assymetric_vf_optimizer": {"state": {0: {"exp_avg": torch.tensor([0.2])}}},
        "env_state": {
            "num_task_steps": 480000,
            "n_bars_active": 65,
            "k_max_cur": 16.0,
            "cfg_lidar_max_range": 12.0,
            "cfg_max_obstacles": 8,
            "cfg_token_fov_deg": 240.0,
            "cfg_obstacle_suppress_deg": 10.0,
            "cfg_lidar_hbeams": 72,
            "cfg_lidar_vbeams": 4,
        },
    }


class NavRLCheckpointPreflightTest(unittest.TestCase):
    def _save(self, checkpoint):
        tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        torch.save(checkpoint, tmp.name)
        return tmp.name

    def test_valid_density_resume(self):
        path = self._save(_checkpoint())
        info = _PREFLIGHT.inspect_checkpoint(
            path,
            max_epochs=45000,
            density_final=110,
            expected_contract={"cfg_token_fov_deg": 240.0, "cfg_max_obstacles": 8},
        )
        self.assertEqual(info["epoch"], 15000)
        self.assertEqual(info["bars"], 65)

    def test_max_epochs_must_extend_checkpoint(self):
        path = self._save(_checkpoint())
        with self.assertRaisesRegex(_PREFLIGHT.CheckpointPreflightError, "must exceed"):
            _PREFLIGHT.inspect_checkpoint(path, max_epochs=15000, density_final=110)

    def test_density_stage_requires_saturated_distance_curriculum(self):
        checkpoint = _checkpoint()
        checkpoint["env_state"]["k_max_cur"] = 14.0
        path = self._save(checkpoint)
        with self.assertRaisesRegex(
            _PREFLIGHT.CheckpointPreflightError, "distance curriculum is not saturated"
        ):
            _PREFLIGHT.inspect_checkpoint(
                path,
                max_epochs=45000,
                density_final=110,
                min_k_max=16.0,
            )

    def test_density_stage_requires_saturated_time_curricula(self):
        checkpoint = _checkpoint()
        checkpoint["env_state"]["num_task_steps"] = 95999
        path = self._save(checkpoint)
        with self.assertRaisesRegex(
            _PREFLIGHT.CheckpointPreflightError, "time-based curricula are not saturated"
        ):
            _PREFLIGHT.inspect_checkpoint(
                path,
                max_epochs=45000,
                density_final=110,
                min_task_steps=96000,
            )

    def test_policy_contract_mismatch_is_rejected(self):
        path = self._save(_checkpoint())
        with self.assertRaisesRegex(_PREFLIGHT.CheckpointPreflightError, "mismatch"):
            _PREFLIGHT.inspect_checkpoint(
                path,
                max_epochs=45000,
                density_final=110,
                expected_contract={"cfg_token_fov_deg": 360.0},
            )

    def test_intentional_same_shape_suppression_override_is_recorded(self):
        path = self._save(_checkpoint())
        info = _PREFLIGHT.inspect_checkpoint(
            path,
            max_epochs=45000,
            density_final=110,
            expected_contract={
                "cfg_token_fov_deg": 240.0,
                "cfg_obstacle_suppress_deg": 15.0,
            },
            allowed_contract_overrides={"cfg_obstacle_suppress_deg"},
        )
        self.assertEqual(
            info["contract_overrides"]["cfg_obstacle_suppress_deg"],
            {"checkpoint": 10.0, "requested": 15.0},
        )

    def test_legacy_checkpoint_implies_greedy_selector(self):
        path = self._save(_checkpoint())
        info = _PREFLIGHT.inspect_checkpoint(
            path,
            max_epochs=45000,
            density_final=110,
            expected_contract={"cfg_obstacle_selector": "greedy_suppress"},
        )
        self.assertEqual(info["contract_overrides"], {})

    def test_cluster_selector_same_shape_override_records_missing_provenance(self):
        path = self._save(_checkpoint())
        info = _PREFLIGHT.inspect_checkpoint(
            path,
            max_epochs=45000,
            density_final=110,
            expected_contract={
                "cfg_obstacle_selector": "cluster_sector",
                "cfg_obstacle_cluster_gap_m": 0.45,
                "cfg_obstacle_sectors": 8,
            },
            allowed_contract_overrides={
                "cfg_obstacle_selector",
                "cfg_obstacle_cluster_gap_m",
                "cfg_obstacle_sectors",
            },
        )
        self.assertEqual(
            info["contract_overrides"]["cfg_obstacle_selector"],
            {"checkpoint": "greedy_suppress", "requested": "cluster_sector"},
        )
        self.assertEqual(
            info["contract_overrides"]["cfg_obstacle_cluster_gap_m"],
            {"checkpoint": None, "requested": 0.45},
        )
        self.assertEqual(
            info["contract_overrides"]["cfg_obstacle_sectors"],
            {"checkpoint": None, "requested": 8},
        )

    def test_selector_mismatch_is_rejected_without_override(self):
        path = self._save(_checkpoint())
        with self.assertRaisesRegex(
            _PREFLIGHT.CheckpointPreflightError, "cfg_obstacle_selector"
        ):
            _PREFLIGHT.inspect_checkpoint(
                path,
                max_epochs=45000,
                density_final=110,
                expected_contract={"cfg_obstacle_selector": "cluster_sector"},
            )

    def test_override_does_not_relax_unlisted_contract_fields(self):
        path = self._save(_checkpoint())
        with self.assertRaisesRegex(_PREFLIGHT.CheckpointPreflightError, "cfg_token_fov_deg"):
            _PREFLIGHT.inspect_checkpoint(
                path,
                max_epochs=45000,
                density_final=110,
                expected_contract={
                    "cfg_token_fov_deg": 360.0,
                    "cfg_obstacle_suppress_deg": 15.0,
                },
                allowed_contract_overrides={"cfg_obstacle_suppress_deg"},
            )

    def test_first_corridor_checkpoint_implies_original_geometry_defaults(self):
        checkpoint = _checkpoint()
        checkpoint["env_state"]["cfg_corridor_tokens"] = 6
        path = self._save(checkpoint)
        info = _PREFLIGHT.inspect_checkpoint(
            path,
            max_epochs=45000,
            density_final=110,
            expected_contract={
                "cfg_corridor_tokens": 6,
                "cfg_corridor_horizon_m": 6.0,
                "cfg_corridor_min_width_m": 0.55,
            },
        )
        self.assertEqual(info["contract_overrides"], {})

    def test_corridor_geometry_change_is_rejected_without_override(self):
        checkpoint = _checkpoint()
        checkpoint["env_state"].update(
            {
                "cfg_corridor_tokens": 6,
                "cfg_corridor_horizon_m": 6.0,
                "cfg_corridor_min_width_m": 0.55,
            }
        )
        path = self._save(checkpoint)
        with self.assertRaisesRegex(
            _PREFLIGHT.CheckpointPreflightError, "cfg_corridor_horizon_m"
        ):
            _PREFLIGHT.inspect_checkpoint(
                path,
                max_epochs=45000,
                density_final=110,
                expected_contract={"cfg_corridor_horizon_m": 5.0},
            )

    def test_legacy_checkpoint_implies_geofence_disabled(self):
        path = self._save(_checkpoint())
        info = _PREFLIGHT.inspect_checkpoint(
            path,
            max_epochs=45000,
            density_final=110,
            expected_contract={"cfg_geofence_actor": 0},
        )
        self.assertEqual(info["contract_overrides"], {})

    def test_geofence_schema_mismatch_is_rejected(self):
        path = self._save(_checkpoint())
        with self.assertRaisesRegex(
            _PREFLIGHT.CheckpointPreflightError, "cfg_geofence_actor"
        ):
            _PREFLIGHT.inspect_checkpoint(
                path,
                max_epochs=45000,
                density_final=110,
                expected_contract={"cfg_geofence_actor": 1},
            )

    def test_nonfinite_checkpoint_is_rejected(self):
        checkpoint = _checkpoint()
        checkpoint["optimizer"]["state"][0]["exp_avg"] = torch.tensor([float("nan")])
        path = self._save(checkpoint)
        with self.assertRaisesRegex(_PREFLIGHT.CheckpointPreflightError, "non-finite"):
            _PREFLIGHT.inspect_checkpoint(path, max_epochs=45000, density_final=110)


if __name__ == "__main__":
    unittest.main()
