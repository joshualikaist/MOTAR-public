"""CPU synthetic contracts for the offline topology difficulty labeler."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/analyze_navrl_topology_labels.py"
SPEC = importlib.util.spec_from_file_location("topology_labels", SCRIPT)
TOPOLOGY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOPOLOGY
SPEC.loader.exec_module(TOPOLOGY)


def layout(name, bars, start=(2.0, 5.0), goal=(18.0, 5.0), arena=(20.0, 10.0)):
    centers = np.asarray([bar[:2] for bar in bars], dtype=float).reshape((-1, 2))
    sizes = np.asarray([bar[2:] for bar in bars], dtype=float).reshape((-1, 2))
    return TOPOLOGY.Layout(
        layout_id=name,
        arena_min_xy_m=np.array([0.0, 0.0]),
        arena_max_xy_m=np.asarray(arena, dtype=float),
        start_xy_m=np.asarray(start, dtype=float),
        goal_xy_m=np.asarray(goal, dtype=float),
        bars_xy_m=centers,
        bars_size_xy_m=sizes,
        bar_size_source="snapshot_exact",
        source_metadata={},
    )


def label(item, sensor=4.0):
    return TOPOLOGY.label_layout(
        item,
        resolution_m=0.10,
        vehicle_half_width_m=0.20,
        side_clearance_m=0.10,
        sensor_range_m=sensor,
        cluster_gap_m=0.45,
        snap_radius_m=0.60,
    )


class TopologyLabelsTest(unittest.TestCase):
    def test_open_layout_has_direct_path_and_no_dead_end(self):
        result = label(
            layout("open", [], start=(10.0, 10.0), goal=(18.0, 10.0), arena=(20.0, 20.0))
        )
        self.assertTrue(result["path_exists"])
        self.assertLess(result["shortest_path_detour_ratio"], 1.02)
        self.assertFalse(result["local_culdesac_proxy"])
        self.assertEqual(result["obstacle_count_within_sensor_range"], 0)

    def test_disconnected_wall_has_no_path(self):
        # One exact full-height rectangle; inflation also seals the arena edges.
        result = label(layout("wall", [(10.0, 5.0, 0.6, 10.0)]))
        self.assertFalse(result["path_exists"])
        self.assertIsNone(result["shortest_path_detour_ratio"])

    def test_corridor_is_not_mislabeled_as_culdesac(self):
        # Horizontal walls create two opposed local exit arcs.
        bars = [(10.0, 2.0, 20.0, 0.5), (10.0, 8.0, 20.0, 0.5)]
        result = label(layout("corridor", bars, start=(10.0, 5.0), goal=(18.0, 5.0)))
        self.assertTrue(result["path_exists"])
        self.assertGreaterEqual(result["local_exit_arc_count"], 2)
        self.assertFalse(result["local_culdesac_proxy"])

    def test_u_shape_with_opening_behind_start_is_culdesac_like(self):
        # Closed on +x, bounded above/below, with the sole local exit toward -x.
        bars = [
            (12.0, 5.0, 0.5, 6.0),
            (9.0, 2.0, 6.0, 0.5),
            (9.0, 8.0, 6.0, 0.5),
        ]
        result = label(layout("dead-end", bars, start=(10.0, 5.0), goal=(18.0, 5.0)), sensor=4.0)
        self.assertEqual(result["local_exit_arc_count"], 1)
        self.assertTrue(result["local_culdesac_proxy"])
        self.assertGreater(result["local_dead_end_severity_proxy"], 0.5)

    def test_visible_obstacles_are_clustered_by_surface_gap(self):
        bars = [
            (4.0, 5.0, 0.6, 0.6),
            (4.8, 5.0, 0.6, 0.6),  # 0.2 m surface gap: same cluster
            (6.5, 5.0, 0.6, 0.6),  # separate cluster
            (15.0, 5.0, 0.6, 0.6), # out of sensor range
        ]
        result = label(layout("clusters", bars), sensor=5.0)
        self.assertEqual(result["obstacle_count_within_sensor_range"], 3)
        self.assertEqual(result["cluster_count_within_sensor_range"], 2)

    def test_metadata_makes_geometry_contract_explicit(self):
        result = label(layout("metadata", []))
        metadata = result["metadata"]
        self.assertEqual(metadata["grid_resolution_m"], 0.10)
        self.assertEqual(metadata["vehicle_half_width_m"], 0.20)
        self.assertEqual(metadata["side_clearance_m"], 0.10)
        self.assertAlmostEqual(metadata["inflation_m"], 0.30)
        self.assertEqual(metadata["cluster_surface_gap_m"], 0.45)


if __name__ == "__main__":
    unittest.main()
