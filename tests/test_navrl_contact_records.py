"""Plan I1/I2: per-contact rows must reproduce the frozen category rule and the H2/H3 features."""

import json
import math
from pathlib import Path
import tempfile
import unittest

import importlib.util

import torch

TASK = Path(__file__).resolve().parents[1] / "aerial_gym/task/navrl_task/navrl_task.py"
# Load by path: importing the aerial_gym package pulls in isaacgym, which these CPU tests avoid.
_SPEC = importlib.util.spec_from_file_location(
    "contact_records_for_test", TASK.with_name("contact_records.py"))
CR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CR)


class CategoryPriority(unittest.TestCase):
    def test_priority_matches_the_preregistered_order(self):
        # rows: pure in_corridor, no_return, lateral, behind, vertical_out, and a row where all
        # flags are set (vertical_out must win), and one where lateral+no_return (lateral wins).
        vo = torch.tensor([0, 0, 0, 0, 1, 1, 0], dtype=torch.bool)
        be = torch.tensor([0, 0, 0, 1, 0, 1, 0], dtype=torch.bool)
        la = torch.tensor([0, 0, 1, 0, 0, 1, 1], dtype=torch.bool)
        nr = torch.tensor([0, 1, 0, 0, 0, 1, 1], dtype=torch.bool)
        got = CR.category_index(vo, be, la, nr).tolist()
        names = [CR.CATEGORIES[i] for i in got]
        self.assertEqual(names, ["in_corridor", "no_return", "lateral", "behind",
                                 "vertical_out", "vertical_out", "lateral"])


class SideGaps(unittest.TestCase):
    def test_left_right_sectors_and_surface_subtraction(self):
        # one row, four bars: dead ahead (excluded), left at 90 deg, right at -45 deg, behind (excluded)
        dist = torch.tensor([[2.0, 1.5, 3.0, 1.0]])
        brg = torch.tensor([[0.0, math.pi / 2, -math.pi / 4, math.pi]])
        circ = torch.tensor([[0.3, 0.5, 0.4, 0.3]])
        left, right, nearest = CR.side_gaps(dist, brg, circ, max_range=12.0)
        self.assertAlmostEqual(float(left), 1.0, places=6)     # 1.5 - 0.5
        self.assertAlmostEqual(float(right), 2.6, places=6)    # 3.0 - 0.4
        self.assertAlmostEqual(float(nearest), 0.7, places=6)  # behind bar 1.0 - 0.3
        # sector edges: 14 deg is "ahead", 166 deg is "behind"
        brg2 = torch.tensor([[math.radians(14.0), math.radians(166.0)]])
        l2, r2, _ = CR.side_gaps(torch.tensor([[1.0, 1.0]]), brg2, torch.zeros(1, 2), 12.0)
        self.assertEqual(float(l2), 12.0)
        self.assertEqual(float(r2), 12.0)


class MemoryWindow(unittest.TestCase):
    def test_seen_then_lost_signature(self):
        seen = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
        in_fov_now = torch.tensor([0, 0, 1], dtype=torch.bool)
        frac, lost = CR.memory_window(seen, in_fov_now)
        self.assertEqual([round(x, 2) for x in frac.tolist()], [0.5, 0.0, 0.25])
        self.assertEqual(lost.tolist(), [True, False, False])

    def test_empty_window_is_safe(self):
        frac, lost = CR.memory_window(torch.zeros(2, 0, dtype=torch.bool), torch.tensor([1, 0], dtype=torch.bool))
        self.assertEqual(frac.tolist(), [0.0, 0.0])
        self.assertEqual(lost.tolist(), [False, False])


class RowsAndJsonl(unittest.TestCase):
    def test_columns_to_rows_and_roundtrip(self):
        cols = {"env": torch.tensor([3, 7]), "flag": torch.tensor([True, False]),
                "x": torch.tensor([1.23456, float("nan")]), "name": ["a", "b"]}
        rows = CR.tensor_columns_to_rows(cols)
        self.assertEqual(rows[0], {"env": 3, "flag": True, "x": 1.2346, "name": "a"})
        self.assertIsNone(rows[1]["x"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sub" / "c.jsonl"
            n, sha = CR.rows_to_jsonl(path, rows, extra={"mode": "riskcap"})
            self.assertEqual(n, 2)
            lines = path.read_text().splitlines()
            self.assertEqual(json.loads(lines[0])["mode"], "riskcap")
            self.assertEqual(json.loads(lines[1])["x"], None)
            self.assertEqual(len(sha), 64)

    def test_ragged_columns_refused(self):
        with self.assertRaises(ValueError):
            CR.tensor_columns_to_rows({"a": torch.zeros(2), "b": torch.zeros(3)})


class TaskWiring(unittest.TestCase):
    """The task must (a) push governor telemetry into the ring buffer after the governor runs,
    (b) append a contact row per contact, (c) reference the sidecars from the payload."""

    def setUp(self):
        self.src = TASK.read_text()

    def test_governor_push_follows_apply(self):
        apply_at = self.src.index("governed, telemetry = apply_speed_governor(")
        push_at = self.src.index("self._push_contact_geometry_governor(telemetry)")
        self.assertGreater(push_at, apply_at)
        self.assertLess(push_at - apply_at, 600)

    def test_contact_rows_and_payload_wired(self):
        self.assertIn("self._cg_contact_rows.append(CR.tensor_columns_to_rows({", self.src)
        self.assertIn('"seen_then_lost": seen_then_lost & mem_ok', self.src)
        self.assertIn("**self._contact_records_payload(),", self.src)
        self.assertIn('.contact_records.jsonl', self.src)
        self.assertIn('.frame_samples.jsonl', self.src)

    def test_contact_instant_columns_are_recorded(self):
        for column in ("gap_min_t05", "gap_left_t0", "gap_right_t0", "nearest_surface_t0", "pinch_t0"):
            self.assertIn(f'"{column}"', self.src, column)
        self.assertIn("pinch_t0 = (left_t0 < CR.PINCH_M) & (right_t0 < CR.PINCH_M)", self.src)
        self.assertEqual(CR.PINCH_M, 0.65)

    def test_memory_window_covers_the_policy_history(self):
        # 5 slots x 0.5 s = 2.5 s at 10 Hz -> 25 steps
        self.assertIn("self._CG_MEMORY_STEPS = 25", self.src)


if __name__ == "__main__":
    unittest.main()
