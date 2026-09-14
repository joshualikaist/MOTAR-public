"""Unit tests for the pooled seed-replication analysis (WORKLOG 2026-09-06, R-B).

CPU-only and hermetic: the statistics are checked against hand-computed values and the loader
against synthesized result JSON, so deleting a result root cannot turn these red. The one test that
reaches outside asserts the tool's thresholds still match the FROZEN predictions in docs/specs --
if a threshold is ever edited without editing the preregistration, that test fails.

Run: PYTHONNOUSERSITE=1 python tests/test_navrl_seed_replication.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stats = _load("navrl_stats", ROOT / "tools/navrl_stats.py")
pool = _load("pool_navrl_seed_replication", ROOT / "tools/pool_navrl_seed_replication.py")
grid = _load("summarize_navrl_grid", ROOT / "tools/summarize_navrl_grid.py")


def _cell(seed, bars, mode, width, crash, total=2049, captured=None, checkpoint="f" * 64):
    return {
        "actual_episodes": total,
        "checkpoint_sha256": checkpoint,
        "condition": {
            "seed": seed,
            "bars": bars,
            "speed_governor_mode": mode,
            "speed_governor_half_width_m": width,
        },
        "outcome": {
            "crash": crash,
            "crash_rate": crash / total,
            "captured": total - crash if captured is None else captured,
            "capture_rate": (total - crash if captured is None else captured) / total,
            "timeout": 0,
            "timeout_rate": 0.0,
        },
    }


def _write_root(directory, cells):
    root = Path(directory)
    for name, cell in cells.items():
        cell_dir = root / name
        cell_dir.mkdir(parents=True)
        (cell_dir / f"{cell['condition']['bars']}bars.json").write_text(json.dumps(cell))
    return root


class StatisticsTest(unittest.TestCase):
    def test_wald_matches_hand_computation(self):
        delta, se = stats.wald_diff(92, 2049, 117, 2050)
        pa, pb = 92 / 2049, 117 / 2050
        self.assertAlmostEqual(delta, 100 * (pa - pb), places=12)
        self.assertAlmostEqual(
            se, 100 * math.sqrt(pa * (1 - pa) / 2049 + pb * (1 - pb) / 2050), places=12
        )
        lo, hi = stats.ci(delta, se)
        self.assertLess(lo, delta)
        self.assertGreater(hi, delta)

    def test_counts_not_rates_drive_the_contrast(self):
        # A rate rounded on the way in would move the estimate; the recorded count must win.
        outcome = {"crash": 285, "crash_rate": 0.1389, "captured": 1706, "capture_rate": 0.83}
        self.assertEqual(stats.outcome_count(outcome, "crash_rate"), 285)
        self.assertEqual(stats.outcome_count(outcome, "capture_rate"), 1706)
        with self.assertRaises(KeyError):
            stats.outcome_count(outcome, "closest_nocrash_mean_m")

    def test_pooling_identical_estimates_keeps_delta_and_shrinks_se(self):
        estimates = [(-1.5, 0.6)] * 4
        delta, se = stats.pool_fixed(estimates)
        self.assertAlmostEqual(delta, -1.5, places=12)
        self.assertAlmostEqual(se, 0.6 / math.sqrt(4), places=12)

    def test_pooling_weights_the_precise_estimate_more(self):
        delta, _ = stats.pool_fixed([(-4.0, 0.5), (0.0, 2.0)])
        self.assertLess(delta, -3.0)

    def test_heterogeneity_is_zero_when_estimates_agree(self):
        q, df, i2 = stats.cochran_q([(-1.5, 0.6)] * 3)
        self.assertAlmostEqual(q, 0.0, places=12)
        self.assertEqual(df, 2)
        self.assertEqual(i2, 0.0)

    def test_random_effect_matches_fixed_when_homogeneous_and_widens_otherwise(self):
        homogeneous = [(-1.5, 0.6)] * 3
        fixed = stats.pool_fixed(homogeneous)
        rdelta, rse, tau2 = stats.pool_random(homogeneous)
        self.assertAlmostEqual(tau2, 0.0, places=12)
        self.assertAlmostEqual(rse, fixed[1], places=12)

        spread = [(-6.0, 0.5), (-1.0, 0.5), (-3.0, 0.5)]
        self.assertGreater(stats.pool_random(spread)[2], 0.0)
        self.assertGreater(stats.pool_random(spread)[1], stats.pool_fixed(spread)[1])

    def test_excludes_zero_reads_the_interval(self):
        self.assertTrue(stats.excludes_zero(-2.58, 0.70))
        self.assertFalse(stats.excludes_zero(-1.22, 0.69))

    def test_holm_is_monotone_and_matches_worked_example(self):
        adjusted = stats.holm([0.01, 0.04, 0.03])
        self.assertAlmostEqual(adjusted[0], 0.03, places=12)
        self.assertAlmostEqual(adjusted[1], 0.06, places=12)
        self.assertAlmostEqual(adjusted[2], 0.06, places=12)
        self.assertTrue(all(a >= p for a, p in zip(adjusted, [0.01, 0.04, 0.03])))

    def test_sign_test_reproduces_the_five_of_five_figure(self):
        # WORKLOG 2026-09-06: five densities all favouring the arc is p = 0.0625, not significance.
        self.assertAlmostEqual(stats.sign_test_p(5, 5), 0.0625, places=12)
        self.assertAlmostEqual(stats.sign_test_p(0, 5), 0.0625, places=12)


class SeedLevelUnitTest(unittest.TestCase):
    """The 2026-09-07 audit's objection: cells inside a seed are not independent replicates."""

    def test_student_t_matches_a_known_quantile(self):
        self.assertAlmostEqual(stats.student_t_ppf(0.975, 2), 4.302653, places=5)
        self.assertAlmostEqual(stats.student_t_ppf(0.975, 10), 2.228139, places=5)
        self.assertAlmostEqual(stats.student_t_sf(0.0, 5), 0.5, places=12)

    def test_seed_level_reproduces_the_audited_main_effect(self):
        # Per-seed pooled arc - riskcap, from the three R-B seeds (523, 527, 531).
        mean, se, lo, hi, p, df = stats.seed_level_t([-1.1482, -1.8886, -1.4459])
        self.assertAlmostEqual(mean, -1.4942, places=3)
        self.assertAlmostEqual(se, 0.2151, places=3)
        self.assertAlmostEqual(lo, -2.4197, places=3)
        self.assertAlmostEqual(hi, -0.5687, places=3)
        self.assertAlmostEqual(p, 0.0201, places=3)
        self.assertEqual(df, 2)
        self.assertLess(hi, 0.0, "the main effect must still exclude zero at the seed level")

    def test_seed_level_is_wider_than_the_cell_level_interval(self):
        """Three observations buy less precision than fifteen; the report must show that cost."""
        per_seed = [-1.1482, -1.8886, -1.4459]
        _, _, lo, hi, _, _ = stats.seed_level_t(per_seed)
        cell_delta, cell_se = stats.pool_fixed([(-1.15, 0.36), (-1.89, 0.36), (-1.45, 0.36)])
        cell_lo, cell_hi = stats.ci(cell_delta, cell_se)
        self.assertGreater(hi - lo, cell_hi - cell_lo)

    def test_the_local_seventy_bar_finding_does_not_survive(self):
        # stopcap - riskcap at 70 bars, per seed. Significant pooled over cells, not over seeds.
        mean, _, lo, hi, p, _ = stats.seed_level_t([-0.34, -1.66, -1.61])
        self.assertLess(mean, 0.0)
        self.assertGreater(hi, 0.0, "this interval must cover zero; the claim is exploratory")
        self.assertGreater(p, 0.05)

    def test_two_replicates_is_the_minimum(self):
        with self.assertRaises(ValueError):
            stats.seed_level_t([-1.5])


class LoaderTest(unittest.TestCase):
    def test_repeated_condition_is_reported_not_dropped(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            _write_root(first, {"a": _cell(523, 70, "dwa_arc", 0.45, 92)})
            _write_root(second, {"b": _cell(523, 70, "dwa_arc", 0.45, 92)})
            cells, repeats = pool.load([first, second])
            self.assertEqual(len(cells), 1)
            self.assertEqual(len(repeats), 1)

    def test_two_checkpoints_for_one_condition_is_refused(self):
        cells = {
            ("aaaaaaaaaaaa", 523, 70, "dwa_arc", 0.45): _cell(523, 70, "dwa_arc", 0.45, 92, checkpoint="a" * 64),
            ("bbbbbbbbbbbb", 523, 70, "dwa_arc", 0.45): _cell(523, 70, "dwa_arc", 0.45, 80, checkpoint="b" * 64),
        }
        with self.assertRaises(SystemExit):
            pool.index_by_condition(cells)

    def test_contrast_pairs_only_within_a_seed_and_density(self):
        cells = {}
        for seed in (523, 527):
            for bars in (70, 205):
                cells[("f" * 12, seed, bars, "dwa_arc", 0.45)] = _cell(seed, bars, "dwa_arc", 0.45, 90)
                cells[("f" * 12, seed, bars, "riskcap", 0.45)] = _cell(seed, bars, "riskcap", 0.45, 120)
        # An arm evaluated at one density only must not borrow another density's baseline.
        cells[("f" * 12, 531, 130, "dwa_arc", 0.45)] = _cell(531, 130, "dwa_arc", 0.45, 90)
        rows = pool.contrast_rows(cells, ("dwa_arc", 0.45), ("riskcap", 0.45))
        self.assertEqual(sorted(rows), [(523, 70), (523, 205), (527, 70), (527, 205)])
        for delta, se in rows.values():
            self.assertLess(delta, 0.0)
            self.assertGreater(se, 0.0)


class PreregistrationSyncTest(unittest.TestCase):
    """The judged thresholds live in code; the frozen text lives in docs/specs. They must agree."""

    SPECS = sorted((ROOT / "docs/specs").glob("grid_r1_seedrep_ep25000_s*.json"))

    def test_specs_exist(self):
        self.assertTrue(self.SPECS, "R-B specs missing")

    def test_thresholds_appear_in_every_frozen_spec(self):
        expected = {
            "P1": ["-0.8"],
            "P2": ["4 of 5"],
            "P3": ["-2 pp at 70 bars", "-4 pp at 205 bars"],
            "P4": ["60%", "20%"],
            "P5": ["includes 0 at every density"],
        }
        self.assertEqual(sorted(expected), sorted(pool.PREDICTIONS))
        for path in self.SPECS:
            predictions = json.loads(path.read_text())["predictions"]
            for key, needles in expected.items():
                for needle in needles:
                    self.assertIn(needle, predictions[key], f"{path.name} {key}")

    def test_code_thresholds_match_the_text(self):
        self.assertEqual(pool.PREDICTIONS["P1"]["pooled_at_most"], -0.8)
        self.assertEqual(pool.PREDICTIONS["P2"]["negative_at_least"], 4)
        self.assertEqual(pool.PREDICTIONS["P2"]["of_densities"], 5)
        self.assertEqual(pool.PREDICTIONS["P3"]["crash_at_most"], {70: -2.0, 205: -4.0})
        self.assertEqual(pool.PREDICTIONS["P4"]["capture_below"], {70: 60.0, 205: 20.0})
        self.assertEqual(pool.PREDICTIONS["P4"]["arm"], ("stopcap", 1.2))


class GridSummaryAgreesWithPoolingTest(unittest.TestCase):
    """The progress view and the final table must read a cell the same way."""

    def test_grid_wald_derives_from_counts_not_the_stored_rate(self):
        # A rate rounded in the JSON must not move the contrast: the count is the record.
        a = _cell(523, 70, "dwa_arc", 0.45, 92)
        b = _cell(523, 70, "riskcap", 0.45, 117, total=2050)
        a["outcome"]["crash_rate"] = 0.04  # deliberately wrong, and deliberately ignored
        delta, lo, hi = grid.wald(a, b)
        expected, se = stats.wald_diff(92, 2049, 117, 2050)
        self.assertAlmostEqual(delta, expected, places=12)
        self.assertAlmostEqual(lo, stats.ci(expected, se)[0], places=12)
        self.assertAlmostEqual(hi, stats.ci(expected, se)[1], places=12)

    def test_grid_and_pool_report_the_same_contrast(self):
        a = _cell(527, 205, "dwa_arc", 0.45, 285, total=2051)
        b = _cell(527, 205, "riskcap", 0.45, 346, total=2050)
        delta, _, _ = grid.wald(a, b)
        cells = {("f" * 12, 527, 205, "dwa_arc", 0.45): a, ("f" * 12, 527, 205, "riskcap", 0.45): b}
        rows = pool.contrast_rows(cells, ("dwa_arc", 0.45), ("riskcap", 0.45))
        self.assertAlmostEqual(delta, rows[(527, 205)][0], places=12)


if __name__ == "__main__":
    unittest.main()
