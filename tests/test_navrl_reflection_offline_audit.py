"""CPU-only contract tests for tools/navrl_reflection_offline_audit.py.

No checkpoint, no Isaac Gym, no GPU, no simulator.  The audit tool keeps every ``aerial_gym`` and
``rl_games`` import inside functions, so importing it by file path (the pattern
tests/test_navrl_ref5in_platform.py uses to stay simulator-free) is enough.

What this guards.  The preregistration
``docs/prereg_2026-08-21_n1_real_frame_reflection_audit.md`` is frozen: its index sets, verdict
thresholds, percentile convention, minimum-sample rule and outcome-code map may not drift.  Every
expectation below is written out from that document independently of the tool, so a change to the
tool that quietly redefines one of them fails here.

Run: PYTHONNOUSERSITE=1 python tests/test_navrl_reflection_offline_audit.py
"""

import importlib.util
import inspect
from pathlib import Path
import sys
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_audit_module():
    path = REPO / "tools/navrl_reflection_offline_audit.py"
    spec = importlib.util.spec_from_file_location("navrl_reflection_offline_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()

# Preregistered schema of the frozen ref5in checkpoint (prereg section 5, Q3/Q4).
HBEAMS = 72
VBEAMS = 4
MAX_OBSTACLES = 8
OBS_DIM = 898


def _independent_expected_operator():
    """Build the preregistered signed permutation straight from the prereg section 5 bullet list.

    Deliberately written out by hand here rather than imported from the tool, so that the tool and
    the preregistration are compared rather than the tool being compared with itself.
    """
    source = list(range(OBS_DIM))
    sign = [1] * OBS_DIM

    # "permutation: [0:288], per ring v in {0,1,2,3} the index v*72 + h, h -> (-h) mod 72"
    for v in range(4):
        for h in range(72):
            source[v * 72 + h] = v * 72 + ((-h) % 72)

    # "sign flip (obstacle [288:768], (hist=5, slot=8, dim=12)): field 1, 4"
    for hist in range(5):
        for slot in range(8):
            base = 288 + (hist * 8 + slot) * 12
            sign[base + 1] = -1
            sign[base + 4] = -1

    # "sign flip (robot [768:818], (hist=5, dim=10)): field 1, 3, 5, 7"
    for hist in range(5):
        base = 768 + hist * 10
        for field in (1, 3, 5, 7):
            sign[base + field] = -1

    # "sign flip (target [818:898], (hist=5, dim=16)): field 1, 4"
    for hist in range(5):
        base = 818 + hist * 16
        sign[base + 1] = -1
        sign[base + 4] = -1

    return source, sign


class PreregisteredIndexSets(unittest.TestCase):
    def test_structured_obs_dim_is_898(self):
        self.assertEqual(AUDIT.structured_obs_dim(HBEAMS, VBEAMS, MAX_OBSTACLES), OBS_DIM)
        self.assertEqual(AUDIT.GATE_STRUCTURED_OBS_DIM, OBS_DIM)

    def test_signed_permutation_matches_preregistration(self):
        expected_source, expected_sign = _independent_expected_operator()
        source, sign = AUDIT.preregistered_signed_permutation(HBEAMS, VBEAMS, MAX_OBSTACLES)
        self.assertEqual(source, expected_source)
        self.assertEqual(sign, expected_sign)

    def test_sign_flip_index_set_matches_preregistration(self):
        _source, expected_sign = _independent_expected_operator()
        expected = set(i for i, s in enumerate(expected_sign) if s == -1)
        self.assertEqual(
            AUDIT.preregistered_sign_flip_indices(HBEAMS, VBEAMS, MAX_OBSTACLES), expected
        )
        # 5*8 obstacle tokens * 2 fields + 5 robot rows * 4 fields + 5 target rows * 2 fields.
        self.assertEqual(len(expected), 5 * 8 * 2 + 5 * 4 + 5 * 2)

    def test_permutation_pairs_match_preregistration(self):
        expected_source, _sign = _independent_expected_operator()
        expected = dict(
            (dest, src) for dest, src in enumerate(expected_source) if src != dest
        )
        self.assertEqual(
            AUDIT.preregistered_permutation_pairs(HBEAMS, VBEAMS, MAX_OBSTACLES), expected
        )
        # Every scan index except the two per-ring fixed points moves.
        self.assertEqual(len(expected), 4 * (72 - 2))

    def test_sign_flip_and_permutation_sets_are_disjoint(self):
        flips = AUDIT.preregistered_sign_flip_indices(HBEAMS, VBEAMS, MAX_OBSTACLES)
        moves = set(AUDIT.preregistered_permutation_pairs(HBEAMS, VBEAMS, MAX_OBSTACLES))
        self.assertEqual(flips & moves, set())

    def test_every_other_index_is_unchanged(self):
        source, sign = AUDIT.preregistered_signed_permutation(HBEAMS, VBEAMS, MAX_OBSTACLES)
        flips = AUDIT.preregistered_sign_flip_indices(HBEAMS, VBEAMS, MAX_OBSTACLES)
        moves = set(AUDIT.preregistered_permutation_pairs(HBEAMS, VBEAMS, MAX_OBSTACLES))
        for index in range(OBS_DIM):
            if index in flips or index in moves:
                continue
            self.assertEqual(source[index], index)
            self.assertEqual(sign[index], 1)

    def test_operator_is_an_involution(self):
        source, sign = AUDIT.preregistered_signed_permutation(HBEAMS, VBEAMS, MAX_OBSTACLES)
        for index in range(OBS_DIM):
            self.assertEqual(source[source[index]], index)
            self.assertEqual(sign[source[index]], sign[index])


class ScanPermutationGate(unittest.TestCase):
    def test_ring_map_is_negation_modulo_hbeams(self):
        ring = AUDIT.preregistered_scan_permutation(HBEAMS)
        self.assertEqual(ring, dict((h, (-h) % HBEAMS) for h in range(HBEAMS)))

    def test_fixed_points_are_exactly_zero_and_thirtysix(self):
        ring = AUDIT.preregistered_scan_permutation(HBEAMS)
        fixed = set(h for h in range(HBEAMS) if ring[h] == h)
        self.assertEqual(fixed, {0, 36})
        self.assertEqual(set(AUDIT.GATE_SCAN_FIXED_POINTS), {0, 36})

    def test_gate_scan_permutation_accepts_the_preregistered_operator(self):
        source, _sign = AUDIT.preregistered_signed_permutation(HBEAMS, VBEAMS, MAX_OBSTACLES)
        passed, detail = AUDIT.gate_scan_permutation(source, HBEAMS, VBEAMS)
        self.assertTrue(passed)
        self.assertEqual(detail["fixed_points"], [0, 36])

    def test_gate_scan_permutation_rejects_the_old_h_minus_2_minus_i_convention(self):
        source, _sign = AUDIT.preregistered_signed_permutation(HBEAMS, VBEAMS, MAX_OBSTACLES)
        broken = list(source)
        for v in range(VBEAMS):
            for h in range(HBEAMS):
                broken[v * HBEAMS + h] = v * HBEAMS + ((HBEAMS - 2 - h) % HBEAMS)
        passed, _detail = AUDIT.gate_scan_permutation(broken, HBEAMS, VBEAMS)
        self.assertFalse(passed)


class VerdictRule(unittest.TestCase):
    """Prereg section 7, including the exact boundary values."""

    def test_thresholds_have_the_preregistered_values(self):
        self.assertEqual(AUDIT.VERDICT_CONFIRM_MEDIAN_MIN, 0.30)
        self.assertEqual(AUDIT.VERDICT_CONFIRM_SIGN_AGREEMENT_MAX, 0.60)
        self.assertEqual(AUDIT.VERDICT_ABSENT_MEDIAN_MAX, 0.10)
        self.assertEqual(AUDIT.VERDICT_ABSENT_SIGN_AGREEMENT_MIN, 0.90)

    def test_confirmed_branch(self):
        self.assertEqual(
            AUDIT.classify_verdict(1.235, 1.0 - 0.7308), AUDIT.VERDICT_CHIRALITY_CONFIRMED
        )

    def test_absent_branch(self):
        self.assertEqual(AUDIT.classify_verdict(0.01, 0.99), AUDIT.VERDICT_CHIRALITY_ABSENT)

    def test_inconclusive_branch(self):
        self.assertEqual(AUDIT.classify_verdict(0.20, 0.75), AUDIT.VERDICT_INCONCLUSIVE)
        # One condition satisfied is not enough: both are ANDed.
        self.assertEqual(AUDIT.classify_verdict(0.50, 0.95), AUDIT.VERDICT_INCONCLUSIVE)
        self.assertEqual(AUDIT.classify_verdict(0.02, 0.10), AUDIT.VERDICT_INCONCLUSIVE)

    def test_exact_boundaries_are_inclusive_as_preregistered(self):
        # median >= 0.30 AND agreement <= 0.60  ->  CONFIRMED, at equality.
        self.assertEqual(AUDIT.classify_verdict(0.30, 0.60), AUDIT.VERDICT_CHIRALITY_CONFIRMED)
        # median <= 0.10 AND agreement >= 0.90  ->  ABSENT, at equality.
        self.assertEqual(AUDIT.classify_verdict(0.10, 0.90), AUDIT.VERDICT_CHIRALITY_ABSENT)
        # Just outside either box on one axis.
        self.assertEqual(
            AUDIT.classify_verdict(0.30, 0.60 + 1e-12), AUDIT.VERDICT_INCONCLUSIVE
        )
        self.assertEqual(
            AUDIT.classify_verdict(0.10 + 1e-12, 0.90), AUDIT.VERDICT_INCONCLUSIVE
        )

    def test_the_two_verdict_regions_cannot_both_be_satisfied(self):
        self.assertLess(AUDIT.VERDICT_ABSENT_MEDIAN_MAX, AUDIT.VERDICT_CONFIRM_MEDIAN_MIN)
        self.assertLess(
            AUDIT.VERDICT_CONFIRM_SIGN_AGREEMENT_MAX, AUDIT.VERDICT_ABSENT_SIGN_AGREEMENT_MIN
        )

    def test_missing_statistics_yield_no_verdict(self):
        self.assertIsNone(AUDIT.classify_verdict(None, 0.5))
        self.assertIsNone(AUDIT.classify_verdict(0.5, None))

    def test_verdict_rule_references_module_constants_not_literals(self):
        source = inspect.getsource(AUDIT.classify_verdict)
        for name in (
            "VERDICT_CONFIRM_MEDIAN_MIN",
            "VERDICT_CONFIRM_SIGN_AGREEMENT_MAX",
            "VERDICT_ABSENT_MEDIAN_MAX",
            "VERDICT_ABSENT_SIGN_AGREEMENT_MIN",
        ):
            self.assertIn(name, source)
        for literal in ("0.30", "0.60", "0.10", "0.90"):
            self.assertNotIn(literal, source)

    def test_constants_are_declared_above_the_measurement_code(self):
        text = (REPO / "tools/navrl_reflection_offline_audit.py").read_text(encoding="utf-8")
        constant_line = text.index("VERDICT_CONFIRM_MEDIAN_MIN = ")
        self.assertLess(constant_line, text.index("def classify_verdict"))
        self.assertLess(constant_line, text.index("def measure_cell"))


class PercentileConvention(unittest.TestCase):
    def test_convention_is_documented_and_linear(self):
        self.assertEqual(AUDIT.PERCENTILE_METHOD, "linear")
        self.assertIn("linear", AUDIT.PERCENTILE_CONVENTION)

    def test_matches_numpy_linear_interpolation(self):
        values = [0.0, 1.0, 2.0, 3.0]
        # Linear convention: position = q/100 * (n-1) = 0.9*3 = 2.7 -> 2.0 + 0.7*(3.0-2.0).
        self.assertAlmostEqual(AUDIT.percentile(values, 90.0), 2.7, places=12)
        for q in (50.0, 90.0, 95.0, 99.0):
            self.assertAlmostEqual(
                AUDIT.percentile(values, q),
                float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear")),
                places=12,
            )

    def test_empty_input_has_no_percentile(self):
        self.assertIsNone(AUDIT.percentile([], 50.0))

    def test_describe_reports_the_preregistered_statistics(self):
        stats = AUDIT.describe([0.0, 1.0, 2.0, 3.0])
        self.assertEqual(
            sorted(stats.keys()), sorted(["median", "p90", "p95", "p99", "mean", "max", "n"])
        )
        self.assertEqual(stats["n"], 4)
        self.assertAlmostEqual(stats["median"], 1.5, places=12)
        self.assertAlmostEqual(stats["mean"], 1.5, places=12)
        self.assertAlmostEqual(stats["max"], 3.0, places=12)

    def test_describe_on_empty_input_is_all_null(self):
        stats = AUDIT.describe([])
        self.assertEqual(stats["n"], 0)
        for key in ("median", "p90", "p95", "p99", "mean", "max"):
            self.assertIsNone(stats[key])


class MinimumSampleRule(unittest.TestCase):
    def test_threshold_is_256_comparable_rows(self):
        self.assertEqual(AUDIT.MIN_CONTEXT_COMPARABLE_ROWS, 256)

    def test_rule_is_inclusive_at_the_threshold(self):
        self.assertFalse(AUDIT.has_sufficient_sample(0))
        self.assertFalse(AUDIT.has_sufficient_sample(255))
        self.assertTrue(AUDIT.has_sufficient_sample(256))
        self.assertTrue(AUDIT.has_sufficient_sample(4096))

    def test_minimum_valid_frame_gate_is_4096(self):
        self.assertEqual(AUDIT.MIN_VALID_FRAMES, 4096)


class OutcomeCodeMap(unittest.TestCase):
    def test_map_is_a_literal_with_the_preregistered_codes(self):
        self.assertEqual(
            AUDIT.OUTCOME_CODES,
            {
                0: "capture",
                1: "crash_bar_contact",
                2: "crash_oob",
                3: "crash_other",
                4: "timeout",
                5: "unattributed",
            },
        )

    def test_map_is_written_as_a_literal_in_the_source(self):
        text = (REPO / "tools/navrl_reflection_offline_audit.py").read_text(encoding="utf-8")
        block = text[text.index("OUTCOME_CODES = {") : text.index("OUTCOME_CONTEXT_CODES")]
        for code, name in AUDIT.OUTCOME_CODES.items():
            self.assertIn('%d: "%s"' % (code, name), block)

    def test_only_the_five_preregistered_outcomes_are_context_cells(self):
        self.assertEqual(AUDIT.OUTCOME_CONTEXT_CODES, (0, 1, 2, 3, 4))
        self.assertNotIn(5, AUDIT.OUTCOME_CONTEXT_CODES)


class ContextSplits(unittest.TestCase):
    def test_first_order_splits_only_and_never_crossed(self):
        frames = {
            "ctx_target_visible": np.array([True, True, False, False]),
            "ctx_front_blocked": np.array([1, 0, -1, 1], dtype=np.int64),
        }
        outcomes = np.array([0, 1, 4, -1], dtype=np.int64)
        names = [name for name, _mask in AUDIT.build_contexts(frames, outcomes)]
        self.assertEqual(
            names,
            [
                "overall",
                "target_visible",
                "target_hidden",
                "front_blocked",
                "front_clear",
                "front_unknown",
                "outcome_capture",
                "outcome_crash_bar_contact",
                "outcome_crash_oob",
                "outcome_crash_other",
                "outcome_timeout",
            ],
        )
        masks = dict(AUDIT.build_contexts(frames, outcomes))
        self.assertEqual(masks["overall"].sum(), 4)
        self.assertEqual(masks["target_visible"].sum(), 2)
        self.assertEqual(masks["target_hidden"].sum(), 2)
        self.assertEqual(masks["front_blocked"].sum(), 2)
        self.assertEqual(masks["front_clear"].sum(), 1)
        self.assertEqual(masks["front_unknown"].sum(), 1)
        self.assertEqual(masks["outcome_capture"].sum(), 1)

    def test_outcome_join_marks_unknown_episodes_with_minus_one(self):
        episodes = {
            "ep_uid": np.array([10, 11], dtype=np.int64),
            "outcome": np.array([0, 4], dtype=np.int8),
        }
        joined = AUDIT.join_outcomes(np.array([11, 10, 99], dtype=np.int64), episodes)
        self.assertEqual(joined.tolist(), [4, 0, -1])


class CellMeasurement(unittest.TestCase):
    """Prereg section 6 definitions on synthetic, hand-checkable actions."""

    def test_perfectly_equivariant_policy_gives_zero_error_and_absent_verdict(self):
        rng = np.random.RandomState(0)
        original = rng.uniform(-1.0, 1.0, size=(512, 4))
        original[:, 1] = np.sign(original[:, 1]) * (np.abs(original[:, 1]) * 0.5 + 0.4)
        mirrored = original.copy()
        mirrored[:, 1] = -mirrored[:, 1]
        mirrored[:, 3] = -mirrored[:, 3]
        cell = AUDIT.measure_cell(original, mirrored, np.ones(512, dtype=bool))
        self.assertAlmostEqual(cell["channels"]["conj_err_lat"]["max"], 0.0, places=12)
        self.assertAlmostEqual(cell["channels"]["conj_err_yaw"]["max"], 0.0, places=12)
        self.assertEqual(cell["lateral_sign_agreement"], 1.0)
        self.assertAlmostEqual(cell["signed_lateral_bias"], 0.0, places=12)
        self.assertFalse(cell["insufficient_sample"])
        self.assertEqual(cell["verdict"], AUDIT.VERDICT_CHIRALITY_ABSENT)

    def test_fully_chiral_policy_gives_confirmed_verdict(self):
        # The policy emits the SAME lateral command in the reflected world: pi(Mo)[1] = pi(o)[1],
        # so e[1] = pi(Mo)[1] + pi(o)[1] = 2*pi(o)[1] and the sign never agrees.
        original = np.zeros((512, 4))
        original[:, 1] = 0.8
        mirrored = original.copy()
        cell = AUDIT.measure_cell(original, mirrored, np.ones(512, dtype=bool))
        self.assertAlmostEqual(cell["channels"]["conj_err_lat"]["median"], 1.6, places=12)
        self.assertEqual(cell["lateral_sign_agreement"], 0.0)
        self.assertAlmostEqual(cell["signed_lateral_bias"], 0.8, places=12)
        self.assertEqual(cell["verdict"], AUDIT.VERDICT_CHIRALITY_CONFIRMED)

    def test_rows_below_the_sign_threshold_are_not_comparable(self):
        original = np.zeros((300, 4))
        mirrored = np.zeros((300, 4))
        original[:, 1] = 0.04  # below LATERAL_SIGN_THRESHOLD on both sides
        mirrored[:, 1] = -0.04
        cell = AUDIT.measure_cell(original, mirrored, np.ones(300, dtype=bool))
        self.assertEqual(cell["lateral_sign_comparable"], 0)
        self.assertIsNone(cell["lateral_sign_agreement"])
        self.assertTrue(cell["insufficient_sample"])
        self.assertIsNone(cell["verdict"])

    def test_small_cell_reports_numbers_but_no_verdict(self):
        original = np.zeros((255, 4))
        original[:, 1] = 0.8
        mirrored = original.copy()
        cell = AUDIT.measure_cell(original, mirrored, np.ones(255, dtype=bool))
        self.assertEqual(cell["lateral_sign_comparable"], 255)
        self.assertTrue(cell["insufficient_sample"])
        self.assertIsNone(cell["verdict"])
        self.assertIsNotNone(cell["channels"]["conj_err_lat"]["median"])

    def test_error_uses_the_action_mirror_on_channels_one_and_three_only(self):
        original = np.array([[0.2, 0.3, 0.4, 0.5]])
        mirrored = np.array([[0.2, -0.3, 0.4, -0.5]])
        cell = AUDIT.measure_cell(original, mirrored, np.ones(1, dtype=bool))
        for name, _axis in AUDIT.ACTION_CHANNELS:
            self.assertAlmostEqual(cell["channels"][name]["max"], 0.0, places=12)


class SymmetrisedNormaliser(unittest.TestCase):
    """Prereg section 6 S1 -- exploratory and non-gating, but its algebra must be right."""

    def _operator(self):
        return AUDIT.preregistered_signed_permutation(HBEAMS, VBEAMS, MAX_OBSTACLES)

    def test_sign_flipped_fields_get_zero_mean_and_keep_their_variance(self):
        import torch

        source, sign = self._operator()
        mean = torch.arange(1, OBS_DIM + 1, dtype=torch.float64)
        var = torch.arange(1, OBS_DIM + 1, dtype=torch.float64) * 2.0
        sym_mean, sym_var = AUDIT.symmetrise_normaliser(mean, var, source, sign)
        flips = AUDIT.preregistered_sign_flip_indices(HBEAMS, VBEAMS, MAX_OBSTACLES)
        for index in sorted(flips)[:16]:
            self.assertEqual(float(sym_mean[index]), 0.0)
            self.assertEqual(float(sym_var[index]), float(var[index]))

    def test_permuted_pairs_share_the_average_mean_and_variance(self):
        import torch

        source, sign = self._operator()
        mean = torch.arange(1, OBS_DIM + 1, dtype=torch.float64)
        var = torch.arange(1, OBS_DIM + 1, dtype=torch.float64) * 2.0
        sym_mean, sym_var = AUDIT.symmetrise_normaliser(mean, var, source, sign)
        for dest, src in list(
            AUDIT.preregistered_permutation_pairs(HBEAMS, VBEAMS, MAX_OBSTACLES).items()
        )[:16]:
            self.assertAlmostEqual(
                float(sym_mean[dest]), (float(mean[dest]) + float(mean[src])) / 2.0, places=12
            )
            self.assertAlmostEqual(float(sym_mean[dest]), float(sym_mean[src]), places=12)
            self.assertAlmostEqual(
                float(sym_var[dest]), (float(var[dest]) + float(var[src])) / 2.0, places=12
            )

    def test_unchanged_indices_are_left_alone(self):
        import torch

        source, sign = self._operator()
        mean = torch.arange(1, OBS_DIM + 1, dtype=torch.float64)
        var = torch.arange(1, OBS_DIM + 1, dtype=torch.float64) * 2.0
        sym_mean, sym_var = AUDIT.symmetrise_normaliser(mean, var, source, sign)
        flips = AUDIT.preregistered_sign_flip_indices(HBEAMS, VBEAMS, MAX_OBSTACLES)
        moves = set(AUDIT.preregistered_permutation_pairs(HBEAMS, VBEAMS, MAX_OBSTACLES))
        untouched = [i for i in range(OBS_DIM) if i not in flips and i not in moves]
        for index in untouched[:16]:
            self.assertEqual(float(sym_mean[index]), float(mean[index]))
            self.assertEqual(float(sym_var[index]), float(var[index]))

    def test_symmetrised_normaliser_commutes_with_the_mirror(self):
        """The defining property: N(M x) == M N(x), clamp included (clamp is odd)."""
        import torch

        source, sign = self._operator()
        generator = torch.Generator().manual_seed(7)
        mean = torch.randn(OBS_DIM, generator=generator, dtype=torch.float64)
        var = torch.rand(OBS_DIM, generator=generator, dtype=torch.float64) + 0.5
        sym_mean, sym_var = AUDIT.symmetrise_normaliser(mean, var, source, sign)

        index = torch.as_tensor(source, dtype=torch.long)
        signs = torch.as_tensor(sign, dtype=torch.float64)

        def mirror(x):
            return x.index_select(1, index) * signs

        def normalise(x):
            y = (x - sym_mean) / torch.sqrt(sym_var + 1.0e-05)
            return torch.clamp(y, min=-5.0, max=5.0)

        x = torch.randn((8, OBS_DIM), generator=generator, dtype=torch.float64) * 3.0
        self.assertLess(float((normalise(mirror(x)) - mirror(normalise(x))).abs().max()), 1e-12)

    def test_raw_normaliser_does_not_commute_with_the_mirror(self):
        """Sanity: the property under test is not vacuous for the unsymmetrised statistics."""
        import torch

        source, sign = self._operator()
        generator = torch.Generator().manual_seed(11)
        mean = torch.randn(OBS_DIM, generator=generator, dtype=torch.float64)
        var = torch.rand(OBS_DIM, generator=generator, dtype=torch.float64) + 0.5
        index = torch.as_tensor(source, dtype=torch.long)
        signs = torch.as_tensor(sign, dtype=torch.float64)

        def mirror(x):
            return x.index_select(1, index) * signs

        def normalise(x):
            return (x - mean) / torch.sqrt(var + 1.0e-05)

        x = torch.randn((8, OBS_DIM), generator=generator, dtype=torch.float64)
        self.assertGreater(float((normalise(mirror(x)) - mirror(normalise(x))).abs().max()), 1e-3)


class MirrorFunctionAgreement(unittest.TestCase):
    """Q4/Q5 against the project's single canonical mirror, when it is importable CPU-only."""

    def setUp(self):
        import os
        import types

        package = sys.modules.get("aerial_gym")
        if package is None:
            package = types.ModuleType("aerial_gym")
            sys.modules["aerial_gym"] = package
        if not hasattr(package, "__path__"):
            package.__path__ = [str(REPO / "aerial_gym")]
        # mirror_navrl_structured_observation reads these at CALL time; pin the 898-D schema.
        os.environ["NAVRL_LIDAR_HBEAMS"] = str(HBEAMS)
        os.environ["NAVRL_LIDAR_VBEAMS"] = str(VBEAMS)
        os.environ["NAVRL_MAX_OBSTACLES"] = str(MAX_OBSTACLES)
        try:
            from aerial_gym.rl_training.rl_games.ppo_update_safety import (
                mirror_navrl_structured_observation,
            )
        except Exception as exc:  # pragma: no cover - environment without torch/rl_games
            raise unittest.SkipTest("canonical mirror not importable: %s" % exc)
        self.mirror = mirror_navrl_structured_observation

    def test_gate_index_sets_passes_on_the_canonical_mirror(self):
        passed, detail, source, sign = AUDIT.gate_index_sets(
            self.mirror, HBEAMS, VBEAMS, MAX_OBSTACLES
        )
        self.assertTrue(passed, detail)
        self.assertTrue(detail["byte_level_equality_on_random_tensor"])
        expected_source, expected_sign = _independent_expected_operator()
        self.assertEqual(source, expected_source)
        self.assertEqual(sign, expected_sign)

    def test_gate_scan_permutation_passes_on_the_canonical_mirror(self):
        _passed, _detail, source, _sign = AUDIT.gate_index_sets(
            self.mirror, HBEAMS, VBEAMS, MAX_OBSTACLES
        )
        passed, detail = AUDIT.gate_scan_permutation(source, HBEAMS, VBEAMS)
        self.assertTrue(passed, detail)


class NpzKeyBinding(unittest.TestCase):
    def test_required_keys_are_reported_when_missing(self):
        with self.assertRaises(AUDIT.AuditPreconditionError) as context:
            AUDIT._bind_keys({"obs": None}, AUDIT.FRAME_KEYS, AUDIT.REQUIRED_FRAME_KEYS)
        self.assertIn("ctx_valid", str(context.exception))

    def test_frame_and_episode_alias_lists_are_disjoint(self):
        frame_aliases = set()
        for aliases in AUDIT.FRAME_KEYS.values():
            frame_aliases.update(aliases)
        episode_aliases = set()
        for aliases in AUDIT.EPISODE_KEYS.values():
            episode_aliases.update(aliases)
        self.assertEqual(frame_aliases & episode_aliases, set())


class OutcomeJoinSentinel(unittest.TestCase):
    """H1 -- the join sentinel and outcome code 5 must not share a name in the payload."""

    def test_sentinel_has_its_own_name(self):
        self.assertEqual(AUDIT.NO_OUTCOME_ROW, -1)
        self.assertNotEqual(AUDIT.NO_OUTCOME_ROW_NAME, AUDIT.OUTCOME_CODES[5])
        self.assertNotIn(AUDIT.NO_OUTCOME_ROW_NAME, set(AUDIT.OUTCOME_CODES.values()))
        self.assertNotIn("unattributed", AUDIT.NO_OUTCOME_ROW_NAME)

    def test_the_ambiguous_payload_key_is_gone(self):
        text = (REPO / "tools/navrl_reflection_offline_audit.py").read_text(encoding="utf-8")
        self.assertNotIn("n_frames_unattributed_outcome", text)
        self.assertIn("n_frames_no_outcome_row", text)

    def test_join_reconciliation_reports_all_three_totals(self):
        codes = np.array([0, 0, 4, 1, -1, -1, 5], dtype=np.int64)
        report = AUDIT.check_outcome_join(codes, codes.size)
        self.assertEqual(report["n_valid_frames"], 7)
        self.assertEqual(report["n_frames_no_outcome_row"], 2)
        self.assertEqual(report["n_frames_with_outcome_row"], 5)
        self.assertTrue(report["reconciles"])
        self.assertEqual(report["sentinel_name"], AUDIT.NO_OUTCOME_ROW_NAME)
        # The code-5 count and the sentinel count are reported under DIFFERENT names.
        self.assertEqual(report["not_to_be_confused_with"]["name"], AUDIT.OUTCOME_CODES[5])
        self.assertEqual(report["not_to_be_confused_with"]["n_frames"], 1)
        # The five context cells exclude code 5, so they do not cover every joined frame.
        self.assertEqual(report["outcome_context_cell_total"], 4)
        self.assertEqual(report["correct_denominator_for_outcome_shares"], "n_frames_with_outcome_row")

    def test_join_that_does_not_reconcile_fails_closed(self):
        codes = np.array([0, -1, 4], dtype=np.int64)
        with self.assertRaises(AUDIT.IntegrityViolation):
            AUDIT.check_outcome_join(codes, 5)

    def test_outcome_code_outside_the_map_fails_closed(self):
        codes = np.array([0, 4, 9], dtype=np.int64)
        with self.assertRaises(AUDIT.IntegrityViolation) as context:
            AUDIT.check_outcome_join(codes, codes.size)
        self.assertIn("9", str(context.exception))

    def test_join_still_marks_missing_episodes_with_the_sentinel(self):
        episodes = {
            "ep_uid": np.array([10, 11], dtype=np.int64),
            "outcome": np.array([0, 4], dtype=np.int8),
        }
        joined = AUDIT.join_outcomes(np.array([11, 10, 99], dtype=np.int64), episodes)
        self.assertEqual(joined.tolist(), [4, 0, AUDIT.NO_OUTCOME_ROW])


class ContextPartitionCompleteness(unittest.TestCase):
    """H2 -- every context family must partition the population it claims to cover."""

    def _frames(self, front, visible=None):
        n = len(front)
        if visible is None:
            visible = np.zeros(n, dtype=bool)
        return {
            "ctx_target_visible": np.asarray(visible),
            "ctx_front_blocked": np.asarray(front, dtype=np.int64),
        }

    def test_complete_split_reconciles(self):
        frames = self._frames([1, 0, -1, 1], [True, True, False, False])
        outcomes = np.array([0, 1, 4, AUDIT.NO_OUTCOME_ROW], dtype=np.int64)
        contexts = AUDIT.build_contexts(frames, outcomes)
        report = AUDIT.check_context_partitions(contexts, 4, 1)
        self.assertTrue(report["partitions_complete"])
        self.assertEqual(report["families"]["front"]["cells_total"], 4)
        self.assertEqual(report["families"]["target"]["cells_total"], 4)
        self.assertEqual(report["families"]["outcome"]["expected_total"], 3)
        self.assertEqual(report["families"]["outcome"]["cells_total"], 3)
        for family in ("front", "target", "outcome"):
            self.assertTrue(report["families"][family]["reconciles"])
            self.assertTrue(report["families"][family]["cells_disjoint"])

    def test_an_out_of_domain_front_label_cannot_vanish_silently(self):
        # The concrete regression: a dump that later emits ctx_front_blocked == 2.  Those frames
        # belong to no front cell, and nothing used to notice.
        front = np.array([1, 0, -1, 2], dtype=np.int64)
        masks = {
            "overall": np.ones(4, dtype=bool),
            "target_visible": np.array([True, True, False, False]),
            "target_hidden": np.array([False, False, True, True]),
            "front_blocked": front == 1,
            "front_clear": front == 0,
            "front_unknown": front == -1,
        }
        for code in AUDIT.OUTCOME_CONTEXT_CODES:
            masks["outcome_" + AUDIT.OUTCOME_CODES[code]] = np.zeros(4, dtype=bool)
        masks["outcome_capture"] = np.array([True, True, True, True])
        with self.assertRaises(AUDIT.IntegrityViolation) as context:
            AUDIT.check_context_partitions(list(masks.items()), 4, 0)
        message = str(context.exception)
        self.assertIn("front", message)
        self.assertIn("NO cell", message)

    def test_overlapping_cells_fail_even_when_the_total_is_right(self):
        masks = {
            "overall": np.ones(4, dtype=bool),
            "target_visible": np.array([True, True, True, False]),
            "target_hidden": np.array([True, False, False, True]),
            "front_blocked": np.array([True, False, False, False]),
            "front_clear": np.array([False, True, True, False]),
            "front_unknown": np.array([False, False, False, True]),
        }
        for code in AUDIT.OUTCOME_CONTEXT_CODES:
            masks["outcome_" + AUDIT.OUTCOME_CODES[code]] = np.zeros(4, dtype=bool)
        masks["outcome_capture"] = np.ones(4, dtype=bool)
        self.assertEqual(
            int(masks["target_visible"].sum()) + int(masks["target_hidden"].sum()), 5
        )
        with self.assertRaises(AUDIT.IntegrityViolation):
            AUDIT.check_context_partitions(list(masks.items()), 4, 0)

    def test_outcome_family_population_is_n_valid_minus_the_excluded_frames(self):
        frames = self._frames([1, 0, -1, 1], [True, True, False, False])
        outcomes = np.array([0, 1, 4, AUDIT.NO_OUTCOME_ROW], dtype=np.int64)
        contexts = AUDIT.build_contexts(frames, outcomes)
        # Claiming no frames were excluded leaves the outcome family one frame short.
        with self.assertRaises(AUDIT.IntegrityViolation) as context:
            AUDIT.check_context_partitions(contexts, 4, 0)
        self.assertIn("outcome", str(context.exception))

    def test_families_cover_exactly_the_preregistered_cells(self):
        self.assertEqual(
            AUDIT.CONTEXT_FAMILY_FRONT, ("front_blocked", "front_clear", "front_unknown")
        )
        self.assertEqual(AUDIT.CONTEXT_FAMILY_TARGET, ("target_visible", "target_hidden"))
        self.assertEqual(
            AUDIT.CONTEXT_FAMILY_OUTCOME,
            tuple("outcome_" + AUDIT.OUTCOME_CODES[c] for c in AUDIT.OUTCOME_CONTEXT_CODES),
        )
        names = set(
            name
            for name, _mask in AUDIT.build_contexts(
                self._frames([1, 0, -1]), np.array([0, 1, -1], dtype=np.int64)
            )
        )
        covered = set(AUDIT.CONTEXT_FAMILY_FRONT) | set(AUDIT.CONTEXT_FAMILY_TARGET) | set(
            AUDIT.CONTEXT_FAMILY_OUTCOME
        )
        self.assertEqual(names - covered, {AUDIT.CONTEXT_OVERALL_CELL})


class GateDelegationAccounting(unittest.TestCase):
    """H3 -- delegated gates must never be folded into an 'evaluated, 0 failed' tally."""

    def _gates(self):
        return {
            "Q1_involution": {"passed": True},
            "Q2_isometry": {"passed": False},
            "Q6_import_origin": {"note": "launcher"},
            "Q7_manifest_schema_version": {"note": "launcher"},
            "Q7_checkpoint_sha": {"passed": True},
        }

    def test_delegated_gates_are_counted_separately(self):
        gates = AUDIT.stamp_gate_states(self._gates())
        summary = AUDIT.summarise_gates(gates)
        self.assertEqual(summary["delegated"], ["Q6_import_origin", "Q7_manifest_schema_version"])
        self.assertEqual(summary["n_delegated"], 2)
        self.assertEqual(summary["evaluated_passed"], ["Q1_involution", "Q7_checkpoint_sha"])
        self.assertEqual(summary["evaluated_failed"], ["Q2_isometry"])
        self.assertEqual(summary["n_evaluated_here"], 3)
        self.assertEqual(summary["n_malformed"], 0)

    def test_a_delegated_gate_cannot_report_passed(self):
        gates = AUDIT.stamp_gate_states(
            {"Q6_import_origin": {"passed": True, "note": "was hardcoded true"}}
        )
        self.assertIsNone(gates["Q6_import_origin"]["passed"])
        self.assertEqual(gates["Q6_import_origin"]["status"], AUDIT.GATE_STATUS_DELEGATED)
        self.assertFalse(gates["Q6_import_origin"]["evaluated_here"])
        summary = AUDIT.summarise_gates(gates)
        self.assertEqual(summary["evaluated_passed"], [])

    def test_a_gate_with_no_result_is_malformed_not_passed(self):
        summary = AUDIT.summarise_gates(
            {"QX_mystery": {"status": AUDIT.GATE_STATUS_EVALUATED, "passed": None}}
        )
        self.assertEqual(summary["malformed"], ["QX_mystery"])
        self.assertEqual(summary["evaluated_passed"], [])

    def test_delegation_contract_is_machine_readable(self):
        self.assertIn("Q6_import_origin", AUDIT.DELEGATED_GATES)
        self.assertIn("Q7_manifest_schema_version", AUDIT.DELEGATED_GATES)
        for name, contract in AUDIT.DELEGATED_GATES.items():
            for key in ("owner", "check", "caller_must_assert"):
                self.assertIn(key, contract, name)
                self.assertTrue(str(contract[key]).strip(), name)

    def test_the_tool_no_longer_claims_to_have_checked_the_manifest(self):
        text = (REPO / "tools/navrl_reflection_offline_audit.py").read_text(encoding="utf-8")
        self.assertNotIn("Q7_checkpoint_sha_and_manifest", text)
        source = inspect.getsource(AUDIT.run_audit)
        self.assertIn("checkpoint_sha_matches = checkpoint_sha ==", source)
        self.assertIn("quality_gate_summary", inspect.getsource(AUDIT.finalise_gate_block))

    def test_failed_gate_list_includes_malformed_gates(self):
        summary = AUDIT.summarise_gates(
            {
                "Q1_involution": {"status": AUDIT.GATE_STATUS_EVALUATED, "passed": True},
                "QX_mystery": {},
            }
        )
        failed = list(summary["evaluated_failed"]) + list(summary["malformed"])
        self.assertEqual(failed, ["QX_mystery"])


class ObservedScanFixedPoints(unittest.TestCase):
    """H4 -- Q5's evidence field must describe the OBSERVED operator, not the tool's own map."""

    def test_identity_operator_reports_every_beam_as_fixed(self):
        identity = list(range(HBEAMS * VBEAMS))
        passed, detail = AUDIT.gate_scan_permutation(identity, HBEAMS, VBEAMS)
        self.assertFalse(passed)
        self.assertEqual(detail["fixed_points"], list(range(HBEAMS)))
        self.assertNotEqual(detail["fixed_points"], sorted(AUDIT.GATE_SCAN_FIXED_POINTS))

    def test_an_extra_fixed_point_is_visible_in_the_evidence(self):
        source, _sign = AUDIT.preregistered_signed_permutation(HBEAMS, VBEAMS, MAX_OBSTACLES)
        broken = list(source)
        for v in range(VBEAMS):
            # h = 1 and h = 71 were swapped by the mirror; pin both to themselves.
            broken[v * HBEAMS + 1] = v * HBEAMS + 1
            broken[v * HBEAMS + (HBEAMS - 1)] = v * HBEAMS + (HBEAMS - 1)
        passed, detail = AUDIT.gate_scan_permutation(broken, HBEAMS, VBEAMS)
        self.assertFalse(passed)
        self.assertEqual(detail["fixed_points"], [0, 1, 36, HBEAMS - 1])

    def test_a_beam_fixed_on_only_one_ring_is_not_a_fixed_point(self):
        source, _sign = AUDIT.preregistered_signed_permutation(HBEAMS, VBEAMS, MAX_OBSTACLES)
        broken = list(source)
        broken[2 * HBEAMS + 5] = 2 * HBEAMS + 5  # fixed on ring 2 only
        self.assertEqual(
            AUDIT.observed_scan_fixed_points(broken, HBEAMS, VBEAMS), {0, 36}
        )
        passed, _detail = AUDIT.gate_scan_permutation(broken, HBEAMS, VBEAMS)
        self.assertFalse(passed)

    def test_evidence_field_states_where_it_came_from(self):
        source, _sign = AUDIT.preregistered_signed_permutation(HBEAMS, VBEAMS, MAX_OBSTACLES)
        _passed, detail = AUDIT.gate_scan_permutation(source, HBEAMS, VBEAMS)
        self.assertIn("observed_source", detail["fixed_points_computed_from"])
        self.assertEqual(detail["fixed_points"], [0, 36])
        gate_source = inspect.getsource(AUDIT.gate_scan_permutation)
        self.assertNotIn("if expected[h] == h", gate_source)


class LatentGuards(unittest.TestCase):
    """H5 -- guards that are unreachable on this npz but would silently corrupt a future run."""

    def test_int8_target_visible_with_unknown_is_rejected_not_coerced(self):
        column = np.array([1, 0, -1, 1], dtype=np.int8)
        # The hazard being closed: plain coercion would count the unknown frame as VISIBLE.
        self.assertTrue(bool(column.astype(bool)[2]))
        with self.assertRaises(AUDIT.IntegrityViolation) as context:
            AUDIT.as_two_valued_bool(
                column, "ctx_target_visible", AUDIT.CTX_TARGET_VISIBLE_DOMAIN
            )
        self.assertIn("astype(bool)", str(context.exception))
        frames = {
            "ctx_target_visible": column,
            "ctx_front_blocked": np.zeros(4, dtype=np.int64),
        }
        with self.assertRaises(AUDIT.IntegrityViolation):
            AUDIT.build_contexts(frames, np.zeros(4, dtype=np.int64))

    def test_two_valued_int_and_bool_columns_are_accepted(self):
        for column in (
            np.array([True, False], dtype=bool),
            np.array([1, 0], dtype=np.int8),
        ):
            values = AUDIT.as_two_valued_bool(column, "ctx_target_visible")
            self.assertEqual(values.tolist(), [True, False])

    def test_float_context_column_is_rejected(self):
        with self.assertRaises(AUDIT.IntegrityViolation):
            AUDIT.as_two_valued_bool(np.array([1.0, 0.0]), "ctx_target_visible")

    def test_front_column_domain_is_asserted(self):
        self.assertEqual(
            AUDIT.as_front_code(np.array([1, 0, -1], dtype=np.int8)).tolist(), [1, 0, -1]
        )
        with self.assertRaises(AUDIT.IntegrityViolation) as context:
            AUDIT.as_front_code(np.array([1, 0, 2], dtype=np.int8))
        self.assertIn("domain", str(context.exception))

    def test_context_column_domains_are_reported(self):
        frames = {
            "ctx_valid": np.ones(3, dtype=bool),
            "ctx_target_visible": np.array([True, False, True]),
            "ctx_front_blocked": np.array([1, 0, -1], dtype=np.int8),
        }
        detail = AUDIT.check_context_columns(frames)
        self.assertEqual(detail["columns"]["ctx_front_blocked"]["dtype"], "int8")
        self.assertEqual(
            detail["columns"]["ctx_front_blocked"]["allowed_domain"], [-1, 0, 1]
        )
        self.assertFalse(detail["columns"]["ctx_target_visible"]["coerced_with_astype_bool"])

    def test_undersampled_overall_cell_gets_no_verdict_not_inconclusive(self):
        original = np.zeros((255, 4))
        original[:, 1] = 0.8
        cell = AUDIT.measure_cell(original, original.copy(), np.ones(255, dtype=bool))
        self.assertIsNone(cell["verdict"])
        verdict, reason = AUDIT.overall_verdict(cell)
        self.assertEqual(verdict, AUDIT.VERDICT_NO_VERDICT_INSUFFICIENT_SAMPLE)
        self.assertNotEqual(verdict, AUDIT.VERDICT_INCONCLUSIVE)
        self.assertIn("255", reason)
        self.assertIn(str(AUDIT.MIN_CONTEXT_COMPARABLE_ROWS), reason)

    def test_a_real_verdict_passes_through_unchanged(self):
        original = np.zeros((512, 4))
        original[:, 1] = 0.8
        cell = AUDIT.measure_cell(original, original.copy(), np.ones(512, dtype=bool))
        verdict, reason = AUDIT.overall_verdict(cell)
        self.assertEqual(verdict, AUDIT.VERDICT_CHIRALITY_CONFIRMED)
        self.assertIsNone(reason)

    def test_driver_no_longer_rewrites_a_missing_verdict_as_inconclusive(self):
        source = inspect.getsource(AUDIT.run_audit)
        self.assertNotIn("else VERDICT_INCONCLUSIVE", source)
        self.assertIn("overall_verdict(overall)", source)

    def test_non_finite_actions_fail_closed_instead_of_reaching_inconclusive(self):
        # Without the gate a NaN reaches the verdict rule, where both comparisons are False.
        self.assertEqual(
            AUDIT.classify_verdict(float("nan"), float("nan")), AUDIT.VERDICT_INCONCLUSIVE
        )
        clean = np.zeros((4, 4), dtype=np.float32)
        detail = AUDIT.check_finite({"original_actions": clean}, "actions")
        self.assertTrue(detail["all_finite"])
        dirty = clean.copy()
        dirty[2, 1] = np.nan
        with self.assertRaises(AUDIT.IntegrityViolation) as context:
            AUDIT.check_finite({"original_actions": dirty}, "actions")
        self.assertIn("non-finite", str(context.exception))

    def test_infinite_observation_fails_closed(self):
        obs = np.zeros((4, 8), dtype=np.float32)
        obs[1, 3] = np.inf
        with self.assertRaises(AUDIT.IntegrityViolation):
            AUDIT.check_finite({"obs": obs}, "frames.obs")

    def test_obs_dtype_is_asserted_not_assumed(self):
        detail = AUDIT.check_frames_obs_dtype(np.zeros((2, 3), dtype=np.float32))
        self.assertEqual(detail["frames_obs_dtype"], "float32")
        with self.assertRaises(AUDIT.IntegrityViolation) as context:
            AUDIT.check_frames_obs_dtype(np.zeros((2, 3), dtype=np.uint8))
        self.assertIn("255", str(context.exception))
        with self.assertRaises(AUDIT.IntegrityViolation):
            AUDIT.check_frames_obs_dtype(np.zeros((2, 3), dtype=np.float64))

    def test_absent_npz_identity_is_stated_explicitly(self):
        report = AUDIT.check_npz_identity({}, "a" * 64)
        self.assertFalse(report["checkpoint_binding_verified"])
        self.assertEqual(report["identity_fields_present"], [])
        self.assertIn("NOT machine-verified", report["note"])

    def test_npz_identity_binds_the_frames_to_the_checkpoint(self):
        report = AUDIT.check_npz_identity({"checkpoint_sha256": "A" * 64}, "a" * 64)
        self.assertTrue(report["checkpoint_binding_verified"])
        with self.assertRaises(AUDIT.IntegrityViolation) as context:
            AUDIT.check_npz_identity({"checkpoint_sha256": "b" * 64}, "a" * 64)
        self.assertIn("dumped from checkpoint", str(context.exception))

    def test_recorded_obs_width_is_checked_against_the_loaded_table(self):
        report = AUDIT.check_npz_identity({"obs_width_recorded": 898}, "a" * 64, obs_width=898)
        self.assertTrue(report["obs_width_recorded_matches_loaded_obs"])
        with self.assertRaises(AUDIT.IntegrityViolation):
            AUDIT.check_npz_identity({"obs_width_recorded": 574}, "a" * 64, obs_width=898)

    def test_run_identity_the_dump_writes_is_recorded(self):
        # The NAVRL_OBS_DUMP hook writes these; the audit must carry them into the JSON.
        for key in ("run_seed", "run_bars", "run_num_envs", "run_pid", "obs_width_recorded"):
            self.assertIn(key, AUDIT.IDENTITY_KEYS, key)
        report = AUDIT.check_npz_identity({"run_seed": 373, "run_bars": 70}, "a" * 64)
        self.assertEqual(report["identity_fields_present"], ["run_bars", "run_seed"])
        self.assertEqual(report["identity_fields"]["run_seed"], 373)

    def test_dump_outcome_code_map_is_cross_checked_not_adopted(self):
        absent = AUDIT.check_dump_outcome_code_map(None)
        self.assertFalse(absent["dump_ships_a_code_map"])
        self.assertIn("NOT be cross-checked", absent["note"])

        agreeing = dict(AUDIT.OUTCOME_CODES)
        agreeing[6] = "crash_below_floor"  # a code the prereg does not know
        detail = AUDIT.check_dump_outcome_code_map(agreeing)
        self.assertEqual(detail["codes_only_in_dump"], [6])
        self.assertEqual(detail["name_disagreements"], {})
        # The tool keeps its preregistered literal; the extra code is reported, never adopted.
        self.assertNotIn(6, AUDIT.OUTCOME_CODES)

        relabelled = dict(AUDIT.OUTCOME_CODES)
        relabelled[2] = "timeout"
        with self.assertRaises(AUDIT.IntegrityViolation) as context:
            AUDIT.check_dump_outcome_code_map(relabelled)
        self.assertIn("2", str(context.exception))

    def test_a_frame_carrying_an_unknown_outcome_code_fails_closed(self):
        codes = np.array([0, 0, 6], dtype=np.int64)
        with self.assertRaises(AUDIT.IntegrityViolation):
            AUDIT.check_outcome_join(codes, codes.size)

    def test_run_audit_wires_every_integrity_check(self):
        source = inspect.getsource(AUDIT.run_audit)
        for name in (
            "check_npz_identity",
            "check_dump_outcome_code_map",
            "check_frames_obs_dtype",
            "check_finite",
            "check_context_columns",
            "check_outcome_join",
            "check_context_partitions",
            "characterise_excluded_frames",
        ):
            self.assertIn(name, source)
        # A failed invariant may not leave policy statistics in the payload.
        recorder = inspect.getsource(AUDIT.record_integrity)
        self.assertIn("measurements_raw_normaliser", recorder)
        self.assertIn(AUDIT.VERDICT_FAIL_CLOSED_INTEGRITY, recorder)


class ExcludedFrameDisclosure(unittest.TestCase):
    """H6 -- the non-random exclusion is characterised from the data, not remembered."""

    def test_tail_concentrated_exclusion_is_flagged_and_described(self):
        call_index = np.arange(100, dtype=np.int64) * 10
        excluded = np.zeros(100, dtype=bool)
        excluded[80:] = True
        report = AUDIT.characterise_excluded_frames(call_index, excluded)
        self.assertTrue(report["concentrated_in_tail"])
        self.assertFalse(report["gating"])
        self.assertEqual(report["n_excluded"], 20)
        self.assertEqual(report["excluded_call_index_min"], 800.0)
        self.assertEqual(report["excluded_call_index_max"], 990.0)
        self.assertEqual(report["population_call_index_max"], 990.0)
        self.assertEqual(report["fraction_of_excluded_at_or_above_population_median"], 1.0)
        self.assertEqual(report["excluded_min_call_index_percentile_of_population"], 80.0)
        self.assertIn("NOT missing at random", report["caveat"])
        self.assertIn("overall cell", report["caveat"])

    def test_uniform_exclusion_is_not_flagged(self):
        call_index = np.arange(100, dtype=np.int64)
        excluded = np.zeros(100, dtype=bool)
        excluded[::2] = True
        report = AUDIT.characterise_excluded_frames(call_index, excluded)
        self.assertFalse(report["concentrated_in_tail"])
        self.assertLess(report["fraction_of_excluded_at_or_above_population_median"], 0.9)

    def test_no_exclusion_is_reported_as_such(self):
        report = AUDIT.characterise_excluded_frames(
            np.arange(10, dtype=np.int64), np.zeros(10, dtype=bool)
        )
        self.assertEqual(report["n_excluded"], 0)
        self.assertFalse(report["concentrated_in_tail"])
        self.assertIn("no frames were excluded", report["caveat"])

    def test_missing_call_index_is_disclosed_not_skipped(self):
        report = AUDIT.characterise_excluded_frames(None, np.array([True, False]))
        self.assertFalse(report["call_index_available"])
        self.assertIsNone(report["concentrated_in_tail"])
        self.assertIn("could NOT be characterised", report["caveat"])

    def test_threshold_is_a_declared_constant(self):
        self.assertEqual(AUDIT.EXCLUDED_TAIL_CONCENTRATION_MIN, 0.9)
        text = (REPO / "tools/navrl_reflection_offline_audit.py").read_text(encoding="utf-8")
        self.assertLess(
            text.index("EXCLUDED_TAIL_CONCENTRATION_MIN = "),
            text.index("def characterise_excluded_frames"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
