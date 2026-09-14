"""CPU-only tests for the NAVRL_OBS_DUMP evaluation-only observation dump (WORKLOG 2026-08-21).

Extracts the pure module-level helpers straight out of navrl_task.py's source via `ast` and execs
just those nodes in an isolated namespace -- no `import aerial_gym`, no Isaac Gym, no torch, no
GPU. Mirrors the technique in tests/test_navrl_outcome_strata_contract.py (which walks the same
file's AST for wiring checks); this file additionally *executes* the extracted nodes so the
sampling/guard helpers can be exercised for real behavior, not just structure.

What this guards:
  - the streaming decimation (`_obs_dump_retain_decision` / `_obs_dump_thin_step`) stays uniform
    over an unknown-length rollout and keeps the row count within [MAX/2, MAX];
  - `OBS_DUMP_OUTCOME_CODES` is exactly the 6-entry literal the dump's per-episode table promises;
  - the fail-closed export guard (`_validate_obs_dump_export`) raises RuntimeError, and only
    RuntimeError, on each of the mismatches it is supposed to catch, and accepts a consistent
    table.

  - the adversarial-audit defects F1-F7 (2026-08-22): sweep dump collision + failure visibility,
    the outcome codes the physical-target merge silently widened, full-reset frame orphans, the
    tautological obs-width check, the uncapped outcome table, the reset/write gate mismatch, and
    atexit being the only flush trigger.

Two extraction techniques are used. Pure module-level helpers are exec'd as before. METHODS whose
bodies contain no torch on the paths under test (`_flush_obs_dump`, `_note_obs_dump_full_reset`)
are lifted out of the class and bound to a stub `self`, which makes their real control flow --
idempotence, the `.FAILED` marker, orphan bookkeeping -- testable without Isaac Gym. The rest is
checked structurally against the AST (call sites, gate conditions), which is what the wiring
defects actually are.

What this does NOT prove: that `_collect_obs_dump_frame` wires the helpers into the live GPU
tensors -- that needs a real (Isaac Gym) run and is out of scope for a CPU-only test.

Run: PYTHONNOUSERSITE=1 python -m unittest discover -s tests -p "test_navrl_obs_dump.py"
"""

import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO / "aerial_gym/task/navrl_task/navrl_task.py"
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

_WANTED = {
    "OBS_DUMP_OUTCOME_CODES",
    "_obs_dump_retain_decision",
    "_obs_dump_thin_step",
    "_validate_obs_dump_export",
    "_obs_dump_outcome_code_table",
    "_obs_dump_assert_free_path",
    "_obs_dump_check_episode_budget",
    "_obs_dump_assign_crash_codes",
    "_obs_dump_drop_reset_orphans",
}


def _is_wanted_assign(node):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in _WANTED for target in node.targets
    )


_nodes = [
    node
    for node in TREE.body
    if _is_wanted_assign(node) or (isinstance(node, ast.FunctionDef) and node.name in _WANTED)
]
if len(_nodes) != len(_WANTED):
    found = {
        (n.targets[0].id if isinstance(n, ast.Assign) else n.name) for n in _nodes
    }
    raise AssertionError(
        "expected module-level definitions %s in navrl_task.py, found %s" % (_WANTED, found)
    )

# `np` / `json` are module globals in navrl_task.py; the extracted helpers close over them.
_namespace = {"np": np, "json": json}
exec(compile(ast.Module(body=_nodes, type_ignores=[]), filename=str(SOURCE_PATH), mode="exec"), _namespace)

OBS_DUMP_OUTCOME_CODES = _namespace["OBS_DUMP_OUTCOME_CODES"]
_obs_dump_retain_decision = _namespace["_obs_dump_retain_decision"]
_obs_dump_thin_step = _namespace["_obs_dump_thin_step"]
_validate_obs_dump_export = _namespace["_validate_obs_dump_export"]
_obs_dump_outcome_code_table = _namespace["_obs_dump_outcome_code_table"]
_obs_dump_assert_free_path = _namespace["_obs_dump_assert_free_path"]
_obs_dump_check_episode_budget = _namespace["_obs_dump_check_episode_budget"]
_obs_dump_assign_crash_codes = _namespace["_obs_dump_assign_crash_codes"]
_obs_dump_drop_reset_orphans = _namespace["_obs_dump_drop_reset_orphans"]


# --------------------------------------------------------------------------------------------
# Method-level extraction (no Isaac Gym): lift a method out of the class and bind it to a stub.
# --------------------------------------------------------------------------------------------

_CLASS = next(
    node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == "NavRLTask"
)


def _method_node(name):
    for node in _CLASS.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("NavRLTask has no method %r" % (name,))


def _method_source(name):
    return ast.get_source_segment(SOURCE, _method_node(name))


def _bind_method(name, extra_globals=None):
    """Compile ONE method as a plain function so it can be called with a stub `self`."""
    globals_ns = {"np": np, "json": json, "Path": Path}
    globals_ns.update(extra_globals or {})
    module = ast.Module(body=[_method_node(name)], type_ignores=[])
    exec(compile(module, filename=str(SOURCE_PATH), mode="exec"), globals_ns)
    return globals_ns[name]


def _attr_names(node):
    return set(child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute))


def _gate_for_assignment(func_name, target_attr):
    """Attribute names appearing in the `if` that guards an assignment to `self.<target_attr>`."""
    func = _method_node(func_name)
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        body_dump = ast.dump(ast.Module(body=list(node.body), type_ignores=[]))
        if "attr='%s'" % target_attr in body_dump or 'attr="%s"' % target_attr in body_dump:
            return _attr_names(node.test)
    return None


class _StubTask(object):
    """Minimal stand-in for NavRLTask, holding only the attributes the lifted methods touch."""

    def __init__(self, path, num_envs=4):
        self._obs_dump_path = str(path)
        self._obs_dump_frames = [{"obs": np.zeros((num_envs, 3), dtype=np.float32)}]
        self._obs_dump_flush_done = False
        self._obs_dump_flush_active = False
        self.num_envs = num_envs
        self._obs_dump_ep_idx_host = np.full(num_envs, -1, dtype=np.int64)
        self._obs_dump_reset_orphan_uids = set()
        self._obs_dump_reset_orphan_cap = 1 << 22
        self.writes = 0
        self.raise_on_write = None

    def _write_obs_dump(self):
        self.writes += 1
        if self.raise_on_write is not None:
            raise self.raise_on_write


class _TaskConfigStub(object):
    def __init__(self, width):
        self.observation_space_dim = width
        self.seed = 373


class _WriterStub(object):
    """Stub with everything the REAL `_write_obs_dump` reads -- none of it is torch, so the whole
    write path (guard, orphan drop, npz kwargs, atomic publish) runs on CPU."""

    def __init__(self, path, num_envs=4, width=8, n_calls=3, ep_idx=None, schema_width=None):
        self._obs_dump_path = str(path)
        self._obs_dump_episode_overflow = ""
        self._obs_dump_max_rows = 16384
        self._obs_dump_stride_eff = 2
        self._obs_dump_decimations = 1
        self._obs_dump_reset_orphan_uids = set()
        self.num_envs = num_envs
        self.n_bars_active = 70
        self.max_bars_available = 150
        self.task_config = _TaskConfigStub(width)
        self._schema_width = schema_width
        self._obs_dump_ep_idx_host = (
            np.zeros(num_envs, dtype=np.int64) if ep_idx is None else np.asarray(ep_idx)
        )
        env_id = np.arange(num_envs, dtype=np.int32)
        self._obs_dump_frames = [
            {
                "obs": np.full((num_envs, width), float(call), dtype=np.float32),
                "env_id": env_id.copy(),
                "call_index": np.full(num_envs, call, dtype=np.int64),
                "episode_uid": env_id.astype(np.int64)
                + num_envs * self._obs_dump_ep_idx_host,
                "ep_step": np.full(num_envs, call, dtype=np.int64),
                "ctx_target_visible": np.zeros(num_envs, dtype=bool),
                "ctx_front_blocked": np.full(num_envs, -1, dtype=np.int8),
                "ctx_valid": np.ones(num_envs, dtype=bool),
            }
            for call in range(n_calls)
        ]
        self._obs_dump_episode_rows = []

    def _obs_dump_schema_obs_width(self):
        return self._schema_width


def _simulate_stream(n_calls, rows_per_call, max_rows):
    """Drive the two pure helpers exactly as `_collect_obs_dump_frame` does: increment a 0-based
    call counter, retain-check it, append, and thin-loop until the row budget is satisfied.
    Returns (retained_call_indices, final_stride_eff, n_decimations).
    """
    retained = []
    stride_eff = 1
    decimations = 0
    for call_index in range(n_calls):
        if not _obs_dump_retain_decision(call_index, stride_eff):
            continue
        retained.append(call_index)
        while True:
            _, new_stride_eff, thinned = _obs_dump_thin_step(
                len(retained), stride_eff, max_rows, rows_per_call
            )
            if not thinned:
                break
            retained = retained[0::2]
            stride_eff = new_stride_eff
            decimations += 1
    return retained, stride_eff, decimations


class TestOutcomeCodeMap(unittest.TestCase):
    def test_exact_literal(self):
        # 0-5 are frozen at their pre-merge (commit cff96c2) meaning so the published seed-373
        # dump stays interpretable; 6-9 were APPENDED by the F2 fix. See TestF2OutcomeCodeSplit.
        self.assertEqual(
            OBS_DUMP_OUTCOME_CODES,
            {
                0: "capture",
                1: "crash_bar_contact",
                2: "crash_oob",
                3: "crash_other",
                4: "timeout",
                5: "unattributed",
                6: "crash_below_floor",
                7: "crash_above_ceiling",
                8: "crash_target_contact",
                9: "crash_target_invalid",
            },
        )


class TestRetainDecision(unittest.TestCase):
    def test_stride_one_retains_every_call(self):
        for call_index in range(50):
            self.assertTrue(_obs_dump_retain_decision(call_index, 1))

    def test_stride_three_retains_only_multiples(self):
        retained = [c for c in range(30) if _obs_dump_retain_decision(c, 3)]
        self.assertEqual(retained, [0, 3, 6, 9, 12, 15, 18, 21, 24, 27])

    def test_first_call_always_retained_regardless_of_stride(self):
        # call_index 0 is retained under every positive stride -- the whole point of counting
        # calls from 0, not 1 (see the comment on _obs_dump_retain_decision).
        for stride in (1, 2, 3, 7, 100):
            self.assertTrue(_obs_dump_retain_decision(0, stride))


class TestThinStep(unittest.TestCase):
    def test_no_thin_under_budget(self):
        new_len, new_stride, thinned = _obs_dump_thin_step(
            retained_len=5, stride_eff=1, max_rows=100, rows_per_call=4
        )
        self.assertFalse(thinned)
        self.assertEqual(new_len, 5)
        self.assertEqual(new_stride, 1)

    def test_thins_and_doubles_stride_over_budget(self):
        new_len, new_stride, thinned = _obs_dump_thin_step(
            retained_len=11, stride_eff=1, max_rows=40, rows_per_call=4
        )
        self.assertTrue(thinned)
        self.assertEqual(new_len, 6)  # len([0..10][0::2])
        self.assertEqual(new_stride, 2)

    def test_single_element_list_never_thins(self):
        # Guards against an infinite loop if one call alone already exceeds the row budget.
        new_len, new_stride, thinned = _obs_dump_thin_step(
            retained_len=1, stride_eff=1, max_rows=1, rows_per_call=1000
        )
        self.assertFalse(thinned)
        self.assertEqual(new_len, 1)
        self.assertEqual(new_stride, 1)


class TestStreamingDecimationEndToEnd(unittest.TestCase):
    """Drives the two pure helpers over a long synthetic rollout, exactly the way
    `_collect_obs_dump_frame` drives them over the real one, and checks the three properties the
    design promises: uniform spacing, a bounded final row count, and no RNG (determinism).
    """

    def test_uniform_after_decimation(self):
        retained, stride_eff, decimations = _simulate_stream(
            n_calls=200_000, rows_per_call=4, max_rows=4096
        )
        self.assertGreater(decimations, 0)
        self.assertTrue(retained, "expected at least one retained call")
        # Exact uniformity: every retained call index is a multiple of the FINAL stride -- not
        # just "on average" spaced, but exactly on the stride_eff grid, with no leftover.
        offenders = [c for c in retained if c % stride_eff != 0]
        self.assertEqual(offenders, [], "non-uniform retained calls: %s" % offenders[:10])
        # And nothing is skipped WITHIN that grid up to the last retained call: the retained set
        # is exactly {0, stride_eff, 2*stride_eff, ...} truncated to what fits in the rollout.
        expected = list(range(0, retained[-1] + 1, stride_eff))
        self.assertEqual(retained, expected)

    def test_row_count_bounded_between_half_and_full_max(self):
        max_rows = 4096
        retained, _, _ = _simulate_stream(n_calls=500_000, rows_per_call=4, max_rows=max_rows)
        n_rows = len(retained) * 4
        self.assertLessEqual(n_rows, max_rows)
        self.assertGreater(n_rows, max_rows // 2)

    def test_final_stride_is_a_power_of_two_multiple_of_initial(self):
        _, stride_eff, decimations = _simulate_stream(
            n_calls=1_000_000, rows_per_call=1, max_rows=1024
        )
        self.assertEqual(stride_eff, 2 ** decimations)

    def test_deterministic_no_rng(self):
        # Same inputs -> byte-identical outputs, twice, and with a totally different call count
        # that still lands on a decimation boundary.
        a = _simulate_stream(n_calls=50_000, rows_per_call=8, max_rows=2048)
        b = _simulate_stream(n_calls=50_000, rows_per_call=8, max_rows=2048)
        self.assertEqual(a, b)

    def test_short_rollout_never_decimates(self):
        # Fewer calls than fit in the budget: every call retained, stride stays 1.
        retained, stride_eff, decimations = _simulate_stream(
            n_calls=100, rows_per_call=4, max_rows=4096
        )
        self.assertEqual(decimations, 0)
        self.assertEqual(stride_eff, 1)
        self.assertEqual(retained, list(range(100)))


def _valid_tables(n=4, m=2, width=898):
    frame_tables = {
        "obs": np.zeros((n, width), dtype=np.float32),
        "env_id": np.arange(n, dtype=np.int32),
        "call_index": np.zeros(n, dtype=np.int64),
        "episode_uid": np.array([0, 1, 2, 3], dtype=np.int64)[:n],
        "ep_step": np.zeros(n, dtype=np.int64),
        "ctx_target_visible": np.zeros(n, dtype=bool),
        "ctx_front_blocked": np.full(n, -1, dtype=np.int8),
        "ctx_valid": np.ones(n, dtype=bool),
    }
    episode_tables = {
        "ep_uid": np.array([0, 1], dtype=np.int64)[:m],
        "ep_env_id": np.array([0, 1], dtype=np.int32)[:m],
        "outcome": np.array([0, 4], dtype=np.int8)[:m],
        "ep_len": np.array([10, 20], dtype=np.int64)[:m],
    }
    return frame_tables, episode_tables


class TestExportGuard(unittest.TestCase):
    def test_accepts_consistent_tables(self):
        frame_tables, episode_tables = _valid_tables()
        # episode_uid 0 and 1 both "finished" (not live); 2 and 3 are still-running envs with no
        # outcome row yet, which is legitimate (mid-episode frames).
        live = {2, 3}
        _validate_obs_dump_export(frame_tables, episode_tables, 898, 16384, live)

    def test_raises_on_frame_array_length_mismatch(self):
        frame_tables, episode_tables = _valid_tables()
        frame_tables["ep_step"] = frame_tables["ep_step"][:-1]  # one row short
        with self.assertRaisesRegex(RuntimeError, "ep_step"):
            _validate_obs_dump_export(frame_tables, episode_tables, 898, 16384, {2, 3})

    def test_raises_on_obs_width_mismatch(self):
        frame_tables, episode_tables = _valid_tables(width=897)
        with self.assertRaisesRegex(RuntimeError, "897"):
            _validate_obs_dump_export(frame_tables, episode_tables, 898, 16384, {2, 3})

    def test_raises_on_row_count_over_cap(self):
        frame_tables, episode_tables = _valid_tables(n=4)
        with self.assertRaisesRegex(RuntimeError, "exceeds the configured cap"):
            _validate_obs_dump_export(frame_tables, episode_tables, 898, max_rows=3, live_episode_uids={2, 3})

    def test_raises_on_episode_array_length_mismatch(self):
        frame_tables, episode_tables = _valid_tables()
        episode_tables["ep_len"] = episode_tables["ep_len"][:-1]
        with self.assertRaisesRegex(RuntimeError, "ep_len"):
            _validate_obs_dump_export(frame_tables, episode_tables, 898, 16384, {2, 3})

    def test_raises_on_finished_episode_missing_outcome_row(self):
        frame_tables, episode_tables = _valid_tables()
        # episode_uid 2 has frame rows and is NOT in `live` (so it must have finished), but the
        # outcome table only covers 0 and 1 -- a dropped-row bug the guard must catch.
        live = {3}
        with self.assertRaisesRegex(RuntimeError, "no outcome"):
            _validate_obs_dump_export(frame_tables, episode_tables, 898, 16384, live)

    def test_missing_obs_key_raises(self):
        frame_tables, episode_tables = _valid_tables()
        del frame_tables["obs"]
        with self.assertRaisesRegex(RuntimeError, "obs"):
            _validate_obs_dump_export(frame_tables, episode_tables, 898, 16384, {2, 3})

    def test_missing_ep_uid_key_raises(self):
        frame_tables, episode_tables = _valid_tables()
        del episode_tables["ep_uid"]
        with self.assertRaisesRegex(RuntimeError, "ep_uid"):
            _validate_obs_dump_export(frame_tables, episode_tables, 898, 16384, {2, 3})


# ============================================================================================
# F1 -- multi-density sweep overwrite / stale-file provenance / invisible atexit failure
# ============================================================================================


class TestF1SweepCollisionAndFailureVisibility(unittest.TestCase):
    def test_refuses_to_overwrite_an_existing_dump_path(self):
        with self.assertRaises(RuntimeError) as caught:
            _obs_dump_assert_free_path("/results/150bars/frames.npz", True)
        message = str(caught.exception)
        self.assertIn("/results/150bars/frames.npz", message)
        self.assertIn("per-condition", message)

    def test_accepts_a_fresh_path(self):
        self.assertIsNone(_obs_dump_assert_free_path("/results/150bars/frames.npz", False))

    def test_overwrite_guard_is_wired_at_construction_and_at_write(self):
        self.assertIn("_obs_dump_assert_free_path(", _method_source("__init__"))
        self.assertIn("_obs_dump_assert_free_path(", _method_source("_write_obs_dump"))

    def test_run_identity_is_recorded_inside_the_npz(self):
        write = _method_source("_write_obs_dump")
        for field in ("run_bars=", "run_seed=", "run_num_envs=", "run_max_bars="):
            self.assertIn(field, write, "npz must carry %s for stale-file detection" % field)

    def test_failed_marker_is_written_when_the_flush_raises(self):
        flush = _bind_method("_flush_obs_dump")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "frames.npz"
            stub = _StubTask(target)
            stub.raise_on_write = RuntimeError("export guard tripped on density 150")
            with self.assertRaises(RuntimeError):
                flush(stub)
            marker = Path(str(target) + ".FAILED")
            self.assertTrue(marker.is_file(), "a failed flush must be visible ON DISK")
            self.assertIn("density 150", marker.read_text(encoding="utf-8"))
            self.assertIn("Traceback", marker.read_text(encoding="utf-8"))
            # And the dump path itself must NOT exist: a downstream require(frames.is_file())
            # must not be able to pass on another condition's stale bytes.
            self.assertFalse(target.exists())
            self.assertFalse(stub._obs_dump_flush_done)

    def test_atexit_failure_does_not_mark_the_flush_done(self):
        flush = _bind_method("_flush_obs_dump")
        with tempfile.TemporaryDirectory() as tmp:
            stub = _StubTask(Path(tmp) / "frames.npz")
            stub.raise_on_write = RuntimeError("boom")
            with self.assertRaises(RuntimeError):
                flush(stub)
            stub.raise_on_write = None
            flush(stub)  # a later, deterministic trigger may still succeed
            self.assertEqual(stub.writes, 2)
            self.assertTrue(stub._obs_dump_flush_done)


# ============================================================================================
# F2 -- outcome codes: `d_contact` was widened to include TARGET-caused terminations
# ============================================================================================


class TestF2OutcomeCodeSplit(unittest.TestCase):
    def test_legacy_codes_0_to_5_are_frozen(self):
        # The published seed-373 dump (commit cff96c2, pre physical-target merge) is on disk with
        # exactly these meanings; renumbering would silently re-label it.
        for code, name in (
            (0, "capture"),
            (1, "crash_bar_contact"),
            (2, "crash_oob"),
            (3, "crash_other"),
            (4, "timeout"),
            (5, "unattributed"),
        ):
            self.assertEqual(OBS_DUMP_OUTCOME_CODES[code], name)

    def test_new_codes_are_appended_not_renumbered(self):
        self.assertEqual(OBS_DUMP_OUTCOME_CODES[6], "crash_below_floor")
        self.assertEqual(OBS_DUMP_OUTCOME_CODES[7], "crash_above_ceiling")
        self.assertEqual(OBS_DUMP_OUTCOME_CODES[8], "crash_target_contact")
        self.assertEqual(OBS_DUMP_OUTCOME_CODES[9], "crash_target_invalid")
        self.assertEqual(sorted(OBS_DUMP_OUTCOME_CODES), list(range(10)))

    def _assign(self, n, **masks):
        code = np.full(n, 5, dtype=np.int8)
        args = dict(
            (key, np.zeros(n, dtype=bool))
            for key in (
                "captured",
                "crashed",
                "target_contact",
                "target_invalid",
                "d_contact",
                "d_oob",
                "d_below",
                "d_above",
            )
        )
        for key, value in masks.items():
            args[key] = np.array(value, dtype=bool)
        return _obs_dump_assign_crash_codes(code, **args)

    def test_target_invalid_is_not_labelled_a_bar_collision(self):
        # THE defect: a clean drone flight whose TARGET left the arena. `d_contact` includes it
        # after the merge, and the old `code[d_contact] = 1` called it a drone-bar collision.
        code = self._assign(
            1, d_contact=[True], target_invalid=[True], crashed=[False]
        )
        self.assertEqual(int(code[0]), 9)
        self.assertNotEqual(int(code[0]), 1)

    def test_target_contact_gets_its_own_code(self):
        code = self._assign(1, d_contact=[True], target_contact=[True], crashed=[False])
        self.assertEqual(int(code[0]), 8)

    def test_drone_body_contact_still_means_code_1(self):
        code = self._assign(1, d_contact=[True], crashed=[True])
        self.assertEqual(int(code[0]), 1)

    def test_drone_contact_wins_over_a_same_step_target_cause(self):
        code = self._assign(
            1,
            d_contact=[True],
            crashed=[True],
            target_contact=[True],
            target_invalid=[True],
        )
        self.assertEqual(int(code[0]), 1)

    def test_below_and_above_are_no_longer_collapsed(self):
        code = self._assign(2, d_below=[True, False], d_above=[False, True])
        self.assertEqual([int(v) for v in code], [6, 7])

    def test_capture_and_oob_and_unattributed_unchanged(self):
        code = self._assign(3, captured=[True, False, False], d_oob=[False, True, False])
        self.assertEqual([int(v) for v in code], [0, 2, 5])

    def test_code_map_is_exported_with_the_data(self):
        values, names, payload = _obs_dump_outcome_code_table()
        self.assertEqual([int(v) for v in values], sorted(OBS_DUMP_OUTCOME_CODES))
        self.assertEqual(
            [str(v) for v in names],
            [OBS_DUMP_OUTCOME_CODES[int(v)] for v in values],
        )
        self.assertEqual(
            json.loads(payload),
            dict((str(k), v) for k, v in OBS_DUMP_OUTCOME_CODES.items()),
        )
        write = _method_source("_write_obs_dump")
        for field in ("outcome_code_values=", "outcome_code_names=", "outcome_code_map_json="):
            self.assertIn(field, write)

    def test_crash_outcome_site_receives_the_raw_masks(self):
        node = _method_node("_record_obs_dump_crash_outcomes")
        params = [arg.arg for arg in node.args.args]
        for name in ("crashed", "target_contact", "target_invalid"):
            self.assertIn(name, params, "the dump cannot split code 1 without %s" % name)
        # And the live call site actually passes them.
        call_src = None
        for call in ast.walk(_method_node("compute_state_reward_and_terminations")):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_record_obs_dump_crash_outcomes"
            ):
                call_src = set(
                    arg.id for arg in call.args if isinstance(arg, ast.Name)
                )
        self.assertIsNotNone(call_src, "no call to _record_obs_dump_crash_outcomes found")
        self.assertTrue({"crashed", "target_contact", "target_invalid"} <= call_src)


# ============================================================================================
# F3 -- frames orphaned by a full reset() used to destroy the whole dump
# ============================================================================================


def _orphan_tables(n_orphan=4, n_live=2, width=8):
    n = n_orphan + n_live
    uids = np.array([100 + i for i in range(n_orphan)] + [900 + i for i in range(n_live)])
    frame_tables = {
        "obs": np.zeros((n, width), dtype=np.float32),
        "episode_uid": uids.astype(np.int64),
        "ctx_valid": np.ones(n, dtype=bool),
    }
    return frame_tables, uids


class TestF3FullResetOrphans(unittest.TestCase):
    def test_full_reset_orphans_are_dropped_with_a_count(self):
        frame_tables, uids = _orphan_tables()
        kept, dropped_rows, dropped_eps = _obs_dump_drop_reset_orphans(
            frame_tables, np.zeros(0, dtype=np.int64), set(uids[:4].tolist())
        )
        self.assertEqual(dropped_rows, 4)
        self.assertEqual(dropped_eps, 4)
        self.assertEqual([int(v) for v in kept["episode_uid"]], [900, 901])
        self.assertEqual(kept["obs"].shape, (2, 8))

    def test_dropping_makes_the_export_guard_pass_where_it_used_to_be_fatal(self):
        frame_tables, uids = _orphan_tables()
        episode_tables = {
            "ep_uid": np.zeros(0, dtype=np.int64),
            "outcome": np.zeros(0, dtype=np.int8),
        }
        live = set(int(v) for v in uids[4:].tolist())
        # Before the fix: reset-orphaned frames are "finished with no outcome row" -> fatal.
        with self.assertRaisesRegex(RuntimeError, "no outcome"):
            _validate_obs_dump_export(frame_tables, episode_tables, 8, 16384, live)
        kept, dropped_rows, _ = _obs_dump_drop_reset_orphans(
            frame_tables, episode_tables["ep_uid"], set(uids[:4].tolist())
        )
        self.assertEqual(dropped_rows, 4)
        _validate_obs_dump_export(kept, episode_tables, 8, 16384, live)

    def test_an_episode_that_finished_normally_is_never_dropped(self):
        frame_tables, uids = _orphan_tables()
        # uid 100 was reset by the full reset AND had already recorded its outcome this step.
        outcome_uids = np.array([100], dtype=np.int64)
        kept, dropped_rows, dropped_eps = _obs_dump_drop_reset_orphans(
            frame_tables, outcome_uids, set(uids[:4].tolist())
        )
        self.assertEqual(dropped_rows, 3)
        self.assertEqual(dropped_eps, 3)
        self.assertIn(100, [int(v) for v in kept["episode_uid"]])

    def test_still_running_episodes_are_untouched(self):
        # The published seed-373 dump has 1530 frames from 117 episodes that were still running
        # when the rollout ended. They are legitimately orphaned and must keep validating.
        n_eps, per_ep, width = 117, 1530 // 117, 898
        uids = np.repeat(np.arange(n_eps, dtype=np.int64), per_ep)
        frame_tables = {
            "obs": np.zeros((uids.shape[0], width), dtype=np.float32),
            "episode_uid": uids,
        }
        kept, dropped_rows, dropped_eps = _obs_dump_drop_reset_orphans(
            frame_tables, np.zeros(0, dtype=np.int64), set()
        )
        self.assertEqual((dropped_rows, dropped_eps), (0, 0))
        self.assertIs(kept, frame_tables)
        _validate_obs_dump_export(
            frame_tables,
            {"ep_uid": np.zeros(0, dtype=np.int64)},
            width,
            16384,
            set(int(v) for v in np.unique(uids).tolist()),
            schema_obs_width=width,
        )

    def test_a_genuinely_missing_outcome_row_is_still_fatal(self):
        frame_tables, uids = _orphan_tables()
        episode_tables = {"ep_uid": np.zeros(0, dtype=np.int64)}
        kept, _, _ = _obs_dump_drop_reset_orphans(
            frame_tables, episode_tables["ep_uid"], set(uids[:4].tolist())
        )
        # uid 900/901 finished (not live) without an outcome row and were NOT reset orphans.
        with self.assertRaisesRegex(RuntimeError, "no outcome"):
            _validate_obs_dump_export(kept, episode_tables, 8, 16384, set())

    def test_full_reset_records_the_orphan_uids(self):
        note = _bind_method("_note_obs_dump_full_reset")
        stub = _StubTask("/tmp/unused.npz", num_envs=4)
        stub._obs_dump_ep_idx_host = np.array([0, 0, 1, 2], dtype=np.int64)
        note(stub)
        self.assertEqual(
            sorted(stub._obs_dump_reset_orphan_uids), [0, 1, 6, 11]
        )  # env + num_envs * ep_idx
        self.assertIn("_note_obs_dump_full_reset", _method_source("reset"))

    def test_orphan_tracking_is_bounded(self):
        note = _bind_method("_note_obs_dump_full_reset")
        stub = _StubTask("/tmp/unused.npz", num_envs=4)
        stub._obs_dump_reset_orphan_cap = 2
        with self.assertRaisesRegex(RuntimeError, "over the cap"):
            note(stub)

    def test_dropped_counts_are_recorded_in_the_npz(self):
        write = _method_source("_write_obs_dump")
        self.assertIn("dropped_reset_orphan_frames=", write)
        self.assertIn("dropped_reset_orphan_episodes=", write)


# ============================================================================================
# F4 -- the obs-width export check was a value compared to itself
# ============================================================================================


class TestF4ObsWidthCheck(unittest.TestCase):
    def test_guard_accepts_a_matching_schema_width(self):
        frame_tables, episode_tables = _valid_tables()
        _validate_obs_dump_export(
            frame_tables, episode_tables, 898, 16384, {2, 3}, schema_obs_width=898
        )

    def test_guard_catches_a_schema_regression_the_allocation_cannot(self):
        # Both the recorded width and `live_obs_width` come from the SAME allocation, so they
        # always agree; only the schema-derived width can disagree.
        frame_tables, episode_tables = _valid_tables(width=898)
        with self.assertRaisesRegex(RuntimeError, "perception schema components"):
            _validate_obs_dump_export(
                frame_tables, episode_tables, 898, 16384, {2, 3}, schema_obs_width=574
            )

    def test_schema_width_is_optional_and_documented_as_such(self):
        node = _method_node("_obs_dump_schema_obs_width")
        source = ast.get_source_segment(SOURCE, node)
        self.assertIn("return None", source)
        # The CODE (docstring excluded -- it names the allocation to explain the difference) must
        # not read the observation allocation at all, or the check is tautological again.
        body = ast.dump(ast.Module(body=list(node.body[1:]), type_ignores=[]))
        self.assertNotIn("observation_space_dim", body)
        self.assertNotIn("task_obs", body)
        # The width is built from schema components, not from the observation allocation.
        for term in ("ROBOT_HISTORY", "TARGET_HISTORY", "OBSTACLE_HISTORY", "CORRIDOR_OBS_DIM"):
            self.assertIn(term, body)
        self.assertIn("depth_range_pixels", body)

    def test_flush_passes_the_independent_width(self):
        write = _method_source("_write_obs_dump")
        self.assertIn("schema_obs_width=schema_obs_width", write)
        self.assertIn("self._obs_dump_schema_obs_width()", write)
        self.assertIn("obs_width_schema=", write)

    def test_seed373_schema_width_arithmetic(self):
        # 5*10 robot + 5*16 target + 5*8*12 obstacles + 4*72 lidar + 0 corridor == 898, the width
        # of the published dump. Pins the component decomposition the guard now recomputes.
        self.assertEqual(5 * 10 + 5 * 16 + 5 * 8 * 12 + 4 * 72 + 0, 898)


# ============================================================================================
# F5 -- the per-episode outcome table had no cap
# ============================================================================================


class TestF5OutcomeTableBudget(unittest.TestCase):
    def test_under_budget_returns_the_running_total(self):
        self.assertEqual(_obs_dump_check_episode_budget(10, 5, 100), 15)

    def test_exactly_at_the_cap_is_allowed(self):
        self.assertEqual(_obs_dump_check_episode_budget(95, 5, 100), 100)

    def test_over_budget_raises_instead_of_truncating(self):
        with self.assertRaises(RuntimeError) as caught:
            _obs_dump_check_episode_budget(99, 5, 100)
        message = str(caught.exception)
        self.assertIn("NAVRL_OBS_DUMP_MAX_EPISODES", message)
        self.assertIn("EVALUATION-only", message)

    def test_budget_is_wired_into_the_outcome_recorder(self):
        record = _method_source("_record_obs_dump_outcome")
        self.assertIn("_obs_dump_check_episode_budget(", record)
        self.assertIn("_obs_dump_episode_overflow", record)
        self.assertIn("NAVRL_OBS_DUMP_MAX_EPISODES", _method_source("__init__"))

    def test_an_overflowed_run_refuses_to_write_a_plausible_dump(self):
        write = _method_source("_write_obs_dump")
        self.assertIn("_obs_dump_episode_overflow", write)
        self.assertIn("refusing to write", write)


# ============================================================================================
# F6 -- reset gate did not match the (widened) write gate
# ============================================================================================


class TestF6ActionDiagResetGate(unittest.TestCase):
    def test_reset_clear_gate_matches_the_write_gate(self):
        guard = next(
            node
            for node in _method_node("_record_action_diagnostics").body
            if isinstance(node, ast.If)
        )
        write_gate = _attr_names(guard.test)
        clear_gate = _gate_for_assignment("reset_idx", "_action_diag_prev_valid")
        self.assertIsNotNone(clear_gate, "reset_idx no longer clears _action_diag_prev_valid")
        self.assertIn("_action_diag_enabled", clear_gate)
        self.assertIn("_obs_dump_enabled", clear_gate)
        self.assertEqual(
            write_gate & {"_action_diag_enabled", "_obs_dump_enabled"},
            clear_gate & {"_action_diag_enabled", "_obs_dump_enabled"},
        )


# ============================================================================================
# F7 -- atexit was the only flush trigger, and it touched a GPU tensor at shutdown
# ============================================================================================


class TestF7FlushLifecycle(unittest.TestCase):
    def test_flush_is_idempotent(self):
        flush = _bind_method("_flush_obs_dump")
        with tempfile.TemporaryDirectory() as tmp:
            stub = _StubTask(Path(tmp) / "frames.npz")
            flush(stub)
            flush(stub)
            flush(stub)
            self.assertEqual(stub.writes, 1, "close() + atexit must not double-write")

    def test_flush_is_a_noop_without_frames(self):
        flush = _bind_method("_flush_obs_dump")
        with tempfile.TemporaryDirectory() as tmp:
            stub = _StubTask(Path(tmp) / "frames.npz")
            stub._obs_dump_frames = []
            flush(stub)
            self.assertEqual(stub.writes, 0)

    def test_close_provides_a_deterministic_in_process_trigger(self):
        close = _method_source("close")
        self.assertIn("_flush_obs_dump()", close)
        # ... and it must run BEFORE the simulator is torn down.
        self.assertLess(close.index("_flush_obs_dump()"), close.index("delete_env"))

    def test_atexit_registration_is_kept(self):
        self.assertIn("atexit.register(self._flush_obs_dump)", _method_source("__init__"))

    def test_write_never_touches_the_device_episode_counter(self):
        write = _method_source("_write_obs_dump")
        self.assertNotIn("self._obs_dump_ep_idx.", write)
        self.assertIn("_obs_dump_ep_idx_host", write)

    def test_write_publishes_atomically(self):
        write = _method_source("_write_obs_dump")
        self.assertIn("os.replace(", write)
        self.assertIn("partial-", write)

    def test_host_mirror_is_updated_in_lockstep_with_the_device_counter(self):
        reset_idx = _method_source("reset_idx")
        self.assertIn("self._obs_dump_ep_idx[env_ids] += 1", reset_idx)
        self.assertIn("self._obs_dump_ep_idx_host[", reset_idx)
        self.assertIn("_obs_dump_ep_idx_host", _method_source("_collect_obs_dump_frame"))


# ============================================================================================
# The write path end to end (F1 + F3 + F7): a REAL npz, on CPU, with no torch involved.
# ============================================================================================


class TestWritePathEndToEnd(unittest.TestCase):
    def _writer(self):
        # The module-level helpers the real writer calls, plus a stand-in for the checksum
        # helper (hashing is exercised by the live run, not by this CPU test).
        return _bind_method(
            "_write_obs_dump",
            {
                "os": os,
                "_sha256_file": lambda path: "sha-not-checked-here",
                "_obs_dump_assert_free_path": _obs_dump_assert_free_path,
                "_obs_dump_drop_reset_orphans": _obs_dump_drop_reset_orphans,
                "_obs_dump_outcome_code_table": _obs_dump_outcome_code_table,
                "_validate_obs_dump_export": _validate_obs_dump_export,
            },
        )

    def test_writes_a_readable_npz_with_run_identity_and_the_code_map(self):
        write = self._writer()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "70bars" / "frames.npz"
            stub = _WriterStub(target, schema_width=8)
            write(stub)
            self.assertTrue(target.is_file())
            self.assertEqual(list(Path(tmp, "70bars").glob("*.partial-*")), [])
            with np.load(str(target), allow_pickle=False) as archive:
                self.assertEqual(archive["obs"].shape, (12, 8))
                self.assertEqual(int(archive["run_bars"]), 70)
                self.assertEqual(int(archive["run_seed"]), 373)
                self.assertEqual(int(archive["run_num_envs"]), 4)
                self.assertEqual(int(archive["obs_width_schema"]), 8)
                self.assertEqual(int(archive["dropped_reset_orphan_frames"]), 0)
                self.assertEqual(int(archive["stride_eff"]), 2)
                restored = json.loads(str(archive["outcome_code_map_json"]))
                self.assertEqual(
                    restored, dict((str(k), v) for k, v in OBS_DUMP_OUTCOME_CODES.items())
                )

    def test_second_condition_of_a_sweep_refuses_to_overwrite(self):
        write = self._writer()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "frames.npz"
            write(_WriterStub(target, schema_width=8))
            digest_before = target.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                write(_WriterStub(target, schema_width=8))
            self.assertEqual(target.read_bytes(), digest_before, "stale bytes must survive intact")

    def test_overflowed_outcome_table_refuses_to_write_anything(self):
        write = self._writer()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "frames.npz"
            stub = _WriterStub(target, schema_width=8)
            stub._obs_dump_episode_overflow = "outcome table would reach 2 rows, over the cap 1"
            with self.assertRaisesRegex(RuntimeError, "refusing to write"):
                write(stub)
            self.assertFalse(target.exists())

    def test_full_reset_orphans_are_dropped_and_counted_in_the_file(self):
        write = self._writer()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "frames.npz"
            # Frames were collected in episode generation 0; a full reset() then bumped every env
            # to generation 1 without emitting outcome rows (the F3 scenario).
            stub = _WriterStub(target, schema_width=8)
            stub._obs_dump_reset_orphan_uids = set(range(4))
            stub._obs_dump_ep_idx_host = np.ones(4, dtype=np.int64)
            write(stub)
            with np.load(str(target), allow_pickle=False) as archive:
                self.assertEqual(int(archive["dropped_reset_orphan_frames"]), 12)
                self.assertEqual(int(archive["dropped_reset_orphan_episodes"]), 4)
                self.assertEqual(archive["obs"].shape, (0, 8))

    def test_schema_width_mismatch_is_fatal_and_leaves_no_file(self):
        write = self._writer()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "frames.npz"
            with self.assertRaisesRegex(RuntimeError, "perception schema components"):
                write(_WriterStub(target, schema_width=9))
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(tmp).glob("*.partial-*")), [])

    def test_flush_wraps_the_real_writer_and_leaves_a_marker(self):
        """The two lifted methods composed: a failing REAL write must leave `.FAILED` behind."""
        writer = self._writer()
        flush = _bind_method("_flush_obs_dump")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "frames.npz"
            stub = _WriterStub(target, schema_width=9)  # deliberate schema mismatch
            stub._obs_dump_flush_done = False
            stub._obs_dump_flush_active = False
            stub._write_obs_dump = lambda: writer(stub)
            with self.assertRaises(RuntimeError):
                flush(stub)
            marker = Path(str(target) + ".FAILED")
            self.assertTrue(marker.is_file())
            self.assertIn("perception schema components", marker.read_text(encoding="utf-8"))
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
