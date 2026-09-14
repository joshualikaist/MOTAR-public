#!/usr/bin/env python3
"""Sensor-model fidelity evaluation (eval-only) -- preregistered launcher.

Preregistration: docs/prereg_2026-08-22_sensor_fidelity.md (frozen; sections 5, 6, 9, 10 binding).

Two evaluation arms of the SAME frozen ref5in D1 ep1900 checkpoint, in the SAME 70-bar arena, on
the SAME held-out seed, differing in exactly one physical axis -- "sensor-model fidelity":

  arm A (baseline)  detect 160x90,   min_pixels 2    -- today's model, unchanged
  arm B (fidelity)  detect 1920x1200, min_pixels 50  -- AR0234-class angular resolution with a
                                                        Johnson/CNN-grounded 8 px-diameter
                                                        detection floor (prereg section 3)

Resolution and threshold are ONE axis, not two (prereg section 2): raising the resolution alone
leaves the threshold below the Johnson floor and merely moves the failure, raising the threshold
alone collapses detection outright at 0.62 px^2.  Splitting them after the fact is forbidden.

What is deliberately NOT touched, because touching it is what confounded seed 367:
``NAVRL_DETECTOR_MAX_RANGE`` stays at its 20.0 m default in BOTH arms.  Seed 367 moved it 20->28 m,
which also renormalises the actor's target token (``rel_pos / max_camera_range``,
navrl_perception.py:1574,1578).  Here the token normalisation is bit-identical across arms and the
manipulation is purely "how honest is detection inside the same 20 m".  This launcher therefore
never sets that variable and asserts, per arm, that it was never set.

Direction, stated before running (prereg section 6): an honest sensor is expected to make detection
HARDER.  Arm B being worse is the EXPECTED outcome, not a failure; its magnitude estimates how much
of the record so far was owed to a sensor that cannot exist.  capture/crash/timeout are reported
RAW and never enter the verdict -- the frozen policy was trained against the dishonest sensor, so a
performance drop is a lineage fact, not a policy defect.

This experiment has NO decision authority.  It cannot revise the P2 STRICT FAIL or the D1 FAIL and
it cannot unlock P3.

Usage:
  python tools/run_navrl_ref5in_sensor_fidelity.py preflight
  python tools/run_navrl_ref5in_sensor_fidelity.py run
  python tools/run_navrl_ref5in_sensor_fidelity.py finalize
  python tools/run_navrl_ref5in_sensor_fidelity.py verify
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys


# ----------------------------------------------------------------------------------------------
# Preregistered contract constants.  Nothing below this block may recompute, relax or re-derive a
# value that appears here; every check compares against these names.  Declared ABOVE any
# measurement so that no number produced by a cell can reach back and change a threshold.
# ----------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "aerial_gym/rl_training/rl_games"
EVALUATOR = RL_ROOT / "eval_navrl_v2_density_sweep.sh"
IMPORT_ORIGIN_GUARD = RL_ROOT / "navrl_import_origin.py"
DETECTOR_SOURCE = ROOT / "aerial_gym/task/navrl_task/navrl_detector.py"
PERCEPTION_SOURCE = ROOT / "aerial_gym/task/navrl_task/navrl_perception.py"
TASK_CONFIG_SOURCE = ROOT / "aerial_gym/config/task_config/navrl_task_config.py"

SEED = 421                      # fresh, exhaustive-search usage count 0
BARS = 70
EPISODES = 2049
OUTPUT = ROOT / "results" / "navrl_ref5in_sensor_fidelity_seed421"
SOURCE_BUNDLE = OUTPUT / "source_bundle"
CHECKPOINT_SHA = "197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e"
CHECKPOINT_REL = (
    "aerial_gym/rl_training/rl_games/runs/"
    "ppo_260813_1636_navrl_v2-ref5in-d1-q3-adapt-s197/nn/"
    "last_gen_ppo_ep_1900_rew_182.11377.pth"
)
GOAL_DIST_MIN_M = 22.5
GOAL_DIST_MAX_M = 28.0
# (arm directory name, detect width, detect height, detector min pixels).  These four values are
# the WHOLE arm axis; every other condition is shared and asserted identical in verify_all().
ARMS = (("baseline", 160, 90, 2), ("fidelity", 1920, 1200, 50))
NEVER_ACQUIRED_COST_THRESHOLD_PP = 10.00
NEVER_ACQUIRED_NEUTRAL_BAND_PP = 3.00

# Prereg section 5-b (amended 2026-08-22, before any arm ran).  The generic evaluator puts
# ``cfg_detector_min_pixels`` in its v2 provenance ``want`` set
# (eval_navrl_v2_density_sweep.sh:675), so it demands that the detection threshold equal the value
# the checkpoint was TRAINED with.  Arm A (=2) passes; arm B (=50) necessarily does not.  That
# mismatch is not a defect -- it is the definition of this experiment, which evaluates a frozen
# policy at a threshold it never saw.
#
# The resolution is the repository's existing narrow-override pattern
# (run_navrl_ref5in_cv_heading_near_open.py:107-120), NOT a blanket force and NOT a new runtime
# flag.  It is STRICTER than a blanket ``NAVRL_V2_FORCE``: a blanket force masks every other
# mismatch as well, whereas this proves AT RUN TIME that the mismatch set is exactly this one
# field before any override is applied.  Two mismatch lines, or a different field, stop the run.
#
# The exact string the evaluator prints, captured verbatim from the unforced preflight.
EXPECTED_MISMATCH = "cfg_detector_min_pixels: checkpoint=2 expected=50.0"
FORCE_ARM = "fidelity"          # the ONLY arm authorised to carry the narrow override
NARROW_OVERRIDE_REASON = (
    "evaluating the frozen ref5in D1 checkpoint at a detection threshold it was not trained at "
    "is the manipulation itself; the evaluator's v2 provenance gate pins cfg_detector_min_pixels "
    "to the training value, so the single-field mismatch is verified at run time and only then "
    "overridden (prereg section 5-b)"
)

# Held fixed across arms and asserted in code, not merely described.  The preregistration turns on
# these staying put, so each one is a verification failure rather than a footnote.
DETECTOR_MAX_RANGE_M = 20.0     # NEVER exported; the config default IS the pinned value
CAMERA_WIDTH = 160              # RGB/perception resolution, identical in both arms
CAMERA_HEIGHT = 90

PREREGISTRATION = "docs/prereg_2026-08-22_sensor_fidelity.md"
PRODUCER = "tools/run_navrl_ref5in_sensor_fidelity.py"
SCOPE = "sensor_model_fidelity_frozen_ref5in_seed421"
ROBOT_NAME = "navrl_ref5in_quad"
ACTION_SELECTION = "deterministic"
REFLECTION_MODE = "original"
SPEED_GOVERNOR_MODE = "off"

VERDICT_COST_CONFIRMED = "FIDELITY_COST_CONFIRMED"
VERDICT_NEUTRAL = "FIDELITY_NEUTRAL"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE_SENSOR_FIDELITY"
VERDICT_FAIL_CLOSED = "FAIL_CLOSED_IMPLEMENTATION"

# The primary measurement is read from ONE evaluator-emitted field, never invented here:
# result["target_motion"]["first_acquisition"][outcome]["never_acquired"], the same per-outcome
# telemetry seed 359 introduced (navrl_task.py first_acquisition_payload) and seed 367 pooled for
# its manipulation check.  Pooling is a sum of counts over the three outcome cohorts, which
# partition the cell exactly -- see arm_measurements() for the accounting assertion.
NEVER_ACQUIRED_SOURCE = (
    'result["target_motion"]["first_acquisition"][outcome]["never_acquired"] '
    "summed over capture/crash/timeout, divided by the same cohorts' episode counts"
)
FIRST_ACQUISITION_OUTCOMES = ("capture", "crash", "timeout")
# The evaluator exports a median but no p90 of the first-visible step: first_acquisition_payload
# reduces its per-outcome histogram to a lower median only, and the histogram itself is not
# exported.  A p90 therefore cannot be derived from any recorded field, and inventing one from the
# mean would be a fabrication.  Recorded as null with this reason attached, rather than silently
# omitted (prereg section 6 asks for median AND p90).
FIRST_ACQUISITION_P90_UNAVAILABLE = (
    "navrl_task.py first_acquisition_payload() exports first_visible_step_mean and a lower "
    "first_visible_step_median only; the underlying per-outcome histogram "
    "(_fa_eval_outcome_first_hist) is not written to the result JSON, so no p90 is derivable "
    "from any recorded field. Recording it would require a runtime-source change, which this "
    "eval-only preregistration does not authorise."
)

# Gates this launcher owns outright -- there is no offline tool to delegate any of them to.  Each
# name maps to the verify_cell()/verify_all() key carrying the launcher's own proof that it ran the
# check.  A gate with no such evidence is a check nobody performed, so build_summary() fails closed
# on it rather than reporting "0 failed".
PER_ARM_GATES = {
    "G1_checkpoint_identity": "checkpoint_identity",
    "G2_result_receipt_binding": "result_receipt_binding",
    "G3_manifest_provenance": "manifest_provenance",
    "G4_runtime_clean": "manifest_provenance",
    "G5_import_origin": "import_origin",
    "G6_episode_contract": "episode_contract",
    "G7_arm_condition_pinned": "arm_condition",
}
CROSS_ARM_GATES = {
    "G8_runtime_byte_map_identity": "runtime_map_identity",
    "G9_single_axis_manipulation": "single_axis",
}

# The `run` log line the fail-closed import-origin guard prints (runner.py:31-33).  A MISSING line
# means the guard never ran, which is itself a failure.  The tree the line must name is NOT the
# verifier's own tree: Q6 asserts one relation internal to the artifact -- the aerial_gym that
# EXECUTED is the tree whose bytes the source manifest hashed -- so the expected path is derived
# per artifact from the manifest's repository_root, never from ROOT.  Compiling it against ROOT
# would make a result produced in a git worktree unverifiable from anywhere else.
ORIGIN_LOG_MARKER = "[origin] aerial_gym "
ORIGIN_LINE_HEAD = r"^\[origin\] aerial_gym "
ORIGIN_LINE_TAIL = r" sha256=(?P<sha256>[0-9a-f]{64}) \(enforced\)$"
ORIGIN_MANIFEST_ENTRY = "aerial_gym/__init__.py"


def origin_line_pattern(repository_root):
    """Matcher for the enforced [origin] line of the tree ``repository_root`` names."""
    return re.compile(
        ORIGIN_LINE_HEAD
        + re.escape(str(Path(repository_root) / ORIGIN_MANIFEST_ENTRY))
        + ORIGIN_LINE_TAIL
    )


# Prereg section 7.  Transcribed, not summarised: the limitations travel with the numbers.
LIMITATIONS = [
    "L1: 잡동사니 배경 미모델링. 렌더가 표적을 평평한 순수 빨강으로 칠하고 분할기가 색 규칙이라"
    " 픽셀 클래스가 자명하게 분리된다. 따라서 지름 8 px 임계는 하늘 배경 기준이며 실기의 도심·"
    "수목 배경(15–20 px)보다 관대하다. 본 실험 결과는 여전히 낙관 편향이다.",
    "L2: 모션 블러 미모델링. 5인치 기체가 고 yaw rate에서 10 ms 노출이면 12 px 표적이 여러"
    " 픽셀로 번진다. 현재 모델에 없다.",
    "L3: 거리 오차 0. navrl_detector.py:130이 정확한 해석적 교차 거리를 쓰고"
    " NAVRL_RANGE_ERROR_M=0이다. 정책은 임의 거리에서 오차 0의 거리를 받는다. 실기에서는"
    " 28 m 스테레오 시차가 1.2–2.4 px로 측정 불가다. 이 실험은 그것을 고치지 않으므로, 결과는"
    " '거리는 여전히 공짜로 주어진 상태에서의 검출 충실도'만 말한다.",
    "L4: 단일 정책·단일 seed·70막대 1조건. 계보나 밀도 전반으로 일반화하지 않는다.",
    "L5: 임계 50 px²는 문헌 중앙값이지 이 시스템에서 측정된 값이 아니다. 감이 아니라 유도된"
    " 값이지만 여전히 외부 기준의 이식이다.",
]

# summary.json comparison contract for `verify`.  created_at_utc is excluded by construction.
SUMMARY_VERIFY_KEYS = (
    "schema_version",
    "producer",
    "scope",
    "decision_authority",
    "p2_verdict_changed",
    "d1_verdict_changed",
    "p3_unlocked",
    "preregistration",
    "checkpoint",
    "checkpoint_sha256",
    "shared_condition",
    "narrow_provenance_override",
    "arms",
    "primary_metric",
    "never_acquired_delta_pp",
    "thresholds_pp",
    "verdict",
    "verdict_basis",
    "quality_gates",
    "failed_gates",
    "held_fixed",
    "import_origin",
    "limitations",
    "sources",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


P2 = load_module("sensor_fidelity_p2", ROOT / "tools/attest_navrl_ref5in_p2.py")
# Reuse -- never re-implement -- the audited contract helpers and the dirty-runtime gate.
BASE = load_module(
    "sensor_fidelity_base", ROOT / "tools/run_navrl_ref5in_cv_heading_diagnostic.py"
)

ContractError = BASE.ContractError
require = BASE.require
load_json = BASE.load_json
verify_runtime_clean_manifest = BASE.verify_runtime_clean_manifest
wilson = BASE.wilson


def _resolve_checkpoint() -> Path:
    """Resolve the pinned checkpoint, following the shared git dir when runs/ is not local.

    Git worktrees intentionally do not duplicate the gitignored multi-GB runs/ tree, so the
    checkpoint exists only in the primary worktree.  ``--git-common-dir`` names that worktree's
    .git; its parent is the primary root.  Identity is still pinned by CHECKPOINT_SHA, which is
    verified before anything is executed -- the path is a lookup, never a trust boundary.
    """
    local = ROOT / CHECKPOINT_REL
    if local.is_file():
        return local.resolve()
    common = Path(
        subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--git-common-dir"], universal_newlines=True
        ).strip()
    ).resolve()
    return (common.parent / CHECKPOINT_REL).resolve()


CHECKPOINT = _resolve_checkpoint()


# ----------------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------------


def cell_dir(arm: str) -> Path:
    return OUTPUT / "cells" / arm


def cell_paths(arm: str) -> dict:
    directory = cell_dir(arm)
    return {
        "result": directory / ("%dbars.json" % BARS),
        "receipt": directory / ("%dbars.receipt.json" % BARS),
        "log": directory / ("%dbars.log" % BARS),
        "snapshot": directory / "checkpoint_snapshot.pth",
        "stdout_log": directory / "sensor_fidelity.log",
    }


SUMMARY_JSON = OUTPUT / "summary.json"
SUMMARY_MD = OUTPUT / "summary.md"


def resolve_recorded_path(recorded, arm: str, label: str) -> Path:
    """Locate a receipt-recorded artifact without trusting the path the producer happened to write.

    The evaluator records ABSOLUTE paths, so a receipt produced in a git worktree names files that
    only exist there.  Migrating the result -- or deleting the worktree once the bytes are merged
    -- would then make the cell unverifiable, even though every byte is present.  Two candidates
    are tried, in order: the copy that travels with the cell, then the absolute path the receipt
    recorded.  Neither is a trust boundary; identity is pinned by the digests the caller checks
    (and P2.manifest_map re-hashes every snapshot it reads).  Nothing is resolved implicitly: if
    neither candidate exists the message names both and the check fails closed.
    """
    raw = str(recorded or "")
    require(bool(raw), f"{arm}: receipt records no {label}")
    recorded_path = Path(raw)
    local = cell_dir(arm) / recorded_path.name
    if local.exists():
        return local.resolve()
    absolute = recorded_path if recorded_path.is_absolute() else (cell_dir(arm) / recorded_path)
    if absolute.exists():
        return absolute.resolve()
    raise ContractError(
        f"{arm}: {label} not found; neither the cell-local copy {local} nor the recorded path "
        f"{absolute} exists"
    )


# ----------------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------------


def arm_requires_force(arm: str) -> bool:
    """Only the fidelity arm may carry the narrow provenance override (prereg section 5-b).

    Expressed as a function of the arm NAME rather than as a caller-supplied flag, so "arm A never
    forces" is a property of this launcher and not of whoever happens to call it.
    """
    return arm == FORCE_ARM


def canonical_env(
    arm: str, detect_w: int, detect_h: int, min_pixels: int, *, preflight: bool, force=None
):
    """P2's closed environment set plus exactly the prereg section 5 additions.

    PYTHONPATH re-injection is the single least obvious line in this file, so the reasoning is
    written out.  ``P2.canonical_env`` DELETES PYTHONPATH from the child environment
    (attest_navrl_ref5in_p2.py:245) to keep the P2 environment closed.  But play_navrl.sh:19 cds
    into ``<worktree>/aerial_gym/rl_training/rl_games``, where no ``aerial_gym/`` package directory
    exists.  With no PYTHONPATH, the PEP 660 editable-install finder -- which hard-codes the
    PRIMARY worktree's absolute path and is consulted after PathFinder finds nothing -- wins, and
    ``import aerial_gym`` resolves to the primary tree instead of this one.  Without the
    re-injection this run would EXECUTE the primary tree while its source manifest hashes THIS
    tree's bytes: a receipt that is internally consistent and describes code that never ran.  The
    re-injection must therefore happen AFTER canonical_env, or it is simply deleted.
    NAVRL_REQUIRE_SOURCE_ROOT below is the independent fail-closed check on the same fact (G5).

    The per-arm block is likewise ordered AFTER canonical_env for a concrete reason:
    ``NAVRL_DETECTOR_MIN_PIXELS=2`` is already inside P2's closed set
    (attest_navrl_ref5in_p2.py:258).  Updating it before the call would be silently overwritten and
    both arms would run at 2 px -- a two-arm experiment with one arm.  The assertion below proves
    the override actually landed.
    """
    env = P2.canonical_env(cell_dir(arm), preflight=preflight)
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "NAVRL_REQUIRE_SOURCE_ROOT": str(ROOT),
            "NAVRL_SEED": str(SEED),
            "NAVRL_V2_DENSITIES": str(BARS),
            "NAVRL_V2_RESULT_DIR": str(cell_dir(arm)),
            "NAVRL_V2_SHARED_SOURCE_BUNDLE": str(SOURCE_BUNDLE),
            # Fixed by the checkpoint contract, not by choice: frozen ref5in D1 ep1900 was trained
            # at 22.5-28 m and the generic evaluator refuses any other goal band (prereg 5).
            "NAVRL_V2_GOAL_DIST_MIN": "22.5",
            "NAVRL_V2_GOAL_DIST_MAX": "28",
            # Held fixed in BOTH arms; exported so the value is a receipt fact, not a default.
            "NAVRL_CAMERA_WIDTH": str(CAMERA_WIDTH),
            "NAVRL_CAMERA_HEIGHT": str(CAMERA_HEIGHT),
            # ---- the arm axis, and the ONLY three variables that differ between arms ----
            "NAVRL_DETECT_WIDTH": str(detect_w),
            "NAVRL_DETECT_HEIGHT": str(detect_h),
            "NAVRL_DETECTOR_MIN_PIXELS": str(min_pixels),
        }
    )
    # The band that is EXPORTED and the band that is later VERIFIED must be one value, or the
    # receipt would be checked against a number the run never used.
    require(
        float(env["NAVRL_V2_GOAL_DIST_MIN"]) == GOAL_DIST_MIN_M
        and float(env["NAVRL_V2_GOAL_DIST_MAX"]) == GOAL_DIST_MAX_M,
        "exported goal band drifted from the pinned constants: "
        f"{env['NAVRL_V2_GOAL_DIST_MIN']}-{env['NAVRL_V2_GOAL_DIST_MAX']} vs "
        f"{GOAL_DIST_MIN_M}-{GOAL_DIST_MAX_M}",
    )
    # Prereg section 5 and section 9: the detector range is NEVER set.  Seed 367 moved it and
    # renormalised the actor's target token along with it; that is the confound this experiment
    # exists to avoid.  Absence is the assertion -- the config default IS 20.0 m
    # (navrl_task_config.py:148), and exporting even the identical number here would make a future
    # per-arm value a one-character edit away.
    require(
        "NAVRL_DETECTOR_MAX_RANGE" not in env,
        "NAVRL_DETECTOR_MAX_RANGE leaked into the arm environment; prereg section 9 forbids "
        "touching the detector range at all (it renormalises the actor's target token)",
    )
    require(
        int(env["NAVRL_DETECTOR_MIN_PIXELS"]) == min_pixels,
        f"{arm}: NAVRL_DETECTOR_MIN_PIXELS is {env['NAVRL_DETECTOR_MIN_PIXELS']!r}, not the arm's "
        f"{min_pixels}; the per-arm update must land AFTER P2.canonical_env, whose closed set "
        "already contains a value for it",
    )
    require(
        int(env["NAVRL_DETECT_WIDTH"]) == detect_w and int(env["NAVRL_DETECT_HEIGHT"]) == detect_h,
        f"{arm}: exported detect resolution drifted from the arm definition",
    )
    require(
        int(env["NAVRL_CAMERA_WIDTH"]) == CAMERA_WIDTH
        and int(env["NAVRL_CAMERA_HEIGHT"]) == CAMERA_HEIGHT,
        "RGB camera resolution must be identical in both arms (prereg section 5)",
    )
    # Prereg section 4: detect != camera is an identity ONLY at zero appearance perturbation, and
    # the runtime fails closed otherwise.  Assert the closed set really is at zero rather than
    # relying on the child to raise after Isaac Gym has been allocated.
    for key in (
        "NAVRL_APP_HUE_DEG",
        "NAVRL_APP_LIGHT_GAIN",
        "NAVRL_APP_ALBEDO_JITTER",
        "NAVRL_APP_TEXTURE_STD",
        "NAVRL_APP_MOTION_BLUR",
        "NAVRL_DETECTION_LATENCY_S",
        "NAVRL_RANGE_ERROR_M",
    ):
        require(
            float(env[key]) == 0.0,
            f"{arm}: {key}={env[key]!r} is non-zero; detect-resolution decoupling is not an "
            "identity under it (prereg section 4)",
        )
    require(env["NAVRL_PERCEPTION_PERTURB"] == "0", f"{arm}: perturbations must be off")
    require(env["NAVRL_SPEED_GOVERNOR"] == SPEED_GOVERNOR_MODE, f"{arm}: governor must be off")
    require(env["NAVRL_V2_ACTION_MODE"] == ACTION_SELECTION, f"{arm}: action must be deterministic")
    require(
        env["NAVRL_EVAL_REFLECTION_MODE"] == REFLECTION_MODE,
        f"{arm}: reflection_mode must be original",
    )
    # Prereg section 5-b.  ``force=None`` means "whatever this arm is authorised to use"; an
    # explicit False is how verify_narrow_override() obtains the UNFORCED run whose refusal is the
    # proof.  An explicit True on an arm that is not FORCE_ARM is refused outright, so arm A can
    # never acquire an override through a caller's mistake.
    use_force = arm_requires_force(arm) if force is None else bool(force)
    require(
        not use_force or arm_requires_force(arm),
        f"{arm}: only the {FORCE_ARM} arm may carry a provenance override (prereg section 5-b)",
    )
    if use_force:
        env["NAVRL_V2_FORCE"] = "1"
    require(
        ("NAVRL_V2_FORCE" in env) == use_force,
        f"{arm}: provenance override state does not match the request (use_force={use_force})",
    )
    require(
        arm_requires_force(arm) or "NAVRL_V2_FORCE" not in env,
        f"{arm}: provenance override leaked into a non-{FORCE_ARM} arm's environment",
    )
    return env


def arm_env_diff() -> dict:
    """The per-arm environment difference, computed from the two environments that will be used.

    Not a comment claiming the arms differ in three variables -- the actual symmetric difference of
    the two dictionaries.  If anything else ever diverges (a stray export, a P2 change that reads
    the result dir, a future knob), this is where it surfaces, before any GPU time is spent.
    """
    environments = {}
    for arm, detect_w, detect_h, min_pixels in ARMS:
        environments[arm] = canonical_env(arm, detect_w, detect_h, min_pixels, preflight=True)
    (name_a, env_a), (name_b, env_b) = environments.items()
    # NAVRL_V2_RESULT_DIR must differ -- each arm writes its own cell -- and NAVRL_V2_FORCE differs
    # because only the fidelity arm carries the narrow override (prereg section 5-b).  Neither is
    # part of the manipulation, so both are listed explicitly rather than filtered silently: an
    # unexplained extra difference must still fail.
    manipulated = {"NAVRL_DETECT_WIDTH", "NAVRL_DETECT_HEIGHT", "NAVRL_DETECTOR_MIN_PIXELS"}
    bookkeeping = {"NAVRL_V2_RESULT_DIR", "NAVRL_V2_FORCE"}
    expected = manipulated | bookkeeping
    differing = {
        key
        for key in set(env_a) | set(env_b)
        if env_a.get(key) != env_b.get(key)
    }
    require(
        differing == expected,
        f"the two arm environments differ in {sorted(differing)}, not in {sorted(expected)}; "
        "sensor-model fidelity must be a single-axis manipulation (prereg section 2)",
    )
    require(
        "NAVRL_V2_FORCE" not in env_a and env_b.get("NAVRL_V2_FORCE") == "1",
        f"the narrow override must sit on {FORCE_ARM} alone: "
        f"{name_a}={env_a.get('NAVRL_V2_FORCE')!r} {name_b}={env_b.get('NAVRL_V2_FORCE')!r}",
    )
    return {key: {name_a: env_a.get(key), name_b: env_b.get(key)} for key in sorted(manipulated)}


# ----------------------------------------------------------------------------------------------
# Prerequisites
# ----------------------------------------------------------------------------------------------


def verify_prerequisites(*, require_clean: bool) -> None:
    require(CHECKPOINT.is_file(), f"pinned ref5in checkpoint missing: {CHECKPOINT}")
    require(
        P2.sha256_file(CHECKPOINT) == CHECKPOINT_SHA,
        "pinned ref5in checkpoint identity mismatch",
    )
    require(EVALUATOR.is_file(), f"canonical evaluator missing: {EVALUATOR}")
    require(IMPORT_ORIGIN_GUARD.is_file(), f"import-origin guard missing: {IMPORT_ORIGIN_GUARD}")
    runner = (RL_ROOT / "runner.py").read_text(encoding="utf-8")
    require(
        "[origin] aerial_gym %s sha256=%s (enforced)" in runner,
        "runner.py no longer prints the enforced import-origin line that G5 verifies",
    )

    # The decoupling this experiment rests on must be present in the runtime that will execute --
    # both the knobs and the fail-closed guard that makes them honest.  A missing guard would let
    # arm B run a high-resolution detection against a perturbed appearance, where the decoupling is
    # NOT an identity, and nothing downstream could tell.
    config = TASK_CONFIG_SOURCE.read_text(encoding="utf-8")
    for literal in (
        'detect_width = _env_int("NAVRL_DETECT_WIDTH", camera_width)',
        'detect_height = _env_int("NAVRL_DETECT_HEIGHT", camera_height)',
        'detector_max_range = _env_float("NAVRL_DETECTOR_MAX_RANGE", 20.0)',
    ):
        require(literal in config, f"detect-resolution decoupling knob missing: {literal}")
    detector = DETECTOR_SOURCE.read_text(encoding="utf-8")
    perception = PERCEPTION_SOURCE.read_text(encoding="utf-8")
    for source, literal in (
        (detector, "self.detect_decoupled = (self.detect_width != self.width) or ("),
        (detector, "NavRL detect-resolution decoupling (%dx%d detect vs %dx%d camera) is NOT "),
        (detector, "equivalent under appearance perturbation"),
        (detector, "NAVRL_DETECT_WIDTH/HEIGHT must be positive"),
        (perception, "self.detect_decoupled = (self.detect_width != self.width) or ("),
    ):
        require(literal in source, f"detect-resolution decoupling guard missing: {literal!r}")

    # The two arm environments must already be a single-axis difference before any GPU time.
    arm_env_diff()

    if require_clean:
        # Same dirty-runtime gate the audited CV diagnostic uses: no uncommitted byte inside the
        # snapshotted runtime roots may enter a receipt-bearing GPU run.
        status = subprocess.check_output(
            [
                "git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no",
                "--", "aerial_gym", "resources/robots", "tools/create_navrl_source_bundle.py",
            ],
            universal_newlines=True,
        )
        require(not status.strip(), "runtime source is dirty; commit the audit before the GPU run")


def run_preflight(arm: str, detect_w: int, detect_h: int, min_pixels: int, *, force):
    """One evaluator preflight for one arm, capturing its combined output.  Never starts a cell."""
    return subprocess.run(
        ["bash", str(EVALUATOR), str(CHECKPOINT), str(EPISODES)],
        cwd=str(ROOT),
        env=canonical_env(arm, detect_w, detect_h, min_pixels, preflight=True, force=force),
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def mismatch_lines(completed) -> list:
    """The evaluator's per-field provenance mismatch lines, as it printed them."""
    return [
        line.strip()
        for line in (completed.stdout or "").splitlines()
        if "checkpoint=" in line and "expected=" in line
    ]


def verify_narrow_override() -> str:
    """Prove the fidelity arm's provenance mismatch is EXACTLY one field, then override that one.

    Copied from the repository's existing narrow-override pattern
    (run_navrl_ref5in_cv_heading_near_open.py:107-120) and adapted to this experiment.

    The order matters and is the whole point.  The UNFORCED preflight must be refused with
    returncode 2, and its mismatch set must be exactly ``[EXPECTED_MISMATCH]`` -- one line, that
    field, nothing else.  Two lines, or a different field, means something OTHER than the
    preregistered manipulation also drifted, and this stops rather than overriding it.  Only after
    that proof is ``NAVRL_V2_FORCE=1`` applied and the preflight required to pass.

    This is stricter than a blanket force, not looser: a blanket force would accept -- and hide --
    any additional mismatch, while this verifies the mismatch set at run time before overriding it.
    """
    arm, detect_w, detect_h, min_pixels = next(row for row in ARMS if row[0] == FORCE_ARM)
    unforced = run_preflight(arm, detect_w, detect_h, min_pixels, force=False)
    lines = mismatch_lines(unforced)
    require(
        unforced.returncode == 2,
        f"{arm}: the generic evaluator accepted the raised detection threshold WITHOUT force "
        f"(returncode={unforced.returncode}); the narrow override would then be overriding "
        "nothing, and the mismatch this experiment is defined by has silently disappeared",
    )
    require(
        lines == [EXPECTED_MISMATCH],
        f"{arm}: forced mismatch set is not exactly one field: {lines}; the narrow override "
        f"authorises {EXPECTED_MISMATCH!r} and nothing else (prereg section 5-b)",
    )
    forced = run_preflight(arm, detect_w, detect_h, min_pixels, force=True)
    require(
        forced.returncode == 0,
        f"{arm}: preflight still failed under the narrow override "
        f"(returncode={forced.returncode}); tail: {(forced.stdout or '').strip().splitlines()[-12:]}",
    )
    require(
        "[eval_v2] PREFLIGHT PASS (evaluation not started)" in (forced.stdout or ""),
        f"{arm}: forced preflight returned 0 without the PREFLIGHT PASS marker",
    )
    return EXPECTED_MISMATCH


def preflight_evaluator() -> str:
    """Preflight every arm, with the narrow override applied to exactly one of them.

    Arm A sets no target pattern and no intervention, so its provenance must already match what the
    generic evaluator expects and it is required to pass with NO override at all -- a non-zero
    return there reports the mismatch lines and STOPS.  Arm B goes through
    verify_narrow_override(), which proves the single-field mismatch before overriding it.

    Called from BOTH ``preflight`` and ``run``, so a cell can never be produced under an
    unverified override.
    """
    for arm, detect_w, detect_h, min_pixels in ARMS:
        if arm_requires_force(arm):
            verify_narrow_override()
            print(
                f"[sensor-fidelity] arm {arm}: narrow provenance override VERIFIED "
                f"(sole mismatch: {EXPECTED_MISMATCH}) then preflight PASS"
            )
            continue
        completed = run_preflight(arm, detect_w, detect_h, min_pixels, force=False)
        if completed.returncode != 0:
            tail = (completed.stdout or "").strip().splitlines()[-12:]
            raise ContractError(
                f"{arm}: generic evaluator preflight did not pass cleanly "
                f"(returncode={completed.returncode}); this arm is preregistered to need NO "
                "provenance override, so it stops here instead of forcing. "
                f"mismatch lines: {mismatch_lines(completed) or 'none'} | tail: {tail}"
            )
        require(
            "[eval_v2] PREFLIGHT PASS (evaluation not started)" in completed.stdout,
            f"{arm}: evaluator preflight returned 0 without the PREFLIGHT PASS marker",
        )
        print(f"[sensor-fidelity] arm {arm}: evaluator preflight PASS (no override)")
    return EXPECTED_MISMATCH


# ----------------------------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------------------------


def tee_run(command: list, env: dict, log_path: Path) -> None:
    """Run a child, streaming its combined output to the console and to log_path.

    The log lands OUTSIDE the cell directory first: the evaluator refuses to start when its
    NAVRL_V2_RESULT_DIR already exists (eval_navrl_v2_density_sweep.sh:822), so nothing may create
    the cell directory ahead of it.  The finished log is moved in afterwards.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as sink:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            sink.write(line)
        process.stdout.close()
        returncode = process.wait()
    require(returncode == 0, f"evaluator arm failed with exit code {returncode}")


def run_arm(arm: str, detect_w: int, detect_h: int, min_pixels: int) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    staged_log = OUTPUT / f"{arm}.log.partial"
    print(
        f"[sensor-fidelity] RUN {arm} | detect {detect_w}x{detect_h} min_pixels={min_pixels} "
        f"| camera {CAMERA_WIDTH}x{CAMERA_HEIGHT} | detector range {DETECTOR_MAX_RANGE_M:.1f} m",
        flush=True,
    )
    tee_run(
        ["bash", str(EVALUATOR), str(CHECKPOINT), str(EPISODES)],
        canonical_env(arm, detect_w, detect_h, min_pixels, preflight=False),
        staged_log,
    )
    require(cell_dir(arm).is_dir(), f"{arm}: evaluator produced no result directory")
    staged_log.replace(cell_paths(arm)["stdout_log"])


# ----------------------------------------------------------------------------------------------
# Cell verification
# ----------------------------------------------------------------------------------------------


def verify_import_origin(arm: str, mapping: dict, metadata: dict) -> dict:
    """G5: prove from the run log that the executing aerial_gym IS the tree the manifest hashed.

    The invariant is internal to the artifact and says nothing about where the verifier stands.
    The expected tree is therefore read from the artifact -- the source manifest's
    ``repository_root`` -- and never from the tree this process happens to be running in.

    Three independent facts must line up.  First, runner.py must have printed the enforced
    `[origin]` line naming exactly <repository_root>/aerial_gym/__init__.py; a MISSING line means
    the guard never ran (NAVRL_REQUIRE_SOURCE_ROOT unset, or a runner without the guard), which
    fails closed rather than passing silently, and an `[origin]` line naming any OTHER tree is a
    failure too.  Second, all such lines must agree.  Third, the sha256 in that line must be the
    digest the source manifest recorded for aerial_gym/__init__.py, which ties the bytes that
    executed to the bytes that were hashed into the receipt.
    """
    recorded_root = str(metadata.get("repository_root") or "")
    require(
        bool(recorded_root) and Path(recorded_root).is_absolute(),
        f"{arm}: G5 source manifest records no absolute repository_root: {recorded_root!r}",
    )
    repository_root = Path(recorded_root)
    expected_origin = repository_root / ORIGIN_MANIFEST_ENTRY
    pattern = origin_line_pattern(repository_root)

    matches = []
    foreign = []
    with cell_paths(arm)["log"].open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            text = line.rstrip("\r\n")
            match = pattern.match(text)
            if match is not None:
                matches.append(match.group("sha256"))
            elif text.startswith(ORIGIN_LOG_MARKER):
                foreign.append(text)
    require(
        bool(matches),
        f"{arm}: G5 run log contains no enforced [origin] line for {expected_origin}; "
        "the import-origin guard did not run",
    )
    require(
        not foreign,
        f"{arm}: G5 run log names an aerial_gym origin that is not the manifest's "
        f"repository_root {repository_root}: {foreign[:4]}",
    )
    require(
        len(set(matches)) == 1,
        f"{arm}: G5 conflicting [origin] digests in the run log: {set(matches)}",
    )
    origin_sha = matches[0]
    entry = mapping.get(ORIGIN_MANIFEST_ENTRY)
    require(
        entry is not None,
        f"{arm}: G5 manifest has no runtime_files entry for {ORIGIN_MANIFEST_ENTRY}",
    )
    require(
        entry[0] == origin_sha,
        f"{arm}: G5 executed aerial_gym/__init__.py sha256 {origin_sha} is not the manifest "
        f"digest {entry[0]}",
    )
    return {
        # The generic per-arm gate loop proves ownership by requiring this marker, the same way
        # manifest_provenance carries it.  Omitting it made the launcher refuse to claim G5 had
        # passed even though every check inside this function had -- which is the assertion doing
        # its job, not a false alarm: evidence in the wrong shape is not evidence.
        "checked_by_launcher": True,
        "enforced": True,
        "required_source_root": str(repository_root),
        "origin": str(expected_origin),
        "origin_sha256": origin_sha,
        "manifest_entry": ORIGIN_MANIFEST_ENTRY,
        "manifest_sha256": entry[0],
        "log_line_occurrences": len(matches),
        "pythonpath_reinjected": str(repository_root),
    }


def require_import_origin_evidence(arm: str, import_origin) -> None:
    """G5 is owned by this launcher; this is the proof the launcher actually ran it.

    verify_import_origin() is the only producer of this shape.  If it were removed -- or never
    called -- the summary would otherwise still print "0 failed" while nobody had judged G5.
    """
    require(
        isinstance(import_origin, dict)
        and import_origin.get("enforced") is True
        and import_origin.get("manifest_entry") == ORIGIN_MANIFEST_ENTRY
        and re.fullmatch(r"[0-9a-f]{64}", str(import_origin.get("origin_sha256", ""))) is not None
        and import_origin.get("origin_sha256") == import_origin.get("manifest_sha256")
        and int(import_origin.get("log_line_occurrences") or 0) >= 1,
        f"{arm}: G5_import_origin is owned by this launcher, but the launcher produced no proof "
        f"it ran the check (verify_import_origin evidence: {import_origin!r})",
    )


def verify_cell(arm: str, detect_w: int, detect_h: int, min_pixels: int) -> dict:
    paths = cell_paths(arm)
    for key in ("result", "receipt", "log", "snapshot"):
        require(paths[key].is_file(), f"{arm}: missing artifact: {paths[key]}")
    result = load_json(paths["result"])
    receipt = load_json(paths["receipt"])

    require(
        P2.sha256_file(paths["result"]) == receipt.get("result_sha256"),
        f"{arm}: result/receipt hash mismatch",
    )
    require(
        P2.sha256_file(paths["snapshot"]) == CHECKPOINT_SHA,
        f"{arm}: evaluated checkpoint snapshot is not the pinned checkpoint",
    )
    require(
        receipt.get("source_checkpoint_sha256") == CHECKPOINT_SHA,
        f"{arm}: receipt source checkpoint mismatch",
    )
    checkpoint_identity = {
        "checked_by_launcher": True,
        "snapshot_sha256": CHECKPOINT_SHA,
        "receipt_source_checkpoint_sha256": receipt.get("source_checkpoint_sha256"),
    }
    result_receipt_binding = {
        "checked_by_launcher": True,
        "result_sha256": receipt.get("result_sha256"),
    }

    pinned = {
        "seed": SEED,
        "bars": BARS,
        "requested_episodes": EPISODES,
        "action_selection": ACTION_SELECTION,
        "reflection_mode": REFLECTION_MODE,
        "speed_governor_mode": SPEED_GOVERNOR_MODE,
        "goal_dist_min_m": GOAL_DIST_MIN_M,
        "goal_dist_max_m": GOAL_DIST_MAX_M,
        # Prereg section 5: the seed-367 confound must be absent from BOTH arms.  This is the
        # receipt-side proof that the detector range really was left at 20 m.
        "target_camera_max_range_m": DETECTOR_MAX_RANGE_M,
    }
    receipt_mismatch = {
        key: (receipt.get(key), value) for key, value in pinned.items() if receipt.get(key) != value
    }
    require(not receipt_mismatch, f"{arm}: receipt condition mismatch: {receipt_mismatch}")

    condition = result.get("condition") or {}
    condition_mismatch = {
        key: (condition.get(key), value)
        for key, value in {
            "seed": SEED,
            "bars": BARS,
            "robot_name": ROBOT_NAME,
            "action_selection": ACTION_SELECTION,
            "reflection_mode": REFLECTION_MODE,
            "speed_governor_mode": SPEED_GOVERNOR_MODE,
            "goal_dist_min_m": GOAL_DIST_MIN_M,
            "goal_dist_max_m": GOAL_DIST_MAX_M,
        }.items()
        if condition.get(key) != value
    }
    require(not condition_mismatch, f"{arm}: result condition mismatch: {condition_mismatch}")

    # The arm axis, proved from the artifact rather than from this launcher's own constants.  The
    # evaluator records detector_min_pixels in the v2 evaluation contract, which is inside the
    # result and therefore hash-bound to the receipt above.  The detect RESOLUTION is not recorded
    # by the evaluator anywhere (see held_fixed()/single-axis notes), so it is carried as a
    # requested condition and is not claimed to be independently attested.
    contract = result.get("v2_evaluation_contract") or {}
    require(
        int(contract.get("detector_min_pixels", -1)) == min_pixels,
        f"{arm}: v2 evaluation contract records detector_min_pixels="
        f"{contract.get('detector_min_pixels')!r}, not the arm's {min_pixels}",
    )
    require(
        float(contract.get("target_camera_max_range_m", -1.0)) == DETECTOR_MAX_RANGE_M,
        f"{arm}: v2 evaluation contract records target_camera_max_range_m="
        f"{contract.get('target_camera_max_range_m')!r}, not the pinned {DETECTOR_MAX_RANGE_M}",
    )
    arm_condition = {
        "checked_by_launcher": True,
        "detect_width_requested": detect_w,
        "detect_height_requested": detect_h,
        "detector_min_pixels_attested": int(contract.get("detector_min_pixels", -1)),
        "target_camera_max_range_m_attested": float(
            contract.get("target_camera_max_range_m", -1.0)
        ),
    }

    # The evaluator drains whole 128-env batches, so a cell finishes at or just past the request.
    # Exact equality is WRONG here (see run_navrl_ref5in_camera_range_control.py:162, where it
    # broke one arm that landed on 2,050).
    actual = int(result.get("actual_episodes", -1))
    require(
        int(result.get("requested_episodes", -1)) == EPISODES and actual >= EPISODES,
        f"{arm}: episode contract mismatch: requested={result.get('requested_episodes')} "
        f"actual={actual}",
    )
    episode_contract = {
        "checked_by_launcher": True,
        "requested_episodes": int(result.get("requested_episodes", -1)),
        "actual_episodes": actual,
        "comparator": "requested == EPISODES and actual >= EPISODES",
    }

    # Schema-2 receipt and the manifest it names.  Both files are located by resolve_recorded_path
    # -- the recorded absolute paths belong to the producing worktree, and a migrated artifact must
    # still verify -- and both are then pinned to the digests the receipt itself recorded, so the
    # location may move but the bytes may not.
    require(receipt.get("schema_version") == 2, f"{arm}: receipt is not a schema_version 2 receipt")
    manifest = resolve_recorded_path(
        receipt.get("runtime_source_manifest"), arm, "runtime source manifest"
    )
    require(
        P2.sha256_file(manifest) == receipt.get("runtime_source_manifest_sha256"),
        f"{arm}: runtime source manifest bytes differ from the receipt digest: {manifest}",
    )
    python_environment = resolve_recorded_path(
        receipt.get("python_environment_manifest"), arm, "python environment manifest"
    )
    require(
        P2.sha256_file(python_environment) == receipt.get("python_environment_manifest_sha256"),
        f"{arm}: python environment manifest bytes differ from the receipt digest: "
        f"{python_environment}",
    )
    mapping, metadata = P2.manifest_map(manifest, 2, require_original=False)
    verify_runtime_clean_manifest(metadata, arm)
    import_origin = verify_import_origin(arm, mapping, metadata)
    manifest_provenance = {
        "checked_by_launcher": True,
        "receipt_schema_version": receipt.get("schema_version"),
        "manifest_schema_version": metadata.get("schema_version"),
        "manifest_sha256": receipt.get("runtime_source_manifest_sha256"),
        "python_environment_manifest_sha256": receipt.get("python_environment_manifest_sha256"),
        "runtime_file_count": metadata.get("runtime_file_count"),
        "runtime_clean_verified": True,
    }

    return {
        "arm": arm,
        "result": result,
        "receipt": receipt,
        "condition": condition,
        "v2_evaluation_contract": contract,
        "runtime_map": mapping,
        "actual_episodes": actual,
        "checkpoint_identity": checkpoint_identity,
        "result_receipt_binding": result_receipt_binding,
        "arm_condition": arm_condition,
        "episode_contract": episode_contract,
        "manifest_provenance": manifest_provenance,
        "import_origin": import_origin,
    }


def verify_all() -> dict:
    """Per-arm verification, then the two cross-arm invariants the experiment rests on."""
    cells = {}
    for arm, detect_w, detect_h, min_pixels in ARMS:
        cells[arm] = verify_cell(arm, detect_w, detect_h, min_pixels)

    (name_a, cell_a), (name_b, cell_b) = (
        (arm, cells[arm]) for arm, _, _, _ in ARMS
    )

    # G8: the two arms must have executed the SAME bytes.  Different runtime maps would make the
    # comparison a comparison of two programs, not of two sensor models.
    require(
        cell_a["runtime_map"] == cell_b["runtime_map"],
        "the two arms used different runtime byte maps; this is not a single-axis manipulation",
    )
    runtime_map_identity = {
        "checked_by_launcher": True,
        "identical": True,
        "runtime_file_count": cell_a["manifest_provenance"]["runtime_file_count"],
    }

    # G9: every held-fixed condition key must match, and the ONLY permitted differences are the
    # detect resolution and the min-pixel threshold.  The v2 evaluation contract is the evaluator's
    # own record of every perception/task knob it ran with, so comparing the whole dictionary --
    # rather than a hand-picked list -- is what makes "only these differ" a measurement instead of
    # a claim.  detector_min_pixels is the sole authorised difference; the detect RESOLUTION does
    # not appear in the contract at all (the evaluator records no detect_width/detect_height), so
    # it can differ without showing up here, which is stated rather than glossed.
    contract_a = cell_a["v2_evaluation_contract"]
    contract_b = cell_b["v2_evaluation_contract"]
    require(
        set(contract_a) == set(contract_b),
        "the two arms' v2 evaluation contracts have different key sets: "
        f"{sorted(set(contract_a) ^ set(contract_b))}",
    )
    differing = sorted(key for key in contract_a if contract_a[key] != contract_b[key])
    require(
        differing == ["detector_min_pixels"],
        f"the arms differ in {differing}, but detector_min_pixels is the only authorised "
        "difference in the evaluation contract (prereg section 5)",
    )
    # Held-fixed conditions named explicitly by the preregistration, asserted individually so a
    # failure message says WHICH one moved rather than only that something did.
    held_fixed = {}
    for key in (
        "target_camera_max_range_m",
        "detector_threshold",
        "perception_perturb",
        "detection_dropout_active",
        "detection_latency_s",
        "range_error_m",
        "appearance_appearance_hue_deg",
        "appearance_appearance_light_gain",
        "appearance_appearance_albedo_jitter",
        "appearance_appearance_texture_std",
        "appearance_appearance_motion_blur",
        "camera_mount_rot_deg",
        "camera_mount_trans_m",
        "camera_fov_scale_err",
        "goal_dist_min_m",
        "goal_dist_max_m",
        "obstacle_selector",
        "action_selection",
        "reflection_mode",
        "speed_governor_mode",
        "seed",
    ):
        require(
            key in contract_a and contract_a[key] == contract_b[key],
            f"held-fixed condition {key} differs between arms: "
            f"{contract_a.get(key)!r} vs {contract_b.get(key)!r}",
        )
        held_fixed[key] = contract_a[key]
    require(
        float(held_fixed["target_camera_max_range_m"]) == DETECTOR_MAX_RANGE_M,
        "both arms must run at the pinned 20.0 m detector range (prereg section 5); "
        f"got {held_fixed['target_camera_max_range_m']!r}",
    )
    for key in (
        "appearance_appearance_hue_deg",
        "appearance_appearance_light_gain",
        "appearance_appearance_albedo_jitter",
        "appearance_appearance_texture_std",
        "appearance_appearance_motion_blur",
        "detection_dropout_active",
        "detection_latency_s",
        "range_error_m",
    ):
        require(
            float(held_fixed[key]) == 0.0,
            f"{key} must be 0 in both arms (prereg section 4/5); got {held_fixed[key]!r}",
        )
    single_axis = {
        "checked_by_launcher": True,
        "evaluation_contract_differences": differing,
        "authorised_differences": ["detector_min_pixels"],
        "detect_resolution_not_recorded_by_evaluator": True,
    }

    return {
        "cells": cells,
        "order": (name_a, name_b),
        "held_fixed": held_fixed,
        "runtime_map_identity": runtime_map_identity,
        "single_axis": single_axis,
    }


# ----------------------------------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------------------------------


def arm_measurements(cell: dict) -> dict:
    """Prereg section 6 measurands for one arm.

    never-acquired is READ, not invented: the evaluator's per-outcome first-acquisition telemetry
    already counts episodes that never acquired the target
    (navrl_task.py first_acquisition_payload -> result.target_motion.first_acquisition).  Pooling is
    a sum of counts across the capture/crash/timeout cohorts, which partition the cell; the
    accounting assertion below is what makes the pool a rate over the whole cell rather than over
    whichever cohorts happened to be present.
    """
    result = cell["result"]
    rows = ((result.get("target_motion") or {}).get("first_acquisition") or {})
    require(
        set(rows) == set(FIRST_ACQUISITION_OUTCOMES),
        f"{cell['arm']}: first-acquisition outcome labels are {sorted(rows)}, expected "
        f"{list(FIRST_ACQUISITION_OUTCOMES)}",
    )
    never = sum(int(rows[label]["never_acquired"]) for label in FIRST_ACQUISITION_OUTCOMES)
    episodes = sum(int(rows[label]["episodes"]) for label in FIRST_ACQUISITION_OUTCOMES)
    acquired = sum(int(rows[label]["acquired"]) for label in FIRST_ACQUISITION_OUTCOMES)
    require(
        episodes == cell["actual_episodes"],
        f"{cell['arm']}: first-acquisition cohorts cover {episodes} episodes but the cell ran "
        f"{cell['actual_episodes']}",
    )
    require(
        never + acquired == episodes,
        f"{cell['arm']}: never_acquired + acquired != episodes ({never} + {acquired} != {episodes})",
    )

    outcome = result["outcome"]
    raw = {}
    for name, count_name in (("capture", "captured"), ("crash", "crash"), ("timeout", "timeout")):
        count = int(outcome[count_name])
        raw[name] = count
        raw[f"{name}_rate"] = count / episodes
        raw[f"{name}_wilson95"] = wilson(count, episodes)

    return {
        "episodes": episodes,
        "never_acquired": never,
        "never_acquired_rate": never / episodes,
        "never_acquired_rate_pp": 100.0 * never / episodes,
        "never_acquired_wilson95": wilson(never, episodes),
        "acquired": acquired,
        "first_acquisition_by_outcome": {
            label: {
                "episodes": int(rows[label]["episodes"]),
                "never_acquired": int(rows[label]["never_acquired"]),
                "never_acquired_rate": rows[label]["never_acquired_rate"],
                "acquired": int(rows[label]["acquired"]),
                "first_visible_step_mean": rows[label]["first_visible_step_mean"],
                "first_visible_step_median": rows[label]["first_visible_step_median"],
                "first_visible_step_p90": None,
                "visible_hidden_transitions_mean_per_episode": rows[label][
                    "visible_hidden_transitions_mean_per_episode"
                ],
            }
            for label in FIRST_ACQUISITION_OUTCOMES
        },
        "first_visible_step_p90_unavailable_reason": FIRST_ACQUISITION_P90_UNAVAILABLE,
        "target_hidden_fraction": float(
            result["action"]["context"]["target_hidden"]["fraction"]
        ),
        # Reported RAW and excluded from the verdict by construction -- see classify_verdict(),
        # which takes a single delta and cannot see these at all (prereg section 6).
        "outcome_raw": raw,
    }


def classify_verdict(delta_pp: float) -> str:
    """Prereg section 6 decision rule, applied to the never-acquired delta in PERCENTAGE POINTS.

    The delta is `fidelity` minus `baseline`: an honest sensor is EXPECTED to raise never-acquired,
    so a large positive delta is the confirmatory outcome, not a failure.

    This function takes exactly one number.  capture, crash and timeout are not parameters, are not
    read from any global, and therefore cannot enter the verdict -- which is the preregistration's
    requirement, expressed as a signature rather than as a promise.
    """
    if delta_pp >= NEVER_ACQUIRED_COST_THRESHOLD_PP:
        return VERDICT_COST_CONFIRMED
    if abs(delta_pp) <= NEVER_ACQUIRED_NEUTRAL_BAND_PP:
        return VERDICT_NEUTRAL
    return VERDICT_INCONCLUSIVE


# ----------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------


def gate_table(verified: dict) -> dict:
    """Build the gate table from EVIDENCE, not from a list of names.

    ``len(gates)`` counts dictionary keys, which is true of any table and therefore reports
    nothing.  Every gate here is owned by this launcher, so each one is marked passed only when the
    verify step that owns it produced its evidence dictionary; a gate whose evidence is missing is
    a check nobody performed and fails closed rather than being tallied as a pass.
    """
    gates = {}
    for gate, evidence_key in sorted(PER_ARM_GATES.items()):
        per_arm = {}
        for arm, _, _, _ in ARMS:
            evidence = verified["cells"][arm].get(evidence_key)
            require(
                isinstance(evidence, dict) and evidence.get("checked_by_launcher") is True,
                f"{arm}: quality gate {gate} is owned by this launcher, but verify_cell() "
                f"produced no {evidence_key} evidence that the launcher ran the check",
            )
            per_arm[arm] = True
        gates[gate] = {"owner": PRODUCER, "scope": "per_arm", "passed": True, "arms": per_arm}
    for gate, evidence_key in sorted(CROSS_ARM_GATES.items()):
        evidence = verified.get(evidence_key)
        require(
            isinstance(evidence, dict) and evidence.get("checked_by_launcher") is True,
            f"quality gate {gate} is owned by this launcher, but verify_all() produced no "
            f"{evidence_key} evidence that the launcher ran the check",
        )
        gates[gate] = {"owner": PRODUCER, "scope": "cross_arm", "passed": True}
    for arm, _, _, _ in ARMS:
        require_import_origin_evidence(arm, verified["cells"][arm].get("import_origin"))
    return gates


def gate_tally(payload: dict) -> tuple:
    """(evaluated, failed) for the summary line -- with every gate's ownership proved.

    A gate carrying no boolean verdict is not counted as evaluated; it is a hole, and it fails
    closed here rather than quietly making the tally read "0 failed".
    """
    gates = payload.get("quality_gates") or {}
    unowned = [
        name
        for name, gate in gates.items()
        if not isinstance(gate, dict)
        or not isinstance(gate.get("passed"), bool)
        or gate.get("owner") != PRODUCER
    ]
    require(
        not unowned,
        f"quality gates {unowned} carry no boolean verdict from their named owner; nobody "
        "judged them",
    )
    failing = {name for name, gate in gates.items() if gate.get("passed") is False}
    recorded_failed = set(payload.get("failed_gates") or [])
    require(
        failing == recorded_failed,
        f"failed_gates {sorted(recorded_failed)} disagrees with the per-gate verdicts "
        f"{sorted(failing)}",
    )
    return sorted(gates), sorted(recorded_failed)


def build_summary(verified: dict) -> dict:
    gates = gate_table(verified)
    failed_gates = sorted(name for name, gate in gates.items() if gate.get("passed") is False)

    measurements = {
        arm: arm_measurements(verified["cells"][arm]) for arm, _, _, _ in ARMS
    }
    baseline_name, fidelity_name = verified["order"]
    delta_pp = (
        measurements[fidelity_name]["never_acquired_rate_pp"]
        - measurements[baseline_name]["never_acquired_rate_pp"]
    )

    # Gate 0 (prereg section 6): implementation feasibility.  If any owned gate failed, nothing may
    # be claimed about the sensor model at all, and the measurements key must be null.  That is a
    # BICONDITIONAL and it is enforced rather than described: a FAIL_CLOSED verdict carrying
    # measurements would publish statistics the prereg says are not to be interpreted, and a null
    # measurements key under any other verdict would publish a verdict with nothing behind it.
    if failed_gates:
        verdict = VERDICT_FAIL_CLOSED
        published = None
        basis = None
        published_delta = None
    else:
        verdict = classify_verdict(delta_pp)
        published = measurements
        published_delta = delta_pp
        basis = {
            "metric": "pooled never-acquired rate over all outcomes",
            "source_field": NEVER_ACQUIRED_SOURCE,
            "baseline_arm": baseline_name,
            "fidelity_arm": fidelity_name,
            "baseline_never_acquired_pp": measurements[baseline_name]["never_acquired_rate_pp"],
            "fidelity_never_acquired_pp": measurements[fidelity_name]["never_acquired_rate_pp"],
            "delta_pp": delta_pp,
            "direction": "fidelity minus baseline; positive means the honest sensor acquires less",
            "outcome_rates_excluded_from_verdict": True,
        }
    require(
        (verdict == VERDICT_FAIL_CLOSED) == (published is None),
        f"fail-closed contract violated: verdict={verdict} with measurements="
        + ("null" if published is None else "present")
        + f"; {VERDICT_FAIL_CLOSED} requires a null arms/measurements payload and a null payload "
        f"requires {VERDICT_FAIL_CLOSED}",
    )

    arms_payload = None
    if published is not None:
        arms_payload = {}
        for arm, detect_w, detect_h, min_pixels in ARMS:
            arms_payload[arm] = {
                "condition": {
                    "detect_width": detect_w,
                    "detect_height": detect_h,
                    "detector_min_pixels": min_pixels,
                    "camera_width": CAMERA_WIDTH,
                    "camera_height": CAMERA_HEIGHT,
                    "target_camera_max_range_m": DETECTOR_MAX_RANGE_M,
                    "actual_episodes": verified["cells"][arm]["actual_episodes"],
                    "num_envs": verified["cells"][arm]["condition"].get("num_envs"),
                    "episode_len_steps": verified["cells"][arm]["condition"].get(
                        "episode_len_steps"
                    ),
                },
                "measurements": published[arm],
            }

    return {
        "schema_version": 1,
        "producer": PRODUCER,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": SCOPE,
        "decision_authority": "none",
        "p2_verdict_changed": False,
        "d1_verdict_changed": False,
        "p3_unlocked": False,
        "preregistration": PREREGISTRATION,
        "checkpoint": CHECKPOINT_REL,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "shared_condition": {
            "seed": SEED,
            "bars": BARS,
            "requested_episodes_per_arm": EPISODES,
            "robot_name": ROBOT_NAME,
            "action_selection": ACTION_SELECTION,
            "reflection_mode": REFLECTION_MODE,
            "speed_governor_mode": SPEED_GOVERNOR_MODE,
            "goal_dist_min_m": GOAL_DIST_MIN_M,
            "goal_dist_max_m": GOAL_DIST_MAX_M,
            "target_camera_max_range_m": DETECTOR_MAX_RANGE_M,
            "camera_width": CAMERA_WIDTH,
            "camera_height": CAMERA_HEIGHT,
            "manipulated_axis": "sensor_model_fidelity(detect_resolution + detector_min_pixels)",
        },
        # Prereg section 5-b.  Per arm, because the whole point is that ONE arm carries it: a
        # single top-level flag would lose the fact that the baseline ran with no override at all.
        "narrow_provenance_override": {
            arm: (
                {
                    "used": True,
                    "arm": arm,
                    "sole_verified_mismatch": EXPECTED_MISMATCH,
                    "reason": NARROW_OVERRIDE_REASON,
                }
                if arm_requires_force(arm)
                else {"used": False}
            )
            for arm, _, _, _ in ARMS
        },
        "arms": arms_payload,
        "primary_metric": "pooled_never_acquired_rate",
        "never_acquired_delta_pp": published_delta,
        "thresholds_pp": {
            "cost_confirmed_at_or_above": NEVER_ACQUIRED_COST_THRESHOLD_PP,
            "neutral_band_abs": NEVER_ACQUIRED_NEUTRAL_BAND_PP,
        },
        "verdict": verdict,
        "verdict_basis": basis,
        "quality_gates": gates,
        "failed_gates": failed_gates,
        "held_fixed": verified["held_fixed"],
        "import_origin": {
            arm: verified["cells"][arm]["import_origin"] for arm, _, _, _ in ARMS
        },
        "limitations": list(LIMITATIONS),
        "sources": {
            arm: {
                "evaluation_result": str(cell_paths(arm)["result"].relative_to(ROOT)),
                "evaluation_receipt": str(cell_paths(arm)["receipt"].relative_to(ROOT)),
                "evaluation_log": str(cell_paths(arm)["log"].relative_to(ROOT)),
                "launcher_log": str(cell_paths(arm)["stdout_log"].relative_to(ROOT)),
            }
            for arm, _, _, _ in ARMS
        },
    }


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _step(value) -> str:
    return "—" if value is None else f"{value:.0f}" if isinstance(value, float) else str(value)


def write_summary(payload: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    evaluated, failed = gate_tally(payload)
    arms = payload.get("arms") or {}

    lines = [
        "# 센서 모델 충실도 (seed 421, 70 bars, eval-only)",
        "",
        f"**판정: `{payload['verdict']}`**",
        "",
    ]
    if arms:
        lines.extend([
            "| arm | detect | min_px | never-acquired | capture | crash | timeout | "
            "target_hidden |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for arm, detect_w, detect_h, min_pixels in ARMS:
            block = arms[arm]
            m = block["measurements"]
            raw = m["outcome_raw"]
            lines.append(
                f"| {arm} | {detect_w}×{detect_h} | {min_pixels} | "
                f"{_pct(m['never_acquired_rate_pp'])} | {_pct(raw['capture_rate'] * 100)} | "
                f"{_pct(raw['crash_rate'] * 100)} | {_pct(raw['timeout_rate'] * 100)} | "
                f"{_pct(m['target_hidden_fraction'] * 100)} |"
            )
        delta = payload["never_acquired_delta_pp"]
        lines.extend([
            "",
            f"**never-acquired 차이 (fidelity − baseline): {delta:+.2f} pp** "
            f"(임계 +{NEVER_ACQUIRED_COST_THRESHOLD_PP:.2f} pp / 중립대 "
            f"±{NEVER_ACQUIRED_NEUTRAL_BAND_PP:.2f} pp)",
            "",
            "| arm | outcome | 최초획득 중앙값 | p90 | never-acq |",
            "|---|---|---:|---:|---:|",
        ])
        for arm, _, _, _ in ARMS:
            for label in FIRST_ACQUISITION_OUTCOMES:
                row = arms[arm]["measurements"]["first_acquisition_by_outcome"][label]
                rate = row["never_acquired_rate"]
                lines.append(
                    f"| {arm} | {label} | {_step(row['first_visible_step_median'])} | "
                    f"{_step(row['first_visible_step_p90'])} | "
                    f"{'—' if rate is None else f'{rate * 100:.2f}%'} |"
                )
        lines.extend([
            "",
            f"- p90은 기록하지 못했다: {FIRST_ACQUISITION_P90_UNAVAILABLE}",
        ])
    else:
        lines.append(
            "게이트 0(구현 타당성)이 실패했으므로 측정값을 게재하지 않는다 "
            "(사전등록 §6)."
        )

    lines.extend([
        "",
        "## 방향 — arm B가 나빠지는 것은 예상된 결과다",
        "",
        "정직해진 센서는 검출을 **더 어렵게** 만든다(면적 임계 2 → 50 px², 25배). 따라서 fidelity",
        "arm의 never-acquired 상승은 **실패가 아니라 예상된 결과**이며, 그 크기가 곧 \"지금까지의",
        "성적 중 얼마가 존재할 수 없는 센서 덕분이었는가\"의 추정치다. 이 실험의 가치는 개선이",
        "아니라 **정직한 기준선의 확립**에 있다.",
        "",
        "**capture/crash/timeout은 원값으로만 보고하며 판정에 쓰지 않는다.** 동결 정책은 부정직한",
        "센서로 학습됐으므로 정직한 센서에서의 성능 저하는 정책의 결함이 아니라 계보의 결과다.",
        "판정 함수 `classify_verdict()`는 never-acquired 차이 하나만 인자로 받으므로 구조적으로",
        "이 값들을 볼 수 없다.",
        "",
        "## 고정된 조건",
        "",
        f"- `detector_max_range` = {DETECTOR_MAX_RANGE_M:.1f} m, **양 arm 동일** — seed 367은 이 값을"
        " 바꾸며 actor 표적 토큰까지 재정규화했고 그것이 교란이었다. 본 실험은 이 변수를 아예"
        " export하지 않는다.",
        f"- RGB 카메라 {CAMERA_WIDTH}×{CAMERA_HEIGHT} 양 arm 동일, appearance 교란 전부 0,"
        " 검출 지연·거리오차 0, governor off, deterministic, reflection_mode original.",
        f"- 두 arm의 runtime 바이트 맵이 동일하고, 평가 계약에서 다른 값은"
        " `detector_min_pixels` 단 하나다.",
        "",
        "## provenance override (사전등록 §5-b)",
        "",
        f"- **baseline arm은 override를 쓰지 않았다** (`used: false`).",
        f"- **{FORCE_ARM} arm만** 좁은 단일 필드 override를 쓴다. 실행 시점에 force 없이 먼저"
        f" 돌려 `returncode == 2`와 불일치 라인 집합이 정확히 `[{EXPECTED_MISMATCH}]` 하나임을"
        " 증명한 **뒤에만** `NAVRL_V2_FORCE=1`을 적용한다. 두 줄이거나 다른 필드면 중단한다.",
        "- 담요식 force보다 **더 엄격하다**: 담요식은 다른 불일치까지 함께 가리지만, 이 절차는"
        " 불일치가 그 한 필드뿐임을 실행 시점에 증명한다. 이 검증은 `preflight`와 `run` 양쪽에서"
        " 수행되므로 검증되지 않은 override 아래에서 셀이 생성될 수 없다.",
        "",
        f"- checkpoint SHA-256 `{payload['checkpoint_sha256'][:16]}…`",
        f"- 품질 게이트: 판정 {len(evaluated)}개, 실패 {len(failed)}개",
        "",
        "## 권한",
        "",
        "이 실험은 **결정 권한이 없다**. P2 STRICT FAIL·D1 FAIL을 소급 변경하지 않으며 P3를 "
        "해제하지 않는다 (`p2_verdict_changed`/`d1_verdict_changed`/`p3_unlocked` 전부 false).",
        "",
        "## 한계 (사전등록 §7)",
        "",
    ])
    lines.extend("- " + item for item in LIMITATIONS)
    lines.append("")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    require(
        mode in {"preflight", "run", "finalize", "verify"},
        "usage: run_navrl_ref5in_sensor_fidelity.py {preflight|run|finalize|verify}",
    )

    if mode == "preflight":
        verify_prerequisites(require_clean=False)
        require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
        diff = arm_env_diff()
        preflight_evaluator()
        print(
            f"[sensor-fidelity] PREFLIGHT PASS | seed={SEED} bars={BARS} "
            f"episodes={EPISODES}/arm | narrow override on {FORCE_ARM} only"
        )
        for key, value in diff.items():
            print(f"[sensor-fidelity]   {key}: {value}")
        print(f"[sensor-fidelity]   NAVRL_V2_FORCE: {{'baseline': None, '{FORCE_ARM}': '1'}}")
        print(
            f"[sensor-fidelity]   NAVRL_DETECTOR_MAX_RANGE: never exported "
            f"(config default {DETECTOR_MAX_RANGE_M:.1f} m, identical in both arms)"
        )
        return 0

    if mode == "run":
        require(not OUTPUT.exists(), f"refusing overwrite: {OUTPUT}")
        verify_prerequisites(require_clean=True)
        # The same narrow-override proof the `preflight` subcommand runs, repeated here rather than
        # trusted from an earlier invocation.  Without this a cell could be produced under an
        # override nobody verified in this process (prereg section 5-b).
        preflight_evaluator()
        for arm, detect_w, detect_h, min_pixels in ARMS:
            run_arm(arm, detect_w, detect_h, min_pixels)
        verified = verify_all()
        payload = build_summary(verified)
        print(f"[sensor-fidelity] RUN COMPLETE | verdict={payload['verdict']} | next: finalize")
        return 0

    verified = verify_all()
    expected = build_summary(verified)
    if mode == "finalize":
        write_summary(expected)
        print(f"[sensor-fidelity] FINALIZE PASS | {expected['verdict']} -> {SUMMARY_JSON}")
        return 0

    require(SUMMARY_JSON.is_file(), f"summary missing: {SUMMARY_JSON}")
    recorded = load_json(SUMMARY_JSON)
    for key in SUMMARY_VERIFY_KEYS:
        require(recorded.get(key) == expected.get(key), f"summary changed: {key}")
    print(f"[sensor-fidelity] VERIFY PASS | {recorded['verdict']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ContractError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[sensor-fidelity] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
