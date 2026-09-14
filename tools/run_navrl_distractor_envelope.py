#!/usr/bin/env python3
"""Appearance-distractor envelope -- preregistered EVALUATION-ONLY launcher.

Preregistration: docs/prereg_2026-09-01_distractor_envelope.md (frozen; sections 3, 4, 5, 6, 8
binding).  Nothing here trains, adapts or replaces a detector, and nothing here may be run with a
detector modification in the tree: this experiment quantifies a defect, it does not fix one
(prereg section 1 and section 6).

A 2 x 4 FACTORIAL of the SAME frozen ref5in D1 ep1900 policy under the SAME arena and sensor
geometry -- eight evaluation cells, two factors, nothing else moving:

  factor 1  detector           default (built-in AppearanceTargetSegmenter, threshold 0.55)
                               v7      (navrl_target_detector_v7_confirmatory.pth, threshold 0.700)
  factor 2  NAVRL_DISTRACTOR_COUNT   0, 1, 3, 5

  cells  default_n0 default_n1 default_n3 default_n5
         v7_n0      v7_n1      v7_n3      v7_n5

WHY BOTH DETECTORS (prereg section 3, "현행 기본값과 학습형 v7 둘 다").  Testing only the default
would measure a five-parameter colour rule failing at colour discrimination, which is very nearly a
tautology.  The open question is whether the LEARNED detector escaped the colour shortcut, and v7
is the artifact carrying the 0.99766 frame-precision claim section 1 puts under examination.  v7 is
evaluated at ITS OWN validation-selected threshold 0.700, because that is the operating point the
claim was measured at; a detector here is an artifact plus an operating point, and the two move
together as one factor level.

Gate F (section 5) is applied PER DETECTOR, each on its own N=5 cell, so the run produces TWO
verdicts.  There is no pooled verdict anywhere: the default detector's result is not evidence about
v7, and v7's is not evidence about the default.

Gate 0.3 (the N=0 lineage regression) applies to `default_n0` alone -- it is the only cell with a
lineage to reproduce.  `v7_n0` is descriptive; see verify_gate0_lineage_regression.

WHAT IS MEASURED (prereg section 4).  False Target Lock Rate: over the frames where the detector
reports visible, the three-way classification of the position the detector REPORTED against the
ground-truth target and distractor centres --

    TARGET_LOCK      within 0.5 m of the target centre
    DISTRACTOR_LOCK  within 0.5 m of some distractor centre
    GHOST_LOCK       neither (the empty space a two-blob centroid average points at)

    FTLR = (DISTRACTOR_LOCK + GHOST_LOCK) / visible frames

The classification lives in navrl_task._record_distractor_lock_frame, which is where the
evaluation-only ground-truth consumers already live (first-acquisition telemetry, OOB exit
forensics).  Ground truth labels the metric and reaches nothing else: no task_obs, no reward, no
termination, no checkpointed field (CLAUDE.md observation contract).  The recorder is not merely
disabled at zero distractors, it is not called and its accumulators are not allocated.

WHY THE CAMERA-RESOLUTION PATH IS THE ONLY LEGAL PATH (prereg section 5, Gate 0.2).  Decoupled
detect resolution replaces the high-resolution RGB render plus segmentation with the
high-resolution TARGET mask.  That substitution is an identity only while the target is the sole
painted object; distractors painted the same colour make perception's segmenter fire on pixels the
target ray-cast never produced, and the two halves then disagree SILENTLY.  navrl_detector.py
refuses the combination outright, and this launcher asserts NAVRL_DETECT_WIDTH/HEIGHT equal the
camera resolution in every cell rather than relying on that refusal.

WHICH DIRECTION IS THE EXPECTED ONE (prereg section 5).  ``COLOR_SHORTCUT_CONFIRMED`` is the
PREDICTED outcome and is NOT an experimental failure -- _detect_rgbd collapses every positive pixel
in the frame to one centroid with no connected-component step, so a frame containing two same
coloured blobs is structurally unable to report either of them.  ``DETECTOR_ROBUST_TO_DISTRACTORS``
would be the surprising result, and the first response to it is to re-check that distractors
actually rendered.

capture / crash / timeout are reported RAW and are excluded from the verdict by construction --
classify_verdict() takes one number and can see nothing else (prereg section 4).  They must
additionally be read against limitation L5: five code paths still read a distractor as free space.

This experiment has NO decision authority.  It cannot revise the P2 STRICT FAIL or the D1 FAIL, it
cannot unlock P3, and it does not retroactively change the v7 offline gate's 8/8 PASS -- that gate
is valid in the condition it measured, and this one measures outside it (prereg section 6).

Usage:
  python tools/run_navrl_distractor_envelope.py preflight
  python tools/run_navrl_distractor_envelope.py evaluate [default_n0|...|v7_n5]
  python tools/run_navrl_distractor_envelope.py finalize
  python tools/run_navrl_distractor_envelope.py verify
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


# ----------------------------------------------------------------------------------------------
# Preregistered contract constants.  Nothing below this block may recompute, relax or re-derive a
# value that appears here.  Declared ABOVE any measurement so that no number produced by a run can
# reach back and change a threshold.
# ----------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "aerial_gym/rl_training/rl_games"
EVALUATOR = RL_ROOT / "eval_navrl_v2_density_sweep.sh"
IMPORT_ORIGIN_GUARD = RL_ROOT / "navrl_import_origin.py"
DETECTOR_SOURCE = ROOT / "aerial_gym/task/navrl_task/navrl_detector.py"
PERCEPTION_SOURCE = ROOT / "aerial_gym/task/navrl_task/navrl_perception.py"
TASK_SOURCE = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"
TASK_CONFIG_SOURCE = ROOT / "aerial_gym/config/task_config/navrl_task_config.py"
ENV_OBJECT_CONFIG_SOURCE = ROOT / "aerial_gym/config/asset_config/env_object_config.py"
DISTRACTOR_TESTS = ROOT / "tests/test_navrl_distractors.py"
CANONICAL_PYTHON = Path("/home/fair/miniconda3/envs/aerialgym/bin/python")

SEED = 479                      # prereg section 3; exhaustive-search usage count 0
BARS = 70
EPISODES = 2049

CHECKPOINT_SHA = "197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e"
CHECKPOINT_REL = (
    "aerial_gym/rl_training/rl_games/runs/"
    "ppo_260813_1636_navrl_v2-ref5in-d1-q3-adapt-s197/nn/"
    "last_gen_ppo_ep_1900_rew_182.11377.pth"
)

# ---- the 2 x 4 factorial (prereg section 3, as amended by section 3-b) ------------------------
#
# Prereg section 3 names BOTH detectors -- "현행 기본값(AppearanceTargetSegmenter)과 학습형 v7
# 둘 다" -- so the detector is a second FACTOR, not a held-fixed condition.  Testing only the
# default would measure a five-parameter colour rule failing at colour discrimination, which is
# very nearly a tautology; the open question is whether the LEARNED detector escaped the colour
# shortcut, and v7 is the artifact carrying the 0.99766 frame-precision claim that section 1 puts
# under examination.
#
# A detector here is an ARTIFACT PLUS ITS OPERATING POINT, not just a file.  v7's 0.99766 was
# measured at its validation-selected threshold 0.700 (tools/train_navrl_target_detector_v2.py
# scores the held-out test set at ``selected_threshold``), so evaluating v7 at the default 0.55
# would measure a different operating point than the one the claim belongs to and the comparison
# would not be with the published number.  The two variables therefore move together, as one
# factor level, and are asserted to move together.
#
# (detector name, artifact path relative to the repository root, artifact sha256, threshold)
DETECTORS = (
    ("default", "", "", 0.55),
    (
        "v7",
        "artifacts/navrl_target_detector_v7_confirmatory.pth",
        "85c7974bcd85c627170c5bd63030144d1c5dc2a11e5d64829cad38f615c5d5d7",
        0.70,
    ),
)
DEFAULT_DETECTOR = "default"    # the bootstrap AppearanceTargetSegmenter: 3R - 2G - 2B - 0.9
V7_DETECTOR = "v7"              # spatial_cnn_wide+focal_dice, offline gate v7 confirmatory
# The v7 gate that produced the claim under examination.  Recorded so the summary can name what it
# is measuring outside of, and checked so the pinned artifact digest cannot drift away from the
# gate that scored it.
V7_GATE_SUMMARY_REL = "results/navrl_detector_offline_gate_v7_confirmatory/summary.json"
V7_GATE_FRAME_PRECISION = 0.99766

DISTRACTOR_COUNTS = (0, 1, 3, 5)
# (cell directory name, detector name, distractor count).  Derived from the two factors rather
# than written out, so a missing or duplicated combination is impossible rather than merely
# unlikely; the shape is asserted immediately below.
CELLS = tuple(
    ("%s_n%d" % (detector, count), detector, count)
    for detector, _, _, _ in DETECTORS
    for count in DISTRACTOR_COUNTS
)
if len(CELLS) != len(DETECTORS) * len(DISTRACTOR_COUNTS) or len(
    set((detector, count) for _, detector, count in CELLS)
) != len(CELLS):
    raise RuntimeError("the distractor envelope cell grid is not a complete 2x4 factorial")

MANIPULATED_VARIABLE = "NAVRL_DISTRACTOR_COUNT"
# The detector factor is selected by these three variables and by nothing else.  They are named as
# a SET because they are one factor level: the artifact, its operating point, and the narrow
# provenance override that the operating point makes necessary.
DETECTOR_SELECTION_VARIABLES = (
    "NAVRL_DETECTOR_CHECKPOINT",
    "NAVRL_DETECTOR_THRESHOLD",
    "NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH",
)
# prereg section 3: the N=0 regression cell.  Only the DEFAULT detector has a lineage to reproduce
# -- see verify_gate0_lineage_regression and the v7 note beside it.
REGRESSION_CELL = "default_n0"
# prereg section 5: Gate F is evaluated on the N=5 cell OF EACH DETECTOR, producing two verdicts.
VERDICT_DISTRACTOR_COUNT = 5


def verdict_cell(detector: str) -> str:
    return "%s_n%d" % (detector, VERDICT_DISTRACTOR_COUNT)


def zero_cell(detector: str) -> str:
    return "%s_n0" % detector

# Prereg section 4.  Frozen before any measurement: 0.15 m target radius plus roughly twice the
# detector's measured 0.178 m range MAE.  The runtime owns the same number
# (navrl_task.DISTRACTOR_LOCK_RADIUS_M) and every cell's exported block is checked against this
# one, so the radius a number was produced under can never differ from the radius it is read under.
CLASSIFICATION_RADIUS_M = 0.5

# Prereg section 5, Gate F.  TWO thresholds, compared by name, on the N=5 cell's FTLR expressed as
# a FRACTION.  0.50 is "more than half of all frames call something that is not the target the
# target"; 0.05 is the same order of magnitude as the complement of the 0.98 frame precision the v7
# gate cleared.  Both fixed before any measurement.
FTLR_SHORTCUT_THRESHOLD = 0.50
FTLR_ROBUST_THRESHOLD = 0.05

# Prereg section 5, applied per detector.  v7 runs at ITS OWN validation-selected threshold, which
# the frozen policy was not trained at, so the evaluator's v2 provenance gate necessarily objects
# to exactly one field.  That objection is not a defect -- it is the definition of evaluating a
# detector the policy never saw -- and it is resolved with the repository's NARROW override
# (run_navrl_ref5in_sensor_fidelity.py:94-104), never a blanket NAVRL_V2_FORCE.  The narrow form
# is STRICTER: a blanket force masks every other mismatch as well, whereas this proves at run time
# that the mismatch set is exactly this one field before any override is applied.  Two mismatch
# lines, or a different field, stop the run.
NARROW_OVERRIDE_DETECTOR = "v7"
NARROW_OVERRIDE_VARIABLE = "NAVRL_V2_ALLOW_DETECTOR_THRESHOLD_MISMATCH"
# The exact string the evaluator prints (eval_navrl_v2_density_sweep.sh:827,
# f"{key}: checkpoint={got} expected={expected}" with got=0.55 from the checkpoint's env_state and
# expected=float(NAVRL_DETECTOR_THRESHOLD)).
EXPECTED_THRESHOLD_MISMATCH = "cfg_detector_threshold: checkpoint=0.55 expected=0.7"
# The evaluator prints the field it LET THROUGH with this prefix, in the same shape as a refusal
# line.  The two are opposite facts, so they are parsed apart (see mismatch_lines below).
ALLOWED_MISMATCH_PREFIX = "[eval_v2] ALLOWED mismatch: "
NARROW_OVERRIDE_REASON = (
    "v7 is evaluated at its own validation-selected operating point 0.700, which is the "
    "threshold its 0.99766 frame precision was measured at; the frozen policy was trained at "
    "0.55, so the evaluator's v2 provenance gate pins cfg_detector_threshold to the training "
    "value and the single-field mismatch is verified at run time and only then overridden"
)

VERDICT_SHORTCUT = "COLOR_SHORTCUT_CONFIRMED"
VERDICT_ROBUST = "DETECTOR_ROBUST_TO_DISTRACTORS"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE_DISTRACTOR_ENVELOPE"
VERDICT_FAIL_CLOSED = "FAIL_CLOSED_IMPLEMENTATION"  # Gate 0 failure: claim nothing about the detector

# Held fixed in EVERY cell (prereg section 3).  This is the sensor condition of the frozen ref5in
# D1 lineage: the camera-resolution detection path at the threshold the checkpoint was trained
# with.  It is also, exactly, the `baseline` arm of the sensor fidelity experiment -- which is what
# makes that arm the lineage reference Gate 0.3 compares the N=0 cell against.
CAMERA_WIDTH = 160
CAMERA_HEIGHT = 90
# Prereg section 5, Gate 0.2 and section 7: distractors > 0 with a DECOUPLED detect resolution is
# refused by the runtime.  Rather than depend on that refusal, the detect resolution here IS the
# camera resolution, and equality is asserted in evaluation_env() for every cell including N=0.
DETECT_WIDTH = CAMERA_WIDTH
DETECT_HEIGHT = CAMERA_HEIGHT
DETECTOR_MIN_PIXELS = 2         # the value the frozen checkpoint was TRAINED at; no override needed
# The detection threshold is NOT here: it is half of the detector factor level (see DETECTORS).
# The default detector's 0.55 is the value the frozen policy was trained at.
DEFAULT_DETECTOR_THRESHOLD = 0.55
GOAL_DIST_MIN_M = 22.5          # fixed by the checkpoint contract, not by choice
GOAL_DIST_MAX_M = 28.0
ROBOT_NAME = "navrl_ref5in_quad"
ACTION_SELECTION = "deterministic"
REFLECTION_MODE = "original"
SPEED_GOVERNOR_MODE = "off"
# Prereg section 9 of the sensor-fidelity lineage, carried forward: the detector range is NEVER
# exported.  Moving it renormalises the actor's target token, which is a second axis.  Absence is
# the assertion -- the config default IS 20.0 m (navrl_task_config.py).
DETECTOR_MAX_RANGE_M = 20.0

# Appearance / injected-error knobs that must be zero in every cell.  Same reasoning as the sensor
# fidelity and detection range launchers: these are asserted on the environment that will actually
# be used, never assumed from a default.
ZERO_PERTURBATION_KEYS = (
    "NAVRL_APP_HUE_DEG",
    "NAVRL_APP_LIGHT_GAIN",
    "NAVRL_APP_ALBEDO_JITTER",
    "NAVRL_APP_TEXTURE_STD",
    "NAVRL_APP_MOTION_BLUR",
    "NAVRL_CAM_MOUNT_ROT_DEG",
    "NAVRL_CAM_MOUNT_TRANS_M",
    "NAVRL_CAM_FOV_SCALE_ERR",
    "NAVRL_DETECTION_LATENCY_S",
    "NAVRL_RANGE_ERROR_M",
)

# ---- Gate 0.3: what "the N=0 cell reproduces the current lineage" means, numerically -----------
#
# The regression cell CANNOT be compared by equality.  It runs a fresh seed (479) against a lineage
# cell that ran seed 421, so the two are independent 2,049-episode samples of the same distribution
# and would differ by sampling noise alone even if the distractor commit changed nothing.  A
# tolerance is therefore required, and it is pinned HERE, before any measurement, because prereg
# section 3 makes a mismatch VOID the entire run.
#
# Derivation.  For two independent proportions at n1 = n2 = 2,049 the standard error of their
# difference is sqrt(p1(1-p1)/n + p2(1-p2)/n), maximised at p = 0.5 where it is
# sqrt(0.25/2049 + 0.25/2049) = 0.015620, i.e. 1.5620 pp.  Three outcome rates are compared
# (capture, crash, timeout), so the z that holds the FAMILY-WISE false-VOID rate at 5% is the
# two-sided 0.05/3 quantile, z = 2.39398.  2.39398 x 1.5620 pp = 3.739 pp, rounded up to 3.75 pp.
#
# Why the worst-case p and the Bonferroni correction both point the same way here: a false VOID
# throws away four cells of GPU time on noise, while a real regression of the kind this gate exists
# to catch is not subtle.  The two defects already found on this axis moved capture by 27.45 pp
# (the flat spawn-clearance bug) and would break the bar slices outright (a wrong _bar_offset).  A
# 3.75 pp band cannot hide either, and it will not fire on sampling noise.
N0_REGRESSION_TOLERANCE_PP = 3.75
# The lineage reference: the sensor-fidelity `baseline` arm, which ran this EXACT sensor condition
# (camera 160x90, detect 160x90, min_pixels 2, detector range 20 m default, goal 22.5-28 m, 70
# bars, deterministic/original, governor off) on this EXACT checkpoint, before the distractor
# commit existed.  Pinned by digest so the constants below cannot silently describe a different
# file; re-derived from the artifact whenever the artifact is reachable.
LINEAGE_RESULT_REL = (
    "results/navrl_ref5in_sensor_fidelity_seed421/cells/baseline/70bars.json"
)
LINEAGE_RESULT_SHA = "57d0bbef819ad0f7566132c985f40d0a49673afb2f53ddba912da39a939188e3"
LINEAGE_SEED = 421
LINEAGE_EPISODES = 2049
LINEAGE_OUTCOME = {"captured": 1445, "crash": 406, "timeout": 198}
LINEAGE_NEVER_ACQUIRED = 387    # pooled over capture/crash/timeout; reported, NOT gated

PREREGISTRATION = "docs/prereg_2026-09-01_distractor_envelope.md"
PRODUCER = "tools/run_navrl_distractor_envelope.py"
SCOPE = "detector_distractor_envelope_frozen_ref5in_seed479"
OUTPUT = ROOT / "results" / "navrl_detector_distractor_envelope_seed479"   # prereg section 8
SOURCE_BUNDLE = OUTPUT / "source_bundle"
SUMMARY_JSON = OUTPUT / "summary.json"
SUMMARY_MD = OUTPUT / "summary.md"

# The primary measurand is READ from one runtime-emitted block, never recomputed here.  The runtime
# owns the classification because that is the only place the detector's reported measurement and
# the ground-truth centres coexist; this launcher checks the block's internal accounting and its
# radius, then divides the two numbers the preregistration names.
FTLR_SOURCE = (
    'result["distractor_lock"]: (distractor_lock + ghost_lock) / visible_frames, produced by '
    "navrl_task._record_distractor_lock_frame and guarded by _validate_distractor_lock_export"
)
FIRST_ACQUISITION_OUTCOMES = ("capture", "crash", "timeout")

# Gate 0 (implementation soundness, prereg section 5) -- checked BEFORE any verdict, and enforced
# here rather than assumed.  Each name maps to the key carrying this launcher's own proof it ran
# the check; a gate with no such evidence is a check nobody performed and fails closed.
GATE0_STATIC = {
    "G0.1_default_off_bit_identity": "default_off",
    "G0.2_decoupling_refusal": "decoupling_refusal",
}
GATE0_MEASURED = {
    "G0.3_zero_cell_reproduces_lineage": "lineage_regression",
}
PER_CELL_GATES = {
    "G1_checkpoint_identity": "checkpoint_identity",
    "G2_result_receipt_binding": "result_receipt_binding",
    "G3_manifest_provenance": "manifest_provenance",
    "G4_runtime_clean": "manifest_provenance",
    "G5_import_origin": "import_origin",
    "G6_episode_contract": "episode_contract",
    "G7_cell_condition_pinned": "cell_condition",
    "G8_camera_resolution_path": "camera_resolution_path",
    "G9_no_provenance_override": "no_override",
}
CROSS_CELL_GATES = {
    "G10_runtime_byte_map_identity": "runtime_map_identity",
    "G11_single_axis_manipulation": "single_axis",
    "G12_distractor_lock_accounting": "lock_accounting",
}

# The two CPU-only test classes that decide Gate 0.1 and Gate 0.2.  They are run as a subprocess by
# this launcher -- their result is this launcher's evidence, not somebody else's promise -- and the
# module they live in is pinned by digest so a passing tally cannot come from a rewritten file.
GATE0_TEST_CLASSES = {
    "default_off": (
        "test_navrl_distractors.DefaultOffIsUnchanged",
        "test_navrl_distractors.DefaultOffIsBitIdentical",
    ),
    "decoupling_refusal": ("test_navrl_distractors.FailsClosedOnDistractorsPlusDecoupling",),
}

ORIGIN_LOG_MARKER = "[origin] aerial_gym "
ORIGIN_LINE_HEAD = r"^\[origin\] aerial_gym "
ORIGIN_LINE_TAIL = r" sha256=(?P<sha256>[0-9a-f]{64}) \(enforced\)$"
ORIGIN_MANIFEST_ENTRY = "aerial_gym/__init__.py"

# Prereg section 6 (L1-L5) and section 3-c (L6) -- transcribed, not summarised: the limitations
# travel with the numbers.  L6 is the one with teeth in code as well as in prose; see
# NO_CROSS_DETECTOR_COMPARISON below.
LIMITATIONS = [
    "L1: distractor가 **정적**이다. 실기의 움직이는 오탐(새 등)은 범위 밖이다.",
    "L2: 색이 표적과 **동일**하다. 색 거리에 따른 성능 곡선은 범위 밖이다.",
    "L3: 카메라 해상도 160×90에서 잰다. 고해상도에서의 FTLR은 다를 수 있다.",
    "L4: 단일 정책·단일 seed·70막대 1조건.",
    "L5: distractor가 일부 코드 경로에서 자유 공간으로 남는다. 자산 배열이"
    " [target?][distractors...][bars...]이고 _bar_offset이 distractor를 건너뛰도록 넓혀졌으므로"
    " [_bar_offset : _bar_offset + n_bars_active]를 읽는 지점들은 distractor를 보지 못한다."
    " FTLR을 직접 오염시키는 두 곳만 고쳤다 — 드론 스폰 clearance(navrl_task.py:~1912)와 표적"
    " 경로 planner(~5858). 고치지 않은 다섯 곳: 정적 goal 배치(~4102), recovery clearance(~6058),"
    " bar-contact probe(~3081, ~3127). 결과적으로 distractor 충돌은 미귀속 contact로 기록되고"
    " 정적 goal이 distractor 안에 놓일 수 있다. **capture/crash/timeout 원값을 해석할 때 이"
    " 사실을 반드시 함께 읽어야 한다** — 그 값들은 §4에서 이미 판정 대상이 아니라 보조 보고다.",
    "L6: **검출기 간 FTLR을 빼서 비교하지 않는다(§3-c).** 동결 정책이 각 검출기의 출력을 보고"
    " 비행하므로 두 검출기에서 궤적이 달라지고, 따라서 프레임 분포 — 거리·베어링·가림, 특히"
    " **표적과 distractor가 동시에 보이는 빈도** — 가 달라진다. FTLR은 바로 그 분포 위에서"
    " 정의되므로 `FTLR_v7 − FTLR_default`는 검출기 강건성과 궤적 분포를 뒤섞은 값이고, 이 셀들로는"
    " 둘을 분리할 수 없다. 따라서 그 차이를 **계산하지도 게재하지도 않는다.** 반면 각 검출기의"
    " FTLR은 그 검출기가 실제로 만든 프레임 위에서 계산되므로 **한 검출기 안에서 N에 따른 비교는"
    " 유효하다** — 그것이 이 실험의 1차 지표다. capture/crash/timeout도 같다: v7 셀의 값은"
    " 계보와도, default 셀과도 비교하지 않는다.",
]
# L6, expressed where it can be checked rather than only where it can be read.  No cross-detector
# FTLR difference is computed anywhere in this launcher -- not in summary.json, not in summary.md,
# and not as an intermediate that a later edit could promote into an output.  The FTLR values live
# in a per-cell mapping and a per-detector verdict; nothing subtracts one detector from another.
NO_CROSS_DETECTOR_COMPARISON = (
    "FTLR is comparable WITHIN a detector across N and NOT between detectors: the two detectors "
    "produce different trajectories, hence different frame distributions, and FTLR is defined "
    "over that distribution (prereg section 3-c)"
)

# Host-absolute locations are not identity.  source + source_sha256 already pin the artifact.
_VERIFY_IGNORE_KEYS = frozenset({"artifact_path"})


def _verify_payload(value):
    """Drop host-absolute path fields before recorded-vs-recomputed summary compare."""
    if isinstance(value, dict):
        return {
            key: _verify_payload(item)
            for key, item in value.items()
            if key not in _VERIFY_IGNORE_KEYS
        }
    if isinstance(value, list):
        return [_verify_payload(item) for item in value]
    return value


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
    "manipulated_variable",
    "cells",
    "primary_metric",
    "design",
    "detectors",
    "verdict_cells",
    "comparability",
    "false_target_lock_rate_by_cell",
    "thresholds",
    "verdicts",
    "verdict_basis",
    "gate0",
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


P2 = load_module("distractor_envelope_p2", ROOT / "tools/attest_navrl_ref5in_p2.py")
# Reuse -- never re-implement -- the audited contract helpers and the dirty-runtime gate.
BASE = load_module(
    "distractor_envelope_base", ROOT / "tools/run_navrl_ref5in_cv_heading_diagnostic.py"
)

ContractError = BASE.ContractError
require = BASE.require
load_json = BASE.load_json
verify_runtime_clean_manifest = BASE.verify_runtime_clean_manifest
wilson = BASE.wilson


def origin_line_pattern(repository_root):
    """Matcher for the enforced [origin] line of the tree ``repository_root`` names."""
    return re.compile(
        ORIGIN_LINE_HEAD
        + re.escape(str(Path(repository_root) / ORIGIN_MANIFEST_ENTRY))
        + ORIGIN_LINE_TAIL
    )


def _resolve_shared_path(relative: str) -> Path:
    """Resolve a gitignored artifact, following the shared git dir when it is not in this tree.

    Git worktrees intentionally do not duplicate the gitignored multi-GB runs/ tree nor the
    results/ tree, so the frozen checkpoint and the lineage reference cell exist only in the
    primary worktree.  Identity is pinned by digest, which is verified before the file is used --
    the path is a lookup, never a trust boundary.
    """
    local = ROOT / relative
    if local.exists():
        return local.resolve()
    common = Path(
        subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--git-common-dir"], universal_newlines=True
        ).strip()
    ).resolve()
    return (common.parent / relative).resolve()


CHECKPOINT = _resolve_shared_path(CHECKPOINT_REL)


# ----------------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------------


def cell_factors(cell: str) -> tuple:
    """(detector name, distractor count) for one cell -- the two factor levels it sits at."""
    for name, detector, count in CELLS:
        if name == cell:
            return detector, count
    raise ContractError(
        f"unknown cell: {cell!r}; the preregistered cells are {[name for name, _, _ in CELLS]}"
    )


def cell_count(cell: str) -> int:
    return cell_factors(cell)[1]


def cell_detector(cell: str) -> str:
    return cell_factors(cell)[0]


def detector_spec(detector: str) -> tuple:
    """(artifact path relative to root, artifact sha256, threshold) for one detector level."""
    for name, relative, sha, threshold in DETECTORS:
        if name == detector:
            return relative, sha, threshold
    raise ContractError(
        f"unknown detector: {detector!r}; the preregistered detectors are "
        f"{[name for name, _, _, _ in DETECTORS]}"
    )


def detector_requires_narrow_override(detector: str) -> bool:
    return detector == NARROW_OVERRIDE_DETECTOR


def cells_for_detector(detector: str) -> list:
    return [name for name, factor, _ in CELLS if factor == detector]


def cells_for_count(count: int) -> list:
    return [name for name, _, factor in CELLS if factor == count]


def cell_dir(cell: str) -> Path:
    return OUTPUT / "cells" / cell


def cell_paths(cell: str) -> dict:
    directory = cell_dir(cell)
    return {
        "result": directory / ("%dbars.json" % BARS),
        "receipt": directory / ("%dbars.receipt.json" % BARS),
        "log": directory / ("%dbars.log" % BARS),
        "snapshot": directory / "checkpoint_snapshot.pth",
        "stdout_log": directory / "distractor_envelope_eval.log",
    }


def resolve_recorded_path(recorded, cell: str, label: str) -> Path:
    """Locate a receipt-recorded artifact without trusting the path the producer happened to write.

    The evaluator records ABSOLUTE paths, so a receipt produced in a git worktree names files that
    only exist there.  Two candidates are tried, in order: the copy that travels with the cell,
    then the absolute path the receipt recorded.  Neither is a trust boundary; identity is pinned
    by the digests the caller checks.  Nothing is resolved implicitly: if neither candidate exists
    the message names both and the check fails closed.
    """
    raw = str(recorded or "")
    require(bool(raw), f"{cell}: receipt records no {label}")
    recorded_path = Path(raw)
    local = cell_dir(cell) / recorded_path.name
    if local.exists():
        return local.resolve()
    absolute = recorded_path if recorded_path.is_absolute() else (cell_dir(cell) / recorded_path)
    if absolute.exists():
        return absolute.resolve()
    raise ContractError(
        f"{cell}: {label} not found; neither the cell-local copy {local} nor the recorded path "
        f"{absolute} exists"
    )


# ----------------------------------------------------------------------------------------------
# Evaluation environment
# ----------------------------------------------------------------------------------------------


def evaluation_env(cell: str, *, preflight: bool, force=None) -> dict:
    """P2's closed evaluation environment plus exactly the prereg section 3 additions.

    Three ordering facts are load-bearing and are asserted rather than commented.

    ``PYTHONPATH`` is DELETED by P2.canonical_env to keep its environment closed, but play_navrl.sh
    cds into aerial_gym/rl_training/rl_games where no aerial_gym/ package directory exists, so
    without re-injection the PEP 660 editable-install finder -- which hard-codes the PRIMARY
    worktree's absolute path -- wins, and the run would EXECUTE one tree while its receipt hashes
    another.  NAVRL_REQUIRE_SOURCE_ROOT is the independent fail-closed check on the same fact, and
    the [origin] log line (G5) is the third.

    ``NAVRL_DISTRACTOR_COUNT`` must land AFTER P2.canonical_env for the same reason
    NAVRL_DETECTOR_MIN_PIXELS does in the sibling launchers: canonical_env builds a CLOSED set by
    dropping every ambient NAVRL_* variable, so anything set before the call is simply erased and
    all four cells would run at zero distractors -- a four-cell experiment with one cell.  The
    assertion below proves the update landed.

    ``NAVRL_DETECT_WIDTH/HEIGHT`` are exported EQUAL to the camera resolution, and the equality is
    asserted here (prereg section 5, Gate 0.2).  navrl_detector.py refuses decoupling whenever
    distractors are present, but a launcher that depends on a downstream raise is a launcher that
    discovers its mistake after Isaac Gym has been allocated -- and, worse, the N=0 cell would sail
    straight past that raise and be measured on a path the other three cells cannot use.
    """
    detector, count = cell_factors(cell)
    relative, detector_sha, threshold = detector_spec(detector)
    env = P2.canonical_env(cell_dir(cell), preflight=preflight)
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "NAVRL_REQUIRE_SOURCE_ROOT": str(ROOT),
            "NAVRL_SEED": str(SEED),
            "NAVRL_V2_DENSITIES": str(BARS),
            "NAVRL_V2_RESULT_DIR": str(cell_dir(cell)),
            "NAVRL_V2_SHARED_SOURCE_BUNDLE": str(SOURCE_BUNDLE),
            # Fixed by the checkpoint contract, not by choice: the frozen ref5in D1 ep1900 policy
            # was trained at 22.5-28 m and the generic evaluator refuses any other goal band.
            "NAVRL_V2_GOAL_DIST_MIN": str(GOAL_DIST_MIN_M),
            "NAVRL_V2_GOAL_DIST_MAX": str(int(GOAL_DIST_MAX_M)),
            # Held fixed in every cell; exported so each value is a receipt fact, not a default.
            "NAVRL_CAMERA_WIDTH": str(CAMERA_WIDTH),
            "NAVRL_CAMERA_HEIGHT": str(CAMERA_HEIGHT),
            "NAVRL_DETECT_WIDTH": str(DETECT_WIDTH),
            "NAVRL_DETECT_HEIGHT": str(DETECT_HEIGHT),
            "NAVRL_DETECTOR_MIN_PIXELS": str(DETECTOR_MIN_PIXELS),
            # ---- factor 1: the distractor count ----
            MANIPULATED_VARIABLE: str(count),
            # ---- factor 2: the detector, as artifact PLUS operating point ----
            # The empty artifact string is not "unset": P2.canonical_env drops every ambient
            # NAVRL_*, and the evaluator treats an empty/absent NAVRL_DETECTOR_CHECKPOINT as "use
            # the built-in AppearanceTargetSegmenter" (eval_navrl_v2_density_sweep.sh:542-561,
            # navrl_task_config.py:261).  Writing it explicitly keeps the two detector levels
            # differing in the SAME key set, which is what makes the per-factor diff below a
            # symmetric difference over values rather than over key presence.
            "NAVRL_DETECTOR_CHECKPOINT": (
                str(_resolve_shared_path(relative)) if relative else ""
            ),
            "NAVRL_DETECTOR_THRESHOLD": f"{threshold:g}",
        }
    )
    require(
        env[MANIPULATED_VARIABLE] == str(count),
        f"{cell}: {MANIPULATED_VARIABLE} is {env.get(MANIPULATED_VARIABLE)!r}, not the cell's "
        f"{count}; the update must land AFTER P2.canonical_env, which drops every ambient NAVRL_* "
        "variable to keep its environment closed",
    )
    require(
        float(env["NAVRL_V2_GOAL_DIST_MIN"]) == GOAL_DIST_MIN_M
        and float(env["NAVRL_V2_GOAL_DIST_MAX"]) == GOAL_DIST_MAX_M,
        "exported goal band drifted from the pinned constants: "
        f"{env['NAVRL_V2_GOAL_DIST_MIN']}-{env['NAVRL_V2_GOAL_DIST_MAX']}",
    )
    # Prereg section 5, Gate 0.2, asserted rather than delegated to the runtime raise.
    require(
        int(env["NAVRL_DETECT_WIDTH"]) == int(env["NAVRL_CAMERA_WIDTH"])
        and int(env["NAVRL_DETECT_HEIGHT"]) == int(env["NAVRL_CAMERA_HEIGHT"]),
        f"{cell}: detect {env['NAVRL_DETECT_WIDTH']}x{env['NAVRL_DETECT_HEIGHT']} is DECOUPLED "
        f"from camera {env['NAVRL_CAMERA_WIDTH']}x{env['NAVRL_CAMERA_HEIGHT']}; with distractors "
        "in the scene that path substitutes the high-resolution TARGET mask for the RGB render, "
        "which stops being an identity the moment anything else is painted the target colour "
        "(navrl_detector.py fails closed on it). The distractor envelope is measured on the "
        "camera-resolution path in every cell, including N=0.",
    )
    require(
        int(env["NAVRL_DETECT_WIDTH"]) == DETECT_WIDTH
        and int(env["NAVRL_DETECT_HEIGHT"]) == DETECT_HEIGHT
        and int(env["NAVRL_CAMERA_WIDTH"]) == CAMERA_WIDTH
        and int(env["NAVRL_CAMERA_HEIGHT"]) == CAMERA_HEIGHT,
        f"{cell}: exported detect/camera resolution drifted from the pinned constants",
    )
    require(
        int(env["NAVRL_DETECTOR_MIN_PIXELS"]) == DETECTOR_MIN_PIXELS,
        f"{cell}: NAVRL_DETECTOR_MIN_PIXELS is {env['NAVRL_DETECTOR_MIN_PIXELS']!r}, not the "
        f"{DETECTOR_MIN_PIXELS} the frozen checkpoint was trained at",
    )
    require(
        "NAVRL_DETECTOR_MAX_RANGE" not in env,
        f"{cell}: NAVRL_DETECTOR_MAX_RANGE leaked into the cell environment; moving the detector "
        "range renormalises the actor's target token and would be a second axis",
    )
    for key in ZERO_PERTURBATION_KEYS:
        require(
            float(env[key]) == 0.0,
            f"{cell}: {key}={env[key]!r} is non-zero; the distractor axis is measured at zero "
            "appearance perturbation so the only thing that changed is what is in the scene",
        )
    require(env["NAVRL_PERCEPTION_PERTURB"] == "0", f"{cell}: perturbations must be off")
    require(env["NAVRL_SPEED_GOVERNOR"] == SPEED_GOVERNOR_MODE, f"{cell}: governor must be off")
    require(
        env["NAVRL_V2_ACTION_MODE"] == ACTION_SELECTION, f"{cell}: action must be deterministic"
    )
    require(
        env["NAVRL_EVAL_REFLECTION_MODE"] == REFLECTION_MODE,
        f"{cell}: reflection_mode must be original",
    )
    # ---- the detector factor, asserted on the environment that will actually be used ----------
    require(
        float(env["NAVRL_DETECTOR_THRESHOLD"]) == threshold,
        f"{cell}: NAVRL_DETECTOR_THRESHOLD is {env['NAVRL_DETECTOR_THRESHOLD']!r}, not this "
        f"detector's operating point {threshold}; the update must land AFTER P2.canonical_env, "
        "whose closed set already contains a value for it",
    )
    if relative:
        artifact = Path(env["NAVRL_DETECTOR_CHECKPOINT"])
        require(
            artifact.is_file(),
            f"{cell}: learned detector artifact missing: {artifact}",
        )
        require(
            P2.sha256_file(artifact) == detector_sha,
            f"{cell}: learned detector artifact identity mismatch; {artifact} is not the pinned "
            f"{detector_sha}",
        )
    else:
        require(
            env["NAVRL_DETECTOR_CHECKPOINT"] == "",
            f"{cell}: the default detector must carry no artifact, but "
            f"NAVRL_DETECTOR_CHECKPOINT={env['NAVRL_DETECTOR_CHECKPOINT']!r}",
        )

    # The narrow provenance override.  ``force=None`` means "whatever this cell is authorised to
    # use"; an explicit False is how verify_narrow_override() obtains the UNFORCED run whose
    # refusal is the proof.  An explicit True on a cell that is not a v7 cell is refused outright,
    # so a default-detector cell can never acquire an override through a caller's mistake.
    use_override = detector_requires_narrow_override(detector) if force is None else bool(force)
    require(
        not use_override or detector_requires_narrow_override(detector),
        f"{cell}: only {NARROW_OVERRIDE_DETECTOR} cells may carry the narrow threshold override",
    )
    env[NARROW_OVERRIDE_VARIABLE] = "1" if use_override else "0"
    require(
        (env[NARROW_OVERRIDE_VARIABLE] == "1") == use_override,
        f"{cell}: narrow override state does not match the request (use_override={use_override})",
    )
    # The BLANKET override stays unreachable in every cell, in both detectors.  The narrow one
    # above admits exactly cfg_detector_threshold and nothing else; NAVRL_V2_FORCE would admit
    # every mismatch in the contract at once.
    require(
        "NAVRL_V2_FORCE" not in env,
        f"{cell}: NAVRL_V2_FORCE leaked into the evaluation environment; the threshold mismatch "
        "the v7 level needs is resolved by the narrow "
        f"{NARROW_OVERRIDE_VARIABLE}, which admits one field, not by a blanket force",
    )
    return env


def evaluation_env_diff() -> dict:
    """The per-factor differences of the eight evaluation environments, computed not claimed.

    A 2 x 4 factorial has TWO single-axis claims, and lumping them into one cross-cell diff would
    let a stray variable hide: any key that happened to move with the detector would be excused as
    "part of the detector axis" when compared across everything at once.  So the two directions are
    asserted separately, and each one is a symmetric difference over every unordered PAIR inside a
    fixed level of the other factor:

      within a fixed DETECTOR      the four cells may differ in NAVRL_DISTRACTOR_COUNT alone
      within a fixed COUNT         the two cells may differ in the detector-selection set alone

    NAVRL_V2_RESULT_DIR is bookkeeping and differs everywhere; it is listed explicitly rather than
    filtered silently, so an unexplained extra difference must still fail.
    """
    environments = {
        cell: evaluation_env(cell, preflight=True) for cell, _, _ in CELLS
    }
    bookkeeping = {"NAVRL_V2_RESULT_DIR"}

    def symmetric_difference(name_a, name_b):
        env_a, env_b = environments[name_a], environments[name_b]
        return {key for key in set(env_a) | set(env_b) if env_a.get(key) != env_b.get(key)}

    # ---- direction 1: the distractor axis, inside each detector -------------------------------
    distractor_expected = {MANIPULATED_VARIABLE} | bookkeeping
    for detector, _, _, _ in DETECTORS:
        names = cells_for_detector(detector)
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                differing = symmetric_difference(name_a, name_b)
                require(
                    differing == distractor_expected,
                    f"within detector {detector}, cells {name_a} and {name_b} differ in "
                    f"{sorted(differing)}, not in {sorted(distractor_expected)}; the distractor "
                    "axis must be a single-axis manipulation (prereg section 3)",
                )

    # ---- direction 2: the detector axis, inside each distractor count -------------------------
    detector_expected = set(DETECTOR_SELECTION_VARIABLES) | bookkeeping
    for count in DISTRACTOR_COUNTS:
        names = cells_for_count(count)
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                differing = symmetric_difference(name_a, name_b)
                require(
                    differing == detector_expected,
                    f"at {MANIPULATED_VARIABLE}={count}, cells {name_a} and {name_b} differ in "
                    f"{sorted(differing)}, not in {sorted(detector_expected)}; the detector axis "
                    "is the artifact, its operating point and the narrow override that operating "
                    "point requires -- and nothing else (prereg section 3-b)",
                )

    # The narrow override must sit on the v7 cells ALONE, which is a claim about each cell rather
    # than a single global flag.
    for cell, detector, _ in CELLS:
        expected = "1" if detector_requires_narrow_override(detector) else "0"
        require(
            environments[cell][NARROW_OVERRIDE_VARIABLE] == expected,
            f"{cell}: narrow override is "
            f"{environments[cell][NARROW_OVERRIDE_VARIABLE]!r}, expected {expected!r}",
        )

    return {
        "distractor_axis": {
            key: {cell: environments[cell].get(key) for cell, _, _ in CELLS}
            for key in sorted({MANIPULATED_VARIABLE})
        },
        "detector_axis": {
            key: {
                detector: environments[zero_cell(detector)].get(key)
                for detector, _, _, _ in DETECTORS
            }
            for key in sorted(DETECTOR_SELECTION_VARIABLES)
        },
    }


# ----------------------------------------------------------------------------------------------
# Gate 0 -- implementation soundness, enforced here and BEFORE any verdict (prereg section 5)
# ----------------------------------------------------------------------------------------------


def runtime_dirty_paths() -> list:
    """Uncommitted paths inside the snapshotted runtime roots, as git reports them."""
    status = subprocess.check_output(
        [
            "git", "-C", str(ROOT), "status", "--porcelain=v1", "--untracked-files=all",
            "--", "aerial_gym", "resources/robots", "tools/create_navrl_source_bundle.py",
        ],
        universal_newlines=True,
    )
    return [line for line in status.splitlines() if line.strip()]


def run_gate0_test_classes(label: str) -> dict:
    """Run the CPU-only test classes that decide one Gate 0 item, and keep the proof.

    The classes are executed from ``tests/`` because that is the directory ``unittest discover``
    puts on sys.path, so the module names below are the same ones the full suite uses.  The module
    is pinned by digest: a tally of passing tests proves nothing if the file that produced it can
    be swapped for a shorter one.  Nothing here touches the GPU.
    """
    classes = GATE0_TEST_CLASSES[label]
    completed = subprocess.run(
        [str(CANONICAL_PYTHON), "-m", "unittest"] + list(classes),
        cwd=str(ROOT / "tests"),
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout or ""
    match = re.search(r"^Ran (\d+) tests? in ", output, flags=re.MULTILINE)
    tests_run = int(match.group(1)) if match is not None else 0
    findings = []
    if completed.returncode != 0:
        findings.append(
            "%s did not pass (returncode=%d): %s"
            % (list(classes), completed.returncode, output.strip().splitlines()[-20:])
        )
    elif tests_run <= 0:
        # A zero-test success is the failure mode that looks most like a pass: a renamed class
        # makes unittest report OK having executed nothing at all.
        findings.append(
            "%s returned 0 but ran no tests: %s"
            % (list(classes), output.strip().splitlines()[-5:])
        )
    return {
        "checked_by_launcher": True,
        "passed": not findings,
        "test_module": str(DISTRACTOR_TESTS.relative_to(ROOT)),
        "test_module_sha256": P2.sha256_file(DISTRACTOR_TESTS),
        "test_classes": list(classes),
        "tests_run": tests_run,
        "returncode": completed.returncode,
        "findings": findings,
    }


def verify_gate0_default_off() -> dict:
    """Gate 0.1: with NAVRL_DISTRACTOR_COUNT unset the runtime is the historical one.

    Two independent halves, because neither alone is sufficient.

    The BEHAVIOURAL half runs the test classes that build a world with and without the knob and
    compare the asset ordering, the placement stream, the spawn sampler's accepted candidates and
    the route planner's obstacle geometry.

    The STATIC half is about the code THIS experiment adds on top of that commit: the
    evaluation-only lock telemetry.  Its enable flag must be a derived conjunction of bulk
    evaluation AND a non-zero distractor count, its accumulators must be allocated only under that
    flag, and its recorder must be called only under that flag.  A telemetry block that ran at zero
    distractors would not corrupt the simulation, but it would make the regression cell measure
    something the lineage never measured, which is exactly what Gate 0.1 exists to prevent.
    """
    evidence = run_gate0_test_classes("default_off")
    findings = list(evidence["findings"])
    task = TASK_SOURCE.read_text(encoding="utf-8")
    for literal in (
        "self._distractor_metric_enabled = bool(\n"
        "            self._bulk_eval_mode and self._num_distractors > 0\n"
        "        )",
        "if self._distractor_metric_enabled:\n"
        "            self._record_distractor_lock_frame(diagnostics)",
        "if self._distractor_metric_enabled:\n"
        "            payload[\"distractor_lock\"] = self._distractor_lock_payload()",
    ):
        if literal not in task:
            findings.append(
                "the evaluation-only lock telemetry is not gated as required; missing exactly: "
                + repr(literal)
            )
    # The accumulators must not exist at all when the flag is False.  Both allocations sit inside
    # the `if self._distractor_metric_enabled:` block, so the only assignments to them in the whole
    # file are the two inside that block plus the in-place updates in the recorder.
    for name in ("self._dl_counts = ", "self._dl_sums = "):
        if task.count(name) != 1:
            findings.append(
                "%s is assigned %d time(s); the accumulators must be allocated exactly once, "
                "inside the enable guard" % (name.strip(), task.count(name))
            )
    # The task must NOT read the knob itself.  ``_num_distractors`` comes from the env config the
    # world was actually built from, so the telemetry and the scene cannot disagree: an env config
    # that has never heard of distractors reports 0 no matter what the environment says.  Prose
    # mentions of the variable name are fine; a read of it is not.
    for read in (
        'os.environ.get("NAVRL_DISTRACTOR_COUNT"',
        'os.environ["NAVRL_DISTRACTOR_COUNT"]',
    ):
        if read in task:
            findings.append(
                "navrl_task.py reads the knob directly (%s); the count must come from the env "
                "config the world was actually BUILT from, or the telemetry can disagree with "
                "the scene" % read
            )
    if 'getattr(getattr(self.sim_env.cfg, "env_config", None), "num_distractors", 0)' not in task:
        findings.append(
            "navrl_task.py no longer derives the distractor count from the built env config"
        )
    evidence.update(
        {
            "passed": not findings,
            "findings": findings,
            "telemetry_enable_expression": "self._bulk_eval_mode and self._num_distractors > 0",
            "accumulators_allocated_under_guard": True,
            "recorder_called_under_guard": True,
            "export_block_under_guard": True,
        }
    )
    return evidence


def verify_gate0_decoupling_refusal() -> dict:
    """Gate 0.2: distractors > 0 combined with a decoupled detect resolution is REFUSED.

    Behavioural half: the test class that constructs the refusal and proves it fires, that it does
    NOT fire on decoupling alone, that distractors alone never reach it, and that the refusal
    precedes the appearance-perturbation offenders (order matters -- a decoupling that is invalid
    for two independent reasons must name the one that is structural).

    Static half: the raise itself, and the fact that the detector knows its own distractor count
    before the guard runs.  Plus the positive form of the same rule -- this launcher's own
    environments must be on the camera-resolution path -- which is asserted per cell in
    evaluation_env() and re-asserted here across all four at once, so the gate does not depend on
    anyone having called that function.
    """
    evidence = run_gate0_test_classes("decoupling_refusal")
    findings = list(evidence["findings"])
    detector = DETECTOR_SOURCE.read_text(encoding="utf-8")
    for literal in (
        "if self.num_distractors > 0:",
        "NavRL detect-resolution decoupling (%dx%d detect vs %dx%d camera) is NOT ",
        "appearance distractor(s) in the scene",
        "self.num_distractors = _distractor_count()",
    ):
        if literal not in detector:
            findings.append("the fail-closed decoupling refusal is missing: " + repr(literal))
    config = TASK_CONFIG_SOURCE.read_text(encoding="utf-8")
    for literal in (
        'detect_width = _env_int("NAVRL_DETECT_WIDTH", camera_width)',
        'detect_height = _env_int("NAVRL_DETECT_HEIGHT", camera_height)',
    ):
        if literal not in config:
            findings.append("detect-resolution knob missing: " + literal)
    decoupled = []
    for cell, _, _ in CELLS:
        env = evaluation_env(cell, preflight=True)
        if (
            int(env["NAVRL_DETECT_WIDTH"]) != int(env["NAVRL_CAMERA_WIDTH"])
            or int(env["NAVRL_DETECT_HEIGHT"]) != int(env["NAVRL_CAMERA_HEIGHT"])
        ):
            decoupled.append(cell)
    if decoupled:
        findings.append(
            "cells %s would run a decoupled detect resolution; every cell must be on the "
            "camera-resolution path" % decoupled
        )
    evidence.update(
        {
            "passed": not findings,
            "findings": findings,
            "refusal_present_in_detector": not findings,
            "cells_on_camera_resolution_path": [cell for cell, _, _ in CELLS],
            "detect_resolution": [DETECT_WIDTH, DETECT_HEIGHT],
            "camera_resolution": [CAMERA_WIDTH, CAMERA_HEIGHT],
        }
    )
    return evidence


def lineage_reference() -> dict:
    """The pre-distractor lineage numbers Gate 0.3 compares the N=0 cell against.

    Constants first, artifact second, and they must AGREE.  The constants make the gate executable
    in a worktree whose gitignored results/ tree was never checked out; the artifact makes the
    constants unable to drift, because whenever it is reachable every number is re-derived from it
    and any disagreement stops the run.
    """
    reference = {
        "seed": LINEAGE_SEED,
        "episodes": LINEAGE_EPISODES,
        "outcome": dict(LINEAGE_OUTCOME),
        "never_acquired": LINEAGE_NEVER_ACQUIRED,
        "source": LINEAGE_RESULT_REL,
        "source_sha256": LINEAGE_RESULT_SHA,
        "artifact_reachable": False,
    }
    path = _resolve_shared_path(LINEAGE_RESULT_REL)
    if not path.is_file():
        reference["note"] = (
            "the lineage artifact is not present in this checkout (results/ is gitignored and is "
            "not duplicated into a worktree); the pinned constants are used and the digest is "
            "recorded so the comparison can be re-derived wherever the artifact lives"
        )
        return reference
    require(
        P2.sha256_file(path) == LINEAGE_RESULT_SHA,
        f"lineage reference {path} does not match the pinned digest {LINEAGE_RESULT_SHA}; the "
        "regression gate would be comparing against a different cell than the one it names",
    )
    result = load_json(path)
    condition = result.get("condition") or {}
    contract = result.get("v2_evaluation_contract") or {}
    # The reference is only a reference if it ran THIS condition.  Everything except the seed --
    # which is deliberately different, so the two cells are independent samples -- must match.
    for key, expected in (
        ("bars", BARS),
        ("robot_name", ROBOT_NAME),
        ("action_selection", ACTION_SELECTION),
        ("reflection_mode", REFLECTION_MODE),
        ("speed_governor_mode", SPEED_GOVERNOR_MODE),
        ("goal_dist_min_m", GOAL_DIST_MIN_M),
        ("goal_dist_max_m", GOAL_DIST_MAX_M),
    ):
        require(
            condition.get(key) == expected,
            f"lineage reference condition {key}={condition.get(key)!r}, not {expected!r}; it is "
            "not the same condition the N=0 cell runs",
        )
    require(
        int(contract.get("detector_min_pixels", -1)) == DETECTOR_MIN_PIXELS
        and float(contract.get("target_camera_max_range_m", -1.0)) == DETECTOR_MAX_RANGE_M,
        "lineage reference sensor contract differs from the cells' "
        f"({contract.get('detector_min_pixels')!r}, "
        f"{contract.get('target_camera_max_range_m')!r})",
    )
    require(
        int(condition.get("distractor_count", 0)) == 0,
        "the lineage reference already contains distractors; it cannot be the pre-distractor "
        "baseline",
    )
    require(
        result.get("checkpoint_sha256") == CHECKPOINT_SHA,
        "lineage reference evaluated a different checkpoint than this experiment does",
    )
    episodes = int(result["actual_episodes"])
    outcome = result["outcome"]
    derived = {
        "captured": int(outcome["captured"]),
        "crash": int(outcome["crash"]),
        "timeout": int(outcome["timeout"]),
    }
    rows = (result.get("target_motion") or {}).get("first_acquisition") or {}
    never = sum(int(rows[label]["never_acquired"]) for label in FIRST_ACQUISITION_OUTCOMES)
    require(
        episodes == LINEAGE_EPISODES
        and derived == LINEAGE_OUTCOME
        and never == LINEAGE_NEVER_ACQUIRED
        and int(condition.get("seed", -1)) == LINEAGE_SEED,
        "the pinned lineage constants disagree with the artifact they name: "
        f"artifact={derived} n={episodes} never={never} seed={condition.get('seed')} vs "
        f"pinned={LINEAGE_OUTCOME} n={LINEAGE_EPISODES} never={LINEAGE_NEVER_ACQUIRED} "
        f"seed={LINEAGE_SEED}",
    )
    reference["artifact_reachable"] = True
    # Identity is source (repo-relative) + source_sha256.  A host-absolute path would make
    # cross-machine verify fail after an otherwise identical tar extract.
    reference["artifact_path"] = LINEAGE_RESULT_REL
    return reference


def verify_gate0_lineage_regression(cell: dict) -> dict:
    """Gate 0.3: the N=0 cell reproduces the current lineage, within the pinned tolerance.

    Prereg section 3: if it does not, introducing distractors changed something unrelated and the
    WHOLE run is VOID.  The comparison is per-outcome and two-sided; never-acquired is computed and
    reported alongside but is NOT gated, because section 4 already lists it as a supporting number
    rather than a lineage statistic.
    """
    reference = lineage_reference()
    episodes = cell["actual_episodes"]
    outcome = cell["result"]["outcome"]
    deltas = {}
    exceeded = []
    for name, key in (("capture", "captured"), ("crash", "crash"), ("timeout", "timeout")):
        cell_pp = 100.0 * int(outcome[key]) / episodes
        lineage_pp = 100.0 * reference["outcome"][key] / reference["episodes"]
        delta = cell_pp - lineage_pp
        deltas[name] = {
            "cell_pp": cell_pp,
            "lineage_pp": lineage_pp,
            "delta_pp": delta,
            "within_tolerance": abs(delta) <= N0_REGRESSION_TOLERANCE_PP,
        }
        if abs(delta) > N0_REGRESSION_TOLERANCE_PP:
            exceeded.append(name)
    rows = (cell["result"].get("target_motion") or {}).get("first_acquisition") or {}
    never = sum(int(rows[label]["never_acquired"]) for label in FIRST_ACQUISITION_OUTCOMES)
    return {
        "checked_by_launcher": True,
        "passed": not exceeded,
        "cell": REGRESSION_CELL,
        "detector": DEFAULT_DETECTOR,
        # Only the DEFAULT detector has a lineage to reproduce.  Every prior 70-bar navigation cell
        # on this checkpoint ran the built-in segmenter; the one existing v7 run
        # (results/navrl_detector_domain_shift_v7/) is an OFFLINE frame-level screen at 205 bars
        # and threshold 0.55, so it is neither the same condition nor the same operating point and
        # cannot serve as an anchor.  v7_n0 is therefore DESCRIPTIVE: it is the within-detector
        # zero-distractor reference that v7's own FTLR trend is read against, and nothing is gated
        # on it.  Its outcome triple is NOT expected to match the lineage -- the frozen policy is
        # flying on a detector it never trained with, so its trajectories legitimately differ.
        # That same fact is limitation L6 (prereg section 3-c) and is the reason no FTLR or outcome
        # rate may be compared BETWEEN the two detectors: different trajectories mean a different
        # frame distribution, and FTLR is defined over that distribution.
        "v7_zero_cell": zero_cell(V7_DETECTOR),
        "v7_zero_cell_role": (
            "descriptive within-detector reference; no lineage anchor exists for v7 at this "
            "condition and operating point, so nothing is gated on it"
        ),
        "tolerance_pp": N0_REGRESSION_TOLERANCE_PP,
        "tolerance_basis": (
            "two-sided Bonferroni (3 outcome rates) 95% band on the difference of two independent "
            "proportions at n1 = n2 = 2049, evaluated at the worst-case p = 0.5: "
            "z(0.05/3) x sqrt(0.25/2049 + 0.25/2049) = 2.39398 x 1.5620 pp = 3.739 pp, rounded up"
        ),
        "lineage_reference": reference,
        "outcomes": deltas,
        "exceeded": sorted(exceeded),
        # Reported, not gated (prereg section 4 lists never-acquired as a supporting number).
        "never_acquired_pp": 100.0 * never / episodes,
        "lineage_never_acquired_pp": (
            100.0 * reference["never_acquired"] / reference["episodes"]
        ),
    }


def verify_detector_artifacts() -> dict:
    """Pin every detector factor level to bytes, before anything is run.

    The default level is pinned by ABSENCE -- no artifact, the built-in AppearanceTargetSegmenter --
    which is asserted rather than assumed so a stray artifact cannot turn the control arm into a
    second learned-detector arm.

    The v7 level is pinned by digest, and the digest is cross-checked against the offline gate that
    SCORED it.  That second check is the one that earns its place: ``artifacts/`` holds seven
    detector checkpoints and the claim under examination belongs to exactly one of them, so
    "the v7 file" must mean "the file whose sha256 the v7 gate summary recorded", not "the file
    whose name contains v7".
    """
    evidence = {}
    for detector, relative, sha, threshold in DETECTORS:
        if not relative:
            require(
                not sha,
                f"detector level {detector} names no artifact but pins a digest {sha!r}",
            )
            evidence[detector] = {
                "artifact": None,
                "artifact_sha256": None,
                "threshold": threshold,
                "segmenter": "AppearanceTargetSegmenter (built in; 3R - 2G - 2B - 0.9)",
            }
            continue
        artifact = _resolve_shared_path(relative)
        require(artifact.is_file(), f"detector artifact missing: {artifact}")
        actual = P2.sha256_file(artifact)
        require(
            actual == sha,
            f"detector level {detector}: {artifact} is {actual}, not the pinned {sha}",
        )
        evidence[detector] = {
            "artifact": relative,
            "artifact_sha256": sha,
            "threshold": threshold,
        }
    # The v7 gate summary: the artifact digest and the operating point must both come from it.
    gate_path = _resolve_shared_path(V7_GATE_SUMMARY_REL)
    _, v7_sha, v7_threshold = detector_spec(V7_DETECTOR)
    if gate_path.is_file():
        gate = load_json(gate_path)
        require(
            gate.get("artifact_sha256") == v7_sha,
            f"the v7 gate scored artifact {gate.get('artifact_sha256')!r}, but this launcher pins "
            f"{v7_sha!r}; they must be the same file or the 0.99766 claim belongs to a different "
            "detector than the one being measured",
        )
        require(
            abs(float(gate.get("selected_threshold", -1.0)) - v7_threshold) < 1e-9,
            f"the v7 gate selected threshold {gate.get('selected_threshold')!r}, but this launcher "
            f"evaluates v7 at {v7_threshold}; the operating point the claim was measured at is "
            "the operating point the claim must be re-examined at",
        )
        require(
            bool(gate.get("gate_passed")),
            "the v7 gate summary does not record a PASS; this experiment examines a claim that "
            "was made, so the claim has to exist",
        )
        precision = float(
            ((gate.get("test_metrics") or {}).get("frame_precision", -1.0))
        )
        evidence[V7_DETECTOR].update(
            {
                "gate_summary": V7_GATE_SUMMARY_REL,
                "gate_frame_precision": precision,
                "gate_selected_threshold": float(gate["selected_threshold"]),
                "gate_reachable": True,
            }
        )
    else:
        evidence[V7_DETECTOR].update(
            {
                "gate_summary": V7_GATE_SUMMARY_REL,
                "gate_frame_precision": V7_GATE_FRAME_PRECISION,
                "gate_selected_threshold": v7_threshold,
                "gate_reachable": False,
                "note": "the gate summary is not present in this checkout (results/ is "
                "gitignored); the artifact digest is still pinned and verified",
            }
        )
    return evidence


def verify_prerequisites() -> dict:
    """Everything cheap and CPU-only, before any GPU second is spent."""
    require(CHECKPOINT.is_file(), f"pinned frozen checkpoint missing: {CHECKPOINT}")
    require(
        P2.sha256_file(CHECKPOINT) == CHECKPOINT_SHA,
        "pinned frozen checkpoint identity mismatch",
    )
    require(EVALUATOR.is_file(), f"canonical evaluator missing: {EVALUATOR}")
    require(IMPORT_ORIGIN_GUARD.is_file(), f"import-origin guard missing: {IMPORT_ORIGIN_GUARD}")
    require(CANONICAL_PYTHON.is_file(), f"canonical Python missing: {CANONICAL_PYTHON}")
    require(DISTRACTOR_TESTS.is_file(), f"Gate 0 test module missing: {DISTRACTOR_TESTS}")
    require(
        (ROOT / PREREGISTRATION).is_file(),
        f"the preregistration this launcher implements is missing: {ROOT / PREREGISTRATION}",
    )
    verify_detector_artifacts()
    runner = (RL_ROOT / "runner.py").read_text(encoding="utf-8")
    require(
        "[origin] aerial_gym %s sha256=%s (enforced)" in runner,
        "runner.py no longer prints the enforced import-origin line that G5 verifies",
    )
    # The measurement this experiment exists to take must be present in the runtime that will
    # execute, together with its fail-closed export guard.  A missing guard would let a miscounted
    # classification ship as a plausible FTLR.
    task = TASK_SOURCE.read_text(encoding="utf-8")
    perception = PERCEPTION_SOURCE.read_text(encoding="utf-8")
    for source, literal in (
        (task, "def _record_distractor_lock_frame(self, diagnostics):"),
        (task, "def _distractor_lock_payload(self):"),
        (task, "def _validate_distractor_lock_export("),
        (task, "DISTRACTOR_LOCK_RADIUS_M = %s" % CLASSIFICATION_RADIUS_M),
        (perception, '"camera_measurement_world": meas_world,'),
        (perception, '"camera_confidence": confidence,'),
    ):
        require(literal in source, f"distractor lock telemetry missing: {literal!r}")
    # The classification radius is a preregistered threshold.  If the runtime's constant and this
    # launcher's constant ever disagree, every cell would be produced under one number and judged
    # under another.
    match = re.search(r"^DISTRACTOR_LOCK_RADIUS_M = ([0-9.]+)$", task, flags=re.MULTILINE)
    require(
        match is not None and float(match.group(1)) == CLASSIFICATION_RADIUS_M,
        "the runtime's classification radius is not the preregistered "
        f"{CLASSIFICATION_RADIUS_M} m",
    )
    env_objects = ENV_OBJECT_CONFIG_SOURCE.read_text(encoding="utf-8")
    require(
        'NAVRL_DISTRACTOR_COUNT = max(0, _env_int("NAVRL_DISTRACTOR_COUNT", 0))' in env_objects,
        "the distractor count knob is missing or no longer defaults to 0",
    )
    gate0 = {
        "default_off": verify_gate0_default_off(),
        "decoupling_refusal": verify_gate0_decoupling_refusal(),
    }
    return gate0


def gate0_static_passed(gate0: dict) -> bool:
    """True when the two pre-run Gate 0 items cleared (prereg section 5).

    Gate 0.3 is not here: it is a property of a cell that does not exist yet.
    """
    return all(bool(gate0[key].get("passed")) for key in GATE0_STATIC.values())


def gate0_failure_report(gate0: dict) -> str:
    return json.dumps(
        {
            key: gate0[key]["findings"]
            for key in GATE0_STATIC.values()
            if not gate0[key].get("passed")
        },
        ensure_ascii=False,
    )


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------


def tee_run(command: list, env: dict, log_path: Path) -> int:
    """Run a child, streaming its combined output to the console and to log_path.

    The log lands OUTSIDE the cell directory first: the evaluator refuses to start when its
    NAVRL_V2_RESULT_DIR already exists, so nothing may create the cell directory ahead of it.  The
    finished log is moved in afterwards.
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
    return returncode


def run_eval_preflight(cell: str, *, force=None):
    return subprocess.run(
        ["bash", str(EVALUATOR), str(CHECKPOINT), str(EPISODES)],
        cwd=str(ROOT),
        env=evaluation_env(cell, preflight=True, force=force),
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _provenance_lines(completed) -> list:
    return [
        line.strip()
        for line in (completed.stdout or "").splitlines()
        if "checkpoint=" in line and "expected=" in line
    ]


def mismatch_lines(completed) -> list:
    """The evaluator's REFUSAL lines -- fields it will not accept.

    Deliberately not "every line mentioning checkpoint= and expected=".  When the narrow override
    is active the evaluator ALSO prints an announcement of the field it let through
    (eval_navrl_v2_density_sweep.sh:822-828), and that line has the same shape as a refusal.
    Treating the announcement as a refusal made the overridden preflight look like a failure; the
    two are opposite facts and are read apart here.
    """
    return [
        line for line in _provenance_lines(completed)
        if not line.startswith(ALLOWED_MISMATCH_PREFIX)
    ]


def allowed_mismatch_lines(completed) -> list:
    """The fields the evaluator announced it let through under the narrow override.

    This is positive evidence, not noise: it names exactly which field was overridden, so the
    launcher can require that the override landed on the authorised field and on no other.
    """
    return [
        line[len(ALLOWED_MISMATCH_PREFIX):]
        for line in _provenance_lines(completed)
        if line.startswith(ALLOWED_MISMATCH_PREFIX)
    ]


def verify_provenance_override(cell: str) -> dict:
    """Establish AT RUN TIME what override this cell is entitled to, and never take more.

    Two different proofs, because the two detector levels are in two different positions.

    A DEFAULT-detector cell evaluates the frozen checkpoint at the sensor condition it was trained
    with, which is the configuration that needs no override at all.  Its unforced preflight is
    required to PASS, and a refusal stops the run and prints the mismatch lines rather than
    reaching for a force.

    A v7 cell evaluates a detector the policy never saw, at that detector's own operating point.
    The evaluator's provenance gate pins cfg_detector_threshold to the training value, so it MUST
    object -- and to exactly one field.  The unforced run is therefore required to FAIL with
    precisely ``EXPECTED_THRESHOLD_MISMATCH`` and nothing else; only then is the narrow override
    applied, and the overridden preflight is required to pass.  If the unforced run ever passes,
    the mismatch this level is defined by has silently disappeared and that is a failure too.
    """
    detector = cell_detector(cell)
    unforced = run_eval_preflight(cell, force=False)
    lines = mismatch_lines(unforced)
    if not detector_requires_narrow_override(detector):
        if unforced.returncode != 0:
            tail = (unforced.stdout or "").strip().splitlines()[-12:]
            raise ContractError(
                f"{cell}: the generic evaluator refused this cell (returncode="
                f"{unforced.returncode}). A {DEFAULT_DETECTOR}-detector cell is preregistered to "
                "need NO provenance override, so it stops here instead of forcing. mismatch "
                f"lines: {lines or 'none'} | tail: {tail}"
            )
        require(
            "[eval_v2] PREFLIGHT PASS (evaluation not started)" in (unforced.stdout or ""),
            f"{cell}: evaluator preflight returned 0 without the PREFLIGHT PASS marker",
        )
        require(
            not lines,
            f"{cell}: evaluator preflight passed but still printed mismatch lines: {lines}",
        )
        print(f"[distractor] cell {cell}: evaluator preflight PASS (no override)")
        return {
            "checked_by_launcher": True,
            "override": "none",
            "blanket_force_used": False,
            "unforced_preflight_returncode": 0,
            "mismatch_lines": [],
        }

    require(
        unforced.returncode == 2,
        f"{cell}: the generic evaluator ACCEPTED v7's operating point without an override "
        f"(returncode={unforced.returncode}); the narrow override would then be overriding "
        "nothing, and the single-field mismatch this level is defined by has silently disappeared",
    )
    require(
        lines == [EXPECTED_THRESHOLD_MISMATCH],
        f"{cell}: the unforced mismatch set is not exactly one field: {lines}; the narrow "
        f"override authorises {EXPECTED_THRESHOLD_MISMATCH!r} and nothing else",
    )
    overridden = run_eval_preflight(cell, force=True)
    require(
        overridden.returncode == 0,
        f"{cell}: preflight still failed under the narrow override (returncode="
        f"{overridden.returncode}); tail: "
        f"{(overridden.stdout or '').strip().splitlines()[-12:]}",
    )
    require(
        "[eval_v2] PREFLIGHT PASS (evaluation not started)" in (overridden.stdout or ""),
        f"{cell}: overridden preflight returned 0 without the PREFLIGHT PASS marker",
    )
    require(
        not mismatch_lines(overridden),
        f"{cell}: the narrow override left REFUSAL lines behind: {mismatch_lines(overridden)}; it "
        "authorises one field and must not have been reached for any other",
    )
    allowed = allowed_mismatch_lines(overridden)
    require(
        allowed == [EXPECTED_THRESHOLD_MISMATCH],
        f"{cell}: the evaluator announced it allowed {allowed}, not exactly "
        f"[{EXPECTED_THRESHOLD_MISMATCH!r}]; the override must land on the authorised field and "
        "on no other",
    )
    print(
        f"[distractor] cell {cell}: narrow override VERIFIED "
        f"(sole mismatch: {EXPECTED_THRESHOLD_MISMATCH}) then preflight PASS"
    )
    return {
        "checked_by_launcher": True,
        "override": "narrow",
        "variable": NARROW_OVERRIDE_VARIABLE,
        "blanket_force_used": False,
        "unforced_preflight_returncode": 2,
        "mismatch_lines": lines,
        "allowed_mismatch_lines": allowed,
        "authorised_mismatch": EXPECTED_THRESHOLD_MISMATCH,
        "reason": NARROW_OVERRIDE_REASON,
    }


def evaluate_cell(cell: str) -> None:
    require(
        not cell_dir(cell).exists(),
        f"refusing overwrite: {cell_dir(cell)} already exists",
    )
    # Gate 0.1 and 0.2 come BEFORE any cell is produced (prereg section 5): a cell measured under a
    # broken implementation is not evidence, and having produced it makes it tempting to keep.
    gate0 = verify_prerequisites()
    require(
        gate0_static_passed(gate0),
        "Gate 0 failed, so this is a FAIL_CLOSED_IMPLEMENTATION and no cell may be produced "
        "(prereg section 5). Nothing may be claimed about the detector. Findings: "
        + gate0_failure_report(gate0),
    )
    verify_provenance_override(cell)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    staged_log = OUTPUT / f"{cell}.eval.log.partial"
    detector, count = cell_factors(cell)
    _, _, threshold = detector_spec(detector)
    print(
        f"[distractor] EVALUATE {cell} | detector={detector} (thr {threshold:g}) | "
        f"{MANIPULATED_VARIABLE}={count} | seed {SEED} | {EPISODES} episodes | "
        f"camera-resolution path {CAMERA_WIDTH}x{CAMERA_HEIGHT}",
        flush=True,
    )
    returncode = tee_run(
        ["bash", str(EVALUATOR), str(CHECKPOINT), str(EPISODES)],
        evaluation_env(cell, preflight=False),
        staged_log,
    )
    require(returncode == 0, f"{cell}: evaluator exited with code {returncode}")
    require(cell_dir(cell).is_dir(), f"{cell}: evaluator produced no result directory")
    staged_log.replace(cell_paths(cell)["stdout_log"])


# ----------------------------------------------------------------------------------------------
# Cell verification
# ----------------------------------------------------------------------------------------------


def verify_import_origin(cell: str, mapping: dict, metadata: dict) -> dict:
    """G5: prove from the run log that the executing aerial_gym IS the tree the manifest hashed."""
    recorded_root = str(metadata.get("repository_root") or "")
    require(
        bool(recorded_root) and Path(recorded_root).is_absolute(),
        f"{cell}: G5 source manifest records no absolute repository_root: {recorded_root!r}",
    )
    repository_root = Path(recorded_root)
    expected_origin = repository_root / ORIGIN_MANIFEST_ENTRY
    pattern = origin_line_pattern(repository_root)

    matches = []
    foreign = []
    with cell_paths(cell)["log"].open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            text = line.rstrip("\r\n")
            match = pattern.match(text)
            if match is not None:
                matches.append(match.group("sha256"))
            elif text.startswith(ORIGIN_LOG_MARKER):
                foreign.append(text)
    require(
        bool(matches),
        f"{cell}: G5 run log contains no enforced [origin] line for {expected_origin}; "
        "the import-origin guard did not run",
    )
    require(
        not foreign,
        f"{cell}: G5 run log names an aerial_gym origin that is not the manifest's "
        f"repository_root {repository_root}: {foreign[:4]}",
    )
    require(
        len(set(matches)) == 1,
        f"{cell}: G5 conflicting [origin] digests in the run log: {set(matches)}",
    )
    origin_sha = matches[0]
    entry = mapping.get(ORIGIN_MANIFEST_ENTRY)
    require(
        entry is not None,
        f"{cell}: G5 manifest has no runtime_files entry for {ORIGIN_MANIFEST_ENTRY}",
    )
    require(
        entry[0] == origin_sha,
        f"{cell}: G5 executed aerial_gym/__init__.py sha256 {origin_sha} is not the manifest "
        f"digest {entry[0]}",
    )
    return {
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


def require_import_origin_evidence(cell: str, import_origin) -> None:
    require(
        isinstance(import_origin, dict)
        and import_origin.get("enforced") is True
        and import_origin.get("manifest_entry") == ORIGIN_MANIFEST_ENTRY
        and re.fullmatch(r"[0-9a-f]{64}", str(import_origin.get("origin_sha256", ""))) is not None
        and import_origin.get("origin_sha256") == import_origin.get("manifest_sha256")
        and int(import_origin.get("log_line_occurrences") or 0) >= 1,
        f"{cell}: G5_import_origin is owned by this launcher, but the launcher produced no proof "
        f"it ran the check (verify_import_origin evidence: {import_origin!r})",
    )


def verify_lock_block(cell: str, result: dict, count: int) -> dict:
    """The exported classification block, re-checked here rather than trusted.

    The runtime already fails closed on a miscount (_validate_distractor_lock_export).  This is the
    independent second reading, on the persisted document rather than on the live tensors: the
    partition, the radius the numbers were produced under, the distractor count the block claims,
    and a positive denominator -- without which FTLR is not a rate at all.

    The N=0 cell has no block, and that is the correct answer rather than a gap: the classifier is
    a structural no-op at zero distractors precisely so the regression cell measures the
    unmodified lineage (prereg section 5, Gate 0.1).
    """
    block = result.get("distractor_lock")
    if count == 0:
        require(
            block is None,
            f"{cell}: a distractor-lock block was exported at zero distractors; the classifier is "
            "supposed to be a structural no-op there, and the regression cell would no longer be "
            "measuring the unmodified lineage",
        )
        return {
            "checked_by_launcher": True,
            "present": False,
            "reason": "structural no-op at zero distractors (prereg section 5, Gate 0.1)",
            "false_target_lock_rate": None,
        }
    require(
        isinstance(block, dict),
        f"{cell}: {MANIPULATED_VARIABLE}={count} but the result carries no distractor_lock block; "
        "the telemetry did not run",
    )
    require(
        int(block.get("schema_version", -1)) == 1,
        f"{cell}: distractor_lock schema_version is {block.get('schema_version')!r}, not 1",
    )
    require(
        float(block.get("classification_radius_m", -1.0)) == CLASSIFICATION_RADIUS_M,
        f"{cell}: the block was produced at radius {block.get('classification_radius_m')!r}, not "
        f"the preregistered {CLASSIFICATION_RADIUS_M} m",
    )
    require(
        int(block.get("distractor_count", -1)) == count,
        f"{cell}: the block reports {block.get('distractor_count')!r} distractor(s), not the "
        f"cell's {count}",
    )
    require(
        block.get("visible_source") == "camera_visible",
        f"{cell}: FTLR is defined over the frames the CAMERA detector reported visible; this block "
        f"used {block.get('visible_source')!r}",
    )
    visible = int(block["visible_frames"])
    categories = {
        name: int(block[name]) for name in ("target_lock", "distractor_lock", "ghost_lock")
    }
    require(
        sum(categories.values()) == visible,
        f"{cell}: the three categories cover {sum(categories.values())} frame(s) but "
        f"{visible} were visible; the classification is not a partition",
    )
    require(
        visible > 0,
        f"{cell}: the detector never reported visible, so FTLR has no denominator. Before reading "
        "this as a detector result, check that the distractors rendered at all (prereg section 5)",
    )
    require(
        0 <= int(block["ambiguous_target_and_distractor"]) <= categories["target_lock"],
        f"{cell}: ambiguous frames ({block['ambiguous_target_and_distractor']}) are not a subset "
        f"of the target-lock frames ({categories['target_lock']})",
    )
    ftlr = (categories["distractor_lock"] + categories["ghost_lock"]) / visible
    require(
        abs(ftlr - float(block["false_target_lock_rate"])) < 1e-12,
        f"{cell}: the exported false_target_lock_rate {block['false_target_lock_rate']!r} is not "
        f"(distractor + ghost) / visible = {ftlr!r}",
    )
    return {
        "checked_by_launcher": True,
        "present": True,
        "distractor_count": count,
        "classification_radius_m": CLASSIFICATION_RADIUS_M,
        "frames_total": int(block["frames_total"]),
        "visible_frames": visible,
        "categories": categories,
        "ambiguous_target_and_distractor": int(block["ambiguous_target_and_distractor"]),
        "false_target_lock_rate": ftlr,
        "false_target_lock_wilson95": wilson(
            categories["distractor_lock"] + categories["ghost_lock"], visible
        ),
    }


def verify_cell(cell: str) -> dict:
    paths = cell_paths(cell)
    for key in ("result", "receipt", "log", "snapshot"):
        require(paths[key].is_file(), f"{cell}: missing artifact: {paths[key]}")
    result = load_json(paths["result"])
    receipt = load_json(paths["receipt"])
    detector, count = cell_factors(cell)
    _, detector_sha, threshold = detector_spec(detector)

    require(
        P2.sha256_file(paths["result"]) == receipt.get("result_sha256"),
        f"{cell}: result/receipt hash mismatch",
    )
    require(
        P2.sha256_file(paths["snapshot"]) == CHECKPOINT_SHA,
        f"{cell}: the evaluated checkpoint snapshot is not the pinned frozen checkpoint",
    )
    require(
        receipt.get("source_checkpoint_sha256") == CHECKPOINT_SHA,
        f"{cell}: receipt source checkpoint is not the pinned frozen checkpoint",
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
    }
    receipt_mismatch = {
        key: (receipt.get(key), value) for key, value in pinned.items() if receipt.get(key) != value
    }
    require(not receipt_mismatch, f"{cell}: receipt condition mismatch: {receipt_mismatch}")

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
    require(not condition_mismatch, f"{cell}: result condition mismatch: {condition_mismatch}")
    # The manipulated axis, attested by the process that BUILT the world.  This is the only field
    # that says what was actually in the scene, as opposed to what was asked for.
    require(
        int(condition.get("distractor_count", -1)) == count,
        f"{cell}: the scene was built with {condition.get('distractor_count')!r} distractor(s), "
        f"not the cell's {count}",
    )

    contract = result.get("v2_evaluation_contract") or {}
    require(
        int(contract.get("detector_min_pixels", -1)) == DETECTOR_MIN_PIXELS,
        f"{cell}: v2 evaluation contract records detector_min_pixels="
        f"{contract.get('detector_min_pixels')!r}, not the lineage {DETECTOR_MIN_PIXELS}",
    )
    require(
        float(contract.get("target_camera_max_range_m", -1.0)) == DETECTOR_MAX_RANGE_M,
        f"{cell}: v2 evaluation contract records target_camera_max_range_m="
        f"{contract.get('target_camera_max_range_m')!r}, not the untouched default "
        f"{DETECTOR_MAX_RANGE_M}",
    )
    # ---- the detector factor, attested by the run rather than by this launcher's intent -------
    # detector_checkpoint_sha256 is the evaluator's echo of NAVRL_EXPECTED_DETECTOR_SHA256, which
    # navrl_perception re-derives from the bytes it actually loads and raises on
    # (navrl_perception.py:937-950).  So a match here is a statement about loaded weights, not
    # about a requested path.
    require(
        abs(float(contract.get("detector_threshold", -1.0)) - threshold) < 1e-9,
        f"{cell}: v2 evaluation contract records detector_threshold="
        f"{contract.get('detector_threshold')!r}, not detector {detector}'s operating point "
        f"{threshold}",
    )
    require(
        str(contract.get("detector_checkpoint_sha256", "")) == detector_sha,
        f"{cell}: the evaluated detector is "
        f"{contract.get('detector_checkpoint_sha256')!r}, not detector {detector}'s pinned "
        f"{detector_sha!r}"
        + ("" if detector_sha else " (the default detector must load no artifact at all)"),
    )
    cell_condition = {
        "checked_by_launcher": True,
        "detector": detector,
        "detector_sha256_attested": str(contract.get("detector_checkpoint_sha256", "")),
        "detector_threshold_attested": float(contract["detector_threshold"]),
        "distractor_count_attested": count,
        "detector_min_pixels_attested": int(contract["detector_min_pixels"]),
        "target_camera_max_range_m_attested": float(contract["target_camera_max_range_m"]),
    }
    # G8: the evaluator does not record the detect resolution, so the camera-resolution path is
    # attested by this launcher's own environment plus the runtime refusal that would have stopped
    # any cell that was not on it.  Recorded as a claim WITH its evidence rather than as a bare
    # boolean.
    camera_resolution_path = {
        "checked_by_launcher": True,
        "detect_width": DETECT_WIDTH,
        "detect_height": DETECT_HEIGHT,
        "camera_width": CAMERA_WIDTH,
        "camera_height": CAMERA_HEIGHT,
        "decoupled": False,
        "evidence": (
            "evaluation_env() asserts NAVRL_DETECT_WIDTH/HEIGHT == NAVRL_CAMERA_WIDTH/HEIGHT for "
            "every cell before the run, and navrl_detector.py raises on decoupling whenever "
            "distractors are present, so a decoupled cell could not have produced this artifact"
        ),
    }

    # The evaluator drains whole 128-env batches, so a cell finishes at or just past the request.
    # Exact equality is WRONG here -- it has already broken one arm that landed on 2,050.
    actual = int(result.get("actual_episodes", -1))
    require(
        int(result.get("requested_episodes", -1)) == EPISODES and actual >= EPISODES,
        f"{cell}: episode contract mismatch: requested={result.get('requested_episodes')} "
        f"actual={actual}",
    )
    episode_contract = {
        "checked_by_launcher": True,
        "requested_episodes": int(result.get("requested_episodes", -1)),
        "actual_episodes": actual,
        "comparator": "requested == EPISODES and actual >= EPISODES",
    }

    require(receipt.get("schema_version") == 2, f"{cell}: receipt is not a schema_version 2 receipt")
    manifest = resolve_recorded_path(
        receipt.get("runtime_source_manifest"), cell, "runtime source manifest"
    )
    require(
        P2.sha256_file(manifest) == receipt.get("runtime_source_manifest_sha256"),
        f"{cell}: runtime source manifest bytes differ from the receipt digest: {manifest}",
    )
    python_environment = resolve_recorded_path(
        receipt.get("python_environment_manifest"), cell, "python environment manifest"
    )
    require(
        P2.sha256_file(python_environment) == receipt.get("python_environment_manifest_sha256"),
        f"{cell}: python environment manifest bytes differ from the receipt digest: "
        f"{python_environment}",
    )
    mapping, metadata = P2.manifest_map(manifest, 2, require_original=False)
    verify_runtime_clean_manifest(metadata, cell)
    import_origin = verify_import_origin(cell, mapping, metadata)
    manifest_provenance = {
        "checked_by_launcher": True,
        "receipt_schema_version": receipt.get("schema_version"),
        "manifest_schema_version": metadata.get("schema_version"),
        "manifest_sha256": receipt.get("runtime_source_manifest_sha256"),
        "python_environment_manifest_sha256": receipt.get("python_environment_manifest_sha256"),
        "runtime_file_count": metadata.get("runtime_file_count"),
        "runtime_clean_verified": True,
    }
    narrow = detector_requires_narrow_override(detector)
    no_override = {
        "checked_by_launcher": True,
        "blanket_force_used": False,
        "override": "narrow" if narrow else "none",
        "authorised_mismatch": EXPECTED_THRESHOLD_MISMATCH if narrow else None,
        "evidence": (
            "evaluation_env() refuses NAVRL_V2_FORCE in every cell; evaluate_cell() requires "
            "verify_provenance_override() to pass first, which for a v7 cell proves the UNFORCED "
            "run refuses with exactly one mismatch line before the narrow override is applied, "
            "and for a default cell requires the unforced run to pass outright"
        ),
    }
    lock = verify_lock_block(cell, result, count)

    return {
        "cell": cell,
        "detector": detector,
        "distractor_count": count,
        "result": result,
        "receipt": receipt,
        "condition": condition,
        "v2_evaluation_contract": contract,
        "runtime_map": mapping,
        "actual_episodes": actual,
        "lock": lock,
        "checkpoint_identity": checkpoint_identity,
        "result_receipt_binding": result_receipt_binding,
        "cell_condition": cell_condition,
        "camera_resolution_path": camera_resolution_path,
        "episode_contract": episode_contract,
        "manifest_provenance": manifest_provenance,
        "import_origin": import_origin,
        "no_override": no_override,
    }


# Held-fixed evaluation-contract keys that must be IDENTICAL in all EIGHT cells.  Two families are
# deliberately absent.  The distractor count lives in `condition`, not here.  The detector's own
# two fields -- `detector_threshold` and `detector_checkpoint_sha256` -- are the second factor, so
# they are held fixed WITHIN a detector and checked against that detector's level across it.
DETECTOR_OWNED_CONTRACT_KEYS = ("detector_threshold", "detector_checkpoint_sha256")
HELD_FIXED_CONTRACT_KEYS = (
    "detector_min_pixels",
    "target_camera_max_range_m",
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
    "arena_xy_m",
    "seed",
)
# `condition` keys that must be identical in all four cells.  `distractor_count` is the ONLY key in
# that dictionary allowed to differ, and that is asserted rather than arranged by omission.
HELD_FIXED_CONDITION_KEYS = (
    "robot_name",
    "robot_config_sha256",
    "robot_asset_sha256",
    "bars",
    "seed",
    "action_selection",
    "reflection_mode",
    "speed_governor_mode",
    "goal_dist_min_m",
    "goal_dist_max_m",
    "target_pattern",
    "target_speed_mode",
    "target_speed_min_mps",
    "target_speed_max_mps",
    "num_envs",
    "episode_len_steps",
)


def verify_all() -> dict:
    """Per-cell verification, then the per-factor cross-cell invariants, then Gate 0.3."""
    cells = {cell: verify_cell(cell) for cell, _, _ in CELLS}
    names = [cell for cell, _, _ in CELLS]
    reference = cells[names[0]]

    # Every cell must have executed the SAME bytes.  Compared against one reference rather than
    # pairwise because byte-map identity is transitive and the map is large.
    differing_maps = [
        name for name in names[1:] if cells[name]["runtime_map"] != reference["runtime_map"]
    ]
    require(
        not differing_maps,
        f"cells {differing_maps} were evaluated against a different runtime byte map than "
        f"{names[0]}; this is not a controlled factorial",
    )
    runtime_map_identity = {
        "checked_by_launcher": True,
        "identical": True,
        "runtime_file_count": reference["manifest_provenance"]["runtime_file_count"],
    }

    # The evaluation contract must have the SAME KEY SET everywhere -- a key that appears in one
    # cell and not another is a failure rather than a silent skip.
    for name in names[1:]:
        require(
            set(cells[name]["v2_evaluation_contract"])
            == set(reference["v2_evaluation_contract"]),
            f"cells {names[0]} and {name} have different evaluation-contract key sets: "
            f"{sorted(set(cells[name]['v2_evaluation_contract']) ^ set(reference['v2_evaluation_contract']))}",
        )

    def contract_differences(group):
        """Contract keys whose value is not identical across a group of cells."""
        return sorted(
            key
            for key in set(reference["v2_evaluation_contract"])
            if len(
                set(
                    json.dumps(cells[name]["v2_evaluation_contract"].get(key), sort_keys=True)
                    for name in group
                )
            )
            > 1
        )

    def condition_differences(group):
        # evaluation_nonce is a per-cell random token (secrets.token_hex, minted per cell by the eval
        # script) -- a within-cell integrity tie, NOT an experimental condition. It differs across
        # every cell by design on ANY machine, so exclude it from the single-axis comparison, exactly
        # as evaluation_env_diff() excludes its NAVRL_V2_RESULT_DIR bookkeeping field.
        bookkeeping = {"evaluation_nonce"}
        return sorted(
            key
            for key in set(reference["condition"]) - bookkeeping
            if len(
                set(
                    json.dumps(cells[name]["condition"].get(key), sort_keys=True)
                    for name in group
                )
            )
            > 1
        )

    for name in names[1:]:
        require(
            set(cells[name]["condition"]) == set(reference["condition"]),
            f"cells {names[0]} and {name} have different condition key sets: "
            f"{sorted(set(cells[name]['condition']) ^ set(reference['condition']))}",
        )

    # ---- direction 1: inside each detector, only the distractor count moved --------------------
    per_detector = {}
    for detector, _, _, threshold in DETECTORS:
        group = cells_for_detector(detector)
        contract_moved = contract_differences(group)
        require(
            not contract_moved,
            f"within detector {detector} the evaluation contracts differ in {contract_moved}; "
            "inside one detector level every contract field is a held-fixed condition",
        )
        condition_moved = condition_differences(group)
        require(
            condition_moved == ["distractor_count"],
            f"within detector {detector} the cells differ in condition {condition_moved}, but "
            f"distractor_count is the only authorised difference; {MANIPULATED_VARIABLE} is the "
            "whole within-detector manipulation (prereg section 3)",
        )
        counts = {name: cells[name]["condition"]["distractor_count"] for name in group}
        require(
            counts == {"%s_n%d" % (detector, c): c for c in DISTRACTOR_COUNTS},
            f"detector {detector} scenes were built with {counts}, not {DISTRACTOR_COUNTS}",
        )
        per_detector[detector] = {
            "cells": group,
            "condition_differences": condition_moved,
            "evaluation_contract_differences": contract_moved,
            "distractor_count_by_cell": counts,
            "detector_threshold": threshold,
        }

    # ---- direction 2: inside each distractor count, only the detector moved --------------------
    per_count = {}
    for count in DISTRACTOR_COUNTS:
        group = cells_for_count(count)
        contract_moved = contract_differences(group)
        require(
            contract_moved == sorted(DETECTOR_OWNED_CONTRACT_KEYS),
            f"at {MANIPULATED_VARIABLE}={count} the evaluation contracts differ in "
            f"{contract_moved}, not in {sorted(DETECTOR_OWNED_CONTRACT_KEYS)}; the detector level "
            "owns its artifact digest and its operating point, and nothing else",
        )
        condition_moved = condition_differences(group)
        require(
            not condition_moved,
            f"at {MANIPULATED_VARIABLE}={count} the cells differ in condition {condition_moved}; "
            "swapping the detector must not change how the world was built",
        )
        per_count[count] = {
            "cells": group,
            "condition_differences": condition_moved,
            "evaluation_contract_differences": contract_moved,
        }

    # Each detector's own two fields must equal that detector's declared level, in every one of
    # its cells -- the check that says WHICH detector ran, not merely that two of them differed.
    for cell, detector, _ in CELLS:
        _, detector_sha, threshold = detector_spec(detector)
        contract = cells[cell]["v2_evaluation_contract"]
        require(
            str(contract.get("detector_checkpoint_sha256", "")) == detector_sha
            and abs(float(contract.get("detector_threshold", -1.0)) - threshold) < 1e-9,
            f"{cell}: ran detector "
            f"({contract.get('detector_checkpoint_sha256')!r}, "
            f"{contract.get('detector_threshold')!r}), not level {detector} "
            f"({detector_sha!r}, {threshold})",
        )

    held_fixed = {}
    for key in HELD_FIXED_CONTRACT_KEYS:
        require(
            key in reference["v2_evaluation_contract"],
            f"held-fixed evaluation-contract key {key} is missing from the cells",
        )
        values = {name: cells[name]["v2_evaluation_contract"].get(key) for name in names}
        require(
            len(set(json.dumps(v, sort_keys=True) for v in values.values())) == 1,
            f"held-fixed condition {key} differs between cells: {values}",
        )
        held_fixed[key] = values[names[0]]
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
            f"{key} must be 0 in every cell; got {held_fixed[key]!r}",
        )
    for key in HELD_FIXED_CONDITION_KEYS:
        require(
            key in reference["condition"],
            f"held-fixed condition key {key} is missing from the cells' condition block",
        )
        values = {name: cells[name]["condition"].get(key) for name in names}
        require(
            len(set(json.dumps(v, sort_keys=True) for v in values.values())) == 1,
            f"held-fixed condition {key} differs between cells: {values}",
        )
        held_fixed.setdefault("condition_" + key, values[names[0]])

    single_axis = {
        "checked_by_launcher": True,
        "design": "2x4 factorial (detector x distractor count)",
        "manipulated_variables": {
            "within_detector": MANIPULATED_VARIABLE,
            "within_distractor_count": list(DETECTOR_SELECTION_VARIABLES),
        },
        "within_detector": per_detector,
        "within_distractor_count": {str(k): v for k, v in per_count.items()},
        "detector_owned_contract_keys": list(DETECTOR_OWNED_CONTRACT_KEYS),
    }

    # Every cell that carries a classification block must have a self-consistent one, and the only
    # cells entitled not to are the two N=0 cells -- one per detector.
    lock_accounting = {
        "checked_by_launcher": True,
        "blocks_present": sorted(name for name in names if cells[name]["lock"]["present"]),
        "blocks_absent": sorted(name for name in names if not cells[name]["lock"]["present"]),
        "classification_radius_m": CLASSIFICATION_RADIUS_M,
    }
    require(
        lock_accounting["blocks_absent"]
        == sorted(zero_cell(detector) for detector, _, _, _ in DETECTORS),
        f"cells {lock_accounting['blocks_absent']} carry no classification block; only the "
        "zero-distractor cell of each detector may be missing one",
    )

    lineage_regression = verify_gate0_lineage_regression(cells[REGRESSION_CELL])

    return {
        "cells": cells,
        "order": tuple(names),
        "held_fixed": held_fixed,
        "runtime_map_identity": runtime_map_identity,
        "single_axis": single_axis,
        "lock_accounting": lock_accounting,
        "lineage_regression": lineage_regression,
    }


# ----------------------------------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------------------------------


def cell_measurements(cell: dict) -> dict:
    """Prereg section 4 measurands for one cell: the primary, then the supporting numbers.

    never-acquired and the outcome triple are READ from the evaluator-emitted telemetry, never
    invented here.  The first-acquisition cohorts partition the cell exactly, which is what makes
    the pooled never-acquired a rate over the whole cell rather than over an unnamed subset -- and
    that partition is asserted, not assumed.
    """
    result = cell["result"]
    rows = (result.get("target_motion") or {}).get("first_acquisition") or {}
    require(
        set(rows) == set(FIRST_ACQUISITION_OUTCOMES),
        f"{cell['cell']}: first-acquisition outcome labels are {sorted(rows)}, expected "
        f"{list(FIRST_ACQUISITION_OUTCOMES)}",
    )
    never = sum(int(rows[label]["never_acquired"]) for label in FIRST_ACQUISITION_OUTCOMES)
    episodes = sum(int(rows[label]["episodes"]) for label in FIRST_ACQUISITION_OUTCOMES)
    acquired = sum(int(rows[label]["acquired"]) for label in FIRST_ACQUISITION_OUTCOMES)
    require(
        episodes == cell["actual_episodes"],
        f"{cell['cell']}: first-acquisition cohorts cover {episodes} episodes but the cell ran "
        f"{cell['actual_episodes']}",
    )
    require(
        never + acquired == episodes,
        f"{cell['cell']}: never_acquired + acquired != episodes "
        f"({never} + {acquired} != {episodes})",
    )

    outcome = result["outcome"]
    raw = {}
    for name, count_name in (("capture", "captured"), ("crash", "crash"), ("timeout", "timeout")):
        count = int(outcome[count_name])
        raw[name] = count
        raw[f"{name}_rate"] = count / episodes
        raw[f"{name}_wilson95"] = wilson(count, episodes)

    lock = cell["lock"]
    block = result.get("distractor_lock") or {}
    return {
        "episodes": episodes,
        # ---- primary (prereg section 4) ----
        "false_target_lock_rate": lock["false_target_lock_rate"],
        "false_target_lock_wilson95": lock.get("false_target_lock_wilson95"),
        "classification": (
            {
                "visible_frames": lock["visible_frames"],
                "frames_total": lock["frames_total"],
                "target_lock": lock["categories"]["target_lock"],
                "distractor_lock": lock["categories"]["distractor_lock"],
                "ghost_lock": lock["categories"]["ghost_lock"],
                "ambiguous_target_and_distractor": lock["ambiguous_target_and_distractor"],
                "target_lock_rate": block.get("target_lock_rate"),
                "distractor_lock_rate": block.get("distractor_lock_rate"),
                "ghost_lock_rate": block.get("ghost_lock_rate"),
            }
            if lock["present"]
            else None
        ),
        # ---- supporting (prereg section 4: reported, never a verdict input) ----
        "detector_pixel_count_mean": block.get("camera_pixel_count_mean"),
        "detector_confidence_mean": block.get("camera_confidence_mean"),
        "measurement_to_target_mean_m": block.get("measurement_to_target_mean_m"),
        "measurement_to_nearest_distractor_mean_m": block.get(
            "measurement_to_nearest_distractor_mean_m"
        ),
        "visible_frame_rate": block.get("visible_frame_rate"),
        "never_acquired": never,
        "never_acquired_rate_pp": 100.0 * never / episodes,
        "never_acquired_wilson95": wilson(never, episodes),
        "first_acquisition_by_outcome": {
            label: {
                "episodes": int(rows[label]["episodes"]),
                "never_acquired": int(rows[label]["never_acquired"]),
                "never_acquired_rate": rows[label]["never_acquired_rate"],
                "acquired": int(rows[label]["acquired"]),
                "first_visible_step_median": rows[label]["first_visible_step_median"],
            }
            for label in FIRST_ACQUISITION_OUTCOMES
        },
        "target_hidden_fraction": float(result["action"]["context"]["target_hidden"]["fraction"]),
        # Reported RAW and excluded from the verdict by construction -- see classify_verdict(),
        # which takes a single rate and cannot see these at all.  Read them against L5.
        "outcome_raw": raw,
    }


def classify_verdict(ftlr: float) -> str:
    """Prereg section 5, Gate F, applied to the N=5 cell's FTLR as a FRACTION.

    This function takes exactly one number.  capture, crash, timeout, never-acquired and the
    per-category counts are not parameters, are not read from any global, and therefore cannot
    enter the verdict -- which is the preregistration's requirement, expressed as a signature
    rather than as a promise.

    Direction note (prereg section 5): COLOR_SHORTCUT_CONFIRMED is the PREDICTED outcome and is not
    an experimental failure.  DETECTOR_ROBUST_TO_DISTRACTORS is the surprising one.
    """
    if ftlr >= FTLR_SHORTCUT_THRESHOLD:
        return VERDICT_SHORTCUT
    if ftlr <= FTLR_ROBUST_THRESHOLD:
        return VERDICT_ROBUST
    return VERDICT_INCONCLUSIVE


# ----------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------


def gate_table(verified: dict, gate0: dict) -> dict:
    """Build the gate table from EVIDENCE, not from a list of names.

    Every gate here is owned by this launcher, so each one is marked passed only when the step that
    owns it produced its evidence dictionary; a gate whose evidence is missing is a check nobody
    performed and fails closed rather than being tallied as a pass.
    """
    gates = {}
    for gate, evidence_key in sorted(GATE0_STATIC.items()):
        evidence = gate0.get(evidence_key)
        require(
            isinstance(evidence, dict)
            and evidence.get("checked_by_launcher") is True
            and isinstance(evidence.get("passed"), bool),
            f"quality gate {gate} is owned by this launcher, but verify_prerequisites() produced "
            f"no {evidence_key} evidence carrying a boolean verdict",
        )
        gates[gate] = {
            "owner": PRODUCER,
            "scope": "gate0_implementation",
            "passed": evidence["passed"],
        }
    for gate, evidence_key in sorted(GATE0_MEASURED.items()):
        evidence = verified.get(evidence_key)
        require(
            isinstance(evidence, dict)
            and evidence.get("checked_by_launcher") is True
            and isinstance(evidence.get("passed"), bool),
            f"quality gate {gate} is owned by this launcher, but verify_all() produced no "
            f"{evidence_key} evidence carrying a boolean verdict",
        )
        gates[gate] = {
            "owner": PRODUCER,
            "scope": "gate0_regression",
            "passed": evidence["passed"],
        }
    for gate, evidence_key in sorted(PER_CELL_GATES.items()):
        per_cell = {}
        for cell, _, _ in CELLS:
            evidence = verified["cells"][cell].get(evidence_key)
            require(
                isinstance(evidence, dict) and evidence.get("checked_by_launcher") is True,
                f"{cell}: quality gate {gate} is owned by this launcher, but verify_cell() "
                f"produced no {evidence_key} evidence that the launcher ran the check",
            )
            per_cell[cell] = True
        gates[gate] = {"owner": PRODUCER, "scope": "per_cell", "passed": True, "cells": per_cell}
    for gate, evidence_key in sorted(CROSS_CELL_GATES.items()):
        evidence = verified.get(evidence_key)
        require(
            isinstance(evidence, dict) and evidence.get("checked_by_launcher") is True,
            f"quality gate {gate} is owned by this launcher, but verify_all() produced no "
            f"{evidence_key} evidence that the launcher ran the check",
        )
        gates[gate] = {"owner": PRODUCER, "scope": "cross_cell", "passed": True}
    for cell, _, _ in CELLS:
        require_import_origin_evidence(cell, verified["cells"][cell].get("import_origin"))
    return gates


def gate_tally(payload: dict) -> tuple:
    """(evaluated, delegated, failed) for the summary line -- with every gate's ownership proved.

    ``delegated`` is empty here and is reported anyway: every gate in this experiment is decided by
    this launcher, including the two Gate 0 items whose behavioural half is a test module -- the
    launcher runs it, pins its digest, and keeps the returncode as its own evidence.  Printing an
    empty delegated list is how a reader can tell that "nothing was delegated" was checked rather
    than that delegation was never considered.
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
    delegated = sorted(name for name, gate in gates.items() if gate.get("owner") != PRODUCER)
    failing = {name for name, gate in gates.items() if gate.get("passed") is False}
    recorded_failed = set(payload.get("failed_gates") or [])
    require(
        failing == recorded_failed,
        f"failed_gates {sorted(recorded_failed)} disagrees with the per-gate verdicts "
        f"{sorted(failing)}",
    )
    return sorted(gates), delegated, sorted(recorded_failed)


def build_summary(verified: dict, gate0: dict) -> dict:
    gates = gate_table(verified, gate0)
    failed_gates = sorted(name for name, gate in gates.items() if gate.get("passed") is False)
    require(
        tuple(verified["order"]) == tuple(name for name, _, _ in CELLS),
        f"cell order drifted: {tuple(verified['order'])} is not "
        f"{tuple(name for name, _, _ in CELLS)}",
    )

    # Gate 0 first (prereg section 5).  If any owned gate failed, nothing may be claimed about the
    # detector and the measurements must be null.  That is a BICONDITIONAL and it is enforced
    # rather than described: a FAIL_CLOSED verdict carrying measurements would publish numbers the
    # preregistration says are not to be interpreted, and a null payload under any other verdict
    # would publish a verdict with nothing behind it.
    #
    # Gate F is evaluated PER DETECTOR, on that detector's own N=5 cell, so the run produces two
    # verdicts.  They are kept in a per-detector mapping with no pooled or "overall" verdict
    # anywhere, because there is no question this experiment asked that a single combined verdict
    # would answer: the default detector's result says nothing about v7 and vice versa.
    if failed_gates:
        verdicts = None
        published = None
        basis = None
        ftlr_by_cell = None
        measurements = {}
    else:
        measurements = {cell: cell_measurements(verified["cells"][cell]) for cell, _, _ in CELLS}
        ftlr_by_cell = {
            cell: measurements[cell]["false_target_lock_rate"] for cell, _, _ in CELLS
        }
        verdicts = {}
        basis = {}
        for detector, _, _, threshold in DETECTORS:
            cell = verdict_cell(detector)
            verdict_ftlr = ftlr_by_cell[cell]
            require(
                verdict_ftlr is not None,
                f"the verdict cell {cell} produced no FTLR; Gate F has nothing to judge for "
                f"detector {detector}",
            )
            verdicts[detector] = classify_verdict(verdict_ftlr)
            classification = measurements[cell]["classification"]
            basis[detector] = {
                "metric": "False Target Lock Rate over the detector-visible frames",
                "source_field": FTLR_SOURCE,
                "detector": detector,
                "detector_threshold": threshold,
                "verdict_cell": cell,
                "distractor_count": VERDICT_DISTRACTOR_COUNT,
                "classification_radius_m": CLASSIFICATION_RADIUS_M,
                "false_target_lock_rate": verdict_ftlr,
                "false_target_lock_rate_pp": 100.0 * verdict_ftlr,
                "visible_frames": classification["visible_frames"],
                "numerator": classification["distractor_lock"] + classification["ghost_lock"],
                "outcome_rates_excluded_from_verdict": True,
                "applies_only_to": detector,
                "expected_direction": (
                    f"{VERDICT_SHORTCUT} is the preregistered PREDICTION and is not a failure; "
                    f"{VERDICT_ROBUST} would be the surprising result and its first response is "
                    "to re-check that the distractors rendered (prereg section 5)"
                ),
            }
        published = measurements
    require(
        (verdicts is None) == (published is None),
        "fail-closed contract violated: verdicts and measurements must be null together",
    )

    cells_payload = None
    if published is not None:
        cells_payload = {}
        for cell, detector, count in CELLS:
            verified_cell = verified["cells"][cell]
            _, detector_sha, threshold = detector_spec(detector)
            cells_payload[cell] = {
                "condition": {
                    "detector": detector,
                    "detector_artifact_sha256": detector_sha,
                    "detector_threshold": threshold,
                    MANIPULATED_VARIABLE: count,
                    "distractor_count_attested": verified_cell["condition"]["distractor_count"],
                    "detect_width": DETECT_WIDTH,
                    "detect_height": DETECT_HEIGHT,
                    "camera_width": CAMERA_WIDTH,
                    "camera_height": CAMERA_HEIGHT,
                    "detector_min_pixels": DETECTOR_MIN_PIXELS,
                    "detector_max_range_m": DETECTOR_MAX_RANGE_M,
                    "seed": SEED,
                    "bars": BARS,
                    "requested_episodes": EPISODES,
                    "actual_episodes": verified_cell["actual_episodes"],
                    "num_envs": verified_cell["condition"].get("num_envs"),
                    "episode_len_steps": verified_cell["condition"].get("episode_len_steps"),
                },
                "measurements": published[cell],
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
            "requested_episodes_per_cell": EPISODES,
            "robot_name": ROBOT_NAME,
            "action_selection": ACTION_SELECTION,
            "reflection_mode": REFLECTION_MODE,
            "speed_governor_mode": SPEED_GOVERNOR_MODE,
            "goal_dist_min_m": GOAL_DIST_MIN_M,
            "goal_dist_max_m": GOAL_DIST_MAX_M,
            "camera_width": CAMERA_WIDTH,
            "camera_height": CAMERA_HEIGHT,
            "detect_width": DETECT_WIDTH,
            "detect_height": DETECT_HEIGHT,
            "detect_resolution_decoupled": False,
            "detector_min_pixels": DETECTOR_MIN_PIXELS,
            "detector_max_range_m": DETECTOR_MAX_RANGE_M,
            "classification_radius_m": CLASSIFICATION_RADIUS_M,
        },
        "manipulated_variable": MANIPULATED_VARIABLE,
        "detector_selection_variables": list(DETECTOR_SELECTION_VARIABLES),
        "provenance_override": {
            cell: {
                "blanket_force_used": False,
                "override": (
                    "narrow" if detector_requires_narrow_override(detector) else "none"
                ),
                "authorised_mismatch": (
                    EXPECTED_THRESHOLD_MISMATCH
                    if detector_requires_narrow_override(detector)
                    else None
                ),
                "reason": (
                    NARROW_OVERRIDE_REASON
                    if detector_requires_narrow_override(detector)
                    else "this cell evaluates the frozen checkpoint at the sensor condition it "
                    "was trained with, so the evaluator's v2 provenance gate passes unforced"
                ),
            }
            for cell, detector, _ in CELLS
        },
        "cells": cells_payload,
        "primary_metric": "false_target_lock_rate",
        "design": "2x4 factorial: detector x distractor count",
        "detectors": {
            name: {
                "artifact": relative or None,
                "artifact_sha256": sha or None,
                "threshold": threshold,
                "verdict_cell": verdict_cell(name),
                "zero_cell": zero_cell(name),
            }
            for name, relative, sha, threshold in DETECTORS
        },
        "verdict_cells": {name: verdict_cell(name) for name, _, _, _ in DETECTORS},
        # Prereg section 3-c / limitation L6.  Stated as a field, and true of this document: the
        # FTLR values below are a per-CELL mapping, and there is deliberately no
        # cross-detector difference key anywhere in this payload.
        "comparability": {
            "within_detector_across_distractor_count": True,
            "between_detectors": False,
            "reason": NO_CROSS_DETECTOR_COMPARISON,
            "cross_detector_difference_computed": False,
            "also_applies_to": ["capture", "crash", "timeout", "never_acquired"],
        },
        "false_target_lock_rate_by_cell": ftlr_by_cell,
        "thresholds": {
            "color_shortcut_confirmed_at_or_above": FTLR_SHORTCUT_THRESHOLD,
            "detector_robust_at_or_below": FTLR_ROBUST_THRESHOLD,
        },
        # PER DETECTOR.  There is deliberately no pooled or "overall" verdict key: one detector's
        # result is not evidence about the other, and a single combined field would invite exactly
        # that misreading.
        "verdicts": verdicts,
        "verdict_basis": basis,
        "detector_artifacts": verify_detector_artifacts(),
        "gate0": {
            "default_off_bit_identity": gate0["default_off"],
            "decoupling_refusal": gate0["decoupling_refusal"],
            "zero_cell_reproduces_lineage": verified["lineage_regression"],
        },
        "quality_gates": gates,
        "failed_gates": failed_gates,
        "held_fixed": verified["held_fixed"],
        "import_origin": {
            cell: verified["cells"][cell]["import_origin"] for cell, _, _ in CELLS
        },
        "limitations": list(LIMITATIONS),
        "sources": {
            cell: {
                "evaluation_result": str(cell_paths(cell)["result"].relative_to(ROOT)),
                "evaluation_receipt": str(cell_paths(cell)["receipt"].relative_to(ROOT)),
                "evaluation_log": str(cell_paths(cell)["log"].relative_to(ROOT)),
                "launcher_log": str(cell_paths(cell)["stdout_log"].relative_to(ROOT)),
            }
            for cell, _, _ in CELLS
        },
    }


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def _num(value, digits=3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _metres(value) -> str:
    return "—" if value is None else f"{value:.3f} m"


def write_summary(payload: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    evaluated, delegated, failed = gate_tally(payload)
    cells = payload.get("cells") or {}
    gate0 = payload["gate0"]
    default_off = gate0["default_off_bit_identity"]
    decoupling = gate0["decoupling_refusal"]
    regression = gate0["zero_cell_reproduces_lineage"]

    lines = [
        f"# distractor envelope — False Target Lock Rate (seed {SEED}, {BARS} bars, "
        f"셀당 {EPISODES} 에피소드)",
        "",
        "**2 × 4 요인설계: 검출기(기본 / v7) × distractor 수(0/1/3/5) = 8 셀.**",
        f"판정(§5 Gate F)은 **검출기마다 따로** 그 검출기의 N={VERDICT_DISTRACTOR_COUNT} 셀에서",
        "내린다. 통합 판정은 없다 — 한쪽 결과는 다른 쪽에 대한 근거가 아니다.",
        "",
    ]
    if cells:
        lines.extend([
            "| 검출기 | 동작점 | 판정 | N=5 FTLR | 가시 프레임 |",
            "|---|---:|---|---:|---:|",
        ])
        for detector, _, _, threshold in DETECTORS:
            b = payload["verdict_basis"][detector]
            lines.append(
                f"| `{detector}` | thr {threshold:g} | **`{payload['verdicts'][detector]}`** | "
                f"{_pct(b['false_target_lock_rate_pp'])} | {b['visible_frames']:,} |"
            )
        lines.extend([
            "",
            f"임계(양쪽 검출기 공통): ≥ {FTLR_SHORTCUT_THRESHOLD * 100:.2f}% → "
            f"`{VERDICT_SHORTCUT}`, ≤ {FTLR_ROBUST_THRESHOLD * 100:.2f}% → `{VERDICT_ROBUST}`, "
            f"그 외 `{VERDICT_INCONCLUSIVE}`. 분류 반경 {CLASSIFICATION_RADIUS_M} m.",
            "",
            "**두 판정은 서로 독립이다.** `default`는 5개 파라미터짜리 색 규칙이고 `v7`은 학습된",
            "spatial CNN이다. 한쪽이 무너졌다고 다른 쪽이 무너지는 것도, 한쪽이 견뎠다고 다른 쪽이",
            "견디는 것도 아니다. 각 판정은 자기 검출기에만 적용된다.",
            "",
            "> ⚠️ **위 두 FTLR을 빼지 마시오 (한계 L6, 사전등록 §3-c).** 동결 정책이 각 검출기의",
            "> 출력을 보고 비행하므로 두 검출기의 **궤적이 다르고**, 따라서 프레임 분포 —",
            "> 거리·베어링·가림, 특히 **표적과 distractor가 동시에 보이는 빈도** — 가 다르다.",
            "> FTLR은 바로 그 분포 위에서 정의되므로 두 값의 차이는 검출기 강건성과 궤적 분포를",
            "> 뒤섞은 값이며 이 셀들로는 둘을 분리할 수 없다. 그래서 이 문서에도 `summary.json`에도",
            "> **그 차이는 없다 — 계산하지 않았다.** 아래 셀별 표에서 유효한 비교는 **같은 검출기",
            "> 행 안에서 N에 따른 변화**뿐이다. capture/crash/timeout도 같다: v7 행은 계보와도,",
            "> `default` 행과도 비교하지 않는다.",
            "",
            "## 셀별 분류 (사전등록 §4)",
            "",
            "| 검출기 | N | FTLR | TARGET | DISTRACTOR | GHOST | 가시 프레임 | count 평균 | conf 평균 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for cell, detector, count in CELLS:
            m = cells[cell]["measurements"]
            classification = m["classification"]
            if classification is None:
                lines.append(f"| `{detector}` | {count} | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| `{detector}` | {count} | {_pct(m['false_target_lock_rate'] * 100)} | "
                f"{classification['target_lock']:,} | {classification['distractor_lock']:,} | "
                f"{classification['ghost_lock']:,} | {classification['visible_frames']:,} | "
                f"{_num(m['detector_pixel_count_mean'], 1)} | "
                f"{_num(m['detector_confidence_mean'])} |"
            )
        lines.extend([
            "",
            "N=0 행이 비어 있는 것은 결측이 아니다. 분류기는 distractor가 0일 때 **호출되지 않는다**",
            "— 회귀 셀이 손대지 않은 계보를 그대로 재는 것이 Gate 0.1의 요구사항이기 때문이다",
            "(사전등록 §5). 따라서 `count`/`confidence`의 N 의존성은 N=1·3·5 세 점으로 읽는다.",
            "",
            "## 보조 수치 (판정에 쓰지 않음, 사전등록 §4)",
            "",
            "| 검출기 | N | capture | crash | timeout | never-acq | target_hidden | 측정↔표적 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for cell, detector, count in CELLS:
            m = cells[cell]["measurements"]
            raw = m["outcome_raw"]
            lines.append(
                f"| `{detector}` | {count} | {_pct(raw['capture_rate'] * 100)} | "
                f"{_pct(raw['crash_rate'] * 100)} | {_pct(raw['timeout_rate'] * 100)} | "
                f"{_pct(m['never_acquired_rate_pp'])} | "
                f"{_pct(m['target_hidden_fraction'] * 100)} | "
                f"{_metres(m['measurement_to_target_mean_m'])} |"
            )
        lines.extend([
            "",
            "**capture/crash/timeout은 원값이며 판정에 쓰지 않는다.** 판정 함수",
            "`classify_verdict()`는 FTLR 하나만 인자로 받으므로 구조적으로 이 값들을 볼 수 없다.",
            "또한 이 세 값은 **한계 L5와 함께 읽어야 한다** — distractor를 자유 공간으로 읽는",
            "코드 경로가 다섯 곳 남아 있으므로 distractor 충돌이 미귀속 contact로 기록된다.",
            "v7 셀의 outcome은 계보와도 `default` 셀과도 비교할 수 없다 (한계 L6): 동결 정책이",
            "**학습한 적 없는 검출기**로 날고 있으므로 궤적 자체가 다르다.",
        ])
    else:
        lines.append(
            "게이트 0(구현 타당성)이 실패했으므로 측정값을 게재하지 않으며 검출기에 대한 어떤 "
            f"주장도 하지 않는다 (사전등록 §5). 실패 게이트: "
            f"{', '.join(payload.get('failed_gates') or []) or '—'}."
        )

    lines.extend([
        "",
        "## 판정 방향 — 이것은 예상된 결과다",
        "",
        f"`{VERDICT_SHORTCUT}`는 **사전등록이 예측한 결과이고 실험 실패가 아니다** (사전등록 §5).",
        "`_detect_rgbd`는 임계를 넘은 이미지 전체 양성 픽셀을 연결 성분 분석 없이 하나의",
        "무게중심으로 축약하므로, 표적과 distractor가 동시에 보이는 프레임에서는 구조적으로",
        "반드시 실패한다. 이 실험의 값어치는 검출기 개선이 아니라 **결함의 정량화**에 있다.",
        f"반대로 `{VERDICT_ROBUST}`가 나오면 그것이 놀라운 결과이며, 그 경우 축하하기 전에",
        "distractor가 실제로 렌더·검출됐는지를 먼저 재확인한다.",
        "",
        "## 게이트 0 — 구현 타당성 (판정보다 먼저)",
        "",
        "| 항목 | 결과 | 근거 |",
        "|---|---|---|",
        f"| 0.1 기본 off bit-identical | {'PASS' if default_off['passed'] else 'FAIL'} | "
        f"{default_off['tests_run']}개 테스트 "
        f"(`{'`, `'.join(default_off['test_classes'])}`) + 텔레메트리 게이트 정적 검사 |",
        f"| 0.2 decoupling 거부 | {'PASS' if decoupling['passed'] else 'FAIL'} | "
        f"{decoupling['tests_run']}개 테스트 + 전 셀 카메라 해상도 경로 확인 |",
        f"| 0.3 N=0 계보 재현 (`{regression['cell']}`) | "
        f"{'PASS' if regression['passed'] else 'FAIL'} | "
        f"허용오차 ±{regression['tolerance_pp']:.2f} pp, 초과: "
        f"{', '.join(regression['exceeded']) or '없음'} |",
        f"| (참고) `{regression['v7_zero_cell']}` | 게이트 아님 | "
        f"{regression['v7_zero_cell_role']} |",
        "",
        f"허용오차 근거: {regression['tolerance_basis']}",
        "",
        "| outcome | N=0 셀 | 계보(seed 421) | 차이 |",
        "|---|---:|---:|---:|",
    ])
    for name in ("capture", "crash", "timeout"):
        row = regression["outcomes"][name]
        lines.append(
            f"| {name} | {_pct(row['cell_pp'])} | {_pct(row['lineage_pp'])} | "
            f"{row['delta_pp']:+.2f} pp |"
        )
    lines.extend([
        "",
        f"계보 참조: `{regression['lineage_reference']['source']}` "
        f"(sha256 `{regression['lineage_reference']['source_sha256'][:16]}…`, "
        f"seed {regression['lineage_reference']['seed']}). 두 셀은 서로 다른 seed의 독립 표본이므로",
        "동치 비교가 아니라 위 허용오차 내의 일치로 판정한다.",
        "",
        f"**`{regression['v7_zero_cell']}`에는 계보 기준점이 없다.** 이 체크포인트의 기존 70막대",
        "항법 셀은 전부 내장 segmenter로 돌았고, 유일한 v7 실행",
        "(`results/navrl_detector_domain_shift_v7/`)은 205막대·thr 0.55의 **오프라인** 프레임",
        "수준 스크리닝이라 조건도 동작점도 다르다. 따라서 그 셀은 **기술적(descriptive)**이며",
        "v7 자신의 FTLR 추이를 읽는 기준으로만 쓰고 아무것도 게이트하지 않는다.",
        "",
        "## 고정된 조건과 두 축",
        "",
        f"- 8셀 전부 동일: camera {CAMERA_WIDTH}×{CAMERA_HEIGHT}, **detect {DETECT_WIDTH}×"
        f"{DETECT_HEIGHT}(= camera, decoupled 금지)**, `min_pixels={DETECTOR_MIN_PIXELS}`,"
        f" 검출 거리 {DETECTOR_MAX_RANGE_M} m(미설정=기본값), {BARS} bars,"
        f" 목표 {GOAL_DIST_MIN_M}–{GOAL_DIST_MAX_M} m, seed {SEED},"
        " deterministic/original, governor off, appearance·지연·거리오차 0.",
        f"- **축 1(검출기 고정 시)**: `{MANIPULATED_VARIABLE}`만 다르다. 각 검출기의 네 환경을",
        "  모든 쌍으로 비교해 확인한다.",
        f"- **축 2(distractor 수 고정 시)**: `{'`, `'.join(DETECTOR_SELECTION_VARIABLES)}`만",
        "  다르다. 세 변수는 하나의 요인 수준이다 — 아티팩트와 그 동작점, 그리고 그 동작점이",
        "  필요로 하는 좁은 override. 각 N에서 두 환경을 비교해 확인한다.",
        "  두 방향을 **따로** 검사한다: 한 번에 뭉뚱그리면 검출기를 따라 움직인 엉뚱한 변수가",
        "  '검출기 축의 일부'로 변명될 수 있다.",
        "- 실행 후에는 결과 문서로 같은 두 방향을 다시 확인한다: 검출기 안에서는 `condition`의",
        "  `distractor_count`만, N 안에서는 evaluation contract의 "
        f"`{'`/`'.join(DETECTOR_OWNED_CONTRACT_KEYS)}`만 움직여야 한다.",
        f"- **담요식 `NAVRL_V2_FORCE`는 어느 셀에서도 쓰지 않는다.** `{DEFAULT_DETECTOR}` 셀은",
        "  동결 체크포인트를 그것이 학습된 센서 조건에서 평가하므로 override 없이 preflight가",
        f"  통과해야 한다. `{V7_DETECTOR}` 셀은 정책이 본 적 없는 동작점(thr "
        f"{detector_spec(V7_DETECTOR)[2]:g})에서 돌므로 evaluator가 **반드시** 한 필드를 문제",
        f"  삼는다 — force 없는 실행이 `{EXPECTED_THRESHOLD_MISMATCH}` **한 줄로만** 거부되는",
        f"  것을 실행 시점에 증명한 뒤에야 좁은 `{NARROW_OVERRIDE_VARIABLE}`를 적용한다.",
        "  두 줄이거나 다른 필드면 중단한다.",
        "",
        "## 권한",
        "",
        "이 실험은 **결정 권한이 없다**. P2 STRICT FAIL·D1 FAIL을 소급 변경하지 않고 P3를",
        "해제하지 않으며, **v7 offline gate의 8/8 PASS도 소급 변경하지 않는다** — 그 게이트는",
        "distractor가 없는 조건에서 frame precision 0.99766을 측정했고 그 조건 안에서 유효하다.",
        "본 실험은 같은 아티팩트를 같은 동작점(thr 0.700)에서 **그 조건 밖**에서 잰다. 두 결과는",
        "모순이 아니라 서로 다른 질문이다. 결과가 어떻든 **검출기 교체를 승인하지 않는다**",
        "(별도 사전등록 필요).",
        "",
        f"- 정책 체크포인트 SHA-256 `{payload['checkpoint_sha256'][:16]}…`",
        "- 검출기 요인 수준:",
    ] + [
        (
            f"  - `{name}`: "
            + (
                f"아티팩트 `{payload['detectors'][name]['artifact']}` "
                f"(sha256 `{payload['detectors'][name]['artifact_sha256'][:16]}…`)"
                if payload["detectors"][name]["artifact"]
                else "내장 `AppearanceTargetSegmenter` (아티팩트 없음, 3R − 2G − 2B − 0.9)"
            )
            + f", 동작점 thr {payload['detectors'][name]['threshold']:g}, "
            f"판정 셀 `{payload['detectors'][name]['verdict_cell']}`"
        )
        for name, _, _, _ in DETECTORS
    ] + [
        f"- 품질 게이트: 판정 {len(evaluated)}개, 위임 {len(delegated)}개, 실패 {len(failed)}개",
        "",
        "## 한계 (사전등록 §6)",
        "",
    ])
    lines.extend("- " + item for item in LIMITATIONS)
    lines.append("")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------


def _cell_argument(argv):
    names = [cell for cell, _, _ in CELLS]
    if len(argv) == 3:
        require(argv[2] in names, f"unknown cell {argv[2]!r}; expected one of {names}")
        return argv[2]
    require(len(argv) == 2, f"usage: {PRODUCER} {argv[1]} [{'|'.join(names)}]")
    return None


def run_preflight() -> int:
    """Everything cheap and CPU-only.  This command starts no GPU work of any kind.

    The evaluator is invoked with NAVRL_PREFLIGHT_ONLY=1, which exits before Isaac Gym is imported
    and before the result directory is created -- it only reads the checkpoint on the CPU and
    compares its provenance.  That is deliberate: a preflight that could disturb a machine already
    running something else would not be run when it is most needed.
    """
    require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
    gate0 = verify_prerequisites()
    detectors = verify_detector_artifacts()
    print("[distractor] PREFLIGHT")
    print(f"[distractor]   checkpoint: {CHECKPOINT_REL}")
    print(f"[distractor]   checkpoint sha256: {CHECKPOINT_SHA}")
    for label, item in (
        ("0.1 default-off bit-identity", gate0["default_off"]),
        ("0.2 decoupling refusal", gate0["decoupling_refusal"]),
    ):
        print(
            "[distractor]   Gate %s: %s (%d tests + static audit)"
            % (label, "PASS" if item["passed"] else "FAIL", item["tests_run"])
        )
        for finding in item["findings"]:
            print(f"[distractor]     {finding}")
    reference = lineage_reference()
    print(
        "[distractor]   Gate 0.3 lineage reference: "
        f"{reference['source']} (artifact reachable: {reference['artifact_reachable']}) "
        f"tolerance +/-{N0_REGRESSION_TOLERANCE_PP:.2f} pp"
    )

    dirty = runtime_dirty_paths()
    if dirty:
        print(f"[distractor]   runtime clean: FAIL ({len(dirty)} path(s))")
        for line in dirty[:8]:
            print(f"[distractor]     {line}")
    else:
        print("[distractor]   runtime clean: PASS")

    print("[distractor]   detector factor levels:")
    for detector, relative, sha, threshold in DETECTORS:
        item = detectors[detector]
        print(
            "[distractor]     %s: %s | thr %g | verdict cell %s"
            % (
                detector,
                (f"{relative} sha256={sha[:16]}..." if relative else "built-in segmenter"),
                threshold,
                verdict_cell(detector),
            )
        )
        if item.get("gate_reachable"):
            print(
                "[distractor]       gate %s: frame precision %.5f at threshold %g"
                % (
                    item["gate_summary"],
                    item["gate_frame_precision"],
                    item["gate_selected_threshold"],
                )
            )
    print("[distractor]   evaluation env diff (measured per factor, not claimed):")
    for axis, mapping in evaluation_env_diff().items():
        print(f"[distractor]     {axis}:")
        for key, value in mapping.items():
            print(f"[distractor]       {key}: {value}")

    for cell, _, _ in CELLS:
        verify_provenance_override(cell)

    require(
        gate0_static_passed(gate0),
        "Gate 0 failed: FAIL_CLOSED_IMPLEMENTATION (prereg section 5). Findings: "
        + gate0_failure_report(gate0),
    )
    require(
        not dirty,
        "runtime source is dirty; the evaluator's schema-2 receipt records a clean-source contract "
        f"and verify would reject the cells. Commit these first: {dirty[:8]}",
    )
    print(
        f"[distractor] PREFLIGHT PASS | seed={SEED} bars={BARS} | {EPISODES} episodes/cell | "
        f"{len(CELLS)} cells ({len(DETECTORS)}x{len(DISTRACTOR_COUNTS)} factorial) "
        f"{[c for c, _, _ in CELLS]} | camera-resolution path {CAMERA_WIDTH}x{CAMERA_HEIGHT} | "
        f"no blanket force; narrow threshold override on {NARROW_OVERRIDE_DETECTOR} cells only"
    )
    return 0


def main() -> int:
    argv = sys.argv
    mode = argv[1] if len(argv) >= 2 else ""
    require(
        mode in {"preflight", "evaluate", "finalize", "verify"},
        f"usage: {PRODUCER} {{preflight|evaluate|finalize|verify}}",
    )

    if mode == "preflight":
        require(len(argv) == 2, f"usage: {PRODUCER} preflight")
        return run_preflight()

    if mode == "evaluate":
        cell = _cell_argument(argv)
        targets = [cell] if cell else [name for name, _, _ in CELLS]
        for name in targets:
            if cell_dir(name).exists():
                print(f"[distractor] cell {name}: cell already exists, skipping")
                continue
            evaluate_cell(name)
        pending = [name for name, _, _ in CELLS if not cell_dir(name).exists()]
        if pending:
            # A per-cell invocation is the normal way to run this: the cells are 2,049 episodes
            # each and may be hours apart.  Summarising now would fail on the cell that does not
            # exist yet and would report that as an error instead of as "not finished".
            print(f"[distractor] EVALUATE COMPLETE {targets} | still pending: {pending}")
            return 0
        gate0 = verify_prerequisites()
        payload = build_summary(verify_all(), gate0)
        print(
            "[distractor] EVALUATE COMPLETE | verdicts="
            f"{payload['verdicts']} | next: finalize"
        )
        return 0

    require(len(argv) == 2, f"usage: {PRODUCER} {mode}")
    gate0 = verify_prerequisites()
    expected = build_summary(verify_all(), gate0)
    if mode == "finalize":
        write_summary(expected)
        print(f"[distractor] FINALIZE PASS | {expected['verdicts']} -> {SUMMARY_JSON}")
        return 0

    require(SUMMARY_JSON.is_file(), f"summary missing: {SUMMARY_JSON}")
    recorded = load_json(SUMMARY_JSON)
    for key in SUMMARY_VERIFY_KEYS:
        require(
            _verify_payload(recorded.get(key)) == _verify_payload(expected.get(key)),
            f"summary changed: {key}",
        )
    print(f"[distractor] VERIFY PASS | {recorded['verdicts']}")
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
        print(f"[distractor] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
