#!/usr/bin/env python3
"""Detection-range stage 1 (screening) -- preregistered TRAINING + evaluation launcher.

Preregistration: docs/prereg_2026-08-22_detection_range_2stage.md (frozen; sections 4, 5, 7, 8
binding).  Stage 2 is NOT implemented here and must not be started from this file.

Two PPO adaptation arms, warm-started from the SAME frozen ref5in D1 ep1900 checkpoint, trained
for the SAME 1,000 epochs / 4.096 M samples, differing in exactly one environment variable:

  arm clip20   NAVRL_DETECTOR_MAX_RANGE = 20.0   -- today's hard clip
  arm clip28   NAVRL_DETECTOR_MAX_RANGE = 28.0   -- the whole 22.5-28 m goal band inside range

Each arm's terminal checkpoint is then evaluated ON ITS OWN ARM'S CLIP (prereg section 4), seed
461, 2,049 episodes.  Both arms train and evaluate under one internally consistent target-token
normalisation (rel_pos / max_camera_range, navrl_perception.py:1574,1578), which is why the
normalisation change is a property of the sensor and not a confound -- unlike seed 367, where a
20 m-trained policy was fed a 28 m normalisation.

WHY THIS IS A SCREEN AND NOT AN ANSWER (prereg section 3): 1,000 epochs of warm-start adaptation
cannot answer "what is achievable at this clip".  Both arms start from a policy whose search
strategy was learned in a world blind past 20 m, so only arm clip28 has anything to unlearn --
the design is BIASED AGAINST clip28.  A positive result therefore punched through a handicap and
is trustworthy; a NEGATIVE result means "undecided at this budget", NOT "no effect".  That
reading is fixed here, before any measurement.

Stage-1 gates:
  Gate 0 (training soundness, BEFORE any verdict)  both arms must reach ``max_epochs`` normally,
      show no KL-driven rollback, and record a terminal SHA.  An arm that fails is VOID and is
      reported as VOID -- never silently folded into the comparison.
  Gate S (screening, primary)  never-acquired delta (clip28 minus clip20, percentage points):
      <= -15.00 pp  ->  RANGE_HELPS ;  otherwise  RANGE_INCONCLUSIVE_AT_THIS_BUDGET.

capture / crash / timeout are reported RAW and are excluded from the verdict by construction --
classify_verdict() takes one number and can see nothing else (prereg section 5).

This experiment has NO decision authority.  It cannot revise the P2 STRICT FAIL or the D1 FAIL and
it cannot unlock P3 (prereg section 7).  `RANGE_HELPS` authorises writing stage 2, nothing more.

Usage:
  python tools/run_navrl_ref5in_detection_range_stage1.py preflight
  python tools/run_navrl_ref5in_detection_range_stage1.py train {clip20|clip28}
  python tools/run_navrl_ref5in_detection_range_stage1.py evaluate [clip20|clip28]
  python tools/run_navrl_ref5in_detection_range_stage1.py finalize
  python tools/run_navrl_ref5in_detection_range_stage1.py verify
"""

from __future__ import annotations

from datetime import datetime, timezone
import glob
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time


# ----------------------------------------------------------------------------------------------
# Preregistered contract constants.  Nothing below this block may recompute, relax or re-derive a
# value that appears here.  Declared ABOVE any measurement so that no number produced by a run can
# reach back and change a threshold.
# ----------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "aerial_gym/rl_training/rl_games"
TRAINER = RL_ROOT / "train_navrl_v2_search.sh"
EVALUATOR = RL_ROOT / "eval_navrl_v2_density_sweep.sh"
IMPORT_ORIGIN_GUARD = RL_ROOT / "navrl_import_origin.py"
SOURCE_BUNDLE_TOOL = ROOT / "tools/create_navrl_source_bundle.py"
DETECTOR_SOURCE = ROOT / "aerial_gym/task/navrl_task/navrl_detector.py"
PERCEPTION_SOURCE = ROOT / "aerial_gym/task/navrl_task/navrl_perception.py"
TASK_CONFIG_SOURCE = ROOT / "aerial_gym/config/task_config/navrl_task_config.py"
CANONICAL_PYTHON = Path("/home/fair/miniconda3/envs/aerialgym/bin/python")

TRAIN_SEED = 457                # exhaustive-search usage count 0 (prereg section 4)
EVAL_SEED = 461                 # exhaustive-search usage count 0 (prereg section 4)
BARS = 70
EPISODES = 2049

# Budget.  MAX_EPOCHS is ABSOLUTE, not a delta: runner.py:850 overwrites the yaml max_epochs and
# the agent resumes at the checkpoint's epoch, so 1,000 epochs of adaptation from ep1900 is
# ``--max_epochs 2900``.  The frozen D1 run is the worked example (ep900 -> --max_epochs 1900 ->
# 1,000 rows in aerial_run/epoch_metrics.csv).
WARM_START_EPOCH = 1900
ADAPT_EPOCHS = 1000
TERMINAL_EPOCH = WARM_START_EPOCH + ADAPT_EPOCHS
NUM_ENVS = 128
PPO_HORIZON = 32
SAMPLES_PER_EPOCH = NUM_ENVS * PPO_HORIZON          # 4,096
ADAPT_SAMPLES = ADAPT_EPOCHS * SAMPLES_PER_EPOCH    # 4,096,000 (prereg section 4)
TERMINAL_FRAME = TERMINAL_EPOCH * SAMPLES_PER_EPOCH  # 11,878,400

CHECKPOINT_SHA = "197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e"
CHECKPOINT_REL = (
    "aerial_gym/rl_training/rl_games/runs/"
    "ppo_260813_1636_navrl_v2-ref5in-d1-q3-adapt-s197/nn/"
    "last_gen_ppo_ep_1900_rew_182.11377.pth"
)

# (arm directory name, detector max range in metres).  The range is the WHOLE arm axis; every
# other condition is shared and asserted identical, per arm, in code.
ARMS = (("clip20", 20.0), ("clip28", 28.0))
TREATMENT_ARM = "clip28"        # the arm whose never-acquired is subtracted FROM
CONTROL_ARM = "clip20"

# Prereg section 5, Gate S.  A SINGLE threshold, compared by name.  -15.00 pp is under half the
# timeout change seed 367 saw (-37.65 pp) and is deliberately conservative because the warm-start
# design handicaps clip28 (prereg section 3).
NEVER_ACQUIRED_HELPS_THRESHOLD_PP = -15.00

VERDICT_HELPS = "RANGE_HELPS"
VERDICT_INCONCLUSIVE = "RANGE_INCONCLUSIVE_AT_THIS_BUDGET"
VERDICT_VOID = "STAGE1_VOID"    # Gate 0 failure: nothing may be claimed about detection range

# Held fixed in BOTH arms, in BOTH training and evaluation (prereg section 4).  These are the
# honest-sensor conditions established by docs/prereg_2026-08-22_sensor_fidelity.md; the point of
# stage 1 is to move the clip while the sensor model stays honest.
DETECT_WIDTH = 1920
DETECT_HEIGHT = 1200
DETECTOR_MIN_PIXELS = 50
CAMERA_WIDTH = 160
CAMERA_HEIGHT = 90
GOAL_DIST_MIN_M = 22.5
GOAL_DIST_MAX_M = 28.0
ROBOT_NAME = "navrl_ref5in_quad"
LEARNING_RATE = "1.5e-5"        # the frozen D1 adaptation LR, unchanged (train_navrl_v2_ref5in_d1_adapt.sh)
SAVE_FREQUENCY = "250"
ACTION_SELECTION = "deterministic"
REFLECTION_MODE = "original"
SPEED_GOVERNOR_MODE = "off"

# Appearance / injected-error knobs that must be zero.  detect != camera decoupling is an identity
# only at zero appearance perturbation and the runtime fails closed otherwise
# (docs/prereg_2026-08-22_sensor_fidelity.md section 4), so these are asserted, not assumed.
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
# Prereg section 4 lists these as "= 0".  train_navrl_v2_search.sh:276 UNSETS both, and
# navrl_task.py:2386,2389 read them as ``os.environ.get(name, "0")`` -- so absence IS zero, and
# exporting a literal 0 would be erased by the launcher anyway.  Absence is therefore the
# assertion, and it is checked on the environment the trainer actually produces.
UNSET_MEANS_ZERO_KEYS = ("NAVRL_REFLECTION_COEF", "NAVRL_LATERAL_BIAS_COEF")

OUTPUT = ROOT / "results" / "navrl_ref5in_detection_range_stage1_s457"
SOURCE_BUNDLE = OUTPUT / "source_bundle"              # shared EVALUATION bundle (schema 2)
SUMMARY_JSON = OUTPUT / "summary.json"
SUMMARY_MD = OUTPUT / "summary.md"
# The shared TRAINING source receipt (schema_version 1, tools/create_navrl_source_bundle.py) --
# not the schema-2 evaluation bundle above.  ONE receipt for both arms, deliberately: it is the
# machine-checkable statement that the two arms executed the same bytes, and the runtime is
# re-hashed against it before the second arm starts.
TRAIN_RECEIPT_DIR = RL_ROOT / "train_source_receipts" / "detection_range_stage1_s457"
TRAIN_RECEIPT_MANIFEST = TRAIN_RECEIPT_DIR / "source_manifest.json"
# Preflight's throwaway VRAM/step-time measurement.  Deliberately OUTSIDE OUTPUT: preflight
# refuses to run once OUTPUT exists, and a measurement that is not an experimental artifact must
# not be able to satisfy that check.
SMOKE_OUTPUT = ROOT / "results" / "navrl_ref5in_detection_range_stage1_s457_preflight"
SMOKE_EPOCHS = 6
SMOKE_ARM = TREATMENT_ARM
VRAM_LIMIT_MIB = 8192           # RTX 3070 board limit

# Adopting a run this launcher did not start.  ``train`` is not the only legal way to produce an
# arm -- the trainer can be driven directly -- and the verification half must not be unusable just
# because it did not press the button.  The operator names the run folder and the tee'd training
# log; every Gate 0 fact is then RE-DERIVED from those artifacts exactly as for a launcher-started
# run.  The one thing adoption cannot recover is which clip the run trained at: navrl_task records
# older checkpoints had no detector-geometry provenance.  Stage 1 now refuses to adopt such a
# checkpoint: the arm assignment must be attested by the checkpoint, not supplied by an operator.
ADOPT_RUN_ROOT_ENV = "DETRANGE_STAGE1_RUN_ROOT_%s"
ADOPT_TRAIN_LOG_ENV = "DETRANGE_STAGE1_TRAIN_LOG_%s"
CLIP_EVIDENCE_LAUNCHER = "set_by_this_launcher_and_verified_in_the_trainer_effective_environment"
CLIP_EVIDENCE_ADOPTED = "checkpoint_env_state_detector_geometry_attestation"

PREREGISTRATION = "docs/prereg_2026-08-22_detection_range_2stage.md"
PRODUCER = "tools/run_navrl_ref5in_detection_range_stage1.py"
SCOPE = "detection_range_stage1_screening_s457_eval_s461"

# The primary measurand is READ from one evaluator-emitted field, never invented here.  Identical
# source and identical pooling to tools/run_navrl_ref5in_sensor_fidelity.py, which documents it:
# navrl_task.py first_acquisition_payload() -> result.target_motion.first_acquisition.
NEVER_ACQUIRED_SOURCE = (
    'result["target_motion"]["first_acquisition"][outcome]["never_acquired"] '
    "summed over capture/crash/timeout, divided by the same cohorts' episode counts"
)
FIRST_ACQUISITION_OUTCOMES = ("capture", "crash", "timeout")

# Gate 0 evidence keys (training soundness, prereg section 5) and the evaluation gates.  Each name
# maps to the verify step's key carrying this launcher's own proof that it ran the check; a gate
# with no such evidence is a check nobody performed and fails closed.
TRAINING_GATES = {
    "T1_budget_reached": "budget",
    "T2_normal_max_epochs_exit": "exit",
    "T3_no_kl_rollback": "rollback",
    "T4_terminal_sha_recorded": "terminal_sha",
    "T5_training_receipt_clean": "training_receipt",
    "T6_training_condition_pinned": "training_condition",
}
PER_ARM_GATES = {
    "G1_checkpoint_identity": "checkpoint_identity",
    "G2_result_receipt_binding": "result_receipt_binding",
    "G3_manifest_provenance": "manifest_provenance",
    "G4_runtime_clean": "manifest_provenance",
    "G5_import_origin": "import_origin",
    "G6_episode_contract": "episode_contract",
    "G7_arm_condition_pinned": "arm_condition",
    "G10_no_provenance_override": "no_override",
}
CROSS_ARM_GATES = {
    "G8_runtime_byte_map_identity": "runtime_map_identity",
    "G9_single_axis_manipulation": "single_axis",
    "G11_shared_training_receipt": "shared_training_receipt",
}

ORIGIN_LOG_MARKER = "[origin] aerial_gym "
ORIGIN_LINE_HEAD = r"^\[origin\] aerial_gym "
ORIGIN_LINE_TAIL = r" sha256=(?P<sha256>[0-9a-f]{64}) \(enforced\)$"
ORIGIN_MANIFEST_ENTRY = "aerial_gym/__init__.py"
ROLLBACK_LOG_MARKER = "[aerial RL] PPO EPOCH ROLLBACK"
KL_SKIP_LOG_MARKER = "[aerial RL] PPO epoch rejection latched"
PPO_ROLLBACK_TOTAL_KEY = "aerial_ppo_rollback_total"
PPO_ROLLBACK_STREAK_KEY = "aerial_ppo_rollback_streak"
MAX_EPOCHS_EXIT_REASON = "max_epochs"

# Every variable the preregistration pins, and what the script chain is allowed to do to it.  The
# launcher sets these; the chain (train_navrl_v2_search.sh -> train_navrl.sh -> runner.py) runs
# afterwards and can quietly erase any of them.  Three outcomes are legal and each must be
# DECLARED here, so a new clobber is a failure rather than a discovery made after a run:
#
#   pass through          the chain never mentions the variable
#   clobbered to the same the chain hard-codes exactly the value this launcher pins, so the run is
#     value we pin        the intended one either way (the literal is checked, not assumed)
#   conditionally erased  the chain unsets it under a condition this launcher pins OFF
#
# NAVRL_NUM_BARS is the live example of the third case and the reason this table exists: the chain
# unsets it inside ``if NAVRL_DENSITY_CURRICULUM == 1``, handing bar count to the density
# curriculum.  Arm clip28 acquires more easily, so it would promote sooner and the arms would have
# differed in DENSITY as well as clip -- two axes, not one, with both runs looking healthy.
CHAIN_SCRIPTS = ("train_navrl_v2_search.sh", "train_navrl.sh")
PINNED_TRAINING_VARIABLES = (
    "NAVRL_DETECTOR_MAX_RANGE",
    "NAVRL_DETECT_WIDTH",
    "NAVRL_DETECT_HEIGHT",
    "NAVRL_DETECTOR_MIN_PIXELS",
    "NAVRL_CAMERA_WIDTH",
    "NAVRL_CAMERA_HEIGHT",
    "NAVRL_NUM_BARS",
    "NAVRL_DENSITY_CURRICULUM",
    "NAVRL_GENERAL_GOAL_DIST_MIN",
    "NAVRL_GENERAL_GOAL_DIST_MAX",
    "NAVRL_SPEED_GOVERNOR",
    "NAVRL_PERCEPTION_PERTURB",
    "NAVRL_ROBOT",
    "NAVRL_LEARNING_RATE",
) + ZERO_PERTURBATION_KEYS + UNSET_MEANS_ZERO_KEYS
# variable -> the exact literal the chain hard-codes, which must equal what this launcher pins.
CHAIN_CLOBBERS_TO_PINNED_VALUE = {
    "NAVRL_PERCEPTION_PERTURB": "0",
}
# variable -> (guard variable, guard value that triggers the erasure, reason it is safe here).
CHAIN_CONDITIONAL_ERASURES = {
    "NAVRL_NUM_BARS": (
        "NAVRL_DENSITY_CURRICULUM",
        "1",
        "train_navrl_v2_search.sh unsets NAVRL_NUM_BARS only while the density curriculum owns the "
        "bar count; this experiment pins NAVRL_DENSITY_CURRICULUM=0, so the fixed 70 bars survive "
        "-- and the effective-environment check proves it did",
    ),
    "NAVRL_REFLECTION_COEF": (None, None, "unset unconditionally; the task reads unset as 0"),
    "NAVRL_LATERAL_BIAS_COEF": (None, None, "unset unconditionally; the task reads unset as 0"),
}

# train_navrl_v2_search.sh must honour these as OVERRIDES rather than hard-coding them, or the
# preregistered held-fixed condition silently does not happen.  ``NAVRL_DETECTOR_MIN_PIXELS`` is
# the live example: hard-coded to 2, it would have trained BOTH arms at the dishonest threshold
# while every log still looked normal.  The literals are checked before any GPU time; the
# effective-environment diff below is the independent, behavioural check on the same fact.
TRAINER_OVERRIDABLE_LITERALS = (
    'export NAVRL_DETECTOR_MIN_PIXELS="${NAVRL_DETECTOR_MIN_PIXELS:-2}"',
)

# Prereg section 7 -- transcribed, not summarised: the limitations travel with the numbers.
LIMITATIONS = [
    "L1: P2 STRICT FAIL / D1 FAIL / P3 BLOCKED를 변경하지 않는다. 2단계는 70막대 고정 10k이므로"
    " P3(70→205막대, 30k, seed 211)가 아니다.",
    "L2: RANGE_CONFIRMED가 나와도 정책을 채택하지 않는다 — 채택은 P2 gate 통과가 필요하다.",
    "L3: 거리 오차가 여전히 0이다(NAVRL_RANGE_ERROR_M=0, 해석적 정확값). 실기 28 m 스테레오 시차는"
    " 1.2–2.4 px로 측정 불가다. 따라서 '실기 준비됨'을 주장하지 않는다.",
    "L4: 잡동사니 배경·모션 블러 미모델링 → 결과는 낙관 편향.",
    "L5: 1단계 음성은 '효과 없음'이 아니라 '이 예산에서 미결'이다(사전등록 §3).",
]

SUMMARY_VERIFY_KEYS = (
    "schema_version",
    "producer",
    "scope",
    "stage",
    "decision_authority",
    "p2_verdict_changed",
    "d1_verdict_changed",
    "p3_unlocked",
    "stage2_authorised",
    "preregistration",
    "warm_start_checkpoint",
    "warm_start_checkpoint_sha256",
    "shared_condition",
    "provenance_override",
    "arms",
    "void_arms",
    "primary_metric",
    "never_acquired_delta_pp",
    "threshold_pp",
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


P2 = load_module("detection_range_p2", ROOT / "tools/attest_navrl_ref5in_p2.py")
# Reuse -- never re-implement -- the audited contract helpers and the dirty-runtime gate.
BASE = load_module(
    "detection_range_base", ROOT / "tools/run_navrl_ref5in_cv_heading_diagnostic.py"
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


def _resolve_checkpoint() -> Path:
    """Resolve the pinned warm-start checkpoint, following the shared git dir when runs/ is remote.

    Git worktrees intentionally do not duplicate the gitignored multi-GB runs/ tree, so the
    checkpoint exists only in the primary worktree.  Identity is pinned by CHECKPOINT_SHA, which is
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


def arm_clip(arm: str) -> float:
    for name, clip in ARMS:
        if name == arm:
            return clip
    raise ContractError(f"unknown arm: {arm!r}; the preregistered arms are {[a for a, _ in ARMS]}")


def train_dir(arm: str) -> Path:
    return OUTPUT / "training" / arm


def cell_dir(arm: str) -> Path:
    return OUTPUT / "cells" / arm


def train_paths(arm: str) -> dict:
    directory = train_dir(arm)
    return {
        "record": directory / "training_record.json",
        "run_summary": directory / "run_summary.json",
        "log": directory / "train.log",
    }


def cell_paths(arm: str) -> dict:
    directory = cell_dir(arm)
    return {
        "result": directory / ("%dbars.json" % BARS),
        "receipt": directory / ("%dbars.receipt.json" % BARS),
        "log": directory / ("%dbars.log" % BARS),
        "snapshot": directory / "checkpoint_snapshot.pth",
        "stdout_log": directory / "detection_range_eval.log",
    }


def resolve_recorded_path(recorded, arm: str, label: str) -> Path:
    """Locate a receipt-recorded artifact without trusting the path the producer happened to write.

    The evaluator records ABSOLUTE paths, so a receipt produced in a git worktree names files that
    only exist there.  Two candidates are tried, in order: the copy that travels with the cell,
    then the absolute path the receipt recorded.  Neither is a trust boundary; identity is pinned
    by the digests the caller checks.  Nothing is resolved implicitly: if neither candidate exists
    the message names both and the check fails closed.
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
# Training environment
# ----------------------------------------------------------------------------------------------


def _closed_process_env() -> dict:
    """A closed base environment: no ambient NAVRL_* or launcher variable may reach a run.

    Same closure discipline as attest_navrl_ref5in_p2.canonical_env, extended with the training
    launcher's own variables.  The training wrappers in this repository scrub these with a
    ``compgen -v`` loop; building the dictionary from scratch is the same guarantee expressed where
    it can be tested.
    """
    blocked = {
        "AERIAL_RUN_TAG",
        "AERIAL_GYM_SIM_NAME",
        "AERIAL_RUN_SUMMARY",
        "ALLOW_CONCURRENT",
        "CKPT",
        "FILE",
        "GPU4GB",
        "HEADLESS",
        "MAX_EPOCHS",
        "NUM_ENVS",
        "PLAY_GAMES_NUM",
        "PYTHON",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTORCH_CUDA_ALLOC_CONF",
        "SEED",
        "TASK",
        "TRAIN_LIVE_LOG",
        "TRAIN_SESSION_LOG",
    }
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("NAVRL_") and key not in blocked
    }


def training_env(arm: str, *, preflight: bool, receipt=None, max_epochs=None, tag=None) -> dict:
    """The environment handed to train_navrl_v2_search.sh for one arm.

    Everything the v2 contract owns (arena, placement, curriculum knots, PPO safety, action policy,
    LiDAR, target motion) is NOT restated here -- the canonical launcher exports it, and
    effective_training_env() below reads back what it actually produced.  This function sets only
    the frozen-lineage conditions the D1 adaptation used, the honest-sensor conditions the sensor
    fidelity preregistration established, and the ONE manipulated variable.

    ``receipt`` is the shared training source receipt as returned by create/verify; when it is
    None the run is a throwaway measurement and declares no receipt, which is why the VRAM smoke
    can execute on a tree whose runtime change is not committed yet while ``train`` cannot.
    """
    clip = arm_clip(arm)
    run_tag = tag or f"v2-ref5in-detrange-{arm}-s{TRAIN_SEED}"
    env = _closed_process_env()
    env.update(
        {
            "PYTHON": str(CANONICAL_PYTHON),
            "PATH": str(CANONICAL_PYTHON.parent) + os.pathsep + os.environ.get("PATH", ""),
            "PYTHONNOUSERSITE": "1",
            # The 128-env PhysX + compiled Transformer footprint sits close to the 8 GB board
            # limit; the D1 adaptation needed this allocator setting to survive its first backward
            # buffer.  It changes memory segmentation only -- not the model, batch or task.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            # An editable install hard-codes ONE worktree's absolute path, so without these the run
            # can execute the primary tree while its receipt hashes this one.  PYTHONPATH makes the
            # right tree win; NAVRL_REQUIRE_SOURCE_ROOT makes runner.py prove it (runner.py:29-36).
            "PYTHONPATH": str(ROOT),
            "NAVRL_REQUIRE_SOURCE_ROOT": str(ROOT),
            # ---- warm start ----
            "CKPT": str(CHECKPOINT),
            "NAVRL_V2_ALLOW_RESUME": "1",
            "MAX_EPOCHS": str(max_epochs if max_epochs is not None else TERMINAL_EPOCH),
            "SEED": str(TRAIN_SEED),
            "NAVRL_V2_PROFILE": "main",
            # ---- frozen ref5in D1 lineage, unchanged ----
            "NAVRL_ROBOT": ROBOT_NAME,
            "NAVRL_YAW_RATE_MAX": "3.0",
            "NAVRL_MAX_TILT_DEG": "45.0",
            "NAVRL_ALT_HOLD_VMAX": "2.5",
            "NAVRL_SPEED_GOVERNOR": SPEED_GOVERNOR_MODE,
            "NAVRL_PERCEPTION_PERTURB": "0",
            "NAVRL_GENERAL_GOAL_DIST_MIN": str(GOAL_DIST_MIN_M),
            "NAVRL_GENERAL_GOAL_DIST_MAX": str(int(GOAL_DIST_MAX_M)),
            "NAVRL_DENSITY_CURRICULUM": "0",
            "NAVRL_NUM_BARS": str(BARS),
            "NAVRL_DENSITY_START": str(BARS),
            "NAVRL_DENSITY_FINAL": str(BARS),
            "NAVRL_LEARNING_RATE": LEARNING_RATE,
            "NAVRL_SAVE_FREQUENCY": SAVE_FREQUENCY,
            # ---- honest sensor, identical in both arms (prereg section 4) ----
            "NAVRL_DETECT_WIDTH": str(DETECT_WIDTH),
            "NAVRL_DETECT_HEIGHT": str(DETECT_HEIGHT),
            "NAVRL_DETECTOR_MIN_PIXELS": str(DETECTOR_MIN_PIXELS),
            "NAVRL_CAMERA_WIDTH": str(CAMERA_WIDTH),
            "NAVRL_CAMERA_HEIGHT": str(CAMERA_HEIGHT),
            # ---- the ONE manipulated variable (prereg section 4) ----
            "NAVRL_DETECTOR_MAX_RANGE": f"{clip:.1f}",
            # ---- bookkeeping: run identity and logs, not conditions ----
            "AERIAL_RUN_TAG": run_tag,
            "TRAIN_SESSION_LOG": f"train_session_logs/{run_tag}.log",
            "TRAIN_LIVE_LOG": f"train_session_logs/current_{run_tag}.log",
            "NAVRL_V2_CONTRACT_PREFLIGHT_ONLY": "1" if preflight else "0",
        }
    )
    for key in ZERO_PERTURBATION_KEYS:
        env[key] = "0"
    if receipt is not None:
        env.update(
            {
                "NAVRL_TRAINING_SOURCE_MANIFEST": str(receipt["manifest"]),
                "NAVRL_TRAINING_SOURCE_MANIFEST_SHA256": str(receipt["manifest_sha256"]),
                "NAVRL_REQUIRE_TRAINING_SOURCE_RECEIPT": "1",
                "NAVRL_REQUIRE_CLEAN_TRAINING_SOURCE": "1",
            }
        )
    require(
        env["NAVRL_DETECTOR_MAX_RANGE"] == f"{clip:.1f}",
        f"{arm}: the manipulated variable did not land: {env.get('NAVRL_DETECTOR_MAX_RANGE')!r}",
    )
    return env


def effective_training_env(arm: str, **kwargs) -> dict:
    """The environment train_navrl_v2_search.sh ACTUALLY produces for one arm.

    Not what this launcher passes in -- what survives the canonical launcher.  The trainer both
    unsets variables and re-exports many of its own, so "the parent set X" is not evidence that
    the run saw X.  The measurement: source the trainer with
    ``NAVRL_V2_CONTRACT_PREFLIGHT_ONLY=1`` (which makes it exit 0 after every export and before
    ``exec``) under an EXIT trap that dumps ``env -0`` to a private file descriptor.

    This is the check that catches a hard-coded condition.  ``NAVRL_DETECTOR_MIN_PIXELS`` was
    hard-coded to 2 in the trainer: the parent environment said 50, the run would have used 2, and
    nothing downstream of "both arms trained normally" could have told the difference.
    """
    dump = SMOKE_OUTPUT / f".effective_env_{arm}"
    dump.parent.mkdir(parents=True, exist_ok=True)
    script = (
        'dump="$1"; shift\n'
        'script="$1"; shift\n'
        "trap 'env -0 > \"$dump\"' EXIT\n"
        'source "$script" "$@"\n'
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "effective_training_env",
            str(dump),
            str(TRAINER),
            "--checkpoint",
            str(CHECKPOINT),
            "--branch_run",
        ],
        cwd=str(RL_ROOT),
        env=training_env(arm, preflight=True, **kwargs),
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    require(
        completed.returncode == 0 and dump.is_file(),
        f"{arm}: could not read the trainer's effective environment "
        f"(returncode={completed.returncode}): {(completed.stdout or '').strip()[-800:]}",
    )
    require(
        "[v2-search] PREFLIGHT PASS (child handoff validated; training not started)"
        in (completed.stdout or ""),
        f"{arm}: the trainer did not reach its own preflight marker; the environment dump would "
        "describe a partially configured run",
    )
    raw = dump.read_bytes()
    dump.unlink()
    effective = {}
    for item in raw.split(b"\0"):
        if not item:
            continue
        text = item.decode("utf-8", "replace")
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        effective[key] = value
    require(bool(effective), f"{arm}: the trainer environment dump was empty")
    return effective


def verify_effective_training_env(arm: str, effective: dict) -> dict:
    """Every preregistered training condition, checked on the environment the trainer produced."""
    clip = arm_clip(arm)
    pinned = {
        "NAVRL_DETECTOR_MAX_RANGE": f"{clip:.1f}",
        "NAVRL_DETECT_WIDTH": str(DETECT_WIDTH),
        "NAVRL_DETECT_HEIGHT": str(DETECT_HEIGHT),
        "NAVRL_DETECTOR_MIN_PIXELS": str(DETECTOR_MIN_PIXELS),
        "NAVRL_CAMERA_WIDTH": str(CAMERA_WIDTH),
        "NAVRL_CAMERA_HEIGHT": str(CAMERA_HEIGHT),
        "NAVRL_NUM_BARS": str(BARS),
        "NAVRL_DENSITY_CURRICULUM": "0",
        "NAVRL_GENERAL_GOAL_DIST_MIN": str(GOAL_DIST_MIN_M),
        "NAVRL_GENERAL_GOAL_DIST_MAX": str(int(GOAL_DIST_MAX_M)),
        "NAVRL_SPEED_GOVERNOR": SPEED_GOVERNOR_MODE,
        "NAVRL_PERCEPTION_PERTURB": "0",
        "NAVRL_ROBOT": ROBOT_NAME,
        "SEED": str(TRAIN_SEED),
        "NAVRL_SEED": str(TRAIN_SEED),
        "NUM_ENVS": str(NUM_ENVS),
        "FILE": "ppo_navrl_perception_transformer.yaml",
        "TASK": "navrl_task",
        "AERIAL_GYM_SIM_NAME": "base_sim",
        "NAVRL_V2_ALLOW_RESUME": "1",
        "CKPT": str(CHECKPOINT),
        "NAVRL_LEARNING_RATE": LEARNING_RATE,
        "NAVRL_REQUIRE_SOURCE_ROOT": str(ROOT),
        "PYTHONPATH": str(ROOT),
    }
    mismatch = {
        key: (effective.get(key), value)
        for key, value in pinned.items()
        if effective.get(key) != value
    }
    require(
        not mismatch,
        f"{arm}: the trainer's effective environment does not carry the preregistered condition "
        f"(effective, expected): {mismatch}",
    )
    for key in ZERO_PERTURBATION_KEYS:
        require(
            float(effective.get(key, "nan")) == 0.0,
            f"{arm}: {key}={effective.get(key)!r} is non-zero; detect-resolution decoupling is "
            "not an identity under it (sensor fidelity prereg section 4)",
        )
    for key in UNSET_MEANS_ZERO_KEYS:
        require(
            key not in effective,
            f"{arm}: {key} survived into the training environment; the trainer unsets it and the "
            "task reads an unset value as 0, so any surviving value is an unrequested condition",
        )
    return {
        "checked_by_launcher": True,
        "detector_max_range_m": float(effective["NAVRL_DETECTOR_MAX_RANGE"]),
        "detector_min_pixels": int(effective["NAVRL_DETECTOR_MIN_PIXELS"]),
        "detect_resolution": [
            int(effective["NAVRL_DETECT_WIDTH"]),
            int(effective["NAVRL_DETECT_HEIGHT"]),
        ],
        "camera_resolution": [
            int(effective["NAVRL_CAMERA_WIDTH"]),
            int(effective["NAVRL_CAMERA_HEIGHT"]),
        ],
        "num_envs": int(effective["NUM_ENVS"]),
        "max_epochs": int(effective["MAX_EPOCHS"]),
    }


def training_env_diff(**kwargs) -> dict:
    """The per-arm difference of the two EFFECTIVE training environments.

    Not a comment claiming the arms differ in one variable -- the actual symmetric difference of
    the two dictionaries the canonical trainer produced.  Anything else that ever diverges (a
    stray export, a future knob, a trainer change) surfaces here, before any GPU time.
    """
    environments = {}
    for arm, _ in ARMS:
        effective = effective_training_env(arm, **kwargs)
        verify_effective_training_env(arm, effective)
        environments[arm] = effective
    (name_a, env_a), (name_b, env_b) = environments.items()
    manipulated = {"NAVRL_DETECTOR_MAX_RANGE"}
    # Run identity and log destinations.  They must differ -- each arm owns its run folder and its
    # tee'd log -- and they are listed explicitly rather than filtered silently, so an unexplained
    # extra difference still fails.
    bookkeeping = {"AERIAL_RUN_TAG", "TRAIN_SESSION_LOG", "TRAIN_LIVE_LOG"}
    expected = manipulated | bookkeeping
    differing = {key for key in set(env_a) | set(env_b) if env_a.get(key) != env_b.get(key)}
    require(
        differing == expected,
        f"the two training environments differ in {sorted(differing)}, not in {sorted(expected)}; "
        "the detection-range screen must be a single-axis manipulation (prereg section 4)",
    )
    return {key: {name_a: env_a.get(key), name_b: env_b.get(key)} for key in sorted(manipulated)}


# ----------------------------------------------------------------------------------------------
# Evaluation environment
# ----------------------------------------------------------------------------------------------


def evaluation_env(arm: str, *, preflight: bool) -> dict:
    """P2's closed evaluation environment plus exactly the prereg section 4 additions.

    Two ordering facts are load-bearing and are asserted rather than commented.  ``PYTHONPATH`` is
    DELETED by P2.canonical_env to keep its environment closed, but play_navrl.sh cds into
    aerial_gym/rl_training/rl_games where no aerial_gym/ package directory exists, so without
    re-injection the editable-install finder resolves the PRIMARY worktree and the run would
    execute one tree while its receipt hashes another.  ``NAVRL_DETECTOR_MIN_PIXELS`` is already
    inside P2's closed set at the value 2, so the honest-sensor threshold must be applied AFTER
    that call or both arms would evaluate at the dishonest threshold.

    NAVRL_V2_FORCE is never set, in either arm.  Each arm is evaluated at the clip it was TRAINED
    at, and the evaluator's conditional provenance gate compares every detector-geometry key
    (eval_navrl_v2_density_sweep.sh:650-698), so there is nothing for an override to override.
    """
    clip = arm_clip(arm)
    env = P2.canonical_env(cell_dir(arm), preflight=preflight)
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "NAVRL_REQUIRE_SOURCE_ROOT": str(ROOT),
            "NAVRL_SEED": str(EVAL_SEED),
            "NAVRL_V2_DENSITIES": str(BARS),
            "NAVRL_V2_RESULT_DIR": str(cell_dir(arm)),
            "NAVRL_V2_SHARED_SOURCE_BUNDLE": str(SOURCE_BUNDLE),
            # Fixed by the checkpoint contract, not by choice: the arms are trained at 22.5-28 m
            # and the generic evaluator refuses any other goal band.
            "NAVRL_V2_GOAL_DIST_MIN": str(GOAL_DIST_MIN_M),
            "NAVRL_V2_GOAL_DIST_MAX": str(int(GOAL_DIST_MAX_M)),
            # Honest sensor, identical in both arms and identical to training.
            "NAVRL_CAMERA_WIDTH": str(CAMERA_WIDTH),
            "NAVRL_CAMERA_HEIGHT": str(CAMERA_HEIGHT),
            "NAVRL_DETECT_WIDTH": str(DETECT_WIDTH),
            "NAVRL_DETECT_HEIGHT": str(DETECT_HEIGHT),
            "NAVRL_DETECTOR_MIN_PIXELS": str(DETECTOR_MIN_PIXELS),
            # ---- the ONE manipulated variable: each arm on ITS OWN clip (prereg section 4) ----
            "NAVRL_DETECTOR_MAX_RANGE": f"{clip:.1f}",
        }
    )
    require(
        float(env["NAVRL_V2_GOAL_DIST_MIN"]) == GOAL_DIST_MIN_M
        and float(env["NAVRL_V2_GOAL_DIST_MAX"]) == GOAL_DIST_MAX_M,
        "exported goal band drifted from the pinned constants: "
        f"{env['NAVRL_V2_GOAL_DIST_MIN']}-{env['NAVRL_V2_GOAL_DIST_MAX']}",
    )
    require(
        int(env["NAVRL_DETECTOR_MIN_PIXELS"]) == DETECTOR_MIN_PIXELS,
        f"{arm}: NAVRL_DETECTOR_MIN_PIXELS is {env['NAVRL_DETECTOR_MIN_PIXELS']!r}, not the "
        f"honest-sensor {DETECTOR_MIN_PIXELS}; the update must land AFTER P2.canonical_env, whose "
        "closed set already contains a value for it",
    )
    require(
        int(env["NAVRL_DETECT_WIDTH"]) == DETECT_WIDTH
        and int(env["NAVRL_DETECT_HEIGHT"]) == DETECT_HEIGHT
        and int(env["NAVRL_CAMERA_WIDTH"]) == CAMERA_WIDTH
        and int(env["NAVRL_CAMERA_HEIGHT"]) == CAMERA_HEIGHT,
        f"{arm}: exported detect/camera resolution drifted from the pinned constants",
    )
    require(
        float(env["NAVRL_DETECTOR_MAX_RANGE"]) == clip,
        f"{arm}: the evaluated clip is {env['NAVRL_DETECTOR_MAX_RANGE']!r}, not the arm's {clip}",
    )
    for key in ZERO_PERTURBATION_KEYS:
        require(
            float(env[key]) == 0.0,
            f"{arm}: {key}={env[key]!r} is non-zero; detect-resolution decoupling is not an "
            "identity under it (sensor fidelity prereg section 4)",
        )
    require(env["NAVRL_PERCEPTION_PERTURB"] == "0", f"{arm}: perturbations must be off")
    require(env["NAVRL_SPEED_GOVERNOR"] == SPEED_GOVERNOR_MODE, f"{arm}: governor must be off")
    require(env["NAVRL_V2_ACTION_MODE"] == ACTION_SELECTION, f"{arm}: action must be deterministic")
    require(
        env["NAVRL_EVAL_REFLECTION_MODE"] == REFLECTION_MODE,
        f"{arm}: reflection_mode must be original",
    )
    # No arm may carry a provenance override.  Evaluating a checkpoint at the clip it was trained
    # with is the ONE configuration that needs none, and that is proved at run time by an unforced
    # preflight (verify_no_override_needed) rather than assumed here.
    require(
        "NAVRL_V2_FORCE" not in env,
        f"{arm}: NAVRL_V2_FORCE leaked into the evaluation environment; each arm is evaluated at "
        "its own training clip precisely so that no override is needed",
    )
    return env


def evaluation_env_diff() -> dict:
    """The per-arm difference of the two evaluation environments, computed not claimed."""
    environments = {}
    for arm, _ in ARMS:
        environments[arm] = evaluation_env(arm, preflight=True)
    (name_a, env_a), (name_b, env_b) = environments.items()
    manipulated = {"NAVRL_DETECTOR_MAX_RANGE"}
    bookkeeping = {"NAVRL_V2_RESULT_DIR"}
    expected = manipulated | bookkeeping
    differing = {key for key in set(env_a) | set(env_b) if env_a.get(key) != env_b.get(key)}
    require(
        differing == expected,
        f"the two evaluation environments differ in {sorted(differing)}, not in "
        f"{sorted(expected)}; the detection-range screen must be a single-axis manipulation",
    )
    return {key: {name_a: env_a.get(key), name_b: env_b.get(key)} for key in sorted(manipulated)}


# ----------------------------------------------------------------------------------------------
# Prerequisites
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


def verify_evaluator_needs_no_range_override() -> dict:
    """Static proof that evaluating each arm at its own clip cannot need a provenance override.

    The generic evaluator's v2 provenance gate compares a fixed ``want`` dictionary against the
    checkpoint's ``env_state`` (eval_navrl_v2_density_sweep.sh:650-698).  Two facts decide the
    question and both are read out of the evaluator's own source:

      1. ``cfg_detector_min_pixels`` is bound to ``os.environ["NAVRL_DETECTOR_MIN_PIXELS"]``.  Both
         arms TRAIN at 50 and are EVALUATED at 50, so this field -- the one that forced the sensor
         fidelity experiment into a narrow single-field override -- matches here by construction.
      2. New checkpoints record the complete detector geometry triplet.  The evaluator compares
         those fields only when present, preserving legacy checkpoint loadability while making
         this new experiment fail closed.  Each arm is evaluated at the exact values it records,
         so no override is necessary.
    """
    evaluator = EVALUATOR.read_text(encoding="utf-8")
    require(
        '"cfg_detector_min_pixels": float(os.environ["NAVRL_DETECTOR_MIN_PIXELS"]),' in evaluator,
        "the evaluator no longer binds cfg_detector_min_pixels to the exported threshold; the "
        "no-override claim rests on training and evaluation using the same value",
    )
    for literal in (
        '"cfg_detector_max_range": float(os.environ["NAVRL_DETECTOR_MAX_RANGE"])',
        '"cfg_detect_width": float(os.environ["NAVRL_DETECT_WIDTH"])',
        '"cfg_detect_height": float(os.environ["NAVRL_DETECT_HEIGHT"])',
        "if present_detector_geometry:",
    ):
        require(literal in evaluator, f"conditional detector provenance gate missing: {literal}")
    task_source = (ROOT / "aerial_gym/task/navrl_task/navrl_task.py").read_text(encoding="utf-8")
    for key in ("cfg_detector_max_range", "cfg_detect_width", "cfg_detect_height"):
        require(key in task_source, f"navrl_task checkpoint provenance missing {key}")
    require(
        '"target_camera_max_range_m": float(os.environ.get("NAVRL_DETECTOR_MAX_RANGE", 20.0)),'
        in evaluator,
        "the evaluator no longer records target_camera_max_range_m; the manipulated axis would "
        "then be attested nowhere at all",
    )
    return {
        "checked_by_launcher": True,
        "override_required": False,
        "reason": (
            "each arm is evaluated at the clip it was trained at; the evaluator's v2 provenance "
            "gate compares the checkpoint detector geometry to that arm's requested geometry"
        ),
        "clip_not_recorded_in_checkpoint_provenance": False,
        "clip_attested_by": "env_state.cfg_detector_max_range + v2_evaluation_contract.target_camera_max_range_m",
    }


def verify_pinned_variables_survive_the_script_chain() -> dict:
    """Mechanically audit every pinned variable against the scripts that run after this launcher.

    Reading what the launcher exports proves nothing: the chain executes afterwards and can
    ``export VAR=literal`` over it or ``unset`` it outright.  Two real cases have already been
    found by hand -- ``NAVRL_DETECTOR_MIN_PIXELS`` hard-coded to 2, and ``NAVRL_NUM_BARS`` unset
    into the density curriculum -- and each would have produced two healthy-looking runs that were
    not the preregistered experiment.  Doing it by hand does not scale to the next experiment, so
    it is done here for every pinned variable, before any GPU time.

    This is the STATIC half.  effective_training_env() is the behavioural half: it reads back the
    environment the chain actually produced.  Neither replaces the other -- the static audit says
    "nothing in the chain can erase this", the dump says "it did not".
    """
    findings = {}
    for name in CHAIN_SCRIPTS:
        path = RL_ROOT / name
        require(path.is_file(), f"script chain member missing: {path}")
        text = path.read_text(encoding="utf-8")
        for variable in PINNED_TRAINING_VARIABLES:
            clobbers = re.findall(
                r"^\s*export\s+%s=(.*)$" % re.escape(variable), text, flags=re.MULTILINE
            )
            hard = [
                value.strip()
                for value in clobbers
                if ("${%s:-" % variable) not in value
            ]
            erasures = re.findall(
                r"^\s*unset\s+([^\n]*\b%s\b[^\n]*)$" % re.escape(variable),
                text,
                flags=re.MULTILINE,
            )
            if not hard and not erasures:
                findings.setdefault(variable, []).append(f"{name}: pass-through")
                continue
            if hard:
                expected = CHAIN_CLOBBERS_TO_PINNED_VALUE.get(variable)
                require(
                    expected is not None and all(value == expected for value in hard),
                    f"{name} hard-codes {variable}={hard!r}, which this launcher pins but has not "
                    "declared as a permitted clobber; the preregistered condition would silently "
                    "not happen (use ${%s:-<default>} in the script, or declare it here with the "
                    "literal it pins)" % variable,
                )
                findings.setdefault(variable, []).append(
                    f"{name}: clobbered to the pinned value {expected!r}"
                )
            if erasures:
                declared = CHAIN_CONDITIONAL_ERASURES.get(variable)
                require(
                    declared is not None,
                    f"{name} unsets {variable} ({erasures!r}) and this launcher has not declared "
                    "why that is safe; the preregistered condition would silently not happen",
                )
                _, _, reason = declared
                findings.setdefault(variable, []).append(f"{name}: erased -- {reason}")
    # The guard this launcher relies on for the conditional erasure must actually be pinned off.
    guard, guard_value, _ = CHAIN_CONDITIONAL_ERASURES["NAVRL_NUM_BARS"]
    pinned_guard = training_env(CONTROL_ARM, preflight=True)[guard]
    require(
        pinned_guard != guard_value,
        f"{guard} is pinned to {pinned_guard!r}, the very value that makes the chain hand "
        "NAVRL_NUM_BARS to the density curriculum; the arms would then differ in density as well "
        "as clip",
    )
    return {
        "checked_by_launcher": True,
        "scripts": list(CHAIN_SCRIPTS),
        "variables_audited": len(PINNED_TRAINING_VARIABLES),
        "findings": {key: sorted(value) for key, value in sorted(findings.items())},
        "density_curriculum_pinned_to": pinned_guard,
    }


def verify_prerequisites() -> dict:
    """Everything cheap, before any GPU second is spent."""
    require(CHECKPOINT.is_file(), f"pinned warm-start checkpoint missing: {CHECKPOINT}")
    require(
        P2.sha256_file(CHECKPOINT) == CHECKPOINT_SHA,
        "pinned warm-start checkpoint identity mismatch",
    )
    require(TRAINER.is_file(), f"canonical trainer missing: {TRAINER}")
    require(EVALUATOR.is_file(), f"canonical evaluator missing: {EVALUATOR}")
    require(SOURCE_BUNDLE_TOOL.is_file(), f"training receipt tool missing: {SOURCE_BUNDLE_TOOL}")
    require(IMPORT_ORIGIN_GUARD.is_file(), f"import-origin guard missing: {IMPORT_ORIGIN_GUARD}")
    require(CANONICAL_PYTHON.is_file(), f"canonical Python missing: {CANONICAL_PYTHON}")
    runner = (RL_ROOT / "runner.py").read_text(encoding="utf-8")
    require(
        "[origin] aerial_gym %s sha256=%s (enforced)" in runner,
        "runner.py no longer prints the enforced import-origin line that G5 verifies",
    )

    # The decoupling this experiment rests on must be present in the runtime that will execute --
    # both the knobs and the fail-closed guard that makes them honest.  A missing guard would let
    # a 1920x1200 detection run against a perturbed appearance, where the decoupling is NOT an
    # identity, and nothing downstream could tell.
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

    # A condition the canonical trainer hard-codes is a condition this experiment cannot set.
    trainer = TRAINER.read_text(encoding="utf-8")
    for literal in TRAINER_OVERRIDABLE_LITERALS:
        require(
            literal in trainer,
            "the canonical trainer hard-codes a preregistered held-fixed condition instead of "
            f"honouring it as an override; expected: {literal}",
        )
    chain_audit = verify_pinned_variables_survive_the_script_chain()
    override = verify_evaluator_needs_no_range_override()
    override["chain_audit"] = chain_audit
    return override


# ----------------------------------------------------------------------------------------------
# Training source receipt (schema_version 1 -- NOT the schema-2 evaluation bundle)
# ----------------------------------------------------------------------------------------------


def _receipt_command(*args) -> dict:
    completed = subprocess.run(
        [str(CANONICAL_PYTHON), str(SOURCE_BUNDLE_TOOL)] + list(args),
        cwd=str(ROOT),
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(
        completed.returncode == 0,
        "training source receipt command failed: "
        f"{' '.join(args)} -> {(completed.stderr or completed.stdout or '').strip()}",
    )
    return json.loads(completed.stdout)


def ensure_training_receipt() -> dict:
    """Create the shared training receipt once, then VERIFY it before every later arm.

    One receipt for both arms is the machine-checkable form of "the two arms executed the same
    bytes".  It is not a cached blessing: ``verify`` re-hashes every original file against the
    manifest, so a runtime edit made between arm A and arm B stops arm B here rather than
    producing two runs of two different programs.
    """
    if TRAIN_RECEIPT_MANIFEST.is_file():
        receipt = _receipt_command("verify", "--manifest", str(TRAIN_RECEIPT_MANIFEST))
        print(f"[detrange] shared training receipt VERIFIED | {TRAIN_RECEIPT_MANIFEST}")
        return receipt
    dirty = runtime_dirty_paths()
    require(
        not dirty,
        "refusing to create a clean-contract training receipt from dirty runtime sources; "
        f"commit first: {dirty[:8]}",
    )
    receipt = _receipt_command(
        "create", "--output", str(TRAIN_RECEIPT_DIR), "--require-clean"
    )
    print(f"[detrange] shared training receipt CREATED | {TRAIN_RECEIPT_MANIFEST}")
    return receipt


# ----------------------------------------------------------------------------------------------
# Running a child with a streamed log
# ----------------------------------------------------------------------------------------------


def tee_run(command: list, env: dict, log_path: Path, cwd: Path, watcher=None) -> int:
    """Run a child, streaming its combined output to the console and to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as sink:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        if watcher is not None:
            watcher.start()
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            sink.write(line)
        process.stdout.close()
        returncode = process.wait()
    if watcher is not None:
        watcher.stop()
    return returncode


class RunWatcher(threading.Thread):
    """Sample GPU memory and epoch progress while a training child runs.

    Both series are polled rather than parsed out of the log: the per-epoch CSV the run recorder
    rewrites (aerial_run/epoch_metrics.csv) is the only per-epoch artifact with a guaranteed
    shape, and nvidia-smi is the only number that answers "does this fit in the 8 GB board"
    including the PhysX and warp allocations torch never sees.
    """

    def __init__(self, run_glob: str, interval: float = 1.0):
        super().__init__(daemon=True)
        self.run_glob = run_glob
        # Folders that already match the tag are EARLIER runs.  Counting their rows would make the
        # series start at some finished epoch count and then jump backwards when this run's folder
        # appears, which silently destroys the per-epoch timing (it did, on the first smoke).
        self.preexisting = set(glob.glob(run_glob))
        self.interval = interval
        self.peak_mib = 0
        self.total_mib = 0
        self.samples = 0
        self.epoch_marks = []          # (monotonic seconds, epoch rows)
        # NOT ``self._stop``: threading.Thread already owns that name (Thread._stop is called by
        # join()), and shadowing it with an Event makes join() raise instead of waiting.
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            used, total = self._gpu_memory()
            if used is not None:
                self.samples += 1
                self.peak_mib = max(self.peak_mib, used)
                self.total_mib = max(self.total_mib, total)
            rows = self._epoch_rows()
            if rows is not None and (not self.epoch_marks or rows != self.epoch_marks[-1][1]):
                self.epoch_marks.append((time.monotonic(), rows))
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5.0)

    @staticmethod
    def _gpu_memory():
        try:
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                universal_newlines=True,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            return None, None
        first = raw.strip().splitlines()[0]
        used, total = (int(part.strip()) for part in first.split(","))
        return used, total

    def _epoch_rows(self):
        matches = sorted(set(glob.glob(self.run_glob)) - self.preexisting)
        if not matches:
            return None
        csv_path = Path(matches[-1]) / "aerial_run" / "epoch_metrics.csv"
        if not csv_path.is_file():
            return None
        try:
            with csv_path.open("r", encoding="utf-8") as stream:
                return max(0, sum(1 for _ in stream) - 1)
        except OSError:
            return None

    def seconds_per_epoch(self):
        """Seconds per epoch over the longest strictly increasing run of row counts.

        Taking first-and-last would be wrong the moment the series is not monotone; the timing is
        only meaningful across samples that are all from the same, still-growing CSV.  Startup is
        excluded by construction because the first mark is the first epoch that finished.
        """
        best = None
        segment = []
        for mark in self.epoch_marks:
            if mark[1] <= 0:
                segment = []
                continue
            if segment and mark[1] <= segment[-1][1]:
                segment = []
            segment.append(mark)
            if len(segment) >= 2:
                span = (segment[-1][0] - segment[0][0]) / (segment[-1][1] - segment[0][1])
                if best is None or len(segment) > best[0]:
                    best = (len(segment), span)
        return None if best is None else best[1]


def find_run_root(tag: str) -> Path:
    """The runs/ folder rl-games created for this tag (runner.py:535-546)."""
    pattern = str(RL_ROOT / "runs" / f"ppo_*_navrl_{tag}")
    matches = sorted(glob.glob(pattern))
    require(bool(matches), f"no run folder matched {pattern}")
    return Path(matches[-1]).resolve()


# ----------------------------------------------------------------------------------------------
# Preflight VRAM / step-time smoke
# ----------------------------------------------------------------------------------------------


def run_vram_smoke() -> dict:
    """A few real training epochs at the full stage-1 condition, to measure before committing.

    1920x1200 detection was measured at 1.92x per step and ~402 MB torch peak in EVALUATION at 128
    envs (WORKLOG 2026-08-22).  Training adds the backward pass and the optimizer to the same 8 GB
    board, and the frozen D1 adaptation already needed ``expandable_segments`` to survive its first
    backward buffer -- so the evaluation figure does not settle the question and a short smoke
    does.  It runs at the smoke tag, declares NO training source receipt (it is a measurement, not
    an artifact), and its run folder is left in the gitignored runs/ tree as evidence.
    """
    SMOKE_OUTPUT.mkdir(parents=True, exist_ok=True)
    tag = f"v2-ref5in-detrange-smoke-{SMOKE_ARM}-s{TRAIN_SEED}"
    max_epochs = WARM_START_EPOCH + SMOKE_EPOCHS
    env = training_env(SMOKE_ARM, preflight=False, max_epochs=max_epochs, tag=tag)
    watcher = RunWatcher(str(RL_ROOT / "runs" / f"ppo_*_navrl_{tag}"))
    log_path = SMOKE_OUTPUT / "vram_smoke.log"
    print(
        f"[detrange] VRAM SMOKE | arm={SMOKE_ARM} clip={arm_clip(SMOKE_ARM):.1f} m "
        f"detect {DETECT_WIDTH}x{DETECT_HEIGHT} camera {CAMERA_WIDTH}x{CAMERA_HEIGHT} "
        f"envs={NUM_ENVS} epochs {WARM_START_EPOCH}->{max_epochs}",
        flush=True,
    )
    started = time.monotonic()
    returncode = tee_run(
        ["bash", str(TRAINER), "--checkpoint", str(CHECKPOINT), "--branch_run"],
        env,
        log_path,
        RL_ROOT,
        watcher=watcher,
    )
    wall_s = time.monotonic() - started
    require(returncode == 0, f"VRAM smoke failed with exit code {returncode}; see {log_path}")
    run_root = find_run_root(tag)
    summary_path = run_root / "aerial_run" / "run_summary.json"
    require(summary_path.is_file(), f"smoke produced no run summary: {summary_path}")
    summary = load_json(summary_path)
    require(
        summary.get("exit_reason") == MAX_EPOCHS_EXIT_REASON,
        f"smoke did not end at max_epochs: exit_reason={summary.get('exit_reason')!r}",
    )
    require(
        watcher.peak_mib > 0,
        "no GPU memory sample was taken; the smoke cannot answer whether the run fits in 8 GB",
    )
    require(
        watcher.peak_mib <= VRAM_LIMIT_MIB,
        f"peak GPU memory {watcher.peak_mib} MiB exceeds the {VRAM_LIMIT_MIB} MiB board",
    )
    payload = {
        "schema_version": 1,
        "producer": PRODUCER,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "stage1_vram_and_step_time_smoke",
        "arm": SMOKE_ARM,
        "detector_max_range_m": arm_clip(SMOKE_ARM),
        "detect_resolution": [DETECT_WIDTH, DETECT_HEIGHT],
        "camera_resolution": [CAMERA_WIDTH, CAMERA_HEIGHT],
        "detector_min_pixels": DETECTOR_MIN_PIXELS,
        "num_envs": NUM_ENVS,
        "epochs_requested": SMOKE_EPOCHS,
        "epochs_logged": summary.get("epochs_logged"),
        "first_epoch": summary.get("first_epoch"),
        "last_epoch": summary.get("last_epoch"),
        "exit_reason": summary.get("exit_reason"),
        "peak_gpu_memory_mib": watcher.peak_mib,
        "gpu_total_mib": watcher.total_mib or VRAM_LIMIT_MIB,
        "gpu_headroom_mib": (watcher.total_mib or VRAM_LIMIT_MIB) - watcher.peak_mib,
        "gpu_samples": watcher.samples,
        "seconds_per_epoch": watcher.seconds_per_epoch(),
        "wall_seconds_total": wall_s,
        "projected_arm_hours": (
            None
            if watcher.seconds_per_epoch() is None
            else watcher.seconds_per_epoch() * ADAPT_EPOCHS / 3600.0
        ),
        "run_root": str(run_root),
        "log": str(log_path),
    }
    (SMOKE_OUTPUT / "vram_smoke.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------


def train_arm(arm: str) -> dict:
    paths = train_paths(arm)
    require(
        not train_dir(arm).exists(),
        f"refusing overwrite: {train_dir(arm)} already exists; a second adaptation of the same "
        "arm would silently replace a preregistered run",
    )
    prerequisites = verify_prerequisites()
    dirty = runtime_dirty_paths()
    require(not dirty, f"runtime source is dirty; commit before a receipt-bearing run: {dirty[:8]}")
    receipt = ensure_training_receipt()
    training_env_diff(receipt=receipt)

    clip = arm_clip(arm)
    env = training_env(arm, preflight=False, receipt=receipt)
    tag = env["AERIAL_RUN_TAG"]
    staged_log = OUTPUT / f"{arm}.train.log.partial"
    staged_log.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[detrange] TRAIN {arm} | clip {clip:.1f} m | warm start ep{WARM_START_EPOCH} -> "
        f"ep{TERMINAL_EPOCH} ({ADAPT_EPOCHS} epochs / {ADAPT_SAMPLES:,} samples) | seed "
        f"{TRAIN_SEED} | bars {BARS}",
        flush=True,
    )
    returncode = tee_run(
        ["bash", str(TRAINER), "--checkpoint", str(CHECKPOINT), "--branch_run"],
        env,
        staged_log,
        RL_ROOT,
    )
    require(returncode == 0, f"{arm}: training exited with code {returncode}")

    run_root = find_run_root(tag)
    terminal = sorted(
        (run_root / "nn").glob(f"last_gen_ppo_ep_{TERMINAL_EPOCH}_rew_*.pth")
    )
    require(
        len(terminal) == 1,
        f"{arm}: expected exactly one terminal checkpoint for epoch {TERMINAL_EPOCH} in "
        f"{run_root / 'nn'}, found {[p.name for p in terminal]}",
    )
    terminal_path = terminal[0].resolve()
    terminal_sha = P2.sha256_file(terminal_path)
    train_dir(arm).mkdir(parents=True, exist_ok=True)
    staged_log.replace(paths["log"])
    shutil.copy2(run_root / "aerial_run" / "run_summary.json", paths["run_summary"])
    record = {
        "schema_version": 1,
        "producer": PRODUCER,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": arm,
        "adopted": False,
        "detector_max_range_m": clip,
        "detector_max_range_evidence": CLIP_EVIDENCE_LAUNCHER,
        "run_root": str(run_root),
        "terminal_checkpoint": str(terminal_path),
        "terminal_checkpoint_sha256": terminal_sha,
        "warm_start_checkpoint": str(CHECKPOINT),
        "warm_start_checkpoint_sha256": CHECKPOINT_SHA,
        "training_source_manifest": str(receipt["manifest"]),
        "training_source_manifest_sha256": str(receipt["manifest_sha256"]),
        "training_source_git_commit": str(receipt["git_commit"]),
        "epoch_metrics_csv": str(run_root / "aerial_run" / "epoch_metrics.csv"),
        "finished_marker": str(run_root / ".aerial_training_finished"),
        "evaluator_needs_no_override": prerequisites,
    }
    paths["record"].write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"[detrange] TRAIN COMPLETE {arm} | {terminal_path.name} sha256={terminal_sha[:16]}… | "
        f"next: evaluate {arm}",
        flush=True,
    )
    return record


def adopt_training_run(arm: str) -> dict:
    """Build training/<arm>/ from a run folder this launcher did not start.

    Named by two environment variables so the assignment is explicit and auditable rather than
    guessed from a glob:

        DETRANGE_STAGE1_RUN_ROOT_CLIP20=<runs/ppo_...>   DETRANGE_STAGE1_TRAIN_LOG_CLIP20=<...log>

    The log is REQUIRED, not optional.  Gate 0's no-rollback check has two independent witnesses --
    the durable counters inside the checkpoint and the ``PPO EPOCH ROLLBACK`` lines in the log --
    and adopting a run without the log would quietly reduce that to one.
    """
    root_key = ADOPT_RUN_ROOT_ENV % arm.upper()
    log_key = ADOPT_TRAIN_LOG_ENV % arm.upper()
    raw_root = os.environ.get(root_key, "").strip()
    raw_log = os.environ.get(log_key, "").strip()
    require(
        bool(raw_root) and bool(raw_log),
        f"{arm}: no training record, and this run was not adopted. Either run `train {arm}`, or "
        f"point the launcher at an existing run:\n  {root_key}=<runs/ppo_..._navrl_...>\n  "
        f"{log_key}=<the tee'd training log for that run>",
    )
    run_root = Path(raw_root).expanduser().resolve()
    log = Path(raw_log).expanduser().resolve()
    require(run_root.is_dir(), f"{arm}: {root_key} is not a directory: {run_root}")
    require(log.is_file(), f"{arm}: {log_key} is not a file: {log}")
    summary_source = run_root / "aerial_run" / "run_summary.json"
    require(summary_source.is_file(), f"{arm}: adopted run has no run summary: {summary_source}")
    terminal = sorted((run_root / "nn").glob(f"last_gen_ppo_ep_{TERMINAL_EPOCH}_rew_*.pth"))
    require(
        len(terminal) == 1,
        f"{arm}: expected exactly one terminal checkpoint for epoch {TERMINAL_EPOCH} in "
        f"{run_root / 'nn'}, found {[path.name for path in terminal]}",
    )
    terminal_path = terminal[0].resolve()

    import torch  # local: CPU-only callers must not pay for torch at import time

    checkpoint = torch.load(str(terminal_path), map_location="cpu", weights_only=False)
    state = checkpoint.get("env_state") or {}
    detector_geometry = {
        "cfg_detector_max_range": arm_clip(arm),
        "cfg_detect_width": DETECT_WIDTH,
        "cfg_detect_height": DETECT_HEIGHT,
    }
    mismatch = {
        key: (state.get(key), expected)
        for key, expected in detector_geometry.items()
        if state.get(key) != expected
    }
    require(
        not mismatch,
        f"{arm}: adopted checkpoint cannot attest the requested detector geometry: {mismatch}",
    )
    manifest_path = Path(str(state.get("cfg_training_source_manifest", "")))
    require(
        manifest_path.is_file(),
        f"{arm}: the adopted checkpoint names a training source manifest that is not present: "
        f"{manifest_path}",
    )
    manifest = load_json(manifest_path)
    train_dir(arm).mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_source, train_paths(arm)["run_summary"])
    shutil.copy2(log, train_paths(arm)["log"])
    record = {
        "schema_version": 1,
        "producer": PRODUCER,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": arm,
        "adopted": True,
        "detector_max_range_m": arm_clip(arm),
        "detector_max_range_evidence": CLIP_EVIDENCE_ADOPTED,
        "adopted_from": {"run_root_env": root_key, "train_log_env": log_key},
        "run_root": str(run_root),
        "terminal_checkpoint": str(terminal_path),
        "terminal_checkpoint_sha256": P2.sha256_file(terminal_path),
        "warm_start_checkpoint": str(CHECKPOINT),
        "warm_start_checkpoint_sha256": CHECKPOINT_SHA,
        "training_source_manifest": str(manifest_path),
        "training_source_manifest_sha256": str(
            state.get("cfg_training_source_manifest_sha256", "")
        ),
        "training_source_git_commit": str(manifest.get("git_commit", "")),
        "epoch_metrics_csv": str(run_root / "aerial_run" / "epoch_metrics.csv"),
        "finished_marker": str(run_root / ".aerial_training_finished"),
    }
    train_paths(arm)["record"].write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[detrange] arm {arm}: ADOPTED existing run {run_root.name}")
    return record


# ----------------------------------------------------------------------------------------------
# Gate 0 -- training soundness
# ----------------------------------------------------------------------------------------------


def verify_training(arm: str) -> dict:
    """Gate 0 (prereg section 5): max_epochs reached normally, no KL rollback, terminal SHA.

    Every fact is re-derived from the artifacts, never trusted from training_record.json -- the
    record is an index, not evidence.  The rollback counters are the durable ones the agent writes
    into the checkpoint itself (ppo_update_safety.PPO_ROLLBACK_TOTAL_KEY), so a rollback cannot be
    hidden by trimming a log, and the log scan is the second, independent witness.
    """
    paths = train_paths(arm)
    for key in ("record", "run_summary", "log"):
        require(paths[key].is_file(), f"{arm}: missing training artifact: {paths[key]}")
    record = load_json(paths["record"])
    require(record.get("arm") == arm, f"{arm}: training record names a different arm")
    require(
        float(record.get("detector_max_range_m", -1.0)) == arm_clip(arm),
        f"{arm}: training record clip {record.get('detector_max_range_m')!r} is not the arm's "
        f"{arm_clip(arm)}",
    )

    terminal_path = Path(str(record.get("terminal_checkpoint", "")))
    require(terminal_path.is_file(), f"{arm}: terminal checkpoint missing: {terminal_path}")
    terminal_sha = P2.sha256_file(terminal_path)
    require(
        terminal_sha == record.get("terminal_checkpoint_sha256"),
        f"{arm}: terminal checkpoint bytes differ from the recorded digest",
    )
    require(
        re.fullmatch(r"[0-9a-f]{64}", str(terminal_sha)) is not None,
        f"{arm}: terminal checkpoint digest is malformed",
    )

    import torch  # local: the CPU-only verification path must not pay for torch at import time

    checkpoint = torch.load(str(terminal_path), map_location="cpu", weights_only=False)
    state = checkpoint.get("env_state") or {}
    # Gate 0 is a VERDICT, not an abort.  A short arm is a real experimental outcome the
    # preregistration has a name for -- VOID -- and a launcher that raised here would report an
    # exception instead of reporting the outcome.  Identity failures above still abort, because a
    # checkpoint that is not the file it claims to be is not evidence of anything.
    epoch = int(checkpoint.get("epoch", -1))
    frame = int(checkpoint.get("frame", -1))
    budget = {
        "checked_by_launcher": True,
        "passed": epoch == TERMINAL_EPOCH and frame == TERMINAL_FRAME,
        "terminal_epoch": epoch,
        "terminal_frame": frame,
        "expected_terminal_epoch": TERMINAL_EPOCH,
        "expected_terminal_frame": TERMINAL_FRAME,
        "adaptation_epochs": ADAPT_EPOCHS,
        "adaptation_samples": ADAPT_SAMPLES,
    }

    summary = load_json(paths["run_summary"])
    marker = Path(str(record.get("finished_marker", "")))
    marker_text = (
        marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
    )
    exit_evidence = {
        "checked_by_launcher": True,
        "passed": (
            summary.get("exit_reason") == MAX_EPOCHS_EXIT_REASON
            and int(summary.get("epochs_logged", -1)) == ADAPT_EPOCHS
            and int(summary.get("first_epoch", -1)) == WARM_START_EPOCH + 1
            and int(summary.get("last_epoch", -1)) == TERMINAL_EPOCH
            and marker_text == f"epoch={TERMINAL_EPOCH}"
        ),
        "exit_reason": summary.get("exit_reason"),
        "expected_exit_reason": MAX_EPOCHS_EXIT_REASON,
        "epochs_logged": int(summary.get("epochs_logged", -1)),
        "first_epoch": int(summary.get("first_epoch", -1)),
        "last_epoch": int(summary.get("last_epoch", -1)),
        "finished_marker": str(marker),
        "finished_marker_text": marker_text,
    }

    rollback_total = int(checkpoint.get(PPO_ROLLBACK_TOTAL_KEY, 0) or 0)
    rollback_streak = int(checkpoint.get(PPO_ROLLBACK_STREAK_KEY, 0) or 0)
    rollback_lines = []
    kl_skip_lines = []
    with paths["log"].open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if ROLLBACK_LOG_MARKER in line:
                rollback_lines.append(line.strip())
            elif KL_SKIP_LOG_MARKER in line:
                kl_skip_lines.append(line.strip())
    rollback = {
        "checked_by_launcher": True,
        # Gate 0 forbids a KL-driven rollback.  Two independent witnesses must agree: the durable
        # counters the agent writes into the checkpoint itself, and the training log.
        "passed": rollback_total == 0 and rollback_streak == 0 and not rollback_lines,
        "rollback_total": rollback_total,
        "rollback_streak": rollback_streak,
        "rollback_log_lines": len(rollback_lines),
        "rollback_log_sample": rollback_lines[:4],
        # KL-rejected minibatches are NOT a rollback and are not a Gate 0 failure: the update is
        # skipped and the epoch commits normally.  Reported so a quiet drift is visible.
        "kl_skipped_log_lines": len(kl_skip_lines),
        "kl_skip_last_line": kl_skip_lines[-1] if kl_skip_lines else None,
    }

    manifest_sha = str(record.get("training_source_manifest_sha256", ""))
    require(
        state.get("cfg_training_source_manifest_sha256") == manifest_sha,
        f"{arm}: the checkpoint binds training receipt "
        f"{state.get('cfg_training_source_manifest_sha256')!r}, not the recorded {manifest_sha!r}",
    )
    require(
        state.get("cfg_training_source_git_dirty") is False,
        f"{arm}: the checkpoint records a dirty training source",
    )
    # The receipt the checkpoint NAMES must still hash to the digest the checkpoint RECORDS.  For
    # an adopted run this is the check that carries T5: the record was written from env_state, so
    # comparing the two would be circular, while re-hashing the manifest on disk is not.
    receipt_manifest = Path(str(state.get("cfg_training_source_manifest", "")))
    require(
        receipt_manifest.is_file()
        and P2.sha256_file(receipt_manifest) == manifest_sha
        and len(manifest_sha) == 64,
        f"{arm}: the training source manifest {receipt_manifest} does not hash to the digest the "
        f"checkpoint records ({manifest_sha!r})",
    )
    training_receipt = {
        "checked_by_launcher": True,
        "passed": True,
        "manifest_sha256": manifest_sha,
        "manifest": str(record.get("training_source_manifest", "")),
        "git_commit": str(record.get("training_source_git_commit", "")),
        "runtime_file_count": int(state.get("cfg_training_source_runtime_file_count", -1)),
        "git_dirty": bool(state.get("cfg_training_source_git_dirty")),
    }

    pinned = {
        "cfg_training_seed": TRAIN_SEED,
        "cfg_training_num_envs": NUM_ENVS,
        "cfg_training_file": "ppo_navrl_perception_transformer.yaml",
        "cfg_training_task": "navrl_task",
        "cfg_training_sim": "base_sim",
        "cfg_training_profile": "main",
        "cfg_ppo_horizon": PPO_HORIZON,
        "cfg_robot_name": ROBOT_NAME,
        "cfg_obstacle_selector": "cluster_sector",
        "cfg_action_policy": "squashed_gaussian",
        "cfg_speed_governor_mode": SPEED_GOVERNOR_MODE,
        "cfg_perception_perturb": False,
        "cfg_detector_min_pixels": DETECTOR_MIN_PIXELS,
        "cfg_detector_max_range": arm_clip(arm),
        "cfg_detect_width": DETECT_WIDTH,
        "cfg_detect_height": DETECT_HEIGHT,
        "cfg_general_goal_dist_min": GOAL_DIST_MIN_M,
        "cfg_general_goal_dist_max": GOAL_DIST_MAX_M,
        "n_bars_active": BARS,
    }
    mismatch = {
        key: (state.get(key), value) for key, value in pinned.items() if state.get(key) != value
    }
    require(not mismatch, f"{arm}: training condition mismatch (recorded, expected): {mismatch}")
    training_condition = {
        "checked_by_launcher": True,
        "passed": True,
        "pinned": {key: state.get(key) for key in sorted(pinned)},
        "detector_max_range_recorded_in_checkpoint": True,
    }

    return {
        "arm": arm,
        "record": record,
        "terminal_checkpoint": terminal_path,
        "terminal_checkpoint_sha256": terminal_sha,
        "run_root": str(record.get("run_root", "")),
        "budget": budget,
        "exit": exit_evidence,
        "rollback": rollback,
        "terminal_sha": {
            "checked_by_launcher": True,
            "passed": True,
            "sha256": terminal_sha,
        },
        "training_receipt": training_receipt,
        "training_condition": training_condition,
        "reward_at_end": summary.get("reward_at_end"),
        "last_captured_rate": summary.get("last_captured_rate"),
    }


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------


def run_eval_preflight(arm: str, checkpoint: Path):
    return subprocess.run(
        ["bash", str(EVALUATOR), str(checkpoint), str(EPISODES)],
        cwd=str(ROOT),
        env=evaluation_env(arm, preflight=True),
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


def verify_no_override_needed(arm: str, checkpoint: Path) -> dict:
    """Prove AT RUN TIME that this arm needs no provenance override, and never force one.

    Evaluating a checkpoint at the clip it was TRAINED at is the configuration that should pass the
    generic evaluator's provenance gate unforced, and the static reading of that gate says why.
    This is where the reading is confirmed against the real checkpoint.  A refusal is not resolved
    with ``NAVRL_V2_FORCE`` -- blanket forcing would hide every other mismatch too -- it stops the
    run and prints the exact mismatch lines, so whoever reads them decides what actually drifted.
    """
    completed = run_eval_preflight(arm, checkpoint)
    if completed.returncode != 0:
        tail = (completed.stdout or "").strip().splitlines()[-12:]
        raise ContractError(
            f"{arm}: the generic evaluator refused this arm's own-clip evaluation "
            f"(returncode={completed.returncode}). This arm is preregistered to need NO "
            "provenance override, so it stops here instead of forcing. mismatch lines: "
            f"{mismatch_lines(completed) or 'none'} | tail: {tail}"
        )
    require(
        "[eval_v2] PREFLIGHT PASS (evaluation not started)" in (completed.stdout or ""),
        f"{arm}: evaluator preflight returned 0 without the PREFLIGHT PASS marker",
    )
    require(
        not mismatch_lines(completed),
        f"{arm}: evaluator preflight passed but still printed mismatch lines: "
        f"{mismatch_lines(completed)}",
    )
    print(f"[detrange] arm {arm}: evaluator preflight PASS (no override)")
    return {
        "checked_by_launcher": True,
        "override_used": False,
        "unforced_preflight_returncode": 0,
        "mismatch_lines": [],
    }


def evaluate_arm(arm: str) -> None:
    if not train_paths(arm)["record"].is_file():
        adopt_training_run(arm)
    require(
        not cell_dir(arm).exists(),
        f"refusing overwrite: {cell_dir(arm)} already exists",
    )
    trained = verify_training(arm)
    require(
        training_gates_passed(trained),
        f"{arm}: Gate 0 failed, so this arm is VOID and must not be evaluated (prereg section 5). "
        "Run `finalize` to publish the VOID summary. Gate 0 evidence: "
        + json.dumps(
            {
                key: trained[key]
                for key in TRAINING_GATES.values()
                if not trained[key].get("passed")
            },
            ensure_ascii=False,
        ),
    )
    checkpoint = trained["terminal_checkpoint"]
    verify_no_override_needed(arm, checkpoint)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    staged_log = OUTPUT / f"{arm}.eval.log.partial"
    print(
        f"[detrange] EVALUATE {arm} | clip {arm_clip(arm):.1f} m (its own training clip) | "
        f"seed {EVAL_SEED} | {EPISODES} episodes | {checkpoint.name}",
        flush=True,
    )
    returncode = tee_run(
        ["bash", str(EVALUATOR), str(checkpoint), str(EPISODES)],
        evaluation_env(arm, preflight=False),
        staged_log,
        ROOT,
    )
    require(returncode == 0, f"{arm}: evaluator exited with code {returncode}")
    require(cell_dir(arm).is_dir(), f"{arm}: evaluator produced no result directory")
    staged_log.replace(cell_paths(arm)["stdout_log"])


# ----------------------------------------------------------------------------------------------
# Cell verification
# ----------------------------------------------------------------------------------------------


def verify_import_origin(arm: str, mapping: dict, metadata: dict) -> dict:
    """G5: prove from the run log that the executing aerial_gym IS the tree the manifest hashed."""
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


def verify_cell(arm: str, trained: dict) -> dict:
    paths = cell_paths(arm)
    for key in ("result", "receipt", "log", "snapshot"):
        require(paths[key].is_file(), f"{arm}: missing artifact: {paths[key]}")
    result = load_json(paths["result"])
    receipt = load_json(paths["receipt"])
    clip = arm_clip(arm)
    terminal_sha = trained["terminal_checkpoint_sha256"]

    require(
        P2.sha256_file(paths["result"]) == receipt.get("result_sha256"),
        f"{arm}: result/receipt hash mismatch",
    )
    require(
        P2.sha256_file(paths["snapshot"]) == terminal_sha,
        f"{arm}: the evaluated checkpoint snapshot is not this arm's terminal checkpoint",
    )
    require(
        receipt.get("source_checkpoint_sha256") == terminal_sha,
        f"{arm}: receipt source checkpoint is not this arm's terminal checkpoint",
    )
    checkpoint_identity = {
        "checked_by_launcher": True,
        "snapshot_sha256": terminal_sha,
        "receipt_source_checkpoint_sha256": receipt.get("source_checkpoint_sha256"),
    }
    result_receipt_binding = {
        "checked_by_launcher": True,
        "result_sha256": receipt.get("result_sha256"),
    }

    pinned = {
        "seed": EVAL_SEED,
        "bars": BARS,
        "requested_episodes": EPISODES,
        "action_selection": ACTION_SELECTION,
        "reflection_mode": REFLECTION_MODE,
        "speed_governor_mode": SPEED_GOVERNOR_MODE,
        "goal_dist_min_m": GOAL_DIST_MIN_M,
        "goal_dist_max_m": GOAL_DIST_MAX_M,
        # The manipulated axis, receipt side: this arm was evaluated at ITS OWN clip.
        "target_camera_max_range_m": clip,
    }
    receipt_mismatch = {
        key: (receipt.get(key), value) for key, value in pinned.items() if receipt.get(key) != value
    }
    require(not receipt_mismatch, f"{arm}: receipt condition mismatch: {receipt_mismatch}")

    condition = result.get("condition") or {}
    condition_mismatch = {
        key: (condition.get(key), value)
        for key, value in {
            "seed": EVAL_SEED,
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

    contract = result.get("v2_evaluation_contract") or {}
    require(
        float(contract.get("target_camera_max_range_m", -1.0)) == clip,
        f"{arm}: v2 evaluation contract records target_camera_max_range_m="
        f"{contract.get('target_camera_max_range_m')!r}, not the arm's {clip}",
    )
    require(
        int(contract.get("detector_min_pixels", -1)) == DETECTOR_MIN_PIXELS,
        f"{arm}: v2 evaluation contract records detector_min_pixels="
        f"{contract.get('detector_min_pixels')!r}, not the honest-sensor {DETECTOR_MIN_PIXELS}",
    )
    arm_condition = {
        "checked_by_launcher": True,
        "target_camera_max_range_m_attested": float(contract["target_camera_max_range_m"]),
        "detector_min_pixels_attested": int(contract["detector_min_pixels"]),
        "detect_width_requested": DETECT_WIDTH,
        "detect_height_requested": DETECT_HEIGHT,
        "trained_at_same_clip": True,
    }

    # The evaluator drains whole 128-env batches, so a cell finishes at or just past the request.
    # Exact equality is WRONG here -- it has already broken one arm that landed on 2,050.
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
    no_override = {
        "checked_by_launcher": True,
        "override_used": False,
        "evidence": "evaluation_env() refuses NAVRL_V2_FORCE and evaluate_arm() requires an "
        "unforced evaluator preflight to pass before the cell is produced",
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
        "no_override": no_override,
    }


def training_gates_passed(training: dict) -> bool:
    """True when this arm cleared every Gate 0 check (prereg section 5)."""
    return all(
        bool((training.get(key) or {}).get("passed")) for key in TRAINING_GATES.values()
    )


def verify_all() -> dict:
    """Per-arm training (Gate 0) and evaluation verification, then the cross-arm invariants."""
    trainings = {}
    for arm, _ in ARMS:
        trainings[arm] = verify_training(arm)
    void_arms = sorted(arm for arm, _ in ARMS if not training_gates_passed(trainings[arm]))
    if void_arms:
        # Gate 0 comes BEFORE the verdict (prereg section 5).  A VOID arm has no cell and must not
        # acquire one, so verification stops here and reports the outcome instead of raising.
        return {
            "trainings": trainings,
            "cells": {},
            "order": tuple(arm for arm, _ in ARMS),
            "void_arms": void_arms,
            "held_fixed": {},
            "runtime_map_identity": None,
            "shared_training_receipt": None,
            "single_axis": None,
        }

    cells = {}
    for arm, _ in ARMS:
        cells[arm] = verify_cell(arm, trainings[arm])

    (name_a, cell_a), (name_b, cell_b) = ((arm, cells[arm]) for arm, _ in ARMS)

    require(
        cell_a["runtime_map"] == cell_b["runtime_map"],
        "the two arms were evaluated against different runtime byte maps; this is not a "
        "single-axis manipulation",
    )
    runtime_map_identity = {
        "checked_by_launcher": True,
        "identical": True,
        "runtime_file_count": cell_a["manifest_provenance"]["runtime_file_count"],
    }

    # The two arms must have TRAINED against the same bytes too, which is what the single shared
    # training receipt is for.  Two receipts with two digests would mean two programs.
    sha_a = trainings[name_a]["training_receipt"]["manifest_sha256"]
    sha_b = trainings[name_b]["training_receipt"]["manifest_sha256"]
    require(
        sha_a == sha_b and len(sha_a) == 64,
        f"the arms trained under different source receipts ({sha_a!r} vs {sha_b!r}); a shared "
        "receipt is what makes 'the same program with one variable changed' checkable",
    )
    require(
        trainings[name_a]["terminal_checkpoint_sha256"]
        != trainings[name_b]["terminal_checkpoint_sha256"],
        "both arms produced the identical terminal checkpoint; the manipulated variable cannot "
        "have reached training",
    )
    shared_training_receipt = {
        "checked_by_launcher": True,
        "manifest_sha256": sha_a,
        "runtime_file_count": trainings[name_a]["training_receipt"]["runtime_file_count"],
        "git_commit": trainings[name_a]["training_receipt"]["git_commit"],
    }

    contract_a = cell_a["v2_evaluation_contract"]
    contract_b = cell_b["v2_evaluation_contract"]
    require(
        set(contract_a) == set(contract_b),
        "the two arms' v2 evaluation contracts have different key sets: "
        f"{sorted(set(contract_a) ^ set(contract_b))}",
    )
    differing = sorted(key for key in contract_a if contract_a[key] != contract_b[key])
    require(
        differing == ["target_camera_max_range_m"],
        f"the arms differ in {differing}, but target_camera_max_range_m is the only authorised "
        "difference in the evaluation contract (prereg section 4)",
    )
    held_fixed = {}
    for key in (
        "detector_min_pixels",
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
        int(held_fixed["detector_min_pixels"]) == DETECTOR_MIN_PIXELS,
        "both arms must run the honest-sensor detection threshold "
        f"{DETECTOR_MIN_PIXELS}; got {held_fixed['detector_min_pixels']!r}",
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
            f"{key} must be 0 in both arms; got {held_fixed[key]!r}",
        )
    single_axis = {
        "checked_by_launcher": True,
        "evaluation_contract_differences": differing,
        "authorised_differences": ["target_camera_max_range_m"],
        "detect_resolution_not_recorded_by_evaluator": True,
    }

    return {
        "trainings": trainings,
        "cells": cells,
        "order": (name_a, name_b),
        "void_arms": [],
        "held_fixed": held_fixed,
        "runtime_map_identity": runtime_map_identity,
        "shared_training_receipt": shared_training_receipt,
        "single_axis": single_axis,
    }


# ----------------------------------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------------------------------


def arm_measurements(cell: dict) -> dict:
    """Prereg section 5 measurands for one arm.

    never-acquired is READ, not invented: it is the same evaluator-emitted per-outcome
    first-acquisition telemetry the sensor fidelity launcher documents and uses.  Pooling is a sum
    of counts across the capture/crash/timeout cohorts, which partition the cell; the accounting
    assertion is what makes the pool a rate over the whole cell.
    """
    result = cell["result"]
    rows = (result.get("target_motion") or {}).get("first_acquisition") or {}
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
        f"{cell['arm']}: never_acquired + acquired != episodes "
        f"({never} + {acquired} != {episodes})",
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
                "visible_hidden_transitions_mean_per_episode": rows[label][
                    "visible_hidden_transitions_mean_per_episode"
                ],
            }
            for label in FIRST_ACQUISITION_OUTCOMES
        },
        "target_hidden_fraction": float(result["action"]["context"]["target_hidden"]["fraction"]),
        # Reported RAW and excluded from the verdict by construction -- see classify_verdict(),
        # which takes a single delta and cannot see these at all (prereg section 5).
        "outcome_raw": raw,
    }


def classify_verdict(delta_pp: float) -> str:
    """Prereg section 5 Gate S, applied to the never-acquired delta in PERCENTAGE POINTS.

    The delta is `clip28` minus `clip20`: opening the clip is expected to LOWER never-acquired, so
    a large NEGATIVE delta is the confirmatory outcome.  Anything else is
    RANGE_INCONCLUSIVE_AT_THIS_BUDGET, which -- per prereg section 3 -- means undecided at 1,000
    warm-start epochs, NOT "no effect".

    This function takes exactly one number.  The outcome rates are not parameters, are not read
    from any global, and therefore cannot enter the verdict -- which is the preregistration's
    requirement, expressed as a signature rather than as a promise.
    """
    if delta_pp <= NEVER_ACQUIRED_HELPS_THRESHOLD_PP:
        return VERDICT_HELPS
    return VERDICT_INCONCLUSIVE


# ----------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------


def gate_table(verified: dict) -> dict:
    """Build the gate table from EVIDENCE, not from a list of names.

    Every gate here is owned by this launcher, so each one is marked passed only when the verify
    step that owns it produced its evidence dictionary; a gate whose evidence is missing is a check
    nobody performed and fails closed rather than being tallied as a pass.
    """
    gates = {}
    for gate, evidence_key in sorted(TRAINING_GATES.items()):
        per_arm = {}
        for arm, _ in ARMS:
            evidence = verified["trainings"][arm].get(evidence_key)
            require(
                isinstance(evidence, dict)
                and evidence.get("checked_by_launcher") is True
                and isinstance(evidence.get("passed"), bool),
                f"{arm}: quality gate {gate} is owned by this launcher, but verify_training() "
                f"produced no {evidence_key} evidence carrying a boolean verdict",
            )
            per_arm[arm] = evidence["passed"]
        gates[gate] = {
            "owner": PRODUCER,
            "scope": "per_arm_training",
            "passed": all(per_arm.values()),
            "arms": per_arm,
        }
    # An arm that failed Gate 0 was never evaluated, so there are no evaluation gates to judge.
    # Emitting them as "failed" would claim checks were run and lost; omitting them says what is
    # true -- the experiment stopped at Gate 0 (prereg section 5).
    if verified["void_arms"]:
        return gates
    for gate, evidence_key in sorted(PER_ARM_GATES.items()):
        per_arm = {}
        for arm, _ in ARMS:
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
    for arm, _ in ARMS:
        require_import_origin_evidence(arm, verified["cells"][arm].get("import_origin"))
    return gates


def gate_tally(payload: dict) -> tuple:
    """(evaluated, failed) for the summary line -- with every gate's ownership proved."""
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
    void_arms = list(verified["void_arms"])
    control_name, treatment_name = (CONTROL_ARM, TREATMENT_ARM)
    require(
        tuple(verified["order"]) == (CONTROL_ARM, TREATMENT_ARM),
        f"arm order drifted: {tuple(verified['order'])} is not "
        f"{(CONTROL_ARM, TREATMENT_ARM)}; the delta sign depends on it",
    )

    # Gate 0 first (prereg section 5).  If any owned gate failed, nothing may be claimed about the
    # detection range and the measurements key must be null.  That is a BICONDITIONAL and it is
    # enforced rather than described: a VOID verdict carrying measurements would publish numbers
    # the preregistration says are not to be interpreted, and a null payload under any other
    # verdict would publish a verdict with nothing behind it.
    if failed_gates or void_arms:
        verdict = VERDICT_VOID
        published = None
        basis = None
        published_delta = None
        measurements = {}
    else:
        measurements = {arm: arm_measurements(verified["cells"][arm]) for arm, _ in ARMS}
        delta_pp = (
            measurements[treatment_name]["never_acquired_rate_pp"]
            - measurements[control_name]["never_acquired_rate_pp"]
        )
        verdict = classify_verdict(delta_pp)
        published = measurements
        published_delta = delta_pp
        basis = {
            "metric": "pooled never-acquired rate over all outcomes",
            "source_field": NEVER_ACQUIRED_SOURCE,
            "control_arm": control_name,
            "treatment_arm": treatment_name,
            "control_never_acquired_pp": measurements[control_name]["never_acquired_rate_pp"],
            "treatment_never_acquired_pp": measurements[treatment_name]["never_acquired_rate_pp"],
            "delta_pp": delta_pp,
            "direction": "clip28 minus clip20; negative means the open clip acquires more often",
            "outcome_rates_excluded_from_verdict": True,
        }
    require(
        (verdict == VERDICT_VOID) == (published is None),
        f"fail-closed contract violated: verdict={verdict} with measurements="
        + ("null" if published is None else "present")
        + f"; {VERDICT_VOID} requires a null arms payload and a null payload requires "
        f"{VERDICT_VOID}",
    )

    arms_payload = None
    if published is not None:
        arms_payload = {}
        for arm, clip in ARMS:
            training = verified["trainings"][arm]
            arms_payload[arm] = {
                "condition": {
                    "detector_max_range_m": clip,
                    "detect_width": DETECT_WIDTH,
                    "detect_height": DETECT_HEIGHT,
                    "detector_min_pixels": DETECTOR_MIN_PIXELS,
                    "camera_width": CAMERA_WIDTH,
                    "camera_height": CAMERA_HEIGHT,
                    "actual_episodes": verified["cells"][arm]["actual_episodes"],
                    "num_envs": verified["cells"][arm]["condition"].get("num_envs"),
                    "episode_len_steps": verified["cells"][arm]["condition"].get(
                        "episode_len_steps"
                    ),
                },
                "training": {
                    "adopted": bool(training["record"].get("adopted")),
                    "detector_max_range_evidence": training["record"].get(
                        "detector_max_range_evidence"
                    ),
                    "run_root": training["run_root"],
                    "terminal_checkpoint": str(training["terminal_checkpoint"]),
                    "terminal_checkpoint_sha256": training["terminal_checkpoint_sha256"],
                    "terminal_epoch": training["budget"]["terminal_epoch"],
                    "terminal_frame": training["budget"]["terminal_frame"],
                    "adaptation_epochs": training["budget"]["adaptation_epochs"],
                    "adaptation_samples": training["budget"]["adaptation_samples"],
                    "exit_reason": training["exit"]["exit_reason"],
                    "ppo_rollback_total": training["rollback"]["rollback_total"],
                    "ppo_rollback_streak": training["rollback"]["rollback_streak"],
                    "ppo_rollback_log_lines": training["rollback"]["rollback_log_lines"],
                    "kl_skipped_log_lines": training["rollback"]["kl_skipped_log_lines"],
                    "training_source_manifest_sha256": training["training_receipt"][
                        "manifest_sha256"
                    ],
                    "reward_at_end": training["reward_at_end"],
                    "last_captured_rate": training["last_captured_rate"],
                },
                "measurements": published[arm],
            }

    return {
        "schema_version": 1,
        "producer": PRODUCER,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": SCOPE,
        "stage": 1,
        "decision_authority": "none",
        "p2_verdict_changed": False,
        "d1_verdict_changed": False,
        "p3_unlocked": False,
        # Prereg section 5: stage 2 runs only on RANGE_HELPS.  Recorded as a fact of this summary
        # so the authorisation cannot be inferred from prose later.
        "stage2_authorised": bool(published is not None and verdict == VERDICT_HELPS),
        "preregistration": PREREGISTRATION,
        "warm_start_checkpoint": CHECKPOINT_REL,
        "warm_start_checkpoint_sha256": CHECKPOINT_SHA,
        "shared_condition": {
            "train_seed": TRAIN_SEED,
            "eval_seed": EVAL_SEED,
            "bars": BARS,
            "requested_episodes_per_arm": EPISODES,
            "adaptation_epochs_per_arm": ADAPT_EPOCHS,
            "adaptation_samples_per_arm": ADAPT_SAMPLES,
            "warm_start_epoch": WARM_START_EPOCH,
            "terminal_epoch": TERMINAL_EPOCH,
            "num_envs": NUM_ENVS,
            "robot_name": ROBOT_NAME,
            "action_selection": ACTION_SELECTION,
            "reflection_mode": REFLECTION_MODE,
            "speed_governor_mode": SPEED_GOVERNOR_MODE,
            "goal_dist_min_m": GOAL_DIST_MIN_M,
            "goal_dist_max_m": GOAL_DIST_MAX_M,
            "detect_width": DETECT_WIDTH,
            "detect_height": DETECT_HEIGHT,
            "detector_min_pixels": DETECTOR_MIN_PIXELS,
            "camera_width": CAMERA_WIDTH,
            "camera_height": CAMERA_HEIGHT,
            "learning_rate": LEARNING_RATE,
            "manipulated_axis": "NAVRL_DETECTOR_MAX_RANGE",
        },
        # No arm carries one, in either half.  Recorded per arm because "nobody forced" is a claim
        # about each run, not a single global flag.
        "provenance_override": {
            arm: {
                "used": False,
                "reason": "each arm is evaluated at the clip it trained at, and the evaluator's v2 "
                "provenance gate matches the checkpoint detector geometry to that arm",
            }
            for arm, _ in ARMS
        },
        "arms": arms_payload,
        "void_arms": void_arms,
        "primary_metric": "pooled_never_acquired_rate",
        "never_acquired_delta_pp": published_delta,
        "threshold_pp": {"range_helps_at_or_below": NEVER_ACQUIRED_HELPS_THRESHOLD_PP},
        "verdict": verdict,
        "verdict_basis": basis,
        "quality_gates": gates,
        "failed_gates": failed_gates,
        "held_fixed": verified["held_fixed"],
        "import_origin": {
            arm: verified["cells"][arm]["import_origin"]
            for arm, _ in ARMS
            if arm in verified["cells"]
        },
        "limitations": list(LIMITATIONS),
        "sources": {
            arm: {
                "training_record": str(train_paths(arm)["record"].relative_to(ROOT)),
                "training_run_summary": str(train_paths(arm)["run_summary"].relative_to(ROOT)),
                "training_log": str(train_paths(arm)["log"].relative_to(ROOT)),
                "evaluation_result": str(cell_paths(arm)["result"].relative_to(ROOT)),
                "evaluation_receipt": str(cell_paths(arm)["receipt"].relative_to(ROOT)),
                "evaluation_log": str(cell_paths(arm)["log"].relative_to(ROOT)),
                "launcher_log": str(cell_paths(arm)["stdout_log"].relative_to(ROOT)),
            }
            for arm, _ in ARMS
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
        "# 검출 거리 1단계 — 스크리닝 (학습 seed 457 / 평가 seed 461, 70 bars)",
        "",
        f"**판정: `{payload['verdict']}`**",
        "",
    ]
    if arms:
        lines.extend([
            "| arm | 클립 | never-acquired | capture | crash | timeout | target_hidden |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for arm, clip in ARMS:
            m = arms[arm]["measurements"]
            raw = m["outcome_raw"]
            lines.append(
                f"| {arm} | {clip:.1f} m | {_pct(m['never_acquired_rate_pp'])} | "
                f"{_pct(raw['capture_rate'] * 100)} | {_pct(raw['crash_rate'] * 100)} | "
                f"{_pct(raw['timeout_rate'] * 100)} | "
                f"{_pct(m['target_hidden_fraction'] * 100)} |"
            )
        delta = payload["never_acquired_delta_pp"]
        lines.extend([
            "",
            f"**never-acquired 차이 (clip28 − clip20): {delta:+.2f} pp** "
            f"(RANGE_HELPS 임계 {NEVER_ACQUIRED_HELPS_THRESHOLD_PP:+.2f} pp 이하)",
            "",
            "| arm | outcome | 최초획득 중앙값 | never-acq |",
            "|---|---|---:|---:|",
        ])
        for arm, _ in ARMS:
            for label in FIRST_ACQUISITION_OUTCOMES:
                row = arms[arm]["measurements"]["first_acquisition_by_outcome"][label]
                rate = row["never_acquired_rate"]
                lines.append(
                    f"| {arm} | {label} | {_step(row['first_visible_step_median'])} | "
                    f"{'—' if rate is None else f'{rate * 100:.2f}%'} |"
                )
        lines.extend([
            "",
            "## 게이트 0 — 학습 건전성",
            "",
            "| arm | 종단 epoch | frame | exit | rollback | KL skip 로그 | 종단 SHA |",
            "|---|---:|---:|---|---:|---:|---|",
        ])
        for arm, _ in ARMS:
            t = arms[arm]["training"]
            lines.append(
                f"| {arm} | {t['terminal_epoch']} | {t['terminal_frame']:,} | "
                f"`{t['exit_reason']}` | {t['ppo_rollback_total']} | "
                f"{t['kl_skipped_log_lines']} | `{t['terminal_checkpoint_sha256'][:16]}…` |"
            )
    else:
        lines.append(
            "게이트 0(학습 건전성)이 실패했으므로 측정값을 게재하지 않는다 (사전등록 §5). "
            f"**VOID arm: {', '.join(payload.get('void_arms') or []) or '—'}** "
            f"실패 게이트: {', '.join(payload.get('failed_gates') or []) or '—'}. "
            "VOID arm은 평가되지 않으므로 평가 게이트는 판정되지 않았고, 판정되지 않은 것을 "
            "'실패'로 적지 않는다."
        )

    lines.extend([
        "",
        "## 이 실험이 답하는 것과 답하지 않는 것",
        "",
        "1,000 epoch warm-start 적응은 **\"이 클립에서 도달 가능한 최선\"을 답하지 못한다**"
        "(사전등록 §3). 양 arm이 20 m 세계에 맞춰진 같은 정책에서 출발하므로 **clip28만 뭔가를",
        "잊어야 하고, 설계가 clip28에 불리하다.** 따라서 양성이면 불리함을 뚫은 것이라 신뢰할 수",
        "있고, **음성이면 \"이 예산에서 미결\"이지 \"효과 없음\"이 아니다.**",
        "",
        "**capture/crash/timeout은 원값으로만 보고하며 판정에 쓰지 않는다** — 서로 다른 센서",
        "정의에서 측정된 값이다. 판정 함수 `classify_verdict()`는 never-acquired 차이 하나만",
        "인자로 받으므로 구조적으로 이 값들을 볼 수 없다.",
        "",
        "## 고정된 조건",
        "",
        f"- 양 arm 동일: detect {DETECT_WIDTH}×{DETECT_HEIGHT}, `min_pixels={DETECTOR_MIN_PIXELS}`,"
        f" RGB {CAMERA_WIDTH}×{CAMERA_HEIGHT}, {BARS} bars, 목표 {GOAL_DIST_MIN_M}–"
        f"{GOAL_DIST_MAX_M} m, appearance·지연·거리오차 0, governor off,"
        " `NAVRL_REFLECTION_COEF`/`NAVRL_LATERAL_BIAS_COEF` 미설정(=0).",
        "- 학습 환경 차이는 실제로 측정된다: 정규 학습 런처가 만들어내는 환경을 양 arm에서 덤프해"
        " 대칭차를 계산하며, 허용되는 차이는 `NAVRL_DETECTOR_MAX_RANGE`와 run 태그·로그 경로뿐이다.",
        "- 양 arm은 **하나의 학습 소스 영수증**(schema 1)을 공유하고, 두 번째 arm 시작 전에 원본"
        " 바이트를 재해싱해 검증한다.",
        "",
        "## provenance override",
        "",
        "- **어느 arm도 override를 쓰지 않는다.** 각 arm은 **자기가 학습한 클립에서** 평가되고,"
        " 체크포인트의 `cfg_detector_max_range/cfg_detect_width/cfg_detect_height`를 평가기의"
        " 요청값과 비교한다. `cfg_detector_min_pixels`도 학습·평가 모두 50이다. 실행 시점에"
        " force 없는 preflight가 통과함을 요구하며,"
        " 거부되면 담요식 force 대신 **중단**한다.",
        "- arm 구분자는 체크포인트와 평가 계약 양쪽에 독립적으로 남는다:"
        " `env_state.cfg_detector_max_range`와 `v2_evaluation_contract.target_camera_max_range_m`.",
        "",
        f"- warm start SHA-256 `{payload['warm_start_checkpoint_sha256'][:16]}…`",
        f"- 품질 게이트: 판정 {len(evaluated)}개, 실패 {len(failed)}개",
        "",
        "## 권한",
        "",
        "이 실험은 **결정 권한이 없다**. P2 STRICT FAIL·D1 FAIL을 소급 변경하지 않으며 P3를",
        "해제하지 않는다. `RANGE_HELPS`일 때에만 2단계를 실행할 자격이 생기며, 그것도 정책 채택",
        f"권한은 아니다 (`stage2_authorised: {str(payload['stage2_authorised']).lower()}`).",
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


def _arm_argument(argv, *, required: bool):
    names = [arm for arm, _ in ARMS]
    if len(argv) == 3:
        require(argv[2] in names, f"unknown arm {argv[2]!r}; expected one of {names}")
        return argv[2]
    require(not required and len(argv) == 2, f"usage: {PRODUCER} {argv[1]} {{{'|'.join(names)}}}")
    return None


def run_preflight() -> int:
    """Everything cheap first, then the one measurement that needs the GPU.

    The runtime-clean gate is CHECKED and REPORTED before the smoke and ENFORCED at the end.  That
    ordering is deliberate: an operator who is one commit away from being able to train still
    wants tonight's VRAM answer, and the smoke writes no receipt-bearing artifact, so it is not
    subject to the clean-source contract that `train` refuses on outright.
    """
    require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
    override = verify_prerequisites()
    print("[detrange] PREFLIGHT")
    print(f"[detrange]   warm start: {CHECKPOINT_REL}")
    print(f"[detrange]   warm start sha256: {CHECKPOINT_SHA}")
    print("[detrange]   decoupling knobs + fail-closed guard: PRESENT")
    print(f"[detrange]   evaluator override required: {override['override_required']}")

    dirty = runtime_dirty_paths()
    if dirty:
        print(f"[detrange]   runtime clean: FAIL ({len(dirty)} path(s))")
        for line in dirty[:8]:
            print(f"[detrange]     {line}")
    else:
        print("[detrange]   runtime clean: PASS")

    training_diff = training_env_diff()
    evaluation_diff = evaluation_env_diff()
    print("[detrange]   training env diff (measured from the trainer's own environment):")
    for key, value in training_diff.items():
        print(f"[detrange]     {key}: {value}")
    print("[detrange]   evaluation env diff:")
    for key, value in evaluation_diff.items():
        print(f"[detrange]     {key}: {value}")

    smoke = run_vram_smoke()
    print(
        "[detrange]   VRAM smoke: peak {peak} MiB / {total} MiB "
        "(headroom {head} MiB), {sec} s/epoch, projected {hours} h/arm".format(
            peak=smoke["peak_gpu_memory_mib"],
            total=smoke["gpu_total_mib"],
            head=smoke["gpu_headroom_mib"],
            sec="n/a" if smoke["seconds_per_epoch"] is None
            else f"{smoke['seconds_per_epoch']:.2f}",
            hours="n/a" if smoke["projected_arm_hours"] is None
            else f"{smoke['projected_arm_hours']:.2f}",
        )
    )
    require(
        not dirty,
        "runtime source is dirty; `train` refuses to create a clean-contract receipt from it. "
        f"Commit these first: {dirty[:8]}",
    )
    print(
        f"[detrange] PREFLIGHT PASS | train_seed={TRAIN_SEED} eval_seed={EVAL_SEED} bars={BARS} "
        f"| {ADAPT_EPOCHS} epochs/{ADAPT_SAMPLES:,} samples per arm | {EPISODES} episodes/arm "
        "| no provenance override in either arm"
    )
    return 0


def main() -> int:
    argv = sys.argv
    mode = argv[1] if len(argv) >= 2 else ""
    require(
        mode in {"preflight", "train", "evaluate", "finalize", "verify"},
        f"usage: {PRODUCER} {{preflight|train|evaluate|finalize|verify}}",
    )

    if mode == "preflight":
        require(len(argv) == 2, f"usage: {PRODUCER} preflight")
        return run_preflight()

    if mode == "train":
        arm = _arm_argument(argv, required=True)
        train_arm(arm)
        return 0

    if mode == "evaluate":
        arm = _arm_argument(argv, required=False)
        targets = [arm] if arm else [name for name, _ in ARMS]
        for name in targets:
            if cell_dir(name).exists():
                print(f"[detrange] arm {name}: cell already exists, skipping")
                continue
            evaluate_arm(name)
        pending = [name for name, _ in ARMS if not cell_dir(name).exists()]
        if pending:
            # A per-arm invocation is the normal way to run this: the arms are 2,049 episodes each
            # and may be days apart.  Summarising now would fail on the arm that does not exist yet
            # and would report that as an error instead of as "not finished".
            print(f"[detrange] EVALUATE COMPLETE {targets} | still pending: {pending}")
            return 0
        verified = verify_all()
        payload = build_summary(verified)
        print(f"[detrange] EVALUATE COMPLETE | verdict={payload['verdict']} | next: finalize")
        return 0

    require(len(argv) == 2, f"usage: {PRODUCER} {mode}")
    verified = verify_all()
    expected = build_summary(verified)
    if mode == "finalize":
        write_summary(expected)
        print(f"[detrange] FINALIZE PASS | {expected['verdict']} -> {SUMMARY_JSON}")
        return 0

    require(SUMMARY_JSON.is_file(), f"summary missing: {SUMMARY_JSON}")
    recorded = load_json(SUMMARY_JSON)
    for key in SUMMARY_VERIFY_KEYS:
        require(recorded.get(key) == expected.get(key), f"summary changed: {key}")
    print(f"[detrange] VERIFY PASS | {recorded['verdict']}")
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
        print(f"[detrange] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
