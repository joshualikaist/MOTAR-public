#!/usr/bin/env python3
"""Offline real-frame reflection audit for the frozen NavRL ref5in policy (N1).

Preregistration: ``docs/prereg_2026-08-21_n1_real_frame_reflection_audit.md`` (frozen).
Every threshold, statistic and context split below is fixed by that document.

What this tool is
-----------------
An OFFLINE evaluator.  It loads (a) a ``.npz`` of real 898-D actor observations dumped during a
standard evaluation rollout and (b) the frozen checkpoint, then runs deterministic forward passes
on each observation and on its exact left-right reflection.  It NEVER creates a simulator, never
imports Isaac Gym, never steps an environment and never feeds a probe action anywhere.  The
temporal separation between rollout and this program is itself the proof of the "side-forward
only" condition (prereg section 4).

Numerical fidelity to the live player
-------------------------------------
``aerial_gym/rl_training/rl_games/navrl_players.py`` (``NavRLPpoPlayerContinuous.get_action``)
does, for the reflection diagnostic:

    original_obs = self._preproc_obs(obs)
    mirrored_obs = self._preproc_obs(mirror_navrl_structured_observation(obs))   # MIRROR FIRST
    ... self.model(...) ...                       # running_mean_std lives INSIDE the model
    action = self._model_action(res_dict, True)   # deterministic_actions, else mus

This tool instantiates the REAL ``NavRLPpoPlayerContinuous`` and calls the REAL ``.restore()``.
No network is hand-rolled and no normaliser is recomputed.  A vecenv is avoided by injecting
``env_info`` into the player config (``rl_games.common.player.BasePlayer.__init__`` only builds an
environment when ``config['env_info']`` is absent), so ``.run()`` is never called and no simulator
exists at any point.

Order matters and is preserved: the observation is mirrored in RAW units and only then handed to
the model, whose ``norm_obs`` applies the checkpoint's ``running_mean_std``.  That order is
physically correct -- a reflected world produces raw observations that pass through the same fixed
normaliser -- and is deliberately NOT commuted (prereg L2).

``deterministic_actions`` (post-tanh, in [-1, 1]) is preferred over ``mus``; for this squashed
Gaussian policy ``mus`` is the PRE-tanh latent mean and is the wrong quantity.  Actions are read
BEFORE any rescale/clamp to ``actions_low``/``actions_high``, i.e. exactly the units of
``navrl_players._record_reflection_pair``.

Usage
-----
    PYTHONNOUSERSITE=1 python tools/navrl_reflection_offline_audit.py \
        --frames results/.../obs_dump.npz \
        --checkpoint <...>/last_gen_ppo_ep_1900_rew_182.11377.pth \
        --checkpoint-sha256 197ea269... \
        --out results/.../reflection_offline_audit.json

Exit codes: 0 = audit completed (any verdict), 2 = precondition failure (bad SHA, bad schema,
unsupported checkpoint), 3 = fail-closed: either a preregistered quality gate failed or a
data-integrity invariant (dtype/domain, finiteness, context partition, outcome-join
reconciliation, npz<->checkpoint identity) was violated.  In both cases no policy statistics and
no policy claim are reported.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np


# --------------------------------------------------------------------------------------------
# PREREGISTERED CONSTANTS.  Declared above every measurement so that no measurement code can
# introduce a second, drifting copy of a threshold.  Prereg sections 5, 6 and 7.
# --------------------------------------------------------------------------------------------

SCHEMA_VERSION = 1
PRODUCER = "tools/navrl_reflection_offline_audit.py"

# Prereg section 7 -- verdict thresholds, applied to the OVERALL cell only.
VERDICT_CONFIRM_MEDIAN_MIN = 0.30
VERDICT_CONFIRM_SIGN_AGREEMENT_MAX = 0.60
VERDICT_ABSENT_MEDIAN_MAX = 0.10
VERDICT_ABSENT_SIGN_AGREEMENT_MIN = 0.90

VERDICT_CHIRALITY_CONFIRMED = "CHIRALITY_CONFIRMED_REAL_FRAME"
VERDICT_CHIRALITY_ABSENT = "CHIRALITY_ABSENT"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE_REAL_FRAME"
VERDICT_FAIL_CLOSED = "FAIL_CLOSED_TRANSFORM_QUALITY"
# Fail-closed outcome of a data-integrity invariant (dtype/domain, finiteness, partition,
# reconciliation, identity binding).  Distinct from the transform-quality failure above so a
# reader can tell WHICH kind of fail-closed happened.
VERDICT_FAIL_CLOSED_INTEGRITY = "FAIL_CLOSED_DATA_INTEGRITY"
# Prereg section 6 says a cell with fewer than MIN_CONTEXT_COMPARABLE_ROWS comparable rows gets NO
# verdict.  When that cell is the OVERALL cell the run therefore has no verdict at all -- which is
# NOT the same statement as INCONCLUSIVE_REAL_FRAME (a verdict the prereg reserves for a cell that
# was measured with enough sample and matched neither region).
VERDICT_NO_VERDICT_INSUFFICIENT_SAMPLE = "NO_VERDICT_INSUFFICIENT_SAMPLE"

# Prereg section 6 -- "meaningful sign" floor, identical to navrl_players._record_reflection_pair.
LATERAL_SIGN_THRESHOLD = 0.05
# Prereg section 6 -- a context cell needs this many comparable rows to receive a verdict.
MIN_CONTEXT_COMPARABLE_ROWS = 256
# Prereg section 5 (Q9) -- minimum valid frames.
MIN_VALID_FRAMES = 4096

# Prereg section 5 -- quality-gate thresholds.
GATE_INVOLUTION_MAX_ABS = 0.0
GATE_ISOMETRY_MAX_ABS = 1.0e-3
GATE_STRUCTURED_OBS_DIM = 898
GATE_SCAN_FIXED_POINTS = frozenset((0, 36))

# Prereg section 5 -- structured schema block layout (the field widths are fixed by
# navrl_perception.py; the beam/token counts come from the checkpoint metadata).
OBSTACLE_HISTORY = 5
OBSTACLE_DIM = 12
OBSTACLE_SIGN_FLIP_FIELDS = (1, 4)
ROBOT_HISTORY = 5
ROBOT_DIM = 10
ROBOT_SIGN_FLIP_FIELDS = (1, 3, 5, 7)
TARGET_HISTORY = 5
TARGET_DIM = 16
TARGET_SIGN_FLIP_FIELDS = (1, 4)

# Prereg section 6 -- reported order statistics and the percentile convention.
PERCENTILE_QUANTILES = (90.0, 95.0, 99.0)
PERCENTILE_METHOD = "linear"
PERCENTILE_CONVENTION = (
    "numpy.percentile(values, q, method='linear') on float64; median is q=50 under the same "
    "convention"
)

# Prereg section 4 -- outcome attribution codes emitted by the NAVRL_OBS_DUMP hook.  Literal on
# purpose: a computed map could silently re-label a stratum.
OUTCOME_CODES = {
    0: "capture",
    1: "crash_bar_contact",
    2: "crash_oob",
    3: "crash_other",
    4: "timeout",
    5: "unattributed",
}
# Prereg section 6 -- the five preregistered outcome context cells (5 = unattributed is NOT a
# context; its frame count is reported as diagnostics only).
OUTCOME_CONTEXT_CODES = (0, 1, 2, 3, 4)

# The OUTCOME-JOIN SENTINEL.  Deliberately NOT called "unattributed": OUTCOME_CODES[5] already owns
# that name and means something else entirely.
#   OUTCOME_CODES[5] == "unattributed": the episode FINISHED and has a row in the outcome table,
#                                       but the dump could not attribute a cause to it.
#   NO_OUTCOME_ROW   == -1            : the frame's episode has NO ROW in the outcome table at all
#                                       (it was still running when the table closed).
# The two used to be reported three lines apart under the same word, so a reader (or an aggregator
# summing outcome_frame_counts) could conclude the split was complete when 9.9% of the frames were
# outside it.  Every surface that reports the sentinel uses NO_OUTCOME_ROW_NAME.
NO_OUTCOME_ROW = -1
NO_OUTCOME_ROW_NAME = "no_outcome_row"
NO_OUTCOME_ROW_MEANING = (
    "the frame's episode has no row in the outcome table (it was still running when the table "
    "closed); NOT the same as outcome code 5 'unattributed', which is a finished episode the dump "
    "could not attribute"
)

# Prereg section 6 -- the context FAMILIES and the population each one must partition.  A family
# whose cells do not sum to its population is a silent data loss (an out-of-domain label falls out
# of every cell of the family with no gate firing), so the sums are asserted, not assumed.
CONTEXT_FAMILY_FRONT = ("front_blocked", "front_clear", "front_unknown")
CONTEXT_FAMILY_TARGET = ("target_visible", "target_hidden")
CONTEXT_FAMILY_OUTCOME = tuple(
    "outcome_" + OUTCOME_CODES[code] for code in OUTCOME_CONTEXT_CODES
)
CONTEXT_OVERALL_CELL = "overall"

# Admissible domains of the dump's context columns.  Asserted rather than coerced: `.astype(bool)`
# on an int8 column would silently turn a future -1 ("unknown") into True (= VISIBLE).
CTX_TARGET_VISIBLE_DOMAIN = (0, 1)
CTX_FRONT_BLOCKED_DOMAIN = (-1, 0, 1)
CTX_VALID_DOMAIN = (0, 1)

# The dtype this tool assumes for the dumped observation.  rl_games' BasePlayer._preproc_obs
# divides uint8 observations by 255; this tool builds a float32 tensor from the npz and would
# silently skip that path, evaluating a differently scaled observation than the live player.
FRAMES_OBS_DTYPE = "float32"

# NON-GATING diagnostic (H6): the fraction of excluded frames that must sit at or above the median
# call_index of the population before the exclusion is flagged as tail-concentrated (i.e. not
# missing at random).  Used for a generated caveat only; it feeds no verdict and no gate.
EXCLUDED_TAIL_CONCENTRATION_MIN = 0.9

# Action channel indices (prereg section 6): x, lateral, z, yaw.
ACTION_CHANNELS = (("conj_err_x", 0), ("conj_err_lat", 1), ("conj_err_z", 2), ("conj_err_yaw", 3))
LATERAL_ACTION_INDEX = 1

DEFAULT_BATCH_SIZE = 1024
CHECKPOINT_ENV_KEYS = (
    "cfg_lidar_hbeams",
    "cfg_lidar_vbeams",
    "cfg_max_obstacles",
    "cfg_corridor_tokens",
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CONFIG = ROOT / "aerial_gym/rl_training/rl_games/ppo_navrl_perception_transformer.yaml"


class AuditPreconditionError(RuntimeError):
    """A precondition that makes the audit meaningless (bad SHA, unsupported schema)."""


class GateFailure(RuntimeError):
    """A fail-closed failure (quality gate or integrity invariant); no policy claim may be made."""

    def __init__(self, gates, payload, kind="quality gates"):
        RuntimeError.__init__(self, "%s failed: %s" % (kind, ", ".join(gates)))
        self.gates = list(gates)
        self.payload = payload
        self.kind = kind


class IntegrityViolation(RuntimeError):
    """A fail-closed data-integrity invariant was violated (dtype, domain, partition, finiteness).

    Carries the same ``detail`` dict the passing path would have reported, so the JSON shows the
    reader exactly what was checked and what was found.
    """

    def __init__(self, message, detail=None):
        RuntimeError.__init__(self, message)
        self.detail = dict(detail or {})


# --------------------------------------------------------------------------------------------
# Quality-gate bookkeeping (prereg section 5).
#
# A gate is in exactly one of THREE states, and they are counted separately:
#   evaluated + passed   -- this tool ran the check and it held;
#   evaluated + failed   -- this tool ran the check and it did not hold (fail-closed);
#   delegated            -- this tool does NOT run the check; the launcher does.
# Folding a delegated gate into the passed tally makes the headline "N evaluated, 0 failed" a
# tautology: it would read identically if the delegated check had been deleted.
# --------------------------------------------------------------------------------------------

GATE_STATUS_EVALUATED = "evaluated"
GATE_STATUS_DELEGATED = "delegated"

# Machine-readable delegation contract.  A caller that quotes this audit's numbers must assert that
# IT performed every check named here; ``caller_must_assert`` says what "performed" means.
DELEGATED_GATES = {
    "Q6_import_origin": {
        "owner": "tools/run_navrl_ref5in_reflection_audit.py",
        "check": "import_origin_enforcement",
        "caller_must_assert": (
            "the executing aerial_gym/__init__.py sha256 printed by the enforced [origin] line "
            "equals the source manifest's runtime_files entry"
        ),
    },
    "Q7_manifest_schema_version": {
        "owner": "tools/run_navrl_ref5in_reflection_audit.py",
        "check": "source_manifest_and_schema_version_2_receipt",
        "caller_must_assert": (
            "the run's source manifest and its schema_version 2 receipt were verified before this "
            "audit's numbers are quoted"
        ),
    },
}


# --------------------------------------------------------------------------------------------
# Preregistered index sets (prereg section 5, Q4/Q5)
#
# Built here from the PREREGISTRATION TEXT, independently of
# ppo_update_safety.mirror_navrl_structured_observation.  Q4 then proves the function agrees.
# --------------------------------------------------------------------------------------------


def structured_obs_dim(hbeams, vbeams, max_obstacles):
    """Structured actor observation width implied by the checkpoint's schema knobs."""
    return (
        int(vbeams) * int(hbeams)
        + OBSTACLE_HISTORY * int(max_obstacles) * OBSTACLE_DIM
        + ROBOT_HISTORY * ROBOT_DIM
        + TARGET_HISTORY * TARGET_DIM
    )


def preregistered_scan_permutation(hbeams):
    """``h -> (-h) mod H`` over one LiDAR ring (prereg section 5, Q5)."""
    return dict((h, (-h) % int(hbeams)) for h in range(int(hbeams)))


def preregistered_signed_permutation(hbeams, vbeams, max_obstacles):
    """The full preregistered mirror operator as an explicit signed permutation.

    Returns ``(source, sign)`` such that the preregistered reflection of ``x`` is

        mirrored[:, j] = sign[j] * x[:, source[j]]

    - permutation: ``[0 : vbeams*hbeams]``, per ring ``v`` the index ``v*hbeams + h`` takes its
      value from ``v*hbeams + ((-h) mod hbeams)``;
    - sign flip, obstacle block ``(hist, slot, dim)``: fields 1 and 4;
    - sign flip, robot block ``(hist, dim)``: fields 1, 3, 5, 7;
    - sign flip, target block ``(hist, dim)``: fields 1 and 4;
    - every other index: unchanged.
    """
    hbeams = int(hbeams)
    vbeams = int(vbeams)
    max_obstacles = int(max_obstacles)
    dim = structured_obs_dim(hbeams, vbeams, max_obstacles)
    source = list(range(dim))
    sign = [1] * dim

    ring_map = preregistered_scan_permutation(hbeams)
    for v in range(vbeams):
        for h in range(hbeams):
            source[v * hbeams + h] = v * hbeams + ring_map[h]

    offset = vbeams * hbeams
    for hist in range(OBSTACLE_HISTORY):
        for slot in range(max_obstacles):
            base = offset + (hist * max_obstacles + slot) * OBSTACLE_DIM
            for field in OBSTACLE_SIGN_FLIP_FIELDS:
                sign[base + field] = -1
    offset += OBSTACLE_HISTORY * max_obstacles * OBSTACLE_DIM

    for hist in range(ROBOT_HISTORY):
        base = offset + hist * ROBOT_DIM
        for field in ROBOT_SIGN_FLIP_FIELDS:
            sign[base + field] = -1
    offset += ROBOT_HISTORY * ROBOT_DIM

    for hist in range(TARGET_HISTORY):
        base = offset + hist * TARGET_DIM
        for field in TARGET_SIGN_FLIP_FIELDS:
            sign[base + field] = -1
    offset += TARGET_HISTORY * TARGET_DIM

    if offset != dim:
        raise AssertionError("preregistered block layout does not cover the observation")
    return source, sign


def preregistered_sign_flip_indices(hbeams, vbeams, max_obstacles):
    _source, sign = preregistered_signed_permutation(hbeams, vbeams, max_obstacles)
    return set(index for index, value in enumerate(sign) if value == -1)


def preregistered_permutation_pairs(hbeams, vbeams, max_obstacles):
    """``{destination: source}`` for every index the mirror actually moves (source != dest)."""
    source, _sign = preregistered_signed_permutation(hbeams, vbeams, max_obstacles)
    return dict(
        (dest, src) for dest, src in enumerate(source) if src != dest
    )


# --------------------------------------------------------------------------------------------
# Statistics helpers (prereg section 6)
# --------------------------------------------------------------------------------------------


def percentile(values, q):
    """Deterministic percentile with the documented convention (numpy linear interpolation)."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.percentile(array, float(q), method=PERCENTILE_METHOD))


def describe(values):
    """median, p90, p95, p99, mean, max, n -- exactly the reported set (prereg section 6)."""
    array = np.asarray(values, dtype=np.float64)
    result = {"n": int(array.size)}
    if array.size == 0:
        result.update(
            {"median": None, "p90": None, "p95": None, "p99": None, "mean": None, "max": None}
        )
        return result
    result["median"] = percentile(array, 50.0)
    for q in PERCENTILE_QUANTILES:
        result["p%d" % int(q)] = percentile(array, q)
    result["mean"] = float(array.mean())
    result["max"] = float(array.max())
    return result


def classify_verdict(median_conj_err_lat, sign_agreement):
    """Prereg section 7 verdict rule.  Thresholds come from module constants only."""
    if median_conj_err_lat is None or sign_agreement is None:
        return None
    if (
        median_conj_err_lat >= VERDICT_CONFIRM_MEDIAN_MIN
        and sign_agreement <= VERDICT_CONFIRM_SIGN_AGREEMENT_MAX
    ):
        return VERDICT_CHIRALITY_CONFIRMED
    if (
        median_conj_err_lat <= VERDICT_ABSENT_MEDIAN_MAX
        and sign_agreement >= VERDICT_ABSENT_SIGN_AGREEMENT_MIN
    ):
        return VERDICT_CHIRALITY_ABSENT
    return VERDICT_INCONCLUSIVE


def has_sufficient_sample(comparable_rows):
    """Prereg section 6: a cell receives a verdict only at >= 256 comparable rows."""
    return int(comparable_rows) >= MIN_CONTEXT_COMPARABLE_ROWS


def overall_verdict(overall_cell):
    """The run's verdict, or an explicit NO-VERDICT when the overall cell is under-sampled.

    ``measure_cell`` returns ``verdict: None`` on purpose below MIN_CONTEXT_COMPARABLE_ROWS
    comparable rows, because the preregistration says such a cell receives NO verdict.  Converting
    that ``None`` into INCONCLUSIVE_REAL_FRAME would assert a verdict the prereg forbids, so the
    driver reports the no-verdict outcome instead.  Returns ``(verdict, reason_or_None)``.
    """
    if overall_cell.get("verdict") is not None:
        return overall_cell["verdict"], None
    comparable = overall_cell.get("lateral_sign_comparable")
    if overall_cell.get("insufficient_sample"):
        reason = (
            "the overall cell has %s comparable rows, below the preregistered minimum of %d; "
            "prereg section 6 assigns NO verdict to such a cell"
            % (comparable, MIN_CONTEXT_COMPARABLE_ROWS)
        )
    else:
        reason = "the overall cell has no reportable statistics (median or sign agreement is null)"
    return VERDICT_NO_VERDICT_INSUFFICIENT_SAMPLE, reason


def summarise_gates(gates):
    """Split the gate block into evaluated-passed / evaluated-failed / delegated / malformed.

    ``malformed`` is counted as a FAILURE: a gate that claims neither state (no status, or a
    non-boolean ``passed`` while claiming to be evaluated) must never be silently absorbed into the
    passed tally.
    """
    evaluated_passed = []
    evaluated_failed = []
    delegated = []
    malformed = []
    for name in sorted(gates):
        info = gates[name] if isinstance(gates[name], dict) else {}
        status = info.get("status")
        if status == GATE_STATUS_DELEGATED:
            delegated.append(name)
            continue
        if status != GATE_STATUS_EVALUATED or not isinstance(info.get("passed"), bool):
            malformed.append(name)
            continue
        if info["passed"]:
            evaluated_passed.append(name)
        else:
            evaluated_failed.append(name)
    return {
        "n_gates_declared": len(gates),
        "evaluated_passed": evaluated_passed,
        "n_evaluated_passed": len(evaluated_passed),
        "evaluated_failed": evaluated_failed,
        "n_evaluated_failed": len(evaluated_failed),
        "delegated": delegated,
        "n_delegated": len(delegated),
        "malformed": malformed,
        "n_malformed": len(malformed),
        "n_evaluated_here": len(evaluated_passed) + len(evaluated_failed),
        "note": (
            "delegated gates are NOT evaluated by this tool and are NOT counted as passed; a "
            "caller quoting these numbers must assert that it performed every gate listed in "
            "delegated_gates"
        ),
    }


def stamp_gate_states(gates):
    """Give every gate exactly one state, and force the delegated ones to report no result."""
    for name, info in gates.items():
        if name in DELEGATED_GATES:
            info["status"] = GATE_STATUS_DELEGATED
            info["passed"] = None
            info["delegation"] = dict(DELEGATED_GATES[name])
            info["evaluated_here"] = False
        else:
            info["status"] = GATE_STATUS_EVALUATED
            info["evaluated_here"] = True
    return gates


# --------------------------------------------------------------------------------------------
# Filesystem / provenance helpers
# --------------------------------------------------------------------------------------------


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(path_argument):
    """Resolve the pinned checkpoint, following the shared git dir when runs/ is not local.

    ``runs/`` is gitignored and exists only in the primary worktree; identity stays pinned by the
    SHA-256 the caller passes.  Same resolution strategy as tools/run_navrl_ref5in_mode_probe.py.
    """
    candidate = Path(path_argument)
    if candidate.is_file():
        return candidate.resolve()
    if candidate.is_absolute():
        raise AuditPreconditionError("checkpoint not found: %s" % candidate)
    local = (ROOT / candidate).resolve()
    if local.is_file():
        return local
    common = Path(
        subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--git-common-dir"], universal_newlines=True
        ).strip()
    ).resolve()
    primary = (common.parent / candidate).resolve()
    if primary.is_file():
        return primary
    raise AuditPreconditionError("checkpoint not found in worktree or primary: %s" % candidate)


def install_cpu_only_aerial_gym_packages():
    """Make the CPU-only subset of ``aerial_gym`` importable WITHOUT Isaac Gym.

    ``aerial_gym/__init__.py`` does ``import isaacgym`` and then pulls in every task/env/control
    module.  This audit needs only three leaf modules (navrl_perception, navrl_transformer_network,
    navrl_action_models / navrl_players), all of which are simulator-free.  Standing in bare
    packages with the same ``__path__`` -- the pattern tests/test_navrl_ref5in_platform.py already
    uses -- keeps the real module files (and therefore the real builders) while guaranteeing that
    no simulator can be constructed.  ``aerial_gym.rl_training`` and its children are implicit
    namespace packages, so they resolve from the stand-in's ``__path__``.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    for name, relative in (
        ("aerial_gym", "aerial_gym"),
        ("aerial_gym.task", "aerial_gym/task"),
        ("aerial_gym.task.navrl_task", "aerial_gym/task/navrl_task"),
    ):
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        if not hasattr(module, "__path__"):
            module.__path__ = [str(ROOT / relative)]
    if not hasattr(sys.modules["aerial_gym"], "AERIAL_GYM_DIRECTORY"):
        sys.modules["aerial_gym"].AERIAL_GYM_DIRECTORY = str(ROOT)


def read_checkpoint_schema(checkpoint_state):
    """Prereg section 5 (Q3): schema knobs come from the CHECKPOINT, never from the environment."""
    env_state = checkpoint_state.get("env_state")
    if not isinstance(env_state, dict):
        raise AuditPreconditionError("checkpoint env_state is missing; cannot read the schema")
    missing = [key for key in CHECKPOINT_ENV_KEYS if env_state.get(key) is None]
    if missing:
        raise AuditPreconditionError(
            "checkpoint lacks schema provenance field(s): %s" % ", ".join(missing)
        )
    schema = {
        "cfg_lidar_hbeams": int(float(env_state["cfg_lidar_hbeams"])),
        "cfg_lidar_vbeams": int(float(env_state["cfg_lidar_vbeams"])),
        "cfg_max_obstacles": int(float(env_state["cfg_max_obstacles"])),
        "cfg_corridor_tokens": int(float(env_state["cfg_corridor_tokens"])),
    }
    if schema["cfg_corridor_tokens"] != 0:
        # mirror_navrl_structured_observation has no corridor-token branch; a nonzero value would
        # silently mirror only a prefix of the observation.
        raise AuditPreconditionError(
            "checkpoint uses %d corridor tokens; the canonical mirror does not support them"
            % schema["cfg_corridor_tokens"]
        )
    schema["structured_obs_dim"] = structured_obs_dim(
        schema["cfg_lidar_hbeams"], schema["cfg_lidar_vbeams"], schema["cfg_max_obstacles"]
    )
    schema["cfg_action_policy"] = str(env_state.get("cfg_action_policy", "") or "").strip()
    schema["cfg_action_std"] = str(env_state.get("cfg_action_std", "") or "").strip()
    schema["cfg_action_mu_scale"] = str(env_state.get("cfg_action_mu_scale", "") or "").strip()
    schema["cfg_truncated_dmin"] = env_state.get("cfg_truncated_dmin")
    schema["cfg_robot_name"] = str(env_state.get("cfg_robot_name", "") or "").strip()
    schema["epoch"] = int(checkpoint_state.get("epoch", -1))
    return schema


def apply_schema_environment(schema):
    """Export the checkpoint's schema so navrl_perception and the mirror agree on ONE source.

    ``mirror_navrl_structured_observation`` reads NAVRL_LIDAR_HBEAMS / NAVRL_LIDAR_VBEAMS /
    NAVRL_MAX_OBSTACLES with defaults 36/4/5 (a 574-D schema) and RAISES on a real 898-D
    observation.  navrl_perception.py freezes the same knobs into module constants AT IMPORT TIME.
    Both are therefore pinned from the checkpoint here, before any of those modules is imported.
    """
    exported = {
        "NAVRL_LIDAR_HBEAMS": str(schema["cfg_lidar_hbeams"]),
        "NAVRL_LIDAR_VBEAMS": str(schema["cfg_lidar_vbeams"]),
        "NAVRL_MAX_OBSTACLES": str(schema["cfg_max_obstacles"]),
        "NAVRL_CORRIDOR_TOKENS": str(schema["cfg_corridor_tokens"]),
    }
    # Action-distribution contract: runner._apply_action_policy_config restores exactly these from
    # the checkpoint for --play.  cfg_action_mu_scale in particular changes the deterministic
    # action (tanh(mu_scale * raw_mu)), so omitting it would evaluate a different controller.
    if schema["cfg_action_std"]:
        exported["NAVRL_ACTION_STD"] = schema["cfg_action_std"]
    if schema["cfg_action_mu_scale"]:
        exported["NAVRL_ACTION_MU_SCALE"] = schema["cfg_action_mu_scale"]
    if schema["cfg_truncated_dmin"] is not None:
        exported["NAVRL_TRUNCATED_DMIN"] = str(schema["cfg_truncated_dmin"])
    if schema["cfg_action_policy"]:
        exported["NAVRL_ACTION_POLICY"] = schema["cfg_action_policy"]
    for key, value in exported.items():
        os.environ[key] = value
    return exported


_ACTION_POLICY_MODELS = {
    "legacy": "continuous_a2c_logstd",
    "fixed_gaussian": "navrl_fixed_gaussian",
    "squashed_gaussian": "navrl_squashed_gaussian",
    "truncated_gaussian": "navrl_truncated_gaussian",
}


def build_player(checkpoint_path, schema, device_name, train_config_path):
    """Instantiate the REAL NavRLPpoPlayerContinuous and restore it -- no environment involved.

    ``BasePlayer.__init__`` builds a vecenv only when ``config['env_info']`` is absent, so the
    env_info that runner.AERIALRLGPUEnv.get_env_info() would have produced is injected directly.
    ``.run()`` is never called.
    """
    import yaml
    import torch
    from gym import spaces
    from rl_games.algos_torch import model_builder

    from aerial_gym.rl_training.rl_games.navrl_transformer_network import NavRLTransformerBuilder
    from aerial_gym.rl_training.rl_games.navrl_action_models import (
        NavRLFixedGaussianModel,
        NavRLSquashedGaussianModel,
        NavRLTruncatedGaussianModel,
    )
    from aerial_gym.rl_training.rl_games.navrl_players import NavRLPpoPlayerContinuous

    # Same performance/precision switches rl_games.torch_runner.Runner.__init__ applies on the
    # live --play path.  They change float32 matmul precision, so replicating them is part of
    # numerical fidelity, not an optimisation.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    model_builder.register_network("navrl_transformer", NavRLTransformerBuilder)
    model_builder.register_model("navrl_fixed_gaussian", NavRLFixedGaussianModel)
    model_builder.register_model("navrl_squashed_gaussian", NavRLSquashedGaussianModel)
    model_builder.register_model("navrl_truncated_gaussian", NavRLTruncatedGaussianModel)

    with open(str(train_config_path), "r") as stream:
        config = yaml.safe_load(stream)
    params = config["params"]

    policy = schema["cfg_action_policy"] or "legacy"
    if policy not in _ACTION_POLICY_MODELS:
        raise AuditPreconditionError("unsupported checkpoint action policy: %r" % policy)
    params["model"]["name"] = _ACTION_POLICY_MODELS[policy]

    obs_dim = int(schema["structured_obs_dim"])
    train_cfg = params["config"]
    train_cfg["env_info"] = {
        "action_space": spaces.Box(-np.ones(4), np.ones(4)),
        "observation_space": spaces.Box(np.ones(obs_dim) * -np.Inf, np.ones(obs_dim) * np.Inf),
        "agents": 1,
        "value_size": 1,
    }
    train_cfg["vec_env"] = None
    train_cfg["device_name"] = device_name
    player_cfg = dict(train_cfg.get("player") or {})
    player_cfg["use_vecenv"] = False
    player_cfg["deterministic"] = True
    player_cfg["print_stats"] = False
    train_cfg["player"] = player_cfg

    player = NavRLPpoPlayerContinuous(params)
    player.restore(str(checkpoint_path))
    player.model.eval()
    if player.model.training:
        raise AuditPreconditionError("restored model is not in eval mode")
    if player.env is not None:
        raise AuditPreconditionError("player unexpectedly constructed an environment")
    return player


# --------------------------------------------------------------------------------------------
# Forward pass -- numerically identical to navrl_players.get_action's diagnostic path
# --------------------------------------------------------------------------------------------


def _running_mean_std(model):
    owner = model
    while hasattr(owner, "_orig_mod"):
        owner = owner._orig_mod
    return owner.running_mean_std


def deterministic_actions(player, obs_batch):
    """One forward pass, returning the same tensor navrl_players would select.

    ``_preproc_obs`` and ``_model_action`` are the player's own methods; ``running_mean_std`` is
    applied inside ``model.forward`` via ``norm_obs``, exactly as in the live path.
    """
    import torch

    from aerial_gym.rl_training.rl_games.navrl_players import NavRLPpoPlayerContinuous

    input_dict = {
        "is_train": False,
        "prev_actions": None,
        "obs": player._preproc_obs(obs_batch),
        "rnn_states": None,
    }
    with torch.no_grad():
        result = player.model(input_dict)
    return NavRLPpoPlayerContinuous._model_action(result, True)


def forward_pairs(player, obs, batch_size, mirror_fn):
    """Deterministic actions for every observation and for its exact reflection.

    MIRROR FIRST, THEN NORMALISE: the raw observation is reflected here and the model's
    running_mean_std is applied afterwards inside the forward, matching navrl_players.py:155.
    """
    import torch

    original_chunks = []
    mirrored_chunks = []
    total = int(obs.shape[0])
    for start in range(0, total, int(batch_size)):
        chunk = obs[start : start + int(batch_size)].to(player.device)
        original_chunks.append(deterministic_actions(player, chunk).detach().cpu())
        mirrored_chunks.append(deterministic_actions(player, mirror_fn(chunk)).detach().cpu())
    if not original_chunks:
        empty = torch.zeros((0, 4), dtype=torch.float32)
        return empty, empty
    return torch.cat(original_chunks, dim=0), torch.cat(mirrored_chunks, dim=0)


# --------------------------------------------------------------------------------------------
# S1 -- exploratory, NON-GATING normaliser-symmetry decomposition (prereg section 6)
# --------------------------------------------------------------------------------------------


def symmetrise_normaliser(running_mean, running_var, source, sign):
    """Symmetrise running_mean_std so that the normaliser COMMUTES with the mirror.

    Write the mirror as a signed permutation ``(M x)_j = s_j * x_{p(j)}`` with ``p`` an involution
    and ``s_{p(j)} = s_j``.  The normaliser is ``N(x)_j = clamp((x_j - m_j) / sqrt(v_j + eps))``.
    Requiring ``N(M x) = M N(x)`` for all ``x`` gives, term by term,

        (s_j x_{p(j)} - m_j) / sqrt(v_j + eps)  ==  s_j * (x_{p(j)} - m_{p(j)}) / sqrt(v_{p(j)}+eps)

    (the clamp is an ODD function on a symmetric interval, so it commutes with the sign and
    imposes no extra condition), which holds for every ``x`` if and only if

        v_j = v_{p(j)}          and          m_j = s_j * m_{p(j)}.

    Hence:
      * VARIANCE is symmetrised by the plain pairwise average, ``v'_j = v'_{p(j)} = (v_j+v_{p(j)})/2``
        -- variance is sign-blind, so a sign-flipped field keeps its own variance and a permuted
        pair shares the average.
      * MEAN is symmetrised as ``m'_j = (m_j + s_j * m_{p(j)}) / 2``.  For an unsigned permuted
        pair (``s=+1``) this is the ordinary average.  For a sign-flipped field the mirror maps the
        index to ITSELF with ``s=-1`` (``p(j)=j``), so the formula collapses to
        ``m'_j = (m_j - m_j)/2 = 0``: the ONLY value a mean can take on a coordinate that must
        change sign under a symmetry of the distribution is zero.  Averaging the raw means there
        instead (``(m_j+m_j)/2 = m_j``) would keep the asymmetry it is supposed to remove.
        The general antisymmetric case ``p(j) != j, s=-1`` is handled by the same expression,
        ``m'_j = (m_j - m_{p(j)})/2 = -m'_{p(j)}``; it does not occur in this schema but costs
        nothing to support.

    Everything is done in float64, the dtype rl_games stores these buffers in.
    """
    import torch

    mean = running_mean.detach().clone().to(torch.float64)
    var = running_var.detach().clone().to(torch.float64)
    index = torch.as_tensor(source, dtype=torch.long, device=mean.device)
    signs = torch.as_tensor(sign, dtype=torch.float64, device=mean.device)
    sym_mean = 0.5 * (mean + signs * mean.index_select(0, index))
    sym_var = 0.5 * (var + var.index_select(0, index))
    return sym_mean, sym_var


# --------------------------------------------------------------------------------------------
# Frame table
# --------------------------------------------------------------------------------------------

# Intended NAVRL_OBS_DUMP schema.  A short, disjoint alias list absorbs harmless naming drift in
# the dump hook; the binding actually used is recorded in the output JSON.
FRAME_KEYS = {
    "obs": ("obs", "observations", "actor_obs"),
    "env_id": ("env_id", "env_ids"),
    "call_index": ("call_index",),
    "episode_uid": ("episode_uid", "frame_episode_uid"),
    "ep_step": ("ep_step", "episode_step"),
    "ctx_target_visible": ("ctx_target_visible", "target_visible"),
    "ctx_front_blocked": ("ctx_front_blocked", "front_blocked"),
    "ctx_valid": ("ctx_valid", "valid_y"),
}
EPISODE_KEYS = {
    "ep_uid": ("ep_uid", "episode_table_uid"),
    "ep_env_id": ("ep_env_id",),
    "outcome": ("outcome", "ep_outcome"),
    "ep_len": ("ep_len", "episode_length"),
}
REQUIRED_FRAME_KEYS = ("obs", "episode_uid", "ctx_target_visible", "ctx_front_blocked", "ctx_valid")
REQUIRED_EPISODE_KEYS = ("ep_uid", "outcome")

# Optional RUN IDENTITY the dump may carry.  Nothing in the original npz schema bound the frames to
# a policy, so the link rested entirely on the caller's word.  Whatever of these the dump writes is
# recorded, and a checkpoint SHA-256 is asserted against the audited checkpoint (check_npz_identity).
IDENTITY_KEYS = {
    "checkpoint_sha256": ("checkpoint_sha256", "policy_sha256", "ckpt_sha256"),
    "checkpoint_path": ("checkpoint_path", "checkpoint"),
    "run_name": ("run_name", "run_id"),
    "dump_schema_version": ("dump_schema_version", "obs_dump_schema_version"),
    "dump_producer": ("dump_producer", "producer"),
    "run_seed": ("run_seed", "seed", "eval_seed"),
    "run_bars": ("run_bars",),
    "run_max_bars": ("run_max_bars",),
    "run_num_envs": ("run_num_envs",),
    "run_num_bars_env": ("run_num_bars_env",),
    "run_pid": ("run_pid",),
    "run_obs_dump_path": ("run_obs_dump_path",),
    "obs_width_recorded": ("obs_width_recorded",),
    "obs_width_live_alloc": ("obs_width_live_alloc",),
    "obs_width_schema": ("obs_width_schema",),
    "dropped_reset_orphan_frames": ("dropped_reset_orphan_frames",),
    "dropped_reset_orphan_episodes": ("dropped_reset_orphan_episodes",),
    "robot_name": ("robot_name",),
}
# The dump may travel with its OWN code -> name map.  This tool keeps its preregistered literal
# (prereg section 4) and cross-checks the two rather than adopting either silently.
OUTCOME_CODE_MAP_KEYS = ("outcome_code_map_json",)
OUTCOME_CODE_ARRAY_KEYS = ("outcome_code_values", "outcome_code_names")


def _bind_keys(archive, table, required):
    available = list(archive.keys())
    binding = {}
    for canonical, aliases in sorted(table.items()):
        for alias in aliases:
            if alias in archive:
                binding[canonical] = alias
                break
    missing = [key for key in required if key not in binding]
    if missing:
        raise AuditPreconditionError(
            "frames npz is missing required key(s) %s; keys present: %s"
            % (", ".join(missing), ", ".join(sorted(available)))
        )
    return binding


def _scalar_value(value):
    """Decode a 0-d / 1-element npz entry into a JSON-safe scalar (str, int or float)."""
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        return None
    item = array[0]
    if isinstance(item, bytes):
        return item.decode("utf-8", "replace")
    if np.issubdtype(array.dtype, np.integer):
        return int(item)
    if np.issubdtype(array.dtype, np.floating):
        return float(item)
    if np.issubdtype(array.dtype, np.bool_):
        return bool(item)
    return str(item)


def load_frames(frames_path):
    archive = np.load(str(frames_path), allow_pickle=False)
    frame_binding = _bind_keys(archive, FRAME_KEYS, REQUIRED_FRAME_KEYS)
    episode_binding = _bind_keys(archive, EPISODE_KEYS, REQUIRED_EPISODE_KEYS)

    frames = dict((key, np.asarray(archive[alias])) for key, alias in frame_binding.items())
    episodes = dict((key, np.asarray(archive[alias])) for key, alias in episode_binding.items())

    obs = frames["obs"]
    if obs.ndim != 2:
        raise AuditPreconditionError("frames obs must be 2-D [N, D], got shape %s" % (obs.shape,))
    n_frames = int(obs.shape[0])
    for key, value in frames.items():
        if key == "obs":
            continue
        if int(value.shape[0]) != n_frames:
            raise AuditPreconditionError(
                "frame column %r has %d rows but obs has %d" % (key, value.shape[0], n_frames)
            )
    n_episodes = int(episodes["ep_uid"].shape[0])
    for key, value in episodes.items():
        if int(value.shape[0]) != n_episodes:
            raise AuditPreconditionError(
                "episode column %r has %d rows but ep_uid has %d"
                % (key, value.shape[0], n_episodes)
            )
    # Provenance scalars the NAVRL_OBS_DUMP writer appends alongside the two tables.  Optional:
    # a dump produced before they existed still audits, it just records nulls.
    extras = {}
    for key in ("stride_eff", "decimations"):
        if key in archive:
            extras[key] = int(np.asarray(archive[key]).reshape(-1)[0])
        else:
            extras[key] = None
    identity = {}
    for canonical, aliases in sorted(IDENTITY_KEYS.items()):
        for alias in aliases:
            if alias in archive:
                value = _scalar_value(archive[alias])
                if value is not None:
                    identity[canonical] = value
                break
    return (
        frames,
        episodes,
        {
            "frames": frame_binding,
            "episodes": episode_binding,
            "dump_provenance": extras,
            "identity_fields": identity,
            "dump_outcome_code_map": read_dump_outcome_code_map(archive),
        },
    )


def read_dump_outcome_code_map(archive):
    """The code -> name map the dump shipped with the data, if it shipped one."""
    for key in OUTCOME_CODE_MAP_KEYS:
        if key in archive:
            text = _scalar_value(archive[key])
            if not text:
                continue
            try:
                loaded = json.loads(str(text))
            except ValueError:
                continue
            return dict((int(code), str(name)) for code, name in loaded.items())
    values_key, names_key = OUTCOME_CODE_ARRAY_KEYS
    if values_key in archive and names_key in archive:
        values = np.asarray(archive[values_key]).reshape(-1).tolist()
        names = np.asarray(archive[names_key]).reshape(-1).tolist()
        if len(values) == len(names):
            return dict(
                (int(code), name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name))
                for code, name in zip(values, names)
            )
    return None


def join_outcomes(frame_episode_uid, episodes):
    """Map each frame to its episode outcome code.

    Frames whose episode has NO ROW in the outcome table get ``NO_OUTCOME_ROW`` (-1).  That
    sentinel is emphatically NOT outcome code 5 ("unattributed"), which is a code the dump itself
    emits for a FINISHED episode it could not attribute; see NO_OUTCOME_ROW_MEANING.
    """
    lookup = {}
    for uid, code in zip(
        np.asarray(episodes["ep_uid"]).tolist(), np.asarray(episodes["outcome"]).tolist()
    ):
        lookup[int(uid)] = int(code)
    return np.array(
        [lookup.get(int(uid), NO_OUTCOME_ROW) for uid in np.asarray(frame_episode_uid).tolist()],
        dtype=np.int64,
    )


def _domain_of(values):
    return sorted(int(value) for value in np.unique(np.asarray(values)).tolist())


def as_two_valued_bool(values, name, domain=CTX_VALID_DOMAIN):
    """Boolean view of a two-valued column, asserting the dtype and the domain (never coercing).

    ``np.asarray(x).astype(bool)`` maps EVERY nonzero to True, so an int8 column that later grows a
    -1 = "unknown" state would silently be counted as True (for ctx_target_visible: as VISIBLE).
    Bool dtype is accepted as-is; an integer dtype is accepted only if every value it actually
    contains is inside ``domain``.
    """
    array = np.asarray(values)
    if array.dtype == np.bool_:
        return array
    if np.issubdtype(array.dtype, np.integer):
        observed = _domain_of(array)
        if not set(observed) <= set(int(value) for value in domain):
            raise IntegrityViolation(
                "column %r is two-valued by contract but contains %s (allowed: %s); refusing to "
                "coerce with .astype(bool), which would map every nonzero value to True"
                % (name, observed, sorted(domain)),
                {"column": name, "dtype": str(array.dtype), "observed_domain": observed,
                 "allowed_domain": sorted(int(value) for value in domain)},
            )
        return array.astype(bool)
    raise IntegrityViolation(
        "column %r has dtype %s; a two-valued context column must be bool or integer"
        % (name, array.dtype),
        {"column": name, "dtype": str(array.dtype)},
    )


def as_front_code(values, name="ctx_front_blocked", domain=CTX_FRONT_BLOCKED_DOMAIN):
    """Three-valued front column (-1 unknown / 0 clear / 1 blocked), domain asserted."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer) and array.dtype != np.bool_:
        raise IntegrityViolation(
            "column %r has dtype %s; the three-valued front column must be an integer column"
            % (name, array.dtype),
            {"column": name, "dtype": str(array.dtype)},
        )
    observed = _domain_of(array)
    if not set(observed) <= set(int(value) for value in domain):
        raise IntegrityViolation(
            "column %r contains %s, outside the preregistered domain %s; an out-of-domain label "
            "would fall out of every cell of the front family without failing any gate"
            % (name, observed, sorted(domain)),
            {"column": name, "dtype": str(array.dtype), "observed_domain": observed,
             "allowed_domain": sorted(int(value) for value in domain)},
        )
    return array.astype(np.int64)


def check_context_columns(frames):
    """Assert the dtype and the domain of every context column the split is built from."""
    detail = {"columns": {}}
    for name, domain, kind in (
        ("ctx_valid", CTX_VALID_DOMAIN, "two_valued"),
        ("ctx_target_visible", CTX_TARGET_VISIBLE_DOMAIN, "two_valued"),
        ("ctx_front_blocked", CTX_FRONT_BLOCKED_DOMAIN, "three_valued"),
    ):
        array = np.asarray(frames[name])
        if kind == "two_valued":
            as_two_valued_bool(array, name, domain)
        else:
            as_front_code(array, name, domain)
        detail["columns"][name] = {
            "dtype": str(array.dtype),
            "allowed_domain": sorted(int(value) for value in domain),
            "observed_domain": (
                [bool(value) for value in np.unique(array).tolist()]
                if array.dtype == np.bool_
                else _domain_of(array)
            ),
            "coerced_with_astype_bool": False,
        }
    return detail


def build_contexts(frames, outcome_by_frame):
    """Prereg section 6 -- first-order context splits only; never crossed.

    The context columns are validated (dtype + domain), never coerced: see as_two_valued_bool.
    """
    visible = as_two_valued_bool(
        frames["ctx_target_visible"], "ctx_target_visible", CTX_TARGET_VISIBLE_DOMAIN
    )
    front = as_front_code(frames["ctx_front_blocked"])
    contexts = [(CONTEXT_OVERALL_CELL, np.ones(visible.shape[0], dtype=bool))]
    contexts.append(("target_visible", visible))
    contexts.append(("target_hidden", ~visible))
    contexts.append(("front_blocked", front == 1))
    contexts.append(("front_clear", front == 0))
    contexts.append(("front_unknown", front == -1))
    for code in OUTCOME_CONTEXT_CODES:
        contexts.append(("outcome_" + OUTCOME_CODES[code], outcome_by_frame == code))
    return contexts


def check_context_partitions(contexts, n_valid, n_no_outcome_row, n_non_context_outcome=0):
    """Every context family must PARTITION the population it claims to cover.  Fail-closed.

    Without this, an out-of-domain label vanishes from every cell of its family silently: a future
    ``ctx_front_blocked == 2`` would leave front_blocked + front_clear + front_unknown short of the
    valid-frame count, with entirely plausible per-cell numbers and no gate firing.

    Populations (prereg section 6):
      * front   {blocked, clear, unknown}  -> every valid frame;
      * target  {visible, hidden}          -> every valid frame;
      * outcome {the five preregistered}   -> every valid frame that HAS an outcome row, i.e.
                                              n_valid minus the excluded no_outcome_row frames.
        (Outcome code 5 "unattributed" IS an outcome row but is not a preregistered context, so it
        is carried as an explicitly named allowance rather than being lost in the arithmetic.)

    Cells are additionally required to be pairwise disjoint, so "sums to the total" cannot be
    reached by double counting.  Returns the reconciliation the JSON reports.
    """
    masks = {}
    for name, mask in contexts:
        masks[name] = np.asarray(mask, dtype=bool)
    n_valid = int(n_valid)
    n_no_outcome_row = int(n_no_outcome_row)
    n_non_context_outcome = int(n_non_context_outcome)

    if CONTEXT_OVERALL_CELL not in masks:
        raise IntegrityViolation("context list has no %r cell" % CONTEXT_OVERALL_CELL, {})
    n_overall = int(masks[CONTEXT_OVERALL_CELL].sum())
    n_outcome_population = n_valid - n_no_outcome_row - n_non_context_outcome

    families = (
        ("front", CONTEXT_FAMILY_FRONT, n_valid, "every valid frame"),
        ("target", CONTEXT_FAMILY_TARGET, n_valid, "every valid frame"),
        (
            "outcome",
            CONTEXT_FAMILY_OUTCOME,
            n_outcome_population,
            "every valid frame that has an outcome row carrying one of the five preregistered "
            "codes (n_valid minus no_outcome_row frames minus non-context outcome codes)",
        ),
    )

    report = {
        "n_valid_frames": n_valid,
        "n_frames_no_outcome_row": n_no_outcome_row,
        "n_frames_non_context_outcome_code": n_non_context_outcome,
        "overall_cell_rows": n_overall,
        "overall_cell_reconciles": n_overall == n_valid,
        "families": {},
    }
    failures = []
    if n_overall != n_valid:
        failures.append(
            "overall cell covers %d rows but there are %d valid frames" % (n_overall, n_valid)
        )
    for family, cell_names, expected_total, covers in families:
        missing = [name for name in cell_names if name not in masks]
        if missing:
            failures.append("family %r is missing cell(s) %s" % (family, ", ".join(missing)))
            report["families"][family] = {
                "covers": covers,
                "missing_cells": missing,
                "reconciles": False,
            }
            continue
        counts = dict((name, int(masks[name].sum())) for name in cell_names)
        total = int(sum(counts.values()))
        covered = np.zeros(masks[cell_names[0]].shape[0], dtype=np.int64)
        for name in cell_names:
            covered += masks[name].astype(np.int64)
        disjoint = bool(covered.size == 0 or int(covered.max()) <= 1)
        reconciles = total == int(expected_total)
        report["families"][family] = {
            "covers": covers,
            "cells": counts,
            "cells_total": total,
            "expected_total": int(expected_total),
            "unassigned_frames": int(expected_total) - total,
            "reconciles": reconciles,
            "cells_disjoint": disjoint,
        }
        if not reconciles:
            failures.append(
                "family %r covers %d rows but must cover %d (%s): %d frame(s) are in NO cell of "
                "the family" % (family, total, expected_total, covers, int(expected_total) - total)
            )
        if not disjoint:
            failures.append("family %r has overlapping cells" % family)
    report["partitions_complete"] = not failures
    if failures:
        raise IntegrityViolation("context partition check failed: " + "; ".join(failures), report)
    return report


def check_outcome_join(outcome_by_valid_frame, n_valid):
    """Reconcile the outcome join: joined + excluded == n_valid, with both totals reported.

    The payload used to place ``outcome_frame_counts["unattributed"] = 0`` (outcome code 5) three
    lines from a sentinel count of 1530 frames reported under the same word.  Emitting the
    joined total, the excluded total and n_valid together -- and asserting they add up -- makes it
    impossible to read the code-5 zero as "the split is complete".
    """
    codes = np.asarray(outcome_by_valid_frame, dtype=np.int64)
    n_valid = int(n_valid)
    counts = dict(
        (OUTCOME_CODES[code], int((codes == code).sum())) for code in sorted(OUTCOME_CODES)
    )
    n_no_row = int((codes == NO_OUTCOME_ROW).sum())
    n_with_row = int(sum(counts.values()))
    known = set(OUTCOME_CODES) | {NO_OUTCOME_ROW}
    unknown_codes = sorted(
        code for code in np.unique(codes).tolist() if int(code) not in known
    )
    n_unknown = int(sum(int((codes == code).sum()) for code in unknown_codes))
    context_total = int(sum(counts[OUTCOME_CODES[code]] for code in OUTCOME_CONTEXT_CODES))

    report = {
        "sentinel": NO_OUTCOME_ROW,
        "sentinel_name": NO_OUTCOME_ROW_NAME,
        "sentinel_meaning": NO_OUTCOME_ROW_MEANING,
        "not_to_be_confused_with": {
            "outcome_code": 5,
            "name": OUTCOME_CODES[5],
            "meaning": (
                "a code the dump emits for a FINISHED episode it could not attribute; it has an "
                "outcome row and is counted in n_frames_with_outcome_row"
            ),
            "n_frames": counts[OUTCOME_CODES[5]],
        },
        "n_valid_frames": n_valid,
        "n_frames_with_outcome_row": n_with_row,
        "n_frames_no_outcome_row": n_no_row,
        "n_frames_unknown_outcome_code": n_unknown,
        "unknown_outcome_codes": [int(code) for code in unknown_codes],
        "outcome_frame_counts": counts,
        "outcome_context_cell_total": context_total,
        "outcome_context_cell_codes": list(OUTCOME_CONTEXT_CODES),
        "reconciliation": (
            "n_frames_with_outcome_row + n_frames_no_outcome_row + n_frames_unknown_outcome_code "
            "== n_valid_frames"
        ),
        "correct_denominator_for_outcome_shares": "n_frames_with_outcome_row",
    }
    reconciles = (n_with_row + n_no_row + n_unknown) == n_valid
    report["reconciles"] = reconciles
    if unknown_codes:
        raise IntegrityViolation(
            "outcome table contains code(s) %s outside OUTCOME_CODES; those frames belong to no "
            "context cell" % (unknown_codes,),
            report,
        )
    if not reconciles:
        raise IntegrityViolation(
            "outcome join does not reconcile: %d joined + %d without a row != %d valid frames"
            % (n_with_row, n_no_row, n_valid),
            report,
        )
    return report


def characterise_excluded_frames(call_index, excluded_mask):
    """Characterise the no_outcome_row exclusion FROM THE DATA (H6).  Non-gating diagnostic.

    The excluded frames are not missing at random: they belong to episodes still running when the
    outcome table closed, so they sit in the tail of the recording.  Computing the call_index range
    of the excluded set against the population's, and flagging tail concentration, means the caveat
    is generated every run instead of being remembered by a human.
    """
    excluded = np.asarray(excluded_mask, dtype=bool)
    n_population = int(excluded.shape[0])
    n_excluded = int(excluded.sum())
    report = {
        "gating": False,
        "n_population": n_population,
        "n_excluded": n_excluded,
        "fraction_excluded": (float(n_excluded) / float(n_population)) if n_population else None,
        "tail_concentration_threshold": EXCLUDED_TAIL_CONCENTRATION_MIN,
        "tail_rule": (
            "concentrated_in_tail is true when at least tail_concentration_threshold of the "
            "excluded frames have call_index >= the population median call_index"
        ),
    }
    if call_index is None:
        report["call_index_available"] = False
        report["concentrated_in_tail"] = None
        report["caveat"] = (
            "the dump carries no call_index column, so whether the excluded frames are missing at "
            "random could NOT be characterised"
        )
        return report
    report["call_index_available"] = True
    values = np.asarray(call_index, dtype=np.float64)
    if n_population == 0 or n_excluded == 0:
        report["concentrated_in_tail"] = False
        report["caveat"] = "no frames were excluded from the outcome split"
        return report
    population_median = percentile(values, 50.0)
    excluded_values = values[excluded]
    fraction_in_tail = float((excluded_values >= population_median).mean())
    below = float((values < float(excluded_values.min())).mean())
    report.update(
        {
            "population_call_index_min": float(values.min()),
            "population_call_index_max": float(values.max()),
            "population_call_index_median": population_median,
            "excluded_call_index_min": float(excluded_values.min()),
            "excluded_call_index_max": float(excluded_values.max()),
            "excluded_call_index_median": percentile(excluded_values, 50.0),
            "fraction_of_excluded_at_or_above_population_median": fraction_in_tail,
            "excluded_min_call_index_percentile_of_population": 100.0 * below,
        }
    )
    concentrated = bool(fraction_in_tail >= EXCLUDED_TAIL_CONCENTRATION_MIN)
    report["concentrated_in_tail"] = concentrated
    if concentrated:
        report["caveat"] = (
            "the %d frames excluded from the outcome split are NOT missing at random: %.1f%% of "
            "them carry call_index >= %g (population median), the lowest being %g of a maximum "
            "%g. They belong to episodes still running when the outcome table closed, so the "
            "OUTCOME-SPLIT cells are biased toward episodes that terminated inside the recording "
            "window. The overall cell is computed on all valid frames and is unaffected."
            % (
                n_excluded,
                100.0 * fraction_in_tail,
                population_median,
                float(excluded_values.min()),
                float(values.max()),
            )
        )
    else:
        report["caveat"] = (
            "the %d excluded frames are spread across the recording (%.1f%% at or above the "
            "population median call_index); no tail concentration flagged"
            % (n_excluded, 100.0 * fraction_in_tail)
        )
    return report


def check_frames_obs_dtype(obs):
    """Assert the observation dtype this tool assumes (H5).

    rl_games' ``BasePlayer._preproc_obs`` divides uint8 observations by 255.  This tool builds a
    float32 tensor straight from the npz, so a uint8 dump would silently SKIP that scaling and
    every number below would describe a different observation than the live path sees.
    """
    dtype = np.dtype(np.asarray(obs).dtype)
    detail = {
        "frames_obs_dtype": str(dtype),
        "expected_dtype": FRAMES_OBS_DTYPE,
        "why": (
            "rl_games BasePlayer._preproc_obs scales uint8 observations by 1/255; this tool casts "
            "the dump to float32 and would skip that path"
        ),
    }
    if dtype != np.dtype(FRAMES_OBS_DTYPE):
        raise IntegrityViolation(
            "frames obs dtype is %s but this tool assumes %s (a uint8 dump would silently skip "
            "the live path's /255)" % (dtype, FRAMES_OBS_DTYPE),
            detail,
        )
    return detail


def _as_numpy(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def check_finite(arrays, label):
    """Fail closed on any non-finite value (H5).

    A single NaN propagates straight through the statistics: ``np.percentile`` returns NaN, and
    both ``NaN >= 0.30`` and ``NaN <= 0.10`` are False, so the verdict rule lands on INCONCLUSIVE
    with no gate firing at all.  An INCONCLUSIVE that means "the arithmetic was corrupt" is
    indistinguishable from one that means "the policy is mildly chiral", so this fails instead.
    """
    detail = {"label": label, "arrays": {}}
    offenders = []
    for name in sorted(arrays):
        values = _as_numpy(arrays[name])
        finite = np.isfinite(values)
        n_bad = int(values.size) - int(finite.sum())
        detail["arrays"][name] = {
            "n_elements": int(values.size),
            "n_non_finite": n_bad,
            "all_finite": n_bad == 0,
        }
        if n_bad:
            offenders.append("%s (%d non-finite of %d)" % (name, n_bad, int(values.size)))
    detail["all_finite"] = not offenders
    if offenders:
        raise IntegrityViolation(
            "%s contains non-finite values: %s" % (label, "; ".join(offenders)), detail
        )
    return detail


def check_dump_outcome_code_map(dump_map):
    """Cross-check the dump's own code -> name map against this tool's preregistered literal.

    OUTCOME_CODES is preregistered and stays a literal (prereg section 4).  When the dump ships its
    map, the two are compared on the codes they SHARE: a code that means something different on the
    two sides would silently re-label a whole stratum.  Codes the dump knows and the prereg does
    not are reported, not adopted -- frames actually carrying one of them fail check_outcome_join,
    which is the correct fail-closed outcome for a stratum with no preregistered cell.
    """
    if not dump_map:
        return {
            "dump_ships_a_code_map": False,
            "tool_outcome_code_map": dict((str(c), n) for c, n in OUTCOME_CODES.items()),
            "note": (
                "this dump ships no outcome-code map, so the tool's preregistered literal could "
                "NOT be cross-checked against the producer's"
            ),
        }
    shared = sorted(set(dump_map) & set(OUTCOME_CODES))
    disagreements = dict(
        (str(code), {"dump": dump_map[code], "tool": OUTCOME_CODES[code]})
        for code in shared
        if str(dump_map[code]) != str(OUTCOME_CODES[code])
    )
    detail = {
        "dump_ships_a_code_map": True,
        "dump_outcome_code_map": dict((str(c), n) for c, n in sorted(dump_map.items())),
        "tool_outcome_code_map": dict((str(c), n) for c, n in OUTCOME_CODES.items()),
        "shared_codes": shared,
        "codes_only_in_dump": sorted(set(dump_map) - set(OUTCOME_CODES)),
        "codes_only_in_tool": sorted(set(OUTCOME_CODES) - set(dump_map)),
        "name_disagreements": disagreements,
        "note": (
            "codes_only_in_dump have no preregistered context cell; frames carrying one of them "
            "fail the outcome-join check rather than being dropped from the split"
        ),
    }
    if disagreements:
        raise IntegrityViolation(
            "the dump's outcome-code map disagrees with the preregistered map on code(s) %s"
            % ", ".join(sorted(disagreements)),
            detail,
        )
    return detail


def check_npz_identity(identity, checkpoint_sha256, obs_width=None):
    """Bind the frames to the checkpoint when the dump carries an identity; say so when it does not.

    Nothing in the npz schema used to tie the frames to a policy: the caller's word was the only
    link.  Whatever run identity the dump carries is recorded here, and a checkpoint SHA-256 in the
    dump is ASSERTED against the audited checkpoint.  When the dump carries none, the JSON says so
    explicitly rather than being silent about it.
    """
    fields = dict(identity or {})
    report = {
        "identity_fields_present": sorted(fields),
        "identity_fields": fields,
        "checkpoint_binding_verified": False,
    }
    recorded_width = fields.get("obs_width_recorded")
    if recorded_width is not None and obs_width is not None:
        report["obs_width_recorded_matches_loaded_obs"] = int(recorded_width) == int(obs_width)
        if int(recorded_width) != int(obs_width):
            raise IntegrityViolation(
                "the dump records obs width %d but the loaded observation table is %d wide"
                % (int(recorded_width), int(obs_width)),
                report,
            )
    dumped_sha = fields.get("checkpoint_sha256")
    if dumped_sha is None:
        report["note"] = (
            "this dump carries no checkpoint identity, so the binding between these frames and "
            "the audited checkpoint is NOT machine-verified here; it rests on the caller, and on "
            "the frames_sha256 / checkpoint_sha256 recorded in this file"
        )
        return report
    matches = str(dumped_sha).strip().lower() == str(checkpoint_sha256).strip().lower()
    report["npz_checkpoint_sha256"] = str(dumped_sha).strip().lower()
    report["audited_checkpoint_sha256"] = str(checkpoint_sha256).strip().lower()
    report["checkpoint_binding_verified"] = matches
    if not matches:
        raise IntegrityViolation(
            "the frames npz was dumped from checkpoint %s but this audit is running checkpoint %s"
            % (report["npz_checkpoint_sha256"], report["audited_checkpoint_sha256"]),
            report,
        )
    report["note"] = "the dump's checkpoint identity matches the audited checkpoint"
    return report


# --------------------------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------------------------


def measure_cell(original, mirrored, mask):
    """All preregistered statistics for one context cell (numpy float64 throughout)."""
    rows = np.asarray(mask, dtype=bool)
    pi_o = np.asarray(original, dtype=np.float64)[rows]
    pi_mo = np.asarray(mirrored, dtype=np.float64)[rows]
    n_rows = int(pi_o.shape[0])

    # e = pi(preproc(M o)) - mirror_navrl_actions(pi(preproc(o))); the action mirror negates
    # channels 1 (lateral) and 3 (yaw).
    expected = pi_o.copy()
    expected[:, 1] = -expected[:, 1]
    expected[:, 3] = -expected[:, 3]
    error = pi_mo - expected

    cell = {"n_rows": n_rows, "channels": {}}
    for name, axis in ACTION_CHANNELS:
        cell["channels"][name] = describe(np.abs(error[:, axis]))

    lat_o = pi_o[:, LATERAL_ACTION_INDEX]
    lat_mo = pi_mo[:, LATERAL_ACTION_INDEX]
    comparable = (np.abs(lat_o) >= LATERAL_SIGN_THRESHOLD) & (
        np.abs(lat_mo) >= LATERAL_SIGN_THRESHOLD
    )
    n_comparable = int(comparable.sum())
    if n_comparable > 0:
        agreeing = np.sign(lat_mo[comparable]) == -np.sign(lat_o[comparable])
        sign_agreement = float(agreeing.sum()) / float(n_comparable)
    else:
        sign_agreement = None
    cell["lateral_sign_agreement"] = sign_agreement
    cell["lateral_sign_comparable"] = n_comparable
    cell["lateral_sign_threshold"] = LATERAL_SIGN_THRESHOLD
    cell["signed_lateral_bias"] = (
        float((lat_o.mean() + lat_mo.mean()) / 2.0) if n_rows > 0 else None
    )
    cell["mean_pi_o_lateral"] = float(lat_o.mean()) if n_rows > 0 else None
    cell["mean_pi_mo_lateral"] = float(lat_mo.mean()) if n_rows > 0 else None

    sufficient = has_sufficient_sample(n_comparable)
    cell["insufficient_sample"] = not sufficient
    cell["min_comparable_rows"] = MIN_CONTEXT_COMPARABLE_ROWS
    if sufficient:
        cell["verdict"] = classify_verdict(
            cell["channels"]["conj_err_lat"]["median"], sign_agreement
        )
    else:
        cell["verdict"] = None
    return cell


def measure_all(original, mirrored, contexts):
    return dict(
        (name, measure_cell(original, mirrored, mask)) for name, mask in contexts
    )


# --------------------------------------------------------------------------------------------
# Quality gates (prereg section 5)
# --------------------------------------------------------------------------------------------


def gate_schema(schema, obs_width):
    """Q3 -- schema from checkpoint metadata, and it must match the collected observation."""
    derived = int(schema["structured_obs_dim"])
    detail = {
        "cfg_lidar_hbeams": schema["cfg_lidar_hbeams"],
        "cfg_lidar_vbeams": schema["cfg_lidar_vbeams"],
        "cfg_max_obstacles": schema["cfg_max_obstacles"],
        "cfg_corridor_tokens": schema["cfg_corridor_tokens"],
        "derived_structured_obs_dim": derived,
        "frames_obs_dim": int(obs_width),
        "expected": GATE_STRUCTURED_OBS_DIM,
    }
    passed = derived == int(obs_width) == GATE_STRUCTURED_OBS_DIM
    return passed, detail


def gate_index_sets(mirror_fn, hbeams, vbeams, max_obstacles):
    """Q4 -- byte-level index-set exactness against the independently built preregistered sets."""
    import torch

    source, sign = preregistered_signed_permutation(hbeams, vbeams, max_obstacles)
    dim = len(source)

    # 1. Recover the operator the function actually implements.  Distinct positive magnitudes make
    #    (source index, sign) unambiguous for every output coordinate.
    probe = torch.arange(1, dim + 1, dtype=torch.float32).view(1, dim)
    probed = mirror_fn(probe)[0]
    observed_source = (probed.abs() - 1.0).round().to(torch.long).tolist()
    observed_sign = [1 if float(value) > 0.0 else -1 for value in probed.tolist()]

    observed_flip = set(i for i, s in enumerate(observed_sign) if s == -1)
    observed_perm = dict((d, s) for d, s in enumerate(observed_source) if s != d)
    expected_flip = set(i for i, s in enumerate(sign) if s == -1)
    expected_perm = dict((d, s) for d, s in enumerate(source) if s != d)
    observed_unchanged = set(
        i
        for i in range(dim)
        if observed_source[i] == i and observed_sign[i] == 1
    )
    expected_unchanged = set(range(dim)) - expected_flip - set(expected_perm)

    # 2. Byte-level equality on a synthetic random tensor against an independently built expected
    #    output (this is what makes "every other index is unchanged" a real assertion).
    generator = torch.Generator().manual_seed(20260821)
    random_obs = torch.randn((8, dim), generator=generator, dtype=torch.float32)
    index = torch.as_tensor(source, dtype=torch.long)
    signs = torch.as_tensor(sign, dtype=torch.float32)
    expected_obs = random_obs.index_select(1, index) * signs
    byte_exact = bool(torch.equal(mirror_fn(random_obs), expected_obs))

    passed = (
        observed_flip == expected_flip
        and observed_perm == expected_perm
        and observed_unchanged == expected_unchanged
        and byte_exact
    )
    detail = {
        "sign_flip_index_count": len(expected_flip),
        "sign_flip_index_sets_match": observed_flip == expected_flip,
        "permutation_index_count": len(expected_perm),
        "permutation_index_sets_match": observed_perm == expected_perm,
        "unchanged_index_count": len(expected_unchanged),
        "unchanged_index_sets_match": observed_unchanged == expected_unchanged,
        "byte_level_equality_on_random_tensor": byte_exact,
        "probe_rows": int(random_obs.shape[0]),
    }
    return passed, detail, observed_source, observed_sign


def observed_scan_fixed_points(observed_source, hbeams, vbeams):
    """Beam indices the OBSERVED operator leaves in place on EVERY ring.

    ``h`` is a fixed point only if ``observed_source[v*H + h] == v*H + h`` for every ring ``v`` --
    a beam that is fixed on one ring and moved on another is not a fixed point of the operator.
    """
    hbeams = int(hbeams)
    vbeams = int(vbeams)
    fixed = set()
    for h in range(hbeams):
        if all(
            int(observed_source[v * hbeams + h]) == v * hbeams + h for v in range(vbeams)
        ):
            fixed.add(h)
    return fixed


def gate_scan_permutation(observed_source, hbeams, vbeams):
    """Q5 -- ring permutation h -> (-h) mod H with fixed points exactly {0, 36}.

    ``fixed_points`` is measured on the OBSERVED operator (the source map recovered from the real
    mirror function by gate_index_sets), never on this tool's own preregistered map.  Reading the
    preregistered map here made the reported evidence a tautology: at hbeams=72 it is the constant
    {0, 36} and could not disagree with ``expected_fixed_points``, so the field a reader would cite
    as proof proved nothing.
    """
    hbeams = int(hbeams)
    vbeams = int(vbeams)
    expected = preregistered_scan_permutation(hbeams)
    rings_match = True
    for v in range(vbeams):
        for h in range(hbeams):
            if int(observed_source[v * hbeams + h]) != v * hbeams + expected[h]:
                rings_match = False
                break
        if not rings_match:
            break
    fixed_points = observed_scan_fixed_points(observed_source, hbeams, vbeams)
    detail = {
        "hbeams": hbeams,
        "vbeams": vbeams,
        "ring_permutation_matches_all_rings": rings_match,
        "fixed_points": sorted(fixed_points),
        "fixed_points_computed_from": (
            "observed_source (the operator recovered from the mirror function), not the tool's "
            "preregistered map"
        ),
        "expected_fixed_points": sorted(GATE_SCAN_FIXED_POINTS),
    }
    passed = rings_match and fixed_points == set(GATE_SCAN_FIXED_POINTS)
    return passed, detail


def gate_involution_and_isometry(obs, mirror_fn, batch_size, device):
    """Q1 -- M(M(x)) == x exactly; Q2 -- ||M x|| == ||x|| to 1e-3.  Over every collected frame."""
    import torch

    max_involution = 0.0
    max_isometry = 0.0
    total = int(obs.shape[0])
    for start in range(0, total, int(batch_size)):
        chunk = obs[start : start + int(batch_size)].to(device)
        mirrored = mirror_fn(chunk)
        round_trip = mirror_fn(mirrored)
        max_involution = max(
            max_involution, float((round_trip - chunk).abs().max().item()) if chunk.numel() else 0.0
        )
        norm_delta = torch.linalg.norm(mirrored, dim=1) - torch.linalg.norm(chunk, dim=1)
        max_isometry = max(
            max_isometry, float(norm_delta.abs().max().item()) if chunk.numel() else 0.0
        )
    involution_passed = max_involution == GATE_INVOLUTION_MAX_ABS
    isometry_passed = max_isometry <= GATE_ISOMETRY_MAX_ABS
    return (
        involution_passed,
        {"max_abs_round_trip_error": max_involution, "threshold": GATE_INVOLUTION_MAX_ABS},
        isometry_passed,
        {"max_abs_norm_difference": max_isometry, "threshold": GATE_ISOMETRY_MAX_ABS},
    )


# --------------------------------------------------------------------------------------------
# Equivalence proof (self-check)
# --------------------------------------------------------------------------------------------


def equivalence_proof(player, checkpoint_state, device):
    """Prove the offline forward reproduces the live path, with reportable numbers."""
    import torch

    stripped = {}
    for key, value in checkpoint_state["model"].items():
        stripped[key[len("_orig_mod.") :] if key.startswith("_orig_mod.") else key] = value

    model_state = player.model.state_dict()
    keys_match = sorted(model_state.keys()) == sorted(stripped.keys())

    def _fingerprint(state):
        digest = hashlib.sha256()
        count = 0
        for key in sorted(state.keys()):
            tensor = state[key].detach().to("cpu").contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(tensor.reshape(-1).numpy().tobytes())
            count += int(tensor.numel())
        return digest.hexdigest(), count

    model_hash, model_elements = _fingerprint(model_state)
    checkpoint_hash, checkpoint_elements = _fingerprint(stripped)

    rms = _running_mean_std(player.model)
    rms_mean_identical = bool(
        torch.equal(rms.running_mean.detach().cpu(), stripped["running_mean_std.running_mean"].cpu())
    )
    rms_var_identical = bool(
        torch.equal(rms.running_var.detach().cpu(), stripped["running_mean_std.running_var"].cpu())
    )
    rms_count_identical = bool(
        torch.equal(rms.count.detach().cpu(), stripped["running_mean_std.count"].cpu())
    )

    from aerial_gym.rl_training.rl_games.navrl_players import NavRLPpoPlayerContinuous

    dim = int(rms.running_mean.numel())
    generator = torch.Generator().manual_seed(1900)
    synthetic = torch.randn((16, dim), generator=generator, dtype=torch.float32).to(device)
    with torch.no_grad():
        res_dict = player.model(
            {
                "is_train": False,
                "prev_actions": None,
                "obs": player._preproc_obs(synthetic),
                "rnn_states": None,
            }
        )
    has_deterministic = "deterministic_actions" in res_dict
    selected = NavRLPpoPlayerContinuous._model_action(res_dict, True)
    picks_deterministic = has_deterministic and bool(
        torch.equal(selected, res_dict["deterministic_actions"])
    )
    tanh_gap = float((selected - torch.tanh(res_dict["mus"])).abs().max().item())
    mus_gap = float((selected - res_dict["mus"]).abs().max().item())

    return {
        "state_dict_keys_match": keys_match,
        "model_parameter_element_count": model_elements,
        "checkpoint_parameter_element_count": checkpoint_elements,
        "model_parameter_sha256": model_hash,
        "checkpoint_parameter_sha256": checkpoint_hash,
        "parameter_fingerprints_identical": model_hash == checkpoint_hash,
        "parameter_fingerprint_recipe": (
            "sha256 over sorted state_dict keys of (key, shape, dtype, raw little-endian bytes of "
            "the flattened tensor)"
        ),
        "running_mean_bitwise_identical_to_checkpoint": rms_mean_identical,
        "running_var_bitwise_identical_to_checkpoint": rms_var_identical,
        "running_count_bitwise_identical_to_checkpoint": rms_count_identical,
        "running_mean_std_source": "checkpoint model state_dict (never recomputed from frames)",
        "res_dict_keys": sorted(res_dict.keys()),
        "model_action_prefers_deterministic_actions": picks_deterministic,
        "max_abs_selected_minus_tanh_mus": tanh_gap,
        "max_abs_selected_minus_mus": mus_gap,
        "action_units": "normalised policy output before rescale/clamp to actions_low/high",
        "model_in_eval_mode": not player.model.training,
        "player_created_environment": player.env is not None,
    }


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def write_json(path, payload):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)


def _base_payload(args, checkpoint_path, checkpoint_sha, frames_sha, schema, exported_env):
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "preregistration": "docs/prereg_2026-08-21_n1_real_frame_reflection_audit.md",
        "scope": "offline_real_frame_reflection_audit_no_simulator",
        "decision_authority": "none",
        "simulator_created": False,
        "environment_stepped": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": schema["epoch"],
        "robot_name": schema["cfg_robot_name"],
        "frames_npz": str(Path(args.frames).resolve()),
        "frames_sha256": frames_sha,
        "checkpoint_schema": dict(
            (key, schema[key])
            for key in (
                "cfg_lidar_hbeams",
                "cfg_lidar_vbeams",
                "cfg_max_obstacles",
                "cfg_corridor_tokens",
                "structured_obs_dim",
                "cfg_action_policy",
                "cfg_action_std",
                "cfg_action_mu_scale",
            )
        ),
        "schema_environment_exported": exported_env,
        "percentile_convention": PERCENTILE_CONVENTION,
        "thresholds": {
            "verdict_confirm_median_min": VERDICT_CONFIRM_MEDIAN_MIN,
            "verdict_confirm_sign_agreement_max": VERDICT_CONFIRM_SIGN_AGREEMENT_MAX,
            "verdict_absent_median_max": VERDICT_ABSENT_MEDIAN_MAX,
            "verdict_absent_sign_agreement_min": VERDICT_ABSENT_SIGN_AGREEMENT_MIN,
            "lateral_sign_threshold": LATERAL_SIGN_THRESHOLD,
            "min_context_comparable_rows": MIN_CONTEXT_COMPARABLE_ROWS,
            "min_valid_frames": MIN_VALID_FRAMES,
            "gate_involution_max_abs": GATE_INVOLUTION_MAX_ABS,
            "gate_isometry_max_abs": GATE_ISOMETRY_MAX_ABS,
        },
        "outcome_code_map": dict((str(code), name) for code, name in OUTCOME_CODES.items()),
    }


def finalise_gate_block(payload, gates):
    """Stamp every gate with its state, publish the block and its three-way count (H3)."""
    stamp_gate_states(gates)
    summary = summarise_gates(gates)
    payload["quality_gates"] = gates
    payload["quality_gate_summary"] = summary
    payload["delegated_gates"] = dict(
        (name, dict(contract)) for name, contract in DELEGATED_GATES.items()
    )
    return summary


def record_integrity(payload, checks, name, thunk):
    """Run one fail-closed integrity check, recording what it examined either way.

    On violation the payload is stripped of every policy statistic (none may be reported behind a
    failed invariant), stamped with VERDICT_FAIL_CLOSED_DATA_INTEGRITY and raised as a GateFailure
    so the driver writes the evidence out and exits 3.
    """
    try:
        detail = thunk()
    except IntegrityViolation as exc:
        failed = dict(exc.detail)
        failed["passed"] = False
        failed["violation"] = str(exc)
        checks[name] = failed
        for key in ("measurements_raw_normaliser", "s1_symmetrised_normaliser", "verdict_basis"):
            payload.pop(key, None)
        payload["integrity_checks"] = checks
        payload["failed_integrity_checks"] = [name]
        payload["verdict"] = VERDICT_FAIL_CLOSED_INTEGRITY
        payload["note"] = (
            "A fail-closed data-integrity invariant was violated. No policy statistics and no "
            "policy claim are reported."
        )
        raise GateFailure([name], payload, kind="integrity checks")
    entry = dict(detail or {})
    entry["passed"] = True
    checks[name] = entry
    payload["integrity_checks"] = checks
    return entry


def run_audit(args):
    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != str(args.checkpoint_sha256).strip().lower():
        raise AuditPreconditionError(
            "checkpoint SHA-256 mismatch: expected %s, got %s"
            % (args.checkpoint_sha256, checkpoint_sha)
        )
    frames_path = Path(args.frames).resolve()
    if not frames_path.is_file():
        raise AuditPreconditionError("frames npz not found: %s" % frames_path)
    frames_sha = sha256_file(frames_path)
    if args.frames_sha256 and frames_sha != str(args.frames_sha256).strip().lower():
        raise AuditPreconditionError(
            "frames SHA-256 mismatch: expected %s, got %s" % (args.frames_sha256, frames_sha)
        )

    import torch  # noqa: F401  (imported here so env vars below are set before aerial_gym loads)

    checkpoint_state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    schema = read_checkpoint_schema(checkpoint_state)
    exported_env = apply_schema_environment(schema)

    install_cpu_only_aerial_gym_packages()
    from aerial_gym.rl_training.rl_games.ppo_update_safety import (
        mirror_navrl_structured_observation,
    )

    frames, episodes, key_binding = load_frames(frames_path)

    payload = _base_payload(args, checkpoint_path, checkpoint_sha, frames_sha, schema, exported_env)
    payload["npz_key_binding"] = key_binding

    # ---- fail-closed data-integrity invariants, BEFORE anything is measured -----------------
    integrity = {}
    payload["integrity_checks"] = integrity
    record_integrity(
        payload,
        integrity,
        "npz_run_identity",
        lambda: check_npz_identity(
            key_binding.get("identity_fields"),
            checkpoint_sha,
            obs_width=int(np.asarray(frames["obs"]).shape[1]),
        ),
    )
    record_integrity(
        payload,
        integrity,
        "dump_outcome_code_map",
        lambda: check_dump_outcome_code_map(key_binding.get("dump_outcome_code_map")),
    )
    record_integrity(payload, integrity, "frames_obs_dtype", lambda: check_frames_obs_dtype(frames["obs"]))
    record_integrity(
        payload, integrity, "frames_obs_finite", lambda: check_finite({"obs": frames["obs"]}, "frames.obs")
    )
    record_integrity(
        payload, integrity, "context_column_domains", lambda: check_context_columns(frames)
    )

    obs_all = torch.as_tensor(np.ascontiguousarray(frames["obs"]), dtype=torch.float32)
    valid = as_two_valued_bool(frames["ctx_valid"], "ctx_valid", CTX_VALID_DOMAIN)

    payload["frames"] = {
        "n_frames_total": int(obs_all.shape[0]),
        "n_frames_valid": int(valid.sum()),
        "obs_dim": int(obs_all.shape[1]),
        "n_episodes": int(np.asarray(episodes["ep_uid"]).shape[0]),
    }

    device = torch.device(args.device)
    gates = {}

    passed, detail = gate_schema(schema, int(obs_all.shape[1]))
    gates["Q3_schema"] = {"passed": passed, **detail}

    if passed:
        q4_passed, q4_detail, observed_source, observed_sign = gate_index_sets(
            mirror_navrl_structured_observation,
            schema["cfg_lidar_hbeams"],
            schema["cfg_lidar_vbeams"],
            schema["cfg_max_obstacles"],
        )
        gates["Q4_index_sets"] = {"passed": q4_passed, **q4_detail}
        q5_passed, q5_detail = gate_scan_permutation(
            observed_source, schema["cfg_lidar_hbeams"], schema["cfg_lidar_vbeams"]
        )
        gates["Q5_scan_permutation"] = {"passed": q5_passed, **q5_detail}
    else:
        observed_source = observed_sign = None
        gates["Q4_index_sets"] = {"passed": False, "skipped": "Q3 failed"}
        gates["Q5_scan_permutation"] = {"passed": False, "skipped": "Q3 failed"}

    n_valid = int(valid.sum())
    gates["Q9_sample_size"] = {
        "passed": n_valid >= MIN_VALID_FRAMES,
        "n_valid_frames": n_valid,
        "threshold": MIN_VALID_FRAMES,
    }

    if gates["Q3_schema"]["passed"]:
        q1_passed, q1_detail, q2_passed, q2_detail = gate_involution_and_isometry(
            obs_all, mirror_navrl_structured_observation, args.batch_size, device
        )
    else:
        q1_passed, q1_detail = False, {"skipped": "Q3 failed"}
        q2_passed, q2_detail = False, {"skipped": "Q3 failed"}
    gates["Q1_involution"] = {"passed": q1_passed, **q1_detail}
    gates["Q2_isometry"] = {"passed": q2_passed, **q2_detail}

    # Q6 and the manifest half of Q7 are NOT evaluated here.  They are declared as delegated (see
    # DELEGATED_GATES) so that they cannot be folded into a "0 failed" tally, and so a caller can
    # assert by name that IT performed them.
    gates["Q6_import_origin"] = {
        "note": "import-origin enforcement is the launcher's gate; not evaluated here",
    }
    gates["Q7_manifest_schema_version"] = {
        "note": (
            "the source manifest and the schema_version 2 receipt are the launcher's gate; not "
            "evaluated here"
        ),
    }
    # The half this tool really does verify: the checkpoint SHA-256, compared in run_audit before
    # anything was loaded.  Recomputed from the actual values rather than hardcoded True.
    checkpoint_sha_matches = checkpoint_sha == str(args.checkpoint_sha256).strip().lower()
    gates["Q7_checkpoint_sha"] = {
        "passed": bool(checkpoint_sha_matches),
        "checkpoint_sha256_matches": bool(checkpoint_sha_matches),
        "expected_checkpoint_sha256": str(args.checkpoint_sha256).strip().lower(),
        "observed_checkpoint_sha256": checkpoint_sha,
        "note": "verified here by re-comparing the digest of the file that was actually loaded",
    }

    summary = finalise_gate_block(payload, gates)
    failed = list(summary["evaluated_failed"]) + list(summary["malformed"])
    if failed:
        payload["verdict"] = VERDICT_FAIL_CLOSED
        payload["failed_gates"] = failed
        payload["note"] = (
            "One or more preregistered transform-quality gates failed. No policy statistics and no "
            "policy claim are reported (prereg section 5)."
        )
        raise GateFailure(failed, payload)

    # ---- forward passes -------------------------------------------------------------------
    player = build_player(checkpoint_path, schema, args.device, args.train_config)
    payload["equivalence_proof"] = equivalence_proof(player, checkpoint_state, device)
    payload["forward_contract"] = {
        "order": "mirror in raw units, then normalise inside model.norm_obs (navrl_players.py:155)",
        "normaliser": "running_mean_std restored from the checkpoint; never recomputed",
        "action_selection": "deterministic_actions (post-tanh) via NavRLPpoPlayerContinuous._model_action",
        "grad_mode": "torch.no_grad",
        "model_mode": "eval",
        "device": str(device),
        "batch_size": int(args.batch_size),
        "tf32": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "note": "replicates rl_games.torch_runner.Runner.__init__ on the live --play path",
        },
    }

    original_a, mirrored_a = forward_pairs(
        player, obs_all, args.batch_size, mirror_navrl_structured_observation
    )
    original_b, mirrored_b = forward_pairs(
        player, obs_all, args.batch_size, mirror_navrl_structured_observation
    )
    q8_passed = bool(torch.equal(original_a, original_b) and torch.equal(mirrored_a, mirrored_b))
    gates["Q8_determinism"] = {
        "passed": q8_passed,
        "original_actions_bitwise_identical": bool(torch.equal(original_a, original_b)),
        "mirrored_actions_bitwise_identical": bool(torch.equal(mirrored_a, mirrored_b)),
        "repeats": 2,
    }
    summary = finalise_gate_block(payload, gates)
    if not q8_passed:
        payload["verdict"] = VERDICT_FAIL_CLOSED
        payload["failed_gates"] = ["Q8_determinism"]
        raise GateFailure(["Q8_determinism"], payload)
    payload["failed_gates"] = list(summary["evaluated_failed"]) + list(summary["malformed"])

    # A NaN here would sail through np.percentile and land on INCONCLUSIVE without failing a gate.
    record_integrity(
        payload,
        integrity,
        "actions_finite",
        lambda: check_finite(
            {"original_actions": original_a, "mirrored_actions": mirrored_a},
            "deterministic actions (raw normaliser)",
        ),
    )

    # ---- measurements ---------------------------------------------------------------------
    outcome_by_frame = join_outcomes(frames["episode_uid"], episodes)
    valid_index = np.nonzero(valid)[0]
    valid_frames = dict(
        (key, np.asarray(value)[valid_index]) for key, value in frames.items() if key != "obs"
    )
    outcome_valid = outcome_by_frame[valid_index]
    contexts = build_contexts(valid_frames, outcome_valid)

    # ---- the join and the context families must reconcile before anything is measured -------
    outcome_join = record_integrity(
        payload, integrity, "outcome_join", lambda: check_outcome_join(outcome_valid, n_valid)
    )
    n_no_outcome_row = int(outcome_join["n_frames_no_outcome_row"])
    n_non_context_outcome = int(
        outcome_join["n_frames_with_outcome_row"] - outcome_join["outcome_context_cell_total"]
    )
    record_integrity(
        payload,
        integrity,
        "context_partition",
        lambda: check_context_partitions(
            contexts, n_valid, n_no_outcome_row, n_non_context_outcome
        ),
    )

    original_np = original_a.numpy()[valid_index]
    mirrored_np = mirrored_a.numpy()[valid_index]
    payload["measurements_raw_normaliser"] = measure_all(original_np, mirrored_np, contexts)

    # H6: characterise the exclusion from the data, every run, rather than relying on memory.
    outcome_join["excluded_frames"] = characterise_excluded_frames(
        valid_frames.get("call_index"), outcome_valid == NO_OUTCOME_ROW
    )
    payload["frames"]["n_frames_no_outcome_row"] = n_no_outcome_row
    payload["frames"]["outcome_join"] = outcome_join
    payload["frames"]["outcome_frame_counts"] = dict(
        (OUTCOME_CODES[code], int((outcome_valid == code).sum()))
        for code in sorted(OUTCOME_CODES)
    )
    payload["frames"]["outcome_frame_counts_note"] = (
        "keys are OUTCOME_CODES, so 'unattributed' here is outcome code 5 (a finished episode the "
        "dump could not attribute) and NOT the join sentinel. These counts cover only the "
        "%d frames that have an outcome row; %d further valid frames have no row at all "
        "(frames.n_frames_no_outcome_row). The denominator for outcome shares is "
        "frames.outcome_join.n_frames_with_outcome_row, not the sum of this dict."
        % (int(outcome_join["n_frames_with_outcome_row"]), n_no_outcome_row)
    )
    caveat = (outcome_join["excluded_frames"] or {}).get("caveat")
    payload["generated_caveats"] = (
        [caveat] if caveat and outcome_join["excluded_frames"].get("concentrated_in_tail") else []
    )

    # ---- S1: exploratory, NON-GATING normaliser-symmetry decomposition ---------------------
    source, sign = preregistered_signed_permutation(
        schema["cfg_lidar_hbeams"], schema["cfg_lidar_vbeams"], schema["cfg_max_obstacles"]
    )
    sym_player = build_player(checkpoint_path, schema, args.device, args.train_config)
    sym_rms = _running_mean_std(sym_player.model)
    sym_mean, sym_var = symmetrise_normaliser(sym_rms.running_mean, sym_rms.running_var, source, sign)
    with torch.no_grad():
        sym_rms.running_mean.copy_(sym_mean)
        sym_rms.running_var.copy_(sym_var)
    sym_original, sym_mirrored = forward_pairs(
        sym_player, obs_all, args.batch_size, mirror_navrl_structured_observation
    )
    record_integrity(
        payload,
        integrity,
        "s1_actions_finite",
        lambda: check_finite(
            {"original_actions": sym_original, "mirrored_actions": sym_mirrored},
            "deterministic actions (symmetrised normaliser, S1)",
        ),
    )
    payload["s1_symmetrised_normaliser"] = {
        "gating": False,
        "exploratory": True,
        "purpose": (
            "prereg section 6 S1: decompose measured chirality into network vs normaliser "
            "asymmetry. e_sym ~ e_raw means the network carries it; e_sym << e_raw means the "
            "normaliser statistics do. NOT used for any verdict (prereg L2)."
        ),
        "symmetrisation": {
            "variance": "v'_j = v'_{p(j)} = (v_j + v_{p(j)}) / 2 (pairwise average; variance is sign-blind)",
            "mean": "m'_j = (m_j + s_j * m_{p(j)}) / 2",
            "reasoning": (
                "The mirror is a signed permutation (M x)_j = s_j x_{p(j)} with p an involution "
                "and s_{p(j)} = s_j. Requiring the normaliser to COMMUTE with it, N(Mx) = M N(x), "
                "forces v_j = v_{p(j)} and m_j = s_j m_{p(j)}. The clamp to [-5, 5] is odd on a "
                "symmetric interval and imposes no further condition. For a sign-flipped field the "
                "mirror maps the index to ITSELF with s = -1, so the mean formula collapses to "
                "m'_j = (m_j - m_j)/2 = 0: a coordinate that must change sign under the symmetry "
                "can only have mean zero. Plain averaging would have left m_j unchanged there and "
                "removed nothing."
            ),
            "dtype": "float64 (the dtype rl_games stores running_mean_std in)",
            "sign_flipped_indices_forced_to_zero_mean": int(sum(1 for s in sign if s == -1)),
        },
        "measurements": measure_all(
            sym_original.numpy()[valid_index], sym_mirrored.numpy()[valid_index], contexts
        ),
    }

    overall = payload["measurements_raw_normaliser"][CONTEXT_OVERALL_CELL]
    verdict, no_verdict_reason = overall_verdict(overall)
    payload["verdict"] = verdict
    payload["verdict_basis"] = {
        "cell": CONTEXT_OVERALL_CELL,
        "median_conj_err_lat": overall["channels"]["conj_err_lat"]["median"],
        "lateral_sign_agreement": overall["lateral_sign_agreement"],
        "comparable_rows": overall["lateral_sign_comparable"],
        "insufficient_sample": overall["insufficient_sample"],
        "verdict_assigned": overall["verdict"] is not None,
        "no_verdict_reason": no_verdict_reason,
    }
    payload["p2_verdict_changed"] = False
    payload["d1_verdict_changed"] = False
    payload["p3_unlocked"] = False
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", required=True, help="NAVRL_OBS_DUMP .npz of real observations")
    parser.add_argument("--checkpoint", required=True, help="frozen policy checkpoint (.pth)")
    parser.add_argument(
        "--checkpoint-sha256", required=True, help="expected checkpoint SHA-256 (verified first)"
    )
    parser.add_argument("--frames-sha256", default="", help="optional expected frames SHA-256")
    parser.add_argument("--out", required=True, help="output summary JSON path")
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device for the forward passes ('auto' = cuda when available)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--train-config",
        default=str(DEFAULT_TRAIN_CONFIG),
        help="rl_games training YAML the checkpoint was produced with",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if str(args.device).strip().lower() == "auto":
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        payload = run_audit(args)
    except GateFailure as failure:
        write_json(args.out, failure.payload)
        print(
            "[reflection-audit] %s | failed %s: %s -> %s"
            % (
                failure.payload.get("verdict", VERDICT_FAIL_CLOSED),
                failure.kind,
                ", ".join(failure.gates),
                args.out,
            ),
            file=sys.stderr,
        )
        return 3
    except AuditPreconditionError as exc:
        print("[reflection-audit] PRECONDITION FAILED | %s" % exc, file=sys.stderr)
        return 2
    write_json(args.out, payload)
    print(
        "[reflection-audit] %s | median(conj_err_lat)=%s sign_agreement=%s -> %s"
        % (
            payload["verdict"],
            payload["verdict_basis"]["median_conj_err_lat"],
            payload["verdict_basis"]["lateral_sign_agreement"],
            args.out,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
