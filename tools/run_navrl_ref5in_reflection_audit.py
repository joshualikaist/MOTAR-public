#!/usr/bin/env python3
"""N1 real-frame reflection audit (eval-only) -- preregistered launcher.

Preregistration: docs/prereg_2026-08-21_n1_real_frame_reflection_audit.md (frozen).

Two temporally separated stages, exactly as prereg section 4 requires:

  1. rollout   -- one standard 70-bar evaluation cell of the frozen ref5in D1 checkpoint.  No
                  intervention of any kind touches the policy.  The task dumps real observation
                  rows, the context masks it already computes, and the per-episode outcome table
                  to ``NAVRL_OBS_DUMP``.
  2. offline   -- tools/navrl_reflection_offline_audit.py reloads the same frozen checkpoint and
                  forwards the ORIGINAL and MIRRORED observations from that npz.  No simulator is
                  created and no probe action is ever executed.  That separation IS the proof of
                  the "side-forward only" condition.

This experiment has NO decision authority.  It cannot revise the P2 STRICT FAIL or the D1 FAIL and
it cannot unlock P3.  The verdict thresholds live in the offline audit tool and are deliberately
NOT duplicated here.

Usage:
  python tools/run_navrl_ref5in_reflection_audit.py preflight
  python tools/run_navrl_ref5in_reflection_audit.py run
  python tools/run_navrl_ref5in_reflection_audit.py finalize
  python tools/run_navrl_ref5in_reflection_audit.py verify
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
# value that appears here; every check compares against these names.
# ----------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "aerial_gym/rl_training/rl_games"
EVALUATOR = RL_ROOT / "eval_navrl_v2_density_sweep.sh"
OFFLINE_AUDIT = ROOT / "tools/navrl_reflection_offline_audit.py"
IMPORT_ORIGIN_GUARD = RL_ROOT / "navrl_import_origin.py"
OBS_DUMP_HOOK = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"

SEED = 373                      # fresh, exhaustive-search usage count 0
BARS = 70
EPISODES = 1024
OUTPUT = ROOT / "results" / "navrl_ref5in_reflection_audit_seed373"
SOURCE_BUNDLE = OUTPUT / "source_bundle"
CELL = "reflection_audit"
CHECKPOINT_SHA = "197ea26999d6bb9cf23c4e5a55acbe945f89985e2384687d60ab1dbae66a278e"
CHECKPOINT_REL = (
    "aerial_gym/rl_training/rl_games/runs/"
    "ppo_260813_1636_navrl_v2-ref5in-d1-q3-adapt-s197/nn/"
    "last_gen_ppo_ep_1900_rew_182.11377.pth"
)
OBS_DUMP_STRIDE = 1
OBS_DUMP_MAX = 16384            # rows; streaming decimation keeps the sample uniform (prereg 3-b)
MIN_FRAMES = 4096               # prereg 5 Q9

PREREGISTRATION = "docs/prereg_2026-08-21_n1_real_frame_reflection_audit.md"
PRODUCER = "tools/run_navrl_ref5in_reflection_audit.py"
SCOPE = "n1_real_frame_reflection_audit_frozen_ref5in_seed373"
ROBOT_NAME = "navrl_ref5in_quad"
ACTION_SELECTION = "deterministic"
REFLECTION_MODE = "original"
SPEED_GOVERNOR_MODE = "off"
# Forced by the checkpoint contract, not chosen: frozen ref5in D1 ep1900 was trained at
# 22.5-28 m and the generic evaluator refuses any other goal band (prereg 3-d).  Pinned here
# so that a drifted band is a verification failure and not a silently different cell.
GOAL_DIST_MIN_M = 22.5
GOAL_DIST_MAX_M = 28.0
VERDICT_FAIL_CLOSED = "FAIL_CLOSED_TRANSFORM_QUALITY"
# Gates the offline audit deliberately does not decide: it records them with this launcher
# named as their owner.  Each name maps to the verify_cell() key carrying the launcher's own
# proof that it ran the check.  A delegated gate with no such proof is a hole in the tally,
# so build_summary() fails closed on it rather than reporting "0 failed".
# The offline audit owns the spelling of its own gate table, so both the name the frozen seed-373
# report uses and the name the current tool uses map to the same evidence; an unrecognised
# delegated gate is a check nobody performed and fails closed.
LAUNCHER_OWNED_GATES = {
    "Q6_import_origin": "import_origin",
    "Q7_checkpoint_sha_and_manifest": "manifest_provenance",
    "Q7_manifest_schema_version": "manifest_provenance",
}
GATE_STATUS_DELEGATED = "delegated"     # mirrors navrl_reflection_offline_audit.py's gate states

# The `run` log line the fail-closed import-origin guard prints (runner.py:31-33).  A MISSING line
# means the guard never ran, which is itself a failure -- see verify_import_origin().
#
# The tree this line must name is NOT the verifier's own tree.  Q6 asserts one relation internal to
# the artifact: the aerial_gym that EXECUTED is the tree whose bytes the source manifest hashed.
# Compiling the pattern against ROOT silently added a second, different claim -- "...and the
# verifier happens to be standing in that tree" -- which makes a result produced in a git worktree
# unverifiable from anywhere else, including after the worktree is deleted.  The expected path is
# therefore derived per artifact from the manifest's repository_root; see verify_import_origin().
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

# Prereg section 2.  Transcribed, not summarised: the limitations travel with the numbers.
LIMITATIONS = [
    "L1: 관측 반사는 세계 반사의 근사다. 장애물 토큰은 부호만 뒤집고 순서를 재배열하지 않는다"
    " (ppo_update_safety.py:398-403). mode-probe가 이 구멍을 slot_permutation_max_abs = 0.0078로"
    " 무시 가능함을 보였으며, 본 실험은 그 값을 재측정하지 않고 인용한다.",
    "L2: 정규화기(running_mean_std)는 좌우 대칭이 아니다. 플레이어는 preproc(M o) 순서로 적용하며"
    " (navrl_players.py:155) 이는 물리적으로 옳지만, 측정된 chirality의 일부는 네트워크가 아니라"
    " 정규화기 통계의 비대칭에서 올 수 있다. 보조 측정 S1로 분해하되 판정에는 쓰지 않는다.",
    "L3: 이 실험은 outcome을 측정하지 않는다. chirality가 있어도 대칭 아레나에서는 성능 손실이 없을 수"
    " 있다(2026-08-02). 따라서 어떤 결과도 'chirality가 성능을 해친다'는 주장의 근거가 될 수 없다.",
    "L4: 단일 checkpoint·단일 조건. 70막대 1셀, seed 1개. 계보 전반이나 밀도 전반으로"
    " 일반화하지 않는다.",
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
    "condition",
    "checkpoint",
    "checkpoint_sha256",
    "frames_sha256",
    "frames_count",
    "frames_valid_count",
    "stride_eff",
    "decimations",
    "quality_gates",
    "verdict",
    "verdict_basis",
    "failed_gates",
    "measurements",
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


P2 = load_module("reflection_audit_p2", ROOT / "tools/attest_navrl_ref5in_p2.py")
# Reuse -- never re-implement -- the audited contract helpers and the dirty-runtime gate.
BASE = load_module(
    "reflection_audit_base", ROOT / "tools/run_navrl_ref5in_cv_heading_diagnostic.py"
)

ContractError = BASE.ContractError
require = BASE.require
load_json = BASE.load_json
verify_runtime_clean_manifest = BASE.verify_runtime_clean_manifest


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


def cell_dir() -> Path:
    return OUTPUT / "cells" / CELL


def cell_paths() -> dict:
    directory = cell_dir()
    return {
        "result": directory / ("%dbars.json" % BARS),
        "receipt": directory / ("%dbars.receipt.json" % BARS),
        "log": directory / ("%dbars.log" % BARS),
        "snapshot": directory / "checkpoint_snapshot.pth",
        "frames": directory / "frames.npz",
        "audit": directory / "reflection_audit.json",
        "stdout_log": directory / "reflection_audit.log",
    }


SUMMARY_JSON = OUTPUT / "summary.json"
SUMMARY_MD = OUTPUT / "summary.md"


def resolve_recorded_path(recorded, label: str) -> Path:
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
    require(bool(raw), f"receipt records no {label}")
    recorded_path = Path(raw)
    local = cell_dir() / recorded_path.name
    if local.exists():
        return local.resolve()
    absolute = recorded_path if recorded_path.is_absolute() else (cell_dir() / recorded_path)
    if absolute.exists():
        return absolute.resolve()
    raise ContractError(
        f"{label} not found; neither the cell-local copy {local} nor the recorded path "
        f"{absolute} exists"
    )


# ----------------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------------


def canonical_env(*, preflight: bool) -> dict:
    """P2's closed environment set plus exactly the prereg section 3 additions.

    PYTHONPATH re-injection is the single least obvious line in this file, so the reasoning is
    written out.  ``P2.canonical_env`` DELETES PYTHONPATH from the child environment
    (attest_navrl_ref5in_p2.py:245) to keep the P2 environment closed.  But play_navrl.sh:19 cds
    into ``<worktree>/aerial_gym/rl_training/rl_games``, where no ``aerial_gym/`` package directory
    exists.  With no PYTHONPATH, the PEP 660 editable-install finder -- which hard-codes the
    PRIMARY worktree's absolute path and is consulted after PathFinder finds nothing -- wins, and
    ``import aerial_gym`` resolves to
    /home/fair/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/__init__.py.  This was
    measured, not assumed (prereg section 3-c, row 3).  Without the re-injection this run would
    EXECUTE the primary tree (currently dirty with unrelated WIP) while its source manifest hashes
    THIS worktree's bytes: a receipt that is internally consistent and describes code that never
    ran.  The re-injection must therefore happen AFTER canonical_env, or it is simply deleted.
    NAVRL_REQUIRE_SOURCE_ROOT below is the independent fail-closed check on the same fact (Q6).
    """
    env = P2.canonical_env(cell_dir(), preflight=preflight)
    paths = cell_paths()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "NAVRL_SEED": str(SEED),
            "NAVRL_V2_DENSITIES": str(BARS),
            # Fixed by the checkpoint contract, not by choice: frozen ref5in D1 ep1900 was
            # trained at 22.5-28 m and the generic evaluator refuses any other goal band
            # (cfg_general_goal_dist_min). Same values as the mode-probe host cell. Prereg 3-d.
            "NAVRL_V2_GOAL_DIST_MIN": "22.5",
            "NAVRL_V2_GOAL_DIST_MAX": "28",
            "NAVRL_V2_RESULT_DIR": str(cell_dir()),
            "NAVRL_V2_SHARED_SOURCE_BUNDLE": str(SOURCE_BUNDLE),
            "NAVRL_OBS_DUMP": str(paths["frames"]),
            "NAVRL_OBS_DUMP_STRIDE": str(OBS_DUMP_STRIDE),
            "NAVRL_OBS_DUMP_MAX": str(OBS_DUMP_MAX),
            "NAVRL_REQUIRE_SOURCE_ROOT": str(ROOT),
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
    return env


def offline_env() -> dict:
    """Environment for the offline audit subprocess.

    The audit derives its own schema environment from the checkpoint (apply_schema_environment),
    so every inherited NAVRL_* variable is dropped rather than pinned -- pinning them here would
    let a shell export silently disagree with the checkpoint.  PYTHONPATH is set for the same
    editable-finder reason as above; the audit additionally stands in CPU-only packages itself.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("NAVRL_") and key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    env.update({"PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1"})
    return env


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
    require(OFFLINE_AUDIT.is_file(), f"offline audit tool missing: {OFFLINE_AUDIT}")
    require(IMPORT_ORIGIN_GUARD.is_file(), f"import-origin guard missing: {IMPORT_ORIGIN_GUARD}")
    require(OBS_DUMP_HOOK.is_file(), f"navrl task missing: {OBS_DUMP_HOOK}")
    hook = OBS_DUMP_HOOK.read_text(encoding="utf-8")
    for literal in (
        'os.environ.get("NAVRL_OBS_DUMP", "")',
        'os.environ.get("NAVRL_OBS_DUMP_STRIDE"',
        'os.environ.get("NAVRL_OBS_DUMP_MAX"',
        "stride_eff=np.int64(self._obs_dump_stride_eff)",
        "decimations=np.int64(self._obs_dump_decimations)",
    ):
        require(literal in hook, f"NAVRL_OBS_DUMP hook missing: {literal}")
    runner = (RL_ROOT / "runner.py").read_text(encoding="utf-8")
    require(
        "[origin] aerial_gym %s sha256=%s (enforced)" in runner,
        "runner.py no longer prints the enforced import-origin line that Q6 verifies",
    )
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


def preflight_evaluator() -> None:
    """Assert the run needs NO provenance override at all.

    This experiment sets no target-pattern and no CV-heading intervention, so the checkpoint's
    cfg_target_pattern must already match what the generic evaluator expects.  Return code 2 means
    it does not.  In that case we report the mismatch lines and STOP -- we never force.
    """
    completed = subprocess.run(
        ["bash", str(EVALUATOR), str(CHECKPOINT), str(EPISODES)],
        cwd=str(ROOT),
        env=canonical_env(preflight=True),
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        mismatch_lines = [
            line.strip()
            for line in completed.stdout.splitlines()
            if "checkpoint=" in line and "expected=" in line
        ]
        tail = completed.stdout.strip().splitlines()[-12:]
        raise ContractError(
            "generic evaluator preflight did not pass cleanly "
            f"(returncode={completed.returncode}); this run is preregistered to need NO "
            f"provenance override, so it stops here instead of forcing. "
            f"mismatch lines: {mismatch_lines or 'none'} | tail: {tail}"
        )
    require(
        "[eval_v2] PREFLIGHT PASS (evaluation not started)" in completed.stdout,
        "evaluator preflight returned 0 without the PREFLIGHT PASS marker",
    )
    require(
        "NAVRL_V2_FORCE" not in canonical_env(preflight=True),
        "provenance override leaked into the canonical environment",
    )


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
    require(returncode == 0, f"evaluator rollout failed with exit code {returncode}")


def run_rollout() -> None:
    paths = cell_paths()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    staged_log = OUTPUT / "reflection_audit.log.partial"
    tee_run(
        ["bash", str(EVALUATOR), str(CHECKPOINT), str(EPISODES)],
        canonical_env(preflight=False),
        staged_log,
    )
    require(cell_dir().is_dir(), "evaluator produced no result directory")
    staged_log.replace(paths["stdout_log"])
    require(paths["frames"].is_file(), f"NAVRL_OBS_DUMP wrote no frames: {paths['frames']}")


def run_offline_audit() -> None:
    paths = cell_paths()
    frames_sha = P2.sha256_file(paths["frames"])
    completed = subprocess.run(
        [
            str(P2.PYTHON),
            str(OFFLINE_AUDIT),
            "--frames", str(paths["frames"]),
            "--checkpoint", str(CHECKPOINT),
            "--checkpoint-sha256", CHECKPOINT_SHA,
            "--frames-sha256", frames_sha,
            "--out", str(paths["audit"]),
        ],
        cwd=str(ROOT),
        env=offline_env(),
        check=False,
    )
    # 0 = audit completed with a verdict; 3 = a preregistered transform-quality gate failed, which
    # writes a FAIL_CLOSED payload and is itself a legitimate recorded outcome (prereg section 5).
    # 2 = precondition failure (bad SHA / bad schema): nothing was measured, so it is fatal.
    require(
        completed.returncode in (0, 3),
        f"offline reflection audit failed with exit code {completed.returncode}",
    )
    require(paths["audit"].is_file(), f"offline audit wrote no report: {paths['audit']}")


# ----------------------------------------------------------------------------------------------
# Cell verification
# ----------------------------------------------------------------------------------------------


def verify_import_origin(mapping: dict, metadata: dict) -> dict:
    """Q6: prove from the run log that the executing aerial_gym IS the tree the manifest hashed.

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
        f"Q6: source manifest records no absolute repository_root: {recorded_root!r}",
    )
    repository_root = Path(recorded_root)
    expected_origin = repository_root / ORIGIN_MANIFEST_ENTRY
    pattern = origin_line_pattern(repository_root)

    paths = cell_paths()
    matches = []
    foreign = []
    with paths["log"].open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            text = line.rstrip("\r\n")
            match = pattern.match(text)
            if match is not None:
                matches.append(match.group("sha256"))
            elif text.startswith(ORIGIN_LOG_MARKER):
                foreign.append(text)
    require(
        bool(matches),
        f"Q6: the run log contains no enforced [origin] line for {expected_origin}; "
        "the import-origin guard did not run",
    )
    require(
        not foreign,
        "Q6: the run log names an aerial_gym origin that is not the manifest's repository_root "
        f"{repository_root}: {foreign[:4]}",
    )
    require(len(set(matches)) == 1, f"Q6: conflicting [origin] digests in the run log: {set(matches)}")
    origin_sha = matches[0]
    entry = mapping.get(ORIGIN_MANIFEST_ENTRY)
    require(
        entry is not None,
        f"Q6: manifest has no runtime_files entry for {ORIGIN_MANIFEST_ENTRY}",
    )
    require(
        entry[0] == origin_sha,
        f"Q6: executed aerial_gym/__init__.py sha256 {origin_sha} is not the manifest digest "
        f"{entry[0]}",
    )
    return {
        "enforced": True,
        "required_source_root": str(repository_root),
        "origin": str(expected_origin),
        "origin_sha256": origin_sha,
        "manifest_entry": ORIGIN_MANIFEST_ENTRY,
        "manifest_sha256": entry[0],
        "log_line_occurrences": len(matches),
        "pythonpath_reinjected": str(repository_root),
    }


def require_import_origin_evidence(import_origin) -> None:
    """Q6 is delegated to this launcher; this is the proof the launcher actually ran it.

    verify_import_origin() is the only producer of this shape.  If it were removed -- or never
    called -- the summary would otherwise still print "0 failed" while nobody had judged Q6.
    """
    require(
        isinstance(import_origin, dict)
        and import_origin.get("enforced") is True
        and import_origin.get("manifest_entry") == ORIGIN_MANIFEST_ENTRY
        and re.fullmatch(r"[0-9a-f]{64}", str(import_origin.get("origin_sha256", ""))) is not None
        and import_origin.get("origin_sha256") == import_origin.get("manifest_sha256")
        and int(import_origin.get("log_line_occurrences") or 0) >= 1,
        "Q6_import_origin is delegated to this launcher, but the launcher produced no proof it "
        f"ran the check (verify_import_origin evidence: {import_origin!r})",
    )


def gate_is_delegated(gate) -> bool:
    """True when the offline audit did NOT decide this gate itself.

    Three independent signals, because the report's shape belongs to the offline tool and one
    missing signal must never quietly turn a delegated gate into a passing one: no boolean
    verdict at all, an explicit delegated status, or an owner that is somebody else -- this
    launcher included.
    """
    table = gate if isinstance(gate, dict) else {}
    owners = {str(table.get("owner", "")), str((table.get("delegation") or {}).get("owner", ""))}
    return (
        not isinstance(table.get("passed"), bool)
        or table.get("status") == GATE_STATUS_DELEGATED
        or bool(owners & {"launcher", PRODUCER})
    )


def classify_gates(gates: dict) -> tuple:
    """Split the offline gate table into gates it DECIDED and gates it delegated elsewhere.

    ``len(gates)`` counts dictionary keys, which is true of any table and therefore reports
    nothing.  A delegated gate carries no verdict from that report and may never be tallied as if
    it did -- that is exactly how such a gate can never be counted as failed.
    """
    delegated = sorted(name for name, gate in gates.items() if gate_is_delegated(gate))
    return sorted(set(gates) - set(delegated)), delegated


def gate_tally(payload: dict) -> tuple:
    """(evaluated, delegated, failed) for the summary line -- with the delegation proved."""
    gates = payload.get("quality_gates") or {}
    evaluated, delegated = classify_gates(gates)
    require_import_origin_evidence(payload.get("import_origin"))
    return evaluated, delegated, list(payload.get("failed_gates") or [])


def verify_cell() -> dict:
    paths = cell_paths()
    for key in ("result", "receipt", "log", "snapshot", "frames", "audit"):
        require(paths[key].is_file(), f"missing reflection-audit artifact: {paths[key]}")
    result = load_json(paths["result"])
    receipt = load_json(paths["receipt"])

    require(
        P2.sha256_file(paths["result"]) == receipt.get("result_sha256"),
        "result/receipt hash mismatch",
    )
    require(
        P2.sha256_file(paths["snapshot"]) == CHECKPOINT_SHA,
        "evaluated checkpoint snapshot is not the pinned checkpoint",
    )
    require(
        receipt.get("source_checkpoint_sha256") == CHECKPOINT_SHA,
        "receipt source checkpoint mismatch",
    )

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
    require(not receipt_mismatch, f"receipt condition mismatch: {receipt_mismatch}")

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
    require(not condition_mismatch, f"result condition mismatch: {condition_mismatch}")

    # The evaluator drains whole 128-env batches, so a cell finishes at or just past the request.
    # Exact equality is WRONG here (see run_navrl_ref5in_camera_range_control.py:162).
    actual = int(result.get("actual_episodes", -1))
    require(
        int(result.get("requested_episodes", -1)) == EPISODES and actual >= EPISODES,
        f"episode contract mismatch: requested={result.get('requested_episodes')} actual={actual}",
    )

    # Q7's other half, delegated to this launcher by the offline audit: the schema-2 receipt and
    # the manifest it names.  Both files are located by resolve_recorded_path -- the recorded
    # absolute paths belong to the producing worktree, and a migrated artifact must still verify --
    # and both are then pinned to the digests the receipt itself recorded, so the location may move
    # but the bytes may not.
    require(receipt.get("schema_version") == 2, "receipt is not a schema_version 2 receipt")
    manifest = resolve_recorded_path(
        receipt.get("runtime_source_manifest"), "runtime source manifest"
    )
    require(
        P2.sha256_file(manifest) == receipt.get("runtime_source_manifest_sha256"),
        f"runtime source manifest bytes differ from the receipt digest: {manifest}",
    )
    python_environment = resolve_recorded_path(
        receipt.get("python_environment_manifest"), "python environment manifest"
    )
    require(
        P2.sha256_file(python_environment) == receipt.get("python_environment_manifest_sha256"),
        f"python environment manifest bytes differ from the receipt digest: {python_environment}",
    )
    mapping, metadata = P2.manifest_map(manifest, 2, require_original=False)
    verify_runtime_clean_manifest(metadata, CELL)
    import_origin = verify_import_origin(mapping, metadata)
    manifest_provenance = {
        "checked_by_launcher": True,
        "receipt_schema_version": receipt.get("schema_version"),
        "manifest_schema_version": metadata.get("schema_version"),
        "manifest_sha256": receipt.get("runtime_source_manifest_sha256"),
        "python_environment_manifest_sha256": receipt.get("python_environment_manifest_sha256"),
        "runtime_file_count": metadata.get("runtime_file_count"),
        "runtime_clean_verified": True,
    }

    frames_sha = P2.sha256_file(paths["frames"])
    audit = load_json(paths["audit"])
    require(
        audit.get("frames_sha256") == frames_sha,
        "offline audit consumed different frame bytes than the rollout wrote: "
        f"{audit.get('frames_sha256')} != {frames_sha}",
    )
    require(
        audit.get("checkpoint_sha256") == CHECKPOINT_SHA,
        "offline audit ran on a different checkpoint",
    )
    require(
        audit.get("preregistration") == PREREGISTRATION,
        "offline audit is bound to a different preregistration",
    )
    require(audit.get("decision_authority") == "none", "offline audit authority drift")
    require(audit.get("simulator_created") is False, "offline audit created a simulator")
    require(audit.get("environment_stepped") is False, "offline audit stepped an environment")

    frames_block = audit.get("frames") or {}
    frames_valid = int(frames_block.get("n_frames_valid", -1))
    require(
        frames_valid >= MIN_FRAMES,
        f"valid frame count {frames_valid} is below the preregistered minimum {MIN_FRAMES}",
    )

    return {
        "result": result,
        "receipt": receipt,
        "audit": audit,
        "frames_sha256": frames_sha,
        "import_origin": import_origin,
        "manifest_provenance": manifest_provenance,
        "actual_episodes": actual,
        "condition": condition,
    }


# ----------------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------------


def build_summary(cell: dict) -> dict:
    audit = cell["audit"]
    condition = cell["condition"]
    frames_block = audit.get("frames") or {}
    provenance = (audit.get("npz_key_binding") or {}).get("dump_provenance") or {}
    verdict = audit.get("verdict")
    require(isinstance(verdict, str) and verdict, "offline audit reported no verdict")
    # Pass-through only.  Under FAIL_CLOSED the offline tool reports no policy statistics at all
    # (prereg section 5), so there is nothing to carry and the key is explicitly null.  That is a
    # BICONDITIONAL, and it is enforced here rather than merely described: a FAIL_CLOSED verdict
    # carrying measurements would publish statistics the prereg says were never computed, and a
    # null measurements key under any other verdict would publish a verdict with nothing behind it.
    measurements = audit.get("measurements_raw_normaliser")
    require(
        (verdict == VERDICT_FAIL_CLOSED) == (measurements is None),
        f"fail-closed contract violated: verdict={verdict} with measurements="
        + ("null" if measurements is None else "present")
        + f"; {VERDICT_FAIL_CLOSED} requires a null measurements key and a null measurements key "
        f"requires {VERDICT_FAIL_CLOSED}",
    )

    # The gate tally must not be a tautology.  Three separate facts are established before the
    # summary may report a count: every gate without a boolean verdict is one this launcher is
    # named the owner of, this launcher really ran each of those checks, and the offline report's
    # failed_gates list agrees with the per-gate verdicts it published.
    gates = audit.get("quality_gates") or {}
    evaluated_gates, delegated_gates = classify_gates(gates)
    unowned = [name for name in delegated_gates if name not in LAUNCHER_OWNED_GATES]
    require(
        not unowned,
        f"quality gates {unowned} carry no boolean verdict and name no owner; nobody judged them",
    )
    for evidence_key in sorted(set(LAUNCHER_OWNED_GATES.values())):
        owned = sorted(name for name, key in LAUNCHER_OWNED_GATES.items() if key == evidence_key)
        require(
            any(name in gates for name in owned),
            f"the offline audit report lists none of {owned}, so the launcher-owned check "
            f"{evidence_key} answers a question this report never asked",
        )
        evidence = cell.get(evidence_key)
        require(
            isinstance(evidence, dict) and evidence,
            f"quality gates {owned} are delegated to this launcher, but verify_cell() produced no "
            f"{evidence_key} evidence that the launcher ran the check",
        )
    require_import_origin_evidence(cell.get("import_origin"))
    provenance_evidence = cell["manifest_provenance"]
    require(
        provenance_evidence.get("checked_by_launcher") is True
        and provenance_evidence.get("receipt_schema_version") == 2
        and provenance_evidence.get("manifest_schema_version") == 2
        and provenance_evidence.get("runtime_clean_verified") is True,
        "Q7_checkpoint_sha_and_manifest delegates the manifest/schema half to this launcher, but "
        f"the launcher's evidence is incomplete: {provenance_evidence!r}",
    )
    failing = {
        name for name, gate in gates.items() if isinstance(gate, dict) and gate.get("passed") is False
    }
    recorded_failed = set(audit.get("failed_gates") or [])
    require(
        failing <= recorded_failed,
        f"quality gates {sorted(failing - recorded_failed)} report passed=false but are missing "
        "from failed_gates",
    )
    require(
        recorded_failed <= set(gates),
        f"failed_gates names gates that are not in the gate table: {sorted(recorded_failed - set(gates))}",
    )
    passing = {
        name for name, gate in gates.items() if isinstance(gate, dict) and gate.get("passed") is True
    }
    require(
        not (recorded_failed & passing),
        f"failed_gates names gates that report passed=true: {sorted(recorded_failed & passing)}",
    )

    paths = cell_paths()
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
        "condition": {
            "seed": SEED,
            "bars": BARS,
            "requested_episodes": EPISODES,
            "actual_episodes": cell["actual_episodes"],
            "robot_name": ROBOT_NAME,
            "action_selection": ACTION_SELECTION,
            "reflection_mode": REFLECTION_MODE,
            "speed_governor_mode": SPEED_GOVERNOR_MODE,
            "episode_len_steps": condition.get("episode_len_steps"),
            "num_envs": condition.get("num_envs"),
            "obs_dump_stride_requested": OBS_DUMP_STRIDE,
            "obs_dump_max_rows": OBS_DUMP_MAX,
            "min_frames_gate": MIN_FRAMES,
            "generic_provenance_override_used": False,
        },
        "checkpoint": CHECKPOINT_REL,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "frames_sha256": cell["frames_sha256"],
        "frames_count": int(frames_block.get("n_frames_total", -1)),
        "frames_valid_count": int(frames_block.get("n_frames_valid", -1)),
        "stride_eff": provenance.get("stride_eff"),
        "decimations": provenance.get("decimations"),
        "quality_gates": audit.get("quality_gates"),
        "verdict": verdict,
        "verdict_basis": audit.get("verdict_basis"),
        "failed_gates": audit.get("failed_gates"),
        "measurements": measurements,
        "import_origin": cell["import_origin"],
        "limitations": list(LIMITATIONS),
        "sources": {
            "evaluation_result": str(paths["result"].relative_to(ROOT)),
            "evaluation_receipt": str(paths["receipt"].relative_to(ROOT)),
            "evaluation_log": str(paths["log"].relative_to(ROOT)),
            "rollout_stdout_log": str(paths["stdout_log"].relative_to(ROOT)),
            "frames_npz": str(paths["frames"].relative_to(ROOT)),
            "offline_audit": str(paths["audit"].relative_to(ROOT)),
        },
    }


def _fmt(value) -> str:
    return "n/a" if value is None else ("%.4f" % value if isinstance(value, float) else str(value))


def write_summary(payload: dict) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    basis = payload.get("verdict_basis") or {}
    measurements = payload.get("measurements") or {}
    overall = (measurements.get("overall") or {}) if isinstance(measurements, dict) else {}
    channels = overall.get("channels") or {}
    lat = channels.get("conj_err_lat") or {}
    # Not len(quality_gates): that counts keys, not verdicts.  gate_tally reports how many gates
    # the offline audit actually decided, how many it delegated to this launcher (and re-proves
    # that the launcher judged them), and how many failed.
    evaluated_gates, delegated_gates, failed = gate_tally(payload)
    lines = [
        "# N1 real-frame reflection audit (seed 373)",
        "",
        f"**판정: `{payload['verdict']}`**",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| median(conj_err_lat) | {_fmt(basis.get('median_conj_err_lat'))} |",
        f"| lateral sign agreement | {_fmt(basis.get('lateral_sign_agreement'))} |",
        f"| 비교가능 행 | {_fmt(basis.get('comparable_rows'))} |",
        f"| conj_err_lat p90 / p95 / p99 | "
        f"{_fmt(lat.get('p90'))} / {_fmt(lat.get('p95'))} / {_fmt(lat.get('p99'))} |",
        f"| 유효 프레임 / 전체 프레임 | {payload['frames_valid_count']:,} / "
        f"{payload['frames_count']:,} (게이트 {MIN_FRAMES:,}) |",
        f"| stride_eff / decimations | {_fmt(payload['stride_eff'])} / "
        f"{_fmt(payload['decimations'])} |",
        f"| 에피소드 (요청/실제) | {payload['condition']['requested_episodes']:,} / "
        f"{payload['condition']['actual_episodes']:,} |",
        f"| 품질 게이트 | 오프라인 판정 {len(evaluated_gates)}개, 런처 위임·검증 "
        f"{len(delegated_gates)}개, 실패 {len(failed)}개 |",
        "",
        f"- checkpoint SHA-256 `{payload['checkpoint_sha256'][:16]}…`, "
        f"frames.npz SHA-256 `{payload['frames_sha256'][:16]}…` (npz는 커밋하지 않는다)",
        f"- 실행된 `aerial_gym/__init__.py` sha256 "
        f"`{payload['import_origin']['origin_sha256'][:16]}…` = manifest 항목 (Q6 강제)",
        "- 조건: 70 bars, seed 373, deterministic, reflection_mode=original, "
        "speed_governor=off, provenance override 미사용",
        "",
        "## 권한",
        "",
        "이 실험은 **결정 권한이 없다**. P2 STRICT FAIL·D1 FAIL을 소급 변경하지 않으며 P3를 "
        "해제하지 않는다 (`p2_verdict_changed`/`d1_verdict_changed`/`p3_unlocked` 전부 false).",
        "",
        "## 한계 (사전등록 §2)",
        "",
    ]
    lines.extend("- " + item for item in payload["limitations"])
    lines.append("")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    require(
        mode in {"preflight", "run", "finalize", "verify"},
        "usage: run_navrl_ref5in_reflection_audit.py {preflight|run|finalize|verify}",
    )

    if mode == "preflight":
        verify_prerequisites(require_clean=False)
        require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
        preflight_evaluator()
        print(
            f"[reflection-audit] PREFLIGHT PASS | seed={SEED} bars={BARS} episodes={EPISODES} "
            "| no provenance override required"
        )
        return 0

    if mode == "run":
        require(not OUTPUT.exists(), f"refusing overwrite: {OUTPUT}")
        verify_prerequisites(require_clean=True)
        run_rollout()
        run_offline_audit()
        cell = verify_cell()
        print(
            f"[reflection-audit] RUN COMPLETE | verdict={cell['audit'].get('verdict')} | "
            "next: finalize"
        )
        return 0

    cell = verify_cell()
    expected = build_summary(cell)
    if mode == "finalize":
        write_summary(expected)
        print(f"[reflection-audit] FINALIZE PASS | {expected['verdict']} -> {SUMMARY_JSON}")
        return 0

    require(SUMMARY_JSON.is_file(), f"summary missing: {SUMMARY_JSON}")
    recorded = load_json(SUMMARY_JSON)
    for key in SUMMARY_VERIFY_KEYS:
        require(recorded.get(key) == expected.get(key), f"summary changed: {key}")
    print(f"[reflection-audit] VERIFY PASS | {recorded['verdict']}")
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
        print(f"[reflection-audit] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
