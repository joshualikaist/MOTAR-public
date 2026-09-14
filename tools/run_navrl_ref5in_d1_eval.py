#!/usr/bin/env python3
"""Run the preregistered held-out decision cell for the ref5in D1 adaptation probe.

The contract is fixed before D1 training: seed 331, at least 8,193 requested episodes, 70 bars,
the full 6..28 m goal distribution, deterministic/original actions and governor off.  D1 cannot
rewrite the failed P2 decision and this evaluator cannot unlock P3 automatically.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "aerial_gym/rl_training/rl_games"
RUN_GLOB = "*_navrl_v2-ref5in-d1-q3-adapt-s197"
P1C_RUN = "ppo_260813_0540_navrl_v2-ref5in-smoke-c-s197"
P1C_CHECKPOINT = RL_ROOT / "runs" / P1C_RUN / "nn/last_gen_ppo_ep_900_rew_137.08087.pth"
P1C_SHA = "f1670a1d74dd92cb00d6a58898e9cc1b96eb9cbe155d1e85812a345e7aaae6bf"
OUTPUT = ROOT / "results/navrl_ref5in_d1_eval_seed331"
CELL = OUTPUT / "ref5in"
SOURCE_BUNDLE = OUTPUT / "source_bundle"
BASELINE_OUTPUT = ROOT / "results/navrl_ref5in_outcome_diagnostic_v2_seed317"
SEED = 331
EPISODES = 8193
SOURCE_EPOCH = 900
TERMINAL_EPOCH = 1900
BRANCH_EPOCHS = TERMINAL_EPOCH - SOURCE_EPOCH


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_module("ref5in_d1_base", ROOT / "tools/run_navrl_ref5in_outcome_diagnostic.py")
V2 = load_module("ref5in_d1_v2", ROOT / "tools/run_navrl_ref5in_outcome_diagnostic_v2.py")
SMOKE = load_module("ref5in_d1_smoke", ROOT / "tools/analyze_navrl_ref5in_smoke.py")
P2 = load_module("ref5in_d1_p2", ROOT / "tools/attest_navrl_ref5in_p2.py")
ORIGINAL_CANONICAL_ENV = BASE.canonical_env


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BASE.ContractError(message)


def rate_difference(
    new_successes: int, new_total: int, old_successes: int, old_total: int
) -> dict:
    """Unpooled Wald interval for a descriptive difference of independent proportions."""
    require(new_total > 0 and old_total > 0, "rate-difference denominator must be positive")
    p_new = new_successes / new_total
    p_old = old_successes / old_total
    delta = p_new - p_old
    se = math.sqrt(
        p_new * (1.0 - p_new) / new_total + p_old * (1.0 - p_old) / old_total
    )
    z = 1.959963984540054
    return {
        "new_rate": p_new,
        "baseline_rate": p_old,
        "difference": delta,
        "difference_ci95": [delta - z * se, delta + z * se],
        "method": "unpooled_normal_approximation_independent_proportions",
    }


def compare_outcomes(new: dict, old: dict) -> dict:
    new_n = int(new["episodes"]) if "episodes" in new else sum(
        int(new[key]) for key in ("captured", "crash", "timeout")
    )
    old_n = int(old["episodes"]) if "episodes" in old else sum(
        int(old[key]) for key in ("captured", "crash", "timeout")
    )
    new_capture = int(new.get("captured", new.get("successes", -1)))
    old_capture = int(old.get("captured", old.get("successes", -1)))
    require(new_capture >= 0 and old_capture >= 0, "capture count missing from comparison cell")
    return {
        "capture": rate_difference(new_capture, new_n, old_capture, old_n),
        "crash": rate_difference(int(new["crash"]), new_n, int(old["crash"]), old_n),
        "timeout": rate_difference(
            int(new["timeout"]), new_n, int(old["timeout"]), old_n
        ),
    }


def configure_base(checkpoint: Path, checkpoint_sha: str) -> None:
    BASE.CHECKPOINT = checkpoint
    BASE.CHECKPOINT_SHA = checkpoint_sha
    BASE.OUTPUT = OUTPUT
    BASE.CELL = CELL
    BASE.SOURCE_BUNDLE = SOURCE_BUNDLE
    BASE.SEED = SEED
    BASE.EPISODES = EPISODES
    # The generic evaluator normally requires the checkpoint's TRAINING goal range to equal the
    # held-out EVALUATION range. D1 intentionally trains on q3 [22.5,28] then tests the full
    # [6,28] distribution.  Its only available bypass is broad, so verify_expected_rejection()
    # proves that this single preregistered field is the sole mismatch before any forced run.
    def d1_env(preflight: bool = False) -> dict[str, str]:
        env = ORIGINAL_CANONICAL_ENV(preflight)
        env["NAVRL_V2_FORCE"] = "1"
        return env

    BASE.canonical_env = d1_env


def verify_expected_rejection(checkpoint: Path) -> str:
    command = ["bash", str(BASE.EVALUATOR), str(checkpoint), str(EPISODES)]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=ORIGINAL_CANONICAL_ENV(preflight=True),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    expected = "cfg_general_goal_dist_min: checkpoint=22.5 expected=6.0"
    mismatch_lines = [line.strip() for line in completed.stdout.splitlines() if "checkpoint=" in line and "expected=" in line]
    require(completed.returncode == 2, f"generic evaluator unexpectedly accepted D1 checkpoint: {completed.returncode}")
    require(mismatch_lines == [expected], f"D1 forced-eval mismatch set is not exactly one field: {mismatch_lines}")
    return expected


def unique_d1_run() -> Path:
    candidates = sorted(path for path in (RL_ROOT / "runs").glob(RUN_GLOB) if path.is_dir())
    completed = []
    for path in candidates:
        summary_path = path / "aerial_run/run_summary.json"
        marker = path / ".aerial_training_finished"
        if not summary_path.is_file() or not marker.is_file():
            continue
        summary = BASE.load_json(summary_path)
        if summary.get("exit_reason") == "max_epochs" and int(summary.get("last_epoch", -1)) == TERMINAL_EPOCH:
            completed.append(path)
    require(
        len(completed) == 1,
        f"expected exactly one completed D1 run; candidates={[p.name for p in candidates]} "
        f"completed={[p.name for p in completed]}",
    )
    return completed[0]


def verify_training() -> dict:
    import torch

    run = unique_d1_run()
    marker = run / ".aerial_training_finished"
    run_summary = BASE.load_json(run / "aerial_run/run_summary.json")
    lineage = (run / "aerial_run/resumed_from.txt").read_text(encoding="utf-8").strip()
    checkpoint = SMOKE.latest_checkpoint(run, TERMINAL_EPOCH)
    checkpoint_sha = BASE.sha256_file(checkpoint)
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("env_state") or {}
    require(marker.is_file(), "D1 completion marker missing")
    require(run_summary.get("exit_reason") == "max_epochs", "D1 did not end at max_epochs")
    require(int(run_summary.get("first_epoch", -1)) == SOURCE_EPOCH + 1, "D1 branch first epoch mismatch")
    require(int(run_summary.get("last_epoch", -1)) == TERMINAL_EPOCH, "D1 terminal epoch mismatch")
    require(int(run_summary.get("epochs_logged", -1)) == BRANCH_EPOCHS, "D1 branch length mismatch")
    require(lineage == P1C_RUN, "D1 resumed_from lineage mismatch")
    require(int(payload.get("epoch", -1)) == TERMINAL_EPOCH, "D1 checkpoint epoch mismatch")
    require(int(payload.get("frame", -1)) == TERMINAL_EPOCH * 32 * 128, "D1 frame mismatch")
    expected_state = {
        "cfg_training_seed": 197,
        "cfg_training_num_envs": 128,
        "cfg_robot_name": "navrl_ref5in_quad",
        "cfg_obstacle_selector": "cluster_sector",
        "cfg_action_policy": "squashed_gaussian",
        "cfg_speed_governor_mode": "off",
        "cfg_perception_perturb": False,
        "cfg_general_goal_dist_min": 22.5,
        "cfg_general_goal_dist_max": 28.0,
        "n_bars_active": 70,
        "num_task_steps": TERMINAL_EPOCH * 32,
    }
    mismatches = {key: (state.get(key), value) for key, value in expected_state.items()
                  if state.get(key) != value}
    require(not mismatches, f"D1 checkpoint contract mismatch: {mismatches}")
    require(math.isclose(float(state.get("cfg_action_learning_rate", 0)), 1.5e-5, abs_tol=1e-12),
            "D1 initial learning rate mismatch")
    require(math.isclose(float(state.get("current_action_learning_rate", 0)), 1.5e-5, abs_tol=1e-12),
            "D1 current learning rate mismatch")
    nonfinite = SMOKE.all_tensors_finite(payload)
    require(not nonfinite, f"non-finite D1 checkpoint tensors: {nonfinite[:8]}")

    log = SMOKE.matching_log(run.name)
    outcome = SMOKE.outcome_window(log, expected_epochs=BRANCH_EPOCHS)
    tb = SMOKE.tensorboard_summary(run)
    require(not tb["all_scalars"]["empty_tags"] and not tb["all_scalars"]["nonfinite_tags"],
            "D1 TensorBoard contains empty/non-finite scalar tags")
    for tag in ("ppo/kl", "ppo/behavior_kl_audit_max"):
        require(tb[tag]["count"] == BRANCH_EPOCHS and tb[tag]["max"] < 0.04,
                f"D1 {tag} safety gate failed: {tb[tag]}")
    for tag in ("ppo/epoch_rollback_total", "ppo/kl_skipped_minibatches"):
        require(tb[tag]["count"] == BRANCH_EPOCHS and tb[tag]["max"] == 0.0,
                f"D1 {tag} is nonzero: {tb[tag]}")
    for axis in ("x", "y", "z", "yaw"):
        item = tb[f"policy_action/raw_oob_{axis}"]
        require(item["count"] == BRANCH_EPOCHS and item["max"] == 0.0,
                f"D1 raw OOB {axis} is nonzero: {item}")

    manifest = Path(str(state.get("cfg_training_source_manifest", ""))).resolve()
    manifest_sha = str(state.get("cfg_training_source_manifest_sha256", ""))
    source = SMOKE.verify_source_manifest(manifest, manifest_sha)
    require(source["git_dirty"] is False, "D1 runtime receipt was dirty")
    training_map, training_manifest = P2.manifest_map(manifest, 1)
    require(P2.current_runtime_map() == training_map, "current runtime differs from D1 training")
    require(training_manifest.get("python_environment_sha256"), "D1 Python receipt missing")
    return {
        "run": run,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha,
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "runtime_map": training_map,
        "python_environment_sha256": training_manifest["python_environment_sha256"],
        "tensorboard": tb,
        "last100_training_outcomes": outcome,
        "session_log": log,
    }


def verify_python_environment_identity(
    training_manifest: dict, training_manifest_path: Path,
    eval_manifest: dict, eval_manifest_path: Path,
) -> dict:
    train_path = training_manifest_path.parent / str(training_manifest["python_environment"])
    eval_path = eval_manifest_path.parent / str(eval_manifest["python_environment"])
    train_text = train_path.read_text(encoding="utf-8")
    eval_text = eval_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(?m)^-e git\+ssh://git@github\.com/joshualikaist/MOTAR\.git@"
        r"([0-9a-f]{40})#egg=aerial_gym$"
    )
    train_match = pattern.search(train_text)
    eval_match = pattern.search(eval_text)
    require(train_match is not None and eval_match is not None,
            "editable aerial_gym package receipt line missing")
    require(train_match.group(1) == training_manifest.get("git_commit"),
            "training editable-package commit differs from training manifest")
    require(eval_match.group(1) == eval_manifest.get("git_commit"),
            "evaluation editable-package commit differs from evaluation manifest")
    placeholder = "-e git+ssh://git@github.com/joshualikaist/MOTAR.git@<MANIFEST_COMMIT>#egg=aerial_gym"
    require(pattern.sub(placeholder, train_text) == pattern.sub(placeholder, eval_text),
            "D1 train/eval Python package environments differ beyond editable Git metadata")
    return {
        "exact_hash_match": training_manifest["python_environment_sha256"]
        == eval_manifest["python_environment_sha256"],
        "normalized_match": True,
        "allowed_difference": "editable aerial_gym VCS commit metadata only",
        "training_git_commit": train_match.group(1),
        "evaluation_git_commit": eval_match.group(1),
    }


def verify_eval_source(training: dict) -> dict:
    receipt = BASE.load_json(CELL / "70bars.receipt.json")
    manifest = Path(str(receipt.get("runtime_source_manifest", ""))).resolve()
    require(manifest == SOURCE_BUNDLE / "source_manifest.json", "non-canonical D1 eval source bundle")
    eval_map, eval_manifest = P2.manifest_map(manifest, 2)
    require(eval_map == training["runtime_map"], "D1 train/eval runtime byte map mismatch")
    training_manifest = BASE.load_json(training["manifest"])
    return verify_python_environment_identity(
        training_manifest, training["manifest"], eval_manifest, manifest
    )


def decision_summary(
    result: dict, training: dict, forced_mismatch: str, python_environment: dict
) -> dict:
    base = BASE.summarize(result)
    base["strata"]["distance_by_pattern"] = V2.enriched_joint(result)
    outcome = result["outcome"]
    q3 = base["strata"]["distance"]["q3"]
    q3_cv = base["strata"]["distance_by_pattern"]["q3"]["cv"]
    baseline = BASE.load_json(BASELINE_OUTPUT / "summary.json")
    baseline_raw = BASE.load_json(BASELINE_OUTPUT / "ref5in/70bars.json")
    comparisons = {
        "baseline_scope": baseline["scope"],
        "note": (
            "Descriptive D0-to-D1 comparison across different evaluation seeds. "
            "It combines the 1,000-epoch adaptation with its q3 training distribution."
        ),
        "global": compare_outcomes(outcome, baseline["outcome"]),
        "q3": compare_outcomes(q3, baseline["strata"]["distance"]["q3"]),
        "q3_cv": compare_outcomes(
            q3_cv, baseline["strata"]["distance_by_pattern"]["q3"]["cv"]
        ),
    }
    action = result["action"]
    baseline_action = baseline_raw["action"]
    action_chirality = {
        "baseline_negative_y_rate": baseline_action["negative_y_rate"],
        "d1_negative_y_rate": action["negative_y_rate"],
        "baseline_signed_mean_y": baseline_action["signed_mean_y"],
        "d1_signed_mean_y": action["signed_mean_y"],
        "baseline_sign_flip_y_rate": baseline_action["sign_flip_y_rate"],
        "d1_sign_flip_y_rate": action["sign_flip_y_rate"],
        "interpretation": (
            "Strong negative-y action chirality predates D1 and persists after adaptation; "
            "this comparison does not establish an outcome penalty."
        ),
    }
    gates = {
        "global_crash_lte_27pct": {
            "pass": float(outcome["crash_rate"]) <= 0.27,
            "value": float(outcome["crash_rate"]), "threshold": 0.27,
        },
        "q3_crash_lte_30pct": {
            "pass": float(q3["crash_rate"]) <= 0.30,
            "value": float(q3["crash_rate"]), "threshold": 0.30,
        },
        "q3_cv_timeout_lte_12pct": {
            "pass": float(q3_cv["timeout_rate"]) <= 0.12,
            "value": float(q3_cv["timeout_rate"]), "threshold": 0.12,
        },
    }
    passed = all(item["pass"] for item in gates.values())
    return {
        "schema_version": 2,
        "producer": "tools/run_navrl_ref5in_d1_eval.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "ref5in_d1_heldout_seed331_decision",
        "verdict": "PASS" if passed else "FAIL",
        "decision_authority": "D1 adaptation probe only",
        "p2_verdict_changed": False,
        "p3_automatically_unlocked": False,
        "checkpoint": str(training["checkpoint"].relative_to(ROOT)),
        "checkpoint_sha256": training["checkpoint_sha256"],
        "training_source_manifest_sha256": training["manifest_sha256"],
        "generic_provenance_override": {
            "used": True,
            "sole_verified_mismatch": forced_mismatch,
            "reason": "q3 training distribution evaluated on preregistered full [6,28] m distribution",
        },
        "python_environment_identity": python_environment,
        "training_safety": {
            "tensorboard": training["tensorboard"],
            "last100_training_outcomes": training["last100_training_outcomes"],
        },
        "condition": result["condition"],
        "outcome": outcome,
        "strata": base["strata"],
        "descriptive_d0_to_d1": comparisons,
        "action_chirality": action_chirality,
        "gates": gates,
        "limitations": [
            "one warm-start training seed and one held-out evaluation seed",
            "70-bar simulation cell only; no hardware claim",
            "PASS permits manual P3 preregistration review, not automatic P3 launch",
            "the original seed313 P2 decision remains strict FAIL",
        ],
    }


def write_summary(payload: dict) -> None:
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    q3 = payload["strata"]["distance"]["q3"]
    cv = payload["strata"]["distance_by_pattern"]["q3"]["cv"]
    outcome = payload["outcome"]
    comparison = payload["descriptive_d0_to_d1"]
    def delta(cell: str, metric: str) -> str:
        item = comparison[cell][metric]
        lo, hi = item["difference_ci95"]
        return f"{item['difference']*100:+.2f} pp [{lo*100:+.2f}, {hi*100:+.2f}]"
    lines = [
        f"# ref5in D1 held-out decision — {payload['verdict']}", "",
        "D1은 P2를 재채점하지 않으며 P3를 자동 해제하지 않는다.", "",
        "| gate | value | limit | pass |", "|---|---:|---:|:---:|",
        f"| global crash | {outcome['crash_rate']*100:.2f}% | ≤27% | {payload['gates']['global_crash_lte_27pct']['pass']} |",
        f"| q3 crash | {q3['crash_rate']*100:.2f}% | ≤30% | {payload['gates']['q3_crash_lte_30pct']['pass']} |",
        f"| q3/CV timeout | {cv['timeout_rate']*100:.2f}% | ≤12% | {payload['gates']['q3_cv_timeout_lte_12pct']['pass']} |",
        "", "## D0 대비 기술적 변화 (서로 다른 평가 seed)", "",
        "| stratum | capture | crash | timeout |", "|---|---:|---:|---:|",
        f"| global | {delta('global', 'capture')} | {delta('global', 'crash')} | {delta('global', 'timeout')} |",
        f"| q3 | {delta('q3', 'capture')} | {delta('q3', 'crash')} | {delta('q3', 'timeout')} |",
        f"| q3/CV | {delta('q3_cv', 'capture')} | {delta('q3_cv', 'crash')} | {delta('q3_cv', 'timeout')} |",
        "", "D1은 유의한 방향의 개선을 만들었지만 q3/CV timeout 절대 gate를 통과하지 못했다. "
        "비교에는 adaptation과 q3 학습분포 변경이 함께 들어 있으므로 둘의 효과를 분리하지 않는다.",
        "", f"Checkpoint SHA-256: `{payload['checkpoint_sha256']}`", "",
    ]
    (OUTPUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    require(mode in {"preflight", "run", "finalize", "verify"},
            "usage: ... {preflight|run|finalize|verify}")
    if mode == "preflight":
        require(BASE.sha256_file(P1C_CHECKPOINT) == P1C_SHA, "P1c checkpoint identity mismatch")
        configure_base(P1C_CHECKPOINT, P1C_SHA)
        require(not OUTPUT.exists(), f"output already exists: {OUTPUT}")
        BASE.run_evaluator(preflight=True)
        require(not OUTPUT.exists(), "preflight created D1 result output")
        print("[ref5in-d1-eval] PREFLIGHT PASS")
        return 0

    training = verify_training()
    configure_base(training["checkpoint"], training["checkpoint_sha256"])
    forced_mismatch = verify_expected_rejection(training["checkpoint"])
    if mode == "run":
        status = subprocess.check_output(
            [
                "git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all",
                "--", "aerial_gym", "resources/robots",
            ],
            text=True,
        )
        # Documentation/result/dashboard drafts cannot change simulator behavior and often coexist
        # with a long GPU campaign.  Gate the exact runtime roots here; verify_training() and
        # verify_eval_source() still require byte-for-byte train/eval maps and Python receipts.
        require(not status.strip(), "runtime source is dirty; commit before D1 evaluation")
        require(not OUTPUT.exists(), f"refusing overwrite: {OUTPUT}")
        BASE.run_evaluator()
    result = BASE.verify_result()
    V2.verify_joint(result)
    python_environment = verify_eval_source(training)
    expected = decision_summary(result, training, forced_mismatch, python_environment)
    if mode in {"run", "finalize"}:
        write_summary(expected)
        print(f"[ref5in-d1-eval] {expected['verdict']}")
        return 0 if expected["verdict"] == "PASS" else 1
    recorded = BASE.load_json(OUTPUT / "summary.json")
    for key in ("scope", "verdict", "decision_authority", "p2_verdict_changed",
                "p3_automatically_unlocked", "checkpoint_sha256", "condition", "outcome",
                "generic_provenance_override", "python_environment_identity", "strata",
                "descriptive_d0_to_d1", "action_chirality", "gates", "limitations"):
        require(recorded.get(key) == expected.get(key), f"D1 summary changed: {key}")
    print(f"[ref5in-d1-eval] VERIFY {recorded['verdict']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BASE.ContractError, OSError, RuntimeError, subprocess.CalledProcessError,
            json.JSONDecodeError) as exc:
        print(f"[ref5in-d1-eval] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
