#!/usr/bin/env python3
"""Single-owner, fail-visible campaign runner for the preregistered detection-range screen.

This wrapper does not own experimental constants or verdict logic.  It imports the frozen stage-1
orchestrator and runs its already-verified phases in order.  Its only job is operational: one PID,
one lock, one durable status file, immediate Gate-0 checks, and a non-zero exit at the first error.

Usage:
  python tools/run_navrl_ref5in_detection_range_stage1_campaign.py run
  python tools/run_navrl_ref5in_detection_range_stage1_campaign.py status
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR = ROOT / "tools/run_navrl_ref5in_detection_range_stage1.py"
STATUS = ROOT / "results/navrl_ref5in_detection_range_stage1_s457_campaign_status.json"
LOCK = ROOT / "results/navrl_ref5in_detection_range_stage1_s457_campaign.lock"
PRODUCER = "tools/run_navrl_ref5in_detection_range_stage1_campaign.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_stage1():
    spec = importlib.util.spec_from_file_location("navrl_detection_range_stage1_campaign", ORCHESTRATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import stage-1 orchestrator: {ORCHESTRATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initial_state(module) -> dict:
    return {
        "schema_version": 1,
        "producer": PRODUCER,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "status": "running",
        "current_phase": None,
        "completed_phases": [],
        "failed_phase": None,
        "error": None,
        "traceback": None,
        "preregistration": module.PREREGISTRATION,
        "stage1_producer": module.PRODUCER,
        "output": str(module.OUTPUT),
        "summary": str(module.SUMMARY_JSON),
        "contract": {
            "arms": {name: clip for name, clip in module.ARMS},
            "train_seed": module.TRAIN_SEED,
            "eval_seed": module.EVAL_SEED,
            "warm_start_epoch": module.WARM_START_EPOCH,
            "terminal_epoch": module.TERMINAL_EPOCH,
            "adaptation_epochs_per_arm": module.ADAPT_EPOCHS,
            "episodes_per_arm": module.EPISODES,
            "goal_distance_m": [module.GOAL_DIST_MIN_M, module.GOAL_DIST_MAX_M],
            "bars": module.BARS,
            "detector_min_pixels": module.DETECTOR_MIN_PIXELS,
            "detect_resolution": [module.DETECT_WIDTH, module.DETECT_HEIGHT],
            "primary_gate": {
                "metric": "never_acquired_delta_pp_clip28_minus_clip20",
                "range_helps_at_or_below": module.NEVER_ACQUIRED_HELPS_THRESHOLD_PP,
            },
        },
    }


def update(state: dict, **values) -> None:
    state.update(values)
    state["updated_at_utc"] = utc_now()
    atomic_json(STATUS, state)


def run_phase(state: dict, name: str, function):
    update(state, current_phase=name)
    print(f"[detrange-campaign] START {name} | {utc_now()}", flush=True)
    value = function()
    state["completed_phases"].append({"name": name, "finished_at_utc": utc_now()})
    update(state, current_phase=None)
    print(f"[detrange-campaign] PASS {name} | {utc_now()}", flush=True)
    return value


def require_gate0(module, arm: str) -> dict:
    evidence = module.verify_training(arm)
    if not module.training_gates_passed(evidence):
        failed = {
            name: evidence[key]
            for name, key in module.TRAINING_GATES.items()
            if not evidence[key]["passed"]
        }
        raise module.ContractError(f"{arm}: Gate 0 failed: {failed}")
    return evidence


def run_campaign() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another detection-range campaign owns {LOCK}") from error
        lock_stream.seek(0)
        lock_stream.truncate()
        lock_stream.write(f"pid={os.getpid()} started={utc_now()}\n")
        lock_stream.flush()

        module = load_stage1()
        if STATUS.exists():
            previous = json.loads(STATUS.read_text(encoding="utf-8"))
            if previous.get("status") == "running":
                raise RuntimeError(
                    f"stale running status exists at {STATUS}; preserve and classify it before rerun"
                )
            raise RuntimeError(
                f"campaign status already exists with status={previous.get('status')!r}; "
                "refusing to overwrite experiment history"
            )
        state = initial_state(module)
        atomic_json(STATUS, state)
        try:
            run_phase(state, "preflight", module.run_preflight)
            run_phase(state, "train_clip20", lambda: module.train_arm("clip20"))
            run_phase(state, "gate0_clip20", lambda: require_gate0(module, "clip20"))
            run_phase(state, "train_clip28", lambda: module.train_arm("clip28"))
            run_phase(state, "gate0_clip28", lambda: require_gate0(module, "clip28"))
            run_phase(state, "evaluate_clip20", lambda: module.evaluate_arm("clip20"))
            run_phase(state, "evaluate_clip28", lambda: module.evaluate_arm("clip28"))

            def finalize():
                verified = module.verify_all()
                payload = module.build_summary(verified)
                module.write_summary(payload)
                return payload

            payload = run_phase(state, "finalize", finalize)

            def verify():
                verified = module.verify_all()
                expected = module.build_summary(verified)
                recorded = module.load_json(module.SUMMARY_JSON)
                for key in module.SUMMARY_VERIFY_KEYS:
                    module.require(recorded.get(key) == expected.get(key), f"summary changed: {key}")
                return recorded

            recorded = run_phase(state, "verify", verify)
            update(
                state,
                status="complete",
                current_phase=None,
                finished_at_utc=utc_now(),
                verdict=recorded["verdict"],
                stage2_authorised=bool(recorded["stage2_authorised"]),
                never_acquired_delta_pp=recorded["never_acquired_delta_pp"],
            )
            print(
                f"[detrange-campaign] COMPLETE | verdict={payload['verdict']} | "
                f"stage2_authorised={payload['stage2_authorised']} | {module.SUMMARY_JSON}",
                flush=True,
            )
            return 0
        except BaseException as error:
            update(
                state,
                status="failed",
                failed_phase=state.get("current_phase"),
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
                finished_at_utc=utc_now(),
            )
            print(
                f"[detrange-campaign] FAILED {state.get('failed_phase')} | "
                f"{type(error).__name__}: {error} | status={STATUS}",
                file=sys.stderr,
                flush=True,
            )
            return 2


def show_status() -> int:
    if not STATUS.is_file():
        print(f"[detrange-campaign] NOT STARTED | {STATUS}")
        return 1
    payload = json.loads(STATUS.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload.get("status") == "complete" else 1


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) == 2 else ""
    if mode == "run":
        return run_campaign()
    if mode == "status":
        return show_status()
    raise SystemExit(f"usage: {PRODUCER} {{run|status}}")


if __name__ == "__main__":
    raise SystemExit(main())
