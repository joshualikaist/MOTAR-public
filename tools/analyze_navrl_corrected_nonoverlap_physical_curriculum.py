#!/usr/bin/env python3
"""Fail-closed live snapshot and post-run verdict for the seed-911 route-off curriculum.

Live mode never emits an official verdict. Completion authorizes only offline analysis and a
separate held-out evaluation preregistration. It never authorizes routed PPO or hardware claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MAX_EPOCHS = 30000
SEED = 911
SCHEDULE = {70: 0.82, 85: 0.77, 100: 0.72}
SCHEDULE_DEFAULT = 0.70
PROMOTION_RE = re.compile(
    r"NavRL density curriculum promoted \| bars (?P<src>\d+) -> (?P<dst>\d+) "
    r"after (?P<episodes>\d+) eps, capture=(?P<capture>[0-9.]+) "
    r"\(threshold=(?P<threshold>[0-9.]+)\) dwell=(?P<dwell>[0-9.]+) epochs"
)
HOLD_RE = re.compile(
    r"NavRL density curriculum held \| bars=(?P<bars>\d+) capture=(?P<capture>[0-9.]+) "
    r"over (?P<episodes>\d+) eps \(threshold=(?P<threshold>[0-9.]+)\)"
)
FAIL_STOP_RE = re.compile(r"FAIL-STOP: same-density capture collapse")
EPOCH_HEAD_RE = re.compile(r"^  epoch\s*:\s*(\d+)\s*/\s*30000\s*$", re.M)
OUTCOME_RE = re.compile(
    r"captured \(success\)\s*:\s*[0-9.]+% \((\d+)/(\d+)\).*?"
    r"crash\s*:\s*[0-9.]+% \((\d+)/(\d+)\).*?"
    r"timeout \(no capture\)\s*:\s*[0-9.]+% \((\d+)/(\d+)\)",
    re.S,
)
BARS_RE = re.compile(r"density bars\s+:\s+(\d+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_for(bars: int) -> float:
    if bars >= 115:
        return SCHEDULE_DEFAULT
    return SCHEDULE[bars]


def _authority(held_out_prereg: bool) -> dict:
    return {
        "held_out_eval_preregistration": held_out_prereg,
        "routed_ppo": False,
        "parameter_search": False,
        "hardware_or_sim_to_real_claim": False,
        "resume_or_warm_start": False,
    }


def parse_density_events(log: Path) -> dict:
    promotions = []
    holds = []
    fail_stop = False
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        promoted = PROMOTION_RE.search(line)
        if promoted:
            promotions.append(
                {
                    "src": int(promoted.group("src")),
                    "dst": int(promoted.group("dst")),
                    "episodes": int(promoted.group("episodes")),
                    "capture": float(promoted.group("capture")),
                    "threshold": float(promoted.group("threshold")),
                    "dwell_epochs": float(promoted.group("dwell")),
                }
            )
            continue
        held = HOLD_RE.search(line)
        if held:
            holds.append(
                {
                    "bars": int(held.group("bars")),
                    "capture": float(held.group("capture")),
                    "episodes": int(held.group("episodes")),
                    "threshold": float(held.group("threshold")),
                }
            )
            continue
        if FAIL_STOP_RE.search(line):
            fail_stop = True
    return {"promotions": promotions, "holds": holds, "fail_stop": fail_stop}


def csv_rows(run: Path) -> list[dict]:
    path = run / "aerial_run/epoch_metrics.csv"
    if not path.is_file():
        raise RuntimeError("epoch_metrics.csv missing")
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if not row.get("epoch"):
                continue
            epoch = int(float(row["epoch"]))
            bars = int(float(row["n_bars_active"])) if row.get("n_bars_active") else None
            reward = float(row["mean_reward"]) if row.get("mean_reward") else None
            capture = float(row["captured_rate"]) if row.get("captured_rate") else None
            crash = float(row["crash_rate"]) if row.get("crash_rate") else None
            timeout = float(row["timeout_rate"]) if row.get("timeout_rate") else None
            if any(
                value is not None and isinstance(value, float) and not math.isfinite(value)
                for value in (reward, capture, crash, timeout)
            ):
                raise RuntimeError(f"non-finite metric at epoch {epoch}")
            rows.append(
                {
                    "epoch": epoch,
                    "n_bars_active": bars,
                    "mean_reward": reward,
                    "captured_rate": capture,
                    "crash_rate": crash,
                    "timeout_rate": timeout,
                }
            )
    if not rows:
        raise RuntimeError("epoch_metrics.csv is empty")
    return rows


def window_mean(rows: list[dict], key: str, count: int) -> dict | None:
    selected = [row[key] for row in rows[-count:] if row.get(key) is not None]
    if not selected:
        return None
    return {
        "count": len(selected),
        "mean": sum(selected) / len(selected),
        "min": min(selected),
        "max": max(selected),
    }


def latest_checkpoint(run: Path) -> Path | None:
    numbered = []
    for path in (run / "nn").glob("last_gen_ppo_ep_*.pth"):
        match = re.search(r"last_gen_ppo_ep_(\d+)_", path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    if not numbered:
        return None
    numbered.sort()
    return numbered[-1][1]


def live_status(run: Path, log: Path) -> dict:
    run, log = run.resolve(), log.resolve()
    if (run / ".aerial_training_finished").is_file():
        raise RuntimeError("run is finished; use official analyze, not --live")
    rows = csv_rows(run)
    last = rows[-1]
    events = parse_density_events(log)
    bars = sorted({row["n_bars_active"] for row in rows if row["n_bars_active"] is not None})
    checkpoint = latest_checkpoint(run)
    current_bars = last["n_bars_active"] or (bars[-1] if bars else None)
    return {
        "schema": "motar_corrected_nonoverlap_physical_off_curriculum_live_v1",
        "status": "LIVE_RUNNING",
        "run": run.name,
        "log": str(log),
        "epoch": last["epoch"],
        "max_epochs": MAX_EPOCHS,
        "seed": SEED,
        "n_bars_active": current_bars,
        "bars_seen": bars,
        "latest_capture": last["captured_rate"],
        "latest_reward": last["mean_reward"],
        "last_100": {
            "capture": window_mean(rows, "captured_rate", 100),
            "reward": window_mean(rows, "mean_reward", 100),
        },
        "density_promotions": events["promotions"],
        "latest_hold": events["holds"][-1] if events["holds"] else None,
        "same_density_fail_stop": events["fail_stop"],
        "promotion_threshold": threshold_for(current_bars) if current_bars else None,
        "latest_last_gen_checkpoint": str(checkpoint) if checkpoint else None,
        "verdict": None,
        "authority": _authority(False),
    }


def tb_summary(run: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(run / "summaries"), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    nonfinite = []
    values = {}
    for tag in tags:
        series = [float(event.value) for event in accumulator.Scalars(tag)]
        if not series or not all(math.isfinite(value) for value in series):
            nonfinite.append(tag)
        values[tag] = series
    required = {}
    for tag in (
        "ppo/kl",
        "ppo/behavior_kl_audit_max",
        "ppo/epoch_rollback_total",
        "ppo/kl_skipped_minibatches",
    ):
        series = values.get(tag, [])
        required[tag] = {
            "count": len(series),
            "max": max(series) if series else None,
            "last": series[-1] if series else None,
        }
    return {"tag_count": len(tags), "nonfinite_tags": nonfinite, "required": required}


def verify_source(manifest: Path, expected_sha: str) -> dict:
    manifest = manifest.resolve()
    if sha256(manifest) != expected_sha:
        raise RuntimeError("source manifest SHA drift")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    repository = Path(payload["repository_root"]).resolve()
    entries = payload.get("runtime_files") or []
    if len(entries) != payload.get("runtime_file_count"):
        raise RuntimeError("source receipt file accounting drift")
    for entry in entries:
        original = (repository / entry["path"]).resolve()
        snapshot = (manifest.parent / entry["snapshot"]).resolve()
        expected = entry["sha256"]
        if not original.is_file() or sha256(original) != expected:
            raise RuntimeError(f"runtime source drift: {entry['path']}")
        if not snapshot.is_file() or sha256(snapshot) != expected:
            raise RuntimeError(f"source snapshot drift: {entry['snapshot']}")
    return {
        "verified": True,
        "manifest": str(manifest),
        "manifest_sha256": expected_sha,
        "git_commit": payload.get("git_commit"),
        "git_dirty": payload.get("git_dirty"),
        "runtime_file_count": len(entries),
    }


def finite_tensors(value, path="checkpoint") -> list[str]:
    import torch

    bad = []
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            bad.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            bad.extend(finite_tensors(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            bad.extend(finite_tensors(item, f"{path}[{index}]"))
    return bad


def analyze(run: Path, log: Path) -> dict:
    import torch

    run, log = run.resolve(), log.resolve()
    summary_path = run / "aerial_run/run_summary.json"
    if not summary_path.is_file():
        raise RuntimeError("run_summary.json missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    events = parse_density_events(log)
    rows = csv_rows(run)
    last = rows[-1]
    checkpoint_path = latest_checkpoint(run)
    if checkpoint_path is None:
        raise RuntimeError("no last_gen_ppo_ep_* checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("env_state") or {}
    tb = tb_summary(run)
    source = verify_source(
        Path(state["cfg_training_source_manifest"]),
        state["cfg_training_source_manifest_sha256"],
    )
    required = tb["required"]
    finished = (run / ".aerial_training_finished").is_file()
    exit_reason = summary.get("exit_reason")
    integrity = {
        "checkpoint_finite": not finite_tensors(checkpoint),
        "tensorboard_finite": not tb["nonfinite_tags"],
        "ppo_kl_below_0p04": required["ppo/kl"]["max"] is not None
        and required["ppo/kl"]["max"] < 0.04,
        "behavior_kl_below_0p04": required["ppo/behavior_kl_audit_max"]["max"] is not None
        and required["ppo/behavior_kl_audit_max"]["max"] < 0.04,
        "zero_rollback": required["ppo/epoch_rollback_total"]["max"] == 0.0,
        "zero_skipped_minibatches": required["ppo/kl_skipped_minibatches"]["max"] == 0.0,
        "source_receipt_valid": source["verified"] and source["git_dirty"] is False,
        "fresh_route_off_contract": (
            state.get("cfg_target_route_mode") == "off"
            and state.get("cfg_target_dynamics") == "physical"
            and state.get("cfg_target_speed_final") == 1.25
            and state.get("cfg_placement_mode") == "footprint_clearance"
            and state.get("cfg_placement_surface_clearance_m") == 0.45
            and state.get("cfg_density_final") == 205
            and state.get("cfg_robot_name") == "navrl_ref5in_v2_quad"
        ),
    }
    if not all(integrity.values()):
        verdict = "VOID_EXECUTION"
    elif events["fail_stop"] or exit_reason == "early_stop_density_capture_collapse":
        verdict = "STOPPED_SAME_DENSITY_GUARD"
    elif finished and exit_reason == "max_epochs" and last["epoch"] == MAX_EPOCHS:
        verdict = "COMPLETE_MAX_EPOCH"
    else:
        verdict = "INCOMPLETE_INTERRUPTED"
    held_out = verdict in {"STOPPED_SAME_DENSITY_GUARD", "COMPLETE_MAX_EPOCH"}
    return {
        "schema": "motar_corrected_nonoverlap_physical_off_curriculum_v1",
        "run": run.name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "verdict": verdict,
        "exit_reason": exit_reason,
        "last_epoch": last["epoch"],
        "n_bars_active": last["n_bars_active"],
        "bars_seen": sorted({row["n_bars_active"] for row in rows if row["n_bars_active"]}),
        "density_promotions": events["promotions"],
        "integrity": integrity,
        "last_100": {
            "capture": window_mean(rows, "captured_rate", 100),
            "reward": window_mean(rows, "mean_reward", 100),
        },
        "tensorboard": tb,
        "source_receipt": source,
        "authority": _authority(held_out),
        "claims": {
            "proves_205_bar_performance": False,
            "uses_gen_ppo_for_density_curve": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.live:
        result = live_status(args.run, args.log)
    else:
        result = analyze(args.run, args.log)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["status"] if args.live else result["verdict"])
    print(json.dumps(
        {
            "epoch": result.get("epoch") or result.get("last_epoch"),
            "n_bars_active": result.get("n_bars_active"),
            "verdict": result.get("verdict"),
            "authority": result["authority"],
            "last_100": result.get("last_100"),
            "density_promotions": result.get("density_promotions"),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
