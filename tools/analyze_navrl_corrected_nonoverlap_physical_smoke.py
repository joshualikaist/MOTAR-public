#!/usr/bin/env python3
"""Fail-closed verdict for the preregistered corrected non-overlap route-off smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def outcome_rows(log: Path) -> list[dict]:
    text = log.read_text(encoding="utf-8", errors="replace")
    heads = list(re.finditer(r"^  epoch\s*:\s*(\d+)\s*/\s*500\s*$", text, re.M))
    rows = []
    pattern = re.compile(
        r"captured \(success\)\s*:\s*[0-9.]+% \((\d+)/(\d+)\).*?"
        r"crash\s*:\s*[0-9.]+% \((\d+)/(\d+)\).*?"
        r"timeout \(no capture\)\s*:\s*[0-9.]+% \((\d+)/(\d+)\)",
        re.S,
    )
    for index, head in enumerate(heads):
        epoch = int(head.group(1))
        end = heads[index + 1].start() if index + 1 < len(heads) else len(text)
        match = pattern.search(text[head.start():end])
        if match is None:
            continue
        captured, cap_n, crash, crash_n, timeout, timeout_n = map(int, match.groups())
        if cap_n != crash_n or cap_n != timeout_n or captured + crash + timeout != cap_n:
            raise RuntimeError(f"outcome accounting drift at epoch {epoch}")
        rows.append(
            {"epoch": epoch, "captured": captured, "crash": crash,
             "timeout": timeout, "episodes": cap_n}
        )
    if len(heads) != 500 or len(rows) != 499 or {row["epoch"] for row in rows} != set(range(2, 501)):
        raise RuntimeError("expected epoch headings 1..500 and outcome blocks 2..500")
    return rows


def pool(rows: list[dict], lo: int, hi: int) -> dict:
    selected = [row for row in rows if lo <= row["epoch"] <= hi]
    if len(selected) != 100:
        raise RuntimeError(f"window {lo}..{hi} has {len(selected)} rows")
    sums = {name: sum(row[name] for row in selected) for name in (
        "captured", "crash", "timeout", "episodes"
    )}
    return {
        "epoch_range": [lo, hi], "epochs": len(selected), **sums,
        "capture_rate": sums["captured"] / sums["episodes"],
        "crash_rate": sums["crash"] / sums["episodes"],
        "timeout_rate": sums["timeout"] / sums["episodes"],
    }


def rewards(run: Path, lo: int, hi: int) -> dict:
    values = []
    bars = set()
    with (run / "aerial_run/epoch_metrics.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["n_bars_active"]:
                bars.add(int(float(row["n_bars_active"])))
            epoch = int(row["epoch"])
            if lo <= epoch <= hi and row["mean_reward"]:
                values.append(float(row["mean_reward"]))
    if len(values) != 100:
        raise RuntimeError(f"reward window {lo}..{hi} has {len(values)} values")
    return {"epoch_range": [lo, hi], "count": len(values), "mean": sum(values) / len(values),
            "min": min(values), "max": max(values), "all_density_bars": sorted(bars)}


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
        "ppo/kl", "ppo/behavior_kl_audit_max", "ppo/epoch_rollback_total",
        "ppo/kl_skipped_minibatches",
    ):
        series = values.get(tag, [])
        required[tag] = {
            "count": len(series), "max": max(series) if series else None,
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
    return {"verified": True, "manifest": str(manifest), "manifest_sha256": expected_sha,
            "git_commit": payload.get("git_commit"), "git_dirty": payload.get("git_dirty"),
            "runtime_file_count": len(entries)}


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
    checkpoints = list((run / "nn").glob("last_gen_ppo_ep_500_*.pth"))
    if len(checkpoints) != 1:
        raise RuntimeError("expected exactly one epoch-500 last checkpoint")
    checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    state = checkpoint.get("env_state") or {}
    summary = json.loads((run / "aerial_run/run_summary.json").read_text(encoding="utf-8"))
    rows = outcome_rows(log)
    early, late = pool(rows, 2, 101), pool(rows, 401, 500)
    early_reward, late_reward = rewards(run, 2, 101), rewards(run, 401, 500)
    tb = tb_summary(run)
    source = verify_source(
        Path(state["cfg_training_source_manifest"]),
        state["cfg_training_source_manifest_sha256"],
    )
    required = tb["required"]
    contract = {
        "normal_completion": (
            (run / ".aerial_training_finished").is_file()
            and summary.get("exit_reason") == "max_epochs"
            and summary.get("last_epoch") == 500
            and checkpoint.get("epoch") == 500
        ),
        "checkpoint_finite": not finite_tensors(checkpoint),
        "tensorboard_finite": not tb["nonfinite_tags"],
        "ppo_kl_below_0p04": required["ppo/kl"]["count"] == 500
        and required["ppo/kl"]["max"] < 0.04,
        "behavior_kl_below_0p04": required["ppo/behavior_kl_audit_max"]["count"] == 500
        and required["ppo/behavior_kl_audit_max"]["max"] < 0.04,
        "zero_rollback": required["ppo/epoch_rollback_total"]["count"] == 500
        and required["ppo/epoch_rollback_total"]["max"] == 0.0,
        "zero_skipped_minibatches": required["ppo/kl_skipped_minibatches"]["count"] == 500
        and required["ppo/kl_skipped_minibatches"]["max"] == 0.0,
        "fixed_70_nonoverlap_contract": (
            early_reward["all_density_bars"] == [70]
            and state.get("n_bars_active") == 70
            and state.get("cfg_density_final") == 70
            and state.get("cfg_placement_mode") == "footprint_clearance"
            and state.get("cfg_placement_surface_clearance_m") == 0.45
            and state.get("cfg_target_dynamics") == "physical"
            and state.get("cfg_target_route_mode") == "off"
            and state.get("cfg_target_speed_final") == 1.25
            and state.get("cfg_robot_name") == "navrl_ref5in_v2_quad"
        ),
        "capture_improves_by_10pp": late["capture_rate"] - early["capture_rate"] >= 0.10,
        "reward_improves": late_reward["mean"] > early_reward["mean"],
        "source_receipt_valid": source["verified"] and source["git_dirty"] is False,
    }
    passed = all(contract.values())
    return {
        "schema": "motar_corrected_nonoverlap_physical_off_smoke_v1",
        "run": run.name,
        "checkpoint": str(checkpoints[0]),
        "checkpoint_sha256": sha256(checkpoints[0]),
        "verdict": "PASS_LEARNING_VIABILITY" if passed else "FAIL_LEARNING_VIABILITY",
        "checks": contract,
        "early_100": early,
        "late_100": late,
        "capture_delta_pp": 100.0 * (late["capture_rate"] - early["capture_rate"]),
        "early_reward": early_reward,
        "late_reward": late_reward,
        "reward_delta": late_reward["mean"] - early_reward["mean"],
        "tensorboard": tb,
        "source_receipt": source,
        "authority": {
            "route_off_curriculum_preregistration": passed,
            "routed_ppo": False,
            "long_training_directly_authorized": False,
            "hardware_or_sim_to_real_claim": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["verdict"])
    print(json.dumps({
        "capture_delta_pp": result["capture_delta_pp"],
        "reward_delta": result["reward_delta"],
        "failed_checks": [key for key, value in result["checks"].items() if not value],
    }, indent=2))


if __name__ == "__main__":
    main()
