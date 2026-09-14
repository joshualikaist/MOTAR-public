#!/usr/bin/env python3
"""Evidence-first postmortem for the canonical NavRL v2 recovery lineage.

The script is read-only.  It joins the three recovery segments without double-counting the
rolled-back ep20701--20746 interval, parses exact density promotion/hold windows and diagnostic
snapshots from session logs, and summarizes TensorBoard policy-health signals.  It may be run on a
live run for a clearly labelled partial report, but publication/next-run decisions require the
normal-completion marker and final held-out evaluation described in RESEARCH_PLAN.md section 8.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean, median


ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "aerial_gym/rl_training/rl_games"
RUNS_ROOT = RL_ROOT / "runs"
LOG_ROOT = RL_ROOT / "train_session_logs"

DEFAULT_SEGMENTS = (
    ("ppo_260801_1150_navrl_v2-recover-smoke-130bars-s1", 9501, 9600),
    ("ppo_260801_1235_navrl_v2-recover-curriculum-s1", 9601, 20700),
    ("ppo_260802_0020_navrl_v2-recover-curriculum-continue-s1", 20701, None),
)

HOLD_RE = re.compile(
    r"density curriculum held \| bars=(?P<bars>\d+) capture=(?P<capture>[0-9.]+) "
    r"over (?P<episodes>\d+) eps \(threshold=(?P<threshold>[0-9.]+)\)"
)
PROMOTE_RE = re.compile(
    r"density curriculum promoted \| bars (?P<src>\d+) -> (?P<dst>\d+) "
    r"after (?P<episodes>\d+) eps, capture=(?P<capture>[0-9.]+) "
    r"\(threshold=(?P<threshold>[0-9.]+)\)"
)
EPOCH_RE = re.compile(r"epoch\s+:\s+(\d+)\s+/\s+(\d+)")
BARS_RE = re.compile(r"density bars\s+:\s+(\d+)")
ACTION_RE = re.compile(
    r"NavRL actiondiag .*?task_input_oob\[x,y,z,yaw\]=\[([^]]+)\].*?"
    r"exec_edge98=\[([^]]+)\].*?signed_y=([0-9.+-]+).*?pos_y=([0-9.]+) "
    r"neg_y=([0-9.]+).*?delta_y=([0-9.+-]+).*?sign_flip_y=([0-9.]+)"
)
MOTION_RE = re.compile(
    r"NavRL motiondiag \| speed=([0-9.]+)m/s command=([0-9.]+)m/s "
    r"low_speed=([0-9.]+) commanded_stall=([0-9.]+)"
)
CRASH_RE = re.compile(
    r"NavRL crashdiag \| bar_contact=([0-9.]+) \(mean_x=([0-9.]+)m steps=([0-9.]+)\) "
    r"below=([0-9.]+).*?above=([0-9.]+) oob=([0-9.]+).*?\(n_crash=(\d+)\)"
)
BAR_RE = re.compile(
    r"NavRL barprobe v2 \| n=(\d+) bars_range=([0-9.]+) bars_fov=([0-9.]+) "
    r"occupied_bins=([0-9.]+).*?hit_fov=([0-9.]+) hit_token=([0-9.]+) "
    r"hit_token_given_fov=([0-9.]+) tokens=([0-9.]+) associated=([0-9.]+) "
    r"unique=([0-9.]+) duplicate=([0-9.]+).*?center_offset=([0-9.]+)m "
    r"cross_track=([0-9.]+)m radial_gap=([0-9.]+)m"
)
STRATA_RE = re.compile(
    r"NavRL density strata \| speed\[([^]]+)\] distance\[([^]]+)\] "
    r"pattern\[([^]]+)\] gate=(\w+) \(([^)]+)\)"
)
STRATUM_ITEM_RE = re.compile(r"([a-zA-Z0-9_]+)=(na|[0-9.]+)\((\d+)\)")


def _finite(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def canonical_rows(segments=DEFAULT_SEGMENTS):
    rows = []
    sources = []
    for run, first, last in segments:
        path = RUNS_ROOT / run / "aerial_run/epoch_metrics.csv"
        selected = []
        for row in _read_csv(path):
            epoch = int(float(row["epoch"]))
            if epoch < first or (last is not None and epoch > last):
                continue
            copy = dict(row)
            copy["epoch"] = epoch
            copy["_run"] = run
            selected.append(copy)
        rows.extend(selected)
        sources.append(
            {
                "run": run,
                "requested_first": first,
                "requested_last": last,
                "rows": len(selected),
                "actual_first": selected[0]["epoch"] if selected else None,
                "actual_last": selected[-1]["epoch"] if selected else None,
            }
        )
    rows.sort(key=lambda row: row["epoch"])
    return rows, sources


def _slope_per_1000(rows, key):
    points = [
        (row["epoch"], _finite(row.get(key)))
        for row in rows
        if _finite(row.get(key)) is not None
    ]
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_bar, y_bar = fmean(xs), fmean(ys)
    denom = sum((x - x_bar) ** 2 for x in xs)
    if denom == 0.0:
        return None
    return 1000.0 * sum((x - x_bar) * (y - y_bar) for x, y in points) / denom


def _metric_stats(rows, key):
    values = [_finite(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return {
        "mean": fmean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "last": values[-1],
    }


def summarize_density(rows):
    by_density = {}
    for row in rows:
        bars = _finite(row.get("n_bars_active"))
        if bars is None:
            continue
        by_density.setdefault(int(round(bars)), []).append(row)
    result = []
    for bars in sorted(by_density):
        group = by_density[bars]
        tail = group[-min(1000, len(group)) :]
        result.append(
            {
                "bars": bars,
                "epochs": len(group),
                "first_epoch": group[0]["epoch"],
                "last_epoch": group[-1]["epoch"],
                "capture": _metric_stats(group, "captured_rate"),
                "crash": _metric_stats(group, "crash_rate"),
                "timeout": _metric_stats(group, "timeout_rate"),
                "reward": _metric_stats(group, "mean_reward"),
                "capture_slope_per_1000_epochs": _slope_per_1000(group, "captured_rate"),
                "tail1000_capture": _metric_stats(tail, "captured_rate"),
                "tail1000_crash": _metric_stats(tail, "crash_rate"),
            }
        )
    return result


def _candidate_logs(run_names):
    paths = []
    for path in LOG_ROOT.glob("*.log"):
        try:
            sample = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(run in sample or run in path.name for run in run_names):
            paths.append((path, sample))
    return sorted(paths, key=lambda item: (item[0].stat().st_mtime, item[0].name))


def parse_logs(run_names):
    gates = []
    diagnostics = {
        "action": [],
        "motion": [],
        "crash": [],
        "barprobe": [],
        "strata": [],
    }
    files = []
    for path, text in _candidate_logs(run_names):
        files.append(str(path.relative_to(RL_ROOT)))
        epoch = None
        bars = None
        for line in text.splitlines():
            if match := EPOCH_RE.search(line):
                epoch = int(match.group(1))
            if match := BARS_RE.search(line):
                bars = int(match.group(1))
            if match := HOLD_RE.search(line):
                gates.append(
                    {
                        "run_log": path.name,
                        "epoch_context": epoch,
                        "result": "held",
                        "bars": int(match.group("bars")),
                        "next_bars": None,
                        "capture": float(match.group("capture")),
                        "episodes": int(match.group("episodes")),
                        "threshold": float(match.group("threshold")),
                    }
                )
            if match := PROMOTE_RE.search(line):
                gates.append(
                    {
                        "run_log": path.name,
                        "epoch_context": epoch,
                        "result": "promoted",
                        "bars": int(match.group("src")),
                        "next_bars": int(match.group("dst")),
                        "capture": float(match.group("capture")),
                        "episodes": int(match.group("episodes")),
                        "threshold": float(match.group("threshold")),
                    }
                )
            if match := ACTION_RE.search(line):
                diagnostics["action"].append(
                    {
                        "epoch": epoch,
                        "bars": bars,
                        "task_input_oob": [float(v) for v in match.group(1).split(",")],
                        "exec_edge98": [float(v) for v in match.group(2).split(",")],
                        "signed_y": float(match.group(3)),
                        "positive_y": float(match.group(4)),
                        "negative_y": float(match.group(5)),
                        "delta_y": float(match.group(6)),
                        "sign_flip_y": float(match.group(7)),
                    }
                )
            if match := MOTION_RE.search(line):
                diagnostics["motion"].append(
                    {
                        "epoch": epoch,
                        "bars": bars,
                        "speed_m_s": float(match.group(1)),
                        "command_m_s": float(match.group(2)),
                        "low_speed": float(match.group(3)),
                        "commanded_stall": float(match.group(4)),
                    }
                )
            if match := CRASH_RE.search(line):
                diagnostics["crash"].append(
                    {
                        "epoch": epoch,
                        "bars": bars,
                        "bar_contact": float(match.group(1)),
                        "mean_x_m": float(match.group(2)),
                        "mean_steps": float(match.group(3)),
                        "below": float(match.group(4)),
                        "above": float(match.group(5)),
                        "oob": float(match.group(6)),
                        "n_crash": int(match.group(7)),
                    }
                )
            if match := BAR_RE.search(line):
                diagnostics["barprobe"].append(
                    {
                        "epoch": epoch,
                        "bars": bars,
                        "n": int(match.group(1)),
                        "bars_range": float(match.group(2)),
                        "bars_fov": float(match.group(3)),
                        "occupied_bins": float(match.group(4)),
                        "hit_fov": float(match.group(5)),
                        "hit_token": float(match.group(6)),
                        "hit_token_given_fov": float(match.group(7)),
                        "tokens": float(match.group(8)),
                        "associated": float(match.group(9)),
                        "unique": float(match.group(10)),
                        "duplicate": float(match.group(11)),
                        "center_offset_m": float(match.group(12)),
                        "cross_track_m": float(match.group(13)),
                        "radial_gap_m": float(match.group(14)),
                    }
                )
            if match := STRATA_RE.search(line):
                def parse_axis(raw):
                    return {
                        label: {
                            "rate": None if rate == "na" else float(rate),
                            "episodes": int(episodes),
                        }
                        for label, rate, episodes in STRATUM_ITEM_RE.findall(raw)
                    }

                diagnostics["strata"].append(
                    {
                        "epoch": epoch,
                        "bars": bars,
                        "speed": parse_axis(match.group(1)),
                        "distance": parse_axis(match.group(2)),
                        "pattern": parse_axis(match.group(3)),
                        "gate": match.group(4),
                        "reason": match.group(5),
                    }
                )
    # Multiple logs may contain the same tee output.  Deduplicate exact gate records while
    # retaining chronological order.
    seen = set()
    unique_gates = []
    for gate in gates:
        key = (
            gate["result"],
            gate["bars"],
            gate["next_bars"],
            gate["capture"],
            gate["episodes"],
            gate["threshold"],
        )
        if key not in seen:
            seen.add(key)
            unique_gates.append(gate)
    for kind, records in diagnostics.items():
        seen_records = set()
        unique_records = []
        for record in records:
            key = json.dumps(record, sort_keys=True, ensure_ascii=True)
            if key not in seen_records:
                seen_records.add(key)
                unique_records.append(record)
        diagnostics[kind] = unique_records
    return {"files": files, "gates": unique_gates, "diagnostics": diagnostics}


def _tb_scalars(run, first, last):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}, "tensorboard package unavailable; use the aerialgym Python"
    path = RUNS_ROOT / run / "summaries"
    if not path.is_dir():
        return {}, f"missing summaries: {path}"
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    data = {}
    for tag in accumulator.Tags().get("scalars", []):
        values = []
        for event in accumulator.Scalars(tag):
            step = int(event.step)
            if step < first or (last is not None and step > last):
                continue
            value = _finite(event.value)
            if value is not None:
                values.append((step, value))
        if values:
            data[tag] = values
    return data, None


def summarize_tensorboard(segments=DEFAULT_SEGMENTS):
    joined = {}
    errors = []
    for run, first, last in segments:
        scalars, error = _tb_scalars(run, first, last)
        if error:
            errors.append(error)
        for tag, points in scalars.items():
            joined.setdefault(tag, []).extend(points)
    for tag in joined:
        joined[tag].sort(key=lambda point: point[0])

    wanted = (
        "ppo/kl",
        "ppo/behavior_kl_audit_max",
        "ppo/behavior_kl_sample_max",
        "ppo/entropy",
        "ppo/explained_variance",
        "ppo/learning_rate",
        "ppo/epoch_rollback",
        "ppo/epoch_rollback_total",
        "ppo/epoch_rollback_streak",
        "ppo/kl_skipped_minibatches",
        "policy_action/raw_oob_x",
        "policy_action/raw_oob_y",
        "policy_action/raw_oob_z",
        "policy_action/raw_oob_yaw",
        "policy_action/edge99_x",
        "policy_action/edge99_y",
        "policy_action/signed_y",
        "policy_action/positive_y",
        "policy_action/negative_y",
    )
    summary = {}
    for tag in wanted:
        points = joined.get(tag, [])
        if not points:
            continue
        rows = [{"epoch": step, "value": value} for step, value in points]
        tail = rows[-min(500, len(rows)) :]
        summary[tag] = {
            "n": len(rows),
            "first_epoch": rows[0]["epoch"],
            "last_epoch": rows[-1]["epoch"],
            "last": rows[-1]["value"],
            "min": min(row["value"] for row in rows),
            "max": max(row["value"] for row in rows),
            "tail500_mean": fmean(row["value"] for row in tail),
            "tail500_slope_per_1000_epochs": _slope_per_1000(tail, "value"),
        }
    return summary, errors


def _latest_checkpoint(run, hash_checkpoint=False):
    paths = list((RUNS_ROOT / run / "nn").glob("last_gen_ppo_ep_*.pth"))
    candidates = []
    for path in paths:
        match = re.search(r"last_gen_ppo_ep_(\d+)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    epoch, path = max(candidates, key=lambda item: (item[0], item[1].stat().st_mtime))
    result = {
        "epoch": epoch,
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
    }
    if hash_checkpoint:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def _checkpoint_contract(checkpoint):
    if not checkpoint:
        return None, "checkpoint missing"
    try:
        import torch
    except ImportError:
        return None, "torch unavailable; use the aerialgym Python"
    path = ROOT / checkpoint["path"]
    try:
        # This is a locally produced trusted rl_games checkpoint. State the pickle choice
        # explicitly so future PyTorch defaults cannot alter this audit silently.
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        return None, f"checkpoint load failed: {exc}"
    env = payload.get("env_state", {})

    def values(name):
        raw = env.get(name, [])
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        return [int(value) for value in raw] if isinstance(raw, (list, tuple)) else []

    def axis(succ_name, fin_name, labels):
        succ, fin = values(succ_name), values(fin_name)
        return {
            label: {
                "successes": succ[index] if index < len(succ) else None,
                "episodes": fin[index] if index < len(fin) else None,
                "rate": (
                    succ[index] / fin[index]
                    if index < len(succ) and index < len(fin) and fin[index] > 0
                    else None
                ),
            }
            for index, label in enumerate(labels)
        }

    cfg_keys = (
        "cfg_arena_xy",
        "cfg_arena_z",
        "cfg_bar_pool",
        "cfg_placement_mode",
        "cfg_placement_touch_m",
        "cfg_placement_gap_m",
        "cfg_obstacle_selector",
        "cfg_obstacle_cluster_gap_m",
        "cfg_density_check_eps",
        "cfg_density_min_epochs",
        "cfg_density_final",
        "cfg_density_step",
        "cfg_density_threshold_schedule",
        "cfg_density_stratified_gate",
        "cfg_density_stratified_floor",
        "cfg_target_motion_model",
        "cfg_target_pattern",
        "cfg_target_speed_min",
        "cfg_target_speed_final",
        "cfg_action_policy",
        "cfg_action_std",
        "current_action_learning_rate",
    )
    fin = int(env.get("density_fin_agg", 0))
    succ = int(env.get("density_succ_agg", 0))
    return {
        "epoch": int(payload.get("epoch", -1)),
        "frame": int(payload.get("frame", -1)),
        "rollback_total": int(payload.get("aerial_ppo_rollback_total", 0)),
        "rollback_streak": int(payload.get("aerial_ppo_rollback_streak", 0)),
        "n_bars_active": int(env.get("n_bars_active", -1)),
        "partial_density_window": {
            "successes": succ,
            "episodes": fin,
            "capture": succ / fin if fin else None,
        },
        "strata": {
            "speed": axis(
                "density_speed_succ", "density_speed_fin", ("q0", "q1", "q2", "q3")
            ),
            "distance": axis(
                "density_dist_succ", "density_dist_fin", ("q0", "q1", "q2", "q3")
            ),
            "pattern": axis(
                "density_pattern_succ", "density_pattern_fin", ("cv", "waypoint", "circle")
            ),
        },
        "config": {key: env.get(key) for key in cfg_keys},
    }, None


def _completion(run, last_epoch):
    run_dir = RUNS_ROOT / run
    marker = run_dir / ".aerial_training_finished"
    summary_path = run_dir / "aerial_run/run_summary.json"
    marker_text = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    process_live = False
    proc = Path("/proc")
    if proc.is_dir():
        for path in proc.glob("[0-9]*/cmdline"):
            try:
                command = path.read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            except OSError:
                continue
            if "runner.py" in command and "--task navrl_task" in command and "--train" in command:
                process_live = True
                break
    exit_reason = summary.get("exit_reason") if isinstance(summary, dict) else None
    normal_completion = bool(
        not process_live
        and marker_text
        and summary
        and exit_reason in ("max_epochs", "completed", "score_to_win")
    )
    if process_live:
        state = "partial-live"
    elif summary and exit_reason == "interrupted":
        state = "stopped-interrupted"
    elif normal_completion:
        state = "normal-complete"
    elif summary:
        state = f"stopped-{exit_reason or 'unknown'}"
    else:
        state = "stopped-without-summary"
    return {
        "process_live": process_live,
        "last_epoch": last_epoch,
        "marker": marker_text,
        "run_summary": summary,
        "state": state,
        "normal_completion": normal_completion,
        "analysis_ready": bool(not process_live and summary),
    }


def analyze(hash_checkpoint=False):
    rows, sources = canonical_rows()
    if not rows:
        raise RuntimeError("canonical recovery lineage has no epoch rows")
    epochs = [row["epoch"] for row in rows]
    duplicates = sorted(epoch for epoch in set(epochs) if epochs.count(epoch) > 1)
    missing = [epoch for epoch in range(min(epochs), max(epochs) + 1) if epoch not in set(epochs)]
    run_names = [segment[0] for segment in DEFAULT_SEGMENTS]
    log_data = parse_logs(run_names)
    tensorboard, tb_errors = summarize_tensorboard()
    active_run = DEFAULT_SEGMENTS[-1][0]
    completion = _completion(active_run, max(epochs))
    checkpoint = _latest_checkpoint(active_run, hash_checkpoint)
    checkpoint_contract, checkpoint_error = _checkpoint_contract(checkpoint)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": completion["state"],
        "canonical_lineage": {
            "segments": sources,
            "first_epoch": min(epochs),
            "last_epoch": max(epochs),
            "rows": len(rows),
            "duplicate_epochs": duplicates,
            "missing_epoch_count": len(missing),
            "missing_epoch_examples": missing[:20],
        },
        "completion": completion,
        "latest_checkpoint": checkpoint,
        "checkpoint_contract": checkpoint_contract,
        "checkpoint_error": checkpoint_error,
        "density_summary": summarize_density(rows),
        "gate_windows": log_data["gates"],
        "diagnostics": log_data["diagnostics"],
        "log_files": log_data["files"],
        "tensorboard": tensorboard,
        "tensorboard_errors": tb_errors,
        "limitations": [
            "epoch_metrics rates are epoch-weighted because per-epoch termination counts are not recorded there",
            "promotion windows are exact episode-count statistics and take precedence over epoch means",
            "a final causal decision requires held-out evaluation and the v2 geometry audit",
            "signed-y is not a chirality verdict without target-bearing and mirrored-layout conditioning",
        ],
    }


def render_markdown(report):
    lines = [
        "# NavRL v2 recovery postmortem",
        "",
        f"- status: **{report['status']}**",
        f"- generated: `{report['generated_at_utc']}`",
        f"- canonical epochs: {report['canonical_lineage']['first_epoch']}–{report['canonical_lineage']['last_epoch']}",
        f"- active process: {report['completion']['process_live']}",
        "",
        "## Density summary (epoch-weighted diagnostics)",
        "",
        "| bars | epochs | range | capture mean | capture tail1000 | crash tail1000 | slope/1000 ep |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["density_summary"]:
        cap = row["capture"]["mean"] if row["capture"] else None
        tail_cap = row["tail1000_capture"]["mean"] if row["tail1000_capture"] else None
        tail_crash = row["tail1000_crash"]["mean"] if row["tail1000_crash"] else None
        slope = row["capture_slope_per_1000_epochs"]
        fmt = lambda value: "—" if value is None else f"{value:.4f}"
        lines.append(
            f"| {row['bars']} | {row['epochs']} | {row['first_epoch']}–{row['last_epoch']} | "
            f"{fmt(cap)} | {fmt(tail_cap)} | {fmt(tail_crash)} | {fmt(slope)} |"
        )
    lines.extend(
        [
            "",
            "## Exact density gate windows",
            "",
            "| result | bars | next | capture | episodes | threshold |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for gate in report["gate_windows"]:
        lines.append(
            f"| {gate['result']} | {gate['bars']} | {gate['next_bars'] or '—'} | "
            f"{gate['capture']:.3f} | {gate['episodes']} | {gate['threshold']:.3f} |"
        )
    contract = report.get("checkpoint_contract") or {}
    partial = contract.get("partial_density_window") or {}
    lines.extend(
        [
            "",
            "## Durable checkpoint residual window",
            "",
            f"- checkpoint epoch: {contract.get('epoch', '—')}",
            f"- bars: {contract.get('n_bars_active', '—')}",
            f"- unfinished gate evidence: {partial.get('successes', '—')}/{partial.get('episodes', '—')} "
            f"(capture {partial.get('capture', float('nan')):.4f})" if partial else "- unfinished gate evidence: —",
            "",
            "| axis | bin | successes | episodes | capture |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for axis, bins in (contract.get("strata") or {}).items():
        for label, values in bins.items():
            rate = values.get("rate")
            lines.append(
                f"| {axis} | {label} | {values.get('successes')} | {values.get('episodes')} | "
                f"{'—' if rate is None else f'{rate:.4f}'} |"
            )
    lines.extend(["", "## Latest diagnostics", ""])
    for key, values in report["diagnostics"].items():
        lines.append(f"- {key}: `{json.dumps(values[-1], ensure_ascii=False) if values else 'missing'}`")
    lines.extend(["", "## PPO/policy health", ""])
    for tag, values in report["tensorboard"].items():
        lines.append(
            f"- `{tag}`: last {values['last']:.6g}, tail500 {values['tail500_mean']:.6g}, "
            f"range [{values['min']:.6g}, {values['max']:.6g}]"
        )
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash-checkpoint", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()
    report = analyze(hash_checkpoint=args.hash_checkpoint)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = render_markdown(report)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    if not args.json_out and not args.markdown_out:
        print(markdown)


if __name__ == "__main__":
    main()
