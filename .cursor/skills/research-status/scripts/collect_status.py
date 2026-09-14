#!/usr/bin/env python3
"""LEGACY run/curve scraper. Do NOT use this to publish the dashboard.

`tools/update_status_snapshot.py` is the canonical dashboard writer: it applies the
evidence gates, emits `research_update`, and keeps `status.json` and the inline
`index.html` fallback identical. This script predates all of that, so its output is
a strict subset -- it silently drops `research_update` and undercounts runs.

Still useful as a cheap, dependency-free dump of run summaries and result CSVs for
analysis. Writes to `docs/status/status.legacy.json` unless `--out` says otherwise.

Usage:
    python collect_status.py [--repo REPO_ROOT] [--out OUT_JSON]

The repo root is derived from this file's location
(<repo>/.cursor/skills/research-status/scripts/), so the skill works in any
clone -- including Cursor cloud/mobile agents, where $HOME differs. Output goes
to <repo>/docs/status/status.json.
"""
import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from glob import glob


def _looks_like_motar_repo(path):
    return os.path.isdir(os.path.join(path, "aerial_gym")) and os.path.isdir(
        os.path.join(path, "docs", "status")
    )


def _default_repo():
    """<repo>/.cursor/skills/research-status/scripts/this_file -> <repo>.

    realpath() is deliberate: the workspace-root copy is a symlink into the
    repo, and resolving it lands on the versioned checkout.
    """
    here = os.path.realpath(__file__)
    candidate = here
    for _ in range(5):
        candidate = os.path.dirname(candidate)
    if _looks_like_motar_repo(candidate):
        return candidate
    legacy = os.path.expanduser("~/workspaces/aerial_gym_ws/src/aerial_gym_simulator")
    if _looks_like_motar_repo(legacy):
        return legacy
    return candidate


DEFAULT_REPO = _default_repo()

# run_summary.json fields we surface (all optional / may be null)
SUMMARY_FIELDS = [
    "finalized_at", "exit_reason", "epochs_logged", "first_epoch", "last_epoch",
    "is_navrl", "last_mean_reward", "last_mean_episode_length",
    "last_captured_rate", "last_crash_rate", "last_timeout_rate",
    "last_closest_approach_m", "last_closest_min_m", "last_curriculum_max_m",
    "last_n_bars_active", "peak_captured_rate", "peak_captured_epoch",
    "peak_mean_reward", "peak_epoch", "reward_collapse",
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _num_or_str(v):
    """Numbers become floats; anything else (e.g. 'mean', a checkpoint name) stays text."""
    if v is None or v == "":
        return None
    f = _f(v)
    return f if f is not None else v


def runs_dir_of(repo):
    return os.path.join(repo, "aerial_gym", "rl_training", "rl_games", "runs")


def collect_runs(repo):
    runs_dir = runs_dir_of(repo)
    runs = []
    for summ in sorted(glob(os.path.join(runs_dir, "*", "aerial_run", "run_summary.json"))):
        run_name = os.path.basename(os.path.dirname(os.path.dirname(summ)))
        try:
            with open(summ) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"warn: skipping {summ}: {exc}", file=sys.stderr)
            continue
        rec = {"run": run_name}
        for k in SUMMARY_FIELDS:
            rec[k] = data.get(k)
        runs.append(rec)
    # sort chronologically by finalized_at (missing sorts first)
    runs.sort(key=lambda r: r.get("finalized_at") or "")
    return runs


def read_result_csv(csv_path):
    """Read a results/*.csv that may be prefixed with '#' provenance comments.

    The leading comment block is provenance we want to keep (checkpoint, eval config,
    caveats), but csv.DictReader would otherwise consume the first comment line as the
    header and turn every data row into garbage.
    """
    notes, data_lines = [], []
    with open(csv_path) as fh:
        for line in fh:
            if not data_lines and line.lstrip().startswith("#"):
                notes.append(line.lstrip().lstrip("#").strip())
            elif line.strip():
                data_lines.append(line)
    rows = [{k: _num_or_str(v) for k, v in row.items()}
            for row in csv.DictReader(data_lines)]
    return notes, rows


def collect_curves(repo):
    """Collect results/*.csv, classified by the axis they sweep.

    density: capture/crash vs number of bars.  speed: capture/crash vs target speed.
    """
    density, speed, other = {}, {}, {}
    results_dir = os.path.join(repo, "results")
    for csv_path in sorted(glob(os.path.join(results_dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(csv_path))[0]
        notes, rows = read_result_csv(csv_path)
        if not rows:
            print(f"warn: no data rows in {csv_path}", file=sys.stderr)
            continue
        cols = set(rows[0])
        entry = {"notes": notes, "rows": rows,
                 "mtime": datetime.fromtimestamp(
                     os.path.getmtime(csv_path), timezone.utc).isoformat()}
        if {"bars", "density_bars"} & cols:
            density[name] = entry
        elif "target_speed_ms" in cols:
            speed[name] = entry
        else:
            other[name] = entry
    return density, speed, other


def collect_active_run(repo, stale_minutes=30.0):
    """The run that is training right now: epoch_metrics.csv but no run_summary.json yet.

    Metrics are per-epoch and noisy, so the headline numbers are means over the last
    `TAIL` epochs rather than the single last row.
    """
    TAIL = 50
    best = None
    for metrics in sorted(glob(os.path.join(
            runs_dir_of(repo), "*", "aerial_run", "epoch_metrics.csv"))):
        run_dir = os.path.dirname(metrics)
        if os.path.exists(os.path.join(run_dir, "run_summary.json")):
            continue  # already finalized -> covered by collect_runs
        mtime = os.path.getmtime(metrics)
        if best is None or mtime > best[0]:
            best = (mtime, metrics)
    if best is None:
        return None
    mtime, metrics = best
    age_min = (datetime.now(timezone.utc).timestamp() - mtime) / 60.0
    with open(metrics) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    tail = rows[-TAIL:]

    def avg(key):
        vals = [_f(r.get(key)) for r in tail]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    last = rows[-1]
    return {
        "run": os.path.basename(os.path.dirname(os.path.dirname(metrics))),
        "is_live": age_min <= stale_minutes,
        "metrics_age_min": round(age_min, 1),
        "epoch": _f(last.get("epoch")),
        "epochs_logged": len(rows),
        "tail_epochs": len(tail),
        "captured_rate": avg("captured_rate"),
        "crash_rate": avg("crash_rate"),
        "timeout_rate": avg("timeout_rate"),
        "mean_reward": avg("mean_reward"),
        "closest_approach_m": avg("closest_approach_m"),
        "n_bars_active": _f(last.get("n_bars_active")),
        "curriculum_max_m": avg("curriculum_max_m"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = os.path.abspath(os.path.expanduser(args.repo))
    # NOT docs/status/status.json. This legacy scraper predates the audit gates in
    # tools/update_status_snapshot.py and omits research_update, so writing the
    # canonical file here silently regressed the published dashboard. Overwriting
    # it now requires an explicit --out.
    out = args.out or os.path.join(repo, "docs", "status", "status.legacy.json")

    runs = collect_runs(repo)
    density, speed, other = collect_curves(repo)
    active = collect_active_run(repo)

    # Latest substantive navrl run. Runs that never learned anything (all-timeout
    # preflights, immediate aborts) are skipped so they can't become the headline.
    navrl_runs = [r for r in runs if r.get("is_navrl")]
    substantive = [r for r in navrl_runs
                   if (r.get("epochs_logged") or 0) >= 200
                   and (r.get("peak_captured_rate") or 0) > 0]
    latest = (substantive or navrl_runs or runs or [None])[-1]

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": repo,
        "n_runs": len(runs),
        "active_run": active,
        "latest_run": latest,
        "runs": runs,
        "density_curves": density,
        "speed_curves": speed,
        "other_curves": other,
    }

    def _jsonable(o):
        """Strip NaN/Inf so status.json stays strict JSON (browsers reject NaN)."""
        if isinstance(o, float):
            return None if (math.isnan(o) or math.isinf(o)) else o
        if isinstance(o, dict):
            return {k: _jsonable(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_jsonable(v) for v in o]
        return o

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(_jsonable(snapshot), fh, indent=2, ensure_ascii=False, allow_nan=False)
    print("wrote %s  (%d runs, %d density + %d speed + %d other curve(s))"
          % (out, len(runs), len(density), len(speed), len(other)))
    if active:
        print("active:", active["run"],
              "live=%s" % active["is_live"],
              "epoch=%s" % active.get("epoch"),
              "captured=%.3f" % (active.get("captured_rate") or 0),
              "bars=%s" % active.get("n_bars_active"))
    if latest:
        print("latest:", latest["run"],
              "captured=%.3f" % (latest.get("last_captured_rate") or 0),
              "crash=%.3f" % (latest.get("last_crash_rate") or 0),
              "bars=%s" % latest.get("last_n_bars_active"))


if __name__ == "__main__":
    main()
