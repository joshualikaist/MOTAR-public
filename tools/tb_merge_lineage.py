#!/usr/bin/env python3
"""Present a warm-start lineage as ONE TensorBoard curve instead of several disjoint ones.

Every ``--branch_run`` resume starts a fresh ``runs/<name>/`` folder. That keeps each run's
metrics clean, but a single continuous training lineage then shows up in TensorBoard as three or
four separate curves that each begin mid-air, which makes the actual learning trend unreadable.

TensorBoard treats *all* event files inside one directory as a single run and orders them by
global step, and the PPO step counter carries across a warm start. So the fix needs no retraining
and no rewriting of event data: symlink the lineage's event files into one directory.

    python tools/tb_merge_lineage.py                       # newest run's lineage
    python tools/tb_merge_lineage.py ppo_260731_2012_...   # a specific run's lineage
    python tools/tb_merge_lineage.py --list                # show every run's step range

The chain is walked backwards through ``<run>/aerial_run/resumed_from.txt``, written by
runner.py at warm-start. Runs that predate that file can be chained explicitly:

    python tools/tb_merge_lineage.py --chain runA runB runC

Symlinks only -- the source runs are never modified, and a link to the *live* run's event file
keeps updating as training continues.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_RUNS = (
    Path(__file__).resolve().parent.parent
    / "aerial_gym"
    / "rl_training"
    / "rl_games"
    / "runs"
)


def _read_resumed_from(run_dir: Path) -> str | None:
    marker = run_dir / "aerial_run" / "resumed_from.txt"
    if not marker.is_file():
        return None
    name = marker.read_text(encoding="utf-8").strip()
    return name or None


def walk_lineage(head: Path, runs_root: Path) -> list[Path]:
    """Oldest-first list of runs ending at ``head``, following resumed_from.txt backwards."""
    chain = [head]
    seen = {head.name}
    cursor = head
    while True:
        parent_name = _read_resumed_from(cursor)
        if parent_name is None:
            break
        if parent_name in seen:
            # A cycle would loop forever; a lineage is a chain by construction, so stop and say so.
            print(f"[tb-merge] cycle at {parent_name!r}, stopping the walk", file=sys.stderr)
            break
        parent = runs_root / parent_name
        if not parent.is_dir():
            print(f"[tb-merge] lineage breaks: {parent_name!r} is gone", file=sys.stderr)
            break
        chain.append(parent)
        seen.add(parent_name)
        cursor = parent
    chain.reverse()
    return chain


def event_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "summaries").glob("events.out.tfevents.*"))


def step_range(event_dir: Path, tag_hint: str = "aerial/mean_reward"):
    """(first, last, n, duplicated) over scalars, or None when TensorBoard is unavailable."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    if not event_dir.is_dir():
        return None
    try:
        acc = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags().get("scalars") or []
        if not tags:
            return None
        tag = tag_hint if tag_hint in tags else tags[0]
        steps = [s.step for s in acc.Scalars(tag)]
    except Exception:
        # Old//partial run folders (deleted summaries, truncated event files) are common in runs/.
        # Reporting them as "no scalars" keeps --list usable instead of aborting the whole listing.
        return None
    if not steps:
        return None
    return min(steps), max(steps), len(steps), len(steps) - len(set(steps))


def build(chain: list[Path], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Drop stale links first: a rerun after the lineage changed must not leave an orphan segment
    # behind, which would silently graft a dead branch onto the merged curve.
    for stale in out_dir.glob("events.out.tfevents.*"):
        if stale.is_symlink():
            stale.unlink()
    linked = 0
    for run_dir in chain:
        for ev in event_files(run_dir):
            (out_dir / ev.name).symlink_to(ev.resolve())
            linked += 1
    return linked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?", help="head run name or path (default: newest run folder)")
    ap.add_argument("--runs-root", default=str(DEFAULT_RUNS), help="runs/ directory")
    ap.add_argument("--out", default=None, help="merged dir name (default: _merged_<head>)")
    ap.add_argument("--chain", nargs="+", help="explicit oldest-first run names, bypassing resumed_from.txt")
    ap.add_argument("--list", action="store_true", help="print every run's step range and exit")
    args = ap.parse_args()

    runs_root = Path(args.runs_root).resolve()
    if not runs_root.is_dir():
        print(f"[tb-merge] no runs directory: {runs_root}", file=sys.stderr)
        return 2

    if args.list:
        for d in sorted(p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            rng = step_range(d / "summaries")
            src = _read_resumed_from(d)
            tail = f"  <- {src}" if src else ""
            print(f"{d.name:58s} {'steps %6d -> %6d  n=%d' % rng[:3] if rng else 'no scalars':34s}{tail}")
        return 0

    if args.chain:
        chain = [runs_root / name for name in args.chain]
        missing = [p.name for p in chain if not p.is_dir()]
        if missing:
            print(f"[tb-merge] not found: {', '.join(missing)}", file=sys.stderr)
            return 2
    else:
        if args.run:
            head = Path(args.run)
            if not head.is_absolute():
                head = runs_root / head.name
        else:
            candidates = [p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("_")]
            if not candidates:
                print("[tb-merge] runs/ is empty", file=sys.stderr)
                return 2
            head = max(candidates, key=lambda p: p.stat().st_mtime)
        if not head.is_dir():
            print(f"[tb-merge] not found: {head}", file=sys.stderr)
            return 2
        chain = walk_lineage(head, runs_root)

    # Named after the lineage ROOT, not the head: the root is the one name that does not change
    # when the lineage grows, so every resume refreshes the same merged view instead of spawning a
    # new one beside it -- which would recreate the exact fragmentation this tool exists to remove.
    out_dir = runs_root / (args.out or f"_merged_{chain[0].name}")
    linked = build(chain, out_dir)

    print(f"[tb-merge] lineage ({len(chain)} runs, oldest first):")
    for run_dir in chain:
        rng = step_range(run_dir / "summaries")
        span = "steps %d -> %d" % rng[:2] if rng else "no scalars"
        print(f"           {run_dir.name:58s} {span}")
    merged = step_range(out_dir)
    print(f"[tb-merge] {linked} event file(s) -> {out_dir}")
    if merged:
        first, last, n, dup = merged
        print(f"[tb-merge] merged view: steps {first} -> {last}  n={n}  duplicated={dup}")
        if dup:
            # Not an error: a resume replays the epochs between the checkpoint and where the
            # previous run actually stopped. Those steps really were trained twice.
            print("[tb-merge] duplicated steps are re-trained epochs from resuming behind the tip")
    print(f"[tb-merge] tensorboard --logdir {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
