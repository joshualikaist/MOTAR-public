#!/usr/bin/env python3
"""Which checkpoints may never be deleted, which are safe, and which are already lost.

Disk pressure keeps forcing checkpoint cleanups, and OPERATIONS.md used to state the rule in
prose. Prose lost: four checkpoints that committed results cite were deleted anyway, which is why
`tests/test_navrl_corrected_nonoverlap_heldout_contract.py` now skips instead of running. This tool
makes the rule mechanical -- it derives the keep-set from the evidence itself, so a cleanup cannot
depend on someone remembering which run mattered.

A checkpoint is KEPT when any of these hold:

  1. a result JSON under results/ names it (that number can then still be re-derived),
  2. a tracked launcher, spec or document names it (an experiment is pinned to it),
  3. it is the highest-epoch `last_gen_*` of its run (the run's terminal state; a run whose final
     weights are gone cannot be evaluated at all, only read about).

Everything else is an intermediate save and is safe to remove.

    python tools/audit_checkpoint_references.py               # report
    python tools/audit_checkpoint_references.py --verify      # exit 1 if a cited checkpoint is gone
    python tools/audit_checkpoint_references.py --list-deletable > /tmp/list.txt

Reads only. It never deletes; hand the list to a human-reviewed `xargs rm`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "aerial_gym/rl_training/rl_games/runs"
# Only real run checkpoints. Test fixtures ("hostile.pth"), documentation placeholders
# ("last_gen_ppo_ep_XXXX.pth"), detector artifacts under artifacts/ and the per-result
# checkpoint_snapshot.pth copies are all out of scope: none of them live under runs/.
RUN_CKPT = re.compile(r"^(?:last_gen_ppo_ep_\d+_rew_[-\d._]+|gen_ppo)\.pth$")
# A periodic save the trainer writes every N epochs. Only these are ever offered for deletion.
INTERMEDIATE = re.compile(r"^last_gen_ppo_ep_\d+_rew_[-\d._]+\.pth$")
PTH = re.compile(r"[A-Za-z0-9_.\-]+\.pth")
EPOCH = re.compile(r"_ep_(\d+)_")


def cited_by_results():
    """{checkpoint file name -> how many result JSONs name it}."""
    cited = {}
    for path in (REPO / "results").rglob("*.json"):
        text = str(path)
        if "source_snapshot" in text or "source_bundle" in text:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        value = data.get("checkpoint")
        # The path must point into runs/: `evaluated_checkpoint_snapshot` names the copy the
        # evaluator froze inside the result root, which is not what a cleanup can touch.
        if isinstance(value, str) and "/runs/" in value:
            name = Path(value).name
            if RUN_CKPT.match(name):
                cited[name] = cited.get(name, 0) + 1
    return cited


def cited_by_tracked_files():
    """Checkpoint names hard-coded in tracked launchers, specs and documents."""
    try:
        tracked = subprocess.check_output(
            ["git", "-C", str(REPO), "ls-files", "*.sh", "*.py", "*.md", "*.json"], text=True
        ).split()
    except subprocess.CalledProcessError:
        return {}
    cited = {}
    for rel in tracked:
        # tests/ names fixtures that never existed; results/ repeats what the JSON scan already has.
        if rel.startswith(("results/", "tests/")) or "source_snapshot" in rel:
            continue
        try:
            text = (REPO / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name in PTH.findall(text):
            if RUN_CKPT.match(name):
                cited.setdefault(name, set()).add(rel)
    return cited


def terminal_checkpoints():
    """The highest-epoch `last_gen_*` of every run: that run's final state.

    A run that saved only `gen_ppo.pth` keeps that instead -- it is the run's only weights, and a
    run with no weights at all can never be evaluated, only read about.
    """
    best, fallback = {}, {}
    for path in RUNS.rglob("*.pth"):
        run = path.relative_to(RUNS).parts[0]
        if path.name == "gen_ppo.pth":
            fallback.setdefault(run, path)
            continue
        if not path.name.startswith("last_gen"):
            continue
        match = EPOCH.search(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        if run not in best or epoch > best[run][0]:
            best[run] = (epoch, path)
    terminal = {run: path for run, (_, path) in best.items()}
    for run, path in fallback.items():
        terminal.setdefault(run, path)
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true",
                        help="exit 1 if any cited checkpoint is missing from disk")
    parser.add_argument("--list-deletable", action="store_true",
                        help="print only the safe-to-delete paths, one per line")
    args = parser.parse_args()

    if not RUNS.is_dir():
        raise SystemExit(f"[ckpt-audit] no runs directory at {RUNS}")

    on_disk = {path.name: path for path in RUNS.rglob("*.pth")}
    by_results = cited_by_results()
    by_files = cited_by_tracked_files()
    terminal = terminal_checkpoints()
    terminal_names = {path.name for path in terminal.values()}
    keep_names = set(by_results) | set(by_files) | terminal_names

    missing = sorted(name for name in set(by_results) | set(by_files) if name not in on_disk)
    # Fail safe: only a file the tool RECOGNISES as a periodic save can be deleted. Anything whose
    # name this tool does not model -- a `_rlnorm` variant, a hand-renamed export, a future naming
    # scheme -- is kept, because "I do not know what this is" must never mean "remove it".
    deletable = sorted(path for name, path in on_disk.items()
                       if name not in keep_names and INTERMEDIATE.match(name))
    unknown = sorted(path for name, path in on_disk.items()
                     if name not in keep_names and not INTERMEDIATE.match(name))

    if args.list_deletable:
        for path in deletable:
            print(path)
        return 0

    size = lambda paths: sum(p.stat().st_size for p in paths) / 1024 ** 3
    kept = [path for name, path in on_disk.items() if name in keep_names]
    print(f"[ckpt-audit] {len(on_disk)} checkpoints under {RUNS.relative_to(REPO)}")
    print(f"  keep      {len(kept):>4}  {size(kept):6.2f} GB   "
          f"(results {len(by_results)}, tracked files {len(by_files)}, terminal {len(terminal)})")
    print(f"  deletable {len(deletable):>4}  {size(deletable):6.2f} GB   intermediate saves")
    if unknown:
        print(f"  unknown   {len(unknown):>4}  {size(unknown):6.2f} GB   kept: name not recognised "
              f"({', '.join(p.name for p in unknown[:3])})")

    if missing:
        print(f"\n[ckpt-audit] MISSING -- cited but not on disk ({len(missing)}). "
              f"Whatever cites these can no longer be re-run:")
        for name in missing:
            where = []
            if name in by_results:
                where.append(f"{by_results[name]} result JSON")
            if name in by_files:
                where.append(", ".join(sorted(by_files[name])[:2]))
            print(f"  {name}  <- {' | '.join(where)}")
    else:
        print("\n[ckpt-audit] every cited checkpoint is present.")

    if args.verify:
        return 1 if missing else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
