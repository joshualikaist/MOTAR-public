#!/usr/bin/env python3
"""Pool governor contrasts across evaluation seeds and densities, and judge the frozen predictions.

The R-B replication (`docs/plans/confirmation_phase_plan_2026-09-06.md` §3) re-runs one grid under
new evaluation seeds. The claim it settles is not "the arc won in this run" but "the arc wins, by
this much, across seeds"; that needs inverse-variance pooling and a heterogeneity statistic, not a
sign count. This tool reads only committed result JSON and the contact-record sidecars, runs
nothing, and prints the same numbers the paper tables carry.

    python tools/pool_navrl_seed_replication.py results/navrl_grid_* --mechanism

Cells are keyed by (checkpoint, seed, bars, mode, half-width), so two roots that evaluated the same
condition are compared rather than silently deduplicated: divergence between them is the
cross-commit drift this project tracks, and it is reported, never averaged away.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("navrl_stats", Path(__file__).with_name("navrl_stats.py"))
stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stats)

# Arms are (governor mode, corridor half-width in metres). The default contrasts are the ones the
# confirmation plan pre-registered; anything else is a --contrast away.
DEFAULT_CONTRASTS = (
    ("arc-riskcap", ("dwa_arc", 0.45), ("riskcap", 0.45)),
    ("arc-stopcap", ("dwa_arc", 0.45), ("stopcap", 0.45)),
    ("stopcap-riskcap", ("stopcap", 0.45), ("riskcap", 0.45)),
    ("arcW-arc", ("dwa_arc", 1.2), ("dwa_arc", 0.45)),
)

# Mirrors the `predictions` block frozen in docs/specs/grid_r1_seedrep_ep25000_s*.json.
# tests/test_navrl_seed_replication.py asserts these numbers still appear in those spec files.
PREDICTIONS = {
    "P1": {"contrast": "arc-riskcap", "pooled_at_most": -0.8, "exclude_zero": True},
    "P2": {"contrast": "arc-riskcap", "negative_at_least": 4, "of_densities": 5},
    "P3": {"contrast": "arcW-arc", "crash_at_most": {70: -2.0, 205: -4.0}, "capture_at_least": 0.0},
    "P4": {"arm": ("stopcap", 1.2), "capture_below": {70: 60.0, 205: 20.0}},
    "P5": {"contrast": "stopcap-riskcap", "include_zero_at_every_density": True},
}

PINCH_NOTE = "share of contacts with a bar surface inside the pinch radius on BOTH sides at impact"


def load(roots):
    """(key -> cell) over every result JSON under the roots, reporting repeated conditions."""
    cells, repeats = {}, []
    for root in roots:
        for path in sorted(Path(root).glob("*/*bars.json")):
            if path.name.endswith(("receipt.json", "manifest.json")):
                continue
            data = json.loads(path.read_text())
            condition = data["condition"]
            key = (
                data.get("checkpoint_sha256", "")[:12],
                int(condition["seed"]),
                int(condition["bars"]),
                condition["speed_governor_mode"],
                round(float(condition["speed_governor_half_width_m"]), 3),
            )
            data["_root"], data["_name"], data["_path"] = str(root), path.parent.name, path
            if key in cells:
                repeats.append((key, cells[key], data))
                continue
            cells[key] = data
    return cells, repeats


def index_by_condition(cells):
    """(seed, bars, mode, width) -> cell, refusing to guess when two checkpoints share a condition."""
    index = {}
    for (ckpt, seed, bars, mode, width), cell in cells.items():
        condition = (seed, bars, mode, width)
        previous = index.get(condition)
        if previous is not None and previous["checkpoint_sha256"][:12] != ckpt:
            raise SystemExit(f"[pool] two checkpoints evaluated {condition}: "
                             f"{previous['checkpoint_sha256'][:12]} and {ckpt}; pass one lineage at a time")
        index[condition] = cell
    return index


def counts(cell, field="crash_rate"):
    return stats.outcome_count(cell["outcome"], field), int(cell["actual_episodes"])


def contrast_rows(cells, arm_a, arm_b, field="crash_rate"):
    """{(seed, bars): (delta, se)} for every seed and density where both arms were evaluated."""
    rows = {}
    for (ckpt, seed, bars, mode, width), cell in cells.items():
        if (mode, width) != arm_a:
            continue
        other = cells.get((ckpt, seed, bars) + arm_b)
        if other is None:
            continue
        rows[(seed, bars)] = stats.wald_diff(*counts(cell, field), *counts(other, field))
    return rows


def records(cell):
    path = (cell.get("contact_geometry") or {}).get("contact_records_path")
    path = Path(path) if path else cell["_path"].with_name(cell["_path"].name.replace(".json", ".contact_records.jsonl"))
    if not path.is_absolute():
        path = REPO / path
    if not path.is_file():
        path = cell["_path"].with_name(cell["_path"].name.replace(".json", ".contact_records.jsonl"))
    return [json.loads(line) for line in path.open()] if path.is_file() else []


def report_pool(name, rows, seeds):
    per_seed = []
    for seed in seeds:
        subset = [v for (s, _), v in sorted(rows.items()) if s == seed]
        if len(subset) >= 2:
            delta, se = stats.pool_fixed(subset)
            q, df, i2 = stats.cochran_q(subset)
            per_seed.append(delta)
            print(f"  {name:<16} seed {seed} n={len(subset):>2}  {stats.format_ci(delta, se):<24}"
                  f"  Q={q:5.1f} ({df})  I2={100 * i2:3.0f}%")
    # The seed-level answer to "would a NEW evaluation seed show this?". Cells inside one seed
    # share a policy, a scene sampler and an RNG stream, so the cell-level interval below is a
    # within-cell precision statement, not a seed-generalisation one. Both are printed on purpose.
    if len(per_seed) >= 2:
        mean, se, lo, hi, p, df = stats.seed_level_t(per_seed)
        print(f"  {name:<16} SEED-LEVEL k={len(per_seed)}  {mean:+.2f} [{lo:+.2f}, {hi:+.2f}]"
              f"      t({df}) p={p:.4f}   <- replication unit = seed")
    allrows = list(rows.values())
    if len(allrows) >= 2:
        delta, se = stats.pool_fixed(allrows)
        q, df, i2 = stats.cochran_q(allrows)
        rdelta, rse, tau2 = stats.pool_random(allrows)
        print(f"  {name:<16} ALL      n={len(allrows):>2}  {stats.format_ci(delta, se):<24}"
              f"  Q={q:5.1f} ({df})  I2={100 * i2:3.0f}%  p={stats.two_sided_p(delta, se):.2e}"
              f"  RE {stats.format_ci(rdelta, rse)} tau2={tau2:.2f}")
    return allrows


def judge(cells, contrasts, seeds, densities, baseline_seed):
    """Frozen predictions P1-P5. A prediction with missing cells is 'pending', never a pass."""
    print("\n== frozen predictions (docs/specs/grid_r1_seedrep_ep25000_s*.json) ==")
    arc_risk = contrasts.get("arc-riskcap", {})
    if len(arc_risk) >= 2:
        delta, se = stats.pool_fixed(list(arc_risk.values()))
        spec = PREDICTIONS["P1"]
        ok = delta <= spec["pooled_at_most"] and stats.excludes_zero(delta, se)
        print(f"P1 pooled arc-riskcap {stats.format_ci(delta, se)} over {len(arc_risk)} cells "
              f"(need <= {spec['pooled_at_most']} and CI excluding 0): {'PASS' if ok else 'FAIL'}")

    for seed in seeds:
        if seed == baseline_seed:
            continue
        present = [b for b in densities if (seed, b) in arc_risk]
        negative = [b for b in present if arc_risk[(seed, b)][0] < 0]
        need = PREDICTIONS["P2"]["negative_at_least"]
        verdict = "PASS" if len(negative) >= need else ("pending" if len(present) < PREDICTIONS["P2"]["of_densities"] else "FAIL")
        print(f"P2 seed {seed}: arc-riskcap negative at {len(negative)}/{len(present)} densities "
              f"(need >= {need}): {verdict}")

    widening = contrasts.get("arcW-arc", {})
    capture = contrast_rows(cells, ("dwa_arc", 1.2), ("dwa_arc", 0.45), field="capture_rate")
    for seed in seeds:
        for bars, threshold in sorted(PREDICTIONS["P3"]["crash_at_most"].items()):
            if (seed, bars) not in widening:
                continue
            dcrash = widening[(seed, bars)][0]
            dcap = capture[(seed, bars)][0]
            ok = dcrash <= threshold and dcap >= PREDICTIONS["P3"]["capture_at_least"]
            print(f"P3 seed {seed} {bars:>3} bars: crash {dcrash:+.2f} (need <= {threshold}), "
                  f"capture {dcap:+.2f} (need >= 0): {'PASS' if ok else 'FAIL'}")

    mode, width = PREDICTIONS["P4"]["arm"]
    index = index_by_condition(cells)
    for seed in seeds:
        for bars, threshold in sorted(PREDICTIONS["P4"]["capture_below"].items()):
            cell = index.get((seed, bars, mode, width))
            if cell is None:
                continue
            captured, total = counts(cell, "capture_rate")
            print(f"P4 seed {seed} {bars:>3} bars: straight {width} m capture {100 * captured / total:.2f}% "
                  f"({captured}/{total}, need < {threshold}): "
                  f"{'PASS' if 100 * captured / total < threshold else 'FAIL'}")

    stop_risk = contrasts.get("stopcap-riskcap", {})
    for seed in seeds:
        rows = [(b, stop_risk[(seed, b)]) for b in densities if (seed, b) in stop_risk]
        if not rows:
            continue
        outside = [(b, stats.format_ci(*v)) for b, v in rows if stats.excludes_zero(*v)]
        verdict = "PASS" if not outside and len(rows) == len(densities) else ("FAIL" if outside else "pending")
        print(f"P5 seed {seed}: stopcap-riskcap CI includes 0 at {len(rows) - len(outside)}/{len(rows)} "
              f"densities: {verdict}" + (f"; outside: {outside}" if outside else ""))


def mechanism(cells):
    print(f"\n== contact records: pinch_t0 ({PINCH_NOTE}), |lateral offset at t-1 s| > 0.45 m ==")
    print(f"{'seed':>5}{'bars':>5}{'mode':>9}{'w':>6}{'contacts':>9}{'pinch':>8}{'off>0.45':>10}{'median':>8}")
    totals = defaultdict(lambda: [0, 0, 0, 0, 0])
    for (ckpt, seed, bars, mode, width) in sorted(cells, key=lambda k: (k[1], k[2], k[3], k[4])):
        cell = cells[(ckpt, seed, bars, mode, width)]
        rows = records(cell)
        if not rows:
            continue
        scored = [r for r in rows if r.get("pinch_t0") is not None]
        pinched = sum(1 for r in scored if r["pinch_t0"])
        offsets = [abs(r["hit_lateral_cmd"]) for r in rows
                   if r.get("hit_lateral_cmd") is not None and r.get("memory_window_valid", True)]
        far = sum(1 for o in offsets if o > 0.45)
        median = sorted(offsets)[len(offsets) // 2] if offsets else float("nan")
        bucket = totals[(mode, width)]
        for index, value in enumerate((len(rows), pinched, len(scored), far, len(offsets))):
            bucket[index] += value
        pinch_pct = 100 * pinched / len(scored) if scored else float("nan")
        far_pct = 100 * far / len(offsets) if offsets else float("nan")
        print(f"{seed:>5}{bars:>5}{mode:>9}{width:>6.2f}{len(rows):>9}{pinch_pct:>7.1f}%{far_pct:>9.1f}%{median:>8.2f}")
    print("-- pooled by arm:")
    for (mode, width), (n, pinched, scored, far, noff) in sorted(totals.items()):
        pinch_pct = 100 * pinched / scored if scored else float("nan")
        far_pct = 100 * far / noff if noff else float("nan")
        print(f"  {mode:>9} {width:.2f} m: contacts {n:>5}  pinch_t0 {pinch_pct:5.1f}% (n={scored})"
              f"  off>0.45 {far_pct:5.1f}% (n={noff})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--field", default="crash_rate")
    parser.add_argument("--baseline-seed", type=int, default=523,
                        help="the seed the predictions were written against; excluded from P2")
    parser.add_argument("--mechanism", action="store_true", help="add the contact-record table")
    parser.add_argument("--json", type=Path, help="write the pooled estimates here")
    args = parser.parse_args()

    cells, repeats = load(args.roots)
    if not cells:
        raise SystemExit("[pool] no result JSON under the given roots")
    seeds = sorted({k[1] for k in cells})
    densities = sorted({k[2] for k in cells})
    print(f"# {len(cells)} cells | seeds {seeds} | densities {densities} | field {args.field}")
    for key, first, second in repeats:
        a, b = counts(first, args.field), counts(second, args.field)
        verdict = "identical" if a == b else "DIVERGENT"
        print(f"# repeated condition {key}: {first['_name']}@{first['_root']} {a} vs "
              f"{second['_name']}@{second['_root']} {b} -> {verdict}")

    print(f"\n== per seed x density: {args.field} as count/n and % ==")
    header = f"{'seed':>5}{'bars':>5}" + "".join(f"{m + '@' + format(w, '.2f'):>18}" for m, w in
                                                 sorted({(k[3], k[4]) for k in cells}))
    print(header)
    arms = sorted({(k[3], k[4]) for k in cells})
    index = index_by_condition(cells)
    for seed in seeds:
        for bars in densities:
            row = f"{seed:>5}{bars:>5}"
            for mode, width in arms:
                cell = index.get((seed, bars, mode, width))
                if cell is None:
                    row += f"{'-':>18}"
                else:
                    c, n = counts(cell, args.field)
                    row += f"{c:>8}/{n:<5}{100 * c / n:>4.1f}"
            print(row)

    print("\n== inverse-variance pooling (fixed) | Cochran Q | DerSimonian-Laird random ==")
    contrasts, pooled = {}, {}
    for name, arm_a, arm_b in DEFAULT_CONTRASTS:
        rows = contrast_rows(cells, arm_a, arm_b, args.field)
        if not rows:
            continue
        contrasts[name] = rows
        report_pool(name, rows, seeds)
        if len(rows) >= 2:
            delta, se = stats.pool_fixed(list(rows.values()))
            q, df, i2 = stats.cochran_q(list(rows.values()))
            entry = {"delta_pp": delta, "se_pp": se, "ci_pp": list(stats.ci(delta, se)),
                     "cells": len(rows), "cochran_q": q, "df": df, "i_squared": i2,
                     "p_two_sided": stats.two_sided_p(delta, se)}
            per_seed = []
            for seed in seeds:
                subset = [v for (s_, _), v in sorted(rows.items()) if s_ == seed]
                if len(subset) >= 2:
                    per_seed.append(stats.pool_fixed(subset)[0])
            if len(per_seed) >= 2:
                mean, se_s, lo_s, hi_s, p_s, df_s = stats.seed_level_t(per_seed)
                entry["seed_level"] = {"mean_pp": mean, "se_pp": se_s, "ci_pp": [lo_s, hi_s],
                                       "p_two_sided": p_s, "df": df_s, "seeds": len(per_seed),
                                       "per_seed_pp": per_seed}
            pooled[name] = entry

    # Per density, both units side by side. A density whose seed-level interval covers zero is a
    # within-cell observation, not a seed-generalisable one; the 2026-09-07 audit's objection to
    # the 70-bar stopcap finding lands exactly here.
    print("\n== per density: cell-level pool (3 cells) vs seed-level t (k=3) ==")
    print(f"{'contrast':<16}{'bars':>5}{'cell-level':>26}{'seed-level t(2)':>28}{'p_seed':>9}")
    for name, rows in contrasts.items():
        for density in densities:
            per_seed = [v[0] for (s_, b_), v in sorted(rows.items()) if b_ == density]
            subset = [v for (s_, b_), v in sorted(rows.items()) if b_ == density]
            if len(per_seed) < 2:
                continue
            delta, se = stats.pool_fixed(subset)
            mean, se_s, lo_s, hi_s, p_s, _ = stats.seed_level_t(per_seed)
            print(f"{name:<16}{density:>5}{stats.format_ci(delta, se):>26}"
                  f"{f'{mean:+.2f} [{lo_s:+.2f}, {hi_s:+.2f}]':>28}{p_s:>9.3f}")

    judge(cells, contrasts, seeds, densities, args.baseline_seed)
    if args.mechanism:
        mechanism(cells)
    if args.json:
        args.json.write_text(json.dumps(
            {"seeds": seeds, "densities": densities, "field": args.field, "pooled": pooled,
             "cells": len(cells), "roots": [str(r) for r in args.roots]}, indent=2, sort_keys=True) + "\n")
        print(f"\n[pool] wrote {args.json}")


if __name__ == "__main__":
    main()
