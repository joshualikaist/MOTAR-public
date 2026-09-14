"""Apply the P10 seed-replication decision rules exactly as preregistered on 2026-09-10.

Both rules were fixed before any training started, in
docs/preregistration_p10_seed_replication_2026-09-10.md:

  R1  sign rule (the R-C precedent): of the two new training seeds, how many have a positive
      within-campaign pooled primary estimand. Zero of two withdraws the claim.
  R2  seed as the replication unit (the arc-geometry precedent): mean of the per-seed pooled primary
      estimand over three training seeds, two-sided 95 % CI from Student's t at df = 2.

REPLICATED needs R1 = 2/2 and an R2 interval excluding zero. WITHDRAWN on R1 = 0/2. Anything else,
including 2/2 with an interval spanning zero, is INCONCLUSIVE and is reported as such.

Every estimand is computed inside one campaign, so a determinism failure in the frozen-policy arms
cannot distort it. Whether those arms reproduce P10's counts is reported either way.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

Z = 1.959963984540054
T_DF2_95 = 4.302652729911275   # two-sided 95 % Student's t, df = 2
ARMS = ("source_clean", "source_p9", "adapted_clean", "adapted_p9")
EVAL_SEEDS = (541, 547)


def pooled_rate(cells, arm, outcome):
    selected = [c for name, c in cells.items() if name.startswith(arm + "_s")]
    episodes = sum(c["episodes"] for c in selected)
    return sum(c[outcome] for c in selected) / episodes, episodes


def contrast(cells, a, b, outcome="captured"):
    pa, na = pooled_rate(cells, a, outcome)
    pb, nb = pooled_rate(cells, b, outcome)
    delta = 100 * (pa - pb)
    se = 100 * math.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    return {"delta_pp": delta, "se_pp": se, "ci95": [delta - Z * se, delta + Z * se],
            "excludes_zero": (delta - Z * se) * (delta + Z * se) > 0,
            "a": "%d/%d" % (round(pa * na), na), "b": "%d/%d" % (round(pb * nb), nb)}


def load_campaign_from_grids(root, seed):
    """Cells for one new training seed, read from the two evaluation-seed grid roots."""
    cells = {}
    for eval_seed in EVAL_SEEDS:
        base = root / ("grid_s%d_eval%d" % (seed, eval_seed))
        for arm in ARMS:
            name = "%s_s%d" % (arm, eval_seed)
            result = json.loads((base / name / "205bars.json").read_text())
            cells[name] = {"captured": result["outcome"]["captured"], "crash": result["outcome"]["crash"],
                           "timeout": result["outcome"]["timeout"], "episodes": result["actual_episodes"],
                           "checkpoint_sha256": result["checkpoint_sha256"],
                           "p9_enabled": result["condition"]["p9_empirical_error_enabled"]}
    return cells


def determinism_check(new_cells, reference_cells):
    """Frozen-policy arms do not depend on the training seed, so they must reproduce exactly."""
    report = {}
    for arm in ("source_clean", "source_p9"):
        for eval_seed in EVAL_SEEDS:
            name = "%s_s%d" % (arm, eval_seed)
            new, old = new_cells.get(name), reference_cells.get(name)
            if new is None or old is None:
                report[name] = {"status": "MISSING"}
                continue
            same = all(new[k] == old[k] for k in ("captured", "crash", "timeout", "episodes"))
            report[name] = {"status": "IDENTICAL" if same else "DIFFERS",
                            "new": {k: new[k] for k in ("captured", "crash", "timeout", "episodes")},
                            "reference": {k: old[k] for k in ("captured", "crash", "timeout", "episodes")}}
    report["all_identical"] = all(v.get("status") == "IDENTICAL" for k, v in report.items() if k != "all_identical")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--new-seeds", type=int, nargs="+", default=[857, 863])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.repo / "results/perception_p10_2026-09-09/summary.json").read_text())
    campaigns = {int(original["training_seed"]): original["cells"]}
    for seed in args.new_seeds:
        campaigns[seed] = load_campaign_from_grids(args.root, seed)
    per_seed = {}
    for seed, cells in campaigns.items():
        per_seed[str(seed)] = {
            "primary_capture": contrast(cells, "adapted_p9", "source_p9"),
            "clean_retention_capture": contrast(cells, "adapted_clean", "source_clean"),
            "p9_cost_on_source": contrast(cells, "source_p9", "source_clean"),
            "p9_cost_on_adapted": contrast(cells, "adapted_p9", "adapted_clean"),
            "crash": contrast(cells, "adapted_p9", "source_p9", "crash"),
            "timeout": contrast(cells, "adapted_p9", "source_p9", "timeout"),
            "episodes": sum(c["episodes"] for c in cells.values())}
    new_deltas = [per_seed[str(s)]["primary_capture"]["delta_pp"] for s in args.new_seeds]
    r1_positive = sum(1 for d in new_deltas if d > 0)
    all_deltas = [per_seed[str(s)]["primary_capture"]["delta_pp"] for s in sorted(campaigns)]
    mean = statistics.mean(all_deltas)
    sd = statistics.stdev(all_deltas)
    half = T_DF2_95 * sd / math.sqrt(len(all_deltas))
    r2 = {"seeds": sorted(campaigns), "per_seed_delta_pp": all_deltas, "mean_pp": mean,
          "sd_pp": sd, "ci95": [mean - half, mean + half],
          "excludes_zero": (mean - half) * (mean + half) > 0, "df": len(all_deltas) - 1}
    if r1_positive == 0:
        verdict = "WITHDRAWN"
    elif r1_positive == len(args.new_seeds) and r2["excludes_zero"] and mean > 0:
        verdict = "REPLICATED"
    else:
        verdict = "INCONCLUSIVE"
    determinism = {str(s): determinism_check(campaigns[s], original["cells"]) for s in args.new_seeds}
    receipt = {"status": verdict, "stage": "P10-seed-replication",
               "preregistration": "docs/preregistration_p10_seed_replication_2026-09-10.md",
               "R1_sign_rule": {"new_seeds": args.new_seeds, "deltas_pp": new_deltas,
                                "positive": r1_positive, "required_for_replicated": len(args.new_seeds),
                                "withdraws_at": 0},
               "R2_seed_level_t": r2, "per_training_seed": per_seed,
               "determinism_of_frozen_arms": determinism,
               "note": "Every estimand is within-campaign. Three training seeds at one density in"
                       " simulation; not a seed-general PPO claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("per_training_seed", "determinism_of_frozen_arms")}, indent=2))


if __name__ == "__main__":
    main()
