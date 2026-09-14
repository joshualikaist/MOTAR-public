"""Independent count-based audit of the f9f3327 verification export; no simulator imports.

Run with Python 3. Outputs are isolated from the supplied evidence. Wilson intervals describe
single proportions; Wald contrasts/IVW summaries explicitly assume independent Bernoulli samples.
Those assumptions are not established by an aggregate CSV or by a common simulator seed.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXPORT = REPO / "results/verification_export_2026-09-06"
Z = 1.959963984540054


def proportion(row, metric="crash"):
    n, count = int(row["actual_episodes"]), int(row["n_" + metric])
    p = count / n
    divisor = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / divisor
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / divisor
    return {"count": count, "n": n, "percent": 100 * p,
            "wilson95_percent": [100 * (center - half), 100 * (center + half)]}


def contrast(a, b, metric="crash"):
    pa, pb = proportion(a, metric), proportion(b, metric)
    p, q = pa["percent"] / 100, pb["percent"] / 100
    d = 100 * (p - q)
    se = 100 * math.sqrt(p * (1 - p) / pa["n"] + q * (1 - q) / pb["n"])
    return {"a": a["root"] + "/" + a["cell"], "b": b["root"] + "/" + b["cell"],
            "metric": metric, "delta_pp": d, "se_pp": se,
            "wald95_pp": [d - Z * se, d + Z * se],
            "p_two_sided": math.erfc(abs(d / se) / math.sqrt(2)) if se else None,
            "cross_root": a["root"] != b["root"]}


def combine(values):
    weights = [1 / v["se_pp"] ** 2 for v in values]
    total = sum(weights)
    normalized = [w / total for w in weights]
    mean = sum(w * v["delta_pp"] for w, v in zip(normalized, values))
    independent_variance = 1 / total
    correlated_max_variance = sum(w * v["se_pp"] for w, v in zip(normalized, values)) ** 2
    sensitivity = {}
    for rho in (0, .25, .5, .75, 1):
        se = math.sqrt((1 - rho) * independent_variance + rho * correlated_max_variance)
        sensitivity[str(rho)] = {"se_pp": se, "wald95_pp": [mean - Z * se, mean + Z * se],
                                 "p_two_sided": math.erfc(abs(mean / se) / math.sqrt(2))}
    return {"delta_pp": mean, "weights": normalized,
            "common_effect_independence_assumption": sensitivity["0"],
            "illustrative_between_density_correlation_sensitivity": sensitivity,
            "rho_where_upper_ci_reaches_zero":
                ((mean / Z) ** 2 - independent_variance) /
                (correlated_max_variance - independent_variance),
            "Q": sum(w * (v["delta_pp"] - mean) ** 2 for w, v in zip(weights, values))}


def holm(values):
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    adjusted = [0.] * len(values)
    last = 0.
    for rank, (index, value) in enumerate(ordered):
        last = max(last, min(1., (len(values) - rank) * value))
        adjusted[index] = last
    return adjusted


def main():
    with (EXPORT / "governor_cells.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_key = {(r["root"], r["cell"]): r for r in rows}
    issues, output_rows = [], []
    for r in rows:
        path = REPO / r["result_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != r["result_sha256"]:
            issues.append([r["cell"], "result SHA mismatch"])
        raw = json.loads(path.read_text())
        n = int(r["actual_episodes"])
        if sum(int(r["n_" + k]) for k in ("captured", "crash", "timeout")) != n:
            issues.append([r["cell"], "outcome counts do not sum to actual_episodes"])
        entry = {"root": r["root"], "cell": r["cell"]}
        for metric in ("captured", "crash", "timeout"):
            p = proportion(r, metric)
            entry[metric] = p
            if int(r["n_" + metric]) != raw["outcome"][metric]:
                issues.append([r["cell"], "CSV vs raw count mismatch", metric])
            if abs(p["percent"] / 100 - float(r["rate_" + metric])) > 1e-12:
                issues.append([r["cell"], "CSV rate mismatch", metric])
            key = "capture_rate" if metric == "captured" else metric + "_rate"
            if abs(p["percent"] / 100 - raw["outcome"][key]) > 1e-12:
                issues.append([r["cell"], "raw rate mismatch", metric])
        if r["contacts"]:
            total = sum(int(r["contact_" + k]) for k in
                        ("vertical_out", "behind", "lateral", "no_return", "in_corridor"))
            if total != int(r["contacts"]):
                issues.append([r["cell"], "contact category sum mismatch"])
            entry["recorded_contacts_per_crash"] = int(r["contacts"]) / int(r["n_crash"])
        output_rows.append(entry)

    d1 = {}
    for baseline in ("riskcap", "stopcap", "off"):
        values = []
        for bars in (70, 100, 130, 160, 205):
            prefix = f"ep25000_d{bars:03d}_"
            a = by_key[("D1p_ep25000_s523", prefix + "dwa_arc")]
            b = by_key[("D1p_ep25000_s523", prefix + baseline)]
            value = contrast(a, b)
            value["bars"] = bars
            values.append(value)
        for v, adjusted in zip(values, holm([v["p_two_sided"] for v in values])):
            v["holm_p_5_densities"] = adjusted
        d1[baseline] = {"individual": values, "combined": combine(values)}

    d4 = []
    for policy in ("T0", "T1"):
        d4.append(contrast(by_key[("D4_coadapt_s521", policy + "_d070_stopcap")],
                           by_key[("D4_coadapt_s521", policy + "_d070_riskcap")]))
    diff = d4[1]["delta_pp"] - d4[0]["delta_pp"]
    se = math.hypot(d4[0]["se_pp"], d4[1]["se_pp"])
    interaction = {"T1_minus_T0_law_effect_pp": diff,
                   "wald95_pp": [diff - Z * se, diff + Z * se],
                   "p_two_sided": math.erfc(abs(diff / se) / math.sqrt(2))}

    width, adaptive = [], []
    for bars in (70, 205):
        prefix = f"ep25000_d{bars:03d}_"
        for mode in ("dwa_arc", "stopcap"):
            root = "L1_width_s523"
            a = by_key[(root, prefix + mode + "_w1p2")]
            b = by_key[(root, prefix + mode + "_w0p45")]
            for metric in ("crash", "captured", "timeout"):
                width.append(contrast(a, b, metric))
        root = "L1b_width_adaptive_s523"
        for end in ("2p0", "3p0"):
            for metric in ("crash", "captured"):
                width.append(contrast(by_key[(root, prefix + "arc_w" + end)],
                                      by_key[(root, prefix + "arc_w1p6")], metric))
        # Explicitly mark the only comparisons spanning the source boundary.
        for end in ("1p6", "3p0"):
            width.append(contrast(by_key[(root, prefix + "arc_w" + end)],
                                  by_key[("L1_width_s523", prefix + "dwa_arc_w1p2")]))
        for k in ("0p3", "0p6", "1p0"):
            adaptive.append(contrast(by_key[(root, prefix + "arc_open" + k)],
                                     by_key[(root, prefix + "arc_w1p6")]))
    summary = {"input_cells": len(rows), "input_columns": len(rows[0]), "integrity_issues": issues,
               "cell_rates_wilson95": output_rows, "D1p": d1, "D4": d4,
               "D4_interaction": interaction, "width_contrasts": width,
               "adaptive_minus_fixed1p6_same_root": adaptive,
               "five_of_five_sign_test_if_independent": {"one_sided": 1/32, "two_sided": 2/32},
               "assumption": "Nominal independent-episode Wald; no seed/layout clustering adjustment; "
                             "IVW additionally assumes independent density effects and a common estimand."}
    (OUT / "recomputed.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "cell_rates.csv").open("w", newline="") as stream:
        fields = ["root", "cell", "n"] + [f"{m}_{k}" for m in ("captured", "crash", "timeout")
                                                 for k in ("count", "percent", "wilson_lo", "wilson_hi")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for e in output_rows:
            flat = {"root": e["root"], "cell": e["cell"], "n": e["crash"]["n"]}
            for m in ("captured", "crash", "timeout"):
                flat.update({m + "_count": e[m]["count"], m + "_percent": e[m]["percent"],
                             m + "_wilson_lo": e[m]["wilson95_percent"][0],
                             m + "_wilson_hi": e[m]["wilson95_percent"][1]})
            writer.writerow(flat)
    print(json.dumps({"cells": len(rows), "issues": issues, "D4_interaction": interaction,
                      "D1p_riskcap_combined": d1["riskcap"]["combined"]}, indent=2))


if __name__ == "__main__":
    main()
