#!/usr/bin/env python3
"""Publication figures for the target-motion result, from the canonical summary."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "results/target_motion_e0_e2_2026-09-18"
S = json.loads((PKG / "canonical_summary.json").read_text())
OUT = PKG / "figures"; OUT.mkdir(exist_ok=True)
DENS = (70, 115, 160, 205); SEEDS = ("4101", "4102", "4103")
ARMS = ("H_historical", "E0_static", "E1_cv", "E2_obstacle_aware")
LBL = {"H_historical": "H historical (in-dist ref)", "E0_static": "E0 static",
       "E1_cv": "E1 CV", "E2_obstacle_aware": "E2 obstacle-aware"}
C = {"H_historical": "#22303f", "E0_static": "#2a9d8f",
     "E1_cv": "#e9c46a", "E2_obstacle_aware": "#e76f51"}
BOUND = ("Frozen policy, simulation only. n=3 evaluation seeds: intervals are preregistered BCa95; "
         "exact permutation cannot reach conventional significance at this seed count.")

def note(fig, extra=""):
    fig.text(0.005, -0.012, BOUND + (("\n" + extra) if extra else ""),
             fontsize=7, color="#555", va="top")

# F1 capture by arm x density, all seed points
d = S["density_descriptive"]["per_arm"]
fig, ax = plt.subplots(figsize=(9, 5))
for a in ARMS:
    ys = [d[a][str(x)]["capture_rate"] * 100 for x in DENS]
    ax.plot(DENS, ys, marker="o", color=C[a], label=LBL[a],
            lw=2.6 if a == "H_historical" else 1.7,
            ls="-" if a == "H_historical" else "--", zorder=3)
ax.set_xlabel("obstacle density (bars)"); ax.set_ylabel("close-approach rate (capture_rate, %)")
ax.set_title("F1 · Close-approach rate by target arm and obstacle density\n"
             "frozen policy, 2048 episodes/cell, 3 seeds pooled per point")
ax.set_xticks(DENS); ax.grid(alpha=.3); ax.legend(fontsize=8)
note(fig); fig.tight_layout(); fig.savefig(OUT / "F1_capture_by_arm_density.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# F2 H-relative paired effect with BCa CI + seed points
caps = [c for c in S["contrasts_vs_H"] if c["metric"] == "capture_rate"]
fig, ax = plt.subplots(figsize=(8, 4.6))
for i, c in enumerate(caps):
    m = c["mean_diff"] * 100; lo, hi = [x * 100 for x in c["bca95"]]
    ax.errorbar([m], [i], xerr=[[m - lo], [hi - m]], fmt="o", color=C[c["arm"]],
                capsize=5, markersize=9, lw=2, zorder=3)
    for s in SEEDS:
        ax.plot(c["seed_diffs"][s] * 100, i, "|", color=C[c["arm"]], markersize=15, alpha=.85, zorder=4)
ax.axvline(0, color="#444", lw=1.2, zorder=1)
ax.set_yticks(range(len(caps))); ax.set_yticklabels([LBL[c["arm"]] for c in caps])
ax.set_xlabel("capture-rate difference vs arm H (pp)")
ax.set_title("F2 · Seed-paired capture effect against the in-distribution reference\n"
             "dot = mean, bar = preregistered BCa95, ticks = the three individual seed effects")
ax.grid(axis="x", alpha=.3)
note(fig, "All three contrasts are sign-consistent across seeds. No significance claim is made.")
fig.tight_layout(); fig.savefig(OUT / "F2_paired_effect_vs_H.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# F3 visibility and timeout
vis = S["observation_side_visibility"]["per_arm"]; am = S["arm_means"]
order = ("E0_static", "E1_cv", "H_historical", "E2_obstacle_aware")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
a1.bar(range(4), [vis[a]["visible_fraction_step_weighted"] for a in order],
       color=[C[a] for a in order], edgecolor="#222")
a1.set_xticks(range(4)); a1.set_xticklabels(["E0", "E1", "H", "E2"])
a1.set_ylabel("target visible fraction (pooled step-weighted)")
a1.set_title("overall target visibility"); a1.grid(axis="y", alpha=.3)
a2.bar(range(4), [am[a]["timeout_rate"] * 100 for a in order],
       color=[C[a] for a in order], edgecolor="#222")
for i, a in enumerate(order):
    a2.text(i, am[a]["timeout_rate"] * 100 + .08,
            f"vis={vis[a]['by_outcome_class']['timeout']:.4f}", ha="center", fontsize=8)
a2.set_xticks(range(4)); a2.set_xticklabels(["E0", "E1", "H", "E2"])
a2.set_ylabel("timeout rate (%)"); a2.set_title("timeout rate, labelled with timeout-episode visibility")
a2.grid(axis="y", alpha=.3)
fig.suptitle("F3 · Observation-side visibility and timeout, by target arm")
note(fig, "Association only. No mediation or counterfactual experiment was run.")
fig.tight_layout(); fig.savefig(OUT / "F3_visibility_and_timeout.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# F4 density attenuation (exploratory)
fig, ax = plt.subplots(figsize=(9, 4.8))
for a in ("E0_static", "E1_cv", "E2_obstacle_aware"):
    ys = [(d[a][str(x)]["capture_rate"] - d["H_historical"][str(x)]["capture_rate"]) * 100 for x in DENS]
    ax.plot(DENS, ys, marker="o", color=C[a], label=f"{LBL[a]} − H", lw=1.8, ls="--")
ax.axhline(0, color="#22303f", lw=2, label="H reference")
ax.set_xlabel("obstacle density (bars)"); ax.set_ylabel("capture difference vs H (pp)")
ax.set_title("F4 · EXPLORATORY · Density attenuation of the target-behavior effect")
ax.set_xticks(DENS); ax.grid(alpha=.3); ax.legend(fontsize=8)
note(fig, "EXPLORATORY: per-density contrasts are not corrected and carry no confirmatory claim "
          "(Amendment 1 A1.3).")
fig.tight_layout(); fig.savefig(OUT / "F4_density_attenuation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote:", *[p.name for p in sorted(OUT.glob("*.png"))], sep="\n  ")
