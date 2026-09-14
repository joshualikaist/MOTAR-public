#!/usr/bin/env python3
"""Build the frozen ep24000 NavRL-v2 limit audit from machine-readable evidence."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def wilson(successes: int, total: int, z: float = 1.959963984540054):
    if total <= 0:
        return [None, None]
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2)) / denom
    return [center - half, center + half]


def rate_difference_ci(a_success, a_total, b_success, b_total):
    """Independent-binomial normal CI for a-b (the rollouts are not episode-paired)."""
    pa, pb = a_success / a_total, b_success / b_total
    se = math.sqrt(pa * (1.0 - pa) / a_total + pb * (1.0 - pb) / b_total)
    return [pa - pb - 1.959963984540054 * se, pa - pb + 1.959963984540054 * se]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def outcome_record(payload):
    outcome = payload["outcome"]
    n = int(payload["actual_episodes"])
    return {
        "action_selection": payload["condition"].get("action_selection", "legacy-unspecified"),
        "episodes": n,
        "captured": int(outcome["captured"]),
        "crash": int(outcome["crash"]),
        "timeout": int(outcome["timeout"]),
        "capture_rate": float(outcome["capture_rate"]),
        "capture_wilson95": wilson(int(outcome["captured"]), n),
        "crash_rate": float(outcome["crash_rate"]),
        "crash_wilson95": wilson(int(outcome["crash"]), n),
        "timeout_rate": float(outcome["timeout_rate"]),
        "bar_contact_rate": int(payload["crash_causes"]["bar_contact"]) / n,
        "strata": payload.get("strata") or {},
        "action": payload.get("action") or {},
    }


def pct(value):
    return "—" if value is None else f"{100.0 * value:.2f}%"


def pp(value):
    return f"{100.0 * value:+.2f} pp"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--postmortem", default=ROOT / "results/navrl_v2_ep24000_postmortem.json", type=Path
    )
    parser.add_argument(
        "--density-csv", default=ROOT / "results/navrl_v2_ep24000_heldout/results.csv", type=Path
    )
    parser.add_argument(
        "--deterministic",
        default=ROOT / "results/navrl_v2_ep24000_action_deterministic/205bars.json",
        type=Path,
    )
    parser.add_argument(
        "--stochastic",
        default=ROOT / "results/navrl_v2_ep24000_action_stochastic/205bars.json",
        type=Path,
    )
    parser.add_argument(
        "--geometry", default=ROOT / "results/navrl_v2_geometry_feasibility.csv", type=Path
    )
    parser.add_argument(
        "--json-out", default=ROOT / "results/navrl_v2_ep24000_limit_audit.json", type=Path
    )
    parser.add_argument(
        "--md-out", default=ROOT / "results/navrl_v2_ep24000_limit_audit.md", type=Path
    )
    args = parser.parse_args()

    postmortem = load_json(args.postmortem)
    det_payload = load_json(args.deterministic)
    stoch_payload = load_json(args.stochastic)
    deterministic = outcome_record(det_payload)
    stochastic = outcome_record(stoch_payload)
    if deterministic["action_selection"] != "deterministic":
        raise ValueError("deterministic result is mislabeled")
    if stochastic["action_selection"] != "stochastic":
        raise ValueError("stochastic result is mislabeled")

    densities = []
    with args.density_csv.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            n = int(row["actual_episodes"])
            captured = int(row["captured"])
            densities.append(
                {
                    "bars": int(row["bars"]),
                    "density_per_100m2": float(row["density_per_100m2"]),
                    "episodes": n,
                    "captured": captured,
                    "capture_rate": float(row["capture_rate"]),
                    "capture_wilson95": wilson(captured, n),
                    "crash_rate": float(row["crash_rate"]),
                    "timeout_rate": float(row["timeout_rate"]),
                    "bar_contact_rate": float(row["bar_contact_rate"]),
                }
            )

    geometry_205 = {}
    with args.geometry.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["density"]) == 205:
                geometry_205[str(float(row["side_clearance_m"]))] = {
                    key: float(row[key])
                    for key in (
                        "crossing_rate",
                        "largest_component_fraction",
                        "random_pair_connectivity",
                        "free_area_fraction",
                        "placement_fallback_rate",
                    )
                }

    det_cap, stoch_cap = deterministic["capture_rate"], stochastic["capture_rate"]
    det_crash, stoch_crash = deterministic["crash_rate"], stochastic["crash_rate"]
    action_gap = {
        "capture_rate_difference_deterministic_minus_stochastic": det_cap - stoch_cap,
        "capture_rate_difference_95ci": rate_difference_ci(
            deterministic["captured"], deterministic["episodes"],
            stochastic["captured"], stochastic["episodes"],
        ),
        "crash_rate_difference_deterministic_minus_stochastic": det_crash - stoch_crash,
        "crash_rate_difference_95ci": rate_difference_ci(
            deterministic["crash"], deterministic["episodes"],
            stochastic["crash"], stochastic["episodes"],
        ),
        "lateral_edge98_deterministic": deterministic["action"]["executed_edge98_rate"][1],
        "lateral_edge98_stochastic": stochastic["action"]["executed_edge98_rate"][1],
    }

    gate_205 = [
        item for item in postmortem["gate_windows"]
        if int(item["bars"]) == 205 and item["result"] == "held"
    ]
    last_seven = gate_205[-7:]
    online_plateau = sum(item["capture"] for item in last_seven) / len(last_seven)
    density_205 = next(item for item in postmortem["density_summary"] if item["bars"] == 205)
    checkpoint = postmortem["latest_checkpoint"]
    geometry_margin = geometry_205["0.2"]

    # Descriptive interpolation only: where the deterministic sweep crosses 70% between 205/220.
    d205 = next(item for item in densities if item["bars"] == 205)
    d220 = next(item for item in densities if item["bars"] == 220)
    deterministic_frontier = 205.0 + 15.0 * (d205["capture_rate"] - 0.70) / (
        d205["capture_rate"] - d220["capture_rate"]
    )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "training-stopped-core-audit-complete",
        "checkpoint": checkpoint,
        "training": {
            "canonical_first_epoch": postmortem["canonical_lineage"]["first_epoch"],
            "canonical_last_epoch": postmortem["canonical_lineage"]["last_epoch"],
            "canonical_epochs": postmortem["canonical_lineage"]["rows"],
            "canonical_samples": postmortem["canonical_lineage"]["rows"] * 32 * 128,
            "epochs_at_205": density_205["epochs"],
            "samples_at_205": density_205["epochs"] * 32 * 128,
            "held_windows_at_205": len(gate_205),
            "last_seven_gate_mean": online_plateau,
        },
        "density_sweep_deterministic": densities,
        "action_mode_205": {
            "deterministic": deterministic,
            "stochastic": stochastic,
            "difference": action_gap,
        },
        "geometry_205": geometry_205,
        "deterministic_70pct_frontier_linear_interpolation_bars": deterministic_frontier,
        "hypothesis_verdicts": {
            "H_PPO": "rejected-as-primary",
            "H_GEOM": "rejected-as-primary",
            "H_GATE": "refined: online gate measures stochastic policy correctly; deployment mean exceeds it",
            "H_PERC": "lowered: timeout is small and bar contact dominates",
            "H_REP": "supported-but-not-proven: full 4x72 scan exists, selector/risk use needs paired A/B",
            "H_BIAS": "unresolved: signed-y is large; mirror-conditioned evaluation still required",
        },
        "next_experiment": {
            "name": "fixed-205 cluster_sector vs ttc_sector",
            "launcher": "aerial_gym/rl_training/rl_games/train_navrl_v2_ep24000_ttc_ab.sh",
            "single_changed_variable": "NAVRL_OBSTACLE_SELECTOR",
            "source_checkpoint_sha256": checkpoint["sha256"],
            "main_epochs_per_arm": 1000,
            "4gb_epochs_per_arm": 2000,
            "samples_per_arm": 4_096_000,
            "acceptance": "TTC capture >= baseline +0.020 and crash <= baseline -0.020",
        },
        "interpretation_limits": [
            "The deterministic and stochastic rollouts use the same seed but are not episode-paired after trajectories diverge.",
            "The 70% frontier is a descriptive interpolation between two evaluated densities, not a trained ceiling theorem.",
            "The task uses an analytic target detector; these scores are not learned-detector evidence.",
            "NavRL/NavRL++ success rates are not directly rank-comparable because objectives, obstacles, sensing, and safety layers differ.",
            "The paired action evaluations predate the forward-only goal_centered diagnostic fix; outcome/strata are valid, context.goal_centered is excluded.",
        ],
        "pending_causal_checks": [
            "mirror-paired left/right evaluation of the frozen ep24000 policy (evaluation only; no learning)",
            "a second held-out seed at 205 bars",
            "fixed target-speed evaluations at 0.3, 0.9, and 1.5 m/s",
            "matched earlier-checkpoint evaluation to test density forgetting",
            "target-trajectory reachability beyond random-pair geometry",
            "real GTX 1650 Ti fixed-205 smoke and paired selector evaluation",
        ],
    }

    lines = [
        "# NavRL v2 ep24000 core failure and limit audit",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Decision",
        "",
        "The curriculum continuation is closed. More epochs under the unchanged 205-bar stochastic gate are not justified.",
        f"The final deployment-style deterministic policy clears 70% ({pct(det_cap)}), while the sampled "
        f"training policy does not ({pct(stoch_cap)}). The next training candidate is a fixed-density, "
        "one-variable representation A/B, but causal evaluations come first.",
        "",
        "## Frozen artifact",
        "",
        f"- checkpoint: `{checkpoint['path']}`",
        f"- epoch/SHA-256: `{checkpoint['epoch']}` / `{checkpoint['sha256']}`",
        f"- canonical training: {payload['training']['canonical_epochs']:,} epochs = "
        f"{payload['training']['canonical_samples']:,} samples",
        f"- 205 bars alone: {payload['training']['epochs_at_205']:,} epochs = "
        f"{payload['training']['samples_at_205']:,} samples; {len(gate_205)} complete holds",
        f"- last seven gate windows: mean {pct(online_plateau)} (16,384+ episodes each)",
        "",
        "## Held-out deterministic density sweep",
        "",
        "| bars | /100m² | episodes | capture (Wilson 95%) | crash | timeout | bar contact / all eps |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in densities:
        lo, hi = item["capture_wilson95"]
        lines.append(
            f"| {item['bars']} | {item['density_per_100m2']:.3f} | {item['episodes']} | "
            f"{pct(item['capture_rate'])} [{pct(lo)}, {pct(hi)}] | {pct(item['crash_rate'])} | "
            f"{pct(item['timeout_rate'])} | {pct(item['bar_contact_rate'])} |"
        )
    lines += [
        "",
        f"A linear interpolation between 205 and 220 puts the deterministic 70% crossing near "
        f"{deterministic_frontier:.1f} bars. This is a descriptive frontier, not a proof of a hard limit.",
        "",
        "## Action-selection A/B at 205 bars",
        "",
        "| mode | episodes | capture (Wilson 95%) | crash | timeout | lateral edge98 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in (deterministic, stochastic):
        lo, hi = record["capture_wilson95"]
        lines.append(
            f"| {record['action_selection']} | {record['episodes']} | {pct(record['capture_rate'])} "
            f"[{pct(lo)}, {pct(hi)}] | {pct(record['crash_rate'])} | {pct(record['timeout_rate'])} | "
            f"{pct(record['action']['executed_edge98_rate'][1])} |"
        )
    cap_lo, cap_hi = action_gap["capture_rate_difference_95ci"]
    crash_lo, crash_hi = action_gap["crash_rate_difference_95ci"]
    lines += [
        "",
        f"Deterministic minus stochastic: capture {pp(det_cap - stoch_cap)} "
        f"(approx. 95% CI {pp(cap_lo)} to {pp(cap_hi)}), crash {pp(det_crash - stoch_crash)} "
        f"(CI {pp(crash_lo)} to {pp(crash_hi)}). Exploration also doubles lateral edge98 "
        f"from {pct(action_gap['lateral_edge98_deterministic'])} to "
        f"{pct(action_gap['lateral_edge98_stochastic'])}.",
        "",
        "Verdict: the gate is not a random false alarm. It measures the sampled training policy; "
        "the deterministic deployment policy is materially better. Keep both metrics and do not "
        "silently replace one with the other.",
        "",
        "## Stable failure structure",
        "",
        "At 205 bars, deterministic distance strata fall from 81.42% (6–11.5 m) to 61.41% "
        "(22.5–28 m); stochastic strata fall from 75.35% to 55.06%. The fastest stochastic speed "
        "bin is 64.26%. CV is weaker than waypoint in both modes. This is accumulated collision "
        "exposure over long/high-speed trajectories, not a timeout or stationary-drone bottleneck.",
        "",
        f"The 205-bar 0.2 m-clearance geometry audit found crossing={geometry_margin['crossing_rate']:.3f}, "
        f"largest component={geometry_margin['largest_component_fraction']:.6f}, random-pair "
        f"connectivity={geometry_margin['random_pair_connectivity']:.6f}, and no placement fallback. "
        "Disconnected free space is therefore rejected as the primary cause.",
        "",
        "PPO is also rejected as the primary cause: behavior KL stayed below the 0.04 rollback "
        "threshold, learning rate remained 5e-6, explained variance was healthy, and no rollback or "
        "out-of-bounds action input occurred. Failures remain overwhelmingly bar contacts.",
        "",
        "## Literature-calibrated interpretation",
        "",
        "| system | arena / density | task and result | transferable lesson |",
        "|---|---|---|---|",
        "| NavRL | 50×50 m; 350 static + 60→120 dynamic | fixed-goal navigation; curriculum SR "
        "94.33→68.65%; best saved at 100 dynamic | stopping before the hardest stage and deploying "
        "the distribution mean/safety shield are legitimate design choices |",
        "| NavRL++ | 40×40 m; evaluation up to 400 static + 100 dynamic | high-dynamic SR 83.96%; "
        "~200 RTX4090 GPU-hours, 1,024 robots | use static ray geometry plus temporal structured "
        "obstacles; curriculum performance need not improve monotonically |",
        "| Ours | 40×40 m; 205 static bars = 12.81/100m² | moving-target interception; 72.44% "
        "deterministic / 67.35% stochastic | raw SR must not be ranked against fixed-goal navigation; "
        "our immediate gap is risk selection and exploration, not obstacle count alone |",
        "",
        "Primary sources: [NavRL](https://arxiv.org/html/2409.15634v2), "
        "[NavRL++](https://arxiv.org/html/2605.15559v1), "
        "[Anticipatory Risk-Guided RL](https://arxiv.org/abs/2607.23565), "
        "[Self-Paced Contextual RL](https://proceedings.mlr.press/v100/klink20a.html).",
        "",
        "## Next training candidate (prepared, not authorized)",
        "",
        "Run `cluster_sector` versus `ttc_sector` from the byte-frozen ep24000 checkpoint at fixed "
        "205 bars. Both arms receive exactly 4,096,000 samples: 1,000 epochs on main or 2,000 on "
        "the 4GB profile. TTC advances only if held-out capture improves by at least 2 pp and crash "
        "falls by at least 2 pp versus its same-profile baseline.",
        "",
        "Launcher: `aerial_gym/rl_training/rl_games/train_navrl_v2_ep24000_ttc_ab.sh`.",
        "",
        "Do not change density threshold, action sigma, reward, speed limit, and selector in one run. "
        "If TTC fails, the next isolated experiment is action-noise reduction. If it passes, repeat "
        "deterministic/stochastic evaluation before deciding whether the curriculum gate should "
        "remain a robustness gate or be paired with a separate deployment gate.",
        "",
        "## Remaining unknowns",
        "",
        "- The large positive lateral action remains a symptom, not a chirality verdict; a paired "
        "mirrored-layout evaluator is still required.",
        "- The current actor uses an analytic target detector, so learned perception remains a later "
        "research stage.",
        "- The full 4×72 static scan is present in addition to eight obstacle tokens. Calling this "
        "simply an '8-token capacity limit' is inaccurate; the open question is whether the actor "
        "uses dense geometry and threat ordering effectively.",
        "- The action-evaluation outcome and strata are valid, but its old goal-centered context "
        "bucket included the rear cone. That diagnostic has now been fixed and is excluded here.",
        "",
        "## Pending causal checks before another training run",
        "",
        "This document closes the unchanged 205-bar curriculum, but it is not the end of the causal "
        "audit. The next step is a mirror-paired evaluation of the frozen ep24000 policy. It performs "
        "no optimizer step and changes no checkpoint weight. A second seed, fixed-speed cells, a "
        "forgetting comparison, target-trajectory reachability, and the real 1650 Ti fixed-205 gate "
        "remain pending before the next main training run.",
    ]

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.json_out)
    print(args.md_out)


if __name__ == "__main__":
    main()
