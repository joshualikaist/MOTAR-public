"""Export every governor cell measured on 2026-09-05/06 as one flat CSV + a manifest.

For independent verification: one row per evaluation cell, with the condition that produced it,
the outcome counts (not just rates, so anyone can recompute rates and intervals), the governor
telemetry, and the contact-category counts. Nothing is aggregated across roots and nothing is
rounded away; the rate columns are recomputed from counts here rather than copied.
"""

import csv
import hashlib
import json
from pathlib import Path

REPO = Path("/home/fair/workspaces/aerial_gym_ws/src/aerial_gym_simulator")
ROOTS = {
    "A7_ref5in_s509": "results/navrl_arc_attribution_seed509",
    "A7_ref5in_s491": "results/navrl_arc_attribution_seed491",
    "A7_ep25000_205_s49": "results/navrl_arc_attribution_205bars_seed49",
    "A8_readapt_s521": "results/navrl_a8_readaptation_seed521",
    "D1gate_T0_s523": "results/navrl_grid_d1_gate_off_T0_seed523",
    "D1p_ep25000_s523": "results/navrl_grid_d1p_ep25000_seed523",
    "D4_coadapt_s521": "results/navrl_grid_d4_coadapt_seed521",
    "L1_width_s523": "results/navrl_grid_l1_ep25000_seed523",
    "L1b_width_adaptive_s523": "results/navrl_grid_l1b_ep25000_seed523",
}
COND = ("bars", "seed", "num_envs", "robot_name", "goal_dist_min_m", "goal_dist_max_m",
        "action_selection", "reflection_mode", "distractor_count",
        "speed_governor_mode", "speed_governor_half_width_m", "speed_governor_margin_m",
        "speed_governor_fixed_mps", "speed_governor_free_mps", "speed_governor_slow_m",
        "speed_governor_release_m", "speed_governor_brake_mps2", "speed_governor_reaction_s",
        "speed_governor_ttc_s", "speed_governor_width_per_mps",
        "speed_governor_width_per_open_m", "speed_governor_width_open_ref_m",
        "speed_governor_width_max_m", "speed_governor_lateral_margin_m",
        "speed_governor_yaw_cap_radps")
CATS = ("vertical_out", "behind", "lateral", "no_return", "in_corridor")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows():
    out, manifest = [], []
    for tag, rel in ROOTS.items():
        root = REPO / rel
        if not root.is_dir():
            continue
        cells_json = root / "cells.json"
        spec_env = {}
        if cells_json.is_file():
            spec = json.loads(cells_json.read_text())
            # cells.json is the authoritative record of what each cell was ASKED to run.
            for cell in spec.get("spec", {}).get("cells", []):
                spec_env[cell["name"]] = cell["env"]
            manifest.append({"root": rel, "commit": spec.get("evaluation_commit"),
                             "contract": spec.get("contract"),
                             "source_fingerprint": spec.get("evaluated_source_fingerprint"),
                             "cells_json_sha256": sha(cells_json)})
        for path in sorted(root.glob("*/*bars.json")):
            if path.name.endswith(("receipt.json", "manifest.json")):
                continue
            data = json.loads(path.read_text())
            cell = path.parent.name
            condition, outcome = data.get("condition", {}), data.get("outcome", {})
            governor = data.get("speed_governor", {}) or {}
            geometry = data.get("contact_geometry", {}) or {}
            commanded = geometry.get("commanded_direction", {}) or {}
            n = data.get("actual_episodes")
            row = {"root": tag, "cell": cell, "result_path": str(path.relative_to(REPO)),
                   "result_sha256": sha(path), "actual_episodes": n,
                   "checkpoint_sha256": (data.get("checkpoint_sha256") or "")[:16],
                   "runtime_git_commit": (data.get("runtime_git_commit") or "")[:12],
                   "runtime_source_manifest_sha256": (data.get("runtime_source_manifest_sha256") or "")[:16],
                   "runtime_git_dirty": data.get("runtime_git_dirty")}
            for key in COND:
                row["cond_" + key] = condition.get(key)
            # requested env from cells.json fills in knobs the result JSON does not record yet
            for key, value in (spec_env.get(cell) or {}).items():
                row["req_" + key.replace("NAVRL_SPEED_GOVERNOR", "gov").lower()] = value
            for key in ("captured", "crash", "timeout"):
                row["n_" + key] = outcome.get(key)
                count = outcome.get(key)
                row["rate_" + key] = (count / n) if (isinstance(count, int) and n) else None
            row["gov_intervention_rate"] = governor.get("intervention_rate")
            row["gov_mean_executed_mps"] = governor.get("mean_executed_speed_mps")
            row["gov_mean_requested_mps"] = governor.get("mean_requested_speed_mps")
            row["gov_samples"] = governor.get("samples")
            row["contacts"] = geometry.get("contacts")
            for cat in CATS:
                row["contact_" + cat] = commanded.get(cat)
            row["contact_records_path"] = geometry.get("contact_records_path")
            row["contact_records_rows"] = geometry.get("contact_records_rows")
            row["frame_samples_rows"] = geometry.get("frame_samples_rows")
            out.append(row)
    return out, manifest


def main():
    data, manifest = rows()
    fields = []
    for row in data:
        for key in row:
            if key not in fields:
                fields.append(key)
    out_dir = REPO / "results" / "verification_export_2026-09-06"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "governor_cells.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)
    (out_dir / "roots_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{len(data)} cells -> {csv_path}")
    print(f"{len(fields)} columns")
    for entry in manifest:
        print("  root", entry["root"], "commit", (entry["commit"] or "")[:8], "contract", entry["contract"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
