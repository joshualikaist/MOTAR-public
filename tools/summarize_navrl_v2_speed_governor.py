"""Validate and summarize the preregistered ep24000 speed-governor screen."""

import hashlib
import json
import math
from pathlib import Path
import sys


TAGS = ("off", "fixed2p0", "fixed1p5", "clearance", "ttc")
EXPECTED_MODE = {
    "off": "off",
    "fixed2p0": "fixed",
    "fixed1p5": "fixed",
    "clearance": "clearance",
    "ttc": "ttc",
}
SOURCE_SHA = "82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f"
COMMON_PARAMETERS = {
    "speed_governor_free_mps": 3.53553390593,
    "speed_governor_half_width_m": 0.45,
    "speed_governor_margin_m": 0.45,
    "speed_governor_slow_m": 3.0,
    "speed_governor_release_m": 5.0,
    "speed_governor_ttc_s": 1.2,
    "speed_governor_brake_mps2": 2.9608856678,
    "speed_governor_reaction_s": 0.1,
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diff_ci(success_a, n_a, success_b, n_b):
    pa, pb = success_a / n_a, success_b / n_b
    se = math.sqrt(pa * (1.0 - pa) / n_a + pb * (1.0 - pb) / n_b)
    return pa - pb, (pa - pb - 1.96 * se, pa - pb + 1.96 * se)


def load_cell(root, tag):
    path = root / tag / "205bars.json"
    receipt_path = root / tag / "205bars.receipt.json"
    log_path = root / tag / "205bars.log"
    snapshot_path = root / tag / "checkpoint_snapshot.pth"
    payload = json.loads(path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    condition = payload.get("condition") or {}
    contract = payload.get("v2_evaluation_contract") or {}
    outcome = payload.get("outcome") or {}
    governor = payload.get("speed_governor") or {}
    actual = int(payload.get("actual_episodes", -1))
    expected_mode = EXPECTED_MODE[tag]
    checks = {
        "schema": payload.get("schema_version") == 1,
        "episodes": int(payload.get("requested_episodes", -1)) == 2049 and actual >= 2049,
        "sha": payload.get("checkpoint_sha256") == SOURCE_SHA
        and payload.get("evaluated_checkpoint_snapshot_sha256") == SOURCE_SHA,
        "condition": int(condition.get("bars", -1)) == 205
        and int(condition.get("seed", -1)) == 42
        and condition.get("action_selection") == "deterministic"
        and condition.get("reflection_mode") == "original"
        and condition.get("target_speed_mode") == "uniform",
        "runtime": contract.get("runtime_profile") == "main"
        and contract.get("runtime_sim") == "base_sim"
        and int(contract.get("runtime_num_envs", -1)) == 128
        and contract.get("obstacle_selector") == "cluster_sector",
        "mode": condition.get("speed_governor_mode") == expected_mode
        and contract.get("speed_governor_mode") == expected_mode
        and governor.get("mode") == expected_mode,
        "sensor_only": governor.get("sensor_only") is True
        and governor.get("direction_preserved") is True
        and governor.get("target_exclusion_source") == "camera_lidar_association"
        and condition.get("speed_governor_target_exclusion") == "camera_lidar_association"
        and contract.get("speed_governor_target_exclusion") == "camera_lidar_association",
        "parameters": all(
            math.isclose(float(condition.get(key, float("nan"))), expected)
            and math.isclose(float(contract.get(key, float("nan"))), expected)
            for key, expected in {
                **COMMON_PARAMETERS,
                "speed_governor_fixed_mps": 1.5 if tag == "fixed1p5" else 2.0,
            }.items()
        ),
        "receipt": receipt.get("schema_version") == 1
        and int(receipt.get("bars", -1)) == 205
        and int(receipt.get("seed", -1)) == 42
        and int(receipt.get("requested_episodes", -1)) == 2049
        and int(receipt.get("actual_episodes", -1)) == actual
        and receipt.get("source_checkpoint_sha256") == SOURCE_SHA
        and receipt.get("evaluated_checkpoint_snapshot_sha256") == SOURCE_SHA
        and receipt.get("speed_governor_mode") == expected_mode
        and receipt.get("speed_governor_target_exclusion") == "camera_lidar_association"
        and receipt.get("evaluation_nonce") == condition.get("evaluation_nonce"),
        "artifact_digests": receipt.get("result_sha256") == sha256(path)
        and receipt.get("log_sha256") == sha256(log_path)
        and sha256(snapshot_path) == SOURCE_SHA,
        "counts": sum(int(outcome.get(k, -actual - 1)) for k in ("captured", "crash", "timeout")) == actual,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"invalid {tag} result {path}: {', '.join(failed)}")
    crash_causes = payload.get("crash_causes") or {}
    return {
        "tag": tag,
        "path": str(path),
        "mode": expected_mode,
        "episodes": actual,
        "captured": int(outcome["captured"]),
        "crashes": int(outcome["crash"]),
        "timeouts": int(outcome["timeout"]),
        "capture_rate": float(outcome["capture_rate"]),
        "crash_rate": float(outcome["crash_rate"]),
        "timeout_rate": float(outcome["timeout_rate"]),
        "bar_contact_rate": int(crash_causes.get("bar_contact", 0)) / actual,
        "intervention_rate": float(governor["intervention_rate"]),
        "near_stop_rate": float(governor["near_stop_rate"]),
        "mean_requested_speed_mps": float(governor["mean_requested_speed_mps"]),
        "mean_executed_speed_mps": float(governor["mean_executed_speed_mps"]),
        "mean_scale": float(governor["mean_scale"]),
        "negative_stopping_margin_requested_rate": float(
            governor["negative_stopping_margin_requested_rate"]
        ),
        "negative_stopping_margin_executed_rate": float(
            governor["negative_stopping_margin_executed_rate"]
        ),
        "contact_actual_speed_mps": float(governor["contact"]["mean_actual_speed_mps"]),
        "contact_requested_speed_mps": float(
            governor["contact"]["mean_requested_speed_mps"]
        ),
        "contact_executed_speed_mps": float(
            governor["contact"]["mean_executed_speed_mps"]
        ),
        "contact_clearance_m": float(governor["contact"]["mean_clearance_m"]),
        "contact_stopping_margin_executed_m": float(
            governor["contact"]["mean_stopping_margin_executed_m"]
        ),
        "contact_step": float(governor["contact"]["mean_step"]),
        "outcome_steps": governor["outcome_steps"],
        "parameters": {
            key: condition[key]
            for key in condition
            if key.startswith("speed_governor_")
        },
    }


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/navrl_v2_ep24000_speed_governor_screen")
    rows = [load_cell(root, tag) for tag in TAGS]
    baseline = rows[0]
    for row in rows:
        row["capture_delta"] = row["capture_rate"] - baseline["capture_rate"]
        row["crash_delta"] = row["crash_rate"] - baseline["crash_rate"]
        row["timeout_delta"] = row["timeout_rate"] - baseline["timeout_rate"]
        _, row["capture_delta_ci95"] = diff_ci(
            row["captured"], row["episodes"], baseline["captured"], baseline["episodes"]
        )
        _, row["crash_delta_ci95"] = diff_ci(
            row["crashes"], row["episodes"], baseline["crashes"], baseline["episodes"]
        )
        row["screen_pass"] = bool(
            row["crash_delta"] <= -0.030
            and row["capture_delta"] >= -0.010
            and row["timeout_rate"] <= 0.050
        )

    adaptive = [row for row in rows if row["tag"] in ("clearance", "ttc") and row["screen_pass"]]
    selected = None
    if adaptive:
        adaptive.sort(key=lambda row: row["capture_rate"], reverse=True)
        selected = adaptive[0]
        close = [row for row in adaptive if selected["capture_rate"] - row["capture_rate"] <= 0.005]
        selected = min(close, key=lambda row: row["crash_rate"])

    summary = {
        "schema_version": 1,
        "experiment": "ep24000_speed_governor_screen",
        "source_checkpoint_sha256": SOURCE_SHA,
        "preregistered_gate": {
            "crash_delta_max": -0.030,
            "capture_delta_min": -0.010,
            "timeout_rate_max": 0.050,
            "adaptive_only_for_training": True,
        },
        "adaptive_go": selected is not None,
        "selected_tag": selected["tag"] if selected else None,
        "rows": rows,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# NavRL v2 ep24000 speed-governor screen",
        "",
        "| condition | n | capture | crash | timeout | bar contact | intervention | executed m/s | Δcapture | Δcrash | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {tag} | {episodes:,} | {capture_rate:.2%} | {crash_rate:.2%} | {timeout_rate:.2%} | "
            "{bar_contact_rate:.2%} | {intervention_rate:.2%} | {mean_executed_speed_mps:.3f} | "
            "{capture_delta:+.2%} | {crash_delta:+.2%} | {gate} |".format(
                **row, gate="PASS" if row["screen_pass"] else "FAIL"
            )
        )
    lines += [
        "",
        "| condition | near-stop | requested m/s | scale | unsafe before | unsafe after | contact actual/executed m/s | contact clearance | contact step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {tag} | {near_stop_rate:.2%} | {mean_requested_speed_mps:.3f} | {mean_scale:.3f} | "
            "{negative_stopping_margin_requested_rate:.2%} | {negative_stopping_margin_executed_rate:.2%} | "
            "{contact_actual_speed_mps:.3f}/{contact_executed_speed_mps:.3f} | "
            "{contact_clearance_m:.3f} m | {contact_step:.1f} |".format(**row)
        )
    lines += [
        "",
        "| condition | capture delta 95% CI | crash delta 95% CI | capture/crash/timeout mean step |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        capture_ci = row["capture_delta_ci95"]
        crash_ci = row["crash_delta_ci95"]
        steps = row["outcome_steps"]
        lines.append(
            "| {tag} | [{cl:+.2%}, {ch:+.2%}] | [{rl:+.2%}, {rh:+.2%}] | {cs}/{rs}/{ts} |".format(
                tag=row["tag"],
                cl=capture_ci[0], ch=capture_ci[1],
                rl=crash_ci[0], rh=crash_ci[1],
                cs="—" if steps["capture"]["mean"] is None else f'{steps["capture"]["mean"]:.1f}',
                rs="—" if steps["crash"]["mean"] is None else f'{steps["crash"]["mean"]:.1f}',
                ts="—" if steps["timeout"]["mean"] is None else f'{steps["timeout"]["mean"]:.1f}',
            )
        )
    lines += [
        "",
        "Adaptive GO: **%s**; selected: **%s**."
        % ("YES" if selected else "NO", selected["tag"] if selected else "none"),
        "",
        "Gate fixed before evaluation: crash delta <= -3.0 pp, capture delta >= -1.0 pp, timeout <= 5%; only adaptive arms authorize training.",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"adaptive_go": selected is not None, "selected": summary["selected_tag"]}))


if __name__ == "__main__":
    main()
