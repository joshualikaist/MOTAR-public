"""Validate and summarize the preregistered seed-44 minimum-intervention riskcap screen."""

import hashlib
import json
import math
from pathlib import Path
import sys


SOURCE_SHA = "82f7978b42d9d9e95adcc638a40ae85fb3736fd897ae39cb8aa8333be39cf23f"
GOVERNOR_PARAMETERS = {
    "speed_governor_fixed_mps": 2.0,
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
    return pa - pb, [pa - pb - 1.96 * se, pa - pb + 1.96 * se]


def load_cell(root, tag, expected_mode):
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
    checks = {
        "schema": payload.get("schema_version") == 1,
        "episodes": int(payload.get("requested_episodes", -1)) == 2049 and actual >= 2049,
        "checkpoint": payload.get("checkpoint_sha256") == SOURCE_SHA
        and payload.get("evaluated_checkpoint_snapshot_sha256") == SOURCE_SHA,
        "condition": int(condition.get("bars", -1)) == 205
        and int(condition.get("seed", -1)) == 44
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
        "sensor": governor.get("sensor_only") is True
        and governor.get("direction_preserved") is True
        and governor.get("target_exclusion_source") == "camera_lidar_association"
        and condition.get("speed_governor_target_exclusion") == "camera_lidar_association"
        and contract.get("speed_governor_target_exclusion") == "camera_lidar_association",
        "parameters": all(
            math.isclose(float(condition.get(key, float("nan"))), expected)
            and math.isclose(float(contract.get(key, float("nan"))), expected)
            for key, expected in GOVERNOR_PARAMETERS.items()
        ),
        "receipt": receipt.get("schema_version") == 1
        and int(receipt.get("bars", -1)) == 205
        and int(receipt.get("seed", -1)) == 44
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
        "counts": sum(int(outcome.get(key, -actual - 1)) for key in ("captured", "crash", "timeout")) == actual,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"invalid {tag} result {path}: {', '.join(failed)}")
    crashes = int(outcome["crash"])
    return {
        "tag": tag,
        "mode": expected_mode,
        "path": str(path),
        "episodes": actual,
        "captured": int(outcome["captured"]),
        "crashes": crashes,
        "timeouts": int(outcome["timeout"]),
        "capture_rate": float(outcome["capture_rate"]),
        "crash_rate": float(outcome["crash_rate"]),
        "timeout_rate": float(outcome["timeout_rate"]),
        "bar_contact_rate": int((payload.get("crash_causes") or {}).get("bar_contact", 0)) / actual,
        "intervention_rate": float(governor["intervention_rate"]),
        "near_stop_rate": float(governor["near_stop_rate"]),
        "mean_requested_speed_mps": float(governor["mean_requested_speed_mps"]),
        "mean_executed_speed_mps": float(governor["mean_executed_speed_mps"]),
        "mean_scale": float(governor["mean_scale"]),
        "unsafe_before_rate": float(governor["negative_stopping_margin_requested_rate"]),
        "unsafe_after_rate": float(governor["negative_stopping_margin_executed_rate"]),
        "contact_actual_speed_mps": float(governor["contact"]["mean_actual_speed_mps"]),
        "contact_executed_speed_mps": float(governor["contact"]["mean_executed_speed_mps"]),
        "contact_step": float(governor["contact"]["mean_step"]),
        "outcome_steps": governor["outcome_steps"],
        "parameters": {
            key: condition[key] for key in condition if key.startswith("speed_governor_")
        },
    }


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/navrl_v2_ep24000_riskcap_seed44_screen")
    baseline = load_cell(root, "off", "off")
    candidate = load_cell(root, "riskcap", "riskcap")
    candidate["capture_delta"], candidate["capture_delta_ci95"] = diff_ci(
        candidate["captured"], candidate["episodes"], baseline["captured"], baseline["episodes"]
    )
    candidate["crash_delta"], candidate["crash_delta_ci95"] = diff_ci(
        candidate["crashes"], candidate["episodes"], baseline["crashes"], baseline["episodes"]
    )
    candidate["timeout_delta"] = candidate["timeout_rate"] - baseline["timeout_rate"]
    gate = {
        "crash_delta_max": -0.030,
        "capture_delta_min": -0.010,
        "timeout_rate_max": 0.050,
        "near_stop_rate_max": 0.050,
    }
    passed = bool(
        candidate["crash_delta"] <= gate["crash_delta_max"]
        and candidate["capture_delta"] >= gate["capture_delta_min"]
        and candidate["timeout_rate"] <= gate["timeout_rate_max"]
        and candidate["near_stop_rate"] <= gate["near_stop_rate_max"]
    )
    candidate["screen_pass"] = passed
    baseline.update(
        capture_delta=0.0, crash_delta=0.0,
        capture_delta_ci95=[0.0, 0.0], crash_delta_ci95=[0.0, 0.0],
        timeout_delta=0.0, screen_pass=False,
    )
    summary = {
        "schema_version": 1,
        "experiment": "ep24000_seed44_riskcap_screen",
        "heldout_seed": 44,
        "source_checkpoint_sha256": SOURCE_SHA,
        "preregistered_gate": gate,
        "adaptive_go": passed,
        "selected_tag": "riskcap" if passed else None,
        "rows": [baseline, candidate],
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# NavRL v2 seed44 minimum-intervention riskcap screen",
        "",
        "| condition | n | capture | crash | timeout | bar contact | intervention | near-stop | executed m/s | Δcapture | Δcrash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (baseline, candidate):
        lines.append(
            "| {tag} | {episodes:,} | {capture_rate:.2%} | {crash_rate:.2%} | {timeout_rate:.2%} | "
            "{bar_contact_rate:.2%} | {intervention_rate:.2%} | {near_stop_rate:.2%} | "
            "{mean_executed_speed_mps:.3f} | {capture_delta:+.2%} | {crash_delta:+.2%} |".format(**row)
        )
    lines += [
        "",
        "Riskcap GO: **%s**." % ("YES" if passed else "NO"),
        "",
        "Gate fixed before seed44 evaluation: crash delta <= -3pp, capture delta >= -1pp, timeout <=5%, near-stop <=5%.",
        "Candidate 95% normal CIs: capture delta [{:+.2%}, {:+.2%}], crash delta [{:+.2%}, {:+.2%}].".format(
            *candidate["capture_delta_ci95"], *candidate["crash_delta_ci95"]
        ),
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"adaptive_go": passed, "selected": summary["selected_tag"]}))


if __name__ == "__main__":
    main()
