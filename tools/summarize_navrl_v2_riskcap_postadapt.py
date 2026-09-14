"""Validate R2b post-adaptation cells, select the winner, and decide generalization."""

import argparse
import hashlib
import json
import math
from pathlib import Path


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


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--trained-sha", required=True)
    parser.add_argument("--select-only", action="store_true")
    return parser.parse_args()


def diff_ci(a, b, field):
    count = {"capture_rate": "captured", "crash_rate": "crashes", "timeout_rate": "timeouts"}[field]
    pa, pb = a[field], b[field]
    se = math.sqrt(pa * (1 - pa) / a["episodes"] + pb * (1 - pb) / b["episodes"])
    return pa - pb, [pa - pb - 1.96 * se, pa - pb + 1.96 * se]


def delta_with_ci(result):
    delta, interval = result
    return f"{delta * 100:+.2f} pp (95% CI {interval[0] * 100:+.2f}..{interval[1] * 100:+.2f})"


def load(root, tag, *, sha, mode, seed, speed=None):
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
    n = int(payload.get("actual_episodes", -1))
    speed_ok = (
        condition.get("target_speed_mode") == "uniform"
        if speed is None
        else condition.get("target_speed_mode") == "fixed"
        and math.isclose(float(condition.get("target_speed_mps", -1)), speed)
    )
    checks = {
        "schema": payload.get("schema_version") == 1,
        "episodes": int(payload.get("requested_episodes", -1)) == 2049 and n >= 2049,
        "sha": payload.get("checkpoint_sha256") == sha
        and payload.get("evaluated_checkpoint_snapshot_sha256") == sha,
        "condition": int(condition.get("bars", -1)) == 205
        and int(condition.get("seed", -1)) == seed
        and condition.get("action_selection") == "deterministic"
        and condition.get("reflection_mode") == "original"
        and speed_ok,
        "runtime": contract.get("runtime_profile") == "main"
        and contract.get("runtime_sim") == "base_sim"
        and int(contract.get("runtime_num_envs", -1)) == 128
        and contract.get("obstacle_selector") == "cluster_sector"
        and math.isclose(float(contract.get("obstacle_effective_fov_deg", -1)), 240.0),
        "mode": condition.get("speed_governor_mode") == mode
        and contract.get("speed_governor_mode") == mode
        and governor.get("mode") == mode,
        "sensor_contract": governor.get("sensor_only") is True
        and governor.get("direction_preserved") is True
        and governor.get("feedback_executed_previous_action") is (mode != "off")
        and governor.get("target_exclusion_source") == "camera_lidar_association"
        and condition.get("speed_governor_target_exclusion") == "camera_lidar_association"
        and contract.get("speed_governor_target_exclusion") == "camera_lidar_association",
        "governor_parameters": all(
            math.isclose(float(condition.get(key, float("nan"))), expected)
            and math.isclose(float(contract.get(key, float("nan"))), expected)
            for key, expected in GOVERNOR_PARAMETERS.items()
        ),
        "receipt": receipt.get("schema_version") == 1
        and int(receipt.get("bars", -1)) == 205
        and int(receipt.get("seed", -1)) == seed
        and int(receipt.get("requested_episodes", -1)) == 2049
        and int(receipt.get("actual_episodes", -1)) == n
        and receipt.get("action_selection") == "deterministic"
        and receipt.get("reflection_mode") == "original"
        and receipt.get("speed_governor_mode") == mode
        and receipt.get("speed_governor_target_exclusion") == "camera_lidar_association"
        and receipt.get("source_checkpoint_sha256") == sha
        and receipt.get("evaluated_checkpoint_snapshot_sha256") == sha
        and receipt.get("evaluation_nonce") == condition.get("evaluation_nonce"),
        "artifact_digests": receipt.get("result_sha256") == sha256(path)
        and receipt.get("log_sha256") == sha256(log_path)
        and sha256(snapshot_path) == sha,
        "counts": sum(int(outcome.get(key, -n - 1)) for key in ("captured", "crash", "timeout")) == n,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"invalid {tag}: {', '.join(failed)}")
    return {
        "tag": tag, "path": str(path), "checkpoint_sha256": sha, "mode": mode,
        "seed": seed, "target_speed_mps": speed, "episodes": n,
        "captured": int(outcome["captured"]), "crashes": int(outcome["crash"]),
        "timeouts": int(outcome["timeout"]), "capture_rate": float(outcome["capture_rate"]),
        "crash_rate": float(outcome["crash_rate"]), "timeout_rate": float(outcome["timeout_rate"]),
        "bar_contact_rate": int((payload.get("crash_causes") or {}).get("bar_contact", 0)) / n,
        "intervention_rate": float(governor["intervention_rate"]),
        "near_stop_rate": float(governor["near_stop_rate"]),
        "mean_requested_speed_mps": float(governor["mean_requested_speed_mps"]),
        "mean_executed_speed_mps": float(governor["mean_executed_speed_mps"]),
        "contact_actual_speed_mps": float(governor["contact"]["mean_actual_speed_mps"]),
    }


def main():
    cfg = args()
    root = cfg.root
    off = load(root, "uniform_off", sha=SOURCE_SHA, mode="off", seed=45)
    source = load(root, "uniform_source_riskcap", sha=SOURCE_SHA, mode="riskcap", seed=45)
    trained = load(root, "uniform_trained_riskcap", sha=cfg.trained_sha, mode="riskcap", seed=45)
    mechanism = {metric: diff_ci(source, off, metric) for metric in ("capture_rate", "crash_rate", "timeout_rate")}
    mechanism_pass = bool(
        mechanism["crash_rate"][0] <= -0.030
        and mechanism["capture_rate"][0] >= -0.010
        and source["timeout_rate"] <= 0.050
        and source["near_stop_rate"] <= 0.050
    )
    adaptation = {metric: diff_ci(trained, source, metric) for metric in ("capture_rate", "crash_rate", "timeout_rate")}
    noninferior = bool(
        adaptation["capture_rate"][0] >= -0.010
        and adaptation["crash_rate"][0] <= 0.010
        and adaptation["timeout_rate"][0] <= 0.010
    )
    intervention_delta = trained["intervention_rate"] - source["intervention_rate"]
    useful = bool(adaptation["capture_rate"][0] >= 0.010 or intervention_delta <= -0.050)
    adaptation_pass = noninferior and useful
    winner_kind = "trained" if adaptation_pass else "source"
    selection = {
        "schema_version": 1, "source_checkpoint_sha256": SOURCE_SHA,
        "trained_checkpoint_sha256": cfg.trained_sha,
        "mechanism_replication_pass": mechanism_pass,
        "adaptation_noninferior": noninferior, "adaptation_useful": useful,
        "adaptation_pass": adaptation_pass, "intervention_delta": intervention_delta,
        "winner_kind": winner_kind,
        "winner_checkpoint_sha256": cfg.trained_sha if winner_kind == "trained" else SOURCE_SHA,
        "uniform_rows": [off, source, trained],
        "mechanism_deltas": mechanism, "adaptation_deltas": adaptation,
    }
    (root / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    if cfg.select_only:
        print(json.dumps({"winner_kind": winner_kind, "adaptation_pass": adaptation_pass}))
        return

    fixed = []
    winner_sha = selection["winner_checkpoint_sha256"]
    for speed, tag in ((0.3, "0p3"), (0.9, "0p9"), (1.5, "1p5")):
        base = load(root, f"fixed_{tag}_off", sha=SOURCE_SHA, mode="off", seed=46, speed=speed)
        candidate = load(root, f"fixed_{tag}_winner", sha=winner_sha, mode="riskcap", seed=46, speed=speed)
        deltas = {metric: diff_ci(candidate, base, metric) for metric in ("capture_rate", "crash_rate", "timeout_rate")}
        direction_pass = deltas["capture_rate"][0] >= 0.0 and deltas["crash_rate"][0] < 0.0
        fixed.append({"target_speed_mps": speed, "off": base, "winner": candidate, "deltas": deltas, "direction_pass": direction_pass})
    generalization_pass = mechanism_pass and all(row["direction_pass"] for row in fixed)
    summary = {**selection, "fixed_speed_rows": fixed, "generalization_pass": generalization_pass}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# NavRL v2 riskcap post-adaptation validation",
        "",
        "## Uniform seed45",
        "",
        "| policy | n | capture | crash | timeout | bar contact | intervention | executed m/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (off, source, trained):
        lines.append("| {tag} | {episodes:,} | {capture_rate:.2%} | {crash_rate:.2%} | {timeout_rate:.2%} | {bar_contact_rate:.2%} | {intervention_rate:.2%} | {mean_executed_speed_mps:.3f} |".format(**row))
    lines += [
        "", f"Mechanism replication: **{'PASS' if mechanism_pass else 'FAIL'}**.",
        "",
        f"- capture: {delta_with_ci(mechanism['capture_rate'])}",
        f"- crash: {delta_with_ci(mechanism['crash_rate'])}",
        f"- timeout: {delta_with_ci(mechanism['timeout_rate'])}",
        "",
        f"Adaptation: non-inferior **{noninferior}**, useful **{useful}**, decision **{'PASS' if adaptation_pass else 'NO ADDITIONAL VALUE'}**.",
        "",
        f"- capture: {delta_with_ci(adaptation['capture_rate'])}",
        f"- crash: {delta_with_ci(adaptation['crash_rate'])}",
        f"- timeout: {delta_with_ci(adaptation['timeout_rate'])}",
        f"- intervention: {intervention_delta * 100:+.2f} pp",
        "",
        f"Winner: **{winner_kind} riskcap**.", "", "## Fixed-speed seed46", "",
        "| speed | off capture/crash | winner capture/crash | Δcapture (95% CI) | Δcrash (95% CI) | direction |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in fixed:
        lines.append(
            "| {speed:.1f} | {oc:.2%}/{orate:.2%} | {wc:.2%}/{wr:.2%} | {dc} | {dr} | {gate} |".format(
                speed=row["target_speed_mps"], oc=row["off"]["capture_rate"], orate=row["off"]["crash_rate"],
                wc=row["winner"]["capture_rate"], wr=row["winner"]["crash_rate"],
                dc=delta_with_ci(row["deltas"]["capture_rate"]),
                dr=delta_with_ci(row["deltas"]["crash_rate"]),
                gate="PASS" if row["direction_pass"] else "FAIL",
            )
        )
    lines += ["", f"Final generalization: **{'PASS' if generalization_pass else 'FAIL'}**."]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"winner_kind": winner_kind, "generalization_pass": generalization_pass}))


if __name__ == "__main__":
    main()
