"""Validate and summarize the preregistered ep25000 seed-49 stopcap screen.

Prereg: docs/prereg_2026-09-02_speed_governor_stopcap_screen.md. Verdict rules were frozen
before results. There is deliberately no singular "verdict" key: verdicts are reported per
question (verdict_m1 / verdict_q1 / verdict_q2 / verdict_q3).
"""

import hashlib
import json
import math
from pathlib import Path
import sys


TAGS = ("off", "fixed2p0", "riskcap", "stopcap", "ttc")
EXPECTED_MODE = {
    "off": "off",
    "fixed2p0": "fixed",
    "riskcap": "riskcap",
    "stopcap": "stopcap",
    "ttc": "ttc",
}
SOURCE_SHA = "f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"
SEED = 49
COMMON_PARAMETERS = {
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
        and int(condition.get("seed", -1)) == SEED
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
            for key, expected in COMMON_PARAMETERS.items()
        ),
        # The canonical evaluator has emitted the source/provenance-complete receipt schema 2
        # since the August provenance hardening.  The result payload remains schema 1.
        "receipt": receipt.get("schema_version") == 2
        and int(receipt.get("bars", -1)) == 205
        and int(receipt.get("seed", -1)) == SEED
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
        "contact_requested_speed_mps": float(governor["contact"]["mean_requested_speed_mps"]),
        "contact_executed_speed_mps": float(governor["contact"]["mean_executed_speed_mps"]),
        "contact_clearance_m": float(governor["contact"]["mean_clearance_m"]),
        "contact_stopping_margin_executed_m": float(
            governor["contact"]["mean_stopping_margin_executed_m"]
        ),
        "contact_step": float(governor["contact"]["mean_step"]),
        "outcome_steps": governor["outcome_steps"],
        "parameters": {
            key: condition[key] for key in condition if key.startswith("speed_governor_")
        },
    }


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/navrl_v2_ep25000_stopcap_seed49_screen")
    rows = [load_cell(root, tag) for tag in TAGS]
    cell = {row["tag"]: row for row in rows}
    baseline = cell["off"]
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

    # --- M1: machinery integrity (fail-closed). stopcap guarantees margin >= 0 by law. ---
    m1_rate = cell["stopcap"]["negative_stopping_margin_executed_rate"]
    verdict_m1 = "PASS" if m1_rate <= 0.01 else "IMPLEMENTATION_VOID"

    # --- Q1: does riskcap's clearance-release mechanism beat a constant 2.0 cap? ---
    q1_delta, q1_ci = diff_ci(
        cell["riskcap"]["captured"], cell["riskcap"]["episodes"],
        cell["fixed2p0"]["captured"], cell["fixed2p0"]["episodes"],
    )
    if q1_ci[0] > 0.0:
        verdict_q1 = "MECHANISM_SUPPORTED"
    elif q1_ci[1] < 0.0:
        verdict_q1 = "MECHANISM_REVERSED"
    else:
        verdict_q1 = "MECHANISM_UNSUPPORTED"

    # --- Q2: stopcap adoption gates (all frozen in the prereg). ---
    q2_crash_delta, q2_crash_ci = diff_ci(
        cell["stopcap"]["crashes"], cell["stopcap"]["episodes"],
        cell["riskcap"]["crashes"], cell["riskcap"]["episodes"],
    )
    q2_capture_delta, _ = diff_ci(
        cell["stopcap"]["captured"], cell["stopcap"]["episodes"],
        cell["riskcap"]["captured"], cell["riskcap"]["episodes"],
    )
    q2_gates = {
        "crash_improvement": bool(q2_crash_delta <= -0.03 and q2_crash_ci[1] < 0.0),
        "liveness": bool(cell["stopcap"]["timeout_rate"] <= 0.05),
        "capture_cost": bool(q2_capture_delta >= -0.02),
    }
    if verdict_m1 != "PASS":
        verdict_q2 = "NOT_JUDGED_M1_VOID"
    elif all(q2_gates.values()):
        verdict_q2 = "GO"
    elif q2_gates["crash_improvement"]:
        verdict_q2 = "SAFETY_ONLY"
    else:
        verdict_q2 = "NO_GO"

    # --- Q3: filter dependence of the riskcap-adapted ep25000 policy. ---
    q3_delta = cell["off"]["crash_rate"] - cell["riskcap"]["crash_rate"]
    if q3_delta >= 0.05:
        verdict_q3 = "FILTER_DEPENDENT"
    elif q3_delta <= 0.02:
        verdict_q3 = "FILTER_INDEPENDENT"
    else:
        verdict_q3 = "INCONCLUSIVE"

    summary = {
        "schema_version": 1,
        "experiment": "ep25000_seed49_stopcap_screen",
        "prereg": "docs/prereg_2026-09-02_speed_governor_stopcap_screen.md",
        "source_checkpoint_sha256": SOURCE_SHA,
        "heldout_seed": SEED,
        "preregistered_gate": {
            "m1_negative_stopping_margin_executed_rate_max": 0.01,
            "q2_crash_delta_vs_riskcap_max": -0.03,
            "q2_timeout_rate_max": 0.05,
            "q2_capture_delta_vs_riskcap_min": -0.02,
            "q3_filter_dependent_crash_delta_min": 0.05,
            "q3_filter_independent_crash_delta_max": 0.02,
        },
        "verdict_m1": verdict_m1,
        "verdict_q1": verdict_q1,
        "verdict_q2": verdict_q2,
        "verdict_q3": verdict_q3,
        "q1_capture_delta_riskcap_vs_fixed2p0": q1_delta,
        "q1_capture_delta_ci95": list(q1_ci),
        "q2_crash_delta_stopcap_vs_riskcap": q2_crash_delta,
        "q2_crash_delta_ci95": list(q2_crash_ci),
        "q2_capture_delta_stopcap_vs_riskcap": q2_capture_delta,
        "q2_gates": q2_gates,
        "q3_crash_delta_off_vs_riskcap": q3_delta,
        "rows": rows,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# NavRL v2 ep25000 seed49 stopcap screen",
        "",
        "Prereg: `docs/prereg_2026-09-02_speed_governor_stopcap_screen.md`",
        "",
        "| condition | n | capture | crash | timeout | bar contact | intervention | executed m/s | Δcapture | Δcrash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {tag} | {episodes:,} | {capture_rate:.2%} | {crash_rate:.2%} | {timeout_rate:.2%} | "
            "{bar_contact_rate:.2%} | {intervention_rate:.2%} | {mean_executed_speed_mps:.3f} | "
            "{capture_delta:+.2%} | {crash_delta:+.2%} |".format(**row)
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
        "## Preregistered verdicts",
        "",
        f"- M1 machinery: **{verdict_m1}** (stopcap unsafe-after {m1_rate:.2%}, limit 1.00%)",
        f"- Q1 release mechanism: **{verdict_q1}** "
        f"(riskcap−fixed2p0 capture {q1_delta:+.2%}, CI95 [{q1_ci[0]:+.2%}, {q1_ci[1]:+.2%}])",
        f"- Q2 stopcap adoption: **{verdict_q2}** "
        f"(crash {q2_crash_delta:+.2%} CI95 [{q2_crash_ci[0]:+.2%}, {q2_crash_ci[1]:+.2%}], "
        f"timeout {cell['stopcap']['timeout_rate']:.2%}, capture {q2_capture_delta:+.2%}; "
        f"gates {q2_gates})",
        f"- Q3 filter dependence: **{verdict_q3}** (off−riskcap crash {q3_delta:+.2%})",
        "",
        "ttc arm is reference-only by prereg (no gate).",
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
