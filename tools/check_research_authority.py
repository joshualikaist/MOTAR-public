#!/usr/bin/env python3
"""Verify the frozen MOTAR execution authority against immutable result summaries.

This tool does not launch training or evaluation.  It makes the current NO-GO/allowed-next
boundary machine-readable so a stale document or a convenient rerun command cannot silently
supersede the preregistered Track A/B decisions.
"""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = ROOT / "docs" / "research_authority_2026-08-26.json"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_authority(receipt_path=DEFAULT_RECEIPT):
    receipt_path = Path(receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "motar_research_execution_authority_v1":
        raise RuntimeError("research authority schema drift")

    evidence = {}
    for record in receipt.get("evidence", []):
        relative = record["path"]
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError("missing frozen evidence: %s" % relative)
        actual_sha = _sha256(path)
        if actual_sha != record["sha256"]:
            raise RuntimeError(
                "frozen evidence SHA drift: %s expected=%s actual=%s"
                % (relative, record["sha256"], actual_sha)
            )
        evidence[relative] = json.loads(path.read_text(encoding="utf-8"))

    stage1 = evidence["results/navrl_ref5in_detection_range_stage1_s457/summary.json"]
    if stage1.get("verdict") != receipt["track_a"]["result"]:
        raise RuntimeError("Track A verdict drift")
    if stage1.get("stage2_authorised") is not False:
        raise RuntimeError("Track A Stage 2 unexpectedly authorized")

    recovery = evidence[
        "results/navrl_physical_target_recovery_v2_gate_lower1p25_seed827/summary.json"
    ]
    verdict = recovery.get("verdict", {})
    if verdict.get("execution_integrity") != receipt["track_b"]["execution_integrity"]:
        raise RuntimeError("Track B execution-integrity drift")
    if verdict.get("route_mechanism") != receipt["track_b"]["route_mechanism"]:
        raise RuntimeError("Track B route-mechanism drift")
    if verdict.get("long_training_authorized") is not False:
        raise RuntimeError("Track B long training unexpectedly authorized")
    if recovery.get("claim_boundary", {}).get("ppo_policy_loaded") is not False:
        raise RuntimeError("Track B diagnostic unexpectedly claims a PPO policy")
    if recovery.get("claim_boundary", {}).get("hardware_validation") is not False:
        raise RuntimeError("Track B diagnostic unexpectedly claims hardware validation")

    followup = evidence[
        "results/navrl_physical_target_recovery_v2_no_connector_forensics_seed827/summary.json"
    ]
    decision = followup.get("decision_rule", {})
    if decision.get("label") != receipt["track_b"]["no_anchor_followup"]:
        raise RuntimeError("Track B no-anchor verdict drift")
    if decision.get("authorizes_retune_or_ppo") is not False:
        raise RuntimeError("Track B no-anchor follow-up unexpectedly authorizes work")
    if followup.get("passes_32_cell_mechanism") is not False:
        raise RuntimeError("Track B no-anchor follow-up unexpectedly supersedes 32-cell FAIL")

    corrected = receipt["corrected_environment_v2_2026_08_27"]
    lower_v3 = corrected["braking_aware_route_v3_lower1p25"]
    authority_path = ROOT / lower_v3["execution_authority"]
    if not authority_path.is_file():
        raise RuntimeError("missing matched-spawn GPU execution authority")
    if _sha256(authority_path) != lower_v3["execution_authority_sha256"]:
        raise RuntimeError("matched-spawn GPU execution-authority SHA drift")
    if lower_v3.get("gpu_authority") is not False or lower_v3.get(
        "gpu_authority_consumed"
    ) is not True:
        raise RuntimeError("matched-spawn pilot GPU authority was not closed after use")
    if lower_v3.get("authorizes_ppo") is not False:
        raise RuntimeError("matched-spawn execution authority unexpectedly authorizes PPO")
    pilot_path = ROOT / lower_v3["pilot_result"]
    if not pilot_path.is_file() or _sha256(pilot_path) != lower_v3["pilot_result_sha256"]:
        raise RuntimeError("matched-spawn pilot result missing or SHA drift")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    pilot_verdict = pilot.get("verdict", {})
    if pilot_verdict.get("execution_integrity") != "PASS_8_CELL_INTEGRITY":
        raise RuntimeError("matched-spawn pilot integrity drift")
    if pilot_verdict.get("gate") != "FAIL_BLOCKS_CONFIRMATORY":
        raise RuntimeError("matched-spawn pilot gate drift")
    if lower_v3.get("confirmatory_authorized") is not False:
        raise RuntimeError("matched-spawn FAIL unexpectedly authorizes confirmatory")
    corrected_gate = corrected["route_physical_gate_r2"]
    corrected_result = evidence[
        "results/navrl_corrected_nonoverlap_route_gate_r2_seed829/summary.json"
    ]
    corrected_verdict = corrected_result.get("verdicts", {})
    for field in ("execution_integrity", "route_mechanism", "full_1p5_contract"):
        if corrected_verdict.get(field) != corrected_gate[field]:
            raise RuntimeError("corrected route gate %s drift" % field)
    corrected_inputs = corrected_verdict.get("route_mechanism_inputs", {})
    for field in (
        "plan_success_fraction_70",
        "fallback_interval_fraction_70",
        "goal_completions_per_env_70_speed_0p6",
    ):
        if corrected_inputs.get(field) != corrected_gate[field]:
            raise RuntimeError("corrected route gate %s drift" % field)
    if corrected_verdict.get("highest_passing_speed_mps_by_density") != (
        corrected_gate["highest_passing_speed_mps_by_density"]
    ):
        raise RuntimeError("corrected route gate density envelope drift")
    if corrected_verdict.get("long_training_authority") is not False:
        raise RuntimeError("corrected route gate unexpectedly authorizes long training")
    if corrected_gate.get("authorizes_ppo") is not False:
        raise RuntimeError("corrected route gate receipt unexpectedly authorizes PPO")
    if corrected_gate.get("fresh_ppo_epochs_run") != 0:
        raise RuntimeError("corrected route gate receipt unexpectedly claims fresh PPO epochs")

    # This narrow route-off smoke does not reverse the frozen routed-mechanism failure.
    smoke = corrected["route_off_learning_viability_smoke"]
    smoke_prereg = ROOT / smoke["preregistration"]
    if not smoke_prereg.is_file() or _sha256(smoke_prereg) != smoke["preregistration_sha256"]:
        raise RuntimeError("route-off learning-smoke preregistration missing or SHA drift")
    if smoke.get("status") != "PASS_LEARNING_VIABILITY":
        raise RuntimeError("route-off learning-smoke status drift")
    if smoke.get("gpu_authority") is not False or smoke.get(
        "gpu_authority_consumed"
    ) is not True or smoke.get("fresh_only") is not True:
        raise RuntimeError("route-off learning-smoke one-shot authority drift")
    if smoke.get("authorizes_routed_ppo") is not False or smoke.get(
        "authorizes_long_training"
    ) is not False:
        raise RuntimeError("route-off smoke unexpectedly authorizes routed/long PPO")
    if smoke.get("fixed_bars") != 70 or smoke.get("max_epochs") != 500:
        raise RuntimeError("route-off learning-smoke execution tuple drift")
    smoke_result_path = ROOT / smoke["result"]
    if not smoke_result_path.is_file() or _sha256(smoke_result_path) != smoke["result_sha256"]:
        raise RuntimeError("route-off learning-smoke result missing or SHA drift")
    smoke_result = json.loads(smoke_result_path.read_text(encoding="utf-8"))
    if smoke_result.get("verdict") != "PASS_LEARNING_VIABILITY" or not all(
        smoke_result.get("checks", {}).values()
    ):
        raise RuntimeError("route-off learning-smoke result no longer passes")
    if smoke_result.get("authority", {}).get(
        "route_off_curriculum_preregistration"
    ) is not True or smoke_result.get("authority", {}).get("routed_ppo") is not False:
        raise RuntimeError("route-off learning-smoke authority boundary drift")

    curriculum = corrected["route_off_curriculum"]
    curriculum_prereg = ROOT / curriculum["preregistration"]
    if not curriculum_prereg.is_file() or _sha256(curriculum_prereg) != curriculum[
        "preregistration_sha256"
    ]:
        raise RuntimeError("route-off curriculum preregistration missing or SHA drift")
    if curriculum.get("status") != "OPERATOR_STOPPED_INCOMPLETE" or curriculum.get(
        "gpu_authority"
    ) is not False or curriculum.get("gpu_authority_consumed") is not True or curriculum.get(
        "fresh_only"
    ) is not True or curriculum.get("resume_forbidden") is not True:
        raise RuntimeError("route-off curriculum execution authority drift")
    if curriculum.get("run") != (
        "ppo_260901_1431_navrl_corrected-nonoverlap-physical-off-curriculum-s911"
    ):
        raise RuntimeError("route-off curriculum live-run identity drift")
    if curriculum.get("authorizes_routed_ppo") is not False or curriculum.get(
        "authorizes_hardware_claim"
    ) is not False:
        raise RuntimeError("route-off curriculum unexpectedly authorizes broader claims")

    heldout = corrected["route_off_heldout_eval"]
    heldout_prereg = ROOT / heldout["preregistration"]
    if not heldout_prereg.is_file() or _sha256(heldout_prereg) != heldout[
        "preregistration_sha256"
    ]:
        raise RuntimeError("route-off held-out eval preregistration missing or SHA drift")
    if heldout.get("eval_seed") != 313 or heldout.get("training_seed_forbidden") != 911:
        raise RuntimeError("route-off held-out eval seed contract drift")
    if heldout.get("densities") != [70, 85, 100, 115, 130, 145]:
        raise RuntimeError("route-off held-out eval density contract drift")
    if heldout.get("ood_205_included") is not False or heldout.get(
        "gen_ppo_forbidden"
    ) is not True:
        raise RuntimeError("route-off held-out eval OOD/gen_ppo contract drift")
    if heldout.get("checkpoint_sha256") != (
        "541b36bdcabacf8bb14c6fbb0ad07054dd9735ad24777a3222655ba8ca9c8132"
    ):
        raise RuntimeError("route-off held-out eval checkpoint digest drift")
    if heldout.get("authorizes_routed_ppo") is not False or heldout.get(
        "authorizes_second_curriculum"
    ) is not False or heldout.get("authorizes_resume") is not False:
        raise RuntimeError("route-off held-out eval unexpectedly authorizes broader work")
    if heldout.get("status") != "COMPLETE_VALID_WITH_METADATA_ERRATUM" or heldout.get(
        "gpu_authority"
    ) is not False or heldout.get("gpu_authority_consumed") is not True:
        raise RuntimeError("route-off held-out completion authority drift")
    heldout_result = ROOT / heldout["result_summary"]
    if not heldout_result.is_file() or _sha256(heldout_result) != heldout[
        "result_summary_sha256"
    ]:
        raise RuntimeError("route-off held-out result summary missing or SHA drift")
    heldout_payload = json.loads(heldout_result.read_text(encoding="utf-8"))
    if heldout_payload.get("status") != "COMPLETE_VALID_WITH_METADATA_ERRATUM":
        raise RuntimeError("route-off held-out result verdict drift")
    if heldout_payload.get("interpretation", {}).get("routed_ppo_authorized") is not False:
        raise RuntimeError("route-off held-out result unexpectedly authorizes routed PPO")
    if heldout.get("target_speed_max_mps") != 1.25 or heldout.get(
        "target_route_mode"
    ) != "off" or heldout.get("placement") != "footprint_clearance":
        raise RuntimeError("route-off held-out eval environment-contract drift")
    launcher = ROOT / heldout["launcher"]
    if not launcher.is_file():
        raise RuntimeError("route-off held-out eval launcher missing")

    if receipt.get("hardware_state") != {
        "assembled_airframe": False,
        "real_sensor_logs": 0,
        "real_flights": 0,
        "sim_to_real_claim_authorized": False,
    }:
        raise RuntimeError("hardware claim boundary drift")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", default=str(DEFAULT_RECEIPT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    receipt = verify_authority(args.receipt)
    summary = {
        "verified": True,
        "track_a": receipt["track_a"]["result"],
        "track_a_stage2_authorised": receipt["track_a"]["stage2_authorised"],
        "track_b": receipt["track_b"]["route_mechanism"],
        "track_b_long_training_authorized": receipt["track_b"]["long_training_authorized"],
        "corrected_route_gate": receipt["corrected_environment_v2_2026_08_27"][
            "route_physical_gate_r2"
        ]["route_mechanism"],
        "corrected_fresh_ppo_epochs_run": receipt["corrected_environment_v2_2026_08_27"][
            "route_physical_gate_r2"
        ]["fresh_ppo_epochs_run"],
        "corrected_nonoverlap_heldout": receipt["corrected_environment_v2_2026_08_27"][
            "route_off_heldout_eval"
        ]["status"],
        "next": receipt["corrected_environment_v2_2026_08_27"]["next_gate"],
    }
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("PASS_RESEARCH_AUTHORITY_FREEZE")
        print("Track A: %s; Stage 2=false" % summary["track_a"])
        print("Track B: %s; long training=false" % summary["track_b"])
        print("Corrected route gate: %s; fresh PPO epochs=0" % summary["corrected_route_gate"])
        print("Corrected route-off held-out: %s" % summary["corrected_nonoverlap_heldout"])
        print("Next: %s" % summary["next"])


if __name__ == "__main__":
    main()
