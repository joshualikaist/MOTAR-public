"""Select current P7, P7b, or P7c on validation under the frozen extension rule."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from perception_candidates import sha256_file
from perception_temporal import selection_utility


TIE_ORDER = (
    "current_transformer_t16",
    "candidate_transformer_t16",
    "candidate_motion_transformer_t16",
)
EXPECTED_ARCHITECTURES = {
    "current_transformer_t16": "transformer",
    "candidate_transformer_t16": "candidate_transformer",
    "candidate_motion_transformer_t16": "candidate_motion_transformer",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-report", type=Path, required=True)
    parser.add_argument("--current-receipt", type=Path, required=True)
    parser.add_argument("--p7b-report", type=Path, required=True)
    parser.add_argument("--p7b-receipt", type=Path, required=True)
    parser.add_argument("--p7c-report", type=Path, required=True)
    parser.add_argument("--p7c-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audited_report(report_path, receipt_path):
    report_path, receipt_path = Path(report_path), Path(receipt_path)
    report = json.loads(report_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if sha256_file(report_path) != receipt["report_sha256"]:
        raise ValueError("P7 extension report hash mismatch: %s" % report_path)
    return report, receipt


def select_arm(arms):
    if set(arms) != set(TIE_ORDER):
        raise ValueError("P7 extension selection requires exactly the three frozen arms")
    return max(TIE_ORDER, key=lambda name: (
        selection_utility(arms[name]), -TIE_ORDER.index(name)))


def main():
    args = parse_args()
    output = args.output.resolve()
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() or receipt_path.exists():
        raise SystemExit("[p7bc-selection] refusing existing output/receipt")
    paths = {
        "current_transformer_t16": (args.current_report, args.current_receipt),
        "candidate_transformer_t16": (args.p7b_report, args.p7b_receipt),
        "candidate_motion_transformer_t16": (args.p7c_report, args.p7c_receipt),
    }
    reports, receipts = {}, {}
    for name, (report_path, receipt_path_in) in paths.items():
        reports[name], receipts[name] = audited_report(report_path, receipt_path_in)
        if reports[name]["architecture"] != EXPECTED_ARCHITECTURES[name]:
            raise SystemExit("[p7bc-selection] architecture mismatch for %s" % name)
    manifest_hashes = {value["validation_manifest_sha256"] for value in receipts.values()}
    candidate_hashes = {value["validation_candidates_sha256"] for value in receipts.values()}
    detector_hashes = {value["detector_weights_sha256"] for value in receipts.values()}
    frame_counts = {value["validation_provenance"]["frames"] for value in reports.values()}
    if any(len(values) != 1 for values in (
            manifest_hashes, candidate_hashes, detector_hashes, frame_counts)):
        raise SystemExit("[p7bc-selection] arm provenance differs")
    arms = {}
    for name in TIE_ORDER:
        metrics = dict(reports[name]["association_metrics"])
        computed = selection_utility(metrics)
        if abs(float(metrics["selection_utility"]) - computed) > 1e-12:
            raise ValueError("P7 extension utility mismatch for %s" % name)
        metrics["selection_utility"] = computed
        arms[name] = metrics
    selected = select_arm(arms)
    result = {
        "schema_version": 1,
        "experiment": "motar.p7b-p7c.validation-selection.v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "selection_rule": "(hit_frames - proxy_false_lock_frames) / frames_with_ground_truth",
        "exact_tie_order": list(TIE_ORDER),
        "selected_arm": selected,
        "selected_utility": arms[selected]["selection_utility"],
        "arms": arms,
        "provenance": {
            "manifest_sha256": next(iter(manifest_hashes)),
            "candidates_sha256": next(iter(candidate_hashes)),
            "detector_weights_sha256": next(iter(detector_hashes)),
            "report_sha256": {
                name: receipts[name]["report_sha256"] for name in TIE_ORDER
            },
            "p7c_validation_motion_sha256": receipts[
                "candidate_motion_transformer_t16"]["validation_motion_sha256"],
        },
        "test_status": "NOT_AN_INPUT_TO_P7B_P7C_TRAINING_OR_SELECTION",
        "limitations": {
            "identity_metrics": "NOT_IDENTIFIABLE_SOURCE_TRACK_IDS_DROPPED",
            "ftlr": "NOT_IDENTIFIABLE_WITHOUT_DESIGNATED_TARGET_ID",
            "proxy_definition": "selected candidate has IoU below 0.3 against every UAV annotation",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": 1,
        "selection_sha256": sha256_file(output),
        "input_report_sha256": result["provenance"]["report_sha256"],
        "manifest_sha256": next(iter(manifest_hashes)),
        "candidates_sha256": next(iter(candidate_hashes)),
        "detector_weights_sha256": next(iter(detector_hashes)),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("[p7bc-selection] PASS: %s utility %.6f -> %s" % (
        selected, result["selected_utility"], output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
