"""Select CNN/KF/GRU/Transformer on NPS validation only under the frozen P6/P7 rule."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from perception_candidates import sha256_file
from perception_temporal import selection_utility


TIE_ORDER = (
    "cnn_only_top1",
    "cnn_plus_kf",
    "cnn_plus_gru_t8",
    "cnn_plus_temporal_transformer_t16",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p5-report", type=Path, required=True)
    parser.add_argument("--p5-receipt", type=Path, required=True)
    parser.add_argument("--gru-report", type=Path, required=True)
    parser.add_argument("--gru-receipt", type=Path, required=True)
    parser.add_argument("--transformer-report", type=Path, required=True)
    parser.add_argument("--transformer-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audited_report(report_path, receipt_path):
    report_path, receipt_path = Path(report_path), Path(receipt_path)
    report, receipt = json.loads(report_path.read_text()), json.loads(receipt_path.read_text())
    if sha256_file(report_path) != receipt["report_sha256"]:
        raise ValueError("association input report hash mismatch: %s" % report_path)
    return report, receipt


def select_arm(arms):
    unknown = set(arms) - set(TIE_ORDER)
    if unknown:
        raise ValueError("unknown association arms: %s" % sorted(unknown))
    if set(arms) != set(TIE_ORDER):
        raise ValueError("selection requires all four preregistered arms")
    return max(TIE_ORDER, key=lambda name: (selection_utility(arms[name]), -TIE_ORDER.index(name)))


def main():
    args = parse_args()
    output = args.output.resolve()
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if output.exists() or receipt_path.exists():
        raise SystemExit("[association-selection] refusing existing output/receipt")
    p5, p5_receipt = audited_report(args.p5_report, args.p5_receipt)
    gru, gru_receipt = audited_report(args.gru_report, args.gru_receipt)
    transformer, transformer_receipt = audited_report(
        args.transformer_report, args.transformer_receipt)
    if p5.get("split") != "val":
        raise SystemExit("[association-selection] P5 input is not validation")
    if gru["architecture"] != "gru" or transformer["architecture"] != "transformer":
        raise SystemExit("[association-selection] temporal architecture receipt mismatch")

    manifest_hashes = {
        p5["manifest_sha256"],
        gru_receipt["validation_manifest_sha256"],
        transformer_receipt["validation_manifest_sha256"],
    }
    candidate_hashes = {
        p5["candidates_sha256"],
        gru_receipt["validation_candidates_sha256"],
        transformer_receipt["validation_candidates_sha256"],
    }
    detector_hashes = {
        p5["detector_weights_sha256"],
        gru_receipt["detector_weights_sha256"],
        transformer_receipt["detector_weights_sha256"],
    }
    if len(manifest_hashes) != 1 or len(candidate_hashes) != 1 or len(detector_hashes) != 1:
        raise SystemExit("[association-selection] arm provenance differs")
    if gru["validation_provenance"]["frames"] != p5["frames"]:
        raise SystemExit("[association-selection] GRU validation frame count differs")
    if transformer["validation_provenance"]["frames"] != p5["frames"]:
        raise SystemExit("[association-selection] Transformer validation frame count differs")

    arms = {
        "cnn_only_top1": p5["arms"]["cnn_only_top1"],
        "cnn_plus_kf": p5["arms"]["cnn_plus_kf"],
        "cnn_plus_gru_t8": gru["association_metrics"],
        "cnn_plus_temporal_transformer_t16": transformer["association_metrics"],
    }
    audited_arms = {}
    for name in TIE_ORDER:
        metrics = dict(arms[name])
        computed = selection_utility(metrics)
        recorded = metrics.get("selection_utility")
        if recorded is not None and abs(float(recorded) - computed) > 1e-12:
            raise ValueError("recorded temporal selection utility mismatch")
        metrics["selection_utility"] = computed
        audited_arms[name] = metrics
    selected = select_arm(audited_arms)
    result = {
        "schema_version": 1,
        "experiment": "motar.p6-p7.validation-association-selection.v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "selection_rule": "(hit_frames - proxy_false_lock_frames) / frames_with_ground_truth",
        "exact_tie_order": list(TIE_ORDER),
        "selected_arm": selected,
        "selected_utility": audited_arms[selected]["selection_utility"],
        "arms": audited_arms,
        "provenance": {
            "manifest_sha256": next(iter(manifest_hashes)),
            "candidates_sha256": next(iter(candidate_hashes)),
            "detector_weights_sha256": next(iter(detector_hashes)),
            "p5_report_sha256": p5_receipt["report_sha256"],
            "gru_report_sha256": gru_receipt["report_sha256"],
            "transformer_report_sha256": transformer_receipt["report_sha256"],
        },
        "test_status": "NOT_AN_INPUT_TO_P6_P7_TRAINING_OR_SELECTION",
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
        "p5_report_sha256": p5_receipt["report_sha256"],
        "gru_report_sha256": gru_receipt["report_sha256"],
        "transformer_report_sha256": transformer_receipt["report_sha256"],
        "manifest_sha256": next(iter(manifest_hashes)),
        "candidates_sha256": next(iter(candidate_hashes)),
        "detector_weights_sha256": next(iter(detector_hashes)),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("[association-selection] PASS: %s utility %.6f -> %s" % (
        selected, result["selected_utility"], output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
