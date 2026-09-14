"""Evaluate CNN-only selection and the frozen P5 KF on the same NPS sequence records."""

import argparse
import gzip
import itertools
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from perception_candidates import canonical_line, read_jsonl, sha256_file, validate_candidate_record
from perception_kf import MultiCandidateKalmanTracker


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "perception_kf_v1.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def box_iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(area_left + area_right - intersection, 1e-12)


def candidate_box(candidate):
    return [
        candidate["u_px"] - candidate["width_px"] / 2.0,
        candidate["v_px"] - candidate["height_px"] / 2.0,
        candidate["u_px"] + candidate["width_px"] / 2.0,
        candidate["v_px"] + candidate["height_px"] / 2.0,
    ]


def selected_result(box, ground_truth, threshold):
    if box is None:
        return {"selected": False, "hit": False, "iou": None, "center_error_px": None}
    overlaps = [box_iou(box, target) for target in ground_truth]
    best_index = int(np.argmax(overlaps)) if overlaps else None
    best_iou = overlaps[best_index] if best_index is not None else 0.0
    center_error = None
    if best_index is not None:
        target = ground_truth[best_index]
        center_error = math.hypot(
            (box[0] + box[2] - target[0] - target[2]) / 2.0,
            (box[1] + box[3] - target[1] - target[3]) / 2.0,
        )
    return {"selected": True, "hit": best_iou >= threshold, "iou": best_iou,
            "center_error_px": center_error}


class ArmMetrics:
    def __init__(self):
        self.frames = 0
        self.selected = 0
        self.hits = 0
        self.center_errors = []
        self.ious = []
        self.loss_events = 0
        self.reacquisition_seconds = []
        self.censored_losses = 0
        self.sequence = None
        self.seen_hit = False
        self.loss_start_ns = None

    def finish_sequence(self):
        if self.loss_start_ns is not None:
            self.censored_losses += 1
        self.seen_hit = False
        self.loss_start_ns = None

    def add(self, sequence, timestamp_ns, result):
        if self.sequence is not None and sequence != self.sequence:
            self.finish_sequence()
        self.sequence = sequence
        self.frames += 1
        self.selected += int(result["selected"])
        self.hits += int(result["hit"])
        if result["hit"]:
            self.ious.append(result["iou"])
            self.center_errors.append(result["center_error_px"])
            if self.loss_start_ns is not None:
                self.reacquisition_seconds.append((timestamp_ns - self.loss_start_ns) / 1e9)
                self.loss_start_ns = None
            self.seen_hit = True
        elif self.seen_hit and self.loss_start_ns is None:
            self.loss_start_ns = timestamp_ns
            self.loss_events += 1

    def report(self):
        self.finish_sequence()
        false_locks = self.selected - self.hits
        return {
            "frames_with_ground_truth": self.frames,
            "selected_frames": self.selected,
            "hit_frames_iou_ge_threshold": self.hits,
            "frame_hit_rate": self.hits / self.frames if self.frames else None,
            "no_lock_frames": self.frames - self.selected,
            "proxy_false_lock_frames": false_locks,
            "proxy_false_lock_rate_among_selected": false_locks / self.selected if self.selected else None,
            "matched_center_error_px_mean": statistics.fmean(self.center_errors) if self.center_errors else None,
            "matched_center_error_px_median": statistics.median(self.center_errors) if self.center_errors else None,
            "matched_iou_mean": statistics.fmean(self.ious) if self.ious else None,
            "loss_events_after_first_hit": self.loss_events,
            "reacquired_events": len(self.reacquisition_seconds),
            "right_censored_loss_events": self.censored_losses,
            "reacquisition_seconds_mean": (
                statistics.fmean(self.reacquisition_seconds) if self.reacquisition_seconds else None),
            "reacquisition_seconds_max": max(self.reacquisition_seconds) if self.reacquisition_seconds else None,
        }


def detection_counts(candidates, ground_truth, confidence, iou_threshold):
    selected = sorted(
        (candidate for candidate in candidates if candidate["confidence"] >= confidence),
        key=lambda candidate: -candidate["confidence"],
    )
    matched_targets = set()
    true_positives = 0
    for candidate in selected:
        overlaps = [box_iou(candidate_box(candidate), target) for target in ground_truth]
        available = [index for index in np.argsort(-np.asarray(overlaps))
                     if int(index) not in matched_targets] if overlaps else []
        if available and overlaps[int(available[0])] >= iou_threshold:
            matched_targets.add(int(available[0]))
            true_positives += 1
    return true_positives, len(selected) - true_positives, len(ground_truth) - true_positives


def main():
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit("[kf-eval] refusing existing output")
    manifest_receipt = json.loads(args.manifest_receipt.read_text())
    candidate_receipt = json.loads(args.candidate_receipt.read_text())
    if sha256_file(args.manifest) != manifest_receipt["manifest_sha256"]:
        raise SystemExit("[kf-eval] manifest hash mismatch")
    if sha256_file(args.candidates) != candidate_receipt["output_sha256"]:
        raise SystemExit("[kf-eval] candidate hash mismatch")
    if candidate_receipt["manifest_sha256"] != manifest_receipt["manifest_sha256"]:
        raise SystemExit("[kf-eval] candidates came from a different manifest")
    config = json.loads(args.config.read_text())
    expected_records = int(candidate_receipt["records"])
    manifest_rows = itertools.islice(read_jsonl(args.manifest), expected_records)
    candidate_rows = read_jsonl(args.candidates)
    tracker = None
    previous_candidate = {}
    previous_sequence = None
    arms = {"cnn_only_top1": ArmMetrics(), "cnn_plus_kf": ArmMetrics()}
    detector_counts = {"tp": 0, "fp": 0, "fn": 0}
    tracker_ms = []
    frame_count = 0
    output.mkdir(parents=True)
    raw_path = output / "tracks.jsonl.gz"
    with gzip.open(str(raw_path), "wt", encoding="utf-8", compresslevel=6) as stream:
        missing = object()
        for frame_count, pair in enumerate(
                itertools.zip_longest(manifest_rows, candidate_rows, fillvalue=missing), 1):
            source, candidate_record = pair
            if source is missing or candidate_record is missing:
                raise RuntimeError("selected manifest/candidate record counts differ")
            identity = ("frame_id", "source_sequence_id", "frame_index", "capture_timestamp_ns")
            if any(source[key] != candidate_record[key] for key in identity):
                raise RuntimeError("manifest/candidate identity mismatch")
            validate_candidate_record(candidate_record, previous_candidate)
            sequence = source["source_sequence_id"]
            if sequence != previous_sequence:
                tracker = MultiCandidateKalmanTracker(config)
                previous_sequence = sequence
            began = time.perf_counter_ns()
            selected_track, tracks = tracker.step(
                candidate_record["candidates"], source["capture_timestamp_ns"],
                source["width_px"], source["height_px"])
            tracker_ms.append((time.perf_counter_ns() - began) / 1e6)
            operating = [candidate for candidate in candidate_record["candidates"]
                         if candidate["confidence"] >= config["operating_confidence"]]
            cnn_box = candidate_box(operating[0]) if operating else None
            kf_box = selected_track["box_xyxy"] if selected_track else None
            ground_truth = source["ground_truth_xyxy"]
            cnn_result = selected_result(cnn_box, ground_truth, config["evaluation_iou"])
            kf_result = selected_result(kf_box, ground_truth, config["evaluation_iou"])
            arms["cnn_only_top1"].add(sequence, source["capture_timestamp_ns"], cnn_result)
            arms["cnn_plus_kf"].add(sequence, source["capture_timestamp_ns"], kf_result)
            tp, fp, fn = detection_counts(
                candidate_record["candidates"], ground_truth,
                config["operating_confidence"], config["evaluation_iou"])
            detector_counts["tp"] += tp
            detector_counts["fp"] += fp
            detector_counts["fn"] += fn
            stream.write(canonical_line({
                "frame_id": source["frame_id"],
                "source_sequence_id": sequence,
                "frame_index": source["frame_index"],
                "capture_timestamp_ns": source["capture_timestamp_ns"],
                "cnn_only": {"box_xyxy": cnn_box, **cnn_result},
                "cnn_plus_kf": {"selected_track": selected_track, **kf_result},
                "tracks": tracks,
                "ground_truth_xyxy": ground_truth,
            }))
    if frame_count != expected_records:
        raise RuntimeError("candidate receipt record count mismatch")
    arm_reports = {name: metric.report() for name, metric in arms.items()}
    tp, fp, fn = (detector_counts[key] for key in ("tp", "fp", "fn"))
    report = {
        "schema_version": 1,
        "experiment": "motar.p5.cnn-versus-kf.v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "split": manifest_receipt["split"],
        "frames": frame_count,
        "sequences": len({row["source_sequence_id"] for row in read_jsonl(args.manifest)})
        if candidate_receipt["complete_manifest"] else None,
        "manifest_sha256": manifest_receipt["manifest_sha256"],
        "candidates_sha256": candidate_receipt["output_sha256"],
        "detector_weights_sha256": candidate_receipt["weights_sha256"],
        "kf_config_sha256": sha256_file(args.config),
        "config": config,
        "top_k_detector_at_operating_confidence": {
            **detector_counts,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        },
        "arms": arm_reports,
        "identity_metrics": {
            "id_switch_count": None,
            "id_switch_rate": None,
            "status": "NOT_IDENTIFIABLE_SOURCE_TRACK_IDS_DROPPED_DURING_NPS_YOLO_PREPARATION",
        },
        "ftlr": {
            "status": "NOT_IDENTIFIABLE_WITHOUT_DESIGNATED_TARGET_ID",
            "reported_proxy": "selected box with IoU < evaluation_iou against every annotated UAV",
        },
        "bearing_error": {"status": "UNAVAILABLE_CAMERA_INTRINSICS_NOT_PROVIDED"},
        "latency_ms": {
            "candidate_producer": candidate_receipt["latency_ms"],
            "kf_association_mean": statistics.fmean(tracker_ms),
            "kf_association_p95": sorted(tracker_ms)[int(0.95 * (len(tracker_ms) - 1))],
            "combined_mean": candidate_receipt["latency_ms"]["mean"] + statistics.fmean(tracker_ms),
        },
        "raw_tracks_file": raw_path.name,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": 1,
        "report_sha256": sha256_file(report_path),
        "raw_tracks_sha256": sha256_file(raw_path),
        "manifest_sha256": manifest_receipt["manifest_sha256"],
        "candidates_sha256": candidate_receipt["output_sha256"],
        "detector_weights_sha256": candidate_receipt["weights_sha256"],
        "kf_config_sha256": sha256_file(args.config),
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("[kf-eval] PASS: %d frames -> %s" % (frame_count, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
