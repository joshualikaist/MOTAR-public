"""Relative range-error proxies measured without any distance ground truth.

Contract: docs/plans/prereg_2026-09-09_range_error_proxy.md, registered before this ran.

A size-based monocular range estimator obeys range = k / s, so the *relative* range error is
the inverse size ratio and the unknown constant k (focal length times true vehicle size) cancels.
That is the only reason this works without intrinsics or range ground truth, and it is also why
nothing here yields absolute metres.

Proxy A is the detector's box-size error against the annotation: a lower bound, because the
annotation itself moves with target attitude. Proxy B is the high-frequency residual of the
annotated size within a clip, which recovers that attitude term but also carries annotation noise
and any genuinely fast range change, so it is an upper bound. The two bracket the quantity.
"""
import argparse
import gzip
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.runtime_fingerprint import runtime_fingerprint  # noqa: E402

# Frozen by the preregistration; these are P8's bin edges verbatim so the two reports stack.
BIN_EDGES = [0.0, 8.0, 16.0, 32.0, 64.0]
SUPPORTED_BINS = (1, 2, 3)
MEDIAN_WINDOWS = (3, 5, 9)
PRIMARY_WINDOW = 5
MAX_SEGMENT_GAP_S = 0.5
HIT_IOU = 0.3


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        for line in stream:
            line = line.strip()
            if line:
                yield json.loads(line)


def box_size(xyxy):
    """sqrt(area) in pixels -- P8's size definition."""
    x0, y0, x1, y1 = xyxy
    return math.sqrt(max(0.0, x1 - x0) * max(0.0, y1 - y0))


def size_bin(size_px):
    for index in range(len(BIN_EDGES) - 1, 0, -1):
        if size_px >= BIN_EDGES[index]:
            return index
    return 0


def quantile(values, q):
    """Linear-interpolation quantile; values need not be sorted."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(values):
    if not values:
        return {"n": 0}
    absolute = [abs(v) for v in values]
    return {
        "n": len(values),
        "median": quantile(values, 0.5),
        "iqr": [quantile(values, 0.25), quantile(values, 0.75)],
        "p5": quantile(values, 0.05),
        "p95": quantile(values, 0.95),
        "abs_p50": quantile(absolute, 0.5),
        "abs_p95": quantile(absolute, 0.95),
    }


def proxy_a(frames):
    """Relative range error implied by the detector's box-size error on hit frames."""
    per_bin = {index: [] for index in SUPPORTED_BINS}
    per_clip = {}
    everything = []
    excluded = {"no_selection": 0, "not_hit": 0, "degenerate_size": 0, "unsupported_bin": 0}

    for frame in frames:
        if frame["pred_xyxy"] is None:
            excluded["no_selection"] += 1
            continue
        if not frame["hit"]:
            excluded["not_hit"] += 1
            continue
        gt_size = frame["gt_size"]
        pred_size = box_size(frame["pred_xyxy"])
        if gt_size <= 0.0 or pred_size <= 0.0:
            excluded["degenerate_size"] += 1
            continue
        # range_hat / range_true = s_true / s_pred for range = k / s.
        relative_error = gt_size / pred_size - 1.0
        bin_index = frame["bin"]
        if bin_index not in per_bin:
            excluded["unsupported_bin"] += 1
            continue
        per_bin[bin_index].append(relative_error)
        per_clip.setdefault(frame["clip"], []).append(relative_error)
        everything.append(relative_error)

    return {
        "estimand": "s_gt / s_pred - 1  (relative range error under range = k / s)",
        "excluded_frames": excluded,
        "overall": summarize(everything),
        "by_bin": {str(k): summarize(v) for k, v in sorted(per_bin.items())},
        "by_clip": {k: summarize(v) for k, v in sorted(per_clip.items())},
        "clip_median_spread": summarize(
            [quantile(v, 0.5) for v in per_clip.values() if v]
        ),
    }


def median_filter_residuals(series, window, max_gap_s):
    """Residual of log-size against a centred running median, skipping discontinuous windows."""
    half = window // 2
    residuals = []
    invalid = 0
    for centre in range(len(series)):
        low = centre - half
        high = centre + half
        if low < 0 or high >= len(series):
            invalid += 1
            continue
        span = series[low:high + 1]
        contiguous = all(
            span[i + 1][0] - span[i][0] <= max_gap_s for i in range(len(span) - 1)
        )
        if not contiguous:
            invalid += 1
            continue
        smooth = statistics.median(value for _, value, _ in span)
        residuals.append((series[centre][2], series[centre][1] - smooth))
    return residuals, invalid


def proxy_b(frames):
    """High-frequency residual of the annotated size: the attitude term, plus annotation noise."""
    by_clip = {}
    for frame in frames:
        if frame["gt_size"] <= 0.0:
            continue
        by_clip.setdefault(frame["clip"], []).append(
            (frame["t_s"], math.log(frame["gt_size"]), frame["bin"])
        )
    for clip in by_clip:
        by_clip[clip].sort(key=lambda row: row[0])

    windows = {}
    for window in MEDIAN_WINDOWS:
        per_bin = {index: [] for index in SUPPORTED_BINS}
        per_clip = {}
        everything = []
        invalid_total = 0
        for clip, series in by_clip.items():
            residuals, invalid = median_filter_residuals(series, window, MAX_SEGMENT_GAP_S)
            invalid_total += invalid
            for bin_index, residual in residuals:
                # Back to a multiplicative size (and hence range) factor.
                relative = math.exp(residual) - 1.0
                if bin_index in per_bin:
                    per_bin[bin_index].append(relative)
                    per_clip.setdefault(clip, []).append(relative)
                    everything.append(relative)
        windows[str(window)] = {
            "invalid_windows": invalid_total,
            "overall": summarize(everything),
            "by_bin": {str(k): summarize(v) for k, v in sorted(per_bin.items())},
            "by_clip": {k: summarize(v) for k, v in sorted(per_clip.items())},
        }

    return {
        "estimand": "exp(log s_gt - running_median(log s_gt)) - 1",
        "assumption": "true range varies smoothly over the window; residual is apparent-size change",
        "primary_window": PRIMARY_WINDOW,
        "max_segment_gap_seconds": MAX_SEGMENT_GAP_S,
        "windows": windows,
    }


def verdict(a_report, b_report):
    """Frozen decision rule: the larger of the two P95s, since independence is not established."""
    a95 = a_report["overall"].get("abs_p95")
    b95 = b_report["windows"][str(PRIMARY_WINDOW)]["overall"].get("abs_p95")
    if a95 is None or b95 is None:
        return {"e95": None, "verdict": "INSUFFICIENT_DATA"}
    e95 = max(abs(a95), abs(b95))
    if e95 <= 0.15:
        label = "RANGE_ERROR_BOUNDED_SMALL"
    elif e95 >= 0.30:
        label = "RANGE_ERROR_LARGE"
    else:
        label = "RANGE_ERROR_INCONCLUSIVE"
    return {
        "proxy_a_abs_p95": a95,
        "proxy_b_abs_p95": b95,
        "e95": e95,
        "rule": "max of the two; <=0.15 small, >=0.30 large",
        "verdict": label,
    }


def load_frames(sequence_path, predictions_path):
    """Single-GT validation frames joined to the selector's chosen box."""
    predictions = {}
    for record in read_jsonl(predictions_path):
        predictions[record["frame_id"]] = record

    frames = []
    multi_gt = 0
    missing_prediction = 0
    for record in read_jsonl(sequence_path):
        boxes = record["ground_truth_xyxy"]
        if len(boxes) != 1:
            multi_gt += 1
            continue
        prediction = predictions.get(record["frame_id"])
        if prediction is None:
            missing_prediction += 1
            continue
        gt_size = box_size(boxes[0])
        iou = prediction.get("iou")
        frames.append({
            "frame_id": record["frame_id"],
            "clip": record["source_sequence_id"],
            "t_s": record["capture_timestamp_ns"] / 1e9,
            "gt_size": gt_size,
            "bin": size_bin(gt_size),
            "pred_xyxy": prediction.get("box_xyxy"),
            "hit": bool(prediction.get("hit")) and iou is not None and iou >= HIT_IOU,
        })
    return frames, {"multi_gt_frames": multi_gt, "frames_missing_prediction": missing_prediction}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="NPS validation sequence manifest")
    parser.add_argument("--predictions", required=True, help="frozen selector validation predictions")
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args()

    sequence_path = Path(args.sequence)
    predictions_path = Path(args.predictions)
    out_dir = Path(args.out)
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {out_dir}")

    frames, join_counts = load_frames(sequence_path, predictions_path)
    a_report = proxy_a(frames)
    b_report = proxy_b(frames)

    report = {
        "schema_version": "motar.range-error-proxy.v1",
        "contract": "docs/plans/prereg_2026-09-09_range_error_proxy.md",
        "test_used": False,
        "split": "nps_validation",
        "conditional_population": "exactly_one_GT",
        "size_definition": "sqrt(box_area_px)",
        "bin_edges_px": BIN_EDGES,
        "supported_bins": list(SUPPORTED_BINS),
        "hit_iou": HIT_IOU,
        "single_gt_frames": len(frames),
        "join_counts": join_counts,
        "proxy_a_detector_box_size": a_report,
        "proxy_b_apparent_size_residual": b_report,
        "verdict": verdict(a_report, b_report),
        "not_measured": [
            "absolute metric range error (no intrinsics)",
            "slow bias from a wrong assumed vehicle size",
            "anything on the sealed test split",
        ],
    }

    out_dir.mkdir(parents=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (out_dir / "receipt.json").write_text(json.dumps({
        "schema_version": "motar.range-error-proxy-receipt.v1",
        "contract": "docs/plans/prereg_2026-09-09_range_error_proxy.md",
        "test_used": False,
        "runtime": runtime_fingerprint(),
        "tool_sha256": sha256(Path(__file__)),
        "input_sha256": {
            str(sequence_path): sha256(sequence_path),
            str(predictions_path): sha256(predictions_path),
        },
    }, indent=2, sort_keys=True) + "\n")

    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {out_dir}/report.json")


if __name__ == "__main__":
    main()
