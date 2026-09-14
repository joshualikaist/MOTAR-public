"""Zero-shot evaluation of a public air-to-air UAV detector on NPS-Drones.

Phase 2 (docs/plans/perception_shape_temporal_redesign_2026-09-03.md is SUPERSEDED; the
current direction is in README.md "External data"). The question this answers is narrow and
decides whether we train anything at all:

    Does a publicly available air-to-air detector, trained on someone else's real data,
    already work well enough on real air-to-air imagery that we can skip training and go
    straight to characterising its error?

This is deliberately a CROSS-DATASET test: the GLAD weights were trained on ARD-MAV and are
evaluated here on NPS-Drones, which they have never seen. That is the honest analogue of
"we will deploy this on our own footage", and it is a harder test than the in-domain numbers
the source papers report.

Nothing here touches the simulator, the actor observation, or any checkpoint. It reads real
images and writes a JSON report.

Licences (see README.md "External data"): NPS-Drones is BSD-3-Clause. The GLAD weights ship
with an MIT badge but NO LICENSE file -- fine for measuring, NOT safe to redistribute.
ultralytics/yolov5 is AGPL-3.0, which is why this runs in an isolated venv and why we do not
vendor its source.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys

# Slice edges on sqrt(box area), the "equivalent side" in pixels. These are frozen here rather
# than chosen after seeing results: <12 px is the regime our own 20 m target occupies at the
# narrow-FOV 320x180 configuration (11.3 px), and 8 px is the Johnson recognition threshold.
SIZE_EDGES_PX = (0.0, 8.0, 12.0, 20.0, 32.0, math.inf)
IOU_MATCH = 0.3  # a 1 px offset on a 10 px box already costs ~0.3 IoU; 0.5 would measure
                 # localisation precision, not detection, at these sizes.


def parse_mot(path):
    """refined_gt/Clip_N_refined.txt -> {frame: [(l, t, w, h, conf), ...]}"""
    per_frame = {}
    with open(path) as fh:
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 7:
                continue
            frame = int(p[0])
            per_frame.setdefault(frame, []).append(
                (float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6]))
            )
    return per_frame


def iou(a, b):
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    return inter / (aw * ah + bw * bh - inter)


def size_bin(w, h):
    side = math.sqrt(max(w * h, 1e-9))
    for i in range(len(SIZE_EDGES_PX) - 1):
        if SIZE_EDGES_PX[i] <= side < SIZE_EDGES_PX[i + 1]:
            return i
    return len(SIZE_EDGES_PX) - 2


def bin_label(i):
    lo, hi = SIZE_EDGES_PX[i], SIZE_EDGES_PX[i + 1]
    return f"{lo:.0f}-{'inf' if hi == math.inf else f'{hi:.0f}'}px"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="directory of NPS clip videos")
    ap.add_argument("--gt", required=True, help="refined_gt directory")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--yolov5", required=True, help="path to the yolov5 source tree")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=30, help="sample every Nth annotated frame")
    ap.add_argument("--max-frames", type=int, default=1500)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    sys.path.insert(0, args.yolov5)
    import cv2
    import torch
    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(args.weights, device=device, fp16=False)
    model.warmup(imgsz=(1, 3, args.imgsz, args.imgsz))

    gt_dir, vid_dir = Path(args.gt), Path(args.videos)
    clips = sorted(gt_dir.glob("Clip_*_refined.txt"), key=lambda p: int(p.name.split("_")[1]))

    # Evenly spread the frame budget over clips so no single long clip dominates the score.
    per_clip_budget = max(1, args.max_frames // max(1, len(clips)))

    tp = [0] * (len(SIZE_EDGES_PX) - 1)
    fn = [0] * (len(SIZE_EDGES_PX) - 1)
    gt_count = [0] * (len(SIZE_EDGES_PX) - 1)
    fp_total = 0
    frames_done = 0
    scores = []  # (confidence, is_true_positive) for AP
    clip_rows = []

    for gt_path in clips:
        idx = int(gt_path.name.split("_")[1])
        cand = [p for ext in ("*.mov", "*.mp4", "*.MOV", "*.MP4", "*.avi")
                for p in vid_dir.glob(ext) if f"Clip_{idx}." in p.name or p.stem in (f"Clip_{idx}", f"clip_{idx}")]
        if not cand:
            continue
        per_frame = parse_mot(gt_path)
        wanted = sorted(per_frame)[:: args.stride][:per_clip_budget]
        if not wanted:
            continue
        cap = cv2.VideoCapture(str(cand[0]))
        clip_tp = clip_gt = clip_fp = 0
        for frame_no in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)  # MOT frames are 1-based
            ok, img0 = cap.read()
            if not ok:
                continue
            img = letterbox(img0, args.imgsz, stride=int(model.stride), auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]  # BGR->RGB, HWC->CHW
            t = torch.from_numpy(img.copy()).to(device).float() / 255.0
            with torch.no_grad():
                pred = model(t[None])
            det = non_max_suppression(pred, args.conf, 0.45, max_det=64)[0]
            preds = []
            if len(det):
                det[:, :4] = scale_boxes(t.shape[1:], det[:, :4], img0.shape).round()
                for *xyxy, cf, _cls in det.tolist():
                    preds.append((xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1], cf))
            preds.sort(key=lambda p: -p[4])

            truths = per_frame[frame_no]
            used = set()
            for pi, pb in enumerate(preds):
                best, best_i = 0.0, -1
                for ti, tb in enumerate(truths):
                    if ti in used:
                        continue
                    v = iou(pb, tb)
                    if v > best:
                        best, best_i = v, ti
                if best >= IOU_MATCH:
                    used.add(best_i)
                    b = size_bin(truths[best_i][2], truths[best_i][3])
                    tp[b] += 1
                    clip_tp += 1
                    scores.append((pb[4], 1))
                else:
                    fp_total += 1
                    clip_fp += 1
                    scores.append((pb[4], 0))
            for ti, tb in enumerate(truths):
                b = size_bin(tb[2], tb[3])
                gt_count[b] += 1
                clip_gt += 1
                if ti not in used:
                    fn[b] += 1
            frames_done += 1
        cap.release()
        clip_rows.append({"clip": idx, "frames": len(wanted), "gt": clip_gt,
                          "tp": clip_tp, "fp": clip_fp})

    # AP by the standard all-point interpolation over the confidence-sorted detections.
    scores.sort(key=lambda s: -s[0])
    total_gt = sum(gt_count)
    ap = 0.0
    if total_gt and scores:
        seen_tp = seen_fp = 0
        prev_recall = 0.0
        for cf, is_tp in scores:
            seen_tp += is_tp
            seen_fp += 1 - is_tp
            recall = seen_tp / total_gt
            precision = seen_tp / (seen_tp + seen_fp)
            ap += (recall - prev_recall) * precision
            prev_recall = recall

    tp_all, fn_all = sum(tp), sum(fn)
    precision = tp_all / (tp_all + fp_total) if (tp_all + fp_total) else 0.0
    recall = tp_all / total_gt if total_gt else 0.0
    report = {
        "schema_version": 1,
        "experiment": "air2air_detector_zeroshot",
        "weights": os.path.basename(args.weights),
        "trained_on": "ARD-MAV (GLAD)" if "GLAD" in args.weights else "unknown",
        "evaluated_on": "NPS-Drones refined_gt v2",
        "cross_dataset": True,
        "iou_match": IOU_MATCH,
        "imgsz": args.imgsz,
        "conf_threshold": args.conf,
        "frames": frames_done,
        "gt_boxes": total_gt,
        "true_positives": tp_all,
        "false_positives": fp_total,
        "false_negatives": fn_all,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "average_precision": ap,
        "by_size": [
            {
                "bin": bin_label(i),
                "gt": gt_count[i],
                "tp": tp[i],
                "recall": (tp[i] / gt_count[i]) if gt_count[i] else None,
            }
            for i in range(len(SIZE_EDGES_PX) - 1)
        ],
        "clips": clip_rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    print(f"frames {frames_done} | GT {total_gt} | P {precision:.3f} R {recall:.3f} "
          f"F1 {report['f1']:.3f} AP {ap:.3f}")
    print("size slice (recall on GT boxes of that equivalent side):")
    for row in report["by_size"]:
        r = "-" if row["recall"] is None else f"{row['recall']:.3f}"
        print(f"  {row['bin']:>10}  gt {row['gt']:>6}  recall {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
