#!/usr/bin/env python3
"""Gate-3 offline dataset, training, calibration, and held-out test for NavRL RGB-D detection.

The frozen PPO policy is never loaded or trained here. Renderer-private target masks are used only
as offline labels. Candidate loss and threshold selection touch train/validation seeds only; the
test seed is evaluated once after selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


# Pin the same v2 renderer/task shape before importing aerial_gym/Isaac Gym.
os.environ.setdefault("NAVRL_VISION", "1")
os.environ.setdefault("NAVRL_PERCEPTION", "1")
os.environ.setdefault("NAVRL_GENERAL_TRAIN", "1")
os.environ.setdefault("NAVRL_NUM_BARS", "205")
os.environ.setdefault("NAVRL_FIXED_BARS", "205")
os.environ.setdefault("NAVRL_MAX_BARS", "300")
os.environ.setdefault("NAVRL_DENSITY_CURRICULUM", "0")
os.environ.setdefault("NAVRL_EVAL_FULL_DISTRIBUTION", "1")
os.environ.setdefault("NAVRL_LIDAR_RANGE", "12")
os.environ.setdefault("NAVRL_LIDAR_HBEAMS", "72")
os.environ.setdefault("NAVRL_LIDAR_VBEAMS", "4")
os.environ.setdefault("NAVRL_MAX_OBSTACLES", "8")
os.environ.setdefault("NAVRL_OBSTACLE_SELECTOR", "cluster_sector")
os.environ.setdefault("NAVRL_OBSTACLE_FOV_DEG", "240")
os.environ.setdefault("NAVRL_PLACEMENT_MODE", "navrl_band")
os.environ.setdefault("NAVRL_PERCEPTION_PERTURB", "0")
os.environ.pop("NAVRL_DETECTOR_CHECKPOINT", None)

SCHEMA_VERSION = 2
TRAIN_SEED = 71
VALIDATION_SEED = 73
TEST_SEED = 79
RANGE_EDGES_M = (2.0, 6.0, 10.0, 14.0, 17.0, 20.0)
THRESHOLDS = tuple(round(0.05 + 0.025 * index, 3) for index in range(37))
TARGET_BATCHES_PER_LAYOUT = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/navrl_target_detector_v2.pth"))
    parser.add_argument(
        "--result-root", type=Path, default=Path("results/navrl_detector_offline_gate_v2")
    )
    parser.add_argument("--train-frames", type=int, default=8192)
    parser.add_argument("--validation-frames", type=int, default=2048)
    parser.add_argument("--test-frames", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--pixel-tolerance-px", type=int, default=0)
    parser.add_argument("--selection-range-mae-max", type=float, default=0.25)
    parser.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--validation-seed", type=int, default=VALIDATION_SEED)
    parser.add_argument("--test-seed", type=int, default=TEST_SEED)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_args(args):
    for name in ("train_frames", "validation_frames", "test_frames", "num_envs", "epochs", "batch_size"):
        if int(getattr(args, name)) <= 0:
            raise SystemExit("[detector-v2] %s must be positive" % name)
    if args.train_frames < args.num_envs or args.validation_frames < args.num_envs:
        raise SystemExit("[detector-v2] each split must contain at least one vector batch")
    if not args.preflight:
        if args.output.exists() or args.output.with_suffix(".receipt.json").exists():
            raise SystemExit("[detector-v2] refusing to overwrite artifact/receipt: %s" % args.output)
        if args.result_root.exists():
            raise SystemExit("[detector-v2] refusing to overwrite result root: %s" % args.result_root)


def _camera_rays(width, height, hfov, vfov, device):
    rows = torch.arange(height, device=device, dtype=torch.float32)
    cols = torch.arange(width, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(rows, cols, indexing="ij")
    fx = width / (2.0 * math.tan(hfov * 0.5))
    fy = height / (2.0 * math.tan(vfov * 0.5))
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    rays = torch.stack(
        [torch.ones_like(uu), -(uu - cx) / fx, -(vv - cy) / fy], dim=-1
    )
    return rays / rays.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def _sample_broad_target_vectors(count, detector, generator, device):
    range_bin = torch.arange(count, device=device) % (len(RANGE_EDGES_M) - 1)
    range_bin = range_bin[torch.randperm(count, generator=generator, device=device)]
    low = torch.tensor(RANGE_EDGES_M[:-1], device=device)[range_bin]
    high = torch.tensor(RANGE_EDGES_M[1:], device=device)[range_bin]
    center_range = low + (high - low) * torch.rand(count, generator=generator, device=device)
    bearing = (
        torch.rand(count, generator=generator, device=device) * 2.0 - 1.0
    ) * detector.half_hfov * 0.92
    elevation = (
        torch.rand(count, generator=generator, device=device) * 2.0 - 1.0
    ) * detector.half_vfov * 0.82
    direction = torch.stack(
        [torch.ones_like(bearing), torch.tan(bearing), torch.tan(elevation)], dim=1
    )
    direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return direction * center_range.unsqueeze(1), center_range, bearing


def _sample_absent_vectors(count, detector, generator, device):
    distance = 4.0 + 14.0 * torch.rand(count, generator=generator, device=device)
    outside = torch.rand(count, generator=generator, device=device) >= 0.5
    sign = torch.where(
        torch.rand(count, generator=generator, device=device) >= 0.5,
        torch.ones(count, device=device),
        -torch.ones(count, device=device),
    )
    bearing = sign * (detector.half_hfov + math.radians(12.0))
    direction = torch.stack(
        [torch.ones_like(bearing), torch.tan(bearing), torch.zeros_like(bearing)], dim=1
    )
    direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-9)
    behind = torch.stack([-torch.ones_like(distance), torch.zeros_like(distance), torch.zeros_like(distance)], dim=1)
    direction = torch.where(outside.unsqueeze(1), direction, behind)
    return direction * distance.unsqueeze(1), distance, bearing


def collect_split(task, frame_count, seed, split_name):
    random.seed(seed)
    torch.manual_seed(seed)
    device = task.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    detector = task.detector
    zero_action = torch.zeros(task.num_envs, 4, device=device)
    obstacle_rays = _camera_rays(
        detector.obstacle_width,
        detector.obstacle_height,
        detector.hfov,
        detector.vfov,
        device,
    ).reshape(-1, 3)

    rgb_parts, depth_parts, mask_parts = [], [], []
    present_parts, forced_occ_parts, range_parts, bearing_parts = [], [], [], []
    collected = 0
    batch_index = 0
    while collected < frame_count:
        # A layout is expensive to generate under navrl_band. Reuse it for four independently
        # sampled target batches; train/validation/test still contain 128/32/64 distinct layouts.
        if batch_index % TARGET_BATCHES_PER_LAYOUT == 0:
            task.reset()
            task.step(zero_action)
        pos = task.obs_dict["robot_position"]
        quat = task.obs_dict["robot_vehicle_orientation"]
        camera_offset = detector.camera_offset_vehicle.expand(task.num_envs, -1)
        camera_origin = pos + quat_rotate(quat, camera_offset)

        # First render with targets behind the camera to obtain actor-invisible obstacle depth.
        probe_vehicle = torch.tensor([-5.0, 0.0, 0.0], device=device).expand(task.num_envs, 3)
        probe_world = camera_origin + quat_rotate(quat, probe_vehicle)
        detector.render_raw_rgbd(pos, quat, probe_world)
        obstacle_depth = detector.obstacle_depth.reshape(task.num_envs, -1)

        scenario = torch.arange(task.num_envs, device=device) % 5
        scenario = scenario[torch.randperm(task.num_envs, generator=generator, device=device)]
        planned_present = scenario != 0  # 20% target-absent.
        forced_occlusion = scenario == 1  # 20% targeted behind a rendered obstacle.

        vehicle_target, center_range, bearing = _sample_broad_target_vectors(
            task.num_envs, detector, generator, device
        )
        absent_vector, absent_range, absent_bearing = _sample_absent_vectors(
            task.num_envs, detector, generator, device
        )
        vehicle_target = torch.where(planned_present.unsqueeze(1), vehicle_target, absent_vector)
        center_range = torch.where(planned_present, center_range, absent_range)
        bearing = torch.where(planned_present, bearing, absent_bearing)

        valid_obstacle = (
            (obstacle_depth > 1.5)
            & (obstacle_depth < min(17.5, detector.obstacle_max_range * 0.995))
        )
        random_rank = torch.rand(obstacle_depth.shape, generator=generator, device=device)
        random_rank = random_rank.masked_fill(~valid_obstacle, -1.0)
        obstacle_index = random_rank.argmax(dim=1)
        has_obstacle = valid_obstacle.any(dim=1)
        obstacle_range = obstacle_depth.gather(1, obstacle_index.unsqueeze(1)).squeeze(1)
        behind_range = (obstacle_range + 0.35 + 1.25 * torch.rand(
            task.num_envs, generator=generator, device=device
        )).clamp(max=19.5)
        occluded_vector = obstacle_rays[obstacle_index] * behind_range.unsqueeze(1)
        use_forced = forced_occlusion & has_obstacle
        vehicle_target = torch.where(use_forced.unsqueeze(1), occluded_vector, vehicle_target)
        center_range = torch.where(use_forced, behind_range, center_range)
        bearing = torch.where(
            use_forced,
            torch.atan2(occluded_vector[:, 1], occluded_vector[:, 0]),
            bearing,
        )
        forced_occlusion = use_forced

        target_world = camera_origin + quat_rotate(quat, vehicle_target)
        rgb, depth = detector.render_raw_rgbd(pos, quat, target_world)
        mask = detector.target_mask > 0

        take = min(task.num_envs, frame_count - collected)
        rgb_parts.append((rgb[:take].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu())
        depth_parts.append(depth[:take].to(torch.float16).cpu())
        mask_parts.append(mask[:take].cpu())
        present_parts.append(planned_present[:take].cpu())
        forced_occ_parts.append(forced_occlusion[:take].cpu())
        range_parts.append(center_range[:take].to(torch.float32).cpu())
        bearing_parts.append(bearing[:take].to(torch.float32).cpu())
        collected += take
        batch_index += 1
        if batch_index % 16 == 0 or collected == frame_count:
            print("[detector-v2] collect %s %d/%d" % (split_name, collected, frame_count))

    dataset = {
        "rgb": torch.cat(rgb_parts),
        "depth": torch.cat(depth_parts),
        "mask": torch.cat(mask_parts),
        "planned_present": torch.cat(present_parts),
        "forced_occlusion": torch.cat(forced_occ_parts),
        "center_range_m": torch.cat(range_parts),
        "bearing_rad": torch.cat(bearing_parts),
        "seed": seed,
        "name": split_name,
    }
    counts = dataset["mask"].sum(dim=(1, 2))
    print(
        "[detector-v2] %s frames=%d positive_pixel_rate=%.6f visible=%d absent=%d "
        "forced_full=%d forced_partial=%d small=%d"
        % (
            split_name,
            frame_count,
            float(dataset["mask"].float().mean()),
            int((counts >= 2).sum()),
            int((~dataset["planned_present"]).sum()),
            int((dataset["forced_occlusion"] & (counts < 2)).sum()),
            int((dataset["forced_occlusion"] & (counts >= 2)).sum()),
            int(((counts >= 2) & (counts <= 5)).sum()),
        )
    )
    return dataset


def _input_batch(dataset, indices, device, max_range, augment, generator):
    rgb = dataset["rgb"][indices].to(device=device, dtype=torch.float32) / 255.0
    depth = dataset["depth"][indices].to(device=device, dtype=torch.float32)
    if augment:
        rgb = (rgb + torch.randn(rgb.shape, generator=generator, device=device) * 0.015).clamp(0, 1)
        depth = (depth + torch.randn(depth.shape, generator=generator, device=device) * 0.02).clamp(0, max_range)
    mask = dataset["mask"][indices].to(device=device, dtype=torch.float32)
    features = torch.cat([(rgb), (depth / max_range).clamp(0, 1).unsqueeze(1)], dim=1)
    return features, depth, mask


def _candidate_loss(logits, target, kind, pos_weight):
    positive_weight = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=positive_weight, reduction="none"
    )
    if kind == "balanced_bce":
        return bce.mean()
    probability = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, probability, 1.0 - probability)
    focal = ((1.0 - pt).square() * bce).mean()
    intersection = (probability * target).sum(dim=(1, 2))
    denominator = probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return focal + 0.25 * dice


def train_candidate(dataset, kind, epochs, batch_size, max_range, device, seed, arch="pixel_1x1"):
    torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    if arch == "spatial_cnn":
        model = SpatialTargetSegmenter()
    elif arch == "spatial_cnn_wide":
        model = SpatialTargetSegmenterWide()
    else:
        model = AppearanceTargetSegmenter()
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    positive = float(dataset["mask"].sum())
    total = float(dataset["mask"].numel())
    pos_weight = min(2048.0, max(1.0, (total - positive) / max(positive, 1.0)))
    n = dataset["rgb"].shape[0]
    for epoch in range(epochs):
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed + epoch))
        total_loss, seen = 0.0, 0
        for start in range(0, n, batch_size):
            indices = order[start : start + batch_size]
            features, _, target = _input_batch(
                dataset, indices, device, max_range, True, generator
            )
            logits = model.forward_logits(features).squeeze(1)
            loss = _candidate_loss(logits, target, kind, pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * indices.numel()
            seen += indices.numel()
        print(
            "[detector-v2] train candidate=%s epoch=%d/%d loss=%.6f pos_weight=%.1f"
            % (kind, epoch + 1, epochs, total_loss / max(seen, 1), pos_weight)
        )
    return model.eval(), pos_weight


def infer_scores(model, dataset, batch_size, max_range, device):
    scores = []
    n = dataset["rgb"].shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            indices = torch.arange(start, min(start + batch_size, n))
            features, _, _ = _input_batch(dataset, indices, device, max_range, False, None)
            scores.append(
                torch.sigmoid(model.forward_logits(features)).squeeze(1).to(torch.float16).cpu()
            )
    return torch.cat(scores)


def detector_metrics(scores, dataset, threshold, min_pixels, fx, pixel_tolerance_px=0):
    predicted = scores.float() >= float(threshold)
    target = dataset["mask"]
    predicted_count = predicted.sum(dim=(1, 2))
    target_count = target.sum(dim=(1, 2))
    predicted_visible = predicted_count >= min_pixels
    target_visible = target_count >= min_pixels
    true_positive_frame = predicted_visible & target_visible

    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    if pixel_tolerance_px > 0:
        # Blur-aware precision: motion blur smears genuine target evidence into pixels the
        # instantaneous GT mask labels background, so EXACT pixel precision under the appearance
        # envelope punishes physics, not the model. A prediction within `tolerance` pixels of the
        # GT mask counts toward precision; sprayed false positives far from the target still
        # count against it. Recall/IoU stay exact.
        kernel = 2 * int(pixel_tolerance_px) + 1
        dilated = (
            F.max_pool2d(
                target.float().unsqueeze(1), kernel_size=kernel, stride=1,
                padding=int(pixel_tolerance_px),
            ).squeeze(1) > 0.5
        )
        tp_tol = int((predicted & dilated).sum())
        fp_tol = int((predicted & ~dilated).sum())
        pixel_precision = tp_tol / max(tp_tol + fp_tol, 1)
    else:
        pixel_precision = tp / max(tp + fp, 1)
    pixel_recall = tp / max(tp + fn, 1)
    pixel_iou = tp / max(tp + fp + fn, 1)
    frame_tp = int(true_positive_frame.sum())
    frame_fp = int((predicted_visible & ~target_visible).sum())
    frame_fn = int((~predicted_visible & target_visible).sum())
    frame_precision = frame_tp / max(frame_tp + frame_fp, 1)
    frame_recall = frame_tp / max(frame_tp + frame_fn, 1)
    frame_f1 = 2 * frame_precision * frame_recall / max(frame_precision + frame_recall, 1e-12)

    planned_absent = ~dataset["planned_present"]
    fully_occluded = dataset["planned_present"] & ~target_visible
    partial = dataset["forced_occlusion"] & target_visible
    small = target_visible & (target_count <= 5)
    far = target_visible & (dataset["center_range_m"] >= 14.0)

    def rate(mask, success):
        count = int(mask.sum())
        return (float((mask & success).sum()) / count if count else None, count)

    absent_fpr, absent_n = rate(planned_absent, predicted_visible)
    full_occ_fpr, full_occ_n = rate(fully_occluded, predicted_visible)
    partial_recall, partial_n = rate(partial, predicted_visible)
    small_recall, small_n = rate(small, predicted_visible)
    far_recall, far_n = rate(far, predicted_visible)

    rows = torch.arange(target.shape[1], dtype=torch.float32).view(1, -1, 1)
    cols = torch.arange(target.shape[2], dtype=torch.float32).view(1, 1, -1)
    pred_den = predicted_count.clamp(min=1).float()
    gt_den = target_count.clamp(min=1).float()
    pred_u = (predicted.float() * cols).sum(dim=(1, 2)) / pred_den
    gt_u = (target.float() * cols).sum(dim=(1, 2)) / gt_den
    bearing_error = (torch.atan((pred_u - gt_u).abs() / fx) * 180.0 / math.pi)
    depth = dataset["depth"].float()
    pred_range = (depth * predicted.float()).sum(dim=(1, 2)) / pred_den
    gt_range = (depth * target.float()).sum(dim=(1, 2)) / gt_den
    if bool(true_positive_frame.any()):
        bearing_mae = float(bearing_error[true_positive_frame].mean())
        range_mae = float((pred_range[true_positive_frame] - gt_range[true_positive_frame]).abs().mean())
    else:
        bearing_mae = float("inf")
        range_mae = float("inf")

    range_recall = {}
    for low, high in zip(RANGE_EDGES_M[:-1], RANGE_EDGES_M[1:]):
        in_bin = target_visible & (dataset["center_range_m"] >= low) & (dataset["center_range_m"] < high)
        value, count = rate(in_bin, predicted_visible)
        range_recall["%.0f-%.0fm" % (low, high)] = {"recall": value, "frames": count}

    return {
        "threshold": float(threshold),
        "pixel_precision": pixel_precision,
        "pixel_recall": pixel_recall,
        "pixel_iou": pixel_iou,
        "frame_precision": frame_precision,
        "frame_recall": frame_recall,
        "frame_f1": frame_f1,
        "absent_false_positive_rate": absent_fpr,
        "absent_frames": absent_n,
        "fully_occluded_false_positive_rate": full_occ_fpr,
        "fully_occluded_frames": full_occ_n,
        "partial_occlusion_recall": partial_recall,
        "partial_occlusion_frames": partial_n,
        "small_target_recall": small_recall,
        "small_target_frames": small_n,
        "far_14_20m_recall": far_recall,
        "far_14_20m_frames": far_n,
        "bearing_mae_deg": bearing_mae,
        "range_mae_m": range_mae,
        "range_bin_recall": range_recall,
    }


def _metric_or_zero(metrics, name):
    value = metrics.get(name)
    return 0.0 if value is None or not math.isfinite(float(value)) else float(value)


def select_validation_operating_point(candidates, pixel_tolerance_px=0, selection_range_mae_max=0.25):
    ranked = []
    for candidate_name, model, pos_weight, scores, dataset, fx in candidates:
        for threshold in THRESHOLDS:
            metrics = detector_metrics(
                scores, dataset, threshold, 2, fx, pixel_tolerance_px=pixel_tolerance_px
            )
            # Feasibility mirrors EVERY precision-side gate check, not only the FPRs.
            # Mirroring just the FPRs let the selector chase recall down to threshold 0.075,
            # where the mask halo dropped pixel precision to 0.80 and the halo's background
            # depth pushed range MAE to 0.88 m -- an operating point that could never pass the
            # gate it was being selected for. Validation-only; the held-out test stays sealed.
            range_mae = metrics.get("range_mae_m")
            feasible = (
                _metric_or_zero(metrics, "absent_false_positive_rate") <= 0.01
                and _metric_or_zero(metrics, "fully_occluded_false_positive_rate") <= 0.01
                and _metric_or_zero(metrics, "frame_precision") >= 0.98
                and _metric_or_zero(metrics, "pixel_precision") >= 0.95
                and range_mae is not None
                and math.isfinite(float(range_mae))
                and float(range_mae) <= selection_range_mae_max
            )
            worst_recall = min(
                _metric_or_zero(metrics, "frame_recall"),
                _metric_or_zero(metrics, "far_14_20m_recall"),
                _metric_or_zero(metrics, "partial_occlusion_recall"),
                _metric_or_zero(metrics, "small_target_recall"),
            )
            rank = (
                int(feasible),
                worst_recall,
                _metric_or_zero(metrics, "frame_f1"),
                _metric_or_zero(metrics, "pixel_iou"),
                -abs(threshold - 0.55),
            )
            ranked.append((rank, candidate_name, model, pos_weight, metrics))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0], [
        {"candidate": item[1], "rank": list(item[0]), "metrics": item[4]}
        for item in ranked[:10]
    ]


def gate_decision(metrics):
    checks = {
        "frame_recall>=0.95": _metric_or_zero(metrics, "frame_recall") >= 0.95,
        "frame_precision>=0.98": _metric_or_zero(metrics, "frame_precision") >= 0.98,
        "absent_fpr<=0.01": _metric_or_zero(metrics, "absent_false_positive_rate") <= 0.01,
        "full_occlusion_fpr<=0.01": _metric_or_zero(metrics, "fully_occluded_false_positive_rate") <= 0.01,
        "far_recall>=0.85": _metric_or_zero(metrics, "far_14_20m_recall") >= 0.85,
        "partial_recall>=0.85": _metric_or_zero(metrics, "partial_occlusion_recall") >= 0.85,
        "small_recall>=0.80": _metric_or_zero(metrics, "small_target_recall") >= 0.80,
        "pixel_precision>=0.95": _metric_or_zero(metrics, "pixel_precision") >= 0.95,
        "bearing_mae<=1.5deg": float(metrics["bearing_mae_deg"]) <= 1.5,
        "range_mae<=0.25m": float(metrics["range_mae_m"]) <= 0.25,
        "test_absent_frames>=500": int(metrics["absent_frames"]) >= 500,
        "test_full_occlusion_frames>=100": int(metrics["fully_occluded_frames"]) >= 100,
        "test_partial_frames>=50": int(metrics["partial_occlusion_frames"]) >= 50,
        "test_small_frames>=100": int(metrics["small_target_frames"]) >= 100,
    }
    return checks, all(checks.values())


def dataset_summary(dataset):
    counts = dataset["mask"].sum(dim=(1, 2))
    return {
        "name": dataset["name"],
        "seed": dataset["seed"],
        "frames": int(counts.numel()),
        "positive_pixel_rate": float(dataset["mask"].float().mean()),
        "visible_frames": int((counts >= 2).sum()),
        "absent_frames": int((~dataset["planned_present"]).sum()),
        "fully_occluded_frames": int((dataset["planned_present"] & (counts < 2)).sum()),
        "forced_partial_frames": int((dataset["forced_occlusion"] & (counts >= 2)).sum()),
        "small_target_frames": int(((counts >= 2) & (counts <= 5)).sum()),
    }


def main():
    args = parse_args()
    validate_args(args)
    if args.preflight:
        print(
            "[detector-v2] PREFLIGHT PASS | train/val/test=%d/%d/%d seeds=%d/%d/%d "
            "candidates=pixel/spatial x bce/focal thresholds=%d"
            % (
                args.train_frames,
                args.validation_frames,
                args.test_frames,
                TRAIN_SEED,
                VALIDATION_SEED,
                TEST_SEED,
                len(THRESHOLDS),
            )
        )
        return

    # Isaac Gym must be imported before torch, but preflight deliberately avoids both so it can
    # validate the frozen campaign contract without initializing a GPU or JIT extension cache.
    # Its legacy gymutil parser otherwise re-parses this tool's already-consumed arguments.
    sys.argv = [sys.argv[0]]
    global task_registry, torch, F, AppearanceTargetSegmenter, SpatialTargetSegmenter, SpatialTargetSegmenterWide, quat_rotate
    from aerial_gym.registry.task_registry import task_registry
    import torch
    import torch.nn.functional as F
    from aerial_gym.task.navrl_task.navrl_perception import (
        AppearanceTargetSegmenter,
        SpatialTargetSegmenter,
        SpatialTargetSegmenterWide,
    )
    from aerial_gym.utils.math import quat_rotate

    task = task_registry.make_task(
        "navrl_task", seed=TRAIN_SEED, num_envs=args.num_envs, headless=True, use_warp=True
    )
    task.tm.speed_fixed = -1.0
    task.tm.pattern = "mixed"
    max_range = float(task.vis_cfg.detector_max_range)
    fx = task.detector.fx
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = collect_split(task, args.train_frames, args.train_seed, "train")
    validation = collect_split(task, args.validation_frames, args.validation_seed, "validation")

    candidates = []
    candidate_receipts = []
    # Gate 3 escalation: the spatial CNN joins the candidate pool ONLY because the 1x1 head
    # failed the offline gate under the appearance envelope (pixel precision 0.17); selection
    # stays validation-only, so a nominal-appearance run can still pick the 1x1 head.
    # v7 pool: the 1x1 head is EXCLUDED -- v3 settled that it is impossible under the envelope
    # (pixel precision 0.17), so training it again spends GPU on a settled negative.
    combos = [
        ("spatial_cnn", "balanced_bce"),
        ("spatial_cnn", "focal_dice"),
        ("spatial_cnn_wide", "balanced_bce"),
        ("spatial_cnn_wide", "focal_dice"),
    ]
    for index, (arch, kind) in enumerate(combos):
        model, pos_weight = train_candidate(
            train, kind, args.epochs, args.batch_size, max_range, device,
            args.train_seed + index, arch=arch,
        )
        scores = infer_scores(model, validation, args.batch_size, max_range, device)
        name = f"{arch}+{kind}"
        candidates.append((name, model, pos_weight, scores, validation, fx))
        candidate_receipts.append({"candidate": name, "pos_weight": pos_weight})

    selected, validation_top10 = select_validation_operating_point(
        candidates,
        pixel_tolerance_px=args.pixel_tolerance_px,
        selection_range_mae_max=args.selection_range_mae_max,
    )
    _, selected_name, selected_model, pos_weight, validation_metrics = selected
    selected_threshold = float(validation_metrics["threshold"])
    print(
        "[detector-v2] selected candidate=%s threshold=%.3f val_frame_f1=%.4f val_far=%.4f"
        % (
            selected_name,
            selected_threshold,
            validation_metrics["frame_f1"],
            validation_metrics["far_14_20m_recall"],
        )
    )

    # Test is generated and touched only after candidate/loss/threshold selection is frozen.
    test = collect_split(task, args.test_frames, args.test_seed, "test")
    test_scores = infer_scores(selected_model, test, args.batch_size, max_range, device)
    test_metrics = detector_metrics(
        test_scores, test, selected_threshold, 2, fx,
        pixel_tolerance_px=args.pixel_tolerance_px,
    )
    checks, passed = gate_decision(test_metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=False)
    created = datetime.now(timezone.utc).isoformat()
    payload = {
        "model": selected_model.state_dict(),
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": created,
            "architecture": (
                "SpatialTargetSegmenterWide/cnn9x9x24-RGBD"
                if selected_name.startswith("spatial_cnn_wide")
                else "SpatialTargetSegmenter/cnn7x7-RGBD"
                if selected_name.startswith("spatial_cnn")
                else "AppearanceTargetSegmenter/1x1-RGBD"
            ),
            "selected_loss": selected_name,
            "selected_threshold": selected_threshold,
            "min_pixels": 2,
            "max_range_m": max_range,
            "train_seed": args.train_seed,
            "validation_seed": args.validation_seed,
            "test_seed": args.test_seed,
            "pixel_tolerance_px": args.pixel_tolerance_px,
            "selection_range_mae_max": args.selection_range_mae_max,
            "gate_passed": passed,
        },
    }
    torch.save(payload, args.output)
    artifact_sha = sha256_file(args.output)
    script_path = Path(__file__).resolve()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "campaign": "navrl_detector_offline_gate_v2",
        "created_at_utc": created,
        "artifact": str(args.output.resolve()),
        "artifact_sha256": artifact_sha,
        # Appearance domain-randomisation envelope active during COLLECTION (검증 2 stage B).
        # All-zero = the nominal pure-red dataset of the original v2 artifact.
        "appearance_envelope": {
            name: float(os.environ.get(name, 0.0))
            for name in (
                "NAVRL_APP_HUE_DEG", "NAVRL_APP_LIGHT_GAIN", "NAVRL_APP_ALBEDO_JITTER",
                "NAVRL_APP_TEXTURE_STD", "NAVRL_APP_MOTION_BLUR",
                "NAVRL_CAM_MOUNT_ROT_DEG", "NAVRL_CAM_MOUNT_TRANS_M",
                "NAVRL_CAM_FOV_SCALE_ERR",
            )
        },
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "selected_candidate": selected_name,
        "selected_threshold": selected_threshold,
        "selected_pos_weight": pos_weight,
        "datasets": [dataset_summary(train), dataset_summary(validation), dataset_summary(test)],
        "candidate_receipts": candidate_receipts,
        "target_batches_per_layout": TARGET_BATCHES_PER_LAYOUT,
        "validation_metrics": validation_metrics,
        "validation_top10": validation_top10,
        "test_metrics": test_metrics,
        "gate_checks": checks,
        "gate_passed": passed,
        "next_step": (
            "frozen-policy analytic-vs-learned navigation A/B"
            if passed
            else "do not run navigation evaluation; revise detector architecture/data"
        ),
    }
    (args.result_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# NavRL detector offline gate v2",
        "",
        "- candidate: `%s`" % selected_name,
        "- validation-selected threshold: `%.3f`" % selected_threshold,
        "- artifact SHA-256: `%s`" % artifact_sha,
        "- held-out test decision: **%s**" % ("PASS" if passed else "FAIL"),
        "",
        "| metric | held-out test |",
        "|---|---:|",
        "| frame precision | %.4f |" % test_metrics["frame_precision"],
        "| frame recall | %.4f |" % test_metrics["frame_recall"],
        "| absent FPR | %.4f |" % test_metrics["absent_false_positive_rate"],
        "| full-occlusion FPR | %.4f |" % test_metrics["fully_occluded_false_positive_rate"],
        "| partial-occlusion recall | %.4f |" % test_metrics["partial_occlusion_recall"],
        "| small-target recall | %.4f |" % test_metrics["small_target_recall"],
        "| far 14-20 m recall | %.4f |" % test_metrics["far_14_20m_recall"],
        "| pixel precision / IoU | %.4f / %.4f |"
        % (test_metrics["pixel_precision"], test_metrics["pixel_iou"]),
        "| bearing MAE | %.3f deg |" % test_metrics["bearing_mae_deg"],
        "| range MAE | %.3f m |" % test_metrics["range_mae_m"],
        "",
        "## Gate checks",
        "",
    ]
    lines.extend("- [%s] %s" % ("x" if value else " ", name) for name, value in checks.items())
    (args.result_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "artifact": str(args.output.resolve()),
        "artifact_sha256": artifact_sha,
        "result_summary": str((args.result_root / "summary.json").resolve()),
        "selected_loss": selected_name,
        "selected_threshold": selected_threshold,
        "train_frames": args.train_frames,
        "validation_frames": args.validation_frames,
        "test_frames": args.test_frames,
        "train_seed": args.train_seed,
        "validation_seed": args.validation_seed,
        "test_seed": args.test_seed,
        "gate_passed": passed,
    }
    args.output.with_suffix(".receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("[detector-v2] GATE %s -> %s" % ("PASS" if passed else "FAIL", args.result_root / "summary.md"))


if __name__ == "__main__":
    main()
