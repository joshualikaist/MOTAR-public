#!/usr/bin/env python3
"""Offline detector brittleness under appearance domain shift (검증 2, stage A).

For each shift axis at several severities, render labelled frames on the live 205-bar scene and
measure BOTH detector heads on identical frames:
  - the analytic bootstrap (fixed +3R-2G-2B-0.9 red rule), and
  - the learned v2 checkpoint (offline-gate PASS artifact, SHA-pinned).

Per cell: frame recall on GT-visible targets, false-positive rate on absent/occluded frames,
pixel precision, and bearing MAE computed with the PERCEPTION module's nominal intrinsics -- so
the calibration axes (mount/FOV) surface exactly as a real back-projection bias would.

Renderer-private target_mask is offline supervision only; nothing here touches the actor path.
No PPO is loaded. Results: results/navrl_detector_domain_shift/summary.{md,json}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import types
from pathlib import Path

os.environ.setdefault("NAVRL_VISION", "1")
os.environ.setdefault("NAVRL_PERCEPTION", "1")
os.environ.setdefault("NAVRL_GENERAL_TRAIN", "1")
os.environ.setdefault("NAVRL_NUM_BARS", "205")
os.environ.setdefault("NAVRL_FIXED_BARS", "205")
os.environ.setdefault("NAVRL_MAX_BARS", "300")
os.environ.setdefault("NAVRL_DENSITY_CURRICULUM", "0")
os.environ.setdefault("NAVRL_LIDAR_RANGE", "12")
os.environ.setdefault("NAVRL_LIDAR_HBEAMS", "72")
os.environ.setdefault("NAVRL_LIDAR_VBEAMS", "4")
os.environ.setdefault("NAVRL_MAX_OBSTACLES", "8")
os.environ.setdefault("NAVRL_OBSTACLE_SELECTOR", "cluster_sector")
os.environ.setdefault("NAVRL_OBSTACLE_FOV_DEG", "240")
os.environ.setdefault("NAVRL_PERCEPTION_PERTURB", "0")

from aerial_gym.registry.task_registry import task_registry  # isaacgym before torch
import torch

from aerial_gym.task.navrl_task.navrl_detector import NavRLTargetDetector
from aerial_gym.task.navrl_task.navrl_perception import (
    AppearanceTargetSegmenter,
    build_target_segmenter,
)
from aerial_gym.utils.math import quat_rotate, quat_rotate_inverse

DEFAULT_ARTIFACT = Path("artifacts/navrl_target_detector_v2.pth")
DEFAULT_SHA = "8da32d6f21bfbd3bdd5ec5de9ef9cb09e8deb4bd5ce511630e19afee33f26f10"
THRESHOLD = 0.55       # validation-selected in the offline gate; runtime default
MIN_PIXELS = 2
MAX_RANGE = 20.0

# Severity ladders. The first entry of every axis list is the nominal control (0), evaluated once.
AXES = {
    "hue_deg": [15.0, 30.0, 60.0, 90.0, 120.0, 180.0],
    "light_gain": [0.15, 0.3, 0.5, 0.7],
    "albedo_jitter": [0.15, 0.3, 0.5],
    "texture_std": [0.1, 0.2, 0.4],
    "motion_blur": [0.3, 0.6, 0.8],
    "mount_rot_deg": [1.0, 3.0, 5.0],
    "mount_trans_m": [0.02, 0.05],
    "fov_scale_err": [0.02, 0.05, 0.10],
}
KNOB_OF = {
    "hue_deg": "appearance_hue_deg",
    "light_gain": "appearance_light_gain",
    "albedo_jitter": "appearance_albedo_jitter",
    "texture_std": "appearance_texture_std",
    "motion_blur": "appearance_motion_blur",
    "mount_rot_deg": "camera_mount_rot_deg",
    "mount_trans_m": "camera_mount_trans_m",
    "fov_scale_err": "camera_fov_scale_err",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=103)  # unused elsewhere in the project
    parser.add_argument("--output", type=Path, default=Path("results/navrl_detector_domain_shift"))
    parser.add_argument("--learned-artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--learned-sha", type=str, default=DEFAULT_SHA)
    return parser.parse_args()


def sample_targets(task, generator):
    """Synthetic camera-frame target poses: full FOV coverage, 2-20 m, 20% absent."""
    n = task.num_envs
    device = task.device
    pos = task.obs_dict["robot_position"]
    quat = task.obs_dict["robot_vehicle_orientation"]
    hfov = math.radians(float(task.vis_cfg.detector_hfov_deg)) * 0.5 * 0.9
    vfov = math.radians(float(task.vis_cfg.detector_vfov_deg)) * 0.5 * 0.9
    bearing = (torch.rand(n, device=device, generator=generator) * 2 - 1) * hfov
    elevation = (torch.rand(n, device=device, generator=generator) * 2 - 1) * vfov
    rng = 2.0 + torch.rand(n, device=device, generator=generator) * (MAX_RANGE - 2.0)
    direction = torch.stack(
        [
            torch.cos(elevation) * torch.cos(bearing),
            torch.cos(elevation) * torch.sin(bearing),
            torch.sin(elevation),
        ],
        dim=1,
    )
    absent = torch.rand(n, device=device, generator=generator) < 0.20
    direction[absent] = torch.tensor([-1.0, 0.0, 0.0], device=device)  # behind the camera
    target = pos + quat_rotate(quat, direction * rng.unsqueeze(1))
    return target, absent


def bearing_of(mask, fx, cx):
    """Pixel-centroid bearing via the PERCEPTION module's nominal pinhole model."""
    count = mask.sum(dim=(1, 2)).clamp(min=1).float()
    cols = torch.arange(mask.shape[2], device=mask.device, dtype=torch.float32)
    u = (mask.float() * cols.view(1, 1, -1)).sum(dim=(1, 2)) / count
    return torch.atan2(-(u - cx), torch.full_like(u, fx))


def evaluate_cell(task, detector, models, batches, generator):
    stats = {name: {"tp": 0, "fn": 0, "fp_absent": 0, "absent_n": 0,
                    "pix_tp": 0, "pix_fp": 0, "bear_err": 0.0, "bear_n": 0}
             for name in models}
    fx = detector.fx  # nominal by construction (fov error perturbs only the ray table)
    cx = detector.cx
    pos = task.obs_dict["robot_position"]
    quat = task.obs_dict["robot_vehicle_orientation"]
    all_envs = torch.arange(task.num_envs, device=task.device)
    for _ in range(batches):
        detector._resample_appearance(all_envs)  # fresh per-episode appearance draw
        target, absent = sample_targets(task, generator)
        if detector.app_motion_blur > 0.0:
            # Temporal axis: expose the trail by evaluating the SECOND frame of a moving camera.
            detector.render_raw_rgbd(pos, quat, target)
            fwd = quat_rotate(quat, torch.tensor([[1.0, 0.0, 0.0]], device=task.device).expand(task.num_envs, 3))
            rgb, depth = detector.render_raw_rgbd(pos + 0.23 * fwd, quat, target)
        else:
            rgb, depth = detector.render_raw_rgbd(pos, quat, target)
        gt_mask = detector.target_mask > 0
        gt_visible = gt_mask.any(dim=1).any(dim=1)
        rel = quat_rotate_inverse(quat, target - pos)
        gt_bearing = torch.atan2(rel[:, 1], rel[:, 0])
        for name, model in models.items():
            with torch.no_grad():
                score = model(rgb, depth, MAX_RANGE)
            pred = (score >= THRESHOLD) & (depth < MAX_RANGE)
            frame_hit = pred.sum(dim=(1, 2)) >= MIN_PIXELS
            s = stats[name]
            s["tp"] += int((frame_hit & gt_visible).sum())
            s["fn"] += int((~frame_hit & gt_visible).sum())
            s["fp_absent"] += int((frame_hit & ~gt_visible).sum())
            s["absent_n"] += int((~gt_visible).sum())
            s["pix_tp"] += int((pred & gt_mask).sum())
            s["pix_fp"] += int((pred & ~gt_mask).sum())
            ok = frame_hit & gt_visible
            if bool(ok.any()):
                est = bearing_of(pred & (depth < MAX_RANGE), fx, cx)
                err = torch.atan2(torch.sin(est - gt_bearing), torch.cos(est - gt_bearing)).abs()
                s["bear_err"] += float(err[ok].sum())
                s["bear_n"] += int(ok.sum())
    out = {}
    for name, s in stats.items():
        seen = s["tp"] + s["fn"]
        out[name] = {
            "frame_recall": s["tp"] / max(1, seen),
            "gt_visible_frames": seen,
            "absent_fpr": s["fp_absent"] / max(1, s["absent_n"]),
            "pixel_precision": s["pix_tp"] / max(1, s["pix_tp"] + s["pix_fp"]),
            "bearing_mae_deg": math.degrees(s["bear_err"] / max(1, s["bear_n"])),
        }
    return out


def main():
    args = parse_args()
    digest = hashlib.sha256(args.learned_artifact.read_bytes()).hexdigest()
    if digest != args.learned_sha:
        raise SystemExit(f"learned detector SHA mismatch: {digest}")

    task = task_registry.make_task(
        "navrl_task", seed=args.seed, num_envs=args.num_envs, headless=True, use_warp=True
    )
    task.reset()
    generator = torch.Generator(device=task.device)
    generator.manual_seed(args.seed)

    bootstrap = AppearanceTargetSegmenter().to(task.device).eval()
    payload = torch.load(args.learned_artifact, map_location=task.device)
    architecture = ""
    if isinstance(payload, dict):
        architecture = str((payload.get("meta") or {}).get("architecture", ""))
    learned = build_target_segmenter(architecture).to(task.device).eval()
    learned.load_state_dict(payload.get("model", payload), strict=True)
    models = {"bootstrap": bootstrap, "learned_v2": learned}

    base_cfg = task.vis_cfg

    def make_detector(**overrides):
        cfg = types.SimpleNamespace(
            **{k: getattr(base_cfg, k) for k in dir(base_cfg) if not k.startswith("_")}
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return NavRLTargetDetector(task.sim_env.warp_env, task.num_envs, task.device, cfg, 0.1)

    cells = []
    nominal = evaluate_cell(task, make_detector(), models, args.batches, generator)
    cells.append({"axis": "nominal", "value": 0.0, "models": nominal})
    print(f"[shift] nominal: boot recall={nominal['bootstrap']['frame_recall']:.3f} "
          f"learned recall={nominal['learned_v2']['frame_recall']:.3f}")
    for axis, values in AXES.items():
        for value in values:
            detector = make_detector(**{KNOB_OF[axis]: value})
            result = evaluate_cell(task, detector, models, args.batches, generator)
            cells.append({"axis": axis, "value": value, "models": result})
            b, l = result["bootstrap"], result["learned_v2"]
            print(f"[shift] {axis}={value}: boot {b['frame_recall']:.3f}/{b['absent_fpr']:.3f} "
                  f"learned {l['frame_recall']:.3f}/{l['absent_fpr']:.3f} "
                  f"bearMAE {b['bearing_mae_deg']:.2f}/{l['bearing_mae_deg']:.2f} deg")
            del detector

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": args.seed,
        "num_envs": args.num_envs,
        "batches": args.batches,
        "frames_per_cell": args.num_envs * args.batches,
        "threshold": THRESHOLD,
        "min_pixels": MIN_PIXELS,
        "learned_detector_artifact": str(args.learned_artifact),
        "learned_detector_sha256": args.learned_sha,
        "bars": 205,
        "cells": cells,
        "note": (
            "Offline stage-A brittleness screen. Frame recall over GT-visible synthetic target "
            "poses (full FOV, 2-20 m, 20% absent, natural bar occlusion); absent_fpr counts "
            "detections on frames with no visible target pixel; bearing MAE uses the perception "
            "module's nominal intrinsics, so calibration axes appear as back-projection bias."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["# detector brittleness under appearance domain shift (stage A)", "",
             f"{args.num_envs * args.batches} frames/cell, threshold {THRESHOLD}, 205 bars, seed {args.seed}",
             "",
             "| axis | value | boot recall | boot FPR | boot bear MAE | learned recall | learned FPR | learned bear MAE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in cells:
        b, l = cell["models"]["bootstrap"], cell["models"]["learned_v2"]
        lines.append(
            f"| {cell['axis']} | {cell['value']} | {b['frame_recall']*100:.1f}% | "
            f"{b['absent_fpr']*100:.1f}% | {b['bearing_mae_deg']:.2f}° | "
            f"{l['frame_recall']*100:.1f}% | {l['absent_fpr']*100:.1f}% | "
            f"{l['bearing_mae_deg']:.2f}° |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[shift] wrote {args.output}/summary.md ({len(cells)} cells)")


if __name__ == "__main__":
    main()
