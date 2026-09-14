#!/usr/bin/env python3
"""Offline bootstrap for the tiny RGB-D target segmenter used by NAVRL perception.

Collects renderer RGB-D plus a renderer-private target mask, then trains
AppearanceTargetSegmenter. The mask never enters the actor path; it is only
used here as offline supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

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
os.environ.setdefault("NAVRL_PERCEPTION_PERTURB", "0")

from aerial_gym.registry.task_registry import task_registry  # isaacgym before torch
import torch
import torch.nn.functional as F

from aerial_gym.task.navrl_task.navrl_perception import AppearanceTargetSegmenter
from aerial_gym.utils.math import quat_rotate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/navrl_target_detector_v1.pth"),
    )
    parser.add_argument("--frames", type=int, default=4096)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def collect_frames(task, *, steps, max_frames):
    act = torch.zeros(task.num_envs, 4, device=task.device)
    rgb_list = []
    depth_list = []
    mask_list = []
    while len(rgb_list) * task.num_envs < max_frames:
        task.reset()
        for _ in range(steps):
            pos = task.obs_dict["robot_position"]
            q_veh = task.obs_dict["robot_vehicle_orientation"]
            fwd_w = quat_rotate(
                q_veh,
                torch.tensor([[1.0, 0.0, 0.0]], device=task.device).expand(task.num_envs, 3),
            )
            distance = torch.empty(task.num_envs, 1, device=task.device).uniform_(2.0, 12.0)
            task.target_position[:] = pos + distance * fwd_w
            task._sync_target_to_sensor()
            task.step(act)
            rgb, depth = task.detector.render_raw_rgbd(pos, q_veh, task.target_position)
            mask = (task.detector.target_mask > 0).float()
            rgb_list.append(rgb.detach().cpu())
            depth_list.append(depth.detach().cpu())
            mask_list.append(mask.detach().cpu())
    rgb = torch.cat(rgb_list, dim=0)[:max_frames]
    depth = torch.cat(depth_list, dim=0)[:max_frames]
    mask = torch.cat(mask_list, dim=0)[:max_frames]
    pos_rate = float(mask.mean())
    if pos_rate <= 0.0:
        raise RuntimeError("detector dataset has zero positive target pixels")
    return rgb, depth, mask


def train_segmenter(rgb, depth, mask, *, max_range, epochs, batch_size, device):
    model = AppearanceTargetSegmenter().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-3)
    n = rgb.shape[0]
    order = torch.arange(n)
    for epoch in range(epochs):
        order = order[torch.randperm(n)]
        total = 0.0
        seen = 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            batch_rgb = rgb[idx].to(device)
            batch_depth = depth[idx].to(device)
            batch_mask = mask[idx].to(device)
            pred = model(batch_rgb, batch_depth, max_range)
            loss = F.binary_cross_entropy(pred, batch_mask)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.item()) * batch_rgb.shape[0]
            seen += batch_rgb.shape[0]
        pos_rate = float(mask.mean())
        print(f"[detector-train] epoch={epoch + 1}/{epochs} loss={total / max(seen, 1):.4f} pos={pos_rate:.3f}")
    return model


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    task = task_registry.make_task(
        "navrl_task",
        seed=args.seed,
        num_envs=args.num_envs,
        headless=True,
        use_warp=True,
    )
    task.tm.speed_fixed = -1.0
    task.tm.pattern = "mixed"
    max_range = float(task.vis_cfg.detector_max_range)

    print(
        f"[detector-train] collecting up to {args.frames} frames from "
        f"{args.num_envs} envs x {args.steps} steps"
    )
    rgb, depth, mask = collect_frames(task, steps=args.steps, max_frames=args.frames)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_segmenter(
        rgb,
        depth,
        mask,
        max_range=max_range,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "frames": int(rgb.shape[0]),
            "seed": args.seed,
            "camera": {
                "width": int(task.vis_cfg.camera_width),
                "height": int(task.vis_cfg.camera_height),
                "max_range": max_range,
            },
        },
    }
    torch.save(payload, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    receipt = {
        "output": str(args.output.resolve()),
        "sha256": digest,
        "frames": int(rgb.shape[0]),
        "positive_pixel_rate": float(mask.mean()),
    }
    receipt_path = args.output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[detector-train] wrote {args.output} sha256={digest}")


if __name__ == "__main__":
    main()
