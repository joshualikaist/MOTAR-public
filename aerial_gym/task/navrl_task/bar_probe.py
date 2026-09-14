"""Geometry helpers for the evaluation-only NavRL bar-contact probe.

Obstacle tokens store LiDAR surface points, not simulator obstacle centers.  A bearing-only match
therefore cannot tell apart two bars on the same ray, and treating surface-to-center distance as a
position error is misleading.  This module associates each token ray with a geometrically plausible
ground-truth bar and exposes the separate lateral and radial components.
"""

from __future__ import annotations

import math
from typing import Dict

import torch


# Generated NavRL bars are at most 0.8 x 0.8 m.  A surface point can therefore be at most the
# half-diagonal from the center, regardless of the bar's yaw.
MAX_BAR_CENTER_TO_SURFACE_M = math.hypot(0.4, 0.4)
DEFAULT_ASSOCIATION_SLACK_M = 0.15


def associate_surface_tokens_to_bars(
    token_positions_xy: torch.Tensor,
    token_valid: torch.Tensor,
    bar_centers_xy: torch.Tensor,
    *,
    max_center_to_surface_m: float = MAX_BAR_CENTER_TO_SURFACE_M,
    association_slack_m: float = DEFAULT_ASSOCIATION_SLACK_M,
) -> Dict[str, torch.Tensor]:
    """Associate token surface points to plausible GT bar centers.

    Args:
        token_positions_xy: ``(N, T, 2)`` current-frame LiDAR surface points.
        token_valid: ``(N, T)`` validity mask.
        bar_centers_xy: ``(N, B, 2)`` current-frame GT bar centers.

    A token can belong to a bar only when its ray passes through the bar's maximum bounding circle
    and its measured surface lies no farther than that circle in front of the center.  This rejects
    the old failure mode where a nearer bar token was matched to a farther bar solely because they
    shared a bearing.
    """
    if token_positions_xy.ndim != 3 or token_positions_xy.shape[-1] != 2:
        raise ValueError("token_positions_xy must have shape (N, T, 2)")
    if token_valid.shape != token_positions_xy.shape[:2]:
        raise ValueError("token_valid must have shape (N, T)")
    if bar_centers_xy.ndim != 3 or bar_centers_xy.shape[0] != token_positions_xy.shape[0]:
        raise ValueError("bar_centers_xy must have shape (N, B, 2)")
    if bar_centers_xy.shape[-1] != 2 or bar_centers_xy.shape[1] == 0:
        raise ValueError("bar_centers_xy must contain at least one 2-D bar center")

    max_surface = float(max_center_to_surface_m)
    slack = float(association_slack_m)
    if not math.isfinite(max_surface) or max_surface <= 0.0:
        raise ValueError("max_center_to_surface_m must be finite and positive")
    if not math.isfinite(slack) or slack < 0.0:
        raise ValueError("association_slack_m must be finite and non-negative")

    token_range = token_positions_xy.norm(dim=2)
    ray = token_positions_xy / token_range.clamp(min=1e-6).unsqueeze(2)
    ray_e = ray.unsqueeze(2)
    bars_e = bar_centers_xy.unsqueeze(1)

    along = (ray_e * bars_e).sum(dim=3)
    cross_track = (ray_e[..., 0] * bars_e[..., 1] - ray_e[..., 1] * bars_e[..., 0]).abs()
    radial_gap = along - token_range.unsqueeze(2)
    center_offset = torch.sqrt(cross_track.square() + radial_gap.square())

    gate = max_surface + slack
    plausible = (
        token_valid.unsqueeze(2)
        & (token_range.unsqueeze(2) > 1e-6)
        & (along > 0.0)
        & (cross_track <= gate)
        & (radial_gap >= -slack)
        & (radial_gap <= gate)
    )
    inf = torch.full_like(center_offset, float("inf"))
    best_offset, bar_index = torch.where(plausible, center_offset, inf).min(dim=2)
    associated = torch.isfinite(best_offset)

    gather_index = bar_index.unsqueeze(2)
    best_cross_track = cross_track.gather(2, gather_index).squeeze(2)
    best_radial_gap = radial_gap.gather(2, gather_index).squeeze(2)
    best_cross_track = torch.where(
        associated, best_cross_track, torch.full_like(best_cross_track, float("nan"))
    )
    best_radial_gap = torch.where(
        associated, best_radial_gap, torch.full_like(best_radial_gap, float("nan"))
    )

    return {
        "associated": associated,
        "bar_index": bar_index,
        "center_offset": best_offset,
        "cross_track": best_cross_track,
        "radial_gap": best_radial_gap,
    }
