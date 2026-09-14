"""Corridor (free-gap) tokens from the fused horizontal LiDAR profile.

Motivation (WORKLOG 2026-07-31): at 100 bars the obstacle-token pipeline is no longer the
bottleneck -- barprobe measured unique=4.9/8, hit_token_given_fov=0.839, duplicate=0.2 -- yet
~95.5% of failures were still bar contact. The policy sees WHERE the obstacle surfaces are but
must implicitly infer WHERE IT CAN PASS. A corridor token answers the affordance question
directly: for each free angular gap between observed surfaces it reports the gap's center
bearing, usable metric width, bounding-surface clearances, and clear depth.

This module is deliberately pure (torch + math only, no simulator imports) so its geometry can
be unit-tested on CPU and validated on GPU against ground-truth bar layouts by
tools/probe_corridor_geometry.py before any policy consumes it.

Frame conventions match navrl_perception: bearings come from lidar_bin_bearings() and DECREASE
with the bin index. The FOV window (away from the +/-pi seam) therefore maps to a contiguous
index range, which this extractor relies on.
"""

import math

import torch

# Feature layout per corridor slot. Fixed (not env-tunable): the network input projection is
# sized from it, so changing it is a schema change requiring an explicit warm-start expansion.
#   [ sin(center), cos(center), width_m/R, left_clear/R, right_clear/R,
#     clear_depth/R, angular_width/pi, valid ]
CORRIDOR_DIM = 8


def extract_corridor_tokens(
    nearest,
    bearings,
    *,
    max_range,
    num_corridors,
    fov_deg,
    horizon_m,
    min_width_m,
):
    """Extract up to ``num_corridors`` free-gap tokens from a horizontal range profile.

    ``nearest``: [batch, beams] fused min-over-vertical ranges in meters (no return = max_range).
    ``bearings``: [beams] body-frame bearings in radians, DECREASING with index.

    A bin is *blocking* when its return is closer than ``horizon_m``; a corridor is a maximal
    contiguous run of non-blocking bins inside the FOV. Per corridor:
      center   -- mid-bearing of the run
      width_m  -- metric chord between the two bounding surface endpoints (interior gaps), or
                  2*sin(ang/2)*min(depth, horizon) at the FOV edge where one bound is missing
      left/right clear -- range of the bounding blocked bins (max_range when at the FOV edge);
                  "left" is the larger-bearing side (lower bin index)
      depth    -- min range inside the run (>= horizon by construction, <= max_range)
    Corridors narrower than ``min_width_m`` are dropped. The widest ``num_corridors`` gaps are
    kept and returned sorted by |center| ascending (forward-first slot semantics).

    Returns (tokens [batch, num_corridors, CORRIDOR_DIM], aux dict of raw per-slot tensors).
    Invalid slots are all-zero (valid flag 0).
    """
    if nearest.ndim != 2 or bearings.ndim != 1 or nearest.shape[1] != bearings.shape[0]:
        raise ValueError("nearest must be [batch, beams] and bearings must be [beams]")
    if num_corridors <= 0:
        raise ValueError("num_corridors must be positive")
    if not 0.0 < float(fov_deg) <= 360.0:
        raise ValueError("fov_deg must be in (0, 360]")
    if not math.isfinite(float(horizon_m)) or float(horizon_m) <= 0.0:
        raise ValueError("horizon_m must be finite and positive")
    if not math.isfinite(float(min_width_m)) or float(min_width_m) < 0.0:
        raise ValueError("min_width_m must be finite and non-negative")

    device = nearest.device
    batch, beams = nearest.shape
    max_range = float(max_range)
    horizon = min(float(horizon_m), max_range)
    half_fov = math.radians(float(fov_deg) * 0.5)
    bin_rad = 2.0 * math.pi / beams

    # FOV window: with the decreasing-bearing convention this is one contiguous index run.
    if float(fov_deg) < 359.9:
        in_fov = bearings.abs() <= half_fov + 1e-7
    else:
        in_fov = torch.ones_like(bearings, dtype=torch.bool)
    fov_idx = torch.nonzero(in_fov, as_tuple=False).squeeze(1)
    if fov_idx.numel() == 0:
        raise ValueError("FOV window selected no beams")
    if not bool((fov_idx[1:] - fov_idx[:-1] == 1).all()):
        raise ValueError("FOV window must be contiguous (keep it away from the +/-pi seam)")
    sub = nearest[:, fov_idx]                       # [batch, m]
    sub_bear = bearings[fov_idx]                    # [m], decreasing
    m = sub.shape[1]

    open_ = sub >= horizon                          # non-blocking bins
    # Run labelling (same cumsum trick as the cluster selector): id 0 = blocked, 1..m = open runs.
    new_run = open_.clone()
    new_run[:, 1:] &= ~open_[:, :-1]
    run_id = torch.cumsum(new_run.long(), dim=1) * open_.long()

    slots = m + 1
    arange = torch.arange(m, device=device).view(1, m).expand(batch, m)
    inf = torch.full((batch, slots), float("inf"), device=device)
    depth_run = inf.clone().scatter_reduce(
        1, run_id, torch.where(open_, sub, torch.full_like(sub, float("inf"))),
        reduce="amin", include_self=True,
    )
    first_bin = torch.full((batch, slots), m, device=device, dtype=torch.long).scatter_reduce(
        1, run_id, torch.where(open_, arange, torch.full_like(arange, m)),
        reduce="amin", include_self=True,
    )
    last_bin = torch.full((batch, slots), -1, device=device, dtype=torch.long).scatter_reduce(
        1, run_id, torch.where(open_, arange, torch.full_like(arange, -1)),
        reduce="amax", include_self=True,
    )
    run_exists = (last_bin >= 0) & (first_bin < m)
    run_exists[:, 0] = False                        # id 0 is the blocked-bin bucket

    fb = first_bin.clamp(0, m - 1)
    lb = last_bin.clamp(0, m - 1)
    b_first = sub_bear[fb]                          # larger bearing (left side)
    b_last = sub_bear[lb]                           # smaller bearing (right side)
    center = 0.5 * (b_first + b_last)
    ang_width = (b_first - b_last) + bin_rad

    has_left = first_bin > 0
    has_right = last_bin < (m - 1)
    left_idx = (first_bin - 1).clamp(0, m - 1)
    right_idx = (last_bin + 1).clamp(0, m - 1)
    rows = torch.arange(batch, device=device).view(batch, 1)
    r_left = torch.where(has_left, sub[rows, left_idx], torch.full_like(center, max_range))
    r_right = torch.where(has_right, sub[rows, right_idx], torch.full_like(center, max_range))
    depth = depth_run.clamp(max=max_range)

    # Metric width. Interior gap: chord between the two bounding surface endpoints. Edge gap
    # (one bound missing): arc opening at the shallower of (depth, horizon) -- conservative.
    bl = sub_bear[left_idx]
    br = sub_bear[right_idx]
    plx = r_left * torch.cos(bl)
    ply = r_left * torch.sin(bl)
    prx = r_right * torch.cos(br)
    pry = r_right * torch.sin(br)
    chord = torch.sqrt((plx - prx).square() + (ply - pry).square())
    edge_ref = torch.minimum(depth, torch.full_like(depth, horizon))
    edge_width = 2.0 * edge_ref * torch.sin((ang_width * 0.5).clamp(max=math.pi / 2.0))
    width = torch.where(has_left & has_right, chord, edge_width)
    width = width.clamp(0.0, 2.0 * max_range)

    valid = run_exists & (width >= float(min_width_m)) & torch.isfinite(depth)

    # Keep the widest K gaps, then order the kept slots forward-first (|center| ascending).
    k = int(num_corridors)
    score = width.masked_fill(~valid, -1.0)
    take = min(k, slots)
    top_score, top_idx = score.topk(take, dim=1)
    if take < k:  # more slots requested than bins+1 -- pad with invalid
        pad = k - take
        top_idx = torch.cat([top_idx, top_idx[:, :1].expand(batch, pad)], dim=1)
        top_score = torch.cat(
            [top_score, torch.full((batch, pad), -1.0, device=device)], dim=1
        )
    sel_valid = top_score > 0.0

    def pick(t):
        return t.gather(1, top_idx)

    c_center = pick(center)
    order = c_center.abs().masked_fill(~sel_valid, float("inf")).argsort(dim=1)

    def ordered(t):
        return pick(t).gather(1, order)

    sel_valid = sel_valid.gather(1, order)
    c_center = c_center.gather(1, order)
    c_width = ordered(width)
    c_left = ordered(r_left)
    c_right = ordered(r_right)
    c_depth = ordered(depth)
    c_ang = ordered(ang_width)

    vf = sel_valid.float()
    tokens = torch.stack(
        [
            torch.sin(c_center),
            torch.cos(c_center),
            (c_width / max_range).clamp(0.0, 1.0),
            (c_left / max_range).clamp(0.0, 1.0),
            (c_right / max_range).clamp(0.0, 1.0),
            (c_depth / max_range).clamp(0.0, 1.0),
            (c_ang / math.pi).clamp(0.0, 1.0),
            torch.ones_like(c_center),
        ],
        dim=2,
    ) * vf.unsqueeze(2)

    aux = {
        "center": c_center * vf,
        "width_m": c_width * vf,
        "left_clear_m": c_left * vf,
        "right_clear_m": c_right * vf,
        "depth_m": c_depth * vf,
        "ang_width": c_ang * vf,
        "valid": sel_valid,
    }
    return tokens, aux
