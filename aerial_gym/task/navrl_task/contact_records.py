"""Per-contact and per-frame feature rows for the corridor forensics (plan I1/I2).

Why this exists: every hypothesis in docs/plans/lateral_contact_density_plan_2026-09-05.md
section 2 (speed-slip, gap width, memory, yaw sweep) needs the DISTRIBUTION of a quantity at
contact, and until now the recorder kept only category counts and four running means. These
functions compute one row per contact (and a sparse sample of non-contact frames as the risk
baseline) from tensors the task already has. Pure torch, Isaac-free, evaluation-only: nothing
here reaches the actor, the critic, the reward or any termination.

Conventions match _record_contact_geometry exactly: bearings are in the vehicle frame at the
lookback pose, "lateral" is measured against the COMMANDED direction, no-return is the nearest
bearing's collapsed scan at >= 99.5% of max range, and category priority is
vertical_out > behind > lateral > no_return > in_corridor.
"""

import hashlib
import json
import math
from pathlib import Path

import torch

CATEGORIES = ("vertical_out", "behind", "lateral", "no_return", "in_corridor")
SIDE_SECTOR_DEG = (15.0, 165.0)   # a bar between 15 and 165 deg off the nose counts as "beside"
# Two-sided pinch at the contact instant: a bar surface within the body+tracking inflation radius
# (geometry audit, 0.65 m) on BOTH sides. Verification of the 09-06 grids noted that "the gap was
# wide a second earlier" says nothing about the contact instant; this column answers that.
PINCH_M = 0.65
SCHEMA_VERSION = 1


def category_index(vertical_out, behind, lateral, no_return):
    """0..4 per CATEGORIES with the frozen priority; everything else is in_corridor (4)."""
    out = torch.full_like(behind, 4, dtype=torch.long)
    out = torch.where(no_return, torch.full_like(out, 3), out)
    out = torch.where(lateral, torch.full_like(out, 2), out)
    out = torch.where(behind, torch.full_like(out, 1), out)
    out = torch.where(vertical_out, torch.full_like(out, 0), out)
    return out


def side_gaps(bar_dist, bar_bearing, bar_circumradius, max_range):
    """Nearest bar SURFACE on the left and right beam, per row.

    bar_dist/bar_bearing: [k, b] centre range and bearing (rad, vehicle frame, +left).
    bar_circumradius: [k, b] per-bar XY half-diagonal, subtracted so the number is a surface gap.
    Returns (left, right, nearest_any) in metres, max_range where no bar qualifies.
    """
    lo, hi = math.radians(SIDE_SECTOR_DEG[0]), math.radians(SIDE_SECTOR_DEG[1])
    surface = (bar_dist - bar_circumradius).clamp(min=0.0)
    ang = bar_bearing
    left = (ang >= lo) & (ang <= hi)
    right = (ang <= -lo) & (ang >= -hi)
    fill = torch.full_like(surface, float(max_range))
    left_gap = torch.where(left, surface, fill).amin(dim=1)
    right_gap = torch.where(right, surface, fill).amin(dim=1)
    nearest = surface.amin(dim=1) if surface.shape[1] else fill[:, 0]
    return left_gap.clamp(max=max_range), right_gap.clamp(max=max_range), nearest.clamp(max=max_range)


def memory_window(seen_steps, in_fov_now):
    """H3 signature from a [k, W] boolean 'bar was in range AND inside the selector FOV' matrix
    over the window [t-2.5 s, t-1.0 s] and the same flag at t-1.0 s.

    seen_frac: fraction of window steps the bar was observable.
    seen_then_lost: it was observable at some point in the window and is NOT in FOV at t-1.
    """
    if seen_steps.numel() == 0 or seen_steps.shape[1] == 0:
        z = torch.zeros(in_fov_now.shape[0], device=in_fov_now.device)
        return z, torch.zeros_like(in_fov_now)
    seen_frac = seen_steps.float().mean(dim=1)
    seen_then_lost = seen_steps.any(dim=1) & ~in_fov_now
    return seen_frac, seen_then_lost


def rows_to_jsonl(path, rows, extra=None):
    """Write a list of dicts as JSON lines; return (row_count, sha256). Atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    digest = hashlib.sha256()
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            if extra:
                row = {**extra, **row}
            line = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    tmp.replace(path)
    return len(rows), digest.hexdigest()


def tensor_columns_to_rows(columns):
    """{name: 1-D tensor or list} -> list of dicts, floats rounded to 4 dp, bools/ints native."""
    names = list(columns)
    if not names:
        return []
    n = len(columns[names[0]])
    prepared = {}
    for name in names:
        value = columns[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.dtype == torch.bool:
                prepared[name] = [bool(x) for x in value.tolist()]
            elif value.dtype in (torch.int32, torch.int64):
                prepared[name] = [int(x) for x in value.tolist()]
            else:
                prepared[name] = [None if (isinstance(x, float) and math.isnan(x)) else round(float(x), 4)
                                  for x in value.tolist()]
        else:
            prepared[name] = list(value)
        if len(prepared[name]) != n:
            raise ValueError(f"column {name} has {len(prepared[name])} rows, expected {n}")
    return [{name: prepared[name][i] for name in names} for i in range(n)]
