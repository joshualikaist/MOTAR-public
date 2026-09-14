"""Star-convex free-space polytope from a single LiDAR scan (A3, shadow-mode only).

Reference: Xu, Wang, Gao et al., "Star-Convex Constrained Optimization for Visibility Planning
with Application to Aerial Inspection", ICRA 2022 (arXiv:2204.04393). We reuse their ball-flip
construction, but for a different purpose: they certify VISIBILITY, we ask whether a commanded
direction leaves the region that the scan actually resolved. The set-theoretic object is the same.

Why this shape, for our failure. The contact forensics measured 57-58% of bar contacts LATERAL to
a 0.45 m half-width corridor and another 20% on rays that returned nothing, i.e. 77-78% outside
what the governor examines at all (docs/prereg_2026-09-04_contact_corridor_forensics.md). Two
properties of the star-convex construction bear on exactly those two numbers:

  * every bearing contributes, so there is no corridor to fall beside;
  * unreturned rays are filled with points on the max-range sphere, so unobserved space bounds the
    region instead of being treated as free.

Star-convexity is about the sensor origin, which is what lets a candidate VELOCITY DIRECTION be
tested in closed form rather than by rolling out a trajectory.

This module is deliberately geometry-only and Isaac-free so the contract is CPU-testable, and it
is used in shadow mode only: nothing here reaches the actor, the critic, the reward, or any
termination.
"""

import math

import torch


# Frozen with the A3 preregistration. Sphere-boundary augmentation is what converts "no return"
# from free space into a bound, so its radius is the sensing horizon itself.
FLIP_EPS = 1e-6
# Frozen by prereg_2026-09-05_a3_star_convex_shadow section 2: equal to the LiDAR horizontal
# beam spacing. Narrower would look between beams; wider degenerates toward omnidirectional.
SC_CONE_HALF_ANGLE_RAD = math.radians(5.0)


def flip_points(points, radius):
    """Ball flip about the origin: x_hat = x * (2r - |x|) / |x|.

    The norm becomes |x_hat| = 2r - |x|, so the flip REVERSES radial order about the sphere: the
    nearest returns land farthest out and therefore end up on the convex hull of the flipped cloud,
    which is what makes that hull the set of directly visible points. Points at |x| = r are fixed,
    which is why max-range sphere samples act as a wall rather than a hole.
    """
    norm = points.norm(dim=-1, keepdim=True).clamp(min=FLIP_EPS)
    return points * ((2.0 * radius - norm) / norm)


def scan_to_points(ranges_m, bearings_rad, elevations_rad, max_range_m):
    """[B,V,H] slant ranges -> [B,N,3] Cartesian points in the sensor frame.

    A ray that returned nothing is NOT dropped: it is placed on the max-range sphere. That single
    choice is the difference between "unknown is free" (the live governor) and "unknown bounds the
    region" (this module).
    """
    b, v, h = ranges_m.shape
    valid = torch.isfinite(ranges_m) & (ranges_m > 0.0) & (ranges_m < max_range_m * 0.995)
    r = torch.where(valid, ranges_m, torch.full_like(ranges_m, max_range_m))
    az = bearings_rad.view(1, 1, h)
    el = elevations_rad.view(1, v, 1)
    cos_el = torch.cos(el)
    x = r * cos_el * torch.cos(az)
    y = r * cos_el * torch.sin(az)
    z = r * torch.sin(el).expand_as(r)
    return torch.stack([x, y, z], dim=-1).reshape(b, v * h, 3), valid.reshape(b, v * h)


def direction_clearance(points, direction_xy, max_range_m, cone_half_angle_rad):
    """Nearest flipped-hull bound along a commanded direction, evaluated for every bearing.

    A full convex hull per environment per step is not affordable inside the sim loop, so we use
    the property that matters here: for a star-convex region about the origin, the bound along a
    direction is set by the points whose bearing is closest to it. We take the minimum range over
    a narrow cone about the commanded direction -- with the cone spanning the whole sphere as the
    angle grows, this degenerates to the omnidirectional nearest return, which is exactly the
    `omni` baseline A4 needs.

    Returns clearance in metres, [B].
    """
    if direction_xy.ndim != 2 or direction_xy.shape[1] != 2:
        raise ValueError("direction_xy must be [batch, 2]")
    d = torch.nn.functional.normalize(
        torch.cat([direction_xy, torch.zeros_like(direction_xy[:, :1])], dim=1), dim=1
    )
    rng = points.norm(dim=-1).clamp(min=FLIP_EPS)
    unit = points / rng.unsqueeze(-1)
    cos = (unit * d.unsqueeze(1)).sum(-1)
    inside = cos >= math.cos(cone_half_angle_rad)
    big = torch.full_like(rng, float(max_range_m))
    return torch.where(inside, rng, big).amin(dim=1).clamp(0.0, float(max_range_m))


def point_in_star_region(points, query_xyz, max_range_m, cone_half_angle_rad):
    """Is `query_xyz` inside the star-convex region carved by the scan?

    True iff the query is nearer than the nearest scan point in its own bearing cone. This is the
    O(K) half-space style test the ICRA-2022 paper gets from its hull; we evaluate it directly
    against the cloud because we only ever ask about one direction at a time.
    """
    q = query_xyz
    qn = q.norm(dim=-1).clamp(min=FLIP_EPS)
    bound = direction_clearance(points, q[:, 0:2], max_range_m, cone_half_angle_rad)
    return qn <= bound
