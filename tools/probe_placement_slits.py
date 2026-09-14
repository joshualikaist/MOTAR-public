"""Measure impassable-slit frequency in the bar placement rule (CPU, no simulator).

Faithfully mirrors AssetManager._random_rejection_xy_spacing (asset_manager.py):
  - uniform candidates in the bar band x=[0.13,0.96]*24, y=[0,24]*[0,1]
  - reject if center distance to any placed bar < min_dist (start 1.5 m)
  - after 128 failed candidate attempts (batches of 32) relax min_dist *= 0.8 -- and the
    relaxed value PERSISTS for all remaining bars of that env (exactly like the GPU code)
Footprints mirror tools/generate_bar_assets.py: per-bar w,d ~ U[0.4, 0.8] axis-aligned boxes.

Reported per density:
  - final relaxed min_dist distribution (how often the guarantee decayed, and to what)
  - surface-gap distribution between nearby bar pairs; impassable slits (gap < 0.40 m =
    drone box diagonal 0.396) and marginal gaps (0.40..0.60 m)
  - drone-accessible free space: fraction of the arena reachable from the spawn strip with
    the drone disk (r=0.20) via 0.1 m grid flood fill, and unreachable-free-space fraction

Also evaluates the PROPOSED NavRL-style rule for comparison: accept a candidate iff its
surface gap to every placed bar is either <= 0 (touching/merged, forms a compound wall) or
>= 0.80 m (comfortably passable); no relaxation of the passable band -- saturation instead
relaxes only the merge acceptance. Usage:
  python tools/probe_placement_slits.py [layouts_per_density]
"""

import sys

import numpy as np

RNG = np.random.default_rng(20260731)

X0, X1 = 0.13 * 24.0, 0.96 * 24.0
Y0, Y1 = 0.0, 24.0
MIN_DIST0 = 1.5
RELAX_AFTER = 128
RELAX = 0.8
BATCH = 32
DRONE_R = 0.20          # half of the 0.28 m box, inflated a little toward its 0.396 diagonal
IMPASS = 0.40           # surface gap below this = the drone box cannot fit at all
MARGINAL = 0.60


def sample_footprints(n):
    return RNG.uniform(0.4, 0.8, size=(n, 2)) * 0.5  # half extents (hx, hy)


def place_current(n):
    """Mirror of _random_rejection_xy_spacing for one env."""
    xs, ys = np.empty(n), np.empty(n)
    xs[0] = RNG.uniform(X0, X1)
    ys[0] = RNG.uniform(Y0, Y1)
    min_dist = MIN_DIST0
    for k in range(1, n):
        attempts = 0
        while True:
            cx = RNG.uniform(X0, X1, size=BATCH)
            cy = RNG.uniform(Y0, Y1, size=BATCH)
            d2 = (xs[:k, None] - cx[None, :]) ** 2 + (ys[:k, None] - cy[None, :]) ** 2
            ok = d2.min(axis=0) >= min_dist * min_dist
            if ok.any():
                j = int(np.argmax(ok))
                xs[k], ys[k] = cx[j], cy[j]
                break
            attempts += BATCH
            if attempts >= RELAX_AFTER:
                min_dist *= RELAX
                attempts = 0
    return xs, ys, min_dist


def box_gap(xs, ys, halves, i, j):
    dx = abs(xs[i] - xs[j]) - (halves[i, 0] + halves[j, 0])
    dy = abs(ys[i] - ys[j]) - (halves[i, 1] + halves[j, 1])
    if dx <= 0.0 and dy <= 0.0:
        return -1.0  # overlapping/touching
    return float(np.hypot(max(dx, 0.0), max(dy, 0.0))) if (dx > 0 and dy > 0) else float(max(dx, dy))


def place_proposed(n, gap_min=0.80):
    """NavRL-style forbidden band on SURFACE gap: accept iff every placed bar is either
    touching (gap <= 0, merge into a wall) or at least gap_min away. Footprint-aware.
    Saturation fallback: if a batch round fails repeatedly, accept the candidate whose
    smallest positive gap is largest (never one inside the forbidden band unless the field
    is truly full, in which case prefer MERGING over slitting)."""
    halves = sample_footprints(n)
    xs, ys = np.empty(n), np.empty(n)
    xs[0] = RNG.uniform(X0, X1)
    ys[0] = RNG.uniform(Y0, Y1)
    for k in range(1, n):
        best_merge = None
        for _ in range(40):  # 40 batches of 32 = 1280 candidates before merge fallback
            cx = RNG.uniform(X0, X1, size=BATCH)
            cy = RNG.uniform(Y0, Y1, size=BATCH)
            dx = np.abs(xs[:k, None] - cx[None, :]) - (halves[:k, 0:1] + halves[k, 0])
            dy = np.abs(ys[:k, None] - cy[None, :]) - (halves[:k, 1:2] + halves[k, 1])
            sep = np.maximum(dx, 0.0) ** 2 + np.maximum(dy, 0.0) ** 2
            gap = np.sqrt(sep)
            touching = (dx <= 0.0) & (dy <= 0.0)
            gap = np.where(touching, -1.0, gap)
            per_cand_ok = ((gap <= 0.0) | (gap >= gap_min)).all(axis=0)
            if per_cand_ok.any():
                j = int(np.argmax(per_cand_ok))
                xs[k], ys[k] = cx[j], cy[j]
                break
            # remember the best merge candidate (most-negative max gap = deepest inside a wall)
            merge_ok = (gap[np.argmin(np.where(gap > 0, gap, np.inf), axis=0), np.arange(BATCH)])
            j = int(np.argmin(np.where(merge_ok > 0, merge_ok, np.inf)))
            if best_merge is None:
                best_merge = (cx[j], cy[j])
        else:
            xs[k], ys[k] = best_merge if best_merge else (RNG.uniform(X0, X1), RNG.uniform(Y0, Y1))
    return xs, ys, halves


def layout_metrics(xs, ys, halves):
    n = len(xs)
    slits = marginal = pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 > 3.0**2:
                continue
            g = box_gap(xs, ys, halves, i, j)
            if g <= 0:
                continue  # merged bars form one wall, not a slit
            pairs += 1
            if g < IMPASS:
                slits += 1
            elif g < MARGINAL:
                marginal += 1
    # flood-fill reachability with the drone disk from the spawn strip (x < 1.5)
    res = 0.1
    gx = np.arange(0.5, 23.5 + res, res)
    gy = np.arange(0.5, 23.5 + res, res)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    free = np.ones_like(GX, dtype=bool)
    for i in range(n):
        dx = np.abs(GX - xs[i]) - (halves[i, 0] + DRONE_R)
        dy = np.abs(GY - ys[i]) - (halves[i, 1] + DRONE_R)
        free &= ~((dx < 0) & (dy < 0))
    from collections import deque

    seen = np.zeros_like(free)
    q = deque()
    for ix in range(min(10, free.shape[0])):  # spawn strip x in [0.5, 1.5]
        for iy in range(free.shape[1]):
            if free[ix, iy] and not seen[ix, iy]:
                seen[ix, iy] = True
                q.append((ix, iy))
    while q:
        ix, iy = q.popleft()
        for nx, ny in ((ix + 1, iy), (ix - 1, iy), (ix, iy + 1), (ix, iy - 1)):
            if 0 <= nx < free.shape[0] and 0 <= ny < free.shape[1] and free[nx, ny] and not seen[nx, ny]:
                seen[nx, ny] = True
                q.append((nx, ny))
    free_frac = free.mean()
    unreachable = (free & ~seen).sum() / max(1, free.sum())
    return slits, marginal, pairs, free_frac, unreachable


def run(density_list, layouts):
    print(f"{'bars':>5} {'rule':>9} | {'relaxed<1.5':>11} {'final_dmin p10':>14} | "
          f"{'slits/layout':>12} {'marginal':>9} | {'free%':>6} {'unreach-free%':>13}")
    for n in density_list:
        for rule in ("current", "proposed"):
            slit_c, marg_c, dmins, freef, unreach = [], [], [], [], []
            for _ in range(layouts):
                if rule == "current":
                    xs, ys, dmin = place_current(n)
                    halves = sample_footprints(n)
                    dmins.append(dmin)
                else:
                    xs, ys, halves = place_proposed(n)
                    dmins.append(np.nan)
                s, m, p, ff, ur = layout_metrics(xs, ys, halves)
                slit_c.append(s)
                marg_c.append(m)
                freef.append(ff)
                unreach.append(ur)
            dm = np.array(dmins)
            relaxed = float(np.mean(dm < MIN_DIST0 - 1e-9)) * 100 if rule == "current" else float("nan")
            p10 = float(np.nanquantile(dm, 0.10)) if rule == "current" else float("nan")
            print(f"{n:>5} {rule:>9} | {relaxed:>10.1f}% {p10:>14.2f} | "
                  f"{np.mean(slit_c):>12.2f} {np.mean(marg_c):>9.2f} | "
                  f"{100 * np.mean(freef):>5.1f}% {100 * np.mean(unreach):>12.2f}%")


if __name__ == "__main__":
    layouts = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    run([85, 100, 110, 150], layouts)
