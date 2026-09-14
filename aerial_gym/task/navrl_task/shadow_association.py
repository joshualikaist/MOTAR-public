"""S1 shadow-mode candidate machinery: connected components + per-component stats.

Prereg: docs/prereg_2026-09-03_s1_structure_fix_shadow.md. This module is READ-ONLY with
respect to the live perception path: it consumes a copy of the thresholded score mask and
depth, and produces candidate lists for the shadow accumulators. Nothing here may write into
observation, reward, termination, or governor tensors.

Connected components are computed on-GPU by iterative 4-neighbour max-propagation of unique
seed labels. Convergence is bounded by the largest component's graph diameter; the iteration
cap is H+W (worst case a single snake-like component), with early exit on fixpoint. The label
values themselves are arbitrary (pixel linear index + 1); only their partition matters.
"""

import torch


def connected_components(mask: torch.Tensor, max_iters: int = 0) -> torch.Tensor:
    """4-connected components of a boolean mask [N, H, W] -> int32 labels [N, H, W].

    Background is 0. Foreground pixels of the same component share one positive label.
    """
    if mask.ndim != 3:
        raise ValueError("mask must be [envs, H, W]")
    n, h, w = mask.shape
    if max_iters <= 0:
        max_iters = h + w
    device = mask.device
    seed = torch.arange(1, h * w + 1, device=device, dtype=torch.int32).view(1, h, w)
    labels = torch.where(mask, seed.expand(n, -1, -1), torch.zeros(1, dtype=torch.int32, device=device))
    if not bool(mask.any()):
        return labels
    for _ in range(max_iters):
        # max over the 4-neighbourhood, zero-padded at the borders
        up = torch.zeros_like(labels); up[:, :-1, :] = labels[:, 1:, :]
        down = torch.zeros_like(labels); down[:, 1:, :] = labels[:, :-1, :]
        left = torch.zeros_like(labels); left[:, :, :-1] = labels[:, :, 1:]
        right = torch.zeros_like(labels); right[:, :, 1:] = labels[:, :, :-1]
        neigh = torch.maximum(torch.maximum(up, down), torch.maximum(left, right))
        new = torch.where(mask, torch.maximum(labels, neigh), labels)
        if torch.equal(new, labels):
            break
        labels = new
    return labels


def component_candidates(mask: torch.Tensor, depth: torch.Tensor, top_k: int):
    """Top-K components per env by pixel count.

    Returns a dict of [N, K] tensors: ``count`` (0 where no candidate), ``u``, ``v``
    (pixel-centroid, invalid where count==0), ``depth`` (mean over component pixels),
    plus ``num_candidates`` [N]. Sparse throughout: cost scales with foreground pixels,
    not with H*W.
    """
    if mask.shape != depth.shape:
        raise ValueError("mask and depth must share [envs, H, W]")
    n, h, w = mask.shape
    device = mask.device
    out = {
        "count": torch.zeros(n, top_k, device=device, dtype=torch.float32),
        "u": torch.full((n, top_k), float("nan"), device=device),
        "v": torch.full((n, top_k), float("nan"), device=device),
        "depth": torch.full((n, top_k), float("nan"), device=device),
        "num_candidates": torch.zeros(n, device=device, dtype=torch.long),
    }
    pts = mask.nonzero(as_tuple=False)
    if pts.numel() == 0:
        return out
    labels = connected_components(mask)
    lab = labels[pts[:, 0], pts[:, 1], pts[:, 2]].long()
    key = pts[:, 0] * (h * w + 1) + lab
    uniq, inv = torch.unique(key, return_inverse=True)
    m = uniq.shape[0]
    ones = torch.ones(pts.shape[0], device=device)
    count = torch.zeros(m, device=device).scatter_add_(0, inv, ones)
    u_sum = torch.zeros(m, device=device).scatter_add_(0, inv, pts[:, 2].float())
    v_sum = torch.zeros(m, device=device).scatter_add_(0, inv, pts[:, 1].float())
    d_sum = torch.zeros(m, device=device).scatter_add_(
        0, inv, depth[pts[:, 0], pts[:, 1], pts[:, 2]].float()
    )
    env_of = uniq // (h * w + 1)
    for env in torch.unique(env_of).tolist():
        sel = (env_of == env).nonzero(as_tuple=False).squeeze(1)
        k = min(top_k, sel.shape[0])
        order = torch.argsort(count[sel], descending=True)[:k]
        chosen = sel[order]
        out["count"][env, :k] = count[chosen]
        out["u"][env, :k] = u_sum[chosen] / count[chosen]
        out["v"][env, :k] = v_sum[chosen] / count[chosen]
        out["depth"][env, :k] = d_sum[chosen] / count[chosen]
        out["num_candidates"][env] = sel.shape[0]
    return out


# χ²(3 dof, 0.99): diagonal-approximation Mahalanobis gate on 3-D world position.
# Frozen by docs/prereg_2026-09-03_s1_structure_fix_shadow.md; never tuned after results.
SHADOW_GATE_CHI2_3DOF_99 = 11.345
SHADOW_TOP_K = 5


class ShadowAssociator:
    """Counterfactual candidate association running beside the live single-centroid path.

    Owns a second BatchedConstantVelocityTracker instance (never the live one). ``step``
    consumes copies of the frame's threshold mask and depth plus the caller-supplied
    vehicle->world lift, and returns what the candidate pipeline WOULD have locked. It never
    writes into any live tensor; the perception module only reads its return values into
    evaluator-facing diagnostics.
    """

    def __init__(self, tracker, *, camera_offset, target_radius, min_pixels,
                 fx, fy, cx, cy, detect_fx, dt,
                 gate_chi2=SHADOW_GATE_CHI2_3DOF_99, top_k=SHADOW_TOP_K):
        self.tracker = tracker
        self.camera_offset = camera_offset          # [3] vehicle frame
        self.target_radius = float(target_radius)
        self.min_pixels = int(min_pixels)
        self.fx, self.fy = float(fx), float(fy)
        self.cx, self.cy = float(cx), float(cy)
        self.detect_fx = float(detect_fx)
        self.dt = float(dt)
        self.gate_chi2 = float(gate_chi2)
        self.top_k = int(top_k)

    def reset_idx(self, env_ids):
        self.tracker.reset_idx(env_ids)

    def step(self, mask, depth, to_world):
        """One shadow frame. ``to_world(vehicle_xyz[N,3]) -> world_xyz[N,3]``.

        Returns (measurement_world [N,3], visible [N] bool, initialized [N] bool,
        num_candidates [N] long). ``initialized`` marks frames whose lock came from track
        INITIALISATION (no active track -> argmax-count seed), the phase where the pipeline
        has no motion history and colour count is its only cue.
        """
        n = mask.shape[0]
        device = mask.device
        cand = component_candidates(mask, depth, self.top_k)
        valid = cand["count"] >= float(self.min_pixels)          # [N, K]

        # Candidate world positions, mirroring _detect_rgbd's ray/center-range lift exactly.
        u, v = cand["u"], cand["v"]
        ray = torch.stack(
            [torch.ones_like(u), -(u - self.cx) / self.fx, -(v - self.cy) / self.fy], dim=2
        )
        ray = ray / ray.norm(dim=2, keepdim=True).clamp(min=1e-6)
        center_range = cand["depth"] + self.target_radius
        meas_vehicle = self.camera_offset.view(1, 1, 3) + ray * center_range.unsqueeze(2)
        flat_world = to_world(torch.nan_to_num(meas_vehicle, nan=0.0).view(n * self.top_k, 3))
        cand_world = flat_world.view(n, self.top_k, 3)

        # Measurement variance per candidate: same formula as the live path
        # (sigma_r = 0.04 + 0.012 r + 0.15/sqrt(px); sigma_lat = 0.03 + r/detect_fx).
        px = cand["count"].clamp(min=1.0)
        sigma_r = 0.04 + 0.012 * cand["depth"] + 0.15 / px.sqrt()
        sigma_lat = 0.03 + cand["depth"] / max(self.detect_fx, 1.0)
        cand_var = torch.stack(
            [sigma_r.square(), sigma_lat.square(), sigma_lat.square()], dim=2
        )
        cand_var = torch.nan_to_num(cand_var, nan=1.0)

        active = self.tracker.active.clone()
        predicted = self.tracker.state[:, :3] + self.tracker.state[:, 3:] * self.dt
        p_diag = torch.diagonal(self.tracker.cov[:, :3, :3], dim1=1, dim2=2)

        err = cand_world - predicted.view(n, 1, 3)
        d2 = (err.square() / (p_diag.view(n, 1, 3) + cand_var)).sum(dim=2)
        d2 = torch.where(valid, d2, torch.full_like(d2, float("inf")))

        # TRACK phase: nearest gated candidate. INIT phase: argmax pixel count (colour-only cue).
        track_best = d2.argmin(dim=1)
        track_ok = active & (d2.gather(1, track_best.view(-1, 1)).squeeze(1) <= self.gate_chi2)
        init_best = cand["count"].argmax(dim=1)
        init_ok = (~active) & valid.gather(1, init_best.view(-1, 1)).squeeze(1)

        chosen = torch.where(track_ok, track_best, init_best)
        visible = track_ok | init_ok
        gather3 = chosen.view(n, 1, 1).expand(-1, 1, 3)
        meas_world = cand_world.gather(1, gather3).squeeze(1)
        meas_var = cand_var.gather(1, gather3).squeeze(1)
        meas_world = torch.where(visible.view(-1, 1), meas_world, torch.zeros_like(meas_world))

        self.tracker.step(meas_world, visible, meas_var)
        return meas_world, visible, init_ok, cand["num_candidates"]
