import math
from typing import Any
from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("asset_manager")
logger.setLevel("DEBUG")


class AssetManager:
    def __init__(
        self,
        global_tensor_dict,
        num_keep_in_env,
        min_xy_spacing=0.0,
        placement_mode="grid",
        placement_attempts_before_relax=128,
        placement_relax_factor=0.8,
        placement_candidate_batch_size=32,
        placement_touch_dist=0.4,
        placement_gap_dist=1.6,
        placement_surface_clearance=0.45,
    ):
        # min_xy_spacing > 0 enforces a minimum ground-plane (XY) center-to-center distance
        # between the kept obstacles so they never overlap. Default 0.0 = off (unchanged behavior).
        self.min_xy_spacing = float(min_xy_spacing)
        self.placement_mode = str(placement_mode).strip().lower()
        self.placement_attempts_before_relax = max(1, int(placement_attempts_before_relax))
        self.placement_relax_factor = float(placement_relax_factor)
        if self.placement_relax_factor <= 0.0 or self.placement_relax_factor >= 1.0:
            self.placement_relax_factor = 0.8
        self.placement_candidate_batch_size = max(1, int(placement_candidate_batch_size))
        # navrl_band mode: forbidden center-distance band (touch, gap). Candidates whose distance
        # to some placed obstacle falls INSIDE the band would create a narrow slit and are
        # rejected; touching (<= touch) merges obstacles into compound walls instead (the
        # reference NavRL terrain generator's good_distance() behavior).
        self.placement_touch_dist = max(0.0, float(placement_touch_dist))
        self.placement_gap_dist = max(self.placement_touch_dist, float(placement_gap_dist))
        self.placement_surface_clearance = float(placement_surface_clearance)
        if not math.isfinite(self.placement_surface_clearance) or self.placement_surface_clearance < 0.0:
            raise ValueError("placement_surface_clearance must be finite and non-negative")
        self.init_tensors(global_tensor_dict, num_keep_in_env)

    def init_tensors(self, global_tensor_dict, num_keep_in_env):
        self.env_asset_state_tensor = global_tensor_dict["env_asset_state_tensor"]
        self.asset_min_state_ratio = global_tensor_dict["asset_min_state_ratio"]
        self.asset_max_state_ratio = global_tensor_dict["asset_max_state_ratio"]
        self.asset_collision_half_extents = global_tensor_dict.get("asset_collision_half_extents")
        self.env_bounds_min = (
            global_tensor_dict["env_bounds_min"]
            .unsqueeze(1)
            .expand(-1, self.env_asset_state_tensor.shape[1], -1)
        )
        self.env_bounds_max = (
            global_tensor_dict["env_bounds_max"]
            .unsqueeze(1)
            .expand(-1, self.env_asset_state_tensor.shape[1], -1)
        )
        self.num_keep_in_env = num_keep_in_env

    def prepare_for_sim(self):
        self.reset(self.num_keep_in_env)
        logger.warning(f"Number of obstacles to be kept in the environment: {self.num_keep_in_env}")
        logger.warning(
            "Obstacle placement | mode=%s min_xy_spacing=%.3f relax_factor=%.3f "
            "candidate_batch=%d touch=%.2f gap=%.2f surface_clearance=%.2f"
            % (
                self.placement_mode,
                self.min_xy_spacing,
                self.placement_relax_factor,
                self.placement_candidate_batch_size,
                self.placement_touch_dist,
                self.placement_gap_dist,
                self.placement_surface_clearance,
            )
        )

    def pre_physics_step(self, actions):
        pass

    def post_physics_step(self):
        pass

    def step(self, actions):
        pass
        # Implement this function if needed.
        # this functionality can do speciic things with the environment assets on stepping.
        # nothing really needs to be done for static environments.
        # if force needs to be applied, it should be done in the other classes and it's
        # better to leave this class to manipulate the state tensors.

    def reset(self, num_obstacles_per_env):
        self.reset_idx(torch.arange(self.env_asset_state_tensor.shape[0]), num_obstacles_per_env)

    def reset_idx(self, env_ids, num_obstacles_per_env=0):
        env_ids = self._env_ids_tensor(env_ids)
        num_obstacles_per_env = int(num_obstacles_per_env)
        if num_obstacles_per_env < self.num_keep_in_env:
            logger.info(
                "Number of obstacles required in the environment by the \
                  code is lesser than the minimum number of obstacles that the environment configuration specifies."
            )
            num_obstacles_per_env = self.num_keep_in_env

        sampled_asset_state_ratio = torch_rand_float_tensor(
            self.asset_min_state_ratio, self.asset_max_state_ratio
        )
        positions = torch_interpolate_ratio(
            min=self.env_bounds_min,
            max=self.env_bounds_max,
            ratio=sampled_asset_state_ratio[..., 0:3],
        )
        if self.min_xy_spacing > 0.0:
            positions = self._enforce_min_xy_spacing(positions, num_obstacles_per_env, env_ids)
        self.env_asset_state_tensor[env_ids, :, 0:3] = positions[env_ids, :, 0:3]
        self.env_asset_state_tensor[env_ids, :, 3:7] = quat_from_euler_xyz_tensor(
            sampled_asset_state_ratio[env_ids, :, 3:6]
        )
        # put those obstacles not needed in the environment outside
        self.env_asset_state_tensor[env_ids, num_obstacles_per_env:, 0:3] = -1000.0

    def _env_ids_tensor(self, env_ids):
        num_envs = self.env_asset_state_tensor.shape[0]
        device = self.env_asset_state_tensor.device
        if env_ids is None:
            return torch.arange(num_envs, device=device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=device, dtype=torch.long).view(-1)
        return torch.as_tensor(env_ids, device=device, dtype=torch.long).view(-1)

    def _enforce_min_xy_spacing(self, positions, num_used, env_ids):
        if self.placement_mode in ("footprint_clearance", "nonoverlap", "surface_clearance"):
            return self._footprint_clearance_xy_spacing(positions, num_used, env_ids)
        if self.placement_mode == "navrl_band":
            return self._navrl_band_xy_spacing(positions, num_used, env_ids)
        if self.placement_mode in ("random", "rejection", "navrl_random"):
            return self._random_rejection_xy_spacing(positions, num_used, env_ids)
        return self._grid_xy_spacing(positions, num_used, env_ids)

    def _placement_band(self):
        span_x = self.env_bounds_max[:, 0, 0] - self.env_bounds_min[:, 0, 0]
        span_y = self.env_bounds_max[:, 0, 1] - self.env_bounds_min[:, 0, 1]
        bx0 = self.env_bounds_min[:, 0, 0] + self.asset_min_state_ratio[:, 0, 0] * span_x
        bx1 = self.env_bounds_min[:, 0, 0] + self.asset_max_state_ratio[:, 0, 0] * span_x
        by0 = self.env_bounds_min[:, 0, 1] + self.asset_min_state_ratio[:, 0, 1] * span_y
        by1 = self.env_bounds_min[:, 0, 1] + self.asset_max_state_ratio[:, 0, 1] * span_y
        return bx0, bx1, by0, by1

    def _footprint_clearance_xy_spacing(self, positions, num_used, env_ids):
        """Place assets without overlap using their real collision footprints.

        Each collision box is conservatively replaced by its XY circumcircle.  Therefore the
        guarantee survives any sampled yaw, unlike the old centre-distance ``navrl_band`` rule.
        Every accepted pair has at least ``placement_surface_clearance`` metres between those
        circumcircles and every circle remains inside the placement band.  There is deliberately
        no merge/overlap fallback: an infeasible layout fails closed instead of silently changing
        the requested obstacle count into a smaller number of compound obstacles.
        """
        num_assets = positions.shape[1]
        n = int(min(num_used, num_assets))
        if n <= 0 or len(env_ids) == 0:
            return positions
        if self.asset_collision_half_extents is None:
            raise RuntimeError("footprint_clearance requires asset_collision_half_extents")
        half = self.asset_collision_half_extents
        if half.ndim != 3 or half.shape[:2] != positions.shape[:2] or half.shape[2] < 2:
            raise RuntimeError("asset_collision_half_extents shape does not match asset positions")
        if not torch.isfinite(half[env_ids, :n, :2]).all() or (half[env_ids, :n, :2] <= 0.0).any():
            raise RuntimeError("footprint_clearance received invalid collision half-extents")

        device = positions.device
        num_envs = len(env_ids)
        clearance = self.placement_surface_clearance
        support = torch.linalg.vector_norm(half[env_ids, :n, :2], dim=2)
        bx0, bx1, by0, by1 = self._placement_band()
        bx0, bx1, by0, by1 = bx0[env_ids], bx1[env_ids], by0[env_ids], by1[env_ids]
        placed_x = torch.empty((num_envs, n), device=device)
        placed_y = torch.empty((num_envs, n), device=device)
        batch = max(32, self.placement_candidate_batch_size)
        # This is a hard engineering limit, not a relaxation trigger. With the canonical
        # 40x40m/300-bar contract it gives 40960 candidates per obstacle before refusing.
        max_candidates = max(batch, self.placement_attempts_before_relax * 320)

        for k in range(n):
            radius = support[:, k]
            lo_x, hi_x = bx0 + radius, bx1 - radius
            lo_y, hi_y = by0 + radius, by1 - radius
            invalid_band = (hi_x <= lo_x) | (hi_y <= lo_y)
            if invalid_band.any():
                bad = env_ids[invalid_band].detach().cpu().tolist()
                raise RuntimeError("collision footprint does not fit placement band for envs %s" % bad)
            pending = torch.ones(num_envs, dtype=torch.bool, device=device)
            attempted = torch.zeros(num_envs, dtype=torch.int64, device=device)
            while pending.any():
                idx = pending.nonzero(as_tuple=False).squeeze(-1)
                local_n = len(idx)
                cand_x = lo_x[idx].unsqueeze(1) + (hi_x - lo_x)[idx].unsqueeze(1) * torch.rand(
                    local_n, batch, device=device
                )
                cand_y = lo_y[idx].unsqueeze(1) + (hi_y - lo_y)[idx].unsqueeze(1) * torch.rand(
                    local_n, batch, device=device
                )
                if k == 0:
                    valid = torch.ones((local_n, batch), dtype=torch.bool, device=device)
                else:
                    dx = placed_x[idx, :k].unsqueeze(1) - cand_x.unsqueeze(2)
                    dy = placed_y[idx, :k].unsqueeze(1) - cand_y.unsqueeze(2)
                    required = support[idx, :k].unsqueeze(1) + radius[idx, None, None] + clearance
                    valid = ((dx * dx + dy * dy) >= required * required).all(dim=2)
                has_valid = valid.any(dim=1)
                if has_valid.any():
                    accepted = idx[has_valid]
                    local = torch.arange(local_n, device=device)[has_valid]
                    chosen = valid[has_valid].to(torch.int64).argmax(dim=1)
                    placed_x[accepted, k] = cand_x[local, chosen]
                    placed_y[accepted, k] = cand_y[local, chosen]
                    pending[accepted] = False
                rejected = idx[~has_valid]
                if len(rejected) > 0:
                    attempted[rejected] += batch
                    exhausted = attempted[rejected] >= max_candidates
                    if exhausted.any():
                        bad = env_ids[rejected[exhausted]].detach().cpu().tolist()
                        raise RuntimeError(
                            "non-overlap placement exhausted %d candidates at asset %d/%d for envs %s; "
                            "refusing overlap fallback" % (max_candidates, k + 1, n, bad)
                        )

        result = positions.clone()
        result[env_ids, :n, 0] = placed_x
        result[env_ids, :n, 1] = placed_y
        return result

    def _grid_xy_spacing(self, positions, num_used, env_ids):
        """Place the kept obstacles on a per-env jittered grid so that no two are closer than
        min_xy_spacing (center-to-center, XY).

        A grid guarantees the spacing deterministically. Pure rejection sampling was tried first
        but stalls once the obstacles fill a large fraction of the field (e.g. 16 bars at a 1.3 m
        minimum spacing). Each obstacle keeps a fixed grid cell across resets; the per-reset
        jitter (bounded so the spacing guarantee always holds) together with the obstacles'
        random footprints gives per-env variety. Only the XY position is set here; z and
        orientation are left as sampled. Requires the placement band to be at least
        (cols * min_xy_spacing) x (rows * min_xy_spacing) -- otherwise the jitter clamps to 0 and
        the nominal cell spacing (still the best achievable) is used.
        """
        _, num_assets, _ = positions.shape
        num_envs = len(env_ids)
        n = int(min(num_used, num_assets))
        if n <= 1 or self.min_xy_spacing <= 0.0:
            return positions
        device = positions.device
        min_dist = self.min_xy_spacing
        # per-env XY placement band from the asset ratio window (identical for every obstacle)
        bx0, bx1, by0, by1 = self._placement_band()
        bx0, bx1, by0, by1 = bx0[env_ids], bx1[env_ids], by0[env_ids], by1[env_ids]
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        k = torch.arange(n, device=device)
        col = (k % cols).float()
        row = torch.div(k, cols, rounding_mode="floor").float()
        cell_x = (bx1 - bx0) / cols  # (num_envs,)
        cell_y = (by1 - by0) / rows
        cx = bx0.unsqueeze(1) + (col.unsqueeze(0) + 0.5) * cell_x.unsqueeze(1)  # (num_envs, n)
        cy = by0.unsqueeze(1) + (row.unsqueeze(0) + 0.5) * cell_y.unsqueeze(1)
        jitter_x = torch.clamp((cell_x - min_dist) * 0.5, min=0.0).unsqueeze(1)  # (num_envs, 1)
        jitter_y = torch.clamp((cell_y - min_dist) * 0.5, min=0.0).unsqueeze(1)
        cx = cx + (2.0 * torch.rand(num_envs, n, device=device) - 1.0) * jitter_x
        cy = cy + (2.0 * torch.rand(num_envs, n, device=device) - 1.0) * jitter_y
        positions = positions.clone()
        positions[env_ids, :n, 0] = cx
        positions[env_ids, :n, 1] = cy
        return positions

    def _random_rejection_xy_spacing(self, positions, num_used, env_ids):
        """NavRL-style random scatter: sample uniform XY positions and reject candidates that are
        too close to already placed obstacles. Candidates are drawn in small batches to avoid the
        slow one-candidate-at-a-time Python loop at high density. If the local field is saturated,
        progressively relax the spacing so high-density runs do not stall, mirroring NavRL's
        obstacle placement trick.
        """
        num_assets = positions.shape[1]
        n = int(min(num_used, num_assets))
        if n <= 1 or self.min_xy_spacing <= 0.0 or len(env_ids) == 0:
            return positions

        device = positions.device
        num_envs = len(env_ids)
        bx0, bx1, by0, by1 = self._placement_band()
        bx0, bx1, by0, by1 = bx0[env_ids], bx1[env_ids], by0[env_ids], by1[env_ids]
        width = (bx1 - bx0).clamp(min=1e-6)
        height = (by1 - by0).clamp(min=1e-6)
        placed_x = torch.empty((num_envs, n), device=device)
        placed_y = torch.empty((num_envs, n), device=device)
        min_dist = torch.full((num_envs,), self.min_xy_spacing, device=device)
        relax_factor = min(max(self.placement_relax_factor, 0.0), 1.0)
        batch = self.placement_candidate_batch_size

        for k in range(n):
            if k == 0:
                placed_x[:, 0] = bx0 + width * torch.rand(num_envs, device=device)
                placed_y[:, 0] = by0 + height * torch.rand(num_envs, device=device)
                continue

            pending = torch.ones(num_envs, dtype=torch.bool, device=device)
            attempts = torch.zeros(num_envs, dtype=torch.int32, device=device)
            while pending.any():
                idx = pending.nonzero(as_tuple=False).squeeze(-1)
                local_n = len(idx)
                cand_x = bx0[idx].unsqueeze(1) + width[idx].unsqueeze(1) * torch.rand(
                    local_n, batch, device=device
                )
                cand_y = by0[idx].unsqueeze(1) + height[idx].unsqueeze(1) * torch.rand(
                    local_n, batch, device=device
                )
                dx = placed_x[idx, :k].unsqueeze(1) - cand_x.unsqueeze(2)
                dy = placed_y[idx, :k].unsqueeze(1) - cand_y.unsqueeze(2)
                nearest_sq = (dx * dx + dy * dy).min(dim=2).values
                valid = nearest_sq >= min_dist[idx].pow(2).unsqueeze(1)
                has_valid = valid.any(dim=1)
                first_valid = valid.to(torch.int64).argmax(dim=1)

                accepted = idx[has_valid]
                if len(accepted) > 0:
                    local = torch.arange(local_n, device=device)[has_valid]
                    chosen = first_valid[has_valid]
                    placed_x[accepted, k] = cand_x[local, chosen]
                    placed_y[accepted, k] = cand_y[local, chosen]
                    pending[accepted] = False

                rejected = idx[~has_valid]
                if len(rejected) > 0:
                    attempts[rejected] += batch
                    relax = attempts[rejected] >= self.placement_attempts_before_relax
                    if relax.any():
                        relax_ids = rejected[relax]
                        min_dist[relax_ids] *= relax_factor
                        attempts[relax_ids] = 0

        positions = positions.clone()
        positions[env_ids, :n, 0] = placed_x
        positions[env_ids, :n, 1] = placed_y
        return positions

    def _navrl_band_xy_spacing(self, positions, num_used, env_ids):
        """Slit-free scatter with a forbidden center-distance band (touch, gap).

        A candidate is accepted only if EVERY already-placed obstacle is either touching it
        (center distance <= touch -> the boxes overlap for all footprints in the 0.4..0.8 m
        pool, forming one compound wall) or at least ``gap`` away. The latter prevents box
        overlap but is not a passability guarantee: diagonal 0.8 m squares at d=1.6 m can leave
        only ~0.469 m corner gap. Distances inside the band can create an even narrower slit and
        are never accepted; when the field saturates, the
        candidate snaps onto a random placed obstacle (guaranteed merge) instead of relaxing
        the forbidden-band rule -- the exact opposite failure mode of the legacy
        "random" rule, whose *=0.8 relaxation produced ~2.2 impassable slits per 150-bar
        layout (tools/probe_placement_slits.py, 2026-07-31).
        """
        num_assets = positions.shape[1]
        n = int(min(num_used, num_assets))
        if n <= 1 or len(env_ids) == 0:
            return positions

        device = positions.device
        num_envs = len(env_ids)
        touch = self.placement_touch_dist
        gap = self.placement_gap_dist
        bx0, bx1, by0, by1 = self._placement_band()
        bx0, bx1, by0, by1 = bx0[env_ids], bx1[env_ids], by0[env_ids], by1[env_ids]
        width = (bx1 - bx0).clamp(min=1e-6)
        height = (by1 - by0).clamp(min=1e-6)
        placed_x = torch.empty((num_envs, n), device=device)
        placed_y = torch.empty((num_envs, n), device=device)
        batch = self.placement_candidate_batch_size
        max_attempts = self.placement_attempts_before_relax * 10  # then merge, never slit

        placed_x[:, 0] = bx0 + width * torch.rand(num_envs, device=device)
        placed_y[:, 0] = by0 + height * torch.rand(num_envs, device=device)
        for k in range(1, n):
            pending = torch.ones(num_envs, dtype=torch.bool, device=device)
            attempts = torch.zeros(num_envs, dtype=torch.int32, device=device)
            while pending.any():
                idx = pending.nonzero(as_tuple=False).squeeze(-1)
                local_n = len(idx)
                cand_x = bx0[idx].unsqueeze(1) + width[idx].unsqueeze(1) * torch.rand(
                    local_n, batch, device=device
                )
                cand_y = by0[idx].unsqueeze(1) + height[idx].unsqueeze(1) * torch.rand(
                    local_n, batch, device=device
                )
                dx = placed_x[idx, :k].unsqueeze(1) - cand_x.unsqueeze(2)
                dy = placed_y[idx, :k].unsqueeze(1) - cand_y.unsqueeze(2)
                d = torch.sqrt(dx * dx + dy * dy)  # (local_n, batch, k)
                # valid iff no placed obstacle sits INSIDE the forbidden band
                in_band = (d > touch) & (d < gap)
                valid = ~in_band.any(dim=2)
                has_valid = valid.any(dim=1)
                first_valid = valid.to(torch.int64).argmax(dim=1)

                accepted = idx[has_valid]
                if len(accepted) > 0:
                    local = torch.arange(local_n, device=device)[has_valid]
                    chosen = first_valid[has_valid]
                    placed_x[accepted, k] = cand_x[local, chosen]
                    placed_y[accepted, k] = cand_y[local, chosen]
                    pending[accepted] = False

                rejected = idx[~has_valid]
                if len(rejected) > 0:
                    attempts[rejected] += batch
                    saturated = attempts[rejected] >= max_attempts
                    if saturated.any():
                        sat_ids = rejected[saturated]
                        # merge fallback: snap onto a random already-placed obstacle with a
                        # sub-touch offset -> guaranteed compound wall, never a slit.
                        pick = torch.randint(0, k, (len(sat_ids),), device=device)
                        ang = 2.0 * math.pi * torch.rand(len(sat_ids), device=device)
                        r = 0.5 * touch * torch.rand(len(sat_ids), device=device)
                        placed_x[sat_ids, k] = (
                            placed_x[sat_ids, pick] + r * torch.cos(ang)
                        ).clamp(bx0[sat_ids], bx1[sat_ids])
                        placed_y[sat_ids, k] = (
                            placed_y[sat_ids, pick] + r * torch.sin(ang)
                        ).clamp(by0[sat_ids], by1[sat_ids])
                        pending[sat_ids] = False

        positions = positions.clone()
        positions[env_ids, :n, 0] = placed_x
        positions[env_ids, :n, 1] = placed_y
        return positions
