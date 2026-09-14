"""Actor-safe spatial memory for NavRL blind search.

The grid consumes only deployable pose, camera depth, and the perception track.  In particular,
there is deliberately no target-state or simulator-semantic input in this API.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


SEARCH_STATES = ("off", "geofence", "coverage", "belief")
SEARCH_MODE_DIM = 4
SEARCH_COVERAGE_DIM = 24
SEARCH_BELIEF_DIM = 25
SEARCH_SECTORS = 12


def search_feature_dim(search_state: str) -> int:
    """Return the appended search-token width for a declared arm."""
    state = str(search_state).strip().lower()
    if state not in SEARCH_STATES:
        raise ValueError("search_state must be off|geofence|coverage|belief")
    if state in ("off", "geofence"):
        return 0
    if state == "coverage":
        return SEARCH_MODE_DIM + SEARCH_COVERAGE_DIM
    return SEARCH_MODE_DIM + SEARCH_COVERAGE_DIM + SEARCH_BELIEF_DIM


def _yaw_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """World yaw for an xyzw quaternion batch."""
    x, y, z, w = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class SearchGrid:
    """Batched 20x20 coverage and target-belief state over a 40 m arena."""

    def __init__(
        self,
        num_envs,
        arena_bounds=None,
        cell_m=2.0,
        device="cpu",
        *,
        search_state="belief",
        step_dt=0.1,
        camera_hfov_rad=math.radians(87.0),
        camera_range_m=20.0,
        depth_far_m=None,
        detection_probability=0.9,
        target_speed_prior_mps=1.25,
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.cell_m = float(cell_m)
        self.search_state = str(search_state).strip().lower()
        self.feature_dim = search_feature_dim(self.search_state)
        self.step_dt = float(step_dt)
        self.camera_hfov_rad = float(camera_hfov_rad)
        self.camera_range_m = float(camera_range_m)
        self.depth_far_m = float(
            self.camera_range_m if depth_far_m is None else depth_far_m
        )
        self.detection_probability = float(detection_probability)
        self.target_speed_prior_mps = float(target_speed_prior_mps)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not math.isfinite(self.cell_m) or self.cell_m <= 0.0:
            raise ValueError("cell_m must be finite and positive")
        if not math.isfinite(self.step_dt) or self.step_dt <= 0.0:
            raise ValueError("step_dt must be finite and positive")
        if not 0.0 < self.camera_hfov_rad < 2.0 * math.pi:
            raise ValueError("camera_hfov_rad must be in (0, 2*pi)")
        if not math.isfinite(self.camera_range_m) or self.camera_range_m <= 0.0:
            raise ValueError("camera_range_m must be finite and positive")
        if not math.isfinite(self.depth_far_m) or self.depth_far_m <= 0.0:
            raise ValueError("depth_far_m must be finite and positive")
        if not 0.0 <= self.detection_probability <= 1.0:
            raise ValueError("detection_probability must be in [0,1]")

        self.bounds_min = None
        self.bounds_max = None
        self.cell_centres = None
        self.nx = 20
        self.ny = 20
        self.num_cells = self.nx * self.ny
        self.arena_diagonal = None
        if arena_bounds is not None:
            if not isinstance(arena_bounds, (tuple, list)) or len(arena_bounds) != 2:
                raise ValueError("arena_bounds must be (bounds_min, bounds_max)")
            self.set_arena_bounds(arena_bounds[0], arena_bounds[1])

        self.viewed = torch.zeros(
            self.num_envs, self.ny, self.nx, dtype=torch.bool, device=self.device
        )
        self.belief = torch.full(
            (self.num_envs, self.ny, self.nx),
            1.0 / self.num_cells,
            dtype=torch.float32,
            device=self.device,
        )
        self.acquired_ever = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.blind_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.blind_steps_before_first_visible = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.coverage_fraction_at_first_visible = torch.full(
            (self.num_envs,), float("nan"), device=self.device
        )
        self.belief_entropy_at_first_visible = torch.full(
            (self.num_envs,), float("nan"), device=self.device
        )
        self._tracker_was_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._last_tracker_state = torch.zeros(
            self.num_envs, 6, dtype=torch.float32, device=self.device
        )
        self._last_tracker_cov = torch.eye(
            6, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._current_visible_cells = torch.zeros_like(self.viewed)

    def set_arena_bounds(self, bounds_min, bounds_max) -> None:
        """Set fixed per-environment bounds and construct world-frame cell centres once."""
        lo = torch.as_tensor(bounds_min, dtype=torch.float32, device=self.device)
        hi = torch.as_tensor(bounds_max, dtype=torch.float32, device=self.device)
        if lo.ndim == 1:
            lo = lo.unsqueeze(0).expand(self.num_envs, -1)
        if hi.ndim == 1:
            hi = hi.unsqueeze(0).expand(self.num_envs, -1)
        if lo.shape[0] != self.num_envs or hi.shape[0] != self.num_envs:
            raise ValueError("arena bounds batch must match num_envs")
        if lo.shape[1] < 2 or hi.shape[1] < 2:
            raise ValueError("arena bounds must contain x and y")
        lo, hi = lo[:, :2].clone(), hi[:, :2].clone()
        span = hi - lo
        if not bool(torch.isfinite(span).all()) or bool((span <= 0.0).any()):
            raise ValueError("arena bounds must be finite and increasing")
        grid_shape = torch.round(span / self.cell_m).to(torch.long)
        if bool(((span / self.cell_m - grid_shape).abs() > 1e-5).any()):
            raise ValueError("arena span must be an integer multiple of cell_m")
        if bool((grid_shape != torch.tensor([20, 20], device=self.device)).any()):
            raise ValueError("S1 search grid requires a 40m/2m = 20x20 arena")

        x_fraction = (torch.arange(20, device=self.device) + 0.5) / 20.0
        y_fraction = (torch.arange(20, device=self.device) + 0.5) / 20.0
        yy, xx = torch.meshgrid(y_fraction, x_fraction, indexing="ij")
        fraction = torch.stack((xx, yy), dim=-1).reshape(1, self.num_cells, 2)
        self.bounds_min = lo
        self.bounds_max = hi
        self.cell_centres = lo[:, None, :] + fraction * span[:, None, :]
        self.arena_diagonal = span.norm(dim=1).clamp(min=1e-6)

    def _require_bounds(self) -> None:
        if self.cell_centres is None:
            raise RuntimeError("search grid arena bounds have not been set")

    def reset(self, env_ids) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.viewed[ids] = False
        self.belief[ids] = 1.0 / self.num_cells
        self.acquired_ever[ids] = False
        self.blind_steps[ids] = 0
        self.blind_steps_before_first_visible[ids] = -1
        self.coverage_fraction_at_first_visible[ids] = float("nan")
        self.belief_entropy_at_first_visible[ids] = float("nan")
        self._tracker_was_active[ids] = False
        self._last_tracker_state[ids] = 0.0
        self._last_tracker_cov[ids] = torch.eye(6, device=self.device)
        self._current_visible_cells[ids] = False

    def _body_geometry(self, drone_pos_w, quat):
        delta_w = self.cell_centres - drone_pos_w[:, None, :2]
        yaw = _yaw_xyzw(quat)
        c = torch.cos(yaw).unsqueeze(1)
        s = torch.sin(yaw).unsqueeze(1)
        body_x = c * delta_w[:, :, 0] + s * delta_w[:, :, 1]
        body_y = -s * delta_w[:, :, 0] + c * delta_w[:, :, 1]
        distance = torch.sqrt(body_x.square() + body_y.square()).clamp(min=1e-7)
        angle = torch.atan2(body_y, body_x)
        return distance, angle

    def _visible_cells(self, drone_pos_w, quat, depth):
        depth = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if depth.ndim != 3 or depth.shape[0] != self.num_envs:
            raise ValueError("depth must have shape [num_envs,height,width]")
        distance, angle = self._body_geometry(drone_pos_w, quat)
        in_frustum = (
            (angle.abs() <= self.camera_hfov_rad * 0.5)
            & (distance <= self.camera_range_m)
        )
        # A cell direction is conservatively occluded by the closest vertical pixel in its image
        # column.  A camera far-plane fill is a no-return, not a wall at depth_far_m.
        col_min = torch.nan_to_num(
            depth, nan=self.depth_far_m, posinf=self.depth_far_m, neginf=0.0
        ).amin(dim=1)
        width = int(col_min.shape[1])
        column = torch.round(
            (self.camera_hfov_rad * 0.5 - angle)
            / self.camera_hfov_rad
            * max(width - 1, 1)
        ).to(torch.long).clamp(0, max(width - 1, 0))
        sampled = col_min.gather(1, column)
        occlusion_limit = torch.where(
            sampled >= self.depth_far_m - 1e-3,
            torch.full_like(sampled, self.camera_range_m + 1e-3),
            sampled,
        )
        visible = in_frustum & (occlusion_limit > distance)
        return visible.reshape(self.num_envs, self.ny, self.nx)

    def _diffuse(self, belief):
        sigma_m = self.target_speed_prior_mps * self.step_dt
        # Four-neighbour random walk chosen so each Cartesian axis has variance sigma_m**2.
        neighbour = min(0.249, (sigma_m / self.cell_m) ** 2 / 2.0)
        kernel = torch.tensor(
            [[0.0, neighbour, 0.0],
             [neighbour, 1.0 - 4.0 * neighbour, neighbour],
             [0.0, neighbour, 0.0]],
            dtype=belief.dtype,
            device=belief.device,
        ).view(1, 1, 3, 3)
        padded = F.pad(belief.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
        result = F.conv2d(padded, kernel).squeeze(1)
        return result / result.sum(dim=(1, 2), keepdim=True).clamp(min=1e-12)

    def _tracker_gaussian(self, state, cov):
        mean = state[:, None, :2]
        covariance = cov[:, :2, :2]
        eye = torch.eye(2, dtype=covariance.dtype, device=covariance.device).unsqueeze(0)
        inverse = torch.linalg.pinv(covariance + eye * 1e-5)
        delta = self.cell_centres - mean
        mahalanobis = torch.einsum("nci,nij,ncj->nc", delta, inverse, delta)
        # Subtracting the per-grid minimum prevents a small KF covariance from underflowing every
        # 2 m cell centre to zero while preserving the discrete Gaussian ratios.
        weights = torch.exp(-0.5 * (mahalanobis - mahalanobis.amin(dim=1, keepdim=True)))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return weights.reshape(self.num_envs, self.ny, self.nx)

    def normalized_entropy(self) -> torch.Tensor:
        flat = self.belief.reshape(self.num_envs, -1).clamp(min=1e-12)
        return (-(flat * flat.log()).sum(dim=1) / math.log(self.num_cells)).clamp(0.0, 1.0)

    def update(self, drone_pos_w, quat, depth, tracker_state, tracker_cov, tracker_active):
        """Advance coverage/belief using only onboard-derived inputs."""
        self._require_bounds()
        drone_pos_w = torch.as_tensor(drone_pos_w, device=self.device)
        quat = torch.as_tensor(quat, device=self.device)
        tracker_state = torch.as_tensor(tracker_state, device=self.device)
        tracker_cov = torch.as_tensor(tracker_cov, device=self.device)
        active = torch.as_tensor(tracker_active, dtype=torch.bool, device=self.device)
        if drone_pos_w.shape[0] != self.num_envs or quat.shape != (self.num_envs, 4):
            raise ValueError("pose batch must match search grid")
        if tracker_state.shape != (self.num_envs, 6) or tracker_cov.shape != (
            self.num_envs, 6, 6
        ):
            raise ValueError("tracker state/covariance shape mismatch")
        if active.shape != (self.num_envs,):
            raise ValueError("tracker_active shape mismatch")

        current = self._visible_cells(drone_pos_w, quat, depth)
        self._current_visible_cells = current
        self.viewed |= current

        self.belief = torch.where(
            current,
            self.belief * (1.0 - self.detection_probability),
            self.belief,
        )
        self.belief /= self.belief.sum(dim=(1, 2), keepdim=True).clamp(min=1e-12)
        self.belief = self._diffuse(self.belief)

        if bool(active.any()):
            gaussian = self._tracker_gaussian(tracker_state, tracker_cov)
            self.belief = torch.where(active[:, None, None], gaussian, self.belief)
            self._last_tracker_state[active] = tracker_state[active]
            self._last_tracker_cov[active] = tracker_cov[active]
        became_inactive = self._tracker_was_active & ~active
        if bool(became_inactive.any()):
            reinjected = self._tracker_gaussian(
                self._last_tracker_state, self._last_tracker_cov
            )
            self.belief = torch.where(
                became_inactive[:, None, None], reinjected, self.belief
            )

        first = active & ~self.acquired_ever
        if bool(first.any()):
            self.blind_steps_before_first_visible[first] = self.blind_steps[first]
            self.coverage_fraction_at_first_visible[first] = self.viewed[first].float().mean(
                dim=(1, 2)
            )
            self.belief_entropy_at_first_visible[first] = self.normalized_entropy()[first]
        self.acquired_ever |= active
        self.blind_steps = torch.where(
            active, torch.zeros_like(self.blind_steps), self.blind_steps + 1
        )
        self._tracker_was_active = active.clone()

    def _sector_index(self, angle):
        width = 2.0 * math.pi / SEARCH_SECTORS
        # Sector zero is centred on body-forward; indices increase counter-clockwise.
        return torch.floor(torch.remainder(angle + width * 0.5, 2.0 * math.pi) / width).long()

    def features(self, drone_pos_w, quat, *, force_invalid=False):
        """Return the configured mode/coverage[/belief] actor feature vector."""
        if self.feature_dim == 0:
            return torch.empty(self.num_envs, 0, device=self.device)
        self._require_bounds()
        drone_pos_w = torch.as_tensor(drone_pos_w, device=self.device)
        quat = torch.as_tensor(quat, device=self.device)
        distance, angle = self._body_geometry(drone_pos_w, quat)
        sector = self._sector_index(angle)

        if force_invalid:
            mode = torch.zeros(self.num_envs, SEARCH_MODE_DIM, device=self.device)
            coverage = torch.zeros(self.num_envs, SEARCH_COVERAGE_DIM, device=self.device)
        else:
            never = ~self.acquired_ever
            tracked = self._tracker_was_active
            stale = self.acquired_ever & ~tracked
            blind_time = (self.blind_steps.float() * self.step_dt / 60.0).clamp(0.0, 1.0)
            mode = torch.stack(
                (never.float(), tracked.float(), stale.float(), blind_time), dim=1
            )
            unviewed = (~self.viewed).reshape(self.num_envs, -1)
            coverage_parts = []
            for index in range(SEARCH_SECTORS):
                mask = sector.eq(index)
                unseen = mask & unviewed
                mass = unseen.sum(dim=1).float() / float(self.num_cells)
                nearest = distance.masked_fill(~unseen, float("inf")).amin(dim=1)
                nearest = torch.where(
                    torch.isfinite(nearest),
                    nearest / self.arena_diagonal,
                    torch.ones_like(nearest),
                ).clamp(0.0, 1.0)
                coverage_parts.extend((mass, nearest))
            coverage = torch.stack(coverage_parts, dim=1)

        parts = [mode, coverage]
        if self.search_state == "belief":
            if force_invalid:
                sector_mass = torch.full(
                    (self.num_envs, SEARCH_SECTORS),
                    1.0 / SEARCH_SECTORS,
                    device=self.device,
                )
                sector_range_moment = torch.zeros(
                    self.num_envs, SEARCH_SECTORS, device=self.device
                )
                entropy = torch.ones(self.num_envs, 1, device=self.device)
            else:
                flat = self.belief.reshape(self.num_envs, -1)
                sector_mass = torch.zeros(
                    self.num_envs, SEARCH_SECTORS, device=self.device
                )
                sector_mass.scatter_add_(1, sector, flat)
                # The preregistered schema reserves 25 belief dimensions although the prose lists
                # only 12 masses + entropy. Use the remaining 12 as a radial first moment per
                # sector: sum(probability * range/arena-diagonal). This is still derived from the
                # same belief and pose, adds no oracle input, and disambiguates equal sector mass
                # concentrated nearby versus at the far boundary.
                sector_range_moment = torch.zeros_like(sector_mass)
                sector_range_moment.scatter_add_(
                    1,
                    sector,
                    flat * (distance / self.arena_diagonal[:, None]).clamp(0.0, 1.0),
                )
                entropy = self.normalized_entropy().unsqueeze(1)
            parts.append(torch.cat((sector_mass, sector_range_moment, entropy), dim=1))
        result = torch.cat(parts, dim=1)
        if result.shape[1] != self.feature_dim:
            raise RuntimeError("search feature schema drift")
        return result

    def diagnostics(self):
        return {
            "search_mode": torch.where(
                self._tracker_was_active,
                torch.ones_like(self.blind_steps),
                torch.where(
                    self.acquired_ever,
                    torch.full_like(self.blind_steps, 2),
                    torch.zeros_like(self.blind_steps),
                ),
            ),
            "blind_steps": self.blind_steps,
            "blind_steps_before_first_visible": self.blind_steps_before_first_visible,
            "coverage_fraction": self.viewed.float().mean(dim=(1, 2)),
            "belief_entropy": self.normalized_entropy(),
            "coverage_fraction_at_first_visible": self.coverage_fraction_at_first_visible,
            "belief_entropy_at_first_visible": self.belief_entropy_at_first_visible,
        }
