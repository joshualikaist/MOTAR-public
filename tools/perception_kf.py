"""Deterministic constant-velocity multi-candidate Kalman association for MOTAR P5."""

import math

import numpy as np


def candidate_measurement(candidate):
    return np.asarray([
        float(candidate["u_px"]), float(candidate["v_px"]),
        math.log(max(float(candidate["width_px"]), 1e-6)),
        math.log(max(float(candidate["height_px"]), 1e-6)),
    ], dtype=np.float64)


def exact_valid_assignment(cost):
    """Maximize valid matches, then minimize total cost; matrices are at most 5x5."""
    cost = np.asarray(cost, dtype=np.float64)
    tracks, candidates = cost.shape
    best_key, best_pairs = None, []

    def visit(track_index, used, pairs, total):
        nonlocal best_key, best_pairs
        if track_index == tracks:
            key = (-len(pairs), total, tuple(pairs))
            if best_key is None or key < best_key:
                best_key, best_pairs = key, list(pairs)
            return
        visit(track_index + 1, used, pairs, total)
        for candidate_index in range(candidates):
            value = cost[track_index, candidate_index]
            if candidate_index in used or not math.isfinite(value):
                continue
            used.add(candidate_index)
            pairs.append((track_index, candidate_index))
            visit(track_index + 1, used, pairs, total + float(value))
            pairs.pop()
            used.remove(candidate_index)

    visit(0, set(), [], 0.0)
    return best_pairs


class KalmanTrack:
    MEASUREMENT = np.concatenate([np.eye(4), np.zeros((4, 4))], axis=1)

    def __init__(self, track_id, candidate, timestamp_ns, config):
        measurement = candidate_measurement(candidate)
        self.track_id = int(track_id)
        self.state = np.concatenate([measurement, np.zeros(4, dtype=np.float64)])
        self.covariance = np.diag([
            25.0 ** 2, 25.0 ** 2, 0.5 ** 2, 0.5 ** 2,
            80.0 ** 2, 80.0 ** 2, 1.0 ** 2, 1.0 ** 2,
        ]).astype(np.float64)
        self.timestamp_ns = int(timestamp_ns)
        self.last_update_timestamp_ns = int(timestamp_ns)
        self.hits = 1
        self.age = 1
        self.last_candidate_rank = int(candidate["rank"])
        self.confidence_ema = float(candidate["confidence"])
        appearance = np.asarray(candidate["appearance_64d"], dtype=np.float64)
        self.appearance = appearance / max(float(np.linalg.norm(appearance)), 1e-12)
        self.config = config

    def transition(self, dt):
        transition = np.eye(8, dtype=np.float64)
        for position in range(4):
            transition[position, position + 4] = dt
        return transition

    @staticmethod
    def cv_process_block(dt, acceleration_std):
        variance = float(acceleration_std) ** 2
        return variance * np.asarray([
            [dt ** 4 / 4.0, dt ** 3 / 2.0],
            [dt ** 3 / 2.0, dt ** 2],
        ], dtype=np.float64)

    def process_noise(self, dt):
        noise = np.zeros((8, 8), dtype=np.float64)
        for position in range(4):
            acceleration = (
                self.config["process_accel_std_px_per_s2"] if position < 2 else
                self.config["process_accel_std_log_size_per_s2"]
            )
            block = self.cv_process_block(dt, acceleration)
            indices = (position, position + 4)
            noise[np.ix_(indices, indices)] = block
        return noise

    def predict(self, timestamp_ns):
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns <= self.timestamp_ns:
            raise ValueError("track timestamp must increase")
        dt = (timestamp_ns - self.timestamp_ns) / 1e9
        transition = self.transition(dt)
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + self.process_noise(dt)
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        self.timestamp_ns = timestamp_ns
        self.age += 1

    def measurement_noise(self, candidate):
        confidence = max(float(candidate["confidence"]), 0.05)
        side = max(min(float(candidate["width_px"]), float(candidate["height_px"])), 1.0)
        center_std = max(2.0, self.config["measurement_center_fraction"] * side) / confidence
        log_std = self.config["measurement_log_size_std"] / confidence
        return np.diag([center_std ** 2, center_std ** 2, log_std ** 2, log_std ** 2])

    def innovation(self, candidate):
        measurement = candidate_measurement(candidate)
        residual = measurement - self.MEASUREMENT @ self.state
        covariance = self.MEASUREMENT @ self.covariance @ self.MEASUREMENT.T
        covariance += self.measurement_noise(candidate)
        return residual, covariance

    def mahalanobis_squared(self, candidate):
        residual, covariance = self.innovation(candidate)
        return float(residual @ np.linalg.solve(covariance, residual))

    def association_cost(self, candidate):
        mahalanobis = self.mahalanobis_squared(candidate)
        gate = self.config["gating_chi2_4d_p99"]
        if mahalanobis > gate:
            return math.inf
        appearance = np.asarray(candidate["appearance_64d"], dtype=np.float64)
        appearance /= max(float(np.linalg.norm(appearance)), 1e-12)
        appearance_distance = 1.0 - float(np.clip(self.appearance @ appearance, -1.0, 1.0))
        confidence_penalty = 1.0 - float(candidate["confidence"])
        return (mahalanobis / gate
                + self.config["appearance_cost_weight"] * appearance_distance
                + self.config["confidence_cost_weight"] * confidence_penalty)

    def update(self, candidate):
        residual, innovation_covariance = self.innovation(candidate)
        measurement_noise = self.measurement_noise(candidate)
        cross = self.covariance @ self.MEASUREMENT.T
        gain = np.linalg.solve(innovation_covariance, cross.T).T
        self.state = self.state + gain @ residual
        identity = np.eye(8)
        residual_map = identity - gain @ self.MEASUREMENT
        self.covariance = (residual_map @ self.covariance @ residual_map.T
                           + gain @ measurement_noise @ gain.T)
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        alpha = self.config["appearance_ema_alpha"]
        appearance = np.asarray(candidate["appearance_64d"], dtype=np.float64)
        appearance /= max(float(np.linalg.norm(appearance)), 1e-12)
        self.appearance = (1.0 - alpha) * self.appearance + alpha * appearance
        self.appearance /= max(float(np.linalg.norm(self.appearance)), 1e-12)
        self.confidence_ema = ((1.0 - alpha) * self.confidence_ema
                               + alpha * float(candidate["confidence"]))
        self.last_update_timestamp_ns = self.timestamp_ns
        self.last_candidate_rank = int(candidate["rank"])
        self.hits += 1

    @property
    def missed_seconds(self):
        return (self.timestamp_ns - self.last_update_timestamp_ns) / 1e9

    @property
    def confirmed(self):
        return self.hits >= self.config["min_confirm_hits"]

    def box_xyxy(self, width, height):
        center_x, center_y = self.state[:2]
        box_w = float(np.clip(math.exp(float(self.state[2])), 1.0, width * 2.0))
        box_h = float(np.clip(math.exp(float(self.state[3])), 1.0, height * 2.0))
        x1, x2 = max(0.0, center_x - box_w / 2.0), min(float(width), center_x + box_w / 2.0)
        y1, y2 = max(0.0, center_y - box_h / 2.0), min(float(height), center_y + box_h / 2.0)
        if x2 <= x1 or y2 <= y1:
            return None
        return [float(x1), float(y1), float(x2), float(y2)]

    def quality(self):
        survival = math.exp(-self.missed_seconds / self.config["max_missed_seconds"])
        return self.confidence_ema * survival + 0.001 * min(self.hits, 10)

    def snapshot(self, width, height):
        return {
            "track_id": self.track_id,
            "box_xyxy": self.box_xyxy(width, height),
            "confirmed": self.confirmed,
            "hits": self.hits,
            "age": self.age,
            "missed_seconds": self.missed_seconds,
            "confidence_ema": self.confidence_ema,
            "quality": self.quality(),
            "last_candidate_rank": self.last_candidate_rank,
        }


class MultiCandidateKalmanTracker:
    def __init__(self, config):
        self.config = dict(config)
        self.tracks = []
        self.next_track_id = 1
        self.timestamp_ns = None

    def step(self, candidates, timestamp_ns, width, height):
        timestamp_ns = int(timestamp_ns)
        if self.timestamp_ns is not None and timestamp_ns <= self.timestamp_ns:
            raise ValueError("tracker timestamps must increase")
        for track in self.tracks:
            track.predict(timestamp_ns)
        # Expiry is evaluated before association. Otherwise a candidate after an arbitrarily long
        # gap could revive a stale track because its propagated covariance also became enormous.
        self.tracks = [track for track in self.tracks
                       if track.missed_seconds <= self.config["max_missed_seconds"]]
        measurements = [candidate for candidate in candidates
                        if candidate["confidence"] >= self.config["min_measurement_confidence"]]
        cost = np.full((len(self.tracks), len(measurements)), math.inf, dtype=np.float64)
        for track_index, track in enumerate(self.tracks):
            for candidate_index, candidate in enumerate(measurements):
                cost[track_index, candidate_index] = track.association_cost(candidate)
        pairs = exact_valid_assignment(cost)
        assigned_candidates = set()
        for track_index, candidate_index in pairs:
            self.tracks[track_index].update(measurements[candidate_index])
            assigned_candidates.add(candidate_index)
        for candidate_index, candidate in enumerate(measurements):
            if candidate_index in assigned_candidates:
                continue
            if candidate["confidence"] < self.config["track_init_confidence"]:
                continue
            if len(self.tracks) >= self.config["max_tracks"]:
                break
            self.tracks.append(KalmanTrack(
                self.next_track_id, candidate, timestamp_ns, self.config))
            self.next_track_id += 1
        self.timestamp_ns = timestamp_ns
        eligible = [track for track in self.tracks if track.confirmed and track.box_xyxy(width, height)]
        selected = max(eligible, key=lambda track: (track.quality(), -track.track_id)) if eligible else None
        snapshots = [track.snapshot(width, height) for track in self.tracks]
        snapshots.sort(key=lambda item: item["track_id"])
        return selected.snapshot(width, height) if selected else None, snapshots
