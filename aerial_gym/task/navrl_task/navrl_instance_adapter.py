"""Offline multi-instance detection contract. Not wired into the control loop.

The live detector still collapses every positive pixel into one centroid
(``navrl_perception._detect_rgbd``). This module exists so a later SAM / lightweight
tracker backend can keep K instance masks instead of repeating that union. Default
operation is CPU-only: ``NAVRL_INSTANCE_ADAPTER`` is 0, and
``NavRLPerceptionModule`` must not import this file.

SAM weights are a separate-process backend. Loading them in-process is fail-closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DECISION_TARGET = "TARGET"
DECISION_AMBIGUOUS = "AMBIGUOUS"
DECISION_REJECT = "REJECT"

DEFAULT_PIXEL_THRESHOLD = 0.55
DEFAULT_MIN_PIXELS = 2
DEFAULT_AMBIGUOUS_MARGIN = 0.10
DEFAULT_MAX_DEPTH = 20.0
TARGET_RGB = (0.88, 0.08, 0.045)

_BACKEND_STUB = "stub"
_BACKEND_SAM = "sam"


class UnionCollapseError(RuntimeError):
    """Raised when K>1 instances would be reduced to a single sum(mask) centroid."""


class SamBackendNotInstalled(RuntimeError):
    """SAM 3 must run in a separate process; in-process load is forbidden here."""


@dataclass(frozen=True)
class InstanceDetection:
    """One connected component. Fields are per-instance; never a union of others."""

    instance_id: int
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # u0, v0, u1, v1 inclusive
    score: float
    depth_median: float
    uv: Tuple[float, float]
    embedding: Optional[Tuple[float, ...]] = None

    def to_json(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["mask_pixels"] = int(np.asarray(self.mask).sum())
        payload.pop("mask")
        return payload


@dataclass(frozen=True)
class Decision:
    status: str
    selected_id: Optional[int]
    margin: float
    n_instances: int
    reason: str = ""
    instances: Tuple[InstanceDetection, ...] = field(default_factory=tuple)

    def to_json(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "selected_id": self.selected_id,
            "margin": self.margin,
            "n_instances": self.n_instances,
            "reason": self.reason,
            "instances": [inst.to_json() for inst in self.instances],
        }


def adapter_enabled() -> bool:
    """Perception must keep the default off. Only the offline tool reads this."""
    return os.environ.get("NAVRL_INSTANCE_ADAPTER", "0").strip() not in ("", "0", "false", "False")


def colour_score(rgb: np.ndarray) -> np.ndarray:
    """Reproduce AppearanceTargetSegmenter logits then sigmoid: 3R − 2G − 2B − 0.9."""
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    logits = 3.0 * red - 2.0 * green - 2.0 * blue - 0.9
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))


def union_collapse_centroid(
    mask: np.ndarray,
    depth: np.ndarray,
    u_grid: Optional[np.ndarray] = None,
    v_grid: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Current-path formula: one centroid from the sum of every positive pixel."""
    binary = np.asarray(mask, dtype=np.bool_)
    height, width = binary.shape
    if u_grid is None:
        u_grid = np.arange(width, dtype=np.float64)[None, :]
    if v_grid is None:
        v_grid = np.arange(height, dtype=np.float64)[:, None]
    count = float(binary.sum())
    denom = max(count, 1.0)
    weights = binary.astype(np.float64)
    return {
        "count": count,
        "u": float((weights * u_grid).sum() / denom),
        "v": float((weights * v_grid).sum() / denom),
        "range": float((weights * np.asarray(depth, dtype=np.float64)).sum() / denom),
    }


def forbid_union_collapse(instances: Sequence[InstanceDetection]) -> None:
    """합집합 금지 가드: K>1이면 sum(mask) centroid를 만들 수 없다."""
    if len(instances) > 1:
        raise UnionCollapseError(
            "refusing union-collapse of %d instances; keep per-candidate uv/depth"
            % len(instances)
        )


def _connected_components(binary: np.ndarray) -> List[np.ndarray]:
    height, width = binary.shape
    seen = np.zeros_like(binary, dtype=np.bool_)
    components: List[np.ndarray] = []
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for start_v in range(height):
        for start_u in range(width):
            if not binary[start_v, start_u] or seen[start_v, start_u]:
                continue
            mask = np.zeros_like(binary, dtype=np.bool_)
            stack = [(start_v, start_u)]
            seen[start_v, start_u] = True
            while stack:
                v, u = stack.pop()
                mask[v, u] = True
                for dv, du in neighbors:
                    nv, nu = v + dv, u + du
                    if nv < 0 or nu < 0 or nv >= height or nu >= width:
                        continue
                    if binary[nv, nu] and not seen[nv, nu]:
                        seen[nv, nu] = True
                        stack.append((nv, nu))
            components.append(mask)
    return components


def instances_from_score(
    score: np.ndarray,
    depth: np.ndarray,
    pixel_threshold: float = DEFAULT_PIXEL_THRESHOLD,
    max_depth: float = DEFAULT_MAX_DEPTH,
    min_pixels: int = DEFAULT_MIN_PIXELS,
) -> List[InstanceDetection]:
    """Keep each connected component as its own instance. Never union them."""
    active = (np.asarray(score) >= pixel_threshold) & (
        np.asarray(depth) < max_depth
    )
    instances: List[InstanceDetection] = []
    for mask in _connected_components(active):
        pixels = int(mask.sum())
        if pixels < min_pixels:
            continue
        vs, us = np.nonzero(mask)
        depth_values = np.asarray(depth, dtype=np.float64)[mask]
        score_values = np.asarray(score, dtype=np.float64)[mask]
        instances.append(
            InstanceDetection(
                instance_id=len(instances),
                mask=mask,
                bbox=(int(us.min()), int(vs.min()), int(us.max()), int(vs.max())),
                score=float(score_values.mean()),
                depth_median=float(np.median(depth_values)),
                uv=(float(us.mean()), float(vs.mean())),
            )
        )
    return instances


def stub_detect(
    rgb: np.ndarray,
    depth: np.ndarray,
    pixel_threshold: float = DEFAULT_PIXEL_THRESHOLD,
    max_depth: float = DEFAULT_MAX_DEPTH,
    min_pixels: int = DEFAULT_MIN_PIXELS,
) -> List[InstanceDetection]:
    """Colour threshold + connected components. Stand-in for a later SAM process."""
    score = colour_score(np.asarray(rgb, dtype=np.float64))
    return instances_from_score(
        score, depth, pixel_threshold=pixel_threshold, max_depth=max_depth, min_pixels=min_pixels
    )


def sam_backend_spec() -> Dict[str, str]:
    """Contract for a future out-of-process SAM 3.1 worker. No weights are loaded here."""
    return {
        "transport": "npz",
        "request_keys": "rgb uint8 HxWx3, depth float32 HxW, prompt_text, pos_exemplar, neg_exemplar",
        "response_keys": "masks uint8 KxHxW, scores float32 K, boxes float32 Kx4",
        "process": "separate conda / CUDA 12.6 worker; never import sam3 inside Isaac Gym",
    }


def run_backend(name: str, rgb: np.ndarray, depth: np.ndarray, **kwargs: Any) -> List[InstanceDetection]:
    backend = str(name).strip().lower()
    if backend == _BACKEND_STUB:
        return stub_detect(rgb, depth, **kwargs)
    if backend == _BACKEND_SAM:
        raise SamBackendNotInstalled(
            "SAM 3 is a separate-process backend (%s); refusing in-process load"
            % sam_backend_spec()["process"]
        )
    raise ValueError("unknown instance-adapter backend %r (use stub|sam)" % name)


def union_from_instances(
    instances: Sequence[InstanceDetection],
    depth: np.ndarray,
) -> Dict[str, float]:
    """Current-path collapse. Forbidden whenever more than one instance exists."""
    forbid_union_collapse(instances)
    if not instances:
        return union_collapse_centroid(
            np.zeros(np.asarray(depth).shape[:2], dtype=np.bool_), depth
        )
    return union_collapse_centroid(instances[0].mask, depth)


def associate_and_decide(
    instances: Sequence[InstanceDetection],
    ambiguous_margin: float = DEFAULT_AMBIGUOUS_MARGIN,
    predicted_uv: Optional[Tuple[float, float]] = None,
) -> Decision:
    """Choose TARGET / AMBIGUOUS / REJECT. Never union masks to break a tie."""
    ordered = sorted(instances, key=lambda inst: inst.score, reverse=True)
    n_instances = len(ordered)
    if n_instances == 0:
        return Decision(
            status=DECISION_REJECT,
            selected_id=None,
            margin=0.0,
            n_instances=0,
            reason="no_instances",
            instances=(),
        )
    if predicted_uv is not None:
        pred = np.asarray(predicted_uv, dtype=np.float64)

        def _gated_score(inst: InstanceDetection) -> float:
            delta = np.linalg.norm(np.asarray(inst.uv, dtype=np.float64) - pred)
            return float(inst.score - 0.01 * delta)

        ordered = sorted(instances, key=_gated_score, reverse=True)
    top = ordered[0]
    if n_instances == 1:
        return Decision(
            status=DECISION_TARGET,
            selected_id=top.instance_id,
            margin=float(top.score),
            n_instances=1,
            reason="single_instance",
            instances=tuple(ordered),
        )
    margin = float(top.score - ordered[1].score)
    if margin < ambiguous_margin:
        return Decision(
            status=DECISION_AMBIGUOUS,
            selected_id=None,
            margin=margin,
            n_instances=n_instances,
            reason="top_two_scores_within_margin",
            instances=tuple(ordered),
        )
    return Decision(
        status=DECISION_TARGET,
        selected_id=top.instance_id,
        margin=margin,
        n_instances=n_instances,
        reason="top_score_margin_ok",
        instances=tuple(ordered),
    )


def two_blob_fixture(
    height: int = 32,
    width: int = 40,
    blob: int = 6,
) -> Dict[str, Any]:
    """Two same-colour squares with a gap so the union centroid falls in empty space."""
    rgb = np.full((height, width, 3), 0.20, dtype=np.float64)
    depth = np.full((height, width), DEFAULT_MAX_DEPTH, dtype=np.float64)
    v0 = (height - blob) // 2
    left_u0, right_u0 = 4, width - blob - 4
    left = (slice(v0, v0 + blob), slice(left_u0, left_u0 + blob))
    right = (slice(v0, v0 + blob), slice(right_u0, right_u0 + blob))
    rgb[left] = TARGET_RGB
    rgb[right] = TARGET_RGB
    depth[left] = 4.0
    depth[right] = 8.0
    return {
        "rgb": rgb,
        "depth": depth,
        "left_bbox": (left_u0, v0, left_u0 + blob - 1, v0 + blob - 1),
        "right_bbox": (right_u0, v0, right_u0 + blob - 1, v0 + blob - 1),
        "left_depth": 4.0,
        "right_depth": 8.0,
    }


def point_in_bbox(uv: Tuple[float, float], bbox: Tuple[int, int, int, int]) -> bool:
    u, v = uv
    u0, v0, u1, v1 = bbox
    return (u0 - 0.5) <= u <= (u1 + 0.5) and (v0 - 0.5) <= v <= (v1 + 0.5)
