"""Shared causal candidate-window data and models for MOTAR P6/P7."""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset

from perception_candidates import read_jsonl, sha256_file, validate_candidate_record


TOP_K = 5
FEATURE_DIMENSION = 69
MOTION_FEATURE_DIMENSION = 12
NO_LOCK_CLASS = TOP_K


def box_iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-12)


def candidate_box(candidate):
    half_width = float(candidate["width_px"]) / 2.0
    half_height = float(candidate["height_px"]) / 2.0
    return [
        float(candidate["u_px"]) - half_width,
        float(candidate["v_px"]) - half_height,
        float(candidate["u_px"]) + half_width,
        float(candidate["v_px"]) + half_height,
    ]


def candidate_feature(candidate, width, height):
    """Normalize geometry while preserving the frozen P4 appearance vector verbatim."""
    appearance = np.asarray(candidate["appearance_64d"], dtype=np.float32)
    if appearance.shape != (64,) or not np.isfinite(appearance).all():
        raise ValueError("temporal candidate appearance must be 64 finite values")
    geometry = np.asarray([
        2.0 * float(candidate["u_px"]) / float(width) - 1.0,
        2.0 * float(candidate["v_px"]) / float(height) - 1.0,
        float(candidate["width_px"]) / float(width),
        float(candidate["height_px"]) / float(height),
        float(candidate["confidence"]),
    ], dtype=np.float32)
    feature = np.concatenate([geometry, appearance])
    if feature.shape != (FEATURE_DIMENSION,) or not np.isfinite(feature).all():
        raise ValueError("temporal candidate feature is invalid")
    return feature


def supervision_target(candidates, ground_truth, iou_threshold):
    """Return the best current UAV candidate rank, or NO_LOCK when none reaches the gate."""
    scored = []
    for candidate in candidates:
        overlaps = [box_iou(candidate_box(candidate), target) for target in ground_truth]
        best_iou = max(overlaps) if overlaps else 0.0
        scored.append((best_iou, float(candidate["confidence"]), -int(candidate["rank"])))
    if not scored:
        return NO_LOCK_CLASS
    best_rank = max(range(len(scored)), key=lambda rank: scored[rank])
    return best_rank if scored[best_rank][0] >= iou_threshold else NO_LOCK_CLASS


def load_aligned_records(manifest, manifest_receipt, candidates, candidate_receipt):
    """Fail closed unless manifest and P4 candidate records have identical provenance and order."""
    manifest, manifest_receipt = Path(manifest), Path(manifest_receipt)
    candidates, candidate_receipt = Path(candidates), Path(candidate_receipt)
    manifest_meta = json.loads(manifest_receipt.read_text())
    candidate_meta = json.loads(candidate_receipt.read_text())
    manifest_sha = sha256_file(manifest)
    candidate_sha = sha256_file(candidates)
    if manifest_sha != manifest_meta["manifest_sha256"]:
        raise ValueError("temporal manifest receipt hash mismatch")
    if candidate_sha != candidate_meta["output_sha256"]:
        raise ValueError("temporal candidate receipt hash mismatch")
    if candidate_meta["manifest_sha256"] != manifest_sha:
        raise ValueError("temporal candidates came from a different manifest")
    if not candidate_meta.get("complete_manifest"):
        raise ValueError("temporal training/evaluation forbids candidate prefixes")

    missing = object()
    aligned = []
    previous = {}
    for pair in itertools.zip_longest(read_jsonl(manifest), read_jsonl(candidates), fillvalue=missing):
        source, record = pair
        if source is missing or record is missing:
            raise ValueError("temporal manifest/candidate record counts differ")
        identity = ("frame_id", "source_sequence_id", "frame_index", "capture_timestamp_ns")
        if any(source[key] != record[key] for key in identity):
            raise ValueError("temporal manifest/candidate identity mismatch")
        validate_candidate_record(record, previous)
        aligned.append({"source": source, "candidate_record": record})
    if len(aligned) != int(candidate_meta["records"]):
        raise ValueError("temporal candidate receipt record count mismatch")
    return aligned, manifest_meta, candidate_meta


def load_aligned_motion_records(motion_path, motion_receipt, aligned_records,
                                manifest_sha256, candidates_sha256):
    """Load one audited P7c motion row per already-aligned candidate frame."""
    motion_path, motion_receipt = Path(motion_path), Path(motion_receipt)
    receipt = json.loads(motion_receipt.read_text())
    if sha256_file(motion_path) != receipt["output_sha256"]:
        raise ValueError("temporal motion receipt hash mismatch")
    if receipt["manifest_sha256"] != manifest_sha256:
        raise ValueError("temporal motion came from a different manifest")
    if receipt["candidates_sha256"] != candidates_sha256:
        raise ValueError("temporal motion came from different candidates")
    if int(receipt["feature_dimension"]) != MOTION_FEATURE_DIMENSION:
        raise ValueError("temporal motion feature dimension mismatch")
    rows = []
    missing = object()
    for pair in itertools.zip_longest(
            aligned_records, read_jsonl(motion_path), fillvalue=missing):
        aligned, row = pair
        if aligned is missing or row is missing:
            raise ValueError("temporal motion/candidate record counts differ")
        source = aligned["source"]
        identity = ("frame_id", "source_sequence_id", "frame_index", "capture_timestamp_ns")
        if any(source[key] != row.get(key) for key in identity):
            raise ValueError("temporal motion/candidate identity mismatch")
        values = np.asarray(row.get("candidate_motion", []), dtype=np.float32)
        if values.shape != (TOP_K, MOTION_FEATURE_DIMENSION):
            raise ValueError("temporal motion row shape mismatch")
        if not np.isfinite(values).all():
            raise ValueError("temporal motion contains non-finite values")
        rows.append(values)
    if len(rows) != int(receipt["records"]):
        raise ValueError("temporal motion receipt record count mismatch")
    return rows, receipt


class TemporalCandidateDataset(Dataset):
    """Causal, clip-bounded windows ending at each current P4 frame."""

    def __init__(self, aligned_records, history_length, evaluation_iou, motion_records=None):
        self.records = list(aligned_records)
        self.history_length = int(history_length)
        self.evaluation_iou = float(evaluation_iou)
        self.motion_records = motion_records
        if self.history_length <= 0:
            raise ValueError("history length must be positive")
        if motion_records is not None and len(motion_records) != len(self.records):
            raise ValueError("motion record count differs from candidate records")
        self.frame_features = []
        self.frame_candidate_masks = []
        self.targets = []
        self.valid_targets = []
        self.windows = []
        history_by_sequence = {}
        for index, aligned in enumerate(self.records):
            source, record = aligned["source"], aligned["candidate_record"]
            features = np.zeros((TOP_K, FEATURE_DIMENSION), dtype=np.float32)
            mask = np.zeros(TOP_K, dtype=np.bool_)
            for candidate in record["candidates"]:
                rank = int(candidate["rank"])
                if rank >= TOP_K:
                    raise ValueError("temporal candidate rank exceeds Top-K")
                features[rank] = candidate_feature(
                    candidate, source["width_px"], source["height_px"])
                mask[rank] = True
            self.frame_features.append(features)
            self.frame_candidate_masks.append(mask)
            self.targets.append(supervision_target(
                record["candidates"], source["ground_truth_xyxy"], self.evaluation_iou))
            valid_targets = np.zeros(TOP_K, dtype=np.float32)
            for candidate in record['candidates']:
                valid_targets[int(candidate['rank'])] = float(any(
                    box_iou(candidate_box(candidate), gt) >= self.evaluation_iou
                    for gt in source['ground_truth_xyxy']))
            self.valid_targets.append(valid_targets)
            sequence = source["source_sequence_id"]
            history = history_by_sequence.setdefault(sequence, [])
            history.append(index)
            self.windows.append(tuple(history[-self.history_length:]))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        indices = self.windows[index]
        length = len(indices)
        features = np.zeros(
            (self.history_length, TOP_K, FEATURE_DIMENSION), dtype=np.float32)
        candidate_mask = np.zeros((self.history_length, TOP_K), dtype=np.bool_)
        frame_mask = np.zeros(self.history_length, dtype=np.bool_)
        delta_seconds = np.zeros(self.history_length, dtype=np.float32)
        age_seconds = np.zeros(self.history_length, dtype=np.float32)
        motion_features = np.zeros(
            (self.history_length, TOP_K, MOTION_FEATURE_DIMENSION), dtype=np.float32)
        previous_timestamp = None
        current_timestamp = int(self.records[indices[-1]]["source"]["capture_timestamp_ns"])
        for position, frame_index in enumerate(indices):
            features[position] = self.frame_features[frame_index]
            candidate_mask[position] = self.frame_candidate_masks[frame_index]
            frame_mask[position] = True
            timestamp = int(self.records[frame_index]["source"]["capture_timestamp_ns"])
            age_seconds[position] = min(max((current_timestamp - timestamp) / 1e9, 0.0), 10.0)
            if previous_timestamp is not None:
                delta_seconds[position] = min(max((timestamp - previous_timestamp) / 1e9, 0.0), 1.0)
            previous_timestamp = timestamp
            if self.motion_records is not None:
                motion_features[position] = self.motion_records[frame_index]
        return {
            "features": torch.from_numpy(features),
            "candidate_mask": torch.from_numpy(candidate_mask),
            "frame_mask": torch.from_numpy(frame_mask),
            "delta_seconds": torch.from_numpy(delta_seconds),
            "age_seconds": torch.from_numpy(age_seconds),
            "motion_features": torch.from_numpy(motion_features),
            "length": torch.tensor(length, dtype=torch.long),
            "target": torch.tensor(self.targets[index], dtype=torch.long),
            "valid_targets": torch.from_numpy(self.valid_targets[index]),
            "record_index": torch.tensor(index, dtype=torch.long),
        }

    def label_counts(self):
        counts = {str(rank): 0 for rank in range(TOP_K)}
        counts["no_lock"] = 0
        for target in self.targets:
            counts[str(target) if target < TOP_K else "no_lock"] += 1
        return counts


class TemporalSelectorBase(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        dimension = int(config["hidden_dim"])
        embedding = int(config["candidate_embedding_dim"])
        if dimension != embedding:
            raise ValueError("v1 requires candidate embedding dim == hidden dim")
        dropout = float(config["dropout"])
        self.candidate_encoder = nn.Sequential(
            nn.Linear(FEATURE_DIMENSION, embedding),
            nn.ReLU(),
            nn.LayerNorm(embedding),
        )
        self.empty_frame = nn.Parameter(torch.zeros(embedding))
        self.frame_projection = nn.Sequential(
            nn.Linear(embedding + 1, dimension),
            nn.ReLU(),
            nn.LayerNorm(dimension),
        )
        self.candidate_scorer = nn.Sequential(
            nn.Linear(4 * dimension, dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 1),
        )
        self.no_lock_scorer = nn.Linear(dimension, 1)

    def encode_frames(self, features, candidate_mask, delta_seconds):
        embeddings = self.candidate_encoder(features)
        confidence = features[..., 4].clamp_min(0.0) + 1e-3
        weights = confidence * candidate_mask.to(features.dtype)
        denominator = weights.sum(dim=2, keepdim=True)
        pooled = (embeddings * weights.unsqueeze(-1)).sum(dim=2)
        pooled = pooled / denominator.clamp_min(1e-9)
        empty = denominator.squeeze(-1) <= 0.0
        pooled = torch.where(empty.unsqueeze(-1), self.empty_frame.view(1, 1, -1), pooled)
        frame_input = self.frame_projection(torch.cat([pooled, delta_seconds.unsqueeze(-1)], dim=-1))
        return embeddings, frame_input

    @staticmethod
    def current_tensor(values, lengths):
        batch = torch.arange(values.shape[0], device=values.device)
        return values[batch, lengths - 1]

    def score_current(self, context, embeddings, candidate_mask, lengths):
        current_embeddings = self.current_tensor(embeddings, lengths)
        current_mask = self.current_tensor(candidate_mask, lengths)
        expanded = context.unsqueeze(1).expand_as(current_embeddings)
        comparison = torch.cat([
            current_embeddings,
            expanded,
            torch.abs(current_embeddings - expanded),
            current_embeddings * expanded,
        ], dim=-1)
        candidate_logits = self.candidate_scorer(comparison).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(~current_mask, -1e9)
        return torch.cat([candidate_logits, self.no_lock_scorer(context)], dim=1)


class GRUCandidateSelector(TemporalSelectorBase):
    def __init__(self, config):
        super().__init__(config)
        hidden = int(config["hidden_dim"])
        self.temporal = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=int(config["num_layers"]),
            batch_first=True,
            dropout=float(config["dropout"]) if int(config["num_layers"]) > 1 else 0.0,
        )

    def forward(self, features, candidate_mask, frame_mask, delta_seconds, lengths,
                age_seconds=None, motion_features=None):
        del frame_mask, age_seconds, motion_features
        embeddings, frame_input = self.encode_frames(features, candidate_mask, delta_seconds)
        packed = pack_padded_sequence(
            frame_input, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.temporal(packed)
        context = hidden[-1]
        return self.score_current(context, embeddings, candidate_mask, lengths)


class TransformerCandidateSelector(TemporalSelectorBase):
    def __init__(self, config):
        super().__init__(config)
        hidden = int(config["hidden_dim"])
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(config["num_heads"]),
            dim_feedforward=int(config["feedforward_dim"]),
            dropout=float(config["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=int(config["num_layers"]))
        self.temporal_position = nn.Embedding(int(config["history_length"]), hidden)
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, features, candidate_mask, frame_mask, delta_seconds, lengths,
                age_seconds=None, motion_features=None):
        del age_seconds, motion_features
        embeddings, frame_input = self.encode_frames(features, candidate_mask, delta_seconds)
        positions = torch.arange(frame_input.shape[1], device=frame_input.device)
        encoded = self.temporal(
            frame_input + self.temporal_position(positions).unsqueeze(0),
            src_key_padding_mask=~frame_mask,
        )
        context = self.output_norm(self.current_tensor(encoded, lengths))
        return self.score_current(context, embeddings, candidate_mask, lengths)


class CandidatePreservingTransformerSelector(nn.Module):
    """Keep every T x K candidate and query history separately for each current candidate."""

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        hidden = int(config["hidden_dim"])
        embedding = int(config["candidate_embedding_dim"])
        if hidden != embedding:
            raise ValueError("candidate transformer requires embedding dim == hidden dim")
        self.uses_motion = config["architecture"] == "candidate_motion_transformer"
        if self.uses_motion and int(config.get("motion_feature_dimension", -1)) != MOTION_FEATURE_DIMENSION:
            raise ValueError("candidate motion feature dimension is not the frozen contract")
        input_dimension = FEATURE_DIMENSION + 1
        if self.uses_motion:
            input_dimension += MOTION_FEATURE_DIMENSION
        self.token_encoder = nn.Sequential(
            nn.Linear(input_dimension, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.temporal_position = nn.Embedding(int(config["history_length"]), hidden)
        self.rank_position = nn.Embedding(TOP_K, hidden)
        self.cold_start_token = nn.Parameter(torch.zeros(hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(config["num_heads"]),
            dim_feedforward=int(config["feedforward_dim"]),
            dropout=float(config["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(config["num_layers"]))
        self.current_to_history = nn.MultiheadAttention(
            hidden, int(config["num_heads"]), dropout=float(config["dropout"]),
            batch_first=True)
        self.output_norm = nn.LayerNorm(hidden)
        self.candidate_scorer = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(float(config["dropout"])),
            nn.Linear(hidden, 1),
        )
        self.no_lock_scorer = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(float(config["dropout"])),
            nn.Linear(hidden, 1),
        )

    def forward(self, features, candidate_mask, frame_mask, delta_seconds, lengths,
                age_seconds=None, motion_features=None):
        del frame_mask, delta_seconds
        if age_seconds is None:
            age_seconds = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device)
        token_fields = [features, age_seconds.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, TOP_K, -1)]
        if self.uses_motion:
            if motion_features is None:
                raise ValueError("candidate motion transformer requires motion features")
            token_fields.append(motion_features)
        tokens = self.token_encoder(torch.cat(token_fields, dim=-1))
        temporal_positions = torch.arange(features.shape[1], device=features.device)
        ranks = torch.arange(TOP_K, device=features.device)
        tokens = (tokens
                  + self.temporal_position(temporal_positions)[None, :, None, :]
                  + self.rank_position(ranks)[None, None, :, :])

        batch_indices = torch.arange(features.shape[0], device=features.device)
        current_positions = lengths - 1
        current_tokens = tokens[batch_indices, current_positions]
        current_mask = candidate_mask[batch_indices, current_positions]

        time_positions = torch.arange(features.shape[1], device=features.device)
        historical_frames = time_positions.unsqueeze(0) < current_positions.unsqueeze(1)
        historical_mask = candidate_mask & historical_frames.unsqueeze(-1)
        history = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        history_mask = historical_mask.reshape(historical_mask.shape[0], -1)
        cold = self.cold_start_token.view(1, 1, -1).expand(tokens.shape[0], -1, -1)
        history = torch.cat([cold, history], dim=1)
        history_mask = torch.cat([
            torch.ones(tokens.shape[0], 1, dtype=torch.bool, device=tokens.device),
            history_mask,
        ], dim=1)
        encoded_history = self.history_encoder(
            history, src_key_padding_mask=~history_mask)
        attended, _ = self.current_to_history(
            current_tokens, encoded_history, encoded_history,
            key_padding_mask=~history_mask, need_weights=False)
        attended = self.output_norm(current_tokens + attended)
        comparison = torch.cat([
            current_tokens,
            attended,
            torch.abs(current_tokens - attended),
            current_tokens * attended,
        ], dim=-1)
        candidate_logits = self.candidate_scorer(comparison).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(~current_mask, -1e9)
        valid = current_mask.to(current_tokens.dtype).unsqueeze(-1)
        current_summary = (current_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        history_summary = encoded_history[:, 0]
        no_lock = self.no_lock_scorer(
            torch.cat([current_summary, history_summary], dim=-1))
        return torch.cat([candidate_logits, no_lock], dim=1)


def build_temporal_model(config):
    architecture = config["architecture"]
    if architecture == "gru":
        return GRUCandidateSelector(config)
    if architecture == "transformer":
        return TransformerCandidateSelector(config)
    if architecture in ("candidate_transformer", "candidate_motion_transformer"):
        return CandidatePreservingTransformerSelector(config)
    raise ValueError("unsupported temporal architecture: %s" % architecture)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def selection_utility(metrics):
    frames = int(metrics["frames_with_ground_truth"])
    if frames <= 0:
        raise ValueError("selection utility requires frames")
    return ((int(metrics["hit_frames_iou_ge_threshold"])
             - int(metrics["proxy_false_lock_frames"])) / frames)
