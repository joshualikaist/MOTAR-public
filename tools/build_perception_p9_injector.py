#!/usr/bin/env python3
"""Compile and validate the preregistered P9 empirical perception injector."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys


STATES = ("HIT", "FALSE_LOCK", "NO_LOCK")
EDGES = (8.0, 16.0, 32.0, 64.0)
SUPPORTED = (1, 2, 3)
FALLBACK = (1, 1, 2, 3, 3)
REPLICAS = 64
SEED = 1701


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values, q):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = q * (len(values) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - position) + values[hi] * (position - lo)


def ks_distance(left, right):
    left, right = sorted(left), sorted(right)
    if not left or not right:
        return None
    i = j = 0
    distance = 0.0
    while i < len(left) or j < len(right):
        if j >= len(right) or (i < len(left) and left[i] <= right[j]):
            value = left[i]
        else:
            value = right[j]
        while i < len(left) and left[i] <= value:
            i += 1
        while j < len(right) and right[j] <= value:
            j += 1
        distance = max(distance, abs(i / len(left) - j / len(right)))
    return distance


def categorical(rng, probabilities):
    draw = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw <= cumulative:
            return index
    return len(probabilities) - 1


def stationary_balance(probabilities, occupancy):
    """Minimum-KL flow balancing with row and column marginals fixed to occupancy."""
    n = len(occupancy)
    flow = [
        [max(1e-12, float(occupancy[i]) * float(probabilities[i][j])) for j in range(n)]
        for i in range(n)
    ]
    for _ in range(10000):
        for i in range(n):
            scale = occupancy[i] / sum(flow[i])
            flow[i] = [value * scale for value in flow[i]]
        columns = [sum(flow[i][j] for i in range(n)) for j in range(n)]
        for j in range(n):
            scale = occupancy[j] / columns[j]
            for i in range(n):
                flow[i][j] *= scale
        row_error = max(abs(sum(flow[i]) - occupancy[i]) for i in range(n))
        column_error = max(
            abs(sum(flow[i][j] for i in range(n)) - occupancy[j]) for j in range(n)
        )
        if max(row_error, column_error) < 1e-13:
            break
    rows = [[flow[i][j] / occupancy[i] for j in range(n)] for i in range(n)]
    return rows


def load_runtime_helpers(repo):
    path = repo / "aerial_gym/task/navrl_task/navrl_empirical_error.py"
    spec = importlib.util.spec_from_file_location("navrl_empirical_error_p9_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_model(error_model, frames, p8_hashes):
    bins = {int(entry["index"]): entry for entry in error_model["bins"]}
    latency_by_bin = defaultdict(list)
    for frame in frames:
        if frame.get("conditional_eligible"):
            latency_by_bin[int(frame["size_bin"])].extend(frame["latency_ms_repeats"])
    conditional_transitions = defaultdict(list)
    previous = None
    for frame in frames:
        if not frame.get("conditional_eligible"):
            previous = None
            continue
        if (previous is not None and previous["sequence"] == frame["sequence"]
                and 0 < (frame["timestamp_ns"] - previous["timestamp_ns"]) / 1e9 <= 0.5):
            key = (int(previous["size_bin"]), previous["state"], int(frame["size_bin"]))
            conditional_transitions[key].append(
                (frame["state"], (frame["timestamp_ns"] - previous["timestamp_ns"]) / 1e9)
            )
        previous = frame

    def nearest_row(target_bin, state_index):
        candidates = []
        for source_bin in SUPPORTED:
            transition = bins[source_bin]["transition_3state"]
            if transition["row_supported"][state_index]:
                candidates.append(source_bin)
        if not candidates:
            raise RuntimeError(f"no supported transition row for {STATES[state_index]}")
        return min(candidates, key=lambda value: (abs(value - target_bin), value))

    def nearest_offsets(target_bin, state):
        candidates = [
            source_bin for source_bin in SUPPORTED
            if bins[source_bin]["error"][state]["joint_du_dv_normalized_samples"]
        ]
        if not candidates:
            raise RuntimeError(f"no paired offsets for {state}")
        return min(candidates, key=lambda value: (abs(value - target_bin), value))

    compiled_bins = []
    fallback_rows = []
    for target_bin in SUPPORTED:
        source = bins[target_bin]
        state_counts = source["state_counts"]
        total = sum(state_counts[state] for state in STATES)
        occupancy = [state_counts[state] / total for state in STATES]
        empirical_rows = []
        empirical_sources = []
        for state_index, state in enumerate(STATES):
            row_bin = nearest_row(target_bin, state_index)
            empirical_sources.append(row_bin)
            empirical_rows.append(bins[row_bin]["transition_3state"]["probabilities"][state_index])
        balanced_rows = stationary_balance(empirical_rows, occupancy)
        rows = []
        for state_index, state in enumerate(STATES):
            row_bin = empirical_sources[state_index]
            transition = bins[row_bin]["transition_3state"]
            intervals = [
                float(sample[2]) for sample in transition["samples"]
                if sample[0] == state
            ]
            if not intervals:
                raise RuntimeError(f"no cadence samples for bin {row_bin}/{state}")
            rows.append({
                "requested_bin": target_bin,
                "source_bin": row_bin,
                "source_state": state,
                "outgoing_observations": int(sum(transition["counts"][state_index])),
                "empirical_probabilities": empirical_rows[state_index],
                "probabilities": balanced_rows[state_index],
                "max_abs_stationary_adjustment": max(
                    abs(a - b) for a, b in zip(
                        empirical_rows[state_index], balanced_rows[state_index]
                    )
                ),
                "reference_dt_s": quantile(intervals, 0.5),
            })
            if row_bin != target_bin:
                fallback_rows.append({"target_bin": target_bin, "state": state, "source_bin": row_bin})
        destination_rows = {}
        for destination_bin in SUPPORTED:
            destination_rows[str(destination_bin)] = []
            for state_index, state in enumerate(STATES):
                samples = conditional_transitions[(target_bin, state, destination_bin)]
                fallback = rows[state_index]
                destination_rows[str(destination_bin)].append({
                    **fallback,
                    "requested_source_bin": target_bin,
                    "requested_destination_bin": destination_bin,
                    "destination_cell_observations": len(samples),
                    "destination_conditioning": False,
                })
        offsets = {}
        for state in STATES[:2]:
            offset_bin = nearest_offsets(target_bin, state)
            offsets[state] = {
                "source_bin": offset_bin,
                "joint_du_dv_normalized_samples": bins[offset_bin]["error"][state][
                    "joint_du_dv_normalized_samples"
                ],
            }
        compiled_bins.append({
            "target_bin": target_bin,
            "initial_state_probabilities": occupancy,
            "transition_rows": rows,
            "transition_rows_by_destination_bin": destination_rows,
            "offsets": offsets,
            "latency_source_bin": target_bin,
            "latency_ms_samples": latency_by_bin[target_bin],
        })
    return {
        "schema_version": 1,
        "name": "p9_nps_validation_pixel_temporal_v1",
        "calibration_split": "NPS validation single-GT only",
        "test_used": False,
        "states": list(STATES),
        "size_definition": "sqrt(clean target bbox area) in detector pixels",
        "size_edges_px": list(EDGES),
        "supported_bins": list(SUPPORTED),
        "fallback_bin_by_raw_bin": list(FALLBACK),
        "unsupported_bin_policy": "nearest supported bin; ties choose lower bin",
        "transition_row_fallbacks": fallback_rows,
        "destination_condition_min_observations": 10,
        "transition_estimator": "minimum-KL stationary flow balancing",
        "cadence_conversion": "competing-risk p_stay(dt)=p_stay(ref_dt)**(dt/ref_dt)",
        "latency_rounding": "floor(latency_ms/(1000*step_dt_s)+0.5)",
        "metric_range_error": "unsupported; retain clean analytic surface range",
        "false_lock_range": "unsupported; retain clean analytic surface range",
        "confidence_error": "unsupported; retain clean analytic confidence",
        "p8_input_sha256": p8_hashes,
        "bins": compiled_bins,
        "latency_ms_samples_by_source_bin": {
            str(index): latency_by_bin[index] for index in SUPPORTED
        },
    }


def simulate(model, frames, seed, helpers, keep_trace=False):
    rng = random.Random(seed)
    bins = {entry["target_bin"]: entry for entry in model["bins"]}
    output = []
    previous = None
    current_state = None
    for frame in frames:
        if not frame.get("conditional_eligible"):
            previous = current_state = None
            continue
        raw_bin = int(frame["size_bin"])
        target_bin = model["fallback_bin_by_raw_bin"][raw_bin]
        entry = bins[target_bin]
        same_segment = (
            previous is not None
            and previous["sequence"] == frame["sequence"]
            and 0 < (frame["timestamp_ns"] - previous["timestamp_ns"]) / 1e9 <= 0.5
        )
        if not same_segment:
            current_state = categorical(rng, entry["initial_state_probabilities"])
        else:
            dt = (frame["timestamp_ns"] - previous["timestamp_ns"]) / 1e9
            source_entry = bins[model["fallback_bin_by_raw_bin"][int(previous["size_bin"])]]
            row = source_entry["transition_rows_by_destination_bin"][str(target_bin)][current_state]
            probabilities = helpers.cadence_adjusted_row(
                row["probabilities"], current_state, row["reference_dt_s"], dt
            )
            current_state = categorical(rng, probabilities)
        offset = None
        if current_state < 2:
            values = entry["offsets"][STATES[current_state]]["joint_du_dv_normalized_samples"]
            offset = values[rng.randrange(len(values))]
        latency = entry["latency_ms_samples"][rng.randrange(len(entry["latency_ms_samples"]))]
        row = {
            "frame_id": frame["frame_id"], "sequence": frame["sequence"],
            "timestamp_ns": frame["timestamp_ns"], "raw_bin": raw_bin,
            "target_bin": target_bin, "state": STATES[current_state],
            "offset": offset, "latency_ms": latency,
        }
        output.append(row)
        previous = frame
    if keep_trace:
        encoded = "\n".join(json.dumps(row, sort_keys=True) for row in output).encode()
        return output, hashlib.sha256(encoded).hexdigest()
    return output


def summarize_validation(model, frames, helpers):
    generated = []
    first_trace, hash_a = simulate(model, frames, SEED, helpers, keep_trace=True)
    _, hash_b = simulate(model, frames, SEED, helpers, keep_trace=True)
    _, hash_c = simulate(model, frames, SEED + 1, helpers, keep_trace=True)
    generated.extend(first_trace)
    for replica in range(1, REPLICAS):
        generated.extend(simulate(model, frames, SEED + replica, helpers))

    bins = {entry["target_bin"]: entry for entry in model["bins"]}
    reports = []
    pass_all = hash_a == hash_b and hash_a != hash_c
    for target_bin in SUPPORTED:
        real = [frame for frame in frames if frame.get("conditional_eligible") and frame["size_bin"] == target_bin]
        synth = [row for row in generated if row["raw_bin"] == target_bin]
        real_counts = [sum(frame["state"] == state for frame in real) for state in STATES]
        synth_counts = [sum(row["state"] == state for row in synth) for state in STATES]
        real_prob = [value / sum(real_counts) for value in real_counts]
        synth_prob = [value / sum(synth_counts) for value in synth_counts]
        occupancy_tv = 0.5 * sum(abs(a - b) for a, b in zip(real_prob, synth_prob))

        transition_errors = []
        for state_index, row in enumerate(bins[target_bin]["transition_rows"]):
            if row["source_bin"] != target_bin:
                continue
            counts = [0, 0, 0]
            outgoing = 0
            for left, right in zip(generated, generated[1:]):
                if (left["raw_bin"] == target_bin and left["sequence"] == right["sequence"]
                        and left["state"] == STATES[state_index]
                        and 0 < (right["timestamp_ns"] - left["timestamp_ns"]) / 1e9 <= 0.5):
                    counts[STATES.index(right["state"])] += 1
                    outgoing += 1
            if outgoing:
                generated_prob = [value / outgoing for value in counts]
                target_prob = row.get("empirical_probabilities", row["probabilities"])
                transition_errors.append({
                    "state": STATES[state_index], "outgoing": outgoing,
                    "target": target_prob, "generated": generated_prob,
                    "max_abs_error": max(abs(a - b) for a, b in zip(target_prob, generated_prob)),
                })
        ks = {}
        for state in STATES[:2]:
            real_offsets = [frame["center_offset_normalized"] for frame in real if frame["state"] == state]
            synth_offsets = [row["offset"] for row in synth if row["state"] == state]
            if real_offsets and synth_offsets:
                ks[state] = {
                    "du_norm": ks_distance([x[0] for x in real_offsets], [x[0] for x in synth_offsets]),
                    "dv_norm": ks_distance([x[1] for x in real_offsets], [x[1] for x in synth_offsets]),
                    "radial_norm": ks_distance(
                        [math.hypot(*x) for x in real_offsets],
                        [math.hypot(*x) for x in synth_offsets],
                    ),
                }
        real_latency = [value for frame in real for value in frame["latency_ms_repeats"]]
        synth_latency = [row["latency_ms"] for row in synth]
        latency_ks = ks_distance(real_latency, synth_latency)
        bin_pass = (
            occupancy_tv <= 0.05
            and all(item["max_abs_error"] <= 0.06 for item in transition_errors)
            and all(value <= 0.10 for state in ks.values() for value in state.values())
            and latency_ks <= 0.05
        )
        pass_all &= bin_pass
        reports.append({
            "bin": target_bin, "real_frames": len(real), "generated_frames": len(synth),
            "state_occupancy_tv": occupancy_tv, "transition_rows": transition_errors,
            "offset_ks": ks, "latency_ks": latency_ks, "passed": bin_pass,
        })
    return {
        "schema_version": 1,
        "replicas": REPLICAS,
        "seeds": [SEED, SEED + REPLICAS - 1],
        "thresholds": {"occupancy_tv": 0.05, "transition_max_abs": 0.06,
                       "offset_ks": 0.10, "latency_ks": 0.05},
        "reproducibility": {"same_seed_sha256_a": hash_a, "same_seed_sha256_b": hash_b,
                            "different_seed_sha256": hash_c,
                            "same_seed_identical": hash_a == hash_b,
                            "different_seed_changed": hash_a != hash_c},
        "bins": reports,
        "passed": bool(pass_all),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p8-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing existing output directory: {args.output_dir}")
    error_path = args.p8_dir / "error_model.json"
    frames_path = args.p8_dir / "frames.jsonl"
    receipt_path = args.p8_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for name, path in (("error_model.json", error_path), ("frames.jsonl", frames_path)):
        expected = receipt["output_sha256"][name]
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"P8 {name} SHA mismatch: {actual} != {expected}")
    error_model = json.loads(error_path.read_text(encoding="utf-8"))
    frames = [json.loads(line) for line in frames_path.read_text(encoding="utf-8").splitlines()]
    if error_model.get("test_used") is not False or receipt.get("test_used") is not False:
        raise SystemExit("P8 inputs must be validation-only")
    model = compile_model(error_model, frames, {
        "error_model.json": sha256_file(error_path), "frames.jsonl": sha256_file(frames_path),
        "receipt.json": sha256_file(receipt_path),
    })
    repo = Path(__file__).resolve().parents[1]
    helpers = load_runtime_helpers(repo)
    validation = summarize_validation(model, frames, helpers)
    args.output_dir.mkdir(parents=True)
    model_path = args.output_dir / "p9_error_injector.json"
    validation_path = args.output_dir / "goodness_of_fit.json"
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_receipt = {
        "schema_version": 1, "stage": "P9", "test_used": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_sha256": sha256_file(Path(__file__)),
        "p8_receipt_sha256": sha256_file(receipt_path),
        "output_sha256": {model_path.name: sha256_file(model_path),
                          validation_path.name: sha256_file(validation_path)},
        "goodness_of_fit_passed": validation["passed"],
        "python": sys.version, "executable": sys.executable,
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(output_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not validation["passed"]:
        raise SystemExit("P9 goodness-of-fit gates failed; P10 is frozen")
    print(json.dumps({"model": str(model_path), "sha256": sha256_file(model_path),
                      "passed": True}, sort_keys=True))


if __name__ == "__main__":
    main()
