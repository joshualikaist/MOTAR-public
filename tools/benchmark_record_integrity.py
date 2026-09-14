#!/usr/bin/env python3
"""Synthetic CPU serialization/hash/parse/validation cost. Never imports a simulator or model."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
import tracemalloc

from check_record_integrity import audit_bytes, strict_loads, write_exclusive

COUNTS = (128, 10000, 100000)
REPEATS = 3
SAMPLES = 5


def fixture(count):
    contract = {"schema_version": 1, "records_key": "records", "expected_count": count,
                "fields": {"row_id": {"type": "integer", "minimum": 0},
                           "seed": {"type": "integer"}, "artifact_sha256": {"type": "string", "format": "sha256"},
                           "value": {"type": "number"}, "valid": {"type": "boolean"}},
                "unique_fields": ["row_id"], "context_key": "context",
                "context_fields": ["seed", "artifact_sha256"]}
    document = {"context": {"seed": 20260914, "artifact_sha256": "a"*64},
                "records": [{"row_id": i, "seed": 20260914, "artifact_sha256": "a"*64,
                             "value": (i % 1024)/1024, "valid": True} for i in range(count)]}
    return document, contract


def encode(document):
    return json.dumps(document, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered)-1)*q
    lower = int(position)
    upper = min(lower+1, len(ordered)-1)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)


def distribution(values):
    return {"n": len(values), "mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
            "p95_ms": percentile(values, .95)}


def benchmark():
    rows = []
    for count in COUNTS:
        print("[record benchmark] synthetic rows:", count, flush=True)
        document, contract = fixture(count)
        blob = encode(document)
        digest = hashlib.sha256(blob).hexdigest()
        if audit_bytes(blob, contract)["status"] != "VALID_FOR_DECLARED_CONTRACT":
            raise RuntimeError("Invalid synthetic fixture")
        operations = {
            "serialize": lambda: encode(document),
            "sha256": lambda: hashlib.sha256(blob).hexdigest(),
            "strict_parse": lambda: strict_loads(blob),
            "hash_parse_validate": lambda: audit_bytes(blob, contract),
            "serialize_hash_parse_validate": lambda: audit_bytes(encode(document), contract),
        }
        # Same process: repeated timing batches, not independent-process replication.
        raw = {name: [] for name in operations}
        for repeat in range(REPEATS):
            order = list(operations)[repeat:] + list(operations)[:repeat]
            for name in order:
                operations[name]()  # One unmeasured warmup per repeat/mode.
                samples = []
                for _ in range(SAMPLES):
                    start = time.perf_counter()
                    output = operations[name]()
                    elapsed = (time.perf_counter()-start)*1000
                    # Destruction/GC outside the timer; no disk or fsync cost is claimed.
                    del output
                    samples.append(elapsed)
                raw[name].append(samples)
        gc.collect()
        tracemalloc.start()
        audited = audit_bytes(blob, contract)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if hashlib.sha256(blob).hexdigest() != digest or audited["status"] != "VALID_FOR_DECLARED_CONTRACT":
            raise RuntimeError("Fixture changed during benchmark")
        rows.append({"records": count, "bytes": len(blob), "fixture_sha256": digest,
                     "raw_ms": raw,
                     "timing": {name: distribution([v for repeat in values for v in repeat]) for name, values in raw.items()},
                     "audit_python_peak_mib": peak/1024**2,
                     "process_high_water_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
    return {"status": "SYNTHETIC_CPU_COST_CHARACTERIZED", "cells": rows,
            "scope": "Synthetic record I/O in memory; no GPU, model, simulator, recorder hooks or domain metrics",
            "repeats": REPEATS, "samples_per_repeat": SAMPLES, "warmup_per_mode_per_repeat": 1,
            "replication": "Repeated timing batches in one process, not independent process/seed experiments",
            "timing_boundary": "Function return; includes allocations, excludes destruction/GC and disk writes",
            "memory_boundary": "tracemalloc audit measured separately outside timers; RSS cumulative process high-water on Linux",
            "not_estimated": ["full evaluation duration", "live instrumentation overhead", "GPU throughput", "disk/fsync cost"],
            "runtime": {"python": platform.python_version(), "executable": sys.executable, "platform": platform.platform()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Never overwrite a benchmark")
    start = time.perf_counter()
    result = benchmark()
    result["elapsed_seconds"] = time.perf_counter()-start
    result["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__), Path(__file__).with_name("check_record_integrity.py"))}
    write_exclusive(args.output, result)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
