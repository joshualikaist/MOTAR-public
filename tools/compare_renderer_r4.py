#!/usr/bin/env python3
"""Combine two independently produced R4 run.json files under the frozen rule."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if len(args.run) != 2 or args.run[0].resolve() == args.run[1].resolve():
        raise ValueError("Exactly two different run paths are required")
    if args.output.exists():
        raise FileExistsError("Comparison output must not exist")
    runs = [json.loads(path.read_text()) for path in args.run]
    exact_fields = ("schema", "git_commit", "seed", "device", "camera", "generic_kernel_sha256",
                    "fixture_name", "stage",
                    "per_scene", "summary", "checks", "fixture",
                    "within_bin_ratio_lambertian_over_depth", "thresholds", "array_sha256", "run_verdict")
    comparisons = {field: runs[0].get(field) == runs[1].get(field) for field in exact_fields}
    runtime_axes = ("python", "executable", "platform", "torch", "cuda", "cudnn", "device_name",
                    "device_capability", "warp")
    runtime_equal = {axis: runs[0]["runtime"].get(axis) == runs[1]["runtime"].get(axis)
                     for axis in runtime_axes}
    passed = (all(comparisons.values()) and all(runtime_equal.values())
              and all(run["run_verdict"] == "PASS" for run in runs))
    output = {"schema": "independent_renderer_r4_comparison_v1",
              "runs": [str(path) for path in args.run], "exact_comparisons": comparisons,
              "runtime_axes_equal": runtime_equal, "verdict": "PASS" if passed else "FAIL"}
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output["verdict"], args.output)


if __name__ == "__main__":
    main()
