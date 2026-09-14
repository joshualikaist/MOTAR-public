#!/usr/bin/env python3
"""Run the offline instance adapter on a fixture. CPU only; no Isaac, SAM, or PPO.

This tool does not change the live perception path. Use --backend stub. --backend sam
is defined as a separate-process interface and refuses to load weights here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "aerial_gym/task/navrl_task/navrl_instance_adapter.py"


def _load_adapter():
    import importlib.util

    spec = importlib.util.spec_from_file_location("navrl_instance_adapter_offline", ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load %s" % ADAPTER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["navrl_instance_adapter_offline"] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="stub", choices=("stub", "sam"))
    parser.add_argument("--fixture", default="two_blob")
    parser.add_argument("--ambiguous-margin", type=float, default=None)
    args = parser.parse_args(argv)

    adapter = _load_adapter()
    if args.fixture != "two_blob":
        print("unknown fixture %r (only two_blob is implemented)" % args.fixture, file=sys.stderr)
        return 2
    fixture = adapter.two_blob_fixture()
    try:
        instances = adapter.run_backend(args.backend, fixture["rgb"], fixture["depth"])
    except adapter.SamBackendNotInstalled as exc:
        print(str(exc), file=sys.stderr)
        return 2
    margin = (
        adapter.DEFAULT_AMBIGUOUS_MARGIN
        if args.ambiguous_margin is None
        else args.ambiguous_margin
    )
    decision = adapter.associate_and_decide(instances, ambiguous_margin=margin)
    score = adapter.colour_score(fixture["rgb"])
    mask = (score >= adapter.DEFAULT_PIXEL_THRESHOLD) & (
        fixture["depth"] < adapter.DEFAULT_MAX_DEPTH
    )
    ghost = adapter.union_collapse_centroid(mask, fixture["depth"])
    payload = {
        "backend": args.backend,
        "fixture": args.fixture,
        "n_instances": len(instances),
        "decision": decision.to_json(),
        "current_path_union": ghost,
        "claim": "CPU contract only; not a SAM or capture result",
        "adapter_enabled_env": adapter.adapter_enabled(),
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
