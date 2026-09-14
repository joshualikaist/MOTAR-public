#!/usr/bin/env python3
"""Explicit public/renderer/metadata regression allowlist; never launches policy evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from check_record_integrity import write_exclusive

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = (
    "test_record_integrity.py", "test_record_benchmark.py", "test_renderer*.py", "test_rc*.py",
    "test_public*.py", "test_cpu_install_evidence.py", "test_runtime_fingerprint.py",
    "test_repository_claims.py", "test_research_overview.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docs-python", default=sys.executable,
                        help="Separate interpreter with requirements-public-validation.txt; never installs into research environment")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selected = [p for pattern in PATTERNS for p in sorted((ROOT/"tests").glob(pattern))]
    if len(selected) != len(set(selected)):
        raise RuntimeError("Overlapping test selection")
    source = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in selected}
    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    env = dict(os.environ, PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    rows = []
    start = time.perf_counter()
    for pattern in PATTERNS:
        if not list((ROOT/"tests").glob(pattern)):
            raise RuntimeError("Empty required test selection: " + pattern)
        command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"]
        begin = time.perf_counter()
        run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        elapsed = time.perf_counter()-begin
        output = run.stdout+run.stderr
        path = args.output/(("%02d" % len(rows))+".log")
        with path.open("x") as stream:
            stream.write(output)
        count = re.search(r"^Ran (\d+) tests? in", output, re.M)
        skipped = re.search(r"^OK \(skipped=(\d+)\)", output, re.M)
        passed = run.returncode == 0 and count is not None and int(count.group(1)) > 0
        rows.append({"pattern": pattern, "command": command, "returncode": run.returncode,
                     "tests_run": int(count.group(1)) if count else None,
                     "skipped": int(skipped.group(1)) if skipped else 0,
                     "status": "PASS" if passed else "FAIL", "elapsed_seconds": elapsed,
                     "log": path.name, "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print("[regression]", pattern, rows[-1]["status"], rows[-1]["tests_run"], flush=True)
    for name, command in (("public_docs", [args.docs_python, "-B", "tools/check_public_docs.py", "--schema"]),
                           ("citation", [args.docs_python, "-c", "from cffconvert.cli.cli import cli; cli()", "--validate"]),
                           ("status_site", ["node", "tests/test_status_site.js"]),
                           ("public_manifest", ["node", "tests/test_public_status_manifest.js"])):
        begin = time.perf_counter()
        run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        path = args.output/(name+".log")
        with path.open("x") as stream:
            stream.write(run.stdout+run.stderr)
        rows.append({"check": name, "command": command, "returncode": run.returncode,
                     "status": "PASS" if run.returncode == 0 else "FAIL", "elapsed_seconds": time.perf_counter()-begin,
                     "log": path.name, "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        print("[public check]", name, rows[-1]["status"], flush=True)
    unchanged = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h for p, h in source.items())
    result = {"scope": "Explicit public/independent-renderer/metadata allowlist; NOT the whole repository suite",
              "status": "PASS" if unchanged and all(r["status"] == "PASS" for r in rows) else "FAIL",
              "tests_run": sum(r.get("tests_run") or 0 for r in rows), "skipped": sum(r.get("skipped", 0) for r in rows),
              "environment": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "PYTHONNOUSERSITE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
              "test_file_hashes": source, "test_sources_unchanged": unchanged,
              "runner_sha256": runner_sha, "docs_python": args.docs_python,
              "whole_repository_suite": "NOT_RUN", "policy_evaluation": "NOT_RUN",
              "checks": rows, "elapsed_seconds": time.perf_counter()-start}
    write_exclusive(args.output/"summary.json", result)
    print(json.dumps({k: result[k] for k in ("status", "tests_run", "skipped", "elapsed_seconds")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
