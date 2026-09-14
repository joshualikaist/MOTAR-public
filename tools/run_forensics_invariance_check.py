#!/usr/bin/env python3
"""Does turning the episode forensics on change the simulated trajectory? Measure it.

Three runs of the same audited evaluation environment, at the same seed and density:

    off_a   forensics off, digest on
    off_b   forensics off, digest on          -- what the harness itself reproduces
    on      forensics on,  digest on          -- the question

`off_a == off_b` establishes that this harness is reproducible at all; without it an `off != on`
difference would be uninterpretable. `off_a == on` is then the invariance claim. The digest is
enabled in every run, so it cancels out of the comparison.

The evaluation environment is not invented here: it is the same closed environment the D8b
sensitivity launcher used, reached through the same loader, so this check runs the audited contract
rather than a lookalike. Nothing is trained, and no policy, reward or controller is touched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "aerial_gym/rl_training/rl_games/eval_navrl_v2_density_sweep.sh"
DIGEST_PREFIX = "NAVRL_TRAJECTORY_DIGEST "
ARMS = ("off_a", "off_b", "on")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_env(base, arm, output, seed, bars, forensics_json):
    env = dict(base)
    # The shared source bundle of the older experiment pins that experiment's tree. Reusing it here
    # makes the evaluator refuse, correctly, as soon as any repository file has moved on since. Each
    # arm snapshots the current tree instead; the three arms run back to back, so they share it.
    env.pop("NAVRL_V2_SHARED_SOURCE_BUNDLE", None)
    env.update({
        "NAVRL_SEED": str(seed),
        "NAVRL_V2_DENSITIES": str(bars),
        "NAVRL_V2_RESULT_DIR": str(output),
        # The digest label enters the hash, so it must be identical in every arm.
        "NAVRL_TRAJECTORY_DIGEST": "1",
        "NAVRL_TRAJECTORY_DIGEST_LABEL": "",
        "NAVRL_TRAJECTORY_DIGEST_JSON": str(Path(output).resolve().parent
                                            / ("%s_trajectory_digest.json" % arm)),
        "NAVRL_EPISODE_FORENSICS": "1" if arm == "on" else "0",
    })
    if arm == "on":
        env["NAVRL_EPISODE_FORENSICS_JSON"] = str(forensics_json)
    else:
        env.pop("NAVRL_EPISODE_FORENSICS_JSON", None)
    return env


def read_digest(log_path, json_path):
    """Prefer the digest file. The runner installs a stdout filter with an allow-list of message
    prefixes, so a printed digest line never reaches the log; the file is the reliable channel."""
    path = Path(json_path)
    if path.is_file():
        return json.loads(path.read_text())
    for line in Path(log_path).read_text(errors="replace").splitlines():
        if line.startswith(DIGEST_PREFIX):
            return json.loads(line[len(DIGEST_PREFIX):])
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--bars", type=int, default=70)
    parser.add_argument("--seed", type=int, default=593)
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    arguments = parser.parse_args(argv)
    # Absolute: the evaluator runs from its own directory, so a relative output path would make
    # the task write its digest beside the launcher script instead of into the result directory.
    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    envelope = load_module("forensics_invariance_env",
                           ROOT / "tools/run_navrl_distractor_envelope.py")
    base = envelope.evaluation_env("default_n0", preflight=False, force=False)
    checkpoint = str(Path(arguments.checkpoint).resolve())
    runs = {}
    for arm in arguments.arms:
        # The evaluator refuses to overwrite an existing result directory, so it creates its own;
        # the launcher's own artefacts live beside it rather than inside it.
        directory = output / arm
        log_path = output / ("%s.log" % arm)
        env = arm_env(base, arm, directory, arguments.seed, arguments.bars,
                      (output / ("%s_episode_forensics.json" % arm)).resolve())
        started = datetime.now(timezone.utc).isoformat()
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                [str(EVALUATOR), checkpoint, str(arguments.games)],
                cwd=str(EVALUATOR.parent), env=env, stdout=stream,
                stderr=subprocess.STDOUT)
        digest = read_digest(log_path, env["NAVRL_TRAJECTORY_DIGEST_JSON"])
        runs[arm] = {"arm": arm, "returncode": completed.returncode, "started_utc": started,
                     "log": str(log_path), "digest": digest,
                     "forensics_json": (str(output / ("%s_episode_forensics.json" % arm))
                                        if arm == "on" else None)}
        print("%-6s exit=%d digest=%s" % (arm, completed.returncode,
                                          (digest or {}).get("sha256", "MISSING")), flush=True)
    hashes = {arm: (runs[arm]["digest"] or {}).get("sha256") for arm in runs}
    reproducible = ("off_a" in hashes and "off_b" in hashes
                    and hashes["off_a"] is not None and hashes["off_a"] == hashes["off_b"])
    invariant = ("off_a" in hashes and "on" in hashes
                 and hashes["off_a"] is not None and hashes["off_a"] == hashes["on"])
    if not reproducible:
        verdict = "HARNESS_NOT_REPRODUCIBLE"
    elif invariant:
        verdict = "TRAJECTORY_INVARIANT"
    else:
        verdict = "TRAJECTORY_CHANGED"
    forensics_summary = None
    forensics_path = output / "on_episode_forensics.json"
    if forensics_path.is_file():
        forensics_summary = json.loads(forensics_path.read_text())["summary"]
    summary = {
        "experiment": "TD trajectory invariance",
        "forensics_smoke_summary": forensics_summary,
        "forensics_smoke_scope": "One density and one seed, produced only to show the recorder "
                                 "runs end to end. This is NOT the preregistered TD-T1 result, "
                                 "which needs the full density axis and the existing protocol.",
        "verdict": verdict,
        "question": "Does enabling the evaluation-only episode forensics change the simulated "
                    "trajectory?",
        "runs": runs,
        "digests": hashes,
        "off_runs_reproduce_each_other": reproducible,
        "forensics_leaves_trajectory_unchanged": invariant,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "evaluator": str(EVALUATOR.relative_to(ROOT)),
        "evaluator_sha256": sha256_file(EVALUATOR),
        "games_per_arm": arguments.games, "bars": arguments.bars, "seed": arguments.seed,
        "scope": "Evaluation-only instrumentation check; no policy, reward, detector, controller "
                 "or termination change, and no training",
        "causality_vs_d8b": "NOT_TESTED",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                         encoding="utf-8")
    print("verdict: %s -> %s" % (verdict, output / "summary.json"))
    return 0 if verdict == "TRAJECTORY_INVARIANT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
