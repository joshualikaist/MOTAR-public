"""S1 shadow-mode launcher: structure-fix counterfactual on fresh seed 481.

Prereg: docs/prereg_2026-09-03_s1_structure_fix_shadow.md. Reuses the distractor-envelope
launcher's CLOSED evaluation environment builder by import, overriding exactly THREE fields:
the seed (479 -> 481), the result directory, and NAVRL_S1_SHADOW=1. Every other guarantee
(v7 checkpoint + threshold + narrow override, camera contract, goal band, gate-0 CPU checks)
is inherited by construction instead of being re-typed.

Modes:
  smoke      M1 bit-identity: two 50-episode runs (shadow off/on), obs dumps must hash equal.
  evaluate   run the four cells (v7_n0, v7_n1, v7_n3, v7_n5) that do not exist yet.
  summarize  validate cells, compute the preregistered RQ0 verdict, write summary.{md,json}.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
ENVELOPE = REPO / "tools" / "run_navrl_distractor_envelope.py"
SEED = 481
RESULT_ROOT = REPO / "results" / "navrl_s1_shadow_seed481"
CELLS = ("v7_n0", "v7_n1", "v7_n3", "v7_n5")
SMOKE_EPISODES = 50

# Frozen by the prereg (section 3). Reading them from the results is forbidden.
GATE_STRUCTURE_DOMINANT_BELOW = 0.30
GATE_RECOGNITION_DOMINANT_ABOVE = 0.60


def _load_envelope():
    spec = importlib.util.spec_from_file_location("distractor_envelope_for_s1", ENVELOPE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cell_env(mod, cell, out_dir, *, shadow, seed=SEED):
    env = mod.evaluation_env(cell, preflight=False)
    env["NAVRL_SEED"] = str(seed)
    env["NAVRL_V2_RESULT_DIR"] = str(out_dir)
    env["NAVRL_S1_SHADOW"] = "1" if shadow else "0"
    # The envelope pins the seed-479 shared source bundle, whose manifest records the 1650 Ti's
    # absolute repository_root -- unverifiable on this machine by construction. S1 gets its own
    # fresh immutable bundle, built by the sweep on first use and shared across the S1 cells.
    env["NAVRL_V2_SHARED_SOURCE_BUNDLE"] = str(RESULT_ROOT / "source_bundle")
    return env


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# npz fields that describe the DUMP FILE rather than the run: they differ between the two smoke
# arms by construction (each arm names its own file and process) and carry no evidence about the
# observation stream. Everything else -- obs, outcomes, episode bookkeeping -- must match bit
# for bit.
_M1_SELF_REFERENTIAL = frozenset({"run_obs_dump_path", "run_pid"})


def _obs_dump_digest(path):
    import numpy as np

    data = np.load(path, allow_pickle=True)
    digest = hashlib.sha256()
    for key in sorted(data.files):
        if key in _M1_SELF_REFERENTIAL:
            continue
        arr = np.ascontiguousarray(data[key])
        digest.update(key.encode())
        digest.update(str(arr.dtype).encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _require(cond, msg):
    if not cond:
        raise SystemExit(f"[s1-shadow] FAIL: {msg}")


def _gate0(mod):
    gate0 = mod.verify_prerequisites()
    _require(
        mod.gate0_static_passed(gate0),
        "envelope gate-0 failed; no S1 cell may be produced: " + mod.gate0_failure_report(gate0),
    )


def run_smoke(mod):
    """M1: the shadow path must not perturb one bit of the actor observation stream."""
    _gate0(mod)
    smoke_root = RESULT_ROOT / "m1_smoke"
    hashes = {}
    smoke_root.mkdir(parents=True, exist_ok=True)
    for label, shadow in (("off", False), ("on", True)):
        out_dir = smoke_root / label
        _require(not out_dir.exists(), f"refusing overwrite: {out_dir}")
        # The evaluator insists on creating NAVRL_V2_RESULT_DIR itself; dump and log live
        # beside it so nothing pre-creates the directory.
        dump = smoke_root / f"frames_{label}.npz"
        _require(not dump.exists(), f"refusing overwrite: {dump}")
        env = _cell_env(mod, "v7_n5", out_dir, shadow=shadow)
        env["NAVRL_OBS_DUMP"] = str(dump)
        code = mod.tee_run(
            ["bash", str(mod.EVALUATOR), str(mod.CHECKPOINT), str(SMOKE_EPISODES)],
            env,
            smoke_root / f"smoke_{label}.log",
        )
        _require(code == 0, f"smoke {label}: evaluator exited {code}")
        _require(dump.is_file(), f"smoke {label}: no obs dump at {dump}")
        hashes[label] = _obs_dump_digest(dump)
    verdict = "PASS" if hashes["off"] == hashes["on"] else "FAIL_SHADOW_PERTURBS_OBS"
    payload = {"episodes": SMOKE_EPISODES, "obs_dump_sha256": hashes, "m1": verdict}
    (smoke_root / "m1.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[s1-shadow] M1 {verdict} | {hashes}")
    _require(verdict == "PASS", "M1 failed: shadow mode changed the observation stream")


def run_evaluate(mod, only=None):
    _gate0(mod)
    m1 = RESULT_ROOT / "m1_smoke" / "m1.json"
    _require(m1.is_file(), "run `smoke` first: M1 bit-identity evidence missing")
    _require(json.loads(m1.read_text())["m1"] == "PASS", "recorded M1 is not PASS")
    for cell in CELLS if only is None else [only]:
        out_dir = RESULT_ROOT / cell
        if out_dir.exists():
            print(f"[s1-shadow] cell {cell}: exists, skipping")
            continue
        env = _cell_env(mod, cell, out_dir, shadow=True)
        print(f"[s1-shadow] EVALUATE {cell} | seed {SEED} | {mod.EPISODES} episodes")
        code = mod.tee_run(
            ["bash", str(mod.EVALUATOR), str(mod.CHECKPOINT), str(mod.EPISODES)],
            env,
            out_dir.parent / f"{cell}.eval.log.partial",
        )
        _require(code == 0, f"{cell}: evaluator exited {code}")
        _require(out_dir.is_dir(), f"{cell}: evaluator produced no result directory")
        (out_dir.parent / f"{cell}.eval.log.partial").replace(out_dir / "eval.log")


def _load_cell(cell):
    path = RESULT_ROOT / cell / "70bars.json"
    _require(path.is_file(), f"missing result {path}")
    result = json.loads(path.read_text())
    cond = result.get("condition") or {}
    n = int(cell.rsplit("n", 1)[1])
    _require(int(cond.get("seed", -1)) == SEED, f"{cell}: seed {cond.get('seed')} != {SEED}")
    _require(int(cond.get("bars", -1)) == 70, f"{cell}: bars != 70")
    _require(int(cond.get("distractor_count", -1)) == n, f"{cell}: distractor_count != {n}")
    _require(cond.get("s1_shadow") is True, f"{cell}: shadow flag not recorded")
    _require(
        result.get("checkpoint_sha256") == _load_cell.checkpoint_sha,
        f"{cell}: policy checkpoint drifted",
    )
    shadow = result.get("s1_shadow")
    _require(isinstance(shadow, dict), f"{cell}: no s1_shadow block")
    online = result.get("distractor_lock")  # absent by design at n0
    return {"cell": cell, "n": n, "shadow": shadow, "online": online, "path": str(path)}


def run_summarize(mod):
    _load_cell.checkpoint_sha = mod.CHECKPOINT_SHA
    rows = [_load_cell(c) for c in CELLS]
    by_n = {r["n"]: r for r in rows}

    n5 = by_n[5]
    shadow_ftlr = n5["shadow"]["false_target_lock_rate"]
    online_ftlr = (n5["online"] or {}).get("false_target_lock_rate")
    _require(shadow_ftlr is not None and online_ftlr is not None, "n5 FTLR missing")
    if shadow_ftlr < GATE_STRUCTURE_DOMINANT_BELOW:
        verdict = "STRUCTURE_DOMINANT"
    elif shadow_ftlr > GATE_RECOGNITION_DOMINANT_ABOVE:
        verdict = "RECOGNITION_DOMINANT"
    else:
        verdict = "MIXED"

    summary = {
        "schema_version": 1,
        "experiment": "s1_shadow_seed481",
        "prereg": "docs/prereg_2026-09-03_s1_structure_fix_shadow.md",
        "seed": SEED,
        "policy_checkpoint_sha256": mod.CHECKPOINT_SHA,
        "preregistered_gate": {
            "structure_dominant_below": GATE_STRUCTURE_DOMINANT_BELOW,
            "recognition_dominant_above": GATE_RECOGNITION_DOMINANT_ABOVE,
            "primary": "shadow false_target_lock_rate at n=5",
        },
        "verdict_rq0": verdict,
        "online_ftlr_n5": online_ftlr,
        "shadow_ftlr_n5": shadow_ftlr,
        "rows": rows,
    }
    (RESULT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# S1 shadow screen (seed 481)",
        "",
        "| cell | online FTLR | shadow FTLR | shadow init-false | shadow tracking-false | shadow visible |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        s = r["shadow"]
        o = (r["online"] or {}).get("false_target_lock_rate")
        fmt = lambda x: ("-" if x is None else f"{x:.2%}")
        lines.append(
            f"| {r['cell']} | {fmt(o)} | {fmt(s['false_target_lock_rate'])} | "
            f"{fmt(s['init_false_lock_rate'])} | {fmt(s['tracking_false_lock_rate'])} | "
            f"{s['visible_frames']:,} |"
        )
    lines += [
        "",
        f"**RQ0 verdict: {verdict}** (shadow FTLR@n5 = {shadow_ftlr:.2%}, "
        f"online = {online_ftlr:.2%}; gates <30% / >60%)",
        "",
    ]
    (RESULT_ROOT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    mod = _load_envelope()
    if mode == "smoke":
        run_smoke(mod)
    elif mode == "evaluate":
        run_evaluate(mod, sys.argv[2] if len(sys.argv) > 2 else None)
    elif mode == "summarize":
        run_summarize(mod)
    else:
        raise SystemExit(f"usage: {sys.argv[0]} smoke|evaluate [cell]|summarize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
