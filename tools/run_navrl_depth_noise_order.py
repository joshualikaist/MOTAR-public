"""Phase 1: depth measurement-noise model ORDER screen, fresh seed 487.

Prereg: docs/prereg_2026-09-04_depth_noise_model_order.md. Reuses the distractor-envelope
launcher's CLOSED evaluation environment builder by import, exactly as the S1 shadow screen
does, so the checkpoint pin, detector pin, camera contract, goal band and gate-0 CPU checks
are inherited by construction rather than re-typed.

The cell that differs from the envelope is N=0 distractors: this experiment isolates the
measurement VARIANCE from the false-target-lock effect that the envelope already measured.

Modes:
  evaluate   run the three cells (linear, stereo_d455, stereo_d435) that do not exist yet.
  summarize  validate cells, compute the preregistered verdict, write summary.{md,json}.
"""

import importlib.util
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
ENVELOPE = REPO / "tools" / "run_navrl_distractor_envelope.py"
SEED = 487
RESULT_ROOT = REPO / "results" / "navrl_depth_noise_order_seed487"

# Frozen by the prereg (section 2). The envelope cell is v7 with zero distractors.
ENVELOPE_CELL = "v7_n0"
CELLS = {
    "linear": {},  # control: knob unset, sigma_r bit-identical to pre-2026-09-04
    "stereo_d455": {"NAVRL_DEPTH_NOISE_MODEL": "stereo", "NAVRL_DEPTH_STEREO_BASELINE_M": "0.095"},
    "stereo_d435": {"NAVRL_DEPTH_NOISE_MODEL": "stereo", "NAVRL_DEPTH_STEREO_BASELINE_M": "0.050"},
}
PRIMARY = "stereo_d455"  # the sensor the camera config models; d435 is a stress cell only

# Frozen by the prereg (section 4). Reading these off the results is forbidden.
GATE_MATTERS_PP = 10.0
GATE_INSENSITIVE_CI_UPPER_PP = 5.0


def _load_envelope():
    spec = importlib.util.spec_from_file_location("distractor_envelope_for_phase1", ENVELOPE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cell_env(mod, cell, out_dir):
    env = mod.evaluation_env(ENVELOPE_CELL, preflight=False)
    env["NAVRL_SEED"] = str(SEED)
    env["NAVRL_V2_RESULT_DIR"] = str(out_dir)
    # Same reason as the S1 screen: the envelope pins seed 479's shared bundle, whose manifest
    # records the 1650 Ti's absolute repository root and is unverifiable on this machine.
    env["NAVRL_V2_SHARED_SOURCE_BUNDLE"] = str(RESULT_ROOT / "source_bundle")
    env.update(CELLS[cell])
    return env


def _require(cond, msg):
    if not cond:
        raise SystemExit(f"[depth-order] FAIL: {msg}")


def run_evaluate(mod, only=None):
    gate0 = mod.verify_prerequisites()
    _require(
        mod.gate0_static_passed(gate0),
        "envelope gate-0 failed; no cell may be produced: " + mod.gate0_failure_report(gate0),
    )
    for cell in (CELLS if only is None else [only]):
        out_dir = RESULT_ROOT / cell
        if out_dir.exists():
            print(f"[depth-order] cell {cell}: exists, skipping")
            continue
        env = _cell_env(mod, cell, out_dir)
        print(f"[depth-order] EVALUATE {cell} | seed {SEED} | {mod.EPISODES} episodes")
        code = mod.tee_run(
            ["bash", str(mod.EVALUATOR), str(mod.CHECKPOINT), str(mod.EPISODES)],
            env,
            out_dir.parent / f"{cell}.eval.log.partial",
        )
        _require(code == 0, f"{cell}: evaluator exited {code}")
        _require(out_dir.is_dir(), f"{cell}: evaluator produced no result directory")
        (out_dir.parent / f"{cell}.eval.log.partial").replace(out_dir / "eval.log")


def _load_cell(cell, checkpoint_sha):
    path = RESULT_ROOT / cell / "70bars.json"
    _require(path.is_file(), f"missing result {path}")
    result = json.loads(path.read_text())
    cond = result.get("condition") or {}
    _require(int(cond.get("seed", -1)) == SEED, f"{cell}: seed {cond.get('seed')} != {SEED}")
    _require(int(cond.get("bars", -1)) == 70, f"{cell}: bars != 70")
    _require(int(cond.get("distractor_count", -1)) == 0, f"{cell}: distractor_count != 0")
    # M3: every cell must share the policy checkpoint.
    _require(
        result.get("checkpoint_sha256") == checkpoint_sha,
        f"{cell}: policy checkpoint drifted",
    )
    out = result["outcome"]
    return {
        "cell": cell,
        "episodes": int(result["actual_episodes"]),
        "capture_rate": float(out["capture_rate"]),
        "captured": int(out["captured"]),
        "crash_rate": float(out["crash_rate"]),
        "timeout_rate": float(out["timeout_rate"]),
        "path": str(path),
    }


def _delta_ci_pp(a, b):
    """95% CI for (a - b) on two independent binomial capture rates, in percentage points."""
    va = a["capture_rate"] * (1.0 - a["capture_rate"]) / a["episodes"]
    vb = b["capture_rate"] * (1.0 - b["capture_rate"]) / b["episodes"]
    se = math.sqrt(va + vb)
    delta = a["capture_rate"] - b["capture_rate"]
    half = 1.959963984540054 * se
    return delta * 100.0, (delta - half) * 100.0, (delta + half) * 100.0


def run_summarize(mod):
    rows = {c: _load_cell(c, mod.CHECKPOINT_SHA) for c in CELLS}
    control, primary = rows["linear"], rows[PRIMARY]
    delta_pp, lo_pp, hi_pp = _delta_ci_pp(control, primary)

    if lo_pp > 0.0 and delta_pp >= GATE_MATTERS_PP:
        verdict = "MODEL_ORDER_MATTERS"
    elif lo_pp > 0.0:
        verdict = "PARTIAL"
    elif hi_pp < GATE_INSENSITIVE_CI_UPPER_PP:
        verdict = "VARIANCE_INSENSITIVE"
    else:
        verdict = "INCONCLUSIVE"

    stress_delta_pp, stress_lo_pp, stress_hi_pp = _delta_ci_pp(control, rows["stereo_d435"])
    monotone = (
        control["capture_rate"] >= primary["capture_rate"] >= rows["stereo_d435"]["capture_rate"]
    )

    summary = {
        "schema_version": 1,
        "experiment": "depth_noise_model_order_seed487",
        "prereg": "docs/prereg_2026-09-04_depth_noise_model_order.md",
        "seed": SEED,
        "distractor_count": 0,
        "policy_checkpoint_sha256": mod.CHECKPOINT_SHA,
        "preregistered_gate": {
            "primary": "capture_rate(linear) - capture_rate(stereo_d455), percentage points",
            "matters_at_or_above_pp": GATE_MATTERS_PP,
            "insensitive_if_ci_upper_below_pp": GATE_INSENSITIVE_CI_UPPER_PP,
        },
        "verdict_rq": verdict,
        "primary_delta_pp": delta_pp,
        "primary_delta_ci95_pp": [lo_pp, hi_pp],
        "stress_delta_pp": stress_delta_pp,
        "stress_delta_ci95_pp": [stress_lo_pp, stress_hi_pp],
        "prediction_1_monotone_capture": monotone,
        "rows": rows,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Depth noise model order (Phase 1, seed 487, 0 distractors)",
        "",
        "| cell | capture | crash | timeout | episodes |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in CELLS:
        r = rows[c]
        lines.append(
            f"| {c} | {r['capture_rate']:.2%} | {r['crash_rate']:.2%} | "
            f"{r['timeout_rate']:.2%} | {r['episodes']:,} |"
        )
    lines += [
        "",
        f"**verdict_rq: {verdict}**",
        "",
        f"- primary (linear - stereo_d455): **{delta_pp:+.2f} pp** "
        f"95% CI [{lo_pp:+.2f}, {hi_pp:+.2f}]",
        f"- stress (linear - stereo_d435): {stress_delta_pp:+.2f} pp "
        f"95% CI [{stress_lo_pp:+.2f}, {stress_hi_pp:+.2f}]",
        f"- prediction 1 (capture monotone decreasing): "
        f"{'HELD' if monotone else 'VIOLATED'}",
        "",
        "Gates (frozen before measurement): MODEL_ORDER_MATTERS if delta >= 10 pp and CI excludes 0;",
        "PARTIAL if CI excludes 0 but delta < 10 pp; VARIANCE_INSENSITIVE if CI upper < 5 pp;",
        "otherwise INCONCLUSIVE.",
        "",
    ]
    (RESULT_ROOT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    mod = _load_envelope()
    if mode == "evaluate":
        run_evaluate(mod, sys.argv[2] if len(sys.argv) > 2 else None)
    elif mode == "summarize":
        run_summarize(mod)
    else:
        raise SystemExit(f"usage: {sys.argv[0]} evaluate [cell]|summarize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
