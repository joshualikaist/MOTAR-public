"""A8 readaptation report: preregistered contrasts and labels from *bars.json only.

Prereg: docs/prereg_2026-09-05_a8_filter_readaptation.md section 2. No simulation import.
"""

import argparse
import csv
import json
import math
from pathlib import Path

try:
    from .build_a5_ablation_table import _delta_ci
except ImportError:
    from build_a5_ablation_table import _delta_ci

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "results/navrl_a8_readaptation_seed521"
RUNS = REPO / "aerial_gym/rl_training/rl_games/runs"
SOURCE_RUN = RUNS / "ppo_260813_1636_navrl_v2-ref5in-d1-q3-adapt-s197"
CELLS = ("S_off", "S_riskcap", "S_dwa_arc", "T0_off", "T0_riskcap", "T0_dwa_arc",
         "T1_riskcap", "T1_off", "T2_dwa_arc", "T2_off")
ARM_MODES = {"T0": "off", "T1": "riskcap", "T2": "dwa_arc"}
METRICS = {
    "crash": ("outcome", "crash_rate", None),
    "capture": ("outcome", "capture_rate", None),
    "timeout": ("outcome", "timeout_rate", None),
    "intervention": ("speed_governor", "intervention_rate", "samples"),
    "below_3m": ("contact_geometry", "corridor_clearance_below_3m_rate", "corridor_clearance_frames"),
}
CONTRASTS = {
    "C_arc": ("T2_dwa_arc", "T0_dwa_arc"),
    "C_risk": ("T1_riskcap", "T0_riskcap"),
    "L_adapt": ("T2_dwa_arc", "T1_riskcap"),
    "D_arc": ("T2_off", "T0_off"),
    "D_risk": ("T1_off", "T0_off"),
    "K_off": ("T0_off", "S_off"),
    "K_risk": ("T0_riskcap", "S_riskcap"),
    "K_arc": ("T0_dwa_arc", "S_dwa_arc"),
}
THRESH = 3.0


def metric(data, name):
    section, key, denom = METRICS[name]
    values = data.get(section) or {}
    rate = values.get(key)
    n = values.get(denom) if denom else data.get("actual_episodes")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not math.isfinite(rate):
        return None
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        return None
    return {"rate": float(rate), "n": n}


def contrast(a, b, name="crash"):
    ma, mb = metric(a, name), metric(b, name)
    if ma is None or mb is None:
        return {"status": "UNAVAILABLE"}
    d, lo, hi = _delta_ci({"outcome": {"r": ma["rate"]}, "actual_episodes": ma["n"]},
                          {"outcome": {"r": mb["rate"]}, "actual_episodes": mb["n"]}, "r")
    return {"status": "AVAILABLE", "delta_pp": d, "ci95_pp": [lo, hi], "excludes_zero": hi < 0 or lo > 0}


def _sig_le(e, thr):
    return e["status"] == "AVAILABLE" and e["delta_pp"] <= -thr and e["excludes_zero"]


def _sig_ge(e, thr):
    return e["status"] == "AVAILABLE" and e["delta_pp"] >= thr and e["excludes_zero"]


def _null(e):
    return e["status"] == "AVAILABLE" and not e["excludes_zero"]


def q1_label(effects):
    t, c = effects["timeout"]["C_arc"], effects["crash"]["C_arc"]
    cap = effects["capture"]["C_arc"]
    if any(e["status"] != "AVAILABLE" for e in (t, c, cap)):
        return "UNAVAILABLE"
    if _sig_ge(c, THRESH):
        return "ADAPTATION_COSTS_SAFETY"
    if _sig_le(t, 2.0) and (_null(c) or c["delta_pp"] < 0):
        return "ADAPTATION_RECOVERS_COST"
    if _null(t) and _null(c) and _null(cap):
        return "NO_ADAPTATION_EFFECT"
    return "INCONCLUSIVE"


def q2_label(e):
    if e["status"] != "AVAILABLE":
        return "UNAVAILABLE"
    if _sig_le(e, THRESH):
        return "LAW_GAIN_PERSISTS"
    if _sig_ge(e, THRESH):
        return "LAW_GAIN_REVERSED"
    if _null(e):
        return "LAW_GAIN_ERASED"
    return "INCONCLUSIVE"  # significant but inside +-3 pp: not assigned by section 2


def q3_label(e):
    if e["status"] != "AVAILABLE":
        return "UNAVAILABLE"
    if _sig_ge(e, THRESH):
        return "FILTER_DEPENDENT"
    if _sig_le(e, THRESH):
        return "INTERNALIZED"
    return "NEUTRAL"


def control_label(effects):
    ks = [effects["crash"][k] for k in ("K_off", "K_risk", "K_arc")]
    if any(e["status"] != "AVAILABLE" for e in ks):
        return "UNAVAILABLE"
    if all(abs(e["delta_pp"]) < THRESH for e in ks):
        return "CONTROL_STABLE"
    return "CONTROL_SHIFTED"


def trailing_crash(run_dir, window=100):
    path = Path(run_dir) / "aerial_run" / "epoch_metrics.csv"
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open()))
    if len(rows) < window:
        return None
    vals = [float(r["crash_rate"]) for r in rows[-window:]]
    return {"epochs": len(rows), "last_epoch": int(rows[-1]["epoch"]), "trailing_crash": sum(vals) / len(vals)}


def training_checks(runs=RUNS, source_run=SOURCE_RUN):
    src = trailing_crash(source_run)
    out = {"source": src, "arms": {}}
    for arm, mode in ARM_MODES.items():
        dirs = sorted(p for p in Path(runs).glob(f"*navrl_v2-ref5in-a8-readapt-{mode}-s197") if p.is_dir())
        if len(dirs) != 1:
            out["arms"][arm] = {"status": "FAIL", "reason": f"{len(dirs)} run dirs"}
            continue
        tc = trailing_crash(dirs[0])
        if tc is None:
            out["arms"][arm] = {"status": "FAIL", "reason": "no epoch metrics"}
            continue
        errors = []
        if tc["last_epoch"] != 2900:
            errors.append(f"last epoch {tc['last_epoch']} != 2900")
        if tc["epochs"] != 1000:
            errors.append(f"{tc['epochs']} epochs logged != 1000")
        if src and tc["trailing_crash"] > src["trailing_crash"] + 0.10:
            errors.append("trailing crash exceeds source + 10 pp (prereg section 5)")
        out["arms"][arm] = dict(status="FAIL" if errors else "PASS", errors=errors, run=str(dirs[0]), **tc)
    return out


def load_root(root):
    root = Path(root)
    records, errors = {}, []
    for name in CELLS:
        path = root / name / "70bars.json"
        if not path.is_file():
            errors.append(f"missing {name}")
            continue
        data = json.loads(path.read_text())
        cond = data.get("condition") or {}
        policy, filt = name.split("_", 1)
        if cond.get("speed_governor_mode") != filt:
            errors.append(f"{name}: governor mode {cond.get('speed_governor_mode')!r} != {filt!r}")
        if cond.get("bars") != 70:
            errors.append(f"{name}: bars != 70")
        if data.get("runtime_git_dirty") is not False:
            errors.append(f"{name}: runtime_git_dirty is not false")
        data["_path"] = str(path)
        records[name] = data
    cells_json = root / "cells.json"
    manifest = json.loads(cells_json.read_text()) if cells_json.is_file() else None
    if manifest:
        for name, data in records.items():
            policy = name.split("_", 1)[0]
            expected = manifest["checkpoints"].get(policy, {}).get("sha256")
            if data.get("checkpoint_sha256") != expected:
                errors.append(f"{name}: checkpoint sha differs from cells.json")
    for field in ("runtime_source_manifest_sha256", "runtime_git_commit"):
        vals = {d.get(field) for d in records.values()}
        if len(vals) > 1:
            errors.append(f"{field} differs within root")
    seeds = {(d.get("condition") or {}).get("seed") for d in records.values()}
    if seeds and seeds != {521}:
        errors.append(f"seed set {sorted(seeds)} != [521]")
    return {"records": records, "errors": errors, "manifest": manifest}


def build_summary(root=DEFAULT_ROOT, runs=RUNS, source_run=SOURCE_RUN):
    loaded = load_root(root)
    rec = loaded["records"]
    effects = {m: {k: contrast(rec.get(a, {}), rec.get(b, {}), m) for k, (a, b) in CONTRASTS.items()} for m in METRICS}
    training = training_checks(runs, source_run)
    blockers = list(loaded["errors"])
    blockers += [f"training {arm}: {v.get('reason') or '; '.join(v.get('errors', []))}"
                 for arm, v in training["arms"].items() if v["status"] != "PASS"]
    labels = {
        "Q1_cost_recovery": q1_label(effects),
        "Q1_riskcap_side": {"timeout": effects["timeout"]["C_risk"], "crash": effects["crash"]["C_risk"]},
        "Q2_law_gain": q2_label(effects["crash"]["L_adapt"]),
        "Q3_filter_dependence": {"T2": q3_label(effects["crash"]["D_arc"]), "T1": q3_label(effects["crash"]["D_risk"])},
        "control": control_label(effects),
    }
    status = "BLOCKED" if blockers else "RESULT_CHECKS_PASSED"
    cells = {}
    for name, d in rec.items():
        cells[name] = {m: metric(d, m) for m in METRICS}
        cells[name].update(actual_episodes=d.get("actual_episodes"), checkpoint_sha256=d.get("checkpoint_sha256"))
    return {"schema_version": 1, "prereg": "docs/prereg_2026-09-05_a8_filter_readaptation.md",
            "status": status, "blockers": blockers, "labels": labels, "effects": effects,
            "training": training, "cells": cells}


def _fmt(e):
    if e["status"] != "AVAILABLE":
        return "unavailable"
    lo, hi = e["ci95_pp"]
    return f"{e['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def render_markdown(r):
    out = ["# A8 — 필터와 함께 재적응 (seed 521, 70 bars, 10 cell)", "", f"Status: `{r['status']}`.", "",
           "생성: `tools/build_a8_readaptation_table.py`. 시뮬레이션 없음.", "",
           "| 판정 | 라벨 |", "|---|---|",
           f"| Q1 비용 회수 (T2/dwa_arc − T0/dwa_arc) | `{r['labels']['Q1_cost_recovery']}` |",
           f"| Q2 법칙 이득 유지 (T2/dwa_arc − T1/riskcap) | `{r['labels']['Q2_law_gain']}` |",
           f"| Q3 필터 의존 T2 (T2/off − T0/off) | `{r['labels']['Q3_filter_dependence']['T2']}` |",
           f"| Q3 필터 의존 T1 (T1/off − T0/off) | `{r['labels']['Q3_filter_dependence']['T1']}` |",
           f"| 대조 건전성 (T0 − S) | `{r['labels']['control']}` |", "",
           "| contrast (pp, 95 % CI) | crash | capture | timeout | 개입률 | <3 m |", "|---|---:|---:|---:|---:|---:|"]
    for k in CONTRASTS:
        out.append(f"| {k} ({CONTRASTS[k][0]} − {CONTRASTS[k][1]}) | " + " | ".join(_fmt(r["effects"][m][k]) for m in METRICS) + " |")
    out += ["", "| cell | ep | crash | capture | timeout | 개입률 | <3 m |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in CELLS:
        c = r["cells"].get(name)
        if not c:
            out.append(f"| {name} | missing | | | | | |")
            continue
        out.append(f"| {name} | {c['actual_episodes']} | " + " | ".join(
            "n/a" if c[m] is None else f"{100 * c[m]['rate']:.2f}%" for m in METRICS) + " |")
    out += ["", "## 학습 arm", "", "| arm | status | epochs | last | trailing-100 crash |", "|---|---|---:|---:|---:|"]
    for arm, v in r["training"]["arms"].items():
        out.append(f"| {arm} | {v['status']} | {v.get('epochs', '-')} | {v.get('last_epoch', '-')} | "
                   f"{v.get('trailing_crash', float('nan')):.3f} |" if v["status"] == "PASS" else
                   f"| {arm} | {v['status']} | | | {v.get('reason') or '; '.join(v.get('errors', []))} |")
    src = r["training"]["source"]
    if src:
        out.append(f"| source ep1900 | ref | {src['epochs']} | {src['last_epoch']} | {src['trailing_crash']:.3f} |")
    out += ["", "## Blockers", ""] + [f"- {b}" for b in r["blockers"]] + ([""] if r["blockers"] else ["- none", ""])
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--runs", type=Path, default=RUNS)
    p.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    p.add_argument("--json-output", type=Path, default=REPO / "results/navrl_a8_readaptation_summary.json")
    p.add_argument("--markdown-output", type=Path, default=REPO / "docs/a8_readaptation_table.md")
    a = p.parse_args(argv)
    r = build_summary(a.root, a.runs, a.source_run)
    for path, text in ((a.json_output, json.dumps(r, indent=2, sort_keys=True) + "\n"), (a.markdown_output, render_markdown(r))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(f"{r['status']}: {a.markdown_output} ; {a.json_output}")
    return 0 if not r["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
