#!/usr/bin/env python3
"""Validate docs/status/data/experiments.json.

experiments.json is hand-curated -- it is the one dataset on the dashboard that a machine cannot
derive, because "what did this experiment actually establish" lives in WORKLOG prose. That makes it
the one dataset that can silently rot, so it gets the same fail-closed discipline as the evaluation
harnesses: a broken record stops the build rather than shipping a plausible-looking dashboard.

What it enforces, and why each rule exists:

  * `verdict` and `validity` are ORTHOGONAL. A result can be PASS + superseded (it passed its gate
    and the numbers are void anyway, e.g. everything before the chirality fix). Collapsing them
    into one enum would force us to relabel retractions as failures, which misreports our own
    record. So both are required, from disjoint enums.
  * `validity == "withdrawn"` requires a `retraction` object. A withdrawal without a stated reason
    is indistinguishable from hiding a result.
  * `superseded_by` / `supersedes` must resolve, and `supersedes` must be reciprocal. A dangling
    pointer means the reader cannot get from a void number to the one that replaced it.
  * exactly one baseline level, so "compared against what" is never ambiguous.
  * `knob`, when set, must exist in parameters.json -- that is the join the two pages rely on.
  * every `results_paths` entry must exist on disk, so provenance links cannot go stale silently.

Run: PYTHONNOUSERSITE=1 python3 tools/lint_experiments.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "docs/status/data/experiments.json"
PARAMETERS = ROOT / "docs/status/data/parameters.json"

VERDICTS = {"PASS", "FAIL", "INCONCLUSIVE"}
VALIDITIES = {"canonical", "superseded", "withdrawn", "exploratory"}
METRICS = {"capture", "crash", "timeout", "bar_contact", "detection_f1", "latency_ms", "other"}
RETRACTION_REASONS = {"bug", "confound", "rng_contamination", "wrong_frame", "spec_error", "scope"}
REQUIRED = ["id", "title_ko", "question", "theme", "levels", "metric", "effect",
            "verdict", "validity", "verdict_note", "seeds", "worklog_date"]


def main():
    errors, warnings = [], []
    data = json.loads(EXPERIMENTS.read_text(encoding="utf-8"))
    exps = data["experiments"]
    theme_ids = {t["id"] for t in data["themes"]}
    ids = [e.get("id") for e in exps]

    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        errors.append(f"duplicate ids: {sorted(dupes)}")
    id_set = set(ids)

    known_knobs = set()
    if PARAMETERS.exists():
        pdata = json.loads(PARAMETERS.read_text(encoding="utf-8"))
        known_knobs = {p["name"] for p in pdata["parameters"]} | {e["name"] for e in pdata["echo_only"]}
    else:
        warnings.append("parameters.json missing -- knob join not checked")

    for e in exps:
        eid = e.get("id", "<no id>")
        for field in REQUIRED:
            if field not in e:
                errors.append(f"{eid}: missing required field `{field}`")

        if e.get("theme") not in theme_ids:
            errors.append(f"{eid}: unknown theme {e.get('theme')!r}")
        if e.get("verdict") not in VERDICTS:
            errors.append(f"{eid}: verdict {e.get('verdict')!r} not in {sorted(VERDICTS)}")
        if e.get("validity") not in VALIDITIES:
            errors.append(f"{eid}: validity {e.get('validity')!r} not in {sorted(VALIDITIES)}")
        if e.get("metric") not in METRICS:
            errors.append(f"{eid}: metric {e.get('metric')!r} not in {sorted(METRICS)}")
        if e.get("metric") == "other" and not e.get("metric_label"):
            errors.append(f"{eid}: metric=='other' requires metric_label")

        levels = e.get("levels") or []
        if len(levels) < 1:
            errors.append(f"{eid}: needs at least one level")
        n_base = sum(1 for l in levels if l.get("is_baseline"))
        if n_base != 1:
            errors.append(f"{eid}: exactly one baseline level required, found {n_base}")

        eff = e.get("effect") or {}
        for field in ("value_pp", "ci95_pp", "ci_method", "direction"):
            if field not in eff:
                errors.append(f"{eid}: effect missing `{field}`")
        if eff.get("direction") not in {"higher_is_better", "lower_is_better"}:
            errors.append(f"{eid}: effect.direction {eff.get('direction')!r} invalid")
        ci = eff.get("ci95_pp")
        if ci is not None:
            if not (isinstance(ci, list) and len(ci) == 2 and ci[0] <= ci[1]):
                errors.append(f"{eid}: ci95_pp must be [lo, hi] with lo <= hi, got {ci!r}")
            elif eff.get("value_pp") is not None and not (ci[0] <= eff["value_pp"] <= ci[1]):
                errors.append(f"{eid}: value_pp {eff['value_pp']} outside its own CI {ci}")

        # validity <-> retraction / supersession consistency
        if e.get("validity") == "withdrawn" and not e.get("retraction"):
            errors.append(f"{eid}: validity=='withdrawn' requires a `retraction` object")
        if e.get("validity") == "superseded" and not e.get("superseded_by"):
            errors.append(f"{eid}: validity=='superseded' requires `superseded_by`")
        r = e.get("retraction")
        if r:
            if r.get("reason") not in RETRACTION_REASONS:
                errors.append(f"{eid}: retraction.reason {r.get('reason')!r} not in {sorted(RETRACTION_REASONS)}")
            if not r.get("text"):
                errors.append(f"{eid}: retraction needs `text` saying what went wrong")

        sb = e.get("superseded_by")
        if sb and sb not in id_set:
            errors.append(f"{eid}: superseded_by -> unknown id {sb!r}")
        for sup in e.get("supersedes") or []:
            if sup not in id_set:
                errors.append(f"{eid}: supersedes -> unknown id {sup!r}")

        knob = e.get("knob")
        if knob and known_knobs and knob not in known_knobs:
            warnings.append(f"{eid}: knob {knob!r} not found in parameters.json")
        for k in e.get("knob_extra") or []:
            if known_knobs and k not in known_knobs:
                warnings.append(f"{eid}: knob_extra {k!r} not found in parameters.json")

        for p in e.get("results_paths") or []:
            if not (ROOT / p).exists():
                warnings.append(f"{eid}: results_path missing on disk -> {p}")
        lp = e.get("launcher")
        if lp and not (ROOT / lp).exists():
            warnings.append(f"{eid}: launcher missing on disk -> {lp}")

    # supersedes must be reciprocal with superseded_by
    by_id = {e["id"]: e for e in exps if "id" in e}
    for e in exps:
        for sup in e.get("supersedes") or []:
            other = by_id.get(sup)
            if other and other.get("superseded_by") not in (e["id"], None):
                errors.append(f"{e['id']}: supersedes {sup}, but {sup}.superseded_by = "
                              f"{other.get('superseded_by')!r}")
            elif other and other.get("superseded_by") is None:
                warnings.append(f"{sup}: superseded by {e['id']} but does not say so")

    counts = {v: sum(1 for e in exps if e.get("validity") == v) for v in sorted(VALIDITIES)}
    verdicts = {v: sum(1 for e in exps if e.get("verdict") == v) for v in sorted(VERDICTS)}
    print(f"{len(exps)} experiments · validity {counts} · verdict {verdicts}")

    for w in warnings:
        print(f"  WARN  {w}")
    for err in errors:
        print(f"  ERROR {err}")
    if errors:
        print(f"\nFAILED: {len(errors)} error(s)")
        return 1

    print("OK" + (f" · {len(warnings)} warning(s)" if warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
