"""CPU-only A7 report, with fail-closed preregistration and provenance gates.

Only *bars.json result files are read. M1/M2 need separately recorded CPU test
evidence; aggregate outcomes cannot establish either implementation property.
No simulation module (or torch) is imported. --check-p1 performs the early gate
without requiring P2/P3. --p1-root accepts a complete four-arm fallback root.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

try:
    from .build_a5_ablation_table import _delta_ci
except ImportError:
    from build_a5_ablation_table import _delta_ci


REPO = Path(__file__).resolve().parents[1]
ARMS = ("riskcap", "stopcap", "dwa_arc", "riskcap_arc")
DEFAULT_ROOTS = {
    "P1": REPO / "results/navrl_arc_attribution_seed509",
    "P2": REPO / "results/navrl_arc_attribution_seed491",
    "P3": REPO / "results/navrl_arc_attribution_205bars_seed49",
    "A4": REPO / "results/navrl_contact_geometry_seed509",
    "SCREEN": REPO / "results/navrl_v2_ep25000_stopcap_seed49_screen",
}
PARAMETERS = {
    "speed_governor_fixed_mps": 2.0,
    "speed_governor_free_mps": 3.53553390593,
    "speed_governor_half_width_m": 0.45,
    "speed_governor_margin_m": 0.45,
    "speed_governor_slow_m": 3.0,
    "speed_governor_release_m": 5.0,
    "speed_governor_brake_mps2": 2.0,
    "speed_governor_reaction_s": 0.1,
    "speed_governor_ttc_s": 1.0,
    "speed_governor_target_exclusion": "camera_lidar_association",
}
METRICS = {
    "crash": ("outcome", "crash_rate", None),
    "capture": ("outcome", "capture_rate", None),
    "timeout": ("outcome", "timeout_rate", None),
    "intervention": ("speed_governor", "intervention_rate", "samples"),
    "below_3m": (
        "contact_geometry", "corridor_clearance_below_3m_rate", "corridor_clearance_frames"
    ),
}
CONTRASTS = {
    "G_stop": ("dwa_arc", "stopcap"),
    "G_risk": ("riskcap_arc", "riskcap"),
    "L_line": ("stopcap", "riskcap"),
    "L_arc": ("dwa_arc", "riskcap_arc"),
}
PRECEDENCE = ("ARC_HURTS_UNDER_RISKCAP", "LAW_CARRIES_SAFETY", "ARC_CARRIES_SAFETY", "INTERACTION")
CONTEXT = (
    "bars", "seed", "num_envs", "action_selection", "reflection_mode",
    "distractor_count", "robot_name", "robot_asset_sha256", "robot_config_sha256",
    "goal_dist_min_m", "goal_dist_max_m", "episode_len_steps",
    "target_motion_model", "target_pattern", "target_speed_mode",
    "target_speed_min_mps", "target_speed_max_mps",
)
CAVEATS = [
    "M1/M2 require external CPU test evidence tied to the evaluation commit; results do not prove them.",
    "M5 is a repeatability check, not proof that historical source was clean or identical.",
    "DENSITY_LIMITED and FLIP_IS_DENSITY are preregistered labels, not isolated density effects: "
    "P3 also changes checkpoint, robot platform, goal range and seed.",
    "Outcome CIs use the A5 independent two-proportion Wald formula and actual episode counts; "
    "same-seed arms are not treated as paired observations.",
    "Intervention and below-3m CIs use recorded frame/sample counts with that same formula. "
    "Frames are temporally dependent; these are nominal unclustered CIs, not episode-level CIs.",
    "Label precedence when rules overlap is frozen in prereg §4: ARC_HURTS_UNDER_RISKCAP beats INTERACTION.",
]


def _check(errors, **details):
    return dict(status="FAIL" if errors else "PASS", errors=errors, **details)


def _valid_rate(value):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 1)


def _valid_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _reject_constant(value):
    raise ValueError("Non-finite JSON number: " + value)


def _load_root(path, bars, required):
    """Read every arm's result in the named root; do not silently select duplicates."""
    path = Path(path)
    records, errors = {}, []
    for filename in sorted(path.glob("*/*bars.json")):
        arm = filename.parent.name
        if filename.name != "{}bars.json".format(bars):
            errors.append("Unexpected density result: {}".format(filename))
            continue
        if arm in records:
            errors.append("Duplicate arm result: {}".format(filename))
            continue
        try:
            raw = filename.read_text()
            data = json.loads(raw, parse_constant=_reject_constant)
            # Preserve numeric literal spellings for the preregistered byte-exact M6 check.
            literal_data = json.loads(raw, parse_float=str, parse_int=str)
            if not isinstance(data, dict):
                raise ValueError("result must be an object")
            condition = data.get("condition", {})
            if not isinstance(condition, dict):
                raise ValueError("condition must be an object")
            data["_path"] = str(filename)
            data["_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
            data["_parameter_literals"] = {
                k: v for k, v in literal_data.get("condition", {}).items()
                if k.startswith("speed_governor_") and k != "speed_governor_mode"
            }
            records[arm] = data
            if condition.get("bars") != bars:
                errors.append("{}: condition.bars mismatch".format(arm))
            if condition.get("speed_governor_mode") != {"fixed2p0": "fixed"}.get(arm, arm):
                errors.append("{}: governor mode mismatch".format(arm))
            n = data.get("actual_episodes")
            if not _valid_count(n):
                errors.append("{}: invalid actual_episodes".format(arm))
            outcome = data.get("outcome", {})
            if not isinstance(outcome, dict):
                errors.append("{}: outcome must be an object".format(arm))
                continue
            for metric, count_key in (("crash", "crash"), ("capture", "captured"), ("timeout", "timeout")):
                rate = outcome.get(metric + "_rate")
                if not _valid_rate(rate):
                    errors.append("{}: invalid {} rate".format(arm, metric))
                elif count_key in outcome and _valid_count(n):
                    count = outcome[count_key]
                    if (not isinstance(count, int) or isinstance(count, bool)
                            or not 0 <= count <= n or abs(rate - count / n) > 1e-10):
                        errors.append("{}: {} count/rate mismatch".format(arm, metric))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            errors.append("{}: {}".format(filename, exc))
    for arm in required:
        if arm not in records:
            errors.append("Missing {}/{}bars.json".format(arm, bars))
    return dict(path=str(path), records=records, load=_check(errors))


def _provenance(root):
    records = root["records"]
    errors = list(root["load"]["errors"])
    for field, length in (("checkpoint_sha256", 64), ("runtime_source_manifest_sha256", 64),
                          ("runtime_git_commit", 40)):
        values = [d.get(field) for d in records.values()]
        if not values or any(not isinstance(v, str) or not re.fullmatch("[0-9a-f]{%d}" % length, v) for v in values):
            errors.append("Missing/invalid {}".format(field))
        elif len(set(values)) != 1:
            errors.append("{} differs within root".format(field))
    dirty = [a for a, d in records.items() if d.get("runtime_git_dirty") is not False]
    if dirty:
        errors.append("runtime_git_dirty is not false: " + ", ".join(dirty))
    return _check(errors, root=root["path"], arms=sorted(records))


def _same_commit(roots):
    values = {name: sorted(set(str(d.get("runtime_git_commit", "")) for d in r["records"].values()))
              for name, r in roots.items()}
    flat = [v for group in values.values() for v in group]
    errors = []
    if any(not group for group in values.values()) or not flat:
        errors.append("Missing evaluation commit")
    if any(not isinstance(v, str) or not re.fullmatch("[0-9a-f]{40}", v) for v in flat):
        errors.append("Invalid evaluation commit")
    if len(set(flat)) != 1:
        errors.append("A7 parts do not share one evaluation commit")
    return _check(errors, commits=values,
                  scope="Recorded evaluation commits; bars.json cannot independently attest all on-disk source bytes.")


def _parameters(roots):
    errors = []
    expected_literals = json.loads(json.dumps(PARAMETERS), parse_float=str, parse_int=str)
    for name, root in roots.items():
        for arm, data in root["records"].items():
            actual = {k: v for k, v in data.get("condition", {}).items()
                      if k.startswith("speed_governor_") and k != "speed_governor_mode"}
            if actual != PARAMETERS or data["_parameter_literals"] != expected_literals:
                errors.append("{}/{}: governor parameters differ from A4 literal values".format(name, arm))
    if not any(r["records"] for r in roots.values()):
        errors.append("No parameter evidence")
    return _check(errors, expected=PARAMETERS, comparison="Exact decoded values and numeric JSON literals; mode varies by arm.")


def _bindings(root, seed, reference):
    errors = []
    expected_sha = reference.get("checkpoint_sha256") if reference else None
    for arm, data in root["records"].items():
        condition = data.get("condition", {})
        for key, value in (("seed", seed), ("num_envs", 128), ("action_selection", "deterministic"),
                           ("reflection_mode", "original")):
            if condition.get(key) != value:
                errors.append("{}: {} must equal {!r}".format(arm, key, value))
        if condition.get("distractor_count", 0) != 0:
            errors.append("{}: distractor_count must be zero".format(arm))
        if not expected_sha or data.get("checkpoint_sha256") != expected_sha:
            errors.append("{}: checkpoint differs from historical reference".format(arm))
        if reference:
            # P2 has a different seed by design. Missing historical optional fields
            # are exposed in condition differences rather than invented.
            ref_condition = reference.get("condition", {})
            for key in CONTEXT:
                if key in ("seed", "distractor_count"):
                    continue
                if key in ref_condition and condition.get(key) != ref_condition[key]:
                    errors.append("{}: {} differs from historical condition".format(arm, key))
    return _check(errors)


def metric(data, name):
    section, key, denominator = METRICS[name]
    values = data.get(section, {})
    if not isinstance(values, dict):
        return None
    rate = values.get(key)
    n = values.get(denominator) if denominator else data.get("actual_episodes")
    if not _valid_rate(rate) or not _valid_count(n):
        return None
    return dict(rate=rate, n=n, denominator=denominator or "actual_episodes")


def contrast(a, b, name="crash"):
    ma, mb = metric(a, name), metric(b, name)
    if ma is None or mb is None:
        return dict(status="UNAVAILABLE", reason="Missing/invalid {} rate or recorded denominator".format(name))
    # Reuse the preregistered A5 formula verbatim, adapting non-outcome denominators.
    d, lo, hi = _delta_ci(
        {"outcome": {"rate": ma["rate"]}, "actual_episodes": ma["n"]},
        {"outcome": {"rate": mb["rate"]}, "actual_episodes": mb["n"]}, "rate")
    return dict(status="AVAILABLE", delta_pp=d, ci95_pp=[lo, hi],
                excludes_zero=hi < 0 or lo > 0, n_a=ma["n"], n_b=mb["n"],
                denominator=ma["denominator"], a=a.get("_path"), b=b.get("_path"))


def classify(effects):
    """Expose every matching rule; no unregistered precedence is imposed."""
    if any(effects.get(k, {}).get("status") != "AVAILABLE" for k in CONTRASTS):
        return dict(status="UNAVAILABLE", label=None, matched_labels=[])
    gs, gr, ll, la = (effects[k] for k in ("G_stop", "G_risk", "L_line", "L_arc"))
    beneficial = lambda e: e["delta_pp"] <= -3 and e["excludes_zero"]
    matches = []
    if beneficial(ll) and beneficial(la) and not gs["excludes_zero"] and not gr["excludes_zero"]:
        matches.append("LAW_CARRIES_SAFETY")
    if beneficial(gs) and beneficial(gr):
        matches.append("ARC_CARRIES_SAFETY")
    if gs["excludes_zero"] != gr["excludes_zero"]:
        matches.append("INTERACTION")
    if gr["delta_pp"] >= 3 and gr["excludes_zero"]:
        matches.append("ARC_HURTS_UNDER_RISKCAP")
    # Prereg §4 precedence (frozen 2026-09-05 before any A7 result): the only rule pair that can
    # both match is ARC_HURTS_UNDER_RISKCAP + INTERACTION; the more specific label wins.
    for label in PRECEDENCE:
        if label in matches:
            return dict(status="CLASSIFIED", label=label, matched_labels=matches)
    return dict(status="CLASSIFIED", label="INCONCLUSIVE", matched_labels=matches)


def _m5(current, reference, expected_percent, prereg_episodes):
    if current is None or reference is None or metric(current, "crash") is None or metric(reference, "crash") is None:
        return dict(status="FAIL", errors=["Missing/invalid riskcap reference or repeat"])
    delta = contrast(current, reference)
    observed = "{:.2f}".format(100 * current["outcome"]["crash_rate"])
    historical = "{:.2f}".format(100 * reference["outcome"]["crash_rate"])
    expected = "{:.2f}".format(expected_percent)
    exact = observed == historical == expected
    if exact:
        status, action = "PASS", "Cross-root zero check passes; does not establish source equivalence."
    elif abs(delta["delta_pp"]) > 0.5 + 1e-12:
        status, action = "FAIL", "Cross-root comparison prohibited; P1 requires a complete four-arm root."
    else:
        # Prereg §7 (frozen): within 0.5 pp the cross-root comparison stays admissible but is
        # reported as inexact; the delta is carried into every label that depends on it.
        status, action = "PASS_INEXACT", "Cross-root comparison admissible; report the repeat delta alongside any dependent label."
    return dict(status=status, expected_percent=expected, observed_percent=observed,
                historical_percent=historical, comparison=delta, action=action,
                actual_episodes=current["actual_episodes"], historical_actual_episodes=reference["actual_episodes"],
                prereg_historical_episodes=prereg_episodes,
                historical_episode_count_matches_prereg=reference["actual_episodes"] == prereg_episodes)


def _cost_decision(effects):
    timeout = effects["timeout"]["G_stop"]
    intervention = effects["intervention"]["G_stop"]
    neutral = effects["timeout"]["G_risk"]
    removal_available = all(x["status"] == "AVAILABLE" for x in (timeout, intervention))
    return {
        "ARC_REMOVES_COST": (all(x["delta_pp"] < 0 and x["excludes_zero"] for x in (timeout, intervention))
                             if removal_available else None),
        "ARC_COST_NEUTRAL_UNDER_RISKCAP": (not neutral["excludes_zero"] if neutral["status"] == "AVAILABLE" else None),
    }


def _part(records, blockers, historical=None):
    effects = {name: {} for name in METRICS}
    for name in METRICS:
        for label, (a, b) in CONTRASTS.items():
            # The prereg explicitly assigns mixed-P1 L_line to the old A4 pair.
            source = historical if historical is not None and label in ("G_stop", "L_line") else records
            da, db = source.get(a, {}), source.get(b, {})
            effects[name][label] = contrast(da, db, name)
    decision = classify(effects["crash"])
    costs = _cost_decision(effects)
    if blockers:
        decision = dict(status="BLOCKED", label=None, matched_labels=[], reasons=blockers)
        costs = {key: None for key in costs}
    return dict(effects=effects, classification=decision, cost_labels=costs, eligible=not blockers,
                arms={arm: _record_view(records[arm]) for arm in ARMS if arm in records})


def _sign(value):
    return (value > 0) - (value < 0)


def replication(p1, p2):
    if any(p["classification"]["label"] is None for p in (p1, p2)):
        return "UNAVAILABLE"
    a, b = p1["effects"]["crash"], p2["effects"]["crash"]
    same_line = _sign(a["L_line"]["delta_pp"]) == _sign(b["L_line"]["delta_pp"])
    same_geometry = _sign(a["G_stop"]["delta_pp"]) == _sign(b["G_stop"]["delta_pp"])
    if p1["classification"]["label"] == p2["classification"]["label"] and same_line:
        return "REPLICATED"
    if p1["classification"]["label"] != p2["classification"]["label"] and same_line and same_geometry:
        return "PARTIAL"
    return "NOT_REPLICATED"


def brake_label(effect):
    if effect["status"] != "AVAILABLE":
        return "UNAVAILABLE"
    if effect["delta_pp"] <= -3 and effect["excludes_zero"]:
        return "FLIP_IS_BRAKE"
    if effect["delta_pp"] >= 0 or not effect["excludes_zero"]:
        return "FLIP_IS_DENSITY"
    return "INCONCLUSIVE"  # Significant -3 < delta < 0 is not assigned by §4.


def _record_view(data):
    return dict(path=data["_path"], file_sha256=data["_sha256"], actual_episodes=data.get("actual_episodes"),
                metrics={name: metric(data, name) for name in METRICS},
                checkpoint_sha256=data.get("checkpoint_sha256"), runtime_git_commit=data.get("runtime_git_commit"),
                runtime_git_dirty=data.get("runtime_git_dirty"),
                runtime_source_manifest_sha256=data.get("runtime_source_manifest_sha256"),
                condition={key: data.get("condition", {}).get(key) for key in CONTEXT},
                descriptive={key: (data.get(section) or {}).get(key) for section, key in (
                    ("contact_geometry", "path_length_mean_m"), ("speed_governor", "mean_executed_speed_mps"),
                    ("contact_geometry", "governor_compute_us_mean_per_env"))})


def build_summary(paths=None, check_p1=False):
    paths = dict(DEFAULT_ROOTS, **(paths or {}))
    roots = {
        "P1": _load_root(paths["P1"], 70, ("riskcap", "riskcap_arc")),
        "A4": _load_root(paths["A4"], 70, ("riskcap", "stopcap", "dwa_arc")),
    }
    if not check_p1:
        roots["P2"] = _load_root(paths["P2"], 70, ARMS)
        roots["P3"] = _load_root(paths["P3"], 205, ARMS)
        roots["SCREEN"] = _load_root(paths["SCREEN"], 205, ("riskcap", "stopcap"))
    a7 = {k: r for k, r in roots.items() if k in ("P1", "P2", "P3")}
    full_p1 = all(a in roots["P1"]["records"] for a in ARMS)
    if not full_p1 and any(a in roots["P1"]["records"] for a in ("stopcap", "dwa_arc")):
        roots["P1"]["load"]["errors"].append("Partial four-arm P1 fallback; both stopcap and dwa_arc are required")
        roots["P1"]["load"]["status"] = "FAIL"
    a4_ref = roots["A4"]["records"].get("riskcap")
    integrity = {
        "M1": dict(status="EXTERNAL_CPU_TEST_REQUIRED", evidence="Cap-law equivalence test tied to evaluation commit"),
        "M2": dict(status="EXTERNAL_CPU_TEST_REQUIRED", evidence="Shared clearance dispatch and zero-yaw equivalence tests tied to evaluation commit"),
        "M3": {k: _provenance(r) for k, r in a7.items()},
        "M4": _same_commit(a7),
        "M5": {"P1": _m5(roots["P1"]["records"].get("riskcap"), a4_ref, 18.77, 2049)},
        "M6": _parameters(a7),
        "historical_A4": _provenance(roots["A4"]),
        "conditions": {"P1": _bindings(roots["P1"], 509, a4_ref)},
    }
    if not check_p1:
        screen_ref = roots["SCREEN"]["records"].get("riskcap")
        integrity["M5"]["P3"] = _m5(roots["P3"]["records"].get("riskcap"), screen_ref, 15.95, 2050)
        integrity["conditions"]["P2"] = _bindings(roots["P2"], 491, a4_ref)
        integrity["conditions"]["P3"] = _bindings(roots["P3"], 49, screen_ref)
        integrity["historical_SCREEN"] = _provenance(roots["SCREEN"])
    parts, blockers = {}, []
    for name, root in a7.items():
        reasons = []
        for gate in ("M3", "conditions"):
            if integrity[gate][name]["status"] != "PASS":
                reasons.append(gate + ": " + name)
        for gate in ("M4", "M6"):
            if integrity[gate]["status"] != "PASS":
                reasons.append(gate)
        records = dict(root["records"])
        historical = None
        if name == "P1" and not full_p1:
            historical = roots["A4"]["records"]
            records.update({a: historical[a] for a in ("stopcap", "dwa_arc") if a in historical})
            for gate in ("historical_A4",):
                if integrity[gate]["status"] != "PASS":
                    reasons.append(gate)
            if integrity["M5"]["P1"]["status"] not in ("PASS", "PASS_INEXACT"):
                reasons.append("M5: P1 cross-root comparison")
        parts[name] = _part(records, reasons, historical=historical)
        parts[name]["design"] = "mixed_A4_controls" if name == "P1" and not full_p1 else "within_root_four_arm"
        parts[name]["root_disposition"] = ("VOID" if "M3: " + name in reasons or "M4" in reasons
                                              else "INELIGIBLE" if reasons else "VALID_RESULT_PROVENANCE")
        blockers.extend("{}: {}".format(name, r) for r in reasons)
        if parts[name]["classification"]["status"] in ("AMBIGUOUS_RULE_OVERLAP", "UNAVAILABLE"):
            blockers.append(name + ": " + parts[name]["classification"]["status"])
    report = dict(schema_version=1, prereg="docs/prereg_2026-09-05_a7_arc_attribution.md",
                  check_p1_only=check_p1, integrity=integrity, parts=parts, caveats=list(CAVEATS),
                  sources={name: {arm: _record_view(d) for arm, d in r["records"].items()} for name, r in roots.items()})
    if full_p1:
        integrity["M5"]["P1"]["required_for_P1_factorial"] = False
        report["caveats"].append("P1 uses all four arms in one root; its factorial does not import A4 controls or require cross-root M5.")
    if not check_p1:
        report["replication"] = replication(parts["P1"], parts["P2"])
        report["withdraw_GEOMETRY_MATTERS"] = report["replication"] == "NOT_REPLICATED"
        p1_label, p3_label = (parts[k]["classification"]["label"] for k in ("P1", "P3"))
        report["density_limited"] = (p1_label != p3_label if p1_label is not None and p3_label is not None else None)
        historical = roots["SCREEN"]["records"]
        current = roots["P3"]["records"]
        cross_ok = (parts["P3"]["eligible"] and integrity["M5"]["P3"]["status"] in ("PASS", "PASS_INEXACT")
                    and roots["SCREEN"]["load"]["status"] == "PASS")
        brake = brake_label(parts["P3"]["effects"]["crash"]["L_line"]) if cross_ok else "BLOCKED"
        if cross_ok and integrity["M5"]["P3"]["status"] == "PASS_INEXACT":
            brake += " (M5 inexact)"
        report["brake_comparison"] = dict(
            label=brake,
            historical_L_line=contrast(historical.get("stopcap", {}), historical.get("riskcap", {})),
            stopcap_new_minus_old=contrast(current.get("stopcap", {}), historical.get("stopcap", {})),
            historical_provenance=integrity["historical_SCREEN"]["status"],
            caveat="Historical screen has its recorded provenance limitations; M5 cannot repair them.")
        if not cross_ok:
            blockers.append("P3: historical brake comparison blocked")
        report["condition_differences_P1_P3"] = {
            key: {"P1": (a4_ref or {}).get("condition", {}).get(key),
                  "P3": (screen_ref or {}).get("condition", {}).get(key)} for key in CONTEXT
            if (a4_ref or {}).get("condition", {}).get(key) != (screen_ref or {}).get("condition", {}).get(key)
        }
    report["blockers"] = list(dict.fromkeys(blockers))
    report["result_checks_passed"] = not report["blockers"]
    report["status"] = "BLOCKED" if report["blockers"] else "RESULT_CHECKS_PASSED_CPU_EVIDENCE_REQUIRED"
    return report


def _fmt_effect(effect):
    if effect["status"] != "AVAILABLE":
        return "unavailable"
    lo, hi = effect["ci95_pp"]
    return "{:+.2f} [{:+.2f}, {:+.2f}]".format(effect["delta_pp"], lo, hi)


def render_markdown(report):
    out = ["# A7 — law × geometry attribution", "", "Status: `{}`.".format(report["status"]), "",
           "Generated by `tools/build_a7_arc_attribution_table.py` from listed `*bars.json` files only.", "",
           "M1/M2: external CPU test evidence required. No aggregate-result PASS is inferred.", "",
           "| gate | result |", "|---|---|"]
    for name, check in report["integrity"].items():
        if "status" in check:
            out.append("| {} | {} |".format(name, check["status"]))
        else:
            for part, item in check.items():
                out.append("| {} / {} | {} |".format(name, part, item["status"]))
    for part, value in report["parts"].items():
        decision = value["classification"]
        out += ["", "## {} ({})".format(part, value["design"]), "",
                "Classification: `{}`; status `{}`; matching rules: {}.".format(
                    decision["label"], decision["status"], ", ".join(decision["matched_labels"]) or "none"), "",
                "| contrast (pp; 95% CI) | crash | capture | timeout | intervention | <3 m frames |",
                "|---|---:|---:|---:|---:|---:|"]
        for effect in CONTRASTS:
            out.append("| {} | {} |".format(effect, " | ".join(_fmt_effect(value["effects"][m][effect]) for m in METRICS)))
        out += ["", "Cost labels: `{}`.".format(json.dumps(value["cost_labels"], sort_keys=True)), "",
                "| arm | actual episodes | crash | capture | timeout | intervention | <3 m |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for arm, data in value["arms"].items():
            rates = ["unavailable" if data["metrics"][m] is None else "{:.2%}".format(data["metrics"][m]["rate"])
                     for m in METRICS]
            out.append("| {} | {} | {} |".format(arm, data["actual_episodes"], " | ".join(rates)))
    historical = [name for name in ("A4", "SCREEN") if name in report["sources"]]
    if historical:
        out += ["", "## Historical references (read-only)", "",
                "| root / arm | actual episodes | crash | capture | timeout | intervention | <3 m |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for name in historical:
            for arm, data in report["sources"][name].items():
                rates = ["unavailable" if data["metrics"][m] is None else "{:.2%}".format(data["metrics"][m]["rate"])
                         for m in METRICS]
                out.append("| {} / {} | {} | {} |".format(name, arm, data["actual_episodes"], " | ".join(rates)))
    for part, value in report["integrity"]["M5"].items():
        out += ["", "M5 {}: `{}`.".format(part, json.dumps(value, sort_keys=True))]
    if not report["check_p1_only"]:
        out += ["", "Replication: `{}`. Withdraw GEOMETRY_MATTERS: `{}`. DENSITY_LIMITED: `{}`.".format(
                    report["replication"], report["withdraw_GEOMETRY_MATTERS"], report["density_limited"]), "",
                "Brake comparison: `{}`. Old L_line: {} pp. Stopcap new − old: {} pp.".format(
                    report["brake_comparison"]["label"], _fmt_effect(report["brake_comparison"]["historical_L_line"]),
                    _fmt_effect(report["brake_comparison"]["stopcap_new_minus_old"])), "",
                "Condition differences: `{}`.".format(json.dumps(report["condition_differences_P1_P3"], sort_keys=True))]
    out += ["", "## Limitations and blockers", ""]
    out.extend("- " + message for message in report["caveats"] + report["blockers"])
    out += ["", "## Sources", "", "| root / arm | result path | SHA256 |", "|---|---|---|"]
    for name, records in report["sources"].items():
        for arm, data in records.items():
            out.append("| {} / {} | `{}` | `{}` |".format(name, arm, data["path"], data["file_sha256"]))
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in DEFAULT_ROOTS:
        parser.add_argument("--" + key.lower() + "-root", type=Path, default=DEFAULT_ROOTS[key])
    parser.add_argument("--check-p1", action="store_true", help="Print P1 gate JSON only; no output files or P2/P3 reads")
    parser.add_argument("--json-output", type=Path, default=REPO / "results/navrl_arc_attribution_summary.json")
    parser.add_argument("--markdown-output", type=Path, default=REPO / "docs/a7_arc_attribution_table.md")
    args = parser.parse_args(argv)
    paths = {k: getattr(args, k.lower() + "_root") for k in DEFAULT_ROOTS}
    report = build_summary(paths, check_p1=args.check_p1)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check_p1:
        print(serialized, end="")
    else:
        for path, content in ((args.json_output, serialized), (args.markdown_output, render_markdown(report))):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        print("{}: {} ; {}".format(report["status"], args.markdown_output, args.json_output))
    return 0 if report["result_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
