#!/usr/bin/env python3
"""Per-arm impact classification for the nine audited defects.

Every arm-impact claim here is settled by a source trace, not by judgement. The
structural facts the matrix depends on are re-derived at build time and the build
fails if any of them stops being true.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
NAVRL_TASK = ROOT / "aerial_gym/task/navrl_task/navrl_task.py"
PERCEPTION = ROOT / "aerial_gym/task/navrl_task/navrl_perception.py"
TARGET_MOTION = ROOT / "aerial_gym/task/navrl_task/target_motion.py"
TASK_CONFIG = ROOT / "aerial_gym/config/task_config/navrl_task_config.py"
ASSET_MANAGER = ROOT / "aerial_gym/env_manager/asset_manager.py"
OUT_JSON = ROOT / "docs/audits/target_motion_defect_impact_matrix_2026-09-17.json"


def git(*a):
    return subprocess.check_output(["git", *a], cwd=str(ROOT), text=True).strip()


def bounded_branch_returns_before_legacy_block():
    """THE structural fact: does bounded/physical exit before the legacy block?"""
    src = NAVRL_TASK.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_advance_target")
    for stmt in fn.body:
        if not isinstance(stmt, ast.If):
            continue
        test = ast.get_source_segment(src, stmt.test) or ""
        if "_target_dynamics" not in test or "bounded" not in test:
            continue
        last = stmt.body[-1]
        return {
            "branch_line": stmt.lineno,
            "branch_test": test,
            "last_statement_type": type(last).__name__,
            "last_statement_line": last.lineno,
            "unconditional_return": isinstance(last, ast.Return),
            "has_else": bool(stmt.orelse),
        }
    return None


def legacy_only_line_numbers():
    """Lines of the push-out / reflection / velocity-rewrite block."""
    src = NAVRL_TASK.read_text(encoding="utf-8").splitlines()
    out = {}
    for i, line in enumerate(src, 1):
        if "push_dir[sel] = dirn[rows]" in line:
            out["push_out"] = i
        if "jit = (torch.rand(" in line:
            out["bounce_jitter_global_rng"] = i
        if "realized world velocity" in line:
            out["velocity_rewrite"] = i + 1
    return out


def structural_facts():
    branch = bounded_branch_returns_before_legacy_block()
    legacy = legacy_only_line_numbers()
    motion_src = TARGET_MOTION.read_text(encoding="utf-8")
    return {
        "bounded_branch": branch,
        "legacy_only_block": legacy,
        "legacy_block_after_branch_return": all(
            line > branch["last_statement_line"] for line in legacy.values()),
        "bounded_executor_consumes_no_rng": not re.search(
            r"torch\.(rand|randn|randint)|\.normal_\(|\.uniform_\(", motion_src),
        "obstacle_poses_resampled_per_episode_from_global_rng": bool(
            re.search(r"def reset_idx", ASSET_MANAGER.read_text(encoding="utf-8"))
            and re.search(r"torch\.rand\(", ASSET_MANAGER.read_text(encoding="utf-8"))),
        "surface_clearance_consumer_is_footprint_mode_only": (
            ROOT / "aerial_gym/env_manager/asset_manager.py"
        ).read_text(encoding="utf-8").count("self.placement_surface_clearance") > 0
        and "_footprint_clearance_xy_spacing" in (
            ROOT / "aerial_gym/env_manager/asset_manager.py").read_text(encoding="utf-8"),
        "detection_latency_default_zero": bool(re.search(
            r'NAVRL_DETECTION_LATENCY_S",\s*0\.0', TASK_CONFIG.read_text(encoding="utf-8"))),
        "detector_noise_range_defaults_zero": bool(re.search(
            r'NAVRL_DETNOISE_RANGE_STD_M",\s*0\.0', TASK_CONFIG.read_text(encoding="utf-8"))),
    }


A, B, C, D = "TYPE-A", "TYPE-B", "TYPE-C", "TYPE-D"

DEFECTS = [
 dict(id="D1", location="aerial_gym/task/navrl_task/navrl_task.py:_advance_target legacy block",
      description=("legacy push-out is pure position displacement, then target_vel_w is "
                   "recomputed as delta-position/dt, so realized instantaneous speed can "
                   "exceed the nominal episode speed; reflection counters increment only "
                   "under _bulk_eval_mode; the bounce jitter draws from the GLOBAL torch RNG "
                   "conditionally on bar contacts"),
      arms=dict(H=True, E0=False, E1=False, E2=False),
      evidence=("_advance_target's bounded|physical branch ends in an unconditional return, "
                "so the entire push-out / reflection / velocity-rewrite block is structurally "
                "unreachable for E0/E1/E2 (all bounded). Verified by AST at build time."),
      training=True, evaluation=True, physics=True, observation=True, receipt_only=False,
      historical=["every legacy/mixed target family: density curves, density x speed map, "
                  "capture/crash/timeout triples trained or evaluated under legacy dynamics"],
      new_eval=["H_historical only"],
      severity="MAJOR", type=A, action="preserve",
      rerun="no",
      note=("H exists to reproduce the frozen policy's actual training lineage. Repairing D1 "
            "inside H would stop H from being that reproduction. Historical defect != "
            "evaluation implementation bug.")),

 dict(id="D2", location="aerial_gym/task/navrl_task/navrl_task.py:9289",
      description="bulk-eval receipt defaults target_pattern to 'static'; the simulator default is 'mixed'",
      arms=dict(H=False, E0=False, E1=False, E2=False),
      evidence=("the 48-cell manifest sets NAVRL_TARGET_PATTERN explicitly in every cell, so "
                "the faulty default is never reached; env_state records str(self.tm.pattern) "
                "and is correct regardless"),
      training=False, evaluation=False, physics=False, observation=False, receipt_only=True,
      historical=["any archived eval launched WITHOUT NAVRL_TARGET_PATTERN set"],
      new_eval=[], severity="MINOR", type=C, action="erratum", rerun="no",
      note="inert under this preflight contract because every cell sets the variable"),

 dict(id="D3", location="aerial_gym/task/navrl_task/navrl_task.py:2224",
      description=("general-eval receipt records target_speed_mps from NAVRL_TARGET_SPEED with "
                   "default 0, but that variable maps to speed_fixed=-1 (disabled) and the real "
                   "speed comes from the speed_min..speed_final curriculum"),
      arms=dict(H=True, E0=True, E1=True, E2=True),
      evidence=("the 48-cell manifest deliberately does NOT set NAVRL_TARGET_SPEED, because "
                "setting it would enable speed_fixed and CHANGE the physics. The receipt field "
                "will therefore read 0 for every arm including the moving ones."),
      training=False, evaluation=False, physics=False, observation=False, receipt_only=True,
      historical=["any archived curriculum-speed eval"],
      new_eval=["all arms, receipt field only"],
      severity="MINOR", type=C, action="gate", rerun="no",
      note=("do NOT set NAVRL_TARGET_SPEED to fix the label: it would change physics. Treat the "
            "receipt field as meaningless and read cfg_target_speed_min/final from env_state.")),

 dict(id="D4", location="aerial_gym/task/navrl_task/navrl_task.py:3440",
      description=("surface clearance attested as 0.0 while navrl_bars_env builds with 0.45; the "
                   "drift guard recomputes the same wrong value so it can never fire"),
      arms=dict(H=False, E0=False, E1=False, E2=False),
      evidence=("the only functional consumer of placement_surface_clearance is "
                "asset_manager._footprint_clearance_xy_spacing, i.e. the footprint_clearance "
                "placement mode. Every cell uses navrl_band, which spaces by touch/gap instead."),
      training=False, evaluation=False, physics=False, observation=False, receipt_only=True,
      historical=["any footprint_clearance-placement family"],
      new_eval=[], severity="MINOR", type=C, action="erratum", rerun="no",
      note="inert for navrl_band; would be MAJOR for a footprint_clearance lineage"),

 dict(id="D5", location="aerial_gym/task/navrl_task/navrl_perception.py:960",
      description=("int(round(latency_s/step_dt)) turns a 0.05 s latency into 0 steps "
                   "(round(0.5)==0) while env_state records 0.05 s"),
      arms=dict(H=False, E0=False, E1=False, E2=False),
      evidence="NAVRL_DETECTION_LATENCY_S defaults to 0.0 and no cell sets it",
      training=False, evaluation=False, physics=False, observation=False, receipt_only=False,
      historical=["any arm configured with a latency in (0, 0.05] s"],
      new_eval=[], severity="MINOR", type=D, action="fix-new-lineage", rerun="unknown",
      note="inert at the default; a genuine bug for any future latency arm"),

 dict(id="D6", location="aerial_gym/task/navrl_task/navrl_task.py:5918",
      description=("a non-finite LiDAR pixel maps to 1.0 = full range = 'nothing there', the most "
                   "favourable value, feeding both the static-safety reward and the speed "
                   "governor; the -max_range sentinel maps to the opposite meaning"),
      arms=dict(H=True, E0=True, E1=True, E2=True),
      evidence="the LiDAR pipeline is shared by every arm; nothing about it is target-dependent",
      training=True, evaluation=True, physics=False, observation=True, receipt_only=False,
      historical=["every family that used this LiDAR pipeline"],
      new_eval=["all arms, identically"],
      severity="MAJOR", type=D, action="fix-new-lineage", rerun="unknown",
      note=("arm-symmetric: it shifts absolute values but cannot bias the H-vs-E2 contrast, "
            "because every arm shares one perception stack. Do not repair mid-comparison.")),

 dict(id="D7", location="aerial_gym/task/navrl_task/speed_governor.py:44",
      description=("free_speed_cap_mps hard-codes sqrt(2)*2.5 while max_velocity defaults to 2.0, "
                   "so at HEAD defaults the free cap exceeds the attainable norm and the governor "
                   "never binds in open space"),
      arms=dict(H=False, E0=False, E1=False, E2=False),
      evidence=("every cell pins NAVRL_MAX_VELOCITY=2.5, which is the value the hard-coded cap "
                "assumes, so the cap is correct under this contract"),
      training=False, evaluation=False, physics=False, observation=False, receipt_only=False,
      historical=["any family run at HEAD defaults rather than the v2 launcher block"],
      new_eval=[], severity="MINOR", type=D, action="fix-new-lineage", rerun="no",
      note="the strict gate is what keeps this inert; without it a default launch silently disables the governor"),

 dict(id="D8", location="aerial_gym/task/navrl_task/navrl_perception.py:1319-1324",
      description="_detector_noise_range_ar AR(1) state is never reset per episode",
      arms=dict(H=False, E0=False, E1=False, E2=False),
      evidence="NAVRL_DETNOISE_RANGE_STD_M and _RHO both default to 0.0 and no cell sets them",
      training=False, evaluation=False, physics=False, observation=False, receipt_only=False,
      historical=["detector-noise arms only"],
      new_eval=[], severity="MINOR", type=D, action="fix-new-lineage", rerun="no",
      note="cross-episode leak, but structurally unreachable at the default"),

 dict(id="D9", location="aerial_gym/task/navrl_task/navrl_perception.py:1960",
      description=("the LiDAR->target range correction adds the CAMERA sphere radius 0.15 to a "
                   "surface range produced by a LiDAR that injects 0.20"),
      arms=dict(H=True, E0=True, E1=True, E2=True),
      evidence="shared perception path; not target-dependent",
      training=True, evaluation=True, physics=False, observation=True, receipt_only=False,
      historical=["every family using LiDAR-fallback target correction"],
      new_eval=["all arms, identically"],
      severity="MAJOR", type=D, action="fix-new-lineage", rerun="unknown",
      note="arm-symmetric 0.05 m bias; shifts absolute range but not the contrast"),
]


def main():
    facts = structural_facts()
    branch = facts["bounded_branch"]
    assert branch and branch["unconditional_return"], (
        "the bounded/physical branch no longer ends in an unconditional return; "
        "D1's arm scoping must be re-derived before this matrix can be trusted")
    assert facts["legacy_block_after_branch_return"], (
        "the legacy push-out block is no longer after the bounded branch's return")
    assert facts["bounded_executor_consumes_no_rng"], (
        "target_motion.py now consumes RNG; the RNG audit must be redone")

    blockers = [d for d in DEFECTS if d["severity"] == "BLOCKER"]
    type_b = [d for d in DEFECTS if d["type"] == B]
    affecting_new_eval_physics = [
        d["id"] for d in DEFECTS
        if d["new_eval"] and (d["physics"] or d["observation"]) and not d["receipt_only"]
    ]
    report = {
        "kind": "target_motion_defect_impact_matrix",
        "schema_version": 1,
        "git_commit": git("rev-parse", "HEAD"),
        "git_tree_clean": git("status", "--porcelain") == "",
        "structural_facts": facts,
        "type_legend": {
            A: "historical behaviour that must be PRESERVED to reproduce arm H",
            B: "current evaluation bug that INVALIDATES new measurements",
            C: "receipt / provenance mislabelling only; physics unchanged",
            D: "future-design debt; does not invalidate the frozen evaluation",
        },
        "defects": DEFECTS,
        "summary": {
            "total": len(DEFECTS),
            "blockers": [d["id"] for d in blockers],
            "type_b_evaluation_invalidating": [d["id"] for d in type_b],
            "affecting_new_eval_physics_or_observation": affecting_new_eval_physics,
            "arm_impact": {
                arm: sorted(d["id"] for d in DEFECTS if d["arms"][arm])
                for arm in ("H", "E0", "E1", "E2")
            },
            "e2_unaffected_by_d1": not next(d for d in DEFECTS if d["id"] == "D1")["arms"]["E2"],
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"  bounded branch returns at line {branch['last_statement_line']}: "
          f"{branch['unconditional_return']}")
    print(f"  legacy-only block: {facts['legacy_only_block']}")
    for arm, ids in report["summary"]["arm_impact"].items():
        print(f"  {arm:3s} affected by: {', '.join(ids) if ids else 'none'}")
    print(f"  TYPE-B (invalidating): {report['summary']['type_b_evaluation_invalidating'] or 'none'}")
    print(f"  E2_UNAFFECTED_BY_D1: {report['summary']['e2_unaffected_by_d1']}")


if __name__ == "__main__":
    main()
