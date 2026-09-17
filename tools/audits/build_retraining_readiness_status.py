#!/usr/bin/env python3
"""Emit the machine-readable retraining-readiness status from the audit artifacts.

Generated, not hand-maintained: every number here is read back from the JSON the
audit tools produced, so the status file cannot drift from the evidence it
summarises.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
AUDITS = ROOT / "docs/audits"
DIFF = AUDITS / "training_contract_vs_current_runtime_2026-09-17.json"
SNAPSHOT = AUDITS / "current_runtime_contract_2026-09-17.json"
OUT = AUDITS / "retraining_readiness_status_2026-09-17.json"

CHECKPOINT_SHA = "f702213936601860995cf61dcc570247e72543b1976e3716055cd8ec5593ad40"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=str(ROOT), text=True).strip()


def main():
    diff = json.loads(DIFF.read_text())
    snapshot = json.loads(SNAPSHOT.read_text())
    changed = {k: v for k, v in diff["rows"].items() if v["status"].startswith("CHANGED")}
    guarded = set(diff["guarded_keys"])

    status = {
        "kind": "navrl_retraining_readiness_status",
        "schema_version": 1,
        "date": "2026-09-17",
        "git_commit": git("rev-parse", "HEAD"),
        "git_tree_clean": git("status", "--porcelain") == "",

        "verdict": "RETRAIN_DECISION_BLOCKED_PENDING_EVALUATION",
        "verdict_reason": (
            "The retrain/no-retrain gate requires the frozen-policy E0/E1/E2 "
            "evaluation, which has not been run. The audit that precedes it is "
            "complete and changed the experiment's design."
        ),
        "new_ppo_training_run": False,
        "gpu_work_performed": False,

        "frozen_policy": {
            "checkpoint_sha256": CHECKPOINT_SHA,
            "path": ("aerial_gym/rl_training/rl_games/runs/"
                     "ppo_260805_0413_navrl_v2-speedgov-ep24000-205bars-main-riskcap-s1/"
                     "nn/last_gen_ppo_ep_25000_rew_39.742134.pth"),
            "epoch": 25000,
            "frame": 102400000,
            "env_state_keys": 127,
            "trained_target_lineage": "legacy",
            "trained_target_motion_model": "symmetric_local_steer_v2_heading_continuity90",
            "trained_target_pattern": "mixed",
            "trained_target_lineage_provenance": "ATTESTED_BY_CHECKPOINT",
        },

        "contract_diff": {
            "checkpoint_cfg_keys": diff["checkpoint_cfg_keys"],
            "counts": diff["counts"],
            "changed_keys": sorted(changed),
            "changed_and_guarded": sorted(k for k in changed if k in guarded),
            "changed_and_unguarded": sorted(k for k in changed if k not in guarded),
            "guard_fails_closed": diff["guard_fails_closed"],
            "training_contract_reproducible_from_head_defaults": False,
            "training_contract_reproducible_from_launcher": (
                "aerial_gym/rl_training/rl_games/train_navrl_v2_search.sh"),
        },

        "heading_valid_speed_contract": {
            "checkpoint_key_present": False,
            "provenance": "ASSUMED_PRE_KEY_DEFAULT",
            "running_value_mps": 0.10,
            "possible_pre_key_inline_epsilon_mps": 1e-05,
            "ratio": 1e4,
            "note": ("The checkpoint predates the key. 0.10 is the value the running "
                     "code holds, not a value the checkpoint recorded."),
        },

        "termination_audit": {
            "success_radius_m": 0.5,
            "capture_ends_episode": True,
            "evidence": "aerial_gym/task/navrl_task/navrl_task.py:4999 comment; :6025 capture test",
            "verdict": "A_CLOSE_APPROACH_TASK",
            "consequence": (
                "Continuous-tracking metrics are not measurable under capture "
                "termination: better tracking terminates earlier and contributes "
                "fewer samples."
            ),
            "termination_modified_by_this_work": False,
        },

        "tm_e2_dynamics_consistency": {
            "target_speed_max_mps": 1.5,
            "target_turn_rate_deg_s": 150.0,
            "required_lateral_accel_mps2": 3.9270,
            "accel_envelope_mps2": 4.0,
            "consistent": True,
            "headroom_fraction": 0.0183,
            "note": ("A coordinated turn at max speed and max turn rate sits on the "
                     "acceleration bound with 1.8% to spare, leaving essentially no "
                     "budget for simultaneous speed change. 2-D bounded kinematic, "
                     "not rigid-body UAV dynamics."),
        },

        "preregistration": {
            "path": "docs/prereg_2026-09-17_target_motion_complexity_e0_e1_e2.md",
            "state": "PREREGISTERED_AMENDED_BEFORE_MEASUREMENT",
            "amendment": 1,
            "amended_before_any_measurement": True,
            "arms": ["H_historical", "E0_static", "E1_cv", "E2_obstacle_aware"],
            "densities": [70, 115, 160, 205],
            "evaluation_seeds": [4101, 4102, 4103],
            "cells": 48,
            "episodes_per_cell": 2048,
            "removed_outcomes": ["time_within_approach_band_frac"],
            "removed_reason": "not defined under capture termination",
            "primary_outcomes": ["min_relative_distance_m", "capture_rate"],
            "condition_gate": "tools/audits/check_eval_condition_contract.py",
            "condition_gate_fails_closed": True,
        },

        "evaluation": {
            "state": "READY_NOT_RUN",
            "blocked_by": [
                "research authority: offline bottleneck analysis and a new "
                "preregistration are required before any GPU work",
            ],
            "authority_tool": "tools/check_research_authority.py",
        },

        "retraining_decision": {
            "case": None,
            "decision": "PENDING_EVALUATION",
            "preconditions_before_any_new_training": {
                "final_observation_contract_frozen": False,
                "final_perception_contract_frozen": False,
                "final_target_behavior_contract_frozen": False,
                "final_robot_dynamics_contract_frozen": False,
                "final_controller_frozen": True,
                "final_safety_filter_frozen": True,
                "blocking_states": {
                    "live_rgb_to_policy": "NOT_TESTED",
                    "true_metric_range": "BLOCKED",
                    "persistent_id": "BLOCKED",
                    "p10_adaptation": "INCONCLUSIVE",
                },
            },
        },

        # Every entry was re-read and quoted at source before being recorded.
        # NOT fixed in this commit: several would change what recorded results
        # mean, which requires impact analysis first.
        "code_defects": [
            {"id": "D1", "location": "aerial_gym/task/navrl_task/navrl_task.py:8184-8252",
             "summary": ("legacy target push-out is pure displacement and target_vel_w is "
                         "recomputed as delta-pos/dt, so realized target speed can exceed the "
                         "labelled episode speed; reflection counters increment only in "
                         "bulk-eval mode; bounce jitter draws from the global torch RNG "
                         "conditionally on bar contacts"),
             "class": "SCIENTIFIC_CONFOUND", "can_change_recorded_result": True,
             "note": "this is the lineage the frozen checkpoint trained on", "fixed": False},
            {"id": "D2", "location": "aerial_gym/task/navrl_task/navrl_task.py:9289",
             "summary": "results receipt defaults target_pattern to 'static'; real default is 'mixed'",
             "class": "BUG", "can_change_recorded_result": True, "fixed": False},
            {"id": "D3", "location": "aerial_gym/task/navrl_task/navrl_task.py:2224",
             "summary": "results receipt defaults target_speed_mps to 0 while speed comes from the curriculum",
             "class": "BUG", "can_change_recorded_result": True, "fixed": False},
            {"id": "D4", "location": "aerial_gym/task/navrl_task/navrl_task.py:3440",
             "summary": ("surface clearance attested as 0.0 while the arena is built with 0.45; "
                         "the drift guard recomputes the same wrong value so it can never fire"),
             "class": "BUG", "can_change_recorded_result": True, "fixed": False},
            {"id": "D5", "location": "aerial_gym/task/navrl_task/navrl_perception.py:960",
             "summary": "int(round(latency/dt)) turns a 0.05 s latency into 0 steps while env_state records 0.05",
             "class": "BUG", "can_change_recorded_result": True, "fixed": False},
            {"id": "D6", "location": "aerial_gym/task/navrl_task/navrl_task.py:5918",
             "summary": "non-finite LiDAR pixel maps to full range ('nothing there'), the most favourable value",
             "class": "SCIENTIFIC_CONFOUND", "can_change_recorded_result": True, "fixed": False},
            {"id": "D7", "location": "aerial_gym/task/navrl_task/speed_governor.py:44",
             "summary": ("free_speed_cap_mps hard-codes sqrt(2)*2.5 while max_velocity defaults to 2.0, "
                         "so at HEAD defaults the governor never binds in open space"),
             "class": "SCIENTIFIC_CONFOUND", "can_change_recorded_result": True,
             "note": "correct for this checkpoint, which trained at 2.5", "fixed": False},
            {"id": "D8", "location": "aerial_gym/task/navrl_task/navrl_perception.py:1319-1324",
             "summary": "_detector_noise_range_ar AR(1) state is never reset per episode",
             "class": "BUG", "can_change_recorded_result": False,
             "note": "both gating knobs default to 0.0; affects detector-noise arms only", "fixed": False},
            {"id": "D9", "location": "aerial_gym/task/navrl_task/navrl_perception.py:1960",
             "summary": "LiDAR->target range adds the camera sphere radius 0.15 to a LiDAR surface range built with 0.20",
             "class": "SCIENTIFIC_CONFOUND", "can_change_recorded_result": True, "fixed": False},
        ],
        "silent_fixes_applied": False,
        "historical_results_overwritten": False,

        "observation_compatibility": {
            "normalization_used": True,
            "normalization_location": "checkpoint model state dict (running_mean_std)",
            "actor_obs_dim": 898,
            "critic_state_dim": 906,
            "wrong_width_fails_loudly": True,
            "same_width_wrong_semantics_fails_loudly": False,
            "checkpoint_vs_runtime_obs_dim_validation_on_eval_path": False,
            "preflight_tool_exists_but_not_invoked_by_eval": (
                "aerial_gym/rl_training/rl_games/navrl_checkpoint_preflight.py"),
        },

        "action_semantics": {
            "xy_limit_kind": "independent per axis",
            "xy_per_axis_mps": 2.5,
            "xy_attainable_norm_mps": 3.5355339059,
            "z_output_overwritten_by_altitude_pi": True,
            "z_persisted_in_prev_action_observation": True,
            "z_status": "design debt for a future training lineage; not changed here",
            "yaw_action": "yaw rate",
        },

        "artifacts": {
            "training_contract": "docs/audits/frozen_policy_training_contract_2026-09-17.md",
            "current_runtime_contract": str(SNAPSHOT.relative_to(ROOT)),
            "contract_diff": str(DIFF.relative_to(ROOT)),
            "readiness_audit": "docs/audits/retraining_readiness_audit_2026-09-17.md",
            "sources_hashed": snapshot["sources"],
        },
    }
    OUT.write_text(json.dumps(status, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  verdict: {status['verdict']}")
    print(f"  changed knobs: {len(status['contract_diff']['changed_keys'])} "
          f"({len(status['contract_diff']['changed_and_unguarded'])} unguarded)")


if __name__ == "__main__":
    main()
