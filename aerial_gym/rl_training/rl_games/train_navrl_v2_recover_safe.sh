#!/usr/bin/env bash
# Recover the v2 density run from a verified last-known-good checkpoint.
#
# Default is a 100-epoch fixed-density smoke test. After its held-out evaluation passes, resume
# the curriculum explicitly with RECOVERY_MODE=curriculum and CKPT=<smoke final checkpoint>.
# If that curriculum run is interrupted, RECOVERY_MODE=continue accepts only its verified lineage.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHON="${PYTHON:-/home/fair/miniconda3/envs/aerialgym/bin/python}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONNOUSERSITE=1

if (( $# != 0 )); then
    echo "[v2-recover] runner arguments are not accepted by the safe wrapper: $*" >&2
    echo "[v2-recover] use only the documented RECOVERY_MODE/CKPT/MAX_EPOCHS environment variables." >&2
    exit 2
fi

RECOVERY_MODE="${RECOVERY_MODE:-smoke}"
case "${RECOVERY_MODE}" in
    smoke|curriculum|continue) ;;
    *)
        echo "[v2-recover] RECOVERY_MODE must be smoke, curriculum, or continue; got: ${RECOVERY_MODE}" >&2
        exit 2
        ;;
esac

# The audited source is seed 1 and the promotion test is held out at seed 42. Letting an inherited
# SEED=42 drive both phases destroys that separation while still producing a syntactically valid
# PASS artifact, so the canonical recovery lineage pins its training seed.
if [[ "${SEED:-1}" != "1" ]]; then
    echo "[v2-recover] canonical recovery training seed is fixed at 1; got SEED=${SEED}." >&2
    exit 2
fi
SEED=1

CKPT_WAS_EXPLICIT=0
if [[ "${CKPT+x}" == "x" && -n "${CKPT}" ]]; then
    CKPT_WAS_EXPLICIT=1
fi
if [[ "${RECOVERY_MODE}" != "smoke" && "${CKPT_WAS_EXPLICIT}" != "1" ]]; then
    echo "[v2-recover] ${RECOVERY_MODE} mode requires an explicit verified CKPT path." >&2
    exit 2
fi

TRUSTED_LKG_SHA256="3a0c167cbf4bc966426488f562da2b6788bd00ca62e3a31f226f5fbe1967578f"
CKPT="${CKPT:-runs/ppo_260731_2012_navrl_v2-search-sched-s1/nn/last_gen_ppo_ep_9500_rew_83.67131.pth}"
export CKPT
if [[ ! -f "${CKPT}" ]]; then
    echo "[v2-recover] checkpoint not found: ${CKPT}" >&2
    exit 2
fi
case "$(basename "${CKPT}")" in
    last_gen_ppo_ep_*.pth) ;;
    *)
        echo "[v2-recover] only last_gen_ppo_ep_*.pth is accepted; best-reward gen_ppo.pth is unsafe for a density continuation." >&2
        exit 2
        ;;
esac

# Recompute the canonical held-out evidence before a smoke checkpoint can unlock curriculum.
# A self-written PASS JSON is not evidence: this verifier rereads the actual result and all 100
# TensorBoard epochs, then requires byte-equivalent fields from the canonical producer.
if [[ "${RECOVERY_MODE}" == "curriculum" ]]; then
    RECOVERY_RUN_ROOT="$(dirname "$(dirname "$(readlink -f -- "${CKPT}")")")"
    RECOVERY_ATTESTATION="${RECOVERY_RUN_ROOT}/.navrl_v2_recovery_eval_pass.json"
    if [[ -f "${RECOVERY_ATTESTATION}" ]]; then
        "${PYTHON}" ../../../tools/navrl_v2_recovery_attestation.py \
            "${CKPT}" --verify-existing "${RECOVERY_ATTESTATION}"
    fi
fi

CHECKPOINT_META="$(
    "${PYTHON}" - "${CKPT}" "${RECOVERY_MODE}" "${TRUSTED_LKG_SHA256}" <<'PY'
import base64
import binascii
import hashlib
import json
import math
from pathlib import Path
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
mode = sys.argv[2]
trusted_lkg_sha256 = sys.argv[3]
state = checkpoint.get("env_state") or {}
epoch = int(checkpoint.get("epoch", -1))
frame = int(checkpoint.get("frame", -1))
bars = int(state.get("n_bars_active", -1))
checkpoint_digest = hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()
attestation_sha256 = "-"
attestation_b64 = "-"
current_learning_rate = 5e-6

# A filename and action-policy match are not enough: a v1 arena or a different sensor/token
# contract can load into the same network shape and train without an obvious exception. Reject
# such semantically incompatible continuations before Isaac Gym is initialized.
expected = {
    "cfg_arena_xy": 40.0,
    "cfg_arena_z": 3.0,
    "cfg_bar_pool": "bars_h3",
    "cfg_placement_mode": "navrl_band",
    "cfg_placement_gap_m": 1.6,
    "cfg_placement_touch_m": 0.4,
    "cfg_bar_x_min": 0.0,
    "cfg_bar_x_max": 1.0,
    "cfg_episode_len_steps": 600.0,
    "cfg_general_goal_dist_min": 6.0,
    "cfg_general_goal_dist_max": 28.0,
    "cfg_lidar_max_range": 12.0,
    "cfg_lidar_hbeams": 72,
    "cfg_lidar_vbeams": 4,
    "cfg_max_obstacles": 8,
    "cfg_token_fov_deg": 240.0,
    "cfg_obstacle_selector": "cluster_sector",
    "cfg_obstacle_cluster_gap_m": 0.45,
    "cfg_obstacle_sectors": 8,
    "cfg_obstacle_suppress_deg": 10.0,
    "cfg_corridor_tokens": 0,
    "cfg_action_policy": "squashed_gaussian",
    "cfg_action_std": "0.35,0.35,0.05,0.08",
    "cfg_action_mu_scale": "1.0,0.4,1.0,1.0",
    # A checkpoint without these fields was trained under the legacy 601-action horizon and
    # without rl_games time-limit bootstrapping.  Joining it to the fixed MDP would create an
    # unlabelled semantic transition, so the safe continuation path rejects it.
    "cfg_episode_limit_comparator": "gte",
    "cfg_rlgames_timeout_info_key": "time_outs",
    "cfg_time_limit_bootstrap_signal": True,
}
problems = []
for key, want in expected.items():
    if key not in state:
        problems.append(f"{key}=missing (expected {want!r})")
        continue
    got = state[key]
    if isinstance(want, float):
        try:
            matches = math.isclose(float(got), want, rel_tol=0.0, abs_tol=1e-6)
        except (TypeError, ValueError):
            matches = False
    else:
        matches = got == want
    if not matches:
        problems.append(f"{key}={got!r} (expected {want!r})")

if mode != "smoke":
    recovery_expected = {
        "cfg_training_seed": 1,
        "cfg_training_num_envs": 128,
        "cfg_training_file": "ppo_navrl_perception_transformer.yaml",
        "cfg_training_task": "navrl_task",
        "cfg_training_sim": "base_sim",
        "cfg_training_profile": "main",
        "cfg_ppo_horizon": 32,
        "k_max_cur": 28.0,
        "k_min_cur": 20.0,
        "cfg_max_velocity": 2.5,
        "cfg_yaw_rate_max": 3.0,
        "cfg_max_tilt_deg": 45.0,
        "cfg_tilt_comp": True,
        "cfg_obstacle_ttc_idle_s": 30.0,
        "cfg_obstacle_ttc_min_speed": 0.15,
        "cfg_corridor_horizon_m": 6.0,
        "cfg_corridor_min_width_m": 0.55,
        "cfg_fov_curriculum_epochs": 3000,
        "cfg_detector_min_pixels": 2,
        "cfg_detector_threshold": 0.55,
        "cfg_detector_checkpoint_name": "",
        "cfg_detector_checkpoint_sha256": "",
        "cfg_perception_perturb": False,
        "cfg_detection_dropout": 0.3,
        "cfg_rgb_noise_std": 0.015,
        "cfg_depth_noise_std": 0.02,
        "cfg_target_motion_model": "symmetric_local_steer_v2_heading_continuity90",
        "cfg_target_pattern": "mixed",
        "cfg_target_speed_min": 0.3,
        "cfg_target_speed_final": 1.5,
        "cfg_target_speed_fixed": -1.0,
        "cfg_target_speed_ramp_epochs": 300,
        "cfg_target_speed_ramp_start_epochs": 0,
        "cfg_general_train": True,
        "cfg_oob_margin": 1.0,
        "cfg_alt_hold_vmax": 2.5,
        "cfg_ppo_log_ratio_clamp": 10.0,
        "cfg_ppo_kl_stop": 0.04,
        "cfg_ppo_epoch_rollback": True,
        "cfg_ppo_rollback_lr_factor": 0.5,
        "cfg_ppo_rollback_min_lr": 1e-6,
        "cfg_ppo_rollback_patience": 5,
        "cfg_density_guard_window_epochs": 50,
        "cfg_density_guard_min_epochs": 100,
        "cfg_density_guard_min_peak": 0.5,
        "cfg_density_guard_drop": 0.25,
        "cfg_density_guard_patience": 25,
        "cfg_latent_margin": "2.0,1.25,2.0,2.0",
        "cfg_lateral_latent_margin_coef": 0.01,
    }
    for key, want in recovery_expected.items():
        got = state.get(key)
        if isinstance(want, float):
            try:
                matches = math.isclose(float(got), want, rel_tol=0.0, abs_tol=1e-9)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = got == want
        if not matches:
            problems.append(f"unsafe recovery metadata {key}={got!r} (expected {want!r})")
    try:
        current_learning_rate = float(state.get("current_action_learning_rate"))
    except (TypeError, ValueError):
        current_learning_rate = float("nan")
    if not math.isfinite(current_learning_rate) or not 1e-6 <= current_learning_rate <= 5e-6:
        problems.append(
            "saved current_action_learning_rate must preserve the rollback range [1e-6,5e-6]; "
            f"got {state.get('current_action_learning_rate')!r}"
        )

def validate_attestation(attestation, expected_checkpoint_digest=None):
    if not isinstance(attestation, dict):
        return ["object"]

    def number_in_range(name, low, high):
        try:
            value = float(attestation.get(name))
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and low <= value <= high

    def integer_equals(name, expected):
        try:
            return int(attestation.get(name)) == expected
        except (TypeError, ValueError):
            return False

    def integer_at_least(name, minimum):
        try:
            return int(attestation.get(name)) >= minimum
        except (TypeError, ValueError):
            return False

    attested_checkpoint_sha = str(attestation.get("checkpoint_sha256", ""))
    checkpoint_sha_ok = (
        attested_checkpoint_sha == expected_checkpoint_digest
        if expected_checkpoint_digest is not None
        else len(attested_checkpoint_sha) == 64
        and all(char in "0123456789abcdef" for char in attested_checkpoint_sha.lower())
    )
    result_sha = str(attestation.get("heldout_result_sha256", ""))
    receipt_sha = str(attestation.get("evaluator_receipt_sha256", ""))
    snapshot_sha = str(
        attestation.get("evaluated_checkpoint_snapshot_sha256", "")
    )
    evaluator_sha = str(attestation.get("evaluator_script_sha256", ""))

    def valid_sha256(value):
        return len(value) == 64 and all(
            char in "0123456789abcdef" for char in value.lower()
        )

    evaluation_contract_expected = {
        "schema_version": 2,
        "episode_limit_steps": 600,
        "episode_limit_comparator": "gte",
        "timeout_observed_at_step": 600,
        "pursuer_speed_limit_semantics": "per_axis_xy",
        "pursuer_per_axis_speed_limit_mps": 2.5,
        "pursuer_max_horizontal_request_norm_mps": math.sqrt(2.0) * 2.5,
        "policy_output_dim": 4,
        "policy_z_output_overwritten_by_altitude_pi": True,
        "policy_z_persisted_in_prev_action_observation": True,
        "runtime_sim": "base_sim",
        "runtime_profile": "main",
        "runtime_num_envs": 128,
        "sim_physics_contract": "base_sim_dt0.01",
        "runtime_sim_config_class": "BaseSimConfig",
        "physics_dt_s": 0.01,
        "physics_substeps": 1,
        "physics_steps_per_rl_step": 10,
        "rl_step_dt_s": 0.1,
        "arena_xy_m": 40.0,
        "goal_dist_min_m": 6.0,
        "goal_dist_max_m": 28.0,
        "full_goal_distribution": True,
        "fov_curriculum_saturated": True,
        "target_speed_distribution": "uniform",
        "target_speed_min_mps": 0.3,
        "target_speed_max_mps": 1.5,
        "target_pattern": "mixed",
        "lidar_beams": [4, 72],
        "lidar_range_m": 12.0,
        "obstacle_tokens": 8,
        "obstacle_fov_deg": 240.0,
        "obstacle_selector": "cluster_sector",
        "obstacle_ttc_idle_s": 30.0,
        "obstacle_ttc_min_speed": 0.15,
        "fov_curriculum_epochs": 3000,
        "detector_checkpoint_sha256": "",
        "detector_min_pixels": 2,
        "detector_threshold": 0.55,
        "perception_perturb": False,
        "detection_dropout": 0.3,
        "detection_dropout_active": 0.0,
        "rgb_noise_std": 0.015,
        "depth_noise_std": 0.02,
        "max_tilt_deg": 45.0,
        "tilt_comp": True,
        "oob_margin_m": 1.0,
        "seed": 42,
    }
    evaluation_contract = attestation.get("evaluation_contract")
    contract_failed = []
    if not isinstance(evaluation_contract, dict):
        contract_failed.append("evaluation_contract")
    else:
        for name, expected in evaluation_contract_expected.items():
            got = evaluation_contract.get(name)
            if isinstance(expected, bool):
                matches = isinstance(got, bool) and got is expected
            elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
                try:
                    number = float(got)
                    matches = math.isfinite(number) and math.isclose(
                        number, float(expected), rel_tol=0.0, abs_tol=1e-9
                    )
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = got == expected
            if not matches:
                contract_failed.append("evaluation_contract." + name)

    try:
        episodes = int(attestation.get("episodes", -1))
        requested = int(attestation.get("requested_episodes", -1))
        counts = [int(attestation.get(name, -1)) for name in ("captured", "crash", "timeout")]
        rates = [float(attestation.get(name)) for name in ("capture_rate", "crash_rate", "timeout_rate")]
        rate_accounting_ok = (
            requested >= 2049
            and episodes >= requested
            and all(value >= 0 for value in counts)
            and sum(counts) == episodes
            and all(math.isfinite(value) for value in rates)
            and all(
                math.isclose(rate, count / episodes, rel_tol=0.0, abs_tol=1e-9)
                for rate, count in zip(rates, counts)
            )
        )
    except (TypeError, ValueError, ZeroDivisionError):
        rate_accounting_ok = False
    checks = {
        "schema_version": attestation.get("schema_version") == 1,
        "verdict": attestation.get("verdict") == "PASS",
        "checkpoint_sha256": checkpoint_sha_ok,
        "checkpoint_epoch": integer_equals("checkpoint_epoch", 9600),
        "checkpoint_frame": integer_equals("checkpoint_frame", 39321600),
        "source_checkpoint_sha256": (
            attestation.get("source_checkpoint_sha256") == trusted_lkg_sha256
        ),
        "source_epoch": integer_equals("source_epoch", 9500),
        "smoke_epochs": integer_equals("smoke_epochs", 100),
        "bars": integer_equals("bars", 130),
        "episodes": integer_at_least("episodes", 2049),
        "outcome_accounting": rate_accounting_ok,
        "seed": integer_equals("seed", 42),
        "capture_rate": number_in_range("capture_rate", 0.65, 1.0),
        "crash_rate": number_in_range("crash_rate", 0.0, 0.35),
        "timeout_rate": number_in_range("timeout_rate", 0.0, 0.10),
        "training_max_kl": number_in_range("training_max_kl", -1e-6, 0.04),
        "max_task_input_oob_rate": number_in_range(
            "max_task_input_oob_rate", 0.0, 1e-9
        ),
        "max_rollback_streak": number_in_range("max_rollback_streak", 0.0, 0.0),
        "final_rollback_streak": number_in_range(
            "final_rollback_streak", 0.0, 0.0
        ),
        "max_epoch_rollback": number_in_range("max_epoch_rollback", 0.0, 0.0),
        "max_rollback_total": number_in_range("max_rollback_total", 0.0, 0.0),
        "final_rollback_total": number_in_range("final_rollback_total", 0.0, 0.0),
        "heldout_result_sha256": valid_sha256(result_sha),
        "evaluator_receipt_sha256": valid_sha256(receipt_sha),
        "evaluated_checkpoint_snapshot_sha256": (
            valid_sha256(snapshot_sha) and snapshot_sha == attested_checkpoint_sha
        ),
        "evaluator_script_sha256": valid_sha256(evaluator_sha),
    }
    return [name for name, passed in checks.items() if not passed] + contract_failed


if mode == "smoke":
    if checkpoint_digest != trusted_lkg_sha256:
        problems.append(
            "checkpoint SHA-256 is not the audited ep9500 last-known-good source "
            f"({checkpoint_digest})"
        )
    if epoch != 9500 or frame != 38912000 or bars != 130:
        problems.append(
            f"smoke source epoch/frame/bars={epoch}/{frame}/{bars} "
            "(expected 9500/38912000/130)"
        )
    if int(state.get("num_task_steps", -1)) != 304000:
        problems.append("smoke source num_task_steps is not the audited 304000")
    if float(state.get("k_max_cur", -1.0)) != 28.0 or float(
        state.get("k_min_cur", -1.0)
    ) != 20.0:
        problems.append("smoke source distance curriculum is not the audited [20,28] m")
else:
    try:
        source_epoch = int(state.get("cfg_recovery_source_epoch", -1))
        required_epochs = int(state.get("cfg_recovery_smoke_required_epochs", -1))
        smoke_bars = int(state.get("cfg_recovery_smoke_bars", -1))
    except (TypeError, ValueError):
        source_epoch = required_epochs = smoke_bars = -1
    if state.get("cfg_recovery_source_sha256") != trusted_lkg_sha256:
        problems.append("checkpoint does not descend from the audited ep9500 source")
    if source_epoch != 9500 or required_epochs != 100 or smoke_bars != 130:
        problems.append(
            "invalid smoke lineage metadata: "
            f"source={source_epoch}, required={required_epochs}, bars={smoke_bars}"
        )
    if epoch < source_epoch + required_epochs:
        problems.append(
            f"smoke is incomplete: epoch={epoch}, need >= {source_epoch + required_epochs}"
        )
    if mode == "curriculum":
        if epoch != 9600 or frame != 39321600:
            problems.append(
                f"curriculum input epoch/frame={epoch}/{frame} "
                "(expected exact smoke final 9600/39321600)"
            )
        if int(state.get("num_task_steps", -1)) != 307200:
            problems.append("curriculum input num_task_steps is not exact smoke final 307200")
        if bars != 130:
            problems.append(f"curriculum input density is {bars}, expected smoke density 130")
        if state.get("cfg_recovery_stage") != "smoke":
            problems.append("checkpoint was not produced by RECOVERY_MODE=smoke")
        run_root = Path(sys.argv[1]).resolve().parents[1]
        finished_marker = run_root / ".aerial_training_finished"
        if not finished_marker.is_file():
            problems.append(f"normal-completion marker is missing: {finished_marker}")
        attestation_path = run_root / ".navrl_v2_recovery_eval_pass.json"
        if not attestation_path.is_file():
            problems.append(
                "held-out PASS attestation is missing; run the documented 130-bar/2049-episode eval: "
                f"{attestation_path}"
            )
        else:
            try:
                attestation_bytes = attestation_path.read_bytes()
                attestation = json.loads(attestation_bytes.decode("utf-8"))
            except (OSError, ValueError) as exc:
                problems.append(f"invalid held-out attestation: {exc}")
            else:
                attestation_sha256 = hashlib.sha256(attestation_bytes).hexdigest()
                attestation_b64 = base64.b64encode(attestation_bytes).decode("ascii")
                failed = validate_attestation(attestation, checkpoint_digest)
                if failed:
                    problems.append("held-out attestation failed fields: " + ", ".join(failed))
                result_path = Path(str(attestation.get("heldout_result_json", ""))).expanduser()
                expected_result_sha = str(attestation.get("heldout_result_sha256", ""))
                if not result_path.is_file():
                    problems.append(f"attested held-out result is missing: {result_path}")
                else:
                    actual_result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
                    if actual_result_sha != expected_result_sha:
                        problems.append("attested held-out result SHA-256 does not match the artifact")
    elif mode == "continue":
        expected_task_steps = 307200 + (epoch - 9600) * 32
        expected_frame = 39321600 + (epoch - 9600) * 4096
        if epoch <= 9600:
            problems.append(f"continue input epoch must be >9600; got {epoch}")
        if int(state.get("num_task_steps", -1)) != expected_task_steps:
            problems.append(
                "continue num_task_steps breaks the smoke-anchor lineage: "
                f"got {state.get('num_task_steps')!r}, expected {expected_task_steps}"
            )
        if frame != expected_frame:
            problems.append(
                f"continue frame breaks the smoke-anchor lineage: got {frame}, "
                f"expected {expected_frame}"
            )
        allowed_bars = set(range(130, 301, 15)) | {300}
        if bars not in allowed_bars:
            problems.append(
                f"continue density {bars} is outside the pinned 130:+15:300 schedule"
            )
        if state.get("cfg_recovery_stage") != "curriculum":
            problems.append("continue checkpoint was not produced by the safe curriculum mode")
        attestation_sha256 = str(
            state.get("cfg_recovery_eval_attestation_sha256", "")
        ).strip()
        if len(attestation_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in attestation_sha256.lower()
        ):
            problems.append("continue checkpoint lacks a valid held-out attestation lineage hash")
        attestation_b64 = str(
            state.get("cfg_recovery_eval_attestation_b64", "")
        ).strip()
        try:
            attestation_bytes = base64.b64decode(attestation_b64, validate=True)
            embedded_digest = hashlib.sha256(attestation_bytes).hexdigest()
            attestation = json.loads(attestation_bytes.decode("utf-8"))
        except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append(f"continue checkpoint has invalid embedded attestation: {exc}")
        else:
            if embedded_digest != attestation_sha256:
                problems.append("embedded attestation does not match its saved lineage hash")
            failed = validate_attestation(attestation)
            if failed:
                problems.append(
                    "embedded held-out attestation failed fields: " + ", ".join(failed)
                )
if problems:
    print("[v2-recover] incompatible checkpoint contract:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    raise SystemExit(2)

print(
    epoch,
    bars,
    str(state.get("cfg_action_policy", "")),
    attestation_sha256,
    current_learning_rate,
    attestation_b64,
)
PY
)"
read -r START_EPOCH START_BARS SAVED_POLICY EVAL_ATTESTATION_SHA256 \
    SAVED_CURRENT_LR EVAL_ATTESTATION_B64 <<< "${CHECKPOINT_META}"
if (( START_EPOCH < 1 || START_BARS < 1 || START_BARS > 300 )); then
    echo "[v2-recover] invalid checkpoint epoch/density: epoch=${START_EPOCH} bars=${START_BARS}" >&2
    exit 2
fi
if [[ "${SAVED_POLICY}" != "squashed_gaussian" ]]; then
    echo "[v2-recover] checkpoint action policy is ${SAVED_POLICY:-missing}, expected squashed_gaussian." >&2
    exit 2
fi

export NAVRL_V2_ALLOW_RESUME=1
# Pin the complete executable contract. A stale interactive shell must not turn the audited
# recovery into a different task/YAML/network, halve its sample budget, bypass the global lock, or
# hide it from the dashboard under an unrelated run tag.
export FILE=ppo_navrl_perception_transformer.yaml
export TASK=navrl_task
export NUM_ENVS=128
export AERIAL_GYM_SIM_NAME=base_sim
export HEADLESS=True
export NAVRL_SEED=1
unset GPU4GB NAVRL_V2_PROFILE ALLOW_CONCURRENT NAVRL_NETWORK_OVERRIDE
unset NAVRL_V2_FORCE
unset NAVRL_GENERAL_EVAL NAVRL_INTERACTIVE NAVRL_BULK_EVAL NAVRL_BULK_EVAL_JSON
unset NAVRL_GENERAL_RESULTS_JSON NAVRL_EVAL_CHECKPOINT NAVRL_EVAL_TARGET_SPEED_FINAL
unset NAVRL_EVAL_FULL_DISTRIBUTION
unset NAVRL_EVAL_RUN_NONCE NAVRL_EVAL_PROFILE NAVRL_SIM_PHYSICS_CONTRACT
unset NAVRL_LEGACY_VISION NAVRL_OOB_PROBE
export NAVRL_OBSTACLE_SELECTOR=cluster_sector
export NAVRL_OBSTACLE_TTC_IDLE_S=30.0
export NAVRL_OBSTACLE_TTC_MIN_SPEED=0.15
export NAVRL_CORRIDOR_HORIZON_M=6.0
export NAVRL_CORRIDOR_MIN_WIDTH_M=0.55
export NAVRL_FOV_CURRICULUM_EPOCHS=3000
export NAVRL_DETECTOR_MIN_PIXELS=2
export NAVRL_DETECTOR_THRESHOLD=0.55
unset NAVRL_DETECTOR_CHECKPOINT
export NAVRL_DETECTION_DROPOUT=0.3
export NAVRL_RGB_NOISE_STD=0.015
export NAVRL_DEPTH_NOISE_STD=0.02
export NAVRL_MAX_TILT_DEG=45.0
export NAVRL_TARGET_SPEED_RAMP_EPOCHS=300
export NAVRL_K_MIN_RAMP_START=2000
export NAVRL_K_MIN_RAMP_EPOCHS=3000
export NAVRL_K_WARMUP=3000
export NAVRL_K_THRESHOLD=0.6
export NAVRL_K_STEP=2.0
export NAVRL_K_CHECK=2048
export NAVRL_DENSITY_START=70
export NAVRL_DENSITY_FINAL=300
export NAVRL_DENSITY_STEP=15
export NAVRL_DENSITY_THRESHOLD=0.80
export NAVRL_DENSITY_THRESHOLD_START=0.80
export NAVRL_DENSITY_THRESHOLD_END=0.70
export NAVRL_DENSITY_THRESHOLD_SCHEDULE=70:0.82,85:0.77,100:0.72,115:0.70
export NAVRL_DENSITY_WARMUP=1000
export NAVRL_DENSITY_CHECK_EPS=16384
export NAVRL_DENSITY_STRATIFIED_GATE=0
export NAVRL_DENSITY_STRATIFIED_FLOOR=0.55
export NAVRL_DENSITY_STRATIFIED_MIN_EPS=512
export NAVRL_DENSITY_EASY_GOAL_MIX=0
export NAVRL_DENSITY_EASY_GOAL_MIN=5.0
export NAVRL_DENSITY_EASY_GOAL_MAX=10.0
# Same likelihood geometry: preserve the competent checkpoint's Adam moments. Resetting this
# optimizer also resets shared Transformer/value-head moments and is only valid for a policy-family
# conversion, not this recovery.
export NAVRL_RESET_ACTOR_OPTIMIZER=0
export NAVRL_ACTION_DIAG=1
export NAVRL_PPO_LOG_RATIO_CLAMP=10.0
export NAVRL_PPO_KL_STOP=0.04
export NAVRL_PPO_EPOCH_ROLLBACK=1
export NAVRL_PPO_ROLLBACK_LR_FACTOR=0.5
export NAVRL_PPO_ROLLBACK_MIN_LR=1e-6
export NAVRL_PPO_ROLLBACK_PATIENCE=5
export NAVRL_DENSITY_GUARD_WINDOW_EPOCHS=50
export NAVRL_DENSITY_GUARD_MIN_EPOCHS=100
export NAVRL_DENSITY_GUARD_MIN_PEAK=0.50
export NAVRL_DENSITY_GUARD_DROP=0.25
export NAVRL_DENSITY_GUARD_PATIENCE=25
export NAVRL_LATENT_MARGIN=2.0,1.25,2.0,2.0
export NAVRL_LATENT_MARGIN_COEF=0.01
if [[ "${RECOVERY_MODE}" == "smoke" ]]; then
    export NAVRL_LEARNING_RATE=5e-6
else
    # Never undo a safety backoff recorded by the preceding smoke/curriculum checkpoint.
    export NAVRL_LEARNING_RATE="${SAVED_CURRENT_LR}"
fi
export NAVRL_RECOVERY_STAGE="${RECOVERY_MODE}"
if [[ "${RECOVERY_MODE}" == "continue" ]]; then
    export NAVRL_RECOVERY_STAGE=curriculum
fi
export NAVRL_RECOVERY_SOURCE_EPOCH=9500
export NAVRL_RECOVERY_SOURCE_SHA256="${TRUSTED_LKG_SHA256}"
export NAVRL_RECOVERY_SMOKE_REQUIRED_EPOCHS=100
export NAVRL_RECOVERY_SMOKE_BARS=130
if [[ "${EVAL_ATTESTATION_SHA256}" != "-" ]]; then
    export NAVRL_RECOVERY_EVAL_ATTESTATION_SHA256="${EVAL_ATTESTATION_SHA256}"
    export NAVRL_RECOVERY_EVAL_ATTESTATION_B64="${EVAL_ATTESTATION_B64}"
else
    unset NAVRL_RECOVERY_EVAL_ATTESTATION_SHA256
    unset NAVRL_RECOVERY_EVAL_ATTESTATION_B64
fi

case "${RECOVERY_MODE}" in
    smoke)
        export NAVRL_DENSITY_CURRICULUM=0
        export NAVRL_NUM_BARS="${START_BARS}"
        export NAVRL_DENSITY_MIN_EPOCHS=0
        export NAVRL_RESET_DENSITY_WINDOW=1
        MAX_EPOCHS="$((START_EPOCH + 100))"
        export NAVRL_DENSITY_RESUME_WARMUP=0
        DEFAULT_TAG="v2-recover-smoke-${NAVRL_NUM_BARS}bars-s${SEED:-1}"
        ;;
    curriculum)
        export NAVRL_DENSITY_CURRICULUM=1
        unset NAVRL_NUM_BARS
        export NAVRL_DENSITY_MIN_EPOCHS=1000
        export NAVRL_RESET_DENSITY_WINDOW=1
        export NAVRL_DENSITY_RESUME_WARMUP=250
        MAX_EPOCHS="${MAX_EPOCHS:-30000}"
        DEFAULT_TAG="v2-recover-curriculum-s${SEED:-1}"
        ;;
    continue)
        export NAVRL_DENSITY_CURRICULUM=1
        unset NAVRL_NUM_BARS
        export NAVRL_DENSITY_MIN_EPOCHS=1000
        export NAVRL_RESET_DENSITY_WINDOW=0
        export NAVRL_DENSITY_RESUME_WARMUP=0
        MAX_EPOCHS="${MAX_EPOCHS:-30000}"
        DEFAULT_TAG="v2-recover-curriculum-continue-s${SEED:-1}"
        ;;
esac
if (( MAX_EPOCHS <= START_EPOCH )); then
    echo "[v2-recover] MAX_EPOCHS=${MAX_EPOCHS} must exceed checkpoint epoch ${START_EPOCH}." >&2
    exit 2
fi

export MAX_EPOCHS
export SEED
export AERIAL_RUN_TAG="${DEFAULT_TAG}"
export TRAIN_SESSION_LOG="train_session_logs/v2_recover_${RECOVERY_MODE}_$(date +%y%m%d_%H%M%S)_$$.log"
export TRAIN_LIVE_LOG=train_session_logs/current_training.log

echo "[v2-recover] mode=${RECOVERY_MODE} LKG=${CKPT} epoch=${START_EPOCH} bars=${START_BARS}"
echo "[v2-recover] contract | file=${FILE} task=${TASK} envs=${NUM_ENVS} selector=${NAVRL_OBSTACLE_SELECTOR} density=${NAVRL_DENSITY_START}->${NAVRL_DENSITY_FINAL}"
echo "[v2-recover] lr=${NAVRL_LEARNING_RATE} optimizer_reset=${NAVRL_RESET_ACTOR_OPTIMIZER} max_epoch=${MAX_EPOCHS}"
echo "[v2-recover] safety | KL=${NAVRL_PPO_KL_STOP:-0.04} epoch rollback=on all-axis margin=${NAVRL_LATENT_MARGIN}@${NAVRL_LATENT_MARGIN_COEF}"
echo "[v2-recover] semantics | seed=${SEED} ttc=${NAVRL_OBSTACLE_TTC_IDLE_S}/${NAVRL_OBSTACLE_TTC_MIN_SPEED} detector=analytic tilt=${NAVRL_MAX_TILT_DEG}deg"

if [[ "${NAVRL_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    # Exercise the real child-launcher handoff too.  Stopping here previously let an unexported
    # CKPT pass recovery preflight and then fail only when the user started the actual run.
    export NAVRL_V2_CONTRACT_PREFLIGHT_ONLY=1
    echo "[v2-recover] recovery contract PASS; validating child-launcher handoff"
fi

exec ./train_navrl_v2_search.sh --checkpoint "${CKPT}" --branch_run
