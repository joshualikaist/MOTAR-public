import math

import numpy as np
import os
import re
import sys
import yaml

from pathlib import Path
from datetime import datetime
from typing import Optional

from aerial_gym.rl_training.rl_games.quiet_rl_io import (
    devnull_stdio,
    install_quiet_print_filter,
    mute_info_logging,
    patch_rlgames_quiet_train_init,
    quiet_startup_enabled,
)

# Fail closed BEFORE isaacgym/torch load and before any GPU work: prove that the aerial_gym
# package actually executing is the one the caller intended. An editable install's meta_path
# finder hard-codes one worktree's absolute path, and play_navrl.sh cds into
# aerial_gym/rl_training/rl_games where no aerial_gym/ package directory exists -- so a run
# started from a different worktree silently executes the primary tree while its source
# manifest hashes the worktree's bytes. Inert unless NAVRL_REQUIRE_SOURCE_ROOT is set.
from aerial_gym.rl_training.rl_games.navrl_import_origin import assert_runtime_origin

_origin_info = assert_runtime_origin()
if _origin_info.get("enforced"):
    print(
        "[origin] aerial_gym %s sha256=%s (enforced)"
        % (_origin_info.get("origin"), _origin_info.get("origin_sha256"))
    )

_real_print = __import__("builtins").print

if quiet_startup_enabled():
    with devnull_stdio():
        import isaacgym

        from aerial_gym.registry.task_registry import task_registry
        from aerial_gym.utils.helpers import parse_arguments
else:
    import isaacgym

    from aerial_gym.registry.task_registry import task_registry
    from aerial_gym.utils.helpers import parse_arguments

import gym
from gym import spaces
from argparse import Namespace

from rl_games.common import env_configurations, vecenv

import torch
import distutils

from aerial_gym.rl_training.rl_games.training_safety import (
    apply_density_capture_guard_overrides,
)
from aerial_gym.rl_training.rl_games.ppo_update_safety import (
    resolve_action_learning_rate,
)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# import warnings
# warnings.filterwarnings("error")

# Vision mode emits obs['states'] for the asymmetric critic during TRAINING. The rl_games PLAYER
# expects a plain actor-obs tensor (observation_space is a Box, not a Dict), so returning the dict
# at play time causes an observation-shape failure. Detect every player alias and hand the player
# only the actor observation; the critic/states are a train-only concern.
_PLAY_MODE = ("--play" in sys.argv) or ("-p" in sys.argv) or ("--eval" in sys.argv)


class ExtractObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    @staticmethod
    def _extract(observations):
        # Asymmetric actor-critic (navrl vision mode): during TRAINING the task also emits
        # privileged critic input under 'states'. rl_games' obs_to_tensors keeps a dict containing
        # an 'obs' key as-is, and get_action_values feeds obs['states'] to the central value net.
        # At PLAY the player wants a plain actor-obs tensor, so drop 'states' there (_PLAY_MODE).
        if "states" in observations and not _PLAY_MODE:
            return {"obs": observations["observations"], "states": observations["states"]}
        return observations["observations"]

    def reset(self, **kwargs):
        observations, *_ = super().reset(**kwargs)
        return self._extract(observations)

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)

        dones = torch.where(
            terminated | truncated,
            torch.ones_like(terminated),
            torch.zeros_like(terminated),
        )

        return (
            self._extract(observations),
            rewards,
            dones,
            infos,
        )

    def render(self, mode="human", **kwargs):
        # Tasks vary: some implement render(), some render(mode=...)
        try:
            return self.env.render(mode=mode, **kwargs)
        except TypeError:
            return self.env.render()


class AERIALRLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.env = ExtractObsWrapper(self.env)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def get_env_state(self):
        # rl_games saves this into the checkpoint ('env_state'); forward to the task (via the
        # gym.Wrapper) so per-task state like navrl_task's curriculum counter survives resume.
        fn = getattr(self.env, "get_env_state", None)
        return fn() if callable(fn) else None

    def set_env_state(self, state):
        fn = getattr(self.env, "set_env_state", None)
        if callable(fn) and state is not None:
            fn(state)

    def render(self, mode="human"):
        """rl_games BasePlayer expects vecenv.render(mode=...) during play."""
        if hasattr(self.env, "render"):
            return self.env.render(mode=mode)
        return None

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = spaces.Box(
            -np.ones(self.env.task_config.action_space_dim),
            np.ones(self.env.task_config.action_space_dim),
        )
        info["observation_space"] = spaces.Box(
            np.ones(self.env.task_config.observation_space_dim) * -np.Inf,
            np.ones(self.env.task_config.observation_space_dim) * np.Inf,
        )
        # navrl vision mode: expose the privileged-critic state space so rl_games'
        # central_value_config builds against the right shape (falls back to observation_space
        # if absent, which would silently defeat the asymmetric critic).
        state_dim = int(getattr(self.env.task_config, "state_space_dim", 0) or 0)
        if state_dim > 0:
            info["state_space"] = spaces.Box(
                np.ones(state_dim) * -np.Inf, np.ones(state_dim) * np.Inf
            )
        if not quiet_startup_enabled():
            print(info["action_space"], info["observation_space"])
        return info


env_configurations.register(
    "position_setpoint_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("position_setpoint_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_sim2real",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_sim2real", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_sim2real_px4",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_sim2real_px4", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_acceleration_sim2real",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_acceleration_sim2real", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "navigation_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("navigation_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_reconfigurable",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_reconfigurable", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_morphy",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_morphy", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_sim2real_end_to_end",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_sim2real_end_to_end", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "shooting_moving_target_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "shooting_moving_target_task", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "navrl_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("navrl_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

vecenv.register(
    "AERIAL-RLGPU",
    lambda config_name, num_actors, **kwargs: AERIALRLGPUEnv(config_name, num_actors, **kwargs),
)


def get_args():
    from isaacgym import gymutil

    custom_parameters = [
        {
            "name": "--seed",
            "type": int,
            "default": 0,
            "required": False,
            "help": "Random seed, if larger than 0 will overwrite the value in yaml config.",
        },
        {
            "name": "--tf",
            "required": False,
            "help": "run tensorflow runner",
            "action": "store_true",
        },
        {
            "name": "--train",
            "required": False,
            "help": "train network",
            "action": "store_true",
        },
        {
            "name": "--play",
            "required": False,
            "help": "play(test) network",
            "action": "store_true",
        },
        {
            "name": "--eval",
            "required": False,
            "help": "same as --play (checkpoint inference; avoids accidental TRAIN branch)",
            "action": "store_true",
        },
        {
            "name": "--checkpoint",
            "type": str,
            "required": False,
            "help": "path to checkpoint",
        },
        {
            "name": "--resume_in_place",
            "action": "store_true",
            "default": False,
            "help": "Reuse the checkpoint's run folder even if that run is already marked "
            "finished. By default, checkpoints from finished runs warm-start a new run folder.",
        },
        {
            "name": "--branch_run",
            "action": "store_true",
            "default": False,
            "help": "Warm-start into a new run folder even when the source run was interrupted. "
            "Use this when changing the environment/config so the source metrics stay clean.",
        },
        {
            "name": "--disable_collapse_early_stop",
            "action": "store_true",
            "default": False,
            "help": "Disable only the reward-drop-from-peak early-stop rule for this run. "
            "NaN/Inf fail-fast remains enabled.",
        },
        {
            "name": "--file",
            "type": str,
            "default": "ppo_aerial_quad.yaml",
            "required": False,
            "help": "path to config",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": "1024",
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--sigma",
            "type": float,
            "required": False,
            "help": "sets new sigma value in case if 'fixed_sigma: True' in yaml config",
        },
        {
            "name": "--track",
            "action": "store_true",
            "help": "if toggled, this experiment will be tracked with Weights and Biases",
        },
        {
            "name": "--wandb-project-name",
            "type": str,
            "default": "rl_games",
            "help": "the wandb's project name",
        },
        {
            "name": "--wandb-entity",
            "type": str,
            "default": None,
            "help": "the entity (team) of wandb's project",
        },
        {
            "name": "--task",
            "type": str,
            "default": "navigation_task",
            "help": "Override task from config file if provided.",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "False",
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--use_warp",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "True",
            "help": "Choose whether to use warp or Isaac Gym rendeing pipeline.",
        },
        {
            "name": "--max_epochs",
            "type": int,
            "default": -1,
            "help": "Override max_epochs in the yaml (> 0). Needed to EXTEND a warm-start: a "
            "--checkpoint resume restores the epoch counter, so a Phase-B run must raise this "
            "above the checkpoint's epoch (e.g. Phase A ended at 6000 -> --max_epochs 12000).",
        },
    ]

    # parse arguments
    args = parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def _prepare_rlgames_argv_dict(cli: dict):
    """rl_games TorchRunner treats both train=False and play=False as TRAIN — easy to misuse."""
    cli.setdefault("train", False)
    cli.setdefault("play", False)
    cli.setdefault("checkpoint", None)
    cli.setdefault("eval", False)
    cli.setdefault("track", False)
    cli["train"] = bool(cli["train"])
    cli["play"] = bool(cli["play"])
    if cli.pop("eval", False):
        cli["play"] = True
    if cli["train"] and cli["play"]:
        raise SystemExit("[aerial RL] --train 과 --eval/--play 은 동시에 쓸 수 없습니다.")

    ck = cli.get("checkpoint")
    ck_set = ck is not None and str(ck).strip() != ""

    if not cli["train"] and not cli["play"]:
        if ck_set and not quiet_startup_enabled():
            print(
                "[aerial RL WARNING] --checkpoint 만 있고 --train / --play / --eval 이 없습니다.\n"
                "  rl-games 기본값은 학습 재개(TRAIN, TensorBoard 에 epoch 기록)입니다.\n"
                "  재생만 하려면:  --play 또는 --checkpoint 를 쓸 때 같은 줄에 반드시 --play (별칭 --eval)\n"
                "  학습 재개 의도면:  --train 명시를 권장합니다."
            )
        if not quiet_startup_enabled():
            print(
                "[aerial RL] Mode = TRAIN (implicit — TorchRunner 의 default)\n"
                "            추론만: runner.py … --play --checkpoint PATH   (별칭: --eval)"
            )
        # Normalize TorchRunner's implicit default into explicit internal state. Downstream
        # checkpoint/LR/provenance logic must not have to rediscover that false/false means train.
        cli["train"] = True
    elif cli["play"]:
        if not quiet_startup_enabled():
            print("[aerial RL] Mode = PLAY (inference, no PPO 학습 루프·TensorBoard epoch)")
            if not ck_set:
                print("[aerial RL WARNING] --play 인데 체크포인트 없음 → 가중치는 초기 상태입니다.")
    elif not quiet_startup_enabled():
        print("[aerial RL] Mode = TRAIN (--train 또는 기본 학습 재개 동작)")
    return cli


def _run_root_from_checkpoint(ckpt_path, base_dir=None):
    """Resolve runs/<run_name>/ from .../runs/<run_name>/nn/<file>.pth."""
    if ckpt_path is None:
        return None
    s = str(ckpt_path).strip()
    if not s:
        return None
    base = Path(base_dir or os.getcwd()).resolve()
    p = Path(s)
    if not p.is_absolute():
        p = (base / p).resolve()
    else:
        p = p.resolve()
    if p.parent.name != "nn":
        return None
    run_root = p.parent.parent
    if run_root.parent.name == "runs":
        return run_root
    return None


def _run_name_from_checkpoint(ckpt_path, base_dir=None):
    run_root = _run_root_from_checkpoint(ckpt_path, base_dir=base_dir)
    return run_root.name if run_root is not None else None


def _record_run_lineage(new_run_name, source_run_root, base_dir=None):
    """Write ``<new run>/aerial_run/resumed_from.txt`` naming the run this one continues.

    Warm-starting into a new folder keeps each run's metrics clean, but it also means one
    continuous training lineage lands in TensorBoard as several disjoint curves, and nothing on
    disk said which folder continued which. Rebuilding the combined view then depended on somebody
    remembering the order. This one line makes the chain walkable by tools/tb_merge_lineage.py.

    Best effort: a failure here must never stop a run from starting.
    """
    try:
        base = Path(base_dir or os.getcwd()).resolve()
        target = base / "runs" / str(new_run_name) / "aerial_run"
        target.mkdir(parents=True, exist_ok=True)
        (target / "resumed_from.txt").write_text(
            f"{source_run_root.name}\n", encoding="utf-8"
        )
    except Exception:
        pass


def _task_run_suffix(task: Optional[str]) -> str:
    """Short label appended to run folder names (fixed vs moving intercept, etc.)."""
    if not task:
        return ""
    mapping = {
        "position_setpoint_task": "fixed",
        "shooting_moving_target_task": "moving",
        "navigation_task": "nav",
        "navrl_task": "navrl",
    }
    if task in mapping:
        return mapping[task]
    # Fallback: strip common _task suffix for readability.
    if task.endswith("_task"):
        return task[: -len("_task")].replace("_", "-")
    return task.replace("_", "-")


def _new_experiment_name(task: Optional[str]) -> str:
    """runs/ folder name: ppo_YYMMDD_HHMM_<fixed|moving|…>."""
    stamp = datetime.now().strftime("ppo_%y%m%d_%H%M")
    suffix = _task_run_suffix(task)
    name = f"{stamp}_{suffix}" if suffix else stamp
    raw_tag = os.environ.get("AERIAL_RUN_TAG", "").strip()
    if raw_tag:
        tag = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw_tag).strip("-._")
        if not tag:
            raise ValueError("AERIAL_RUN_TAG contains no usable filename characters")
        name = f"{name}_{tag}"
    return name


_ACTION_POLICY_MODELS = {
    "fixed_gaussian": "navrl_fixed_gaussian",
    "squashed_gaussian": "navrl_squashed_gaussian",
    "truncated_gaussian": "navrl_truncated_gaussian",
}


def _canonical_action_policy(value):
    raw = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "": "",
        "legacy": "legacy",
        "legacy_gaussian": "legacy",
        "gaussian": "legacy",
        "fixed": "fixed_gaussian",
        "fixed_gaussian": "fixed_gaussian",
        "tanh": "squashed_gaussian",
        "tanh_gaussian": "squashed_gaussian",
        "squashed": "squashed_gaussian",
        "squashed_gaussian": "squashed_gaussian",
        "truncated": "truncated_gaussian",
        "sa_truncated": "truncated_gaussian",
        "truncated_gaussian": "truncated_gaussian",
    }
    if raw not in aliases:
        raise ValueError(
            "NAVRL_ACTION_POLICY must be one of legacy, fixed_gaussian, "
            "squashed_gaussian, truncated_gaussian; got %r" % value
        )
    return aliases[raw]


def _checkpoint_action_contract(checkpoint):
    """Read action-distribution provenance without modifying the checkpoint."""
    if not checkpoint or not os.path.isfile(str(checkpoint)):
        return {}
    try:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        env_state = state.get("env_state")
        return env_state if isinstance(env_state, dict) else {}
    except Exception:
        return {}


def _apply_action_policy_config(config, args):
    """Select a local bounded-action model and restore its checkpoint metadata for evaluation."""
    checkpoint_contract = _checkpoint_action_contract(args.get("checkpoint"))
    explicit_policy = os.environ.get("NAVRL_ACTION_POLICY", "").strip()
    saved_policy = checkpoint_contract.get("cfg_action_policy", "")
    policy = _canonical_action_policy(explicit_policy or saved_policy or "legacy")

    if not explicit_policy and saved_policy:
        os.environ["NAVRL_ACTION_POLICY"] = policy
    if not os.environ.get("NAVRL_ACTION_STD", "").strip():
        saved_std = str(checkpoint_contract.get("cfg_action_std", "")).strip()
        if saved_std:
            os.environ["NAVRL_ACTION_STD"] = saved_std
    if not os.environ.get("NAVRL_ACTION_MU_SCALE", "").strip():
        saved_mu_scale = str(checkpoint_contract.get("cfg_action_mu_scale", "")).strip()
        if saved_mu_scale:
            os.environ["NAVRL_ACTION_MU_SCALE"] = saved_mu_scale
    if not os.environ.get("NAVRL_TRUNCATED_DMIN", "").strip():
        saved_dmin = checkpoint_contract.get("cfg_truncated_dmin")
        if saved_dmin is not None:
            os.environ["NAVRL_TRUNCATED_DMIN"] = str(saved_dmin)

    if explicit_policy and saved_policy:
        saved_canonical = _canonical_action_policy(saved_policy)
        if saved_canonical != policy:
            print(
                "[aerial RL WARNING] action-policy branch changes checkpoint contract: "
                f"{saved_canonical} -> {policy}.",
                flush=True,
            )

    train_cfg = config["params"]["config"]
    if policy in _ACTION_POLICY_MODELS:
        config["params"]["model"]["name"] = _ACTION_POLICY_MODELS[policy]
        os.environ.setdefault("NAVRL_ACTION_STD", "0.35,0.35,0.05,0.08")
        # A pre-tanh mean may legitimately exceed 1.1.  Truncated locations are already bounded.
        if policy in ("squashed_gaussian", "truncated_gaussian"):
            train_cfg["bounds_loss_coef"] = 0.0

    entropy_override = os.environ.get("NAVRL_ENTROPY_COEF", "").strip()
    if entropy_override:
        entropy = float(entropy_override)
        if not np.isfinite(entropy) or entropy < 0.0:
            raise ValueError("NAVRL_ENTROPY_COEF must be finite and >= 0")
        train_cfg["entropy_coef"] = entropy

    learning_rate_override = os.environ.get("NAVRL_LEARNING_RATE", "").strip()
    saved_current_lr = checkpoint_contract.get("current_action_learning_rate")
    # TorchRunner's historical default is TRAIN even when both flags are false.  Treat every
    # non-play checkpoint path as a training resume, otherwise the implicit resume route silently
    # discards a saved rollback LR while the explicit --train route preserves it.
    resume_training = not bool(args.get("play")) and bool(checkpoint_contract)
    learning_rate = resolve_action_learning_rate(
        train_cfg.get("learning_rate", 0.0),
        explicit_override=learning_rate_override,
        saved_current=saved_current_lr,
        resume_training=resume_training,
    )
    train_cfg["learning_rate"] = learning_rate
    if resume_training and not learning_rate_override and saved_current_lr is not None:
        print(
            "[aerial RL] Resume optimizer LR restored from checkpoint safety state: %.3g."
            % learning_rate,
            flush=True,
        )

    guard_cfg = train_cfg.get("early_stop_density_capture")
    if isinstance(guard_cfg, dict):
        guard_cfg = apply_density_capture_guard_overrides(guard_cfg, os.environ)
        train_cfg["early_stop_density_capture"] = guard_cfg
        print(
            "[aerial RL] same-density guard | window=%d min_epochs=%d "
            "min_peak=%.3f drop=%.3f patience=%d"
            % (
                int(guard_cfg.get("window_epochs", 50)),
                int(guard_cfg.get("min_epochs_at_density", 100)),
                float(guard_cfg.get("min_peak_capture", 0.50)),
                float(guard_cfg.get("drop_absolute", 0.25)),
                int(guard_cfg.get("patience_epochs", 25)),
            ),
            flush=True,
        )

    os.environ.setdefault("NAVRL_ACTION_POLICY", policy)
    os.environ["NAVRL_ENTROPY_COEF"] = str(float(train_cfg.get("entropy_coef", 0.0)))
    # Checkpoint provenance must record the optimizer value that is actually active, including the
    # YAML default. Previously this environment key was set only for an explicit override, so a
    # real 1e-4 optimizer was falsely stored as cfg_action_learning_rate=0.
    os.environ["NAVRL_LEARNING_RATE"] = str(
        float(train_cfg.get("learning_rate", 0.0))
    )
    os.environ["NAVRL_CURRENT_LEARNING_RATE"] = os.environ["NAVRL_LEARNING_RATE"]
    print(
        "[aerial RL] action policy | mode=%s std=%s mu_scale=%s entropy=%g "
        "bounds_loss=%g lr=%g"
        % (
            policy,
            os.environ.get("NAVRL_ACTION_STD", "checkpoint/legacy"),
            os.environ.get("NAVRL_ACTION_MU_SCALE", "1"),
            float(train_cfg.get("entropy_coef", 0.0)),
            float(train_cfg.get("bounds_loss_coef", 0.0)),
            float(train_cfg.get("learning_rate", 0.0)),
        ),
        flush=True,
    )


def _normalize_vf_checkpoint(path):
    """Warm-start fix: torch.compile prefixes the CENTRAL-VALUE (critic) state_dict keys with
    '_orig_mod.' when the checkpoint is saved, but the freshly-built central_value_net at restore
    expects no prefix -> set_full_state_weights' strict load_state_dict('assymetric_vf_nets')
    raises on a `--checkpoint --train` resume of a vision (asymmetric-critic) run. Strip the prefix
    into a sibling *_rlnorm.pth and return that path (or the original if there is nothing to fix).
    The actor 'model' keys are left untouched -- rl_games restores those fine."""
    if not path or not os.path.isfile(path):
        return path
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return path
    cv = ck.get("assymetric_vf_nets")
    if not isinstance(cv, dict) or not any("_orig_mod." in k for k in cv):
        return path
    ck["assymetric_vf_nets"] = {k.replace("_orig_mod.", ""): v for k, v in cv.items()}
    out = (path[:-4] if path.endswith(".pth") else path) + "_rlnorm.pth"
    torch.save(ck, out)
    if not quiet_startup_enabled():
        print(f"[aerial RL] normalized central-value checkpoint keys -> {out}")
    return out


def _expand_corridor_checkpoint(path):
    """Rewrite a pre-corridor (17-token) checkpoint for the corridor-token schema.

    Gated on NAVRL_CORRIDOR_WARMSTART=1 with NAVRL_CORRIDOR_TOKENS>0. Corridor features are
    APPENDED at the end of the actor observation (before the privileged tail of the critic
    state), so the expansion is purely structural -- no trained weight is reinterpreted:

      actor model : position_embedding (1,17,E) -> (1,18,E) with one fresh N(0,0.02) row;
                    corridor_project.* injected at nn.Linear default init;
                    input running_mean_std mean/var padded with 0/1 for the new features.
      critic nets : first MLP weight [H, obs+priv] -> [H, obs+add+priv] with the privileged
                    COLUMNS MOVED to the new tail and zero columns for corridor features
                    (zero-init = critic initially ignores them; no value shock);
                    critic input running_mean_std remapped the same way.
      optimizers  : actor Adam state cleared and param count grown by the 4 corridor_project
                    tensors; critic Adam state cleared (its first-layer shape changed).
                    Fresh moments are the correct choice for a schema change.

    All fresh values come from a fixed-seed generator so the expansion is reproducible. The
    original file is never modified; a *_corridor<K>.pth sibling is written and returned.
    env_state provenance (cfg_corridor_tokens absent/0) is intentionally left as trained --
    the preflight override + set_env_state warning document the schema change loudly.
    """
    flag = os.environ.get("NAVRL_CORRIDOR_WARMSTART", "0").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return path
    corridor_tokens = int(os.environ.get("NAVRL_CORRIDOR_TOKENS", "0").strip() or 0)
    if corridor_tokens <= 0 or not path or not os.path.isfile(path):
        return path
    from aerial_gym.task.navrl_task.navrl_corridor import CORRIDOR_DIM

    add = corridor_tokens * CORRIDOR_DIM
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ck.get("model")
    if not isinstance(model, dict):
        return path
    pe_key = next(
        (k for k in model if k.endswith("a2c_network.position_embedding")), None
    )
    if pe_key is None:
        return path
    pe = model[pe_key]
    if pe.shape[1] == 18:
        return path  # already expanded (idempotent)
    if pe.shape[1] != 17:
        raise RuntimeError(
            f"corridor warm-start expects a 17-token checkpoint, found {tuple(pe.shape)}"
        )
    prefix = pe_key[: -len("position_embedding")]

    gen = torch.Generator().manual_seed(20260731)
    new_row = torch.empty(1, 1, pe.shape[2], dtype=pe.dtype).normal_(std=0.02, generator=gen)
    model[pe_key] = torch.cat([pe, new_row], dim=1)

    def _fresh_linear(fan_out, fan_in):
        # nn.Linear.reset_parameters: kaiming_uniform(a=sqrt(5)) == U(-1/sqrt(fan_in), +...),
        # same law for the bias. Spelled out so a seeded generator can be used.
        bound = 1.0 / math.sqrt(fan_in)
        w = torch.empty(fan_out, fan_in).uniform_(-bound, bound, generator=gen)
        b = torch.empty(fan_out).uniform_(-bound, bound, generator=gen)
        return w, b

    w0, b0 = _fresh_linear(128, add)
    w2, b2 = _fresh_linear(pe.shape[2], 128)
    model[prefix + "corridor_project.0.weight"] = w0
    model[prefix + "corridor_project.0.bias"] = b0
    model[prefix + "corridor_project.2.weight"] = w2
    model[prefix + "corridor_project.2.bias"] = b2

    # Actor input normalizer: new features start at identity normalization (mean 0, var 1).
    old_obs = None
    for k in list(model.keys()):
        if "value_mean_std" in k:
            continue
        if k.endswith("running_mean_std.running_mean"):
            old_obs = model[k].numel()
            model[k] = torch.cat([model[k], torch.zeros(add, dtype=model[k].dtype)])
        elif k.endswith("running_mean_std.running_var"):
            model[k] = torch.cat([model[k], torch.ones(add, dtype=model[k].dtype)])
    if old_obs is None:
        raise RuntimeError("corridor warm-start: actor running_mean_std not found")

    def _remap_cols(t, old_o, priv):
        # [.., old_o | priv] -> [.., old_o | add(new) | priv]
        head, tail = t[..., :old_o], t[..., old_o:]
        mid_shape = list(t.shape[:-1]) + [add]
        mid = torch.zeros(*mid_shape, dtype=t.dtype)
        return torch.cat([head, mid, tail], dim=-1), priv

    vf = ck.get("assymetric_vf_nets")
    if isinstance(vf, dict):
        for k in list(vf.keys()):
            v = vf[k]
            if "value_mean_std" in k or not hasattr(v, "shape"):
                continue
            if k.endswith("running_mean_std.running_mean") and v.numel() > old_obs:
                priv = v.numel() - old_obs
                vf[k], _ = _remap_cols(v, old_obs, priv)
            elif k.endswith("running_mean_std.running_var") and v.numel() > old_obs:
                head, tail = v[:old_obs], v[old_obs:]
                vf[k] = torch.cat([head, torch.ones(add, dtype=v.dtype), tail])
            elif v.ndim == 2 and v.shape[1] == old_obs + 8:
                vf[k], _ = _remap_cols(v, old_obs, v.shape[1] - old_obs)

    opt = ck.get("optimizer")
    if isinstance(opt, dict) and opt.get("param_groups"):
        opt["state"] = {}
        for pg in opt["param_groups"]:
            pg["params"] = list(range(len(pg["params"]) + 4))
            break  # single param group; the 4 corridor_project tensors join it
    vopt = ck.get("assymetric_vf_optimizer")
    if isinstance(vopt, dict):
        vopt["state"] = {}  # critic first-layer shape changed; param count is unchanged

    out = (path[:-4] if path.endswith(".pth") else path) + f"_corridor{corridor_tokens}.pth"
    torch.save(ck, out)
    print(
        f"[aerial RL] corridor warm-start: expanded 17->18 tokens, obs {old_obs}->"
        f"{old_obs + add} (K={corridor_tokens}); optimizer moments reset -> {out}",
        flush=True,
    )
    return out


def update_config(config, args):

    if args.get("max_epochs", -1) and int(args.get("max_epochs", -1)) > 0:
        config["params"]["config"]["max_epochs"] = int(args["max_epochs"])

    if args.get("disable_collapse_early_stop"):
        train_cfg = config["params"]["config"]
        collapse_cfg = train_cfg.get("early_stop_collapse")
        collapse_cfg = dict(collapse_cfg) if isinstance(collapse_cfg, dict) else {}
        collapse_cfg["enable"] = False
        collapse_cfg["disabled_by"] = "--disable_collapse_early_stop"
        train_cfg["early_stop_collapse"] = collapse_cfg

    if args.get("task") is not None:
        config["params"]["config"]["env_name"] = args["task"]
    if args.get("experiment_name") is not None:
        config["params"]["config"]["name"] = args["experiment_name"]
    config["params"]["config"]["env_config"]["headless"] = args["headless"]
    config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    config["params"]["config"]["env_config"]["use_warp"] = args["use_warp"]
    if args["num_envs"] > 0:
        config["params"]["config"]["num_actors"] = args["num_envs"]
        # config['params']['config']['num_envs'] = args['num_envs']
        config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    if args["seed"] > 0:
        config["params"]["seed"] = args["seed"]
        config["params"]["config"]["env_config"]["seed"] = args["seed"]

    _apply_action_policy_config(config, args)

    player_cfg = config["params"]["config"].get("player")
    if not isinstance(player_cfg, dict):
        player_cfg = {}
    player_cfg = dict(player_cfg)
    player_cfg["use_vecenv"] = True
    if args.get("play"):
        # Task 쪽 PLAY 대시보드만 쓰고 rl-games 기본 플레이 로그는 끈다.
        player_cfg["print_stats"] = False
        # Evaluation must be able to reproduce both the deployed mean action and the stochastic
        # policy that generated the on-policy density gate.  The YAML default is deterministic;
        # require an explicit, validated mode only when the evaluator requests an override.
        eval_action_mode = os.environ.get("NAVRL_EVAL_ACTION_MODE", "").strip().lower()
        if eval_action_mode:
            if eval_action_mode not in ("deterministic", "stochastic"):
                raise ValueError(
                    "NAVRL_EVAL_ACTION_MODE must be deterministic or stochastic, got %r"
                    % eval_action_mode
                )
            player_cfg["deterministic"] = eval_action_mode == "deterministic"
        try:
            _gn = int(os.environ.get("PLAY_GAMES_NUM", "64"))
        except ValueError:
            _gn = 64
        player_cfg["games_num"] = max(1, _gn)
    config["params"]["config"]["player"] = player_cfg

    network_override = os.environ.get("NAVRL_NETWORK_OVERRIDE", "").strip()
    if network_override:
        config["params"]["network"]["name"] = network_override

    # Runs folder: ppo_YYMMDD_HHMM_<fixed|moving|…> (local time).
    # Interrupted runs have no finished marker and resume in place. A checkpoint from a completed
    # run is a warm-start branch: putting it back into the source folder mixes old/future epochs,
    # TensorBoard events and checkpoint names, so it gets a new folder by default. PLAY is
    # read-only and may keep the source name; --resume_in_place is the explicit escape hatch.
    _cfg = config["params"]["config"]
    fe = _cfg.get("full_experiment_name", None)
    if isinstance(fe, str) and fe.strip():
        pass
    else:
        resumed_root = _run_root_from_checkpoint(args.get("checkpoint"))
        source_finished = bool(
            resumed_root is not None
            and (resumed_root / ".aerial_training_finished").is_file()
        )
        reuse_source = bool(
            resumed_root is not None
            and not args.get("branch_run")
            and (
                args.get("play")
                or args.get("resume_in_place")
                or not source_finished
            )
        )
        if reuse_source:
            _cfg["full_experiment_name"] = resumed_root.name
        else:
            task = args.get("task") or _cfg.get("env_name")
            _cfg["full_experiment_name"] = _new_experiment_name(task)
            if resumed_root is not None:
                _record_run_lineage(_cfg["full_experiment_name"], resumed_root)
                if not quiet_startup_enabled():
                    print(
                        "[aerial RL] warm-starting a new run folder "
                        f"({_cfg['full_experiment_name']}). Use --resume_in_place to override."
                    )

    return config


if __name__ == "__main__":
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    install_quiet_print_filter()
    mute_info_logging()

    args = vars(get_args())
    args = _prepare_rlgames_argv_dict(args)

    # Warm-start (--checkpoint --train) of a vision run needs the central-value keys normalized
    # (torch.compile '_orig_mod.' prefix); rewrite the path to the normalized copy before restore.
    if args.get("checkpoint") and not args.get("play"):
        # Corridor expansion first: it matches keys by suffix, so it works on the raw prefixes,
        # and the normalizer then sees the already-expanded tensors.
        args["checkpoint"] = _expand_corridor_checkpoint(args["checkpoint"])
        args["checkpoint"] = _normalize_vf_checkpoint(args["checkpoint"])

    config_name = args["file"]

    if not quiet_startup_enabled():
        print("Loading config: ", config_name)
    with open(config_name, "r") as stream:
        config = yaml.safe_load(stream)

        config = update_config(config, args)

        from rl_games.torch_runner import Runner

        from aerial_gym.rl_training.rl_games.early_stop_a2c_agent import EarlyStopA2CAgent

        runner = Runner()
        # Reward-stability early stop — see YAML `early_stop_stable`.
        runner.algo_factory.register_builder(
            "a2c_continuous", lambda **kwargs: EarlyStopA2CAgent(**kwargs)
        )

        # Custom networks (select via params.network.name in the YAML).
        from rl_games.algos_torch import model_builder

        from aerial_gym.rl_training.rl_games.navrl_network import NavRLCNNBuilder
        from aerial_gym.rl_training.rl_games.navrl_vision_network import NavRLVisionBuilder
        from aerial_gym.rl_training.rl_games.navrl_vision_legacy_network import (
            NavRLVisionLegacyBuilder,
        )
        from aerial_gym.rl_training.rl_games.navrl_transformer_network import (
            NavRLTransformerBuilder,
        )
        from aerial_gym.rl_training.rl_games.navrl_action_models import (
            NavRLFixedGaussianModel,
            NavRLSquashedGaussianModel,
            NavRLTruncatedGaussianModel,
        )
        from aerial_gym.rl_training.rl_games.navrl_players import NavRLPpoPlayerContinuous

        model_builder.register_network("navrl_cnn", NavRLCNNBuilder)
        model_builder.register_network("navrl_vision", NavRLVisionBuilder)
        model_builder.register_network("navrl_vision_legacy", NavRLVisionLegacyBuilder)
        model_builder.register_network("navrl_transformer", NavRLTransformerBuilder)
        model_builder.register_model("navrl_fixed_gaussian", NavRLFixedGaussianModel)
        model_builder.register_model("navrl_squashed_gaussian", NavRLSquashedGaussianModel)
        model_builder.register_model("navrl_truncated_gaussian", NavRLTruncatedGaussianModel)
        # Backward compatible for legacy models; bounded models add an explicit deterministic action.
        runner.player_factory.register_builder(
            "a2c_continuous", lambda **kwargs: NavRLPpoPlayerContinuous(**kwargs)
        )

        # Task-aware dashboard config: banner title, episode-length cap and the optional
        # reward "design range" hint all follow the task being trained.
        try:
            _dash_task = str(args.get("task") or config["params"]["config"].get("env_name", ""))
            from aerial_gym.rl_training.rl_games.run_header import _TASK_TITLES

            # Surface the run folder name (ppo_YYMMDD_HHMM_<task>) so the dashboard box title AND
            # the recurring "NavRL progress" line show which run this terminal is — avoids mixing up
            # concurrent runs. AERIAL_RUN_NAME is read by navrl_task._log_progress.
            _run_name = str(config["params"]["config"].get("full_experiment_name") or "").strip()
            os.environ.setdefault("AERIAL_RUN_NAME", _run_name)
            os.environ.setdefault(
                "AERIAL_RUN_TITLE",
                f"Aerial RL  ·  {_TASK_TITLES.get(_dash_task, _dash_task or 'PPO')}",
            )
            os.environ.setdefault(
                "AERIAL_TASK_CONFIG_MODULE",
                f"aerial_gym.config.task_config.{_dash_task}_config",
            )
            if _dash_task in ("position_setpoint_task", "shooting_moving_target_task"):
                os.environ.setdefault("AERIAL_REWARD_DESIGN_MAX", "1000")
        except Exception:
            pass

        # Prettier boxed stats only during TRAIN; PLAY 에서는 학습처럼 보이게 하지 않음.
        if not args.get("play"):
            try:
                _rl_here = Path(__file__).resolve().parent
                if str(_rl_here) not in sys.path:
                    sys.path.insert(0, str(_rl_here))
                from pretty_train_stats import install_boxed_statistics

                install_boxed_statistics()
            except Exception:
                pass

        try:
            if quiet_startup_enabled() and not args.get("play"):
                with devnull_stdio():
                    runner.load(config)
            else:
                runner.load(config)
        except yaml.YAMLError as exc:
            _real_print(exc)

        from aerial_gym.rl_training.rl_games.run_header import print_run_header

        print_run_header(
            args,
            config,
            mode="play" if args.get("play") else "train",
        )

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if args["track"] and rank == 0:
        import wandb

        wandb.init(
            project=args["wandb_project_name"],
            entity=args["wandb_entity"],
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )
    if not args.get("play"):
        patch_rlgames_quiet_train_init()
    runner.run(args)

    if args["track"] and rank == 0:
        wandb.finish()
