"""Suppress startup / rl-games chatter; keep compact PPO dashboard prints."""

from __future__ import annotations

import builtins
import logging
import os
import sys
from contextlib import contextmanager
from typing import Iterator

_REAL_PRINT = builtins.print
_PATCHED = False

# Bypass quiet print filter (run banner, errors).
real_print = _REAL_PRINT


def quiet_startup_enabled() -> bool:
    return os.environ.get("AERIAL_RL_QUIET_STARTUP", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _allow_print(*args: object) -> bool:
    if not args:
        return False
    msg = " ".join(str(a) for a in args)
    if "Aerial RL" in msg and ("PPO" in msg or "PLAY" in msg):
        return True
    if "body-center intercept" in msg:
        return True
    if "epoch" in msg and "step time" in msg:
        return True
    if "mean reward" in msg:
        return True
    if "intercept success" in msg:
        return True
    if msg.strip().startswith("─") or msg.strip().startswith("━"):
        return True
    if "[aerial RL] Early stop" in msg:
        return True
    if msg.startswith("[aerial RL]") and any(
        marker in msg
        for marker in (
            "PPO epoch rejection",
            "PPO EPOCH ROLLBACK",
            "FAIL-STOP",
            "Resume optimizer",
            "Actor optimizer",
            "action policy",
            "policy-actiondiag",
            "Best-checkpoint guard",
            "density ",
        )
    ):
        return True
    if msg.strip() in ("MAX EPOCHS NUM!", "MAX FRAMES NUM!"):
        return True
    if msg.startswith("WARNING: Max epochs") or msg.startswith("WARNING: Max frames"):
        return True
    if "[record_train_params]" in msg:
        return True
    if "Evaluation did not reach" in msg or "Heuristic checklist" in msg:
        return True
    if msg.startswith("[train]"):
        return True
    if msg.startswith("NAVRL_BULK_EVAL_RESULT "):
        return True
    if msg.startswith(" >>"):
        return True
    return False


def _filtered_print(*args, **kwargs) -> None:
    if _allow_print(*args):
        _REAL_PRINT(*args, **kwargs)


def install_quiet_print_filter() -> None:
    global _PATCHED
    if _PATCHED or not quiet_startup_enabled():
        return
    builtins.print = _filtered_print
    _PATCHED = True


def mute_info_logging() -> None:
    if not quiet_startup_enabled():
        return
    logging.disable(logging.INFO)


def patch_rlgames_quiet_train_init() -> None:
    """Skip rl-games startup prints while the sim/env/agent is constructed."""
    if not quiet_startup_enabled():
        return
    try:
        import torch
        from rl_games import torch_runner as tr
    except ImportError:
        return
    if getattr(tr.Runner.run_train, "_aerial_quiet_patched", False):
        return

    _orig = tr.Runner.run_train

    def run_train(self, args):
        if not quiet_startup_enabled():
            return _orig(self, args)

        with devnull_stdio():
            agent = self.algo_factory.create(self.algo_name, base_name="run", params=self.params)
            tr._restore(agent, args)
            tr._override_sigma(agent, args)

            compile_config = self.params.get("config", {}).get("torch_compile", True)
            if compile_config is not False:
                if isinstance(compile_config, bool):
                    actor_mode = "default"
                    critic_mode = "default"
                elif isinstance(compile_config, str):
                    actor_mode = compile_config
                    critic_mode = compile_config
                elif isinstance(compile_config, dict):
                    actor_mode = compile_config.get("mode", "default")
                    critic_mode = compile_config.get("critic_mode", actor_mode)
                else:
                    actor_mode = "default"
                    critic_mode = "default"
                agent.model = torch.compile(agent.model, mode=actor_mode)
                if hasattr(agent, "central_value_net") and agent.central_value_net is not None:
                    agent.central_value_net.model = torch.compile(
                        agent.central_value_net.model, mode=critic_mode
                    )

        agent.train()

    run_train._aerial_quiet_patched = True
    tr.Runner.run_train = run_train


@contextmanager
def devnull_stdio() -> Iterator[None]:
    """Redirect stdout/stderr (e.g. Isaac Gym import spam)."""
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull, devnull
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        devnull.close()
