#!/usr/bin/env python3
"""Clean rerun of the frozen seed-367 camera cells with OOB exit forensics.

This is descriptive instrumentation, not a new policy comparison and not authority to change
P2/D1/P3. It reuses the already frozen 20 m / 28 m single-variable contract and adds only exit-time
telemetry, stratified by whether the target had ever been acquired before leaving the arena.

Usage: tools/run_navrl_ref5in_oob_exit_forensics.py {preflight|run|finalize|verify}
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
# The host environment has an editable aerial_gym installation pointing at the primary (dirty)
# workspace. Put this clean evidence worktree first for this process and every evaluator child.
sys.path.insert(0, str(ROOT))


def load_base():
    path = ROOT / "tools/run_navrl_ref5in_camera_range_control.py"
    spec = importlib.util.spec_from_file_location("navrl_seed367_camera_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = load_base()
OUTPUT = ROOT / "results/navrl_ref5in_oob_exit_forensics_seed367"
SOURCE_BUNDLE = OUTPUT / "source_bundle"
PRIMARY_ROOT = Path(
    "/home/fair/workspaces/aerial_gym_ws/src/aerial_gym_simulator"
)
CHECKPOINT = PRIMARY_ROOT / (
    "aerial_gym/rl_training/rl_games/runs/"
    "ppo_260813_1636_navrl_v2-ref5in-d1-q3-adapt-s197/nn/"
    "last_gen_ppo_ep_1900_rew_182.11377.pth"
)
CHECKPOINT_ROBOT_CONFIG_SHA256 = (
    "ebb71802f19b630ba6c2ac4c04b113c269d8bbd3e40e094e126913caa8731297"
)
ROBOT_CONFIG = ROOT / "aerial_gym/config/robot_config/navrl_ref5in_quad_config.py"

# Override storage/report identity only. Arms, checkpoint, seed, episode count, gates and every
# simulator setting remain owned by the audited base orchestrator.
BASE.OUTPUT = OUTPUT
BASE.SOURCE_BUNDLE = SOURCE_BUNDLE
BASE.RUN.OUTPUT = OUTPUT
BASE.RUN.SOURCE_BUNDLE = SOURCE_BUNDLE
BASE.RUN.CHECKPOINT = CHECKPOINT
BASE.RUN.BASE.CHECKPOINT = CHECKPOINT
BASE.PRODUCER = "tools/run_navrl_ref5in_oob_exit_forensics.py"
BASE.SUMMARY_SCOPE = "frozen_seed367_oob_exit_forensics_20m_28m"
BASE.INCLUDE_OOB_FORENSICS = True

_base_canonical_env = BASE.canonical_env


def canonical_env_from_clean_worktree(*args, **kwargs):
    env = _base_canonical_env(*args, **kwargs)
    inherited = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + inherited if inherited else "")
    return env


BASE.canonical_env = canonical_env_from_clean_worktree


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"preflight", "run", "finalize", "verify"}:
        raise SystemExit("usage: ... {preflight|run|finalize|verify}")
    if BASE.P2.sha256_file(ROBOT_CONFIG) != CHECKPOINT_ROBOT_CONFIG_SHA256:
        raise SystemExit("refusing robot config bytes that differ from the frozen checkpoint")
    package_spec = importlib.util.find_spec("aerial_gym")
    if package_spec is None or package_spec.origin is None:
        raise SystemExit("refusing missing aerial_gym import spec")
    loaded_package = Path(package_spec.origin).resolve()
    if ROOT not in loaded_package.parents:
        raise SystemExit(f"refusing aerial_gym imported outside clean worktree: {loaded_package}")
    return BASE.main()


if __name__ == "__main__":
    raise SystemExit(main())
