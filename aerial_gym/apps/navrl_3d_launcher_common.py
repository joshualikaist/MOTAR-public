"""Shared helpers for the NavRL 3-D launcher and runtime."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
APP = REPO / "aerial_gym" / "apps" / "navrl_3d.py"
RL_DIR = REPO / "aerial_gym" / "rl_training" / "rl_games"
RL_RUNS = RL_DIR / "runs"
CHECKPOINTS_SAVED = RL_DIR / "checkpoints_saved"
RESULTS_DIR = REPO / "results"

DEFAULT_TARGET_SPEED = 0.75
DEFAULT_DRONE_SPEED = 2.0
DEFAULT_NUM_TRIALS = 10
DEFAULT_DENSITY_MIN = 25
DEFAULT_DENSITY_MAX = 110

EXPECTED_OBS_DIM = {
    "transformer": 574,
    "legacy_vision_305": 305,
    "vision_1265": 1265,
}

CHECKPOINT_KIND_LABELS = {
    "transformer": "NavRL++ Target Transformer | 574D | Recommended",
    "legacy_vision_305": "Legacy semantic Vision CNN | 305D | Baseline playback",
    "vision_1265": "RGB-D + semantic LiDAR Vision CNN | 1265D",
}

CHECKPOINT_KIND_SHORT = {
    "transformer": "574D Transformer",
    "legacy_vision_305": "305D Legacy",
    "vision_1265": "1265D Vision",
}

VIEWER_CONTROL_CHIPS = (
    ("G", "LiDAR"),
    (", / .", "Target speed"),
    ("- / =", "Drone speed"),
    ("N", "New trial"),
    ("M", "Manual"),
    ("I K J L", "Move"),
    ("U O", "Yaw"),
    ("Space", "Pause"),
)

VIEWER_CONTROLS_HELP = (
    "Target speed  , / .       Drone speed  - / =       LiDAR  G       New trial  N\n"
    "Policy / Manual  M       Move  I K J L       Yaw  U O       Pause  Space"
)

_CHECKPOINT_PROBE = r"""
import json, sys, torch
try:
    ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    model = ck.get('model', {})
    keys = list(model.keys())
    dim = None
    for key, value in model.items():
        if key.endswith('running_mean_std.running_mean') and getattr(value, 'ndim', 0) == 1:
            dim = int(value.shape[0])
            if dim > 1:
                break
    if any('.cls_token' in key for key in keys):
        kind = 'transformer'
    elif any('.scan_cnn.' in key for key in keys):
        kind = 'legacy_vision_305'
    elif any('.lidar_cnn.' in key for key in keys):
        kind = 'vision_1265'
    else:
        kind = 'unsupported'
    print(json.dumps({'ok': True, 'kind': kind, 'obs_dim': dim, 'epoch': ck.get('epoch')}))
except Exception as exc:
    print(json.dumps({'ok': False, 'error': str(exc)}))
"""


def inspect_checkpoint(path: Path | str, python: str | None = None) -> dict[str, Any]:
    """Inspect checkpoint ABI in a subprocess so torch is not imported before Isaac Gym."""
    runtime = python or sys.executable
    result = subprocess.run(
        [runtime, "-c", _CHECKPOINT_PROBE, str(Path(path).expanduser())],
        text=True,
        capture_output=True,
        check=False,
        env=dict(os.environ, PYTHONNOUSERSITE="1"),
    )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"ok": False, "error": result.stderr.strip() or "checkpoint inspection failed"}


def validate_checkpoint_info(info: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (label, error). Label is set when the checkpoint is supported."""
    if not info.get("ok"):
        return None, str(info.get("error", "unknown"))
    kind = info.get("kind")
    label = CHECKPOINT_KIND_LABELS.get(kind)
    if label is None:
        return None, "Unsupported checkpoint architecture."
    expected = EXPECTED_OBS_DIM.get(kind)
    observed = info.get("obs_dim")
    if expected is not None and observed not in (None, expected):
        return None, "Checkpoint observation is %sD, but the detected model expects %sD." % (
            observed,
            expected,
        )
    epoch = info.get("epoch")
    if epoch is not None:
        label += " | epoch %s" % epoch
    return label, None


def find_recent_checkpoints(limit: int = 8) -> list[Path]:
    """Return recently modified .pth files from saved checkpoints and run folders."""
    roots = [CHECKPOINTS_SAVED, RL_RUNS]
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        candidates.extend(path for path in root.rglob("*.pth") if path.is_file())
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    seen: set[str] = set()
    recent: list[Path] = []
    for path in candidates:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        recent.append(path.resolve())
        if len(recent) >= max(1, int(limit)):
            break
    return recent


def default_results_path() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / "general_eval_results.json"


def apply_runtime_environment(args) -> None:
    """Configure NavRL 3-D runtime env vars from parsed launcher/CLI args."""
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["AERIAL_GYM_SIM_NAME"] = "navrl_viewer_sim"
    os.environ["NAVRL_INTERACTIVE"] = "1"
    os.environ["NAVRL_VISION"] = "1"
    os.environ["NAVRL_DENSITY_CURRICULUM"] = "0"
    os.environ.pop("NAVRL_NUM_BARS", None)
    os.environ["NAVRL_GENERAL_EVAL"] = "1"
    os.environ["NAVRL_GENERAL_DENSITY_MIN"] = str(int(args.density_min))
    os.environ["NAVRL_GENERAL_DENSITY_MAX"] = str(int(args.density_max))
    os.environ["NAVRL_GENERAL_NUM_TRIALS"] = str(int(args.num_trials))
    os.environ["NAVRL_TARGET_SPEED"] = str(max(0.0, float(args.target_speed)))
    os.environ["NAVRL_TARGET_PATTERN"] = "mixed"
    os.environ["NAVRL_MAX_VELOCITY"] = str(max(0.25, float(args.drone_speed)))
    os.environ["NAVRL_3D_HUD"] = "1"
    results_path = getattr(args, "results_json", None) or default_results_path()
    os.environ["NAVRL_GENERAL_RESULTS_JSON"] = str(Path(results_path).expanduser().resolve())

    kind = getattr(args, "policy_kind", "transformer")
    if kind == "legacy_vision_305":
        os.environ["NAVRL_PERCEPTION"] = "0"
        os.environ["NAVRL_LEGACY_VISION"] = "1"
        os.environ["NAVRL_NETWORK_OVERRIDE"] = "navrl_vision_legacy"
    elif kind == "vision_1265":
        os.environ["NAVRL_PERCEPTION"] = "0"
        os.environ["NAVRL_LEGACY_VISION"] = "0"
        os.environ.pop("NAVRL_NETWORK_OVERRIDE", None)
    else:
        os.environ["NAVRL_PERCEPTION"] = "1"
        os.environ["NAVRL_LEGACY_VISION"] = "0"
        os.environ.pop("NAVRL_NETWORK_OVERRIDE", None)


def configure_policy_checkpoint(args, python: str | None = None) -> dict[str, Any]:
    path = Path(args.checkpoint).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Checkpoint file not found.")
    info = inspect_checkpoint(path, python=python)
    _, error = validate_checkpoint_info(info)
    if error:
        raise ValueError(error)
    args.checkpoint = path
    args.policy_kind = info["kind"]
    args.checkpoint_info = info
    return info
