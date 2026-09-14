"""Run the frozen NPS + Det-Fly v1 detector training contract.

The launcher refuses changed inputs and an existing output directory.  This keeps a failed run
from silently becoming a different experiment.  A CUDA-OOM recovery or any other contract change
requires a new run name and an amended, committed contract.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
REPOSITORY = Path(__file__).resolve().parents[1]
DATASET = WORKSPACE / "datasets" / "nps_detfly_joint_v1"
YOLOV5 = WORKSPACE / "datasets" / "yolov5"
PYTHON = WORKSPACE / "detector_runs" / "venv" / "bin" / "python"
WEIGHTS = (
    WORKSPACE / "detector_runs" / "runs" / "nps_det" / "s_tiles640_b8_e40"
    / "weights" / "best.pt"
)
HYP = REPOSITORY / "configs" / "perception_joint_detector_hyp_v1.yaml"
PROJECT = WORKSPACE / "detector_runs" / "runs" / "nps_detfly_joint"
RUN_NAME = "yolov5s_ms_b8_e30_s0"

EXPECTED = {
    "dataset_receipt_sha256": "ef9babdafdca3b198c265f05c80a158d1b51761b20f9b256b0a19d2143f8b0b2",
    "training_yaml_sha256": "08025f7948e0e05256aa6193303049db41a39def87d860a4696e46e4e436bfd8",
    "initial_weights_sha256": "ccd65dc37ec2fce765e0860232922287323c49179a4ea0e12cb6ad1b4f28f580",
    "yolov5_git_commit": "35b48237aef6d71ca9de2c5dea345d7536eb7fa7",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Verify and print without training.")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(directory, *args):
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True
    ).strip()


def frozen_command():
    return [
        str(PYTHON), str(YOLOV5 / "train.py"),
        "--weights", str(WEIGHTS),
        "--data", str(DATASET / "joint_detector.yaml"),
        "--hyp", str(HYP),
        "--epochs", "30",
        "--batch-size", "8",
        "--imgsz", "640",
        "--optimizer", "SGD",
        "--workers", "8",
        "--patience", "10",
        "--seed", "0",
        "--device", "0",
        "--multi-scale",
        "--project", str(PROJECT),
        "--name", RUN_NAME,
        "--exist-ok",
    ]


def verify_inputs():
    required = [
        PYTHON, YOLOV5 / "train.py", DATASET / "receipt.json",
        DATASET / "joint_detector.yaml", WEIGHTS, HYP,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("missing required inputs: " + ", ".join(missing))

    observed = {
        "dataset_receipt_sha256": sha256_file(DATASET / "receipt.json"),
        "training_yaml_sha256": sha256_file(DATASET / "joint_detector.yaml"),
        "initial_weights_sha256": sha256_file(WEIGHTS),
        "yolov5_git_commit": git_output(YOLOV5, "rev-parse", "HEAD"),
    }
    for key, expected in EXPECTED.items():
        if observed[key] != expected:
            raise RuntimeError(
                "%s changed: expected %s, observed %s" % (key, expected, observed[key])
            )
    yaml_text = (DATASET / "joint_detector.yaml").read_text()
    if any(line.lstrip().startswith("test:") for line in yaml_text.splitlines()):
        raise RuntimeError("sealed test unexpectedly appears in training YAML")

    subprocess.run(
        [str(PYTHON), str(REPOSITORY / "tools" / "verify_nps_detfly_joint_dataset.py"),
         "--dataset", str(DATASET)],
        check=True,
    )
    return observed


def main():
    args = parse_args()
    observed = verify_inputs()
    command = frozen_command()
    output = PROJECT / RUN_NAME
    receipt = {
        "schema_version": 1,
        "experiment": "motar.p3-followup.nps-detfly-joint-detector.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "contract": {
            "epochs_max": 30,
            "early_stopping_patience": 10,
            "batch_size": 8,
            "image_size_px": 640,
            "multi_scale": {"enabled": True, "range_px": [320, 960]},
            "optimizer": "SGD",
            "seed": 0,
            "workers": 8,
            "autoanchor": True,
            "freeze_layers": 0,
            "model_selection": "YOLOv5 validation fitness; test is absent",
            "sealed_test_access_during_training": False,
        },
        "inputs": observed,
        "hyperparameters_sha256": sha256_file(HYP),
        "launcher_sha256": sha256_file(Path(__file__)),
        "repository_git_commit": git_output(REPOSITORY, "rev-parse", "HEAD"),
        "repository_tracked_diff": bool(git_output(REPOSITORY, "status", "--short", "--untracked-files=no")),
        "output": str(output),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if args.dry_run:
        return
    if output.exists():
        raise RuntimeError("refusing existing output directory: %s" % output)
    output.mkdir(parents=True)
    (output / "preflight_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    os.chdir(YOLOV5)
    completed = subprocess.run(command, check=False)
    (output / "launcher_exit.json").write_text(json.dumps({
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": completed.returncode,
    }, indent=2, sort_keys=True) + "\n")
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
