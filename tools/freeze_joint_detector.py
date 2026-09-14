"""Freeze the validation-selected joint detector before any sealed-test access."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_RUN = (WORKSPACE / "detector_runs" / "runs" / "nps_detfly_joint" /
               "yolov5s_ms_b8_e30_s0")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def validation_fitness(row):
    """YOLOv5 fitness(): 0.1*mAP50 + 0.9*mAP50-95 for detection."""
    return (0.1 * float(row["     metrics/mAP_0.5"])
            + 0.9 * float(row["metrics/mAP_0.5:0.95"]))


def main():
    args = parse_args()
    run = args.run.resolve()
    output = args.output.resolve() if args.output else run / "model_receipt.json"
    if output.exists():
        raise SystemExit("[detector-freeze] refusing existing receipt: %s" % output)
    required = [run / "preflight_receipt.json", run / "launcher_exit.json",
                run / "results.csv", run / "weights" / "best.pt", run / "weights" / "last.pt"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("[detector-freeze] incomplete training run: %s" % missing)
    exit_receipt = json.loads((run / "launcher_exit.json").read_text())
    if exit_receipt.get("returncode") != 0:
        raise SystemExit("[detector-freeze] launcher did not exit successfully")
    preflight = json.loads((run / "preflight_receipt.json").read_text())
    if preflight.get("experiment") != "motar.p3-followup.nps-detfly-joint-detector.v1":
        raise SystemExit("[detector-freeze] unexpected training experiment")
    with (run / "results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("[detector-freeze] training has no completed validation epoch")
    fitness = [validation_fitness(row) for row in rows]
    best_index = max(range(len(rows)), key=lambda index: fitness[index])
    best_row = rows[best_index]
    maximum_epochs = int(preflight["contract"]["epochs_max"])
    receipt = {
        "schema_version": 1,
        "experiment": preflight["experiment"],
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "selection_data": "joint train/validation only; sealed Det-Fly test absent",
        "selection_rule": "YOLOv5 fitness = 0.1*mAP50 + 0.9*mAP50-95",
        "completed_epochs": len(rows),
        "maximum_epochs": maximum_epochs,
        "ended_before_maximum": len(rows) < maximum_epochs,
        "best_epoch_zero_based": int(best_row["               epoch"]),
        "best_validation": {
            "precision": float(best_row["   metrics/precision"]),
            "recall": float(best_row["      metrics/recall"]),
            "map50": float(best_row["     metrics/mAP_0.5"]),
            "map50_95": float(best_row["metrics/mAP_0.5:0.95"]),
            "fitness": fitness[best_index],
        },
        "best_weights": str(run / "weights" / "best.pt"),
        "best_weights_sha256": sha256_file(run / "weights" / "best.pt"),
        "last_weights_sha256": sha256_file(run / "weights" / "last.pt"),
        "results_csv_sha256": sha256_file(run / "results.csv"),
        "preflight_receipt_sha256": sha256_file(run / "preflight_receipt.json"),
        "launcher_exit_receipt_sha256": sha256_file(run / "launcher_exit.json"),
        "training_inputs": preflight["inputs"],
        "training_contract": preflight["contract"],
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("[detector-freeze] epoch %d, SHA-256 %s -> %s" % (
        receipt["best_epoch_zero_based"], receipt["best_weights_sha256"], output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
