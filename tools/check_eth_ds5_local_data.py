"""Check externally acquired ETH ds5 paths, without downloading or running analysis."""
import argparse
import json
from pathlib import Path

REQUIRED = ("dataset5/pose/fused_pose.txt", "dataset5/camera-locations/campos.txt",
            "dataset5/videos/cam0/cam0_frame_ts.txt", "calibration/sony5100/sony5100.json")


def check(dataset, video):
    paths = [Path(dataset) / name for name in REQUIRED] + [Path(video)]
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError("ETH ds5 is not distributed with MOTAR. Obtain it separately and provide "
                         "--dataset and --video; see docs/external_data/ETH_DS5.md. "
                         "Missing or empty: " + ", ".join(missing))
    return {"status": "LOCAL_FILES_PRESENT_ONLY", "dataset": str(Path(dataset).resolve()),
            "video": str(Path(video).resolve()), "scientific_validation": "NOT_RUN"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check(args.dataset, args.video)
    except ValueError as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
