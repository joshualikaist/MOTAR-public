"""Run the isolated R4 background-v1 technical contract, with complete source snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/renderer_background_v1_contract.md"


def source_bytes():
    paths = [Path(__file__).resolve(), CONTRACT, ROOT / "tools/runtime_fingerprint.py",
             ROOT / "aerial_gym/sensors/warp/warp_kernels/warp_camera_kernels.py"]
    paths += sorted((ROOT / "tools/renderer_validation").glob("*.py"))
    return {str(p.relative_to(ROOT)): p.read_bytes() for p in paths}


def isolation_guard():
    if any(n == "aerial_gym" or n.startswith("aerial_gym.") for n in sys.modules):
        raise RuntimeError("Renderer validation must not load the simulator")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("A new output directory is required")
    isolation_guard()
    before = source_bytes()
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    tracked_clean = all(subprocess.run(["git", "-C", str(ROOT), "diff"] + flags + ["--quiet"]).returncode == 0
                        for flags in ([], ["--cached"]))
    started = time.time()
    from renderer_validation.scene import Camera, sample_camera_poses
    from renderer_validation.background_scene import background_fixture
    from renderer_validation.background_validation import background_appearance, evaluate_background
    from renderer_validation.gbuffer import WarpGBufferRenderer
    from runtime_fingerprint import runtime_fingerprint
    fixture, camera = background_fixture(), Camera(width=240, height=135)
    appearance = background_appearance(4)
    positions, orientations = sample_camera_poses(619, 4)
    renderer = WarpGBufferRenderer(fixture.mesh, camera, 4, args.device)
    renderer.set_camera_poses(positions, orientations)
    result, arrays = evaluate_background(fixture, camera, appearance, renderer.render(), renderer.render())
    runtime = runtime_fingerprint()
    runtime["warp"] = renderer.wp.__version__
    if args.device.startswith("cuda"):
        runtime["nvidia_driver_version"] = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    isolation_guard()
    if before != source_bytes() or commit != subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip():
        raise RuntimeError("Source changed during execution")
    source_record = {}
    for name, value in before.items():
        old = subprocess.run(["git", "-C", str(ROOT), "show", commit+":"+name], capture_output=True)
        source_record[name] = {"sha256": hashlib.sha256(value).hexdigest(),
                               "tracked_at_head": old.returncode == 0,
                               "matches_head": old.returncode == 0 and old.stdout == value}
    result.update({"schema": "renderer_background_validation_v1", "seed": 619, "views": 4,
                   "camera": vars(camera), "fixture": fixture.as_dict(), "appearance": appearance.as_dict(),
                   "camera_poses": {"positions": positions.tolist(), "orientations": orientations.tolist()},
                   "runtime": runtime, "source": {"base_commit": commit, "tracked_tree_clean": tracked_clean,
                                                    "files": source_record, "source_unchanged_during_run": True},
                   "started_unix_s": started, "finished_unix_s": time.time(), "command": sys.argv})
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in before.items():
        destination = args.output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value)
    # Raw arrays are retained for an independent statistics audit; never model inputs.
    import numpy as np
    np.savez_compressed(args.output / "arrays.npz", **{name: a.detach().cpu().numpy() for name, a in arrays.items()})
    result["arrays_npz_sha256"] = hashlib.sha256((args.output / "arrays.npz").read_bytes()).hexdigest()
    with (args.output / "run.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(result["status"], args.output)
    if result["status"] != "TECHNICAL_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
