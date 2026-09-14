"""Independent renderer-only audit and fresh-process end-to-end benchmark.

This supplements, never replaces, the frozen R3/R4/R5 runners. No task or model imports.
Run one cell per process. Uncommitted audit code is disclosed and pinned by content SHA.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/renderer_reverification_protocol_2026-09-11.md"
REPEATS, WARMUP, SAMPLES = 3, 3, 10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    files = [Path(__file__).resolve(), PROTOCOL, ROOT / "tools/runtime_fingerprint.py",
             ROOT / "aerial_gym/sensors/warp/warp_kernels/warp_camera_kernels.py"]
    files += sorted((ROOT / "tools/renderer_validation").glob("*.py"))
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    sources = {}
    for path in files:
        relative = str(path.relative_to(ROOT))
        old = subprocess.run(["git", "-C", str(ROOT), "show", commit + ":" + relative], capture_output=True)
        sha = digest(path)
        sources[relative] = {"sha256": sha, "tracked_at_head": old.returncode == 0,
                             "matches_head": old.returncode == 0 and hashlib.sha256(old.stdout).hexdigest() == sha}
    return {"base_commit": commit, "source_files": sources,
            "note": "Content hashes, not base_commit alone, identify supplemental execution source."}


def isolation_guard():
    if any(n == "aerial_gym" or n.startswith("aerial_gym.") for n in sys.modules):
        raise RuntimeError("Independent renderer must not import the simulator")


def timing_summary(samples):
    import numpy as np
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Finite positive timing samples required")
    return {"n": len(values), "mean_ms": float(values.mean()),
            "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
            "min_ms": float(values.min()), "max_ms": float(values.max())}


def numpy_statistics(rgb, ranges, valid):
    """Independent from r4_metrics: NumPy least-squares, population std, stable rank bins."""
    import numpy as np
    x = ranges[valid].astype(np.float64)
    y = rgb[valid].astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    if len(x) < 40:
        raise ValueError("Insufficient pixels for 20 bins")
    design = np.column_stack([np.ones_like(x), x])
    fit = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    r2 = 0.0 if np.ptp(y) < 1e-12 else float(1 - np.sum((y-fit)**2) / np.sum((y-y.mean())**2))
    bins = np.array_split(np.argsort(x, kind="stable"), 20)
    return {"range_luminance_r_squared": r2,
            "within_range_bin_luminance_std_median": float(np.median([np.std(y[b]) for b in bins])),
            "luminance_mean": float(y.mean()), "luminance_variance": float(y.var())}


class Telemetry:
    """Sampled nvidia-smi observations, not exact peaks or exclusive renderer utilisation."""
    def __init__(self):
        self.samples, self.errors = [], []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _query(self, category, fields):
        value = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-" + category + "=" + fields,
                                         "--format=csv,noheader,nounits"], text=True, timeout=3)
        return list(csv.reader(io.StringIO(value), skipinitialspace=True))

    def _run(self):
        while not self.stop.is_set():
            try:
                gpu = self._query("gpu", "uuid,driver_version,utilization.gpu,memory.used,temperature.gpu")
                processes = self._query("compute-apps", "pid,used_gpu_memory")
                own = [float(row[1]) for row in processes if row[0] == str(os.getpid())]
                self.samples.append({"monotonic_s": time.perf_counter(), "gpu_raw": gpu,
                                     "compute_processes_raw": processes,
                                     "process_memory_mib": own[0] if own else None})
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.errors.append(str(exc))
            self.stop.wait(0.1)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=8)
        if self.thread.is_alive():
            raise RuntimeError("Telemetry did not stop")
        return {"samples": self.samples, "errors": self.errors,
                "scope": "Device metrics include other processes; process memory is sampled, not exact peak."}


def benchmark(args):
    import gc
    import numpy as np
    import torch
    from renderer_validation.scene import Camera, box_fixture, sample_appearance
    from renderer_validation.gbuffer import WarpGBufferRenderer
    from renderer_validation.shading import shade
    scene, camera = box_fixture(), Camera(width=args.width, height=args.height)
    renderer = WarpGBufferRenderer(scene, camera, args.batch, "cuda:0")
    positions = np.zeros((args.batch, 3), dtype=np.float32)
    orientations = np.tile(np.array([0, 0, 0, 1], np.float32), (args.batch, 1))
    renderer.set_camera_poses(positions, orientations)
    appearance = sample_appearance(0, args.batch, scene.material_count)
    monitor = Telemetry()
    monitor.start()
    records = []
    try:
        for repeat in range(REPEATS):
            order = ["flat", "lambertian"] if repeat % 2 == 0 else ["lambertian", "flat"]
            for mode in order:
                # No G-buffer survives an invocation or an arm. Both arms warm up identically.
                for _ in range(WARMUP):
                    shade(renderer.render(), scene, appearance, mode=mode)
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                samples = []
                started = time.perf_counter()
                for _ in range(SAMPLES):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    shade(renderer.render(), scene, appearance, mode=mode)
                    torch.cuda.synchronize()
                    samples.append(1000 * (time.perf_counter() - start))
                records.append({"repeat": repeat, "mode": mode, "start_monotonic_s": started,
                                "end_monotonic_s": time.perf_counter(), "samples_ms": samples,
                                "timing": timing_summary(samples),
                                "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                                "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2})
    finally:
        telemetry = monitor.finish()
    arms = {mode: timing_summary([s for r in records if r["mode"] == mode for s in r["samples_ms"]])
            for mode in ("flat", "lambertian")}
    return {"stage": "R5_DIRECT", "status": "MEASURED_NO_SPEED_GATE", "batch": args.batch,
            "camera": vars(camera), "scene": scene.as_dict(), "appearance": appearance.as_dict(),
            "camera_poses": {"positions": positions.tolist(), "orientations": orientations.tolist()},
            "protocol": {"repeats": REPEATS, "warmup_per_arm_repeat": WARMUP,
                         "samples_per_arm_repeat": SAMPLES, "percentile": "numpy linear interpolation",
                         "measurement": "synchronised shade(renderer.render()) wall clock; new process per cell"},
            "records": records, "arms": arms,
            "lambertian_over_flat": arms["lambertian"]["mean_ms"] / arms["flat"]["mean_ms"],
            "images_per_second_lambertian": 1000 * args.batch / arms["lambertian"]["mean_ms"],
            "environment_step_latency": None, "environment_step_reason": "N/A: no environment or policy",
            "telemetry": telemetry}


def audit(args):
    import numpy as np
    from renderer_validation.scene import (Camera, box_fixture, mixed_material_box_fixture,
                                           sample_appearance, sample_camera_poses)
    from renderer_validation.gbuffer import WarpGBufferRenderer
    from renderer_validation.appearance_models import render_arms
    from renderer_validation.validation import tensor_hash
    expected = json.loads(args.reference.read_text())
    scene = box_fixture() if args.fixture == "box" else mixed_material_box_fixture()
    camera = Camera(width=480, height=270)
    appearance = sample_appearance(409, 8, scene.material_count)
    renderer = WarpGBufferRenderer(scene, camera, 8, "cuda:0")
    renderer.set_camera_poses(*sample_camera_poses(409, 8))
    buffer = renderer.render()
    arms = render_arms(buffer, scene, appearance, camera)
    hashes = {name + "_rgb": tensor_hash(rgb) for name, rgb in arms.items()}
    hashes.update({name: tensor_hash(value) for name, value in
                   [("range", buffer.range_m), ("normal", buffer.normal_world),
                    ("face", buffer.face_id), ("valid", buffer.valid)]})
    ranges, valid = buffer.range_m.cpu().numpy(), buffer.valid.cpu().numpy()
    observed, differences, checks = {}, {}, {}
    for name, tensor in arms.items():
        rgb = tensor.cpu().numpy()
        observed[name] = [numpy_statistics(rgb[i], ranges[i], valid[i]) for i in range(8)]
        for i, record in enumerate(observed[name]):
            for metric, value in record.items():
                reference = expected["per_scene"][name][i][metric]
                key = "%s/%d/%s" % (name, i, metric)
                differences[key] = abs(value-reference)
                checks[key] = bool(np.isclose(value, reference, atol=1e-10, rtol=1e-10))
    ratios = [observed["lambertian"][i]["within_range_bin_luminance_std_median"] /
              observed["depth_gradient"][i]["within_range_bin_luminance_std_median"] for i in range(8)]
    checks["within_bin_ratios"] = bool(np.allclose(ratios, expected["within_bin_ratio_lambertian_over_depth"],
                                                  atol=1e-10, rtol=1e-10))
    checks["array_sha256_exact"] = hashes == expected["array_sha256"]
    faces = buffer.face_id.cpu().numpy()
    a, b = box_fixture().face_material, mixed_material_box_fixture().face_material
    retention = [float(np.mean(a[faces[i][valid[i]]] == b[faces[i][valid[i]]])) for i in range(8)]
    return {"stage": "R4_NUMPY_AUDIT", "fixture": args.fixture,
            "reference": str(args.reference), "reference_sha256": digest(args.reference),
            "array_sha256": hashes, "per_scene": observed, "absolute_differences": differences,
            "within_bin_ratios": ratios, "material_retention_per_scene": retention, "checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
            "scope": "Recomputation only; does not alter R4 FAIL or identify a causal mechanism"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "audit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--fixture", choices=("box", "mixed_material_box"), default="box")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output path")
    if args.mode == "audit" and args.reference is None:
        parser.error("audit requires --reference")
    before = provenance()
    isolation_guard()
    started = time.time()
    result = benchmark(args) if args.mode == "benchmark" else audit(args)
    from runtime_fingerprint import runtime_fingerprint
    import warp
    runtime = runtime_fingerprint()
    runtime["warp"] = warp.__version__
    runtime["nvidia_driver_version"] = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    # The shared fingerprint's driver_cuda is a torch build alias, not an actual driver query.
    runtime["driver_cuda_field_note"] = "Legacy driver_cuda equals torch build CUDA; use nvidia_driver_version for driver"
    isolation_guard()
    after = provenance()
    if before != after:
        raise RuntimeError("Execution source changed during measurement; refusing a valid receipt")
    result.update({"schema": "renderer_measurement_verification_v1", "runtime": runtime,
                   "provenance": before, "source_unchanged_during_run": True,
                   "started_unix_s": started, "finished_unix_s": time.time(), "command": sys.argv})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(result["status"], args.output, result.get("lambertian_over_flat", ""))
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
