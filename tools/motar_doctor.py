#!/usr/bin/env python3
"""Read-only environment inventory. Never installs packages or imports the simulator."""
import argparse
import contextlib
import importlib
import importlib.metadata as metadata
import importlib.util
import io
import json
from pathlib import Path
import platform
import site
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
URDF_COMMIT = "5466842899b33bd549e8f9e2a9a987bd5e37373b"


def inventory():
    packages = {}
    for distribution, module in (("torch", "torch"), ("warp-lang", "warp"),
                                  ("urdfpy", "urdfpy"), ("trimesh", "trimesh"),
                                  ("numpy", "numpy")):
        row = {}
        try:
            dist = metadata.distribution(distribution)
            row.update(version=dist.version, path=str(Path(dist.locate_file("")).resolve()),
                       source=json.loads(dist.read_text("direct_url.json") or "{}"))
            # Capture library chatter: stdout remains a single parseable JSON document.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                imported = importlib.import_module(module)
            row["import"] = "AVAILABLE"
            if module == "torch":
                row.update(cuda_build=imported.version.cuda, cuda_available=imported.cuda.is_available())
        except Exception as exc:
            row.update(import_status="UNAVAILABLE", error_type=type(exc).__name__)
        packages[distribution] = row
    try:
        isaac = importlib.util.find_spec("isaacgym") is not None
    except (ImportError, ValueError):
        isaac = False
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        gpu_info = {"status": "AVAILABLE" if gpu.returncode == 0 else "UNAVAILABLE",
                    "inventory": gpu.stdout.strip() if gpu.returncode == 0 else None}
    except (OSError, subprocess.SubprocessError):
        gpu_info = {"status": "UNAVAILABLE", "inventory": None}
    return {"python": platform.python_version(), "python_minor": list(sys.version_info[:2]),
            "platform": platform.system(), "machine": platform.machine(),
            "prefix": sys.prefix, "isolated_venv": sys.prefix != sys.base_prefix,
            "user_site_enabled": site.ENABLE_USER_SITE, "packages": packages,
            "isaacgym_discoverable_not_imported": isaac, "gpu": gpu_info,
            "paths": {p: (ROOT / p).is_file() for p in
                      ("requirements-renderer-cpu.txt", "tools/run_renderer_validation.py",
                       "aerial_gym/sensors/warp/warp_kernels/warp_camera_kernels.py")}}


def assess(data, profile):
    checks = {}
    if profile == "simulator":
        # Inventory is deliberately not certification of the historical simulator environment.
        missing = not data["isaacgym_discoverable_not_imported"]
        return {"status": "UNAVAILABLE" if missing else "INVENTORY_ONLY",
                "checks": {"isaacgym_discoverable": not missing},
                "limit": "No simulator import, execution, repair or deployment certification"}
    p = data["packages"]
    checks["python38_linux_x86_64"] = (data["python_minor"] == [3, 8]
        and data["platform"] == "Linux" and data["machine"] == "x86_64")
    checks["isolated_venv"] = data["isolated_venv"] and data["user_site_enabled"] is False
    prefix = Path(data["prefix"]).resolve()
    for name in p:
        location = Path(p[name].get("path", "/")).resolve()
        checks[name + "_import_and_origin"] = (p[name].get("import") == "AVAILABLE"
            and (location == prefix or prefix in location.parents))
    checks["torch_cpu"] = (p["torch"].get("version") == "2.4.1+cpu"
                           and p["torch"].get("cuda_build") is None)
    source = p["urdfpy"].get("source", {})
    checks["urdf_source"] = (source.get("url") == "https://github.com/mmatl/urdfpy.git"
        and source.get("vcs_info", {}).get("commit_id") == URDF_COMMIT
        and not source.get("dir_info", {}).get("editable", False))
    for name, version in (("warp-lang", "1.0.0"), ("trimesh", "4.11.5"), ("numpy", "1.23.0")):
        checks[name + "_version"] = p[name].get("version") == version
    checks["required_paths"] = all(data["paths"].values())
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "limit": "Dependency checks only; run pip check, tests and an actual CPU smoke separately"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("renderer-cpu", "simulator"), required=True)
    args = parser.parse_args(argv)
    data = inventory()
    result = dict(schema="motar_doctor_v1", profile=args.profile, inventory=data, **assess(data, args.profile))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
