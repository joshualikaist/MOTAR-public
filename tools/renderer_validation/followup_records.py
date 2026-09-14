"""Independent RC follow-up evidence, never imports a simulator or model."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "docs/preregistration_renderer_followup_2026-09-14.md"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def isolation():
    if any(name == "aerial_gym" or name.startswith("aerial_gym.") for name in sys.modules):
        raise RuntimeError("Simulator import is outside the independent renderer contract")


def manifest():
    isolation()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = sorted((ROOT / "tools/renderer_validation").glob("*.py"))
    files += [ROOT / p for p in (
        "tools/benchmark_renderer_transfer.py", "tools/measure_renderer_contract.py",
        "tools/runtime_fingerprint.py", "aerial_gym/sensors/warp/warp_kernels/warp_camera_kernels.py",
        "resources/models/environment_assets/objects/navrl_target_drone_v3.urdf") if (ROOT / p).exists()]
    files += [PREREG]
    sources = {}
    for path in files:
        relative = str(path.relative_to(ROOT))
        committed = subprocess.check_output(["git", "show", commit + ":" + relative], cwd=ROOT)
        if path.read_bytes() != committed:
            raise RuntimeError("Uncommitted measurement source: " + relative)
        sources[relative] = sha(path)
    frozen = subprocess.check_output(["git", "log", "-1", "--format=%H", "--",
                                      str(PREREG.relative_to(ROOT))], cwd=ROOT, text=True).strip()
    if frozen == commit:
        raise RuntimeError("Implementation must follow preregistration commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", frozen, commit], cwd=ROOT, check=True)
    return {"commit": commit, "preregistration_commit": frozen, "source_sha256": sources}


def finish(output, before, config, summary, prose):
    """Folder may contain progress evidence; final files use exclusive creation."""
    from runtime_fingerprint import runtime_fingerprint
    import warp
    output = Path(output)
    after = manifest()
    if before != after:
        raise RuntimeError("Source changed during measurement; refusing final result")
    summary["causality_vs_d8b"] = "NOT_TESTED"
    summary["scope"] = "Independent static renderer; no detector, policy, physics or causal D8b claim"
    config["seed"] = 20260914
    dump(output / "config.json", config)
    dump(output / "source_manifest.json", before)
    dump(output / "summary.json", summary)
    with (output / "README.md").open("x", encoding="utf-8") as stream:
        stream.write(prose.rstrip() + "\n\nCausality versus D8b: `NOT_TESTED`.\n")
    with (output / "PREREGISTRATION.md").open("x", encoding="utf-8") as stream:
        stream.write("Frozen protocol: `docs/preregistration_renderer_followup_2026-09-14.md`\n\n"
                     "SHA-256: `%s`\n" % sha(PREREG))
    paths = sorted(p for p in output.rglob("*") if p.is_file())
    dump(output / "receipt.json", {"runtime": runtime_fingerprint(), "warp": warp.__version__,
         "source_manifest": before, "verdict": summary["verdict"], "causality_vs_d8b": "NOT_TESTED",
         "files_sha256": {str(p.relative_to(output)): sha(p) for p in paths}})


def verify(output):
    output = Path(output)
    receipt = json.loads((output / "receipt.json").read_text())
    for path, expected in receipt["files_sha256"].items():
        if sha(output / path) != expected:
            raise RuntimeError("Evidence hash mismatch: " + path)
    sources = receipt["source_manifest"]
    for path, expected in sources["source_sha256"].items():
        data = subprocess.check_output(["git", "show", sources["commit"] + ":" + path], cwd=ROOT)
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError("Historical Git blob mismatch: " + path)
    return receipt["verdict"]
