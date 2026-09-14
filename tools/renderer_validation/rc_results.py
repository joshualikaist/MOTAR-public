"""Result directories and provenance for the renderer characterization track.

Every RC experiment writes the same six files, so a reader never has to guess where a threshold or
a source version came from: PREREGISTRATION.md (a hash pin to the track document plus the thresholds
actually read out of the code at run time), README.md, config.json, summary.json, receipt.json and
source_manifest.json.

The source manifest is taken twice, before and after the run, and a run whose own source changed
underneath it refuses to write a verdict.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TRACK_DIR = ROOT / "results/renderer_characterization_2026-09-13"
PREREGISTRATION = TRACK_DIR / "PREREGISTRATION.md"
SOURCE_FILES = tuple(
    ["tools/renderer_validation/%s.py" % name for name in (
        "view_grid", "analytic_primitives", "target_arms", "geometry_metrics", "image_statistics",
        "area_match", "characterization_pipeline", "rc_results", "scene", "gbuffer", "shading",
        "reference_raycast", "urdf_asset")]
    + ["tools/%s.py" % name for name in (
        "run_renderer_characterization", "export_renderer_dataset",
        "benchmark_renderer_characterization", "runtime_fingerprint")]
    + ["aerial_gym/sensors/warp/warp_kernels/warp_camera_kernels.py",
       "resources/models/environment_assets/objects/navrl_target_drone_v3.urdf",
       "results/renderer_characterization_2026-09-13/PREREGISTRATION.md"])


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def source_manifest():
    """Git state plus a hash per source file this track's numbers depend on."""
    hashes = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        hashes[relative] = file_sha256(path) if path.exists() else "MISSING"
    return {"commit": git("rev-parse", "HEAD"),
            "tracked_tree_dirty": bool(git("status", "--porcelain")),
            "untracked_present": bool(git("status", "--porcelain", "--untracked-files=all")),
            "preregistration_sha256": file_sha256(PREREGISTRATION),
            "source_sha256": hashes}


def verify_unchanged(before):
    after = source_manifest()
    if before["commit"] != after["commit"] or before["source_sha256"] != after["source_sha256"]:
        raise RuntimeError("Source changed while the experiment was running; no verdict is written")
    return after


def write_json(path, record):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def preregistration_pin(experiment, thresholds, extra=""):
    """The per-experiment PREREGISTRATION.md: a pin to the track document, plus applied thresholds."""
    return "\n".join([
        "# %s — preregistration pin" % experiment,
        "",
        "This experiment is governed by the track preregistration:",
        "",
        "* path: `results/renderer_characterization_2026-09-13/PREREGISTRATION.md`",
        "* sha256: `%s`" % file_sha256(PREREGISTRATION),
        "",
        "No threshold was chosen or changed after a result was seen. The values below were read out",
        "of the code at run time, so this file records what was actually applied rather than what",
        "was intended:",
        "",
        "```json",
        json.dumps(thresholds, indent=2, sort_keys=True),
        "```",
        "",
        extra,
        "",
        "Out of scope, as in the track document: detector behaviour, tracking, policy performance,",
        "and any causal account of the closed D8b result. Causality with respect to D8b is",
        "`NOT_TESTED` here.",
    ]).rstrip() + "\n"


def write_experiment(folder, experiment, config, summary, readme, thresholds, runtime=None,
                     prereg_extra="", exist_ok=False):
    """Write the six standard files. Refuses to overwrite an existing result by default."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=exist_ok)
    before = config.get("source_manifest")
    if before is None:
        raise ValueError("config must carry the source manifest taken before the run")
    after = verify_unchanged(before)
    receipt = {"experiment": experiment, "verdict": summary.get("verdict", "NOT_EVALUATED"),
               "thresholds": thresholds, "source_manifest": after, "runtime": runtime,
               "files": ["PREREGISTRATION.md", "README.md", "config.json", "summary.json",
                         "receipt.json", "source_manifest.json"]}
    (folder / "PREREGISTRATION.md").write_text(
        preregistration_pin(experiment, thresholds, prereg_extra), encoding="utf-8")
    (folder / "README.md").write_text(readme, encoding="utf-8")
    write_json(folder / "config.json", config)
    write_json(folder / "summary.json", summary)
    write_json(folder / "receipt.json", receipt)
    write_json(folder / "source_manifest.json", after)
    return receipt
