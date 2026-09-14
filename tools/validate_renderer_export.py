#!/usr/bin/env python3
"""Actual CPU/Warp graphics metamorphic checks in independent exporter processes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Output must not exist")
    from renderer_validation.public_pipeline import verify_export, write_json
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    cases = {"base": [], "repeat": [], "light": ["--light-seed", "2"],
             "material": ["--material-seed", "2"], "renumber": ["--instance-offset", "100"]}
    records = {}
    for name, extra in cases.items():
        subprocess.run([sys.executable, "-B", str(root / "tools/export_renderer_dataset.py"),
                        "--output", str(args.output / name), "--device", "cpu", "--width", "80",
                        "--height", "60", *extra], check=True, cwd=str(root))
        records[name] = verify_export(args.output / name)
    a = {name: record["files"][0]["arrays"] for name, record in records.items()}
    geometry = {"depth_m", "range_m", "normal_world", "face_id", "valid"}
    checks = {"same_source_all_runs": all(r["source"] == records["base"]["source"] for r in records.values()),
              "same_seed_same_decoded_arrays": a["base"] == a["repeat"],
              "light_changes_rgb": a["base"]["rgb"] != a["light"]["rgb"],
              "material_changes_rgb": a["base"]["rgb"] != a["material"]["rgb"],
              "renumber_preserves_rgb": a["base"]["rgb"] == a["renumber"]["rgb"],
              "renumber_changes_labels": a["base"]["instance_id"] != a["renumber"]["instance_id"]}
    for name in ("light", "material", "renumber"):
        checks[name + "_preserves_geometry"] = all(a["base"][k] == a[name][k] for k in geometry)
    checks["appearance_preserves_instance_ids"] = all(a["base"]["instance_id"] == a[k]["instance_id"]
                                                     for k in ("light", "material"))
    passed = all(checks.values())
    result = {"schema": "generic_export_checks_v1", "status": "TECHNICAL_PASS" if passed else "FAIL",
              "checks": checks, "source": records["base"]["source"],
              "scope": "Five static generic box exports; not training data validation or statistical shortcut reduction"}
    write_json(args.output / "validation.json", result)
    print(json.dumps(result["checks"]))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
