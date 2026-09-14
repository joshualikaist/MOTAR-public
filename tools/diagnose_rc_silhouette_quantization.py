#!/usr/bin/env python3
"""Post-hoc diagnostic for the RC-R1 gates that were not met. No gate, no verdict, no threshold.

RC-R1 recorded G3, G4 and G7 as not met for the polyhedral specimens while G1, G2 and G5 held. This
tool asks one falsifiable question about that: is the disagreement dominated by binary-silhouette
quantization, meaning pixels whose centres fall near the outline, rather than by anything in the
intersection path?

The prediction, stated before the numbers are read, is that a quantization-dominated error **halves
when the focal length doubles**. If the measured ratio between consecutive resolutions sits near
2.0, the explanation holds; if it sits near 1.0, the error is resolution-independent and the
explanation is wrong. This is a diagnostic, so no verdict is attached either way, and nothing here
changes an RC-R1 threshold or an RC-R1 number.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

RESOLUTIONS = ((320, 240), (640, 480), (1280, 960))
PREDICTED_RATIO = 2.0
ARMS = ("analytic_sphere", "box_proxy", "area_matched_box", "quadrotor_mesh")


def boundary_pixels(mask):
    """Silhouette pixels with at least one 4-neighbour outside the silhouette."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    inner = (padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    return int((mask & ~inner).sum())


def measure(arm, scale, resolution, grid, device):
    from renderer_validation.characterization_pipeline import CharacterizationCell
    from renderer_validation.geometry_metrics import equivalent_diameter_px
    from renderer_validation.scene import Camera
    camera = Camera(width=resolution[0], height=resolution[1], far_range_m=20.0)
    cell = CharacterizationCell(arm, camera, grid, device=device,
                                scale=scale if arm == "area_matched_box" else None)
    valid = cell.gbuffer().valid.detach().cpu().numpy()
    rows = []
    for index in range(len(grid)):
        mask = valid[index]
        area = int(mask.sum())
        rows.append({"view": index, "area_px": area,
                     "perimeter_px": boundary_pixels(mask),
                     "equivalent_diameter_px": equivalent_diameter_px(area) if area else 0.0,
                     "angular_diameter": (equivalent_diameter_px(area) / camera.focal_px
                                          if area else 0.0)})
    del cell
    return {"resolution": list(resolution), "focal_px": camera.focal_px, "views": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    arguments = parser.parse_args(argv)
    from renderer_validation.rc_results import source_manifest, write_experiment
    from renderer_validation.view_grid import primary_grid
    from run_renderer_characterization import fitted_scale
    from runtime_fingerprint import runtime_fingerprint
    before = source_manifest()
    before["diagnostic_tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    scale, _ = fitted_scale(arguments.device)
    grid = primary_grid()
    arms = {}
    for arm in ARMS:
        levels = [measure(arm, scale, resolution, grid, arguments.device)
                  for resolution in RESOLUTIONS]
        reference = levels[-1]
        for level in levels:
            differences = []
            for row, target in zip(level["views"], reference["views"]):
                if target["angular_diameter"] <= 0.0 or row["angular_diameter"] <= 0.0:
                    continue
                differences.append(abs(row["angular_diameter"] / target["angular_diameter"] - 1.0))
            level["max_relative_difference_vs_finest"] = max(differences) if differences else None
            level["median_relative_difference_vs_finest"] = (float(np.median(differences))
                                                             if differences else None)
            level["mean_perimeter_over_area"] = float(np.mean(
                [row["perimeter_px"] / row["area_px"] for row in level["views"] if row["area_px"]]))
        ratios = []
        for coarse, fine in zip(levels, levels[1:]):
            if coarse["max_relative_difference_vs_finest"] and fine["max_relative_difference_vs_finest"]:
                ratios.append(coarse["max_relative_difference_vs_finest"]
                              / fine["max_relative_difference_vs_finest"])
        arms[arm] = {"levels": levels, "consecutive_error_ratios": ratios,
                     "predicted_ratio_if_quantization_dominates": PREDICTED_RATIO}
    summary = {
        "experiment": "RC-R1-diagnostic",
        "verdict": "DIAGNOSTIC_NO_VERDICT",
        "question": "Is the RC-R1 G3/G4/G7 disagreement dominated by binary-silhouette "
                    "quantization rather than by the intersection path?",
        "prediction_stated_before_reading": "a quantization-dominated relative error halves when "
                                            "the focal length doubles (ratio near %.1f)"
                                            % PREDICTED_RATIO,
        "reference_resolution": list(RESOLUTIONS[-1]),
        "arms": arms,
        "reading": {arm: {"consecutive_error_ratios": record["consecutive_error_ratios"]}
                    for arm, record in arms.items()},
        "relationship_to_rc_r1": "Explanatory only. No RC-R1 threshold, number or verdict is "
                                 "changed by this file; RC-R1 stands as recorded.",
        "scope": "Renderer measurement behaviour only.",
        "causality_vs_d8b": "NOT_TESTED",
    }
    config = {"command": "diagnose_silhouette_quantization", "device": arguments.device,
              "source_manifest": before, "resolutions": [list(r) for r in RESOLUTIONS],
              "views": grid.as_dict(), "arms": list(ARMS), "isotropic_scale": scale}
    readme = diagnostic_readme(summary)
    thresholds = {"none": "this is a diagnostic; it has no pass or fail threshold",
                  "prediction": summary["prediction_stated_before_reading"]}
    receipt = write_experiment(arguments.output, "RC-R1-diagnostic", config, summary, readme,
                               thresholds,
                               runtime_fingerprint(include_device=arguments.device != "cpu"))
    print("RC-R1-diagnostic written to %s" % arguments.output)
    for arm, record in arms.items():
        print("  %-20s ratios %s" % (arm, [round(value, 2)
                                           for value in record["consecutive_error_ratios"]]))
    return 0


def diagnostic_readme(summary):
    lines = ["# RC-R1 diagnostic — is the unmet-gate disagreement quantization?", "",
             "**This is a diagnostic, not an experiment.** It carries no gate and no verdict, and it",
             "changes nothing in RC-R1, which stands exactly as recorded.", "",
             "Question: %s" % summary["question"], "",
             "Prediction, written before the numbers were read: %s."
             % summary["prediction_stated_before_reading"], "",
             "| arm | error vs finest at 320x240 | at 640x480 | consecutive ratios | perimeter/area at 320x240 |",
             "|---|---|---|---|---|"]
    for arm, record in summary["arms"].items():
        levels = record["levels"]
        lines.append("| `%s` | %.4f | %.4f | %s | %.4f |" % (
            arm, levels[0]["max_relative_difference_vs_finest"],
            levels[1]["max_relative_difference_vs_finest"],
            ", ".join("%.2f" % value for value in record["consecutive_error_ratios"]),
            levels[0]["mean_perimeter_over_area"]))
    lines += ["", "A ratio near 2 means the error halves as the focal length doubles, which is what",
              "silhouette quantization does. A ratio near 1 would mean the error does not depend on",
              "resolution, and the quantization reading would be wrong.", "",
              "Nothing here is evidence about the detector, the policy, or the cause of the closed",
              "D8b result (`causality_vs_d8b: NOT_TESTED`)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
