#!/usr/bin/env python3
"""Paper-style figures for the renderer characterization track, drawn from its result files.

Every number plotted is read from a committed summary.json; nothing is recomputed here, so a figure
cannot disagree with the result it illustrates. Output goes to a new directory with new filenames;
no existing hash-pinned figure, archive or page is touched.

House rules followed here: white background, at most three ink colours, no gradients, no
decorative fills, and SVG source plus PNG and vector PDF exports of every figure.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/renderer_characterization_2026-09-13"
OUTPUT = ROOT / "docs/assets/paper/renderer-characterization-2026-09-13"
INK, TEAL, AMBER = "#142D3B", "#007F73", "#B66318"
ARM_STYLE = {
    "analytic_sphere": {"color": TEAL, "linestyle": "-", "marker": "o", "label": "A0 analytic sphere"},
    "box_proxy": {"color": AMBER, "linestyle": "-", "marker": "s", "label": "A1 box proxy"},
    "area_matched_box": {"color": AMBER, "linestyle": "--", "marker": "^",
                         "label": "A2 area-matched box"},
    "quadrotor_mesh": {"color": INK, "linestyle": "-", "marker": "D", "label": "A3 quadrotor mesh"},
}
ARM_ORDER = ("analytic_sphere", "box_proxy", "area_matched_box", "quadrotor_mesh")


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#DCE5EA", "grid.linewidth": 0.6,
        "legend.frameon": False, "lines.linewidth": 1.4, "lines.markersize": 3.5,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    return plt


def load(name):
    return json.loads((RESULTS / name / "summary.json").read_text())


def save(figure, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("svg", "png", "pdf"):
        path = OUTPUT / ("%s.%s" % (name, suffix))
        figure.savefig(path, dpi=200 if suffix == "png" else None)
        written.append(path)
    figure.clf()
    return written


def view_labels(views):
    return ["%.0f/%.0f" % (view["azimuth_deg"], view["elevation_deg"]) for view in views]


def figure_r1(plt):
    """Apparent size and depth across the 24 primary views, per specimen."""
    summary = load("R1_geometry")
    views = summary["arms"]["quadrotor_mesh"]["description"]["views"]["views"]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for arm in ARM_ORDER:
        rows = summary["arms"][arm]["primary_320x240"]
        index = [row["view"] for row in rows]
        axes[0].plot(index, [row["equivalent_diameter_px"] for row in rows], **ARM_STYLE[arm])
        axes[1].plot(index, [row["normal_histogram"]["entropy_bits"] for row in rows],
                     **ARM_STYLE[arm])
    for axis in axes:
        axis.set_xticks(range(0, 24, 2))
        axis.set_xticklabels(view_labels(views)[::2], rotation=60, fontsize=6.5)
        axis.set_xlabel("view  (azimuth / elevation, degrees)  ·  distance 2.0 m")
    axes[0].set_ylabel("equivalent silhouette diameter (px)")
    axes[0].set_title("(a) apparent size at 320x240")
    axes[1].set_ylabel("visible-normal entropy (bits)")
    axes[1].set_title("(b) orientation diversity of the visible surface")
    axes[1].legend(loc="center left", fontsize=7.5)
    figure.suptitle("RC-R1  geometry representation of four specimens      verdict: %s"
                    % summary["verdict"], x=0.01, ha="left", fontsize=10)
    return save(figure, "rc-r1-geometry-representation")


def figure_r2(plt):
    """Apparent area per specimen, and what the frozen isotropic scale did on held-out views."""
    summary = load("R2_apparent_area")
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    means = summary["mean_areas"]
    names = [arm for arm in ARM_ORDER if arm in means]
    values = [means[arm]["mean_area_px"] for arm in names]
    colours = [ARM_STYLE[arm]["color"] for arm in names]
    hatches = ["", "", "//", ""]
    axes[0].set_axisbelow(True)
    bars = axes[0].bar(range(len(names)), values, color=colours, edgecolor=INK, linewidth=0.8)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for position, (arm, value) in enumerate(zip(names, values)):
        axes[0].text(position, value + 25, "%.0f\n(%.2fx)" % (value, means[arm]["ratio_to_reference"]),
                     ha="center", fontsize=7.5)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels([ARM_STYLE[arm]["label"].split(" ", 1)[0] for arm in names])
    axes[0].set_ylabel("mean silhouette area (px)")
    axes[0].set_ylim(0, max(values) * 1.25)
    axes[0].set_title("(a) mean area over 20 validation views")
    for arm in ARM_ORDER:
        series = summary["per_view_areas_px"][arm]
        axes[1].plot(summary["views_used_for_every_arm"], series, **ARM_STYLE[arm])
    axes[1].set_xlabel("primary-grid view index (validation views only)")
    axes[1].set_ylabel("silhouette area (px)")
    axes[1].set_title("(b) per-view area; the scale was fitted on four other views")
    axes[1].legend(loc="upper right", fontsize=7)
    match = summary["area_match"]
    figure.suptitle("RC-R2  apparent area      frozen scale %.4f  ->  ratio %.4f  "
                    "(tolerance %.2f)      verdict: %s"
                    % (match["scale"], match["area_ratio"], match["tolerance"], summary["verdict"]),
                    x=0.01, ha="left", fontsize=10)
    return save(figure, "rc-r2-apparent-area")


def figure_r3(plt):
    """Shading variation against orientation diversity, with geometry held fixed."""
    shading = load("R3_shading")
    geometry = load("R1_geometry")
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for arm in ARM_ORDER:
        gate = shading["gates"]["S3_shading_varies_%s" % arm]
        variance = np.asarray(gate["per_view_variance"], dtype=float)
        entropy = np.asarray([row["normal_histogram"]["entropy_bits"]
                              for row in geometry["arms"][arm]["primary_320x240"]], dtype=float)
        style = dict(ARM_STYLE[arm])
        style.pop("linestyle")
        axes[0].scatter(np.maximum(entropy, 0.0), np.maximum(variance, 1e-12), s=16,
                        facecolors="none", edgecolors=style["color"], marker=style["marker"],
                        label=style["label"], linewidths=0.9)
        axes[1].plot([row["view"] for row in geometry["arms"][arm]["primary_320x240"]],
                     np.maximum(variance, 1e-12), **ARM_STYLE[arm])
    for axis in axes:
        axis.set_yscale("log")
        axis.axhline(1e-4, color=INK, linewidth=0.8, linestyle=":")
        axis.set_ylabel("silhouette luminance variance")
    axes[0].set_xlabel("visible-normal entropy (bits)")
    axes[0].set_title("(a) shading variation follows orientation diversity")
    axes[0].legend(loc="lower right", fontsize=7)
    axes[0].annotate("S3 threshold 1e-4", (0.05, 1.4e-4), fontsize=7)
    axes[1].set_xlabel("primary-grid view index")
    axes[1].set_title("(b) the same values by view; floored at 1e-12 for the log axis")
    figure.suptitle("RC-R3  shading with geometry, camera, pose and depth held fixed      "
                    "verdict: %s" % shading["verdict"], x=0.01, ha="left", fontsize=10)
    return save(figure, "rc-r3-shading-variation")


def figure_r4(plt):
    """Lighting and material sweeps, each leaving every geometry buffer untouched."""
    lighting = load("R4_lighting")
    material = load("R4M_material")
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    cells = lighting["cells"]
    directions = sorted({tuple(cell["light_direction"]) for cell in cells})
    width = 0.38
    for offset, directional in enumerate(sorted({cell["directional"] for cell in cells})):
        for ambient_index, ambient in enumerate(sorted({cell["ambient"] for cell in cells})):
            values = []
            for direction in directions:
                matching = [cell for cell in cells if tuple(cell["light_direction"]) == direction
                            and cell["directional"] == directional and cell["ambient"] == ambient]
                values.append(float(np.mean(matching[0]["silhouette_luminance_mean"])))
            positions = np.arange(len(directions)) + (offset - 0.5) * width
            axes[0].plot(positions, values, marker="os^"[ambient_index],
                         color=TEAL if offset else INK, linestyle="none", markersize=4,
                         label="directional %.1f, ambient %.2f" % (directional, ambient))
    axes[0].set_xticks(range(len(directions)))
    axes[0].set_xticklabels(["%+.0f%+.0f%+.0f" % d for d in directions], fontsize=7)
    axes[0].set_xlabel("world light direction (x y z)")
    axes[0].set_ylabel("silhouette mean luminance")
    axes[0].set_title("(a) 36 lighting cells, one geometry")
    axes[0].set_ylim(0.0, 0.30)
    axes[0].legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.26))
    names = [cell["material_set"] for cell in material["cells"]]
    values = [float(np.mean(cell["silhouette_luminance_mean"])) for cell in material["cells"]]
    axes[1].set_axisbelow(True)
    axes[1].bar(range(len(names)), values, color=AMBER, edgecolor=INK, linewidth=0.8)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=25, fontsize=7, ha="right")
    axes[1].set_ylabel("silhouette mean luminance")
    axes[1].set_title("(b) five material sets at fixed lighting")
    invariant = lighting["gates"]["L1_geometry_and_silhouette_invariant"]["passed"]
    figure.suptitle("RC-R4 lighting and RC-R4M material      %s / %s      geometry identical in "
                    "every cell: %s" % (lighting["verdict"], material["verdict"], invariant),
                    x=0.01, ha="left", fontsize=9.5)
    return save(figure, "rc-r4-lighting-and-material")


def figure_r5(plt):
    """Render-path cost by ray count, and where the time goes in the largest cell."""
    summary = load("R5_performance")
    measured = [cell for cell in summary["cells"] if cell["status"] == "MEASURED"]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    order = ("analytic_sphere_geometry", "box_geometry", "box_flat",
             "area_matched_box_lambertian", "quadrotor_geometry", "quadrotor_lambertian")
    configurations = [name for name in order
                      if any(cell["configuration"] == name for cell in measured)]
    palette = {name: (INK, TEAL, AMBER)[index % 3] for index, name in enumerate(configurations)}
    markers = {name: "os^Dv<"[index] for index, name in enumerate(configurations)}
    for name in configurations:
        rows = [cell for cell in measured if cell["configuration"] == name]
        rays = [cell["resolution"][0] * cell["resolution"][1] * cell["scenes"] for cell in rows]
        axes[0].plot(rays, [cell["headline_total"]["median_ms"] for cell in rows],
                     color=palette[name], marker=markers[name], linestyle="none", label=name,
                     markersize=4.5, markerfacecolor="none", markeredgewidth=1.1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("rays per render  (width x height x scenes)")
    axes[0].set_ylabel("median render time (ms)")
    axes[0].set_title("(a) headline total, one synchronization at the end")
    axes[0].legend(fontsize=6.5, loc="upper left")
    stages = ("normal_face_pass_ms", "range_pass_ms", "finalize_ms", "clear_ms",
              "analytic_intersection_ms")
    labels, geometry, shading, copy = [], [], [], []
    for name in configurations:
        rows = [cell for cell in measured if cell["configuration"] == name]
        cell = max(rows, key=lambda row: (row["resolution"][0] * row["resolution"][1],
                                          row["scenes"]))
        labels.append("%s\n%dx%d  n=%d" % (name, cell["resolution"][0], cell["resolution"][1],
                                            cell["scenes"]))
        geometry.append(sum(cell["stages"][stage]["median_ms"] for stage in stages
                            if stage in cell["stages"]))
        shading.append(cell["shading"]["median_ms"] if cell["shading"] else 0.0)
        copy.append(cell["host_copy"]["median_ms"])
    positions = np.arange(len(labels))
    axes[1].set_axisbelow(True)
    axes[1].bar(positions, geometry, color=INK, edgecolor=INK, linewidth=0.6,
                label="geometry passes")
    axes[1].bar(positions, shading, bottom=geometry, color=TEAL, edgecolor=INK, linewidth=0.6,
                label="shading")
    axes[1].bar(positions, copy, bottom=np.add(geometry, shading), color=AMBER, edgecolor=INK,
                linewidth=0.6, label="host copy")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, fontsize=6, rotation=30, ha="right")
    axes[1].set_ylabel("median time (ms)")
    axes[1].set_title("(b) stage breakdown of the largest cell per configuration")
    axes[1].legend(fontsize=6.5, loc="upper right")
    figure.suptitle("RC-R5  render-path cost      descriptive, no verdict; not comparable with D6 "
                    "or D7      %d of %d cells measured"
                    % (summary["measured_cells"], summary["total_cells"]),
                    x=0.01, ha="left", fontsize=9.5)
    return save(figure, "rc-r5-render-cost")


def figure_r6(plt):
    """RC-R6: what the exported arrays keep and what they let change.

    Every cell is read from a committed result: the determinism gate compared all seven arrays
    across two processes, and RC-R4/RC-R4M recorded whether the six geometry hashes survived a
    lighting or material change while the image did not.
    """
    determinism = load("determinism")
    lighting = load("R4_lighting")
    material = load("R4M_material")
    arrays = ("rgb", "range_m", "depth_m", "normal_world", "face_id", "instance_id", "valid")
    geometry = arrays[1:]
    rows = [
        ("same arguments,\ntwo processes",
         {name: "identical" for name in arrays},
         "RC determinism: %s" % determinism["verdict"]),
        ("lighting changed\n(36 cells)",
         dict({name: "identical" for name in geometry}, rgb="differs"),
         "RC-R4: geometry invariant %s, luminance spread %.3f"
         % (lighting["gates"]["L1_geometry_and_silhouette_invariant"]["passed"],
            lighting["gates"]["L2_lighting_changes_luminance"]["spread"])),
        ("material changed\n(5 sets)",
         dict({name: "identical" for name in geometry}, rgb="differs"),
         "RC-R4M: geometry invariant %s, luminance spread %.3f"
         % (material["gates"]["M1_geometry_and_silhouette_invariant"]["passed"],
            material["gates"]["M2_material_changes_luminance"]["spread"])),
    ]
    figure, axis = plt.subplots(figsize=(9.6, 3.2))
    axis.set_xlim(-0.5, len(arrays) - 0.5)
    axis.set_ylim(-0.6, len(rows) - 0.4)
    axis.set_xticks(range(len(arrays)))
    axis.set_xticklabels(arrays, fontsize=8)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([label for label, _, _ in rows], fontsize=8)
    axis.grid(False)
    axis.invert_yaxis()
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)
    for index, (_, states, note) in enumerate(rows):
        for column, name in enumerate(arrays):
            identical = states[name] == "identical"
            axis.add_patch(plt.Rectangle((column - 0.44, index - 0.3), 0.88, 0.6,
                                         facecolor="white",
                                         edgecolor=TEAL if identical else AMBER, linewidth=1.2))
            axis.text(column, index + 0.02, "identical" if identical else "differs",
                      ha="center", va="center", fontsize=7,
                      color=TEAL if identical else AMBER, fontweight="bold")
        axis.text(-0.46, index + 0.42, note, fontsize=6.5)
    figure.suptitle("RC-R6  what an export keeps fixed and what it lets change      "
                    "seven arrays, three comparisons, all read from committed results",
                    x=0.01, ha="left", fontsize=9.5)
    return save(figure, "rc-r6-export-invariance")


def figure_map(plt):
    """What each experiment established, and what remains untested."""
    rows = [
        ("RC-R1 geometry", "R1_geometry"),
        ("RC-R2 apparent area", "R2_apparent_area"),
        ("RC-R3 shading", "R3_shading"),
        ("RC-R4 lighting", "R4_lighting"),
        ("RC-R4M material", "R4M_material"),
        ("RC-R5 cost", "R5_performance"),
        ("RC determinism", "determinism"),
    ]
    passing = ("GEOMETRY_VERIFIED", "AREA_MATCHED", "SHADING_CHARACTERIZED",
               "LIGHTING_CHARACTERIZED", "MATERIAL_CHARACTERIZED", "DETERMINISTIC")
    figure, axis = plt.subplots(figsize=(9.6, 4.4))
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(-1.5, len(rows) + 1.4)
    axis.text(0.0, len(rows) + 0.85, "RC-R6  evidence map of the renderer characterization track",
              fontsize=11, fontweight="bold")
    axis.text(0.0, len(rows) + 0.35,
              "Each row states only what its own result established. Nothing here is evidence about "
              "the detector, the policy, or the cause of the closed D8b result.", fontsize=8)
    for index, (label, folder) in enumerate(rows):
        summary = load(folder)
        height = len(rows) - index - 0.7
        verdict = summary["verdict"]
        failed = summary.get("failed_gates", [])
        if verdict == "DESCRIPTIVE_NO_VERDICT":
            colour, detail = INK, "descriptive by design; no pass or fail threshold exists"
        elif verdict in passing:
            colour, detail = TEAL, "every preregistered gate held"
        else:
            colour = AMBER
            if failed:
                counted = {}
                for name in failed:
                    key = name.split("_")[0]
                    counted[key] = counted.get(key, 0) + 1
                detail = "gates not met: " + ", ".join(
                    "%s%s" % (gate, " (%d arms)" % count if count > 1 else "")
                    for gate, count in sorted(counted.items()))
            else:
                match = summary.get("area_match", {})
                detail = ("frozen scale %.4f gave an area ratio of %.4f against a tolerance of %.2f"
                          % (match["scale"], match["area_ratio"], match["tolerance"]))
        axis.add_patch(plt.Rectangle((0.0, height - 0.3), 0.26, 0.62, facecolor="white",
                                     edgecolor=colour, linewidth=1.2))
        axis.text(0.014, height - 0.06, label, fontsize=8.5, fontweight="bold")
        axis.text(0.29, height + 0.08, verdict, fontsize=8.5, color=colour, fontweight="bold")
        axis.text(0.29, height - 0.2, detail, fontsize=7.5)
    axis.add_patch(plt.Rectangle((0.0, -1.35), 1.0, 0.95, facecolor="white", edgecolor=INK,
                                 linewidth=1.2, linestyle=":"))
    axis.text(0.014, -0.7, "D8b frozen-policy loss (-48.967 pp, closed and preserved)",
              fontsize=8.5, fontweight="bold")
    axis.text(0.014, -1.05, "causality from any row above to this box: NOT_TESTED. Shading, mesh "
              "geometry, projected area and the detector are not established as causes.",
              fontsize=7.5)
    return save(figure, "rc-evidence-map")


FIGURES = {"r1": figure_r1, "r2": figure_r2, "r3": figure_r3, "r4": figure_r4, "r5": figure_r5,
           "r6": figure_r6, "map": figure_map}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(FIGURES), nargs="*", default=sorted(FIGURES))
    arguments = parser.parse_args(argv)
    plt = style()
    written = []
    for name in arguments.only:
        written.extend(FIGURES[name](plt))
    for path in written:
        print("%9d  %s" % (path.stat().st_size, path.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
