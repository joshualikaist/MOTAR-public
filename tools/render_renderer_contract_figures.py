#!/usr/bin/env python3
"""Paper figures for Renderer Contract v1, drawn only from committed follow-up results.

Five figures, each read from a `summary.json` that is already in the repository: nothing is
re-rendered, re-measured or recomputed here, so a figure cannot drift away from the result it
illustrates. The manifest written beside them pins the source result hashes, this generator's own
hash and every emitted asset, so a later reader can tell whether a figure still matches its data.

House rules: white background, three ink colours, no gradients, SVG source plus PNG and vector PDF,
new filenames only. No existing hash-pinned figure, archive or page is touched.
`causality_vs_d8b = NOT_TESTED` for every panel here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/assets/paper/renderer-contract-v1-2026-09-14"
RESULTS = {
    "r1b": ROOT / "results/renderer_characterization_r1b_2026-09-14/summary.json",
    "r2b": ROOT / "results/renderer_characterization_r2b_2026-09-14/summary.json",
    "r3b": ROOT / "results/renderer_characterization_r3b_2026-09-14/summary.json",
    "r5b": ROOT / "results/renderer_characterization_r5b_2026-09-14/summary.json",
}
INK, TEAL, AMBER = "#142D3B", "#007F73", "#B66318"
SPECIMEN_STYLE = {
    "analytic_sphere": {"color": TEAL, "marker": "o", "label": "analytic sphere"},
    "box_proxy": {"color": AMBER, "marker": "s", "label": "box"},
    "quadrotor_mesh": {"color": INK, "marker": "D", "label": "quadrotor mesh"},
}
# RC-R3 and RC-R3b both record views whose luminance variance is exactly zero. A log axis cannot
# show zero, so those views are drawn at this floor and the caption says so.
VARIANCE_FLOOR = 1e-12


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
        "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.4,
        "lines.markersize": 4.0, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    return plt


def load(stage):
    return json.loads(RESULTS[stage].read_text())


def save(figure, name, written):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png", "pdf"):
        path = OUTPUT / ("%s.%s" % (name, suffix))
        figure.savefig(path, dpi=200 if suffix == "png" else None)
        written.append(path)
    figure.clf()


def figure_rc1(plt, written):
    """Fig RC-1: how silhouette measurements converge towards the sampled reference."""
    summary = load("r1b")
    levels = sorted(summary["levels"], key=lambda row: row["resolution"][0])
    tested = [row for row in levels if row["resolution"] != summary["reference_resolution"]]
    widths = [row["resolution"][0] for row in tested]
    series = (("width_over_f", "width", INK, "-", 2.0),
              ("height_over_f", "height", INK, "--", 2.0),
              ("area_over_f2", "area", TEAL, "-", 2.0),
              ("centroid_over_f", "centroid", AMBER, "-", 1.0),
              ("depth_median_m", "median depth", AMBER, "--", 1.0))
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    for key, label, colour, dash, _ in series:
        axes[0].plot(widths, [100.0 * row["worst_relative"][key] for row in tested],
                     color=colour, linestyle=dash, marker="o", label=label)
    axes[0].axhline(2.0, color=INK, linestyle=":", linewidth=0.9)
    axes[0].axhline(1.0, color=AMBER, linestyle=":", linewidth=0.9)
    axes[0].annotate("registered 2 % (width, height, area)", (widths[0], 2.15), fontsize=6.5)
    axes[0].annotate("registered 1 % (centroid, depth)", (widths[0], 1.08), fontsize=6.5,
                     color=AMBER)
    recommended = summary["recommended_resolution"]
    axes[0].axvline(recommended[0], color=TEAL, linewidth=0.9, linestyle="-.")
    axes[0].annotate("qualified\n%d x %d" % tuple(recommended), (recommended[0] * 0.62, 0.11),
                     fontsize=7, color=TEAL)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xticks(widths)
    axes[0].set_xticklabels(["%dx%d" % tuple(row["resolution"]) for row in tested], fontsize=7.5)
    # A log axis adds its own minor decade labels, which land on top of these resolution labels.
    axes[0].xaxis.set_minor_locator(plt.NullLocator())
    axes[0].xaxis.set_minor_formatter(plt.NullFormatter())
    axes[0].set_xlabel("tested resolution   (reference sampled at %d x %d)"
                       % tuple(summary["reference_resolution"]))
    axes[0].set_ylabel("worst relative discrepancy over 72 views (%)")
    axes[0].set_title("(a) every view must pass, per quantity")
    axes[0].legend(fontsize=7, loc="lower left", ncol=2)
    for specimen, style_row in SPECIMEN_STYLE.items():
        for index, row in enumerate(tested):
            values = [100.0 * entry["errors"]["area_over_f2"]["relative"]
                      for entry in summary["rows"]
                      if entry["specimen"] == specimen
                      and entry["resolution"] == row["resolution"]]
            offset = (list(SPECIMEN_STYLE).index(specimen) - 1) * 0.12
            axes[1].scatter(np.full(len(values), index + offset), values, s=11,
                            facecolors="none", edgecolors=style_row["color"],
                            marker=style_row["marker"], linewidths=0.8,
                            label=style_row["label"] if index == 0 else None)
    axes[1].axhline(2.0, color=INK, linestyle=":", linewidth=0.9)
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(tested)))
    axes[1].set_xticklabels(["%dx%d" % tuple(row["resolution"]) for row in tested], fontsize=7.5)
    axes[1].set_xlabel("tested resolution")
    axes[1].set_ylabel("per-view relative area discrepancy (%)")
    axes[1].set_title("(b) all 24 views per specimen, not only the worst")
    axes[1].legend(fontsize=7, loc="lower left")
    axes[1].annotate("the sphere's 24 views coincide:\nits silhouette is the same from\nevery "
                     "direction", (1.35, 2.0e-3), fontsize=6.5, color=TEAL)
    figure.suptitle("Fig RC-1  silhouette measurement convergence      verdict: %s      "
                    "quantization estimate, not an accuracy proof" % summary["verdict"],
                    x=0.01, ha="left", fontsize=9.5)
    save(figure, "fig-rc-1-resolution-convergence", written)


def figure_rc2(plt, written):
    """Fig RC-2: an area control fitted on six views, frozen, and then held out."""
    summary = load("r2b")
    gates = {"median": 5.0, "p90": 10.0, "worst": 20.0}
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    phases = ("fit", "validation")
    controls = ("C0", "C1")
    colours = {"C0": AMBER, "C1": TEAL}
    labels = {"C0": "C0 historical isotropic scale", "C1": "C1 three-axis scale (fitted here)"}
    for control in controls:
        for index, phase in enumerate(phases):
            values = [100.0 * value for value in
                      summary["phases"][phase]["statistics"][control]
                      ["per_view_absolute_relative_error"]]
            offset = (controls.index(control) - 0.5) * 0.26
            axes[0].scatter(np.full(len(values), index + offset), values, s=14,
                            facecolors="none", edgecolors=colours[control], linewidths=0.9,
                            marker="os"[controls.index(control)],
                            label=labels[control] if index == 0 else None)
            axes[0].plot([index + offset - 0.09, index + offset + 0.09],
                         [np.median(values)] * 2, color=colours[control], linewidth=1.6)
    for name, value in gates.items():
        axes[0].axhline(value, color=INK, linestyle=":", linewidth=0.9)
        axes[0].annotate("registered %s %.0f %%" % (name, value), (-0.18, value + 0.6),
                         fontsize=6.5)
    axes[0].set_xticks(range(len(phases)))
    axes[0].set_xticklabels(["fit (6 views)\nparameters chosen here",
                             "held out (24 views)\nfrozen before rendering"], fontsize=7.5)
    axes[0].set_ylabel("absolute relative area error (%)")
    axes[0].set_title("(a) per-view error; bars mark the median")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].set_ylim(0, 40)
    statistics = ("median_absolute_relative_error", "p90_absolute_relative_error",
                  "worst_absolute_relative_error")
    names = ("median", "P90", "worst view")
    positions = np.arange(len(statistics))
    width = 0.36
    for index, control in enumerate(controls):
        values = [100.0 * summary["phases"]["validation"]["statistics"][control][key]
                  for key in statistics]
        axes[1].bar(positions + (index - 0.5) * width, values, width, color=colours[control],
                    edgecolor=INK, linewidth=0.7, label=labels[control])
        for position, value in zip(positions + (index - 0.5) * width, values):
            axes[1].text(position, value + 0.6, "%.1f" % value, ha="center", fontsize=7)
    for position, key in zip(positions, ("median", "p90", "worst")):
        axes[1].plot([position - 0.55, position + 0.55], [gates[key]] * 2, color=INK,
                     linestyle=":", linewidth=1.1)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("held-out absolute relative area error (%)")
    axes[1].set_title("(b) held-out statistics against the registered maxima (dotted)")
    axes[1].legend(fontsize=7, loc="upper left")
    axes[1].set_ylim(0, 40)
    figure.suptitle("Fig RC-2  multi-view area control      verdict: %s      "
                    "no refit after the held-out views were seen" % summary["verdict"],
                    x=0.01, ha="left", fontsize=9.5)
    save(figure, "fig-rc-2-area-control-validation", written)


def figure_rc3(plt, written):
    """Fig RC-3: shading variation against the entropy of the visible normal field."""
    summary = load("r3b")
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    for specimen, style_row in SPECIMEN_STYLE.items():
        rows = [row for row in summary["rows"] if row["specimen"] == specimen]
        entropy = [max(row["normal_histogram"]["entropy_bits"], 0.0) for row in rows]
        variance = [max(row["luminance_variance"], VARIANCE_FLOOR) for row in rows]
        occupied = [row["normal_histogram"]["occupied_bins"] for row in rows]
        axes[0].scatter(entropy, variance, s=16, facecolors="none",
                        edgecolors=style_row["color"], marker=style_row["marker"],
                        linewidths=0.9, label="%s  (r = %+.3f)"
                        % (style_row["label"],
                           summary["descriptive_correlations"][specimen]["pearson_r"]))
        axes[1].scatter(occupied, variance, s=16, facecolors="none",
                        edgecolors=style_row["color"], marker=style_row["marker"],
                        linewidths=0.9, label=style_row["label"])
    for axis in axes:
        axis.set_yscale("log")
        axis.set_ylabel("silhouette luminance variance")
    axes[0].set_xlabel("visible-normal entropy (bits, frozen 8 x 4 histogram)")
    axes[0].set_title("(a) entropy does not determine the variance")
    axes[0].legend(fontsize=6.8, loc="lower right")
    axes[0].annotate("views at exactly zero variance\nare drawn at %g" % VARIANCE_FLOOR,
                     (0.08, 2.5e-12), fontsize=6.5)
    axes[0].annotate("the sphere's 24 views span only 1.6e-9\nof variance: its correlation is\n"
                     "numerical, not a physical trend",
                     (2.05, 2.0e-4), fontsize=6.5, color=TEAL)
    axes[1].set_xlabel("occupied normal bins of 32")
    axes[1].set_title("(b) the same views by occupied bins")
    axes[1].legend(fontsize=7, loc="lower right")
    figure.suptitle("Fig RC-3  visible normal field and shading      verdict: %s      "
                    "N1 and N2 UNSUPPORTED: no authored vertex or smooth normals exist"
                    % summary["verdict"], x=0.01, ha="left", fontsize=9.5)
    save(figure, "fig-rc-3-normal-entropy-vs-shading", written)


def arm_table(summary, fixture):
    arms, rows = sorted({cell["arm"] for cell in summary["cells"]}), {}
    for arm in arms:
        cells = [cell for cell in summary["cells"]
                 if cell["arm"] == arm and tuple(cell["fixture"]) == fixture]
        rows[arm] = {
            "geometry": float(np.mean([cell["timing"]["geometry"]["mean_ms"] for cell in cells])),
            "shading": float(np.mean([cell["timing"]["shading"]["mean_ms"] for cell in cells])),
            "host": float(np.mean([cell["timing"]["host_strategy"]["mean_ms"] for cell in cells])),
            "headline": float(np.mean([cell["timing"]["headline_total"]["mean_ms"]
                                       for cell in cells])),
            "p95": float(np.mean([cell["timing"]["headline_total"]["p95_ms"] for cell in cells])),
            "repeats": len(cells),
        }
    return arms, rows


def figure_rc4(plt, written):
    """Fig RC-4: where the time goes once the host strategy is separated from compute."""
    summary = load("r5b")
    labels = {"A0": "A0 individual\nfull owned copy", "A1": "A1 resident\n+ scalars only",
              "A2": "A2 dtype-grouped\ncopy", "A3": "A3 pinned async\n+ completion wait"}
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for axis, fixture, title in ((axes[0], (640, 480, 32), "(a) 640x480, 32 scenes"),
                                 (axes[1], (320, 240, 8), "(b) 320x240, 8 scenes")):
        arms, rows = arm_table(summary, fixture)
        positions = np.arange(len(arms))
        geometry = [rows[arm]["geometry"] for arm in arms]
        shading = [rows[arm]["shading"] for arm in arms]
        host = [rows[arm]["host"] for arm in arms]
        axis.bar(positions, geometry, color=INK, edgecolor=INK, linewidth=0.6, label="geometry")
        axis.bar(positions, shading, bottom=geometry, color=TEAL, edgecolor=INK, linewidth=0.6,
                 label="shading")
        axis.bar(positions, host, bottom=np.add(geometry, shading), color=AMBER, edgecolor=INK,
                 linewidth=0.6, label="host strategy")
        for position, arm in zip(positions, arms):
            total = geometry[position] + shading[position] + host[position]
            axis.text(position, total + total * 0.04, "%.1f ms\nhost %.0f %%"
                      % (rows[arm]["headline"], 100.0 * rows[arm]["host"] / total),
                      ha="center", fontsize=6.8)
        axis.set_xticks(positions)
        axis.set_xticklabels([labels[arm] for arm in arms], fontsize=6.2)
        axis.set_ylabel("mean of three repeat means (ms)")
        axis.set_title(title)
        axis.set_ylim(0, max(np.add(np.add(geometry, shading), host)) * 1.35)
    axes[0].legend(fontsize=7, loc="upper right")
    figure.suptitle("Fig RC-4  renderer timing with the host strategy separated      verdict: %s"
                    "      all seven output hashes identical across arms" % summary["verdict"],
                    x=0.01, ha="left", fontsize=9.5)
    save(figure, "fig-rc-4-timing-breakdown", written)


def figure_rc5(plt, written):
    """Fig RC-5: the contract itself - what determines what, and where evidence stops."""
    r1b, r2b, r3b, r5b = (load(name) for name in ("r1b", "r2b", "r3b", "r5b"))
    figure, axis = plt.subplots(figsize=(9.6, 4.8))
    axis.set_axis_off()
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 60)

    def box(x, y, w, h, title, lines, colour=INK, dashed=False):
        axis.add_patch(plt.Rectangle((x, y), w, h, facecolor="white", edgecolor=colour,
                                     linewidth=1.2, linestyle=":" if dashed else "-"))
        axis.text(x + 1.6, y + h - 3.6, title, fontsize=8.5, fontweight="bold", color=colour)
        for index, line in enumerate(lines):
            axis.text(x + 1.6, y + h - 7.6 - index * 2.9, line, fontsize=7)

    def arrow(x1, y1, x2, y2, colour=INK):
        axis.annotate("", xy=(x2, y2), xytext=(x1, y1),
                      arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=1.1))

    axis.text(0, 57.5, "Fig RC-5  Independent Renderer Contract v1", fontsize=11,
              fontweight="bold")
    axis.text(0, 54.4, "Static geometry to exported arrays. Every box states what this track "
              "measured; the dotted box states what it did not.", fontsize=7.5)
    box(0, 33, 26, 18, "geometry + camera pose",
        ["3 specimens, 24 views", "world up -Y, +Z forward",
         "integer pixel sampling,", "principal point (W/2, H/2)"])
    box(31, 33, 30, 18, "G-buffer (6 arrays)",
        ["range_m, depth_m, normal_world,", "face_id, instance_id, valid",
         "measure at %d x %d (R1b)" % tuple(r1b["recommended_resolution"]),
         "worst area discrepancy 1.14 %"], TEAL)
    box(66, 33, 34, 18, "shading -> RGB",
        ["RGB = clamp(colour x (ambient +", "  kd x directional x max(0, n.l)), 0, 1)",
         "face_id addresses material;", "instance_id never enters shading"], AMBER)
    arrow(26, 42, 31, 42)
    arrow(61, 42, 66, 42)
    box(0, 13, 48, 17, "invariants held on the recorded grids",
        ["material or light change -> RGB only;", "  6 geometry hashes byte-identical (R4, R4M)",
         "transfer strategy change -> no output change:",
         "  7 hashes identical across A0-A3 (R5b)"], TEAL)
    box(52, 13, 48, 17, "measured boundaries, not blanket passes",
        ["area control: held-out %.1f %% median vs 5 %% gate" %
         (100 * r2b["phases"]["validation"]["statistics"]["C1"]
          ["median_absolute_relative_error"]),
         "  -> %s, no control adopted" % r2b["verdict"],
         "normal arms N1/N2 UNSUPPORTED by the asset",
         "host strategy is %.0f %% of the large staged total" %
         (100 * max(r5b["a0_large_host_fractions"]))], AMBER)
    arrow(24, 33, 24, 30)
    arrow(76, 33, 76, 30)
    box(0, 0, 100, 11, "downstream causal contribution",
        ["No experiment in this track connects a detector, association module or policy. "
         "The closed D8b frozen-policy loss is NOT explained here:",
         "area mismatch, shading and mesh geometry are NOT_TESTED as causes.        "
         "causality_vs_d8b = NOT_TESTED"], INK, dashed=True)
    arrow(50, 13, 50, 11)
    save(figure, "fig-rc-5-contract-diagram", written)


FIGURES = {"rc1": figure_rc1, "rc2": figure_rc2, "rc3": figure_rc3, "rc4": figure_rc4,
           "rc5": figure_rc5}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(written):
    record = {
        "manifest": "renderer_contract_v1_figures",
        "generated_from": {name: {"path": path.relative_to(ROOT).as_posix(),
                                  "sha256": sha256(path)}
                           for name, path in sorted(RESULTS.items())},
        "generator": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                      "sha256": sha256(Path(__file__).resolve())},
        "assets": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                   for path in sorted(written)},
        "rules": ("white background, three ink colours, no gradients, SVG source with PNG and "
                  "vector PDF, new filenames only"),
        "scope": "Independent static renderer figures",
        "causality_vs_d8b": "NOT_TESTED",
    }
    path = OUTPUT / "manifest.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=sorted(FIGURES), nargs="*", default=sorted(FIGURES))
    arguments = parser.parse_args(argv)
    plt = style()
    written = []
    for name in arguments.only:
        FIGURES[name](plt, written)
    manifest = write_manifest(written)
    for path in written + [manifest]:
        print("%9d  %s" % (path.stat().st_size, path.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
