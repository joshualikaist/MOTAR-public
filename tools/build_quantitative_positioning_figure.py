#!/usr/bin/env python3
"""Render the quantitative positioning figure from the registry and ledger.

The numbers in the figure are read from `docs/quantitative_positioning_registry.json`
so the SVG cannot drift from the recorded evidence. The right-hand panel marks which
axes each published work *studies*; it is explicitly not a performance score.

    python3 tools/build_quantitative_positioning_figure.py [--check]
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs/quantitative_positioning_registry.json"
OUT = ROOT / "docs/assets/paper/quantitative-positioning-2026-09-18.svg"

# Which registry entry supplies each stage of the measured chain.
CHAIN = [
    ("Measured perception error", "e3s_size_range",
     "held-out median absolute relative range error"),
    ("Simulator error injection", None, "P8 distribution driving the P9 injector"),
    ("Frozen-policy sensitivity", "perception_error_cost_frozen", "capture cost, CI excludes zero"),
    ("Readaptation", "p10_policy_readaptation", "net benefit not established"),
    ("Safety-filter geometry", "safety_filter_geometry", "crash difference, 15/15 cells"),
    ("Observation rendering", "d8b_mesh_observation", "causality NOT_TESTED"),
]

# Axes each work studies. Presence means "this work reports on this axis",
# never "this work is better on this axis".
AXES = ["pursuit /\ntracking", "dense\nclutter", "perception\nerror", "temporal\nassociation",
        "safety\nfilter", "observation\nrendering"]
COVERAGE = {
    "NavRL":            [0, 1, 0, 1, 1, 0],
    "NavRL++":          [0, 1, 1, 1, 1, 0],
    "YOPO":             [0, 1, 0, 0, 0, 0],
    "YOPOv2-Tracker":   [1, 1, 0, 1, 0, 0],
    "OPEN":             [1, 1, 0, 1, 0, 0],
    "Fast-Tracker":     [1, 1, 0, 1, 0, 0],
    "Elastic Tracker":  [1, 1, 0, 1, 0, 0],
    "Temporal Barrier": [1, 0, 0, 0, 1, 0],
    "MOTAR":            [1, 1, 1, 1, 1, 1],
}

INK, MUTE, RULE = "#1a1a1a", "#5a5a5a", "#c8c8c8"
ACCENT, PEND = "#1f4e79", "#8a8a8a"


def effect_label(reg, key):
    if key is None:
        return ""
    entry = next(e for e in reg["internal_motar"] if e["id"] == key)
    if entry.get("value_pp") is not None:
        return f"{entry['value_pp']:+g} pp"
    if entry.get("value_percent") is not None:
        return f"{entry['value_percent']:g} %"
    return ""


def render() -> str:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    W, H = 1600, 900
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         'font-family="Helvetica,Arial,sans-serif" role="img" '
         'aria-label="MOTAR measured evidence chain beside a coverage matrix of published systems">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']

    s.append(f'<text x="60" y="56" font-size="27" font-weight="700" fill="{INK}">'
             'Quantitative positioning across published systems</text>')
    s.append(f'<text x="60" y="86" font-size="17" fill="{MUTE}">'
             'Reported numbers · not head-to-head. Values on the left are MOTAR measurements; '
             'the right panel marks studied axes, not performance.</text>')

    # ---- left: the measured chain -----------------------------------------
    s.append(f'<text x="60" y="140" font-size="19" font-weight="700" fill="{INK}">'
             'MOTAR measured chain</text>')
    s.append(f'<text x="60" y="164" font-size="14" fill="{MUTE}">'
             'each stage bound to a result path</text>')

    x, y0, bw, bh, gap = 60, 186, 520, 82, 26
    for i, (title, key, note) in enumerate(CHAIN):
        y = y0 + i * (bh + gap)
        s.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="6" fill="#f5f7fa" '
                 f'stroke="{RULE}" stroke-width="1.2"/>')
        s.append(f'<text x="{x+18}" y="{y+31}" font-size="17" font-weight="600" fill="{INK}">'
                 f'{html.escape(title)}</text>')
        s.append(f'<text x="{x+18}" y="{y+56}" font-size="13.5" fill="{MUTE}">'
                 f'{html.escape(note)}</text>')
        label = effect_label(reg, key)
        if label:
            s.append(f'<text x="{x+bw-18}" y="{y+40}" font-size="21" font-weight="700" '
                     f'text-anchor="end" fill="{ACCENT}">{html.escape(label)}</text>')
        if i < len(CHAIN) - 1:
            ay = y + bh
            s.append(f'<path d="M{x+bw/2} {ay} L{x+bw/2} {ay+gap-7}" stroke="{RULE}" '
                     'stroke-width="1.8" fill="none"/>')
            s.append(f'<path d="M{x+bw/2-5} {ay+gap-12} L{x+bw/2} {ay+gap-5} '
                     f'L{x+bw/2+5} {ay+gap-12}" fill="{RULE}"/>')

    ylast = y0 + len(CHAIN) * (bh + gap)
    s.append(f'<text x="{x}" y="{ylast+18}" font-size="13.5" fill="{PEND}">'
             'Target-motion generalization (H / E0 / E1 / E2): RESULT PENDING</text>')

    # ---- right: coverage matrix -------------------------------------------
    mx, my = 720, 186
    label_w = 232
    # Columns must terminate inside the viewBox with a margin; 128 px overflowed
    # the right edge once the marker radius and row rule were added.
    right_margin = 60
    last_center = W - right_margin - 34
    col_w = (last_center - (mx + label_w)) / (len(AXES) - 1)
    row_h = 56
    rule_x2 = last_center + 30
    s.append(f'<text x="{mx}" y="140" font-size="19" font-weight="700" fill="{INK}">'
             'Studied axes by system</text>')
    s.append(f'<text x="{mx}" y="164" font-size="14" fill="{MUTE}">'
             'a mark means the axis is studied — it is not a performance score</text>')

    for j, axis in enumerate(AXES):
        cx = mx + label_w + j * col_w
        for k, line in enumerate(axis.split("\n")):
            s.append(f'<text x="{cx}" y="{my - 26 + k*15}" font-size="12.5" text-anchor="middle" '
                     f'fill="{MUTE}">{html.escape(line)}</text>')

    for i, (work, marks) in enumerate(COVERAGE.items()):
        ry = my + i * row_h
        is_motar = work == "MOTAR"
        if is_motar:
            s.append(f'<rect x="{mx-12}" y="{ry-4}" width="{rule_x2 - (mx-12)}" '
                     f'height="{row_h-8}" rx="5" fill="#eef3f9"/>')
        s.append(f'<text x="{mx}" y="{ry+26}" font-size="15.5" '
                 f'font-weight="{"700" if is_motar else "400"}" fill="{INK}">'
                 f'{html.escape(work)}</text>')
        for j, on in enumerate(marks):
            cx = mx + label_w + j * col_w
            if on:
                s.append(f'<circle cx="{cx}" cy="{ry+20}" r="8.5" '
                         f'fill="{ACCENT if is_motar else "#7d93ad"}"/>')
            else:
                s.append(f'<circle cx="{cx}" cy="{ry+20}" r="8.5" fill="none" '
                         f'stroke="{RULE}" stroke-width="1.3"/>')
        s.append(f'<line x1="{mx}" y1="{ry+row_h-8}" x2="{rule_x2}" '
                 f'y2="{ry+row_h-8}" stroke="{RULE}" stroke-width="0.7"/>')

    by = my + len(COVERAGE) * row_h + 26
    s.append(f'<text x="{mx}" y="{by}" font-size="14" font-weight="600" fill="{INK}">'
             'Class A matched external comparisons: 0</text>')
    for k, line in enumerate([
            "No system above has been run under MOTAR's arena, sensors, target and success",
            "definition, so no value from either panel may be subtracted from the other.",
            "Absence of a mark means the work does not report that axis, not that it would fail it."]):
        s.append(f'<text x="{mx}" y="{by + 22 + k*19}" font-size="13.5" fill="{MUTE}">'
                 f'{html.escape(line)}</text>')

    s.append('</svg>')
    svg = "\n".join(s) + "\n"
    _assert_within_bounds(svg, W, H)
    return svg


def _assert_within_bounds(svg: str, W: int, H: int) -> None:
    """Fail loudly if any drawn coordinate leaves the canvas.

    An SVG that overflows its viewBox still parses as valid XML and still renders
    a plausible-looking picture with its right-hand column quietly clipped, so
    XML validity is not evidence that the figure is correct.
    """
    import re
    bad = []
    for m in re.finditer(r'\b(x|x1|x2|cx)="([-\d.]+)"', svg):
        v = float(m.group(2))
        if v < 0 or v > W:
            bad.append(f"{m.group(1)}={v} outside 0..{W}")
    for m in re.finditer(r'\b(y|y1|y2|cy)="([-\d.]+)"', svg):
        v = float(m.group(2))
        if v < 0 or v > H:
            bad.append(f"{m.group(1)}={v} outside 0..{H}")
    for m in re.finditer(r'x="([-\d.]+)"\s+y="[-\d.]+"\s+width="([-\d.]+)"', svg):
        right = float(m.group(1)) + float(m.group(2))
        if right > W:
            bad.append(f"rect right edge {right} > {W}")
    if bad:
        raise SystemExit("figure geometry leaves the canvas:\n  " + "\n  ".join(sorted(set(bad))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = render()
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print(f"{OUT} is stale or missing", file=sys.stderr)
            return 2
        print("figure is current")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
