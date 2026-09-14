# Research overview figures — 2026-09-13

New explanatory block diagrams; no algorithm changes or new performance results.
SVG is authoritative. Each same-stem PNG is 3840×2160; PDF is exported from vector SVG.

The paper-style status page displays six of these diagrams as Figures 1 and 3–7 and
uses the interactive WebGL arena as Figure 2. The observation/Transformer diagram is
kept here as a supplementary figure rather than removed from the dated package.

Package inventory (not article numbering):

- [Research question](research-overview-block-diagram.svg)
- [Existing simulation control stack](system-control-block-diagram.svg)
- [Separate real-image perception pipeline](perception-tracking-block-diagram.svg)
- [Observation and policy Transformer — supplementary](observation-transformer-block-diagram.svg)
- [Safety-filter diagnosis](safety-filter-block-diagram.svg)
- [Appearance/rendering boundaries](appearance-rendering-block-diagram.svg)
- [Evidence and reproducibility](evidence-reproducibility-block-diagram.svg)

[Download all 21 artifacts and hashes](research-overview-figures.zip).
Muted blue denotes learned modules, white fixed modules, and light gray
sensing/evidence or control.
Dashed borders/arrows denote unimplemented links. Historical task names are retained;
none of these figures establishes real-flight or integrated real-image-to-policy performance.

Rebuild: `python tools/render_research_overview.py --export` from the repository root,
with Chrome and websocket-client installed. Existing parent-directory figures and their
hash-pinned packages are untouched. The manifest covers SVG/PNG/PDF, not this README.
