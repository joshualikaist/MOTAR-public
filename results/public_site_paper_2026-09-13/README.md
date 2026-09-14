# Public research-paper page validation — 2026-09-13

## Scope

This package validates the presentation-only redesign of `docs/status/index.html`
on source commit `be896750dfd4e2fb665944952e18b9a1510b4edc`. The captured tree was dirty
because it contained the redesign under review. No simulator, detector, policy,
controller, research experiment or historical result was run or changed.

The page was changed from a research-dashboard layout into a single-column paper
layout with an abstract, numbered sections, seven numbered figures, captions,
compact parameter/data/evidence tables, bounded discussion, and end matter. The
existing interactive WebGL arena is retained as Figure 2. Six static diagrams are
shown in the article; one additional observation/Transformer diagram remains in the
download package as supplementary material.

## Browser validation

The final capture is in [`browser/`](browser/) and its machine-readable receipt is
[`browser/preview.json`](browser/preview.json). Chrome was exercised at:

| Viewport | Document width | Result |
|---|---:|---|
| Desktop | 1360 px | PASS |
| Tablet | 800 px | PASS |
| Mobile | 390 px | PASS |

At every viewport the six navigation links were visible, four manifest-backed status
rows were populated, all static images decoded, the WebGL canvas rendered, and the
arena route check reported 12 points / 31.9 m / no repeated-goal blocks. The mobile
document width remained 390 px. Blocking `status_manifest.json` produced the
fail-closed state instead of inferring successful experiment states.

Representative captures:

- [desktop title and abstract](browser/desktop-top.png)
- [desktop paper evidence table](browser/desktop-evidence.png)
- [tablet interactive arena](browser/tablet-arena.png)
- [mobile title and abstract](browser/mobile-top.png)
- [mobile stacked evidence table](browser/mobile-evidence.png)
- [mobile perception figure](browser/mobile-perception.png)

## Failures found during visual review

The first browser pass exposed inherited responsive CSS that hid all but one
navigation link on tablet. A later mobile pass revealed that the evidence table's
760 px minimum width enlarged the effective mobile layout to 774 px. Both were
corrected in `overview.css`: the paper navigation wraps at small widths, and evidence
rows become labeled vertical records below 640 px. One diagnostic run also hit a
Chrome cleanup race after completing its page checks; the preview helper now closes
the browser explicitly and tolerates an already-removed temporary profile. Failed
diagnostic captures were kept outside the repository and are not presented as final
evidence.

## Text and interface comparison

Counts compare the committed `be89675` page with the paper revision after stripping
markup and scripts:

| Measure | Dashboard page | Paper page | Interpretation |
|---|---:|---:|---|
| HTML bytes | 24,974 | 27,062 | Paper metadata, tables, captions and prose add source text |
| Visible characters | 6,291 | 14,380 | Full abstract/method/limitations replace terse cards |
| Visible words | 1,235 | 1,929 | Includes figure captions and table text |
| Paragraph elements | 44 | 30 | 31.8% fewer paragraphs |
| Dashboard card elements | 20 | 0 | Card presentation removed |
| Numbered figures | 7 | 7 | Six static diagrams plus one interactive figure |

The revision therefore does not claim a smaller document. It reduces dashboard UI
and paragraph fragmentation while adding the requested paper narrative and explicit
claim boundaries. The English abstract contains 140 words.

## Test record

- Full repository suite: **1,584 tests, 0 failures/errors, 4 existing skips**
  (`aerialgym`, CUDA device 0, 43.503 s).
- Paper-page contracts: 10 tests PASS.
- Status manifest and static-site Node contracts: PASS.
- Arena fixed-clock, route determinism and headless WebGL contracts: PASS.
- Declared public local links and Draft-7 manifest schema: PASS.
- `CITATION.cff`: valid CFF 1.2.0.
- Seven SVGs parsed by the Python XML tests; all seven PDFs contain no raster image
  rows according to `pdfimages -list`; `git diff --check`: PASS.

The full suite printed existing dependency/resource warnings and intentional
fault-injection tracebacks. Its process exit code was 0; those messages were not
reclassified as failures.

## Reproduction

From the repository root:

```bash
python tools/render_research_overview.py --export
python tools/preview_research_overview.py \
  --output results/public_site_paper_2026-09-13/browser
node tests/test_public_status_manifest.js
node tests/test_status_site.js
node tests/test_status_arena_motion.js
node tests/test_status_arena_route.js
node tests/test_status_webgl_headless.js
python -m unittest discover -s tests -p 'test_research_overview.py' -v
```

The whole repository suite is recorded in `WORKLOG.md`. PDF checks confirm that the
derived PDFs remain vector exports. This package validates local presentation and
links, not hosted deployment, publication, real-flight behavior or a new experiment.
