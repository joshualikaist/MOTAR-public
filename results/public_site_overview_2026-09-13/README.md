# Public research overview — presentation verification

## Scope and source

Documentation/figure/site changes on top of `52b5dd878b28a15e0391efdee7cc5b2e01bdca65`.
Tests and screenshots were generated on the explicitly **dirty development tree**;
the [browser receipt](browser/preview.json) pins the actual HTML, CSS, JS, SVG and
preview tool hashes. This is not a D7 rerun, research result, or clean-tree benchmark.
D7's four existing commits were pushed by ordinary fast-forward from `77d885b` to
`52b5dd8`, after fetch, clean-tree and ancestry checks. No force push was used.

## Checks

| Check | Observed result |
|---|---|
| Full unittest discovery, aerialgym Python 3.8, user-site disabled | 1,582 run; 1,578 passed; 4 existing skips; 0 failures/errors; first run 42.352 s |
| New active overview tests | 8 passed: sections, links/HTML anchors, viewer DOM, bounded claims, defaults/lineages, assets/hashes/ZIP, SVG/accessibility, historical preservation |
| Historical detail site Node contract | PASS, content assertions relocated with archived content |
| Node arena motion / route / manifest | PASS |
| Headless WebGL startup and routed preview | PASS |
| Manifest Draft-7 schema + public local paths | PASS |
| CFF 1.2.0 validation | PASS |
| SVG export | 7 diagrams; text inside slide and inside blocks checked by Chrome |
| Original paper manifest and ZIP | SHA-256 unchanged |
| git diff --check | PASS |

The full suite emits expected fault-injection tracebacks plus existing dependency /
resource warnings. They are not counted as test failures. No tests were removed to
obtain a pass. No new skip was added. CPU CI now includes the new active overview tests;
hosted CI has not been run for the unpushed site commit.

## Browser / visual review

Three emulated viewport sizes: desktop 1360×1000, tablet 800×1000, mobile 390×844.
All have nine manifest rows, eight visible navigation links, decoded figure images,
a live WebGL canvas and no document-level horizontal overflow. The mobile figure
container intentionally scrolls sideways and links to its full SVG.
Blocking the manifest request produces “no PASS inferred”, not stale success output.

Screenshots: [desktop](browser/desktop-top.png), [arena](browser/desktop-arena.png),
[tablet arena](browser/tablet-arena.png), [mobile](browser/mobile-top.png),
[mobile perception](browser/mobile-perception.png), [mobile evidence](browser/mobile-evidence.png).
Twelve viewport screenshots are preserved under `browser/`. All seven diagram PNGs
were visually inspected, as were desktop/mobile/tablet samples.

First visual review found that inherited CSS hid mobile navigation except GitHub.
The overview-only CSS now shows all eight links; the final browser check explicitly
requires all eight at each viewport. The first diagnostic screenshots remain local
under `/tmp/motar-site-overview-20260913-preview1`, not research evidence.

## Reproduce locally

From the repository root, in the documented environments:

```bash
PYTHONNOUSERSITE=1 python -B -m unittest discover -s tests -q
node tests/test_status_site.js
node tests/test_public_status_manifest.js
node tests/test_status_webgl_headless.js
python tools/check_public_docs.py --schema
python tools/preview_research_overview.py --output /tmp/motar-overview-new-preview
python -m http.server 8000 --bind 127.0.0.1
```

Use aerialgym for full historical unittest discovery; the preview tool requires
Chrome and websocket-client, and schema checking requires the public-validation
dependencies. Preview: `http://127.0.0.1:8000/docs/status/`. The output directory must
not exist. The preview's “top” capture scrolls to main content, so the first few header
pixels may be above the capture; this is a screenshot position, not hidden navigation.

## Claim and historical preservation

[Source/migration map](../../docs/status/overview-sources-2026-09-13.md) records every
detail destination and schema 1→2 naming migration. D6 INCONCLUSIVE; D7 GO for
shadow cost/non-interference only; D8 NOT_STARTED; D9 NOT_RUN. E3-S is a defined
single-flight size-proxy range error, not general sensor accuracy. E3-P limitations,
P10 uncertainty, R4/R4b FAIL and withdrawn C3 statements remain explicit.

The archived previous HTML matches the original bytes at `52b5dd8` (SHA-256
`a96982824b82293172b73ac226ade7396b8773c62366c21e32a5f4694accf2bb`). Shared scripts
remain live; the archive is not an executable environment snapshot.

No D8 integration, geometry choice, perception adaptation, control change or new
training/evaluation campaign was implemented. Public deployment of the site changes
is not claimed by these local checks.
