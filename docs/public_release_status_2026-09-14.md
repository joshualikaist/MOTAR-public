# Public release status — 2026-09-14

Release-layer follow-up to [the earlier audit](public_release_audit_2026-09-14.md), not a new
scientific result or legal clearance. Final machine audit and test receipts are linked below.
No policy, detector, controller, threshold or scientific result is changed.

| Axis | Status / boundary |
| --- | --- |
| CURRENT_TREE_DATASET_REDISTRIBUTION | RESOLVED: exact 13 ETH JPEGs and one ZIP removed; [inventory](eth_ds5_public_release_inventory_2026-09-14.md) |
| HISTORICAL_GIT_DATASET_REACHABILITY | OPEN: removal does not rewrite history |
| LICENSE_STATUS | ETH ds5 CC BY-NC-SA 4.0; not covered by root BSD. No blanket clearance |
| REPRODUCIBILITY_STATUS | EXTERNAL_DATA_REQUIRED; numerical evidence retained; exact old review pack remains unreproducible due to missing complete centres |
| OPEN_CONFIRMATIONS | NavRL copy/adaptation provenance, fork asset ownership, Det-Fly hosted-data terms, NPS derived-content provenance |

## Retained confirmations, not inferred away

- **NavRL:** source-name/comment/history searches show numerous NavRL-named local adaptations,
  but a name is not proof of copying or its absence. No reliable line-by-line upstream comparison
  or complete copied-file inventory has been established. **NEEDS_CONFIRMATION**, not NOT_REDISTRIBUTED.
- **Fork copyright:** root LICENSE still names Autonomous Robots Lab, NTNU. NOTICE credits the
  fork without inventing a legal owner. Git authorship is not ownership certification.
  Asset-level and contributor ownership remains **NEEDS_CONFIRMATION**.
- **Det-Fly:** the repository MIT licence does not establish terms for separately hosted images
  or annotations. No new permission obtained: **NEEDS_CONFIRMATION / BLOCKED_BY_LICENSE** remains.
- **NPS-Drones:** no tracked JPEGs remain after ETH removal; tracked PNGs are diagram/plot/site
  screenshot paths, not a tracked raw NPS directory. Arrays, models and adapted/embedded content
  are not cleared by a filename search. Exact-content ZIP inspection is documented separately;
  full dataset-derived provenance remains **NEEDS_CONFIRMATION**.

Bounded inventory: `git ls-files '*nps*' '*NPS*'` finds four numerical
`results/air2air_zeroshot/glad_on_nps*.json` files and five preparation/verification scripts;
no NPS-named image/archive. This does not exclude NPS-derived content under other names.
`rg 'Zhefan|zhefan|github.com/Zhefan-Xu|copied from|adapted from' aerial_gym tools NOTICE.md`
finds no explicit upstream NavRL copying header; the one unrelated “copied from” comment is
not attribution. Absence of a header is not proof of no copying. No code ownership is inferred.

Three remaining ZIPs contain 29, 22 and 20 members respectively (diagram exports and associated
manifests). No removed ETH JPEG byte stream is duplicated among them. The two public E3 PNG
figures were visually inspected: they contain diagrams/text/aggregate values, not dataset frames.
SVG source search found no `<image` / `data:image` embeds in `docs/assets`.

Corrections to the earlier audit: its claim that all 18 weights were project-trained was too
broad. The existing third-party inventory distinguishes seven project weights from eleven
inherited examples. Project generation alone does not clear training-data or third-party rights.
Archive totals below include `.zip/.tar/.gz/.7z`, not ZIP alone; no Git-history size reduction is claimed.

## Release choices when historical blobs remain

A. Preserve this history and state that only the current tree excludes data.
B. With separate approval, export permitted files to a **clean public-release repository**.
   This is the recommended choice if distribution of historical image blobs must stop while
   preserving the research repository.
C. Rewrite history only with explicit approval and coordinated migration.

Neither B nor C is executed here. A normal push preserves historical reachability.

## Verification

[Post-removal audit at commit 79209d8](eth_ds5_current_tree_audit_2026-09-14.json) and
[code-complete audit at d6bccb4](release_validation_2026-09-14/current_tree_audit.json) found
zero unexpected evidence changes and all 14 historical blobs reachable. Historical receipts and
summaries are not rewritten to reflect availability; [external_data_manifest.json](external_data_manifest.json)
is the release-layer source of truth. Missing unrelated required evidence must remain an error.

Full canonical `unittest discover`: **1,882 run, 1,878 passed, 4 existing skips, 0 failures/errors**,
66.638 s. This is not a pytest result: pytest is absent from the canonical historical environment;
no package was installed into it. Public docs/Draft-7 schema, CFF citation validation and all five
JavaScript site checks (including headless WebGL) passed. New external-data regressions: 13 passed.
No tests were newly skipped. The direct-ZIP test changed to the explicit non-redistribution contract;
synthetic ZIP member-preservation checks remain enabled.

[Validation receipt and commands](release_validation_2026-09-14/README.md),
[full unit log](release_validation_2026-09-14/full_unittest.txt),
[removed-link audit](release_validation_2026-09-14/removed_link_audit.json).
The link scan covered 394 tracked Markdown/HTML/SVG/notebook documents and found zero live links
to removed assets; it does not claim all unrelated historical links are valid.

| Tracked category | Before f749600 | After d6bccb4 |
| --- | ---: | ---: |
| JPEG | 13 / 1,962,706 bytes | 0 / 0 bytes |
| ZIP | 4 | 3 |
| All archives (ZIP plus existing gzip) | 5 / 9,351,020 bytes | 4 / 7,991,842 bytes |
| Checkpoints | 18 / 51,891,688 bytes | unchanged |
| PNG | 64 / 13,186,634 bytes | unchanged |
| NPZ | 10 / 3,321,708 bytes | unchanged |
| Total tracked files / bytes | 5,146 / 514,769,555 | 5,141 / 511,546,261 |

Totals are exact for the named measured commits, before adding these final validation logs/docs.
The asset-only reduction is 3,321,884 bytes. This is current-tree content, **not Git history size**.
