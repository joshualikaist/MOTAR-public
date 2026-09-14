# ETH ds5 release inventory — 2026-09-14

Baseline: `f7496004a1027ecf21b666d6da2d040adb16dec3`. Inventory was captured before removal.
[Machine inventory](eth_ds5_public_release_inventory_2026-09-14.json) includes full hashes,
last-change commits, baseline line-numbered lexical reference candidates and every ZIP member.
All 14 assets below are **derived dataset assets (A)**, not numerical figures.

| Path | Type | Bytes | SHA-256 | Last change |
| --- | --- | ---: | --- | --- |
| `results/eth_ds5_intake_2026-09-10/eth_ds5_human_review_pack.zip` | ZIP | 1359178 | `8fe6499c7206caa0b66a218fd7de50802f601475e0259fbd452675b1295a3952` | `b3e157cb8b0df2ef6b43fb56659341590a125a81` |
| `results/eth_ds5_intake_2026-09-10/review/contact_sheet.jpg` | JPG | 220066 | `c640f6e823998d4ead187aac503d652f6d9057052a66be9b8ae4538a82a2824d` | `016e2229aa63b54c71152c782d3f163a91d8e4f3` |
| `results/eth_ds5_intake_2026-09-10/review_alignment/contact_sheet.jpg` | JPG | 106114 | `f2c654afc6c29db94dcb5d7b2151364679998a68221a8ca1cfcc31de884fd080` | `d3a77717ec5788da3661c9f4399ccc931913c694` |
| `results/eth_ds5_intake_2026-09-10/review_offset0/contact_sheet.jpg` | JPG | 220260 | `97a521ef35489f387ff17a672d2bebad3d9ec82a1ee5a85fdc637dc77b65b2e2` | `e2775fb85e1cb8b9a97c40cf9cbcb264efb18e15` |
| `results/eth_ds5_intake_2026-09-10/review_pack/01_frame000901_18m.jpg` | JPG | 139458 | `90aa68e73883066ceabd523bd539165904a01499a956ae380f55fff5bbf9a4ea` | `17e4a82f185787c080ef8fea52f3c4cc4870a580` |
| `results/eth_ds5_intake_2026-09-10/review_pack/02_frame001051_31m.jpg` | JPG | 126469 | `4fd1ae027ff378c127a546c17c6fb637cabefd11fe0ededd99ab773e59980a4d` | `17e4a82f185787c080ef8fea52f3c4cc4870a580` |
| `results/eth_ds5_intake_2026-09-10/review_pack/03_frame004937_62m.jpg` | JPG | 128213 | `1d24837bf3551f9ae888baffadeff73bf2b84d49d6dd8b2a5f7158439a918f07` | `17e4a82f185787c080ef8fea52f3c4cc4870a580` |
| `results/eth_ds5_intake_2026-09-10/review_pack/04_frame005251_61m.jpg` | JPG | 120100 | `f93ff3acaa26212a2155ca01b90dbcd9aadd86eab71448cdf694c50cae3adc65` | `17e4a82f185787c080ef8fea52f3c4cc4870a580` |
| `results/eth_ds5_intake_2026-09-10/review_pack/05_frame001605_82m.jpg` | JPG | 156806 | `e411d69a5002e8adfba94af3f458aa62b8d015b4bc323812fce2346603c9b7a7` | `17e4a82f185787c080ef8fea52f3c4cc4870a580` |
| `results/eth_ds5_intake_2026-09-10/review_pack/06_frame003909_97m.jpg` | JPG | 125834 | `2213677315b398b9f9a0678fe6ee14cf271212f65500a5ef485ebee56a893490` | `17e4a82f185787c080ef8fea52f3c4cc4870a580` |
| `results/eth_ds5_intake_2026-09-10/review_pack/07_frame001651_87m.jpg` | JPG | 162392 | `b0f445b165afa1c8225eafec9e98bd6c2d852374b907bbc79eddcd6f86652a73` | `17e4a82f185787c080ef8fea52f3c4cc4870a580` |
| `results/eth_ds5_intake_2026-09-10/review_pack/08_frame001778_88m.jpg` | JPG | 170630 | `444f37de8a4bb8f721c33aa6bfdccc4c52c9c6ce0b8ff61b027091a047b70f88` | `17e4a82f185787c080ef8fea52f3c4cc4870a580` |
| `results/eth_ds5_intake_2026-09-10/review_pack/09_frame000001_6m.jpg` | JPG | 132053 | `0084debc88632a4814c6f2dc6fee0c8e4ca7c3c6dcbdaac1970e7de98e982293` | `17e4a82f185787c080ef8fea52f3c4cc4870a580` |
| `results/eth_ds5_intake_2026-09-10/review_pack/10_frame003451_55m.jpg` | JPG | 154311 | `0ffc9dc73b970e6f30c5e9b2ff2f90aec4abc1d6356edd9590758a59f6dfbff6` | `17e4a82f185787c080ef8fea52f3c4cc4870a580` |

## Dependency map and retained evidence

- Ten review-frame JPEGs → `review_pack/answers.csv`, human review answers and archive notice
  receipt → intake interpretation and review corrections → E3-S/E3-P records.
- Three contact sheets → review material receipts and historical handoff → intake review.
  The handoff's two live image links now point to the external-data contract.
- Review ZIP → archive notice receipt (full archive and original member hashes) →
  historical packaging audit. Its direct-file test is migrated to explicit non-redistribution
  and preserved manifest/receipt hashes; generic ZIP notice-preservation tests still execute.
- E3-S numerical track → `run/e3s_result.json` → result tables and E3 evidence diagrams.
  These numeric/aggregate **scientific results (B)** are retained, not recomputed.
- Input filenames/hashes, review CSVs, receipts, notices and old worklogs are
  **historical provenance (C)**. They describe inputs at run time, not available downloads.
  Annotations are not automatically relicensed as BSD.

The inventory pins 46 non-image ETH result files. The only allowed editorial exception is
`HUMAN_REVIEW_HANDOFF.md` (availability note and two links, no measurement edits).
Every other pinned file, including all JSON receipts/summaries/tracks and CSVs, must remain byte-identical.
Historical source scripts and licence notices are also preserved.

Lexical references are deliberately over-inclusive: a shared basename such as
`contact_sheet.jpg` can match a generator output, not a dependency on the deleted file.
No production analysis code reads these removed repository images. Existing preparation tools take
explicit `--video`, `--dataset`, `--pilot-dir` and `--output` inputs.
The only direct tracked-ZIP consumer was the packaging regression test.

Public E3 diagrams are vector/text/aggregate graphics, not embedded ds5 frames; ZIP member inspection
checks all remaining tracked ZIPs for renamed byte-identical copies. Exact-byte checks do not prove
absence of arbitrary resized or re-encoded images. The release audit reports that boundary.

## Availability and history

See [external contract](external_data/ETH_DS5.md) and [release status](public_release_status_2026-09-14.md).
Removing the current paths does not remove reachable historical Git blobs or shrink Git history.
The missing original review-centre file prevents exact historical review-pack regeneration;
this was already recorded in the retained builder before removal.
