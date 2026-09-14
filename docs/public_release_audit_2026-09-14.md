# Public release audit — 14 September 2026

An inventory and a status list. **Nothing is deleted, relicensed or removed by this audit**; items
whose terms are unresolved are named as blockers for the owner to decide.

Supersedes nothing: the [2026-09-12 audit](public_release_audit_2026-09-12.md) stands as the earlier
record, and the licence inventory in [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md) remains
authoritative for terms.

## Repository release files

| Present | State |
| --- | --- |
| [`LICENSE`](../LICENSE) | BSD-3-Clause, Autonomous Robots Lab, NTNU — upstream notice retained |
| [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md) | Inventory with per-item VERIFIED / NOT_REDISTRIBUTED labels |
| [`CITATION.cff`](../CITATION.cff) | Software citation, repository-verifiable identity only |
| [`NOTICE.md`](../NOTICE.md) | Present |
| [`README.md`](../README.md) | Project entrance |

## Tracked binary inventory

Computed from `git ls-files` at this commit.

| Category | Files | Size | Release note |
| --- | ---: | ---: | --- |
| Model weights (`.pth`) | 18 | 49.5 MiB | Project-trained checkpoints; no third-party weights |
| PNG images | 64 | 12.6 MiB | Project-generated figures |
| Archives (`.zip`) | 4 | 8.8 MiB | Includes one ETH-derived review pack, see blockers |
| `.npz` arrays | 10 | 3.2 MiB | Project-generated renderer/exporter outputs |
| JPEG images | 13 | 1.9 MiB | **All ETH ds5 derived**, see blockers |

Raw datasets are not tracked: `datasets/` (ETH ds5, Det-Fly, NPS-Drones, detenv) is outside Git.

## Status

### CLEAR

* **Aerial Gym Simulator** — BSD-3-Clause notice retained, upstream copyright intact.
* **NavRL** — upstream MIT licence text verified 2026-09-12.
* **Project-generated figures, `.npz` exports and trained checkpoints** — produced by this
  repository; no third-party content identified in them.
* **Installed dependencies** (rl_games, Warp, urdfpy, trimesh, Isaac Gym) — `NOT_REDISTRIBUTED`;
  nothing vendored by this repository.

### NEEDS_CONFIRMATION

| Item | What is unresolved |
| --- | --- |
| NavRL copying inventory | The licence is verified, but no line-by-line inventory of what was copied or adapted exists. Attribution is present; completeness is unproven. |
| Fork contributor copyright | Asset-by-asset provenance inside the fork is not certified; the upstream notice covers the fork as a whole. |
| Det-Fly **dataset** terms | Only the *repository* MIT licence is verified. The dataset's own terms are not. |
| NPS-Drones derived artefacts | The data licence (BSD-3-Clause, 2022 C. A. Bouman) is verified; derived artefacts inside this repository have no separate inventory. |

These stay `NEEDS_CONFIRMATION`. None is resolved by inference.

### BLOCKERS

| Item | Why it blocks | Size |
| --- | --- | ---: |
| 13 tracked ETH ds5 JPEGs (10 review frames + 3 contact sheets) | Derived from a **CC BY-NC-SA 4.0** dataset. Non-commercial and share-alike terms constrain redistribution from this repository; the [notice](../results/eth_ds5_intake_2026-09-10/THIRD_PARTY_NOTICE.md) records the licence but not a redistribution decision. | 1.9 MiB |
| `eth_ds5_human_review_pack.zip` | Same source and same constraint; it also embeds those images. | part of 8.8 MiB |

**Recommended handling, not taken here:** the owner decides between (a) keeping them under an
explicit CC BY-NC-SA 4.0 attribution and share-alike statement in the repository, (b) removing them
from the published tree while keeping the analysis that references them, or (c) obtaining permission.
Deleting them automatically would destroy evidence referenced by a committed result, and relicensing
them is not this repository's to do.

## What this audit did not do

No file was deleted, moved or relicensed. No licence was inferred from a sibling file, a dataset
homepage or a repository badge. Hosted CI remains unverified, and a green local suite is not
release clearance.
