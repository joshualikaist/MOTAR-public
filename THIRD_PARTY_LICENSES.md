# Third-party materials

This is a source/redistribution inventory, **not legal advice or blanket release clearance**.
`VERIFIED` means the stated licence text/source was directly checked, not that every use or every
historical file has received a legal audit. Conditions can differ between code, data, weights and images.

## Code and attribution

| Component | Licence/source checked | Repository boundary | Status |
|---|---|---|---|
| Aerial Gym Simulator | Root [BSD-3-Clause notice](LICENSE), copyright Autonomous Robots Lab, NTNU | Fork contains upstream code/assets; original notice retained | VERIFIED notice; asset-by-asset provenance not certified |
| NavRL | [Upstream MIT licence](https://github.com/Zhefan-Xu/NavRL/blob/main/LICENSE), copyright 2025 Zhefan Xu; checked 2026-09-12 | Acknowledged related software; this pass did not establish a line-by-line copying inventory | VERIFIED licence text |
| rl_games | [Upstream MIT](https://github.com/Denys88/rl_games/blob/master/LICENSE) | Installed dependency; no new vendoring in this pass | NOT_REDISTRIBUTED by this change |
| Isaac Gym Preview 4 | NVIDIA distribution terms | External proprietary prerequisite; not installed by CPU CI | NOT_REDISTRIBUTED |
| NVIDIA Warp | Apache-2.0 package | Installed dependency, no new source vendoring | NOT_REDISTRIBUTED |
| urdfpy | [MIT source](https://github.com/mmatl/urdfpy/blob/5466842899b33bd549e8f9e2a9a987bd5e37373b/LICENSE) | CPU profile pins a source revision, not merely version 0.0.22 | NOT_REDISTRIBUTED |
| trimesh | MIT package | Installed dependency | NOT_REDISTRIBUTED |
| YOLOv5 | [Upstream AGPL-3.0](https://github.com/ultralytics/yolov5/blob/master/LICENSE) | Historical external detector dependency; no source added in this release pass | Separate licence boundary |

The former assertion “no vendored path named yolov5 means the AGPL boundary is not crossed” was
not a sufficient legal or provenance test and is withdrawn. Obligations depend on actual use,
modification and distribution, not the filename or whether measured results were produced.
This release pass does not certify the historical application's AGPL status.

The original BSD notice is unchanged. [NOTICE.md](NOTICE.md) attributes the fork and explains
why no new legal owner/name was invented. A separate named fork copyright line is a maintainer
ownership decision; it is not substituted for the upstream notice.

## Data and derived materials

| Material | Direct source and conditions | Redistributed here? | Status |
|---|---|---|---|
| NPS-Drones | [Official Purdue distribution page](https://engineering.purdue.edu/~bouman/UAV_Dataset/) explicitly links a **data** [BSD-3-Clause licence](https://engineering.purdue.edu/~bouman/UAV_Dataset/pubs/LICENSE.txt), copyright 2022 Charles A. Bouman | Raw dataset not under tracked `datasets/`; derived artefacts still need their own inventory | VERIFIED data licence, checked 2026-09-12 |
| Det-Fly repository | [MIT repository licence](https://github.com/Jake-WU/Det-Fly/blob/main/LICENSE) | External source | VERIFIED repository licence only |
| Det-Fly images/annotations | [Official README](https://github.com/Jake-WU/Det-Fly/blob/main/README.md) links separately hosted downloads and requests scholarly citation; inspected text does not explicitly extend MIT to those downloads | Raw data not added here | **BLOCKED_BY_LICENSE** for image redistribution; obtain explicit image/annotation terms |
| ETH ds5 source and derived images | [Pinned dataset licence](https://github.com/CenekAlbl/drone-tracking-datasets/blob/2c857c97be71834d0791ae8ee4984ffb62b7680a/dataset5/LICENSE): **CC BY-NC-SA 4.0** | **NOT_REDISTRIBUTED in current tree**; user obtains separately; review JPEGs/ZIP excluded | Licence checked 2026-09-14; numerical/aggregate evidence retained, not blanket relicensed |
| Project weights | Seven historical `artifacts/*.pth` | Yes | Project-produced; no new weights added; training-data/third-party obligations are not inferred away |
| Upstream example weights | Eleven inherited `.pth` files | Yes | Classified as inherited examples; individual weight provenance/licence coverage not fully audited |
| Project diagrams and generic synthetic arrays | Project-generated graphics | Yes | Root terms, except third-party embedded material if any; generic new exports contain no real imagery |
| Historical URDF/mesh assets | Inherited and project-generated files | Yes | Mixed provenance; no blanket asset-level VERIFIED claim |

“Nothing is tracked under datasets/” is a path check, **not** proof that no dataset-derived content
appears elsewhere. The history inventory explicitly includes images, archives and weights.

## ETH ds5: external acquisition, historical packaging evidence retained

The former inventory said 34 tracked JPEGs. The pre-removal tree had **13**, not 34.
All 13 and the review ZIP are excluded from the current public tree under the owner's
non-redistribution policy. See [external acquisition](docs/external_data/ETH_DS5.md),
[exact inventory](docs/eth_ds5_public_release_inventory_2026-09-14.md) and
[current release status](docs/public_release_status_2026-09-14.md).
This does not assert that CC BY-NC-SA prohibits all sharing; MOTAR elects not to bundle images.

Historically the ZIP originally had 14 members, then 16: its original members plus
`THIRD_PARTY_NOTICE.md` and `LICENSE_DATASET.txt`.
All original member bytes were unchanged by that packaging pass; the preserved [packaging receipt](results/eth_ds5_intake_2026-09-10/archive_notice_receipt.json)
records before/after archive hashes and every member hash.

The adjacent [notice](results/eth_ds5_intake_2026-09-10/THIRD_PARTY_NOTICE.md) identifies creators,
source/revision, alterations, licence, disclaimer and NonCommercial/ShareAlike scope.
The [full licence text](results/eth_ds5_intake_2026-09-10/LICENSE_DATASET.txt) differs from the
upstream byte stream only by a trailing blank line; it is not claimed to have the upstream SHA.
The root BSD licence **does not cover these image adaptations**.
The CC [licence conditions](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en)
allow sharing subject to their conditions; this is not unrestricted/commercial clearance.

Earlier commits still contain JPEGs and ZIPs, including ZIPs without bundled notices. No history rewrite was performed.
The packaging fix does not retroactively repair downloads of an earlier revision, certify third-party
rights outside the licence, or extend BSD to extracted annotations.

## Remaining release review

- Det-Fly hosted-image/annotation terms: BLOCKED_BY_LICENSE.
- Historical archives, inherited weights and miscellaneous third-party assets: BLOCKED_BY_EVIDENCE
  for blanket redistribution clearance.
- Personal absolute paths and incomplete binary secret scanning remain in the
  [history audit](docs/public_release_audit_2026-09-12.md).
- No bespoke author permission, commercial permission or ownership confirmation was obtained.

See [PUBLIC_RELEASE_CHECKLIST](docs/PUBLIC_RELEASE_CHECKLIST.md) for narrower checks that did pass.
