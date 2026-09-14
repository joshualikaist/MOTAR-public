# ETH ds5 — external data contract

MOTAR does not redistribute ETH ds5 source or derived image assets in its current tree.
Obtain the dataset separately from the [original distributor](https://github.com/CenekAlbl/drone-tracking-datasets).
The intake used revision `2c857c97be71834d0791ae8ee4984ffb62b7680a`.
Its [dataset5 licence](https://github.com/CenekAlbl/drone-tracking-datasets/blob/2c857c97be71834d0791ae8ee4984ffb62b7680a/dataset5/LICENSE)
is CC BY-NC-SA 4.0, checked again on 2026-09-14. The root BSD-3-Clause licence does not replace it.
This is a repository distribution policy, not a claim that the CC licence forbids every redistribution.

Creators: Cenek Albl, Jingtong Li, Jesse Murray, Chen-Chieh Liao, Dorina Ismalii,
Mudathir Awadaljeed and Yue Pan; ETH Zurich Photogrammetry and Remote Sensing group.
Yue Pan is credited for ds5 collection. Retained [attribution and disclaimer](../../results/eth_ds5_intake_2026-09-10/THIRD_PARTY_NOTICE.md)
and [licence text](../../results/eth_ds5_intake_2026-09-10/LICENSE_DATASET.txt) describe historical adaptations.

## Local acquisition and preparation

Choose a user-controlled directory, not a tracked results directory. Obtain files at the pinned
revision from the distributor after reviewing its terms; this workflow never downloads or accepts
terms for you. Existing CLI names are `--dataset` (repository-shaped data root) and `--video`
(extracted cam0 MP4), not an invented `--eth-ds5-root` alias.

Expected data-root layout includes:

```text
dataset5/pose/fused_pose.txt
dataset5/camera-locations/campos.txt
dataset5/videos/cam0/cam0_frame_ts.txt
calibration/sony5100/sony5100.json
```

If you obtained the split video archives, the existing preparation command below verifies the
recorded archive hashes before extracting them. `--sevenzip` is the path to your installed 7z binary.
All output paths below must be new directories; never overwrite historical evidence.

```bash
python tools/extract_eth_ds5.py --dataset /path/to/eth-source \
  --receipt results/eth_ds5_intake_2026-09-10/video_receipt.json \
  --output artifacts/private/eth_ds5/extracted --sevenzip /path/to/7z
python tools/check_eth_ds5_local_data.py --dataset /path/to/eth-source \
  --video artifacts/private/eth_ds5/extracted/cam0.mp4
```

The local-data check is filesystem-only; it neither decodes images nor evaluates a model.
Missing data produces `ETH ds5 is not distributed with MOTAR`, names the missing paths and
directs the user to `--dataset` / `--video`. It does not certify timing, pose or analysis eligibility.

## Analysis reproduction, not a new experiment

The existing numerical analysis accepts external data and a retained numerical track:

```bash
python tools/measure_eth_ds5_size_range.py \
  --track results/eth_ds5_e3s_2026-09-10/track_adaptive.json \
  --dataset /path/to/eth-source \
  --calibration /path/to/eth-source/calibration/sony5100/sony5100.json \
  --output artifacts/private/eth_ds5/e3s-reproduction --shift-frames 0
```

This release cleanup does not run that command or alter its estimator, thresholds or verdict.
See the preserved [E3-S result](../../results/eth_ds5_e3s_2026-09-10/README.md) and
[preregistration](../../results/eth_ds5_e3s_2026-09-10/PREREGISTRATION.md) for scope and limitations.
Raw-video track regeneration additionally requires the historical environment and seeds documented
there. E3-P remains blocked; file availability is not a scientific gate.

For new local review panels, existing `tools/prepare_eth_ds5_frames.py` and
`tools/prepare_eth_ds5_review.py` require explicit input paths and `--output`; direct their outputs
under `artifacts/private/eth_ds5/`, which Git ignores. Historical pilot/frame receipts are not the
missing image bytes and cannot substitute for locally regenerated inputs.

**Exact old ten-image review-pack regeneration is unavailable:** its complete centre-coordinate
input was already missing before this cleanup. Six-frame `track_seeds.json` is not a replacement.
Do not claim byte-identical regeneration from the retained builder or invent missing centres.

## Historical availability

`current_release_availability = EXTERNAL_DATA_REQUIRED`.
Earlier commits contained 13 derived JPEGs and one review ZIP. Current releases exclude them;
old Git blobs remain reachable because this change does not rewrite history.
Historical input names/hashes and experiment receipts remain provenance, not download links.
The [inventory](../eth_ds5_public_release_inventory_2026-09-14.md) maps removed assets to references.
Numerical results, annotations, tables and figures are retained without blanket relicensing of
third-party-derived content. A separate clean-public-history decision is still required if old blobs
must not be distributed at all.
