# ETH ds5: E1–E2 intake and conditional E3–E5 plan

User authorized implementation after the high-level plan. Scope now: CPU-only intake,
cam0 acquisition and annotation preparation. No detector fitting, P8/P9 changes, PPO,
external messages or application forms. Existing NPS test/P10 receipts remain unchanged.

## Frozen input and known limitations

Upstream: https://github.com/CenekAlbl/drone-tracking-datasets
commit `2c857c97be71834d0791ae8ee4984ffb62b7680a`, CC BY-NC-SA 4.0 (retain both licenses).
Pose applies to Pixhawk drone0, not the other DJI drones. Root Sony5100 calibration
is a candidate for cam0, not automatically a validated ds5 calibration. Match resolution,
crop, lens and reprojection; resolution equality alone does not prove compatibility.

Published project-time frame timestamps must NOT receive the sync affine transform again.
Pose support ends near 187.6 s; video timestamps extend near 710.2 s. Use their intersection,
then quality filters, not the whole video. Record status counts without assuming 0/1 means
valid/invalid. Verify the upstream processing code and attitude/frame conventions first.
Do not interpolate across long gaps or extrapolate outside pose support.

## Execution and gates

| Stage | Output | Gate |
|---|---|---|
| E1 | Pinned files, hashes, schema/time-support audit | Source integrity passes; unresolved scientific checks remain explicit |
| E2 | cam0 archives; deterministic pilot annotation queue; reviewed drone0 boxes | Identity, visibility, annotation quality and camera compatibility verified |
| E3 | Separate detector-box error and pose/viewpoint-associated size variation | Preregister estimator, quality filters, time splits and exclusions before fitting |
| E4 | Range model including bias and temporal dependence | Held-out continuous segments; uncertainty includes annotation/GT limits |
| E5 | New versioned P9 range arm and frozen-policy evaluation | Injection validation and separate experiment preregistration; no automatic PPO training |

The initial queue samples every 150 frames WITHIN temporal overlap for feasibility only.
It is not a train/test split or a representative error estimate. Every row initially has
null box, unverified identity and measurement_eligible=false. No detector output becomes GT.
Dense continuous segments for E3 are selected and registered later, before error inspection.

Separate camera-to-target slant range from optical-axis depth. Analyze viewing direction
relative to body orientation, not absolute RPY alone. Do not double-count detector error when
combining an end-to-end ETH range model with NPS measurements. One flight/multiple views are
not independent flight replications. Fixed-camera ETH does not validate airborne-camera transfer.

## Commands

Metadata and pending annotation queue (standard-library Python; no GPU):

```bash
PYTHONNOUSERSITE=1 /home/fair/miniconda3/envs/aerialgym/bin/python tools/prepare_eth_ds5.py \
  --output /home/fair/workspaces/aerial_gym_ws/datasets/eth_ds5 \
  --report results/eth_ds5_intake_2026-09-10/receipt.json
```

Add `--video` to acquire cam0's 43 split-archive files (~4.47 GB). The tool reserves an
additional archive-sized extraction budget plus 2 GiB headroom and never deletes user files.
Downloads resume only with matching HTTP Content-Range; corrupt files are preserved and refused.
All completed files are verified against Git blob SHA-1 and recorded with SHA-256.
An interrupted intake can rerun; no report claims complete until all requested files verify.

Extract with a split-ZIP-capable tool into a NEW external directory, validate archive members
before extraction, then verify video dimensions/FPS/frame count against timestamps. Extraction
and frame decoding do not imply calibration or GT acceptance. Do not clone the whole dataset.

```bash
PYTHONNOUSERSITE=1 /home/fair/miniconda3/envs/aerialgym/bin/python -m unittest discover -s tests -p 'test_eth_ds5.py'
```

## E2 status (2026-09-10, after extraction)

Receipts: [results/eth_ds5_intake_2026-09-10/README.md](../../results/eth_ds5_intake_2026-09-10/README.md).

- Extraction: cam0.mp4 4,520,030,301 B, 7-Zip CRC OK, decoded fully (20970 frames, 1920x1080, 30000/1001).
- **Open blocker A — frame/time alignment (root cause found, mapping still open)**: the video track's
  edit list starts one frame into the media (audio and metadata start at zero), which is exactly the
  missing row; ffmpeg and OpenCV keep that frame (20970 decoded with and without `-ignore_editlist`),
  MATLAB's `readFrame` loop did not. Three whole-frame mappings survive (`frame_id`, `frame_id - 1`,
  `frame_id + 1`) and timestamps cannot separate them. The published stamps also carry cam1's
  `Time_scale` with cam0's `Time_shift`, because they were committed 2021-03-08 and the coefficient
  table was revised 2022-02-07 without regenerating them; at the favoured frame origin of 2 the
  revision is worth −3.0 to +1.7 ms, and the integer origin itself is worth at most ~2 frames. One
  frame of error displaces drone0 by 2.2 px median / 7.6 px p95, which the 4 px reprojection gate can
  separate. Pilot images were rendered under an explicit `UNRESOLVED_count_mismatch` flag.
- **Open blocker B — identity**: 36 pilot panels plus 17 fast-motion panels, GT geometry and motion cues
  are ready; no box exists. **The six-arm cue recorded here on 2026-09-10 is WITHDRAWN**: all three
  aircraft are quadrotors, frame 901 resolves four arms plus two landing legs, and rotor count separates
  nothing (`results/eth_ds5_intake_2026-09-10/CORRECTION_2026-09-11_rotor_count.md`). The review
  instructions now cite airframe style at close range instead, with E3-S's size-versus-range agreement
  (6.2% held out over 3,107 frames, no camera orientation needed) as the strongest surviving evidence.
- **Open blocker C — resolved in diagnosis, not in fix.** The published calibration is verified correct
  for its own 122 chessboard images (held-out 1.245 px versus 1.262 px recomputed), so the file is not
  stale. Against drone0, with tracking noise of 0.198 px over 3495 tracked frames, it leaves 5.6 px, and
  no camera-model hypothesis survives held-out testing (best 3.04 px, winner differs by split, a radial
  fit on the first half worsens the second). The inconsistency is more likely in the published timestamps
  or the fused pose; settling it needs the ds5 calibration session or the authors. **Apparent size versus
  range is stable to about 7 % over 18-108 m, so a size-only arm of E3 is feasible under separate
  preregistration while absolute-geometry uses stay blocked.** Earlier provisional reading: A provisional AI review of the 17
  alignment frames fits best at −3.53 frames (−117.6 ms), which is not an integer and so not a frame
  relabelling, and even there the RMS is 6.9 px against the 4 px gate. Freeing the radial coefficient
  alone (−0.100 versus the published −0.0112) brings it to 2.34 px with the focal length unchanged. A
  usable ds5 cam0 calibration has to come from the dataset or the authors; refitting from these boxes and
  then declaring the calibration valid would be circular. Original text: resolution/FPS match only. `tools/check_eth_ds5_reprojection.py`
  (rotation-only Kabsch fit + PnP centre vs surveyed cam0 position, shifts −3…+3 frames) decides after
  review; thresholds were fixed before any box: pixel RMS < 4 px, centre < 2 m, ≥ 6 boxes. The check now
  refuses points behind the camera or past the radial model's fold-back radius, scores every shift on the
  same frames, and separates "calibration consistent" from "alignment resolved" — only one shift passing
  counts as resolved. The pilot frames alone cannot resolve it; the 17-frame alignment set can.
- TrackingStatus = Leica TPS status (0 fine, 1 warning), attitude = ArduPilot EKF body→NED; still
  uninterpreted in any measurement. E3 not started.

Additional commands:

```bash
PY="env PYTHONNOUSERSITE=1 /home/fair/miniconda3/envs/aerialgym/bin/python"; FF=/home/fair/miniconda3/envs/aerialgym/bin
DS=/home/fair/workspaces/aerial_gym_ws/datasets/eth_ds5; V=/home/fair/workspaces/aerial_gym_ws/datasets/eth_ds5_cam0_extracted
R=results/eth_ds5_intake_2026-09-10
$PY tools/extract_eth_ds5.py --dataset $DS --receipt $R/video_receipt.json --output $V --sevenzip <7za>
$PY tools/verify_eth_ds5_video.py --video $V/cam0.mp4 --extraction-receipt $V/extraction_receipt.json --dataset $DS \
  --calibration $DS/calibration/sony5100/sony5100.json --output $R/video_verification --ffprobe $FF/ffprobe --ffmpeg $FF/ffmpeg
$PY tools/prepare_eth_ds5_frames.py --video $V/cam0.mp4 --queue $R/annotation_queue.json --intake-receipt $R/video_receipt.json \
  --calibration $DS/calibration/sony5100/sony5100.json --output $V/pilot_frames --ffprobe $FF/ffprobe --allow-count-mismatch
$PY tools/prepare_eth_ds5_review.py --pilot-dir $V/pilot_frames --video $V/cam0.mp4 --dataset $DS --output $V/review_material
$PY tools/check_eth_ds5_reprojection.py --review-csv <filled review_template.csv> --dataset $DS \
  --calibration $DS/calibration/sony5100/sony5100.json --output $R/reprojection_check   # after human review
```

## Schedule / fallback

E1: 0.5–1 day; E2–E4: provisional 3–6 days, revised after annotation feasibility.
E5 GPU budget is not fixed yet. If calibration/identity/synchronization cannot be validated,
stop quantitative pose-error claims. Author contact requires separate approval. In parallel,
the already-planned P8-support versus simulator-size audit can proceed under its own contract.
Missing range errors do NOT mathematically establish that previous P10 loss is a lower bound;
that monotonicity has to be tested, not assumed.
