# ETH ds5 cam0 intake, E1–E2 (2026-09-10)

Status: **E2 material ready, gates NOT passed.** drone0 identity, frame-time alignment and the
Sony5100 calibration candidate remain unverified; E3 (error modelling) has not started.

| Receipt | Content | Result |
|---|---|---|
| `receipt.json`, `video_receipt.json` | E1 pinned files, Git blob SHA-1 + SHA-256, 43 split-archive parts (4,471,880,421 B) | verified |
| `extraction_receipt.json` | cam0.mp4 4,520,030,301 B, SHA-256 `d3e6727e…c1ec4`, 7-Zip CRC OK | extracted outside Git: `datasets/eth_ds5_cam0_extracted/` |
| `video_verification/video_verification.json` | h264 1920x1080 30000/1001, 20970 packets = 20970 decoded = nb_frames | **20969 published timestamp rows: off by one** |
| `review_offset0/` | 36 pilot panels at `container_index = frame_id + 0`, GT geometry, motion cues, contact sheet, empty reviewer table | identity unverified; too slow to resolve the alignment |
| `review_alignment/` + `alignment_queue.json` | **17 fast-motion frames** (4.1-6.9 px of image motion per frame) selected to resolve the alignment | identity unverified |
| `review/` | the first set, rendered at `container_index = frame_id - 1` before the edit list was analysed | superseded, kept for provenance |

## Root causes of the two timestamp blockers (2026-09-10, investigated)

### Why 20969 published rows for 20970 container frames

The MP4 video track carries an edit list `media_time = 1001` at a 30000 timescale, i.e. exactly one
frame period, while the audio and metadata tracks carry `media_time = 0`. The edit therefore says
presentation begins one frame into the media, so a reader honouring it presents 20969 frames. Upstream
generated the file with a MATLAB `VideoReader` `readFrame` loop, which honours edit lists, and got
20969 rows.

ffmpeg and OpenCV do **not** drop that frame: full decodes return 20970 both by default and with
`-ignore_editlist 1`, and OpenCV's first three frames are byte-identical to ffmpeg's default output.
So the missing row is at one end of the container, but three candidate mappings remain, and the
timestamps cannot separate them because each is a whole-frame relabelling:

| candidate | container index of `frame_id` | evidence |
|---|---|---|
| edit list skips the first frame | `frame_id` | container edit list, quantitatively exact |
| reader dropped the last frame | `frame_id - 1` | documented MATLAB `readFrame`/`NumFrames` discrepancy, no container evidence |
| `CurrentTime` is the next frame's time | `frame_id + 1` | MathWorks semantics with the fitted origin; predicts one row too few, kept only as a bound |

The first pilot was rendered with `frame_id - 1`, which was hard-coded before the edit list was known.
The mapping is now an explicit `--index-offset` argument, and the set to review (`review_offset0/`) uses
offset 0, the candidate the container supports. One frame of timing error displaces drone0 in the image
by a median of 2.2 px, a 95th percentile of 7.6 px and at most 20.3 px, so the mapping is immaterial for
deciding visibility and material for measurement.

The reprojection check recovers the true offset: images rendered at `container_index = frame_id + offset`
fit at `shift = offset - d`, so `d = offset - shift`. With offset 0 a winning shift of 0 confirms the
edit-list candidate and +1 would mean the reader dropped the last frame instead.

### Why the stamps carry a scale that is not cam0's

The published stamps are exactly affine in frame id (maximum residual 33 ns):
`t(j) = 0.033367703956 * j + 10.551275280`. Writing this as `scale * (j + origin) * period + shift`
and testing every row of `sync_coefficients_cam2pc.txt`:

- the slope matches **cam1's** `Time_scale` (1.000031087) to 5.9e-10; every other camera is off by at
  least 2.1e-6;
- the intercept only yields a near-integer frame origin for **cam0's** `Time_shift`: origin 1.9486,
  i.e. 0.051 frames from 2. Every other row lands 0.25 to 0.46 frames from an integer.
- at origin 2 the implied shift is 10.48454 s, which is 1.7 ms from cam0's current 10.48625553 s.

The stamps were committed on 2021-03-08; the coefficient table was revised on 2022-02-07 and the stamps
were never regenerated. The pre-revision table displayed rounded values `1.000` and `10.48`, which are
consistent with the pair the stamps actually used (1.000031087, 10.48454) and inconsistent with origins
1 or 3 (which would need 10.518 or 10.451). So the file predates the revision rather than being
internally corrupt.

Practical size of the discrepancy, recomputing the stamps from cam0's current coefficients:

| assumed origin | change over the pose overlap |
|---|---|
| 0 | −69.7 to −65.0 ms (−2.09 to −1.95 frames) |
| 1 | −36.4 to −31.7 ms (−1.09 to −0.95 frames) |
| **2 (favoured)** | **−3.0 to +1.7 ms (−0.09 to +0.05 frames)** |
| 3 | +30.4 to +35.1 ms (+0.91 to +1.05 frames) |

The scale difference alone contributes at most 4.7 ms across the whole pose overlap. So the coefficient
revision is harmless once the integer origin is fixed; the residual uncertainty is the integer, worth
at most about two frames. The reprojection check searches whole-frame shifts of −3…+3, which covers
every combination above, and its 4 px RMS gate is below the 7.6 px that one frame of error produces.

## Code back-derivation, 2026-09-10

Every derivation in the tools was re-checked against an independent computation before going further.
Reconstructing the published timestamp file from the fitted scale, origin and shift reproduces it to
4.4e-7 s, which is exactly the 9-digit rounding of the published coefficient. Projection, ray recovery
and the Kabsch rotation fit round-trip to 1e-12 or better against a hand-written pinhole-plus-Brown model.
Interpolation, bearing and speed match a direct bracket computation exactly. The mapping
`d = offset - shift` was confirmed end to end for six combinations of rendering offset and true offset.

That pass also found six real defects, all now fixed and covered by tests:

1. **Edit-list frames were computed from an assumed 30000/1001 rate**, correct here only by coincidence.
   The parser now reads each track's own `mdhd` timescale and `stts` sample duration.
2. **The reprojection check could not resolve the alignment at all.** With the pilot frames, three
   neighbouring shifts pass the 4 px gate even with perfect boxes, because the target moves only about
   1.5 px per frame there. The tool now reports `..._ALIGNMENT_RESOLVED` only when exactly one shift
   passes, and publishes the measured discriminating power alongside.
3. **Shifts were scored on different frame subsets**, since a shift near the edge of pose support
   silently dropped frames. All shifts are now scored on the common set.
4. **Ground-truth points behind the fitted camera were accepted**; such a point still projects to a
   plausible in-image pixel. They are now refused.
5. **Points past the radial model's fold-back radius were accepted.** For this calibration the radial
   polynomial stops increasing at a normalized radius of 1.023 while the image corner is at 0.713, so a
   target 70 degrees off axis folds back into the picture and `undistortPoints` inverts it onto the wrong
   branch. The limit is now derived from the coefficients and such points are refused.
6. **Interpolated attitude could leave the [-180, 180] range** after wrapping.

## Resolving the alignment needs fast frames

Image motion per frame of timing error, over the pose overlap: median 1.61 px, 90th percentile 4.41 px,
maximum 20.23 px. The stride-150 pilot sits at 1.40 px median, so it cannot separate neighbouring shifts
however carefully it is annotated. `tools/select_eth_ds5_alignment_frames.py` picks frames on motion
instead, subject to a minimum range of 25 m and a predicted-in-view test; the 17 it selected span 4.07 to
6.94 px per frame at 62 to 99 m. In simulation that set resolves the alignment uniquely with box-centre
noise up to 3 px, and fails closed at 4 px. Combining it with the pilot is better for the calibration
question and worse for the alignment question, so the check is meant to be run on both.

The in-view test needs an approximate camera pointing, which is an explicit argument recorded as an
unverified assumption (azimuth 25.8, elevation 17.3 degrees, roll zero, fitted to the airframe in frames
901 and 1651 with a 0.11 degree residual; that airframe is a quadrotor, not the six-armed one the note
originally described, and the fit used its image coordinates rather than its arm count). It only chooses which frames a human is shown. Eight
of the twenty frames selected before this test existed had drone0 outside the picture entirely.

## Provisional review result (2026-09-10)

A provisional review of the 17 alignment frames by Claude, not by the independent human reviewer the plan
requires, is in [provisional_review_2026-09-10/](provisional_review_2026-09-10/README.md). It answers the
design question and changes the blocker.

- **The alignment set discriminates.** The pixel RMS curve has a single minimum and rises 2 to 3 px per
  frame of shift. The stride-150 pilot could not have done this.
- **The mismatch is not a frame relabelling.** The refined optimum sits at −3.53 frames, −117.6 ms, which
  is 0.47 frames from any integer. None of the three index candidates explains it. The old ±3 search
  would have missed it; the search now spans ±8, refines the continuous optimum, and refuses to report an
  index offset when the optimum is not an integer.
- **The published calibration fails.** Best RMS 6.9 px against the 4 px gate, with residuals correlating
  0.78 with radial distance. Freeing the radial coefficient alone, to −0.100 versus the published
  −0.0112, brings RMS to 2.34 px while the focal length stays unchanged. The Sony5100 file does not
  describe ds5 cam0's lens as published.

## Distortion investigation (2026-09-10): the calibration is not the fault

Full write-up in [distortion_investigation_2026-09-10/](distortion_investigation_2026-09-10/README.md).
This supersedes the provisional review's reading that the distortion coefficients are wrong.

- **The published calibration is correct for its own images.** The dataset's 122 a5100 chessboard images
  were fetched and hash-verified; recomputing the intrinsics from half of them and scoring on the other
  half gives 1.262 px, against 1.245 px for the published file. It is not stale or mis-assigned.
- **The residual against drone0 is 28 times the measurement noise.** The 41 reviewed centres were
  propagated to 3495 tracked positions; tracking noise is 0.198 px, measured from trajectory smoothness
  with no ground truth. The published calibration leaves 5.6 px RMS on 678 of them.
- **No camera-model hypothesis survives held-out testing.** Radial distortion, focal length, principal
  point, camera position, time drift, an 8 s rotation refit and the prism lever arm were each fitted on
  one split and scored on another, with the rotation refitted on the held-out split. The winner differs
  by split and the best held-out result is 3.04 px, still 15 times the noise. A radial model fitted on
  the first half makes the second half worse.
- **An unexplained −3.6 frame (−120 ms) offset is needed by every variant.**
- **What still works**: apparent size against range is stable to about 7 % over 18 to 108 m, because a
  5 px position error does not corrupt a size measurement. A size-versus-range arm of E3 is feasible
  under separate preregistration; absolute-geometry uses stay blocked.

## Findings that still block measurement

1. **Frame/time alignment**: three candidate mappings, unresolved without reviewed boxes (above).
2. **Identity**: three drones flew and upstream publishes no 2D labels for ds5. GT is drone0 (Pixhawk)
   only. GT places drone0 5.8 m from cam0 on the ground until about 30 s, then 5–108 m away.
3. **Calibration candidate**: not usable for absolute geometry, but the file itself is verified correct
   for its own images, see the distortion investigation; resolution 1920x1080 and 29.97 fps match `calibration/sony5100/sony5100.json`;
   the file carries no camera-model, lens or zoom tag (XAVC brand only) and ds5 publishes no per-camera
   calibration. Compatibility is testable only by reprojecting reviewed drone0 boxes
   (`tools/check_eth_ds5_reprojection.py`; thresholds fixed before any box existed: pixel RMS < 4 px,
   PnP centre within 2 m of the surveyed cam0 position, at least 6 boxes, shifts −3…+3 frames).
4. **TrackingStatus** is the Leica total-station status (0 fine, 1 warning that fast motion may reduce
   accuracy, 2 lost and already filtered upstream), not a visibility flag. Attitude is the ArduPilot
   EKF roll/pitch/yaw, body to local NED, while positions are ENU. Both stay uninterpreted here.

## Local environment notes

Extraction used p7zip 16.02 from Ubuntu `p7zip-full_16.02+dfsg-7build1_amd64.deb`
(SHA-256 `efc2d2795fe6c707183a4b7f4146477fc410c478131bf451914d534246d06896`) unpacked under `/tmp`
without installation. ffprobe/ffmpeg 7.1 and OpenCV 4.13 come from the `aerialgym` conda environment.

A previous session had already extracted the same video to `datasets/eth_ds5_extracted_cam0/`; this
session extracted it again to `datasets/eth_ds5_cam0_extracted/`. Both files hashed to the same
SHA-256 `d3e6727e…c1ec4` over their full length, so the duplicate directory was removed on the user's
instruction and the verified copy kept. Unrelated installers were removed from the user's Downloads at
the same time. Earlier, to make room for the extraction, only the conda package tarball cache and the
pip cache were purged. No dataset, receipt or run file was removed at any point.

## Next (E2 completion, not E3)

Human review of the 36 panels in `review_offset0/` → filled `review_template.csv` →
`check_eth_ds5_reprojection.py --container-index-offset 0`. Only if that reports
`CALIBRATION_CANDIDATE_CONSISTENT` for some alignment shift may dense segments be registered for E3,
and the chosen shift must then be preregistered, not tuned.

Identity note for the reviewer, corrected 2026-09-11: **all three aircraft are quadrotors, so rotor count
separates nothing.** Frame 901's crop at 17.9 m resolves four thick arms carrying rotors plus two thin
landing legs and an antenna mast; the legs were previously miscounted as arms and the "six-armed" cue is
withdrawn (`CORRECTION_2026-09-11_rotor_count.md`). What survives, strongest first: apparent size follows
drone0's ground-truth range, with a held-out median relative error of 6.2% over 3,107 frames and 9 blocks
from 30.7 to 108.4 m (E3-S, and it needs no camera orientation); and the custom-build airframe style at
close range against the two moulded DJI consumer airframes. The 0.6 degree angular residual cited earlier
is the residual of a camera pose fitted to the same 15 reviewed boxes, and that fit fails the calibration
gate at 7.10 px RMS, so it is corroboration at most.
