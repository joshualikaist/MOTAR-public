# ETH ds5 cam0 alignment set: 17 fast-motion frames

Panels: `.../datasets/eth_ds5_cam0_extracted/alignment_review/review_XXXXXX.jpg`
Clean PNGs: `.../alignment_frames/frame_XXXXXX.png`. Hashes in `review_material_receipt.json`.

## Why this set exists

The 36-frame pilot in `review_offset0/` samples the flight every 150 frames, where drone0 moves a median
of 1.5 px between consecutive frames. Neighbouring alignment shifts then fit the reviewed boxes almost
equally well, so the reprojection check returns several consistent alignments and resolves nothing. This
set was chosen for image motion instead: 4.1 to 6.9 px per frame, median 4.9. With that, one frame of
timing error costs more than the annotation noise and exactly one alignment survives, provided box
centres are accurate to about 3 px.

Selection used ground-truth geometry only, plus an approximate camera pointing (azimuth 25.8, elevation
17.3 degrees, roll assumed zero) fitted to the airframe in frames 901 and 1651. (That airframe was
described as six-armed when the fit was made; it is a quadrotor. The fit used its image coordinates, not
its arm count, so the pointing is unaffected.)
That pointing decides only which frames you are shown. It is recorded as unverified, and the reprojection
check re-fits the orientation from your boxes alone, so a wrong pointing wastes your time but cannot make
a wrong result pass.

## What to expect

- drone0 is 62 to 99 m away, so the airframe is roughly 8 to 15 px across. Zoom in.
- It is moving fast, so it will be motion-blurred and may be elongated. Box the whole blur, and put the
  centre where the airframe centre is, not where the blur is brightest.
- **Correction, 2026-09-11: do not count rotors.** All three aircraft are quadrotors, so the count
  separates nothing; the earlier instruction is withdrawn (`../CORRECTION_2026-09-11_rotor_count.md`).
  Where the crop resolves the airframe, the cue is style: drone0 is a custom Pixhawk build with exposed
  landing legs and an antenna mast, while the Phantom and Mavic are moulded consumer airframes. At these
  ranges it will usually not resolve. Then rely on continuity with neighbouring frames and on the
  ground-truth range and bearing printed in the header, and say in `notes` what you relied on. That much
  confirms an aircraft of the right apparent size at the predicted place, not that it is drone0 rather
  than one of the other two. `unsure` remains a valid answer.
- Yellow `m#` boxes are temporal-difference motion cues. At these ranges they often latch onto fence
  posts and branches because the camera itself shakes. Do not copy them without looking.

## Fill `review_template.csv`

Same columns as the pilot set. Leave the box empty unless you are marking `drone0_visible=yes`.

## After review

Run the check twice and report both:

```bash
tools/check_eth_ds5_reprojection.py --review-csv <this set's filled csv> ... --container-index-offset 0
tools/check_eth_ds5_reprojection.py --review-csv <pilot + this set combined> ... --container-index-offset 0
```

The first answers the alignment question, because only these frames discriminate. The second answers the
calibration question with boxes spread more widely over the image. `CALIBRATION_CONSISTENT_ALIGNMENT_RESOLVED`
with exactly one consistent shift is the gate. `..._AMBIGUOUS` means the frames used do not separate the
shifts, and `CALIBRATION_CANDIDATE_NOT_VALIDATED` means no alignment fits at all, which points at the
calibration candidate, the surveyed position, or the identity.
