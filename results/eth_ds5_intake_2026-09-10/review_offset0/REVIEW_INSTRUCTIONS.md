# ETH ds5 cam0 pilot review: is drone0 visible, and where?

**Review this set.** Panels:
`/home/fair/workspaces/aerial_gym_ws/datasets/eth_ds5_cam0_extracted/review_material_offset0/review_XXXXXX.jpg`
Clean 1920x1080 PNGs: `.../pilot_frames_offset0/frame_XXXXXX.png`. Hashes are pinned in
`review_material_receipt.json`.

An earlier set exists under `review_material/` and `pilot_frames/`. It is the same 36 frame ids
rendered one container frame earlier, before the container's edit list was analysed. It is kept for
provenance only. Do not review both.

## Which container frame each image is

The published timestamp file has 20969 rows for 20970 container frames, because the video track's edit
list starts one frame into the media. This set uses `container_index = frame_id + 0`, the mapping that
edit list supports. Two other integer mappings remain possible, so the panels still say
`alignment UNRESOLVED_count_mismatch`. A one-frame error moves drone0 by about 2 px typically and 8 px
at the 95th percentile, which does not change whether you can see it.

## What each panel shows

- Header: frame id, container index, published project time, alignment status.
- `drone0 GT`: slant range from cam0, bearing (degrees from East, counter-clockwise, ENU), elevation and
  ground speed, interpolated from `fused_pose.txt`. The camera orientation is unknown, so no pixel
  position is predicted. Use range, elevation and speed as plausibility checks only.
- Yellow `m#` boxes are temporal-difference motion cues (frames -3/+3). They are hints, not detections,
  and they do not know which drone, person, bird or snowfall moved. Zoomed crops of the first three
  cues are appended below the frame.

## Three drones flew; only drone0 (Pixhawk) has GT

`drones.txt`: drone0 Pixhawk, drone1 DJI Phantom, drone2 DJI Mavic. A visible aircraft is NOT
automatically drone0.

**Correction, 2026-09-11: rotor count is NOT a cue and the earlier instruction to use it is withdrawn.**
All three aircraft are quadrotors. Frame 901 at 14x resolves four thick arms carrying rotors, two thin
landing legs and an antenna mast; the legs were previously miscounted as arms. See
`../CORRECTION_2026-09-11_rotor_count.md`.

**Airframe style is the cue where the crop resolves it.** drone0 is a custom Pixhawk build with exposed
landing legs and an antenna mast. The Phantom and the Mavic are moulded consumer airframes and the Mavic
folds its arms and has no exposed legs. Record what you actually saw in `notes`; where it is a dark blob,
do not guess from motion alone. Mark `drone0_visible=yes` only when you can argue identity, for example continuity
with take-off near cam0 (GT: 5.8 m from cam0 on the ground until about 30 s, climbing from about 30.6 s),
apparent size consistent with the GT range across frames, or other drones being visible elsewhere at the
same time. Write the argument in `notes`. `unsure` is a valid answer.

## Fill `review_template.csv`

- `drone0_visible`: yes / no / unsure
- `box_x1,box_y1,box_x2,box_y2`: tight pixel box in the 1920x1080 clean PNG (x right, y down), only when
  visible; leave empty otherwise
- `other_drones_visible_count`, `confidence` (high/medium/low), `reviewer`, `notes`

Do not edit `project_timestamp_s` or the GT columns. Never copy a motion cue into a box without looking
at the crop yourself. Keep the file next to the panels and do not overwrite the template.

## After review

```bash
tools/check_eth_ds5_reprojection.py --review-csv <filled csv> --dataset <datasets/eth_ds5> \
  --calibration <.../sony5100.json> --output <results/.../reprojection_check> --container-index-offset 0
```

It needs at least 6 confident boxes spread across the frame. It tests whether the reviewed boxes, the
Sony5100 calibration candidate, the surveyed cam0 position and the GT trajectory are mutually consistent,
over whole-frame shifts of -3 to +3. A winning shift of 0 confirms the edit-list mapping; +1 would mean
the reader dropped the last frame instead. If nothing is consistent, the calibration candidate, the
synchronisation or the identity is wrong, and E3 does not start.
