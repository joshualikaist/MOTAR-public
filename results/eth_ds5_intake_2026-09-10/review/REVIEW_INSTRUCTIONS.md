# ETH ds5 cam0 pilot review: is drone0 visible, and where?

Material (outside Git, hash-pinned in `review_material_receipt.json`):
`/home/fair/workspaces/aerial_gym_ws/datasets/eth_ds5_cam0_extracted/review_material/review_XXXXXX.jpg`
(one panel per pilot frame; the clean 1920x1080 PNG is `pilot_frames/frame_XXXXXX.png`).

## What each panel shows

- Header: frame id, OpenCV index, published project time, and the index-alignment status.
  The container has 20970 frames but the published timestamp file has 20969 rows, so the mapping
  frame_id -> container index is uncertain by one frame (33 ms). This does not matter for deciding
  visibility; it does matter for measurement, which stays blocked.
- `drone0 GT` line: slant range from cam0, bearing (degrees from East, counter-clockwise, ENU),
  elevation and ground speed interpolated from `fused_pose.txt`. The camera orientation is unknown,
  so no pixel location is predicted. Use range/elevation/speed only as plausibility checks.
- Yellow `m#` boxes are temporal-difference motion cues (frames -3/+3). They are hints, not
  detections, and they do not know which drone (or person, bird, snow) moved. Zoomed crops of
  the first three cues are appended below the frame.

## Three drones flew; only drone0 (Pixhawk) has GT

`drones.txt`: drone0 Pixhawk, drone1 DJI Phantom, drone2 DJI Mavic. A visible aircraft is NOT
automatically drone0. Mark `drone0_visible=yes` only when you can argue identity, e.g. continuity
with take-off near cam0 (GT: 5.8 m from cam0 on the ground until ~30 s, climbing from ~30.6 s),
consistent range-driven apparent size across frames, or other drones being simultaneously visible
elsewhere. Write the argument in `notes`. `unsure` is a valid answer.

## Fill `review_template.csv`

- `drone0_visible`: yes / no / unsure
- `box_x1,box_y1,box_x2,box_y2`: tight pixel box in the 1920x1080 clean PNG (x right, y down),
  only when visible; leave empty otherwise
- `other_drones_visible_count`, `confidence` (high/medium/low), `reviewer`, `notes`

Do not edit `project_timestamp_s` or GT columns. Never derive a box from the motion cue without
looking at the crop yourself. Keep the file next to the panels and do not overwrite the template.

## After review

`tools/check_eth_ds5_reprojection.py --review-csv <filled csv> ...` tests whether the reviewed
boxes, the Sony5100 calibration candidate, the surveyed cam0 position and the GT trajectory are
mutually consistent (needs >= 6 confident boxes spread across the frame). A failure there means the
calibration candidate, the sync, or the identity is wrong, and E3 does not start.

> Superseded by `../review_offset0/`. This set was rendered at `container_index = frame_id - 1`
> before the container edit list was analysed; it is kept for provenance only.
