"""Cut the 1920x1080 NPS-Drones frames into 640x640 tiles so targets keep their native scale.

Why tile instead of resize: the prepared train split has a median target of 19.5 px equivalent
side and a 5th percentile of 9.7 px. Feeding 1920x1080 to a 640 network scales those to 6.5 px and
3.2 px, which is the regime where the published GLAD appearance branch already failed on this data
(recall 0.152, results/air2air_zeroshot/). A 640 tile keeps 19.5 px at 19.5 px and fits a 4 GB
card at a batch size that still trains, which a 1920 input does not.

Negative tiles are sampled on purpose. A detector trained only on tiles that contain a drone has
never seen empty sky or clutter at training scale and will fire on both.

Test is deliberately NOT tiled: the final judgement has to be made on whole frames through a
sliding window, otherwise the tiling grid itself becomes part of the reported number.
"""

import argparse
import collections
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2

TILE = 640
OVERLAP = 128
MIN_VISIBLE_FRACTION = 0.6   # a box clipped below this by a tile edge is dropped, not truncated
MIN_SIDE_PX = 4.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("../../datasets/nps_yolo"))
    p.add_argument("--output", type=Path, default=Path("../../datasets/nps_yolo_tiles640"))
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--frame-stride", type=int, default=2, help="use every Nth prepared frame")
    p.add_argument("--negatives-per-frame", type=int, default=1)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--seed", type=int, default=1729)
    return p.parse_args()


def tile_origins(extent, tile=TILE, overlap=OVERLAP):
    """Left/top offsets covering `extent` with `tile`-wide windows; the last one is flush right."""
    if extent <= tile:
        return [0]
    step = tile - overlap
    origins = list(range(0, extent - tile + 1, step))
    if origins[-1] != extent - tile:
        origins.append(extent - tile)
    return origins


def boxes_in_tile(boxes, ox, oy, width, height):
    """Reproject absolute boxes into a tile; keep only those still mostly visible."""
    kept = []
    for cx, cy, bw, bh in boxes:
        left, top = cx - bw / 2.0, cy - bh / 2.0
        right, bottom = cx + bw / 2.0, cy + bh / 2.0
        vl, vt = max(left, ox), max(top, oy)
        vr, vb = min(right, ox + TILE), min(bottom, oy + TILE)
        if vr - vl <= MIN_SIDE_PX or vb - vt <= MIN_SIDE_PX:
            continue
        if (vr - vl) * (vb - vt) < MIN_VISIBLE_FRACTION * bw * bh:
            continue
        kept.append((((vl + vr) / 2.0 - ox) / TILE, ((vt + vb) / 2.0 - oy) / TILE,
                     (vr - vl) / TILE, (vb - vt) / TILE))
    return kept


def read_labels(path, width, height):
    boxes = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = parts
        boxes.append((float(cx) * width, float(cy) * height, float(bw) * width, float(bh) * height))
    return boxes


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    source, out_root = Path(args.source), Path(args.output)
    if out_root.exists():
        raise SystemExit(f"[tile] refusing to overwrite: {out_root}")
    stats = collections.defaultdict(lambda: {"frames": 0, "pos_tiles": 0, "neg_tiles": 0,
                                             "boxes": 0, "bytes": 0})
    for split in args.splits:
        images = sorted((source / "images" / split).glob("*.jpg"))
        if not images:
            raise SystemExit(f"[tile] no images in {source / 'images' / split}")
        image_dir = out_root / "images" / split
        label_dir = out_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index, image_path in enumerate(images):
            if index % args.frame_stride:
                continue
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            height, width = frame.shape[:2]
            boxes = read_labels(source / "labels" / split / f"{image_path.stem}.txt", width, height)
            bucket = stats[split]
            bucket["frames"] += 1
            empty = []
            for oy in tile_origins(height):
                for ox in tile_origins(width):
                    kept = boxes_in_tile(boxes, ox, oy, width, height)
                    if not kept:
                        empty.append((ox, oy))
                        continue
                    name = f"{image_path.stem}_x{ox}_y{oy}"
                    path = image_dir / f"{name}.jpg"
                    cv2.imwrite(str(path), frame[oy:oy + TILE, ox:ox + TILE],
                                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                    (label_dir / f"{name}.txt").write_text(
                        "".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in kept))
                    bucket["pos_tiles"] += 1
                    bucket["boxes"] += len(kept)
                    bucket["bytes"] += path.stat().st_size
            rng.shuffle(empty)
            for ox, oy in empty[:args.negatives_per_frame]:
                name = f"{image_path.stem}_x{ox}_y{oy}"
                path = image_dir / f"{name}.jpg"
                cv2.imwrite(str(path), frame[oy:oy + TILE, ox:ox + TILE],
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                (label_dir / f"{name}.txt").write_text("")   # explicit background sample
                bucket["neg_tiles"] += 1
                bucket["bytes"] += path.stat().st_size
        b = stats[split]
        print(f"[tile] {split}: {b['frames']} frames -> {b['pos_tiles']} positive + "
              f"{b['neg_tiles']} negative tiles, {b['boxes']} boxes, {b['bytes'] / 1e9:.2f} GB")

    (out_root / "nps_tiles.yaml").write_text(
        "# NPS-Drones 640 tiles (BSD-3 source). tools/tile_nps_yolo_dataset.py\n"
        f"path: {out_root.resolve()}\n"
        "train: images/train\nval: images/val\n"
        "nc: 1\nnames: [uav]\n"
    )
    source_receipt = source / "receipt.json"
    (out_root / "receipt.json").write_text(json.dumps({
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source.resolve()),
        "source_receipt_sha256": (hashlib.sha256(source_receipt.read_bytes()).hexdigest()
                                  if source_receipt.is_file() else None),
        "tile": TILE, "overlap": OVERLAP,
        "min_visible_fraction": MIN_VISIBLE_FRACTION, "min_side_px": MIN_SIDE_PX,
        "frame_stride": args.frame_stride, "negatives_per_frame": args.negatives_per_frame,
        "jpeg_quality": args.jpeg_quality, "seed": args.seed,
        "test_split": "not tiled on purpose; judge whole frames with a sliding window",
        "splits": {k: dict(v) for k, v in sorted(stats.items())},
    }, indent=2, sort_keys=True) + "\n")
    print(f"[tile] wrote {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
