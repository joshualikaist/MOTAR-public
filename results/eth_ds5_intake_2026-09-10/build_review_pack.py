"""Build the 10-frame human-review pack: full frame beside a large crop, nothing else.

Corrected 2026-09-11: the first two questions asked about a six-armed airframe. All three
aircraft in this dataset are quadrotors, so that question could not separate them and the
answer column that asked for an arm count is gone. See CORRECTION_2026-09-11_rotor_count.md.

WARNING (2026-09-11): the centres file used for the shipped pack is NOT in this repository and could
not be recovered. track_seeds.json covers only six of the ten frames and stores bare [x, y] lists,
which this loader does not accept. Re-running this script therefore does NOT reproduce the shipped
images: four crops would fall back to the whole-frame view and lose the recorded centre marks the
reviewer already commented on. The 2026-09-11 correction rebuilt the archive around the existing
images rather than re-rendering them.

The earlier panels carried motion cues and small thumbnails, which made the one question a reviewer has
to answer harder to see, not easier. Each image here shows the frame on the left with the inspected
region marked, and that region enlarged on the right with the recorded centre crossed. The reviewer
answers one question per image.

Headers are drawn with PIL and a CJK font: OpenCV's putText cannot render Korean and silently emits
question marks, which is exactly the kind of unreadable output this pack exists to avoid.
"""
import argparse
import csv
import json
from pathlib import Path
import zipfile

CJK_FONTS = ("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/nanum/NanumGothic.ttf")

FRAMES = [
    (901, 17.9, "회전익 항공기인가 (팔 4개 + 착륙 다리)", "positive"),
    (1051, 30.7, "901과 같은 기체인가", "positive"),
    (4937, 61.5, "회전익 형상이 보이는가", "positive"),
    (5251, 61.4, "비행 후반, 같은 기체인가", "positive"),
    (1605, 82.3, "흐릿하지만 항공기인가", "positive"),
    (3909, 97.0, "최원거리, 항공기인가", "positive"),
    (1651, 87.2, "표시가 나무에 있지 않은가", "excluded"),
    (1778, 87.5, "표시가 나무 꼭대기에 있지 않은가", "excluded"),
    (1, 5.8, "표적이 화면에 없는 것이 맞는가", "absent"),
    (3451, 54.9, "표시된 것이 드론이 아니라 나무인가", "negative"),
]
EXPECTED = {"positive": "yes 예상", "excluded": "제외 대상 — 표시가 기체가 아님", 
            "absent": "no 예상 — 화면 밖", "negative": "no 예상 — 나무"}


def load_font(size):
    from PIL import ImageFont
    for path in CJK_FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise ValueError("no CJK font available; headers would render as question marks")


def header_image(width, lines):
    """Header band drawn with PIL so Korean renders, returned as BGR for OpenCV."""
    import numpy as np
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (width, 124), (22, 22, 22))
    draw_on = ImageDraw.Draw(image)
    for text, size, colour, xy in lines:
        draw_on.text(xy, text, font=load_font(size), fill=colour)
    return np.asarray(image)[:, :, ::-1].copy()


def draw(video, frame_id, centre, order, total, distance, question, kind, out_dir, half=52, zoom=6):
    import cv2
    import numpy as np
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)      # container index = frame_id + 0
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError("decode failed at %d" % frame_id)
    full = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    if centre is None:
        crop = cv2.convertScaleAbs(cv2.resize(frame, (540, 304), interpolation=cv2.INTER_AREA), alpha=1.8, beta=-60)
        crop = cv2.copyMakeBorder(crop, 118, 118, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    else:
        cx, cy = int(round(centre[0])), int(round(centre[1]))
        x0, y0 = max(0, min(frame.shape[1] - 2 * half, cx - half)), max(0, min(frame.shape[0] - 2 * half, cy - half))
        patch = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
        crop = cv2.convertScaleAbs(cv2.resize(patch, (2 * half * zoom, 2 * half * zoom),
                                              interpolation=cv2.INTER_NEAREST), alpha=1.9, beta=-70)
        px, py = int((cx - x0) * zoom), int((cy - y0) * zoom)
        for d in (-1, 1):
            cv2.line(crop, (px + d * 34, py), (px + d * 96, py), (0, 0, 255), 2)
            cv2.line(crop, (px, py + d * 34), (px, py + d * 96), (0, 0, 255), 2)
        sx, sy = 960 / frame.shape[1], 540 / frame.shape[0]
        cv2.rectangle(full, (int((cx - half) * sx), int((cy - half) * sy)),
                      (int((cx + half) * sx), int((cy + half) * sy)), (0, 220, 255), 2)
    height = max(full.shape[0], crop.shape[0])
    pad = lambda img: cv2.copyMakeBorder(img, 0, height - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    body = np.concatenate([pad(full), np.full((height, 16, 3), 40, np.uint8), pad(crop)], axis=1)
    header = header_image(body.shape[1], [
        ("%d / %d      frame %d      GT 거리 %.1f m" % (order, total, frame_id, distance),
         34, (255, 255, 255), (20, 12)),
        (question, 28, (255, 220, 0), (20, 68)),
        (EXPECTED[kind], 22, (160, 160, 160), (body.shape[1] - 430, 74))])
    image = np.concatenate([header, body], axis=0)
    path = out_dir / ("%02d_frame%06d_%.0fm.jpg" % (order, frame_id, distance))
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--centres", type=Path, nargs="+", required=True,
                        help="JSON files mapping frame id to a recorded centre")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    centres = {}
    for path in args.centres:
        for key, value in json.loads(path.read_text()).items():
            if value is None:
                continue
            point = value.get("centre") if "centre" in value else [value["cx"], value["cy"]]
            centres.setdefault(int(key), point)
    args.output.mkdir(parents=True, exist_ok=True)
    made = []
    for order, (frame_id, distance, question, kind) in enumerate(FRAMES, 1):
        made.append(draw(args.video, frame_id, centres.get(frame_id), order, len(FRAMES),
                         distance, question, kind, args.output))
    sheet = args.output / "answers.csv"
    with sheet.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["no", "image", "frame_id", "gt_range_m", "question",
                         "your_answer_yes_no_unsure", "what_you_saw", "notes"])
        for order, (frame_id, distance, question, _) in enumerate(FRAMES, 1):
            writer.writerow([order, made[order - 1].name, frame_id, distance, question, "", "", ""])
    archive = args.output.parent / "eth_ds5_human_review_pack.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for path in made + [sheet, args.output / "READ_ME_FIRST.txt"]:
            if path.exists():
                zipped.write(path, path.name)
    print(json.dumps({"images": len(made), "zip": str(archive),
                      "zip_bytes": archive.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
