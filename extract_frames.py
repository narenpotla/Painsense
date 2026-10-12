"""
extract_frames.py - data prep step 1: split SynPAIN videos into frames.

    data/videos/raw/SynPain/Videos/<name>.mp4
        -> data/videos/frames/<name>/frame_000001.jpg, frame_000002.jpg, ...

Frame numbers start at 1 and are used later to build the 0-10 target
(linear in frame index for Pain clips). Videos whose frame folder
already exists are skipped unless --overwrite is given.

Run:
    python extract_frames.py
Next step: python crop_faces.py
"""

import argparse
from pathlib import Path

import cv2

from painsense import config


def extract_video(video_path, out_dir):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: could not open {video_path.name}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        count += 1
        cv2.imwrite(str(out_dir / f"frame_{count:06d}.jpg"), frame)

    cap.release()
    return count


def main():
    ap = argparse.ArgumentParser(
        description="Split SynPAIN videos into frames."
    )
    ap.add_argument("--videos", default=config.RAW_VIDEO_DIR)
    ap.add_argument("--out", default=config.VIDEO_FRAME_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    videos = sorted(Path(args.videos).glob("*.mp4"))
    out_root = Path(args.out)

    if not videos:
        raise SystemExit(f"No .mp4 files found in {args.videos}")

    print(f"Found {len(videos)} videos")

    for video in videos:
        out_dir = out_root / video.stem

        if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
            print(f"  skip {video.stem} (frames exist)")
            continue

        n = extract_video(video, out_dir)
        print(f"  {video.stem}: {n} frames")

    print(f"Frames in {out_root}")


if __name__ == "__main__":
    main()
