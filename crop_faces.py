"""
crop_faces.py - data prep step 2: cut the face out of every frame.

    data/videos/frames/<name>/frame_*.jpg
        -> data/videos/face_frames/<name>/frame_*.jpg   (128 x 128)

Uses the single crop recipe in painsense/face.py (largest Haar face,
margin 0.15 * max(w, h), resize to 128). Frames without a detected face
are skipped, never center-cropped. Folders that already have crops are
skipped unless --overwrite is given, so the existing training crops are
never replaced by accident.

Run:
    python crop_faces.py
"""

import argparse
from pathlib import Path

import cv2

from painsense import config
from painsense.face import crop_from_box, detect_face_box


def main():
    ap = argparse.ArgumentParser(
        description="Cut the face out of every extracted frame."
    )
    ap.add_argument("--frames", default=config.VIDEO_FRAME_DIR)
    ap.add_argument("--out", default=config.VIDEO_FACE_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    in_root, out_root = Path(args.frames), Path(args.out)
    folders = sorted(f for f in in_root.iterdir() if f.is_dir())
    print(f"Found {len(folders)} frame folders")

    processed = faces = skipped = 0

    for folder in folders:
        out_dir = out_root / folder.name

        if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
            print(f"  skip {folder.name} (crops exist)")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        for frame_path in sorted(folder.glob("*.jpg")):
            image = cv2.imread(str(frame_path))
            processed += 1

            box = detect_face_box(image) if image is not None else None
            crop = crop_from_box(image, box) if box is not None else None

            if crop is None:
                skipped += 1
                continue

            cv2.imwrite(str(out_dir / frame_path.name), crop)
            faces += 1

        print(f"  {folder.name}")

    print()
    print(f"Frames processed : {processed}")
    print(f"Faces saved      : {faces}")
    print(f"No face (skipped): {skipped}")
    print(f"Output folder    : {out_root}")


if __name__ == "__main__":
    main()
