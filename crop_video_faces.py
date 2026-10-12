import cv2
from pathlib import Path

import config
from data_prep import detect_face_box, crop_from_box
from utils import seed_everything

seed_everything(config.SEED)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Input and output folders
INPUT_DIR = PROJECT_ROOT / "data" / "videos" / "frames"
OUTPUT_DIR = PROJECT_ROOT / "data" / "videos" / "face_frames"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

video_folders = sorted(
    [folder for folder in INPUT_DIR.iterdir() if folder.is_dir()]
)

print(f"Found {len(video_folders)} video frame folders")

total_processed = 0
total_faces = 0
total_failed = 0

for video_folder in video_folders:

    output_folder = OUTPUT_DIR / video_folder.name
    output_folder.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(video_folder.glob("*.jpg"))

    print()
    print(f"Processing: {video_folder.name}")
    print(f"Frames found: {len(frame_files)}")

    for frame_path in frame_files:

        image = cv2.imread(str(frame_path))

        if image is None:
            total_failed += 1
            continue

        total_processed += 1

        box = detect_face_box(image)

        if box is None:
            total_failed += 1
            continue

        face = crop_from_box(image, box, size=config.CROP_SIZE)

        if face is None:
            total_failed += 1
            continue

        output_path = output_folder / frame_path.name

        cv2.imwrite(str(output_path), face)

        total_faces += 1

print()
print("======================================")
print("FACE CROPPING COMPLETE")
print("======================================")
print(f"Frames processed : {total_processed}")
print(f"Faces extracted  : {total_faces}")
print(f"Frames skipped   : {total_failed}")
print(f"Output folder    : {OUTPUT_DIR}")