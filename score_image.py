"""
score_image.py - score the face in one or more photos.

For each photo: find the largest face, crop it like the live demo and
score it with the single-photo model (checkpoints/pain_intensity_image.pt,
built by cross_validate.py --final). Prints the 0-10 score (expression
intensity on a synthetic morph scale, not pain) and the face box.
The same scoring is available in the dashboard ("Score a photo").

Run:
    python score_image.py photo.jpg
    python score_image.py a.jpg b.png --save-crops crops/
"""

import argparse
import base64
import os

from painsense import config
from painsense.photo import score_photo
from painsense.utils import seed_everything


def main():
    ap = argparse.ArgumentParser(
        description="Score the face in one or more photos."
    )
    ap.add_argument("images", nargs="+", help="image files (jpg, png, ...)")
    ap.add_argument("--save-crops", default=None,
                    help="folder to save the 128x128 face crop the model saw")
    args = ap.parse_args()

    seed_everything(config.SEED)

    if args.save_crops:
        os.makedirs(args.save_crops, exist_ok=True)

    for path in args.images:
        try:
            result = score_photo(path)
        except (ValueError, OSError) as error:
            print(f"{path}: ERROR {error}")
            continue

        if not result["face_found"]:
            print(f"{path}: no face found")
            continue

        x, y, w, h = result["box"]
        print(
            f"{path}: score {result['score']:.2f} / 10"
            f"{'  (above alert line)' if result['above_alert'] else ''}"
            f"   face {w}x{h} at ({x},{y})   model {result['model']}"
        )

        if args.save_crops and result["crop_png"]:
            name = os.path.splitext(os.path.basename(path))[0] + "_face.png"
            with open(os.path.join(args.save_crops, name), "wb") as handle:
                handle.write(base64.b64decode(result["crop_png"]))

    print(
        "\nScores are expression intensity on a synthetic morph scale "
        "(0-10), not a pain measurement."
    )


if __name__ == "__main__":
    main()
