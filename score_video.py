"""
score_video.py - run the live model on one video file and print a timeline.

Uses exactly the live pipeline (painsense.live.run, headless, no logging):
face crop every 0.25 s -> ResNet18 features -> LSTM -> 0-10 score ->
AlertState. If the file name contains _Pain_ or _NoPain_, the generated
morph target is printed next to every score and an MAE is reported.

Use it to check the pipeline without a webcam. The live model is trained
on all 40 SynPAIN clips, so on those files this is a pipeline check,
not a test score (use cross_validate.py for honest numbers).

Run:
    python score_video.py data/videos/raw/SynPain/Videos/1100031270_Pain_man_Young.mp4
"""

import argparse
from pathlib import Path

import cv2

from painsense import config
from painsense.data import frame_intensity
from painsense.live import run
from painsense.utils import seed_everything


def score_video(video_path):
    """Print the score timeline for one video. Returns the records."""

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found:\n{video_path}")

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    name = video_path.name.lower()
    label = 1 if "_pain_" in name else 0 if "_nopain_" in name else None

    records = run(
        video_path=video_path,
        headless=True,
        log=False,
        verbose=False
    )

    print()
    print("======================================")
    print("VIDEO SCORE - expression intensity (synthetic morph scale)")
    print("======================================")
    print(f"Video: {video_path.name} ({total} frames)")
    print(
        "Expected: "
        + ("Pain morph (target 0 -> 10)" if label == 1
           else "NoPain (target 0)" if label == 0 else "unlabeled")
    )
    print()
    print(f"{'t (s)':>7} {'frame':>6} {'score':>6} {'median':>7} "
          f"{'target':>7} alert")

    errors = []

    for r in records:
        if not r["face"]:
            print(f"{r['t']:7.2f} {r['frame']:6d}   no face")
            continue

        if r["score"] is None:
            continue

        target = None

        if label is not None:
            target = frame_intensity(label, r["frame"], 0, max(total - 1, 1))
            errors.append(abs(r["score"] - target))

        print(
            f"{r['t']:7.2f} {r['frame']:6d} {r['score']:6.2f} "
            f"{r['median']:7.2f} "
            + (f"{target:7.2f}" if target is not None else "      -")
            + ("  ALERT" if r["alert"] else "")
        )

    scored = [r for r in records if r["score"] is not None]
    faces = sum(r["face"] for r in records)

    print()
    print(f"Samples: {len(records)}, with face: {faces}, scored: {len(scored)}")

    if not scored:
        print(f"No score: fewer than {config.SEQ_LEN} consecutive face samples.")
        return records

    print(f"Final score: {scored[-1]['score']:.2f}   "
          f"max score: {max(r['score'] for r in scored):.2f}")
    print(f"Alert ever on: {any(r['alert'] for r in scored)}")

    if errors:
        print(f"MAE vs morph target: {sum(errors) / len(errors):.3f}")

    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("video", help="path to an .mp4 (or any OpenCV video)")
    args = ap.parse_args()

    seed_everything(config.SEED)
    score_video(args.video)


if __name__ == "__main__":
    main()
