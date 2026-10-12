"""
live_camera.py - run the live demo on a webcam (or replay a video file).

Camera -> face crop every 0.25 s -> ResNet18 features -> LSTM -> score
(0-10, synthetic morph scale) -> alert (median of last 5 >= 6, off < 4)
-> SQLite log + dashboard state. Start dashboard.py in a second terminal
to watch it in the browser.

Run:
    python live_camera.py                       # webcam 0, OpenCV window
    python live_camera.py --camera 1            # another webcam
    python live_camera.py --video clip.mp4      # replay a file
    python live_camera.py --video clip.mp4 --headless

Press 'q' in the window to quit (Ctrl+C with --headless).
The loop itself lives in painsense/live.py.
"""

import argparse

from painsense import config
from painsense.live import run
from painsense.utils import seed_everything


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--camera", type=int, default=0,
                    help="webcam index (default 0)")
    ap.add_argument("--video", default=None,
                    help="replay a video file instead of the webcam")
    ap.add_argument("--headless", action="store_true",
                    help="no OpenCV window (Ctrl+C stops a webcam run)")
    ap.add_argument("--no-model", action="store_true",
                    help="camera and dashboard only, no scoring")
    args = ap.parse_args()

    seed_everything(config.SEED)
    print(f"Using device: {config.DEVICE}")

    run(
        camera_index=args.camera,
        use_model=not args.no_model,
        video_path=args.video,
        headless=args.headless
    )


if __name__ == "__main__":
    main()
