"""
PainSense - run a model on one video file.

Default (intensity, the model the camera uses):
    video -> same live loop as realtime_infer.py (headless, no logging)
          -> face crop every 0.25 s -> ResNet features -> LSTM
          -> 0-10 score (expression intensity, synthetic morph scale)
          -> AlertState
    If the file name contains _Pain_ / _NoPain_, the generated morph
    target is printed next to each score and an MAE is reported.
    NOTE: the final feature checkpoint was trained on all 40 SynPAIN
    videos, so on those files this is a pipeline check, not a test.

--binary: the older Pain / NoPain CNN-LSTM classifier (cnn_lstm.pt).

Usage:
    python test_video.py "path_to_video.mp4"
    python test_video.py "path_to_video.mp4" --binary
"""

import argparse
import sys
from pathlib import Path
from collections import deque

import cv2
import torch
from PIL import Image

import config
from data_prep import crop_face
from dataset import frame_transform
from utils import seed_everything
from model import PainCNNLSTM


# Face detection + crop live in data_prep (shared with training crops).


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    print(
        f"Using device: {config.DEVICE}"
    )

    model = PainCNNLSTM().to(
        config.DEVICE
    )

    checkpoint = Path(
        config.CNNLSTM_CKPT
    )

    if not checkpoint.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found:\n"
            f"{checkpoint}"
        )

    state = torch.load(
        checkpoint,
        map_location=config.DEVICE
    )

    model.load_state_dict(
        state
    )

    model.eval()

    print(
        f"Loaded trained model:\n"
        f"{checkpoint}"
    )

    return model


# =========================================================
# TEST VIDEO
# =========================================================

def test_video(video_path):

    video_path = Path(
        video_path
    )

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found:\n"
            f"{video_path}"
        )

    model = load_model()

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print()
    print(
        "======================================"
    )
    print(
        "SYNPAIN VIDEO TEST"
    )
    print(
        "======================================"
    )

    print(
        f"Video: {video_path.name}"
    )

    print(
        f"Total video frames: {total_frames}"
    )

    print(
        f"Video FPS: {fps:.2f}"
    )

    # -----------------------------------------------------
    # Determine expected label from filename.
    # -----------------------------------------------------

    name_lower = (
        video_path.name.lower()
    )

    if "_pain_" in name_lower:

        expected_label = "PAIN"

    elif "_nopain_" in name_lower:

        expected_label = "NO PAIN"

    else:

        expected_label = "UNKNOWN"

    print(
        f"Expected dataset label: "
        f"{expected_label}"
    )

    # -----------------------------------------------------
    # 8-frame rolling buffer.
    # -----------------------------------------------------

    frame_buffer = deque(
        maxlen=config.SEQ_LEN
    )

    pain_probabilities = []

    processed_frames = 0
    skipped_frames = 0
    sequences_tested = 0

    # -----------------------------------------------------
    # Read every real video frame.
    # -----------------------------------------------------

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        processed_frames += 1

        # Detect/crop face.
        face = crop_face(
            frame
        )

        if face is None:

            skipped_frames += 1
            frame_buffer.clear()

            continue

        # Add real consecutive face frame.
        frame_buffer.append(
            face
        )

        # -------------------------------------------------
        # Once we have 8 real consecutive frames,
        # run the CNN-LSTM.
        # -------------------------------------------------

        if len(frame_buffer) == config.SEQ_LEN:

            tensors = []

            for f in frame_buffer:

                rgb = cv2.cvtColor(
                    f,
                    cv2.COLOR_BGR2RGB
                )

                image = Image.fromarray(
                    rgb
                )

                tensor = frame_transform(
                    image
                )

                tensors.append(
                    tensor
                )

            # (T,C,H,W)
            sequence = torch.stack(
                tensors
            )

            # (1,T,C,H,W)
            sequence = (
                sequence
                .unsqueeze(0)
                .to(config.DEVICE)
            )

            with torch.no_grad():

                logits = model(
                    sequence
                )

                # Final timestep.
                final_logits = logits[
                    0,
                    -1
                ]

                probabilities = torch.softmax(
                    final_logits,
                    dim=0
                )

                pain_probability = float(
                    probabilities[1].item()
                )

                predicted_class = int(
                    probabilities.argmax().item()
                )

            pain_probabilities.append(
                pain_probability
            )

            sequences_tested += 1

            if sequences_tested % 10 == 0:

                prediction = (
                    "PAIN"
                    if predicted_class == 1
                    else
                    "NO PAIN"
                )

                print(
                    f"Sequence {sequences_tested:4d} | "
                    f"Prediction: {prediction:7s} | "
                    f"Pain probability: "
                    f"{pain_probability * 100:6.2f}%"
                )

    cap.release()

    # =====================================================
    # FINAL RESULT
    # =====================================================

    print()
    print(
        "======================================"
    )
    print(
        "TEST COMPLETE"
    )
    print(
        "======================================"
    )

    print(
        f"Frames processed: {processed_frames}"
    )

    print(
        f"Frames without face: {skipped_frames}"
    )

    print(
        f"8-frame sequences tested: "
        f"{sequences_tested}"
    )

    if not pain_probabilities:

        print()
        print(
            "ERROR: No valid face sequences "
            "were created."
        )

        return

    # Average probability over the complete video.
    average_probability = (
        sum(pain_probabilities)
        / len(pain_probabilities)
    )

    # Percentage of sequences predicted as Pain.
    pain_sequence_count = sum(
        p >= 0.5
        for p in pain_probabilities
    )

    pain_percentage = (
        pain_sequence_count
        / len(pain_probabilities)
        * 100
    )

    final_prediction = (
        "PAIN"
        if average_probability >= 0.5
        else
        "NO PAIN"
    )

    print()
    print(
        f"Average pain probability: "
        f"{average_probability * 100:.2f}%"
    )

    print(
        f"Sequences predicted as Pain: "
        f"{pain_sequence_count}/"
        f"{len(pain_probabilities)} "
        f"({pain_percentage:.2f}%)"
    )

    print()
    print(
        f"FINAL MODEL PREDICTION: "
        f"{final_prediction}"
    )

    print(
        f"DATASET EXPECTED LABEL: "
        f"{expected_label}"
    )

    print(
        "======================================" 
    )


# =========================================================
# MAIN
# =========================================================

# =========================================================
# INTENSITY MODEL ON A VIDEO
# =========================================================

def test_video_intensity(video_path):

    import realtime_infer
    from dataset import frame_intensity

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found:\n{video_path}")

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    name = video_path.name.lower()
    label = 1 if "_pain_" in name else 0 if "_nopain_" in name else None

    records = realtime_infer.run(
        video_path=video_path,
        headless=True,
        log=False,
        verbose=False
    )

    print()
    print("======================================")
    print("VIDEO TEST - expression intensity (synthetic morph scale)")
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
        print("No score: fewer than 8 consecutive face samples.")
        return

    print(f"Final score: {scored[-1]['score']:.2f}   "
          f"max score: {max(r['score'] for r in scored):.2f}")
    print(f"Alert ever on: {any(r['alert'] for r in scored)}")

    if errors:
        print(f"MAE vs morph target: {sum(errors) / len(errors):.3f}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument(
        "--binary",
        action="store_true",
        help="test the older Pain / NoPain classifier instead"
    )
    args = ap.parse_args()

    seed_everything(config.SEED)

    if args.binary:
        test_video(args.video)
    else:
        test_video_intensity(args.video)
