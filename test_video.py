"""
PainSense - test CNN-LSTM on a real SynPAIN video.

This script:
    Real SynPAIN .mp4
        -> real video frames
        -> face detection
        -> face crop
        -> 8 consecutive frames
        -> trained CNN-LSTM
        -> Pain / NoPain probability

Usage:
    python test_video.py "path_to_video.mp4"
"""

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

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            'python test_video.py '
            '"path_to_video.mp4"'
        )

        sys.exit(1)

    seed_everything(config.SEED)

    test_video(
        sys.argv[1]
    )