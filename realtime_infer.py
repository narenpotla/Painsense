"""
PainSense - real-time inference + web dashboard integration.

CURRENT MODEL OUTPUT:
    Expression intensity from 0 to 10, on a synthetic morph scale.

The score estimates progress along the generated SynPAIN
neutral-to-pain morph. It is not a clinical PSPI or VAS score.

Pipeline:

    Camera
       |
    Face detection + crop (same recipe as the training crops)
       |
    ResNet18 features, computed ONCE per new frame and cached
       |
    LSTM over the last SEQ_LEN cached features
       |
    Score 0-10
       |
    Alert rule (median of last 5 scores + hysteresis)
       |
    SQLite log + dashboard state

Frames are sampled every CAPTURE_INTERVAL_SEC (= SEQ_STRIDE / 24 s),
the same spacing the training windows use.

If no face is detected:
    - nothing is added to the feature buffer and the buffer is cleared
    - the LSTM is NOT executed
    - prediction, score and alert are cleared
    - the alert state (median history, hysteresis) is reset

Model choice:
    checkpoints/pain_intensity_features.pt   (preferred; made by
        cv_intensity.py --final, trained at the live frame spacing)
    checkpoints/pain_intensity.pt            (fallback; trained at 1/24 s
        frame spacing, so live scores are not trustworthy)

Run:
    python realtime_infer.py

Press 'q' to quit.
"""

import argparse
import json
import os
import time
from collections import deque

import cv2
import torch
from PIL import Image

import config
import db
import alert_rules
import dashboard_state

from data_prep import crop_face
from dataset import frame_transform
from model import (
    FeatureIntensityModel,
    PainIntensityLSTM,
    imagenet_backbone
)
from utils import seed_everything


# =========================================================
# CONFIGURATION
# =========================================================

SCORE_THRESHOLD = config.ALERT_THRESHOLD


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

def load_calibration():
    path = config.calibration_path_for(config.INTENSITY_CKPT)

    if not os.path.exists(path):
        return 1.0, 0.0

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return (
        float(payload.get("scale", 1.0)),
        float(payload.get("bias", 0.0))
    )


def load_model():
    """Load the old full CNN-LSTM checkpoint (fallback model)."""

    model = PainIntensityLSTM().to(config.DEVICE)

    try:

        state = torch.load(
            config.INTENSITY_CKPT,
            map_location=config.DEVICE
        )

        model.load_state_dict(state)

        print(
            f"Loaded trained checkpoint: "
            f"{config.INTENSITY_CKPT}"
        )

    except FileNotFoundError:

        print(
            "WARNING: No trained CNN-LSTM checkpoint found."
        )

        print(
            "The model will be untrained."
        )

    model.eval()

    return model


class LiveScorer:
    """
    Two-step scorer: encode(face) once per new frame, then
    score(list_of_features) runs only the temporal head.
    """

    def __init__(self, encode_fn, score_fn, scale, bias, description):
        self._encode = encode_fn
        self._score = score_fn
        self.scale = scale
        self.bias = bias
        self.description = description

    @torch.no_grad()
    def encode(self, face_bgr):
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        x = frame_transform(Image.fromarray(rgb)).unsqueeze(0)
        return self._encode(x.to(config.DEVICE))[0].cpu()

    @torch.no_grad()
    def score(self, features):
        """features: list of SEQ_LEN tensors (512,), oldest first."""

        window = torch.stack(list(features)).unsqueeze(0).to(config.DEVICE)
        raw = float(self._score(window)[0, -1].item())
        value = self.scale * raw + self.bias

        return max(
            float(config.PAIN_SCORE_MIN),
            min(float(config.PAIN_SCORE_MAX), value)
        )


def load_scorer():
    """
    Prefer the feature checkpoint trained at the live frame spacing.
    Fall back to the old checkpoint with a warning.
    """

    if os.path.exists(config.FEATURE_CKPT):
        payload = torch.load(
            config.FEATURE_CKPT,
            map_location=config.DEVICE
        )

        if payload.get("seq_len") != config.SEQ_LEN:
            raise RuntimeError(
                f"{config.FEATURE_CKPT} was trained with seq_len="
                f"{payload.get('seq_len')} but config.SEQ_LEN="
                f"{config.SEQ_LEN}"
            )

        if payload.get("stride") != config.SEQ_STRIDE:
            print(
                f"WARNING: feature checkpoint stride "
                f"{payload.get('stride')} != config.SEQ_STRIDE "
                f"{config.SEQ_STRIDE}; live spacing differs from training."
            )

        backbone = imagenet_backbone().to(config.DEVICE)
        head = FeatureIntensityModel(
            payload.get("variant", "lstm")
        ).to(config.DEVICE)
        head.load_state_dict(payload["state_dict"])
        head.eval()

        print(f"Loaded feature checkpoint: {config.FEATURE_CKPT}")

        return LiveScorer(
            backbone,
            head,
            float(payload.get("scale", 1.0)),
            float(payload.get("bias", 0.0)),
            "features:" + payload.get("variant", "lstm")
        )

    print(
        f"WARNING: {config.FEATURE_CKPT} not found. Falling back to "
        f"{config.INTENSITY_CKPT}."
    )
    print(
        "WARNING: that model was trained on frames 1/24 s apart, but live "
        f"frames are {config.CAPTURE_INTERVAL_SEC:.2f} s apart, so its "
        "scores are not trustworthy. Create the feature checkpoint with: "
        "python cv_intensity.py --epochs 30 --seeds 3 --final"
    )

    model = load_model()
    scale, bias = load_calibration()

    return LiveScorer(
        model.encode_backbone,
        model.score_features,
        scale,
        bias,
        "full model (legacy)"
    )


# =========================================================
# DRAW CAMERA OVERLAY
# =========================================================

def draw_overlay(
    frame,
    prediction,
    probability,
    face_detected,
    capturing,
    alert,
    alert_reason,
    buffer_fill,
    buffer_target
):

    h, w = frame.shape[:2]

    # -----------------------------------------------------
    # TOP INFORMATION PANEL
    # -----------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (w, 125),
        (30, 30, 30),
        -1
    )

    # -----------------------------------------------------
    # CAPTURE INDICATOR
    # -----------------------------------------------------

    dot_color = (
        (0, 200, 0)
        if capturing and face_detected
        else (100, 100, 100)
    )

    cv2.circle(
        frame,
        (25, 25),
        10,
        dot_color,
        -1
    )

    cv2.putText(
        frame,
        "CAPTURE ACTIVE"
        if capturing and face_detected
        else "IDLE",
        (45, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    # -----------------------------------------------------
    # NO FACE DETECTED
    # -----------------------------------------------------

    if not face_detected:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (25, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2
        )

        cv2.putText(
            frame,
            "Waiting for a valid face...",
            (25, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1
        )

    # -----------------------------------------------------
    # FACE DETECTED BUT BUFFERING
    # -----------------------------------------------------

    elif prediction is None:

        cv2.putText(
            frame,
            f"Buffering valid face frames: "
            f"{buffer_fill}/{buffer_target}",
            (25, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Waiting for LSTM inference...",
            (25, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1
        )

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    else:

        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (25, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        probability_text = (
            f"Intensity (synthetic scale): "
            f"{probability:.1f} / {config.PAIN_SCORE_MAX:.0f}"
        )

        cv2.putText(
            frame,
            probability_text,
            (25, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    # -----------------------------------------------------
    # ALERT
    # -----------------------------------------------------

    if alert:

        cv2.rectangle(
            frame,
            (0, h - 45),
            (w, h),
            (0, 0, 200),
            -1
        )

        cv2.putText(
            frame,
            f"ALERT: {alert_reason}",
            (15, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return frame


# =========================================================
# MAIN INFERENCE
# =========================================================

def _clear_dashboard(sequence_current=0):
    dashboard_state.write_state(
        score=None,
        probability=None,
        prediction=None,
        sequence_current=sequence_current,
        sequence_total=config.SEQ_LEN,
        alert=False,
        alert_reason=None
    )


def run(
    camera_index=0,
    use_model=True
):

    # -----------------------------------------------------
    # INITIALIZE DATABASE
    # -----------------------------------------------------

    db.init_db()

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    scorer = load_scorer() if use_model else None

    alert_state = alert_rules.AlertState()

    # -----------------------------------------------------
    # OPEN WEBCAM
    # -----------------------------------------------------

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open camera "
            f"index {camera_index}"
        )

    # Cached backbone features of the last SEQ_LEN valid faces.
    feature_buffer = deque(maxlen=config.SEQ_LEN)

    last_sample_t = 0.0

    last_prediction = None
    last_probability = None

    last_alert = False
    last_reason = None

    face_detected = False

    _clear_dashboard()

    print()
    print("PainSense live inference running.")
    print("Dashboard state integration active.")
    print(
        f"Sampling every {config.CAPTURE_INTERVAL_SEC:.2f} s; "
        f"first score after {config.SEQ_LEN} valid faces "
        f"({config.SEQ_LEN * config.CAPTURE_INTERVAL_SEC:.1f} s)."
    )
    print(
        f"Alert: median of last {config.ALERT_MEDIAN_WINDOW} scores "
        f">= {SCORE_THRESHOLD:.1f}, off below "
        f"{config.ALERT_OFF_THRESHOLD:.1f} "
        f"(score range 0-{config.PAIN_SCORE_MAX:.0f}, synthetic morph scale)"
    )
    print("Press 'q' to quit.")
    print()

    # =====================================================
    # CAMERA LOOP
    # =====================================================

    while True:

        ok, frame = cap.read()

        if not ok:

            print("Camera read failed, stopping.")

            break

        now = time.time()

        capturing = (
            now - last_sample_t
        ) >= config.CAPTURE_INTERVAL_SEC

        if capturing:

            last_sample_t = now

            face = crop_face(frame)

            # =============================================
            # NO FACE DETECTED
            # =============================================

            if face is None:

                face_detected = False

                feature_buffer.clear()
                alert_state.reset()

                last_prediction = None
                last_probability = None
                last_alert = False
                last_reason = None

                _clear_dashboard()

                print(
                    "[DASHBOARD] No face detected -> buffer cleared "
                    "-> no inference -> alert reset"
                )

            # =============================================
            # FACE DETECTED
            # =============================================

            else:

                face_detected = True

                # ResNet runs ONCE per new frame; only the
                # temporal head sees the whole window.
                if scorer is not None:
                    feature_buffer.append(scorer.encode(face))
                else:
                    feature_buffer.append(None)

                if len(feature_buffer) < config.SEQ_LEN:

                    last_prediction = None
                    last_probability = None
                    last_alert = False
                    last_reason = None

                    _clear_dashboard(len(feature_buffer))

                elif scorer is not None:

                    score = scorer.score(feature_buffer)

                    last_alert, last_reason = alert_state.update(
                        score,
                        face_detected=True
                    )

                    prediction = "PAIN" if last_alert else "NO PAIN"

                    last_prediction = prediction
                    last_probability = score

                    try:

                        db.log_score(
                            score,
                            last_alert,
                            last_reason,
                            prediction=prediction
                        )

                    except Exception as db_error:

                        print(f"[DATABASE] Warning: {db_error}")

                    dashboard_state.write_state(
                        score=round(score, 2),
                        probability=score / float(config.PAIN_SCORE_MAX),
                        prediction=prediction,
                        sequence_current=config.SEQ_LEN,
                        sequence_total=config.SEQ_LEN,
                        alert=last_alert,
                        alert_reason=last_reason
                    )

                    print(
                        f"[DASHBOARD] "
                        f"Score={score:.1f} "
                        f"Median={alert_state.median:.1f} "
                        f"Prediction={prediction} "
                        f"Alert={last_alert}"
                    )

        # =================================================
        # DRAW CAMERA WINDOW
        # =================================================

        frame = draw_overlay(
            frame,
            last_prediction,
            last_probability,
            face_detected,
            capturing,
            last_alert,
            last_reason,
            len(feature_buffer),
            config.SEQ_LEN
        )

        cv2.imshow(
            "PainSense - live inference",
            frame
        )

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            break

    # =====================================================
    # CLEANUP
    # =====================================================

    cap.release()

    cv2.destroyAllWindows()

    _clear_dashboard()

    print()
    print("PainSense live inference stopped.")


# =========================================================
# PROGRAM ENTRY
# =========================================================

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--camera",
        type=int,
        default=0
    )

    ap.add_argument(
        "--no-model",
        action="store_true",
        help=(
            "Run camera and dashboard "
            "without CNN-LSTM inference"
        )
    )

    args = ap.parse_args()

    seed_everything(config.SEED)

    print(f"Using device: {config.DEVICE}")

    run(
        camera_index=args.camera,
        use_model=not args.no_model
    )
