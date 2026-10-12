"""
painsense.live - the live scoring loop (webcam or video file).

    load_scorer   live model: checkpoints/pain_intensity_features.pt
                  (offline, ensemble-aware); falls back to the older
                  pain_intensity.pt with a frame-spacing warning
    LiveScorer    encode(face) once per new frame -> 512-d feature;
                  score(last SEQ_LEN features) -> 0-10
    run           camera/video loop: face crop every 0.25 s, feature
                  buffer, score, AlertState, SQLite log, dashboard state,
                  OpenCV overlay. Returns one record per sample.

If no face is found the buffer is cleared, nothing is scored and the
alert history resets. Used by live_camera.py and score_video.py.
"""

import json
import os
import time
from collections import deque

import cv2
import torch
from PIL import Image

from painsense import alerts, config, db, face, live_state
from painsense.data import frame_transform
from painsense.models import PainIntensityLSTM, load_feature_checkpoint


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
        backbone, head, payload = load_feature_checkpoint(config.FEATURE_CKPT)

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

        n_heads = len(payload.get("state_dicts") or [None])
        variant = payload.get("variant", "lstm")
        print(
            f"Loaded feature checkpoint: {config.FEATURE_CKPT} "
            f"({variant}, {n_heads} head(s))"
        )

        return LiveScorer(
            backbone,
            head,
            float(payload.get("scale", 1.0)),
            float(payload.get("bias", 0.0)),
            f"features:{variant} x{n_heads}"
        )

    print(
        f"WARNING: {config.FEATURE_CKPT} not found. Falling back to "
        f"{config.INTENSITY_CKPT}."
    )
    print(
        "WARNING: that model was trained on frames 1/24 s apart, but live "
        f"frames are {config.CAPTURE_INTERVAL_SEC:.2f} s apart, so its "
        "scores are not trustworthy. Create the feature checkpoint with: "
        "python cross_validate.py --epochs 30 --seeds 3 --final"
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
    live_state.write_state(
        score=None,
        probability=None,
        prediction=None,
        sequence_current=sequence_current,
        sequence_total=config.SEQ_LEN,
        alert=False,
        alert_reason=None
    )


def _open_source(camera_index, video_path):
    """Open a webcam or a video file. Returns (cap, is_file, fps)."""

    if video_path is not None:
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or float(config.VIDEO_FPS)
        return cap, True, float(fps)

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Try --camera 1 (or 2), close other apps using the camera, "
            "or check the OS camera privacy setting."
        )

    return cap, False, None


def run(
    camera_index=0,
    use_model=True,
    video_path=None,
    headless=False,
    log=True,
    verbose=True
):
    """
    Live loop.

    camera_index: webcam index (ignored when video_path is given).
    video_path:   replay a video file instead of a webcam. Frames are
                  sampled every round(fps * CAPTURE_INTERVAL_SEC) frames,
                  so spacing matches training regardless of CPU speed.
    headless:     no OpenCV window (stop a webcam run with Ctrl+C).
    log:          write the SQLite log and the dashboard state.

    Returns a list of per-sample records:
        {"frame": i, "t": seconds, "face": bool, "score": float|None,
         "median": float|None, "alert": bool}
    """

    if log:
        db.init_db()

    scorer = load_scorer() if use_model else None

    alert_state = alerts.AlertState()

    cap, is_file, fps = _open_source(camera_index, video_path)

    # A webcam uses the stricter live face size; a video file uses the
    # same detector settings as the stored training crops.
    min_face = (
        face.DETECT_MIN_SIZE if is_file else config.LIVE_MIN_FACE_SIZE
    )
    step = (
        max(1, round(fps * config.CAPTURE_INTERVAL_SEC)) if is_file else None
    )

    # Cached backbone features of the last SEQ_LEN valid faces.
    feature_buffer = deque(maxlen=config.SEQ_LEN)

    last_sample_t = 0.0
    frame_index = -1
    records = []

    last_prediction = None
    last_probability = None

    last_alert = False
    last_reason = None

    face_detected = False

    def clear_dashboard(sequence_current=0):
        if log:
            _clear_dashboard(sequence_current)

    clear_dashboard()

    if verbose:
        print()
        print("PainSense live inference running.")
        if is_file:
            print(f"Source: {video_path} ({fps:.1f} fps, every {step} frames)")
        print(
            f"Sampling every {config.CAPTURE_INTERVAL_SEC:.2f} s; "
            f"first score after {config.SEQ_LEN} valid faces "
            f"({config.SEQ_LEN * config.CAPTURE_INTERVAL_SEC:.1f} s)."
        )
        print(
            f"Alert: median of last {config.ALERT_MEDIAN_WINDOW} scores "
            f">= {SCORE_THRESHOLD:.1f}, off below "
            f"{config.ALERT_OFF_THRESHOLD:.1f} "
            f"(score range 0-{config.PAIN_SCORE_MAX:.0f}, "
            "synthetic morph scale)"
        )
        print("Press 'q' to quit." if not headless else "Ctrl+C to stop.")
        print()

    # =====================================================
    # CAMERA LOOP
    # =====================================================

    try:
        while True:

            ok, frame = cap.read()

            if not ok:
                if verbose:
                    print(
                        "End of video." if is_file
                        else "Camera read failed, stopping."
                    )
                break

            frame_index += 1

            if is_file:
                capturing = frame_index % step == 0
                now = frame_index / fps
            else:
                now = time.time()
                capturing = (
                    now - last_sample_t
                ) >= config.CAPTURE_INTERVAL_SEC

            if capturing:

                last_sample_t = now

                face_crop = face.crop_face(frame, min_size=min_face)
                record = {
                    "frame": frame_index,
                    "t": round(frame_index / fps, 3) if is_file else now,
                    "face": face_crop is not None,
                    "score": None,
                    "median": None,
                    "alert": False,
                }

                # =========================================
                # NO FACE DETECTED
                # =========================================

                if face_crop is None:

                    if face_detected and verbose:
                        print(
                            "[LIVE] No face detected -> buffer cleared "
                            "-> no inference -> alert reset"
                        )

                    face_detected = False

                    feature_buffer.clear()
                    alert_state.reset()

                    last_prediction = None
                    last_probability = None
                    last_alert = False
                    last_reason = None

                    clear_dashboard()

                # =========================================
                # FACE DETECTED
                # =========================================

                else:

                    face_detected = True

                    # ResNet runs ONCE per new frame; only the
                    # temporal head sees the whole window.
                    if scorer is not None:
                        feature_buffer.append(scorer.encode(face_crop))
                    else:
                        feature_buffer.append(None)

                    if len(feature_buffer) < config.SEQ_LEN:

                        last_prediction = None
                        last_probability = None
                        last_alert = False
                        last_reason = None

                        clear_dashboard(len(feature_buffer))

                    elif scorer is not None:

                        score = scorer.score(feature_buffer)

                        last_alert, last_reason = alert_state.update(
                            score,
                            face_detected=True
                        )

                        prediction = "PAIN" if last_alert else "NO PAIN"

                        last_prediction = prediction
                        last_probability = score

                        record.update(
                            score=score,
                            median=alert_state.median,
                            alert=last_alert
                        )

                        if log:
                            try:
                                db.log_score(
                                    score,
                                    last_alert,
                                    last_reason,
                                    prediction=prediction
                                )
                            except Exception as db_error:
                                print(f"[DATABASE] Warning: {db_error}")

                            live_state.write_state(
                                score=round(score, 2),
                                probability=score / float(
                                    config.PAIN_SCORE_MAX
                                ),
                                prediction=prediction,
                                sequence_current=config.SEQ_LEN,
                                sequence_total=config.SEQ_LEN,
                                alert=last_alert,
                                alert_reason=last_reason
                            )

                        if verbose:
                            print(
                                f"[LIVE] "
                                f"Score={score:.1f} "
                                f"Median={alert_state.median:.1f} "
                                f"Prediction={prediction} "
                                f"Alert={last_alert}"
                            )

                records.append(record)

            if headless:
                continue

            # =================================================
            # DRAW CAMERA WINDOW
            # =================================================

            frame = draw_overlay(
                frame,
                last_prediction,
                last_probability,
                face_detected,
                True,   # loop is running; indicator follows face_detected
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

    except KeyboardInterrupt:
        if verbose:
            print("Stopped by user.")

    # =====================================================
    # CLEANUP
    # =====================================================

    cap.release()

    if not headless:
        cv2.destroyAllWindows()

    clear_dashboard()

    if verbose:
        print()
        print("PainSense live inference stopped.")

    return records
