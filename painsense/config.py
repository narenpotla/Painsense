"""
painsense.config - every path and setting in one place.

All scripts and modules import their settings from here, so a change
(paths, window length, alert thresholds, ...) is made once.

The score everywhere is "expression intensity on a synthetic morph
scale" (0-10): 0 = start of a SynPAIN neutral-to-pain morph (or any
NoPain clip), 10 = end of a Pain morph. It is NOT PSPI or VAS pain.
"""

import os

import torch

# =========================================================
# Paths
# =========================================================

# Project root = the folder that contains the painsense/ package.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SynPAIN source videos (.mp4) and the frames / face crops made from them.
RAW_VIDEO_DIR = os.path.join(ROOT, "data", "videos", "raw", "SynPain", "Videos")
VIDEO_FRAME_DIR = os.path.join(ROOT, "data", "videos", "frames")
VIDEO_FACE_DIR = os.path.join(ROOT, "data", "videos", "face_frames")

# Generated 0-10 target per labeled face frame (written by train_full_model.py).
INTENSITY_DIR = os.path.join(ROOT, "data", "intensity")
INTENSITY_MANIFEST = os.path.join(INTENSITY_DIR, "manifest.csv")

# Cached ResNet18 features (cross_validate.py / robustness_study.py).
FEATURE_DIR = os.path.join(ROOT, "data", "features")

CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
# Live model: frozen ImageNet ResNet18 features -> temporal head(s).
# Written by cross_validate.py --final. Used by the camera.
FEATURE_CKPT = os.path.join(CHECKPOINT_DIR, "pain_intensity_features.pt")
# Older full CNN-LSTM (train_full_model.py). Camera fallback only.
INTENSITY_CKPT = os.path.join(CHECKPOINT_DIR, "pain_intensity.pt")

RESULTS_DIR = os.path.join(ROOT, "results")
DB_PATH = os.path.join(ROOT, "logs", "painsense.db")
LIVE_STATE_PATH = os.path.join(ROOT, "logs", "dashboard_state.json")


def calibration_path_for(ckpt_path):
    """Calibration JSON that belongs to a full-model checkpoint file."""
    if os.path.abspath(ckpt_path) == os.path.abspath(INTENSITY_CKPT):
        return os.path.join(CHECKPOINT_DIR, "pain_intensity_calibration.json")
    stem, _ = os.path.splitext(ckpt_path)
    return stem + "_calibration.json"


# =========================================================
# Images, windows, model sizes
# =========================================================

CROP_SIZE = 128           # stored face crop in data/videos/face_frames
IMG_SIZE = 224            # model input (frame_transform upsamples crops)
FEATURE_DIM = 512         # ResNet18 pooled feature size
EMBED_DIM = 128           # projection fed into the temporal head
LSTM_HIDDEN = 64

# The SynPAIN videos are 24 fps. A window is SEQ_LEN frames taken
# SEQ_STRIDE video frames apart (6 frames = 0.25 s); the live camera
# is sampled at the same spacing.
VIDEO_FPS = 24
SEQ_LEN = 8
SEQ_STRIDE = 6
CAPTURE_INTERVAL_SEC = SEQ_STRIDE / VIDEO_FPS

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Training
# =========================================================

# Video-level split for train_full_model.py (stratified Pain/NoPain).
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Head learning rate; ResNet layer4 (opt-in fine-tuning) is gentler.
LR_HEAD = 1e-3
LR_LAYER4 = 1e-4

# Webcam-style feature variants per frame: N_TRAIN_AUG for training
# augmentation, N_HELDOUT_AUG from a different (harsher) family for the
# "simulated webcam" test only. See painsense.data.apply_webcam.
N_TRAIN_AUG = 4
N_HELDOUT_AUG = 2

# Settings of the live model, chosen by robustness_study.py
# (results/robustness_study.json). cross_validate.py --final uses them.
FINAL_VARIANT = "lstm"      # "frame", "pool" or "lstm"
FINAL_AUG = False           # train with webcam-style variants
FINAL_STRIDES = (6,)        # training window spacings (frames)
FINAL_ENSEMBLE = 1          # heads averaged in the final model

# =========================================================
# Score and alert
# =========================================================

PAIN_SCORE_MIN = 0.0
PAIN_SCORE_MAX = 10.0

# Alert ON when the median of the last ALERT_MEDIAN_WINDOW scores
# reaches ALERT_THRESHOLD, OFF when it falls below ALERT_OFF_THRESHOLD.
ALERT_THRESHOLD = 6.0
ALERT_OFF_THRESHOLD = 4.0
ALERT_MEDIAN_WINDOW = 5

# =========================================================
# Live demo, log, dashboard
# =========================================================

# Smallest webcam face (pixels). Larger than the 40 px used for the
# SynPAIN crops, to cut Haar false positives on background clutter.
LIVE_MIN_FACE_SIZE = (60, 60)

DB_MAX_ROWS = 100_000          # newest rows kept in the SQLite log
DASHBOARD_HISTORY = 120        # points in the trend chart (30 s)
DASHBOARD_STALE_SEC = 3.0      # dashboard shows OFFLINE after this
