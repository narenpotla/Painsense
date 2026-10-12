"""
PainSense - central configuration.
Edit paths here once; every other script imports from this file.
"""
import os
import torch

# ---------------- Paths ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))

RAW_DATASET_DIR = os.path.join(ROOT, "data", "raw", "SynPain", "Images")   # SynPAIN .jpg files go here
FRAMES_DIR = os.path.join(ROOT, "data", "frames")
SEQ_DIR = os.path.join(ROOT, "data", "sequences")

# Real SynPAIN video face frames
VIDEO_FACE_DIR = os.path.join(ROOT, "data", "videos", "face_frames")

# Generated 0-10 labels for those videos: "expression intensity on a
# synthetic morph scale". Pain videos are neutral-to-pain morphs, so the
# target is linear in frame index (0 -> 10). NoPain videos stay at 0.
# This is NOT real pain intensity (not PSPI, not VAS).
INTENSITY_DIR = os.path.join(ROOT, "data", "intensity")
INTENSITY_MANIFEST = os.path.join(INTENSITY_DIR, "manifest.csv")

CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
CNN_CKPT = os.path.join(CHECKPOINT_DIR, "cnn_encoder.pt")
CNNLSTM_CKPT = os.path.join(CHECKPOINT_DIR, "cnn_lstm.pt")
INTENSITY_CKPT = os.path.join(CHECKPOINT_DIR, "pain_intensity.pt")
# Feature-based model: frozen ImageNet ResNet18 features -> LSTM head.
# Saved by cv_intensity.py --final.
FEATURE_CKPT = os.path.join(CHECKPOINT_DIR, "pain_intensity_features.pt")
FEATURE_DIR = os.path.join(ROOT, "data", "features")
RESULTS_DIR = os.path.join(ROOT, "results")


def calibration_path_for(ckpt_path):
    """Calibration JSON that belongs to a checkpoint file."""
    if os.path.abspath(ckpt_path) == os.path.abspath(INTENSITY_CKPT):
        return os.path.join(CHECKPOINT_DIR, "pain_intensity_calibration.json")
    stem, _ = os.path.splitext(ckpt_path)
    return stem + "_calibration.json"


DB_PATH = os.path.join(ROOT, "logs", "painsense.db")

# ---------------- Image / model ----------------
IMG_SIZE = 224            # model input size (frame_transform resizes crops to this)
CROP_SIZE = 128           # stored face-crop size in data/videos/face_frames
EMBED_DIM = 128           # CNN embedding dimension fed into LSTM
LSTM_HIDDEN = 64
SEQ_LEN = 8               # frames per temporal clip (both training windows + live inference buffer)
FEATURE_DIM = 512         # ResNet18 pooled feature size

# The training videos are 24 fps. Training windows take SEQ_LEN frames
# SEQ_STRIDE video frames apart (6 frames = 0.25 s), and live inference
# samples the camera at the same spacing.
VIDEO_FPS = 24
SEQ_STRIDE = 6
SEED = 42

# ---------------- Training ----------------
BATCH_SIZE = 32
CNN_EPOCHS = 6
LSTM_EPOCHS = 10
LR = 1e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15         # intensity stage: video-level train/val/test split

# Intensity stage learning rates (opt-in layer4 fine-tuning is gentler).
LR_HEAD = 1e-3
LR_LAYER4 = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Intensity / Alerting ----------------
# 0 = start of a neutral-to-expression transition (or any NoPain video)
# 10 = end of a Pain video transition
# "Expression intensity on a synthetic morph scale": generated from SynPAIN
# morph progress, not PSPI or VAS.

PAIN_SCORE_MAX = 10.0
PAIN_SCORE_MIN = 0.0

# Alert turns ON when the median of the last ALERT_MEDIAN_WINDOW scores
# reaches ALERT_THRESHOLD and turns OFF when it falls below
# ALERT_OFF_THRESHOLD (hysteresis). See alert_rules.AlertState.
ALERT_THRESHOLD = 6.0
ALERT_OFF_THRESHOLD = 4.0
ALERT_MEDIAN_WINDOW = 5

# Kept so older binary scripts still import.
PAIN_PROBABILITY_THRESHOLD = 0.50

# Number of consecutive valid face frames required
# before CNN-LSTM inference.
MIN_VALID_SEQUENCE = SEQ_LEN


# Seconds between frames sampled into the sequence buffer during live
# inference. Derived from the training stride so live spacing = training spacing.
CAPTURE_INTERVAL_SEC = SEQ_STRIDE / VIDEO_FPS

# ---------------- Feature model (cv_intensity.py / live demo) ----------------
# Webcam-style feature-cache variants per frame: used for training
# augmentation, and a separate held-out family for the "simulated
# webcam" test (see dataset.sample_webcam_params).
N_TRAIN_AUG = 4
N_HELDOUT_AUG = 2

# Settings of the final live model. Chosen by cv_robustness.py
# (results/cv_robustness.json); cv_intensity.py --final uses them.
FINAL_VARIANT = "lstm"
FINAL_AUG = False
FINAL_STRIDES = (6,)        # training window spacings (frames)
FINAL_ENSEMBLE = 1          # heads averaged in the final model

# Smallest face (pixels) accepted from a live webcam. Larger than the
# 40 px used for the SynPAIN crops to cut Haar false positives on
# background clutter. The crop recipe itself is unchanged.
LIVE_MIN_FACE_SIZE = (60, 60)

# ---------------- Logging / dashboard ----------------
# The live loop logs ~4 scores per second; keep the newest rows only.
DB_MAX_ROWS = 100_000
# Points shown in the dashboard trend chart (120 = 30 s at 0.25 s).
DASHBOARD_HISTORY = 120
# Dashboard shows OFFLINE when the live state is older than this.
DASHBOARD_STALE_SEC = 3.0
