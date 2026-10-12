# PainSense

Research prototype that estimates a facial **expression intensity** score (0–10) from a webcam, raises an alert when it stays high, logs every score, and shows it on a web dashboard.

```
Camera → face crop every 0.25 s → ResNet18 features → LSTM → score 0–10 → alert → SQLite log → dashboard
```

## What the score means (read this first)

The score is **expression intensity on a synthetic morph scale**. It is **not** a clinical pain measurement (not VAS, not PSPI).

The model is trained on 40 SynPAIN clips (synthetic faces, 24 fps, about 122 frames each):

- **Pain** clips morph from a mildly tense face to a pain face. The target rises linearly with frame index: 0 at the first frame, 10 at the last.
- **NoPain** clips are 0 on every frame.

So the model learns "how far along that morph is this face", not how much pain a person feels. A real webcam face differs from these synthetic faces, so live scores can be lower or noisier. A real pain model would need real labels (e.g. UNBC-McMaster PSPI), which this project does not have.

## Project structure

```
PainSense/
├── live_camera.py          Run the live demo (webcam or video file)
├── dashboard.py            Web dashboard for the live demo
├── score_video.py          Run the live model on one video, print a timeline
├── cross_validate.py       Honest accuracy (video-wise CV) + builds the live model
├── robustness_study.py     Which training settings hold up on webcam-like input
├── evaluate.py             Score a checkpoint on held-out videos
├── train_full_model.py     Optional: train the full CNN-LSTM on images (slow)
├── extract_frames.py       Data prep 1: videos → frames
├── crop_faces.py           Data prep 2: frames → 128×128 face crops
│
├── painsense/              Library code (imported by the scripts above)
│   ├── config.py           Every path and setting
│   ├── face.py             Face detection + the single crop recipe
│   ├── data.py             Labels, 0–10 targets, windows, augmentation, splits
│   ├── models.py           Networks + checkpoint loading
│   ├── features.py         Feature cache + cross-validation engine + metrics
│   ├── live.py             Live scoring loop (used by live_camera / score_video)
│   ├── alerts.py           AlertState: median of 5 + hysteresis
│   ├── live_state.py       Latest result shared with the dashboard
│   ├── db.py               SQLite score log
│   └── utils.py            seed_everything
│
├── templates/index.html    Dashboard page
├── static/css, static/js   Dashboard style and polling script
├── tests/test_core.py      Unit tests
│
├── checkpoints/            Trained models (not in git)
├── data/                   Videos, frames, face crops, cached features (not in git)
├── results/                CV results written by the scripts (do not edit by hand)
└── logs/                   SQLite log + live state (created at run time)
```

### Scripts (run these)

| File | What it does |
|---|---|
| `live_camera.py` | The live demo. Opens the webcam (`--camera N`) or replays a file (`--video clip.mp4`). Crops the face every 0.25 s, scores it with the live model, applies the alert rule, logs to SQLite and writes the dashboard state. `--headless` runs without a window. |
| `dashboard.py` | Flask dashboard at http://127.0.0.1:5000 (localhost only). Shows the current score, alert and capture status (OFFLINE when `live_camera.py` isn't running), plus a 30 s trend. |
| `score_video.py` | Runs the exact live pipeline on one video file without a camera and prints every score next to the expected morph value. Use it to check the setup. |
| `cross_validate.py` | Stratified 5-fold cross-validation **by video**, 3 seeds. Compares frame-only, mean-pool and LSTM heads against no-image baselines on clean, simulated-webcam and off-timing test frames. Writes `results/cross_validation.json`. `--final` trains the live model on all 40 videos → `checkpoints/pain_intensity_features.pt`. |
| `robustness_study.py` | Ablation with fixed selection rules: plain vs webcam-style augmentation vs timing jitter vs both, then an ensemble. Writes `results/robustness_study.json` and prints the `FINAL_*` values for `painsense/config.py`. |
| `evaluate.py` | Scores any checkpoint on held-out videos: MAE, RMSE, Pearson, Spearman, alert macro-F1, per-video MAE, always-0 baseline. |
| `train_full_model.py` | Optional, slow: trains the full CNN-LSTM on face images with a 28/6/6 video split, clip augmentation and opt-in layer4 fine-tuning (`--finetune-layer4`). Not the live model; only a fallback. |
| `extract_frames.py` | Data prep step 1: `data/videos/raw/SynPain/Videos/*.mp4` → `data/videos/frames/<clip>/frame_000001.jpg …`. Skips clips already done. |
| `crop_faces.py` | Data prep step 2: frames → 128×128 face crops in `data/videos/face_frames/`. Same crop recipe as the live demo. Never overwrites existing crops unless `--overwrite`. |

### Library (`painsense/`)

| Module | What it contains |
|---|---|
| `config.py` | All paths (data, checkpoints, results, logs), window length/spacing, model sizes, learning rates, `FINAL_*` live-model settings, alert thresholds, dashboard settings. |
| `face.py` | Haar face detector, `detect_face_box`, `crop_from_box` (margin 0.15 × max(w, h), resize to 128) and `crop_face`. Clear error if OpenCV 5 is installed. |
| `data.py` | Pain/NoPain label from folder names, the generated 0–10 target, window helpers (8 frames, 6 apart), clip augmentation, webcam-style degradation, image dataset for the full model, stratified video splits and k-fold. |
| `models.py` | `PainIntensityLSTM` (full CNN-LSTM), `FeatureIntensityModel` (frame / pool / lstm heads on cached features), `FeatureEnsemble`, `imagenet_backbone`, `load_feature_checkpoint`. |
| `features.py` | Feature cache (clean, flipped, webcam-style variants), `FeatureBank`, training of one setting, metrics, the CV loop and table printing. Shared by `cross_validate.py` and `robustness_study.py`. |
| `live.py` | `load_scorer` (live model, offline, ensemble-aware; falls back to the old model with a warning), `LiveScorer` (ResNet once per frame, head on the cached window), OpenCV overlay and the `run()` loop. |
| `alerts.py` | `AlertState`: alert ON when the median of the last 5 scores reaches 6, OFF when it falls below 4, reset when the face is lost. |
| `live_state.py` | Writes/reads `logs/dashboard_state.json` atomically, with a timestamp so the dashboard knows whether the camera is running. |
| `db.py` | SQLite log (`logs/painsense.db`) in WAL mode; keeps the newest 100,000 rows. |
| `utils.py` | `seed_everything` for reproducible runs. |

### Dashboard, tests, outputs

| Path | What it is |
|---|---|
| `templates/index.html`, `static/css/dashboard.css`, `static/js/dashboard.js` | Dashboard page. Polls `/api/state` every second and `/api/history` every 2 s. Chart.js comes from a CDN (pinned 4.4.1); without internet only the chart is skipped. |
| `tests/test_core.py` | 19 unit tests: alert rule, windows, splits, augmentations, model causality / BatchNorm / ensemble, live checkpoint, crop, SQLite, dashboard state. |
| `checkpoints/pain_intensity_features.pt` | **Live model.** Feature head(s) + ResNet18 weights (runs offline). Made by `cross_validate.py --final`. |
| `checkpoints/pain_intensity.pt` (+ `_calibration.json`) | Older full CNN-LSTM, trained on frames 1/24 s apart. Fallback only. |
| `checkpoints/cnn_encoder.pt`, `cnn_lstm.pt` | From the removed Pain/NoPain classifier; no code uses them any more. Safe to delete. |
| `results/cross_validation.json`, `results/robustness_study.json` | Written by the scripts. Never edit by hand. |
| `data/videos/raw/SynPain/Videos/` | The 40 SynPAIN clips (`.mp4`) and a still of each clip's first and last frame (`.jpg`). |
| `data/videos/frames/`, `data/videos/face_frames/` | Extracted frames and 128×128 face crops. |
| `data/features/` | Cached ResNet features (rebuilt automatically; git-ignored). |
| `data/intensity/manifest.csv` | The 0–10 target for every labeled face crop. |

Folders named `video_001` … `video_040` (and `data/videos/unlabeled/`) are leftover label-free copies of the same 40 clips. They are not a test set and no code reads them; you can delete them.

## Setup

Windows, from this folder:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

OpenCV must be **4.x** (5.x removed `CascadeClassifier`). Install only one OpenCV package. If `python -c "import cv2; print(cv2.__version__)"` prints 5.x:

```powershell
pip uninstall -y opencv-python opencv-python-headless
pip install "opencv-python>=4.8,<5"
```

## Run the demo

1. (Once) build the live model. The first run caches ResNet features for every face crop, including webcam-style copies; later runs reuse the cache:

   ```powershell
   python cross_validate.py --epochs 30 --seeds 3 --final
   ```

2. Terminal 1: `python live_camera.py`
3. Terminal 2: `python dashboard.py`, then open http://127.0.0.1:5000
4. Press `q` in the camera window to stop.

What you should see:

- No face: buffer clears, nothing is scored, alert resets. Faces smaller than 60 px are ignored, so sit reasonably close.
- Face: one crop every 0.25 s; the first score after 8 crops (about 2 s). ResNet runs once per new frame; only the small head sees the 8-frame window.
- Alert: ON when the median of the last 5 scores reaches 6, OFF when it drops below 4.

No camera? Check the whole pipeline on a clip:

```powershell
python score_video.py data\videos\raw\SynPain\Videos\1100031270_Pain_man_Young.mp4
python live_camera.py --video data\videos\raw\SynPain\Videos\1100031270_Pain_man_Young.mp4
```

## Results

From `results/cross_validation.json`: stratified 5-fold CV **by video**, 3 seeds (15 fold-runs), 40 videos / 3085 windows, frozen ImageNet ResNet18 features, 30 epochs. Metrics are on the **last frame** of every window, mean ± std over the 15 fold-runs.

| Model | MAE | RMSE | Pearson | Spearman | Alert macro-F1 |
|---|---|---|---|---|---|
| frame-only | 1.932 ± 0.258 | 2.617 ± 0.276 | 0.726 ± 0.071 | 0.711 ± 0.092 | 0.641 ± 0.078 |
| causal mean-pool | 1.947 ± 0.472 | 2.606 ± 0.530 | 0.750 ± 0.146 | 0.723 ± 0.186 | 0.662 ± 0.152 |
| LSTM | 1.782 ± 0.590 | 2.536 ± 0.608 | 0.750 ± 0.164 | 0.734 ± 0.172 | 0.758 ± 0.110 |
| always 0 (no image) | 3.410 ± 0.181 | 4.971 ± 0.139 | n/a | n/a | 0.409 ± 0.007 |
| train-mean (no image) | 3.353 ± 0.029 | 3.622 ± 0.029 | n/a | n/a | 0.409 ± 0.007 |

All three image models are far better than the no-image baselines. The LSTM has the lowest mean MAE, but the fold-to-fold spread is larger than the gaps between models (LSTM − frame-only: −0.149 ± 0.376, LSTM better in 10/15 runs), so this data does **not** establish that the LSTM beats frame-only or pooling.

These numbers predate the robustness study (webcam-style augmentation, timing jitter, ensemble); this section is updated when that study finishes.

Original model for reference (`pain_intensity.pt`, original 6-video validation split, consecutive frames, `python evaluate.py --ckpt checkpoints/pain_intensity.pt --split old_val --stride 1`): MAE 1.689 over all timesteps, 1.723 on the last frame; always-0 baseline 2.374 / 2.517. Six videos only, so noisy.

## Main settings (`painsense/config.py`)

| Setting | Value |
|---|---|
| Face crop / model input | 128×128 stored, upsampled to 224×224 |
| Window | 8 frames, 6 video frames apart (0.25 s at 24 fps) |
| Live sampling | every 0.25 s (derived from the window spacing) |
| Alert | median of last 5 scores ≥ 6 on, < 4 off |
| Live min face size | 60×60 px |
| Full-model split | stratified by video: 28 train / 6 val / 6 test |
| Learning rate | head 1e-3, ResNet layer4 1e-4 (opt-in) |
| Seed | 42 |
| SQLite log | newest 100,000 rows |

## Rebuild the data from scratch (only if `data/videos/face_frames` is missing)

```powershell
python extract_frames.py
python crop_faces.py
```

## Limits

- The 0–10 value is morph progress on synthetic faces, not a measured pain score.
- Only 40 clips (40 people), so any single split is noisy; trust the cross-validation tables, not one split.
- No patient study; no real pain labels.
- There is no hardware/LED code in this repository.
