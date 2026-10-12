# PainSense

Prototype that estimates a facial **expression intensity** score from a webcam, logs it, and raises an alert.

Camera → face detection → ResNet18 features → LSTM → score (0–10) → alert → SQLite log → Flask dashboard

## What the score means

The score is **expression intensity on a synthetic morph scale**. It is not a clinical pain measurement (not VAS and not PSPI).

Training targets come from SynPAIN video face frames in `data/videos/face_frames`:

- **Pain** videos are morphs from a neutral face to a pain expression. The target is linear in frame index: the first frame is 0 and the last frame is 10. The model learns "how far along the morph is this face", not how much pain a person feels.
- **NoPain** videos are 0 on every frame.

SynPAIN faces are synthetic, and a laptop webcam is a different kind of image, so live scores can stay low or behave oddly. Do not read the number as pain.

## Setup

Windows, from this folder:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`opencv-python` must be 4.x. OpenCV 5.x removed `cv2.CascadeClassifier`, which the face crop uses. Install only **one** OpenCV package: `opencv-python` and `opencv-python-headless` write to the same `cv2` folder, and the last one installed wins. If `python -c "import cv2; print(cv2.__version__)"` prints 5.x, run:

```powershell
pip uninstall -y opencv-python opencv-python-headless
pip install "opencv-python>=4.8,<5"
```

Check the install:

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Models

| Checkpoint | What it is | Made by |
|---|---|---|
| `checkpoints/pain_intensity_features.pt` | Frozen ImageNet ResNet18 features → LSTM head. Trained at the live frame spacing (0.25 s). Contains the backbone weights too, so the demo runs offline. **Used by the camera when present.** | `cv_intensity.py --final` |
| `checkpoints/pain_intensity.pt` (+ `pain_intensity_calibration.json`) | Older full CNN-LSTM, trained on frames 1/24 s apart. The camera falls back to it with a warning because its frame spacing does not match live sampling. | `train.py --stage intensity` |
| `checkpoints/cnn_encoder.pt`, `cnn_lstm.pt` | Older binary Pain / NoPain classifier. Not used by the camera. | `train.py --stage cnn` / `lstm` |

## Run the demo

Create the feature checkpoint once. The first run caches ResNet features for about 9,700 face crops (original + flipped; roughly 25 ms each on a laptop CPU), then CV and the final model take a few minutes:

```powershell
venv\Scripts\python.exe cv_intensity.py --epochs 30 --seeds 3 --final
```

Then use two terminals.

Terminal 1 — camera, model, logging:

```powershell
venv\Scripts\python.exe realtime_infer.py
```

Options: `--camera 1` for another webcam; `--video clip.mp4` replays a file (sampled every `round(fps × 0.25 s)` frames, so spacing matches training); `--headless` runs without a window (Ctrl+C to stop).

Terminal 2 — dashboard (binds to 127.0.0.1 only):

```powershell
venv\Scripts\python.exe dashboard_web.py
```

Open http://127.0.0.1:5000

Press `q` in the camera window to stop.

The dashboard shows **OFFLINE** when `realtime_infer.py` has not written a state for 3 s. The trend chart uses Chart.js from a CDN (pinned 4.4.1); without internet the chart is skipped and the cards and table still update.

What you should see:

- No face: the feature buffer clears, the model does not run, and the alert and its history reset.
- Face present: one crop is taken every 0.25 s (`CAPTURE_INTERVAL_SEC = SEQ_STRIDE / 24`). ResNet runs once per new frame; only the LSTM sees the 8-frame window. The first score appears after 8 successful face crops (about 2 s).
- Alert: ON when the **median of the last 5 scores** reaches 6 (only once 5 scores exist); OFF only when that median falls below 4. Every score is still logged.
- Webcam faces smaller than 60×60 px are ignored (`LIVE_MIN_FACE_SIZE`) to cut false detections; sit closer if you only ever see "No face".

The SQLite log runs in WAL mode (the dashboard can read while the camera writes ~4 rows/s) and keeps the newest 100,000 rows (`DB_MAX_ROWS`).

Check the whole pipeline on a SynPAIN clip without a camera:

```powershell
venv\Scripts\python.exe test_video.py dataideosaw\SynPain\VideosH0031270_Pain_man_Young.mp4
```

It prints every sampled score next to the morph target. The final checkpoint was trained on all 40 clips, so this is a pipeline check, not a test score.

## LEDs / hardware

There is no `hardware.py` in this repository and no LED code anywhere in it. Earlier versions of this README described Raspberry Pi LEDs; that code does not exist here. `gpiozero` is not needed and is commented out in `requirements.txt`.

## Train / evaluate

Cross-validation and the live-demo checkpoint (frozen features, fast):

```powershell
venv\Scripts\python.exe cv_intensity.py --epochs 30 --seeds 3 --final
```

Full CNN-LSTM intensity model (slow; default 3 epochs). It uses a video-level train / val / test split: the checkpoint is chosen and the calibration is fitted on val, test is scored once. ResNet `layer4` is frozen unless you pass `--finetune-layer4` (lr 1e-4; head lr 1e-3):

```powershell
venv\Scripts\python.exe train.py --stage intensity --epochs 3
```

Older Pain / No Pain classifier (not used by the camera):

```powershell
venv\Scripts\python.exe train.py --stage cnn
venv\Scripts\python.exe train.py --stage lstm
```

Evaluation:

```powershell
venv\Scripts\python.exe evaluate.py --stage intensity                      # 3-way split test videos
venv\Scripts\python.exe evaluate.py --stage intensity --ckpt checkpoints\pain_intensity.pt --split old_val --stride 1
venv\Scripts\python.exe evaluate.py --stage cnn
venv\Scripts\python.exe evaluate.py --stage lstm
```

`--split old_val --stride 1` reproduces the original 6-video validation split and frame spacing of `pain_intensity.pt`. `evaluate.py --stage intensity` on `pain_intensity_features.pt` is in-sample (that model saw every video); use the CV table below for honest numbers.

`test_video.py clip.mp4` runs the camera's intensity model on a file; `test_video.py clip.mp4 --binary` runs the older classifier.

Tests:

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Results

From `results/cv_intensity.json` (written by `cv_intensity.py`; do not edit by hand):
stratified 5-fold CV **by video**, 3 seeds (15 fold-runs), 40 videos / 3085 windows,
frozen ImageNet ResNet18 features, 30 epochs. Metrics are on the **last frame** of every window.
Mean ± std over the 15 fold-runs. Target: expression intensity on a synthetic morph scale (0–10).

| Model | MAE | RMSE | Pearson | Spearman | Alert macro-F1 |
|---|---|---|---|---|---|
| frame-only | 1.932 ± 0.258 | 2.617 ± 0.276 | 0.726 ± 0.071 | 0.711 ± 0.092 | 0.641 ± 0.078 |
| causal mean-pool | 1.947 ± 0.472 | 2.606 ± 0.530 | 0.750 ± 0.146 | 0.723 ± 0.186 | 0.662 ± 0.152 |
| LSTM | 1.782 ± 0.590 | 2.536 ± 0.608 | 0.750 ± 0.164 | 0.734 ± 0.172 | 0.758 ± 0.110 |
| always 0 (no image) | 3.410 ± 0.181 | 4.971 ± 0.139 | n/a | n/a | 0.409 ± 0.007 |
| train-mean (no image) | 3.353 ± 0.029 | 3.622 ± 0.029 | n/a | n/a | 0.409 ± 0.007 |

Paired MAE difference on identical folds (negative = first model is better):

| Comparison | Mean diff ± std | First better in |
|---|---|---|
| LSTM − frame-only | -0.149 ± 0.376 | 10/15 runs |
| LSTM − mean-pool | -0.164 ± 0.285 | 12/15 runs |
| mean-pool − frame-only | +0.015 ± 0.247 | 6/15 runs |

How to read this: all three image models are far better than the no-image baselines. The LSTM has the lowest mean MAE, but the fold-to-fold std (0.59 for the LSTM) is larger than the gaps between models, and the paired differences have std larger than their means. The ordering varies from fold to fold, so this data does **not** establish that the LSTM beats frame-only or pooling. The score is a morph-progress proxy; on it, temporal context may add little because one frame already shows how far the morph has gone.

Calibration: a least-squares scale/bias fit on out-of-fold LSTM predictions raised MAE (1.790 → 1.972), so it is **not** applied; `pain_intensity_features.pt` stores scale 1, bias 0.

Original model for reference (`pain_intensity.pt`, original 6-video validation split, frame spacing 1/24 s, reproduced with `evaluate.py --stage intensity --ckpt checkpoints/pain_intensity.pt --split old_val --stride 1`): MAE 1.689 over all timesteps (raw), 1.723 on the last frame; always-0 baseline 2.374 / 2.517. These are measured on 6 videos and are noisy; the README used to quote about 1.7.

## Main settings (`config.py`)

| Setting | Value |
|---|---|
| Stored face crop | 128×128 (upscaled to the 224×224 model input) |
| Model input | 224×224 |
| Embedding / LSTM hidden | 128 / 64 |
| Frames per window (`SEQ_LEN`) | 8 |
| Frame spacing in a window (`SEQ_STRIDE`) | 6 video frames = 0.25 s at 24 fps |
| Seconds between live samples | 0.25 (derived from `SEQ_STRIDE`) |
| Alert on / off | median of last 5 scores ≥ 6.0 / < 4.0 |
| Split | stratified by video: 28 train / 6 val / 6 test |
| LR (head / layer4) | 1e-3 / 1e-4 (layer4 opt-in) |
| Seed | 42 |
| Live min face size | 60×60 px |
| DB rows kept | 100,000 newest |
| Dashboard chart | last 120 scores (30 s) |

`probability` stored in the database is `score / 10`. It is the same score written as a fraction. The alert uses the 0–10 score, not that fraction.

## Data layout

- `data/videos/face_frames/` — cropped faces, 128×128. Folder names contain `_Pain_` or `_NoPain_`.
- `data/intensity/manifest.csv` — generated 0–10 target for each labeled frame.
- `data/features/` — cached ResNet features (made by `cv_intensity.py`, git-ignored).
- Folders `video_001` … `video_040` are the **same 40 clips** as the labeled folders with the names removed (`unlabeled_videos.py` copies them in sorted order). They are **not** a held-out test set. Intensity training and evaluation skip them.
- There are 40 labeled videos (20 Pain, 20 NoPain, 24 fps, about 122 frames each).

## Files

| File | Role |
|---|---|
| `config.py` | Paths, sizes, sequence length / stride, alert thresholds, seed |
| `utils.py` | `seed_everything` |
| `model.py` | ResNet18 encoder, old classifier, `PainIntensityLSTM`, feature-only `FeatureIntensityModel` |
| `dataset.py` | Frame / sequence / intensity datasets, augmentation, video-level splits |
| `data_prep.py` | Face detection and the shared `crop_from_box()` / `crop_face()` |
| `train.py` | CNN stage, LSTM stage, intensity stage |
| `cv_intensity.py` | Feature cache, video-wise k-fold CV, `--final` live checkpoint |
| `evaluate.py` | Evaluation: `cnn`, `lstm`, `intensity` |
| `test_video.py` | Intensity model (default) or old classifier (`--binary`) on one video file |
| `realtime_infer.py` | Webcam demo (`--video` to replay a file, `--headless`) |
| `alert_rules.py` | `AlertState`: median of 5 + hysteresis |
| `db.py` | SQLite log at `logs/painsense.db` |
| `dashboard_state.py` | Latest result in `logs/dashboard_state.json` |
| `dashboard_web.py` | Dashboard at http://127.0.0.1:5000 (`templates/`, `static/`) |
| `extract_video_frames.py` | Saves frames from SynPAIN mp4 files |
| `crop_video_faces.py` | Haar face crop into `face_frames` |
| `extract_frames.py`, `unlabeled_videos.py` | Make / extract the label-free copies |
| `tests/test_core.py` | Unit tests (alert rule, windows, splits, augmentation, causality, crop, DB, state) |

`frontend.zip` is an older copy of `templates/` and `static/`; the folders are current.

## Limits

This is a feasibility prototype.

- The 0–10 value is morph progress (a synthetic target), not a measured clinical score.
- Only 40 labeled videos exist, so any single split is noisy. Use the cross-validation table, not one split.
- There is no patient study in this repository.
- UNBC-McMaster or BioVid labels would be required for a real intensity model.
- `data_prep.py` can still build alpha-blended sequences from static SynPAIN photos. That path is not what the current models were trained on.
