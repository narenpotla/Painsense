# PainSense

Prototype that estimates a facial pain score from a webcam, logs it, and raises an alert.

Camera → face detection → CNN (ResNet18) → LSTM → score (0–10) → alert → SQLite log → Flask dashboard

## What the score means

The score is not a clinical pain measurement (not VAS and not PSPI).

Training labels come from SynPAIN video face frames in `data/videos/face_frames`:

- **Pain** videos are morphs from a neutral face to a pain expression. The first frame is labeled 0 and the last frame is labeled 10.
- **NoPain** videos are labeled 0 on every frame.

The live model estimates that generated target. A held-out error of about 1.7 points was measured on that target. Full-pain frames are often predicted lower than 10, so a strong expression may stay under the alert line.

SynPAIN faces are synthetic. A laptop webcam is a different kind of image, so live scores can stay low.

## Setup

Windows, from this folder:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The trained demo weights are `checkpoints/pain_intensity.pt`.
Calibration is `checkpoints/pain_intensity_calibration.json`.

## Run the demo

Use two terminals.

Terminal 1 — camera, model, logging, LEDs:

```powershell
venv\Scripts\python.exe realtime_infer.py
```

Terminal 2 — dashboard:

```powershell
venv\Scripts\python.exe dashboard_web.py
```

Open http://127.0.0.1:5000

Press `q` in the camera window to stop.

What you should see:

- No face: the buffer clears, the model does not run, and the alert turns off.
- Face present: one crop is taken each second. The first score appears after 8 successful face crops.
- Score below 6: prediction `NO PAIN`, alert stays off. The score is still logged.
- Score 6 or above: prediction `PAIN` and an alert.

Do not also start `dashboard.py`. It uses the same port, 5000.

`gpiozero` is only needed on a Raspberry Pi. On Windows the LED calls do nothing and the camera still runs.

## LEDs (Raspberry Pi only)

`hardware.py` uses these pins:

| LED | GPIO | When it is on |
|---|---|---|
| Green | 17 | A face is being captured |
| Blue | 27 | Capturing, score under 6 |
| Red | 22 | Capturing, score at or above 6 |

On the Pi:

```bash
pip install gpiozero
```

## Train again

Intensity model (this is what the camera loads):

```powershell
venv\Scripts\python.exe train.py --stage intensity --epochs 6
```

Older Pain / No Pain classifier (not used by the camera):

```powershell
venv\Scripts\python.exe train.py --stage cnn
venv\Scripts\python.exe train.py --stage lstm
```

`evaluate.py` and `test_video.py` score that older classifier. They do not evaluate `pain_intensity.pt`.

## Main settings (`config.py`)

| Setting | Value |
|---|---|
| Image size | 224×224 |
| CNN embedding | 128 |
| LSTM hidden size | 64 |
| Frames per sequence | 8 |
| Seconds between live samples | 1.0 |
| Alert threshold | 6.0 |

`probability` stored in the database is `score / 10`. It is the same score written as a fraction. The alert uses the 0–10 score, not that fraction.

## Data layout

- `data/videos/face_frames/` — cropped faces. Folder names contain `_Pain_` or `_NoPain_`.
- `data/intensity/manifest.csv` — generated 0–10 label for each labeled frame.
- Folders named `video_001` … `video_040` have no Pain/NoPain name. Intensity training skips them.

## Files

| File | Role |
|---|---|
| `config.py` | Paths, image size, sequence length, alert threshold |
| `model.py` | ResNet18 encoder, old classifier, live `PainIntensityLSTM` |
| `dataset.py` | Loads frames and builds the 0–10 targets |
| `train.py` | CNN stage, LSTM stage, and intensity stage |
| `realtime_infer.py` | Webcam demo |
| `alert_rules.py` | Alert when the score reaches 6 |
| `db.py` | SQLite log at `logs/painsense.db` |
| `dashboard_state.py` | Latest result in `logs/dashboard_state.json` |
| `dashboard_web.py` | Dashboard at http://127.0.0.1:5000 |
| `hardware.py` | Raspberry Pi LEDs |
| `extract_video_frames.py` | Saves frames from SynPAIN mp4 files |
| `crop_video_faces.py` | Haar face crop into `face_frames` |

## Limits

This is a feasibility prototype.

- The 0–10 value is morph progress, not a measured clinical score.
- There is no patient study in this repository.
- UNBC-McMaster or BioVid labels would be required for a real intensity model.
- `data_prep.py` can still build alpha-blended sequences from static SynPAIN photos. That path is not what the current checkpoint was trained on.
