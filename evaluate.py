"""
PainSense - evaluation using REAL SynPAIN video data.

Usage:

    python evaluate.py --stage cnn
    python evaluate.py --stage lstm
    python evaluate.py --stage intensity [--ckpt PATH] [--split test|old_val]

CNN:
    Evaluates individual real face-frame classification.

LSTM:
    Evaluates real consecutive video-frame sequences.

Intensity:
    Evaluates the 0-10 "expression intensity on a synthetic morph
    scale" regressor: MAE, RMSE, Pearson, Spearman, alert-level
    macro-F1, per-video MAE and the "always predict 0" baseline.
    Metrics use the LAST frame of every window (what the live demo
    reports); all-timestep MAE is printed too.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)

import config

from dataset import (
    FrameDataset,
    SequenceDataset,
    IntensitySequenceDataset,
    split_dataset,
    split_dataset_3way
)

from model import (
    CNNClassifier,
    PainCNNLSTM,
    PainIntensityLSTM,
    FeatureIntensityModel,
    imagenet_backbone
)
from utils import seed_everything


# =========================================================
# CNN Evaluation
# =========================================================

def eval_cnn():

    print()
    print("======================================")
    print("CNN EVALUATION")
    print("======================================")

    # Real SynPAIN face frames.
    ds = FrameDataset(
        train=False
    )

    # Use the same video-level split
    # used during training.
    _, val_ds = split_dataset(
        ds
    )

    loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = CNNClassifier().to(
        config.DEVICE
    )

    model.load_state_dict(
        torch.load(
            config.CNN_CKPT,
            map_location=config.DEVICE
        )
    )

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(
                config.DEVICE
            )

            logits = model(x)

            predictions = logits.argmax(
                dim=1
            )

            y_pred.extend(
                predictions.cpu().tolist()
            )

            y_true.extend(
                y.tolist()
            )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print()
    print(
        f"CNN Accuracy: {accuracy:.4f}"
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

    print()
    print(
        "Classification report:"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "NoPain",
                "Pain"
            ],
            zero_division=0
        )
    )


# =========================================================
# CNN-LSTM Evaluation
# =========================================================

def eval_lstm():

    print()
    print("======================================")
    print("REAL VIDEO CNN-LSTM EVALUATION")
    print("======================================")

    # Real consecutive video sequences.
    ds = SequenceDataset(
        train=False
    )

    # Same video-level validation split.
    _, val_ds = split_dataset(
        ds
    )

    loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    model = PainCNNLSTM().to(
        config.DEVICE
    )

    model.load_state_dict(
        torch.load(
            config.CNNLSTM_CKPT,
            map_location=config.DEVICE
        )
    )

    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(
                config.DEVICE
            )

            logits = model(x)

            # logits:
            # (B, T, 2)
            #
            # Convert to predicted class.
            predictions = logits.argmax(
                dim=-1
            )

            all_pred.extend(
                predictions.cpu().reshape(-1).tolist()
            )

            all_true.extend(
                y.long().reshape(-1).tolist()
            )

    accuracy = accuracy_score(
        all_true,
        all_pred
    )

    print()
    print(
        f"CNN-LSTM Accuracy: {accuracy:.4f}"
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            all_true,
            all_pred
        )
    )

    print()
    print(
        "Classification report:"
    )

    print(
        classification_report(
            all_true,
            all_pred,
            target_names=[
                "NoPain",
                "Pain"
            ],
            zero_division=0
        )
    )


# =========================================================
# Intensity evaluation
# =========================================================

def _load_intensity_scorer(ckpt):
    """
    Returns (fn, scale, bias, description).
    fn maps (B, T, C, H, W) images to (B, T) raw scores.
    """

    payload = torch.load(ckpt, map_location=config.DEVICE)

    if isinstance(payload, dict) and "variant" in payload:
        backbone = imagenet_backbone(
            payload.get("backbone_state_dict")
        ).to(config.DEVICE)
        head = FeatureIntensityModel(payload["variant"]).to(config.DEVICE)
        head.load_state_dict(payload["state_dict"])
        head.eval()

        def fn(x):
            b, t = x.shape[:2]
            feats = backbone(x.reshape(b * t, *x.shape[2:]))
            return head(feats.reshape(b, t, -1))

        return (
            fn,
            float(payload.get("scale", 1.0)),
            float(payload.get("bias", 0.0)),
            f"feature model ({payload['variant']})"
        )

    model = PainIntensityLSTM().to(config.DEVICE)
    model.load_state_dict(payload)
    model.eval()

    scale, bias = 1.0, 0.0
    cal_path = config.calibration_path_for(ckpt)

    if os.path.exists(cal_path):
        with open(cal_path, "r", encoding="utf-8") as handle:
            cal = json.load(handle)
        scale, bias = float(cal["scale"]), float(cal["bias"])

    return model, scale, bias, "full CNN-LSTM"


def intensity_metrics(pred, target):
    err = pred - target
    out = {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
    }

    if np.std(pred) < 1e-9 or np.std(target) < 1e-9:
        out["pearson"] = float("nan")
        out["spearman"] = float("nan")
    else:
        out["pearson"] = float(pearsonr(pred, target)[0])
        out["spearman"] = float(spearmanr(pred, target)[0])

    out["alert_macro_f1"] = float(f1_score(
        target >= config.ALERT_THRESHOLD,
        pred >= config.ALERT_THRESHOLD,
        average="macro",
        zero_division=0
    ))

    return out


def eval_intensity(ckpt=None, split="test", stride=None, seed=None):

    seed = seed_everything(seed)

    if ckpt is None:
        ckpt = (
            config.FEATURE_CKPT
            if os.path.exists(config.FEATURE_CKPT)
            else config.INTENSITY_CKPT
        )

    print()
    print("======================================")
    print("INTENSITY EVALUATION")
    print("expression intensity on a synthetic morph scale (0-10)")
    print("======================================")
    print(f"Checkpoint: {ckpt}")

    fn, scale, bias, description = _load_intensity_scorer(ckpt)
    print(f"Model: {description}, calibration scale={scale:.3f} bias={bias:.3f}")

    ds = IntensitySequenceDataset(train=False, stride=stride)

    if split == "old_val":
        # The original 6-video validation split (seed 42, 15 %).
        _, eval_ds = split_dataset(ds, seed=42)
        print("Split: original 6-video validation split")
    else:
        _, _, eval_ds, names = split_dataset_3way(ds, seed=seed)
        print(f"Split: test videos {names['test']}")

        if description.startswith("feature model"):
            print(
                "NOTE: cv_intensity.py --final trains on ALL videos, so "
                "these test videos are not held out for this checkpoint. "
                "Use results/cv_intensity.json for honest numbers."
            )

    loader = DataLoader(eval_ds, batch_size=4, shuffle=False, num_workers=0)
    video_of = [ds.groups[i] for i in eval_ds.indices]

    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            out = fn(x.to(config.DEVICE)).cpu()
            preds.append(out)
            targets.append(y)

    raw = torch.cat(preds).numpy()
    target = torch.cat(targets).numpy()
    lo, hi = config.PAIN_SCORE_MIN, config.PAIN_SCORE_MAX

    rows = {
        "model (raw, clamped)": np.clip(raw, lo, hi),
        "model (calibrated)": np.clip(scale * raw + bias, lo, hi),
        "always predict 0": np.zeros_like(raw),
    }

    print()
    print(f"{len(target)} windows, {len(set(video_of))} videos "
          f"(last-frame metrics)")
    print()
    print(f"{'':<24}{'MAE':>8}{'RMSE':>8}{'Pearson':>9}{'Spearman':>10}"
          f"{'alertF1':>9}{'MAE(all t)':>12}")

    report = {}

    for name, pred in rows.items():
        m = intensity_metrics(pred[:, -1], target[:, -1])
        m["mae_all_timesteps"] = float(np.abs(pred - target).mean())
        report[name] = m
        fmt = lambda v: "   n/a" if v != v else f"{v:6.3f}"
        print(
            f"{name:<24}{m['mae']:8.3f}{m['rmse']:8.3f}"
            f"{fmt(m['pearson']):>9}{fmt(m['spearman']):>10}"
            f"{m['alert_macro_f1']:9.3f}{m['mae_all_timesteps']:12.3f}"
        )

    print()
    print("Per-video MAE (last frame, calibrated vs always-0):")

    by_video = defaultdict(list)

    for i, v in enumerate(video_of):
        by_video[v].append(i)

    cal = rows["model (calibrated)"][:, -1]

    for v in sorted(by_video):
        idx = by_video[v]
        e_model = float(np.abs(cal[idx] - target[idx, -1]).mean())
        e_zero = float(np.abs(target[idx, -1]).mean())
        print(f"  {v:<34} model {e_model:6.3f}   always-0 {e_zero:6.3f}")

    print()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        choices=[
            "cnn",
            "lstm",
            "intensity"
        ],
        required=True
    )

    parser.add_argument(
        "--ckpt",
        default=None,
        help="intensity stage: checkpoint to evaluate"
    )

    parser.add_argument(
        "--split",
        choices=["test", "old_val"],
        default="test",
        help=(
            "intensity stage: 'test' = 3-way split test videos; "
            "'old_val' = the original 6-video validation split"
        )
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "intensity stage: frame spacing inside a window "
            f"(default {config.SEQ_STRIDE}; use 1 to reproduce the "
            "original pain_intensity.pt validation number)"
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=config.SEED
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    if args.stage == "cnn":

        eval_cnn()

    elif args.stage == "intensity":

        eval_intensity(
            ckpt=args.ckpt,
            split=args.split,
            stride=args.stride,
            seed=args.seed
        )

    else:

        eval_lstm()
