"""
PainSense - video-wise cross-validation of the intensity models.

Target: expression intensity on a synthetic morph scale (0-10). It is
linear in frame index for Pain videos and 0 for NoPain videos. It is
NOT a clinical pain score.

What it does:
    1. Caches frozen ImageNet ResNet18 features once per labeled video
       (original and horizontally flipped frames) in data/features/.
    2. Builds windows of SEQ_LEN frames, SEQ_STRIDE video frames apart.
    3. Runs stratified k-fold CV by VIDEO (never by window), repeated
       over several seeds, for:
           frame-only      one frame at a time, no temporal context
           causal mean-pool running mean of the window's features
           LSTM            LSTM over the window
           always 0        no-image baseline
           train-mean      no-image baseline
    4. Every metric is computed on the LAST frame of each window.
    5. Writes results/cv_intensity.json and prints a table.

    --final trains the LSTM on all videos and saves
    checkpoints/pain_intensity_features.pt (state dict, variant,
    seq_len, stride, scale, bias). scale/bias come from the
    out-of-fold LSTM predictions of this run, not from the final
    model's own training data.

Run:
    python cv_intensity.py --epochs 30 --seeds 3 --final
"""

import argparse
import glob
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

import config
from dataset import (
    frame_transform,
    frame_intensity,
    get_frame_number,
    list_labeled_videos,
    stratified_kfold_videos,
    valid_window_starts,
    window_numbers,
)
from model import FeatureIntensityModel, imagenet_backbone
from utils import seed_everything

warnings.filterwarnings("ignore", message=".*constant.*")

TRAINED_VARIANTS = ("frame", "pool", "lstm")
BASELINES = ("zero", "train_mean")
DISPLAY = {
    "frame": "frame-only",
    "pool": "causal mean-pool",
    "lstm": "LSTM",
    "zero": "always 0 (no image)",
    "train_mean": "train-mean (no image)",
}


# =========================================================
# Feature cache
# =========================================================

def cache_path(video_name):
    return os.path.join(config.FEATURE_DIR, f"{video_name}.npz")


@torch.no_grad()
def _encode(backbone, images, batch_size=64):
    out = []

    for i in range(0, len(images), batch_size):
        batch = torch.stack(
            [frame_transform(im) for im in images[i:i + batch_size]]
        ).to(config.DEVICE)
        out.append(backbone(batch).cpu())

    return torch.cat(out).numpy().astype(np.float32)


def build_feature_cache(rebuild=False):
    """Frozen ImageNet ResNet18 features for every labeled video."""

    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    videos = list_labeled_videos()

    if not videos:
        raise RuntimeError(
            f"No labeled videos found in {config.VIDEO_FACE_DIR}"
        )

    backbone = None

    for index, (name, label, folder) in enumerate(videos, 1):
        path = cache_path(name)

        if os.path.exists(path) and not rebuild:
            continue

        if backbone is None:
            backbone = imagenet_backbone().to(config.DEVICE)

        files = sorted(
            glob.glob(os.path.join(folder, "*.jpg")),
            key=get_frame_number
        )
        numbers = np.array([get_frame_number(p) for p in files])
        images = [Image.open(p).convert("RGB") for p in files]

        feats = _encode(backbone, images)
        feats_flip = _encode(backbone, [ImageOps.mirror(im) for im in images])

        lo, hi = int(numbers.min()), int(numbers.max())
        target = np.array(
            [frame_intensity(label, int(n), lo, hi) for n in numbers],
            dtype=np.float32
        )

        np.savez(
            path,
            numbers=numbers,
            feats=feats,
            feats_flip=feats_flip,
            target=target,
            label=label
        )
        print(
            f"[cache {index}/{len(videos)}] {name}: "
            f"{len(files)} frames -> {path}"
        )

    return [name for name, _, _ in videos]


def load_windows(names, stride):
    """
    Per video: windows of cached features.

    Returns {video: dict(X, Xf, y)} with
        X, Xf : (W, T, 512) original / flipped features
        y     : (W, T) targets
    """

    data = {}

    for name in names:
        z = np.load(cache_path(name))
        numbers = z["numbers"].tolist()
        index = {n: i for i, n in enumerate(numbers)}
        starts = valid_window_starts(numbers, config.SEQ_LEN, stride)

        if not starts:
            continue

        idx = np.array([
            [index[n] for n in window_numbers(s, config.SEQ_LEN, stride)]
            for s in starts
        ])

        data[name] = {
            "X": z["feats"][idx],
            "Xf": z["feats_flip"][idx],
            "y": z["target"][idx],
        }

    return data


# =========================================================
# Training / metrics
# =========================================================

def _stack(data, videos, key):
    return np.concatenate([data[v][key] for v in videos], axis=0)


def train_model(variant, data, train_videos, epochs, seed, lr=1e-3, batch=128):
    """Fit one FeatureIntensityModel on windows of train_videos."""

    seed_everything(seed)

    X = torch.from_numpy(_stack(data, train_videos, "X"))
    Xf = torch.from_numpy(_stack(data, train_videos, "Xf"))
    y = torch.from_numpy(_stack(data, train_videos, "y"))

    model = FeatureIntensityModel(variant).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = len(X)

    for _ in range(epochs):
        model.train()

        # Each window is used flipped or not, chosen per window per epoch.
        flip = (torch.rand(n) < 0.5).view(-1, 1, 1)
        inputs = torch.where(flip, Xf, X)
        order = torch.randperm(n)

        for i in range(0, n, batch):
            sel = order[i:i + batch]
            optimizer.zero_grad()
            pred = model(inputs[sel].to(config.DEVICE))
            loss = loss_fn(pred, y[sel].to(config.DEVICE))
            loss.backward()
            optimizer.step()

    model.eval()
    return model


@torch.no_grad()
def predict_last(model, X):
    """Last-frame prediction, clamped to the 0-10 scale."""

    out = []
    tensor = torch.from_numpy(X)

    for i in range(0, len(tensor), 512):
        out.append(model(tensor[i:i + 512].to(config.DEVICE))[:, -1].cpu())

    return torch.cat(out).numpy().clip(
        config.PAIN_SCORE_MIN, config.PAIN_SCORE_MAX
    )


def metrics(pred, target):
    """MAE, RMSE, Pearson, Spearman, alert-level macro-F1."""

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


def fit_linear(pred, target):
    """Least-squares scale/bias so that scale * pred + bias ~ target."""

    var = float(np.var(pred))

    if var < 1e-12:
        return 1.0, 0.0

    scale = float(np.cov(pred, target, bias=True)[0, 1] / var)
    bias = float(target.mean() - scale * pred.mean())
    return scale, bias


# =========================================================
# Cross-validation
# =========================================================

def run_cv(data, n_folds, n_seeds, epochs, base_seed):
    videos = sorted(data)
    runs = []
    oof_pred, oof_target = [], []
    t0 = time.time()

    for s in range(n_seeds):
        seed = base_seed + s
        folds = stratified_kfold_videos(videos, n_folds, seed)

        for k, (train_v, test_v) in enumerate(folds):
            Xte = _stack(data, test_v, "X")
            yte = _stack(data, test_v, "y")[:, -1]
            ytr_last = _stack(data, train_v, "y")[:, -1]
            run = {"seed": seed, "fold": k, "test_videos": test_v}

            for variant in TRAINED_VARIANTS:
                model = train_model(
                    variant, data, train_v, epochs, seed * 100 + k
                )
                pred = predict_last(model, Xte)
                run[variant] = metrics(pred, yte)

                if variant == "lstm":
                    oof_pred.append(pred)
                    oof_target.append(yte)

            run["zero"] = metrics(np.zeros_like(yte), yte)
            run["train_mean"] = metrics(
                np.full_like(yte, ytr_last.mean()), yte
            )
            runs.append(run)

            print(
                f"seed {seed} fold {k + 1}/{n_folds}  MAE "
                + "  ".join(
                    f"{m}={run[m]['mae']:.3f}"
                    for m in TRAINED_VARIANTS + BASELINES
                )
                + f"   [{time.time() - t0:.0f}s]",
                flush=True
            )

    return runs, np.concatenate(oof_pred), np.concatenate(oof_target)


def summarize(runs):
    summary = {}

    for model in TRAINED_VARIANTS + BASELINES:
        entry = {}

        for key in ("mae", "rmse", "pearson", "spearman", "alert_macro_f1"):
            values = np.array([r[model][key] for r in runs], dtype=float)

            if np.isnan(values).all():
                entry[key] = {"mean": None, "std": None}
            else:
                entry[key] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                }

        summary[model] = entry

    return summary


def paired(runs, a, b):
    """MAE difference a - b over the same folds (negative: a is better)."""

    diff = np.array([r[a]["mae"] - r[b]["mae"] for r in runs])
    return {
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std()),
        "a_better_in": int((diff < 0).sum()),
        "runs": int(len(diff)),
    }


def _cell(entry, key):
    v = entry[key]

    if v["mean"] is None:
        return "n/a"

    return f"{v['mean']:.3f} +- {v['std']:.3f}"


def print_table(summary, n_runs):
    cols = ("mae", "rmse", "pearson", "spearman", "alert_macro_f1")
    heads = ("MAE", "RMSE", "Pearson", "Spearman", "alert macroF1")
    width = 18

    print()
    print(f"Mean +- std over {n_runs} fold-runs; metrics on the LAST frame "
          f"of each window.")
    print("Target: expression intensity on a synthetic morph scale (0-10).")
    print()
    print(f"{'model':<24}" + "".join(f"{h:>{width}}" for h in heads))
    print("-" * (24 + width * len(heads)))

    for model in TRAINED_VARIANTS + BASELINES:
        row = f"{DISPLAY[model]:<24}"
        row += "".join(f"{_cell(summary[model], c):>{width}}" for c in cols)
        print(row)

    print()


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stride", type=int, default=config.SEQ_STRIDE)
    ap.add_argument("--seed", type=int, default=config.SEED,
                    help="first seed; seeds are seed, seed+1, ...")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--final", action="store_true",
                    help="train the LSTM on all videos and save "
                         f"{os.path.basename(config.FEATURE_CKPT)}")
    ap.add_argument("--out", default=os.path.join(
        config.RESULTS_DIR, "cv_intensity.json"))
    args = ap.parse_args()

    seed_everything(args.seed)
    print(f"Device: {config.DEVICE}")

    names = build_feature_cache(rebuild=args.rebuild_cache)
    data = load_windows(names, args.stride)

    n_windows = sum(len(v["y"]) for v in data.values())
    print(
        f"{len(data)} videos, {n_windows} windows "
        f"({config.SEQ_LEN} frames, stride {args.stride} = "
        f"{args.stride / config.VIDEO_FPS:.2f} s)"
    )

    runs, oof_pred, oof_target = run_cv(
        data, args.folds, args.seeds, args.epochs, args.seed
    )

    summary = summarize(runs)
    comparisons = {
        "lstm_minus_frame": paired(runs, "lstm", "frame"),
        "lstm_minus_pool": paired(runs, "lstm", "pool"),
        "pool_minus_frame": paired(runs, "pool", "frame"),
    }

    print_table(summary, len(runs))
    print("Paired MAE difference on identical folds (negative = first is better):")
    for key, c in comparisons.items():
        print(
            f"  {key:<18} {c['mean_diff']:+.3f} +- {c['std_diff']:.3f}   "
            f"first better in {c['a_better_in']}/{c['runs']} runs"
        )

    # Least squares minimizes MSE, not MAE, so the fit is only kept when
    # it lowers the pooled out-of-fold MAE. Otherwise identity.
    fit_scale, fit_bias = fit_linear(oof_pred, oof_target)
    cal = (fit_scale * oof_pred + fit_bias).clip(
        config.PAIN_SCORE_MIN, config.PAIN_SCORE_MAX
    )
    mae_raw = float(np.abs(oof_pred - oof_target).mean())
    mae_cal = float(np.abs(cal - oof_target).mean())
    use_fit = mae_cal < mae_raw
    scale, bias = (fit_scale, fit_bias) if use_fit else (1.0, 0.0)
    calibration = {
        "fitted_scale": fit_scale,
        "fitted_bias": fit_bias,
        "oof_mae_raw": mae_raw,
        "oof_mae_with_fit": mae_cal,
        "applied": bool(use_fit),
        "scale": scale,
        "bias": bias,
        "note": "least-squares fit on pooled out-of-fold LSTM predictions; "
                "applied only if it lowers out-of-fold MAE, else identity",
    }
    print(
        f"\nLSTM out-of-fold calibration: fitted scale={fit_scale:.3f} "
        f"bias={fit_bias:.3f}; OOF MAE raw {mae_raw:.3f} vs with fit "
        f"{mae_cal:.3f} -> "
        f"{'applied' if use_fit else 'NOT applied (identity)'}"
    )

    result = {
        "target": "expression intensity on a synthetic morph scale (0-10)",
        "metrics_on": "last frame of each window",
        "n_videos": len(data),
        "n_windows": n_windows,
        "seq_len": config.SEQ_LEN,
        "stride_frames": args.stride,
        "epochs": args.epochs,
        "folds": args.folds,
        "seeds": [args.seed + i for i in range(args.seeds)],
        "features": "frozen ImageNet ResNet18, flip copies used as augmentation",
        "summary": summary,
        "paired_mae_diff": comparisons,
        "calibration_oof": calibration,
        "runs": runs,
    }

    if args.final:
        print("\n--final: training LSTM on all videos ...")
        model = train_model(
            "lstm", data, sorted(data), args.epochs, args.seed
        )
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "variant": "lstm",
                "seq_len": config.SEQ_LEN,
                "stride": args.stride,
                "scale": scale,
                "bias": bias,
            },
            config.FEATURE_CKPT
        )
        result["final_checkpoint"] = config.FEATURE_CKPT
        print(f"Saved {config.FEATURE_CKPT}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Results -> {args.out}")


if __name__ == "__main__":
    main()
