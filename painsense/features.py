"""
painsense.features - feature cache and the cross-validation engine.

Shared by cross_validate.py and robustness_study.py.

    build_feature_cache    frozen ImageNet ResNet18 features of every
                           labeled face frame (+ flipped) -> data/features/
    build_aug_cache        webcam-style variants: N_TRAIN_AUG for training,
                           N_HELDOUT_AUG (harsher family) for testing only
    FeatureBank            all cached features in one table; windows are
                           index arrays, so nothing is copied per window
    make_spec / train_spec train one setting (variant, aug, strides, ensemble)
    evaluate_models        metrics on clean, sim-webcam and spacing 5 / 7
    run_cv                 stratified k-fold by VIDEO x seeds, identical
                           folds for every setting
    summarize / paired / print_table   reporting

Every metric is computed on the LAST frame of each window (what the live
demo shows). Target: expression intensity on a synthetic morph scale.
"""

import glob
import os
import time
import warnings
import zlib

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

from painsense import config
from painsense.data import (
    apply_webcam,
    frame_transform,
    frame_intensity,
    get_frame_number,
    list_labeled_videos,
    sample_webcam_params,
    stratified_kfold_videos,
    valid_window_starts,
    window_numbers,
)
from painsense.models import FeatureIntensityModel, imagenet_backbone
from painsense.utils import seed_everything

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
AUG_CACHE_VERSION = 1
EVAL_STRIDES = (5, 6, 7)


# =========================================================
# Feature cache
# =========================================================

def cache_path(video_name):
    return os.path.join(config.FEATURE_DIR, f"{video_name}.npz")


def aug_cache_path(video_name):
    return os.path.join(config.FEATURE_DIR, "aug", f"{video_name}.npz")


@torch.inference_mode()
def _encode(backbone, images, batch_size=64):
    out = []

    for i in range(0, len(images), batch_size):
        batch = torch.stack(
            [frame_transform(im) for im in images[i:i + batch_size]]
        ).to(config.DEVICE)
        out.append(backbone(batch).cpu())

    return torch.cat(out).numpy().astype(np.float32)


def _video_frames(folder):
    files = sorted(
        glob.glob(os.path.join(folder, "*.jpg")),
        key=get_frame_number
    )
    numbers = np.array([get_frame_number(p) for p in files])
    images = [Image.open(p).convert("RGB") for p in files]
    return numbers, images


def build_feature_cache(rebuild=False):
    """Clean + flipped frozen ImageNet ResNet18 features per video."""

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

        numbers, images = _video_frames(folder)
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
        print(f"[cache {index}/{len(videos)}] {name}: {len(numbers)} frames")

    return [name for name, _, _ in videos]


def _variant_features(backbone, name, images, k, held_out):
    """Features of one webcam-degraded copy of a whole video."""

    base = zlib.crc32(name.encode("utf-8")) % 1_000_000
    offset = 500_000 if held_out else 0
    param_rng = np.random.RandomState(base + offset + 1000 * k)
    frame_rng = np.random.RandomState(base + offset + 1000 * k + 1)
    params = sample_webcam_params(param_rng, held_out=held_out)
    degraded = [apply_webcam(im, params, frame_rng) for im in images]
    return _encode(backbone, degraded)


def build_aug_cache(names, n_train, n_heldout, rebuild=False):
    """Webcam-style training variants and held-out test variants."""

    os.makedirs(os.path.join(config.FEATURE_DIR, "aug"), exist_ok=True)
    folders = {name: folder for name, _, folder in list_labeled_videos()}
    backbone = None
    t0 = time.time()

    for index, name in enumerate(names, 1):
        path = aug_cache_path(name)

        if os.path.exists(path) and not rebuild:
            z = np.load(path)
            if (
                int(z["version"]) == AUG_CACHE_VERSION
                and z["train"].shape[0] >= n_train
                and z["heldout"].shape[0] >= n_heldout
            ):
                continue

        if backbone is None:
            backbone = imagenet_backbone().to(config.DEVICE)

        numbers, images = _video_frames(folders[name])
        train = np.stack([
            _variant_features(backbone, name, images, k, held_out=False)
            for k in range(n_train)
        ])
        heldout = np.stack([
            _variant_features(backbone, name, images, k, held_out=True)
            for k in range(n_heldout)
        ])

        np.savez(
            path,
            numbers=numbers,
            train=train,
            heldout=heldout,
            version=AUG_CACHE_VERSION
        )
        print(
            f"[aug cache {index}/{len(names)}] {name}: "
            f"{n_train}+{n_heldout} variants   [{time.time() - t0:.0f}s]",
            flush=True
        )


class FeatureBank:
    """
    All cached features in one table.

    sources[s, i] is the feature of global frame i from source s:
        s = 0 clean, 1 flipped, 2.. webcam-style training variants
    heldout[e, i] is a held-out "simulated webcam" variant.
    Windows are arrays of global frame indices, so nothing is copied
    per window.
    """

    def __init__(self, names, n_train_aug=0, n_heldout_aug=0):
        sources, heldout, targets = [], [], []
        self.offset, self.numbers = {}, {}
        start = 0

        for name in names:
            z = np.load(cache_path(name))
            numbers = z["numbers"]
            per_video = [z["feats"], z["feats_flip"]]
            held = []

            if n_train_aug or n_heldout_aug:
                a = np.load(aug_cache_path(name))

                if not np.array_equal(a["numbers"], numbers):
                    raise RuntimeError(f"aug cache out of date: {name}")

                per_video += list(a["train"][:n_train_aug])
                held = list(a["heldout"][:n_heldout_aug])

            sources.append(np.stack(per_video))
            if held:
                heldout.append(np.stack(held))
            targets.append(z["target"])
            self.offset[name] = start
            self.numbers[name] = numbers.tolist()
            start += len(numbers)

        self.names = list(names)
        self.sources = np.concatenate(sources, axis=1)
        self.heldout = np.concatenate(heldout, axis=1) if heldout else None
        self.target = np.concatenate(targets)
        self.n_train_aug = n_train_aug
        self._windows = {}

    def windows(self, name, stride):
        key = (name, stride)

        if key not in self._windows:
            numbers = self.numbers[name]
            index = {n: i for i, n in enumerate(numbers)}
            starts = valid_window_starts(numbers, config.SEQ_LEN, stride)
            idx = np.array(
                [
                    [index[n] for n in window_numbers(s, config.SEQ_LEN, stride)]
                    for s in starts
                ],
                dtype=np.int64
            ).reshape(-1, config.SEQ_LEN)
            self._windows[key] = idx + self.offset[name]

        return self._windows[key]

    def window_index(self, videos, strides):
        return np.concatenate(
            [self.windows(v, s) for v in videos for s in strides]
        )


# =========================================================
# Training / prediction
# =========================================================

def make_spec(variant, aug=False, strides=(6,), ensemble=1):
    return {
        "variant": variant,
        "aug": bool(aug),
        "strides": tuple(int(s) for s in strides),
        "ensemble": int(ensemble),
    }


def spec_name(spec):
    parts = [spec["variant"]]
    if spec["aug"]:
        parts.append("aug")
    if spec["strides"] != (config.SEQ_STRIDE,):
        parts.append("stride" + "/".join(map(str, spec["strides"])))
    if spec["ensemble"] > 1:
        parts.append(f"ens{spec['ensemble']}")
    return "+".join(parts)


def _train_one(variant, bank, pool_idx, n_per_epoch, use_aug, epochs, seed,
               lr=1e-3, batch=128):
    """
    One head. Each epoch draws n_per_epoch windows from pool_idx (all
    training strides) and, per window, a feature source: clean or
    flipped, or (use_aug) a webcam-style variant half of the time.
    """

    seed_everything(seed)
    rng = np.random.RandomState(seed)

    model = FeatureIntensityModel(variant).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n_aug = bank.sources.shape[0] - 2

    for _ in range(epochs):
        model.train()

        if len(pool_idx) > n_per_epoch:
            chosen = pool_idx[rng.choice(len(pool_idx), n_per_epoch, replace=False)]
        else:
            chosen = pool_idx[rng.permutation(len(pool_idx))]

        src = rng.randint(0, 2, len(chosen))

        if use_aug and n_aug > 0:
            use = rng.rand(len(chosen)) < 0.5
            src = np.where(use, 2 + rng.randint(0, n_aug, len(chosen)), src)

        for i in range(0, len(chosen), batch):
            idx = chosen[i:i + batch]
            x = torch.from_numpy(bank.sources[src[i:i + batch, None], idx])
            y = torch.from_numpy(bank.target[idx])

            optimizer.zero_grad()
            loss = loss_fn(model(x.to(config.DEVICE)), y.to(config.DEVICE))
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def train_spec(spec, bank, train_videos, epochs, seed):
    """Train spec["ensemble"] heads. Returns a list of models."""

    pool = bank.window_index(train_videos, spec["strides"])
    n_per_epoch = len(bank.window_index(train_videos, (config.SEQ_STRIDE,)))

    return [
        _train_one(
            spec["variant"], bank, pool, n_per_epoch, spec["aug"],
            epochs, seed + 7919 * e
        )
        for e in range(spec["ensemble"])
    ]


@torch.no_grad()
def predict_last(models, feats):
    """Mean last-frame prediction of `models`, clamped to 0-10."""

    tensor = torch.from_numpy(np.ascontiguousarray(feats))
    total = None

    for model in models:
        out = torch.cat([
            model(tensor[i:i + 1024].to(config.DEVICE))[:, -1].cpu()
            for i in range(0, len(tensor), 1024)
        ])
        total = out if total is None else total + out

    return (total / len(models)).numpy().clip(
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


def evaluate_models(models, bank, test_videos):
    """Metrics on clean, simulated-webcam and off-spacing test windows."""

    out = {}
    idx6 = bank.window_index(test_videos, (config.SEQ_STRIDE,))
    y6 = bank.target[idx6[:, -1]]
    clean_pred = predict_last(models, bank.sources[0][idx6])
    out["clean"] = metrics(clean_pred, y6)

    if bank.heldout is not None:
        preds = [predict_last(models, h[idx6]) for h in bank.heldout]
        out["sim"] = metrics(np.concatenate(preds), np.tile(y6, len(preds)))

    for stride in EVAL_STRIDES:
        if stride == config.SEQ_STRIDE:
            continue
        idx = bank.window_index(test_videos, (stride,))
        pred = predict_last(models, bank.sources[0][idx])
        out[f"clean_s{stride}"] = {
            "mae": float(np.abs(pred - bank.target[idx[:, -1]]).mean())
        }

    return out, clean_pred, y6


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

def run_cv(bank, specs, n_folds, n_seeds, epochs, base_seed,
           oof_keys=(), baselines=True, label=""):
    """
    specs: {name: spec}. Every spec sees the same folds and seeds, so
    results can be compared pairwise.

    Returns (runs, oof) where oof = {name: (pred, target)} holds the
    pooled out-of-fold last-frame predictions of every name in oof_keys.
    """

    videos = bank.names
    runs = []
    oof_parts = {key: ([], []) for key in oof_keys}
    t0 = time.time()

    for s in range(n_seeds):
        seed = base_seed + s
        folds = stratified_kfold_videos(videos, n_folds, seed)

        for k, (train_v, test_v) in enumerate(folds):
            run = {"seed": seed, "fold": k, "test_videos": test_v}

            for name, spec in specs.items():
                models = train_spec(spec, bank, train_v, epochs, seed * 100 + k)
                run[name], pred, y6 = evaluate_models(models, bank, test_v)

                if name in oof_parts:
                    oof_parts[name][0].append(pred)
                    oof_parts[name][1].append(y6)

            if baselines:
                idx6 = bank.window_index(test_v, (config.SEQ_STRIDE,))
                y6 = bank.target[idx6[:, -1]]
                ytr = bank.target[
                    bank.window_index(train_v, (config.SEQ_STRIDE,))[:, -1]
                ]
                zero = metrics(np.zeros_like(y6), y6)
                mean = metrics(np.full_like(y6, ytr.mean()), y6)
                # No-image baselines do not see the image, so the
                # degraded copies give the same numbers.
                run["zero"] = {"clean": zero, "sim": zero}
                run["train_mean"] = {"clean": mean, "sim": mean}

            runs.append(run)

            shown = list(specs)[:6]
            print(
                f"{label}seed {seed} fold {k + 1}/{n_folds}  clean MAE "
                + "  ".join(f"{n}={run[n]['clean']['mae']:.3f}" for n in shown)
                + f"   [{time.time() - t0:.0f}s]",
                flush=True
            )

    oof = {
        key: (np.concatenate(p), np.concatenate(t))
        for key, (p, t) in oof_parts.items()
        if p
    }
    return runs, oof


def decide_calibration(pred, target):
    """
    Least-squares scale/bias on out-of-fold predictions. Least squares
    minimizes MSE, not MAE, so the fit is only kept when it lowers the
    pooled out-of-fold MAE; otherwise identity (scale 1, bias 0).
    """

    fit_scale, fit_bias = fit_linear(pred, target)
    cal = (fit_scale * pred + fit_bias).clip(
        config.PAIN_SCORE_MIN, config.PAIN_SCORE_MAX
    )
    mae_raw = float(np.abs(pred - target).mean())
    mae_cal = float(np.abs(cal - target).mean())
    use_fit = mae_cal < mae_raw

    return {
        "fitted_scale": fit_scale,
        "fitted_bias": fit_bias,
        "oof_mae_raw": mae_raw,
        "oof_mae_with_fit": mae_cal,
        "applied": bool(use_fit),
        "scale": fit_scale if use_fit else 1.0,
        "bias": fit_bias if use_fit else 0.0,
    }


def save_feature_checkpoint(path, models, variant, settings, scale, bias,
                            purpose):
    """Save head(s) + frozen backbone weights (loads offline)."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": models[0].state_dict(),
            "state_dicts": [m.state_dict() for m in models],
            "variant": variant,
            "purpose": purpose,
            "seq_len": config.SEQ_LEN,
            "stride": config.SEQ_STRIDE,
            "training_settings": settings,
            "scale": scale,
            "bias": bias,
            # Frozen ImageNet weights stored with the head so nothing
            # is downloaded at run time.
            "backbone_state_dict": imagenet_backbone().state_dict(),
        },
        path
    )


def summarize(runs, names):
    """mean/std over runs for every model, condition and metric."""

    summary = {}

    for name in names:
        summary[name] = {}

        for cond in runs[0][name]:
            entry = {}

            for key in runs[0][name][cond]:
                values = np.array(
                    [r[name][cond][key] for r in runs], dtype=float
                )

                if np.isnan(values).all():
                    entry[key] = {"mean": None, "std": None}
                else:
                    entry[key] = {
                        "mean": float(np.nanmean(values)),
                        "std": float(np.nanstd(values)),
                    }

            summary[name][cond] = entry

    return summary


def paired(runs, a, b, cond="clean"):
    """MAE difference a - b over the same folds (negative: a is better)."""

    diff = np.array([r[a][cond]["mae"] - r[b][cond]["mae"] for r in runs])
    return {
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std()),
        "a_better_in": int((diff < 0).sum()),
        "runs": int(len(diff)),
    }


def _cell(entry, key):
    v = entry.get(key)

    if not v or v["mean"] is None:
        return "n/a"

    return f"{v['mean']:.3f} +- {v['std']:.3f}"


def print_table(summary, names, n_runs, display=None):
    display = display or {}
    width = 18

    print()
    print(f"Mean +- std over {n_runs} fold-runs; metrics on the LAST frame "
          "of each window.")
    print("Target: expression intensity on a synthetic morph scale (0-10).")
    print()
    print("Clean test frames (spacing 6):")
    heads = ("MAE", "RMSE", "Pearson", "Spearman", "alert macroF1")
    cols = ("mae", "rmse", "pearson", "spearman", "alert_macro_f1")
    print(f"{'model':<30}" + "".join(f"{h:>{width}}" for h in heads))
    print("-" * (30 + width * len(heads)))

    for name in names:
        row = f"{display.get(name, name):<30}"
        row += "".join(
            f"{_cell(summary[name]['clean'], c):>{width}}" for c in cols
        )
        print(row)

    print()
    print("Robustness (MAE):")
    heads = ("clean", "sim-webcam", "spacing 5", "spacing 7", "sim alertF1")
    print(f"{'model':<30}" + "".join(f"{h:>{width}}" for h in heads))
    print("-" * (30 + width * len(heads)))

    for name in names:
        s = summary[name]
        row = f"{display.get(name, name):<30}"
        row += f"{_cell(s['clean'], 'mae'):>{width}}"
        row += f"{_cell(s.get('sim', {}), 'mae'):>{width}}"
        row += f"{_cell(s.get('clean_s5', {}), 'mae'):>{width}}"
        row += f"{_cell(s.get('clean_s7', {}), 'mae'):>{width}}"
        row += f"{_cell(s.get('sim', {}), 'alert_macro_f1'):>{width}}"
        print(row)

    print()


def prepare_bank(rebuild_cache=False, n_train_aug=None, n_heldout_aug=None):
    if n_train_aug is None:
        n_train_aug = config.N_TRAIN_AUG
    if n_heldout_aug is None:
        n_heldout_aug = config.N_HELDOUT_AUG

    names = build_feature_cache(rebuild=rebuild_cache)
    build_aug_cache(names, n_train_aug, n_heldout_aug, rebuild=rebuild_cache)
    bank = FeatureBank(names, n_train_aug, n_heldout_aug)

    n_windows = len(bank.window_index(names, (config.SEQ_STRIDE,)))
    print(
        f"{len(names)} videos, {len(bank.target)} frames, {n_windows} windows "
        f"({config.SEQ_LEN} frames, stride {config.SEQ_STRIDE} = "
        f"{config.SEQ_STRIDE / config.VIDEO_FPS:.2f} s); "
        f"{bank.sources.shape[0]} train sources, "
        f"{0 if bank.heldout is None else bank.heldout.shape[0]} held-out"
    )
    return bank, n_windows
