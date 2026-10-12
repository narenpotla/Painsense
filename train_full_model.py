"""
train_full_model.py - train the full CNN-LSTM on face images (slow, optional).

ResNet18 on every face crop -> LSTM -> 0-10 score per frame, trained on
image windows (SEQ_LEN frames, SEQ_STRIDE apart) with one augmentation
per clip. Video-level stratified split 28 train / 6 val / 6 test:
the checkpoint is chosen and the score calibration is fitted on val,
test is scored once and written next to the checkpoint.

ResNet layer4 stays frozen unless --finetune-layer4 (lr 1e-4; head 1e-3).
Backbone BatchNorm layers stay in eval mode.

This is NOT the live model (that is cross_validate.py --final). The live
demo only falls back to this checkpoint if the feature model is missing.

Run:
    python train_full_model.py --epochs 3
    python train_full_model.py --epochs 3 --finetune-layer4 --ckpt checkpoints/full_l4.pt
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from painsense import config
from painsense.data import (
    IntensitySequenceDataset,
    export_intensity_manifest,
    split_dataset_3way
)
from painsense.models import PainIntensityLSTM
from painsense.utils import seed_everything


# =========================================================
# Utility
# =========================================================

def maybe_limit(dataset, limit, seed=None):
    """
    Limit the number of samples AFTER the video-level
    train/validation split.

    This is important because split_dataset() needs access
    to the original dataset's video-group information.

    The subset is drawn with its own seeded generator, so it is
    reproducible and does not disturb the global RNG stream.
    """

    if limit is not None and limit < len(dataset):

        generator = torch.Generator()
        generator.manual_seed(
            config.SEED if seed is None else int(seed)
        )

        idx = torch.randperm(
            len(dataset),
            generator=generator
        )[:limit].tolist()

        return Subset(
            dataset,
            idx
        )

    return dataset


# =========================================================
# Intensity regression on generated 0-10 targets
# ("expression intensity on a synthetic morph scale")
# =========================================================

def _predict_all(model, loader):
    """Run the model; return (pred, target) tensors of shape (N, T)."""

    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(config.DEVICE)).cpu())
            targets.append(y)

    return torch.cat(preds), torch.cat(targets)


def train_intensity(
    epochs,
    limit=None,
    sequences_per_video=6,
    seed=None,
    finetune_layer4=False,
    ckpt_path=None
):
    """
    Train the full CNN-LSTM intensity model.

    Video-level train / val / test split (stratified):
      * the checkpoint is chosen by validation MAE,
      * the score calibration is fitted on validation windows,
      * the test videos are scored once at the end.

    layer4 stays frozen unless finetune_layer4=True (then lr 1e-4,
    head lr 1e-3).
    """

    seed = seed_everything(seed)

    if ckpt_path is None:
        ckpt_path = config.INTENSITY_CKPT

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    print()
    print("======================================")
    print("INTENSITY: expression intensity, synthetic morph scale (0-10)")
    print("======================================")
    print(f"Seed: {seed}")

    export_intensity_manifest()

    full_ds = IntensitySequenceDataset(
        sequences_per_video=sequences_per_video,
        train=True
    )

    train_ds, val_ds, test_ds, names = split_dataset_3way(
        full_ds, seed=seed
    )
    train_ds = maybe_limit(train_ds, limit, seed=seed)

    print(f"Training sequences used: {len(train_ds)}")
    print(f"Validation sequences: {len(val_ds)}")
    print(f"Test sequences: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=4, shuffle=False, num_workers=0
    )

    model = PainIntensityLSTM().to(config.DEVICE)

    # ImageNet features, not the binary pain checkpoint.
    # That checkpoint collapsed toward No Pain.
    print("Encoder: ImageNet ResNet18")

    # Backbone frozen by default. layer4 is opt-in and uses a small lr.
    model.encoder.freeze_backbone(True)

    param_groups = [{
        "params": (
            list(model.encoder.proj.parameters())
            + list(model.lstm.parameters())
            + list(model.head.parameters())
        ),
        "lr": config.LR_HEAD
    }]

    if finetune_layer4:
        layer4 = []

        for name, param in model.encoder.backbone.named_parameters():
            if name.startswith("layer4"):
                param.requires_grad = True
                layer4.append(param)

        param_groups.append({"params": layer4, "lr": config.LR_LAYER4})
        print(
            f"Fine-tuning layer4 (lr {config.LR_LAYER4}), "
            f"head lr {config.LR_HEAD}"
        )
    else:
        print(f"Backbone fully frozen, head lr {config.LR_HEAD}")

    optimizer = torch.optim.Adam(param_groups)

    loss_fn = nn.MSELoss()
    best_mae = float("inf")

    for epoch in range(epochs):
        model.train()   # backbone BN layers stay in eval mode
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / max(len(train_ds), 1)

        pred, target = _predict_all(model, val_loader)
        val_mae = (pred - target).abs().mean().item()

        print(
            f"[INTENSITY] epoch {epoch + 1}/{epochs} "
            f"train_mse={train_loss:.4f} val_mae={val_mae:.4f}"
        )

        if val_mae <= best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved checkpoint -> {ckpt_path}")

    print()
    print(f"Best validation MAE: {best_mae:.4f}")

    model.load_state_dict(torch.load(
        ckpt_path,
        map_location=config.DEVICE
    ))

    # Calibration is fitted on VALIDATION windows, never on train.
    scale, bias = save_score_calibration(
        model, val_loader, ckpt_path=ckpt_path
    )

    # Test videos: scored once.
    pred, target = _predict_all(model, test_loader)
    lo, hi = config.PAIN_SCORE_MIN, config.PAIN_SCORE_MAX
    raw = pred.clamp(lo, hi)
    cal = (scale * pred + bias).clamp(lo, hi)

    report = {
        "seed": seed,
        "finetune_layer4": bool(finetune_layer4),
        "epochs": epochs,
        "val_mae_best": best_mae,
        "test_videos": names["test"],
        "test_mae_last_frame_raw": (raw[:, -1] - target[:, -1]).abs().mean().item(),
        "test_mae_last_frame_calibrated": (cal[:, -1] - target[:, -1]).abs().mean().item(),
        "test_mae_all_steps_raw": (raw - target).abs().mean().item(),
        "test_mae_always_zero_last_frame": target[:, -1].abs().mean().item(),
        "scale": scale,
        "bias": bias,
    }

    print()
    print("TEST (scored once):")
    for key in (
        "test_mae_last_frame_raw",
        "test_mae_last_frame_calibrated",
        "test_mae_all_steps_raw",
        "test_mae_always_zero_last_frame",
    ):
        print(f"  {key}: {report[key]:.4f}")

    out = os.path.splitext(ckpt_path)[0] + "_test.json"

    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"  report -> {out}")


def save_score_calibration(model, loader, ckpt_path=None):
    """
    Fit score = scale * raw + bias by least squares on the last frame
    of every window in `loader` so the 0-10 output uses the full
    range. Pass the VALIDATION loader, not the training loader.

    Returns (scale, bias).
    """

    if ckpt_path is None:
        ckpt_path = config.INTENSITY_CKPT

    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(config.DEVICE))[:, -1]
            target = y.to(config.DEVICE)[:, -1]
            preds.extend(pred.detach().cpu().tolist())
            targets.extend(target.detach().cpu().tolist())

    count = max(len(preds), 1)
    mean_pred = sum(preds) / count
    mean_target = sum(targets) / count
    variance = sum((pred - mean_pred) ** 2 for pred in preds)
    covariance = sum(
        (pred - mean_pred) * (target - mean_target)
        for pred, target in zip(preds, targets)
    )
    scale = covariance / variance if variance else 1.0

    if scale <= 0:
        print(
            "WARNING: fitted calibration slope is not positive; "
            "using identity calibration."
        )
        scale = 1.0
        bias = 0.0
    else:
        bias = mean_target - scale * mean_pred

    path = config.calibration_path_for(ckpt_path)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"scale": scale, "bias": bias}, handle)

    print(
        f"Score calibration (fit on validation): "
        f"scale={scale:.3f} bias={bias:.3f} -> {path}"
    )

    return scale, bias


def main():
    ap = argparse.ArgumentParser(
        description="Train the full CNN-LSTM on face images."
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None,
                    help="limit training windows for a quick test run "
                         "(applied after the video-level split)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--finetune-layer4", action="store_true",
                    help="also train ResNet layer4 (lr 1e-4; head lr 1e-3)")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path to write "
                         "(default: checkpoints/pain_intensity.pt)")
    args = ap.parse_args()

    seed_everything(args.seed)
    print(f"Using device: {config.DEVICE}")

    train_intensity(
        epochs=args.epochs,
        limit=args.limit,
        seed=args.seed,
        finetune_layer4=args.finetune_layer4,
        ckpt_path=args.ckpt
    )


if __name__ == "__main__":
    main()
