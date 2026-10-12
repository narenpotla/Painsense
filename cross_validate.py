"""
cross_validate.py - honest accuracy numbers + the live model.

Video-wise stratified 5-fold cross-validation, repeated over 3 seeds,
on cached frozen ResNet18 features. Compares frame-only, causal
mean-pool and LSTM heads (trained with the FINAL_* settings in
painsense/config.py) against two no-image baselines (always 0,
train-mean). Scores every model on clean test frames, on held-out
simulated-webcam frames and at frame spacing 5 / 7.

Writes results/cross_validation.json and prints the tables.

--final also trains FINAL_VARIANT on all 40 videos and saves the live
model checkpoints/pain_intensity_features.pt (head(s) + backbone
weights, so the demo runs offline).

Run:
    python cross_validate.py --epochs 30 --seeds 3 --final
"""

import argparse
import json
import os

import numpy as np
import torch

from painsense import config
from painsense.features import (
    BASELINES,
    DISPLAY,
    TRAINED_VARIANTS,
    fit_linear,
    make_spec,
    paired,
    prepare_bank,
    print_table,
    run_cv,
    summarize,
    train_spec,
)
from painsense.models import imagenet_backbone
from painsense.utils import seed_everything


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=config.SEED,
                    help="first seed; seeds are seed, seed+1, ...")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--final", action="store_true",
                    help=f"train {config.FINAL_VARIANT} on all videos and save "
                         f"{os.path.basename(config.FEATURE_CKPT)}")
    ap.add_argument("--out", default=os.path.join(
        config.RESULTS_DIR, "cross_validation.json"))
    args = ap.parse_args()

    seed_everything(args.seed)
    print(f"Device: {config.DEVICE}")

    bank, n_windows = prepare_bank(args.rebuild_cache)

    settings = {
        "aug": config.FINAL_AUG,
        "strides": tuple(config.FINAL_STRIDES),
        "ensemble": config.FINAL_ENSEMBLE,
    }
    specs = {v: make_spec(v, **settings) for v in TRAINED_VARIANTS}
    print(
        f"Training settings (config.FINAL_*): aug={settings['aug']} "
        f"strides={settings['strides']} ensemble={settings['ensemble']}"
    )

    runs, oof_pred, oof_target = run_cv(
        bank, specs, args.folds, args.seeds, args.epochs, args.seed,
        oof_key=config.FINAL_VARIANT
    )

    names = list(TRAINED_VARIANTS) + list(BASELINES)
    summary = summarize(runs, names)
    comparisons = {}

    for cond in ("clean", "sim"):
        comparisons[cond] = {
            "lstm_minus_frame": paired(runs, "lstm", "frame", cond),
            "lstm_minus_pool": paired(runs, "lstm", "pool", cond),
            "pool_minus_frame": paired(runs, "pool", "frame", cond),
        }

    print_table(summary, names, len(runs), DISPLAY)

    for cond in ("clean", "sim"):
        print(f"Paired MAE difference, {cond} (negative = first is better):")
        for key, c in comparisons[cond].items():
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
        "note": "least-squares fit on pooled out-of-fold predictions of "
                "FINAL_VARIANT; applied only if it lowers out-of-fold MAE",
    }
    print(
        f"\n{config.FINAL_VARIANT} out-of-fold calibration: fitted "
        f"scale={fit_scale:.3f} bias={fit_bias:.3f}; OOF MAE raw "
        f"{mae_raw:.3f} vs with fit {mae_cal:.3f} -> "
        f"{'applied' if use_fit else 'NOT applied (identity)'}"
    )

    result = {
        "target": "expression intensity on a synthetic morph scale (0-10)",
        "metrics_on": "last frame of each window",
        "conditions": {
            "clean": "real test frames, spacing 6",
            "sim": "held-out webcam-style degraded copies of the test frames "
                   "(different distortion family from training)",
            "clean_s5": "clean test frames, spacing 5",
            "clean_s7": "clean test frames, spacing 7",
        },
        "n_videos": len(bank.names),
        "n_windows": n_windows,
        "seq_len": config.SEQ_LEN,
        "stride_frames": config.SEQ_STRIDE,
        "epochs": args.epochs,
        "folds": args.folds,
        "seeds": [args.seed + i for i in range(args.seeds)],
        "features": "frozen ImageNet ResNet18",
        "training_settings": {**settings, "strides": list(settings["strides"])},
        "summary": summary,
        "paired_mae_diff": comparisons,
        "calibration_oof": calibration,
        "runs": runs,
    }

    if args.final:
        print(f"\n--final: training {config.FINAL_VARIANT} on all videos ...")
        spec = make_spec(config.FINAL_VARIANT, **settings)
        models = train_spec(spec, bank, bank.names, args.epochs, args.seed)
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        torch.save(
            {
                "state_dict": models[0].state_dict(),
                "state_dicts": [m.state_dict() for m in models],
                "variant": config.FINAL_VARIANT,
                "seq_len": config.SEQ_LEN,
                "stride": config.SEQ_STRIDE,
                "training_settings": result["training_settings"],
                "scale": scale,
                "bias": bias,
                # Frozen ImageNet weights stored with the head so the
                # live demo needs no download.
                "backbone_state_dict": imagenet_backbone().state_dict(),
            },
            config.FEATURE_CKPT
        )
        result["final_checkpoint"] = config.FEATURE_CKPT
        print(f"Saved {config.FEATURE_CKPT} ({len(models)} head(s))")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Results -> {args.out}")


if __name__ == "__main__":
    main()
