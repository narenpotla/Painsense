"""
cross_validate.py - honest accuracy numbers + the live model.

Video-wise stratified 5-fold cross-validation, repeated over 3 seeds,
on cached frozen ResNet18 features. Compares frame-only, causal
mean-pool and LSTM heads (trained with the FINAL_* settings in
painsense/config.py) against two no-image baselines (always 0,
train-mean). Scores every model on clean test frames, on held-out
simulated-webcam frames and at frame spacing 5 / 7.

Writes results/cross_validation.json and prints the tables.

--final also trains on all 40 videos and saves
    checkpoints/pain_intensity_features.pt  live model (FINAL_VARIANT)
    checkpoints/pain_intensity_image.pt     single-photo model ("frame")
(head(s) + backbone weights, so both run offline).

Run:
    python cross_validate.py --epochs 30 --seeds 3 --final
"""

import argparse
import json
import os

from painsense import config
from painsense.features import (
    BASELINES,
    DISPLAY,
    TRAINED_VARIANTS,
    decide_calibration,
    make_spec,
    paired,
    prepare_bank,
    print_table,
    run_cv,
    save_feature_checkpoint,
    summarize,
    train_spec,
)
from painsense.utils import seed_everything

# Saved by --final: (checkpoint path, variant, purpose).
FINAL_MODELS = (
    (config.FEATURE_CKPT, config.FINAL_VARIANT, "live"),
    (config.IMAGE_CKPT, "frame", "photo"),
)


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
                    help="train the live and photo models on all videos")
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

    runs, oof = run_cv(
        bank, specs, args.folds, args.seeds, args.epochs, args.seed,
        oof_keys=tuple({v for _, v, _ in FINAL_MODELS})
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

    calibration = {}

    for _, variant, purpose in FINAL_MODELS:
        cal = decide_calibration(*oof[variant])
        calibration[purpose] = {"variant": variant, **cal}
        print(
            f"\n{purpose} model ({variant}) out-of-fold calibration: fitted "
            f"scale={cal['fitted_scale']:.3f} bias={cal['fitted_bias']:.3f}; "
            f"OOF MAE raw {cal['oof_mae_raw']:.3f} vs with fit "
            f"{cal['oof_mae_with_fit']:.3f} -> "
            f"{'applied' if cal['applied'] else 'NOT applied (identity)'}"
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
        result["final_checkpoints"] = {}

        for path, variant, purpose in FINAL_MODELS:
            print(f"\n--final: training {purpose} model ({variant}) on all videos ...")
            spec = make_spec(variant, **settings)
            models = train_spec(spec, bank, bank.names, args.epochs, args.seed)
            save_feature_checkpoint(
                path, models, variant, result["training_settings"],
                calibration[purpose]["scale"], calibration[purpose]["bias"],
                purpose
            )
            result["final_checkpoints"][purpose] = path
            print(f"Saved {path} ({len(models)} head(s))")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Results -> {args.out}")


if __name__ == "__main__":
    main()
