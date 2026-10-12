"""
PainSense - robustness ablation for the live feature model.

Only the existing SynPAIN data is used. The question is which training
settings make the model hold up better on webcam-like input and on
imperfect live frame timing, judged by the same video-wise CV as
cv_intensity.py (5 folds x 3 seeds, identical folds for every setting).

Stage A, 12 settings = {frame, pool, lstm} x
    base         clean + flipped features, spacing 6
    aug          + webcam-style feature variants (half of the windows)
    jitter       training windows at spacing 5, 6 and 7
    aug+jitter   both
Stage B: the Stage-A winner with an ensemble of N heads.

Every setting is scored on clean test frames, on HELD-OUT simulated
webcam copies (a harsher distortion family than training uses, with
gamma and motion blur that training never sees), and at spacing 5 / 7.

Selection rules (fixed before running):
    A: average over the three variants; keep configs whose clean MAE is
       at most 0.05 above base; pick the lowest sim-webcam MAE.
    B: adopt the ensemble if it lowers sim-webcam MAE and clean MAE is
       at most 0.02 worse (both averaged over variants).
    Final variant: lowest sim-webcam MAE under the adopted settings.

Picking the best of several settings on the same folds is slightly
optimistic; the README says so.

Writes results/cv_robustness.json and prints the config.py values to
use. It does NOT edit config.py.

Run:
    python cv_robustness.py --epochs 30 --seeds 3
"""

import argparse
import json
import os

import numpy as np

import config
from cv_intensity import (
    TRAINED_VARIANTS,
    make_spec,
    paired,
    prepare_bank,
    print_table,
    run_cv,
    spec_name,
    summarize,
)
from utils import seed_everything

CONFIGS = {
    "base": {"aug": False, "strides": (6,)},
    "aug": {"aug": True, "strides": (6,)},
    "jitter": {"aug": False, "strides": (5, 6, 7)},
    "aug+jitter": {"aug": True, "strides": (5, 6, 7)},
}
CLEAN_TOLERANCE_A = 0.05
CLEAN_TOLERANCE_B = 0.02


def _avg(summary, names, cond):
    return float(np.mean([summary[n][cond]["mae"]["mean"] for n in names]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--ensemble", type=int, default=5,
                    help="ensemble size tried in stage B")
    ap.add_argument("--out", default=os.path.join(
        config.RESULTS_DIR, "cv_robustness.json"))
    args = ap.parse_args()

    seed_everything(args.seed)
    bank, n_windows = prepare_bank()

    # ---------------- Stage A ----------------
    specs_a = {}
    by_config = {}

    for cname, c in CONFIGS.items():
        by_config[cname] = []
        for variant in TRAINED_VARIANTS:
            spec = make_spec(variant, **c)
            specs_a[spec_name(spec)] = spec
            by_config[cname].append(spec_name(spec))

    print(f"\nSTAGE A: {len(specs_a)} settings")
    runs_a, _, _ = run_cv(
        bank, specs_a, args.folds, args.seeds, args.epochs, args.seed,
        baselines=False, label="[A] "
    )
    summary_a = summarize(runs_a, list(specs_a))
    print_table(summary_a, list(specs_a), len(runs_a))

    config_scores = {
        cname: {
            "clean_mae": _avg(summary_a, names, "clean"),
            "sim_mae": _avg(summary_a, names, "sim"),
            "clean_s5_mae": _avg(summary_a, names, "clean_s5"),
            "clean_s7_mae": _avg(summary_a, names, "clean_s7"),
        }
        for cname, names in by_config.items()
    }
    base_clean = config_scores["base"]["clean_mae"]
    eligible = [
        c for c, s in config_scores.items()
        if s["clean_mae"] <= base_clean + CLEAN_TOLERANCE_A
    ]
    winner = min(eligible, key=lambda c: config_scores[c]["sim_mae"])

    print("Stage A, averaged over frame / pool / LSTM (MAE):")
    print(f"{'config':<12}{'clean':>9}{'sim':>9}{'sp5':>9}{'sp7':>9}")
    for c, s in config_scores.items():
        mark = " <- winner" if c == winner else (
            "" if c in eligible else "  (clean too much worse)")
        print(
            f"{c:<12}{s['clean_mae']:9.3f}{s['sim_mae']:9.3f}"
            f"{s['clean_s5_mae']:9.3f}{s['clean_s7_mae']:9.3f}{mark}"
        )

    paired_a = {}
    for cname, names in by_config.items():
        if cname == "base":
            continue
        for name, base_name in zip(names, by_config["base"]):
            paired_a[f"{name} vs {base_name}"] = {
                cond: paired(runs_a, name, base_name, cond)
                for cond in ("clean", "sim")
            }

    # ---------------- Stage B ----------------
    win = CONFIGS[winner]
    specs_b = {}
    single = {}

    for variant in TRAINED_VARIANTS:
        spec = make_spec(variant, ensemble=args.ensemble, **win)
        specs_b[spec_name(spec)] = spec
        single[spec_name(spec)] = spec_name(make_spec(variant, **win))

    print(f"\nSTAGE B: '{winner}' with an ensemble of {args.ensemble} heads")
    runs_b, _, _ = run_cv(
        bank, specs_b, args.folds, args.seeds, args.epochs, args.seed,
        baselines=False, label="[B] "
    )
    summary_b = summarize(runs_b, list(specs_b))
    print_table(summary_b, list(specs_b), len(runs_b))

    ens_clean = _avg(summary_b, list(specs_b), "clean")
    ens_sim = _avg(summary_b, list(specs_b), "sim")
    one_clean = config_scores[winner]["clean_mae"]
    one_sim = config_scores[winner]["sim_mae"]
    use_ensemble = (
        ens_sim < one_sim and ens_clean <= one_clean + CLEAN_TOLERANCE_B
    )

    # Paired ensemble vs single head: same seeds/folds in the same order.
    paired_b = {}
    for ens_name, one_name in single.items():
        paired_b[f"{ens_name} vs {one_name}"] = {}
        for cond in ("clean", "sim"):
            diff = np.array([
                rb[ens_name][cond]["mae"] - ra[one_name][cond]["mae"]
                for ra, rb in zip(runs_a, runs_b)
            ])
            paired_b[f"{ens_name} vs {one_name}"][cond] = {
                "mean_diff": float(diff.mean()),
                "std_diff": float(diff.std()),
                "a_better_in": int((diff < 0).sum()),
                "runs": int(len(diff)),
            }

    print(
        f"Ensemble vs single head (avg over variants): clean "
        f"{ens_clean:.3f} vs {one_clean:.3f}, sim {ens_sim:.3f} vs "
        f"{one_sim:.3f} -> {'ADOPT' if use_ensemble else 'do not adopt'}"
    )

    # ---------------- Final variant ----------------
    if use_ensemble:
        final_summary, final_names = summary_b, list(specs_b)
    else:
        final_summary, final_names = summary_a, by_config[winner]

    best = min(final_names, key=lambda n: final_summary[n]["sim"]["mae"]["mean"])
    final_variant = (specs_b if use_ensemble else specs_a)[best]["variant"]

    recommended = {
        "FINAL_VARIANT": final_variant,
        "FINAL_AUG": win["aug"],
        "FINAL_STRIDES": list(win["strides"]),
        "FINAL_ENSEMBLE": args.ensemble if use_ensemble else 1,
    }

    print("\nVariants under the adopted settings (sim-webcam MAE):")
    for n in final_names:
        s = final_summary[n]
        print(
            f"  {n:<28} clean {s['clean']['mae']['mean']:.3f}  "
            f"sim {s['sim']['mae']['mean']:.3f} +- "
            f"{s['sim']['mae']['std']:.3f}"
        )

    print("\nRecommended config.py values:")
    for key, value in recommended.items():
        print(f"  {key} = {tuple(value) if isinstance(value, list) else repr(value)}")

    result = {
        "target": "expression intensity on a synthetic morph scale (0-10)",
        "metrics_on": "last frame of each window",
        "n_videos": len(bank.names),
        "n_windows": n_windows,
        "epochs": args.epochs,
        "folds": args.folds,
        "seeds": [args.seed + i for i in range(args.seeds)],
        "n_train_aug": config.N_TRAIN_AUG,
        "n_heldout_aug": config.N_HELDOUT_AUG,
        "rules": {
            "stage_a": f"configs averaged over variants; clean MAE <= base "
                       f"+ {CLEAN_TOLERANCE_A}; lowest sim-webcam MAE wins",
            "stage_b": f"adopt ensemble if sim-webcam MAE drops and clean "
                       f"MAE <= single + {CLEAN_TOLERANCE_B}",
            "variant": "lowest sim-webcam MAE under adopted settings",
        },
        "stage_a": {
            "summary": summary_a,
            "config_scores": config_scores,
            "eligible": eligible,
            "winner": winner,
            "paired_vs_base": paired_a,
        },
        "stage_b": {
            "ensemble": args.ensemble,
            "summary": summary_b,
            "paired_vs_single": paired_b,
            "adopted": bool(use_ensemble),
        },
        "recommended": recommended,
        "runs_a": runs_a,
        "runs_b": runs_b,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Results -> {args.out}")


if __name__ == "__main__":
    main()
