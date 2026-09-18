"""
downsample_study.py
===================
Uds_xPass — Is rare-type miscalibration just scarcity? Downsample Ordinary
passes to rare-type sample sizes and recompute ECE/Brier.

If Ordinary passes at n=1,200 calibrate as badly as Through balls at
n=1,200, scarcity alone explains the gap. If they stay clean, something
intrinsic to the pass type matters.

Method: train the current event-only model on the largest cached seasons,
pool held-out predictions, then repeatedly subsample Ordinary predictions
to each rare type's n (200 resamples) and report mean ECE/Brier with 95%
intervals. No retraining per resample.

USAGE
-----
    conda run -n xpass python downsample_study.py
    conda run -n xpass python downsample_study.py --ids 90 42 --resamples 100

OUTPUTS (./poster_outputs/)
    downsample_summary.csv  - mean ECE/Brier per size + full-sample rare-type rows
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

from all_seasons import (  # noqa: E402
    DATA_DIR, OUT_DIR,
    pull_season_passes, add_weak_foot_flag, build_features,
    BASE_FEATS, train_and_predict, expected_calibration_error,
)

TYPE_FLAGS = {"Cross": "pass_cross_f", "Switch": "pass_switch_f",
              "Through ball": "pass_through_ball_f"}


def ptype(row):
    for name, flag in TYPE_FLAGS.items():
        if row.get(flag) == 1:
            return name
    return "Ordinary"


def largest_cached_ids(k=5):
    files = sorted(DATA_DIR.glob("passes_*.pkl"), key=lambda p: p.stat().st_size)
    return [p.stem.split("_")[1] for p in files[-k:]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--resamples", type=int, default=200)
    args = ap.parse_args()
    ids = args.ids or largest_cached_ids()
    print(f"Seasons (by cache id): {ids}")

    pooled = []
    for sid in ids:
        passes = pull_season_passes(11, int(sid), f"id={sid}")
        if passes.empty:
            continue
        out = train_and_predict(
            build_features(add_weak_foot_flag(passes)), BASE_FEATS, seed=42)
        if out is None:
            continue
        preds, _, _ = out
        preds = preds.copy()
        preds["pass_type"] = preds.apply(ptype, axis=1)
        pooled.append(preds)
    all_preds = pd.concat(pooled, ignore_index=True)

    rare_ns = {t: len(all_preds[all_preds.pass_type == t])
               for t in ["Cross", "Switch", "Through ball"]}
    print("Rare-type ns:", rare_ns)
    ord_preds = all_preds[all_preds.pass_type == "Ordinary"].reset_index(drop=True)
    yo = ord_preds["actual"].values
    po = ord_preds["pred"].values
    print(f"Ordinary pool: n={len(ord_preds):,} "
          f"full ECE={expected_calibration_error(yo, po):.4f}")

    rng = np.random.RandomState(0)
    rows = []
    sizes = sorted(set(rare_ns.values()) | {500, 2000, 10000})
    sizes = [s for s in sizes if s <= len(ord_preds)]
    for s in sizes:
        eces, briers = [], []
        for _ in range(args.resamples):
            idx = rng.choice(len(ord_preds), size=s, replace=False)
            eces.append(expected_calibration_error(yo[idx], po[idx]))
            briers.append(brier_score_loss(yo[idx], po[idx]))
        rows.append({"subset": f"Ordinary@n={s}", "n": s,
                     "brier_mean": float(np.mean(briers)),
                     "ece_mean": float(np.mean(eces)),
                     "ece_lo": float(np.percentile(eces, 2.5)),
                     "ece_hi": float(np.percentile(eces, 97.5))})
        print(f"Ordinary@n={s:<6,} ECE={np.mean(eces):.4f} "
              f"[{np.percentile(eces,2.5):.4f}, {np.percentile(eces,97.5):.4f}]")

    print("\nFull-sample rare types (same pool):")
    for t, n in rare_ns.items():
        sub = all_preds[all_preds.pass_type == t]
        yt, yp = sub["actual"].values, sub["pred"].values
        rows.append({"subset": t, "n": n,
                     "brier_mean": float(brier_score_loss(yt, yp)),
                     "ece_mean": float(expected_calibration_error(yt, yp)),
                     "ece_lo": float("nan"), "ece_hi": float("nan")})
        print(f"{t:12s} n={n:<6,} Brier={brier_score_loss(yt,yp):.4f} "
              f"ECE={expected_calibration_error(yt,yp):.4f}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "downsample_summary.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'downsample_summary.csv'}")


if __name__ == "__main__":
    main()
