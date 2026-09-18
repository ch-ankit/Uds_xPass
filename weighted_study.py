"""
weighted_study.py
=================
Uds_xPass — Exploration: does upweighting rare pass types help calibration?

Three identical XGBoost models except for training sample weights:
  - unweighted   : w = 1 (replicates the `current` ablation variant)
  - inv_type     : w ~ 1 / type_frequency (rare types count more in the loss)
  - sqrt_inv_type: w ~ 1 / sqrt(type_frequency) (milder version)

Held-out predictions are pooled across seasons; Brier/ECE per pass type
decide whether weighting helps the cause (rare-type calibration) and at
what cost to Ordinary passes.

USAGE
-----
    conda run -n xpass python weighted_study.py            # 5 largest cached seasons
    conda run -n xpass python weighted_study.py --ids 90 42 41

OUTPUTS (./poster_outputs/)
    weighted_summary.csv  - Brier/ECE/n per variant per pass type (+ overall AUC)
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")

from all_seasons import (  # noqa: E402
    OUT_DIR,
    pull_season_passes, add_weak_foot_flag, build_features,
    BASE_FEATS, expected_calibration_error,
)
import xgboost as xgb  # noqa: E402

# Mirrors calibration_study.py: Cut-back excluded (flag never populates).
TYPE_FLAGS = {
    "Cross": "pass_cross_f",
    "Switch": "pass_switch_f",
    "Through ball": "pass_through_ball_f",
}
TYPES = ["Ordinary", "Cross", "Switch", "Through ball"]


def pass_type_label(row):
    for name, flag in TYPE_FLAGS.items():
        if row.get(flag) == 1:
            return name
    return "Ordinary"


def type_weights(train_df, mode):
    """Sample weights from TRAIN-split type frequencies (no leakage)."""
    types = train_df.apply(pass_type_label, axis=1)
    counts = types.value_counts()
    n = len(train_df)
    if mode == "unweighted":
        return np.ones(n)
    if mode == "inv_type":
        w = n / (len(counts) * types.map(counts))
    elif mode == "sqrt_inv_type":
        w = np.sqrt(n / types.map(counts))
    else:
        raise ValueError(mode)
    return (w / w.mean()).values  # mean 1: same effective learning rate


def train_weighted(df, feats, weights, seed=42):
    match_ids = df["match_id"].unique()
    rng = np.random.RandomState(seed)
    shuffled = match_ids.copy()
    rng.shuffle(shuffled)
    split = max(1, int(0.8 * len(shuffled)))
    train_ids, test_ids = shuffled[:split], shuffled[split:]

    train = df[df.match_id.isin(train_ids)]
    test = df[df.match_id.isin(test_ids)]
    if test["completed"].nunique() < 2 or len(test) < 30:
        return None

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, gamma=1.0, reg_alpha=1.0,
        eval_metric="logloss", random_state=seed,
    )
    model.fit(train[feats].values, train["completed"].values,
              sample_weight=type_weights(train, weights))
    preds = model.predict_proba(test[feats].values)[:, 1]

    out = test.copy()
    out["pred"] = preds
    out["actual"] = out["completed"].values
    return out


# Fixed default: the five largest season caches (deterministic; size-based
# discovery flipped between ties across runs, changing the pooled results).
DEFAULT_IDS = ["1", "90", "26", "23", "27"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="*", default=None,
                        help="Season ids to use (default: five largest caches).")
    args = parser.parse_args()
    ids = args.ids or DEFAULT_IDS
    print(f"Seasons (by cache id): {ids}")

    variants = ["unweighted", "inv_type", "sqrt_inv_type"]
    pooled = {v: [] for v in variants}

    for sid in ids:
        passes = pull_season_passes(11, int(sid), f"id={sid}")
        if passes.empty:
            continue
        sname = passes["season_name"].iloc[0] if "season_name" in passes else sid
        passes = add_weak_foot_flag(passes)
        df = build_features(passes)
        if len(df) < 200:
            continue
        print(f"[{sname}] n={len(df):,}")
        for v in variants:
            out = train_weighted(df, BASE_FEATS, v)
            if out is None:
                continue
            out["pass_type"] = out.apply(pass_type_label, axis=1)
            pooled[v].append(out)

    rows = []
    for v in variants:
        if not pooled[v]:
            continue
        all_preds = pd.concat(pooled[v], ignore_index=True)
        print(f"\n{v}: overall AUC={roc_auc_score(all_preds['actual'], all_preds['pred']):.4f}")
        for ptype in TYPES:
            sub = all_preds[all_preds["pass_type"] == ptype]
            if len(sub) < 30:
                continue
            yt, yp = sub["actual"].values, sub["pred"].values
            brier = brier_score_loss(yt, yp)
            ece = expected_calibration_error(yt, yp)
            rows.append({"variant": v, "pass_type": ptype, "n": len(sub),
                         "brier_score": brier, "ece": ece})
            print(f"  {ptype:12s} n={len(sub):6,}  Brier={brier:.4f}  ECE={ece:.4f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "weighted_summary.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'weighted_summary.csv'}")
    print(summary.pivot(index="pass_type", columns="variant", values="ece").to_string())


if __name__ == "__main__":
    main()
