"""
poster_ablation_and_recalibration.py
======================================
Uds_xPass — Poster Extension: Does Footedness Help Calibration? + The Fix
----------------------------------------------------------------------------

Two follow-up questions from the poster review:

  1. ABLATION — "Does feeding footedness help calibration, especially on the
     rare/hard pass types (Cross, Switch, Through ball) where it's worst?"
     We train three versions of the event-only xPass model, identical except
     for footedness features, and compare Brier/ECE per pass type:
       - no_footedness      : weak_foot_feat and wf_x_cross removed entirely
       - current             : the notebook's existing feature set
                                (weak_foot_feat + a single wf_x_cross interaction)
       - full_interactions   : current + wf_x_switch + wf_x_through_ball
                                (the model can now learn a *type-specific*
                                weak-foot penalty instead of one blanket one)

  2. RECALIBRATION — "What can actually be done about the miscalibration?"
     Feature engineering alone rarely fully fixes calibration on classes with
     only 1-4k examples. We fit per-pass-type isotonic regression on half of
     the held-out predictions (the best-performing variant from the ablation)
     and evaluate the correction on the other half, producing a clean
     before/after reliability diagram.

This script REUSES the cached per-season data from
`poster_all_seasons_analysis.py` (./data/passes_<season_id>.parquet) and
imports its helper functions directly, so nothing needs to be re-pulled from
StatsBomb — run poster_all_seasons_analysis.py at least once first (or just
make sure ./data/ is populated).

Note on scope: per the poster decision, Cut-back is EXCLUDED from this
analysis (StatsBomb's `pass_cut_back` flag never populates in this open-data
pull, so there's effectively no signal to ablate or recalibrate there).

USAGE
-----
    # Run from the same directory as poster_all_seasons_analysis.py, with
    # ./data/ already populated from a prior run of that script.
    python poster_ablation_and_recalibration.py

OUTPUTS  (written to ./poster_outputs/)
----------------------------------------
    ablation_calibration_by_pass_type.png   - grouped bars: Brier & ECE per pass type, 3 feature variants
    ablation_summary.csv                    - same data as the plot, in table form
    recalibration_before_after.png          - isotonic-corrected reliability diagrams, best variant
    recalibration_summary.csv               - Brier/ECE before vs. after recalibration, per pass type
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

# ── Reuse everything from the main poster script instead of duplicating it ──
from all_seasons import (   # noqa: E402
    COMP_ID, DATA_DIR, OUT_DIR,
    get_laliga_seasons, pull_season_passes, add_weak_foot_flag, build_features,
    BASE_FEATS, train_and_predict, expected_calibration_error,
)

# Explicitly excludes Cut-back — see module docstring.
TYPES = ["Ordinary", "Cross", "Switch", "Through ball"]
TYPE_FLAGS = {
    "Cross": "pass_cross_f",
    "Switch": "pass_switch_f",
    "Through ball": "pass_through_ball_f",
}


def pass_type_label(row):
    for name, flag in TYPE_FLAGS.items():
        if row.get(flag) == 1:
            return name
    return "Ordinary"


# ─────────────────────────────────────────────────────────────────────────
# STEP 1 — Extra interaction features build_features() doesn't already make
# ─────────────────────────────────────────────────────────────────────────
def extend_features(df):
    d = df.copy()
    d["wf_x_switch"] = d["weak_foot_feat"] * d["pass_switch_f"]
    d["wf_x_through_ball"] = d["weak_foot_feat"] * d["pass_through_ball_f"]
    return d


FEATS_NO_FOOT = [f for f in BASE_FEATS if f not in ("weak_foot_feat", "wf_x_cross")]
FEATS_CURRENT = list(BASE_FEATS)
FEATS_FULL_INTERACT = list(BASE_FEATS) + ["wf_x_switch", "wf_x_through_ball"]

VARIANTS = {
    "no_footedness": FEATS_NO_FOOT,
    "current": FEATS_CURRENT,
    "full_interactions": FEATS_FULL_INTERACT,
}
VARIANT_LABELS = {
    "no_footedness": "No footedness",
    "current": "Current (weak_foot + wf×cross)",
    "full_interactions": "Full interactions (weak_foot × all types)",
}
VARIANT_COLORS = {
    "no_footedness": "#9CA3AF",
    "current": "#1D4ED8",
    "full_interactions": "#059669",
}


# ─────────────────────────────────────────────────────────────────────────
# STEP 2 — Load every cached season once, train all 3 variants per season
# ─────────────────────────────────────────────────────────────────────────
def load_all_seasons_featurized():
    """Pull (from cache) + featurize every season once, so we don't redo
    feature engineering 3x per variant."""
    seasons = get_laliga_seasons()
    dfs = []
    for _, row in seasons.iterrows():
        comp_id, season_id, season_name = (
            row["competition_id"], row["season_id"], row["season_name"]
        )
        passes = pull_season_passes(comp_id, season_id, season_name)
        if passes.empty:
            continue
        passes = add_weak_foot_flag(passes)
        df = build_features(passes)
        if len(df) < 200:
            continue
        df = extend_features(df)
        dfs.append((season_name, df))
    return dfs


def run_ablation(season_dfs):
    """For each variant, pool held-out predictions across all seasons, then
    compute Brier/ECE per pass type. Returns (summary_df, pooled_preds_by_variant)."""
    pooled_preds = {v: [] for v in VARIANTS}

    for season_name, df in season_dfs:
        print(f"  Training all 3 variants for {season_name}...")
        for variant, feats in VARIANTS.items():
            out = train_and_predict(df, feats)
            if out is None:
                continue
            preds, auc, brier = out
            preds = preds.copy()
            preds["pass_type"] = preds.apply(pass_type_label, axis=1)
            pooled_preds[variant].append(preds)

    rows = []
    for variant in VARIANTS:
        if not pooled_preds[variant]:
            continue
        all_preds = pd.concat(pooled_preds[variant], ignore_index=True)
        for ptype in TYPES:
            sub = all_preds[all_preds["pass_type"] == ptype]
            if len(sub) < 30:
                continue
            y_true, y_pred = sub["actual"].values, sub["pred"].values
            rows.append({
                "variant": variant,
                "pass_type": ptype,
                "n": len(sub),
                "brier_score": brier_score_loss(y_true, y_pred),
                "ece": expected_calibration_error(y_true, y_pred),
            })
        pooled_preds[variant] = all_preds  # collapse list -> single df

    return pd.DataFrame(rows), pooled_preds


def plot_ablation(summary_df, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    variants = [v for v in VARIANTS if v in summary_df["variant"].unique()]
    x = np.arange(len(TYPES))
    width = 0.8 / len(variants)

    for metric, ax in zip(["brier_score", "ece"], axes):
        for i, variant in enumerate(variants):
            vals = [
                summary_df.query("variant == @variant and pass_type == @t")[metric].mean()
                for t in TYPES
            ]
            ax.bar(
                x + i * width, vals, width,
                label=VARIANT_LABELS[variant], color=VARIANT_COLORS[variant],
            )
        ax.set_xticks(x + width * (len(variants) - 1) / 2)
        ax.set_xticklabels(TYPES)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{metric.replace('_', ' ').title()} by Pass Type — Feature Ablation")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(loc="upper left", fontsize=9)
    plt.suptitle(
        "Does Footedness Help Calibration? — event-only model, pooled across all La Liga seasons\n"
        "(lower = better; Cut-back excluded — see script docstring)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# STEP 3 — Isotonic recalibration of the best variant, per pass type
# ─────────────────────────────────────────────────────────────────────────
def pick_best_variant(summary_df):
    """Best = lowest mean ECE across the three hard pass types (Cross,
    Switch, Through ball) — Ordinary is already near-perfect everywhere,
    so it shouldn't drive the choice."""
    hard_types = ["Cross", "Switch", "Through ball"]
    scores = (
        summary_df[summary_df["pass_type"].isin(hard_types)]
        .groupby("variant")["ece"].mean()
        .sort_values()
    )
    print("\nMean ECE on hard pass types (Cross/Switch/Through ball), by variant:")
    print(scores.to_string())
    return scores.index[0]


def recalibrate_and_plot(all_preds, path_plot, path_csv, seed=42):
    """Split each pass type's pooled held-out predictions 50/50: fit an
    isotonic mapping on one half, evaluate original vs. recalibrated
    calibration on the other half (so the 'after' numbers aren't just
    fit-set overfitting)."""
    rng = np.random.RandomState(seed)
    rows = []

    fig, axes = plt.subplots(1, len(TYPES), figsize=(4 * len(TYPES), 4), sharey=True)

    for ax, ptype in zip(axes, TYPES):
        sub = all_preds[all_preds["pass_type"] == ptype].sample(
            frac=1.0, random_state=seed
        )  # shuffle
        if len(sub) < 60:
            ax.set_title(f"{ptype}\n(n={len(sub)}, too few to split)")
            continue

        half = len(sub) // 2
        fit_set, eval_set = sub.iloc[:half], sub.iloc[half:]

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(fit_set["pred"].values, fit_set["actual"].values)
        eval_recal = iso.predict(eval_set["pred"].values)

        y_true = eval_set["actual"].values
        y_raw = eval_set["pred"].values

        brier_before = brier_score_loss(y_true, y_raw)
        brier_after = brier_score_loss(y_true, eval_recal)
        ece_before = expected_calibration_error(y_true, y_raw)
        ece_after = expected_calibration_error(y_true, eval_recal)

        frac_raw, mean_raw = calibration_curve(y_true, y_raw, n_bins=8, strategy="quantile")
        frac_recal, mean_recal = calibration_curve(y_true, eval_recal, n_bins=8, strategy="quantile")

        ax.plot(mean_raw, frac_raw, "o-", color="#DC2626", lw=2, ms=5, label="Before")
        ax.plot(mean_recal, frac_recal, "o-", color="#059669", lw=2, ms=5, label="After (isotonic)")
        ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", lw=1)
        ax.set_title(
            f"{ptype}\nn_eval={len(eval_set):,}\n"
            f"Brier {brier_before:.3f}→{brier_after:.3f}  ECE {ece_before:.3f}→{ece_after:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("Mean predicted")
        ax.spines[["top", "right"]].set_visible(False)

        rows.append({
            "pass_type": ptype,
            "n_eval": len(eval_set),
            "brier_before": brier_before,
            "brier_after": brier_after,
            "ece_before": ece_before,
            "ece_after": ece_after,
        })

    axes[0].set_ylabel("Fraction of positives")
    axes[0].legend(fontsize=9, loc="upper left")
    plt.suptitle(
        "Isotonic Recalibration by Pass Type — before vs. after "
        "(fit on half the held-out predictions, evaluated on the other half)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(path_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(path_csv, index=False)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────
def main():
    print("Loading + featurizing every cached season...")
    season_dfs = load_all_seasons_featurized()
    print(f"Loaded {len(season_dfs)} seasons.\n")

    print("=== Ablation: no_footedness vs current vs full_interactions ===")
    summary_df, pooled_preds = run_ablation(season_dfs)
    summary_df.to_csv(OUT_DIR / "ablation_summary.csv", index=False)
    plot_ablation(summary_df, OUT_DIR / "ablation_calibration_by_pass_type.png")
    print(f"\nSaved: {OUT_DIR / 'ablation_summary.csv'}")
    print(f"Saved: {OUT_DIR / 'ablation_calibration_by_pass_type.png'}")
    print(summary_df.pivot(index="pass_type", columns="variant", values="ece").to_string())

    best_variant = pick_best_variant(summary_df)
    print(f"\nBest variant on hard pass types: {best_variant}")

    print(f"\n=== Isotonic recalibration of '{best_variant}' ===")
    all_preds = pooled_preds[best_variant]
    recal_df = recalibrate_and_plot(
        all_preds,
        OUT_DIR / "recalibration_before_after.png",
        OUT_DIR / "recalibration_summary.csv",
    )
    print(f"Saved: {OUT_DIR / 'recalibration_before_after.png'}")
    print(f"Saved: {OUT_DIR / 'recalibration_summary.csv'}")
    print(recal_df.to_string(index=False))

    print(f"\nAll ablation/recalibration assets written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()