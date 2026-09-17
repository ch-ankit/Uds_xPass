"""
all_seasons.py
==============
Uds_xPass — Poster Extension: All La Liga Seasons + Pass-Type Calibration
--------------------------------------------------------------------------

This script extends `laliga_360_xpass.ipynb` (which only covers La Liga
2020/21, the one season with StatsBomb 360 freeze-frame data) in two ways
for the final poster:

  1. HYPOTHESIS TEST ACROSS ALL SEASONS
     "Passes hit with a player's weak foot are completed less often than
     passes hit with their strong foot" — tested on every La Liga season
     available in StatsBomb's open data (not just 2020/21), using the
     event-only feature set (no 360 data needed/available outside 2020/21).
     Per season we report:
       - raw completion-rate gap (strong vs weak foot)
       - a chi-square test of independence
       - a logistic-regression coefficient on `weak_foot`, controlling for
         pass difficulty (length, pressure, cross/switch/through-ball/cut-back,
         zone, angle) — this is the effect size that matters, since weak-foot
         passes are also more likely to be attempted in easier situations.
     Results are pooled into a forest plot + summary CSV for the poster.

  2. CALIBRATION BY PASS TYPE, ACROSS ALL SEASONS
     For each season we train an event-only xPass model (same feature
     recipe as the notebook's `build_features(use_360=False)`), get held-out
     predictions, then break out calibration (predicted vs. actual
     completion rate, Brier score, Expected Calibration Error) by pass
     type: Ordinary / Cross / Switch / Through ball / Cut-back, and pool
     across all seasons for one clean reliability-diagram figure per type.

  Optionally (--with-360), it also reruns the notebook's original
  event-only vs. 360-enriched comparison for La Liga 2020/21 specifically,
  since 360 freeze-frame data is only available for that season in the
  open dataset — useful as a "does richer context change the story"
  sanity check on the poster.

USAGE
-----
    pip install statsbombpy xgboost scikit-learn scipy statsmodels pandas numpy matplotlib seaborn

    # Full run across every La Liga season in the open data (slow — pulls
    # event data for every match in every season the first time; cached
    # to ./data/ afterwards)
    python all_seasons.py

    # Quicker iteration: limit to a few seasons while testing the script
    python all_seasons.py --seasons 2020/2021 2019/2020 2018/2019

    # Include the bonus 2020/21 event-only vs 360 comparison
    python all_seasons.py --with-360

OUTPUTS  (written to ./poster_outputs/)
----------------------------------------
    weak_foot_hypothesis_by_season.csv     - per-season completion gap, chi2, logistic coef/CI/p
    weak_foot_forest_plot.png              - forest plot of the controlled weak-foot effect, all seasons
    calibration_by_pass_type.png           - reliability diagrams, one panel per pass type, pooled seasons
    calibration_summary_by_season.csv      - Brier / ECE / AUC per season (event-only model)
    calibration_summary_by_pass_type.csv   - Brier / ECE / n per pass type, pooled across seasons
    [optional] event_vs_360_2020_21.png    - only with --with-360

NOTES
-----
- Cut-back: StatsBomb's `pass_cut_back` flag never populates in this
  open-data pull, so it is kept as a (zero-variance) feature but dropped
  from the per-season logit controls and excluded from the ablation in
  calibration_study.py.
- 1973/74: n=469 with incomplete body-part tagging — kept in the CSV for
  transparency but excluded from the headline claim (see report).
- Canonical runnable. The archived notebook
  (archive/poster_all_seasons_analysis.ipynb) is a frozen copy; edit here.

NOTE ON RUNTIME
----------------
StatsBomb's open data has ~20 La Liga seasons on record (mostly Messi-era
Barcelona matches, ~30-40 matches per season, plus the full 2020/21
season at ~35 matches). Pulling + featurizing everything can take a
while on a slow connection — that's why every season's raw events are
cached to ./data/passes_<season_id>.parquet (or .pkl if pyarrow is
missing) the first time they're pulled, so re-running the script (e.g.
after tweaking a plot) is fast.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss
from statsbombpy import sb

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("This script needs xgboost: pip install xgboost") from e

try:
    import statsmodels.api as sm
except ImportError as e:
    raise ImportError("This script needs statsmodels: pip install statsmodels") from e

COMP_ID = 11  # La Liga, per StatsBomb open data
DATA_DIR = Path("data")
OUT_DIR = Path("poster_outputs")
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

PASS_TYPE_FLAGS = {
    "Cross": "pass_cross_f",
    "Switch": "pass_switch_f",
    "Through ball": "pass_through_ball_f",
    "Cut-back": "pass_cut_back_f",
}


# ─────────────────────────────────────────────────────────────────────────
# STEP 0 — Discover every La Liga season in the open data
# ─────────────────────────────────────────────────────────────────────────
def get_laliga_seasons():
    """Return a DataFrame of every (competition_id, season_id, season_name)
    for men's La Liga available in StatsBomb's open data."""
    comps = sb.competitions()
    laliga = comps[
        (comps["competition_id"] == COMP_ID) & (comps["competition_gender"] == "male")
    ].sort_values("season_name")
    return laliga[["competition_id", "season_id", "season_name"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
# STEP 1 — Pull (and cache) all passes for one season
# ─────────────────────────────────────────────────────────────────────────
def pull_season_passes(competition_id, season_id, season_name):
    cache_parquet = DATA_DIR / f"passes_{season_id}.parquet"
    cache_pkl = DATA_DIR / f"passes_{season_id}.pkl"
    if cache_parquet.exists():
        return pd.read_parquet(cache_parquet)
    if cache_pkl.exists():
        return pd.read_pickle(cache_pkl)

    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    print(f"  [{season_name}] {len(matches)} matches — pulling events...")

    all_events = []
    for i, mid in enumerate(matches.match_id):
        try:
            ev = sb.events(match_id=mid)
        except Exception as e:
            print(f"    match {mid} failed: {e}")
            continue
        ev["match_id"] = mid
        all_events.append(ev[ev["type"] == "Pass"].copy())
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(matches)} matches pulled")

    if not all_events:
        return pd.DataFrame()

    passes = pd.concat(all_events, ignore_index=True)
    passes["season_id"] = season_id
    passes["season_name"] = season_name

    # Parquet needs list-columns (location, pass_end_location) to be pickled
    # as object — pandas/pyarrow handles this fine, but fall back to pickle
    # if pyarrow isn't available (this is why ./data/ holds .pkl files).
    try:
        passes.to_parquet(cache_parquet)
    except Exception:
        passes.to_pickle(cache_pkl)

    return passes


# ─────────────────────────────────────────────────────────────────────────
# STEP 2 — Preferred foot / weak-foot flag (same logic as notebook Cell 4)
#          Computed PER SEASON, from that season's own pass volume, since
#          player squads / minutes shift across seasons.
# ─────────────────────────────────────────────────────────────────────────
def add_weak_foot_flag(passes, min_attempts=20):
    foot_passes = passes[passes["pass_body_part"].isin(["Right Foot", "Left Foot"])].copy()

    player_foot = (
        foot_passes.groupby(["player_id", "pass_body_part"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"Right Foot": "n_right", "Left Foot": "n_left"})
    )
    for col in ("n_right", "n_left"):
        if col not in player_foot.columns:
            player_foot[col] = 0
    player_foot["total"] = player_foot["n_right"] + player_foot["n_left"]
    player_foot["right_pct"] = player_foot["n_right"] / player_foot["total"]
    player_foot["preferred_foot"] = np.where(
        player_foot["right_pct"] >= 0.55,
        "Right",
        np.where(player_foot["right_pct"] <= 0.45, "Left", "Both"),
    )

    passes = passes.merge(
        player_foot[["preferred_foot", "total"]].reset_index(), on="player_id", how="left"
    )

    def flag_weak(row):
        if row["pass_body_part"] not in ("Right Foot", "Left Foot"):
            return np.nan
        if row.get("preferred_foot") not in ("Right", "Left"):
            return np.nan
        if pd.isna(row.get("total")) or row["total"] < min_attempts:
            return np.nan
        used = "Right" if row["pass_body_part"] == "Right Foot" else "Left"
        return int(used != row["preferred_foot"])

    passes["weak_foot"] = passes.apply(flag_weak, axis=1)
    return passes


# ─────────────────────────────────────────────────────────────────────────
# STEP 3 — Feature engineering (event-only recipe from the notebook,
#          i.e. build_features(use_360=False) — no 360 data needed, so
#          this works identically for every season)
# ─────────────────────────────────────────────────────────────────────────
BASE_FEATS = [
    "pass_length", "pass_angle", "dist_to_goal", "dist_to_sideline",
    "end_dist_to_goal", "forward_progress", "lateral_dist",
    "angle_to_goal", "pass_direction",
    "height_num", "len_short", "len_medium", "len_long",
    "under_pressure", "pass_cross_f", "pass_switch_f",
    "pass_through_ball_f", "pass_cut_back_f", "weak_foot_feat",
    "zone_x", "zone_y",
    "pressure_x_length", "cross_x_pressure", "wf_x_cross",
]


# StatsBomb open data uses a 120x80 coordinate system (verified on cache:
# start x spans 0.1-120.9, y 0.1-80.8). Goal centre is (120, 40); pitch
# zones below are approximate thirds.
PITCH_L, PITCH_W = 120, 80
GOAL_X, GOAL_Y = 120, 40


def build_features(df):
    d = df.copy()
    def coordinate(location, axis):
        return (location[axis] if isinstance(location, (list, tuple, np.ndarray))
                and len(location) > axis else np.nan)

    d["start_x"] = d["location"].apply(lambda loc: coordinate(loc, 0))
    d["start_y"] = d["location"].apply(lambda loc: coordinate(loc, 1))
    d["end_x"] = d["pass_end_location"].apply(lambda loc: coordinate(loc, 0))
    d["end_y"] = d["pass_end_location"].apply(lambda loc: coordinate(loc, 1))
    d["dist_to_goal"] = np.sqrt((GOAL_X - d["start_x"]) ** 2 + (GOAL_Y - d["start_y"]) ** 2)
    d["dist_to_sideline"] = np.minimum(d["start_y"], PITCH_W - d["start_y"])
    d["end_dist_to_goal"] = np.sqrt((GOAL_X - d["end_x"]) ** 2 + (GOAL_Y - d["end_y"]) ** 2)
    d["forward_progress"] = d["end_x"] - d["start_x"]
    d["lateral_dist"] = abs(d["end_y"] - d["start_y"])
    d["angle_to_goal"] = np.arctan2(GOAL_Y - d["start_y"], GOAL_X - d["start_x"])
    d["pass_direction"] = np.sign(d["forward_progress"])
    ht_map = {"Ground Pass": 0, "Low Pass": 1, "High Pass": 2}
    d["height_num"] = d["pass_height"].map(ht_map).fillna(0)
    d["len_short"] = (d["pass_length"] < 10).astype(int)
    d["len_medium"] = ((d["pass_length"] >= 10) & (d["pass_length"] < 32)).astype(int)
    d["len_long"] = (d["pass_length"] >= 32).astype(int)
    d["under_pressure"] = d["under_pressure"].fillna(False).astype(int)
    d["pass_cross_f"] = d["pass_cross"].fillna(False).astype(int) if "pass_cross" in d else 0
    d["pass_switch_f"] = d["pass_switch"].fillna(False).astype(int) if "pass_switch" in d else 0
    d["pass_through_ball_f"] = (
        d["pass_through_ball"].fillna(False).astype(int) if "pass_through_ball" in d else 0
    )
    d["pass_cut_back_f"] = d["pass_cut_back"].fillna(False).astype(int) if "pass_cut_back" in d else 0
    d["weak_foot_feat"] = d["weak_foot"].fillna(0)
    d["zone_x"] = pd.cut(d["start_x"], bins=[0, 40, 80, 120], labels=[0, 1, 2]).astype(float)
    d["zone_y"] = pd.cut(d["start_y"], bins=[0, 27, 53, 80], labels=[0, 1, 2]).astype(float)
    d["pressure_x_length"] = d["under_pressure"] * d["pass_length"]
    d["cross_x_pressure"] = d["pass_cross_f"] * d["under_pressure"]
    d["wf_x_cross"] = d["weak_foot_feat"] * d["pass_cross_f"]
    d["completed"] = d["pass_outcome"].isna().astype(int)

    d = d.dropna(subset=BASE_FEATS + ["completed", "match_id"])
    return d


def pass_type_label(row):
    for name, flag in PASS_TYPE_FLAGS.items():
        if row.get(flag) == 1:
            return name
    return "Ordinary"


# ─────────────────────────────────────────────────────────────────────────
# STEP 4 — Train event-only XGBoost model for one season, get held-out preds
# ─────────────────────────────────────────────────────────────────────────
def train_and_predict(df, feats, seed=42):
    match_ids = df["match_id"].unique()
    if len(match_ids) < 5:
        return None  # too few matches for a meaningful split

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
    model.fit(train[feats].values, train["completed"].values)
    preds = model.predict_proba(test[feats].values)[:, 1]

    out = test.copy()
    out["pred"] = preds
    out["actual"] = out["completed"].values
    auc = roc_auc_score(out["actual"], out["pred"])
    brier = brier_score_loss(out["actual"], out["pred"])
    return out, auc, brier


# ─────────────────────────────────────────────────────────────────────────
# STEP 5 — Hypothesis test: weak foot -> lower completion, per season
# ─────────────────────────────────────────────────────────────────────────
CANDIDATE_CONTROLS = [
    "pass_length", "under_pressure", "pass_cross_f", "pass_switch_f",
    "pass_through_ball_f", "pass_cut_back_f", "dist_to_goal", "angle_to_goal",
]


def weak_foot_hypothesis_test(df, season_name, verbose=True):
    """
    Returns a dict with:
      - raw completion rates for strong vs weak foot
      - chi-square test of independence (weak_foot x completed)
      - a logistic-regression coefficient on weak_foot_feat, controlling
        for pass difficulty (this is the effect that should go on the poster
        — the raw gap alone conflates weak-foot passes being systematically
        shorter/safer with an actual accuracy penalty).
    """
    d = df.dropna(subset=["weak_foot"]).copy()
    if d["weak_foot"].nunique() < 2 or len(d) < 100:
        return None

    strong = d[d.weak_foot == 0]["completed"]
    weak = d[d.weak_foot == 1]["completed"]

    # Chi-square test of independence
    table = pd.crosstab(d["weak_foot"], d["completed"])
    chi2, p_chi2, _, _ = stats.chi2_contingency(table)

    # Controlled logistic regression: completed ~ weak_foot + difficulty controls
    # Drop any control with zero variance in THIS season (e.g. pass_cut_back
    # never occurs) — a constant column makes X'X singular.
    controls = [c for c in CANDIDATE_CONTROLS if c in d.columns and d[c].nunique() > 1]
    dropped = [c for c in CANDIDATE_CONTROLS if c not in controls]
    if dropped and verbose:
        print(f"    [{season_name}] dropping zero-variance controls: {dropped}")

    coef = ci_low = ci_high = p_val = np.nan
    logit_error = ""
    fit_method = "statsmodels_logit"

    X = sm.add_constant(d[["weak_foot"] + controls].astype(float))
    y = d["completed"].astype(float)
    try:
        logit = sm.Logit(y, X).fit(disp=0)
        coef = logit.params["weak_foot"]
        ci_low, ci_high = logit.conf_int().loc["weak_foot"]
        p_val = logit.pvalues["weak_foot"]
    except Exception as e:
        logit_error = str(e)
        if verbose:
            print(f"    [{season_name}] statsmodels Logit failed: {e} — falling back to sklearn")
        # Fallback: L2-regularized logistic regression handles near-singular
        # / (quasi-)separated designs more gracefully than unregularized MLE.
        # ponytail: bootstrap CI, not exact p-values. Upgrade only if a
        # season keeps hitting this path.
        try:
            from sklearn.linear_model import LogisticRegression

            feat_cols = ["weak_foot"] + controls
            Xs = d[feat_cols].astype(float).values
            ys = d["completed"].astype(float).values

            clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
            clf.fit(Xs, ys)
            coef = clf.coef_[0][feat_cols.index("weak_foot")]

            rng = np.random.RandomState(0)
            boot_coefs = []
            n = len(d)
            for _ in range(200):
                idx = rng.randint(0, n, n)
                try:
                    c2 = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
                    c2.fit(Xs[idx], ys[idx])
                    boot_coefs.append(c2.coef_[0][feat_cols.index("weak_foot")])
                except Exception:
                    continue
            if boot_coefs:
                ci_low, ci_high = np.percentile(boot_coefs, [2.5, 97.5])
                p_val = 2 * min(
                    np.mean(np.array(boot_coefs) <= 0), np.mean(np.array(boot_coefs) >= 0)
                )
            fit_method = "sklearn_l2_fallback"
            logit_error += " | recovered via sklearn L2 fallback"
        except Exception as e2:
            logit_error += f" | sklearn fallback also failed: {e2}"

    return {
        "season": season_name,
        "n_passes": len(d),
        "strong_foot_completion": strong.mean(),
        "weak_foot_completion": weak.mean(),
        "raw_gap_pp": (strong.mean() - weak.mean()) * 100,
        "chi2": chi2,
        "chi2_pvalue": p_chi2,
        "logit_weak_foot_coef": coef,
        "logit_ci_low": ci_low,
        "logit_ci_high": ci_high,
        "logit_pvalue": p_val,
        "fit_method": fit_method,
        "controls_used": ",".join(controls),
        "controls_dropped": ",".join(dropped),
        "logit_error": logit_error,
    }


def plot_weak_foot_forest(results_df, path):
    """Forest plot of the controlled weak-foot logistic coefficient, one row
    per season, ordered chronologically. A coefficient < 0 means weak-foot
    passes are less likely to complete even after controlling for pass
    difficulty — negative CIs that exclude 0 support the hypothesis."""
    d = results_df.dropna(subset=["logit_weak_foot_coef"]).copy()
    d = d.sort_values("season")

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(d))))
    y_pos = np.arange(len(d))

    is_fallback = d.get("fit_method", pd.Series(["statsmodels_logit"] * len(d))) == "sklearn_l2_fallback"
    colors = np.where(is_fallback, "#D97706", "#1D4ED8")
    ax.errorbar(
        d["logit_weak_foot_coef"], y_pos,
        xerr=[d["logit_weak_foot_coef"] - d["logit_ci_low"],
              d["logit_ci_high"] - d["logit_weak_foot_coef"]],
        fmt="none", ecolor="#93C5FD", capsize=3, zorder=1,
    )
    ax.scatter(d["logit_weak_foot_coef"], y_pos, c=colors, zorder=2, s=40)
    ax.axvline(0, color="#DC2626", ls="--", lw=1.2, label="No effect")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(d["season"])
    if is_fallback.any():
        ax.scatter([], [], c="#1D4ED8", s=40, label="statsmodels logit")
        ax.scatter([], [], c="#D97706", s=40, label="sklearn L2 fallback")
    ax.set_xlabel("Logistic coefficient on weak_foot (controlling for pass difficulty)")
    ax.set_title(
        "Weak-foot passes are completed less often — across every La Liga season\n"
        "(negative = weak foot hurts completion probability, controls held constant)",
        fontsize=11, fontweight="bold",
    )
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# STEP 6 — Calibration by pass type, pooled across all seasons
# ─────────────────────────────────────────────────────────────────────────
def expected_calibration_error(y_true, y_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_pred, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_pred[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece


def calibration_by_pass_type(all_preds, path_plot, path_csv):
    all_preds = all_preds.copy()
    all_preds["pass_type"] = all_preds.apply(pass_type_label, axis=1)

    types = ["Ordinary"] + list(PASS_TYPE_FLAGS.keys())
    rows = []
    fig, axes = plt.subplots(1, len(types), figsize=(4 * len(types), 4), sharey=True)
    if len(types) == 1:
        axes = [axes]

    for ax, ptype in zip(axes, types):
        sub = all_preds[all_preds["pass_type"] == ptype]
        if len(sub) < 30:
            ax.set_title(f"{ptype}\n(n={len(sub)}, too few)")
            continue
        y_true = sub["actual"].values
        y_pred = sub["pred"].values
        frac_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", color="#1D4ED8", lw=2, ms=5)
        ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", lw=1)
        brier = brier_score_loss(y_true, y_pred)
        ece = expected_calibration_error(y_true, y_pred)
        ax.set_title(f"{ptype}\nn={len(sub):,}  Brier={brier:.3f}  ECE={ece:.3f}", fontsize=10)
        ax.set_xlabel("Mean predicted")
        ax.spines[["top", "right"]].set_visible(False)

        rows.append({
            "pass_type": ptype,
            "n": len(sub),
            "actual_rate": y_true.mean(),
            "predicted_rate": y_pred.mean(),
            "gap_pp": (y_true.mean() - y_pred.mean()) * 100,
            "brier_score": brier,
            "ece": ece,
        })

    axes[0].set_ylabel("Fraction of positives (actual completion rate)")
    plt.suptitle(
        "xPass Calibration by Pass Type — pooled across all La Liga seasons "
        "(event-only model)", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(path_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(path_csv, index=False)


def calibration_by_season(results_by_season, path_csv):
    rows = []
    for season_name, (preds, auc, brier) in results_by_season.items():
        ece = expected_calibration_error(preds["actual"].values, preds["pred"].values)
        rows.append({
            "season": season_name,
            "n_test_passes": len(preds),
            "auc": auc,
            "brier_score": brier,
            "ece": ece,
        })
    pd.DataFrame(rows).sort_values("season").to_csv(path_csv, index=False)


# ─────────────────────────────────────────────────────────────────────────
# BONUS (optional) — La Liga 2020/21 event-only vs 360-enriched, side by side
# (mirrors the notebook's Cells 6-9, kept here so the poster can show
#  "does adding 360 context change the weak-foot story on the one season
#  where we have it")
# ─────────────────────────────────────────────────────────────────────────
def fetch_360(match_id):
    url = (
        f"https://raw.githubusercontent.com/statsbomb/"
        f"open-data/master/data/three-sixty/{match_id}.json"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    rows = [
        {"id": rec["event_uuid"], "freeze_frame": rec["freeze_frame"],
         "visible_area": rec["visible_area"]}
        for rec in data
    ]
    return pd.DataFrame(rows)


def defenders_in_lane(passer_loc, end_loc, opponents, width=3.0):
    px, py = passer_loc
    ex, ey = end_loc
    dx, dy = ex - px, ey - py
    length = np.sqrt(dx ** 2 + dy ** 2)
    if length == 0:
        return 0
    count = 0
    for op in opponents:
        ox, oy = op["location"]
        t = ((ox - px) * dx + (oy - py) * dy) / (length ** 2)
        if 0 < t < 1:
            perp = abs((oy - py) * dx - (ox - px) * dy) / length
            if perp <= width:
                count += 1
    return count


def extract_360_features(freeze_frame, passer_loc, end_loc):
    if not isinstance(freeze_frame, list) or not freeze_frame:
        return {}
    opponents = [p for p in freeze_frame if not p["teammate"] and not p["actor"]]
    teammates = [p for p in freeze_frame if p["teammate"] and not p["actor"]]
    if not opponents:
        return {}

    def dist(p, ref):
        return np.sqrt((p["location"][0] - ref[0]) ** 2 + (p["location"][1] - ref[1]) ** 2)

    return {
        "closest_def_passer": min(dist(o, passer_loc) for o in opponents),
        "closest_def_receiver": min(dist(o, end_loc) for o in opponents),
        "n_defenders_in_lane": defenders_in_lane(passer_loc, end_loc, opponents),
        "n_opponents_visible": len(opponents),
        "n_teammates_visible": len(teammates),
        "n_def_behind": sum(1 for o in opponents if o["location"][0] < passer_loc[0]),
    }


FEATS_360 = [
    "closest_def_passer", "closest_def_receiver", "n_defenders_in_lane",
    "n_opponents_visible", "n_teammates_visible", "n_def_behind",
]


def run_2020_21_event_vs_360_bonus():
    print("\n[BONUS] Event-only vs 360-enriched — La Liga 2020/21")
    season_row = get_laliga_seasons().query("season_name == '2020/2021'")
    if season_row.empty:
        print("  Could not find 2020/2021 season_id — skipping bonus section")
        return
    season_id = int(season_row.iloc[0]["season_id"])

    passes = pull_season_passes(COMP_ID, season_id, "2020/2021")
    passes = add_weak_foot_flag(passes)
    matches = sb.matches(competition_id=COMP_ID, season_id=season_id)

    print("  Pulling 360 freeze frames...")
    all_360 = []
    for i, mid in enumerate(matches.match_id):
        df360 = fetch_360(mid)
        if df360.empty:
            continue
        match_passes = passes[passes.match_id == mid]
        merged = match_passes.merge(df360, on="id", how="left")
        for _, row in merged.iterrows():
            loc, end_loc = row["location"], row["pass_end_location"]
            if not all(isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2
                       for p in (loc, end_loc)):
                continue
            f = extract_360_features(row.get("freeze_frame"), loc, end_loc)
            f["id"], f["match_id"] = row["id"], mid
            all_360.append(f)
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(matches)} matches processed")

    df_360_feats = pd.DataFrame(all_360)
    if df_360_feats.empty or not set(FEATS_360).issubset(df_360_feats.columns):
        print("  No usable 360 features available — skipping bonus comparison")
        return
    df = passes.merge(df_360_feats, on=["id", "match_id"], how="left")
    df_base = build_features(df)
    df_360_built = df_base.dropna(subset=FEATS_360)

    out_base = train_and_predict(df_base, BASE_FEATS)
    out_360 = train_and_predict(df_360_built, BASE_FEATS + FEATS_360)
    if out_base is None or out_360 is None:
        print("  Not enough data for bonus comparison — skipping")
        return

    preds_base, auc_base, brier_base = out_base
    preds_360, auc_360, brier_360 = out_360

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, preds, color in [
        ("event_only", preds_base, "#6B7280"), ("with_360", preds_360, "#1D4ED8")
    ]:
        frac_pos, mean_pred = calibration_curve(preds["actual"], preds["pred"], n_bins=10)
        ax.plot(mean_pred, frac_pos, "o-", color=color, lw=2, ms=5,
                label=f"{label} (AUC={roc_auc_score(preds['actual'], preds['pred']):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", lw=1.5, label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("La Liga 2020/21 — Event-only vs 360-enriched Calibration",
                 fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "event_vs_360_2020_21.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  event_only  AUC={auc_base:.4f}  Brier={brier_base:.4f}")
    print(f"  with_360    AUC={auc_360:.4f}  Brier={brier_360:.4f}")


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons", nargs="*", default=None,
        help="Limit to specific season_name values (e.g. 2020/2021 2019/2020). "
             "Default: every La Liga season in the open data.",
    )
    parser.add_argument(
        "--with-360", action="store_true",
        help="Also run the bonus event-only vs 360 comparison for 2020/21.",
    )
    args = parser.parse_args()

    seasons = get_laliga_seasons()
    if args.seasons:
        seasons = seasons[seasons["season_name"].isin(args.seasons)]
    print(f"Found {len(seasons)} La Liga season(s) to process:")
    print(seasons.to_string(index=False))

    hypothesis_rows = []
    results_by_season = {}
    pooled_preds = []

    for _, row in seasons.iterrows():
        comp_id, season_id, season_name = (
            row["competition_id"], row["season_id"], row["season_name"]
        )
        print(f"\n=== Season {season_name} (season_id={season_id}) ===")

        passes = pull_season_passes(comp_id, season_id, season_name)
        if passes.empty:
            print("  No events pulled — skipping")
            continue

        passes = add_weak_foot_flag(passes)
        df = build_features(passes)
        if len(df) < 200:
            print(f"  Only {len(df)} usable passes after feature build — skipping")
            continue

        # Hypothesis test on the FULL season's passes (not just the test split)
        h = weak_foot_hypothesis_test(df, season_name)
        if h:
            hypothesis_rows.append(h)
            print(
                f"  Weak-foot completion gap: {h['raw_gap_pp']:+.2f}pp raw | "
                f"controlled logit coef={h['logit_weak_foot_coef']:.3f} "
                f"(p={h['logit_pvalue']:.4f})"
            )

        # Train + evaluate event-only model, collect held-out preds for calibration
        out = train_and_predict(df, BASE_FEATS)
        if out is None:
            print("  Not enough matches for a train/test split — skipping model eval")
            continue
        preds, auc, brier = out
        results_by_season[season_name] = (preds, auc, brier)
        pooled_preds.append(preds)
        print(f"  Event-only model: AUC={auc:.4f}  Brier={brier:.4f}  n_test={len(preds):,}")

    # ── Save hypothesis-test results + forest plot ──
    if hypothesis_rows:
        hyp_df = pd.DataFrame(hypothesis_rows)
        hyp_df.to_csv(OUT_DIR / "weak_foot_hypothesis_by_season.csv", index=False)
        plot_weak_foot_forest(hyp_df, OUT_DIR / "weak_foot_forest_plot.png")
        print(f"\nSaved: {OUT_DIR / 'weak_foot_hypothesis_by_season.csv'}")
        print(f"Saved: {OUT_DIR / 'weak_foot_forest_plot.png'}")
    else:
        print("\nNo seasons produced valid hypothesis-test results.")

    # ── Save calibration analyses ──
    if pooled_preds:
        all_preds = pd.concat(pooled_preds, ignore_index=True)
        calibration_by_pass_type(
            all_preds,
            OUT_DIR / "calibration_by_pass_type.png",
            OUT_DIR / "calibration_summary_by_pass_type.csv",
        )
        calibration_by_season(results_by_season, OUT_DIR / "calibration_summary_by_season.csv")
        print(f"Saved: {OUT_DIR / 'calibration_by_pass_type.png'}")
        print(f"Saved: {OUT_DIR / 'calibration_summary_by_pass_type.csv'}")
        print(f"Saved: {OUT_DIR / 'calibration_summary_by_season.csv'}")
    else:
        print("No seasons produced held-out predictions for calibration analysis.")

    if args.with_360:
        run_2020_21_event_vs_360_bonus()

    print(f"\nAll poster assets written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()