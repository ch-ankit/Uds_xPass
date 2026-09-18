# Uds_xPass — Expected Pass (xPass) for La Liga

Can we predict every pass? This project builds an event-only expected pass model on StatsBomb open data (18 La Liga seasons) and asks two questions: does a player's weak foot hurt completion, and why does the model miscalibrate on rare pass types?

Short answer: weak-foot passes complete about 15–20% less often at fixed difficulty, and rare types (crosses, switches, through balls) miscalibrate mostly because there are so few of them. An equal-n check shows Ordinary passes degrade roughly eightfold when downsampled to rare-type sizes. Details and numbers are in `report/main.tex` and the poster PDF.

## Setup

```bash
conda activate xpass
```

Everything runs from the repo root. The StatsBomb event caches live in `data/` (gitignored, ~1 GB) — if you don't have them, the scripts will try to download from StatsBomb instead, which is slow. The verified environment is Python 3.14 with statsbombpy, xgboost, scikit-learn, scipy, statsmodels, pandas, numpy and matplotlib.

## Running the pipeline

```bash
python all_seasons.py          # base models, weak-foot test, calibration by type
python calibration_study.py    # footedness ablation + isotonic recalibration
python weighted_study.py       # type-weighted training (negative result)
python downsample_study.py     # equal-n downsampling (scarcity decomposition)
python build_dashboard.py      # rebuilds xpass_dashboard.html
```

Quicker smoke test on two seasons:

```bash
python all_seasons.py --seasons 2020/2021 2019/2020
python weighted_study.py --ids 90 42
python build_dashboard.py --ids 90 42
```

Note these overwrite the full-run outputs, so back up `poster_outputs/` first if you need the submitted figures. `python -m unittest discover -s tests` runs three quick offline checks (feature building, missing freeze frames, the 360 path on synthetic data).

The report compiles with `cd report && latexmk -pdf main.tex`, but you'll need a TeX distribution for that. The prose and tables in the report are maintained by hand — rerunning the scripts won't update them.

## What's where

- `all_seasons.py` — the main pipeline: per-season XGBoost (match-level 80/20 split), controlled weak-foot test, pooled calibration by pass type. `calibration_study.py`, `weighted_study.py` and `downsample_study.py` import from it.
- `build_dashboard.py` — generates `xpass_dashboard.html`, a self-contained page with the findings charts, a calibration map and player cards. Just open it in a browser.
- `laliga_360_xpass.ipynb` — the original 2020/21 pilot notebook, kept for reference. The `--with-360` flag in `all_seasons.py` covers the 360 comparison.
- `poster_outputs/` — all figures and CSVs behind the report and poster.
- `preferred_foot_fifa.csv` — FIFA 21 preferred-foot labels used to sanity-check our usage-inferred foot flags (`players_21.csv` is the raw dump).
- `archive/` — old notebooks, foot-source scrapers (FBref/Transfermarkt/Wikidata experiments) and early single-season figures. Not part of the run sequence.

## Things worth knowing

- Preferred foot is inferred from each player's own pass mix per season (55/45 split, minimum 20 attempts), so two-footed and low-minute players stay unlabelled.
- Cut-backs barely exist in this data (145 pooled passes), so they're shown in the calibration table but left out of the ablation.
- 1973/74 has only 555 usable passes with patchy body-part tagging and gets excluded from the headline weak-foot claim.
- Early seasons are mostly Barcelona games, so don't over-generalise across eras.
- The isotonic recalibration and the equal-n experiment split individual passes rather than matches, which flatters them slightly — flagged here so nobody oversells it.
