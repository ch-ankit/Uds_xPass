# Uds_xPass

**Can We Predict Every Pass?** An event-only Expected Pass (xPass) model for La Liga, studying weak-foot completion differences and calibration across pass types.

## Verification status

The **cached core pipeline runs end to end** in the existing `xpass` environment:

| Stage | Verification |
|---|---|
| `all_seasons.py` | Completed using 18 existing season caches; generated hypothesis and calibration outputs |
| `calibration_study.py` | Completed all three feature variants and isotonic recalibration |
| `weighted_study.py` | Completed all three weighting variants on the default five cached seasons |
| `build_dashboard.py` | Generated `xpass_dashboard.html`; browser interactions not tested |
| Optional 360 comparison | Offline regression test trains both models and writes a figure; live 360 downloads not verified |
| `report/main.tex` | Build not verified in this environment: `pdflatex` and `latexmk` are unavailable |
| Fresh download / clean installation | Not tested end to end; the verified run reused existing pickle caches |

Verification ran in temporary directories, preserving the existing `poster_outputs/` and dashboard. Five of the six regenerated CSVs matched the saved results within `rtol=1e-6, atol=1e-8`. Through-ball recalibration ECE after correction was **0.029324** versus **0.028602** in the saved CSV; exact historical reproduction is therefore not established. Runtime fixes were then rerun through all four scripts, with all six CSVs unchanged relative to the first verification run.

**Execution success is not scientific validation.** The coordinate-scale issue below must be resolved before treating regenerated model results as corrected findings.

## Setup

Run commands from the repository root: data and output paths are relative to the working directory.

Existing local environment:

```bash
conda activate xpass
```

Alternatively, create a separate environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install statsbombpy xgboost scikit-learn scipy statsmodels pandas numpy matplotlib requests
```

The verified environment used Python 3.14.4, statsbombpy 1.13.0, XGBoost 3.2.0, scikit-learn 1.9.0, SciPy 1.18.1, statsmodels 0.14.6, pandas 3.0.5, NumPy 2.5.3, Matplotlib 3.11.0, and requests 2.34.2. These are observed versions, not a dependency lockfile. PyArrow was not installed; caches were `.pkl` files.

Only load trusted pickle caches. `data/` is gitignored, so a fresh clone must download events or receive trusted caches separately.

## Run the core pipeline

```bash
# Optional on headless machines; thread limits avoid CPU oversubscription.
export MPLBACKEND=Agg
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

python all_seasons.py
python calibration_study.py
python weighted_study.py
python build_dashboard.py
```

Open `xpass_dashboard.html` in a browser; no server is needed. The scripts overwrite their output files. Back up submission artifacts before running experiments.

- `all_seasons.py` discovers seasons through StatsBomb, then reads or downloads each season's events. Even with event caches, season discovery uses the StatsBomb client and may require network access.
- `calibration_study.py` also discovers seasons and downloads missing caches. It is **not** a strictly offline, cache-only command and does not inherit the base script's season filter.
- `weighted_study.py` defaults to the five largest **pickle cache files by file size**, not the five largest datasets by row count. In the verified run these IDs were `1 90 26 23 27`.
- `build_dashboard.py` reads the summary CSVs and retrains the current model for its pitch sample (default season IDs `90 42`). Run it after the three analysis scripts; absent summary CSVs can leave sections empty.

For a smaller base-model run and explicit weighting/dashboard seasons:

```bash
python all_seasons.py --seasons 2020/2021 2019/2020
python weighted_study.py --ids 90 42
python build_dashboard.py --ids 90 42
```

These commands still overwrite the corresponding full-run outputs. There is currently no season-filter CLI for `calibration_study.py`.

### Regression tests

```bash
python -m unittest discover -s tests -v
```

Three offline tests check list/array feature equivalence, missing freeze frames, and the optional 360 path using synthetic passes with real feature building and XGBoost training. They do not validate live data availability, coordinate units, or research conclusions.

### Optional 360 comparison

```bash
python all_seasons.py --seasons 2020/2021 --with-360
```

This attempts to write `poster_outputs/event_vs_360_2020_21.png` when usable freeze frames are available. The runtime merge now preserves `match_id`, and missing freeze-frame input is handled. This command was not verified against live 360 data; it is not required for the event-only results.


## Analysis design

1. **Base model:** `all_seasons.py` trains a per-season XGBoost classifier with a match-level 80/20 split, seed 42, and 24 event-only features. `laliga_360_xpass.ipynb` is a pilot/reference notebook, not a prerequisite.
2. **Footedness:** preferred foot is inferred per season using a 55/45 usage rule and at least 20 footed attempts. Controlled logistic regression tests the weak-foot association. FIFA files and source-search notebooks provide separate provenance; the core scripts do not rerun FIFA validation.
3. **Calibration:** held-out predictions are pooled and grouped as Ordinary, Cross, Switch, or Through ball. Flags can overlap; labeling prioritizes Cross, then Switch, then Through ball. Cut-back is retained as a base-model placeholder but absent from the saved type summaries.
4. **Ablation:** no footedness (22 features), current (24), and full interactions (26). The additional interactions are weak foot × switch and weak foot × through ball.
5. **Recalibration:** the variant with the lowest mean rare-type ECE is selected; per-type isotonic regression is fitted on half its held-out passes and evaluated on the other half.
6. **Weighting:** unweighted, inverse-type-frequency, and square-root-inverse-frequency training. Overall AUC is printed to the console, not saved in `weighted_summary.csv`.

## Outputs

| File | Contents |
|---|---|
| `poster_outputs/weak_foot_forest_plot.png` + `weak_foot_hypothesis_by_season.csv` | Controlled weak-foot association by season |
| `poster_outputs/calibration_by_pass_type.png` + `calibration_summary_by_pass_type.csv` | Pooled pass-type reliability, Brier score and ECE |
| `poster_outputs/calibration_summary_by_season.csv` | AUC, Brier score and ECE by evaluated season |
| `poster_outputs/ablation_calibration_by_pass_type.png` + `ablation_summary.csv` | Three footedness variants |
| `poster_outputs/recalibration_before_after.png` + `recalibration_summary.csv` | Isotonic before/after comparison |
| `poster_outputs/weighted_summary.csv` | Weighting variants: sample counts, Brier score and ECE |
| `poster_outputs/downsample_summary.csv` | Equal-n Ordinary subsamples vs rare types (scarcity decomposition) |
| `xpass_dashboard.html` | Standalone dashboard, summary data and sampled held-out passes |
| `report/main.tex` | Manually maintained seminar report source |

There are 18 cached seasons, but only 17 appear in the saved model-calibration table. The 1973/74 cache has one match and is skipped by the model's minimum-match check; it remains in the hypothesis table. The 2020/21 cache contains **35 matches**, not a full league season; 2015/16 contains 380.

Saved results show a significant negative weak-foot coefficient in **14 of 17 modern seasons**. Ordinary ECE is about 0.003 versus 0.058 for Through balls. Isotonic reduces Through-ball ECE to about 0.029 but increases Cross/Switch ECE. The verification run confirmed weighting AUC of **0.8934 unweighted versus 0.8837 inverse-weighted**. These patterns do not by themselves prove sample scarcity is the cause of miscalibration.

## Known limitations and remaining blockers
- **Parquet workflow is not fully verified:** shared feature building now accepts NumPy-array coordinates, but the dashboard's sample extraction still accepts lists only. Weighting's default cache discovery finds only `.pkl` files; for Parquet-only caches explicit `--ids` are needed. An actual Parquet round trip was not tested.
- **360 comparison:** live availability remains unverified. Event-only and 360 models currently use potentially different eligible rows/splits; this is not a controlled like-for-like feature comparison.
- **Evaluation:** preferred foot is inferred before the train/test split. Isotonic fitting/evaluation splits individual passes, not matches, and variant selection uses the same held-out prediction pool. These are methodological limitations, not resolved by successful execution.
- **Data pulls:** failed matches are skipped and the remaining partial season can be cached. Review logs; a zero exit code alone does not establish complete coverage.
- **Report/poster:** figures can be regenerated, but prose, tables, citations and historical presentation files are not automatically synchronized or validated by these scripts. No fresh report compilation was performed in this environment.

## Repository layout

```text
all_seasons.py             # canonical event-only pipeline and optional 360 path
calibration_study.py       # footedness ablation and isotonic recalibration
weighted_study.py          # rare-type weighting study
build_dashboard.py        # CSV summaries and pitch sample to HTML
tests/test_pipeline.py    # offline runtime regression checks
laliga_360_xpass.ipynb     # historical pilot/reference notebook
preferred_foot_fifa.csv    # matched FIFA preferred-foot labels
players_21.csv             # FIFA source data retained for provenance
data/                     # local StatsBomb caches (gitignored)
poster_outputs/           # generated figures and CSVs
report/main.tex            # report source
xPass_Poster_Presentation.pdf
Expected_Passs_in_Football.pptx
archive/                  # frozen notebooks, source-search scrapers, old figures
```

Archived scrapers document the FBref/Transfermarkt/FIFA/Wikidata foot-source search; they are not part of the supported core run sequence.
