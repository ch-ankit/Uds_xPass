# Uds_xPass

**Uds_xPass** is an Expected Pass (xPass) modeling project built on StatsBomb 360 event data for La Liga (2020/21). It estimates the probability that a given pass attempt will be completed, using contextual features derived from freeze-frame (360°) tracking data, and investigates how a player's **preferred foot** affects pass completion (i.e., whether players are less accurate when passing with their weak foot).

## Overview

The project pipeline covers:

1. **Data acquisition** — pulling event and 360° freeze-frame data for La Liga matches via [`statsbombpy`](https://github.com/statsbomb/statsbombpy).
2. **Feature engineering** — deriving pass-level features (e.g., distance, angle, pressure, body part used) from the raw event/360 data.
3. **Preferred-foot enrichment** — since StatsBomb data doesn't include a player's natural preferred foot, this project scrapes/derives ground-truth preferred-foot labels from external sources and merges them in to flag "weak foot" passes.
4. **Modeling** — training a classification model to predict pass completion probability (xPass).
5. **Evaluation** — ROC/calibration analysis, feature importance, and a specialist-vs-generalist breakdown of players.

## Repository Structure

```
Uds_xPass/
├── laliga_360_xpass.ipynb          # Main notebook: data prep, feature engineering, xPass model training & evaluation
├── get_preferred_foot.ipynb        # Notebook for fetching/inspecting preferred-foot data for players
├── scraper.py                      # Multi-approach scraper/merger for ground-truth preferred foot
│                                    #   (FBref via soccerdata, Transfermarkt, FIFA21 Kaggle dataset, Wikidata SPARQL)
├── xpass_get_scrapper.py           # Refined FBref scraping approach (fixes/extends Approach A in scraper.py)
├── downloaded_files/               # Cached/downloaded data used by the notebooks
├── feature_importance_comparison.png  # Feature importance plot for the xPass model
├── roc_calibration_laliga.png         # ROC curve + calibration plot for model evaluation
└── specialist_generalist_laliga.png   # Specialist vs. generalist passer analysis plot
```

## Data Sources

| Source | Purpose |
|---|---|
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) (via `statsbombpy`) | Event data + 360° freeze frames for La Liga 2020/21 |
| [FBref](https://fbref.com) (via [`soccerdata`](https://github.com/probberechts/soccerdata)) | Player biographical data, including preferred foot |
| [Transfermarkt](https://www.transfermarkt.com) | Alternative source for preferred-foot data |
| [FIFA 21 Kaggle dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-21-complete-player-dataset) | Offline preferred-foot reference, fuzzy-matched to StatsBomb player names |
| [Wikidata](https://www.wikidata.org) (SPARQL, property `P2354`) | Open/citable preferred-foot data (partial coverage) |

> **Note:** Web scraping code (Transfermarkt/FBref approaches) is provided for research/educational purposes. Respect each site's terms of service and `robots.txt`, use polite request delays, and prefer the official/open data sources (StatsBomb, Wikidata, Kaggle) where possible.

## Getting Started

### Requirements

```bash
pip install pandas numpy requests beautifulsoup4 lxml fuzzywuzzy python-Levenshtein
pip install statsbombpy soccerdata
pip install scikit-learn matplotlib seaborn  # for modeling & plots
```

### Workflow

1. **Fetch match/event data and build the base pass dataset**
   Open `laliga_360_xpass.ipynb` and run the early cells to pull StatsBomb events (competition/season for La Liga 2020/21) and construct the passes dataframe.

2. **Get preferred-foot ground truth**
   Use `get_preferred_foot.ipynb`, `scraper.py`, or `xpass_get_scrapper.py` to fetch preferred-foot labels for the players in your dataset. `scraper.py` includes several interchangeable approaches:
   - **Approach A:** FBref via `soccerdata` (most reliable/citable)
   - **Approach B:** Transfermarkt scraping (broadest player coverage)
   - **Approach C:** Offline FIFA 21 dataset with fuzzy name matching
   - **Approach D:** Wikidata SPARQL query (open, partial coverage)

   Once you have foot data from any approach, merge it back into your passes dataframe with `merge_real_foot()`, which replaces the inferred weak-foot flag with the ground-truth value and reports coverage. Use `validate_inference()` to compare model-inferred preferred foot against the ground truth and see match accuracy.

3. **Train and evaluate the xPass model**
   Continue in `laliga_360_xpass.ipynb` to engineer pass features, train the completion-probability model, and generate the evaluation plots (ROC/calibration, feature importance, specialist vs. generalist breakdown).

## Outputs

- **`roc_calibration_laliga.png`** — model discrimination (ROC/AUC) and probability calibration for predicted pass completion.
- **`feature_importance_comparison.png`** — relative importance of features driving the xPass prediction.
- **`specialist_generalist_laliga.png`** — comparison of players who specialize in certain pass types/zones versus more all-around passers.

## Notes & Limitations

- Preferred-foot data is not part of the core StatsBomb feed and must be sourced externally; coverage and accuracy vary by source, so cross-validating with `validate_inference()` is recommended.
- Scraping cells are commented out by default — uncomment and run deliberately, and expect scraping runs to take a few minutes depending on the number of players and source site.
- Notebooks assume you have valid access to StatsBomb's open/360 competition data for the specified `competition_id`/`season_id`.

## License

No license has been specified for this repository. Please contact the repository owner before reuse.