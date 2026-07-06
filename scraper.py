# =============================================================================
# Scrape preferred foot for La Liga 2020/21 players
# Run this locally — requires internet access to FBref / Transfermarkt
# pip install soccerdata requests beautifulsoup4 fuzzywuzzy pandas
# =============================================================================

from time import time

import pandas as pd
import requests
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from statsbombpy import sb
from fuzzywuzzy import process, fuzz

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Get all unique players from your La Liga data
# (Run after Cell 2 in your notebook so `passes` already exists)
# ─────────────────────────────────────────────────────────────────────────────
COMP_ID, SEASON_ID = 11, 90
matches = sb.matches(competition_id=COMP_ID, season_id=SEASON_ID)

all_events = []
for mid in matches.match_id:
    ev = sb.events(match_id=mid)
    ev['match_id'] = mid
    all_events.append(ev[ev.type == 'Pass'][['player', 'player_id']].drop_duplicates())

players_df = (pd.concat(all_events)
              .drop_duplicates('player_id')
              .reset_index(drop=True)
              .sort_values('player'))

print(f"Unique players to look up: {len(players_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# APPROACH A — soccerdata (wraps FBref — most reliable, citable)
# pip install soccerdata
# ─────────────────────────────────────────────────────────────────────────────
try:
    import soccerdata as sd

    # FBref La Liga 2020/21 player bio data
    fbref = sd.FBref(leagues='ESP-La Liga', seasons='2020-2021')
    bio   = fbref.read_player_season_stats(stat_type='standard')

    # The bio table has a 'foot' column
    if 'foot' in bio.columns:
        foot_fbref = (bio[['player', 'foot']]
                      .drop_duplicates('player')
                      .rename(columns={'foot': 'preferred_foot_fbref'}))
        print(f"\nFBref foot data: {len(foot_fbref)} players")
        print(foot_fbref.head(5))
    else:
        print("Columns:", bio.columns.tolist())

except ImportError:
    print("soccerdata not installed — pip install soccerdata")
except Exception as e:
    print(f"FBref error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH B — Transfermarkt scraper (most complete player coverage)
# ─────────────────────────────────────────────────────────────────────────────
# import requests
# from bs4 import BeautifulSoup
# import time

# def get_foot_transfermarkt(player_name, session):
#     """
#     Search Transfermarkt for a player and return their preferred foot.
#     Returns 'Right', 'Left', 'Both', or None.
#     """
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
#         'Accept-Language': 'en-US,en;q=0.9',
#     }
#     search_url = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"
#     try:
#         r = session.get(
#             search_url,
#             params={'query': player_name, 'Spieler_page': '0'},
#             headers=headers,
#             timeout=10
#         )
#         soup = BeautifulSoup(r.text, 'lxml')

#         # First result link
#         result = soup.select_one('table.items td.hauptlink a')
#         if not result:
#             return None
#         player_url = "https://www.transfermarkt.com" + result['href']

#         time.sleep(0.8)  # be polite
#         r2 = session.get(player_url, headers=headers, timeout=10)
#         soup2 = BeautifulSoup(r2.text, 'lxml')

#         # Find the 'foot' data item in the player profile
#         for li in soup2.select('.info-table__content'):
#             txt = li.get_text(strip=True)
#             if txt in ('Right', 'Left', 'Both feet', 'both'):
#                 return txt.replace(' feet', '').title()

#         # Alternative location
#         for span in soup2.find_all('span', class_='info-table__content--bold'):
#             txt = span.get_text(strip=True)
#             if txt in ('Right', 'Left', 'Both feet'):
#                 return txt.replace(' feet', '').title()

#     except Exception as e:
#         print(f"  Error for {player_name}: {e}")
#     return None


# def scrape_transfermarkt(players_df, n_players=None):
#     """
#     Scrape preferred foot from Transfermarkt for all players.
#     n_players: limit for testing (None = all)
#     """
#     session = requests.Session()
#     session.headers.update({
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
#     })

#     results = []
#     subset  = players_df.head(n_players) if n_players else players_df

#     for i, (_, row) in enumerate(subset.iterrows()):
#         name = row['player']
#         print(f"  [{i+1}/{len(subset)}] {name}...", end=' ', flush=True)
#         foot = get_foot_transfermarkt(name, session)
#         print(foot or "not found")
#         results.append({'player_id': row['player_id'], 'player': name,
#                         'preferred_foot_tm': foot})
#         time.sleep(0.5)

#     return pd.DataFrame(results)

# Uncomment to run (takes ~2-3 min for full squad):
# foot_tm = scrape_transfermarkt(players_df)
# foot_tm.to_csv('preferred_foot_transfermarkt.csv', index=False)
# print(foot_tm['preferred_foot_tm'].value_counts())


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH C — FIFA 21 Dataset from Kaggle (fastest, offline)
# Download: https://www.kaggle.com/datasets/stefanoleone992/fifa-21-complete-player-dataset
# File: players_21.csv
# ─────────────────────────────────────────────────────────────────────────────
# def load_fifa_foot(fifa_csv_path='players_21.csv'):
#     """
#     Load preferred foot from FIFA 21 dataset and fuzzy-match to StatsBomb players.
#     """
#     fifa = pd.read_csv(fifa_csv_path, usecols=['short_name', 'long_name', 'preferred_foot'],
#                        low_memory=False)
#     fifa = fifa.drop_duplicates('long_name')
#     print(f"FIFA 21: {len(fifa)} players loaded")
#     print(f"Foot distribution:\n{fifa['preferred_foot'].value_counts()}")
#     return fifa


# def match_fifa_to_statsbomb(statsbomb_players, fifa_df, threshold=85):
#     """
#     Fuzzy name match StatsBomb players to FIFA dataset.
#     Returns dataframe with preferred_foot added.
#     """
#     fifa_names   = fifa_df['long_name'].tolist()
#     matched      = []

#     for _, row in statsbomb_players.iterrows():
#         sb_name = row['player']

#         # Try exact match first
#         exact = fifa_df[fifa_df['long_name'].str.lower() == sb_name.lower()]
#         if len(exact) > 0:
#             foot = exact.iloc[0]['preferred_foot']
#             matched.append({'player_id': row['player_id'], 'player': sb_name,
#                             'preferred_foot': foot, 'match_score': 100,
#                             'matched_name': exact.iloc[0]['long_name']})
#             continue

#         # Fuzzy match
#         best_match, score = process.extractOne(
#             sb_name, fifa_names,
#             scorer=fuzz.token_sort_ratio
#         )
#         if score >= threshold:
#             foot = fifa_df[fifa_df['long_name'] == best_match].iloc[0]['preferred_foot']
#             matched.append({'player_id': row['player_id'], 'player': sb_name,
#                             'preferred_foot': foot, 'match_score': score,
#                             'matched_name': best_match})
#         else:
#             matched.append({'player_id': row['player_id'], 'player': sb_name,
#                             'preferred_foot': None, 'match_score': score,
#                             'matched_name': best_match})

#     df = pd.DataFrame(matched)
#     matched_count = df['preferred_foot'].notna().sum()
#     print(f"\nMatched: {matched_count}/{len(df)} players ({matched_count/len(df):.1%})")
#     print(f"Unmatched: {df[df.preferred_foot.isna()]['player'].tolist()}")
#     return df

# # Uncomment when you have players_21.csv downloaded:
# # fifa_df  = load_fifa_foot('players_21.csv')
# # foot_df  = match_fifa_to_statsbomb(players_df, fifa_df)
# # foot_df.to_csv('preferred_foot_fifa.csv', index=False)


# # ─────────────────────────────────────────────────────────────────────────────
# # APPROACH D — Wikidata SPARQL (free, open, citable — partial coverage)
# # ─────────────────────────────────────────────────────────────────────────────
# ### def fetch_wikidata_foot():
#     """
#     Fetch preferred foot for all footballers from Wikidata public SPARQL endpoint.
#     Property P2354 = preferred foot.
#     """
#     endpoint = "https://query.wikidata.org/sparql"
#     query    = """
#     SELECT DISTINCT ?playerLabel ?foot ?footLabel WHERE {
#       ?player wdt:P106 wd:Q937857 .   # occupation: footballer
#       ?player wdt:P2354 ?foot .        # preferred foot
#       SERVICE wikibase:label {
#         bd:serviceParam wikibase:language "en" .
#       }
#     }
#     """
#     r = requests.get(
#         endpoint,
#         params={'query': query, 'format': 'json'},
#         headers={
#             'User-Agent': 'xPassResearch/1.0 (academic; contact: your@email.com)',
#             'Accept':     'application/sparql-results+json',
#         },
#         timeout=60
#     )
#     if r.status_code != 200:
#         print(f"Wikidata error: {r.status_code}")
#         return pd.DataFrame()

#     results = r.json()['results']['bindings']
#     rows    = [{'player_wiki':    p['playerLabel']['value'],
#                 'preferred_foot': p['footLabel']['value'].replace(' foot','')}
#                for p in results]
#     df = pd.DataFrame(rows).drop_duplicates('player_wiki')
#     print(f"Wikidata: {len(df)} players with foot data")
#     return df


#  def match_wikidata_to_statsbomb(statsbomb_players, wikidata_df, threshold=85):
#     """Fuzzy match Wikidata player names to StatsBomb names."""
#     wiki_names = wikidata_df['player_wiki'].tolist()
#     matched    = []

#     for _, row in statsbomb_players.iterrows():
#         sb_name    = row['player']
#         best, score = process.extractOne(sb_name, wiki_names, scorer=fuzz.token_sort_ratio)
#         if score >= threshold:
#             foot = wikidata_df[wikidata_df['player_wiki'] == best].iloc[0]['preferred_foot']
#             matched.append({'player_id': row['player_id'], 'player': sb_name,
#                             'preferred_foot': foot, 'score': score, 'matched': best})
#         else:
#             matched.append({'player_id': row['player_id'], 'player': sb_name,
#                             'preferred_foot': None, 'score': score, 'matched': best})

#     df = pd.DataFrame(matched)
#     print(f"Wikidata match: {df.preferred_foot.notna().sum()}/{len(df)} players")
#     return df

# Uncomment to run:
# wiki_df   = fetch_wikidata_foot()
# foot_wiki = match_wikidata_to_statsbomb(players_df, wiki_df)
# foot_wiki.to_csv('preferred_foot_wikidata.csv', index=False)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Merge whichever foot source worked back into your passes dataframe
# Replace foot_df with whichever approach returned data
# ─────────────────────────────────────────────────────────────────────────────
def merge_real_foot(passes_df, foot_df, foot_col='preferred_foot'):
    """
    Replace the inferred weak_foot with ground-truth preferred foot.
    foot_df must have columns: player_id, preferred_foot ('Right'/'Left'/'Both')
    """
    # Drop old inferred columns
    drop_cols = ['preferred_foot', 'total', 'weak_foot']
    passes_clean = passes_df.drop(columns=[c for c in drop_cols if c in passes_df.columns])

    # Merge real foot
    passes_clean = passes_clean.merge(
        foot_df[['player_id', foot_col]].rename(columns={foot_col: 'preferred_foot'}),
        on='player_id', how='left'
    )

    # Flag weak foot using real data
    def flag_weak_real(row):
        if row['pass_body_part'] not in ('Right Foot', 'Left Foot'):
            return np.nan
        if row.get('preferred_foot') not in ('Right', 'Left'):
            return np.nan
        used = 'Right' if row['pass_body_part'] == 'Right Foot' else 'Left'
        return int(used != row['preferred_foot'])

    passes_clean['weak_foot'] = passes_clean.apply(flag_weak_real, axis=1)

    # Coverage report
    total_foot = passes_clean['pass_body_part'].isin(['Right Foot','Left Foot']).sum()
    covered    = passes_clean[passes_clean['pass_body_part'].isin(
                     ['Right Foot','Left Foot'])]['preferred_foot'].notna().sum()
    print(f"\nCoverage: {covered:,}/{total_foot:,} foot passes "
          f"({covered/total_foot:.1%}) have ground-truth preferred foot")
    print(f"Weak foot dist: {passes_clean['weak_foot'].value_counts(dropna=False).to_dict()}")

    return passes_clean

# After running one of the approaches above:
# passes = merge_real_foot(passes, foot_df)
# Then re-run Cell 5 onwards — everything else stays the same


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Validate: compare inferred vs ground-truth preferred foot
# (Tells you how accurate your inference was)
# ─────────────────────────────────────────────────────────────────────────────
def validate_inference(passes_inferred, foot_ground_truth):
    """
    Compare inferred preferred foot (from pass usage) against ground truth.
    Shows accuracy and where the inference goes wrong.
    """
    merged = passes_inferred[['player_id','player','preferred_foot']]\
             .drop_duplicates('player_id')\
             .rename(columns={'preferred_foot': 'inferred'})\
             .merge(
                 foot_ground_truth[['player_id','preferred_foot']]
                 .rename(columns={'preferred_foot': 'ground_truth'}),
                 on='player_id', how='inner'
             )

    valid    = merged[merged['inferred'].isin(['Right','Left'])].copy()
    correct  = (valid['inferred'] == valid['ground_truth']).sum()
    accuracy = correct / len(valid)

    print(f"\n── Foot Inference Validation ──")
    print(f"Players compared: {len(valid)}")
    print(f"Accuracy:         {accuracy:.1%}")
    print(f"\nMismatches (inferred ≠ ground truth):")
    mismatches = valid[valid['inferred'] != valid['ground_truth']]
    print(mismatches[['player','inferred','ground_truth']].to_string(index=False))
    return accuracy

# After getting ground truth:
# validate_inference(passes, foot_df)