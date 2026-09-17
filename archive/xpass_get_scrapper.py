# =============================================================================
# FBref Preferred Foot — Fixed approach
# Paste this into your notebook after the failed Approach A cell
# =============================================================================

import soccerdata as sd
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# ── Step 1: Get player list + FBref URLs from soccerdata ──────────────────
fbref   = sd.FBref(leagues='ESP-La Liga', seasons='2020-2021')
bio     = fbref.read_player_season_stats(stat_type='standard')

# Reset index so we can inspect what's there
bio_reset = bio.reset_index()
print("Index/columns available:")
print(bio_reset.columns.tolist())
print(f"\nSample rows:")
print(bio_reset.head(3).to_string())

# ── Step 2: Get unique player names from the index ────────────────────────
# soccerdata uses MultiIndex — player name is usually one of the levels
if isinstance(bio.index, pd.MultiIndex):
    print("\nMultiIndex levels:", bio.index.names)
    # Player name is typically the last level
    player_level = [n for n in bio.index.names if 'player' in str(n).lower()]
    print("Player levels:", player_level)

# ── Step 3: Scrape foot from FBref player pages via soccerdata's session ──
# soccerdata caches FBref pages locally — we can reuse its scraper session

def get_foot_from_fbref_page(player_name, session, base_url="https://fbref.com"):
    """
    Search FBref for a player and return their preferred foot.
    Uses soccerdata-style scraping with polite delays.
    """
    # FBref search endpoint
    search_url = f"{base_url}/search/search.fcgi"
    try:
        r = session.get(
            search_url,
            params={'search': player_name, 'pid': 'search'},
            timeout=15
        )
        soup = BeautifulSoup(r.text, 'lxml')

        # If redirected directly to a player page
        if '/players/' in r.url:
            return _parse_foot_from_page(soup)

        # Otherwise find first player result link
        result = soup.select_one('div.search-item-name a')
        if not result:
            # Try alternate selector
            result = soup.select_one('div#players div.search-item a')
        if not result:
            return None

        time.sleep(1.0)
        r2   = session.get(base_url + result['href'], timeout=15)
        soup2 = BeautifulSoup(r2.text, 'lxml')
        return _parse_foot_from_page(soup2)

    except Exception as e:
        print(f"  Error: {e}")
        return None


def _parse_foot_from_page(soup):
    """Extract foot preference from a FBref player profile page."""
    # Foot is in the player biographical info section
    # Look for 'Footed:' label in the page
    full_text = soup.get_text()

    # Pattern: "Footed:\n Right" or "Footed: Right"
    match = re.search(r'Footed[:\s]+([A-Za-z]+)', full_text)
    if match:
        foot = match.group(1).strip().title()
        if foot in ('Right', 'Left', 'Both'):
            return foot

    # Alternative: look in the meta/bio div
    for p in soup.find_all('p'):
        txt = p.get_text()
        if 'Footed' in txt or 'footed' in txt:
            match = re.search(r'(Right|Left|Both)', txt)
            if match:
                return match.group(1)

    return None


# ── Step 4: Build a session reusing soccerdata's approach ─────────────────
session = requests.Session()
session.headers.update({
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://fbref.com',
})
# Warm up session with a landing page hit
session.get('https://fbref.com/en/', timeout=10)
time.sleep(2)


# ── Step 5: Scrape foot for all La Liga players ───────────────────────────
def scrape_fbref_foot(players_df, session, n_players=None, delay=1.5):
    """
    Scrape preferred foot from FBref player profile pages.
    players_df: DataFrame with 'player' and 'player_id' columns.
    """
    subset  = players_df.head(n_players) if n_players else players_df
    results = []

    for i, (_, row) in enumerate(subset.iterrows()):
        name = row['player']
        print(f"  [{i+1}/{len(subset)}] {name}...", end=' ', flush=True)
        foot = get_foot_from_fbref_page(name, session)
        print(foot or "not found")
        results.append({
            'player_id':     row['player_id'],
            'player':        name,
            'preferred_foot': foot
        })
        time.sleep(delay)   # polite delay between requests

    df = pd.DataFrame(results)
    print(f"\n── Results ──")
    print(f"Found:    {df.preferred_foot.notna().sum()}/{len(df)} players")
    print(f"Not found: {df[df.preferred_foot.isna()]['player'].tolist()}")
    print(f"\nDistribution:\n{df['preferred_foot'].value_counts(dropna=False)}")
    return df


# ── Run it ────────────────────────────────────────────────────────────────
# Quick test with 5 players first:
# foot_test = scrape_fbref_foot(players_df, session, n_players=5)

# Full run (~3-4 min for ~100 players):
# foot_fbref = scrape_fbref_foot(players_df, session)
# foot_fbref.to_csv('preferred_foot_fbref.csv', index=False)

# Then merge into your passes dataframe:
# passes = merge_real_foot(passes, foot_fbref)


# ── Alternative: check what other stat_types soccerdata supports ──────────
# Sometimes 'misc' or a roster read has biographical fields
print("\n── Checking other soccerdata stat types ──")
for stat_type in ['misc', 'shooting', 'passing', 'defense']:
    try:
        df_check = fbref.read_player_season_stats(stat_type=stat_type)
        df_r     = df_check.reset_index()
        foot_col = [c for c in df_r.columns if 'foot' in str(c).lower()]
        if foot_col:
            print(f"  ✓ '{stat_type}' HAS foot column: {foot_col}")
        else:
            print(f"  ✗ '{stat_type}': no foot column ({len(df_r)} rows, {len(df_r.columns)} cols)")
    except Exception as e:
        print(f"  ✗ '{stat_type}': error — {e}")