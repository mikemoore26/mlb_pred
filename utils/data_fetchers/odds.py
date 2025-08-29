# utils/data_fetchers/odds.py
from __future__ import annotations
import os
import requests
import pandas as pd

API_KEY = os.getenv("THE_ODDS_API_KEY")  # set in your env
BASE_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

def cc_odds(market: str = "h2h", regions: str = "us") -> pd.DataFrame:
    """
    Fetch MLB odds (default: moneyline).
    market: 'h2h' (moneyline), 'spreads', 'totals'
    regions: 'us', 'uk', 'eu', 'au'
    """
    url = f"{BASE_URL}?apiKey={API_KEY}&regions={regions}&markets={market}&oddsFormat=decimal"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    # Flatten into DataFrame
    rows = []
    for g in data:
        gid = g["id"]
        teams = g["teams"]
        commence = g["commence_time"]
        for b in g["bookmakers"]:
            book = b["title"]
            for mk in b["markets"]:
                if mk["key"] != market:
                    continue
                for out in mk["outcomes"]:
                    rows.append({
                        "game_id": gid,
                        "commence_time": commence,
                        "bookmaker": book,
                        "team": out["name"],
                        "price": out["price"],
                        "point": out.get("point"),
                        "market": mk["key"]
                    })
    return pd.DataFrame(rows)
