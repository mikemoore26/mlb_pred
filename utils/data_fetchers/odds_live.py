# utils/data_fetchers/odds_live.py
from __future__ import annotations
import os
from typing import Iterable, List, Optional
import requests
import pandas as pd
from datetime import datetime, timezone

# Use your existing canonicalizer if you have one
try:
    from utils.data_fetchers.park_venue import norm_team_name
except Exception:
    def norm_team_name(x: str) -> str:
        return str(x).strip()

ODDS_API_KEY_ENV = "THE_ODDS_API_KEY"
BASE_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

BOOKMAKER_PRIORITY = ["FanDuel", "DraftKings", "BetMGM"]

def _to_date_utc(iso_str: str) -> str:
    # commence_time is ISO UTC; return YYYY-MM-DD (UTC) to align with your game_date
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    return dt.date().isoformat()

def _pick_bookmaker(bookmakers: List[dict], want: Optional[Iterable[str]]) -> Optional[dict]:
    if not bookmakers:
        return None
    if want:
        wanted = list(want)
    else:
        wanted = BOOKMAKER_PRIORITY
    # exact title match in priority order
    title_to_b = {b.get("title"): b for b in bookmakers}
    for name in wanted:
        if name in title_to_b:
            return title_to_b[name]
    # fall back to the first one
    return bookmakers[0]

def _moneyline_pair_from_market(mkt: dict) -> Optional[tuple[dict, dict]]:
    """
    Returns (home_outcome, away_outcome) from a 'h2h' market if possible.
    Outcomes look like {"name": "New York Yankees", "price": 1.85} for decimal.
    """
    if not mkt or mkt.get("key") != "h2h":
        return None
    outs = mkt.get("outcomes") or []
    if len(outs) != 2:
        return None
    return outs[0], outs[1]

def fetch_live_odds_df(
    *,
    market: str = "h2h",
    regions: str = "us",
    bookmakers: Optional[List[str]] = None,
    odds_format: str = "decimal",
) -> pd.DataFrame:
    """
    Fetch current MLB odds. Default: moneyline (h2h), US books, decimal prices.
    Returns rows with:
      ['game_date','bookmaker','home_team','away_team','home_price_dec','away_price_dec']
    """
    api_key = os.getenv(ODDS_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{ODDS_API_KEY_ENV} not set. Get a key and export it in your env.")

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": market,
        "oddsFormat": odds_format,
    }
    r = requests.get(BASE_URL, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()

    rows = []
    for g in payload:
        commence = g.get("commence_time")
        gdate = _to_date_utc(commence) if commence else None
        b = _pick_bookmaker(g.get("bookmakers") or [], bookmakers)
        if not b:
            continue
        # pick the moneyline market
        mkts = b.get("markets") or []
        m = next((m for m in mkts if m.get("key") == market), None)
        pair = _moneyline_pair_from_market(m)
        if not pair:
            continue

        a, b_ = pair  # two team outcomes (unordered)
        # normalize names
        t1 = norm_team_name(a.get("name"))
        t2 = norm_team_name(b_.get("name"))

        # The API doesn't mark home/away; we’ll align later when merging
        rows.append({
            "game_date": gdate,
            "bookmaker": b.get("title"),
            "team1": t1,
            "team2": t2,
            "price1": float(a.get("price")) if a.get("price") is not None else None,
            "price2": float(b_.get("price")) if b_.get("price") is not None else None,
        })

    df = pd.DataFrame(rows)
    # ensure date type
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype(str)
    return df
