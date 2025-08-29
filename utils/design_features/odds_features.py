# utils/design_features/odds_features.py
from __future__ import annotations
import pandas as pd
import numpy as np

try:
    from utils.data_fetchers.park_venue import norm_team_name
except Exception:
    def norm_team_name(x: str) -> str:
        return str(x).strip()

def _de_vig_implied(prob_home_raw: float, prob_away_raw: float) -> tuple[float, float]:
    """
    Remove the overround: normalize raw implied probs so they sum to 1.
    """
    s = (prob_home_raw or 0.0) + (prob_away_raw or 0.0)
    if s <= 0:
        return np.nan, np.nan
    return prob_home_raw / s, prob_away_raw / s

def add_live_odds_features(games: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align odds to games by (game_date, home_team, away_team) after canonicalization.
    Adds:
      odds_bookmaker, odds_decimal_home, odds_decimal_away,
      odds_implied_home_raw, odds_implied_away_raw,
      odds_implied_home (de-vig), odds_implied_away (de-vig)
    """
    if games is None or games.empty or odds_df is None or odds_df.empty:
        # add empty columns so downstream code doesn't break
        g = games.copy()
        for c in [
            "odds_bookmaker","odds_decimal_home","odds_decimal_away",
            "odds_implied_home_raw","odds_implied_away_raw",
            "odds_implied_home","odds_implied_away"
        ]:
            g[c] = np.nan
        return g

    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce").dt.date.astype(str)
    g["home_team_norm"] = g["home_team"].map(norm_team_name)
    g["away_team_norm"] = g["away_team"].map(norm_team_name)

    # Build both “home=team1/away=team2” and swapped keys because API team1/team2 are unordered
    odds = odds_df.copy()
    odds["team1_norm"] = odds["team1"].map(norm_team_name)
    odds["team2_norm"] = odds["team2"].map(norm_team_name)

    # Try direct alignment
    key_games = g[["game_date","home_team_norm","away_team_norm"]].copy()
    key_games["k"] = key_games["game_date"] + "|" + key_games["home_team_norm"] + "|" + key_games["away_team_norm"]

    key_odds_direct = odds[["game_date","team1_norm","team2_norm","bookmaker","price1","price2"]].copy()
    key_odds_direct["k"] = key_odds_direct["game_date"] + "|" + key_odds_direct["team1_norm"] + "|" + key_odds_direct["team2_norm"]

    key_odds_swapped = odds[["game_date","team1_norm","team2_norm","bookmaker","price1","price2"]].copy()
    # swap to cover (away, home)
    key_odds_swapped = key_odds_swapped.rename(columns={
        "team1_norm":"team2_norm","team2_norm":"team1_norm","price1":"price2","price2":"price1"
    })
    key_odds_swapped["k"] = key_odds_swapped["game_date"] + "|" + key_odds_swapped["team1_norm"] + "|" + key_odds_swapped["team2_norm"]

    # combine
    odds_all = pd.concat([key_odds_direct, key_odds_swapped], ignore_index=True)

    merged = g.merge(
        odds_all[["k","bookmaker","price1","price2"]],
        left_on="k", right_on="k", how="left"
    ).rename(columns={
        "price1": "odds_decimal_home",
        "price2": "odds_decimal_away",
        "bookmaker": "odds_bookmaker"
    })

    # raw implied probs from decimal odds
    merged["odds_implied_home_raw"] = 1.0 / merged["odds_decimal_home"].astype(float)
    merged["odds_implied_away_raw"] = 1.0 / merged["odds_decimal_away"].astype(float)

    # de-vig normalization
    merged[["odds_implied_home","odds_implied_away"]] = merged.apply(
        lambda r: pd.Series(_de_vig_implied(r["odds_implied_home_raw"], r["odds_implied_away_raw"])),
        axis=1
    )

    # cleanup
    merged = merged.drop(columns=["home_team_norm","away_team_norm","k"], errors="ignore")
    return merged
