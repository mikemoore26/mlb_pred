# utils/design_features/elo_features.py
from __future__ import annotations
import pandas as pd
from utils.data_fetchers.elo import cc_elo_from_games

def elo_feature(games: pd.DataFrame) -> pd.DataFrame:
    """
    Attach Elo features to the *entire* games DataFrame (chronologically).
    Requires columns:
      - 'game_date','home_team','away_team'
      - plus either 'home_win' OR ('home_runs','away_runs') for finals.

    Adds:
      - 'elo_home','elo_away','elo_diff','elo_prob_home'
    """
    if games is None or games.empty:
        return games

    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce")
    g = g.dropna(subset=["game_date"]).sort_values("game_date").reset_index(drop=True)

    elo_df = cc_elo_from_games(g)
    elo_df.index = g.index
    out = pd.concat([g, elo_df], axis=1)
    return out
