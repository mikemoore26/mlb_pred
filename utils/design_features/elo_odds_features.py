# utils/design_feature/elo_odds_features.py
from __future__ import annotations
from typing import Dict, Optional
from utils import data_fetchers as DF
from utils.design_features._common import f

DEBUG_ELO = False

def elo_feature(home_team: str, away_team: str, game_iso: str) -> Dict[str, float]:
    """
    Returns: {"elo_diff": float}
    """
    ed = DF.cc_elo(home_team, away_team, game_iso)
    out = {"elo_diff": f(ed, 0.0)}
    if DEBUG_ELO:
        print("[elo_feature]", home_team, away_team, ed, "->", out)
    return out

def odds_feature(game_pk: Optional[int], include_odds: bool) -> Dict[str, float]:
    """
    Returns {} if include_odds=False or no data, else {"odds_implied_home_close": float}
    """
    if not include_odds or not game_pk:
        return {}
    val = DF.cc_odds(game_pk)
    return {} if val is None else {"odds_implied_home_close": f(val, 0.0)}
