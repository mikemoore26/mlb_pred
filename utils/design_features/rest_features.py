# utils/design_feature/rest_features.py
from __future__ import annotations
from typing import Dict
from utils import data_fetchers as DF
from utils.design_features.utils import i

DEBUG_REST = False

def rest_b2b(home_team: str, away_team: str, game_iso: str) -> Dict[str, int]:
    """
    Returns:
      - home_days_rest
      - away_days_rest
      - b2b_flag (1 if either has 0 days rest, else 0)
    """
    h = i(DF.cc_days_rest(home_team, game_iso), 1)
    a = i(DF.cc_days_rest(away_team, game_iso), 1)
    out = {
        "home_days_rest": h,
        "away_days_rest": a,
        "b2b_flag": 1 if (h == 0 or a == 0) else 0
    }
    if DEBUG_REST:
        print("[rest_b2b]", home_team, h, "|", away_team, a, "->", out)
    return out
