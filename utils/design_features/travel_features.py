# utils/design_feature/travel_features.py
from __future__ import annotations
from typing import Dict
from utils import data_fetchers as DF
from utils.design_features.utils import f

DEBUG_TRAVEL = False

def travel_km(home_team: str, away_team: str, game_iso: str) -> Dict[str, float]:
    """
    Returns:
      - travel_km_home_prev_to_today
      - travel_km_away_prev_to_today
    """
    km_h, km_a = DF.cc_travel(home_team, away_team, game_iso)
    out = {
        "travel_km_home_prev_to_today": f(km_h, 0.0),
        "travel_km_away_prev_to_today": f(km_a, 0.0),
    }
    if DEBUG_TRAVEL:
        print("[travel_km]", home_team, km_h, "|", away_team, km_a, "->", out)
    return out
