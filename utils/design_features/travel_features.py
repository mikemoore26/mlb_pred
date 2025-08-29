# utils/design_features/travel_feature.py
from __future__ import annotations
from typing import Dict
# Central fetch layer (may or may not have cc_travel; we guard below)
from utils import data_fetchers as DF

def travel_km(home_team: str, away_team: str, game_iso_date: str) -> Dict[str, float]:
    """
    Feature wrapper: returns travel distance from each team's previous venue to today's venue (km).
    Keys:
      - travel_km_home_prev_to_today
      - travel_km_away_prev_to_today
    If the fetcher can't resolve, returns 0.0s (never throws).
    """
    try:
        if hasattr(DF, "cc_travel"):
            out = DF.cc_travel(home_team, away_team, game_iso_date) or {}
            # Normalize output & never throw
            return {
                "travel_km_home_prev_to_today": float(out.get("travel_km_home_prev_to_today", 0.0) or 0.0),
                "travel_km_away_prev_to_today": float(out.get("travel_km_away_prev_to_today", 0.0) or 0.0),
            }
    except Exception:
        pass
    # Safe fallback
    return {
        "travel_km_home_prev_to_today": 0.0,
        "travel_km_away_prev_to_today": 0.0,
    }
