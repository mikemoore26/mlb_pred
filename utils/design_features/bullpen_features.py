# utils/design_features/bullpen_features.py
from __future__ import annotations
from typing import Dict

from utils import data_fetchers as DF

def _f(x, d: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(d)

def bullpen_diff_era14(home_team: str, away_team: str) -> Dict[str, float]:
    """
    Feature: bullpen_era14_diff = home_bullpen_era14 - away_bullpen_era14
    (lower is better, so negative values favor HOME bullpen)
    """
    bp = DF.cc_bullpen_era14() or {}
    home = _f(bp.get(home_team), 4.00)
    away = _f(bp.get(away_team), 4.00)
    return {"bullpen_era14_diff": home - away}

def bullpen_ip_last3(home_team: str, away_team: str, game_iso_date: str) -> Dict[str, float]:
    """
    Feature block for bullpen fatigue: relief IP accrued over last 3 days.
    """
    hip = _f(DF.cc_bullpen_ip_last3(home_team, game_iso_date))
    aip = _f(DF.cc_bullpen_ip_last3(away_team, game_iso_date))
    return {
        "bullpen_ip_last3_home": hip,
        "bullpen_ip_last3_away": aip,
    }
