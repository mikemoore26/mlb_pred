# utils/design_feature/bullpen_features.py
from __future__ import annotations
from typing import Dict
from utils import data_fetchers as DF
from utils.design_features.utils import f

DEBUG_BULLPEN = False

def bullpen_diff_era14(home_team: str, away_team: str) -> Dict[str, float]:
    """
    Returns: {"bullpen_era14_diff": float}
    Uses DF.cc_bullpen_era14() dict {team -> era_last14}, treats missing as 0.0 for the diff.
    """
    bp14 = DF.cc_bullpen_era14() or {}
    h = bp14.get(home_team, None)
    a = bp14.get(away_team, None)
    diff = f((0.0 if h is None else h) - (0.0 if a is None else a))
    out = {"bullpen_era14_diff": diff}
    if DEBUG_BULLPEN:
        print("[bullpen_diff_era14]", home_team, h, "|", away_team, a, "->", out)
    return out

def bullpen_ip_last3(home_team: str, away_team: str, game_iso: str) -> Dict[str, float]:
    """
    Returns: {"bullpen_ip_last3_home": float, "bullpen_ip_last3_away": float}
    Uses DF.cc_bullpen_ip_last3(team, game_iso) → Optional[float]
    """
    ip_h = DF.cc_bullpen_ip_last3(home_team, game_iso)
    ip_a = DF.cc_bullpen_ip_last3(away_team, game_iso)
    out = {
        "bullpen_ip_last3_home": f(ip_h, 0.0),
        "bullpen_ip_last3_away": f(ip_a, 0.0),
    }
    if DEBUG_BULLPEN:
        print("[bullpen_ip_last3]", home_team, ip_h, "|", away_team, ip_a, "->", out)
    return out
