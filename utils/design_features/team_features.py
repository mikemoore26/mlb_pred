# utils/design_feature/team_features.py
from __future__ import annotations
from typing import Dict
from utils import data_fetchers as DF
from utils.design_features.utils import f, i

DEBUG_TEAM = False

def team_form(home_team: str, away_team: str) -> Dict[str, float]:
    """
    Returns:
      - home_advantage (always 1)
      - team_wpct_diff_season
      - team_wpct_diff_30d
      - offense_runs_pg_30d_diff
    """
    wp_season = DF.cc_wpct_season() or {}
    wp_30     = DF.cc_wpct_last30() or {}
    off_30    = DF.cc_offense30() or {}

    out = {
        "home_advantage": 1.0,
        "team_wpct_diff_season": f((wp_season.get(home_team, 0.5) or 0.5) - (wp_season.get(away_team, 0.5) or 0.5)),
        "team_wpct_diff_30d":    f((wp_30.get(home_team, 0.5)     or 0.5) - (wp_30.get(away_team, 0.5)     or 0.5)),
        "offense_runs_pg_30d_diff": f((off_30.get(home_team, 0.0) or 0.0) - (off_30.get(away_team, 0.0)    or 0.0)),
    }
    if DEBUG_TEAM:
        print("[team_form]", home_team, away_team, "->", out)
    return out
