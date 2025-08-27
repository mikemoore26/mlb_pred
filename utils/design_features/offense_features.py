# utils/design_feature/offense_features.py
from __future__ import annotations
from typing import Dict
from utils import data_fetchers as DF
from utils.design_features._common import f

DEBUG_OFFENSE = False

def offense_30d_diff(home_team: str, away_team: str) -> Dict[str, float]:
    """
    (Kept for symmetry; team_features.team_form already includes this.)
    Returns: {"offense_runs_pg_30d_diff": float}
    """
    off_30 = DF.cc_offense30() or {}
    out = {"offense_runs_pg_30d_diff": f((off_30.get(home_team, 0.0) or 0.0) - (off_30.get(away_team, 0.0) or 0.0))}
    if DEBUG_OFFENSE:
        print("[offense_30d_diff]", home_team, away_team, "->", out)
    return out
