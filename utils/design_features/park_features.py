# utils/design_feature/park_features.py
from __future__ import annotations
from typing import Dict
from utils import data_fetchers as DF
from utils.design_features._common import f

DEBUG_PARK = False

def park_factor_feature(home_team: str) -> Dict[str, float]:
    """
    Returns: {"park_factor": float}
    """
    pf = DF.cc_park_factor(home_team)
    out = {"park_factor": f(pf, 1.0)}
    if DEBUG_PARK:
        print("[park_factor_feature]", home_team, pf, "->", out)
    return out
