# utils/design_feature/pitcher_features.py
from __future__ import annotations
from typing import Dict, Optional

from utils import data_fetchers as DF
from utils.design_features._common import f

# Toggle a little extra logging while you’re wiring things up
DEBUG_PITCHERS = False

def _norm_pitcher_season(payload: Optional[dict]) -> dict:
    """Normalize season stat payload -> guaranteed keys with floats."""
    payload = payload or {}
    return {
        "era": f(payload.get("era"), None),
        "k9":  f(payload.get("k9"),  None),
        "bb9": f(payload.get("bb9"), None),
    }

def _norm_last30(payload: Optional[dict]) -> dict:
    """Normalize last-30 range stats -> {'era30': float|None}."""
    payload = payload or {}
    val = payload.get("era30", None)
    return {"era30": (f(val) if val is not None else None)}

def _norm_last3(payload: Optional[dict]) -> dict:
    """Normalize last-3 starts -> {'era3': float|None, 'kbb3': float|None}."""
    payload = payload or {}
    e3 = payload.get("era3", None)
    kbb = payload.get("kbb3", None)
    return {
        "era3": (f(e3) if e3 is not None else None),
        "kbb3": (f(kbb) if kbb is not None else None),
    }

def pitcher_diffs(home_pitcher_id: Optional[int], away_pitcher_id: Optional[int]) -> Dict[str, float]:
    """
    Compute starter feature diffs, gracefully handling None / missing data.
    Returns 6 numeric features (floats; missing treated as 0.0 for diffs):
      starter_era_diff, starter_k9_diff, starter_bb9_diff,
      starter_era30_diff, starter_era3_diff, starter_kbb3_diff
    """
    # Season
    s_h = _norm_pitcher_season(DF.cc_pitcher_season(home_pitcher_id))
    s_a = _norm_pitcher_season(DF.cc_pitcher_season(away_pitcher_id))

    # Last-30 ERA (may be None)
    l30_h = _norm_last30(DF.cc_pitcher_last30(home_pitcher_id))
    l30_a = _norm_last30(DF.cc_pitcher_last30(away_pitcher_id))

    # Last-3 starts (may be None)
    l3_h = _norm_last3(DF.cc_pitcher_last3(home_pitcher_id))
    l3_a = _norm_last3(DF.cc_pitcher_last3(away_pitcher_id))

    # If either side is None, treat as 0.0 *for the diff only* (keeps model stable)
    era30_h = 0.0 if l30_h["era30"] is None else f(l30_h["era30"])
    era30_a = 0.0 if l30_a["era30"] is None else f(l30_a["era30"])
    era3_h  = 0.0 if l3_h["era3"]   is None else f(l3_h["era3"])
    era3_a  = 0.0 if l3_a["era3"]   is None else f(l3_a["era3"])
    kbb3_h  = 0.0 if l3_h["kbb3"]   is None else f(l3_h["kbb3"])
    kbb3_a  = 0.0 if l3_a["kbb3"]   is None else f(l3_a["kbb3"])

    out = {
        "starter_era_diff":   f((0 if s_h["era"] is None else s_h["era"]) - (0 if s_a["era"] is None else s_a["era"])),
        "starter_k9_diff":    f((0 if s_h["k9"]  is None else s_h["k9"])  - (0 if s_a["k9"]  is None else s_a["k9"])),
        "starter_bb9_diff":   f((0 if s_h["bb9"] is None else s_h["bb9"]) - (0 if s_a["bb9"] is None else s_a["bb9"])),
        "starter_era30_diff": f(era30_h - era30_a),
        "starter_era3_diff":  f(era3_h  - era3_a),
        "starter_kbb3_diff":  f(kbb3_h  - kbb3_a),
    }

    if DEBUG_PITCHERS:
        print("[pitcher_diffs] s_h:", s_h, "s_a:", s_a)
        print("[pitcher_diffs] l30_h:", l30_h, "l30_a:", l30_a)
        print("[pitcher_diffs] l3_h:", l3_h, "l3_a:", l3_a)
        print("[pitcher_diffs] out:", out)

    return out
