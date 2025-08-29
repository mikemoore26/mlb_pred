# utils/design_features/pitcher_features.py
from __future__ import annotations
from typing import Dict, Any, Optional
from utils import data_fetchers as DF
from utils.design_features._common import f as _f

def _safe_kbb_from_rates(k9: Optional[float], bb9: Optional[float]) -> float:
    k9 = _f(k9, 0.0); bb9 = _f(bb9, 0.0)
    if bb9 > 0:
        return k9 / bb9
    return k9 if k9 > 0 else 0.0

def pitcher_diffs(home_pid: Optional[int], away_pid: Optional[int]) -> Dict[str, float]:
    # Season
    hs = DF.cc_pitcher_season(home_pid or 0) if home_pid else {}
    as_ = DF.cc_pitcher_season(away_pid or 0) if away_pid else {}

    h_era = _f((hs or {}).get("era"), 0.0)
    a_era = _f((as_ or {}).get("era"), 0.0)
    h_k9  = _f((hs or {}).get("k9"),  0.0)
    a_k9  = _f((as_ or {}).get("k9"),  0.0)
    h_bb9 = _f((hs or {}).get("bb9"), 0.0)
    a_bb9 = _f((as_ or {}).get("bb9"), 0.0)

    # Last 30 (rolling)
    hl30 = DF.cc_pitch_lastx(home_pid, days=30) if home_pid else {}
    al30 = DF.cc_pitch_lastx(away_pid, days=30) if away_pid else {}
    h_era30 = _f((hl30 or {}).get("era"), 0.0)
    a_era30 = _f((al30 or {}).get("era"), 0.0)

    # Last 3 starts (rates + safe K/BB)
    hl3 = DF.cc_pitch_lastx(home_pid, days=60, only_starts=True, limit_games=3) if home_pid else {}
    al3 = DF.cc_pitch_lastx(away_pid, days=60, only_starts=True, limit_games=3) if away_pid else {}
    h_era3  = _f((hl3 or {}).get("era"), 0.0)
    a_era3  = _f((al3 or {}).get("era"), 0.0)

    # Prefer direct totals if fetcher provides them, else derive from rates
    h_kbb3 = _safe_kbb_from_rates((hl3 or {}).get("k9"), (hl3 or {}).get("bb9"))
    a_kbb3 = _safe_kbb_from_rates((al3 or {}).get("k9"), (al3 or {}).get("bb9"))

    return {
        "starter_era_diff":   h_era - a_era,
        "starter_k9_diff":    h_k9  - a_k9,
        "starter_bb9_diff":   h_bb9 - a_bb9,
        "starter_era30_diff": h_era30 - a_era30,
        "starter_era3_diff":  h_era3  - a_era3,
        "starter_kbb3_diff":  h_kbb3  - a_kbb3,
    }
