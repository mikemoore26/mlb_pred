# utils/data_fetchers/odds_elo.py
from __future__ import annotations
import datetime as _dt
from typing import Optional
import utils.cache as cache
from utils.safe_get_json import _safe_get_json

def _call(method: str, *args, default=None):
    """
    Mockable external calls (Elo, odds). Replace URLs with your providers later.
    """
    try:
        if method == "get_elo_diff":
            home, away, game_date = args
            url = f"https://api.example.com/elo?home={home}&away={away}&date={game_date.isoformat()}"
            data = _safe_get_json(url)
            return (data or {}).get("elo_diff", default)
        elif method == "get_closing_odds_implied_home":
            (game_pk,) = args
            url = f"https://api.example.com/odds?gamePk={game_pk}"
            data = _safe_get_json(url)
            return (data or {}).get("implied_home_odds", default)
        return default
    except Exception:
        return default

def cc_elo(home: str, away: str, game_iso_date: str) -> float:
    k = cache._make_key("elo", home, away, game_iso_date)
    hit = cache.load_json("elo", k, max_age_days=7)
    if hit is not None:
        try:
            return float(hit)
        except Exception:
            return 0.0
    d = _dt.date.fromisoformat(game_iso_date)
    data = _call("get_elo_diff", home, away, d, default=0.0)
    cache.save_json("elo", data, k)
    try:
        return float(data)
    except Exception:
        return 0.0

def cc_odds(game_pk: Optional[int]) -> Optional[float]:
    if not game_pk:
        return None
    k = cache._make_key("odds_imp_home", game_pk)
    hit = cache.load_json("odds_imp_home", k, max_age_days=1)
    if hit is not None:
        try:
            return float(hit) if hit is not None else None
        except Exception:
            return None
    data = _call("get_closing_odds_implied_home", game_pk, default=None)
    cache.save_json("odds_imp_home", data, k)
    try:
        return float(data) if data is not None else None
    except Exception:
        return None
