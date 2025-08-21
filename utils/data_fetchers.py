# utils/data_fetchers.py
from __future__ import annotations
from typing import Any, Optional, Iterable, Tuple, Dict, List
import importlib
import hashlib

# Disk cache helpers (you already have utils/cache.py with load_json/save_json)
from utils import cache

# Dynamically load the live-data sources (StatsAPI, etc.)
S = importlib.import_module("utils.sources")


# ---------- tiny helpers ----------
def _call(name: str, *args, default=None, **kwargs):
    """Call utils.sources.<name>(...) safely; return default on any error."""
    fn = getattr(S, name, None)
    if not callable(fn):
        return default
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


def _hash_list(x: Iterable[Any]) -> str:
    s = ",".join(str(xx) for xx in x)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


# ---------- CACHED FETCHERS (RAW DATA ONLY) ----------

def cc_schedule(start_date: str, end_date: str) -> List[dict]:
    """
    Cached schedule for an inclusive date range.
    Refresh daily.
    """
    cached = cache.load_json("schedule", start_date, end_date, max_age_days=1)
    if cached is not None:
        return cached
    data = _call("get_schedule_range", start_date, end_date, default=[]) or []
    cache.save_json("schedule", data, start_date, end_date)
    return data


def cc_wpct_season(season: Optional[int] = None) -> Dict[str, float]:
    """
    Cached season-long win% by team.
    Refresh every 3 days.
    """
    tag = season or "cur"
    cached = cache.load_json("wpct_season", tag, max_age_days=3)
    if cached is not None:
        return cached
    data = _call("get_team_wpct_season", season, default={}) or {}
    cache.save_json("wpct_season", data, tag)
    return data


def cc_wpct_last30() -> Dict[str, float]:
    """Cached last-30-days win% by team. Refresh daily."""
    cached = cache.load_json("wpct_last30", "global", max_age_days=1)
    if cached is not None:
        return cached
    data = _call("get_team_wpct_last30", default={}) or {}
    cache.save_json("wpct_last30", data, "global")
    return data


def cc_bullpen_era14() -> Dict[str, float]:
    """Cached team bullpen ERA (approx) last 14 days. Refresh daily."""
    cached = cache.load_json("bp14", "global", max_age_days=1)
    if cached is not None:
        return cached
    data = _call("get_bullpen_era_last14", default={}) or {}
    cache.save_json("bp14", data, "global")
    return data


def cc_offense30() -> Dict[str, float]:
    """Cached team offense (runs/game) last 30 days. Refresh daily."""
    cached = cache.load_json("offense30", "global", max_age_days=1)
    if cached is not None:
        return cached
    data = (_call("get_team_offense_runs_pg_30d", default=None)
            or _call("get_offense_runs_pg_30d", default={})
            or {})
    cache.save_json("offense30", data, "global")
    return data


def cc_probables(game_pks: List[int]) -> Dict[int, dict]:
    """Cached probable pitchers by gamePk set. Refresh same day."""
    key = _hash_list(sorted([pk for pk in game_pks if pk]))
    cached = cache.load_json("probables", key, max_age_days=1)
    if cached is not None:
        return cached
    data = _call("get_probable_pitchers", game_pks, default={}) or {}
    cache.save_json("probables", data, key)
    return data


def cc_pitcher_season(player_id: Optional[int]) -> dict:
    """Cached pitcher season stats {era,k9,bb9}. Refresh weekly."""
    if not player_id:
        return {"era": None, "k9": None, "bb9": None}
    cached = cache.load_json("p_season", player_id, max_age_days=7)
    if cached is not None:
        return cached
    data = _call("get_pitcher_season_stats", player_id, default={"era": None, "k9": None, "bb9": None}) or {}
    cache.save_json("p_season", data, player_id)
    return data


def cc_pitcher_last3(player_id: Optional[int]) -> dict:
    """Cached last-3-starts {era3,kbb3}. Refresh every 2 days."""
    if not player_id:
        return {"era3": None, "kbb3": None}
    cached = cache.load_json("p_last3", player_id, max_age_days=2)
    if cached is not None:
        return cached
    data = _call("get_pitcher_last3_starts", player_id, default={"era3": None, "kbb3": None}) or {}
    cache.save_json("p_last3", data, player_id)
    return data


def cc_weather(game_dict: dict):
    """Cached weather payload per gamePk. Refresh same day."""
    pk = (game_dict or {}).get("gamePk") or 0
    cached = cache.load_json("weather", pk, max_age_days=1)
    if cached is not None:
        return cached
    data = _call("get_weather_for_game", game_dict, default=None)
    cache.save_json("weather", data, pk)
    return data


def cc_days_rest(team_name: str, game_iso_date: str) -> int:
    """Cached days rest per team for a specific ISO date. Refresh every 5 days."""
    cached = cache.load_json("rest", team_name, game_iso_date, max_age_days=5)
    if cached is not None:
        return int(cached)
    from datetime import date as _date
    d = _date.fromisoformat(game_iso_date)
    data = _call("get_days_rest", team_name, d, default=1)
    cache.save_json("rest", data, team_name, game_iso_date)
    return int(data or 1)


def cc_travel(home: str, away: str, game_iso_date: str) -> Tuple[float, float]:
    """Cached (home_km, away_km) travel to today's park. Refresh every 5 days."""
    cached = cache.load_json("travel", home, away, game_iso_date, max_age_days=5)
    if cached is not None:
        try:
            a, b = cached
            return float(a), float(b)
        except Exception:
            pass
    from datetime import date as _date
    d = _date.fromisoformat(game_iso_date)
    data = _call("get_travel_km_to_today", home, away, d, default=(0.0, 0.0)) or (0.0, 0.0)
    cache.save_json("travel", data, home, away, game_iso_date)
    a, b = data
    return float(a), float(b)


def cc_park_factor(home_team: str) -> float:
    """Cached park factor per home team. Refresh every 30 days."""
    cached = cache.load_json("park", home_team, max_age_days=30)
    if cached is not None:
        try:
            return float(cached)
        except Exception:
            return 0.0
    data = _call("get_park_factor", home_team, default=0.0)
    cache.save_json("park", data, home_team)
    try:
        return float(data)
    except Exception:
        return 0.0


def cc_elo(home: str, away: str, game_iso_date: str) -> float:
    """Cached elo diff for a game. Refresh weekly."""
    cached = cache.load_json("elo", home, away, game_iso_date, max_age_days=7)
    if cached is not None:
        try:
            return float(cached)
        except Exception:
            return 0.0
    from datetime import date as _date
    d = _date.fromisoformat(game_iso_date)
    data = _call("get_elo_diff", home, away, d, default=0.0)
    cache.save_json("elo", data, home, away, game_iso_date)
    try:
        return float(data)
    except Exception:
        return 0.0


def cc_odds(game_pk: int):
    """Cached closing odds implied home probability (optional). Refresh same day."""
    cached = cache.load_json("odds_imp_home", game_pk, max_age_days=1)
    if cached is not None:
        return cached
    data = _call("get_closing_odds_implied_home", game_pk, default=None)
    cache.save_json("odds_imp_home", data, game_pk)
    return data
