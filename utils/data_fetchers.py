from __future__ import annotations
from typing import Any, Optional, Iterable, Tuple, Dict, List
import importlib
import hashlib
from typing import Optional

from utils import cache
from utils.sources import get_bullpen_ip_last3

S = importlib.import_module("utils.sources")

def _call(name: str, *args, default=None, **kwargs):
    fn = getattr(S, name, None)
    if not callable(fn):
        print(f"Error: No function '{name}' in utils.sources")
        return default
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"Error calling '{name}' with args {args}, kwargs {kwargs}: {str(e)}")
        return default

def _hash_list(x: Iterable[Any]) -> str:
    s = ",".join(str(xx) for xx in x)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

def cc_schedule(start_date: str, end_date: str) -> List[dict]:
    cached = cache.load_json("schedule", start_date, end_date, max_age_days=1)
    if cached is not None:
        return cached
    data = _call("get_schedule_range", start_date, end_date, default=[]) or []
    cache.save_json("schedule", data, start_date, end_date)
    return data

def cc_wpct_season(season: Optional[int] = None) -> Dict[str, float]:
    tag = season or "cur"
    cached = cache.load_json("wpct_season", tag, max_age_days=3)
    if cached is not None:
        return cached
    data = _call("get_team_wpct_season", season, default={}) or {}
    cache.save_json("wpct_season", data, tag)
    return data

def cc_wpct_last30() -> Dict[str, float]:
    cached = cache.load_json("wpct_last30", "global", max_age_days=1)
    if cached is not None:
        return cached
    data = _call("get_team_wpct_last30", default={}) or {}
    cache.save_json("wpct_last30", data, "global")
    return data

def cc_offense30() -> Dict[str, float]:
    cached = cache.load_json("offense30", "global", max_age_days=1)
    if cached is not None:
        return cached
    data = _call("get_team_offense_runs_pg_30d", default={}) or {}
    cache.save_json("offense30", data, "global")
    return data

def cc_bullpen_era14() -> Dict[str, float]:
    cached = cache.load_json("bp14", "global", max_age_days=1)
    if cached is not None:
        return cached
    data = _call("get_bullpen_era_last14", default={}) or {}
    cache.save_json("bp14", data, "global")
    return data

def cc_bullpen_ip_last3(team: str, game_date: str) -> Optional[float]:
    """Cached wrapper for get_bullpen_ip_last3."""
    pk = cache._make_key("bullpen_ip_last3", team, game_date)  # Use _make_key
    cached = cache.load_json("bullpen_ip_last3", pk, max_age_days=7)
    if cached is not None:
        return cached
    data = get_bullpen_ip_last3(team, game_date)
    if data is not None:
        cache.save_json("bullpen_ip_last3", data, pk)
    return data

def cc_probables(game_pks: List[int]) -> Dict[int, dict]:
    key = _hash_list(sorted([pk for pk in game_pks if pk]))
    cached = cache.load_json("probables", key, max_age_days=1)
    if cached is not None:
        return cached
    data = _call("get_probable_pitchers", game_pks, default={}) or {}
    cache.save_json("probables", data, key)
    return data

def cc_pitcher_season(player_id: Optional[int]) -> dict:
    if not player_id:
        return {"era": None, "k9": None, "bb9": None}
    cached = cache.load_json("p_season", player_id, max_age_days=7)
    if cached is not None:
        return cached
    data = _call("get_pitcher_season_stats", player_id, default={"era": None, "k9": None, "bb9": None}) or {}
    cache.save_json("p_season", data, player_id)
    return data



def cc_bullpen_ip_last3(team: str, game_date: str) -> Optional[float]:
    """Cached wrapper for get_bullpen_ip_last3."""
    pk = cache._make_key("bullpen_ip_last3", team, game_date)  # Use _make_key
    cached = cache.load_json("bullpen_ip_last3", pk, max_age_days=7)
    if cached is not None:
        return cached
    data = _call("get_bullpen_ip_last3", team, game_date, default=None)
    if data is not None:
        cache.save_json("bullpen_ip_last3", data, pk)
    return data
def cc_pitcher_last30(player_id: Optional[int]) -> dict:
    if not player_id:
        return {"era30": None}
    cached = cache.load_json("p_last30", player_id, max_age_days=1)
    if cached is not None:
        return cached
    data = _call("get_pitcher_last30", player_id, default={"era30": None}) or {}
    cache.save_json("p_last30", data, player_id)
    return data

def cc_weather(game_dict: dict):
    pk = (game_dict or {}).get("gamePk") or 0
    cached = cache.load_json("weather", pk, max_age_days=1)
    if cached is not None:
        return cached
    data = _call("get_weather_for_game", game_dict, default=None)
    if data is None:
        print(f"Warning: Using default weather for gamePk {pk}")
        data = {"temp": 72.0, "wind_speed": 5.0, "wind_out_to_cf": False, "wind_cf_x_park": 0.0}
    cache.save_json("weather", data, pk)
    return data

def cc_days_rest(team_name: str, game_iso_date: str) -> int:
    cached = cache.load_json("rest", team_name, game_iso_date, max_age_days=5)
    if cached is not None:
        return int(cached)
    from datetime import date as _date
    d = _date.fromisoformat(game_iso_date)
    data = _call("get_days_rest", team_name, d, default=1)
    cache.save_json("rest", data, team_name, game_iso_date)
    return int(data or 1)

def cc_travel(home: str, away: str, game_iso_date: str) -> Tuple[float, float]:
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
    cached = cache.load_json("odds_imp_home", game_pk, max_age_days=1)
    if cached is not None:
        return cached
    data = _call("get_closing_odds_implied_home", game_pk, default=None)
    cache.save_json("odds_imp_home", data, game_pk)
    return data


if __name__ == "__main__":
    from data_fetchers import cc_weather
    print(cc_weather({"officialDate": "2024-09-14", "venue": {"name": "Yankee Stadium"}}))