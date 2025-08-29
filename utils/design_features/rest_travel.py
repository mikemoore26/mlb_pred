# utils/data_fetchers/rest_travel.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
import math
import datetime as _dt

import utils.cache as cache
from utils.safe_get_json import _safe_get_json
from utils.data_fetchers.park_venue import cc_venue_latlon_from_gamepk

def _haversine_km(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    try:
        lat1, lon1 = map(float, a); lat2, lon2 = map(float, b)
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        x = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
        return float(2*R*math.asin(math.sqrt(max(0.0, min(1.0, x)))))
    except Exception:
        return 0.0

def _today_gamepk_for(home: str, away: str, date_iso: str) -> Optional[int]:
    url = (f"https://statsapi.mlb.com/api/v1/schedule?startDate={date_iso}&endDate={date_iso}&sportId=1")
    sched = _safe_get_json(url) or {}
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            ht = (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name","")
            at = (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name","")
            if ht == home and at == away:
                return g.get("gamePk")
    return None

def _prev_gamepk_for_team(team: str, date_iso: str, lookback_days: int = 10) -> Optional[int]:
    try:
        d = _dt.date.fromisoformat(date_iso)
    except Exception:
        return None
    start = (d - _dt.timedelta(days=lookback_days)).isoformat()
    end   = (d - _dt.timedelta(days=1)).isoformat()
    url = (f"https://statsapi.mlb.com/api/v1/schedule?startDate={start}&endDate={end}&sportId=1")
    sched = _safe_get_json(url) or {}
    last_pk = None
    last_date = None
    for dd in sched.get("dates", []):
        for g in dd.get("games", []):
            ht = (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name","")
            at = (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name","")
            if team in (ht, at):
                try:
                    gd = _dt.date.fromisoformat(dd.get("date"))
                except Exception:
                    continue
                if last_date is None or gd > last_date:
                    last_date = gd
                    last_pk = g.get("gamePk")
    return last_pk

def cc_travel(home_team: str, away_team: str, game_date_iso: str) -> Dict[str, float]:
    """
    Returns:
      {"travel_km_home_prev_to_today": float, "travel_km_away_prev_to_today": float}
    """
    ck = cache._make_key("travel", home_team, away_team, game_date_iso)
    hit = cache.load_json("travel", ck, max_age_days=3)
    if isinstance(hit, dict):
        try:
            return {
                "travel_km_home_prev_to_today": float(hit.get("travel_km_home_prev_to_today", 0.0)),
                "travel_km_away_prev_to_today": float(hit.get("travel_km_away_prev_to_today", 0.0)),
            }
        except Exception:
            pass

    # Resolve today's venue via schedule→gamePk→lat/lon
    pk_today = _today_gamepk_for(home_team, away_team, game_date_iso)
    lat_today = lon_today = None
    if pk_today:
        lat_today, lon_today = cc_venue_latlon_from_gamepk(int(pk_today))

    def _dist_for(team: str) -> float:
        if lat_today is None or lon_today is None:
            return 0.0
        prev_pk = _prev_gamepk_for_team(team, game_date_iso, lookback_days=10)
        if not prev_pk:
            return 0.0
        lat_prev, lon_prev = cc_venue_latlon_from_gamepk(int(prev_pk))
        if lat_prev is None or lon_prev is None:
            return 0.0
        return _haversine_km((lat_prev, lon_prev), (lat_today, lon_today))

    out = {
        "travel_km_home_prev_to_today": float(_dist_for(home_team)),
        "travel_km_away_prev_to_today": float(_dist_for(away_team)),
    }
    cache.save_json("travel", out, ck)
    return out
