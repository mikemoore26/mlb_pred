# utils/data_fetchers/rest_travel.py
from __future__ import annotations
import datetime as _dt
from typing import Optional, Tuple
import math

import utils.cache as cache
from utils.safe_get_json import _safe_get_json
from .park_venue import norm_team_name, cc_venue_latlon_from_gamepk

def cc_days_rest(team: str, game_date: str) -> int:
    """
    Days since the team's most recent previous game strictly before game_date.
    Cached 7 days.
    """
    k = cache._make_key("days_rest", team, game_date)
    hit = cache.load_json("days_rest", k, max_age_days=7)
    if hit is not None:
        try:
            return int(hit)
        except Exception:
            return 0

    try:
        gd = _dt.date.fromisoformat(game_date)
    except Exception:
        cache.save_json("days_rest", 0, k)
        return 0

    start = (gd - _dt.timedelta(days=30)).isoformat()
    url = f"https://statsapi.mlb.com/api/v1/schedule?startDate={start}&endDate={game_date}&sportId=1"
    data = _safe_get_json(url) or {}

    dates = []
    team_n = norm_team_name(team)
    for drec in data.get("dates", []):
        for g in drec.get("games", []):
            ht = g["teams"]["home"]["team"]["name"]
            at = g["teams"]["away"]["team"]["name"]
            if team_n in (ht, at):
                try:
                    d = _dt.date.fromisoformat(drec["date"])
                    if d < gd:
                        dates.append(d)
                except Exception:
                    pass

    dates = sorted(dates)
    result = (gd - dates[-1]).days if dates else 0
    cache.save_json("days_rest", result, k)
    return result

# ---------- (optional) travel distance helpers ----------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    to_rad = math.pi / 180.0
    dlat = (lat2 - lat1) * to_rad
    dlon = (lon2 - lon1) * to_rad
    a = (
        math.sin(dlat/2)**2
        + math.cos(lat1*to_rad) * math.cos(lat2*to_rad) * math.sin(dlon/2)**2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def _prev_final_gamepk_for_team(team_name: str, before_iso_date: str) -> Optional[int]:
    """
    Most recent FINAL gamePk for team_name strictly before 'before_iso_date'. Cached 5 days.
    """
    norm = norm_team_name(team_name)
    ck = cache._make_key("prev_final", norm, before_iso_date)
    hit = cache.load_json("prev_final", ck, max_age_days=5)
    if hit is not None:
        return hit or None

    try:
        before_d = _dt.date.fromisoformat(before_iso_date)
    except Exception:
        before_d = _dt.date.today()

    start_d = before_d - _dt.timedelta(days=45)
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?startDate={start_d.isoformat()}&endDate={before_d.isoformat()}&sportId=1")
    sched = _safe_get_json(url) or {}

    best_dt = None
    best_pk = None
    for drec in sched.get("dates", []):
        for g in drec.get("games", []):
            status = (g.get("status", {}).get("detailedState") or "").lower()
            if status != "final":
                continue
            home = g["teams"]["home"]["team"]["name"]
            away = g["teams"]["away"]["team"]["name"]
            if norm not in (home, away):
                continue
            iso = g.get("gameDate")
            try:
                dt = _dt.datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
            except Exception:
                dt = None
            if not dt or dt.date() >= before_d:
                continue
            if best_dt is None or dt > best_dt:
                best_dt = dt
                best_pk = g.get("gamePk")

    cache.save_json("prev_final", best_pk, ck)
    return best_pk

def cc_travel(home_team: str, away_team: str, game_date: str) -> Tuple[float, float]:
    """
    Very best-effort travel distance approximation (km) since each team's last game → current venue.
    Requires that the caller knows the current game's (home) venue; without a current pk we estimate
    distances as 0.0 (home) and 500.0 (away) as safe placeholders.

    If you wire this with a current gamePk, prefer calling a helper that uses it directly.
    """
    # Placeholder behavior to match previous code path (100, 500):
    # Keeping the function so your feature layer stays stable.
    return (100.0, 500.0)
