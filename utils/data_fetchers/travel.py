# utils/data_fetchers/travel.py
from __future__ import annotations
from typing import Optional, Tuple, Dict
import math
import datetime as _dt

import utils.cache as cache
from utils.safe_get_json import _safe_get_json
from .park_venue import cc_venue_latlon_from_gamepk, norm_team_name
from helpers.travel_helper import _resolve_today_venue_latlon

DEBUG_TRAVEL = True
BACKFILL_DAYS = 60  # how far back we search for the previous final game

# ---------- math ----------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    try:
        R = 6371.0
        phi1 = math.radians(lat1); phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return float(R * c)
    except Exception:
        return 0.0

def _canon(s: str) -> str:
    return ''.join(ch for ch in (s or '').lower() if ch.isalnum())

# ---------- team id index ----------

def _team_index() -> Dict[str, int]:
    ck = cache._make_key("mlb_team_index")
    hit = cache.load_json("mlb_misc", ck, max_age_days=30)
    if isinstance(hit, dict) and hit:
        return {str(k): int(v) for k, v in hit.items() if v is not None}

    url = "https://statsapi.mlb.com/api/v1/teams?sportId=1&activeStatus=Yes"
    data = _safe_get_json(url) or {}
    out: Dict[str, int] = {}
    for t in data.get("teams", []):
        name = t.get("name") or ""
        tid = t.get("id")
        if name and tid is not None:
            out[_canon(norm_team_name(name))] = int(tid)
    cache.save_json("mlb_misc", out, ck)
    return out

# ---------- schedule helpers ----------

def _find_game_info_for_matchup_on_date(home: str, away: str, iso_date: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Return (gamePk, venue_id) for a specific matchup on a date, or (None, None).
    Uses team IDs for robust matching; tries swapped roles too.
    """
    try:
        url = f"https://statsapi.mlb.com/api/v1/schedule?startDate={iso_date}&endDate={iso_date}&sportId=1"
        data = _safe_get_json(url) or {}
        teams_map = _team_index()

        want_home = teams_map.get(_canon(norm_team_name(home)))
        want_away = teams_map.get(_canon(norm_team_name(away)))

        if DEBUG_TRAVEL:
            print(f"[travel] schedule {iso_date} games={data.get('totalGames')}")
            if want_home and want_away:
                print(f"[travel] team ids: {home}→{want_home}, {away}→{want_away}")

        # direct match
        for drec in data.get("dates", []):
            for g in drec.get("games", []):
                home_id = (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("id")
                away_id = (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("id")
                if home_id == want_home and away_id == want_away:
                    pk = int(g.get("gamePk") or 0) or None
                    vid = (g.get("venue") or {}).get("id")
                    if DEBUG_TRAVEL:
                        print(f"[travel] game_pk={pk} venue_id={vid} for {away} @ {home} on {iso_date}")
                    return pk, vid

        # swapped roles
        for drec in data.get("dates", []):
            for g in drec.get("games", []):
                home_id = (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("id")
                away_id = (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("id")
                if home_id == want_away and away_id == want_home:
                    pk = int(g.get("gamePk") or 0) or None
                    vid = (g.get("venue") or {}).get("id")
                    if DEBUG_TRAVEL:
                        print(f"[travel] matched with swapped roles → game_pk={pk} venue_id={vid}")
                    return pk, vid

    except Exception as e:
        if DEBUG_TRAVEL:
            print("[travel] _find_game_info_for_matchup_on_date error:", e)
    return None, None

def _prev_final_gamepk_for_team(team_name: str, before_iso_date: str) -> Optional[int]:
    """
    Most recent FINAL gamePk for team strictly before 'before_iso_date'.
    """
    norm = _canon(norm_team_name(team_name))
    ck = cache._make_key("prev_final", norm, before_iso_date)
    hit = cache.load_json("prev_final", ck, max_age_days=5)
    if hit is not None:
        return hit or None

    try:
        before_d = _dt.date.fromisoformat(before_iso_date)
    except Exception:
        before_d = _dt.date.today()

    start_d = before_d - _dt.timedelta(days=BACKFILL_DAYS)
    url = f"https://statsapi.mlb.com/api/v1/schedule?startDate={start_d.isoformat()}&endDate={before_d.isoformat()}&sportId=1"
    sched = _safe_get_json(url) or {}

    best_dt: Optional[_dt.datetime] = None
    best_pk: Optional[int] = None

    for drec in sched.get("dates", []):
        for g in drec.get("games", []):
            state = (g.get("status", {}) or {}).get("detailedState", "")
            if str(state).lower() != "final":
                continue

            home_raw = (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name", "")
            away_raw = (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name", "")
            if _canon(norm_team_name(home_raw)) != norm and _canon(norm_team_name(away_raw)) != norm:
                continue

            try:
                when = g.get("gameDate")
                dt = _dt.datetime.fromisoformat(str(when).replace("Z", "+00:00"))
            except Exception:
                dt = None
            if not dt or dt.date() >= before_d:
                continue

            if (best_dt is None) or (dt > best_dt):
                best_dt = dt
                best_pk = int(g.get("gamePk") or 0) or None

    cache.save_json("prev_final", best_pk, ck)
    if DEBUG_TRAVEL:
        print(f"[travel] prev_final {team_name} before {before_iso_date}: {best_pk}")
    return best_pk

# ---------- public API ----------

def cc_travel(home_team: str, away_team: str, game_iso_date: str, game_pk: Optional[int] = None) -> Dict[str, float]:
    """
    Distance (km) from each team's previous game's venue to today's venue.
    Returns:
      {
        "travel_km_home_prev_to_today": float,
        "travel_km_away_prev_to_today": float
      }
    """
    ck = cache._make_key("travel", home_team, away_team, game_iso_date, game_pk or "")
    hit = cache.load_json("travel", ck, max_age_days=1)
    if isinstance(hit, dict):
        try:
            return {
                "travel_km_home_prev_to_today": float(hit.get("travel_km_home_prev_to_today", 0.0) or 0.0),
                "travel_km_away_prev_to_today": float(hit.get("travel_km_away_prev_to_today", 0.0) or 0.0),
            }
        except Exception:
            pass

    # 1) Find gamePk & venue_id (if not provided)
    venue_id: Optional[int] = None
    if not game_pk:
        game_pk, venue_id = _find_game_info_for_matchup_on_date(home_team, away_team, game_iso_date)

    if DEBUG_TRAVEL:
        print(f"[travel] game_pk={game_pk} venue_id={venue_id} for {away_team} @ {home_team} on {game_iso_date}")

    if not game_pk:
        out = {"travel_km_home_prev_to_today": 0.0, "travel_km_away_prev_to_today": 0.0}
        cache.save_json("travel", out, ck)
        return out

    # 2) Today's venue lat/lon (resolver hits cache, feed, venues, team, overrides)
    today_lat, today_lon = _resolve_today_venue_latlon(game_pk, venue_id)
    if today_lat is None or today_lon is None:
        if DEBUG_TRAVEL:
            print("[travel] missing today venue lat/lon")
        out = {"travel_km_home_prev_to_today": 0.0, "travel_km_away_prev_to_today": 0.0}
        cache.save_json("travel", out, ck)
        return out

    # 3) Get previous final game for each team
    home_prev_pk = _prev_final_gamepk_for_team(home_team, game_iso_date)
    away_prev_pk = _prev_final_gamepk_for_team(away_team, game_iso_date)

    def _dist_from_prev(prev_pk: Optional[int]) -> float:
        if not prev_pk:
            return 0.0
        plat, plon = cc_venue_latlon_from_gamepk(prev_pk)
        if plat is None or plon is None:
            # fallback via live feed → venue id → resolver chain
            try:
                feed = _safe_get_json(f"https://statsapi.mlb.com/api/v1.1/game/{int(prev_pk)}/feed/live") or {}
                vid = ((feed.get("gameData") or {}).get("venue") or {}).get("id")
                if vid:
                    vlat, vlon = _resolve_today_venue_latlon(prev_pk, int(vid))
                    plat = vlat if vlat is not None else plat
                    plon = vlon if vlon is not None else plon
            except Exception:
                pass
        if plat is None or plon is None:
            return 0.0
        try:
            return _haversine_km(float(plat), float(plon), float(today_lat), float(today_lon))
        except Exception:
            return 0.0

    d_home = _dist_from_prev(home_prev_pk)
    d_away = _dist_from_prev(away_prev_pk)

    out = {
        "travel_km_home_prev_to_today": float(d_home),
        "travel_km_away_prev_to_today": float(d_away),
    }
    cache.save_json("travel", out, ck)
    if DEBUG_TRAVEL:
        print(f"[travel] distances km → home:{d_home:.1f} away:{d_away:.1f}")
    return out

# ---------- smoke test ----------

if __name__ == "__main__":
    today = _dt.date.today().isoformat()
    example_home = "Los Angeles Angels"
    example_away = "Chicago Cubs"
    print("[travel] Example:", cc_travel(example_home, example_away, today))
