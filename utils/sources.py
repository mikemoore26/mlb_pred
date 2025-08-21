# utils/sources.py
import os
import json
import math
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

# ---------------------------
# Shared HTTP session + caches
# ---------------------------
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "mlb-model/1.0"})

_DAY_SCHEDULE_CACHE: Dict[str, List[dict]] = {}
_GAME_FEED_CACHE: Dict[int, dict] = {}
_STANDINGS_CACHE: Dict[str, Dict[str, float]] = {}
_LAST_GAME_DATE_CACHE: Dict[str, date] = {}
_PARK_DATA: Optional[dict] = None
_BALLPARK_COORDS: Optional[dict] = None


def _safe_get_json(url: str, timeout: int = 12):
    try:
        r = _SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ---------------------------
# Schedule utilities
# ---------------------------
def get_schedule_range(start_date: str, end_date: str) -> List[dict]:
    """Return raw StatsAPI schedule game dicts for [start_date, end_date] (inclusive)."""
    out: List[dict] = []
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    d = start
    while d <= end:
        d_iso = d.isoformat()
        if d_iso in _DAY_SCHEDULE_CACHE:
            out.extend(_DAY_SCHEDULE_CACHE[d_iso])
        else:
            url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={d_iso}"
            data = _safe_get_json(url) or {}
            games = []
            for dayblk in (data.get("dates") or []):
                games.extend(dayblk.get("games") or [])
            _DAY_SCHEDULE_CACHE[d_iso] = games
            out.extend(games)
        d += timedelta(days=1)
    return out


def _get_feed(game_pk: int) -> dict:
    if game_pk in _GAME_FEED_CACHE:
        return _GAME_FEED_CACHE[game_pk]
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    data = _safe_get_json(url) or {}
    _GAME_FEED_CACHE[game_pk] = data
    return data


# ---------------------------
# Team strength: season & last30
# ---------------------------
def get_team_wpct_season(season: Optional[int] = None) -> Dict[str, float]:
    """Return current season win% per team name."""
    season = season or date.today().year
    cache_key = f"{season}|season"
    if cache_key in _STANDINGS_CACHE:
        return _STANDINGS_CACHE[cache_key]

    url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={season}&standingsTypes=regularSeason"
    data = _safe_get_json(url) or {}
    wpct: Dict[str, float] = {}
    for rec_set in (data.get("records") or []):
        for tr in (rec_set.get("teamRecords") or []):
            tname = tr["team"]["name"]
            w = tr.get("wins", 0) or 0
            l = tr.get("losses", 0) or 0
            wpct[tname] = (w/(w+l)) if (w+l) > 0 else 0.5
    _STANDINGS_CACHE[cache_key] = wpct
    return wpct


def get_team_wpct_last30() -> Dict[str, float]:
    """Approximate last-30d win% per team by scanning schedule results."""
    end = date.today()
    start = end - timedelta(days=30)
    games = get_schedule_range(start.isoformat(), end.isoformat())
    wins, total = {}, {}
    for g in games:
        status = ((g.get("status") or {}).get("detailedState") or "").lower()
        if status != "final":
            continue
        home = g["teams"]["home"]["team"]["name"]
        away = g["teams"]["away"]["team"]["name"]
        hs = g["teams"]["home"].get("score") or 0
        as_ = g["teams"]["away"].get("score") or 0
        for t in (home, away):
            total[t] = total.get(t, 0) + 1
        if hs > as_:
            wins[home] = wins.get(home, 0) + 1
        else:
            wins[away] = wins.get(away, 0) + 1
    return {t: (wins.get(t, 0) / n if n > 0 else 0.5) for t, n in total.items()}


# ---------------------------
# Probable pitchers + pitcher stats
# ---------------------------
def get_probable_pitchers(game_pks: List[int]) -> Dict[int, dict]:
    """
    Return {gamePk: {"home_id": id|None, "away_id": id|None}}
    Uses live feed (more reliable for probables).
    """
    out: Dict[int, dict] = {}
    for pk in game_pks:
        if not pk:
            continue
        feed = _get_feed(pk)
        pp = ((feed.get("gameData") or {}).get("probablePitchers") or {})
        out[pk] = {"home_id": (pp.get("home") or {}).get("id"),
                   "away_id": (pp.get("away") or {}).get("id")}
    return out


def _ip_to_float(ip_str):
    # "123.1" = 123 + 1/3, "123.2" = 123 + 2/3
    try:
        s = str(ip_str)
        if "." in s:
            whole, frac = s.split(".", 1)
            thirds = {"0": 0.0, "1": 1/3, "2": 2/3}.get(frac, 0.0)
            return int(whole) + thirds
        return float(s)
    except Exception:
        return 0.0


def get_pitcher_season_stats(player_id: Optional[int], season: Optional[int] = None) -> dict:
    """Returns {era, k9, bb9} with proper IP guards."""
    if not player_id:
        return {"era": None, "k9": None, "bb9": None}
    season = season or date.today().year
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&group=pitching&season={season}"
    d = _safe_get_json(url) or {}
    era = k9 = bb9 = None
    for blk in (d.get("stats") or []):
        for sp in (blk.get("splits") or []):
            st = sp.get("stat") or {}
            try:
                e = float(st["era"]) if st.get("era") not in (None, "") else None
            except Exception:
                e = None
            so = st.get("strikeOuts"); bb = st.get("baseOnBalls")
            ip = _ip_to_float(st.get("inningsPitched"))
            k9v = (float(so)/ip*9.0) if (so is not None and ip > 0) else None
            bb9v = (float(bb)/ip*9.0) if (bb is not None and ip > 0) else None
            if era is None and e is not None: era = e
            if k9  is None and k9v is not None: k9 = k9v
            if bb9 is None and bb9v is not None: bb9 = bb9v
    if era is not None or k9 is not None or bb9 is not None:
        return {"era": era, "k9": k9, "bb9": bb9}

    # fallback hydrate
    url2 = f"https://statsapi.mlb.com/api/v1/people/{player_id}?hydrate=stats(group=pitching,stats=season)"
    d2 = _safe_get_json(url2) or {}
    for blk in (d2.get("people", [{}])[0].get("stats") or []):
        for sp in (blk.get("splits") or []):
            st = sp.get("stat") or {}
            try:
                e = float(st["era"]) if st.get("era") not in (None, "") else None
            except Exception:
                e = None
            so = st.get("strikeOuts"); bb = st.get("baseOnBalls")
            ip = _ip_to_float(st.get("inningsPitched"))
            k9v = (float(so)/ip*9.0) if (so is not None and ip > 0) else None
            bb9v = (float(bb)/ip*9.0) if (bb is not None and ip > 0) else None
            if era is None and e is not None: era = e
            if k9  is None and k9v is not None: k9 = k9v
            if bb9 is None and bb9v is not None: bb9 = bb9v
    return {"era": era, "k9": k9, "bb9": bb9}


def get_pitcher_last3_starts(player_id: Optional[int]) -> dict:
    """
    Return last-3-starts metrics for a pitcher:
      {'era3': float|None, 'kbb3': float|None}
    Uses gameLog; filters gamesStarted == 1; takes latest 3.
    """
    if not player_id:
        return {"era3": None, "kbb3": None}
    season = date.today().year
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&group=pitching&season={season}"
    d = _safe_get_json(url) or {}
    games: List[dict] = []
    for blk in (d.get("stats") or []):
        for sp in (blk.get("splits") or []):
            st = sp.get("stat") or {}
            if (st.get("gamesStarted") or 0) >= 1:
                games.append(st)

    def _when(g):
        dt = g.get("gameDate") or g.get("date")
        try:
            return datetime.fromisoformat(str(dt).replace("Z","+00:00"))
        except Exception:
            return datetime.min

    games = sorted(games, key=_when)[-3:]
    if not games:
        return {"era3": None, "kbb3": None}

    er = 0.0; outs = 0; k = 0.0; bb = 0.0
    for st in games:
        try: er += float(st.get("earnedRuns") or 0)
        except Exception: pass
        outs += int(st.get("outsRecorded") or 0)
        k += float(st.get("strikeOuts") or 0)
        bb += float(st.get("baseOnBalls") or 0)

    ip = outs/3.0
    era3 = (er*9.0/ip) if ip > 0 else None
    kbb3 = (k/bb) if (bb and bb > 0) else None
    return {"era3": era3, "kbb3": kbb3}


# ---------------------------
# Bullpen usage/ERA helpers
# ---------------------------
def _bullpen_ip_in_game(feed: dict, team_side: str) -> float:
    box = ((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
    team_blob = box.get(team_side) or {}
    players = team_blob.get("players") or {}
    max_outs = 0; total_outs = 0
    for pdata in players.values():
        stat = ((pdata.get("stats") or {}).get("pitching") or {})
        outs = int(stat.get("outsRecorded") or 0)
        total_outs += outs
        if outs > max_outs:
            max_outs = outs
    bp_outs = max(total_outs - max_outs, 0)
    return bp_outs / 3.0

def get_bullpen_ip_last3_days(team_name: str, on_date: date) -> float:
    start = on_date - timedelta(days=3)
    end = on_date - timedelta(days=1)
    games = get_schedule_range(start.isoformat(), end.isoformat())
    total_ip = 0.0
    for g in games:
        status = ((g.get("status") or {}).get("detailedState") or "").lower()
        if status != "final":
            continue
        home = g["teams"]["home"]["team"]["name"]; away = g["teams"]["away"]["team"]["name"]
        if team_name not in (home, away):
            continue
        pk = g.get("gamePk")
        if not pk:
            continue
        feed = _get_feed(pk)
        side = "home" if team_name == home else "away"
        total_ip += _bullpen_ip_in_game(feed, side)
    return total_ip

def get_bullpen_era_last14() -> Dict[str, float]:
    end = date.today()
    start = end - timedelta(days=14)
    games = get_schedule_range(start.isoformat(), end.isoformat())
    outs_by_team: Dict[str, int] = {}
    er_by_team: Dict[str, float] = {}

    for g in games:
        status = ((g.get("status") or {}).get("detailedState") or "").lower()
        if status != "final":
            continue
        home = g["teams"]["home"]["team"]["name"]
        away = g["teams"]["away"]["team"]["name"]
        pk = g.get("gamePk")
        if not pk:
            continue
        feed = _get_feed(pk)
        box = ((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
        for side, team_name in (("home", home), ("away", away)):
            players = (box.get(side) or {}).get("players") or {}
            er = 0.0; outs = 0
            for pdata in players.values():
                stat = ((pdata.get("stats") or {}).get("pitching") or {})
                try:
                    er += float(stat.get("earnedRuns") or 0)
                except Exception:
                    pass
                o = stat.get("outsRecorded")
                if o is not None:
                    try: outs += int(o)
                    except Exception: pass
            er_by_team[team_name] = er_by_team.get(team_name, 0.0) + er
            outs_by_team[team_name] = outs_by_team.get(team_name, 0) + outs

    bp14 = {}
    for t, outs in outs_by_team.items():
        ip = outs/3.0
        bp14[t] = (er_by_team[t]*9.0/ip) if ip > 0 else None
    return bp14


# ---------------------------
# Park factor
# ---------------------------
def _load_park_fallback() -> dict:
    global _PARK_DATA
    if _PARK_DATA is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(here, "..", "data", "park_factors_fallback.json"))
        with open(path, "r", encoding="utf-8") as f:
            _PARK_DATA = json.load(f)
    return _PARK_DATA

def get_park_factor(home_team: str) -> float:
    data = _load_park_fallback()
    aliases = data.get("aliases", {})
    canon = aliases.get(home_team, home_team)
    try:
        return float(data["park_factor"].get(canon, 0.0))
    except Exception:
        return 0.0


# ---------------------------
# Rest days
# ---------------------------
def _last_game_date(team_name: str, on_or_before: date) -> Optional[date]:
    key = f"{team_name}|{on_or_before.isoformat()}"
    if key in _LAST_GAME_DATE_CACHE:
        return _LAST_GAME_DATE_CACHE[key]
    start = on_or_before - timedelta(days=15)
    games = get_schedule_range(start.isoformat(), on_or_before.isoformat())
    last_played: Optional[date] = None
    for g in games:
        status = ((g.get("status") or {}).get("detailedState") or "").lower()
        if status not in ("final", "completed early", "game over"):
            continue
        home = g["teams"]["home"]["team"]["name"]
        away = g["teams"]["away"]["team"]["name"]
        if team_name not in (home, away):
            continue
        game_dt = datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00")).date()
        if last_played is None or game_dt > last_played:
            last_played = game_dt
    _LAST_GAME_DATE_CACHE[key] = last_played
    return last_played

def get_days_rest(team_name: str, game_day: date) -> int:
    lgd = _last_game_date(team_name, game_day - timedelta(days=1))
    if lgd is None:
        return 3
    return max(0, (game_day - lgd).days)


# ---------------------------
# Travel distance to today's game
# ---------------------------
def _load_ballparks() -> dict:
    global _BALLPARK_COORDS
    if _BALLPARK_COORDS is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(here, "..", "data", "ballpark_coords.json"))
        with open(path, "r", encoding="utf-8") as f:
            _BALLPARK_COORDS = json.load(f)
    return _BALLPARK_COORDS

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    try:
        R = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dlat = p2 - p1
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
        return 2*R*math.asin(math.sqrt(a))
    except Exception:
        return 0.0

def _team_prev_game_city_coords(team_name: str, on_date: date) -> Tuple[Optional[float], Optional[float]]:
    games = get_schedule_range((on_date - timedelta(days=7)).isoformat(),
                               (on_date - timedelta(days=1)).isoformat())
    last_home = None
    last_date = None
    for g in games:
        status = ((g.get("status") or {}).get("detailedState") or "").lower()
        if status != "final":
            continue
        home = g["teams"]["home"]["team"]["name"]
        away = g["teams"]["away"]["team"]["name"]
        if team_name not in (home, away):
            continue
        gdt = datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00")).date()
        if last_date is None or gdt > last_date:
            last_date = gdt
            last_home = home
    if not last_home:
        return (None, None)
    parks = _load_ballparks()
    latlon = parks.get(last_home)
    return (latlon[0], latlon[1]) if latlon else (None, None)

def get_travel_km_to_today(home_team: str, away_team: str, game_date: date) -> Tuple[float, float]:
    parks = _load_ballparks()
    latlon_home_today = parks.get(home_team)
    if not latlon_home_today:
        return (0.0, 0.0)
    th_lat, th_lon = latlon_home_today
    h_lat, h_lon = _team_prev_game_city_coords(home_team, game_date)
    a_lat, a_lon = _team_prev_game_city_coords(away_team, game_date)
    home_km = _haversine_km(h_lat, h_lon, th_lat, th_lon) if (h_lat and h_lon) else 0.0
    away_km = _haversine_km(a_lat, a_lon, th_lat, th_lon) if (a_lat and a_lon) else 0.0
    return (home_km, away_km)


# ---------------------------
# Weather (return None if unknown/closed)
# ---------------------------
def get_weather_for_game(game_dict: dict) -> Optional[dict]:
    """
    Minimal placeholder: returns None if roof likely closed; else a light neutral default.
    Replace with a real weather API if you want true signal.
    """
    venue = ((game_dict.get("venue") or {}).get("name")) or ""
    closed_likely = any(k in venue.lower() for k in (
        "tropicana", "minute maid", "rogers centre", "american family field"
    ))
    if closed_likely:
        return None
    # lightweight neutral defaults — your fetcher can use these or override
    return {"temp": 72.0, "wind_speed": 5.0, "wind_out_to_cf": False}


# ---------------------------
# Odds / Elo (left unimplemented by default)
# ---------------------------
def get_closing_odds_implied_home(game_pk: int) -> Optional[float]:
    return None

def get_elo_diff(home_team: str, away_team: str, on_date) -> Optional[float]:
    return None
