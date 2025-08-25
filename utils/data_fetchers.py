# utils/data_fetchers.py
from __future__ import annotations
import os
import json
import math
import datetime as _dt
import requests
from typing import Optional, List, Dict, Tuple, Any

from . import cache
from .sources import get_bullpen_ip_last3, get_weather_for_game
from .safe_get_json import _safe_get_json

# Persistent HTTP session
_SESSION = requests.Session()

# --------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088  # km
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def _parse_iso_dt(s) -> Optional[_dt.datetime]:
    try:
        return _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        try:
            return _dt.datetime.fromisoformat(str(s))
        except Exception:
            return None

def _parse_ip_to_decimal(ip_val) -> float:
    """Convert innings pitched '5.1'→5.333..., '5.2'→5.666..."""
    try:
        if ip_val and isinstance(ip_val, str):
            parts = ip_val.split(".")
            base = float(parts[0])
            frac = 0.0 if len(parts) == 1 else (float(parts[1]) / 3.0)
            return base + frac
        return float(ip_val or 0.0)
    except Exception:
        return 0.0

def _call(method: str, *args, default: Any = None) -> Any:
    """Mockable external calls (Elo, odds)."""
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

        else:
            return default
    except Exception:
        return default


# --------------------------------------------------------------------
# Static data (park factors + ballpark coords) – loaded once, no disk cache
# --------------------------------------------------------------------
_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))

_PARK_FACTORS_JSON = os.path.join(_PROJ_ROOT, "data", "park_factors_fallback.json")
_BALLPARK_COORDS_JSON = os.path.join(_PROJ_ROOT, "data", "ballpark_coords.json")

_PARK_FACTORS = None
_BALLPARK_COORDS = None

def _load_park_factors() -> dict:
    global _PARK_FACTORS
    if _PARK_FACTORS is not None:
        return _PARK_FACTORS
    try:
        with open(_PARK_FACTORS_JSON, "r", encoding="utf-8") as f:
            _PARK_FACTORS = json.load(f)
    except Exception:
        _PARK_FACTORS = {"aliases": {}, "park_factor": {}}
    return _PARK_FACTORS

def _load_ballpark_coords() -> dict:
    global _BALLPARK_COORDS
    if _BALLPARK_COORDS is not None:
        return _BALLPARK_COORDS
    try:
        with open(_BALLPARK_COORDS_JSON, "r", encoding="utf-8") as f:
            _BALLPARK_COORDS = json.load(f)
    except Exception:
        _BALLPARK_COORDS = {}
    return _BALLPARK_COORDS

# --- Team alias helpers (normalize to MLB full names) ---
# Try to leverage park_factors aliases; extend with a few common nicknames.
def _team_alias_map() -> Dict[str, str]:
    pf = _load_park_factors()
    aliases = pf.get("aliases", {})  # often maps Team->Stadium; we want team-to-team too
    # We’ll keep a manual, minimal team alias map for common nicknames:
    manual = {
        "yankees": "New York Yankees",
        "red sox": "Boston Red Sox",
        "bos": "Boston Red Sox",
        "nyy": "New York Yankees",
        "mets": "New York Mets",
        "dodgers": "Los Angeles Dodgers",
        "giants": "San Francisco Giants",
        "cards": "St. Louis Cardinals",
        "dbacks": "Arizona Diamondbacks",
        "d-backs": "Arizona Diamondbacks",
        "chi cubs": "Chicago Cubs",
        "cubs": "Chicago Cubs",
        "white sox": "Chicago White Sox",
        "chi white sox": "Chicago White Sox",
        # add more as needed
    }
    # Normalize keys to lower once
    final_map = {k.lower(): v for k, v in manual.items()}
    # If your park_factors has team aliases, fold them in (best-effort)
    # (Some users store team->stadium here; if so, we can't use it for team normalization.)
    # We'll only include entries that look like Team->Team (i.e., value has a space and no "Park"/"Field")
    for k, v in aliases.items():
        if isinstance(k, str) and isinstance(v, str):
            if "park" not in v.lower() and "field" not in v.lower() and "stadium" not in v.lower():
                final_map[k.lower()] = v
    return final_map

def _norm_team_name(name: str) -> str:
    if not name:
        return name
    m = _team_alias_map()
    return m.get(name.lower().strip(), name.strip())


def _resolve_home_destination_coords(
    game_date: str,
    home_team: str,
    away_team: Optional[str] = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    Try hard to resolve the home park (lat,lon) for (home_team vs away_team) on date.
    Fallbacks:
      - If exact matchup not found, take ANY game that day with the same home_team.
      - If no schedule match, try team->stadium alias -> ballpark_coords.json directly.
    """
    # 1) try schedule exact match
    sched = cc_schedule_range(game_date, game_date)
    home_n = _norm_team_name(home_team)
    away_n = _norm_team_name(away_team) if away_team else None

    game_pk_today = None
    chosen_venue_name = None

    for drec in sched.get("dates", []):
        for g in drec.get("games", []):
            hname = ((g.get("teams", {}) or {}).get("home", {}) or {}).get("team", {}) or {}
            aname = ((g.get("teams", {}) or {}).get("away", {}) or {}).get("team", {}) or {}
            home_api = hname.get("name", "")
            away_api = aname.get("name", "")
            if home_api == home_n and (away_n is None or away_api == away_n):
                game_pk_today = g.get("gamePk")
                chosen_venue_name = (g.get("venue") or {}).get("name")
                break
        if game_pk_today:
            break

    # 2) if no exact match, try any game with that home team that day
    if not game_pk_today:
        for drec in sched.get("dates", []):
            for g in drec.get("games", []):
                home_api = ((g.get("teams", {}) or {}).get("home", {}) or {}).get("team", {}) or {}
                if home_api.get("name") == home_n:
                    game_pk_today = g.get("gamePk")
                    chosen_venue_name = (g.get("venue") or {}).get("name")
                    break
            if game_pk_today:
                break

    # 3) live-feed coords or fallback to venue_name->coords
    if game_pk_today:
        lat, lon = _cc_venue_latlon_from_gamepk(game_pk_today)
        if (lat is None or lon is None) and chosen_venue_name:
            lat, lon = _coords_from_venue_name(chosen_venue_name)
        if lat is not None and lon is not None:
            return lat, lon

    # 4) last resort: team -> stadium name (from park_factors aliases) -> coords
    pf = _load_park_factors()
    pf_aliases = pf.get("aliases", {})  # often team->stadium
    stadium_guess = pf_aliases.get(home_n, home_n)
    lat, lon = _coords_from_venue_name(stadium_guess)
    return lat, lon


def _coords_from_venue_name(venue_name: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Fallback coords from data/ballpark_coords.json (value = [lat, lon, orientation]).
    """
    if not venue_name:
        return (None, None)
    row = _load_ballpark_coords().get(venue_name)
    if not row or len(row) < 2:
        return (None, None)
    try:
        return float(row[0]), float(row[1])
    except Exception:
        return (None, None)


# --------------------------------------------------------------------
# Cached wrappers around data sources
# --------------------------------------------------------------------
def cc_schedule_range(start_date: str, end_date: str) -> dict:
    """
    Cached MLB schedule JSON for a date range (1-day TTL).
    """
    k = cache._make_key("schedule", f"{start_date}_{end_date}")
    hit = cache.load_json("schedule", k, max_age_days=1)
    if hit is not None:
        return hit
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?startDate={start_date}&endDate={end_date}&sportId=1&hydrate=probablePitchers"
    )
    data = _safe_get_json(url) or {}
    cache.save_json("schedule", data, k)
    return data

def cc_bullpen_ip_last3(team: str, game_date: str) -> float:
    k = cache._make_key("bullpen_ip_last3", team, game_date)
    cached = cache.load_json("bullpen_ip_last3", k, max_age_days=7)
    if cached is not None:
        try:
            return float(cached)
        except Exception:
            return 0.0
    val = get_bullpen_ip_last3(team, game_date)
    out = 0.0
    try:
        out = float(val) if val is not None else 0.0
    except Exception:
        out = 0.0
    cache.save_json("bullpen_ip_last3", out, k)
    return out

def cc_wpct_season() -> Dict[str, float]:
    k = cache._make_key("wpct_season", "cur")
    cached = cache.load_json("wpct_season", k, max_age_days=1)
    if cached is not None:
        return cached
    year = _dt.date.today().year
    url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={year}"
    data = _safe_get_json(url) or {}
    wpct: Dict[str, float] = {}
    for rec in data.get("records", []):
        for t in rec.get("teamRecords", []):
            wins = t.get("wins", 0)
            losses = t.get("losses", 0)
            name = t.get("team", {}).get("name")
            if not name:
                continue
            wpct[name] = (wins / (wins + losses)) if (wins + losses) > 0 else 0.5
    cache.save_json("wpct_season", wpct, k)
    return wpct

def cc_wpct_last30() -> Dict[str, float]:
    k = cache._make_key("wpct_last30", "global")
    cached = cache.load_json("wpct_last30", k, max_age_days=1)
    if cached is not None:
        return cached
    end_date = _dt.date.today()
    start_date = end_date - _dt.timedelta(days=30)
    url = f"https://statsapi.mlb.com/api/v1/schedule?startDate={start_date}&endDate={end_date}&sportId=1"
    data = _safe_get_json(url) or {}
    counts: Dict[str, Dict[str, int]] = {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if (g.get("status", {}).get("detailedState") or "").lower() != "final":
                continue
            ht = g["teams"]["home"]["team"]["name"]
            at = g["teams"]["away"]["team"]["name"]
            hs = g["teams"]["home"].get("score", 0)
            as_ = g["teams"]["away"].get("score", 0)
            winner = ht if hs > as_ else at
            for t in (ht, at):
                counts.setdefault(t, {"wins": 0, "games": 0})
                counts[t]["games"] += 1
                if t == winner:
                    counts[t]["wins"] += 1
    result = {t: (v["wins"] / v["games"]) if v["games"] else 0.5 for t, v in counts.items()}
    cache.save_json("wpct_last30", result, k)
    return result

def cc_bullpen_era14() -> Dict[str, float]:
    k = cache._make_key("bp14", "global")
    cached = cache.load_json("bp14", k, max_age_days=1)
    if cached is not None:
        return cached
    # Placeholder neutral values (replace with real bullpen calc when available)
    end_date = _dt.date.today()
    start_date = end_date - _dt.timedelta(days=14)
    url = f"https://statsapi.mlb.com/api/v1/schedule?startDate={start_date}&endDate={end_date}&sportId=1"
    data = _safe_get_json(url) or {}
    teams = set()
    for d in data.get("dates", []):
        for g in d.get("games", []):
            teams.add(g["teams"]["home"]["team"]["name"])
            teams.add(g["teams"]["away"]["team"]["name"])
    result = {t: 4.00 for t in teams}
    cache.save_json("bp14", result, k)
    return result

def cc_offense30() -> Dict[str, float]:
    k = cache._make_key("offense30", "global")
    cached = cache.load_json("offense30", k, max_age_days=1)
    if cached is not None:
        return cached
    end_date = _dt.date.today()
    start_date = end_date - _dt.timedelta(days=30)
    url = f"https://statsapi.mlb.com/api/v1/schedule?startDate={start_date}&endDate={end_date}&sportId=1"
    data = _safe_get_json(url) or {}
    raw: Dict[str, Dict[str, float]] = {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if (g.get("status", {}).get("detailedState") or "").lower() != "final":
                continue
            ht = g["teams"]["home"]["team"]["name"]
            at = g["teams"]["away"]["team"]["name"]
            hs = g["teams"]["home"].get("score", 0)
            as_ = g["teams"]["away"].get("score", 0)
            for t, r in ((ht, hs), (at, as_)):
                raw.setdefault(t, {"runs": 0.0, "games": 0})
                raw[t]["runs"] += float(r)
                raw[t]["games"] += 1
    result = {t: (v["runs"] / v["games"]) if v["games"] else 0.0 for t, v in raw.items()}
    cache.save_json("offense30", result, k)
    return result

def cc_probables(game_pks: List[int]) -> Dict[int, Dict[str, int]]:
    """
    Probables for provided game_pks. Uses per-PK cache, a batch snapshot, and a short lookahead schedule.
    Writes negative caches {} for missing Pks to avoid repeated fetches.
    """
    wanted = [int(pk) for pk in game_pks if pk]
    if not wanted:
        return {}
    result: Dict[int, Dict[str, int]] = {}
    missing: List[int] = []

    # per-PK cache
    for pk in wanted:
        per_key = cache._make_key("probables", pk)
        cached = cache.load_json("probables", per_key, max_age_days=1)
        if isinstance(cached, dict):
            result[pk] = cached
        else:
            missing.append(pk)
    if not missing:
        return result

    # batch cache
    batch_key = cache._make_key("probables_batch", *sorted(wanted))
    batch_cached = cache.load_json("probables", batch_key, max_age_days=1)
    if isinstance(batch_cached, dict):
        for pk, rec in batch_cached.items():
            per_key = cache._make_key("probables", pk)
            cache.save_json("probables", rec, per_key)
            result[pk] = rec
        missing = [pk for pk in wanted if pk not in result]

    # 3-day lookahead
    if missing:
        today = _dt.date.today()
        start_date = today.isoformat()
        end_date = (today + _dt.timedelta(days=3)).isoformat()
        url = (
            "https://statsapi.mlb.com/api/v1/schedule"
            f"?startDate={start_date}&endDate={end_date}&sportId=1&hydrate=probablePitchers"
        )
        data = _safe_get_json(url) or {}
        found: Dict[int, Dict[str, int]] = {}
        for d in data.get("dates", []):
            for g in d.get("games", []):
                pk = g.get("gamePk")
                if pk in missing:
                    home_id = (g.get("teams", {}).get("home", {}).get("probablePitcher", {}) or {}).get("id")
                    away_id = (g.get("teams", {}).get("away", {}).get("probablePitcher", {}) or {}).get("id")
                    if home_id or away_id:
                        found[pk] = {"home_id": home_id, "away_id": away_id}
        for pk, rec in found.items():
            per_key = cache._make_key("probables", pk)
            cache.save_json("probables", rec, per_key)
            result[pk] = rec
        still_missing = [pk for pk in missing if pk not in result]
        for pk in still_missing:
            per_key = cache._make_key("probables", pk)
            cache.save_json("probables", {}, per_key)
            result[pk] = {}
        batch_out = {pk: result.get(pk, {}) for pk in wanted}
        cache.save_json("probables", batch_out, batch_key)

    return result

def cc_pitcher_season(player_id: Optional[int]) -> Dict[str, Optional[float]]:
    if not player_id:
        return {"era": None, "k9": None, "bb9": None}
    ck = cache._make_key("p_season", player_id)
    hit = cache.load_json("p_season", ck, max_age_days=7)
    if hit is not None:
        return hit
    year = _dt.date.today().year
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&group=pitching&season={year}"
    data = _safe_get_json(url) or {}
    stat = (((data.get("stats") or [{}])[0].get("splits") or [{}])[0].get("stat") or {})
    def _fnum(v):
        try: return float(v)
        except Exception: return None
    out = {"era": _fnum(stat.get("era")),
           "k9": _fnum(stat.get("strikeoutsPer9Inn")),
           "bb9": _fnum(stat.get("walksPer9Inn"))}
    cache.save_json("p_season", out, ck)
    return out

def cc_pitcher_last30(player_id: Optional[int]) -> Dict[str, Optional[float]]:
    if not player_id:
        return {"era30": None}
    ck = cache._make_key("p_last30", player_id)
    hit = cache.load_json("p_last30", ck, max_age_days=2)
    if hit is not None:
        return hit
    end_d = _dt.date.today()
    start_d = end_d - _dt.timedelta(days=30)
    url = (f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
           f"?stats=gameLog&group=pitching&season={end_d.year}")
    data = _safe_get_json(url) or {}
    splits = (data.get("stats") or [{}])[0].get("splits") or []
    ER = 0.0; IP = 0.0
    for s in splits:
        dt = _parse_iso_dt(s.get("date"))
        if not dt or not (start_d <= dt.date() <= end_d):
            continue
        st = (s.get("stat") or {})
        IP += _parse_ip_to_decimal(st.get("inningsPitched"))
        ER += float(st.get("earnedRuns", 0.0) or 0.0)
    era30 = (ER * 9.0 / IP) if IP > 0 else None
    out = {"era30": era30}
    cache.save_json("p_last30", out, ck)
    return out

def cc_pitcher_last3(player_id: Optional[int]) -> Dict[str, Optional[float]]:
    if not player_id:
        return {"era3": None, "kbb3": None}
    ck = cache._make_key("p_last3", player_id)
    hit = cache.load_json("p_last3", ck, max_age_days=2)
    if hit is not None:
        return hit
    year = _dt.date.today().year
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=gameLog&group=pitching&season={year}"
    data = _safe_get_json(url) or {}
    splits = (data.get("stats") or [{}])[0].get("splits") or []
    starts = [s for s in splits if (s.get("stat") or {}).get("gamesStarted", 0) > 0]
    starts.sort(key=lambda s: s.get("date", ""), reverse=True)
    starts = starts[:3]
    if not starts:
        out = {"era3": None, "kbb3": None}
        cache.save_json("p_last3", out, ck)
        return out
    ER = 0.0; IP = 0.0; K = 0.0; BB = 0.0
    for s in starts:
        st = s.get("stat") or {}
        IP += _parse_ip_to_decimal(st.get("inningsPitched"))
        ER += float(st.get("earnedRuns", 0.0) or 0.0)
        K  += float(st.get("strikeOuts", st.get("strikeouts", 0.0)) or 0.0)
        BB += float(st.get("baseOnBalls", st.get("walks", 0.0)) or 0.0)
    era3 = (ER * 9.0 / IP) if IP > 0 else None
    kbb3 = (K / BB) if BB > 0 else (K if (K > 0 and BB == 0) else None)
    out = {"era3": era3, "kbb3": kbb3}
    cache.save_json("p_last3", out, ck)
    return out

# ---------------- Park factors ----------------
def cc_park_factor(team: str) -> float:
    """
    Resolve team -> stadium (via 'aliases') -> numeric park factor.
    Returns neutral 100.0 if unknown.
    """
    pf = _load_park_factors()
    aliases = pf.get("aliases", {})
    park_map = pf.get("park_factor", {})
    key = (team or "").strip()
    stadium = aliases.get(key, key)
    val = park_map.get(stadium)
    if val is None:
        # try case-insensitive alias match
        for k, v in aliases.items():
            if k.lower() == key.lower():
                val = park_map.get(v)
                break
    try:
        return float(val) if val is not None else 100.0
    except Exception:
        return 100.0

# ---------------- Days rest ----------------
def cc_days_rest(team: str, game_date: str) -> int:
    k = cache._make_key("days_rest", team, game_date)
    cached = cache.load_json("days_rest", k, max_age_days=7)
    if cached is not None:
        try:
            return int(cached)
        except Exception:
            return 0
    try:
        gd = _dt.date.fromisoformat(game_date)
        start = (gd - _dt.timedelta(days=30)).isoformat()
        url = f"https://statsapi.mlb.com/api/v1/schedule?startDate={start}&endDate={game_date}&sportId=1"
        data = _safe_get_json(url) or {}
        dates = []
        for drec in data.get("dates", []):
            for g in drec.get("games", []):
                ht = g["teams"]["home"]["team"]["name"]
                at = g["teams"]["away"]["team"]["name"]
                if ht == team or at == team:
                    try:
                        dates.append(_dt.date.fromisoformat(drec["date"]))
                    except Exception:
                        pass
        dates = sorted([d for d in dates if d < gd])
        result = (gd - dates[-1]).days if dates else 0
    except Exception:
        result = 0
    cache.save_json("days_rest", result, k)
    return result

# ---------------- Venue coords helpers (with fallbacks) ----------------
def _cc_venue_latlon_from_gamepk(game_pk: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Resolve venue coords via live feed; if absent, fallback to ballpark_coords by venue name.
    Cached for 60 days.
    """
    if not game_pk:
        return (None, None)
    vkey = cache._make_key("venue_coord", game_pk)
    hit = cache.load_json("venue_coord", vkey, max_age_days=60)
    if isinstance(hit, dict) and ("lat" in hit or "lon" in hit):
        try:
            lat = float(hit["lat"]) if hit["lat"] is not None else None
            lon = float(hit["lon"]) if hit["lon"] is not None else None
            return (lat, lon)
        except Exception:
            pass

    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/feed/live"
    data = _safe_get_json(url)
    if not data:
        cache.save_json("venue_coord", {"lat": None, "lon": None}, vkey)
        return (None, None)

    # Try live coordinates first
    loc = (((data.get("gameData") or {}).get("venue") or {}).get("location") or {})
    lat = loc.get("latitude")
    lon = loc.get("longitude")
    latf = None
    lonf = None
    try:
        latf = float(lat) if lat is not None else None
        lonf = float(lon) if lon is not None else None
    except Exception:
        latf = lonf = None

    # Fallback: venue name -> coords JSON
    if latf is None or lonf is None:
        venue_name = ((data.get("gameData") or {}).get("venue") or {}).get("name")
        latf, lonf = _coords_from_venue_name(venue_name)

    cache.save_json("venue_coord", {"lat": latf, "lon": lonf}, vkey)
    return (latf, lonf)

def _cc_prev_final_gamepk_for_team(team_name: str, before_iso_date: str) -> Optional[int]:
    """
    Most recent FINAL gamePk for team_name strictly before 'before_iso_date'. Cached 5 days.
    """
    norm = _norm_team_name(team_name)
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

    best_dt: Optional[_dt.datetime] = None
    best_pk: Optional[int] = None
    for drec in sched.get("dates", []):
        for g in drec.get("games", []):
            status = (g.get("status", {}).get("detailedState") or "").lower()
            if status != "final":
                continue
            home = (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name", "")
            away = (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name", "")
            if norm not in (home, away):
                continue
            dt = _parse_iso_dt(g.get("gameDate"))
            if not dt or dt.date() >= before_d:
                continue
            if (best_dt is None) or (dt > best_dt):
                best_dt = dt
                best_pk = g.get("gamePk")

    cache.save_json("prev_final", best_pk, ck)
    return best_pk

# ---------------- Travel distance ----------------
def cc_travel(home_team: str, away_team: str, game_date: str) -> Tuple[float, float]:
    """
    Great-circle travel km from each team's previous FINAL game venue to today's HOME park.
    Robust name normalization + multiple fallbacks for destination coords.
    """
    # Normalize names first
    home_n = _norm_team_name(home_team)
    away_n = _norm_team_name(away_team)

    ck = cache._make_key("travel", home_n, away_n, game_date)
    hit = cache.load_json("travel", ck, max_age_days=5)
    if isinstance(hit, (list, tuple)) and len(hit) == 2:
        try:
            return (float(hit[0]), float(hit[1]))
        except Exception:
            pass

    # 1) Resolve destination (home park) coords
    dest_lat, dest_lon = _resolve_home_destination_coords(game_date, home_n, away_n)
    if dest_lat is None or dest_lon is None:
        cache.save_json("travel", (0.0, 0.0), ck)
        return (0.0, 0.0)

    # 2) Previous final → coords → haversine
    def _km_from_prev(team: str) -> float:
        prev_pk = _cc_prev_final_gamepk_for_team(team, game_date)
        if not prev_pk:
            return 0.0
        lat, lon = _cc_venue_latlon_from_gamepk(prev_pk)
        if lat is None or lon is None:
            # if prev venue coords missing, try venue name fallback via live feed:
            # (We already do this inside _cc_venue_latlon_from_gamepk, so just return 0)
            return 0.0
        return float(_haversine_km(lat, lon, dest_lat, dest_lon))

    km_home = _km_from_prev(home_n)
    km_away = _km_from_prev(away_n)

    cache.save_json("travel", (km_home, km_away), ck)
    return (km_home, km_away)

# ---------------- Weather, Elo, Odds ----------------
def cc_weather(game_dict: dict) -> Optional[dict]:
    pk = (game_dict or {}).get("gamePk", "unknown")
    k = cache._make_key("weather", pk)
    cached = cache.load_json("weather", k, max_age_days=1)
    if cached is not None:
        return cached
    data = get_weather_for_game(game_dict)
    if data is not None:
        cache.save_json("weather", data, k)
    return data

def cc_elo(home: str, away: str, game_iso_date: str) -> float:
    k = cache._make_key("elo", home, away, game_iso_date)
    cached = cache.load_json("elo", k, max_age_days=7)
    if cached is not None:
        try:
            return float(cached)
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
    cached = cache.load_json("odds_imp_home", k, max_age_days=1)
    if cached is not None:
        try:
            return float(cached) if cached is not None else None
        except Exception:
            return None
    data = _call("get_closing_odds_implied_home", game_pk, default=None)
    cache.save_json("odds_imp_home", data, k)
    try:
        return float(data) if data is not None else None
    except Exception:
        return None

if __name__ == "__main__":
    # Tiny smoke test
    print(cc_weather({"gamePk": 123, "officialDate": "2024-09-14", "venue": {"name": "Yankee Stadium"}}))
