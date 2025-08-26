# utils/data_fetchers.py
from __future__ import annotations
import os
import json
import re
import math
import datetime as _dt
from typing import Optional, List, Dict, Tuple, Any

import requests

import utils.cache as cache
from utils.safe_get_json import _safe_get_json

# --------------- Small helpers ---------------
# --- Team offense last 30 days (runs per game) -------------------------------
def cc_offense30() -> Dict[str, float]:
    """
    Returns {team_name: runs_per_game_last_30_days} with a 1-day cache TTL.
    We compute this from final games in the last 30 days using StatsAPI scores.
    """
    k = cache._make_key("offense30", "global")
    cached = cache.load_json("offense30", k, max_age_days=1)
    if cached is not None:
        # Ensure float conversion in case it was saved as strings
        try:
            return {t: float(v) for t, v in cached.items()}
        except Exception:
            return cached  # fallback to whatever is there

    end_d = _dt.date.today()
    start_d = end_d - _dt.timedelta(days=30)
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?startDate={start_d.isoformat()}&endDate={end_d.isoformat()}&sportId=1"
    )

    data = _safe_get_json(url) or {}

    # Aggregate runs and games per team
    agg: Dict[str, Dict[str, float]] = {}
    for drec in data.get("dates", []):
        for g in drec.get("games", []):
            status = (g.get("status", {}).get("detailedState") or "").lower()
            if status != "final":
                continue
            home = (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name")
            away = (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name")
            h_runs = float(((g.get("teams") or {}).get("home") or {}).get("score") or 0)
            a_runs = float(((g.get("teams") or {}).get("away") or {}).get("score") or 0)

            if home:
                agg.setdefault(home, {"runs": 0.0, "games": 0.0})
                agg[home]["runs"] += h_runs
                agg[home]["games"] += 1.0
            if away:
                agg.setdefault(away, {"runs": 0.0, "games": 0.0})
                agg[away]["runs"] += a_runs
                agg[away]["games"] += 1.0

    # Convert to runs per game; default 0.0 if no games
    out = {}
    for team, d in agg.items():
        games = float(d.get("games") or 0.0)
        runs = float(d.get("runs") or 0.0)
        out[team] = (runs / games) if games > 0 else 0.0

    cache.save_json("offense30", out, k)
    return out

def _parse_iso_dt(s) -> Optional[_dt.datetime]:
    try:
        return _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        try:
            return _dt.datetime.fromisoformat(str(s))
        except Exception:
            return None

_IP_RE = re.compile(r"^\s*(\d+)(?:\.(\d))?\s*$")

def _parse_ip_to_float(ip_str: Any) -> float:
    """
    Convert MLB-style innings string to decimal innings.
    .1 -> 1/3, .2 -> 2/3
    """
    if ip_str is None:
        return 0.0
    try:
        if isinstance(ip_str, (int, float)):
            whole = int(ip_str)
            tenths = int(round((float(ip_str) - whole) * 10))
            tenths = 0 if tenths not in (0, 1, 2) else tenths
            return whole + tenths / 3.0
        s = str(ip_str).strip()
        m = _IP_RE.match(s)
        if not m:
            return 0.0
        whole = int(m.group(1))
        tenths = int(m.group(2) or 0)
        tenths = 0 if tenths not in (0, 1, 2) else tenths
        return whole + tenths / 3.0
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

def _team_alias_map() -> Dict[str, str]:
    pf = _load_park_factors()
    aliases = pf.get("aliases", {})
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
        "cubs": "Chicago Cubs",
        "white sox": "Chicago White Sox",
    }
    final_map = {k.lower(): v for k, v in manual.items()}
    # Only fold in team->team aliases (not stadium names)
    for k, v in aliases.items():
        if isinstance(k, str) and isinstance(v, str):
            if not any(tok in v.lower() for tok in ("park", "field", "stadium")):
                final_map[k.lower()] = v
    return final_map

def _norm_team_name(name: str) -> str:
    if not name:
        return name
    m = _team_alias_map()
    return m.get(name.lower().strip(), name.strip())

def _coords_from_venue_name(venue_name: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
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
# Cached wrappers / Core fetches
# --------------------------------------------------------------------

def cc_schedule_range(start_date: str, end_date: str) -> dict:
    """
    Cached MLB schedule JSON for a date range (1-day TTL).
    """
    k = cache._make_key("schedule", f"{start_date}_{end_date}")
    hit = cache.load_json("schedule", k, max_age_days=1)
    if hit is not None:
        print(f"[df] schedule cache HIT {start_date}..{end_date}")
        return hit
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?startDate={start_date}&endDate={end_date}&sportId=1&hydrate=probablePitchers"
    )
    data = _safe_get_json(url) or {}
    print(f"[df] schedule cache MISS {start_date}..{end_date} -> dates={len(data.get('dates', []))}")
    cache.save_json("schedule", data, k)
    return data


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
        out = float(val) if val is not None else 100.0
    except Exception:
        out = 100.0
    return out


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


# ---------------- Probables (robust, with batch + negative cache) ----------------
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

    # 3-day lookahead via schedule hydrate
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


# ---------------- Venue coords helpers ----------------
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

    if latf is None or lonf is None:
        venue_name = ((data.get("gameData") or {}).get("venue") or {}).get("name")
        latf, lonf = _coords_from_venue_name(venue_name)

    cache.save_json("venue_coord", {"lat": latf, "lon": lonf}, vkey)
    return (latf, lonf)


# ---------------- Pitchers (robust) ----------------

def _year_from_date_iso(iso: str) -> int:
    try:
        return _dt.date.fromisoformat(iso).year
    except Exception:
        return _dt.date.today().year

def cc_pitcher_season(player_id: Optional[int]) -> Dict[str, Optional[float]]:
    """
    Season-level for current year. Returns dict with possibly None values:
    {'era': float|None, 'k9': float|None, 'bb9': float|None}
    """
    if not player_id:
        return {"era": None, "k9": None, "bb9": None}
    k = cache._make_key("p_season", player_id)
    hit = cache.load_json("p_season", k, max_age_days=7)
    if isinstance(hit, dict):
        return hit

    year = _dt.date.today().year
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&group=pitching&season={year}"
    data = _safe_get_json(url) or {}
    era = k9 = bb9 = None
    try:
        stats_blocks = data.get("stats", [])
        if stats_blocks:
            splits = stats_blocks[0].get("splits", [])
            if splits:
                stat = (splits[0] or {}).get("stat", {}) or {}
                era = float(stat["era"]) if stat.get("era") not in (None, "") else None
                k9 = float(stat["strikeoutsPer9Inn"]) if stat.get("strikeoutsPer9Inn") not in (None, "") else None
                bb9 = float(stat["walksPer9Inn"]) if stat.get("walksPer9Inn") not in (None, "") else None
    except Exception:
        pass

    out = {"era": era, "k9": k9, "bb9": bb9}
    cache.save_json("p_season", out, k)
    print(f"[df] cc_pitcher_season({player_id}) -> {out}")
    return out

def _aggregate_pitching_from_gamelog(splits: list, max_games: Optional[int] = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Aggregate ERA, K, BB from gamelog splits.
    Returns (ERA, K, BB) where ERA is computed from ER and IP.
    """
    # splits are usually newest first, but be defensive: sort by date descending
    try:
        splits = sorted(splits, key=lambda s: s.get("date", ""), reverse=True)
    except Exception:
        splits = splits or []

    if max_games is not None:
        splits = splits[:max_games]

    ER = 0.0
    IP = 0.0
    K = 0
    BB = 0
    GS = 0

    for s in splits:
        stat = (s or {}).get("stat", {}) or {}
        # Games started flag:
        gs = stat.get("gamesStarted", 0) or 0
        GS += int(gs)
        # innings pitched as MLB tenths string
        ip = _parse_ip_to_float(stat.get("inningsPitched"))
        IP += ip
        ER += float(stat.get("earnedRuns", 0) or 0)
        K  += int(stat.get("strikeouts", 0) or 0)
        BB += int(stat.get("baseOnBalls", 0) or 0)

    ERA = None
    if IP > 0:
        ERA = 9.0 * ER / IP

    return (ERA, float(K), float(BB))

def cc_pitcher_last30(player_id: Optional[int]) -> Dict[str, Optional[float]]:
    """
    30-day pitching ERA (best-effort).
    Tries byDateRange → fallback to gameLog aggregate over last 30 days.
    """
    if not player_id:
        return {"era30": None}

    k = cache._make_key("p_last30", player_id)
    hit = cache.load_json("p_last30", k, max_age_days=1)
    if isinstance(hit, dict):
        return hit

    end_d = _dt.date.today()
    start_d = end_d - _dt.timedelta(days=30)

    # 1) byDateRange
    url = (f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
           f"?stats=byDateRange&group=pitching&startDate={start_d.isoformat()}&endDate={end_d.isoformat()}")
    data = _safe_get_json(url) or {}
    era = None
    try:
        blocks = data.get("stats", [])
        if blocks and blocks[0].get("splits"):
            stat = (blocks[0]["splits"][0] or {}).get("stat", {}) or {}
            e = stat.get("era")
            era = float(e) if e not in (None, "") else None
    except Exception:
        pass

    # 2) fallback to gameLog over the same window
    if era is None:
        url2 = (f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
                f"?stats=gameLog&group=pitching&startDate={start_d.isoformat()}&endDate={end_d.isoformat()}")
        data2 = _safe_get_json(url2) or {}
        splits = []
        try:
            splits = (data2.get("stats", []) or [{}])[0].get("splits", []) or []
        except Exception:
            splits = []
        ERA, _, _ = _aggregate_pitching_from_gamelog(splits, max_games=None)
        era = ERA

    out = {"era30": (float(era) if era is not None else None)}
    cache.save_json("p_last30", out, k)
    print(f"[df] cc_pitcher_last30({player_id}) -> {out}")
    return out

def cc_pitcher_last3(player_id: Optional[int]) -> Dict[str, Optional[float]]:
    """
    Last-3 *starts* (not relief outings) from gameLog.
    Returns {'era3': float|None, 'kbb3': float|None}
    """
    if not player_id:
        return {"era3": None, "kbb3": None}

    k = cache._make_key("p_last3", player_id)
    hit = cache.load_json("p_last3", k, max_age_days=1)
    if isinstance(hit, dict):
        return hit

    end_d = _dt.date.today()
    start_d = end_d - _dt.timedelta(days=60)  # give ourselves enough room
    url = (f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
           f"?stats=gameLog&group=pitching&startDate={start_d.isoformat()}&endDate={end_d.isoformat()}")
    data = _safe_get_json(url) or {}
    splits_all = (data.get("stats", []) or [{}])[0].get("splits", []) or []

    # filter only starts (gamesStarted > 0), newest 3
    try:
        splits_all = sorted(splits_all, key=lambda s: s.get("date", ""), reverse=True)
    except Exception:
        pass
    starts = [s for s in splits_all if ((s.get("stat", {}) or {}).get("gamesStarted", 0) or 0) > 0][:3]

    ERA, K, BB = _aggregate_pitching_from_gamelog(starts, max_games=None)
    kbb = None
    if BB > 0:
        kbb = float(K) / float(BB)
    elif K > 0:
        kbb = float(K)

    out = {"era3": (float(ERA) if ERA is not None else None),
           "kbb3": (float(kbb) if kbb is not None else None)}
    cache.save_json("p_last3", out, k)
    print(f"[df] cc_pitcher_last3({player_id}) -> {out}")
    return out


# ---------------- Bullpen IP last 3 days (boxscore-based) ----------------

def _cc_games_for_team_in_window(team_name: str, start_date: str, end_date: str) -> List[int]:
    """Return gamePks where team played between start_date..end_date (inclusive)."""
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?startDate={start_date}&endDate={end_date}&sportId=1")
    sched = _safe_get_json(url) or {}
    pks: List[int] = []
    tnorm = _norm_team_name(team_name)
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            ht = (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name", "")
            at = (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name", "")
            if tnorm in (ht, at):
                pk = g.get("gamePk")
                if pk:
                    pks.append(int(pk))
    return pks

def _sum_relief_ip_from_boxscore(game_pk: int, team_side: str) -> float:
    """
    Sum innings pitched for relievers (non-starters) for the given team_side ('home'/'away').
    Heuristic: gamesStarted == 0 counts as relief. Uses live boxscore.
    """
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    data = _safe_get_json(url) or {}
    tm = (data.get("teams", {}) or {}).get(team_side, {}) or {}
    players = tm.get("players", {}) or {}
    total_ip = 0.0
    for pid, p in players.items():
        stat = (p.get("stats", {}) or {}).get("pitching", {}) or {}
        gs = int(stat.get("gamesStarted", 0) or 0)
        ip = _parse_ip_to_float(stat.get("inningsPitched"))
        if ip <= 0:
            continue
        # count relief: not a starter in this game
        if gs == 0:
            total_ip += ip
    return total_ip

def get_bullpen_ip_last3(team: str, game_date: str) -> float:
    """
    Sum relief IP in the 3-day window BEFORE game_date for the given team (home+away games).
    """
    try:
        g = _dt.date.fromisoformat(game_date)
    except Exception:
        return 0.0
    start = (g - _dt.timedelta(days=3)).isoformat()
    end = (g - _dt.timedelta(days=1)).isoformat()

    pks = _cc_games_for_team_in_window(team, start, end)
    if not pks:
        return 0.0

    total = 0.0
    for pk in pks:
        # Need to figure out if team was home/away for this game
        live = _safe_get_json(f"https://statsapi.mlb.com/api/v1/game/{pk}/feed/live") or {}
        home = (((live.get("gameData") or {}).get("teams") or {}).get("home") or {}).get("name", "")
        away = (((live.get("gameData") or {}).get("teams") or {}).get("away") or {}).get("name", "")
        side = "home" if _norm_team_name(team) == _norm_team_name(home) else "away"
        total += _sum_relief_ip_from_boxscore(pk, side)
    return float(total)

def cc_bullpen_ip_last3(team: str, game_date: str) -> float:
    k = cache._make_key("bullpen_ip_last3", team, game_date)
    cached = cache.load_json("bullpen_ip_last3", k, max_age_days=1)
    if cached is not None:
        try:
            return float(cached)
        except Exception:
            return 0.0
    val = get_bullpen_ip_last3(team, game_date)
    out = float(val) if val is not None else 0.0
    cache.save_json("bullpen_ip_last3", out, k)
    print(f"[df] cc_bullpen_ip_last3({team}, {game_date}) -> {out}")
    return out


# ---------------- Team form placeholders (season/last30) ----------------

def cc_wpct_season() -> Dict[str, float]:
    k = cache._make_key("wpct_season", "cur")
    cached = cache.load_json("wpct_season", k, max_age_days=1)
    if cached is not None:
        return cached
    year = _dt.date.today().year
    url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={year}"
    data = _safe_get_json(url) or {}
    wpct = {}
    for standing in data.get("records", []):
        for team in standing.get("teamRecords", []):
            team_name = team["team"]["name"]
            wins = team.get("wins", 0) or 0
            losses = team.get("losses", 0) or 0
            wpct[team_name] = wins / (wins + losses) if (wins + losses) > 0 else 0.5
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
    wpct = {}
    for date in data.get("dates", []):
        for game in date.get("games", []):
            if (game.get("status", {}).get("detailedState") or "").lower() != "final":
                continue
            home_team = game["teams"]["home"]["team"]["name"]
            away_team = game["teams"]["away"]["team"]["name"]
            home_score = game["teams"]["home"].get("score", 0) or 0
            away_score = game["teams"]["away"].get("score", 0) or 0
            winner = home_team if home_score > away_score else away_team
            for team in (home_team, away_team):
                wpct.setdefault(team, {"wins": 0, "games": 0})
                wpct[team]["games"] += 1
                if team == winner:
                    wpct[team]["wins"] += 1
    result = {t: (d["wins"] / d["games"] if d["games"] > 0 else 0.5) for t, d in wpct.items()}
    cache.save_json("wpct_last30", result, k)
    return result

def cc_bullpen_era14() -> Dict[str, float]:
    k = cache._make_key("bp14", "global")
    cached = cache.load_json("bp14", k, max_age_days=1)
    if cached is not None:
        return cached
    # Placeholder neutral values (until you compute true bullpen ERA)
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


# ---------------- Weather, Elo, Odds ----------------

def cc_weather(game_dict: dict) -> Optional[dict]:
    """
    Placeholder that returns an empty or simple weather record.
    You can wire a real provider later; this keeps feature builder stable.
    """
    try:
        game_pk = game_dict.get("gamePk")
        vlat, vlon = _cc_venue_latlon_from_gamepk(game_pk) if game_pk else (None, None)
        # Return simple, neutral defaults
        out = {"temp": 75.0, "wind_speed": 8.0, "wind_out_to_cf": False, "wind_cf_x_park": 0.0}
        cache.save_json("weather", out, cache._make_key("weather", game_pk or "unknown"))
        return out
    except Exception:
        return {"temp": 75.0, "wind_speed": 8.0, "wind_out_to_cf": False, "wind_cf_x_park": 0.0}

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


# ----------------- Tiny demo block -----------------
if __name__ == "__main__":
    print("[df] weather smoke test:")
    print(cc_weather({"gamePk": 123, "officialDate": "2024-09-14", "venue": {"name": "Yankee Stadium"}}))

    print("\n[df] Pitcher smoke: Gerrit Cole (543037)")
    print("SEASON:", cc_pitcher_season(543037))
    print("LAST30:", cc_pitcher_last30(543037))
    print("LAST3 :", cc_pitcher_last3(543037))

    # bullpen last-3 days example (team string must match StatsAPI schedule names)
    today = _dt.date.today().isoformat()
    print("\n[df] Bullpen IP last3 (New York Yankees):", cc_bullpen_ip_last3("New York Yankees", today))
