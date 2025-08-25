import os
import json
import math
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from utils import cache

from utils.safe_get_json import _safe_get_json

load_dotenv()  # Load .env variables
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # Loaded from .env

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
_BALLPARK_COORDS: Optional[dict] = None  # Format: {"Venue Name": [lat, lon, cf_direction_degrees]}

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
            wpct[tname] = (w / (w + l)) if (w + l) > 0 else 0.5
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

def get_team_offense_runs_pg_30d() -> Dict[str, float]:
    """Returns runs per game over last 30 days per team."""
    end = date.today()
    start = end - timedelta(days=30)
    games = get_schedule_range(start.isoformat(), end.isoformat())
    runs, games_played = {}, {}
    for g in games:
        status = ((g.get("status") or {}).get("detailedState") or "").lower()
        if status != "final":
            continue
        home = g["teams"]["home"]["team"]["name"]
        away = g["teams"]["away"]["team"]["name"]
        hs = g["teams"]["home"].get("score") or 0
        as_ = g["teams"]["away"].get("score") or 0
        runs[home] = runs.get(home, 0) + hs
        runs[away] = runs.get(away, 0) + as_
        games_played[home] = games_played.get(home, 0) + 1
        games_played[away] = games_played.get(away, 0) + 1
    return {t: (runs.get(t, 0) / n if n > 0 else 0.0) for t, n in games_played.items()}

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
            so = st.get("strikeOuts")
            bb = st.get("baseOnBalls")
            ip = _ip_to_float(st.get("inningsPitched"))
            k9v = (float(so) / ip * 9.0) if (so is not None and ip > 0) else None
            bb9v = (float(bb) / ip * 9.0) if (bb is not None and ip > 0) else None
            if era is None and e is not None: era = e
            if k9 is None and k9v is not None: k9 = k9v
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
            so = st.get("strikeOuts")
            bb = st.get("baseOnBalls")
            ip = _ip_to_float(st.get("inningsPitched"))
            k9v = (float(so) / ip * 9.0) if (so is not None and ip > 0) else None
            bb9v = (float(bb) / ip * 9.0) if (bb is not None and ip > 0) else None
            if era is None and e is not None: era = e
            if k9 is None and k9v is not None: k9 = k9v
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
            return datetime.fromisoformat(str(dt).replace("Z", "+00:00"))
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


def get_pitcher_last30(player_id: Optional[int]) -> dict:
    """Returns {era30} for last 30 days using byDateRange, with season stats fallback."""
    if not player_id:
        return {"era30": None}
    today = date.today()
    end = today.isoformat()
    start = (today - timedelta(days=30)).isoformat()
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=byDateRange&group=pitching&startDate={start}&endDate={end}"
    d = _safe_get_json(url) or {}
    era30 = None
    for blk in (d.get("stats") or []):
        for sp in (blk.get("splits") or []):
            st = sp.get("stat") or {}
            try:
                e = float(st["era"]) if st.get("era") not in (None, "") else None
            except Exception:
                e = None
            if era30 is None and e is not None:
                era30 = e
    if era30 is None:
        # Fallback to season stats
        season_stats = get_pitcher_season_stats(player_id)
        era30 = season_stats.get("era")
        print(f"Warning: No last-30-day stats for pitcher {player_id}, using season ERA: {era30}")
    return {"era30": era30}

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

def get_bullpen_ip_last3(team: str, game_date: str) -> Optional[float]:
    """Returns total innings pitched by bullpen in team's last 3 games before game_date."""
    if not team or not game_date:
        print(f"Warning: Invalid team {team} or date {game_date} for bullpen_ip_last3")
        return None
    try:
        game_date_obj = date.fromisoformat(game_date)
    except ValueError:
        print(f"Warning: Invalid date format {game_date} for bullpen_ip_last3")
        return None

    # Team ID mapping
    team_map = {
        "Los Angeles Dodgers": 119, "New York Yankees": 147, "San Francisco Giants": 137,
        "Oakland Athletics": 133, "Tampa Bay Rays": 139, "Los Angeles Angels": 108,
        "Baltimore Orioles": 110, "Boston Red Sox": 111, "Chicago White Sox": 145,
        "Cleveland Guardians": 114, "Detroit Tigers": 116, "Kansas City Royals": 118,
        "Minnesota Twins": 142, "Texas Rangers": 140, "Toronto Blue Jays": 141,
        "Arizona Diamondbacks": 109, "Atlanta Braves": 144, "Chicago Cubs": 112,
        "Cincinnati Reds": 113, "Colorado Rockies": 115, "Miami Marlins": 146,
        "Houston Astros": 117, "Milwaukee Brewers": 158, "Washington Nationals": 120,
        "New York Mets": 121, "Philadelphia Phillies": 143, "Pittsburgh Pirates": 134,
        "St. Louis Cardinals": 138, "San Diego Padres": 135, "Seattle Mariners": 136
    }
    team_id = team_map.get(team)
    if not team_id:
        print(f"Warning: No team ID for {team} in bullpen_ip_last3")
        return None

    # Fetch schedule for last 30 days
    cache_key = cache._make_key("schedule", team, game_date)
    schedule = cache.load_json("schedule", cache_key, max_age_days=7)
    if schedule is None:
        start_date = (game_date_obj - timedelta(days=30)).isoformat()
        url = f"https://statsapi.mlb.com/api/v1/schedule?teamId={team_id}&startDate={start_date}&endDate={game_date}&sportId=1"
        schedule = _safe_get_json(url)
        if schedule is None:
            print(f"Error: Failed to fetch schedule for team {team} (ID {team_id}) from {start_date} to {game_date}")
            return None
        cache.save_json("schedule", schedule, cache_key)

    # Get last 3 finalized games
    games = []
    for day in (schedule.get("dates") or []):
        for game in (day.get("games") or []):
            game_status = game.get("status", {}).get("detailedState")
            if game.get("gameDate") < game_date_obj.isoformat() and game_status == "Final":
                games.append(game)
    games = sorted(games, key=lambda x: x["gameDate"], reverse=True)[:3]

    if len(games) < 3:
        print(f"Warning: Only {len(games)} games found for {team} before {game_date}")
        return None

    total_ip = 0.0
    for game in games:
        game_pk = game.get("gamePk")
        boxscore_cache_key = cache._make_key("boxscore", game_pk)
        boxscore = cache.load_json("boxscore", boxscore_cache_key, max_age_days=7)
        if boxscore is None:
            boxscore_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
            boxscore = _safe_get_json(boxscore_url)
            if boxscore is None:
                print(f"Warning: Failed to fetch boxscore for gamePk {game_pk}")
                continue
            cache.save_json("boxscore", boxscore, boxscore_cache_key)

        team_key = "home" if game["teams"]["home"]["team"]["id"] == team_id else "away"
        players = boxscore.get("teams", {}).get(team_key, {}).get("players", {})
        for player in players.values():
            stats = player.get("stats", {}).get("pitching", {})
            if stats.get("gamesStarted", 0) == 0:  # Reliever, not starter
                innings = stats.get("inningsPitched", 0.0)
                try:
                    total_ip += float(innings)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid inningsPitched {innings} for gamePk {game_pk}")
                    continue

    if total_ip == 0.0:
        print(f"Warning: No bullpen IP for {team} in last 3 games before {game_date}")
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
        try:
            with open(path, "r", encoding="utf-8") as f:
                _BALLPARK_COORDS = json.load(f)
            print(f"Loaded ballpark coords for {len(_BALLPARK_COORDS)} venues from JSON")
        except Exception as e:
            print(f"Error loading ballpark_coords.json: {e}")
            _BALLPARK_COORDS = {
                "Angel Stadium of Anaheim": [33.8003, -117.8827, 90],
                "Angel Stadium": [33.8003, -117.8827, 90],
                "Oriole Park at Camden Yards": [39.2839, -76.6217, 31],
                "Camden Yards": [39.2839, -76.6217, 31],
                "Fenway Park": [42.3467, -71.0972, 37],
                "Guaranteed Rate Field": [41.83, -87.6339, 315],
                "Rate Field": [41.83, -87.6339, 315],
                "Progressive Field": [41.4962, -81.6853, 45],
                "Comerica Park": [42.339, -83.0485, 355],
                "Kauffman Stadium": [39.0513, -94.4805, 315],
                "Target Field": [44.9817, -93.2784, 315],
                "Yankee Stadium": [40.8296, -73.9262, 0],
                "Oakland Coliseum": [37.7516, -122.2005, 330],
                "T-Mobile Park": [47.5913, -122.3325, 355],
                "Tropicana Field": [27.7683, -82.6534, 315],
                "Globe Life Field": [32.7473, -97.0945, 45],
                "Rogers Centre": [43.6414, -79.3894, 315],
                "Chase Field": [33.4455, -112.0667, 0],
                "Truist Park": [33.8908, -84.4678, 158],
                "Wrigley Field": [41.9484, -87.6553, 45],
                "Great American Ball Park": [39.0979, -84.5066, 315],
                "Coors Field": [39.756, -104.9942, 315],
                "loanDepot park": [25.7781, -80.2196, 0],
                "Minute Maid Park": [29.7573, -95.3554, 315],
                "Dodger Stadium": [34.0739, -118.2401, 315],
                "American Family Field": [43.028, -87.9712, 315],
                "Nationals Park": [38.8729, -77.0074, 315],
                "Citi Field": [40.7571, -73.8458, 315],
                "Citizens Bank Park": [39.9058, -75.1665, 315],
                "PNC Park": [40.4469, -80.0057, 315],
                "Busch Stadium": [38.6226, -90.1928, 315],
                "Petco Park": [32.7073, -117.1573, 315],
                "Oracle Park": [37.7786, -122.3893, 315],
                "Sutter Health Park": [38.5801, -121.5133, 0],
                "George M. Steinbrenner Field": [27.9795, -82.5066, 315],
                "Journey Bank Ballpark": [41.2412, -77.0469, 315],
                "Daikin Park": [37.7516, -122.2005, 330]
            }
    # Case-insensitive aliasing
    aliased_coords = {}
    for venue, coords in _BALLPARK_COORDS.items():
        aliased_coords[venue] = coords
        aliased_coords[venue.lower()] = coords  # Add lowercase version
    aliases = {
        "angel stadium": "Angel Stadium of Anaheim",
        "rate field": "Guaranteed Rate Field",
        "camden yards": "Oriole Park at Camden Yards",
        "daikin park": "Daikin Park",
        "sutter health park": "Sutter Health Park",
        "george m. steinbrenner field": "George M. Steinbrenner Field",
        "journey bank ballpark": "Journey Bank Ballpark"
    }
    for alias, canonical in aliases.items():
        if canonical in _BALLPARK_COORDS:
            aliased_coords[alias] = _BALLPARK_COORDS[canonical]
            aliased_coords[alias.lower()] = _BALLPARK_COORDS[canonical]
    return aliased_coords

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
    Fetches weather for a game using WeatherAPI.com.
    Returns {'temp': float, 'wind_speed': float, 'wind_out_to_cf': bool, 'wind_cf_x_park': float}
    or None on failure (with defaults applied in preprocessing).
    """
    venue = (game_dict.get("venue") or {}).get("name") or ""
    game_date_str = game_dict.get("officialDate") or date.today().isoformat()
    game_date = date.fromisoformat(game_date_str)
    parks = _load_ballparks()  # Loads ballpark_coords.json or fallback
    latlon_cf = parks.get(venue)
    if not latlon_cf or len(latlon_cf) < 3:
        print(f"Warning: No coords or CF direction for {venue}; skipping weather fetch")
        return None
    lat, lon, cf_dir = latlon_cf
    q = f"{lat},{lon}"
    
    # Choose endpoint: history for past, current for today/future
    if game_date < date.today():
        url = f"http://api.weatherapi.com/v1/history.json?key={WEATHER_API_KEY}&q={q}&dt={game_date_str}&hour=15"  # 3 PM game time
    else:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={q}"
    
    data = _safe_get_json(url)
    if not data:
        print(f"Warning: No weather data for {venue} on {game_date_str}")
        return None
    
    # Extract from 'current' or 'forecastday' > 'hour' (for history)
    current = data.get("current") or data.get("forecast", {}).get("forecastday", [{}])[0].get("hour", [{}])[0]
    if not current:
        return None
    
    temp = float(current.get("temp_f", 72.0))  # Fahrenheit
    wind_speed = float(current.get("wind_mph", 5.0))  # mph
    wind_dir_deg = float(current.get("wind_degree", 0))  # Degrees
    
    # Compute wind effects
    dir_diff = abs((wind_dir_deg - cf_dir + 180) % 360 - 180)
    wind_out_to_cf = dir_diff <= 45  # Blowing out to CF ±45°
    wind_cf_x_park = 1.0 if dir_diff <= 45 else (0.5 if dir_diff <= 135 else -1.0)
    
    return {
        "temp": temp,
        "wind_speed": wind_speed,
        "wind_out_to_cf": wind_out_to_cf,
        "wind_cf_x_park": wind_cf_x_park
    }


# ---------------------------
# Odds / Elo (left unimplemented by default)
# ---------------------------
def get_closing_odds_implied_home(game_pk: int) -> Optional[float]:
    return None

def get_elo_diff(home: str, away: str, game_iso_date: str) -> float:
    return 0.0

# Odds / Elo
def get_closing_odds_implied_home(game_pk: int) -> Optional[float]:
    return None

def get_elo_diff(home_team: str, away_team: str, on_date: date) -> float:
    # Simple Elo: initialize teams at 1500, update based on game outcomes
    elo_ratings = {}  # Load from cache or file
    if not elo_ratings:
        teams = get_team_wpct_season().keys()
        elo_ratings = {t: 1500.0 for t in teams}
    games = get_schedule_range((on_date - timedelta(days=30)).isoformat(), (on_date - timedelta(days=1)).isoformat())
    for g in games:
        status = ((g.get("status") or {}).get("detailedState") or "").lower()
        if status != "final":
            continue
        home = g["teams"]["home"]["team"]["name"]
        away = g["teams"]["away"]["team"]["name"]
        hs = g["teams"]["home"].get("score", 0)
        as_ = g["teams"]["away"].get("score", 0)
        winner = home if hs > as_ else away
        loser = away if hs > as_ else home
        expected = 1 / (1 + 10 ** ((elo_ratings[loser] - elo_ratings[winner]) / 400))
        k = 32  # Adjust K-factor as needed
        elo_ratings[winner] += k * (1 - expected)
        elo_ratings[loser] -= k * expected
    return elo_ratings.get(home_team, 1500.0) - elo_ratings.get(away_team, 1500.0)


if __name__ == "__main__":
    from utils.sources import get_weather_for_game
    game_dict = {
        "officialDate": "2024-09-14",
        "venue": {"name": "Yankee Stadium"}
    }
    weather = get_weather_for_game(game_dict)
    print(weather)  # Expect: {'temp': ~75.0, 'wind_speed': ~8.0, 'wind_out_to_cf': True/False, 'wind_cf_x_park': 1.0/0.5/-1.0}