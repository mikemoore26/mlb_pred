# utils/data_fetchers/__init__.py
"""
Unified entrypoint for all MLB data fetchers.
Re-exports functions from submodules so existing imports keep working:
    from utils import data_fetchers as DF
"""

# Schedule
from utils.data_fetchers.schedule import cc_schedule_range

# Teams
from utils.data_fetchers.teams import cc_wpct_season, cc_wpct_last30, cc_offense30

# Pitchers
from utils.data_fetchers.pitchers import cc_pitcher_season, cc_pitcher_last30, cc_pitcher_last3

# Bullpen
from utils.data_fetchers.bullpen import cc_bullpen_ip_last3, cc_bullpen_era14

# Park/Venue
from utils.data_fetchers.park_venue import cc_park_factor, norm_team_name, cc_venue_latlon_from_gamepk

# Rest/Travel
from utils.data_fetchers.rest_travel import cc_days_rest

# Probables
from utils.data_fetchers.probables import cc_probables

# Weather
from utils.data_fetchers.weather import cc_weather

# Elo & Odds
from utils.data_fetchers.odds_elo import cc_elo, cc_odds

from utils.safe_get_json import _safe_get_json
def _stats_url_yearbyyear(pid: int) -> str:
    # Year-by-year pitching lines for the player
    return (f"https://statsapi.mlb.com/api/v1/people/{pid}"
            f"?hydrate=stats(group=[pitching],type=[yearByYear])")

from typing import Any
import re
_IP_RE = re.compile(r"^\s*(\d+)(?:\.(\d))?\s*$")

def _parse_ip_to_float(ip_val: Any) -> float:
    """
    MLB tenths: .1 = 1/3, .2 = 2/3
    """
    if ip_val is None:
        return 0.0
    try:
        if isinstance(ip_val, (int, float)):
            whole = int(ip_val)
            tenths = int(round((float(ip_val) - whole) * 10))
            tenths = tenths if tenths in (0, 1, 2) else 0
            return whole + tenths / 3.0
        s = str(ip_val).strip()
        m = _IP_RE.match(s)
        if not m:
            return 0.0
        whole = int(m.group(1))
        tenths = int(m.group(2) or 0)
        tenths = tenths if tenths in (0, 1, 2) else 0
        return whole + tenths / 3.0
    except Exception:
        return 0.0

def _best_recent_season_pitching_line(pid: int) -> dict:
    """
    Returns dict like {'era': float|None, 'k9': float|None, 'bb9': float|None,
                      'ip': float|None, 'games': int, 'season': int}
    Uses yearByYear; picks the most recent season with >0 games.
    """
    data = _safe_get_json(_stats_url_yearbyyear(pid)) or {}
    stats_blocks = (((data.get("people") or [{}])[0]).get("stats") or [])
    # Find the 'yearByYear' block
    yby = next((b for b in stats_blocks if (b.get("type", {}) or {}).get("displayName") == "yearByYear"), None)
    splits = (yby or {}).get("splits") or []
    # Newest first just in case
    try:
        splits = sorted(splits, key=lambda s: int(s.get("season", 0)), reverse=True)
    except Exception:
        pass

    for s in splits:
        stat = (s.get("stat") or {})
        games = int(stat.get("gamesPlayed") or 0)
        if games <= 0:
            continue
        season = int(s.get("season") or 0)
        # ERA/K9/BB9 may be strings; coerce carefully
        def _flt(v):
            try:
                return float(v)
            except Exception:
                return None
        era = _flt(stat.get("era"))
        k9  = _flt(stat.get("strikeoutsPer9Inn"))
        bb9 = _flt(stat.get("walksPer9Inn"))
        ip  = _parse_ip_to_float(stat.get("inningsPitched"))
        return {
            "era": era, "k9": k9, "bb9": bb9,
            "ip": (ip if ip is not None else None),
            "games": games, "season": season
        }
    # nothing found
    return {"era": None, "k9": None, "bb9": None, "ip": None, "games": 0, "season": None}

# ---------------- Smoke Test ----------------
if __name__ == "__main__":
    import datetime as _dt

    print("=== [SMOKE TEST] data_fetchers unified entrypoint ===")

    # 1. Team form
    print("\n[Teams]")
    print("Season WPCT (sample):", list(cc_wpct_season().items())[:3])
    print("Last30 WPCT (sample):", list(cc_wpct_last30().items())[:3])
    print("Offense30 (sample):", list(cc_offense30().items())[:3])

    # 2. Pitchers
    print("\n[Pitchers] Gerrit Cole (543037)")
    print("Season:", cc_pitcher_season(543037))
    print("Last30:", cc_pitcher_last30(543037))
    print("Last3 :", cc_pitcher_last3(543037))

    # 3. Bullpen
    today = _dt.date.today().isoformat()
    print("\n[Bullpen] Yankees bullpen")
    print("IP last3:", cc_bullpen_ip_last3("New York Yankees", today))
    print("ERA14 sample:", list(cc_bullpen_era14().items())[:3])

    # 4. Park/Venue
    print("\n[Park] Factor Coors Field (Colorado Rockies):", cc_park_factor("Colorado Rockies"))

    # 5. Schedule
    print("\n[Schedule] Upcoming 2 days")
    sched = cc_schedule_range(today, ( _dt.date.today() + _dt.timedelta(days=2) ).isoformat())
    print("Dates in schedule:", len(sched.get("dates", [])))

    # 6. Weather
    print("\n[Weather] Stub test")
    wx = cc_weather({"gamePk": 123, "venue": {"name": "Yankee Stadium"}})
    print("Weather block:", wx)

    # 7. Elo & Odds (mocked)
    print("\n[Elo/Odds] Example calls (will return defaults unless real API wired)")
    print("Elo:", cc_elo("New York Yankees", "Boston Red Sox", today))
    print("Odds:", cc_odds(776708))

    print("\n=== [END SMOKE TEST] ===")
