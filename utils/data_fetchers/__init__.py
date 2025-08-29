# utils/data_fetchers/__init__.py
"""
Unified entrypoint for all MLB data fetchers.

Keeps backward-compatible imports:
    from utils import data_fetchers as DF
"""

# -------- Schedule --------
from utils.data_fetchers.schedule import cc_schedule_range

# -------- Teams --------
from utils.data_fetchers.teams import (
    cc_wpct_season,
    cc_wpct_last30,
    cc_offense30,
)

# -------- Pitchers --------
from utils.data_fetchers.pitchers import (
    cc_pitcher_season,
    cc_pitcher_last30,
    cc_pitcher_last3,
    cc_pitch_lastx,  # <- new generic rolling-window helper
)

# -------- Bullpen --------
from utils.data_fetchers.bullpen import (
    cc_bullpen_ip_last3,
    cc_bullpen_era14,
)

# -------- Park / Venue --------
from utils.data_fetchers.park_venue import (
    cc_park_factor,
    norm_team_name,
    cc_venue_latlon_from_gamepk,
)

# -------- Rest / Travel --------
from utils.data_fetchers.rest_travel import cc_days_rest

# -------- Probables --------
from utils.data_fetchers.probables import cc_probables

# -------- Weather --------
from utils.data_fetchers.weather import cc_weather

# -------- Elo & Odds (mocked unless you wire real APIs) --------
from utils.data_fetchers.odds_elo import cc_elo, cc_odds

from utils.data_fetchers.travel import cc_travel

from utils.data_fetchers.odds_live import fetch_live_odds_df

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
    COLE = 543037  # Gerrit Cole
    print("\n[Pitchers] Gerrit Cole (543037)")
    print("Season:", cc_pitcher_season(COLE))
    print("Last30:", cc_pitcher_last30(COLE))
    print("Last3 :", cc_pitcher_last3(COLE))

    # 3. Bullpen
    today = _dt.date.today().isoformat()
    print("\n[Bullpen] Yankees bullpen")
    print("IP last3:", cc_bullpen_ip_last3("New York Yankees", today))
    print("ERA14 sample:", list(cc_bullpen_era14().items())[:3])

    # 4. Park/Venue
    print("\n[Park] Factor Coors Field (Colorado Rockies):", cc_park_factor("Colorado Rockies"))

    # 5. Schedule
    print("\n[Schedule] Upcoming 2 days")
    sched = cc_schedule_range(today, (_dt.date.today() + _dt.timedelta(days=2)).isoformat())
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
