# utils/feature_builder.py
from __future__ import annotations
from typing import Any, Optional, List, Dict
import pandas as pd
from dateutil import parser

# Pull cached/raw inputs from the fetch layer
from utils import data_fetchers as DF

META_COLS = ["game_date", "home_team", "away_team", "home_win"]


# -------- tiny numeric guards --------
def _f(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(d)

def _i(x: Any, d: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(d)


# -------- single-game feature builder --------
def build_features_for_game(g: dict,
                            include_odds: bool = False,
                            include_weather: bool = True) -> Dict[str, Any]:
    """
    Convert one raw game dict (from DF.cc_schedule) into a flat feature row.
    Uses only cached fetchers from utils.data_fetchers (DF).
    """
    status = ((g.get("status") or {}).get("detailedState") or "").lower()

    home = g["teams"]["home"]["team"]["name"]
    away = g["teams"]["away"]["team"]["name"]
    game_dt = parser.isoparse(g["gameDate"])
    game_pk = g.get("gamePk")
    game_iso = game_dt.date().isoformat()

    # META
    if status == "final":
        hs = _i(g["teams"]["home"].get("score"))
        as_ = _i(g["teams"]["away"].get("score"))
        home_win = 1 if hs > as_ else 0
    else:
        home_win = None

    # Pre-cached dictionaries (already warmed in calling range)
    # (We still re-look them up to keep build isolated when used standalone.)
    wp_season = DF.cc_wpct_season()
    wp_30     = DF.cc_wpct_last30()
    bp14      = DF.cc_bullpen_era14()
    offense30 = DF.cc_offense30()

    # Probables
    probables = DF.cc_probables([game_pk]) if game_pk else {}
    ids = (probables.get(game_pk) or {})
    hid, aid = ids.get("home_id"), ids.get("away_id")

    # Pitchers
    s_h = DF.cc_pitcher_season(hid)
    s_a = DF.cc_pitcher_season(aid)

    # last-30 ERA (if your sources implements it; default None→0.0 diff)
    # we call sources via DF._call only inside DF; here assume unknown & treat as 0.0
    l30_h = {"era30": None}
    l30_a = {"era30": None}

    # last-3 starts
    l3_h = DF.cc_pitcher_last3(hid)
    l3_a = DF.cc_pitcher_last3(aid)

    # Team context
    team_wpct_diff_season = _f((wp_season.get(home, 0.5) or 0.5) - (wp_season.get(away, 0.5) or 0.5))
    team_wpct_diff_30d    = _f((wp_30.get(home, 0.5) or 0.5) - (wp_30.get(away, 0.5) or 0.5))
    home_advantage = 1

    # Starters
    starter_era_diff   = _f((s_h.get("era")   or 0.0) - (s_a.get("era")   or 0.0))
    starter_k9_diff    = _f((s_h.get("k9")    or 0.0) - (s_a.get("k9")    or 0.0))
    starter_bb9_diff   = _f((s_h.get("bb9")   or 0.0) - (s_a.get("bb9")   or 0.0))
    starter_era30_diff = _f((l30_h.get("era30") or 0.0) - (l30_a.get("era30") or 0.0))

    starter_era3_diff = _f((l3_h.get("era3") or 0.0) - (l3_a.get("era3") or 0.0))
    starter_kbb3_diff = _f((l3_h.get("kbb3") or 0.0) - (l3_a.get("kbb3") or 0.0))

    # Bullpen + Park
    bullpen_era14_diff = _f((bp14.get(home) if bp14.get(home) is not None else 0.0) -
                            (bp14.get(away) if bp14.get(away) is not None else 0.0))
    park_factor = _f(DF.cc_park_factor(home))

    # Rest / B2B
    home_days_rest = _i(DF.cc_days_rest(home, game_iso))
    away_days_rest = _i(DF.cc_days_rest(away, game_iso))
    b2b_flag = 1 if (home_days_rest == 0 or away_days_rest == 0) else 0

    # Travel
    km_home, km_away = DF.cc_travel(home, away, game_iso)
    travel_km_home_prev_to_today = _f(km_home)
    travel_km_away_prev_to_today = _f(km_away)

    # Bullpen IP last 3d (kept as uncached disk; quick and already API-cached in sources)
    # If you want to disk-cache, you can mirror cc_* pattern; usually not necessary.
    # We'll call sources directly via DF._call isn't exposed here; keep zeros or implement later.

    bullpen_ip_last3_home = 0.0
    bullpen_ip_last3_away = 0.0

    # Offense 30d diff
    offense_runs_pg_30d_diff = _f((offense30.get(home, 0.0) or 0.0) - (offense30.get(away, 0.0) or 0.0))

    # Weather
    wx_temp = wx_wind_speed = wx_wind_out_to_cf = 0.0
    if include_weather:
        wx = DF.cc_weather(g)
        if isinstance(wx, dict) and wx:
            wx_temp = _f(wx.get("temp", 0.0))
            wx_wind_speed = _f(wx.get("wind_speed", 0.0))
            wx_wind_out_to_cf = 1.0 if wx.get("wind_out_to_cf") else 0.0
    wind_cf_x_park = _f(wx_wind_out_to_cf * park_factor)

    # Elo + Odds
    elo_diff = _f(DF.cc_elo(home, away, game_iso))
    odds_implied_home_close = DF.cc_odds(game_pk) if include_odds else None

    row = {
        # META
        "game_date": game_iso,
        "home_team": home,
        "away_team": away,
        "home_win": home_win,

        # FEATURES
        "home_advantage": home_advantage,
        "team_wpct_diff_season": team_wpct_diff_season,
        "team_wpct_diff_30d": team_wpct_diff_30d,

        "starter_era_diff": starter_era_diff,
        "starter_k9_diff": starter_k9_diff,
        "starter_bb9_diff": starter_bb9_diff,
        "starter_era30_diff": starter_era30_diff,
        "starter_era3_diff": starter_era3_diff,
        "starter_kbb3_diff": starter_kbb3_diff,

        "bullpen_era14_diff": bullpen_era14_diff,
        "park_factor": park_factor,

        "home_days_rest": home_days_rest,
        "away_days_rest": away_days_rest,
        "b2b_flag": b2b_flag,

        "travel_km_home_prev_to_today": travel_km_home_prev_to_today,
        "travel_km_away_prev_to_today": travel_km_away_prev_to_today,
        "bullpen_ip_last3_home": bullpen_ip_last3_home,
        "bullpen_ip_last3_away": bullpen_ip_last3_away,

        "offense_runs_pg_30d_diff": offense_runs_pg_30d_diff,
        "wx_temp": wx_temp,
        "wx_wind_speed": wx_wind_speed,
        "wx_wind_out_to_cf": wx_wind_out_to_cf,
        "wind_cf_x_park": wind_cf_x_park,

        "elo_diff": elo_diff,
    }
    if include_odds and odds_implied_home_close is not None:
        row["odds_implied_home_close"] = _f(odds_implied_home_close)

    return row


# -------- batch/range builder (TRAIN or PREDICT) --------
def build_features_for_range(
    start_date: str,
    end_date: str,
    include_odds: bool = False,
    include_weather: bool = True,
    only_finals: bool = True,
    required_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame of meta + features for games in [start_date, end_date].
      - only_finals=True: training (keeps only completed games and sets home_win).
      - only_finals=False: prediction (keeps scheduled/live games; home_win=None).
    """
    games = DF.cc_schedule(start_date, end_date) or []
    if not games:
        return pd.DataFrame(columns=(required_features or []) + META_COLS)

    # warm common caches once (to avoid re-calling inside each row)
    _ = DF.cc_wpct_season()
    _ = DF.cc_wpct_last30()
    _ = DF.cc_bullpen_era14()
    _ = DF.cc_offense30()
    _ = DF.cc_probables([g.get("gamePk") for g in games])

    rows: List[Dict[str, Any]] = []
    for g in games:
        status = ((g.get("status") or {}).get("detailedState") or "").lower()
        if only_finals and status != "final":
            continue
        rows.append(build_features_for_game(g, include_odds=include_odds, include_weather=include_weather))

    df = pd.DataFrame(rows)

    # If a subset is requested, create any missing columns and order them
    if required_features:
        want = list(dict.fromkeys(list(required_features) + META_COLS))
        for c in want:
            if c not in df.columns:
                if c in ("home_team", "away_team"):
                    df[c] = ""
                elif c == "game_date":
                    df[c] = None
                elif c == "home_win":
                    df[c] = None
                else:
                    df[c] = 0.0
        df = df[want]

    return df


# -------- compatibility wrapper (old name) --------
def fetch_games_with_features(
    start_date: str,
    end_date: str,
    include_odds: bool = False,
    include_weather: bool = True,
    required_features: Optional[List[str]] = None,
    only_finals: bool = True,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper so older imports keep working.
    """
    return build_features_for_range(
        start_date=start_date,
        end_date=end_date,
        include_odds=include_odds,
        include_weather=include_weather,
        only_finals=only_finals,
        required_features=required_features,
    )



"""
Separates 'fetch raw games + feature engineering' from routes.
This calls your advanced cached fetcher in utils.data_fetchers.
"""
from typing import Optional
import pandas as pd
from utils.data_fetchers import fetch_games_with_features

def build_features_for_range(
    start_date: str,
    end_date: str,
    include_odds: bool = False,
    include_weather: bool = True,
    required_features: Optional[list[str]] = None,
    only_finals: bool = False,  # for predictions we want scheduled too
) -> pd.DataFrame:
    return fetch_games_with_features(
        start_date=start_date,
        end_date=end_date,
        include_odds=include_odds,
        include_weather=include_weather,
        required_features=required_features,
        only_finals=only_finals,
    )

