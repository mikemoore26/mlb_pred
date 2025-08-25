from __future__ import annotations
from typing import Any, Optional, List, Dict
import pandas as pd
from dateutil import parser
import datetime
import sys

# Pull cached/raw inputs from the fetch layer
from utils import data_fetchers as DF
from utils import cache
from utils.safe_get_json import _safe_get_json  # <-- FIX: use safe_get_json directly

META_COLS = ["game_date", "home_team", "away_team", "home_win"]

# -------- tiny numeric guards --------
def _f(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return float(d)

def _i(x: Any, d: int = 0) -> int:
    try:
        return int(x)
    except (ValueError, TypeError):
        return int(d)

# -------- single-game feature builder --------
def build_features_for_game(
    g: dict,
    include_odds: bool = False,
    include_weather: bool = True
) -> Dict[str, Any]:
    status = ((g.get("status") or {}).get("detailedState") or "").lower()

    home = g["teams"]["home"]["team"]["name"]
    away = g["teams"]["away"]["team"]["name"]
    game_dt = parser.isoparse(g["gameDate"])
    game_pk = g.get("gamePk")
    game_iso = game_dt.date().isoformat()

    # META label
    if status == "final":
        hs = _i(g["teams"]["home"].get("score"))
        as_ = _i(g["teams"]["away"].get("score"))
        home_win = 1 if hs > as_ else 0
    else:
        home_win = None

    # cached dicts (always present; may be partial)
    wp_season = DF.cc_wpct_season() or {}
    wp_30     = DF.cc_wpct_last30() or {}
    bp14      = DF.cc_bullpen_era14() or {}
    offense30 = DF.cc_offense30() or {}

    # probables + pitcher stats
    probables = DF.cc_probables([game_pk]) if game_pk else {}
    ids = (probables.get(game_pk) or {}) if isinstance(probables, dict) else {}
    hid, aid = ids.get("home_id"), ids.get("away_id")

    s_h  = DF.cc_pitcher_season(hid) or {}
    s_a  = DF.cc_pitcher_season(aid) or {}
    l30h = DF.cc_pitcher_last30(hid) or {"era30": None}
    l30a = DF.cc_pitcher_last30(aid) or {"era30": None}
    l3h  = DF.cc_pitcher_last3(hid)  or {"era3": None, "kbb3": None}
    l3a  = DF.cc_pitcher_last3(aid)  or {"era3": None, "kbb3": None}

    # context diffs (home - away) — COERCE EACH SIDE FIRST
    team_wpct_diff_season = _f(wp_season.get(home, 0.5)) - _f(wp_season.get(away, 0.5))
    team_wpct_diff_30d    = _f(wp_30.get(home, 0.5))     - _f(wp_30.get(away, 0.5))
    home_advantage = 1

    starter_era_diff   = _f(s_h.get("era"))    - _f(s_a.get("era"))
    starter_k9_diff    = _f(s_h.get("k9"))     - _f(s_a.get("k9"))
    starter_bb9_diff   = _f(s_h.get("bb9"))    - _f(s_a.get("bb9"))
    starter_era30_diff = _f(l30h.get("era30")) - _f(l30a.get("era30"))
    starter_era3_diff  = _f(l3h.get("era3"))   - _f(l3a.get("era3"))
    starter_kbb3_diff  = _f(l3h.get("kbb3"))   - _f(l3a.get("kbb3"))

    bullpen_era14_diff = _f(bp14.get(home)) - _f(bp14.get(away))
    park_factor = _f(DF.cc_park_factor(home))

    home_days_rest = _i(DF.cc_days_rest(home, game_iso))
    away_days_rest = _i(DF.cc_days_rest(away, game_iso))
    b2b_flag = 1 if (home_days_rest == 0 or away_days_rest == 0) else 0

    km_home, km_away = DF.cc_travel(home, away, game_iso)
    travel_km_home_prev_to_today = _f(km_home)
    travel_km_away_prev_to_today = _f(km_away)

    bullpen_ip_last3_home = _f(DF.cc_bullpen_ip_last3(home, game_iso))
    bullpen_ip_last3_away = _f(DF.cc_bullpen_ip_last3(away, game_iso))

    offense_runs_pg_30d_diff = _f(offense30.get(home)) - _f(offense30.get(away))

    # weather
    wx_temp = wx_wind_speed = wx_wind_out_to_cf = 0.0
    if include_weather:
        wx = DF.cc_weather(g)
        if isinstance(wx, dict) and wx:
            wx_temp = _f(wx.get("temp"))
            wx_wind_speed = _f(wx.get("wind_speed"))
            wx_wind_out_to_cf = 1.0 if wx.get("wind_out_to_cf") else 0.0
    wind_cf_x_park = _f(wx_wind_out_to_cf * park_factor)

    # elo + odds
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

# -------- batch/range builder --------
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
    """
    # Future date guard: if training but end is today/future, flip to prediction mode
    try:
        end_date_obj = datetime.date.fromisoformat(end_date)
        if end_date_obj >= datetime.date.today() and only_finals:
            only_finals = False
    except Exception:
        pass

    # Cached schedule
    cache_key = cache._make_key("schedule", start_date, end_date)
    cached_schedule = cache.load_json("schedule", cache_key, max_age_days=1)
    if cached_schedule is not None:
        games = [g for d in cached_schedule.get("dates", []) for g in d.get("games", [])]
    else:
        url = (
            "https://statsapi.mlb.com/api/v1/schedule"
            f"?startDate={start_date}&endDate={end_date}&sportId=1&hydrate=probablePitchers"
        )
        schedule = _safe_get_json(url)  # <-- FIX: use imported helper
        if schedule is None:
            return pd.DataFrame(columns=(required_features or []) + META_COLS)
        cache.save_json("schedule", schedule, cache_key)
        games = [g for d in schedule.get("dates", []) for g in d.get("games", [])]

    if not games:
        return pd.DataFrame(columns=(required_features or []) + META_COLS)
    
        # ... after you computed `games` list from schedule ...

    # Seed probables cache directly from the schedule (quietly)
    try:
        batch_seed = {}
        for g in games:
            pk = g.get("gamePk")
            if not pk:
                continue
            home_id = (g.get("teams", {}).get("home", {}).get("probablePitcher", {}) or {}).get("id")
            away_id = (g.get("teams", {}).get("away", {}).get("probablePitcher", {}) or {}).get("id")
            if home_id or away_id:
                rec = {"home_id": home_id, "away_id": away_id}
            else:
                rec = {}  # negative cache if schedule has none
            per_key = cache._make_key("probables", pk)
            cache.save_json("probables", rec, per_key)
            batch_seed[pk] = rec
        if batch_seed:
            batch_key = cache._make_key("probables_batch", *sorted(batch_seed.keys()))
            cache.save_json("probables", batch_seed, batch_key)
    except Exception:
        pass


    # warm caches once
    _ = DF.cc_wpct_season()
    _ = DF.cc_wpct_last30()
    _ = DF.cc_bullpen_era14()
    _ = DF.cc_offense30()
    _ = DF.cc_probables([g.get("gamePk") for g in games if g.get("gamePk")])

    rows: List[Dict[str, Any]] = []
    for g in games:
        status = ((g.get("status") or {}).get("detailedState") or "").lower()
        if only_finals and status != "final":
            continue
        rows.append(build_features_for_game(g, include_odds=include_odds, include_weather=include_weather))

    df = pd.DataFrame(rows)

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

# compatibility alias
def fetch_games_with_features(*args, **kwargs) -> pd.DataFrame:
    return build_features_for_range(*args, **kwargs)
