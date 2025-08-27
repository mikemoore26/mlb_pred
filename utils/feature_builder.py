from __future__ import annotations
from typing import Any, Optional, List, Dict
import traceback
import pandas as pd
from dateutil import parser
import datetime

# Pull cached/raw inputs from the fetch layer
from utils import data_fetchers as DF
from utils import cache
from utils.safe_get_json import _safe_get_json

# Composed feature blocks
from utils.design_features import (
    f, i,  # coercers that never throw (f(None)->0.0, i(None)->0)
    team_form,               # -> {"home_advantage", "team_wpct_diff_season", "team_wpct_diff_30d"}
    park_factor_feature,     # -> {"park_factor"}
    pitcher_diffs,           # -> {"starter_era_diff", "starter_k9_diff", "starter_bb9_diff", "starter_era30_diff", "starter_era3_diff", "starter_kbb3_diff"}
    cc_bullpen_era14,      # -> {"bullpen_era14_diff"}
    cc_bullpen_ip_last3,        # -> {"bullpen_ip_last3_home", "bullpen_ip_last3_away"}
    rest_b2b,                # -> {"home_days_rest", "away_days_rest", "b2b_flag"}
    travel_km,               # -> {"travel_km_home_prev_to_today", "travel_km_away_prev_to_today"}
    weather_block,           # -> {"wx_temp","wx_wind_speed","wx_wind_out_to_cf","wind_cf_x_park"}
    elo_feature,             # -> {"elo_diff"}
    odds_feature,            # -> {"odds_implied_home_close"} or {}
)

META_COLS = ["game_date", "home_team", "away_team", "home_win"]

def _status_of(g: dict) -> str:
    return ((g.get("status") or {}).get("detailedState") or "").lower()

def _probable_ids_from_schedule(g: dict) -> tuple[Optional[int], Optional[int]]:
    """Best-effort probable pitcher IDs from the schedule item itself."""
    try:
        hid = ((g["teams"]["home"].get("probablePitcher") or {}).get("id"))
        aid = ((g["teams"]["away"].get("probablePitcher") or {}).get("id"))
        return hid, aid
    except Exception:
        return None, None

# -------- single-game feature builder --------
def build_features_for_game(
    g: dict,
    include_odds: bool = False,
    include_weather: bool = True,
) -> Dict[str, Any]:
    """
    Build all features for a single schedule JSON game record.
    This function is defensive: any None/invalid numeric value is coerced via f()/i().
    """
    # Minimal meta
    status = _status_of(g)
    home = (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name", "")
    away = (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name", "")
    game_pk = g.get("gamePk")
    try:
        game_dt = parser.isoparse(g["gameDate"])
        game_iso = game_dt.date().isoformat()
    except Exception:
        # if gameDate is malformed, default to today's date to avoid key errors downstream
        game_iso = datetime.date.today().isoformat()

    # META label
    if status == "final":
        hs = i(((g.get("teams") or {}).get("home") or {}).get("score"))
        as_ = i(((g.get("teams") or {}).get("away") or {}).get("score"))
        home_win = 1 if hs > as_ else 0
    else:
        home_win = None

    # Probables (best-effort): seed with schedule, then enrich from cached probables
    sch_hid, sch_aid = _probable_ids_from_schedule(g)
    prob = DF.cc_probables([game_pk]) if game_pk else {}
    ids = prob.get(game_pk) or {}
    hid = ids.get("home_id", sch_hid)
    aid = ids.get("away_id", sch_aid)

    # Collect each block with try/except; keep track for debug
    blocks: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []

    def _run_block(name: str, fn, *args, **kwargs):
        try:
            out = fn(*args, **kwargs) or {}
            if not isinstance(out, dict):
                raise TypeError(f"{name} returned non-dict: {type(out)}")
            # Coerce all numerics inside this block
            safe_out: Dict[str, Any] = {}
            for k, v in out.items():
                # meta & booleans can stay as-is
                if k in ("home_team", "away_team", "game_date", "odds_note"):
                    safe_out[k] = v
                else:
                    # Try numeric coercion; if it fails, fallback 0.0
                    try:
                        # preserve ints for *_flag or *_days like fields; else float
                        if isinstance(v, bool):
                            safe_out[k] = int(v)
                        elif isinstance(v, int):
                            safe_out[k] = int(v)
                        else:
                            safe_out[k] = f(v, 0.0)
                    except Exception:
                        safe_out[k] = 0.0
            blocks[name] = safe_out
        except Exception as e:
            errors.append(f"[{name}] {type(e).__name__}: {e}")
            blocks[name] = {}

    # Blocks (arguments chosen to minimize repeated lookups)
    _run_block("team_form",         team_form,         home, away)
    _run_block("park_factor",       park_factor_feature, home)
    _run_block("pitcher_diffs",     pitcher_diffs,     hid, aid)  # handles None IDs internally
    _run_block("bullpen_era14",     cc_bullpen_era14, home, away)
    _run_block("bullpen_ip_last3",  cc_bullpen_ip_last3,  home, away, game_iso)
    _run_block("rest_b2b",          rest_b2b,          home, away, game_iso)
    _run_block("travel_km",         travel_km,         home, away, game_iso)
    _run_block("weather",           weather_block,     g, blocks.get("park_factor", {}).get("park_factor", 100.0), include_weather)
    _run_block("elo",               elo_feature,       home, away, game_iso)
    _run_block("odds",              odds_feature,      game_pk, include_odds)

    # Merge blocks into one row
    row: Dict[str, Any] = {
        "game_date": game_iso,
        "home_team": home,
        "away_team": away,
        "home_win": home_win,
    }
    for b in ("team_form", "pitcher_diffs", "bullpen_era14", "park_factor", "rest_b2b",
              "travel_km", "bullpen_ip_last3", "weather", "elo", "odds"):
        row.update(blocks.get(b, {}))

    # Final global coercion pass to bulletproof numbers
    for k, v in list(row.items()):
        if k in ("game_date", "home_team", "away_team", "home_win"):
            continue
        # flags/days can be ints; others as float
        try:
            if k.endswith("_flag") or k.endswith("_days_rest"):
                row[k] = int(v) if v is not None else 0
            else:
                row[k] = f(v, 0.0)
        except Exception:
            row[k] = 0.0

    # If there were block errors, print one compact line for debugging
    if errors:
        print(
            f"[feature_builder][WARN] Block errors for gamePk={game_pk} "
            f"({home} vs {away} on {game_iso}; status={status}): "
            + " | ".join(errors)
        )

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
    # If training (only_finals) but end is today/future, flip to prediction mode
    try:
        end_d = datetime.date.fromisoformat(end_date)
        if end_d >= datetime.date.today() and only_finals:
            print(f"[feature_builder] end_date={end_date} >= today; forcing only_finals=False")
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
        schedule = _safe_get_json(url)
        if not schedule:
            print(f"[feature_builder][ERROR] schedule fetch failed [{start_date}..{end_date}]")
            return pd.DataFrame(columns=(required_features or []) + META_COLS)
        cache.save_json("schedule", schedule, cache_key)
        games = [g for d in schedule.get("dates", []) for g in d.get("games", [])]

    if not games:
        print(f"[feature_builder][WARN] No games in range {start_date}..{end_date}")
        return pd.DataFrame(columns=(required_features or []) + META_COLS)

    # Seed probables cache from schedule (quietly)
    try:
        batch_seed = {}
        for g in games:
            pk = g.get("gamePk")
            if not pk:
                continue
            home_id = ((g.get("teams", {}).get("home", {}).get("probablePitcher", {}) or {}).get("id"))
            away_id = ((g.get("teams", {}).get("away", {}).get("probablePitcher", {}) or {}).get("id"))
            rec = {"home_id": home_id, "away_id": away_id} if (home_id or away_id) else {}
            per_key = cache._make_key("probables", pk)
            cache.save_json("probables", rec, per_key)
            batch_seed[pk] = rec
        if batch_seed:
            batch_key = cache._make_key("probables_batch", *sorted(batch_seed.keys()))
            cache.save_json("probables", batch_seed, batch_key)
    except Exception:
        pass

    # Warm common caches once
    _ = DF.cc_wpct_season()
    _ = DF.cc_wpct_last30()
    _ = DF.cc_bullpen_era14()
    _ = DF.cc_offense30()
    _ = DF.cc_probables([g.get("gamePk") for g in games if g.get("gamePk")])

    rows: List[Dict[str, Any]] = []
    skipped = 0

    for g in games:
        try:
            status = _status_of(g)
            if only_finals and status != "final":
                continue
            rows.append(build_features_for_game(g, include_odds=include_odds, include_weather=include_weather))
        except Exception as e:
            skipped += 1
            pk = g.get("gamePk")
            home = (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name", "")
            away = (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name", "")
            print(
                f"[feature_builder][ERROR] build row failed (gamePk={pk}, {home} vs {away}): {type(e).__name__}: {e}"
            )
            traceback.print_exc(limit=1)

    df = pd.DataFrame(rows)

    if skipped:
        print(f"[feature_builder][WARN] Skipped {skipped} game(s) due to errors; see logs above.")

    # Column alignment if caller requests a specific set
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
