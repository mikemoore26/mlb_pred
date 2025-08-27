# utils/data_fetchers/pitchers.py
from __future__ import annotations
import re
import datetime as _dt
from typing import Optional, Dict, Any, Tuple

import utils.cache as cache
from utils.safe_get_json import _safe_get_json

# ---------------- URL helpers ----------------
def _stats_url_season(pid: int, season: int) -> str:
    return (f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
            f"?stats=season&group=pitching&season={season}")

def _stats_url_by_date(pid: int, start: _dt.date, end: _dt.date) -> str:
    return (f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
            f"?stats=byDateRange&group=pitching&startDate={start.isoformat()}&endDate={end.isoformat()}")

def _stats_url_gamelog(pid: int, start: _dt.date, end: _dt.date) -> str:
    return (f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
            f"?stats=gameLog&group=pitching&startDate={start.isoformat()}&endDate={end.isoformat()}")

def _stats_url_yearbyyear(pid: int) -> str:
    return (f"https://statsapi.mlb.com/api/v1/people/{pid}"
            f"?hydrate=stats(group=[pitching],type=[yearByYear])")

# ---------------- innings parsing ----------------
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

# ---------------- aggregations ----------------
def _aggregate_from_splits(splits: list,
                           only_starts: Optional[bool],
                           limit_games: Optional[int]) -> Tuple[float, float, float, float, int, int]:
    """
    Aggregate IP, ER, K, BB (and counts) from gameLog splits.
    Returns: (IP, ER, K, BB, games_counted, starts_counted)
    """
    try:
        splits = sorted(splits, key=lambda s: s.get("date", ""), reverse=True)
    except Exception:
        splits = splits or []

    if limit_games is not None:
        splits = splits[:limit_games]

    IP = ER = K = BB = 0.0
    games = starts = 0

    for s in splits:
        st = (s.get("stat") or {})
        started = int(st.get("gamesStarted", 0) or 0) > 0
        if only_starts is True and not started:
            continue
        ip = _parse_ip_to_float(st.get("inningsPitched"))
        er = float(st.get("earnedRuns", 0) or 0)
        so = float(st.get("strikeouts", 0) or 0)
        bb = float(st.get("baseOnBalls", 0) or 0)
        if ip == 0 and er == 0 and so == 0 and bb == 0:
            continue
        IP += ip; ER += er; K += so; BB += bb
        games += 1
        if started:
            starts += 1

    return IP, ER, K, BB, games, starts

def _rates_from_totals(IP: float, ER: float, K: float, BB: float):
    ERA = 9.0 * ER / IP if IP > 0 else None
    K9  = 9.0 * K  / IP if IP > 0 else None
    BB9 = 9.0 * BB / IP if IP > 0 else None
    return ERA, K9, BB9

# ---------------- fallbacks ----------------
def _best_recent_season_pitching_line(pid: int) -> dict:
    """
    Most-recent pitched MLB season (via yearByYear). Returns:
    {'era','k9','bb9','ip','games','season'} (values may be None).
    """
    data = _safe_get_json(_stats_url_yearbyyear(pid)) or {}
    people = data.get("people") or []
    blocks = (people[0].get("stats") or []) if people else []
    yby = next((b for b in blocks if (b.get("type", {}) or {}).get("displayName") == "yearByYear"), None)
    splits = (yby or {}).get("splits") or []
    try:
        splits = sorted(splits, key=lambda s: int(s.get("season", 0)), reverse=True)
    except Exception:
        pass

    for s in splits:
        st = (s.get("stat") or {})
        games = int(st.get("gamesPlayed") or 0)
        if games <= 0:
            continue
        season = int(s.get("season") or 0)
        def _flt(v):
            try: return float(v)
            except Exception: return None
        era = _flt(st.get("era"))
        k9  = _flt(st.get("strikeoutsPer9Inn"))
        bb9 = _flt(st.get("walksPer9Inn"))
        ip  = _parse_ip_to_float(st.get("inningsPitched"))
        return {"era": era, "k9": k9, "bb9": bb9, "ip": ip, "games": games, "season": season}
    return {"era": None, "k9": None, "bb9": None, "ip": None, "games": 0, "season": None}

# ---------------- public API ----------------
def cc_pitcher_season(player_id: Optional[int]) -> Dict[str, Optional[float]]:
    """
    Current-season line; falls back to most recent season (yearByYear) if current is empty.
    Returns {'era','k9','bb9','ip','games','season', 'note'?}
    """
    if not player_id:
        return {"era": None, "k9": None, "bb9": None, "ip": None, "games": 0, "season": None}

    k = cache._make_key("p_season", player_id)
    hit = cache.load_json("p_season", k, max_age_days=7)
    if isinstance(hit, dict):
        return hit

    season = _dt.date.today().year
    data = _safe_get_json(_stats_url_season(int(player_id), season)) or {}

    era = k9 = bb9 = None
    ip = None
    games = 0

    try:
        blocks = data.get("stats", [])
        if blocks and blocks[0].get("splits"):
            st = (blocks[0]["splits"][0] or {}).get("stat", {}) or {}
            era = float(st["era"]) if st.get("era") not in (None, "") else None
            k9  = float(st["strikeoutsPer9Inn"]) if st.get("strikeoutsPer9Inn") not in (None, "") else None
            bb9 = float(st["walksPer9Inn"]) if st.get("walksPer9Inn") not in (None, "") else None
            ip  = _parse_ip_to_float(st.get("inningsPitched"))
            games = int(st.get("gamesPlayed", 0) or 0)
    except Exception:
        pass

    if ((ip is None or ip == 0) and (games == 0 or era is None)):
        yby = _best_recent_season_pitching_line(int(player_id))
        if (yby["games"] or 0) > 0:
            out = {
                "era": yby["era"], "k9": yby["k9"], "bb9": yby["bb9"],
                "ip": yby["ip"], "games": yby["games"], "season": yby["season"],
                "note": "yearByYear fallback (no current-season data yet)",
            }
            cache.save_json("p_season", out, k)
            return out

    out = {"era": era, "k9": k9, "bb9": bb9, "ip": ip, "games": games, "season": season}
    cache.save_json("p_season", out, k)
    return out


def cc_pitch_lastx(
    player_id: Optional[int],
    days: int,
    *,
    only_starts: Optional[bool] = None,
    limit_games: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """
    Rolling-window pitching over `days`.
    - Try byDateRange
    - Fallback to gameLog aggregation over `days`
    - Widen to 45 if still empty
    - Final fallback: most-recent season via yearByYear
    Returns:
      {'era','k9','bb9','ip','games','starts','window_days', 'season_fallback'?}
    """
    if not player_id:
        return {"era": None, "k9": None, "bb9": None, "ip": None,
                "games": 0, "starts": 0, "window_days": int(days)}

    k = cache._make_key("p_lastx", player_id, days, only_starts, limit_games)
    hit = cache.load_json("p_lastx", k, max_age_days=1)
    if isinstance(hit, dict):
        return hit

    end_d = _dt.date.today()
    start_d = end_d - _dt.timedelta(days=int(days))

    # 1) byDateRange
    era = k9 = bb9 = None
    ip = None
    games = starts = 0
    try:
        data = _safe_get_json(_stats_url_by_date(int(player_id), start_d, end_d)) or {}
        blocks = data.get("stats", [])
        if blocks and blocks[0].get("splits"):
            st = (blocks[0]["splits"][0] or {}).get("stat", {}) or {}
            ip = _parse_ip_to_float(st.get("inningsPitched"))
            er = float(st.get("earnedRuns", 0) or 0)
            so = float(st.get("strikeouts", 0) or 0)
            bb = float(st.get("baseOnBalls", 0) or 0)
            games = int(st.get("gamesPlayed", 0) or 0)
            starts = int(st.get("gamesStarted", 0) or 0)
            era, k9, bb9 = _rates_from_totals(ip, er, so, bb)
    except Exception:
        pass

    # 2) gameLog aggregation helper
    def _agg(window_days: int):
        end = _dt.date.today()
        start = end - _dt.timedelta(days=window_days)
        gl = _safe_get_json(_stats_url_gamelog(int(player_id), start, end)) or {}
        splits = (gl.get("stats", []) or [{}])[0].get("splits", []) or []
        IP, ER, K, BB, G, GS = _aggregate_from_splits(splits, only_starts, limit_games)
        ERA, K9, BB9 = _rates_from_totals(IP, ER, K, BB)
        return ERA, K9, BB9, IP, G, GS

    if (ip is None or ip == 0) and games == 0:
        era, k9, bb9, ip, games, starts = _agg(days)

    if (ip is None or ip == 0) and games == 0:
        era, k9, bb9, ip, games, starts = _agg(45)

    season_fallback = None
    if (ip is None or ip == 0) and games == 0 and era is None:
        yby = _best_recent_season_pitching_line(int(player_id))
        if (yby["games"] or 0) > 0:
            era = yby["era"]; k9 = yby["k9"]; bb9 = yby["bb9"]
            ip = yby["ip"]; games = yby["games"]; starts = None
            season_fallback = yby["season"]

    out = {
        "era": (float(era) if era is not None else None),
        "k9": (float(k9) if k9 is not None else None),
        "bb9": (float(bb9) if bb9 is not None else None),
        "ip": (float(ip) if ip is not None else None),
        "games": int(games or 0),
        "starts": int(starts or 0) if starts is not None else 0,
        "window_days": int(days),
    }
    if season_fallback is not None:
        out["season_fallback"] = int(season_fallback)

    cache.save_json("p_lastx", out, k)
    return out


# --------- Back-compat wrappers ---------
def cc_pitcher_last30(player_id: Optional[int]) -> Dict[str, Optional[float]]:
    res = cc_pitch_lastx(player_id, 30, only_starts=None, limit_games=None)
    return {"era30": res["era"]}

def cc_pitcher_last3(player_id: Optional[int]) -> Dict[str, Optional[float]]:
    # Use last 60 days, only starts, limit to 3 starts
    res = cc_pitch_lastx(player_id, 60, only_starts=True, limit_games=3)
    # k/bb ratio from rates if available; graceful None handling
    kbb = None
    if res["k9"] is not None and res["bb9"] is not None:
        if res["bb9"] > 0:
            kbb = res["k9"] / res["bb9"]
        elif res["k9"] > 0:
            kbb = res["k9"]
    return {"era3": res["era"], "kbb3": (float(kbb) if kbb is not None else None)}


# ---------------- Tests ----------------
if __name__ == "__main__":
    COLE = 543037
    print("[Pitchers] Gerrit Cole (543037)")
    print("Season:", cc_pitcher_season(COLE))
    print("Last30 (generic):", cc_pitch_lastx(COLE, 30))
    print("Last3 (wrapper):", cc_pitcher_last3(COLE))
