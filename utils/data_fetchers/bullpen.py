# utils/data_fetchers/bullpen.py
from __future__ import annotations
from typing import Optional, Dict, List, Tuple, Any
import datetime as _dt

import utils.cache as cache
from utils.safe_get_json import _safe_get_json
from utils.data_fetchers._common import parse_ip_to_float as _parse_ip

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _team_from_live(game_pk: int, side: str) -> Optional[str]:
    """
    From /game/<pk>/feed/live get the official team name for 'home'/'away'.
    """
    data = _safe_get_json(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/feed/live") or {}
    return (((data.get("gameData") or {}).get("teams") or {}).get(side) or {}).get("name")

def _sum_relief_ip_er_from_boxscore(game_pk: int, side: str) -> Tuple[float, float]:
    """
    Sum relief IP and ER for the given team side ('home'/'away') in a game.
    Relief = appearance with gamesStarted == 0.
    """
    data = _safe_get_json(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore") or {}
    tm = (data.get("teams", {}) or {}).get(side, {}) or {}
    players = tm.get("players", {}) or {}

    ip_total = 0.0
    er_total = 0.0
    for _, p in players.items():
        st = (p.get("stats", {}) or {}).get("pitching", {}) or {}
        ip = _parse_ip(st.get("inningsPitched"))
        if ip <= 0:
            continue
        gs = int(st.get("gamesStarted", 0) or 0)
        if gs == 0:  # relief
            ip_total += ip
            er_total += float(st.get("earnedRuns", 0) or 0)

    return (float(ip_total), float(er_total))

def _games_for_team_in_window(team_name: str, start_date: str, end_date: str) -> List[int]:
    """
    Return all gamePks where team played between start_date..end_date (inclusive).
    """
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?startDate={start_date}&endDate={end_date}&sportId=1")
    sched = _safe_get_json(url) or {}
    pks: List[int] = []
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            ht = (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name", "")
            at = (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name", "")
            if team_name in (ht, at):
                pk = g.get("gamePk")
                if pk:
                    pks.append(int(pk))
    return pks

# -------------------------------------------------------------------
# Public: Bullpen IP last 3 days (for fatigue)
# -------------------------------------------------------------------

def get_bullpen_ip_last3(team: str, game_date: str) -> float:
    """
    Sum relief IP during the 3 days BEFORE 'game_date' for 'team' (home/away games).
    """
    try:
        gd = _dt.date.fromisoformat(game_date)
    except Exception:
        return 0.0
    start = (gd - _dt.timedelta(days=3)).isoformat()
    end   = (gd - _dt.timedelta(days=1)).isoformat()

    pks = _games_for_team_in_window(team, start, end)
    if not pks:
        return 0.0

    total = 0.0
    for pk in pks:
        # Determine side in that game
        home_name = _team_from_live(pk, "home") or ""
        side = "home" if home_name == team else "away"
        ip, _ = _sum_relief_ip_er_from_boxscore(pk, side)
        total += ip
    return float(total)

def cc_bullpen_ip_last3(team: str, game_date: str) -> float:
    ck = cache._make_key("bullpen_ip_last3", team, game_date)
    hit = cache.load_json("bullpen_ip_last3", ck, max_age_days=1)
    if hit is not None:
        try:
            return float(hit)
        except Exception:
            return 0.0
    out = get_bullpen_ip_last3(team, game_date)
    cache.save_json("bullpen_ip_last3", float(out), ck)
    return float(out)

# -------------------------------------------------------------------
# Public: Real bullpen ERA over last 14 days
# -------------------------------------------------------------------

def cc_bullpen_era14() -> Dict[str, float]:
    """
    Compute bullpen ERA for each team over the last 14 days (FINAL games only),
    by summing relief ER/IP from boxscores. Cached for 1 day.
    If a team has 0 relief IP in that window, returns 4.00 neutral.
    """
    ck = cache._make_key("bp14", "global_real")
    hit = cache.load_json("bp14", ck, max_age_days=1)
    if isinstance(hit, dict):
        try:
            return {t: float(v) for t, v in hit.items()}
        except Exception:
            return hit

    end_d = _dt.date.today()
    start_d = end_d - _dt.timedelta(days=14)
    sched_url = (f"https://statsapi.mlb.com/api/v1/schedule"
                 f"?startDate={start_d.isoformat()}&endDate={end_d.isoformat()}&sportId=1")
    sched = _safe_get_json(sched_url) or {}

    # team -> (IP, ER)
    agg: Dict[str, Tuple[float, float]] = {}

    for d in sched.get("dates", []):
        for g in d.get("games", []):
            if (g.get("status", {}).get("detailedState") or "").lower() != "final":
                continue
            pk = g.get("gamePk")
            if not pk:
                continue

            # HOME
            home = (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name")
            if home:
                ip, er = _sum_relief_ip_er_from_boxscore(pk, "home")
                ip0, er0 = agg.get(home, (0.0, 0.0))
                agg[home] = (ip0 + ip, er0 + er)

            # AWAY
            away = (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name")
            if away:
                ip, er = _sum_relief_ip_er_from_boxscore(pk, "away")
                ip0, er0 = agg.get(away, (0.0, 0.0))
                agg[away] = (ip0 + ip, er0 + er)

    out: Dict[str, float] = {}
    for team, (ip, er) in agg.items():
        if ip and ip > 0:
            out[team] = float(9.0 * er / ip)
        else:
            out[team] = 4.00

    cache.save_json("bp14", out, ck)
    return out

# -------------------------------------------------------------------
# Smoke test
# -------------------------------------------------------------------

if __name__ == "__main__":
    today = _dt.date.today().isoformat()
    team = "New York Yankees"
    print(f"[Bullpen] IP last3 for {team} before {today}:", cc_bullpen_ip_last3(team, today))
    sample = list(cc_bullpen_era14().items())[:5]
    print("[Bullpen] ERA14 sample:", sample)
