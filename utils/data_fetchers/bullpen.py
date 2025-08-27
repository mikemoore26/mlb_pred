from __future__ import annotations
import datetime as _dt
from typing import List, Dict
import utils.cache as cache
from utils.safe_get_json import _safe_get_json
from .park_venue import norm_team_name
from ._common import parse_ip

import datetime as _dt
from typing import Dict, Optional, Tuple

from utils.safe_get_json import _safe_get_json
import utils.cache as cache


def _games_for_team(team: str, start: str, end: str) -> List[int]:
    data = _safe_get_json(
        f"https://statsapi.mlb.com/api/v1/schedule?startDate={start}&endDate={end}&sportId=1"
    ) or {}
    tnorm = norm_team_name(team)
    pks=[]
    for d in data.get("dates",[]):
        for g in d.get("games",[]):
            home = g["teams"]["home"]["team"]["name"]; away = g["teams"]["away"]["team"]["name"]
            if tnorm in (home, away): pks.append(int(g.get("gamePk")))
    return pks

def _relief_ip_from_boxscore(game_pk: int, side: str) -> float:
    box = _safe_get_json(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore") or {}
    tm = (box.get("teams",{}) or {}).get(side,{}) or {}
    players = tm.get("players",{}) or {}
    total=0.0
    for _,p in players.items():
        st = (p.get("stats",{}) or {}).get("pitching",{}) or {}
        gs = int(st.get("gamesStarted",0) or 0)
        ip = parse_ip(st.get("inningsPitched"))
        if ip>0 and gs==0: total += ip
    return total

def get_bullpen_ip_last3(team: str, game_date: str) -> float:
    try: gd = _dt.date.fromisoformat(game_date)
    except Exception: return 0.0
    start = (gd - _dt.timedelta(days=3)).isoformat()
    end   = (gd - _dt.timedelta(days=1)).isoformat()
    pks = _games_for_team(team, start, end)
    if not pks: return 0.0
    total=0.0
    for pk in pks:
        live = _safe_get_json(f"https://statsapi.mlb.com/api/v1/game/{pk}/feed/live") or {}
        home = (((live.get("gameData") or {}).get("teams") or {}).get("home") or {}).get("name","")
        away = (((live.get("gameData") or {}).get("teams") or {}).get("away") or {}).get("name","")
        side = "home" if norm_team_name(team)==norm_team_name(home) else "away"
        total += _relief_ip_from_boxscore(pk, side)
    return float(total)

def cc_bullpen_ip_last3(team: str, game_date: str) -> float:
    k = cache._make_key("bullpen_ip_last3", team, game_date)
    hit = cache.load_json("bullpen_ip_last3", k, max_age_days=1)
    if hit is not None:
        try: return float(hit)
        except Exception: return 0.0
    val = get_bullpen_ip_last3(team, game_date)
    out = float(val) if val is not None else 0.0
    cache.save_json("bullpen_ip_last3", out, k)
    return out

# If this helper isn't already in the same module, import it from where you defined it.
# from utils.data_fetchers import _parse_ip_to_float  # if exposed
# Otherwise paste a local version:
def _parse_ip_to_float(ip_str) -> float:
    try:
        s = str(ip_str).strip()
        if not s:
            return 0.0
        if "." in s:
            whole, tenths = s.split(".", 1)
            whole = int(whole or 0)
            tenths = int(tenths or 0)
            if tenths not in (0, 1, 2):
                tenths = 0
            return whole + tenths / 3.0
        return float(s)
    except Exception:
        return 0.0


def _sum_relief_er_ip_from_boxscore(game_pk: int, team_side: str) -> Tuple[float, float]:
    """
    Return (ER, IP) for relievers (non-starters) on the given team_side ('home'|'away')
    using the boxscore endpoint for a single game.
    """
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    data = _safe_get_json(url) or {}
    tm = (data.get("teams", {}) or {}).get(team_side, {}) or {}
    players = tm.get("players", {}) or {}

    ER = 0.0
    IP = 0.0
    for _, p in players.items():
        stat = (p.get("stats", {}) or {}).get("pitching", {}) or {}
        ip = _parse_ip_to_float(stat.get("inningsPitched"))
        if ip <= 0:
            continue
        gs = int(stat.get("gamesStarted", 0) or 0)
        if gs == 0:  # relief
            IP += ip
            ER += float(stat.get("earnedRuns", 0) or 0)
    return ER, IP


def cc_bullpen_era14() -> Dict[str, float]:
    """
    Compute bullpen ERA for each MLB team over the last 14 days (FINAL games only).
    Uses boxscores to sum reliever ER/IP. Caches the full map for 1 day.
    """
    k = cache._make_key("bp14", "global")
    hit = cache.load_json("bp14", k, max_age_days=1)
    if isinstance(hit, dict):
        # ensure floats
        try:
            return {t: (float(v) if v is not None else 4.00) for t, v in hit.items()}
        except Exception:
            return hit

    end = _dt.date.today()
    start = end - _dt.timedelta(days=14)
    sched_url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?startDate={start.isoformat()}&endDate={end.isoformat()}&sportId=1"
    )
    sched = _safe_get_json(sched_url) or {}

    # Accumulators per team
    # team_totals[team_name] = {"ER": ..., "IP": ...}
    team_totals: Dict[str, Dict[str, float]] = {}

    for drec in sched.get("dates", []):
        for g in drec.get("games", []):
            status = (g.get("status", {}).get("detailedState") or "").lower()
            if status != "final":
                continue

            game_pk = g.get("gamePk")
            if not game_pk:
                continue

            home = (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name")
            away = (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name")
            if not home or not away:
                continue

            # Home relievers
            ER_h, IP_h = _sum_relief_er_ip_from_boxscore(game_pk, "home")
            if home:
                agg = team_totals.setdefault(home, {"ER": 0.0, "IP": 0.0})
                agg["ER"] += ER_h
                agg["IP"] += IP_h

            # Away relievers
            ER_a, IP_a = _sum_relief_er_ip_from_boxscore(game_pk, "away")
            if away:
                agg = team_totals.setdefault(away, {"ER": 0.0, "IP": 0.0})
                agg["ER"] += ER_a
                agg["IP"] += IP_a

    # Convert to ERA
    out: Dict[str, float] = {}
    for team, totals in team_totals.items():
        ER = float(totals.get("ER") or 0.0)
        IP = float(totals.get("IP") or 0.0)
        if IP > 0:
            era = 9.0 * ER / IP
            out[team] = float(era)
        else:
            # No relief IP in window → neutral-ish default (you can choose None instead)
            out[team] = 4.00

    cache.save_json("bp14", out, k)
    return out

