from __future__ import annotations
import datetime as _dt
from typing import Dict
import utils.cache as cache
from utils.safe_get_json import _safe_get_json

def cc_wpct_season() -> Dict[str,float]:
    k = cache._make_key("wpct_season","cur")
    hit = cache.load_json("wpct_season", k, max_age_days=1)
    if hit is not None: return hit
    year = _dt.date.today().year
    url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={year}"
    data = _safe_get_json(url) or {}
    out = {}
    for rec in data.get("records",[]):
        for t in rec.get("teamRecords",[]):
            name = t["team"]["name"]; w = t.get("wins",0) or 0; l = t.get("losses",0) or 0
            out[name] = (w/(w+l)) if (w+l)>0 else 0.5
    cache.save_json("wpct_season", out, k)
    return out

def cc_wpct_last30() -> Dict[str,float]:
    k = cache._make_key("wpct_last30","global")
    hit = cache.load_json("wpct_last30", k, max_age_days=1)
    if hit is not None: return hit
    end_d = _dt.date.today(); start_d = end_d - _dt.timedelta(days=30)
    data = _safe_get_json(
        f"https://statsapi.mlb.com/api/v1/schedule?startDate={start_d}&endDate={end_d}&sportId=1"
    ) or {}
    agg = {}
    for d in data.get("dates",[]):
        for g in d.get("games",[]):
            if (g.get("status",{}).get("detailedState") or "").lower() != "final": continue
            home = g["teams"]["home"]["team"]["name"]
            away = g["teams"]["away"]["team"]["name"]
            hs = g["teams"]["home"].get("score",0) or 0
            as_ = g["teams"]["away"].get("score",0) or 0
            win = home if hs>as_ else away
            for t in (home,away):
                agg.setdefault(t, {"wins":0,"games":0})
                agg[t]["games"] += 1
                if t==win: agg[t]["wins"] += 1
    out = {t: (d["wins"]/d["games"] if d["games"]>0 else 0.5) for t,d in agg.items()}
    cache.save_json("wpct_last30", out, k)
    return out

def cc_offense30() -> Dict[str,float]:
    k = cache._make_key("offense30","global")
    hit = cache.load_json("offense30", k, max_age_days=1)
    if hit is not None:
        try: return {t: float(v) for t,v in hit.items()}
        except Exception: return hit
    end = _dt.date.today(); start = end - _dt.timedelta(days=30)
    data = _safe_get_json(
        f"https://statsapi.mlb.com/api/v1/schedule?startDate={start}&endDate={end}&sportId=1"
    ) or {}
    agg = {}
    for d in data.get("dates",[]):
        for g in d.get("games",[]):
            if (g.get("status",{}).get("detailedState") or "").lower() != "final": continue
            home = g["teams"]["home"]["team"]["name"]; away = g["teams"]["away"]["team"]["name"]
            hs = float(g["teams"]["home"].get("score",0) or 0)
            as_ = float(g["teams"]["away"].get("score",0) or 0)
            agg.setdefault(home, {"runs":0.0,"games":0.0}); agg[home]["runs"] += hs; agg[home]["games"] += 1
            agg.setdefault(away, {"runs":0.0,"games":0.0}); agg[away]["runs"] += as_; agg[away]["games"] += 1
    out = {t: (d["runs"]/d["games"] if d["games"]>0 else 0.0) for t,d in agg.items()}
    cache.save_json("offense30", out, k)
    return out
