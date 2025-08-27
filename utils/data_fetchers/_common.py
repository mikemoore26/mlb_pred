from __future__ import annotations
import os, json, re, datetime as _dt
from typing import Optional, Any, Tuple, Dict
import utils.cache as cache
from utils.safe_get_json import _safe_get_json

# innings parser (MLB tenths)
_IP_RE = re.compile(r"^\s*(\d+)(?:\.(\d))?\s*$")

def parse_ip(ip_val: Any) -> float:
    if ip_val is None:
        return 0.0
    try:
        if isinstance(ip_val, (int, float)):
            whole = int(ip_val)
            tenths = int(round((float(ip_val) - whole) * 10))
            tenths = tenths if tenths in (0,1,2) else 0
            return whole + tenths/3.0
        s = str(ip_val).strip()
        m = _IP_RE.match(s)
        if not m:
            return 0.0
        whole = int(m.group(1))
        tenths = int(m.group(2) or 0)
        tenths = tenths if tenths in (0,1,2) else 0
        return whole + tenths/3.0
    except Exception:
        return 0.0

def parse_iso_dt(s) -> Optional[_dt.datetime]:
    try:
        return _dt.datetime.fromisoformat(str(s).replace("Z","+00:00"))
    except Exception:
        try:
            return _dt.datetime.fromisoformat(str(s))
        except Exception:
            return None

def call_external(method: str, *args, default=None):
    # Keep your mockable Elo/Odds hook
    try:
        from utils.safe_get_json import _safe_get_json as gj
        if method == "get_elo_diff":
            home, away, game_date = args
            url = f"https://api.example.com/elo?home={home}&away={away}&date={game_date.isoformat()}"
            data = gj(url) or {}
            return data.get("elo_diff", default)
        if method == "get_closing_odds_implied_home":
            (game_pk,) = args
            url = f"https://api.example.com/odds?gamePk={game_pk}"
            data = gj(url) or {}
            return data.get("implied_home_odds", default)
        return default
    except Exception:
        return default
