from __future__ import annotations
import os, json
from typing import Optional, Tuple
from utils.safe_get_json import _safe_get_json
import utils.cache as cache

_THIS_DIR = os.path.dirname(__file__)
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_PARK_FACTORS_JSON = os.path.join(_PROJ_ROOT, "data", "park_factors_fallback.json")
_BALLPARK_COORDS_JSON = os.path.join(_PROJ_ROOT, "data", "ballpark_coords.json")

_PARK_FACTORS = None
_BALLPARK_COORDS = None

def _load_park_factors() -> dict:
    global _PARK_FACTORS
    if _PARK_FACTORS is None:
        try:
            with open(_PARK_FACTORS_JSON,"r",encoding="utf-8") as f:
                _PARK_FACTORS = json.load(f)
        except Exception:
            _PARK_FACTORS = {"aliases":{}, "park_factor":{}}
    return _PARK_FACTORS

def _load_ballpark_coords() -> dict:
    global _BALLPARK_COORDS
    if _BALLPARK_COORDS is None:
        try:
            with open(_BALLPARK_COORDS_JSON,"r",encoding="utf-8") as f:
                _BALLPARK_COORDS = json.load(f)
        except Exception:
            _BALLPARK_COORDS = {}
    return _BALLPARK_COORDS

def norm_team_name(name: str) -> str:
    if not name: return name
    manual = {
        "yankees":"New York Yankees","red sox":"Boston Red Sox","bos":"Boston Red Sox",
        "nyy":"New York Yankees","mets":"New York Mets","dodgers":"Los Angeles Dodgers",
        "giants":"San Francisco Giants","cards":"St. Louis Cardinals",
        "dbacks":"Arizona Diamondbacks","d-backs":"Arizona Diamondbacks",
        "cubs":"Chicago Cubs","white sox":"Chicago White Sox",
    }
    m = {k.lower():v for k,v in manual.items()}
    pf = _load_park_factors()
    aliases = pf.get("aliases",{})
    for k,v in aliases.items():
        if isinstance(k,str) and isinstance(v,str):
            if not any(tok in v.lower() for tok in ("park","field","stadium")):
                m[k.lower()] = v
    return m.get(name.lower().strip(), name.strip())

def cc_park_factor(team: str) -> float:
    pf = _load_park_factors()
    aliases = pf.get("aliases",{})
    park_map = pf.get("park_factor",{})
    key = (team or "").strip()
    stadium = aliases.get(key, key)
    val = park_map.get(stadium)
    if val is None:
        for k,v in aliases.items():
            if k.lower() == key.lower():
                val = park_map.get(v); break
    try:
        return float(val) if val is not None else 100.0
    except Exception:
        return 100.0

def coords_from_venue_name(venue_name: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not venue_name: return (None,None)
    row = _load_ballpark_coords().get(venue_name)
    if not row or len(row)<2: return (None,None)
    try: return float(row[0]), float(row[1])
    except Exception: return (None,None)

def cc_venue_latlon_from_gamepk(game_pk: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (lat, lon) for a game's venue. Uses cache, then:
      - live feed: /api/v1.1/game/{gamePk}/feed/live → gameData.venue.location
      - venues endpoint if needed
    """
    ck = cache._make_key("venue_coord", game_pk)
    hit = cache.load_json("venue_coord", ck, max_age_days=365)
    if isinstance(hit, dict):
        return hit.get("lat"), hit.get("lon")

    lat = lon = None
    try:
        feed = _safe_get_json(f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live") or {}
        venue = (feed.get("gameData") or {}).get("venue") or {}
        loc = venue.get("location") or {}
        lat = loc.get("latitude"); lon = loc.get("longitude")
        if lat is None or lon is None:
            vid = venue.get("id")
            if vid:
                # venue endpoint fallback
                vdat = _safe_get_json(f"https://statsapi.mlb.com/api/v1/venues/{int(vid)}") or {}
                vlist = vdat.get("venues") or []
                if vlist:
                    vloc = (vlist[0].get("location") or {})
                    lat = lat or vloc.get("latitude")
                    lon = lon or vloc.get("longitude")
    except Exception:
        pass

    out = {"lat": (float(lat) if lat is not None else None),
           "lon": (float(lon) if lon is not None else None)}
    cache.save_json("venue_coord", out, ck)
    return out["lat"], out["lon"]

# utils/data_fetchers/park_venue.py  (append near bottom)


def _venue_latlon_from_venue_id(venue_id: int) -> Tuple[Optional[float], Optional[float]]:
    """
    Look up (lat, lon) from the MLB venues endpoint. Cached ~1 year.
    """
    ck = cache._make_key("venue_coord", f"venue_{int(venue_id)}")
    hit = cache.load_json("venue_coord", ck, max_age_days=365)
    if isinstance(hit, dict):
        return hit.get("lat"), hit.get("lon")

    url = f"https://statsapi.mlb.com/api/v1/venues/{int(venue_id)}"
    data = _safe_get_json(url) or {}
    venues = data.get("venues") or []
    if venues:
        loc = (venues[0].get("location") or {})
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        out = {"lat": (float(lat) if lat is not None else None),
               "lon": (float(lon) if lon is not None else None)}
        cache.save_json("venue_coord", out, ck)
        return out["lat"], out["lon"]

    return None, None
