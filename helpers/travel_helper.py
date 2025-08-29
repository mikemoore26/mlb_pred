# helpers/travel_helper.py
from __future__ import annotations
from typing import Optional, Tuple, Dict
import os, json, pathlib

import utils.cache as cache
from utils.safe_get_json import _safe_get_json
from utils.data_fetchers.park_venue import cc_venue_latlon_from_gamepk

DEBUG_TRAVEL = True

# ---------- paths & override loading ----------

_THIS = pathlib.Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]  # adjust if your layout differs

_NAME_REL = "data/static/venue_latlon_overrides.json"
_ID_REL   = "data/static/venue_id_latlon_overrides.json"

def _candidate_paths(rel_path: str):
    # 1) env var can point to a file path (absolute or relative)
    env_val = os.getenv(rel_path)  # optional: set env var = full path
    if env_val:
        yield pathlib.Path(env_val).expanduser().resolve()
    # 2) project root default
    yield (_PROJECT_ROOT / rel_path).resolve()
    # 3) cwd fallback
    yield pathlib.Path(rel_path).resolve()

def _load_json(kind: str, rel_path: str) -> Dict[str, object]:
    for p in _candidate_paths(rel_path):
        try:
            if p.exists():
                if DEBUG_TRAVEL:
                    print(f"[travel] {kind} overrides FOUND at: {p}")
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            else:
                if DEBUG_TRAVEL:
                    print(f"[travel] {kind} overrides not found at: {p}")
        except Exception as e:
            if DEBUG_TRAVEL:
                print(f"[travel] {kind} overrides load error at {p}: {e}")
    if DEBUG_TRAVEL:
        print(f"[travel] {kind} overrides ultimately not found. Using defaults.")
    return {}

_OVERRIDES_NAME: Dict[str, object] = _load_json("name", _NAME_REL)
_OVERRIDES_ID: Dict[str, object]   = _load_json("id",   _ID_REL)

# Basic built-ins to avoid total failure if files missing
if not _OVERRIDES_ID:
    _OVERRIDES_ID.update({
        "1":    [33.8003, -117.8827],   # Angel Stadium
        "3313": [40.8296,  -73.9262],   # Yankee Stadium
        "2680": [32.7073, -117.1573],   # Petco Park
    })

def _canon(s: str) -> str:
    return ''.join(ch for ch in (s or '').lower() if ch.isalnum())

# normalized index for name overrides (tolerant to spacing/punct)
_OVERRIDES_NAME_NORM: Dict[str, object] = {}
for k, v in _OVERRIDES_NAME.items():
    try:
        _OVERRIDES_NAME_NORM[_canon(k)] = v
    except Exception:
        pass

# ---------- lat/lon extract helpers ----------

def _coerce_latlon(d: object) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(d, dict):
        return None, None
    for keypair in (("latitude", "longitude"), ("lat", "lon"), ("lat", "lng")):
        lat = d.get(keypair[0])
        lon = d.get(keypair[1])
        if lat is not None and lon is not None:
            try:
                return float(lat), float(lon)
            except Exception:
                return None, None
    return None, None

def _overrides_lookup_by_id(venue_id: Optional[int]) -> Tuple[Optional[float], Optional[float]]:
    if not venue_id:
        return None, None
    rec = _OVERRIDES_ID.get(str(int(venue_id)))
    if isinstance(rec, (list, tuple)) and len(rec) >= 2:
        try:
            return float(rec[0]), float(rec[1])
        except Exception:
            return None, None
    if isinstance(rec, dict):
        coords = rec.get("coords")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            try:
                return float(coords[0]), float(coords[1])
            except Exception:
                return None, None
    return None, None

def _overrides_lookup_by_name(venue_name: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not venue_name:
        return None, None
    rec = _OVERRIDES_NAME.get(str(venue_name))
    if rec is None:
        rec = _OVERRIDES_NAME_NORM.get(_canon(venue_name))
    if isinstance(rec, (list, tuple)) and len(rec) >= 2:
        try:
            return float(rec[0]), float(rec[1])
        except Exception:
            return None, None
    if isinstance(rec, dict):
        coords = rec.get("coords")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            try:
                return float(coords[0]), float(coords[1])
            except Exception:
                return None, None
    return None, None

# ---------- venue lookups via MLB API ----------

def _venue_latlon_from_venue_id(venue_id: int) -> Tuple[Optional[float], Optional[float]]:
    ck = cache._make_key("venue_coord", f"venue_{int(venue_id)}")
    hit = cache.load_json("venue_coord", ck, max_age_days=365)
    if isinstance(hit, dict):
        lat = hit.get("lat"); lon = hit.get("lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon)

    for url in (
        f"https://statsapi.mlb.com/api/v1/venues?venueIds={int(venue_id)}",
        f"https://statsapi.mlb.com/api/v1/venues/{int(venue_id)}",
    ):
        data = _safe_get_json(url) or {}
        venues = data.get("venues") or []
        if not venues:
            continue
        v = venues[0] or {}
        lat, lon = _coerce_latlon(v.get("location") or {})
        if lat is None or lon is None:
            lat, lon = _coerce_latlon(v.get("coordinates") or {})
        if lat is not None and lon is not None:
            out = {"lat": float(lat), "lon": float(lon)}
            cache.save_json("venue_coord", out, ck)
            return float(lat), float(lon)

    return None, None

def _home_team_id_from_feed(game_pk: int) -> Optional[int]:
    try:
        feed = _safe_get_json(f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live") or {}
        return int(((feed.get("gameData") or {}).get("teams") or {}).get("home", {}).get("id") or 0) or None
    except Exception:
        return None

def _venue_latlon_from_team(team_id: int) -> Tuple[Optional[float], Optional[float]]:
    ck = cache._make_key("venue_coord", f"team_{int(team_id)}")
    hit = cache.load_json("venue_coord", ck, max_age_days=365)
    if isinstance(hit, dict):
        lat = hit.get("lat"); lon = hit.get("lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon)

    url = f"https://statsapi.mlb.com/api/v1/teams/{int(team_id)}?hydrate=venue"
    data = _safe_get_json(url) or {}
    teams = data.get("teams") or []
    if teams:
        venue = teams[0].get("venue") or {}
        lat, lon = _coerce_latlon(venue.get("location") or {})
        if lat is None or lon is None:
            lat, lon = _coerce_latlon(venue.get("coordinates") or {})
        if lat is not None and lon is not None:
            out = {"lat": float(lat), "lon": float(lon)}
            cache.save_json("venue_coord", out, ck)
            return float(lat), float(lon)
    return None, None

# ---------- main resolver ----------

def _resolve_today_venue_latlon(game_pk: int, venue_id: Optional[int]) -> Tuple[Optional[float], Optional[float]]:
    """
    Order:
      1) cc_venue_latlon_from_gamepk(game_pk)
      2) live feed: gameData.venue.location / coordinates (+ capture venue name/id)
      3) venues endpoint by venue_id
      4) overrides by ID
      5) team endpoint (hydrate=venue) using home team id
      6) overrides by NAME
    """
    # 1) cached helper
    lat, lon = cc_venue_latlon_from_gamepk(game_pk)
    if lat is not None and lon is not None:
        return float(lat), float(lon)

    venue_name: Optional[str] = None

    # 2) live feed
    try:
        feed = _safe_get_json(f"https://statsapi.mlb.com/api/v1.1/game/{int(game_pk)}/feed/live") or {}
        v = (feed.get("gameData") or {}).get("venue") or {}
        venue_name = v.get("name")
        if DEBUG_TRAVEL:
            print(f"[travel] live feed venue → name={venue_name!r} id={v.get('id')!r}")
        lat, lon = _coerce_latlon(v.get("location") or {})
        if lat is None or lon is None:
            lat, lon = _coerce_latlon(v.get("coordinates") or {})
        if lat is not None and lon is not None:
            return float(lat), float(lon)
        if venue_id is None:
            try:
                venue_id = int(v.get("id") or 0) or None
            except Exception:
                venue_id = None
    except Exception:
        pass

    # 3) venues endpoint
    if venue_id:
        vlat, vlon = _venue_latlon_from_venue_id(int(venue_id))
        if vlat is not None and vlon is not None:
            return float(vlat), float(vlon)

    # 4) overrides by ID
    ilat, ilon = _overrides_lookup_by_id(venue_id)
    if ilat is not None and ilon is not None:
        if DEBUG_TRAVEL:
            print(f"[travel] using ID override for venue_id={venue_id}")
        return ilat, ilon

    # 5) team endpoint
    home_tid = _home_team_id_from_feed(game_pk)
    if home_tid:
        tlat, tlon = _venue_latlon_from_team(int(home_tid))
        if tlat is not None and tlon is not None:
            return float(tlat), float(tlon)

    # 6) overrides by NAME
    nlat, nlon = _overrides_lookup_by_name(venue_name)
    if nlat is not None and nlon is not None:
        if DEBUG_TRAVEL:
            print(f"[travel] using NAME override for venue_name={venue_name!r}")
        return nlat, nlon

    if DEBUG_TRAVEL:
        print("[travel] no venue lat/lon found after all sources")
    return None, None
