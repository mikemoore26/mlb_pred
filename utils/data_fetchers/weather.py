# utils/data_fetchers/weather.py
from __future__ import annotations
import os
import math
import datetime as _dt
from typing import Optional, Dict, Any, Tuple

import utils.cache as cache
from utils.safe_get_json import _safe_get_json
from .park_venue import cc_venue_latlon_from_gamepk

from dotenv import load_dotenv

# Load .env once when this module is imported
load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "").strip()

def _game_iso_date(g: dict) -> str:
    # Try gameDate first; fall back to officialDate; else today
    for k in ("gameDate", "officialDate", "date"):
        v = g.get(k)
        if v:
            try:
                # Accept YYYY-mm-dd or full ISO
                return str(v)[:10]
            except Exception:
                pass
    return _dt.date.today().isoformat()

def _target_hour_local_default() -> int:
    # Reasonable default MLB first pitch if we can't map to local tz/hour
    return 19  # 7pm

def _ballpark_cf_azimuth_from_coords(lat: Optional[float], lon: Optional[float]) -> Optional[float]:
    """
    If your ballpark_coords.json stores [lat, lon, orientation_deg],
    cc_venue_latlon_from_gamepk returns only lat/lon. We can’t fetch the 3rd item
    here, so weather_block will just use raw wind component.
    This stub returns None so callers know we don't have the park's CF azimuth.
    """
    return None

def _wind_to_component_mph(wind_mph: float, wind_deg_from: Optional[float], cf_azimuth_deg: Optional[float]) -> float:
    """
    Project wind vector onto the CF axis. WeatherAPI 'wind_degree' is the
    direction FROM which wind blows (meteorological). The TO direction is +180°.
    If we don't have CF azimuth, return 0.0 so caller can still use speed/dir.
    """
    if wind_mph is None or wind_deg_from is None or cf_azimuth_deg is None:
        return 0.0
    wind_to = (float(wind_deg_from) + 180.0) % 360.0
    delta = math.radians(wind_to - float(cf_azimuth_deg))
    return float(wind_mph) * math.cos(delta)

def _weatherapi_history(lat: float, lon: float, ymd: str) -> Optional[dict]:
    if not WEATHER_API_KEY:
        return None
    url = (
        "https://api.weatherapi.com/v1/history.json"
        f"?key={WEATHER_API_KEY}&q={lat},{lon}&dt={ymd}"
    )
    return _safe_get_json(url)

def _weatherapi_forecast(lat: float, lon: float, ymd: str) -> Optional[dict]:
    if not WEATHER_API_KEY:
        return None
    # Forecast endpoint also returns hourly for specific dates (including near-future)
    url = (
        "https://api.weatherapi.com/v1/forecast.json"
        f"?key={WEATHER_API_KEY}&q={lat},{lon}&dt={ymd}&days=1&aqi=no&alerts=no"
    )
    return _safe_get_json(url)

def _extract_hour_block(day_obj: dict, hour_pref: int) -> Optional[dict]:
    hrs = (day_obj or {}).get("hour") or []
    # Prefer requested local hour (e.g., 19 == 7pm)
    for h in hrs:
        try:
            # WeatherAPI hour[].time is like "2025-07-05 19:00"
            if int(str(h.get("time", "")).split(" ")[-1].split(":")[0]) == hour_pref:
                return h
        except Exception:
            pass
    # Fallback to middle of day or first hour available
    return hrs[12] if len(hrs) > 12 else (hrs[0] if hrs else None)

def _pull_day_and_hour(payload: dict, ymd: str, hour_pref: int) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Return (day_obj, hour_obj) where day_obj has 'day' aggregate and 'hour' list.
    Handles both history and forecast payload shapes.
    """
    # history.json → forecast.forecastday[0]
    # forecast.json → forecast.forecastday[0]
    fdays = (((payload or {}).get("forecast") or {}).get("forecastday") or [])
    chosen = None
    for d in fdays:
        if str(d.get("date")) == ymd:
            chosen = d
            break
    if not chosen and fdays:
        chosen = fdays[0]
    if not chosen:
        return None, None

    hour_obj = _extract_hour_block(chosen, hour_pref)
    return chosen, hour_obj

def _coerce_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default

def cc_weather(game_dict: dict) -> Optional[Dict[str, Any]]:
    """
    Fetch real weather via WeatherAPI for a given game dict.
    Returns a dict with:
      - temp (°F)
      - wind_speed (mph)
      - wind_out_to_cf (bool-ish)
      - wind_cf_component (signed mph along CF axis; positive → blowing out)
      - wind_cf_x_park (kept 0.0 here; design_features.weather_block will scale by park)
    Caching: 1 day TTL.
    """
    # If no API key, gracefully fall back to a neutral-but-not-None record
    if not WEATHER_API_KEY:
        return {"temp": 70.0, "wind_speed": 7.0, "wind_out_to_cf": 0, "wind_cf_component": 0.0, "wind_cf_x_park": 0.0}

    game_pk = game_dict.get("gamePk")
    ymd = _game_iso_date(game_dict)
    lat, lon = (None, None)
    try:
        lat, lon = cc_venue_latlon_from_gamepk(int(game_pk)) if game_pk else (None, None)
    except Exception:
        pass

    if not lat or not lon:
        # Cannot resolve coords → neutral-but-not-None output so model can still train
        return {"temp": 70.0, "wind_speed": 7.0, "wind_out_to_cf": 0, "wind_cf_component": 0.0, "wind_cf_x_park": 0.0}

    # Cache lookup
    ck = cache._make_key("weatherapi", f"{int(game_pk) if game_pk else 'no_pk'}_{ymd}")
    hit = cache.load_json("weatherapi", ck, max_age_days=1)
    if isinstance(hit, dict) and "temp" in hit:
        return hit

    # Decide history vs forecast
    today = _dt.date.today()
    this_day = _dt.date.fromisoformat(ymd)
    if this_day <= today:
        payload = _weatherapi_history(lat, lon, ymd)
    else:
        payload = _weatherapi_forecast(lat, lon, ymd)

    if not payload:
        out = {"temp": 70.0, "wind_speed": 7.0, "wind_out_to_cf": 0, "wind_cf_component": 0.0, "wind_cf_x_park": 0.0}
        cache.save_json("weatherapi", out, ck)
        return out

    # Prefer 7pm local if available
    hour_pref = _target_hour_local_default()
    day_obj, hour_obj = _pull_day_and_hour(payload, ymd, hour_pref)

    # Pull temp & wind (hourly preferred; else daily aggregates)
    if hour_obj:
        temp_f   = _coerce_float(hour_obj.get("temp_f"), default=_coerce_float((((day_obj or {}).get("day") or {}).get("avgtemp_f")), 70.0))
        wind_mph = _coerce_float(hour_obj.get("wind_mph"), default=_coerce_float((((day_obj or {}).get("day") or {}).get("maxwind_mph")), 7.0))
        wind_deg_from = _coerce_float(hour_obj.get("wind_degree"), default=None)
    else:
        d = (day_obj or {}).get("day") or {}
        temp_f   = _coerce_float(d.get("avgtemp_f"), default=70.0)
        wind_mph = _coerce_float(d.get("maxwind_mph"), default=7.0)
        wind_deg_from = None  # no hourly direction, so we can't project

    # Compute out-to-CF component (signed); if we don't know CF azimuth, just set 0
    cf_az = _ballpark_cf_azimuth_from_coords(lat, lon)  # currently returns None (no third value)
    cf_component = _wind_to_component_mph(wind_mph or 0.0, wind_deg_from, cf_az)

    out = {
        "temp": float(temp_f if temp_f is not None else 70.0),
        "wind_speed": float(wind_mph if wind_mph is not None else 7.0),
        "wind_out_to_cf": 1 if cf_component > 0.0 else 0,
        "wind_cf_component": float(cf_component),   # raw signed mph along CF axis
        "wind_cf_x_park": 0.0,                      # scaled in design_features.weather_block
    }
    cache.save_json("weatherapi", out, ck)
    return out
