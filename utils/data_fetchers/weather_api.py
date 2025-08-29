# utils/data_fetchers/weather_api.py
from __future__ import annotations
import os, json, sqlite3, time
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None

load_dotenv()

API_KEY = os.getenv("WEATHERAPI_KEY")
BASE = "https://api.weatherapi.com/v1"
DB_PATH = os.getenv("WEATHER_CACHE_DB", "cache/weather_cache.sqlite")
DEFAULT_TZ = os.getenv("WEATHER_DEFAULT_TZ", "America/New_York")
PROVIDER = "weatherapi"

# ----------------------
# SQLite cache helpers
# ----------------------

def _ensure_db():
    cache_dir = os.path.dirname(DB_PATH) or "."
    os.makedirs(cache_dir, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_cache (
            provider TEXT NOT NULL,
            stadium_id TEXT NOT NULL,
            iso_hour TEXT NOT NULL,        -- local hour bucket, e.g., 2025-08-27T19:00-04:00
            is_forecast INTEGER NOT NULL,  -- 1 forecast, 0 history
            payload_json TEXT NOT NULL,
            updated_ts REAL NOT NULL,
            PRIMARY KEY (provider, stadium_id, iso_hour)
        );
        """
    )
    con.close()

_ensure_db()

# ----------------------
# Dataclass for extracted hour
# ----------------------

@dataclass
class WeatherHour:
    temp_f: Optional[float]
    wind_mph: Optional[float]
    wind_dir_deg: Optional[float]  # direction wind is FROM, meteorological degrees
    source: str  # "history" | "forecast" | "cache" | "fallback"

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "wx_temp": self.temp_f,
            "wx_wind_speed": self.wind_mph,
            "wx_wind_dir_deg": self.wind_dir_deg,
        }

# ----------------------
# Core client
# ----------------------

class WeatherClient:
    """Thin client around WeatherAPI with caching and hourly extraction.

    Use get_hour_for_game(stadium_id, lat, lon, game_dt_local, tz_str) to fetch a WeatherHour.
    """

    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None):
        self.api_key = api_key or API_KEY
        self.session = session or requests.Session()

    # ---------- Public API ----------
    def get_hour_for_game(
        self,
        stadium_id: str,
        lat: float,
        lon: float,
        game_dt_local: dt.datetime,
        tz_str: Optional[str] = None,
    ) -> WeatherHour:
        tz = tz_str or DEFAULT_TZ
        if game_dt_local.tzinfo is None:
            if ZoneInfo:
                game_dt_local = game_dt_local.replace(tzinfo=ZoneInfo(tz))
            else:
                raise ValueError("game_dt_local must be timezone-aware when zoneinfo is unavailable")

        # bucket to top of hour for caching
        bucket = game_dt_local.replace(minute=0, second=0, microsecond=0)
        iso_hour = bucket.isoformat(timespec="minutes")

        # 1) cache read
        cached = self._cache_read(stadium_id, iso_hour)
        if cached is not None:
            return cached

        # 2) decide endpoint: history for past, forecast for now/future
        now_local = dt.datetime.now(bucket.tzinfo)
        is_future_or_today = bucket >= now_local.replace(minute=0, second=0, microsecond=0)
        kind = "forecast" if is_future_or_today else "history"

        if not self.api_key:
            wh = WeatherHour(temp_f=None, wind_mph=None, wind_dir_deg=None, source="fallback")
            # cache fallback to avoid repeated calls
            self._cache_write(stadium_id, iso_hour, wh, is_forecast=1 if is_future_or_today else 0)
            return wh

        payload = self._fetch_day(kind, lat, lon, bucket.date())
        wh = self._extract_hour(payload, bucket)
        wh.source = kind

        # 3) write cache (short-lived for forecasts)
        self._cache_write(stadium_id, iso_hour, wh, is_forecast=1 if is_future_or_today else 0)
        return wh

    # ---------- Internals ----------
    def _fetch_day(self, kind: str, lat: float, lon: float, date_obj: dt.date) -> Dict[str, Any]:
        assert kind in {"history", "forecast"}
        base = f"{BASE}/{ 'history.json' if kind=='history' else 'forecast.json' }"
        q = f"{lat},{lon}"
        params = {
            "key": self.api_key,
            "q": q,
            "dt": date_obj.isoformat(),
            "aqi": "no",
            "alerts": "no",
            # forecast.json respects dt for specific day forecasts; no 'hour' param here
        }
        url = f"{base}?{urlencode(params)}"
        # simple polite throttle
        time.sleep(0.2)
        r = self.session.get(url, timeout=15)
        r.raise_for_status()
        return r.json()

    def _extract_hour(self, payload: Dict[str, Any], target_local_dt: dt.datetime) -> WeatherHour:
        # WeatherAPI returns forecast.forecastday[0].hour (list of 24 hours)
        try:
            hours = payload["forecast"]["forecastday"][0]["hour"]
        except Exception:
            return WeatherHour(temp_f=None, wind_mph=None, wind_dir_deg=None, source="fallback")

        # choose the hour closest to our bucket
        target_epoch = int(target_local_dt.timestamp())
        best = min(
            hours,
            key=lambda h: abs(int(h.get("time_epoch", 0)) - target_epoch)
        )
        return WeatherHour(
            temp_f=best.get("temp_f"),
            wind_mph=best.get("wind_mph"),
            wind_dir_deg=best.get("wind_degree"),
            source="api",
        )

    def _cache_read(self, stadium_id: str, iso_hour: str) -> Optional[WeatherHour]:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute(
            "SELECT is_forecast, payload_json, updated_ts FROM weather_cache WHERE provider=? AND stadium_id=? AND iso_hour=?",
            (PROVIDER, stadium_id, iso_hour),
        )
        row = cur.fetchone()
        con.close()
        if not row:
            return None
        is_forecast, payload_json, updated_ts = row
        # if it's a forecast and older than 3 hours, ignore cache (forecasts change a lot)
        if is_forecast and (time.time() - float(updated_ts) > 3 * 3600):
            return None
        try:
            obj = json.loads(payload_json)
            return WeatherHour(**obj)
        except Exception:
            return None

    def _cache_write(self, stadium_id: str, iso_hour: str, wh: WeatherHour, is_forecast: int):
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "REPLACE INTO weather_cache (provider, stadium_id, iso_hour, is_forecast, payload_json, updated_ts) VALUES (?,?,?,?,?,?)",
            (PROVIDER, stadium_id, iso_hour, int(is_forecast), json.dumps(wh.__dict__), time.time()),
        )
        con.commit()
        con.close()