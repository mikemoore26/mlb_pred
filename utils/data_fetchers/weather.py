# utils/data_fetchers/weather.py
from __future__ import annotations
from typing import Optional
import utils.cache as cache
from .park_venue import cc_venue_latlon_from_gamepk

def cc_weather(game_dict: dict) -> Optional[dict]:
    """
    Stubbed weather provider. Returns neutral defaults but caches per gamePk.
    Replace with a real API when ready (use venue lat/lon if needed).
    """
    try:
        game_pk = game_dict.get("gamePk")
        k = cache._make_key("weather", game_pk or "unknown")
        hit = cache.load_json("weather", k, max_age_days=1)
        if isinstance(hit, dict):
            return hit

        # we could use coords for a real provider:
        _lat, _lon = cc_venue_latlon_from_gamepk(game_pk) if game_pk else (None, None)

        out = {
            "temp": 75.0,
            "wind_speed": 8.0,
            "wind_out_to_cf": False,
            "wind_cf_x_park": 0.0,
        }
        cache.save_json("weather", out, k)
        return out
    except Exception:
        return {"temp": 75.0, "wind_speed": 8.0, "wind_out_to_cf": False, "wind_cf_x_park": 0.0}
