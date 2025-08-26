# utils/design_feature/weather_features.py
from __future__ import annotations
from typing import Dict
from utils import data_fetchers as DF
from utils.design_features.utils import f

DEBUG_WEATHER = False

def weather_block(game_dict: dict, park_factor: float, include_weather: bool) -> Dict[str, float]:
    """
    Returns:
      - wx_temp, wx_wind_speed, wx_wind_out_to_cf, wind_cf_x_park
    If include_weather=False or no data, returns zeros.
    """
    if not include_weather:
        return {
            "wx_temp": 0.0, "wx_wind_speed": 0.0,
            "wx_wind_out_to_cf": 0.0, "wind_cf_x_park": 0.0,
        }
    wx = DF.cc_weather(game_dict) or {}
    temp = f(wx.get("temp"), 0.0)
    wind = f(wx.get("wind_speed"), 0.0)
    out_to_cf = 1.0 if wx.get("wind_out_to_cf") else 0.0
    wind_x_park = f(out_to_cf * (park_factor if park_factor is not None else 0.0), 0.0)
    out = {
        "wx_temp": temp,
        "wx_wind_speed": wind,
        "wx_wind_out_to_cf": out_to_cf,
        "wind_cf_x_park": wind_x_park,
    }
    if DEBUG_WEATHER:
        print("[weather_block]", out)
    return out
