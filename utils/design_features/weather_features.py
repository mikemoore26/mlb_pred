# utils/design_features/weather_features.py
from __future__ import annotations
from typing import Dict, Any
from utils import data_fetchers as DF

def weather_block(game_dict: dict, park_factor: float = 100.0, include_weather: bool = True) -> Dict[str, Any]:
    """
    Produces standardized weather features the trainer expects.
    - wx_temp (°F)
    - wx_wind_speed (mph)
    - wx_wind_out_to_cf (0/1)  -> sign of component toward CF
    - wind_cf_x_park (component mph scaled by park factor)
    """
    if not include_weather:
        return {"wx_temp": 70.0, "wx_wind_speed": 7.0, "wx_wind_out_to_cf": 0, "wind_cf_x_park": 0.0}

    wx = DF.cc_weather(game_dict) or {}
    temp = wx.get("temp", 70.0)
    spd  = wx.get("wind_speed", 7.0)
    out  = int(wx.get("wind_out_to_cf", 0))
    comp = float(wx.get("wind_cf_component", 0.0))

    # Scale the CF wind component by park factor (e.g., 105 → 1.05x)
    pf_scale = float(park_factor or 100.0) / 100.0
    comp_x_park = comp * pf_scale

    return {
        "wx_temp": float(temp),
        "wx_wind_speed": float(spd),
        "wx_wind_out_to_cf": out,
        "wind_cf_x_park": float(comp_x_park),
    }
