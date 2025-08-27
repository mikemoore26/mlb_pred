# utils/data_fetchers/schedule.py
from __future__ import annotations
import datetime as _dt
import utils.cache as cache
from utils.safe_get_json import _safe_get_json

def cc_schedule_range(start_date: str, end_date: str) -> dict:
    """
    Cached MLB schedule JSON for a date range (1-day TTL).
    """
    k = cache._make_key("schedule", f"{start_date}_{end_date}")
    hit = cache.load_json("schedule", k, max_age_days=1)
    if hit is not None:
        return hit

    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?startDate={start_date}&endDate={end_date}&sportId=1&hydrate=probablePitchers"
    )
    data = _safe_get_json(url) or {}
    cache.save_json("schedule", data, k)
    return data
