# utils/data_fetchers/probables.py
from __future__ import annotations
import datetime as _dt
from typing import Dict, List
import utils.cache as cache
from utils.safe_get_json import _safe_get_json

def cc_probables(game_pks: List[int]) -> Dict[int, Dict[str, int]]:
    """
    Probables for provided game_pks. Uses per-PK cache, a batch snapshot, and a short lookahead schedule.
    Writes negative caches {} for missing Pks to avoid repeated fetches.
    """
    wanted = [int(pk) for pk in game_pks if pk]
    if not wanted:
        return {}
    result: Dict[int, Dict[str, int]] = {}
    missing: List[int] = []

    # per-PK cache
    for pk in wanted:
        per_key = cache._make_key("probables", pk)
        cached = cache.load_json("probables", per_key, max_age_days=1)
        if isinstance(cached, dict):
            result[pk] = cached
        else:
            missing.append(pk)
    if not missing:
        return result

    # batch cache
    batch_key = cache._make_key("probables_batch", *sorted(wanted))
    batch_cached = cache.load_json("probables", batch_key, max_age_days=1)
    if isinstance(batch_cached, dict):
        for pk, rec in batch_cached.items():
            per_key = cache._make_key("probables", pk)
            cache.save_json("probables", rec, per_key)
            result[pk] = rec
        missing = [pk for pk in wanted if pk not in result]

    # 3-day lookahead via schedule hydrate
    if missing:
        today = _dt.date.today()
        start_date = today.isoformat()
        end_date = (today + _dt.timedelta(days=3)).isoformat()
        url = (
            "https://statsapi.mlb.com/api/v1/schedule"
            f"?startDate={start_date}&endDate={end_date}&sportId=1&hydrate=probablePitchers"
        )
        data = _safe_get_json(url) or {}
        found: Dict[int, Dict[str, int]] = {}
        for d in data.get("dates", []):
            for g in d.get("games", []):
                pk = g.get("gamePk")
                if pk in missing:
                    home_id = (g.get("teams", {}).get("home", {}).get("probablePitcher", {}) or {}).get("id")
                    away_id = (g.get("teams", {}).get("away", {}).get("probablePitcher", {}) or {}).get("id")
                    if home_id or away_id:
                        found[pk] = {"home_id": home_id, "away_id": away_id}
        for pk, rec in found.items():
            per_key = cache._make_key("probables", pk)
            cache.save_json("probables", rec, per_key)
            result[pk] = rec
        still_missing = [pk for pk in missing if pk not in result]
        for pk in still_missing:
            per_key = cache._make_key("probables", pk)
            cache.save_json("probables", {}, per_key)
            result[pk] = {}
        batch_out = {pk: result.get(pk, {}) for pk in wanted}
        cache.save_json("probables", batch_out, batch_key)

    return result
