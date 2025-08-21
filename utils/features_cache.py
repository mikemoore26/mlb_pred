# utils/feature_cache.py
from __future__ import annotations
import os, json, hashlib
from datetime import date
from typing import Optional, List, Dict, Any
import pandas as pd

from utils.data_fetchers import fetch_games_with_features

DEFAULT_CACHE_DIR = os.path.join("data", "feature_cache")
os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)

def _features_fingerprint(cols: Optional[List[str]]) -> str:
    if not cols: return "none"
    return hashlib.sha1(",".join(cols).encode("utf-8")).hexdigest()[:10]

def _cache_paths(cache_dir: str,
                 start: str, end: str,
                 include_odds: bool, include_weather: bool,
                 req: Optional[List[str]]):
    fp = _features_fingerprint(req)
    key = f"{start}_{end}_odds{int(include_odds)}_wx{int(include_weather)}_{fp}"
    parq = os.path.join(cache_dir, f"{key}.parquet")
    manifest = os.path.join(cache_dir, f"{key}.json")
    return parq, manifest, key

def _read_manifest(path: str) -> Dict[str, Any]:
    if not os.path.exists(path): return {}
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception:
        return {}

def _write_manifest(path: str, meta: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, path)

def get_games_frame_cached(
    start_date: str,
    end_date: str,
    include_odds: bool = False,
    include_weather: bool = True,
    required_features: Optional[List[str]] = None,
    refresh_if_before: Optional[str] = None,   # e.g. date.today().isoformat()
    force_refresh: bool = False,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load cached feature frame if fresh; otherwise rebuild and store.
    - refresh_if_before: if manifest['last_updated'] < this ISO date, refresh.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    parq, manifest_path, key = _cache_paths(cache_dir, start_date, end_date, include_odds, include_weather, required_features)
    manifest = _read_manifest(manifest_path)
    last_updated = manifest.get("last_updated")
    today_iso = date.today().isoformat()

    should_refresh = force_refresh or (not os.path.exists(parq))
    if refresh_if_before and not should_refresh:
        if not last_updated or last_updated < refresh_if_before:
            should_refresh = True

    if should_refresh:
        df = fetch_games_with_features(
            start_date=start_date,
            end_date=end_date,
            include_odds=include_odds,
            include_weather=include_weather,
            required_features=required_features,
        )
        tmp = parq + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, parq)
        _write_manifest(manifest_path, {
            "key": key,
            "start_date": start_date,
            "end_date": end_date,
            "include_odds": include_odds,
            "include_weather": include_weather,
            "required_features": required_features or [],
            "last_updated": today_iso,
            "rows": int(len(df)),
            "schema": list(df.columns),
        })
        return df

    return pd.read_parquet(parq)
