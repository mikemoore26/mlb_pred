# utils/cache.py
from __future__ import annotations
import json
import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional, List
import os

import pandas as pd

# utils/cache.py (top)
import os, json, hashlib
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Optional, List
import pandas as pd

# Base cache root (env-overridable)
CACHE_DIR = Path(os.getenv("CACHE_DIR", Path(__file__).resolve().parent.parent / "cache"))

# Namespaces
RAW_DIR         = CACHE_DIR / "raw"          # raw inputs (API JSON, boxscores, etc.)
ENGINEERED_DIR  = CACHE_DIR / "engineered"   # feature tables, model-ready datasets
STATE_DIR       = CACHE_DIR / "state"        # derived states (e.g., Elo), safe to reset

for p in (RAW_DIR, ENGINEERED_DIR, STATE_DIR):
    p.mkdir(parents=True, exist_ok=True)


# Default features cache (Parquet preferred)
FEATURES_CACHE_PATH = ENGINEERED_DIR / "mlb_features.parquet"

# ----------------------
# Filename / key helpers
# ----------------------
def _make_key(prefix: str, *parts: Any) -> str:
    prefix = prefix.replace(".json", "")
    raw = prefix + "|" + "|".join(str(p) for p in parts)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}.json"

def load_json(cache_type: str, key: str, max_age_days: int = 7, *, folder: Path = RAW_DIR) -> Optional[dict]:
    folder.mkdir(parents=True, exist_ok=True)
    key = key.replace(".json.json", ".json")
    filename = folder / f"{cache_type}_{key}"
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        if max_age_days:
            mtime = datetime.fromtimestamp(os.path.getmtime(filename))
            if (datetime.now() - mtime).days > max_age_days:
                print(f"Cache expired for {filename}")
                return None
        return data
    except FileNotFoundError:
        print(f"Cache miss for {filename}")
        return None
    except Exception as e:
        print(f"Error loading cache {filename}: {e}")
        return None

def save_json(cache_type: str, data: Any, key: str, *, folder: Path = RAW_DIR) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    key = key.replace(".json.json", ".json")
    filename = folder / f"{cache_type}_{key}"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving cache {filename}: {e}")

# ---------------------------
# Table cache (Parquet / CSV)
# ---------------------------
# ---------- Table I/O ----------
def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "game_date" not in df.columns:
        low = {c.lower(): c for c in df.columns}
        if "date" in low: df = df.rename(columns={low["date"]: "game_date"})
        elif "gamedate" in low: df = df.rename(columns={low["gamedate"]: "game_date"})
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.dropna(subset=["game_date"]).copy()
    return df

def _read_table(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists(): return None
    try:
        return pd.read_parquet(path) if path.suffix.lower()==".parquet" else pd.read_csv(path)
    except Exception as e:
        print(f"[TABLE CACHE] read error {path}: {e}")
        return None

def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        (df.to_parquet if path.suffix.lower()==".parquet" else df.to_csv)(path, index=False)
    except Exception as e:
        print(f"[TABLE CACHE] write error {path}: {e}")

def _choose_upsert_keys(df: pd.DataFrame) -> List[str]:
    """Prefer unique game id if present; otherwise fallback to a stable triplet."""
    if "gamePk" in df.columns:
        return ["gamePk"]
    return ["game_date", "home_team", "away_team"]

def latest_cached_date(cache_path: Path = FEATURES_CACHE_PATH) -> Optional[date]:
    """Return the max cached game_date as a date, if cache exists."""
    cached = _read_table(cache_path)
    if cached is None or cached.empty or "game_date" not in cached.columns:
        return None
    cached = _normalize_dates(cached)
    if cached.empty:
        return None
    return cached["game_date"].max().date()

def upsert_table(new_rows: pd.DataFrame, cache_path: Path = FEATURES_CACHE_PATH) -> pd.DataFrame:
    """
    Upsert `new_rows` into the table cache using key columns.
    Returns the full, deduped, sorted dataframe.
    """
    new_rows = _normalize_dates(new_rows)
    cached = _read_table(cache_path)
    if cached is None or cached.empty:
        combined = new_rows.sort_values("game_date").reset_index(drop=True)
        _write_table(combined, cache_path)
        print(f"[CACHE] Initialized cache with {len(new_rows)} rows -> {cache_path}")
        return combined

    cached = _normalize_dates(cached)
    keys = _choose_upsert_keys(new_rows if "gamePk" in new_rows.columns else cached)
    combined = pd.concat([cached, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=keys, keep="last")
    combined = combined.sort_values("game_date").reset_index(drop=True)
    _write_table(combined, cache_path)
    print(f"[CACHE] Upserted {len(new_rows)} rows. Cache now {len(combined)} -> {cache_path}")
    return combined

# -------------------------------------------------------
# Incremental feature dataset for MLB (only fetch “new”)
# -------------------------------------------------------
def get_incremental_features(
    *,
    start_date: str,
    end_date: Optional[str] = None,
    include_odds: bool = False,
    required_features: Optional[List[str]] = None,
    include_weather: bool = True,
    cache_path: Path = FEATURES_CACHE_PATH,
    force_until_today: bool = True,
) -> pd.DataFrame:
    """
    Load cached features and only fetch NEW rows after the last cached game_date.
    If no cache exists, fetch start_date..end_date initially.
    If `force_until_today` is True, will automatically fetch through today's date
    when the cache is stale (i.e., last_cached_date < today).
    """
    # Lazy import to avoid circulars
    from utils.feature_builder import fetch_games_with_features

    # Resolve end_date
    if end_date is None:
        # If training labels require completed games, you might want (today-1)
        # but per request we'll push to today so cache is always "fresh today".
        end_dt = date.today()
        end_date = end_dt.isoformat()
    else:
        end_dt = date.fromisoformat(end_date)

    # Load what we have
    current = _read_table(cache_path)
    if current is not None and not current.empty:
        current = _normalize_dates(current)
        last_dt = current["game_date"].max().date()
        target_end = max(end_dt, date.today()) if force_until_today else end_dt

        if last_dt >= target_end:
            print(f"[CACHE] Up to date through {last_dt}. No fetch needed.")
            # Clip to requested end_date (not strictly necessary but nice)
            return current[current["game_date"] <= pd.to_datetime(end_date)].copy()

        # Fetch only the missing tail
        fetch_start = (last_dt + timedelta(days=1)).isoformat()
        fetch_end = target_end.isoformat()
        print(f"[CACHE] Fetching increment {fetch_start} .. {fetch_end}")
        new_df = fetch_games_with_features(
            start_date=fetch_start,
            end_date=fetch_end,
            include_odds=include_odds,
            include_weather=include_weather,
            required_features=list(dict.fromkeys(required_features or [])),
        )
        if new_df is None or len(new_df) == 0:
            print("[CACHE] No new rows returned by fetcher.")
            return current[current["game_date"] <= pd.to_datetime(end_date)].copy()

        combined = upsert_table(new_df, cache_path=cache_path)
        # Clip to requested end_date for the returned frame
        return combined[combined["game_date"] <= pd.to_datetime(end_date)].copy()

    # No cache yet: full initial fetch
    print(f"[CACHE] No cache found. Full fetch {start_date} .. {end_date}")
    df_all = fetch_games_with_features(
        start_date=start_date,
        end_date=end_date,
        include_odds=include_odds,
        include_weather=include_weather,
        required_features=list(dict.fromkeys(required_features or [])),
    )
    if df_all is None or len(df_all) == 0:
        raise SystemExit("No games fetched. Check date range or fetcher.")
    df_all = _normalize_dates(df_all).sort_values("game_date").reset_index(drop=True)
    _write_table(df_all, cache_path)
    print(f"[CACHE] Wrote new cache -> {cache_path}")
    return df_all
