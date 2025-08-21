# utils/cache.py
from __future__ import annotations
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Base cache folder
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def _make_key(prefix: str, *parts: Any) -> str:
    """
    Build a short, unique filename from prefix + parts.
    Every part is stringified and hashed (to avoid long filenames).
    """
    raw = prefix + "|" + "|".join(str(p) for p in parts)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}.json"

def save_json(prefix: str, data: Any, *parts: Any) -> str:
    """
    Save JSON-serializable 'data' to cache file.
    """
    path = CACHE_DIR / _make_key(prefix, *parts)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str)
    return str(path)

def load_json(prefix: str, *parts: Any, max_age_days: int = 1) -> Optional[Any]:
    """
    Load JSON from cache if it exists and is not older than max_age_days.
    Returns None on miss or stale.
    """
    path = CACHE_DIR / _make_key(prefix, *parts)
    if not path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return None
    if (datetime.now() - mtime).days > max_age_days:
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
