from __future__ import annotations
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import os

# Base cache folder
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def _make_key(prefix: str, *parts: Any) -> str:
    """
    Build a short, unique filename from prefix + parts.
    Every part is stringified and hashed (to avoid long filenames).
    Ensures no double .json extension.
    """
    prefix = prefix.replace(".json", "")
    raw = prefix + "|" + "|".join(str(p) for p in parts)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}.json"

def load_json(cache_type: str, key: str, max_age_days: int = 7) -> Optional[dict]:
    """Load cached JSON data if it exists and is recent."""
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    key = key.replace(".json.json", ".json")
    filename = os.path.join(cache_dir, f"{cache_type}_{key}")
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

def save_json(cache_type: str, data: Any, key: str) -> None:
    """Save data to cache as JSON."""
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    key = key.replace(".json.json", ".json")
    filename = os.path.join(cache_dir, f"{cache_type}_{key}")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving cache {filename}: {e}")