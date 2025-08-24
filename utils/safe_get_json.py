# Persistent HTTP session for efficient requests
import requests
from typing import Optional
_SESSION = requests.Session()

def _safe_get_json(url: str, timeout: int = 12) -> Optional[dict]:
    """
    Safely fetch JSON from a URL with error handling.
    Uses a persistent session to improve performance.
    """
    try:
        r = _SESSION.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return None