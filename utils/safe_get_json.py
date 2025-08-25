# utils/safe_get_json.py
import requests

def _safe_get_json(url: str, timeout: float = 15.0, quiet_404: bool = True):
    try:
        r = requests.get(url, timeout=timeout)
        if quiet_404 and r.status_code == 404:
            # treat as missing, but silently
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        # keep this silent to avoid console spam; return None for caller to handle
        return None
