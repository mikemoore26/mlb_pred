# utils/_common.py
from __future__ import annotations
from typing import Any
import re

_IP_RE = re.compile(r"^\s*(\d+)(?:\.(\d))?\s*$")

def parse_ip_to_float(ip_val: Any) -> float:
    """
    Parse MLB IP string like '1.2' into decimal innings (1 + 2/3).
    Accepts int/float and strings. Returns 0.0 on any failure.
    """
    try:
        if ip_val is None:
            return 0.0
        if isinstance(ip_val, (int, float)):
            whole = int(ip_val)
            tenths = int(round((float(ip_val) - whole) * 10))
            if tenths not in (0, 1, 2):
                tenths = 0
            return whole + tenths / 3.0
        s = str(ip_val).strip()
        m = _IP_RE.match(s)
        if not m:
            return 0.0
        whole = int(m.group(1))
        tenths = int(m.group(2) or 0)
        if tenths not in (0, 1, 2):
            tenths = 0
        return whole + tenths / 3.0
    except Exception:
        return 0.0
    

def i(x: Any, d: int = 0) -> int:
    try:
        return int(x)
    except (ValueError, TypeError):
        return int(d)
    
def f(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return float(d)

# short alias used in your code
parse_ip = parse_ip_to_float

