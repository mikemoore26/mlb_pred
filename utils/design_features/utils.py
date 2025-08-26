# utils/design_feature/utils.py
from typing import Any

def f(x: Any, d: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(d)

def i(x: Any, d: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return int(d)
