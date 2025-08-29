# utils/cli_dates.py
from __future__ import annotations
from datetime import date, datetime, timedelta
from typing import Tuple

ISO = "%Y-%m-%d"

def _to_date(s: str) -> date:
    return datetime.strptime(s, ISO).date()

def parse_date_input(value: str, *, yesterday_offset: int = 1) -> date:
    """
    Parse CLI date inputs:
      - "auto"      -> today - yesterday_offset (useful for labeled games)
      - "yesterday" -> today - 1 day
      - "today"     -> today
      - "YYYY-MM-DD"
    """
    v = (value or "").strip().lower()
    today = date.today()
    if v == "auto":
        return today - timedelta(days=yesterday_offset)
    if v == "yesterday":
        return today - timedelta(days=1)
    if v == "today":
        return today
    return _to_date(value)

def resolve_training_dates(
    start: str,
    end: str,
    val_start: str,
    *,
    default_start: str = "2024-03-28",
    label_lag_days: int = 1,
    val_back_days: int = 30,
) -> Tuple[str, str, str]:
    """
    Returns ISO strings (start, end, val_start) with consistent defaults:
      - end:   "auto" -> today - label_lag_days
      - start: "auto" -> default_start
      - val_start: "auto" -> end - val_back_days
    """
    # end first (depends on label_lag_days)
    end_dt = parse_date_input(end, yesterday_offset=label_lag_days) if end.lower() == "auto" else parse_date_input(end, yesterday_offset=label_lag_days)
    start_dt = _to_date(default_start) if start.lower() == "auto" else parse_date_input(start, yesterday_offset=label_lag_days)

    if val_start.lower() == "auto":
        val_start_dt = end_dt - timedelta(days=val_back_days)
    else:
        val_start_dt = parse_date_input(val_start, yesterday_offset=label_lag_days)

    # sanity: ensure ordering
    if not (start_dt <= val_start_dt <= end_dt):
        # If out of order, nudge with safe defaults
        if val_start_dt > end_dt:
            val_start_dt = end_dt
        if start_dt > val_start_dt:
            start_dt = val_start_dt

    return start_dt.isoformat(), end_dt.isoformat(), val_start_dt.isoformat()
