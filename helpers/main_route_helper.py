# helpers/main_helpers.py
from __future__ import annotations
import os, json, glob, joblib
from datetime import date, datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from utils.feature_builder import build_features_for_range
import utils.sources as S  # for schedule start times

# --------- persisted model selection ----------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CONFIG_PATH = os.path.join(DATA_DIR, "model_selection.json")
DEFAULT_MODEL = os.environ.get("MODEL_PATH", "models/full_model_ml2.pkl")

def read_selection() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"model_path": DEFAULT_MODEL}

def write_selection(model_path: str) -> None:
    payload = {"model_path": model_path}
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, CONFIG_PATH)

def list_models() -> List[dict]:
    """Return [{value, label, has_report, report_path}] for models/ *.pkl"""
    items: List[dict] = []
    for p in sorted(glob.glob(os.path.join("models", "*.pkl"))):
        base = os.path.basename(p)
        report = os.path.splitext(p)[0] + "_report.txt"
        items.append({
            "value": p,
            "label": base,
            "has_report": os.path.exists(report),
            "report_path": report
        })
    return items

def load_model_payload(path: str):
    """Return (estimator, required_features, meta_dict) from payload or raw estimator."""
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        est = obj["model"]
        req = obj.get("required_features") or list(getattr(est, "feature_names_in_", []))
        meta = obj.get("meta", {})
    else:
        est = obj
        req = list(getattr(est, "feature_names_in_", []))
        meta = {}
    return est, req, meta

def _align(df: pd.DataFrame, req: List[str]) -> pd.DataFrame:
    X = df.copy()
    for c in req:
        if c not in X.columns:
            X[c] = 0.0
    return X[req].apply(pd.to_numeric, errors="coerce").fillna(0.0)

from datetime import date, datetime
import pytz   # <--- add this import (pip install pytz)

# inside today_schedule_rows()
def today_schedule_rows() -> list[dict]:
    """Return today's games with start time in Eastern Time (ET)."""
    today = date.today().isoformat()
    games = S.get_schedule_range(today, today) or []
    rows = []
    utc = pytz.UTC
    eastern = pytz.timezone("US/Eastern")

    for g in games:
        try:
            home = g["teams"]["home"]["team"]["name"]
            away = g["teams"]["away"]["team"]["name"]
            game_dt_iso = g.get("gameDate")

            try:
                # parse ISO (UTC)
                dt_utc = datetime.fromisoformat(str(game_dt_iso).replace("Z", "+00:00"))
                dt_utc = dt_utc.astimezone(utc)
                # convert to ET
                dt_et = dt_utc.astimezone(eastern)
                hhmm = dt_et.strftime("%I:%M %p")  # 07:05 PM format
            except Exception:
                dt_et = None
                hhmm = ""

            rows.append({
                "start_dt_utc": game_dt_iso,
                "start_time": hhmm,
                "home_team": home,
                "away_team": away,
                "_sort_dt": dt_et,
            })
        except Exception:
            continue

    # sort by ET start time, fallback to home team
    rows.sort(key=lambda r: (r["_sort_dt"] is None, r["_sort_dt"], r["home_team"]))
    for r in rows:
        r.pop("_sort_dt", None)
    return rows

def read_model_report(path: str) -> str:
    report_path = os.path.splitext(path)[0] + "_report.txt"
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "No report available for this model."

def _safe_meta(path: str) -> dict:
    meta = {"name": None, "variant": None, "created": None}
    try:
        obj = joblib.load(path)
        if isinstance(obj, dict):
            m = obj.get("meta", {})
            meta["name"] = m.get("name")
            meta["variant"] = m.get("variant")
            meta["created"] = m.get("created")
    except Exception:
        pass
    return meta

def collect_model_rows() -> List[dict]:
    """Rows for admin table with meta + short report preview."""
    sel = read_selection()
    selected_path = sel.get("model_path") or DEFAULT_MODEL
    rows: List[dict] = []
    for m in list_models():
        path = m["value"]
        meta = _safe_meta(path)
        report_text = read_model_report(path)
        lines = report_text.splitlines()
        if len(lines) > 8:
            preview = "\n".join(lines[:8]) + "\n…"
        else:
            preview = report_text[:600] + ("…" if len(report_text) > 600 else "")
        rows.append({
            "path": path,
            "filename": m["label"],
            "is_selected": (path == selected_path),
            "name": meta.get("name"),
            "variant": meta.get("variant"),
            "created": meta.get("created"),
            "has_report": m["has_report"],
            "report_preview": preview or "—",
        })
    return rows

def predict_today_selected_model() -> pd.DataFrame:
    """
    Predict today’s games using the currently selected model.
    Returns columns: game_date, start_time, home_team, away_team, proba_home_win, pick_team, pick_prob
    """
    today = date.today().isoformat()
    df = build_features_for_range(
        start_date=today,
        end_date=today,
        include_odds=False,
        include_weather=True,
        only_finals=False,
        required_features=None,
    )
    if df.empty:
        return df

    # add start_time for sort/display
    sched = pd.DataFrame(today_schedule_rows())
    if not sched.empty:
        df = df.merge(sched, on=["home_team", "away_team"], how="left")
    else:
        df["start_time"] = ""

    sel = read_selection()
    model_path = sel.get("model_path") or DEFAULT_MODEL
    if not os.path.exists(model_path):
        model_path = DEFAULT_MODEL

    est, req, _ = load_model_payload(model_path)
    X = _align(df, req)
    proba = est.predict_proba(X)[:, 1] if hasattr(est, "predict_proba") else est.predict(X).astype(float)

    out = df[["game_date", "start_time", "home_team", "away_team"]].copy()
    out["proba_home_win"] = proba
    out["pick_team"] = np.where(proba >= 0.5, out["home_team"], out["away_team"])
    out["pick_prob"] = np.where(proba >= 0.5, proba, 1 - proba)

    out = out.sort_values(["start_time", "home_team"]).reset_index(drop=True)
    return out

