# main.py
from __future__ import annotations
import os
import json
import glob
import joblib
import numpy as np
import pandas as pd
from datetime import date, datetime

from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user

# If you log predictions to DB, import your model(s) here:
# from model import Prediction

from utils.feature_builder import build_features_for_range
import utils.sources as S  # for exact schedule start times

main_bp = Blueprint("main", __name__)

# --------- Simple persisted selection ----------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CONFIG_PATH = os.path.join(DATA_DIR, "model_selection.json")
DEFAULT_MODEL = "models/full_model_ml2.pkl"


def _read_selection() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"model_path": DEFAULT_MODEL}


def _write_selection(model_path: str) -> None:
    payload = {"model_path": model_path}
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, CONFIG_PATH)


def _list_models() -> list[dict]:
    """
    Find all .pkl in models/ and expose:
      {value: path, label: filename, has_report: bool, report_path: str}
    """
    items = []
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


def _load_model_payload(path: str):
    """
    Load a payload or raw estimator.
    Returns (estimator, required_feature_list, meta_dict)
    """
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


def _align(df: pd.DataFrame, req: list[str]) -> pd.DataFrame:
    X = df.copy()
    for c in req:
        if c not in X.columns:
            X[c] = 0.0
    return X[req].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _today_schedule_rows() -> list[dict]:
    """
    Return today's games with exact start time in UTC if available.
    Sorted by start time (then home team).
    """
    today = date.today().isoformat()
    games = S.get_schedule_range(today, today) or []
    rows = []
    for g in games:
        try:
            home = g["teams"]["home"]["team"]["name"]
            away = g["teams"]["away"]["team"]["name"]
            game_dt_iso = g.get("gameDate")
            try:
                dt = datetime.fromisoformat(str(game_dt_iso).replace("Z", "+00:00"))
                hhmm = dt.strftime("%H:%M")
            except Exception:
                dt = None
                hhmm = ""
            rows.append({
                "start_dt_utc": game_dt_iso,
                "start_time": hhmm,
                "home_team": home,
                "away_team": away,
            })
        except Exception:
            continue

    if rows:
        for r in rows:
            try:
                r["_sort_dt"] = datetime.fromisoformat(str(r["start_dt_utc"]).replace("Z", "+00:00"))
            except Exception:
                r["_sort_dt"] = None
        rows.sort(key=lambda r: (r["_sort_dt"] is None, r["_sort_dt"], r["home_team"]))
        for r in rows:
            r.pop("_sort_dt", None)
    return rows


def _read_model_report(path: str) -> str:
    report_path = os.path.splitext(path)[0] + "_report.txt"
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "No report available for this model."


def _safe_load_meta(path: str) -> dict:
    """
    Try to read minimal meta from the model payload (name, variant, created).
    Never crashes the page if a model can't be loaded.
    """
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


def _collect_model_rows() -> list[dict]:
    """
    Return list of rows for the admin table:
      { path, filename, is_selected, name, variant, created, has_report, report_preview }
    """
    sel = _read_selection()
    selected_path = sel.get("model_path") or DEFAULT_MODEL
    rows = []
    for m in _list_models():
        path = m["value"]
        filename = m["label"]
        meta = _safe_load_meta(path)
        report_text = _read_model_report(path)
        # Short preview (first ~8 lines or 600 chars)
        lines = report_text.splitlines()
        if len(lines) > 8:
            preview = "\n".join(lines[:8]) + "\n…"
        else:
            preview = report_text[:600] + ("…" if len(report_text) > 600 else "")
        rows.append({
            "path": path,
            "filename": filename,
            "is_selected": (path == selected_path),
            "name": meta.get("name"),
            "variant": meta.get("variant"),
            "created": meta.get("created"),
            "has_report": m["has_report"],
            "report_preview": preview or "—",
        })
    return rows


def _predict_today_selected_model() -> pd.DataFrame:
    """
    Use the admin-selected single model for predictions.
    Returns columns: game_date, start_time, home_team, away_team, proba, pick, pick_prob
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

    # attach start_time for sort and display
    sched = pd.DataFrame(_today_schedule_rows())
    if not sched.empty:
        df = df.merge(sched, on=["home_team", "away_team"], how="left")
    else:
        df["start_time"] = ""

    sel = _read_selection()
    model_path = sel.get("model_path") or DEFAULT_MODEL

    A, reqA, _ = _load_model_payload(model_path)
    XA = _align(df, reqA)
    p = (A.predict_proba(XA)[:, 1] if hasattr(A, "predict_proba")
         else A.predict(XA).astype(float))

    cols = [c for c in ["game_date", "start_time", "home_team", "away_team"] if c in df.columns]
    out = df[cols].copy()
    out["proba"] = p
    out["pick"] = np.where(p >= 0.5, "HOME", "AWAY")
    out["pick_prob"] = np.where(p >= 0.5, p, 1 - p)

    if "start_time" in out.columns:
        out = out.sort_values(["start_time", "home_team"]).reset_index(drop=True)
    else:
        out = out.sort_values(["game_date", "home_team"]).reset_index(drop=True)
    return out


# --------- Routes ----------
@main_bp.route("/")
@login_required
def index():
    rows = _today_schedule_rows()
    # rows is a Python list; Jinja can iterate it directly
    return render_template("index.html", games=rows)


@main_bp.route("/predictions")
@login_required
def predictions():
    preds = _predict_today_selected_model()
    # Convert to list of dicts for easy templating
    preds_rows = (preds.to_dict(orient="records") if isinstance(preds, pd.DataFrame) else [])
    return render_template("predictions.html", preds=preds_rows)


@main_bp.route("/admin", methods=["GET", "POST"])
@login_required
def admin():
    if not getattr(current_user, "is_admin", False):
        flash("Admin access required.", "warning")
        return redirect(url_for("main.index"))

    models = _list_models()
    sel = _read_selection()
    selected_path = sel.get("model_path") or DEFAULT_MODEL

    if request.method == "POST":
        new_path = request.form.get("model_path") or ""
        if not new_path or not os.path.exists(new_path):
            flash("Invalid model selection.", "error")
            return redirect(url_for("main.admin"))
        _write_selection(new_path)
        flash("Model selection saved.", "success")
        return redirect(url_for("main.admin"))

    table_rows = _collect_model_rows()
    return render_template(
        "admin.html",
        models=models,
        selected_path=selected_path,
        table_rows=table_rows
    )


@main_bp.route("/contact")
@login_required
def contact():
    return render_template("contact.html")
