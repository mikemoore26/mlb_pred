from datetime import date
import os
import joblib
import numpy as np
import pandas as pd
from flask import Blueprint, render_template, flash, request, redirect, url_for
from flask_login import login_required, current_user
from models import db, User, Prediction
from utils.feature_builder import build_features_for_range as fetch_games_with_features

main_bp = Blueprint("main", __name__)

# All pages require login
@main_bp.before_request
@login_required
def require_login():
    # If not logged in, @login_required will redirect to login_view
    pass

@main_bp.route("/index")
def index():
    return render_template("index.html")

@main_bp.route("/users")
def users():
    if not current_user.is_admin:
        flash("Admins only.", "warning")
        return redirect(url_for("main.index"))
    rows = User.query.order_by(User.created_at.desc()).all()
    return render_template("users.html", rows=rows)

@main_bp.route("/admin")
def admin():
    if not current_user.is_admin:
        flash("Admins only.", "warning")
        return redirect(url_for("main.index"))
    # very simple admin page for now
    today = date.today()
    today_preds = Prediction.query.filter_by(game_date=today).count()
    return render_template("admin.html", today_preds=today_preds, today=today.isoformat())

@main_bp.route("/predictions")
def predictions():
    """
    Show today's predictions. If none exist:
      - Check if there are any MLB games scheduled today.
      - If so, run model to generate + store predictions.
    """
    today = date.today()

    # First, check if we already have predictions
    existing = Prediction.query.filter_by(game_date=today, sport="mlb").order_by(Prediction.created_at.desc()).all()
    if not existing:
        # Pull today's slate (scheduled, not finals-only)
        df = fetch_games_with_features(
            start_date=today.isoformat(),
            end_date=today.isoformat(),
            include_odds=False,
            include_weather=True,
            required_features=None,
            only_finals=False,
        )
        if df.empty:
            flash("No MLB games scheduled for today.", "info")
            return render_template("predictions.html", rows=[])

        # Try to run prediction if model exists
        model_path = request.args.get("model", os.getenv("DEFAULT_MODEL", "models/full_model_ml2.pkl"))
        if not os.path.exists(model_path):
            flash(f"Model not found at '{model_path}'. Put your trained model there.", "danger")
            return render_template("predictions.html", rows=[])

        try:
            rows = _predict_and_store(df, model_path, game_date=today)
            if not rows:
                flash("No predictions produced.", "warning")
                return render_template("predictions.html", rows=[])
            flash(f"Generated {len(rows)} predictions for today.", "success")
            existing = Prediction.query.filter_by(game_date=today, sport="mlb").order_by(Prediction.created_at.desc()).all()
        except Exception as e:
            flash(f"Prediction failed: {e}", "danger")
            return render_template("predictions.html", rows=[])

    # Render from DB
    return render_template("predictions.html", rows=existing)

# ---------------- helpers ----------------
def _load_model(path):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        est = obj["model"]
        req = obj.get("required_features") or list(getattr(est, "feature_names_in_", []))
        model_name = obj.get("meta", {}).get("name") or "unknown"
    else:
        est = obj
        req = list(getattr(est, "feature_names_in_", []))
        model_name = getattr(est, "__class__", type("X",(object,),{})).__name__
    if not req:
        raise RuntimeError("Model missing required_features / feature_names_in_. Re-train with that metadata.")
    return est, req, model_name

def _align(df, req):
    X = df.copy()
    for c in req:
        if c not in X.columns:
            X[c] = 0.0
    return X[req].apply(pd.to_numeric, errors="coerce").fillna(0.0)

def _predict_and_store(df: pd.DataFrame, model_path: str, game_date: date):
    est, req, model_name = _load_model(model_path)
    X = _align(df, req)
    if hasattr(est, "predict_proba"):
        proba = est.predict_proba(X)[:, 1]
    else:
        proba = np.clip(est.predict(X).astype(float), 0.0, 1.0)

    out_rows = []
    for i, row in df.iterrows():
        home = str(row.get("home_team", "HOME"))
        away = str(row.get("away_team", "AWAY"))
        p = float(proba[i])
        pick = "HOME" if p >= 0.5 else "AWAY"
        pick_prob = p if p >= 0.5 else (1 - p)

        pred = Prediction(
            sport="mlb",
            game_date=game_date,
            home_team=home,
            away_team=away,
            model_name=model_name,
            proba_home=p,
            pick=pick,
            pick_prob=pick_prob,
        )
        db.session.add(pred)
        out_rows.append(pred)
    db.session.commit()
    return out_rows
