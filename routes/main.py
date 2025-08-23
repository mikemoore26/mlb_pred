# routes/main.py
from __future__ import annotations
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
import os
import pandas as pd

from helpers.main_route_helper import (
    today_schedule_rows,
    predict_today_selected_model,
    list_models,
    read_selection,
    write_selection,
    collect_model_rows,
)

main_bp = Blueprint("main", __name__)

# ----------------- Routes -----------------

@main_bp.route("/")
@login_required
def index():
    games = today_schedule_rows()  # list[dict]
    return render_template("index.html", games=games)

@main_bp.route("/predictions")
@login_required
def predictions():
    preds_df = predict_today_selected_model()
    preds = preds_df.to_dict(orient="records") if isinstance(preds_df, pd.DataFrame) else []
    return render_template("predictions.html", preds=preds)

@main_bp.route("/admin", methods=["GET", "POST"])
@login_required
def admin():
    if not getattr(current_user, "is_admin", False):
        flash("Admin access required.", "warning")
        return redirect(url_for("main.index"))

    models = list_models()
    sel = read_selection()
    selected_path = sel.get("model_path")

    if request.method == "POST":
        new_path = request.form.get("model_path") or ""
        if not new_path or not os.path.exists(new_path):
            flash("Invalid model selection.", "error")
            return redirect(url_for("main.admin"))
        write_selection(new_path)
        flash("Model selection saved.", "success")
        return redirect(url_for("main.admin"))

    table_rows = collect_model_rows()
    return render_template("admin.html",
                           models=models,
                           selected_path=selected_path,
                           table_rows=table_rows)

@main_bp.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        # plug in real settings handling as needed
        flash("Settings saved.", "success")
        return redirect(url_for("main.settings"))
    return render_template("settings.html")

@main_bp.route("/contact")
@login_required
def contact():
    return render_template("contact.html")
