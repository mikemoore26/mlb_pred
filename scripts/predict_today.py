# scripts/predict_today.py
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from utils.feature_builder import build_features_for_range as fetch_games_with_features

META_COLS = ["game_date", "home_team", "away_team"]

def _load_model(path):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        est = obj["model"]
        req = obj.get("required_features") or list(getattr(est, "feature_names_in_", []))
    else:
        est = obj
        req = list(getattr(est, "feature_names_in_", []))
    if not req:
        raise SystemExit(f"Model at {path} is missing required_features / feature_names_in_.")
    return est, req

def _align(df, req):
    X = df.copy()
    for c in req:
        if c not in X.columns:
            X[c] = 0.0
    return X[req].apply(pd.to_numeric, errors="coerce").fillna(0.0)

def _predict(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    # fallback for regressors or margin models
    p = est.predict(X).astype(float)
    # clip into [0,1] just in case
    return np.clip(p, 0.0, 1.0)

def main(args):
    # yyyy-mm-dd in UTC (good enough for daily slate)
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # pull **scheduled/live+final** games for today
    df = fetch_games_with_features(
        start_date=today,
        end_date=today,
        include_odds=False,
        include_weather=True,
        required_features=None,   # build everything; we’ll align to model
        only_finals=False,        # <-- important for prediction
    )

    if df.empty:
        print("No scheduled games found for today.")
        return

    # primary model
    est_a, req_a = _load_model(args.model)
    Xa = _align(df, req_a)
    pa = _predict(est_a, Xa)

    # optional second model for an ensemble
    if args.model_b:
        est_b, req_b = _load_model(args.model_b)
        Xb = _align(df, req_b)
        pb = _predict(est_b, Xb)

        # optional gate: only blend when model B is confident (|p-0.5| >= gate_b)
        conf_b = np.abs(pb - 0.5)
        use_b = conf_b >= args.gate_b

        # weighted blend
        w_a = float(args.w_a)
        w_b = float(args.w_b)
        p = np.where(use_b, w_a * pa + w_b * pb, pa)
    else:
        p = pa

    out = df[META_COLS].copy()
    out["proba_home_win"] = p
    out["pick"] = np.where(p >= 0.5, "HOME", "AWAY")
    out["pick_prob"] = np.where(p >= 0.5, p, 1 - p)

    # nice print + save
    print(out.to_string(index=False))
    out.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict today's MLB games.")
    ap.add_argument("--model", required=True, help="Path to primary model (payload or estimator).")
    ap.add_argument("--model_b", default=None, help="Optional: second model for blending.")
    ap.add_argument("--w_a", type=float, default=0.5, help="Weight for model A (when blending).")
    ap.add_argument("--w_b", type=float, default=0.5, help="Weight for model B (when blending).")
    ap.add_argument("--gate_b", type=float, default=0.0, help="Confidence gate for model B (0–0.5).")
    ap.add_argument("--out", default="predictions_today.csv", help="Output CSV.")
    args = ap.parse_args()
    main(args)



"""
python -m scripts.predict_today --model models/full_model_ml1.pkl

"""