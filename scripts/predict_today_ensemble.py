# scripts/predict_today_ensemble.py
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import date
from utils.feature_builder import build_features_for_range as fetch_games_with_features

META_COLS = ["game_date", "home_team", "away_team"]

def _load(path):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        est = obj["model"]
        feats = obj.get("required_features") or list(getattr(est, "feature_names_in_", []))
    else:
        est = obj
        feats = list(getattr(est, "feature_names_in_", []))
    if not feats:
        raise SystemExit(f"Model at {path} is missing required_features/feature_names_in_.")
    return est, feats

def _align(df, req):
    X = df.copy()
    for c in req:
        if c not in X.columns:
            X[c] = 0.0
    return X[req].apply(pd.to_numeric, errors="coerce").fillna(0.0)

def _predict(est, X):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    p = est.predict(X).astype(float)
    return np.clip(p, 0.0, 1.0)

def main(args):
    day = args.date or date.today().isoformat()

    # Build features for **today**, include scheduled games
    df = fetch_games_with_features(
        start_date=day,
        end_date=day,
        include_odds=False,
        include_weather=True,
        required_features=None,
        only_finals=False,   # important for prediction
    )
    if df.empty:
        print("No scheduled games found for today.")
        return

    # Load both models
    A, reqA = _load(args.model_a)
    B, reqB = _load(args.model_b)

    # Align features + predict
    XA = _align(df, reqA); pA = _predict(A, XA)
    XB = _align(df, reqB); pB = _predict(B, XB)

    wa, wb = float(args.wa), float(args.wb)
    if wa < 0 or wb < 0 or (wa == 0 and wb == 0):
        wa, wb = 0.5, 0.5
    # normalize for the blend branch
    s = wa + wb
    wa_n, wb_n = wa / s, wb / s

    # Gate behavior:
    # - If gate_b == 0: always blend p = wa*pA + wb*pB (normalized)
    # - If gate_b > 0: use A unless B is confident (|pB - 0.5| >= gate); then blend with wa/wb
    if args.gate_b > 0:
        confB = np.abs(pB - 0.5)
        mask = confB >= float(args.gate_b)
        p = np.where(mask, wa_n * pA + wb_n * pB, pA)
    else:
        p = wa_n * pA + wb_n * pB

    out = df[META_COLS].copy()
    out["proba_A"] = pA
    out["proba_B"] = pB
    out["proba_ens"] = p
    out["pick"] = np.where(p >= 0.5, "HOME", "AWAY")
    out["pick_prob"] = np.where(p >= 0.5, p, 1 - p)

    # Sort nicely by team names
    out = out.sort_values(["home_team", "away_team"]).reset_index(drop=True)

    print(out.to_string(index=False))
    out.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Predict today's MLB games with an ensemble of two models.")
    ap.add_argument("--model_a", required=True, help="e.g., models/full_model_ml1.pkl")
    ap.add_argument("--model_b", required=True, help="e.g., models/full_model_ml2.pkl")
    ap.add_argument("--wa", type=float, default=0.5, help="Weight for model A (used when blending)")
    ap.add_argument("--wb", type=float, default=0.5, help="Weight for model B (used when blending)")
    ap.add_argument("--gate_b", type=float, default=0.0, help="Confidence gate for model B (0–0.5).")
    ap.add_argument("--date", default=None, help="Override date YYYY-MM-DD (default: today, UTC).")
    ap.add_argument("--out", default="predictions_today_ens.csv")
    args = ap.parse_args()
    main(args)



"""

# Simple blend (always blend A & B with 60/40)
python -m scripts.predict_today_ensemble \
  --model_a models/full_model_ml1.pkl \
  --model_b models/full_model_ml2.pkl \
  --wa 0.6 --wb 0.4

# Gate model B unless it’s pretty confident (>= 0.65 away from 0.5)
python -m scripts.predict_today_ensemble --model_a models/full_model_ml1.pkl --model_b models/full_model_ml2.pkl --wa 0.5 --wb 0.5 --gate_b 0.15






"""