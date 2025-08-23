#!/usr/bin/env python
"""
Unified MLB CLI:
  - train:    train a model variant and save a calibrated payload
  - ensemble: evaluate/produce predictions from two models combined

Examples:
  Train:
    python -m scripts.ml train \
      --variant full_model_ml2 \
      --start 2024-03-28 --end 2025-08-12 --val_start 2025-06-15 \
      --out models/full_model_ml2.pkl

  Ensemble on a CSV:
    python -m scripts.ml ensemble \
      --model_a models/full_model_ml1.pkl \
      --model_b models/full_model_ml2.pkl \
      --data data/validation.csv --label home_win \
      --w_a 0.6 --w_b 0.4 --out ensemble_preds.csv
"""
import os
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import date
from typing import Dict, List, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    log_loss, brier_score_loss, precision_score, recall_score, f1_score
)

# -------------------------
# Data builder
# -------------------------
try:
    from utils.data_fetchers import fetch_games_with_features
except Exception as e:
    raise SystemExit(
        f"[FATAL] Could not import data fetcher: {e}\n"
        "Run from project root, e.g. 'python -m scripts.ml train ...'"
    )

META_COLS = ["game_date", "home_team", "away_team", "home_win"]

# -------------------------
# Model builders & registry
# -------------------------
def _build_lr_isotonic() -> Tuple[Pipeline, str]:
    base = Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))
    ])
    return base, "isotonic"

def _build_gbdt_isotonic() -> Tuple[GradientBoostingClassifier, str]:
    base = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=600,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
    return base, "isotonic"

MODEL_REGISTRY: Dict[str, dict] = {
    "full_model_ml1": {
        "name": "Full Model ML1 (LR+Iso)",
        "features": [
            "team_wpct_diff_season", "team_wpct_diff_30d", "home_advantage",
            "starter_era_diff", "starter_k9_diff", "starter_bb9_diff",
            "starter_era30_diff", "bullpen_era14_diff", "park_factor",
            "home_days_rest", "away_days_rest", "b2b_flag",
            "wx_temp", "wx_wind_speed", "wx_wind_out_to_cf",
        ],
        "builder": _build_lr_isotonic,
        "default_out": "models/full_model_ml1.pkl",
    },
    "full_model_ml2": {
        "name": "Full Model ML2 (GBDT+Iso)",
        "features": [
            "home_advantage", "team_wpct_diff_season", "team_wpct_diff_30d",
            "starter_era_diff", "starter_k9_diff", "bullpen_era14_diff",
            "park_factor", "travel_km_home_prev_to_today",
            "travel_km_away_prev_to_today", "bullpen_ip_last3_home",
            "bullpen_ip_last3_away", "offense_runs_pg_30d_diff",
        ],
        "builder": _build_gbdt_isotonic,
        "default_out": "models/full_model_ml2.pkl",
    },
}

# -------------------------
# Shared utils (eval/align)
# -------------------------
def _evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_true, proba)),
        "precision": float(precision_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
    }

def _bucket_eval(y_true: np.ndarray, proba: np.ndarray,
                 thresholds=(0.55, 0.60, 0.65, 0.70)) -> pd.DataFrame:
    rows = []
    abs_p = np.where(proba >= 0.5, proba, 1 - proba)
    picks = (proba >= 0.5).astype(int)
    for t in thresholds:
        mask = abs_p >= t
        cov = float(mask.mean()) if mask.size else 0.0
        acc = float((picks[mask] == y_true[mask]).mean()) if mask.any() else None
        rows.append({"threshold": t, "coverage": cov, "pick_acc": acc, "n": int(len(y_true))})
    return pd.DataFrame(rows)

def _normalize_game_date(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    low = {c.lower(): c for c in df.columns}
    if "gamedate" in low and "game_date" not in df.columns:
        rename_map[low["gamedate"]] = "game_date"
    if "date" in low and "game_date" not in df.columns:
        rename_map[low["date"]] = "game_date"
    if rename_map:
        df = df.rename(columns=rename_map)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["game_date"]).reset_index(drop=True)
    if before - len(df):
        print(f"[WARN] Dropped {before - len(df)} rows without valid game_date.")
    return df

def _build_xy(frame: pd.DataFrame, FEATURES: List[str]):
    X = frame.copy()
    for col in FEATURES:
        if col not in X.columns:
            print(f"[WARN] Missing feature '{col}', filling with 0.0")
            X[col] = 0.0
    X = X[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    target_candidates = ["home_win", "label", "target", "y"]
    tcol = next((c for c in target_candidates if c in frame.columns), None)
    if tcol is None:
        raise ValueError(f"Could not find target column. Got: {list(frame.columns)}")

    y = pd.to_numeric(frame[tcol], errors="coerce").fillna(0).astype(int).to_numpy().ravel()
    return X, y

def _load_payload_or_estimator(path: str):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        est = obj["model"]
        rf = obj.get("required_features") or list(getattr(est, "feature_names_in_", []))
    else:
        est = obj
        rf = list(getattr(est, "feature_names_in_", []))
    return est, list(rf or [])

def _align_features(X_all: pd.DataFrame, req: list[str]) -> pd.DataFrame:
    X = X_all.copy()
    # shims for older/newer names
    if "home_is_home" in req and "home_is_home" not in X.columns and "home_advantage" in X.columns:
        X["home_is_home"] = X["home_advantage"]
    if "team_wpct_diff" in req and "team_wpct_diff" not in X.columns and "team_wpct_diff_season" in X.columns:
        X["team_wpct_diff"] = X["team_wpct_diff_season"]
    if "bullpen_era_diff_14d" in req and "bullpen_era_diff_14d" not in X.columns and "bullpen_era14_diff" in X.columns:
        X["bullpen_era_diff_14d"] = X["bullpen_era14_diff"]
    for c in req:
        if c not in X.columns:
            X[c] = 0.0
    return X[req].apply(pd.to_numeric, errors="coerce").fillna(0.0)

def _predict_proba_one(est, X):
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)
        return p[:, 1]
    yhat = est.predict(X)
    return (yhat.astype(int) == 1).astype(float)

# -------------------------
# Subcommand: train
# -------------------------
def cmd_train(args):
    if args.variant not in MODEL_REGISTRY:
        raise SystemExit(f"[FATAL] Unknown variant '{args.variant}'. Options: {list(MODEL_REGISTRY.keys())}")

    spec = MODEL_REGISTRY[args.variant]
    FEATURES = spec["features"]
    model_name = spec["name"]
    default_out = spec["default_out"]
    builder = spec["builder"]

    out_path = args.out or default_out
    save_copy_name = args.save_copy_name or f"{args.variant}_plain.pkl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    request_cols = list(dict.fromkeys(FEATURES + META_COLS))
    print(f"[INFO] Fetching games {args.start} .. {args.end} for {model_name}")
    df = fetch_games_with_features(
        start_date=args.start,
        end_date=args.end,
        include_odds=args.include_odds,
        include_weather=True,
        required_features=request_cols
    )
    if df is None or df.empty:
        raise SystemExit("[FATAL] No games fetched. Adjust your date range or fetcher.")

    df = df.loc[:, ~df.columns.duplicated()].copy()
    for c in META_COLS:
        if c not in df.columns:
            raise RuntimeError(f"[FATAL] Missing required meta column '{c}'")

    df = _normalize_game_date(df).sort_values("game_date").reset_index(drop=True)

    val_start_dt = pd.to_datetime(args.val_start)
    train_df = df[df["game_date"] < val_start_dt].copy()
    val_df   = df[df["game_date"] >= val_start_dt].copy()
    if train_df.empty or val_df.empty:
        raise SystemExit("[FATAL] Train/val split produced empty sets. Check --val_start.")

    X_train, y_train = _build_xy(train_df, FEATURES)
    X_val,   y_val   = _build_xy(val_df, FEATURES)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    base_estimator, calib_method = builder()
    base_estimator.fit(X_train, y_train)
    calib = CalibratedClassifierCV(estimator=base_estimator, method=calib_method, cv="prefit")
    calib.fit(X_val, y_val)

    proba_va = calib.predict_proba(X_val)[:, 1]
    m_va = _evaluate(y_val, proba_va)
    buckets = _bucket_eval(y_val, proba_va)

    print(f"\n[{model_name}] Validation metrics:")
    for k, v in m_va.items():
        print(f"  {k:>12}: {v:.3f}")
    print("\nCoverage/Accuracy by confidence threshold:")
    print(buckets.to_string(index=False))

    payload = {
        "model": calib,
        "required_features": FEATURES,
        "trained_on": {"start": args.start, "end": args.end, "val_start": args.val_start},
        "calibrated": True,
        "meta": {"created": date.today().isoformat(), "variant": args.variant, "name": model_name}
    }
    joblib.dump(payload, out_path)
    print(f"[OK] Saved calibrated payload -> {out_path}")

    # save plain copy
    os.makedirs("models", exist_ok=True)
    model_copy_path = os.path.join("models", save_copy_name)
    joblib.dump(calib, model_copy_path, compress=3)
    print(f"[OK] Saved plain model copy -> {model_copy_path}")

    pd.Series(FEATURES, name="features").to_csv(os.path.join("models", f"{args.variant}_features.csv"), index=False)
    print(f"[OK] Saved feature list -> models/{args.variant}_features.csv")

# -------------------------
# Subcommand: ensemble
# -------------------------
def cmd_ensemble(args):
    est_a, req_a = _load_payload_or_estimator(args.model_a)
    est_b, req_b = _load_payload_or_estimator(args.model_b)

    df = pd.read_csv(args.data)
    if args.label not in df.columns:
        raise SystemExit(f"[FATAL] Label '{args.label}' not in dataset.")
    y = df[args.label].astype(int).to_numpy()

    Xa = _align_features(df, req_a)
    Xb = _align_features(df, req_b)
    pa = _predict_proba_one(est_a, Xa)
    pb = _predict_proba_one(est_b, Xb)

    if args.gate_b > 0:
        conf_b = np.maximum(pb, 1 - pb)
        w_b = (conf_b >= args.gate_b).astype(float) * args.w_b
        w_a = 1.0 - w_b
        p_ens = w_a * pa + w_b * pb
    else:
        s = args.w_a + args.w_b
        w_a = args.w_a / s if s > 0 else 0.5
        w_b = args.w_b / s if s > 0 else 0.5
        p_ens = w_a * pa + w_b * pb

    m = _evaluate(y, p_ens)
    print("\nEnsemble validation metrics:")
    for k, v in m.items():
        print(f"  {k:>12}: {v:.3f}")
    print("\nCoverage/Accuracy by confidence threshold:")
    print(_bucket_eval(y, p_ens).to_string(index=False))

    out = args.out or "ensemble_preds.csv"
    out_df = df.copy()
    out_df["proba_A"] = pa
    out_df["proba_B"] = pb
    out_df["proba_ens"] = p_ens
    out_df["pick"] = np.where(p_ens >= 0.5, "HOME", "AWAY")
    out_df["pick_prob"] = np.where(p_ens >= 0.5, p_ens, 1 - p_ens)
    out_df.to_csv(out, index=False)
    print(f"[OK] Saved predictions: {out}")

# -------------------------
# CLI
# -------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="MLB CLI: train and ensemble")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train
    ap_tr = sub.add_parser("train", help="Train a model variant")
    ap_tr.add_argument("--variant", required=True, help=f"Options: {list(MODEL_REGISTRY.keys())}")
    ap_tr.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    ap_tr.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    ap_tr.add_argument("--val_start", required=True, help="YYYY-MM-DD (first day of validation window)")
    ap_tr.add_argument("--out", default=None, help="Output pickle path (payload). Defaults per variant.")
    ap_tr.add_argument("--include_odds", action="store_true", help="Include odds if fetcher supports it.")
    ap_tr.add_argument("--save_copy_name", default=None, help="Also save plain estimator to models/<name>.pkl")
    ap_tr.set_defaults(func=cmd_train)

    # ensemble
    ap_en = sub.add_parser("ensemble", help="Ensemble two models on a CSV dataset")
    ap_en.add_argument("--model_a", required=True)
    ap_en.add_argument("--model_b", required=True)
    ap_en.add_argument("--data", required=True, help="CSV file with features + label column")
    ap_en.add_argument("--label", default="home_win")
    ap_en.add_argument("--w_a", type=float, default=0.5)
    ap_en.add_argument("--w_b", type=float, default=0.5)
    ap_en.add_argument("--gate_b", type=float, default=0.0)
    ap_en.add_argument("--out", default=None)
    ap_en.set_defaults(func=cmd_ensemble)

    return ap

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
