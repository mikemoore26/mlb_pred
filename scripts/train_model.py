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

# scripts/train_model.py
import os, argparse, joblib, numpy as np, pandas as pd
from datetime import date
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle

# Feature fetcher
from utils.data_fetchers import fetch_games_with_features
# Import central registry
from models.registry import MODEL_REGISTRY


# Core feature builder
try:
    from utils.data_fetchers import fetch_games_with_features
except Exception as e:
    raise SystemExit(
        f"Import failed: {e}\nRun from project root, e.g. 'python -m scripts.train_model ...'"
    )

META_COLS = ["game_date", "home_team", "away_team", "home_win"]



# -------------------------
# Utilities
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
            X[col] = 0.0
    X = X[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    target_candidates = ["home_win", "label", "target", "y"]
    tcol = next((c for c in target_candidates if c in frame.columns), None)
    if tcol is None:
        lower_map = {c.lower(): c for c in frame.columns}
        for k in target_candidates:
            if k in lower_map:
                tcol = lower_map[k]
                break
    if tcol is None:
        raise ValueError(f"Could not find target column among {target_candidates}. "
                         f"Got columns: {list(frame.columns)}")

    col_obj = frame[tcol]
    if getattr(col_obj, "ndim", 1) > 1:
        col_obj = col_obj.iloc[:, 0]
    y = pd.to_numeric(col_obj, errors="coerce").fillna(0).astype(int).to_numpy().ravel()
    if y.ndim != 1:
        raise ValueError(f"Target y must be 1-D. Got shape {y.shape} from column '{tcol}'.")
    return X, y


# -------------------------
# Trainer
# -------------------------
def train_variant(variant_key: str,
                  start: str, end: str, val_start: str,
                  include_odds: bool, out_path: str,
                  save_copy_name: str | None = None) -> None:

    if variant_key not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown variant '{variant_key}'. Options: {list(MODEL_REGISTRY.keys())}")

    spec = MODEL_REGISTRY[variant_key]
    FEATURES = spec["features"]
    model_name = spec["name"]
    default_out = spec["default_out"]
    builder = spec["builder"]

    out_path = out_path or default_out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    request_cols = list(dict.fromkeys(FEATURES + META_COLS))
    print(f"[{model_name}] Fetching games {start} .. {end}")
    df = fetch_games_with_features(
        start_date=start,
        end_date=end,
        include_odds=include_odds,
        include_weather=True,
        required_features=request_cols
    )
    if df is None or df.empty:
        raise SystemExit("No games fetched. Adjust your date range or fetcher.")

    df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    for c in META_COLS:
        if c not in df.columns:
            raise RuntimeError(f"Missing required meta column '{c}' in dataset.")

    df = _normalize_game_date(df)
    df = df.sort_values("game_date").reset_index(drop=True)

    val_start_dt = pd.to_datetime(val_start)
    train_df = df[df["game_date"] < val_start_dt].copy()
    val_df   = df[df["game_date"] >= val_start_dt].copy()
    if train_df.empty or val_df.empty:
        raise SystemExit("Train/val split produced empty sets. Check --val_start.")

    X_train, y_train = _build_xy(train_df, FEATURES)
    X_val,   y_val   = _build_xy(val_df, FEATURES)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    base_estimator, calib_method = builder()
    base_estimator.fit(X_train, y_train)
    calib = CalibratedClassifierCV(estimator=base_estimator, method=calib_method, cv="prefit")
    calib.fit(X_val, y_val)

    proba_tr = calib.predict_proba(X_train)[:, 1]
    proba_va = calib.predict_proba(X_val)[:, 1]
    m_tr = _evaluate(y_train, proba_tr)
    m_va = _evaluate(y_val, proba_va)
    buckets = _bucket_eval(y_val, proba_va)

    print(f"\n[{model_name}]")
    print(f"Window: {start} .. {end}")
    print(f"Train N: {len(y_train)} | Val N: {len(y_val)}")
    print("\nValidation metrics:")
    for k, v in m_va.items():
        print(f"  {k}: {v:.3f}")
    print("\nCoverage/Accuracy by confidence threshold:")
    print(buckets.to_string(index=False))

    try:
        setattr(calib, "feature_names_in_", np.array(FEATURES, dtype=object))
    except Exception:
        pass

    payload = {
        "model": calib,
        "required_features": FEATURES,
        "trained_on": {"start": start, "end": end, "val_start": val_start},
        "calibrated": True,
        "meta": {
            "created": date.today().isoformat(),
            "variant": variant_key,
            "name": model_name
        }
    }
    joblib.dump(payload, out_path)
    print(f"\nSaved calibrated payload to: {out_path}")

    report_path = os.path.splitext(out_path)[0] + "_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{model_name}\n")
        f.write(f"Variant: {variant_key}\n")
        f.write(f"Window: {start} .. {end}\n")
        f.write(f"Train N: {len(y_train)} | Val N: {len(y_val)}\n\n")
        f.write("Validation metrics:\n")
        for k, v in m_va.items():
            f.write(f"{k}: {v:.3f}\n")
        f.write("\nCoverage/Accuracy by confidence threshold:\n")
        f.write(buckets.to_string(index=False))
        f.write("\n")
    print(f"Wrote report: {report_path}")

    os.makedirs("models", exist_ok=True)
    if save_copy_name:
        model_copy_path = os.path.join("models", save_copy_name)
        joblib.dump(calib, model_copy_path, compress=3)
        print("Saved plain model copy to:", model_copy_path)
    pd.Series(FEATURES).to_csv(os.path.join("models", f"{variant_key}_features.csv"),
                               index=False, header=False)
    print("Saved feature list CSV ->", os.path.join("models", f"{variant_key}_features.csv"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Unified trainer for MLB models (LR/GBDT + isotonic calibration)."
    )
    ap.add_argument("--variant", required=True,
                    help=f"Which model to train. Options: {list(MODEL_REGISTRY.keys())}")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--val_start", required=True, help="YYYY-MM-DD (first day of validation window)")
    ap.add_argument("--out", default=None, help="Output pickle path (payload). Defaults per variant.")
    ap.add_argument("--include_odds", action="store_true",
                    help="If your fetcher populates closing odds, include them in training.")
    ap.add_argument("--save_copy_name", default=None,
                    help="Optional: also save a plain estimator copy to models/<name>.pkl")
    args = ap.parse_args()

    train_variant(
        variant_key=args.variant,
        start=args.start,
        end=args.end,
        val_start=args.val_start,
        include_odds=args.include_odds,
        out_path=args.out,
        save_copy_name=args.save_copy_name,
    )


"""
python -m scripts.train_model --variant full_model_ml1 --start 2024-03-28 --end 2025-08-12 --val_start 2025-06-15 --out models/full_model_ml1.pkl
python -m scripts.train_model --variant full_model_ml2 --start 2024-03-28 --end 2025-08-12 --val_start 2025-06-15 --out models/full_model_ml2.pkl


"""