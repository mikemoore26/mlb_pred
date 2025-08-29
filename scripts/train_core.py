# scripts/train_core.py
import os, joblib, numpy as np, pandas as pd
from typing import List, Tuple, Dict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    log_loss, brier_score_loss, precision_score, recall_score, f1_score
)

META_COLS = ["game_date", "home_team", "away_team", "home_win"]

# -------------------------
# Metrics & utils
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

def _build_xy(frame: pd.DataFrame, FEATURES: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
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

def _print_dataset_info(df: pd.DataFrame, features: List[str], tag: str):
    print(f"[{tag}] rows={len(df)}, cols={len(df.columns)}")
    if not df.empty:
        gmin, gmax = df["game_date"].min(), df["game_date"].max()
        print(f"[{tag}] game_date span: {gmin.date()} .. {gmax.date()}")
    missing = [c for c in features if c not in df.columns]
    if missing:
        print(f"[{tag}] MISSING features (filled with 0.0): {missing[:8]}{' ...' if len(missing)>8 else ''}")

# -------------------------
# Public API: train_from_df
# -------------------------
def train_from_df(df: pd.DataFrame,
                  FEATURES: List[str],
                  builder_fn,
                  start: str, end: str, val_start: str,
                  model_name: str,
                  out_path: str,
                  variant_key: str,
                  include_calibration: bool = True,
                  quick: bool = False,
                  save_copy_name: str | None = None) -> Dict:
    """
    Train + evaluate from an already-built dataframe.
    Returns the payload dict; also persists to disk at out_path.
    """

    if df is None or df.empty:
        raise SystemExit("Empty dataframe passed to train_from_df.")

    for c in META_COLS:
        if c not in df.columns:
            raise RuntimeError(f"Missing required meta column '{c}' in dataset.")

    df = _normalize_game_date(df)
    df = df.sort_values("game_date").reset_index(drop=True)

    _print_dataset_info(df, FEATURES, tag="ALL")

    val_start_dt = pd.to_datetime(val_start)
    train_df = df[df["game_date"] < val_start_dt].copy()
    val_df   = df[df["game_date"] >= val_start_dt].copy()
    if train_df.empty or val_df.empty:
        n_all = len(df)
        gmin, gmax = (df["game_date"].min().date(), df["game_date"].max().date()) if n_all else ("-", "-")
        raise SystemExit(
            "Train/val split produced empty sets.\n"
            f"  val_start={val_start_dt.date()}  | data span={gmin}..{gmax}  | n_all={n_all}"
        )

    _print_dataset_info(train_df, FEATURES, tag="TRAIN")
    _print_dataset_info(val_df,   FEATURES, tag="VAL")

    X_train, y_train = _build_xy(train_df, FEATURES)
    X_val,   y_val   = _build_xy(val_df, FEATURES)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    if quick:
        from sklearn.linear_model import LogisticRegression
        base_estimator = LogisticRegression(max_iter=500, n_jobs=None, solver="lbfgs")
        calib_method = "sigmoid"
        print("[QUICK] Using fast LogisticRegression baseline.")
    else:
        base_estimator, calib_method = builder_fn()

    if include_calibration:
        base_estimator.fit(X_train, y_train)
        calib = CalibratedClassifierCV(estimator=base_estimator, method=calib_method, cv="prefit")
        calib.fit(X_val, y_val)
        proba_tr = calib.predict_proba(X_train)[:, 1]
        proba_va = calib.predict_proba(X_val)[:, 1]
        fitted = calib
    else:
        print("[FAST] Fitting without calibration.")
        base_estimator.fit(X_train, y_train)
        proba_tr = base_estimator.predict_proba(X_train)[:, 1]
        proba_va = base_estimator.predict_proba(X_val)[:, 1]
        fitted = base_estimator

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
        setattr(fitted, "feature_names_in_", np.array(FEATURES, dtype=object))
    except Exception:
        pass

    payload = {
        "model": fitted,
        "required_features": FEATURES,
        "trained_on": {"start": start, "end": end, "val_start": val_start},
        "calibrated": bool(include_calibration),
        "meta": {
            "variant": variant_key,
            "name": model_name,
        }
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(payload, out_path)
    print(f"\nSaved payload to: {out_path}")

    # text report
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

    if save_copy_name:
        os.makedirs("models", exist_ok=True)
        model_copy_path = os.path.join("models", save_copy_name)
        joblib.dump(fitted, model_copy_path, compress=3)
        print("Saved plain model copy to:", model_copy_path)
    return payload
