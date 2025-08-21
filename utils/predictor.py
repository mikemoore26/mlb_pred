# utils/predictor.py
from __future__ import annotations
import os, joblib, numpy as np, pandas as pd

def _load_payload_or_estimator(path: str):
    """Return (estimator, required_features_list). Handles payloads {'model', 'required_features'} or plain estimators."""
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        est = obj["model"]
        req = obj.get("required_features") or list(getattr(est, "feature_names_in_", []))
    else:
        est = obj
        req = list(getattr(est, "feature_names_in_", []))
    return est, (req or [])

def _align_features(df: pd.DataFrame, req: list[str]) -> pd.DataFrame:
    X = df.copy()
    # Common name shims (extend as needed)
    if "home_is_home" in req and "home_is_home" not in X.columns and "home_advantage" in X.columns:
        X["home_is_home"] = X["home_advantage"]
    if "team_wpct_diff" in req and "team_wpct_diff" not in X.columns and "team_wpct_diff_season" in X.columns:
        X["team_wpct_diff"] = X["team_wpct_diff_season"]
    if "bullpen_era_diff_14d" in req and "bullpen_era_diff_14d" not in X.columns and "bullpen_era14_diff" in X.columns:
        X["bullpen_era_diff_14d"] = X["bullpen_era14_diff"]

    # Create any missing columns as 0.0
    for c in req:
        if c not in X.columns:
            X[c] = 0.0
    return X[req].apply(pd.to_numeric, errors="coerce").fillna(0.0)

def _predict_proba(est, X: pd.DataFrame) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)
        return p[:, 1]
    # fallback: label -> probability 0/1
    yhat = est.predict(X)
    return (np.asarray(yhat).astype(int) == 1).astype(float)

def predict_with_model(df_features: pd.DataFrame, model_path: str) -> pd.DataFrame:
    est, req = _load_payload_or_estimator(model_path)
    X = _align_features(df_features, req)
    proba = _predict_proba(est, X)
    out = df_features.copy()
    out["proba"] = proba
    out["pick_side"] = np.where(proba >= 0.5, "HOME", "AWAY")
    out["pick_prob"] = np.where(proba >= 0.5, proba, 1 - proba)
    return out

def predict_with_ensemble(
    df_features: pd.DataFrame,
    model_a_path: str,
    model_b_path: str,
    w_a: float = 0.5,
    w_b: float = 0.5,
    gate_b: float = 0.0,  # if >0, only trust model B when its confidence >= gate
) -> pd.DataFrame:
    ea, req_a = _load_payload_or_estimator(model_a_path)
    eb, req_b = _load_payload_or_estimator(model_b_path)

    Xa = _align_features(df_features, req_a)
    Xb = _align_features(df_features, req_b)

    pa = _predict_proba(ea, Xa)
    pb = _predict_proba(eb, Xb)

    if gate_b and gate_b > 0:
        conf_b = np.maximum(pb, 1 - pb)
        use_b = (conf_b >= gate_b).astype(float)
        # when B is not used, A gets full weight
        p = (1 - use_b) * pa + use_b * pb
    else:
        s = (w_a + w_b) if (w_a + w_b) > 0 else 1.0
        w_a, w_b = w_a / s, w_b / s
        p = w_a * pa + w_b * pb

    out = df_features.copy()
    out["proba_A"] = pa
    out["proba_B"] = pb
    out["proba_ens"] = p
    out["pick_side"] = np.where(p >= 0.5, "HOME", "AWAY")
    out["pick_prob"] = np.where(p >= 0.5, p, 1 - p)
    return out
