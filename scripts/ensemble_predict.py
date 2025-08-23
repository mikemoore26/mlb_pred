# scripts/ensemble_predict.py
import os, argparse, joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    log_loss, brier_score_loss, precision_score, recall_score, f1_score
)

def _load_payload_or_estimator(path):
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
    # fill missing
    for c in req:
        if c not in X.columns:
            X[c] = 0.0
    return X[req].apply(pd.to_numeric, errors="coerce").fillna(0.0)

def _predict_proba_one(est, X):
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)
        return p[:, 1]
    # fallback
    yhat = est.predict(X)
    return (yhat.astype(int) == 1).astype(float)

def _eval(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-6, 1-1e-6))),
        "brier": float(brier_score_loss(y_true, proba)),
        "precision": float(precision_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
    }

def _bucket_eval(y_true, proba, thresholds=(0.55, 0.60, 0.65, 0.70)):
    rows = []
    abs_p = np.where(proba >= 0.5, proba, 1 - proba)
    pred = (proba >= 0.5).astype(int)
    for t in thresholds:
        m = abs_p >= t
        cov = float(m.mean()) if m.size else 0.0
        acc = float((pred[m] == y_true[m]).mean()) if m.any() else None
        rows.append({"threshold": t, "coverage": cov, "pick_acc": acc, "n": int(len(y_true))})
    return pd.DataFrame(rows)

def main(args):
    est_a, req_a = _load_payload_or_estimator(args.model_a)
    est_b, req_b = _load_payload_or_estimator(args.model_b)

    df = pd.read_csv(args.data)
    if args.label not in df.columns:
        raise SystemExit(f"Label '{args.label}' not in dataset.")
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

    m = _eval(y, p_ens)
    print("\nEnsemble validation metrics:")
    for k, v in m.items():
        print(f"  {k}: {v:.3f}")
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
    print(f"\nSaved predictions: {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True)
    ap.add_argument("--model_b", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default="home_win")
    ap.add_argument("--w_a", type=float, default=0.5)
    ap.add_argument("--w_b", type=float, default=0.5)
    ap.add_argument("--gate_b", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(args)


"""
python scripts/ensemble_predict.py --model_a models/model_A.pkl --model_b models/model_B.pkl --data PATH/TO/data.csv --label home_win --w_a 0.5 --w_b 0.5 --gate_b 0.0 --out ensemble_preds.csv


"""