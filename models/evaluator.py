import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from utils.model_registry import load_latest_model

def evaluate_latest_model(model_path: str, df: pd.DataFrame):
    model, _ = load_latest_model()

    features = [c for c in df.columns if c not in ("home_win", "game_date")]
    X = df[features]
    y = df["home_win"]

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    print(f"Evaluation — Accuracy: {acc:.3f}, AUC: {auc:.3f}")
    return acc, auc
