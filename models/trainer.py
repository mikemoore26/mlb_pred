import pandas as pd
from xgboost import XGBClassifier
from utils.model_registry import save_model

def train_and_save_model(df: pd.DataFrame, model_name: str = "xgb_default"):
    features = [c for c in df.columns if c not in ("home_win", "game_date")]
    X = df[features]
    y = df["home_win"]

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y)

    path = save_model(model, model_name)
    print(f"Model saved at {path}")
    return path
