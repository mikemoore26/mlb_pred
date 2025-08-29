# utils/datasets.py
from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np

# Your existing unified feature fetcher (already returns engineered data)
from utils.feature_builder import fetch_games_with_features
# Reuse your existing transformer used by training
from scripts.train_core import _build_xy

META_COLS = ["game_date", "home_team", "away_team", "home_win"]

def get_raw_df(start: str, end: str) -> pd.DataFrame:
    """
    'Raw' baseline table. If you have lower-level cc_* fetchers you prefer,
    wire them here. For now we return minimal meta via feature_builder
    (no odds/weather, only META_COLS) as the stable, early-stage view.
    """
    df = fetch_games_with_features(
        start_date=start,
        end_date=end,
        include_odds=False,
        include_weather=False,
        required_features=META_COLS
    )
    return df

def get_engineered_df(start: str, end: str,
                      required_features: List[str],
                      include_odds: bool = True,
                      include_weather: bool = True) -> pd.DataFrame:
    req = list(dict.fromkeys(required_features))
    df = fetch_games_with_features(
        start_date=start,
        end_date=end,
        include_odds=include_odds,
        include_weather=include_weather,
        required_features=req
    )
    return df

def get_transformed_xy(df: pd.DataFrame, FEATURES: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Numeric X (pandas DataFrame) + y (numpy array).
    """
    X, y = _build_xy(df, FEATURES)
    return X, y
