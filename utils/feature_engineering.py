import pandas as pd

def build_features(df: pd.DataFrame, include_odds: bool, include_weather: bool) -> pd.DataFrame:
    """
    Build engineered features. Stub implementation: extend with your logic.
    """
    df["home_advantage"] = 1.0  # placeholder
    df["team_wpct_diff_season"] = df.get("home_win", 0) * 0  # placeholder

    if include_weather:
        df["wx_temp"] = 72
        df["wx_wind_speed"] = 5
        df["wx_wind_out_to_cf"] = 0

    return df
