# models/registry.py
from typing import Dict, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# -------------------------
# Model builders
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

# -------------------------
# Registry
# -------------------------
MODEL_REGISTRY: Dict[str, dict] = {
    "full_model_ml1": {
        "name": "Full Model ML1 (LR+Iso)",
        "features": [
            "team_wpct_diff_season",
            "team_wpct_diff_30d",
            "home_advantage",
            "starter_era_diff",
            "starter_k9_diff",
            "starter_bb9_diff",
            "starter_era30_diff",
            "bullpen_era14_diff",
            "park_factor",
            "home_days_rest",
            "away_days_rest",
            "b2b_flag",
            "wx_temp",
            "wx_wind_speed",
            "wx_wind_out_to_cf",
        ],
        "builder": _build_lr_isotonic,
        "default_out": "models/full_model_ml1.pkl",
    },
    "full_model_ml2": {
        "name": "Full Model ML2 (GBDT+Iso)",
        "features": [
            "home_advantage",
            "team_wpct_diff_season",
            "team_wpct_diff_30d",
            "starter_era_diff",
            "starter_k9_diff",
            "bullpen_era14_diff",
            "park_factor",
            "travel_km_home_prev_to_today",
            "travel_km_away_prev_to_today",
            "bullpen_ip_last3_home",
            "bullpen_ip_last3_away",
            "offense_runs_pg_30d_diff",
        ],
        "builder": _build_gbdt_isotonic,
        "default_out": "models/full_model_ml2.pkl",
    },
}
