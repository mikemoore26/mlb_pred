# utils/data_fetchers/elo.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np

# ---- Tunables ----
ELO_MEAN   = 1500.0      # base rating
K_BASE     = 20.0        # base K-factor
HOME_ADV   = 35.0        # home-field Elo points
REGRESS_PCT = 0.25       # off-season regression toward mean (25%)

# Persist Elo state under cache/state/
try:
    from utils.cache import STATE_DIR
    STATE_PATH = Path(STATE_DIR) / "elo_state.json"
except Exception:
    STATE_PATH = Path("cache/state/elo_state.json")


# ---------- helpers ----------
def _k_with_margin(k_base: float, home_runs: float | None, away_runs: float | None) -> float:
    """Gentle margin-of-victory scaling."""
    if home_runs is None or away_runs is None:
        return float(k_base)
    margin = abs(float(home_runs) - float(away_runs))
    return float(k_base * (1.0 + np.log2(margin + 1.0) / 5.0))

def _expected(elo_home_adj: float, elo_away_adj: float) -> float:
    """Elo logistic expected score for home side."""
    return 1.0 / (1.0 + 10 ** (-(elo_home_adj - elo_away_adj) / 400.0))

def _season_of(ts: pd.Timestamp) -> int:
    return int(pd.to_datetime(ts).year)

def _load_state() -> Dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"ratings": {}, "last_date": None, "last_season": None}

def _save_state(state: Dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))

def _ensure_team(state: Dict, team: str) -> None:
    state["ratings"].setdefault(team, ELO_MEAN)

def _regress_offseason(state: Dict) -> None:
    for t, r in list(state["ratings"].items()):
        state["ratings"][t] = float(ELO_MEAN + (r - ELO_MEAN) * (1.0 - REGRESS_PCT))

def _infer_home_win(row: pd.Series) -> int:
    """Return 1 if home won, 0 if lost. Requires home_win or both runs."""
    if "home_win" in row and pd.notna(row["home_win"]):
        return int(row["home_win"])
    if {"home_runs", "away_runs"} <= set(row.index):
        return int(float(row["home_runs"]) > float(row["away_runs"]))
    raise ValueError("Need 'home_win' or both 'home_runs' and 'away_runs' to update Elo.")


# ---------- public API ----------
def cc_elo_from_games(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute/extend Elo from finalized games.
    Returns per-row pregame Elo & win prob:

      ['elo_home','elo_away','elo_diff','elo_prob_home']

    Required game columns:
      - 'game_date','home_team','away_team'
      - and either 'home_win' OR ('home_runs','away_runs')
    """
    if games is None or games.empty:
        return pd.DataFrame(index=(games.index if games is not None else None))

    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce")
    g = g.dropna(subset=["game_date"]).sort_values("game_date").reset_index(drop=True)

    n = len(g)
    elo_home = np.zeros(n, dtype=float)
    elo_away = np.zeros(n, dtype=float)
    elo_diff = np.zeros(n, dtype=float)
    elo_prob = np.zeros(n, dtype=float)

    state = _load_state()
    last_date = pd.to_datetime(state["last_date"]) if state["last_date"] else None
    last_season = state.get("last_season")

    for i, row in g.iterrows():
        dt = row["game_date"]
        season = _season_of(dt)
        ht = str(row["home_team"])
        at = str(row["away_team"])

        # Off-season regression when crossing seasons
        if last_season is not None and season > int(last_season):
            _regress_offseason(state)

        _ensure_team(state, ht)
        _ensure_team(state, at)

        eh = float(state["ratings"][ht])
        ea = float(state["ratings"][at])

        # Pregame values (HOME_ADV applied only for probability)
        p_home = _expected(eh + HOME_ADV, ea)

        elo_home[i] = eh
        elo_away[i] = ea
        elo_diff[i] = eh - ea
        elo_prob[i] = p_home

        # If this date already processed, skip updating ratings
        if last_date is not None and dt <= last_date:
            continue

        # Only update on rows with a final result
        try:
            s_home = float(_infer_home_win(row))
        except Exception:
            continue

        K = _k_with_margin(K_BASE, row.get("home_runs"), row.get("away_runs"))
        delta = K * (s_home - p_home)
        state["ratings"][ht] = eh + delta
        state["ratings"][at] = ea - delta

        state["last_date"] = dt.isoformat()
        state["last_season"] = int(season)
        last_date = dt
        last_season = int(season)

    _save_state(state)

    return pd.DataFrame({
        "elo_home": elo_home,
        "elo_away": elo_away,
        "elo_diff": elo_diff,
        "elo_prob_home": elo_prob,
    })
