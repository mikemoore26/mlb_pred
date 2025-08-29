# scripts/train_model.py
from __future__ import annotations
import os
import argparse
import pandas as pd
from pathlib import Path

from models.registry import MODEL_REGISTRY
from scripts.train_core import train_from_df
from utils.cache import get_incremental_features
from utils.cli_dates import resolve_training_dates

META_COLS = ["game_date", "home_team", "away_team", "home_win"]

def _print_model_options():
    return f"Options: {list(MODEL_REGISTRY.keys())}"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Incremental trainer (requires --variant).")
    ap.add_argument("--variant", required=True, help=f"Which model to train. {_print_model_options()}")
    ap.add_argument("--start", default="auto", help="YYYY-MM-DD | auto")
    ap.add_argument("--end", default="auto", help="YYYY-MM-DD | auto | today | yesterday")
    ap.add_argument("--val_start", default="auto", help="YYYY-MM-DD | auto")
    ap.add_argument("--out", default=None, help="Output payload path. Defaults per variant.")
    ap.add_argument("--include_odds", action="store_true", help="Include odds if available.")
    ap.add_argument("--save_copy_name", default=None, help="Also save plain estimator to models/<name>.pkl")
    ap.add_argument("--quick", action="store_true", help="Use fast baseline (LogReg).")
    ap.add_argument("--no_calibrate", action="store_true", help="Skip calibration.")
    ap.add_argument("--cache_path", default="cache/mlb_features.parquet", help="Dataset cache file.")
    # optional knobs
    ap.add_argument("--label_lag_days", type=int, default=1, help="How many days behind to set 'auto' end.")
    ap.add_argument("--val_back_days", type=int, default=30, help="Validation starts this many days before end.")
    args = ap.parse_args()

    if args.variant not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown variant '{args.variant}'. {_print_model_options()}")

    # Dates via shared helper
    start_iso, end_iso, val_start_iso = resolve_training_dates(
        args.start, args.end, args.val_start,
        default_start="2024-03-28",
        label_lag_days=args.label_lag_days,
        val_back_days=args.val_back_days,
    )
    print(f"[DATES] start={start_iso} end={end_iso} val_start={val_start_iso}")

    spec = MODEL_REGISTRY[args.variant]
    FEATURES = spec["features"]
    model_name = spec["name"]
    default_out = spec["default_out"]
    builder = spec["builder"]

    out_path = args.out or default_out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    request_cols = list(dict.fromkeys(FEATURES + META_COLS))

    # Fetch incrementally
    df_all = get_incremental_features(
        start_date=start_iso,
        end_date=end_iso,
        include_odds=args.include_odds,
        required_features=request_cols,
        include_weather=True,
        cache_path=Path(args.cache_path),
        force_until_today=False,  # respect resolved end
    )

    # Train
    _ = train_from_df(
        df=df_all,
        FEATURES=FEATURES,
        builder_fn=builder,
        start=start_iso,
        end=end_iso,
        val_start=val_start_iso,
        model_name=model_name,
        out_path=out_path,
        variant_key=args.variant,
        include_calibration=(not args.no_calibrate),
        quick=args.quick,
        save_copy_name=args.save_copy_name,
    )

    # Save feature list
    os.makedirs("models", exist_ok=True)
    pd.Series(FEATURES).to_csv(
        os.path.join("models", f"{args.variant}_features.csv"),
        index=False, header=False
    )
    print("Saved feature list CSV ->", os.path.join("models", f"{args.variant}_features.csv"))
