# scripts/get_data.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from utils.cli_dates import resolve_training_dates
from utils.datasets import get_raw_df, get_engineered_df, get_transformed_xy, META_COLS
from utils.cache import get_incremental_features
from models.registry import MODEL_REGISTRY  # only used when --variant is provided


def _load_features_list(variant: str | None, features_csv: str | None) -> list[str]:
    """
    Determine the FEATURES list for transformed mode.
    Priority: --features_csv > --variant
    """
    if features_csv:
        feats = pd.read_csv(features_csv, header=None).iloc[:, 0].astype(str).tolist()
        if not feats:
            raise SystemExit(f"--features_csv provided, but no features found in {features_csv}")
        return feats

    if variant:
        if variant not in MODEL_REGISTRY:
            raise SystemExit(f"Unknown variant '{variant}'. Options: {list(MODEL_REGISTRY.keys())}")
        return list(MODEL_REGISTRY[variant]["features"])

    raise SystemExit("For --mode transformed, provide either --features_csv or --variant.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fetch MLB data at three stages: raw, engineered (cached incrementally), or transformed (X,y)."
    )
    ap.add_argument("--mode", choices=["raw", "engineered", "transformed"], default="engineered",
                    help="Which layer to materialize. Default: engineered")
    ap.add_argument("--start", default="2024-03-28", help="YYYY-MM-DD | auto")
    ap.add_argument("--end", default="yesterday", help="YYYY-MM-DD | auto | today | yesterday")
    ap.add_argument("--include_odds", action="store_true", help="Include odds (engineered only).")
    ap.add_argument("--include_weather", action="store_true", help="Include weather (engineered only).")
    ap.add_argument("--cache_path", default="cache/mlb_features.parquet",
                    help="Where to store/read the engineered dataset cache (.parquet or .csv).")
    ap.add_argument("--out", default=None,
                    help="Output file path. If omitted, a sensible default is used per mode.")

    # transformed-mode options
    ap.add_argument("--variant", default=None,
                    help="Model variant to load feature list from MODEL_REGISTRY (transformed mode).")
    ap.add_argument("--features_csv", default=None,
                    help="CSV file with one feature name per line (overrides --variant).")

    # knobs shared with training date logic
    ap.add_argument("--label_lag_days", type=int, default=1,
                    help="How many days behind to set 'auto' end.")
    ap.add_argument("--val_back_days", type=int, default=30,
                    help="Unused here; kept for compatibility.")

    args = ap.parse_args()

    # We reuse the same resolver; val_start is irrelevant here.
    start_iso, end_iso, _ = resolve_training_dates(
        args.start, args.end, "auto",
        default_start="2024-03-28",
        label_lag_days=args.label_lag_days,
        val_back_days=args.val_back_days,
    )
    print(f"[DATES] start={start_iso} end={end_iso}")

    # ---------- RAW ----------
    if args.mode == "raw":
        df_raw = get_raw_df(start_iso, end_iso)
        if df_raw is None or df_raw.empty:
            raise SystemExit("No rows returned for RAW.")

        # Output path
        out = args.out or "cache/raw_games.parquet"
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        if out_path.suffix.lower() == ".csv":
            df_raw.to_csv(out_path, index=False)
        else:
            df_raw.to_parquet(out_path, index=False)

        print(f"[RAW] rows={len(df_raw):,} cols={len(df_raw.columns)} -> {out_path}")
        raise SystemExit(0)

    # ---------- ENGINEERED (incremental cache) ----------
    if args.mode == "engineered":
        # For engineered layer, we leverage incremental cache so reruns only add new dates
        req_cols = META_COLS  # you can add more cols if you always want them
        df_eng = get_incremental_features(
            start_date=start_iso,
            end_date=end_iso,
            include_odds=args.include_odds,
            required_features=req_cols,
            include_weather=args.include_weather,
            cache_path=Path(args.cache_path),
            force_until_today=False,  # respect end_iso
        )

        if df_eng is None or df_eng.empty:
            raise SystemExit("No rows returned for ENGINEERED.")

        out = args.out or args.cache_path  # by default, write to the same cache path
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to desired format
        if out_path.suffix.lower() == ".csv":
            df_eng.to_csv(out_path, index=False)
        else:
            df_eng.to_parquet(out_path, index=False)

        print(f"[ENGINEERED] rows={len(df_eng):,} cols={len(df_eng.columns)} -> {out_path}")
        raise SystemExit(0)

    # ---------- TRANSFORMED (X, y) ----------
    if args.mode == "transformed":
        # We need the engineered df first (not necessarily cached to disk, just materialized)
        # For transformed you typically want the full feature set used by the model.
        FEATURES = _load_features_list(args.variant, args.features_csv)

        # We can reuse incremental cache to build df quickly
        req_cols = list(dict.fromkeys(FEATURES + META_COLS))
        df_eng = get_incremental_features(
            start_date=start_iso,
            end_date=end_iso,
            include_odds=args.include_odds,
            required_features=req_cols,
            include_weather=args.include_weather,
            cache_path=Path(args.cache_path),
            force_until_today=False,
        )
        if df_eng is None or df_eng.empty:
            raise SystemExit("No rows returned to transform.")

        X, y = get_transformed_xy(df_eng, FEATURES)
        print(f"[TRANSFORMED] X.shape={X.shape}, y.shape={y.shape}")

        # Output(s)
        base_out = args.out or "cache/transformed"
        base = Path(base_out)
        base.parent.mkdir(parents=True, exist_ok=True)

        # Option 1: save X and y separately
        X_path = base.with_suffix(".X.parquet")
        y_path = base.with_suffix(".y.csv")
        X.to_parquet(X_path, index=False)
        pd.Series(y, name="y").to_csv(y_path, index=False)

        print(f"Saved X -> {X_path}")
        print(f"Saved y -> {y_path}")
        raise SystemExit(0)


# # RAW (earliest layer; minimal meta)

# python -m scripts.get_data --mode raw --out cache/raw_2024.parquet

# # ENGINEERED (incremental; also updates cache)

# python -m scripts.get_data --mode engineered --include_odds --include_weather --cache_path cache/mlb_features.parquet

# # TRANSFORMED using a variant's feature list

# python -m scripts.get_data --mode transformed --variant full_model_ml1 --out cache/tx_full_model_ml1

# # TRANSFORMED using a CSV feature list (one feature per line)

# python -m scripts.get_data --mode transformed --features_csv models/full_model_ml1_features.csv --out cache/tx_custom