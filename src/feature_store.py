# src/feature_store.py
import os
import pandas as pd
from pathlib import Path
from src.config import FEATURE_STORE_DIR

# ----------------------------------------------------
# ✅ Local-only feature storage (Parquet)
# ----------------------------------------------------

def insert_features(df: pd.DataFrame, city: str):
    """
    Save prepared features to local feature store folder.
    """
    city_dir = FEATURE_STORE_DIR / city
    city_dir.mkdir(parents=True, exist_ok=True)

    file_path = city_dir / f"{city}_{pd.Timestamp.now().strftime('%Y%m%d%H%M')}.parquet"
    df.to_parquet(file_path)

    print(f"✅ Saved locally → {file_path}")
    return True


def read_feature_history(city: str):
    """
    Read all local parquet files for a city.
    """
    city_dir = FEATURE_STORE_DIR / city
    if not city_dir.exists():
        return pd.DataFrame()

    files = sorted(Path(city_dir).glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    if "observed_at" in df.columns:
        df["observed_at"] = pd.to_datetime(df["observed_at"])

    return df
