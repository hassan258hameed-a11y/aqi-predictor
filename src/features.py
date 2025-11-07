# src/features.py
import pandas as pd
from pathlib import Path
from src.config import FEATURE_STORE_DIR


def load_features():
    """Load all collected feature records."""
    parquet_path = FEATURE_STORE_DIR / "features.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No feature data found at {parquet_path}")
    return pd.read_parquet(parquet_path)


def create_training_dataset(df: pd.DataFrame, target_col="pm2_5"):
    """
    Prepare a dataset suitable for ML:
    - sorts by timestamp
    - creates lag features (previous readings)
    - drops NaNs after shifting
    """
    df = df.sort_values("observed_at")

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)

    df["target"] = df[target_col].shift(-1)  # predict next reading
    df = df.dropna()
    return df


def save_training_dataset(train_df):
    """Save processed dataset as a Parquet file."""
    out_path = FEATURE_STORE_DIR / "training_data.parquet"
    train_df.to_parquet(out_path, index=False)
    print(f"✅ Training dataset saved → {out_path}")


def run_feature_engineering():
    df = load_features()
    print(f"Loaded {len(df)} feature rows")
    train_df = create_training_dataset(df)
    save_training_dataset(train_df)
    print("✅ Feature engineering complete.")
    
def prepare_features_for_prediction(df: pd.DataFrame, target_col="pm2_5"):
    """
    Prepare a single-row DataFrame for prediction.
    This recreates lag features consistent with training.
    """
    df = df.sort_values("observed_at").copy()
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    df = df.fillna(method="bfill")
    return df


if __name__ == "__main__":
    run_feature_engineering()
