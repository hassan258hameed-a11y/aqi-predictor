# src/train.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json

from src.feature_store import read_feature_history
from src.config import FEATURE_STORE_DIR, MODEL_SAVE_DIR, REGISTRY_FILE


CITIES = ["Karachi", "Delhi", "Paris", "New York"]


# ---------------------------
# ✅ Utility Helpers
# ---------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------
# ✅ Load data from Hopsworks (all cities)
# ---------------------------
def load_from_hopsworks():
    dfs = []
    for city in CITIES:
        try:
            df = read_feature_history(city)
            if df is not None and not df.empty:
                print(f"✅ Loaded {len(df)} rows from Hopsworks for {city}")
                dfs.append(df)
            else:
                print(f"⚠️ No Hopsworks data for {city}")
        except Exception as e:
            print(f"⚠️ Hopsworks read failed for {city}: {e}")

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


# ---------------------------
# ✅ Load data locally (fallback)
# ---------------------------
def load_local_training_data():
    all_dfs = []

    for city_dir in FEATURE_STORE_DIR.iterdir():
        if not city_dir.is_dir():
            continue

        for f in city_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(f)
                df["city_name"] = city_dir.name
                all_dfs.append(df)
            except Exception as e:
                print(f"⚠️ Could not load {f}: {e}")

    if not all_dfs:
        raise FileNotFoundError("❌ No local data found. Run fetch_all_cities.py first.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ Loaded {len(df_all)} local rows")
    return df_all


# ---------------------------
# ✅ Model Training
# ---------------------------
def train_model(df: pd.DataFrame, target_col="pm2_5"):
    drop_cols = ["observed_at", "city", "country", "city_name", "target", "aqi"]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"✅ Model trained → MAE={mae:.3f}, R²={r2:.3f}")
    return model, mae, r2


# ---------------------------
# ✅ Model Saving
# ---------------------------
def save_model(model, mae, r2):
    ensure_dir(MODEL_SAVE_DIR)
    model_path = MODEL_SAVE_DIR / "aqi_model.pkl"
    joblib.dump(model, model_path)

    registry = {
        "model_path": str(model_path),
        "mae": mae,
        "r2": r2,
    }
    save_json(REGISTRY_FILE, registry)

    print(f"✅ Model saved → {model_path}")
    print(f"✅ Registry saved → {REGISTRY_FILE}")


# ---------------------------
# ✅ Training Entry Point
# ---------------------------
def run_training():
    print("🚀 Starting training...")

    # ✅ Step 1: Try Hopsworks
    df = load_from_hopsworks()

    # ✅ Step 2: Fallback → local parquet files
    if df is None:
        print("⚠️ No Hopsworks data found — using local parquet files.")
        df = load_local_training_data()

    print(f"📦 Final training dataset: {df.shape}")

    # ✅ Step 3: Train model
    model, mae, r2 = train_model(df)

    # ✅ Step 4: Save model + registry
    save_model(model, mae, r2)

    print("✅ Training complete.")


if __name__ == "__main__":
    run_training()
