# src/predict.py
import joblib
import pandas as pd
from pathlib import Path
from src.config import MODEL_SAVE_DIR

def load_model():
    """Load trained model from disk."""
    model_path = MODEL_SAVE_DIR / "aqi_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("❌ Trained model not found. Run train.py first.")
    model = joblib.load(model_path)
    return model

def predict_next_hour(df_features: pd.DataFrame):
    """
    Given prepared feature DataFrame, predict next-hour PM2.5.
    """
    model = load_model()
    drop_cols = ["observed_at", "city", "country", "target", "aqi"]
    X = df_features.drop(columns=drop_cols, errors="ignore")
    preds = model.predict(X)
    return preds
