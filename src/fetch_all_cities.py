# src/fetch_all_cities.py
from datetime import datetime
import pandas as pd
from src.feature_store import insert_features
from src.config import WEATHERAPI_KEY
from src.fetch_features import fetch_now, make_features
from src.features import prepare_features_for_prediction

# Cities to fetch
CITIES = ["Karachi", "Delhi", "Paris", "New York"]

def collect_city_data(city):
    print(f"🌆 Fetching data for {city}...")

    try:
        df_now = fetch_now(city, WEATHERAPI_KEY)
        df_raw = make_features(df_now)
        df_prepared = prepare_features_for_prediction(df_raw)

        # ✅ Save to local feature store
        insert_features(df_prepared, city)

        print(f"✅ Done for {city}")
    except Exception as e:
        print(f"❌ Error fetching {city}: {e}")


def run_all():
    print(f"🚀 Starting fetch for {len(CITIES)} cities...")
    for c in CITIES:
        collect_city_data(c)
    print("🎯 All cities updated successfully.")


if __name__ == "__main__":
    run_all()
