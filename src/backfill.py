# src/backfill.py
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from src.config import WEATHERAPI_KEY, FEATURE_STORE_DIR
from src.fetch_features import make_features
from src.features import prepare_features_for_prediction

def fetch_historical(city, days=5):
    """Fetch last `days` of hourly data using WeatherAPI's history endpoint."""
    all_data = []
    base_url = "http://api.weatherapi.com/v1/history.json"
    for i in range(days):
        date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        url = f"{base_url}?key={WEATHERAPI_KEY}&q={city}&dt={date}"
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            for hour in data["forecast"]["forecastday"][0]["hour"]:
                all_data.append({
                    "observed_at": hour["time"],
                    "city": data["location"]["name"],
                    "country": data["location"]["country"],
                    "temp_c": hour["temp_c"],
                    "humidity": hour["humidity"],
                    "wind_kph": hour["wind_kph"],
                    "pressure_mb": hour["pressure_mb"],
                    "feelslike_c": hour["feelslike_c"],
                    "pm2_5": hour.get("air_quality", {}).get("pm2_5", None),
                    "pm10": hour.get("air_quality", {}).get("pm10", None),
                })
        except Exception as e:
            print(f"⚠️ Failed for {date}: {e}")

    df = pd.DataFrame(all_data)
    print(f"✅ Collected {len(df)} hourly rows for {city}")
    return df

def run_backfill(city="Karachi", days=5):
    df = fetch_historical(city, days)
    if df.empty:
        print("⚠️ No data fetched.")
        return
    df = make_features(df)
    df = prepare_features_for_prediction(df)
    out_path = FEATURE_STORE_DIR / f"backfill_{city}_{datetime.now().strftime('%Y%m%d')}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Backfill saved → {out_path}")

if __name__ == "__main__":
    run_backfill("Karachi", days=7)
