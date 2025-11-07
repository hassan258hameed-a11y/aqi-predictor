# src/fetch_features.py
import requests
import pandas as pd
from datetime import datetime


def fetch_now(city_name, api_key):
    """Fetch current weather + air quality data for a city using WeatherAPI."""
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city_name}&aqi=yes"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    loc = data["location"]
    cur = data["current"]
    air = cur["air_quality"]

    df = pd.DataFrame([{
        "observed_at": datetime.now(),
        "city": loc.get("name"),
        "country": loc.get("country"),
        "temp_c": cur.get("temp_c"),
        "humidity": cur.get("humidity"),
        "wind_kph": cur.get("wind_kph"),
        "pressure_mb": cur.get("pressure_mb"),
        "feelslike_c": cur.get("feelslike_c"),
        "is_day": cur.get("is_day"),
        "co": air.get("co"),
        "no2": air.get("no2"),
        "o3": air.get("o3"),
        "so2": air.get("so2"),
        "pm2_5": air.get("pm2_5"),
        "pm10": air.get("pm10"),
        "us_epa_index": air.get("us-epa-index"),
        "gb_defra_index": air.get("gb-defra-index"),
    }])

    return df


def make_features(df):
    """Generate extra time-based features for model input."""
    df = df.copy()
    if "observed_at" in df.columns:
        df["observed_at"] = pd.to_datetime(df["observed_at"])
        df["hour"] = df["observed_at"].dt.hour
        df["day"] = df["observed_at"].dt.day
        df["month"] = df["observed_at"].dt.month
        df["weekday"] = df["observed_at"].dt.weekday
    return df


def fetch_features(df_now: pd.DataFrame):
    """
    Wrapper used by fetch_all_cities.py and Streamlit app.
    Converts raw fetched data into model-ready features.
    """
    df = make_features(df_now)
    df["target"] = df["pm2_5"]  # placeholder for next-hour PM2.5 target
    return df
