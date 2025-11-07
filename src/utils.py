# src/utils.py
import os
from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("FEATURE_DB", "data/features.db")
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

def ensure_db():
    # Creates DB file and table if not exists
    with engine.connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            ts_utc TEXT,
            hour INTEGER,
            day INTEGER,
            month INTEGER,
            temp REAL,
            humidity REAL,
            pressure REAL,
            wind_kph REAL,
            pm25 REAL,
            pm10 REAL,
            aqi_change_rate REAL,
            target_pm25_next_hour REAL
        );
        """)

def save_features_df(df: pd.DataFrame):
    ensure_db()
    df.to_sql("features", engine, if_exists="append", index=False)

def read_features(city=None, limit=10000):
    ensure_db()
    query = "SELECT * FROM features"
    if city:
        query += f" WHERE city = :city"
        return pd.read_sql_query(query, engine, params={"city": city})
    return pd.read_sql_query(query, engine)
