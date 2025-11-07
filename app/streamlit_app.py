import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from datetime import datetime
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------
# ✅ Safe Prophet Import
# -------------------------------------
try:
    from prophet import Prophet
    PROPHET_OK = True
except Exception as e:
    PROPHET_OK = False
    import logging
    logging.warning(f"Prophet import failed: {e}")

from src.config import MODEL_SAVE_DIR, FEATURE_STORE_DIR, WEATHERAPI_KEY
from src.fetch_features import fetch_now, make_features
from src.features import prepare_features_for_prediction
from src.feature_store import read_feature_history, insert_features

# -------------------------------------
# ✅ AQI Helpers
# -------------------------------------
BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

def pm25_to_aqi(pm25: float) -> int:
    for c_low, c_high, aqi_low, aqi_high in BREAKPOINTS:
        if c_low <= pm25 <= c_high:
            return int(round((aqi_high - aqi_low) / (c_high - c_low) * (pm25 - c_low) + aqi_low))
    return 500

def aqi_alert(aqi: int):
    if aqi >= 300:
        st.error("☠️ Hazardous AQI — Avoid outdoor activity.")
    elif aqi >= 200:
        st.error("😷 Very Unhealthy AQI for everyone.")
    elif aqi >= 150:
        st.warning("⚠️ Unhealthy AQI — limit outdoor activity.")
    elif aqi >= 100:
        st.warning("😐 Moderate AQI — some risk to sensitive groups.")
    else:
        st.success("✅ Good AQI — clean air.")

# -------------------------------------
# ✅ 3-Day Forecast (Prophet or fallback)
# -------------------------------------
def forecast_3_days(df: pd.DataFrame):
    if df is None or df.empty:
        return pd.DataFrame(columns=["ds", "yhat"])
    try:
        if PROPHET_OK:
            df_prophet = df[['observed_at', 'pm2_5']].dropna().rename(
                columns={"observed_at": "ds", "pm2_5": "y"}
            )
            model = Prophet()
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=72, freq="H")
            fc = model.predict(future)
            return fc[['ds', 'yhat']]
        else:
            raise ImportError("Prophet not available.")
    except Exception as e:
        import logging
        logging.warning(f"Prophet forecast failed: {e}. Using simple fallback.")
        s = df.set_index("observed_at")["pm2_5"].dropna()
        if len(s) == 0:
            return pd.DataFrame(columns=["ds", "yhat"])
        last_val = float(s.iloc[-1])
        idx = pd.date_range(start=s.index[-1], periods=73, freq="H", closed="right")
        return pd.DataFrame({"ds": idx, "yhat": [last_val]*72})

# -------------------------------------
# ✅ Streamlit App
# -------------------------------------
st.set_page_config(page_title="🌍 AQI Dashboard", layout="wide")
st.title("🌍 Any-City AQI Prediction Dashboard")

city = st.text_input("🏙️ Enter city name:", value="Karachi")
view = st.sidebar.radio("View", ["Prediction", "3-Day Forecast", "Data History"])

@st.cache_data(ttl=3600)
def get_city_data(city_name, api_key):
    df_now = fetch_now(city_name, api_key)
    df_features = make_features(df_now)
    df_prepared = prepare_features_for_prediction(df_features.copy())
    return df_now, df_features, df_prepared

# -------------------------------------
# ✅ Prediction
# -------------------------------------
if view == "Prediction":
    if st.button("🔮 Predict Next Hour PM2.5"):
        df_now, df_features, df_prepared = get_city_data(city, WEATHERAPI_KEY)

        # Save locally
        insert_features(df_prepared, city)

        # Load model
        model_path = MODEL_SAVE_DIR / "aqi_model.pkl"
        model = joblib.load(model_path)

        drop_cols = ["observed_at", "aqi", "city", "country", "target"]
        X = df_prepared.drop(columns=drop_cols, errors="ignore")
        pred = float(model.predict(X.tail(1))[0])

        st.metric(f"Next Hour PM2.5 ({city})", round(pred, 2))

        aqi_val = pm25_to_aqi(pred)
        st.write(f"**Estimated AQI:** {aqi_val}")
        aqi_alert(aqi_val)

        # SHAP
        with st.spinner("Explaining model..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            plt.figure(figsize=(7, 3.5))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
            st.pyplot(plt.gcf(), clear_figure=True)

        fig = px.bar(
            df_features.tail(1).T,
            title=f"Weather Snapshot - {city}",
            labels={"value": "Value", "index": "Feature"},
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------
# ✅ 3-Day Forecast
# -------------------------------------
elif view == "3-Day Forecast":
    hist = read_feature_history(city)
    if hist.empty:
        st.warning("No history available. Run prediction first.")
    else:
        fc = forecast_3_days(hist)
        st.line_chart(fc.set_index("ds"))

# -------------------------------------
# ✅ Data History
# -------------------------------------
else:
    hist = read_feature_history(city)
    if hist.empty:
        st.info("No historical data yet.")
    else:
        st.dataframe(hist.tail(500))
