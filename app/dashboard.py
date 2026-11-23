import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from pathlib import Path

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Delhi AQI Forecast",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ Delhi AQI Forecasting Dashboard")
st.caption("Historical air quality, next-day predictions, and health alerts")


# --------------------------------------------------
# 2. Locate project files
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "clean.csv"
MODEL_PATH = ROOT_DIR / "models" / "xgb_aqi.pkl"
FEATURES_PATH = ROOT_DIR / "models" / "feature_cols.pkl"


# --------------------------------------------------
# 3. Load data and model
# --------------------------------------------------

@st.cache_data
def load_data(path):
    data = pd.read_csv(path, parse_dates=["Date"])
    data = data.sort_values("Date")

    if "City" in data.columns:
        data = data[data["City"] == "Delhi"]

    data = data.drop_duplicates(subset=["Date"], keep="last")
    data = data.set_index("Date").sort_index()

    required = ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data[required] = data[required].apply(
        pd.to_numeric, errors="coerce"
    )

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required)

    return data


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_features(path):
    return joblib.load(path)


try:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()

    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        st.error(
            "Saved model files are missing. "
            "Run Phase 7 to train and save the model."
        )
        st.stop()

    df = load_data(str(DATA_PATH))
    model = load_model(str(MODEL_PATH))
    feature_cols = load_features(str(FEATURES_PATH))

except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()


if len(df) < 20:
    st.error("Not enough historical observations to generate forecasts.")
    st.stop()


# --------------------------------------------------
# 4. AQI category and health advice
# --------------------------------------------------

def category(aqi):
    if not np.isfinite(aqi) or aqi < 0:
        return "Invalid AQI", "A valid non-negative AQI is required."

    if aqi <= 50:
        return "Good", "Air quality is satisfactory. Enjoy outdoor activities."

    elif aqi <= 100:
        return (
            "Satisfactory",
            "Sensitive people should limit prolonged outdoor exertion."
        )

    elif aqi <= 200:
        return (
            "Moderate",
            "People with asthma or heart disease should reduce outdoor activity."
        )

    elif aqi <= 300:
        return (
            "Poor",
            "Avoid prolonged outdoor exertion; consider a well-fitting mask."
        )

    elif aqi <= 400:
        return (
            "Very Poor",
            "Avoid outdoor activity where possible and consider cleaner indoor air."
        )

    return (
        "Severe",
        "Minimize outdoor exposure. Vulnerable people should take extra precautions."
    )


# --------------------------------------------------
# 5. Generate features matching Phase 6
# --------------------------------------------------

pollutants = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene"
]


def build_feature_row(history, columns):
    """Generate the exact features expected by the trained XGBoost model."""

    row = {}
    origin_date = history.index[-1]
    target_date = origin_date + pd.Timedelta(days=1)

    # Calendar features for the forecast date
    row["month"] = target_date.month
    row["dayofweek"] = target_date.dayofweek
    row["dayofyear"] = target_date.dayofyear
    row["is_weekend"] = int(target_date.dayofweek >= 5)
    row["month_sin"] = np.sin(2 * np.pi * target_date.month / 12)
    row["month_cos"] = np.cos(2 * np.pi * target_date.month / 12)

    for col in columns:

        # Calendar features are already calculated
        if col in row:
            continue

        # AQI lag features
        match = re.fullmatch(r"AQI_lag(\d+)", col)

        if match:
            lag = int(match.group(1))

            if len(history) < lag:
                raise ValueError(f"Insufficient history for {col}")

            row[col] = history["AQI"].iloc[-lag]
            continue

        # Rolling AQI mean/std features
        match = re.fullmatch(r"AQI_roll_(mean|std)(\d+)", col)

        if match:
            statistic = match.group(1)
            window = int(match.group(2))

            # Use the previous observations, including the latest known AQI
            values = history["AQI"].tail(window)

            if len(values) < window or values.isna().any():
                raise ValueError(f"Insufficient history for {col}")

            if statistic == "mean":
                row[col] = values.mean()
            else:
                row[col] = values.std(ddof=1)

            continue

        # Pollutant lag features, e.g. PM2.5_lag1 or NO2_lag1
        match = re.fullmatch(r"(.+)_lag1", col)

        if match:
            pollutant = match.group(1)

            if pollutant not in history.columns:
                raise ValueError(
                    f"Column '{pollutant}' required for {col} is missing."
                )

            row[col] = history[pollutant].iloc[-1]
            continue

        # Current known pollutant values, e.g. NO, NOx, NH3, Benzene
        if col in history.columns:
            row[col] = history[col].iloc[-1]
            continue

        raise ValueError(f"Cannot generate required feature: {col}")

    X_next = pd.DataFrame([row], columns=columns)

    if X_next.isna().any().any():
        raise ValueError("Generated features contain missing values.")

    if not np.isfinite(X_next.to_numpy(dtype=float)).all():
        raise ValueError("Generated features contain invalid numeric values.")

    return X_next, target_date
# --------------------------------------------------
# 6. Forecast next N days recursively
# --------------------------------------------------

def forecast_next_days(history, model, columns, horizon):
    working = history.copy()
    forecasts = []

    for _ in range(horizon):

        X_next, target_date = build_feature_row(working, columns)

        prediction = float(model.predict(X_next)[0])
        prediction = max(0.0, prediction)

        forecasts.append({
            "Date": target_date,
            "Predicted_AQI": prediction
        })

        # Add the predicted AQI so the next iteration can
        # use it as part of the historical AQI sequence.
        # Pollutants are held at their last known values.
        new_row = working.iloc[-1].copy()
        new_row["AQI"] = prediction

        working.loc[target_date] = new_row

    return pd.DataFrame(forecasts).set_index("Date")


# --------------------------------------------------
# 7. Sidebar controls
# --------------------------------------------------

st.sidebar.header("Forecast Settings")

horizon = st.sidebar.slider(
    "Forecast days",
    min_value=1,
    max_value=7,
    value=3
)

st.sidebar.caption(
    "Multi-day forecasts are recursive estimates, not live measurements."
)


# --------------------------------------------------
# 8. Generate forecasts
# --------------------------------------------------

try:
    forecasts = forecast_next_days(
        history=df,
        model=model,
        columns=feature_cols,
        horizon=horizon
    )

except Exception as e:
    st.error(f"Could not generate forecast: {e}")
    st.stop()


# --------------------------------------------------
# 9. Dashboard metrics
# --------------------------------------------------

latest_aqi = float(df["AQI"].iloc[-1])
tomorrow_aqi = float(forecasts["Predicted_AQI"].iloc[0])

latest_category, latest_advice = category(latest_aqi)
tomorrow_category, tomorrow_advice = category(tomorrow_aqi)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest Recorded AQI",
    f"{latest_aqi:.0f}",
    help=f"Latest dataset date: {df.index[-1].date()}"
)

col2.metric(
    "Tomorrow's Forecast",
    f"{tomorrow_aqi:.0f}",
    delta=f"{tomorrow_aqi - latest_aqi:+.0f} AQI points"
)

col3.metric(
    "Forecast Category",
    tomorrow_category
)

st.caption(
    f"Latest historical date: {df.index[-1].date()} | "
    f"Forecast date: {forecasts.index[0].date()}"
)


# --------------------------------------------------
# 10. Health alert
# --------------------------------------------------

st.subheader("🩺 Health Alert")

if tomorrow_aqi <= 100:
    st.success(f"**{tomorrow_category}:** {tomorrow_advice}")
elif tomorrow_aqi <= 200:
    st.info(f"**{tomorrow_category}:** {tomorrow_advice}")
elif tomorrow_aqi <= 300:
    st.warning(f"**{tomorrow_category}:** {tomorrow_advice}")
else:
    st.error(f"**{tomorrow_category}:** {tomorrow_advice}")


# --------------------------------------------------
# 11. Historical and forecast chart
# --------------------------------------------------

st.subheader("📈 Historical AQI and Forecast")

historical = df["AQI"].tail(90)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=historical.index,
        y=historical.values,
        mode="lines",
        name="Historical AQI"
    )
)

fig.add_trace(
    go.Scatter(
        x=forecasts.index,
        y=forecasts["Predicted_AQI"],
        mode="lines+markers",
        name="Forecast AQI",
        line=dict(dash="dash")
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="AQI",
    hovermode="x unified",
    legend_title="Series"
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 12. Forecast table
# --------------------------------------------------

st.subheader("📅 Forecast Details")

forecast_table = forecasts.copy()

forecast_table["Category"] = forecast_table["Predicted_AQI"].apply(
    lambda value: category(value)[0]
)

forecast_table["Health Advice"] = forecast_table["Predicted_AQI"].apply(
    lambda value: category(value)[1]
)

forecast_table.index.name = "Forecast Date"

st.dataframe(
    forecast_table.round({"Predicted_AQI": 2}),
    use_container_width=True
)

csv_data = forecast_table.to_csv().encode("utf-8")

st.download_button(
    label="Download Forecast CSV",
    data=csv_data,
    file_name="aqi_forecast.csv",
    mime="text/csv"
)


# --------------------------------------------------
# 13. Monthly AQI trend analysis
# --------------------------------------------------

st.subheader("📊 Monthly AQI Trend Analysis")

monthly_aqi = df["AQI"].groupby(df.index.month).mean()

month_names = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

monthly_aqi.index = [
    month_names[month - 1] for month in monthly_aqi.index
]

st.bar_chart(monthly_aqi)

st.caption(
    "Monthly values are historical averages across all available years, "
    "not forecasts for the coming months."
)


# --------------------------------------------------
# 14. Forecast assumptions
# --------------------------------------------------

with st.expander("How should I interpret these forecasts?"):
    st.write(
        "The first forecast uses the saved XGBoost model and the "
        "features expected by Phase 7."
    )

    st.write(
        "For forecasts beyond tomorrow, the model is run recursively. "
        "Future pollutant concentrations are held at their last known "
        "values because future pollution measurements are unavailable."
    )

    st.write(
        "Actual future AQI can differ substantially due to weather, "
        "traffic, emissions, and other changing conditions. These "
        "predictions are estimates, not official AQI readings or medical advice."
    )

st.caption("AQI Forecasting Project | Model: Tuned XGBoost")