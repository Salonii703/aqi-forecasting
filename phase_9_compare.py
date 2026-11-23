import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = Path("data")


# --------------------------------------------------
# 1. Load Phase 5 predictions
# --------------------------------------------------

phase5 = pd.read_csv(
    DATA_DIR / "phase5_predictions.csv",
    index_col=0,
    parse_dates=True
)

phase5.index = pd.to_datetime(phase5.index).normalize()

# --------------------------------------------------
# 2. Reconstruct Phase 7 test dates
#    Phase 7 predictions were saved without dates.
# --------------------------------------------------

df = pd.read_csv(
    DATA_DIR / "clean.csv",
    parse_dates=["Date"],
    index_col="Date"
).sort_index()

if "City" in df.columns:
    df = df[df["City"] == "Delhi"]

df = df[~df.index.duplicated(keep="first")]

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

def make_features(data):
    d = data.copy()

    # Predict the next row's AQI, following Phase 6
    d["target_AQI"] = d["AQI"].shift(-1)

    for lag in [1, 2, 3, 7, 14]:
        d[f"AQI_lag{lag}"] = d["AQI"].shift(lag)

    past_aqi = d["AQI"].shift(1)

    for window in [3, 7, 14]:
        d[f"AQI_roll_mean{window}"] = past_aqi.rolling(window).mean()
        d[f"AQI_roll_std{window}"] = past_aqi.rolling(window).std()

    target_dates = d.index + pd.Timedelta(days=1)

    d["month"] = target_dates.month
    d["dayofweek"] = target_dates.dayofweek
    d["dayofyear"] = target_dates.dayofyear
    d["is_weekend"] = (target_dates.dayofweek >= 5).astype(int)

    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)

    for col in pollutants:
        d[f"{col}_lag1"] = d[col].shift(1)

    d["target_date"] = target_dates
    d = d.dropna()

    return d

feat = make_features(df)

exclude_cols = [
    "AQI", "target_AQI", "AQI_Bucket",
    "City", "target_date"
] + pollutants

X = feat.drop(
    columns=[c for c in exclude_cols if c in feat.columns]
)

X = X.select_dtypes(include=[np.number])
y = feat["target_AQI"]

valid = (
    np.isfinite(X.to_numpy()).all(axis=1)
    & np.isfinite(y.to_numpy())
)

feat = feat.loc[valid].copy()
X = X.loc[valid]

split = int(len(X) * 0.8)

phase7_dates = pd.DatetimeIndex(
    pd.to_datetime(feat["target_date"].iloc[split:])
).normalize()

phase7 = pd.read_csv(DATA_DIR / "phase7_predictions.csv")

if len(phase7) != len(phase7_dates):
    raise ValueError(
        f"Phase 7 prediction count ({len(phase7)}) does not match "
        f"reconstructed test-date count ({len(phase7_dates)}). "
        "Check that Phase 6 and Phase 7 used the same data and features."
    )

phase7.index = phase7_dates


# --------------------------------------------------
# 3. Load Phase 8 LSTM predictions
# --------------------------------------------------

phase8 = pd.read_csv(
    DATA_DIR / "phase8_lstm_predictions.csv",
    index_col=0,
    parse_dates=True
)

phase8.index = pd.to_datetime(phase8.index).normalize()


# --------------------------------------------------
# 4. Collect prediction series
# --------------------------------------------------

prediction_series = {}
actual_series = []

# Phase 5: Naive and SARIMA
actual_series.append(phase5["Actual_AQI"].rename("Phase5_Actual"))

for col, model_name in [
    ("Naive_Prediction", "Naive"),
    ("SARIMA_Prediction", "SARIMA")
]:
    if col in phase5.columns:
        prediction_series[model_name] = phase5[col].rename(model_name)

# Phase 7: Random Forest and XGBoost
actual_series.append(phase7["Actual_AQI"].rename("Phase7_Actual"))

for col, model_name in [
    ("RandomForest_Prediction", "Random Forest"),
    ("XGBoost_Prediction", "XGBoost"),
    ("Tuned_XGBoost_Prediction", "Tuned XGBoost")
]:
    if col in phase7.columns:
        prediction_series[model_name] = phase7[col].rename(model_name)

# Phase 8: LSTM
actual_series.append(phase8["Actual_AQI"].rename("Phase8_Actual"))

prediction_series["LSTM"] = phase8["LSTM_Prediction"].rename("LSTM")


# --------------------------------------------------
# 5. Align all models on common test dates
# --------------------------------------------------

predictions = pd.concat(
    list(prediction_series.values()),
    axis=1,
    join="inner"
)

actual_values = pd.concat(actual_series, axis=1)
actual_values = actual_values.bfill(axis=1).iloc[:, 0]

comparison = predictions.join(
    actual_values.rename("Actual_AQI"),
    how="inner"
)

comparison = comparison.replace([np.inf, -np.inf], np.nan).dropna()

if comparison.empty:
    raise ValueError(
        "No common test dates found across the models. "
        "Check the dates in your prediction files."
    )

model_names = list(prediction_series.keys())

print("\nCommon test period:")
print("Start:", comparison.index.min().date())
print("End:  ", comparison.index.max().date())
print("Number of common test dates:", len(comparison))


# --------------------------------------------------
# 6. Calculate MAE and RMSE
# --------------------------------------------------

results = []

for model_name in model_names:
    y_true = comparison["Actual_AQI"].to_numpy()
    y_pred = comparison[model_name].to_numpy()

    results.append({
        "Model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    })

results_df = pd.DataFrame(results).sort_values("RMSE")

print("\nModel Comparison:")
print(results_df.round(2).to_string(index=False))

results_df.to_csv(DATA_DIR / "phase9_model_comparison.csv", index=False)
comparison.to_csv(DATA_DIR / "phase9_aligned_predictions.csv")


# --------------------------------------------------
# 7. Plot MAE and RMSE comparison
# --------------------------------------------------

results_df.set_index("Model")[["MAE", "RMSE"]].plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("AQI Forecasting: Model Comparison")
plt.ylabel("Error (lower is better)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(DATA_DIR / "phase9_model_comparison.png", dpi=300)
plt.show()


# --------------------------------------------------
# 8. Plot actual vs predicted AQI for each model
# --------------------------------------------------

for model_name in model_names:
    plt.figure(figsize=(12, 5))

    plt.plot(
        comparison.index,
        comparison["Actual_AQI"],
        label="Actual AQI"
    )

    plt.plot(
        comparison.index,
        comparison[model_name],
        label=f"{model_name} Prediction"
    )

    plt.title(f"Actual vs Predicted AQI: {model_name}")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    plt.savefig(
        DATA_DIR / f"phase9_actual_vs_{safe_name}.png",
        dpi=300
    )

    plt.show()


# --------------------------------------------------
# 9. Residual plots
#    Residual = Actual AQI - Predicted AQI
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for model_name in model_names:
    residuals = (
        comparison["Actual_AQI"] - comparison[model_name]
    )

    plt.scatter(
        comparison[model_name],
        residuals,
        alpha=0.5,
        label=model_name
    )

plt.axhline(y=0, linestyle="--")

plt.title("Residual Plot: All Models")
plt.xlabel("Predicted AQI")
plt.ylabel("Residual (Actual - Predicted)")
plt.legend()
plt.tight_layout()

plt.savefig(DATA_DIR / "phase9_residual_plot.png", dpi=300)
plt.show()


# --------------------------------------------------
# 10. Save residuals for further analysis
# --------------------------------------------------

residual_df = pd.DataFrame(index=comparison.index)

for model_name in model_names:
    residual_df[model_name + "_Residual"] = (
        comparison["Actual_AQI"] - comparison[model_name]
    )

residual_df.to_csv(DATA_DIR / "phase9_residuals.csv")

print("\nPhase 9 completed successfully!")
print("Check the data folder for comparison tables and plots.")