import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ============================================================
# 1. LOAD TRAINING AND TESTING DATA
# ============================================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(
    DATA_DIR / "train.csv",
    parse_dates=["Date"],
    index_col="Date"
)

test = pd.read_csv(
    DATA_DIR / "test.csv",
    parse_dates=["Date"],
    index_col="Date"
)

train = train.sort_index()
test = test.sort_index()

train = train.dropna(subset=["AQI"])
test = test.dropna(subset=["AQI"])

if train.empty or test.empty:
    raise ValueError("Training or testing dataset is empty.")

# Keep original dates for the final prediction output.
test_dates = test.index

# Use numeric values and a regular integer index for modeling.
y_train = pd.Series(
    train["AQI"].to_numpy(dtype=float),
    index=pd.RangeIndex(start=0, stop=len(train)),
    name="AQI"
)

y_test = test["AQI"].to_numpy(dtype=float)

print("Training observations:", len(y_train))
print("Testing observations:", len(y_test))
print("Training period:", train.index.min(), "to", train.index.max())
print("Testing period:", test.index.min(), "to", test.index.max())


# ============================================================
# 2. EVALUATION FUNCTION
# ============================================================

def evaluate(y_true, y_pred, name):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"{name}: actual and predicted lengths do not match."
        )

    if not np.isfinite(y_pred).all():
        raise ValueError(f"{name}: predictions contain invalid values.")

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"{name:15s} MAE={mae:.2f}  RMSE={rmse:.2f}")

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse
    }


results = []


# ============================================================
# 3. NAIVE BASELINE
# ============================================================

# Fixed-origin forecast:
# Use the last AQI observed in training for every test observation.
naive_pred = np.repeat(y_train.iloc[-1], len(y_test))

results.append(
    evaluate(y_test, naive_pred, "Naive")
)


# ============================================================
# 4. AUTOMATIC SARIMA ORDER SELECTION
# ============================================================

print("\nSelecting SARIMA parameters...")

auto = pm.auto_arima(
    y_train,
    seasonal=True,
    m=7,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    trace=True,
    max_p=3,
    max_q=3,
    max_P=2,
    max_Q=2
)

print("\nSelected SARIMA order:", auto.order)
print("Selected seasonal order:", auto.seasonal_order)


# ============================================================
# 5. FIT SARIMAX AND FORECAST
# ============================================================

# Refit the selected orders with statsmodels directly.
# The explicit RangeIndex avoids the unsupported-index issue
# encountered with auto.predict() in your environment.

sarima_model = SARIMAX(
    endog=y_train,
    order=auto.order,
    seasonal_order=auto.seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fit = sarima_model.fit(disp=False)

print("\nSARIMA model fitted successfully.")

# Forecast the entire test period.
forecast_result = sarima_fit.get_forecast(
    steps=len(y_test)
)

sarima_pred = np.asarray(
    forecast_result.predicted_mean,
    dtype=float
).reshape(-1)

# Evaluate SARIMA exactly once.
results.append(
    evaluate(y_test, sarima_pred, "SARIMA")
)


# ============================================================
# 6. COMPARE MODELS
# ============================================================

results_df = pd.DataFrame(results).sort_values("MAE")

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))


# ============================================================
# 7. SAVE PREDICTIONS AND METRICS
# ============================================================

predictions = pd.DataFrame(
    {
        "Actual_AQI": y_test,
        "Naive_Prediction": naive_pred,
        "SARIMA_Prediction": sarima_pred
    },
    index=test_dates
)

predictions.index.name = "Date"

predictions.to_csv(DATA_DIR / "phase5_predictions.csv")
results_df.to_csv(DATA_DIR / "phase5_results.csv", index=False)

print("\nPhase 5 completed successfully!")
print("Predictions saved to: data/phase5_predictions.csv")
print("Metrics saved to: data/phase5_results.csv")