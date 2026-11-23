import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1. Load Phase 6 training and testing data
# --------------------------------------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

# Ensure target values are numeric
y_train = pd.to_numeric(y_train)
y_test = pd.to_numeric(y_test)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Check that training and testing features match
if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and testing features do not match.")

results = []


# --------------------------------------------------
# 2. Evaluation function
# --------------------------------------------------

def evaluate(y_true, y_pred, model_name):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"{model_name:15s} MAE = {mae:.2f} | RMSE = {rmse:.2f}")

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse
    }


# --------------------------------------------------
# 3. Train Random Forest
# --------------------------------------------------

print("\nTraining Random Forest...")

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)

results.append(
    evaluate(y_test, rf_predictions, "RandomForest")
)


# --------------------------------------------------
# 4. Train XGBoost
# --------------------------------------------------

print("\nTraining XGBoost...")

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

xgb_predictions = xgb.predict(X_test)

results.append(
    evaluate(y_test, xgb_predictions, "XGBoost")
)


# --------------------------------------------------
# 5. Tune XGBoost using time-series cross-validation
# --------------------------------------------------

print("\nTuning XGBoost. This may take some time...")

param_dist = {
    "n_estimators": [300, 500, 800],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.9, 1.0]
}

tscv = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    estimator=XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    ),
    param_distributions=param_dist,
    n_iter=15,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

best_xgb = search.best_estimator_

print("\nBest XGBoost parameters:")
print(search.best_params_)

print("\nBest cross-validation MAE:", round(-search.best_score_, 2))


# --------------------------------------------------
# 6. Evaluate tuned XGBoost on the test set
# --------------------------------------------------

best_predictions = best_xgb.predict(X_test)

results.append(
    evaluate(y_test, best_predictions, "Tuned XGBoost")
)


# --------------------------------------------------
# 7. Compare model results
# --------------------------------------------------

results_df = pd.DataFrame(results).sort_values("MAE")

print("\nFinal Model Comparison:")
print(results_df.to_string(index=False))

results_df.to_csv(DATA_DIR / "phase7_results.csv", index=False)

print("\nResults saved to data/phase7_results.csv")


# --------------------------------------------------
# 8. Plot top 15 feature importances
# --------------------------------------------------

importance = pd.Series(
    best_xgb.feature_importances_,
    index=X_train.columns
).nlargest(15).sort_values()

plt.figure(figsize=(10, 7))
importance.plot(kind="barh")

plt.title("Top 15 Features for Next-Day AQI Prediction")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig(DATA_DIR / "phase7_feature_importance.png", dpi=300)
plt.show()

print("Feature importance graph saved to data/phase7_feature_importance.png")


# --------------------------------------------------
# 9. Save tuned model predictions
# --------------------------------------------------

predictions_df = pd.DataFrame({
    "Actual_AQI": np.asarray(y_test),
    "RandomForest_Prediction": rf_predictions,
    "XGBoost_Prediction": xgb_predictions,
    "Tuned_XGBoost_Prediction": best_predictions
})

predictions_df.to_csv(
    DATA_DIR / "phase7_predictions.csv",
    index=False
)
joblib.dump(best_xgb, MODEL_DIR / "xgb_aqi.pkl")


joblib.dump(list(X_train.columns), MODEL_DIR / "feature_cols.pkl")

print("Tuned XGBoost model saved successfully!")
print("Predictions saved to data/phase7_predictions.csv")