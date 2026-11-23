import joblib
import pandas as pd

from phase_11_health_alerts import aqi_category

# Load the trained model and feature names
model = joblib.load("models/xgb_aqi.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

# Load one example feature row from the test dataset
X_test = pd.read_csv("data/X_test.csv")

# Ensure the feature order matches the model's training data
X_test = X_test[feature_cols]

# Predict AQI for the first test observation
predicted_aqi = float(model.predict(X_test.iloc[[0]])[0])

# AQI cannot be negative
predicted_aqi = max(0, predicted_aqi)

category, advice = aqi_category(predicted_aqi)

print("\nAQI FORECAST AND HEALTH ALERT")
print("=" * 40)
print(f"Predicted AQI: {predicted_aqi:.2f}")
print(f"Category: {category}")
print(f"Health Advice: {advice}")