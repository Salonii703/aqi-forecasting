import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================

DATA_DIR = Path("data")

df = pd.read_csv(
    DATA_DIR / "clean.csv",
    parse_dates=["Date"],
    index_col="Date"
)

df = df.sort_index()

# Keep Delhi data only, if the City column exists.
if "City" in df.columns:
    df = df[df["City"] == "Delhi"]

# Remove duplicate dates, if any.
df = df[~df.index.duplicated(keep="first")]

# Required columns
pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

required = ["AQI"] + pollutants
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("Original dataset shape:", df.shape)


# ============================================================
# 2. CREATE FEATURES FOR NEXT-DAY AQI FORECASTING
# ============================================================

def make_features(data):
    d = data.copy()

    # Target: tomorrow's AQI
    d["target_AQI"] = d["AQI"].shift(-1)

    # AQI lag features
    for lag in [1, 2, 3, 7, 14]:
        d[f"AQI_lag{lag}"] = d["AQI"].shift(lag)

    # Rolling statistics using past observations only
    for window in [3, 7, 14]:
        past_aqi = d["AQI"].shift(1)

        d[f"AQI_roll_mean{window}"] = (
            past_aqi.rolling(window).mean()
        )

        d[f"AQI_roll_std{window}"] = (
            past_aqi.rolling(window).std()
        )

    # Calendar features for the day being predicted
    target_dates = d.index + pd.Timedelta(days=1)

    d["month"] = target_dates.month
    d["dayofweek"] = target_dates.dayofweek
    d["dayofyear"] = target_dates.dayofyear
    d["is_weekend"] = (target_dates.dayofweek >= 5).astype(int)

    # Cyclical encoding of month
    d["month_sin"] = np.sin(
        2 * np.pi * d["month"] / 12
    )

    d["month_cos"] = np.cos(
        2 * np.pi * d["month"] / 12
    )

    # Previous day's pollutant measurements
    for col in pollutants:
        d[f"{col}_lag1"] = d[col].shift(1)

    # Keep the date of the AQI being predicted.
    d["target_date"] = target_dates

    # Remove rows with missing features or target.
    d = d.dropna()

    return d


feat = make_features(df)

print("Feature dataset shape:", feat.shape)


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

# AQI for the following day is the prediction target.
y = feat["target_AQI"]

# Exclude current-day AQI, target, raw pollutants,
# and metadata from the input features.
exclude_cols = [
    "AQI",
    "target_AQI",
    "AQI_Bucket",
    "City",
    "target_date"
] + pollutants

X = feat.drop(
    columns=[c for c in exclude_cols if c in feat.columns]
)

# Convert features to numeric values.
X = X.select_dtypes(include=[np.number])

# Ensure features and targets are finite.
valid = (
    np.isfinite(X.to_numpy()).all(axis=1)
    & np.isfinite(y.to_numpy())
)

X = X.loc[valid]
y = y.loc[valid]
feat = feat.loc[valid]


# ============================================================
# 4. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

split = int(len(X) * 0.8)

X_train = X.iloc[:split].copy()
X_test = X.iloc[split:].copy()

y_train = y.iloc[:split].copy()
y_test = y.iloc[split:].copy()

print("\n========== DATA SPLIT ==========")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nTraining target dates:")
print(
    feat["target_date"].iloc[:split].min(),
    "to",
    feat["target_date"].iloc[:split].max()
)

print("\nTesting target dates:")
print(
    feat["target_date"].iloc[split:].min(),
    "to",
    feat["target_date"].iloc[split:].max()
)


# ============================================================
# 5. SAVE OUTPUTS
# ============================================================

X_train.to_csv(DATA_DIR / "X_train.csv", index=False)
X_test.to_csv(DATA_DIR / "X_test.csv", index=False)

y_train.to_frame("AQI_tomorrow").to_csv(
    DATA_DIR / "y_train.csv",
    index=False
)

y_test.to_frame("AQI_tomorrow").to_csv(
    DATA_DIR / "y_test.csv",
    index=False
)

print("\nPhase 6 completed successfully!")
print("Saved:")
print("data/X_train.csv")
print("data/X_test.csv")
print("data/y_train.csv")
print("data/y_test.csv")