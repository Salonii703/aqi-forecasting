# import pandas as pd
# import numpy as np

# df = pd.read_csv("data/city_day.csv")
# df["Date"] = pd.to_datetime(df["Date"])
# df = df[df["City"] == "Delhi"].sort_values("Date").set_index("Date")

# print(df.info())
# print(df.isna().mean().sort_values(ascending=False))   # % missing per column

# # Ensure a continuous daily index
# df = df.asfreq("D")

# # Handle missing values (time-aware, no leakage from the future)
# pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
# df[pollutants] = df[pollutants].interpolate(method="time", limit=7)
# df = df.dropna(subset=["AQI"])

# # Outliers: clip extreme values rather than delete (breaks time continuity)
# for c in pollutants:
#     df[c] = df[c].clip(upper=df[c].quantile(0.995))

# df.to_csv("data/clean.csv")
import pandas as pd
from pathlib import Path

# 1. Load dataset
df = pd.read_csv("data/city_day.csv")
df["Date"] = pd.to_datetime(df["Date"])

# 2. Filter Delhi and sort chronologically
df = (
    df[df["City"] == "Delhi"]
    .sort_values("Date")
    .set_index("Date")
)

# 3. Ensure continuous daily dates
df = df.asfreq("D")

# 4. Remove rows where the target AQI is missing
df = df.dropna(subset=["AQI"])

# 5. Chronological 80/20 train-test split
split = int(len(df) * 0.8)

train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

# 6. Define pollutant columns
pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# 7. Fill missing pollutant values using past observations only
for c in pollutants:
    train[c] = train[c].ffill(limit=7)
    test[c] = test[c].ffill(limit=7)

# Use the last available training value to initialize test imputation
for c in pollutants:
    if pd.notna(train[c].iloc[-1]):
        test[c] = test[c].fillna(train[c].iloc[-1], limit=7)

# 8. Clip outliers using training data thresholds only
for c in pollutants:
    upper_limit = train[c].quantile(0.995)
    train[c] = train[c].clip(upper=upper_limit)
    test[c] = test[c].clip(upper=upper_limit)

# 9. Save cleaned training and testing datasets
Path("data").mkdir(parents=True, exist_ok=True)

train.to_csv("data/train.csv")
test.to_csv("data/test.csv")

# 10. Display results
print("Total observations:", len(df))
print("Training observations:", len(train))
print("Testing observations:", len(test))

print("\nTraining period:")
print(train.index.min(), "to", train.index.max())

print("\nTesting period:")
print(test.index.min(), "to", test.index.max())

print("\nMissing values in training data:")
print(train[pollutants + ["AQI"]].isna().sum())

print("\nMissing values in testing data:")
print(test[pollutants + ["AQI"]].isna().sum())

print("\nDatasets saved successfully:")
print("data/train.csv")
print("data/test.csv")