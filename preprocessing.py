import pandas as pd
import numpy as np

df = pd.read_csv("data/city_day.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df[df["City"] == "Delhi"].sort_values("Date").set_index("Date")

print(df.info())
print(df.isna().mean().sort_values(ascending=False))   # % missing per column

# Ensure a continuous daily index
df = df.asfreq("D")

# Handle missing values (time-aware, no leakage from the future)
pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
df[pollutants] = df[pollutants].interpolate(method="time", limit=7)
df = df.dropna(subset=["AQI"])

# Outliers: clip extreme values rather than delete (breaks time continuity)
for c in pollutants:
    df[c] = df[c].clip(upper=df[c].quantile(0.995))

df.to_csv("data/clean.csv")