import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Load data
df = pd.read_csv("data/clean.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

pollutants = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]


# 1. AQI trend
df["AQI"].plot(figsize=(14, 4), title="AQI over time")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.tight_layout()
plt.savefig("graphs/aqi_trend.png", dpi=300)
plt.show()
plt.close()


# 2. Monthly AQI pattern
df.groupby(df.index.month)["AQI"].mean().plot(
    kind="bar",
    title="Average AQI by Month"
)
plt.xlabel("Month")
plt.ylabel("Average AQI")
plt.tight_layout()
plt.savefig("graphs/monthly_aqi.png", dpi=300)
plt.show()
plt.close()


# 3. Seasonal decomposition
result = seasonal_decompose(
    df["AQI"],
    model="additive",
    period=365
)

result.plot()
plt.tight_layout()
plt.savefig("graphs/seasonal_decomposition.png", dpi=300)
plt.show()
plt.close()


# 4. Correlation heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(
    df[pollutants + ["AQI"]].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Pollutants vs AQI Correlation")
plt.tight_layout()
plt.savefig("graphs/correlation_heatmap.png", dpi=300)
plt.show()
plt.close()


# 5. ACF
plot_acf(df["AQI"], lags=40)
plt.title("AQI Autocorrelation")
plt.tight_layout()
plt.savefig("graphs/acf.png", dpi=300)
plt.show()
plt.close()


# 6. PACF
plot_pacf(df["AQI"], lags=40)
plt.title("AQI Partial Autocorrelation")
plt.tight_layout()
plt.savefig("graphs/pacf.png", dpi=300)
plt.show()
plt.close()


# 7. Stationarity test
p = adfuller(df["AQI"])[1]

print("ADF p-value:", p)

if p < 0.05:
    print("Stationary")
else:
    print("Non-stationary — differencing may be needed")