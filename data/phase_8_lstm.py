import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------
# 1. Load Delhi AQI data
# --------------------------------------------------

DATA_DIR = Path("data")

df = pd.read_csv(
    DATA_DIR / "clean.csv",
    parse_dates=["Date"],
    index_col="Date"
)

df = df.sort_index()

if "City" in df.columns:
    df = df[df["City"] == "Delhi"]

df = df[~df.index.duplicated(keep="first")]

cols = ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

data = df[cols].apply(pd.to_numeric, errors="coerce")
data = data.replace([np.inf, -np.inf], np.nan).dropna()
data = data.sort_index()

if len(data) < 100:
    raise ValueError("Not enough complete observations to train the LSTM.")

print("Dataset shape:", data.shape)
print("Date range:", data.index.min(), "to", data.index.max())


# --------------------------------------------------
# 2. Create chronological train/test split
# --------------------------------------------------

WINDOW = 14

n = int(len(data) * 0.8)

if n <= WINDOW or n >= len(data):
    raise ValueError("Not enough observations for the selected split.")

# Fit the scaler ONLY on training data
scaler = MinMaxScaler()
scaler.fit(data.iloc[:n])

# Transform all observations using the training scaler
scaled = scaler.transform(data)

print("Training observations:", n)
print("Testing observations:", len(data) - n)


# --------------------------------------------------
# 3. Create sequences of previous 14 days
# --------------------------------------------------

def make_windows(arr, window):
    X_sequences = []
    y_targets = []

    for i in range(window, len(arr)):
        # Previous 14 days of AQI and pollutants
        X_sequences.append(arr[i - window:i])

        # AQI for the target day
        y_targets.append(arr[i, 0])

    return np.array(X_sequences), np.array(y_targets)


Xl, yl = make_windows(scaled, WINDOW)

# Target indices correspond to data rows WINDOW onward
target_indices = np.arange(WINDOW, len(data))

# Training targets must be before the chronological split
train_mask = target_indices < n
test_mask = target_indices >= n

Xl_train = Xl[train_mask]
yl_train = yl[train_mask]

Xl_test = Xl[test_mask]
yl_test = yl[test_mask]

test_dates = data.index[target_indices[test_mask]]

print("LSTM training sequences:", Xl_train.shape)
print("LSTM testing sequences:", Xl_test.shape)


# --------------------------------------------------
# 4. Build the LSTM model
# --------------------------------------------------

model = Sequential([
    Input(shape=(WINDOW, len(cols))),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()


# --------------------------------------------------
# 5. Train the model
# --------------------------------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    Xl_train,
    yl_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    shuffle=False,
    verbose=1
)


# --------------------------------------------------
# 6. Convert scaled AQI back to original AQI units
# --------------------------------------------------

def inverse_aqi(values):
    values = np.asarray(values).reshape(-1)

    # AQI is the first column in cols
    return values * (
        scaler.data_max_[0] - scaler.data_min_[0]
    ) + scaler.data_min_[0]


lstm_pred_scaled = model.predict(Xl_test, verbose=0).reshape(-1)

lstm_pred = inverse_aqi(lstm_pred_scaled)
lstm_true = inverse_aqi(yl_test)


# --------------------------------------------------
# 7. Evaluate the LSTM model
# --------------------------------------------------

mae = mean_absolute_error(lstm_true, lstm_pred)
rmse = np.sqrt(mean_squared_error(lstm_true, lstm_pred))

print("\nLSTM Test Results")
print("-----------------")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


# --------------------------------------------------
# 8. Save predictions
# --------------------------------------------------

predictions = pd.DataFrame({
    "Actual_AQI": lstm_true,
    "LSTM_Prediction": lstm_pred
}, index=test_dates)

predictions.index.name = "Date"

predictions.to_csv(DATA_DIR / "phase8_lstm_predictions.csv")

print("\nPredictions saved to data/phase8_lstm_predictions.csv")


# --------------------------------------------------
# 9. Plot actual vs predicted AQI
# --------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    test_dates,
    lstm_true,
    label="Actual AQI"
)

plt.plot(
    test_dates,
    lstm_pred,
    label="LSTM Predicted AQI"
)

plt.title("LSTM: Actual vs Predicted Delhi AQI")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    DATA_DIR / "phase8_lstm_plot.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 10. Plot training and validation loss
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("LSTM Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    DATA_DIR / "phase8_training_loss.png",
    dpi=300
)

plt.show()

print("Graphs saved in the data folder.")