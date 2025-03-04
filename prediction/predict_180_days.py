import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import load_data, train_xgb, train_lstm, create_lstm_dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

#Load data
file_path = "youtube_shorts_translated.csv"
df_growth = load_data(file_path)

#Train XGBoost
features = ["year", "month", "day", "weekday"] + [f"lag_{lag}" for lag in range(1, 8)] + ["rolling_mean_7"]
X = df_growth[features]
y = df_growth["y"]
xgb_model = train_xgb(X, y)

# Generate data for the next 180 days
future_dates = pd.date_range(start=df_growth["ds"].max() + pd.Timedelta(days=1), periods=180, freq='D')
future_features = pd.DataFrame({
 "year": future_dates.year, "month": future_dates.month,
 "day": future_dates.day,
 "weekday": future_dates.weekday
})

# XGBoost prediction
future_xgb_predictions = xgb_model.predict(future_features)

# Normalized data for LSTM
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_growth["y"].values.reshape(-1, 1))

X_lstm, y_lstm = create_lstm_dataset(scaled_data, time_steps=60)
lstm_model = train_lstm(X_lstm, y_lstm)

# LSTM predicts the next 180 days
future_lstm_predictions = []
last_input = scaled_data[-60:].reshape(1, 60, 1)

for _ in range(180):
 next_pred = lstm_model.predict(last_input)
 future_lstm_predictions.append(next_pred[0, 0])
 last_input = np.append(last_input[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

future_lstm_predictions = scaler.inverse_transform(np.array(future_lstm_predictions).reshape(-1, 1))

# Combined Hybrid predictions
alpha=0.6
future_hybrid_predictions = alpha * future_xgb_predictions + (1 - alpha) * future_lstm_predictions.flatten()

# Draw prediction results
plt.figure(figsize=(12, 6))
plt.plot(future_dates, future_xgb_predictions, label="XGBoost Prediction", linestyle="dashed")
plt.plot(future_dates, future_lstm_predictions, label="LSTM Prediction", linestyle="dashed")
plt.plot(future_dates, future_hybrid_predictions, label="Hybrid Prediction", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Subscribers")
plt.title("180-Day Prediction")
plt.legend()
plt.show()


min_length = min(len(y_test), len(xgb_predictions), len(lstm_predictions), len(hybrid_predictions))

y_test = y_test[:min_length]
xgb_predictions = xgb_predictions[:min_length]
lstm_predictions = lstm_predictions[:min_length]
hybrid_predictions = hybrid_predictions[:min_length]


# Calculate RMSE and MAE
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
lstm_mae = mean_absolute_error(y_test, lstm_predictions)
hybrid_mae = mean_absolute_error(y_test, hybrid_predictions)

xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))
hybrid_rmse = np.sqrt(mean_squared_error(y_test, hybrid_predictions))


print(f"XGBoost MAE: {xgb_mae}, RMSE: {xgb_rmse}")
print(f"LSTM MAE: {lstm_mae}, RMSE: {lstm_rmse}")
print(f"Hybrid MAE: {hybrid_mae}, RMSE: {hybrid_rmse}")

plt.figure(figsize=(12, 6))

plt.plot(future_dates, abs(future_xgb_predictions - future_hybrid_predictions), label="XGBoost-Hybrid Error", linestyle="dashed", color="blue")
plt.plot(future_dates, abs(future_lstm_predictions - future_hybrid_predictions), label="LSTM-Hybrid Error", linestyle="dashed", color="green")

plt.xlabel("Date")
plt.ylabel("Absolute Error")
plt.title("Error Comparison Over Time")
plt.legend()
plt.show()

# save predicted data
#import pickle

#with open("future_predictions.pkl", "wb") as f:
#    pickle.dump({"future_dates": future_dates, "future_hybrid_predictions": future_hybrid_predictions}, f)

