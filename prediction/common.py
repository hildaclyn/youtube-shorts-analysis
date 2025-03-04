import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 数据读取
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df_growth = df.groupby("date")["subscribers"].sum().reset_index()
    df_growth = df_growth.rename(columns={"date": "ds", "subscribers": "y"})

    # 生成时间特征
    df_growth["year"] = df_growth["ds"].dt.year
    df_growth["month"] = df_growth["ds"].dt.month
    df_growth["day"] = df_growth["ds"].dt.day
    df_growth["weekday"] = df_growth["ds"].dt.weekday

    for lag in range(1, 8):
        df_growth[f"lag_{lag}"] = df_growth["y"].shift(lag)

    df_growth["rolling_mean_7"] = df_growth["y"].rolling(window=7).mean()

    df_growth = df_growth.dropna()
    return df_growth

# XGBoost 训练
def train_xgb(X_train, y_train):
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=300, learning_rate=0.05, max_depth=7)
    xgb_model.fit(X_train, y_train)
    return xgb_model

# LSTM 数据预处理
def create_lstm_dataset(data, time_steps=60):
    X_lstm, y_lstm = [], []
    for i in range(len(data) - time_steps):
        X_lstm.append(data[i : i + time_steps])
        y_lstm.append(data[i + time_steps])
    return np.array(X_lstm), np.array(y_lstm)

# LSTM 训练
def train_lstm(X_train_lstm, y_train_lstm, time_steps=60):
    model_lstm = Sequential([
        LSTM(128, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model_lstm.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=8)
    return model_lstm
