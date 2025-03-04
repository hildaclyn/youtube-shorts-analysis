import pandas as pd
from prophet import Prophet

# 🔹 Read data
df = pd.read_csv("youtube_shorts_translated.csv")

# 🔹 Process the time column
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None) # Make sure it is in datetime format
df = df.sort_values("date") # Sort by time

# 🔹 Select columns for Prophet
df_prophet = df[["date", "subscribers", "views", "likes", "comments", "watch_duration", "watch_completion_rate"]].copy()
df_prophet.rename(columns={"date": "ds", "subscribers": "y"}, inplace=True)

# 🔹 Prophet Need numerical data, remove NaN values
df_prophet.dropna(inplace=True)

# 🔹 Initialize Prophet model and add regression variables
model = Prophet()
model.add_regressor("views")
model.add_regressor("likes")
model.add_regressor("comments")
model.add_regressor("watch_duration")
model.add_regressor("watch_completion_rate")

# 🔹 Train Prophet model
model.fit(df_prophet)

# 🔹 Forecast the next 180 days
future = model.make_future_dataframe(periods=180)
future["views"] = df["views"].mean() # Fill future data with historical mean
future["likes"] = df["likes"].mean()
future["comments"] = df["comments"].mean()
future["watch_duration"] = df["watch_duration"].mean()
future["watch_completion_rate"] = df["watch_completion_rate"].mean()

# 🔹 Make predictions
forecast = model.predict(future)

# 🔹 Draw prediction results
import matplotlib.pyplot as plt
fig = model.plot(forecast)
plt.title("YouTube Shorts Subscriber Growth Prediction")
plt.show()

model.plot_components(forecast)
