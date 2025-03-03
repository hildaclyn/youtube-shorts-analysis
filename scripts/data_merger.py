# Read Shorts video & channel data
# If in the same cell, comment it
df_shorts = pd.read_csv("youtube_shorts_videos.csv")
df_channels = pd.read_csv("youtube_shorts_creators.csv")

# Merge data
df_combined = df_shorts.merge(df_channels, on="channel_id", how="left")
df_combined["subscribers"] = df_combined["subscribers_y"].fillna(df_combined["subscribers_x"])
df_combined = df_combined.drop(columns=["subscribers_x", "subscribers_y"])

# Save analysis data
df_combined.to_csv("youtube_shorts_analysis.csv", index=False)
print("The merger of Shorts video data and creator growth data is complete!")
