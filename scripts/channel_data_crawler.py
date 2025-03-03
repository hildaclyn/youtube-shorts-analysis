def get_channel_growth(channel_ids):
    """
    Get subscription growth data for Shorts creators
    """
    channel_data = []
    
    for channel_id in channel_ids:
        channel_request = youtube.channels().list(
            part="statistics",
            id=channel_id
        ).execute()

        stats = channel_request["items"][0]["statistics"]

        channel_data.append({
            "channel_id": channel_id,
            "subscribers": int(stats.get("subscriberCount", 0)),
            "total_views": int(stats.get("viewCount", 0)),
            "video_count": int(stats.get("videoCount", 0))
        })

        time.sleep(1)  # Avoid API restrictions
    return pd.DataFrame(channel_data)

# Retrieve the Shorts creator's channel ID
channel_ids = df_shorts["channel_id"].unique()

# Get channel subscription growth data
df_channels = get_channel_growth(channel_ids)
df_channels.to_csv("youtube_shorts_creators.csv", index=False)
print(f"Crawling completed, a total of {len(df_channels)} creator data were obtained")

