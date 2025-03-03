from googleapiclient.discovery import build
import pandas as pd
import time
import isodate

# Your YouTube API Key 
#API_KEY = 
youtube = build("youtube", "v3", developerKey=API_KEY)

def get_shorts_data(query="Shorts", max_pages=10):
    """
    Fetch trending YouTube Shorts video data, including:
    - video_id, title, views, likes, comments, video duration, channel subscribers
    """

    all_videos = []  # List to store video data
    next_page_token = None  # Initialize pagination token

    for _ in range(max_pages):  # Iterate through multiple pages
        request = youtube.search().list(
            part="snippet", q=query, type="video", videoDuration="short",
            order="viewCount", maxResults=50, pageToken=next_page_token
        )
        response = request.execute()

        # Process each video in the search results
        for item in response.get("items", []):
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            published_at = item["snippet"]["publishedAt"]
            channel_id = item["snippet"]["channelId"]

            # Fetch video statistics & content details
            video_stats = youtube.videos().list(
                part="statistics,contentDetails", id=video_id
            ).execute()

            stats = video_stats["items"][0]["statistics"]
            details = video_stats["items"][0]["contentDetails"]

            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))
            comments = int(stats.get("commentCount", 0))
            duration = details.get("duration", "PT0S")

            # Convert ISO 8601 duration format to seconds
            video_length = isodate.parse_duration(duration).total_seconds()

            # Fetch channel statistics (subscriber count)
            channel_info = youtube.channels().list(part="statistics", id=channel_id).execute()
            subscribers = int(channel_info["items"][0]["statistics"].get("subscriberCount", 0))

            # Compute sub_growth_rate (Prevent division by zero)
            sub_growth_rate = subscribers / views if views > 0 else 0

            # Compute estimated watch duration (Assume 60% completion rate)
            watch_duration = views * 0.6 * video_length
            watch_completion_rate = watch_duration / video_length if video_length > 0 else 0

            # Append extracted data to the list
            all_videos.append({
                "date": published_at,
                "video_id": video_id,
                "title": title,
                "channel_id": channel_id,
                "views": views,
                "likes": likes,
                "comments": comments,
                "video_length": video_length,
                "subscribers": subscribers,
                "sub_growth_rate": sub_growth_rate,  # Added sub growth rate
                "watch_duration": watch_duration,  # Added watch duration
                "watch_completion_rate": watch_completion_rate  # Added watch completion rate
            })

        # Handle pagination
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break  # Exit loop if no more pages

        time.sleep(1)  # Add delay to avoid exceeding API rate limits

    return pd.DataFrame(all_videos)  # Convert list to Pandas DataFrame

# Execute the script to fetch Shorts video data
df_shorts = get_shorts_data(max_pages=10)
df_shorts.to_csv("youtube_shorts_videos.csv", index=False)  # Save results as CSV
print(f"✅ Data collection completed. A total of {len(df_shorts)} Shorts video records were obtained.")

# Display first few rows to verify
print(df_shorts[["title", "sub_growth_rate", "watch_completion_rate"]].head())
