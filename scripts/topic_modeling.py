import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# Read translated title data
df_analysis = pd.read_csv("youtube_shorts_translated.csv")

# Preprocess text (remove empty values)
df_analysis["title_translated"] = df_analysis["title_translated"].astype(str).fillna("")

# Processing English titles
vectorizer = TfidfVectorizer(stop_words="english", max_features=15)
tfidf_matrix = vectorizer.fit_transform(df_analysis["title_translated"])

# Keyword Importance
feature_names = vectorizer.get_feature_names_out()
print("Top 10 most common keywords:", feature_names[:10])

# Vectorize text (LDA requires CountVectorizer instead of TF-IDF)
vectorizer = CountVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["title_translated"])

# Train LDA model
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)  # 5 个主题
lda_model.fit(X)

# Capture topics
def print_topics(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx+1}: ", [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])

print("🎯 YouTube Shorts Topics Analysis：")
print_topics(lda_model, vectorizer.get_feature_names_out())

# Get the topic distribution of each video
topic_distributions = lda_model.transform(X)
df["dominant_topic"] = np.argmax(topic_distributions, axis=1)  # 找出每个视频的主主题

# Calculate the average subscription growth per topic
df["sub_growth_rate"] = df["subscribers"] / df["views"]  # 订阅增长率
topic_growth = df.groupby("dominant_topic")["sub_growth_rate"].mean().reset_index()

# Output the subscription growth rate for each topic
print("Average subscription growth rate per Shorts topic：")
print(topic_growth)

