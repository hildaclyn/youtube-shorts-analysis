import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
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
X = vectorizer.fit_transform(df_analysis["title_translated"])

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
df_analysis["dominant_topic"] = np.argmax(topic_distributions, axis=1)  # 找出每个视频的主主题

# Calculate the average subscription growth per topic
df_analysis["sub_growth_rate"] = df_analysis["subscribers"] / df_analysis["views"]  # 订阅增长率
topic_growth = df_analysis.groupby("dominant_topic")["sub_growth_rate"].mean().reset_index()

plt.figure(figsize=(10,6))
sns.barplot(x=topic_growth["dominant_topic"], y=topic_growth["sub_growth_rate"], palette="Blues_d")
plt.title("Different Shorts topics vs. subscription growth rate")
plt.xlabel("Shorts Topic")
plt.ylabel("Average subscription growth rate")
plt.xticks(rotation=45)
plt.show()

# Output the subscription growth rate for each topic
print("Average subscription growth rate per Shorts topic：")
print(topic_growth)

# ------------------------------------- #
"""           A / B Testing           """
# ------------------------------------- #

# Get the topic distribution of each video
topic_distributions = lda_model.transform(X)

# Find the main theme of each video
df_analysis["dominant_topic"] = np.argmax(topic_distributions, axis=1)

# check dominant_topic
#print(df_analysis.head())

# Select data for high growth (topics 1 & 0) and low growth (topics 2 & 3)
high_growth = df_analysis[df["dominant_topic"].isin([0, 1])]["sub_growth_rate"]
low_growth = df_analysis[df_analysis["dominant_topic"].isin([2, 4])]["sub_growth_rate"]

# 进行 t-test
t_stat, p_value = stats.ttest_ind(high_growth, low_growth)
print(f"🔥 A/B 测试结果: t={t_stat:.3f}, p-value={p_value:.3f}")

if p_value < 0.05:
    print("The difference in growth rates between the high growth topics (Topics 1 & 0) and the low growth topics (Topics 2 & 4) is significant!")
else:
    print("The difference between high growth & low growth themes is not significant")
