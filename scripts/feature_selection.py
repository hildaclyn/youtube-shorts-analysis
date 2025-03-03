import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression

# Read Data
df_analysis = pd.read_csv("youtube_shorts_analysis.csv")

# Calculating Interaction Rate Features
df_analysis["like_rate"] = df_analysis["likes"] / df_analysis["views"]
df_analysis["comment_rate"] = df_analysis["comments"] / df_analysis["views"]
df_analysis["engagement_rate"] = (df_analysis["likes"] + df_analysis["comments"]) / df_analysis["views"]

# Selecting variables for feature selection
features = ["views", "likes", "comments", "like_rate", "comment_rate", "engagement_rate", "watch_completion_rate"]
X = df_analysis[features].fillna(0)  # 填充缺失值
y = df_analysis["sub_growth_rate"].fillna(0)

# Feature selection using SelectKBest
selector = SelectKBest(score_func=f_regression, k="all")  # 选择所有特征，评估其重要性
selector.fit(X, y)
scores = selector.scores_

# Create a DataFrame to store feature scores
feature_scores = pd.DataFrame({"Feature": features, "Score": scores})
feature_scores = feature_scores.sort_values(by="Score", ascending=False)

# Visualizing feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Score", y="Feature", data=feature_scores, palette="coolwarm")
plt.title("Feature Importance for Predicting Subscription Growth Rate")
plt.xlabel("F-Score")
plt.ylabel("Feature")
plt.show()

# Create a DataFrame for feature importance scores
feature_scores_df = pd.DataFrame({"Feature": features, "Score": scores}).sort_values(by="Score", ascending=False)

# Show the DataFrame
import IPython.display as display
display.display(feature_scores_df)
