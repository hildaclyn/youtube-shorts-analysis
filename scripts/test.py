import scipy.stats as stats

# Select data for high growth (topics 1 & 0) and low growth (topics 2 & 4)
high_growth = df[df["dominant_topic"].isin([0, 1])]["sub_growth_rate"]
low_growth = df[df["dominant_topic"].isin([2, 4])]["sub_growth_rate"]

# t-test
t_stat, p_value = stats.ttest_ind(high_growth, low_growth)
print(f"A/B test result: t={t_stat:.3f}, p-value={p_value:.3f}")

if p_value < 0.05:
    print("The difference in growth rates between the high growth topics (Topics 1 & 0) and the low growth topics (Topics 2 & 4) is significant!")
else:
    print("The difference between high growth & low growth themes is not significant")
