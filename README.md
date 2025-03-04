# YouTube Shorts Growth Analysis

This project analyzes YouTube Shorts user growth trends, exploring key factors that impact subscriber growth. It includes data collection, feature engineering, NLP-based topic modeling, A/B testing, forecasting, and data visualization.

## 📌 Project Overview

The goal of this project is to:
- Collect and process **YouTube Shorts video data** using the YouTube API.
- Perform **topic modeling (LDA, BERT)** to identify key video categories.
- Conduct **A/B testing** to evaluate high vs. low subscriber growth content.
- Build a **hybrid forecasting model (XGBoost + LSTM)** to predict future subscriber growth.
- Generate **data visualizations** and insights into user engagement trends.

---

## 🚀 Features

### 🔹 **Data Collection**
- Extracting YouTube Shorts data (views, likes, comments, video duration, and subscribers).
- Cleaning and preprocessing the data.

### 🔹 **Feature Engineering**
- Calculating engagement metrics such as watch completion rate and subscriber growth rate.
- Creating time-series features (e.g., lag features, rolling averages).

### 🔹 **Natural Language Processing (NLP)**
- Using **LDA** & **BERT** for **topic modeling** on video titles.
- Identifying the most engaging content categories.

### 🔹 **A/B Testing**
- Conducting **statistical hypothesis testing** to compare subscriber growth across different content types.
- Evaluating the impact of **video duration, posting time, and engagement metrics**.

### 🔹 **Forecasting Subscriber Growth**  🆕
This project integrates **XGBoost + LSTM** to predict future YouTube Shorts subscriber growth.

#### **🔹 Model Components**
- **XGBoost:** Captures short-term trends and feature-based relationships.
- **LSTM (Long Short-Term Memory):** Captures long-term dependencies and temporal patterns.
- **Hybrid Model:** Combines XGBoost and LSTM using a weighted ensemble approach.

#### **🔹 Forecasting Results**
- **Performance Metrics:**
  - XGBoost **MAE**: `6,197,075` | **RMSE**: `9,508,780`
  - LSTM **MAE**: `7,082,377` | **RMSE**: `12,889,239`
  - Hybrid **MAE**: `6,908,920` | **RMSE**: `9,774,821`
  
- **Trend Insights:**
  - Predicted subscriber growth fluctuates cyclically, influenced by **weekly engagement patterns**.
  - **Peak growth:** `2025-03-9` 🚀
  - **Lowest growth:** `2025-05-6` 📉
  
- **Visualization:**
  - 📈 **Historical vs. Future Growth:** Comparison of past trends with future predictions.
  - 📊 **Error Analysis:** Evaluates prediction error over time.
  - 🔍 **Trend Forecasting:** Identifies upcoming peaks and declines.

---

## 🎯 Result

### 🔹 **Five Top Content Topics** (From NLP Analysis)
1. **Gaming & School Vlogs** 🎮📚
2. **School & Teacher Life** 📖🏫 (🌟🌟🔝)
3. **Comedy & Trending Entertainment** 🤣📺
4. **Classic Cartoons & Viral Clips** 🎞️🐭
5. **School & Fun Challenges** 🏆😆

### 🔹 **Subscriber Growth Forecasting Insights** (From XGBoost + LSTM)
- Hybrid model predicts **stable subscriber growth** with periodic fluctuations.
- Future content planning can leverage predicted **growth peaks** to optimize posting schedules.
- Data-driven strategies for **A/B testing** can align with **forecasted high-growth periods**.

---

## 📊 Visualization Examples
- **Engagement trends across different video categories**
- **A/B test results for different content strategies**
- **Forecasted subscriber growth with peak/low periods**

---

## 🛠️ Tech Stack
- **Data Processing:** `Pandas`, `NumPy`, `Scikit-learn`
- **Machine Learning:** `XGBoost`, `TensorFlow (LSTM)`
- **NLP:** `spaCy`, `Gensim (LDA)`, `BERT`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Forecasting:** `XGBoost + LSTM Hybrid Model`

---

## 🔥 Future Work
- Incorporate **seasonality adjustments** for long-term forecasting.
- Implement **real-time trend monitoring** using **YouTube API + Streaming Data**.
- Explore **reinforcement learning** for content optimization.

---


