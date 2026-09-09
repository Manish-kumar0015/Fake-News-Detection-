# 📰 Fake News Detection System

An end-to-end Fake News Detection and Verification System built using
Natural Language Processing (NLP), TF-IDF, Machine Learning, NewsAPI,
and Streamlit.

The system classifies news as **Likely Real**, **Likely Fake**, or
**Uncertain** by combining machine-learning confidence with external
evidence retrieved from related news articles.

---

## 🚀 Project Overview

Fake news can spread rapidly through online platforms and can be
difficult to identify manually.

This project develops a machine-learning based system that analyzes
news text and provides a prediction. To improve the verification
process, the system also retrieves related articles using NewsAPI and
calculates their textual similarity with the input news.

The final decision is generated using a confidence-based decision
engine.

### System Architecture

```text
                    User Input
                        |
                        v
                Text Preprocessing
                        |
                        v
                   TF-IDF
                        |
                        v
              Logistic Regression
                        |
                 ML Probabilities
                        |
              +---------+---------+
              |                   |
              v                   v
        ML Prediction         NewsAPI
                                  |
                                  v
                       Related News Articles
                                  |
                                  v
                         TF-IDF Similarity
                                  |
              +-------------------+
              |
              v
        Decision Engine
              |
       +------+------+------+
       |             |      |
       v             v      v
   Likely Real   Likely Fake  Uncertain