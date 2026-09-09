# ============================================================
# FAKE NEWS DETECTION SYSTEM - STREAMLIT APPLICATION
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import streamlit as st
import joblib
import re
import requests
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# ============================================================
# 3. LOAD TRAINED ML MODELS
# ============================================================

# Load the TF-IDF vectorizer that was trained in the notebook
vectorization = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

# Load the trained Logistic Regression model
LR = joblib.load(
    "models/logistic_regression.pkl"
)


# ============================================================
# 4. TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    # Convert input to string and lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r'https?://\S+',
        ' ',
        text
    )

    # Remove www URLs
    text = re.sub(
        r'www\.\S+',
        ' ',
        text
    )

    # Remove HTML tags
    text = re.sub(
        r'<.*?>',
        ' ',
        text
    )

    # Keep only English letters and spaces
    text = re.sub(
        r'[^a-zA-Z\s]',
        ' ',
        text
    )

    # Replace multiple spaces with one space
    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    # Remove extra spaces from beginning/end
    return text.strip()


# ============================================================
# 5. MACHINE LEARNING PREDICTION
# ============================================================

def get_ml_prediction(news):

    # Clean the input news
    news = clean_text(news)

    # Convert cleaned text into TF-IDF features
    vector = vectorization.transform([news])

    # Get predicted class
    pred = LR.predict(vector)[0]

    # Get probability for each class
    proba = LR.predict_proba(vector)[0]

    # According to your trained model:
    # class 0 = Fake
    # class 1 = Real

    fake_prob = float(proba[0])
    real_prob = float(proba[1])

    return pred, fake_prob, real_prob


# ============================================================
# 6. NEWSAPI CONFIGURATION
# ============================================================

# Read NewsAPI key from Streamlit secrets.
#
# In local development:
# .streamlit/secrets.toml
#
# NEWS_API_KEY = "YOUR_NEW_API_KEY"

try:
    API_KEY = st.secrets["NEWS_API_KEY"]

except Exception:
    API_KEY = None


# ============================================================
# 7. SEARCH FOR RELATED NEWS USING NEWSAPI
# ============================================================

def search_news(query):

    # If API key is not available,
    # return an empty list.
    if not API_KEY:
        return []

    url = "https://newsapi.org/v2/everything"

    # Use a shorter query instead of sending
    # the entire article to NewsAPI.
    cleaned_query = clean_text(query)

    # Take approximately the first 20 words
    # to create a manageable search query.
    query_words = cleaned_query.split()[:20]

    search_query = " ".join(query_words)

    params = {
        "q": search_query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 10,
        "apiKey": API_KEY
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        # If NewsAPI returns an error,
        # return an empty list.
        if response.status_code != 200:
            return []

        data = response.json()

        return data.get(
            "articles",
            []
        )

    except requests.RequestException:

        return []


# ============================================================
# 8. CALCULATE EXTERNAL EVIDENCE SCORE
# ============================================================

def evidence_score(news, articles):

    # If no articles were found,
    # there is no external evidence.
    if len(articles) == 0:
        return 0.0, None

    # First document is the user's news.
    docs = [news]

    # Add title + description of every
    # retrieved article.
    for article in articles:

        title = article.get(
            "title",
            ""
        )

        description = article.get(
            "description",
            ""
        )

        docs.append(
            title + " " + description
        )

    # Create a temporary TF-IDF vectorizer
    # for comparing the input news with
    # retrieved articles.
    temp_vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    # Convert all documents into TF-IDF vectors.
    matrix = temp_vectorizer.fit_transform(
        docs
    )

    # Calculate cosine similarity between
    # user's news and all retrieved articles.
    sims = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    )[0]

    # Find the article with highest similarity.
    best = np.argmax(sims)

    return (
        float(sims[best]),
        articles[best]
    )


# ============================================================
# 9. FINAL DECISION ENGINE
# ============================================================

def final_prediction(news):

    # --------------------------------------------------------
    # Step 1: Get ML prediction
    # --------------------------------------------------------

    pred, fake_prob, real_prob = get_ml_prediction(
        news
    )

    # --------------------------------------------------------
    # Step 2: Search for external evidence
    # --------------------------------------------------------

    articles = search_news(
        news
    )

    # --------------------------------------------------------
    # Step 3: Calculate evidence similarity
    # --------------------------------------------------------

    evidence, best_article = evidence_score(
        news,
        articles
    )

    # --------------------------------------------------------
    # Step 4: Make final decision
    # --------------------------------------------------------

    # Strong external evidence + strong real probability
    if (
        len(articles) > 0
        and evidence > 0.50
        and real_prob > 0.70
    ):

        verdict = "Likely Real"

    # Strongly fake prediction + very low
    # external similarity
    elif (
        len(articles) > 0
        and evidence < 0.10
        and fake_prob > 0.70
    ):

        verdict = "Likely Fake"

    # Very high ML confidence
    elif real_prob > 0.90:

        verdict = "Likely Real"

    elif fake_prob > 0.90:

        verdict = "Likely Fake"

    # Otherwise, do not force a decision.
    else:

        verdict = "Uncertain"

    return {
        "prediction": pred,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "evidence": evidence,
        "best_article": best_article,
        "verdict": verdict
    }


# ============================================================
# 10. STREAMLIT USER INTERFACE
# ============================================================

st.title(
    "📰 Fake News Detection System"
)

st.write(
    "Enter a news headline or article text "
    "to analyze whether it is likely real or fake."
)


# ============================================================
# 11. NEWS INPUT BOX
# ============================================================

news_text = st.text_area(
    "Enter News",
    height=250,
    placeholder=(
        "Paste a news headline or article here..."
    )
)


# ============================================================
# 12. ANALYZE NEWS BUTTON
# ============================================================

if st.button("Analyze News"):

    # --------------------------------------------------------
    # Check whether user entered anything
    # --------------------------------------------------------

    if not news_text.strip():

        st.warning(
            "Please enter some news text."
        )

    else:

        # ----------------------------------------------------
        # Run complete hybrid prediction system
        # ----------------------------------------------------

        result = final_prediction(
            news_text
        )

        # Extract results
        fake_prob = result["fake_prob"]
        real_prob = result["real_prob"]
        evidence = result["evidence"]
        best_article = result["best_article"]
        verdict = result["verdict"]


        # ====================================================
        # 13. DISPLAY FINAL VERDICT
        # ====================================================

        st.subheader(
            "Final Verdict"
        )

        if verdict == "Likely Real":

            st.success(
                "🟢 Likely Real"
            )

        elif verdict == "Likely Fake":

            st.error(
                "🔴 Likely Fake"
            )

        else:

            st.warning(
                "🟡 Uncertain"
            )


        # ====================================================
        # 14. DISPLAY ML PROBABILITIES
        # ====================================================

        st.subheader(
            "ML Model Confidence"
        )

        st.write(
            f"Real Probability: {real_prob:.2%}"
        )

        st.write(
            f"Fake Probability: {fake_prob:.2%}"
        )


        # ====================================================
        # 15. DISPLAY EVIDENCE SCORE
        # ====================================================

        st.subheader(
            "External Evidence"
        )

        st.write(
            f"Evidence Similarity Score: "
            f"{evidence:.2%}"
        )


        # ====================================================
        # 16. DISPLAY SUPPORTING ARTICLE
        # ====================================================

        if best_article:

            st.subheader(
                "Most Similar Retrieved Article"
            )

            article_title = best_article.get(
                "title",
                "No title available"
            )

            article_description = best_article.get(
                "description",
                ""
            )

            article_url = best_article.get(
                "url",
                ""
            )

            source = best_article.get(
                "source",
                {}
            )

            source_name = source.get(
                "name",
                "Unknown source"
            )

            st.write(
                f"**Source:** {source_name}"
            )

            st.write(
                f"**Title:** {article_title}"
            )

            if article_description:

                st.write(
                    f"**Description:** "
                    f"{article_description}"
                )

            if article_url:

                st.markdown(
                    f"[Read Supporting Article]({article_url})"
                )

        else:

            st.info(
                "No related external article "
                "was found. The result is based "
                "mainly on the ML model."
            )


# ============================================================
# END OF APPLICATION
# ============================================================