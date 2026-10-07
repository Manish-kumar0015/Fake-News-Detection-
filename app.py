# ============================================================
# FAKE NEWS DETECTION SYSTEM - STREAMLIT APPLICATION
# ============================================================

import json
import re
from pathlib import Path

import joblib
import streamlit as st

from fact_checker import (
    extract_claim,
    search_fact_checks,
    match_fact_checks,
    combine_ml_and_factcheck,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
)


# ============================================================
# 2. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# 3. LOAD THE AUTOMATICALLY SELECTED ML MODEL
# ============================================================

@st.cache_resource
def load_ml_artifacts():
    vectorizer = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.pkl"
    )

    model = joblib.load(
        MODEL_DIR / "best_model.pkl"
    )

    metadata_path = MODEL_DIR / "ml_metadata.json"

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return vectorizer, model, metadata


try:
    vectorizer, ml_model, ml_metadata = load_ml_artifacts()
    MODEL_LOAD_ERROR = None
except Exception as exc:
    vectorizer = None
    ml_model = None
    ml_metadata = {}
    MODEL_LOAD_ERROR = str(exc)


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        r"https?://\S+",
        " ",
        text,
    )

    text = re.sub(
        r"www\.\S+",
        " ",
        text,
    )

    text = re.sub(
        r"<.*?>",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# 5. ML PREDICTION
# ============================================================

def get_ml_prediction(news_text):
    if vectorizer is None or ml_model is None:
        raise RuntimeError(
            "ML model artifacts could not be loaded."
        )

    cleaned = clean_text(news_text)

    if not cleaned:
        raise ValueError(
            "The input does not contain usable text."
        )

    features = vectorizer.transform([cleaned])

    prediction = int(
        ml_model.predict(features)[0]
    )

    classes = list(
        getattr(
            ml_model,
            "classes_",
            [0, 1],
        )
    )

    if hasattr(
        ml_model,
        "predict_proba",
    ):
        probabilities = ml_model.predict_proba(
            features
        )[0]

        probability_map = {
            int(cls): float(prob)
            for cls, prob in zip(
                classes,
                probabilities,
            )
        }

        fake_probability = probability_map.get(
            0,
            0.0,
        )

        real_probability = probability_map.get(
            1,
            0.0,
        )

    else:
        # This should not happen because the training
        # notebook calibrates the selected model.
        decision = float(
            ml_model.decision_function(
                features
            )[0]
        )

        # Fallback conversion only.
        import math

        real_probability = 1.0 / (
            1.0 + math.exp(-decision)
        )

        fake_probability = (
            1.0 - real_probability
        )

    return {
        "prediction": prediction,
        "label": (
            "REAL"
            if prediction == 1
            else "FAKE"
        ),
        "fake_probability": fake_probability,
        "real_probability": real_probability,
    }


# ============================================================
# 6. API KEY CONFIGURATION
# ============================================================

FACTCHECK_API_KEY = st.secrets.get(
    "FACTCHECK_API_KEY",
    "",
)


# ============================================================
# 7. SIDEBAR
# ============================================================

st.sidebar.title("⚙️ System Information")

if MODEL_LOAD_ERROR:
    st.sidebar.error(
        "ML model loading failed."
    )
    st.sidebar.code(
        MODEL_LOAD_ERROR
    )
else:
    st.sidebar.success(
        "ML model loaded"
    )

st.sidebar.write(
    "**Selected model:** "
    + ml_metadata.get(
        "best_model_name",
        "Unknown",
    )
)

if ml_metadata.get("test_f1_macro") is not None:
    st.sidebar.write(
        "**Test Macro-F1:** "
        f"{ml_metadata['test_f1_macro']:.4f}"
    )

if FACTCHECK_API_KEY:
    st.sidebar.success(
        "Google Fact Check API connected"
    )
else:
    st.sidebar.warning(
        "Google Fact Check API key not configured"
    )


# ============================================================
# 8. MAIN UI
# ============================================================

st.title(
    "📰 Fake News Detection System"
)

st.write(
    "This system combines an automatically selected "
    "TF-IDF machine-learning classifier with "
    "Google's existing professional fact-check reviews."
)

st.info(
    "Important: Google Fact Check searches claims that "
    "have already been fact-checked. It is not a general "
    "Google News search. A new claim may therefore have "
    "no external fact-check even when it is true."
)


news_text = st.text_area(
    "Enter News",
    height=260,
    placeholder=(
        "Paste a news headline or article text here..."
    ),
)


# ============================================================
# 9. ANALYZE
# ============================================================

if st.button(
    "🔍 Analyze News",
    type="primary",
):

    if not news_text.strip():
        st.warning(
            "Please enter a news headline or article."
        )
        st.stop()

    if MODEL_LOAD_ERROR:
        st.error(
            "The trained ML model could not be loaded. "
            "Run the training notebook first."
        )
        st.stop()

    # --------------------------------------------------------
    # STEP 1: CLAIM
    # --------------------------------------------------------

    claim = extract_claim(
        news_text
    )

    st.subheader(
        "🔎 Claim Being Checked"
    )

    st.write(claim)

    # --------------------------------------------------------
    # STEP 2: ML MODEL
    # --------------------------------------------------------

    with st.spinner(
        "Running the selected ML model..."
    ):

        ml_result = get_ml_prediction(
            news_text
        )

    st.subheader(
        "🤖 Machine Learning Prediction"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "ML Prediction",
            ml_result["label"],
        )

    with col2:
        st.metric(
            "Fake Probability",
            f"{ml_result['fake_probability']:.2%}",
        )

    with col3:
        st.metric(
            "Real Probability",
            f"{ml_result['real_probability']:.2%}",
        )

    # --------------------------------------------------------
    # STEP 3: GOOGLE FACT CHECK
    # --------------------------------------------------------

    with st.spinner(
        "Searching Google Fact Check..."
    ):

        fact_checks, factcheck_status = (
            search_fact_checks(
                claim=claim,
                api_key=FACTCHECK_API_KEY,
                max_results=10,
            )
        )

    # --------------------------------------------------------
    # STEP 4: SEMANTIC MATCHING OF FACT CHECKS
    # --------------------------------------------------------

    with st.spinner(
        "Matching fact-check reviews to the claim..."
    ):

        matched_fact_checks = match_fact_checks(
            claim,
            fact_checks,
        )

    # --------------------------------------------------------
    # STEP 5: FINAL DECISION
    # --------------------------------------------------------

    final_result = combine_ml_and_factcheck(
        ml_result=ml_result,
        matched_fact_checks=matched_fact_checks,
        factcheck_status=factcheck_status,
    )

    # --------------------------------------------------------
    # FINAL VERDICT
    # --------------------------------------------------------

    st.subheader(
        "### Final Fact Check"
    )

    if final_result["verdict"] == "LIKELY REAL":
        st.success(
            "✅ LIKELY REAL"
        )

    elif final_result["verdict"] == "LIKELY FAKE":
        st.error(
            "❌ LIKELY FAKE"
        )

    elif final_result["verdict"] == "CONFLICTING":
        st.warning(
            "⚠️ CONFLICTING EVIDENCE"
        )

    else:
        st.info(
            "ℹ️ NOT ENOUGH EVIDENCE"
        )

    st.write(
        final_result["explanation"]
    )

    st.progress(
        final_result["decision_confidence"]
    )

    st.caption(
        "Decision confidence is a system confidence score, "
        "not a mathematical probability that the claim is true."
    )

    # --------------------------------------------------------
    # EVIDENCE SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "📊 Evidence Summary"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Fact Checks Found",
            len(matched_fact_checks),
        )

    with col2:
        st.metric(
            "Supporting Reviews",
            final_result["supporting_reviews"],
        )

    with col3:
        st.metric(
            "Refuting Reviews",
            final_result["refuting_reviews"],
        )

    # --------------------------------------------------------
    # FACT CHECK DETAILS
    # --------------------------------------------------------

    st.subheader(
        "🔍 Google Fact Check Evidence"
    )

    if factcheck_status["status"] == "missing_key":

        st.warning(
            "Google Fact Check API key is not configured."
        )

    elif factcheck_status["status"] == "api_error":

        st.error(
            factcheck_status["message"]
        )

    elif not matched_fact_checks:

        st.info(
            "No sufficiently matching professional "
            "fact-check review was found."
        )

    else:

        for index, review in enumerate(
            matched_fact_checks,
            start=1,
        ):

            title = (
                review.get("title")
                or f"Fact Check Review {index}"
            )

            with st.expander(
                f"{index}. {title}"
            ):

                st.write(
                    "**Publisher:** "
                    + (
                        review.get(
                            "publisher",
                            "Unknown",
                        )
                        or "Unknown"
                    )
                )

                st.write(
                    "**Rating:** "
                    + (
                        review.get(
                            "rating",
                            "Not available",
                        )
                        or "Not available"
                    )
                )

                st.write(
                    "**Semantic match:** "
                    f"{review.get('semantic_score', 0.0):.2%}"
                )

                st.write(
                    "**NLI match:** "
                    + review.get(
                        "nli_match",
                        "UNKNOWN",
                    )
                )

                st.write(
                    "**Fact-checked claim:**"
                )

                st.write(
                    review.get(
                        "claim",
                        "",
                    )
                )

                if review.get("review_date"):
                    st.write(
                        "**Review date:** "
                        + review["review_date"]
                    )

                if review.get("url"):
                    st.markdown(
                        "[Open professional fact-check]("
                        + review["url"]
                        + ")"
                    )

    # --------------------------------------------------------
    # DECISION BREAKDOWN
    # --------------------------------------------------------

    st.subheader(
        "🧩 Decision Breakdown"
    )

    st.write(
        f"**ML prediction:** "
        f"{ml_result['label']}"
    )

    st.write(
        f"**ML real probability:** "
        f"{ml_result['real_probability']:.2%}"
    )

    st.write(
        f"**ML fake probability:** "
        f"{ml_result['fake_probability']:.2%}"
    )

    st.write(
        f"**Matching professional fact checks:** "
        f"{len(matched_fact_checks)}"
    )

    st.write(
        f"**Fact-check support score:** "
        f"{final_result['factcheck_support_score']:.2f}"
    )

    st.write(
        f"**Fact-check refute score:** "
        f"{final_result['factcheck_refute_score']:.2f}"
    )

    # --------------------------------------------------------
    # LIMITATION
    # --------------------------------------------------------

    st.caption(
        "The ML classifier learns patterns from the training "
        "dataset. Google Fact Check contributes previously "
        "published human fact-check reviews. Neither source "
        "should be interpreted as a guarantee of truth."
    )
