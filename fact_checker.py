# ============================================================
# GOOGLE FACT CHECKING ENGINE
# ============================================================

import re
import requests
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ============================================================
# 1. LOAD SEMANTIC MODELS
# ============================================================

print("Loading Sentence-BERT model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading DeBERTa MNLI model...")

nli_model = pipeline(
    "text-classification",
    model="microsoft/deberta-base-mnli",
)

print("Fact-checking models loaded.")


# ============================================================
# 2. TEXT CLEANING
# ============================================================

def clean_text(text):
    text = str(text)

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
        r"[^a-zA-Z0-9\s]",
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
# 3. CLAIM EXTRACTION
# ============================================================

def extract_claim(text):
    """
    Create a compact factual claim from the user's input.

    For a headline, the complete headline is retained.
    For longer articles, the first two non-empty sentences
    are used so that the Google search query stays focused.
    """

    text = str(text).strip()

    if not text:
        return ""

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:
        return clean_text(text)[:800]

    if len(sentences) == 1:
        return clean_text(sentences[0])[:800]

    claim = " ".join(
        sentences[:2]
    )

    return clean_text(claim)[:1000]


# ============================================================
# 4. GOOGLE FACT CHECK API
# ============================================================

def search_fact_checks(
    claim,
    api_key,
    max_results=10,
):
    """
    Search Google's Fact Check Tools API.

    Returns:
        (results, status)

    The API searches claims that have already been
    fact-checked. It is not a general news-search API.
    """

    if not api_key:
        return [], {
            "status": "missing_key",
            "message": (
                "Google Fact Check API key is missing."
            ),
        }

    claim = str(claim).strip()

    if not claim:
        return [], {
            "status": "empty_query",
            "message": (
                "The claim is empty."
            ),
        }

    endpoint = (
        "https://factchecktools.googleapis.com/"
        "v1alpha1/claims:search"
    )

    params = {
        "query": claim,
        "languageCode": "en",
        "pageSize": int(max_results),
    }

    headers = {
        "x-goog-api-key": api_key,
    }

    try:
        response = requests.get(
            endpoint,
            params=params,
            headers=headers,
            timeout=20,
        )

        if response.status_code != 200:
            return [], {
                "status": "api_error",
                "message": (
                    "Google Fact Check API returned "
                    f"HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                ),
            }

        data = response.json()

        results = []

        for claim_data in data.get(
            "claims",
            [],
        ):

            claim_text = (
                claim_data.get(
                    "text",
                    "",
                )
                or ""
            )

            claimant = (
                claim_data.get(
                    "claimant",
                    "",
                )
                or ""
            )

            claim_date = (
                claim_data.get(
                    "claimDate",
                    "",
                )
                or ""
            )

            reviews = claim_data.get(
                "claimReview",
                [],
            )

            for review in reviews:

                publisher = review.get(
                    "publisher",
                    {},
                ) or {}

                results.append(
                    {
                        "type": "fact_check",
                        "claim": claim_text,
                        "claimant": claimant,
                        "claim_date": claim_date,
                        "publisher": (
                            publisher.get(
                                "name",
                                "",
                            )
                            or ""
                        ),
                        "publisher_site": (
                            publisher.get(
                                "site",
                                "",
                            )
                            or ""
                        ),
                        "title": (
                            review.get(
                                "title",
                                "",
                            )
                            or ""
                        ),
                        "rating": (
                            review.get(
                                "textualRating",
                                "",
                            )
                            or ""
                        ),
                        "url": (
                            review.get(
                                "url",
                                "",
                            )
                            or ""
                        ),
                        "review_date": (
                            review.get(
                                "reviewDate",
                                "",
                            )
                            or ""
                        ),
                        "language": (
                            review.get(
                                "languageCode",
                                "en",
                            )
                            or "en"
                        ),
                    }
                )

        return results, {
            "status": "success",
            "message": (
                f"Google returned {len(results)} "
                "fact-check review(s)."
            ),
        }

    except requests.exceptions.Timeout:
        return [], {
            "status": "api_error",
            "message": (
                "Google Fact Check API request timed out."
            ),
        }

    except requests.exceptions.RequestException as exc:
        return [], {
            "status": "api_error",
            "message": (
                "Google Fact Check API request failed: "
                + str(exc)
            ),
        }

    except ValueError:
        return [], {
            "status": "api_error",
            "message": (
                "Google Fact Check API returned invalid JSON."
            ),
        }

    except Exception as exc:
        return [], {
            "status": "api_error",
            "message": (
                "Unexpected Fact Check API error: "
                + str(exc)
            ),
        }


# ============================================================
# 5. RATING NORMALIZATION
# ============================================================

def normalize_rating(rating):
    """
    Convert different publishers' rating language into
    a conservative category.

    Because fact-checking organizations use different
    rating systems, this is intentionally conservative.
    """

    value = (
        str(rating)
        .strip()
        .lower()
    )

    if not value:
        return "UNKNOWN"

    # Strong false / refuting language.
    if any(
        phrase in value
        for phrase in [
            "pants on fire",
            "false",
            "fake",
            "incorrect",
            "not true",
            "wrong",
            "fabricated",
            "hoax",
        ]
    ):
        return "REFUTES"

    # Strong true / supporting language.
    if any(
        phrase in value
        for phrase in [
            "true",
            "correct",
            "accurate",
            "verified",
        ]
    ):
        return "SUPPORTS"

    # Intermediate / mixed ratings.
    if any(
        phrase in value
        for phrase in [
            "mostly false",
            "mostly true",
            "half true",
            "partly true",
            "partially true",
            "misleading",
            "mixed",
            "missing context",
            "out of context",
        ]
    ):
        return "MIXED"

    return "UNKNOWN"


# ============================================================
# 6. SEMANTIC MATCHING
# ============================================================

def _nli_match(
    fact_checked_claim,
    user_claim,
):
    """
    Verify that the Google fact-checked claim and the
    user's claim are semantically related.

    Premise    = Google fact-checked claim
    Hypothesis = user's claim

    NLI here is used for claim matching, not as the
    truth source. The Google review rating remains the
    truth assessment.
    """

    try:
        result = nli_model(
            {
                "text": fact_checked_claim,
                "text_pair": user_claim,
            },
            truncation=True,
        )[0]

        raw_label = str(
            result.get(
                "label",
                "",
            )
        ).upper()

        label_map = {
            "LABEL_0": "CONTRADICTION",
            "LABEL_1": "NEUTRAL",
            "LABEL_2": "ENTAILMENT",
        }

        label = label_map.get(
            raw_label,
            raw_label,
        )

        score = float(
            result.get(
                "score",
                0.0,
            )
        )

        return label, score

    except Exception as exc:
        print(
            "NLI claim-match error:",
            exc,
        )
        return "UNKNOWN", 0.0


def match_fact_checks(
    user_claim,
    fact_checks,
    semantic_threshold=0.65,
):
    """
    Match Google's returned fact-check claims to the
    user's claim.

    SBERT provides semantic similarity.
    DeBERTa checks the relationship between the two claims.

    Only sufficiently relevant reviews are retained.
    """

    if not fact_checks:
        return []

    user_claim_clean = clean_text(
        user_claim
    )

    if not user_claim_clean:
        return []

    fact_claims = [
        clean_text(
            item.get(
                "claim",
                "",
            )
        )
        for item in fact_checks
    ]

    valid_pairs = [
        (
            index,
            claim_text,
        )
        for index, claim_text in enumerate(
            fact_claims
        )
        if claim_text
    ]

    if not valid_pairs:
        return []

    texts = [
        user_claim_clean
    ] + [
        claim_text
        for _, claim_text in valid_pairs
    ]

    try:
        embeddings = embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        user_embedding = embeddings[0]

        fact_embeddings = embeddings[1:]

        similarity_scores = np.dot(
            fact_embeddings,
            user_embedding,
        )

    except Exception as exc:
        print(
            "SBERT matching error:",
            exc,
        )
        return []

    matched = []

    for position, (
        original_index,
        fact_claim,
    ) in enumerate(valid_pairs):

        semantic_score = float(
            similarity_scores[position]
        )

        if semantic_score < semantic_threshold:
            continue

        item = fact_checks[
            original_index
        ].copy()

        nli_label, nli_score = _nli_match(
            fact_claim,
            user_claim_clean,
        )

        # The claims can be phrased differently.
        # Entailment is strongest for a direct match.
        # Neutral can still be a valid match when SBERT
        # similarity is high.
        if (
            nli_label == "CONTRADICTION"
            and nli_score >= 0.80
        ):
            continue

        item["semantic_score"] = (
            semantic_score
        )

        item["nli_match"] = nli_label

        item["nli_match_score"] = (
            nli_score
        )

        item["normalized_rating"] = (
            normalize_rating(
                item.get(
                    "rating",
                    "",
                )
            )
        )

        # Relevance score is used only for ranking.
        item["match_score"] = (
            0.70 * semantic_score
            + 0.30 * nli_score
        )

        matched.append(item)

    matched.sort(
        key=lambda item: item.get(
            "match_score",
            0.0,
        ),
        reverse=True,
    )

    return matched


# ============================================================
# 7. FINAL DECISION ENGINE
# ============================================================

def _factcheck_strength(review):
    """
    Give a conservative strength to a fact-check review.

    This is not a probability. It is only a weight used
    to combine multiple reviews.
    """

    rating_type = review.get(
        "normalized_rating",
        "UNKNOWN",
    )

    match_score = float(
        review.get(
            "match_score",
            0.0,
        )
    )

    if rating_type == "SUPPORTS":
        return match_score

    if rating_type == "REFUTES":
        return match_score

    if rating_type == "MIXED":
        return 0.5 * match_score

    return 0.0


def combine_ml_and_factcheck(
    ml_result,
    matched_fact_checks,
    factcheck_status,
):
    """
    Combine:

        1. Automatically selected ML model
        2. Google professional fact-check reviews

    Decision priority:

        - Strong, relevant professional fact-checks
        - Agreement between ML and fact-checks
        - ML-only result when no professional review exists
        - Uncertain when evidence conflicts or is weak

    This function deliberately does NOT treat a missing
    Google fact-check as evidence that the claim is false.
    """

    real_prob = float(
        ml_result["real_probability"]
    )

    fake_prob = float(
        ml_result["fake_probability"]
    )

    supports = [
        item
        for item in matched_fact_checks
        if item.get(
            "normalized_rating"
        ) == "SUPPORTS"
    ]

    refutes = [
        item
        for item in matched_fact_checks
        if item.get(
            "normalized_rating"
        ) == "REFUTES"
    ]

    mixed = [
        item
        for item in matched_fact_checks
        if item.get(
            "normalized_rating"
        ) == "MIXED"
    ]

    support_score = 0.0

    if supports:
        support_score = max(
            _factcheck_strength(item)
            for item in supports
        )

    refute_score = 0.0

    if refutes:
        refute_score = max(
            _factcheck_strength(item)
            for item in refutes
        )

    ml_label = ml_result["label"]

    # --------------------------------------------------------
    # API failure
    # --------------------------------------------------------

    if factcheck_status.get(
        "status"
    ) in {
        "missing_key",
        "api_error",
        "empty_query",
    }:

        if (
            real_prob >= 0.90
            or fake_prob >= 0.90
        ):

            confidence = max(
                real_prob,
                fake_prob,
            )

            verdict = (
                "LIKELY REAL"
                if real_prob >= fake_prob
                else "LIKELY FAKE"
            )

            return {
                "verdict": verdict,
                "explanation": (
                    "The ML model is highly confident, "
                    "but Google professional fact-check "
                    "verification was unavailable."
                ),
                "decision_confidence": float(
                    confidence
                ),
                "supporting_reviews": len(
                    supports
                ),
                "refuting_reviews": len(
                    refutes
                ),
                "factcheck_support_score": (
                    support_score
                ),
                "factcheck_refute_score": (
                    refute_score
                ),
            }

        return {
            "verdict": "NOT ENOUGH EVIDENCE",
            "explanation": (
                "The ML prediction is not sufficiently "
                "confident and professional fact-check "
                "verification was unavailable."
            ),
            "decision_confidence": 0.50,
            "supporting_reviews": len(
                supports
            ),
            "refuting_reviews": len(
                refutes
            ),
            "factcheck_support_score": (
                support_score
            ),
            "factcheck_refute_score": (
                refute_score
            ),
        }

    # --------------------------------------------------------
    # Conflicting professional fact checks
    # --------------------------------------------------------

    if (
        support_score >= 0.70
        and refute_score >= 0.70
    ):

        return {
            "verdict": "CONFLICTING",
            "explanation": (
                "Relevant professional fact-check "
                "reviews disagree. The system will not "
                "force a Real/Fake decision."
            ),
            "decision_confidence": 0.50,
            "supporting_reviews": len(
                supports
            ),
            "refuting_reviews": len(
                refutes
            ),
            "factcheck_support_score": (
                support_score
            ),
            "factcheck_refute_score": (
                refute_score
            ),
        }

    # --------------------------------------------------------
    # Strong professional support
    # --------------------------------------------------------

    if support_score >= 0.75:

        if real_prob >= 0.50:

            confidence = min(
                0.98,
                0.55 * support_score
                + 0.45 * real_prob,
            )

            return {
                "verdict": "LIKELY REAL",
                "explanation": (
                    "A relevant professional "
                    "fact-check supports the claim, "
                    "and the ML model also leans Real."
                ),
                "decision_confidence": float(
                    confidence
                ),
                "supporting_reviews": len(
                    supports
                ),
                "refuting_reviews": len(
                    refutes
                ),
                "factcheck_support_score": (
                    support_score
                ),
                "factcheck_refute_score": (
                    refute_score
                ),
            }

        return {
            "verdict": "NOT ENOUGH EVIDENCE",
            "explanation": (
                "A professional fact-check supports "
                "the claim, but the ML model strongly "
                "disagrees. The system will not blindly "
                "override the disagreement."
            ),
            "decision_confidence": 0.50,
            "supporting_reviews": len(
                supports
            ),
            "refuting_reviews": len(
                refutes
            ),
            "factcheck_support_score": (
                support_score
            ),
            "factcheck_refute_score": (
                refute_score
            ),
        }

    # --------------------------------------------------------
    # Strong professional refutation
    # --------------------------------------------------------

    if refute_score >= 0.75:

        if fake_prob >= 0.50:

            confidence = min(
                0.98,
                0.55 * refute_score
                + 0.45 * fake_prob,
            )

            return {
                "verdict": "LIKELY FAKE",
                "explanation": (
                    "A relevant professional "
                    "fact-check refutes the claim, "
                    "and the ML model also leans Fake."
                ),
                "decision_confidence": float(
                    confidence
                ),
                "supporting_reviews": len(
                    supports
                ),
                "refuting_reviews": len(
                    refutes
                ),
                "factcheck_support_score": (
                    support_score
                ),
                "factcheck_refute_score": (
                    refute_score
                ),
            }

        return {
            "verdict": "NOT ENOUGH EVIDENCE",
            "explanation": (
                "A professional fact-check refutes "
                "the claim, but the ML model strongly "
                "disagrees. The system will not blindly "
                "override the disagreement."
            ),
            "decision_confidence": 0.50,
            "supporting_reviews": len(
                supports
            ),
            "refuting_reviews": len(
                refutes
            ),
            "factcheck_support_score": (
                support_score
            ),
            "factcheck_refute_score": (
                refute_score
            ),
        }

    # --------------------------------------------------------
    # Mixed / intermediate ratings
    # --------------------------------------------------------

    if mixed and not supports and not refutes:

        return {
            "verdict": "NOT ENOUGH EVIDENCE",
            "explanation": (
                "A relevant professional fact-check "
                "contains an intermediate or mixed rating. "
                "The system will not convert it into a "
                "binary Real/Fake verdict."
            ),
            "decision_confidence": 0.50,
            "supporting_reviews": len(
                supports
            ),
            "refuting_reviews": len(
                refutes
            ),
            "factcheck_support_score": (
                support_score
            ),
            "factcheck_refute_score": (
                refute_score
            ),
        }

    # --------------------------------------------------------
    # No useful professional fact check:
    # use ML conservatively.
    # --------------------------------------------------------

    if not matched_fact_checks:

        if real_prob >= 0.90:

            return {
                "verdict": "LIKELY REAL",
                "explanation": (
                    "No matching professional "
                    "fact-check was found, but the "
                    "automatically selected ML model "
                    "is highly confident that the text "
                    "resembles Real news."
                ),
                "decision_confidence": real_prob,
                "supporting_reviews": 0,
                "refuting_reviews": 0,
                "factcheck_support_score": 0.0,
                "factcheck_refute_score": 0.0,
            }

        if fake_prob >= 0.90:

            return {
                "verdict": "LIKELY FAKE",
                "explanation": (
                    "No matching professional "
                    "fact-check was found, but the "
                    "automatically selected ML model "
                    "is highly confident that the text "
                    "resembles Fake news."
                ),
                "decision_confidence": fake_prob,
                "supporting_reviews": 0,
                "refuting_reviews": 0,
                "factcheck_support_score": 0.0,
                "factcheck_refute_score": 0.0,
            }

        return {
            "verdict": "NOT ENOUGH EVIDENCE",
            "explanation": (
                "No matching professional fact-check "
                "was found and the ML model is not "
                "confident enough to make a reliable "
                "binary decision."
            ),
            "decision_confidence": max(
                real_prob,
                fake_prob,
            ),
            "supporting_reviews": 0,
            "refuting_reviews": 0,
            "factcheck_support_score": 0.0,
            "factcheck_refute_score": 0.0,
        }

    # --------------------------------------------------------
    # Relevant review exists but rating is not decisive.
    # --------------------------------------------------------

    return {
        "verdict": "NOT ENOUGH EVIDENCE",
        "explanation": (
            "Relevant fact-check material was found, "
            "but its rating is not decisive enough to "
            "produce a safe binary verdict."
        ),
        "decision_confidence": max(
            0.50,
            0.50 * max(
                real_prob,
                fake_prob,
            ),
        ),
        "supporting_reviews": len(
            supports
        ),
        "refuting_reviews": len(
            refutes
        ),
        "factcheck_support_score": (
            support_score
        ),
        "factcheck_refute_score": (
            refute_score
        ),
    }
