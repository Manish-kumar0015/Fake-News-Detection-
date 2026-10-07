Fake News Detection & Claim Verification System

A multi-stage Fake News Detection and Claim Verification system built with NLP, classical machine learning, semantic embeddings, Natural Language Inference (NLI), and the Google Fact Check Tools API.

The system does not use NewsAPI or GDELT. Instead, it combines:

TF-IDF + automatically tuned ML + Google Fact Check + SBERT + DeBERTa MNLI + a conservative decision engine

to produce one of four outcomes:

LIKELY REAL

LIKELY FAKEs

NOT ENOUGH EVIDENCE

CONFLICTING

Important: This project is a verification assistant, not an absolute truth oracle. The ML model learns patterns from labeled training data, while Google Fact Check provides previously published fact-check evidence when a relevant fact-check exists. A missing Google Fact Check result does not mean a claim is false.

1. Project Overview

A simple fake-news classifier often looks like:

News Article
    -> TF-IDF
    -> Classifier
    -> Fake / Real

This project extends that into an evidence-fusion pipeline:

                     USER NEWS / CLAIM
                            |
                            v
                     Claim Extraction
                            |
                            v
                       Text Cleaning
                            |
                            v
                          TF-IDF
                            |
                            v
              Automatically Tuned ML Model
                            |
                       ML Probability
                            |
             +--------------+--------------+
             |                             |
             v                             v
       ML Evidence                Google Fact Check API
                                           |
                                           v
                                Retrieved Fact-Check Claims
                                           |
                                           v
                                         SBERT
                                           |
                                           v
                                Semantic Similarity
                                           |
                                           v
                                      DeBERTa MNLI
                                           |
                                           v
                                Entailment / Neutral /
                                Contradiction Relationship
             |                             |
             +--------------+--------------+
                            v
                     Decision Engine
                            |
                            v
          +------------------------------------------+
          | LIKELY REAL                             |
          | LIKELY FAKE                             |
          | NOT ENOUGH EVIDENCE                     |
          | CONFLICTING                             |
          +------------------------------------------+

2. Key Features

Supervised fake-news classification

Uses a fitted TF-IDF vectorizer and an automatically selected classical ML classifier.

Candidate models:

Logistic Regression

Linear SVM (LinearSVC)

Random Forest

Multinomial Naive Bayes

The final classifier is not manually hard-coded. Hyperparameters are searched automatically using RandomizedSearchCV, and the best configuration is selected using cross-validated Macro-F1.

Probability calibration

The selected model is wrapped with CalibratedClassifierCV so the application can use probability estimates even when a winning classifier such as LinearSVC does not natively expose predict_proba().

Google Fact Check integration

The application queries the Google Fact Check Tools API for previously published fact-checks related to the user's claim.

Semantic claim matching

Retrieved fact-check claims are compared with the user's claim using Sentence-BERT (SBERT) to handle different wording with similar meaning.

NLI verification

DeBERTa MNLI analyzes the relationship between the user's claim and a retrieved fact-checked claim:

Entailment

Neutral

Contradiction

SBERT answers approximately:

Are these claims semantically related?

DeBERTa answers approximately:

What is the logical relationship between them?

Conservative decision engine

The final decision combines:

ML class probabilities

Fact-check rating

Semantic similarity

NLI relationship/strength

Google API availability

When evidence is weak or conflicting, the system can return NOT ENOUGH EVIDENCE instead of forcing a binary answer.

3. What Kind of News Can It Detect?

The current implementation is for text-based news articles and factual claims.

Example categories include:

Politics: The government announced a new policy today.

Health misinformation: COVID-19 vaccines contain microchips used to track people.

Science: A certain drink completely cures cancer.

Technology: Company X announced a new technology yesterday.

Economics: The central bank announced a major policy change.

The system works best when the input is clear, similar kinds of examples are represented in training data, and a relevant professional fact-check exists when external evidence is needed.

4. What the System Does NOT Guarantee

Google Fact Check is not Google News

The Google Fact Check Tools API searches previously fact-checked claims and reviews. It is not a general search engine for every current news article.

Therefore:

No Google Fact Check result
        !=
False claim

A breaking or newly emerging story may have no existing fact-check.

ML probability is not factual truth probability

For example:

Fake = 0.92
Real = 0.08

does not literally mean that the claim has a 92% probability of being factually false. It means the calibrated classifier strongly favors the Fake class based on patterns learned from the training data.

The system is intentionally conservative

When evidence is missing or strongly conflicting, the system can return:

NOT ENOUGH EVIDENCE

or

CONFLICTING

rather than pretending to know the answer.

5. Technology Stack

Component

Technology

Purpose

UI

Streamlit

Interactive web application

Language

Python

End-to-end implementation

Text representation

TF-IDF

Sparse numerical text features

Candidate classifier

Logistic Regression

Linear text classification

Candidate classifier

Linear SVM

Maximum-margin classification for sparse text

Candidate classifier

Random Forest

Nonlinear ensemble baseline

Candidate classifier

Multinomial Naive Bayes

Fast probabilistic text baseline

Hyperparameter tuning

RandomizedSearchCV

Automatic model/hyperparameter search

Evaluation metric

Macro-F1

Equal importance to Fake and Real

Probability calibration

CalibratedClassifierCV

Consistent probability interface

Semantic matching

Sentence-BERT

Semantic claim similarity

NLI

DeBERTa MNLI

Entailment / neutral / contradiction

External evidence

Google Fact Check Tools API

Existing professional fact-check reviews

Serialization

Joblib

Save/load trained scikit-learn artifacts

6. Dataset Preparation

The training pipeline supports datasets such as:

data/
├── Fake.csv
├── True.csv
├── news.csv
└── news2.csv

The standardized label mapping is:

0 = Fake
1 = Real

When title and body fields are available, they can be combined so the classifier sees headline and article information.

The preprocessing pipeline also:

normalizes text

removes URLs/HTML and unnecessary noise

removes empty examples

removes exact duplicates

standardizes labels

performs a stratified train/test split

Why remove exact duplicates?

If the same article appears in both training and test data, evaluation can become artificially high because the model has effectively seen the same content before.

7. TF-IDF Feature Engineering

The current vectorizer is:

TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=80000,
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
)

Why TF-IDF?

TF-IDF is an efficient representation for traditional text classification because it converts text into high-dimensional sparse numerical features.

Why unigrams and bigrams?

ngram_range=(1, 2) captures both individual words and short phrases such as:

miracle cure
official announcement
secret conspiracy

Why fit only on training data?

The vectorizer is fitted on training data:

X_train_tfidf = vectorizer.fit_transform(X_train)

and only transforms test data:

X_test_tfidf = vectorizer.transform(X_test)

This prevents test-set information from leaking into the learned TF-IDF vocabulary and IDF statistics.

8. Automatic Model and Hyperparameter Selection

The training notebook evaluates multiple candidate algorithms automatically instead of manually selecting one classifier.

The search uses:

RandomizedSearchCV

randomized hyperparameter combinations

3-fold cross-validation

f1_macro scoring

a fixed random seed for reproducibility

Flow:

Candidate Models
      |
      v
Randomized Hyperparameter Search
      |
      v
3-Fold Cross-Validation
      |
      v
Macro-F1 Comparison
      |
      v
Best Model + Best Hyperparameters

Why RandomizedSearchCV?

An exhaustive grid can become expensive as the number of parameters and models increases. Randomized search allows the project to control the number of configurations evaluated.

Why use a tuning subset?

The notebook can use a stratified tuning subset of up to:

TUNE_MAX_SAMPLES = 40000

This reduces search cost. After the best configuration is identified, the winning model is refit on the full training set.

9. Why Macro-F1?

For binary classification:

Macro-F1 = (F1_fake + F1_real) / 2

Macro-F1 gives both Fake and Real equal importance, so the search is not driven only by overall accuracy.

10. Candidate Models

Logistic Regression

A linear classifier that is efficient for high-dimensional sparse TF-IDF features.

Linear SVM

A maximum-margin linear classifier. It is a strong traditional approach for high-dimensional sparse text.

Random Forest

An ensemble of decision trees. It is included as a nonlinear baseline so the search does not assume linear models must win.

Multinomial Naive Bayes

A fast probabilistic text-classification baseline.

The project does not claim that one of these models is universally best. The actual winner is selected automatically from the search results.

11. Probability Calibration

Some classifiers, especially LinearSVC, do not provide predict_proba() directly.

The application therefore uses:

CalibratedClassifierCV

around the selected classifier.

This creates a consistent probability interface for the deployed application and decision layer. These estimates are still model probabilities, not guarantees about real-world truth.

12. Google Fact Check Tools API

The external evidence layer searches the Google Fact Check Tools API for previously fact-checked claims.

The returned data can contain:

claim text

claimant

claim date

fact-check publisher

review title

review URL

review date

textual rating

The application uses this information as evidence after relevance checking rather than blindly trusting the first API response.

13. Fact-Check Rating Normalization

Different fact-check publishers can use different labels. The project maps them into common categories:

SUPPORTS
REFUTES
MIXED
UNKNOWN

Typical examples:

True / Correct / Accurate / Verified
             -> SUPPORTS

False / Fake / Incorrect / Wrong / Hoax
             -> REFUTES

Mostly False / Mostly True / Misleading /
Missing Context / Out of Context / Mixed
             -> MIXED

This makes the decision engine easier to implement consistently.

14. SBERT Semantic Matching

A user claim and a fact-check can use very different wording.

Example:

User:
Vaccines contain tracking chips.

Fact-check:
COVID-19 vaccines do not contain microchips for tracking individuals.

SBERT converts the sentences into embeddings and compares them using semantic similarity.

The system applies a minimum similarity threshold so unrelated fact-checks are not allowed to influence the final decision.

15. DeBERTa MNLI

SBERT measures semantic relevance, but similarity alone does not tell us whether two claims agree or contradict each other.

For example:

Vaccines contain microchips.
Vaccines do not contain microchips.

These statements are semantically related but logically contradictory.

DeBERTa MNLI therefore evaluates:

Entailment

Neutral

Contradiction

The application uses:

Premise    = Google fact-checked claim
Hypothesis = user's claim

16. Why Use SBERT and DeBERTa Together?

They solve different problems:

SBERT
  -> semantic relevance / retrieval

DeBERTa MNLI
  -> logical relationship / inference

This is stronger than simple keyword overlap because it handles paraphrasing and also checks for contradiction.

17. Decision Engine

The final verdict combines:

ML probability
      +
Fact-check rating
      +
SBERT match strength
      +
DeBERTa NLI
      +
API availability
      |
      v
Decision Engine

LIKELY REAL

Strong supporting evidence and sufficient ML agreement.

LIKELY FAKE

Strong refuting evidence and/or a strong ML fake signal under the decision rules.

NOT ENOUGH EVIDENCE

Used when fact-check evidence is unavailable or the combined evidence is not strong enough.

CONFLICTING

Used when strong evidence sources disagree.

The goal is to avoid forced binary decisions when the available evidence does not justify them.

18. Project Structure

Fake-News-Detection/
|
├── app.py
├── fact_checker.py
├── requirements.txt
├── README.md
|
├── Fake_News_Auto_Hyperparameter_Tuning.ipynb
|
├── data/
|   ├── Fake.csv
|   ├── True.csv
|   ├── news.csv
|   └── news2.csv
|
├── models/
|   ├── tfidf_vectorizer.pkl
|   ├── best_model.pkl
|   ├── ml_metadata.json
|   └── model_search_results.csv
|
└── .streamlit/
    └── secrets.toml          # local only; do NOT commit

19. Installation and Training

Create a virtual environment

Windows PowerShell:

python -m venv .venv
.venv\Scripts\activate

Install dependencies

pip install -r requirements.txt

Place datasets

Put the CSV files under:

data/

Run the training notebook

Open:

Fake_News_Auto_Hyperparameter_Tuning.ipynb

Run cells from the beginning. The notebook produces/updates:

models/tfidf_vectorizer.pkl
models/best_model.pkl
models/ml_metadata.json
models/model_search_results.csv

Run the notebook from the project root so relative paths such as data/ and models/ resolve correctly.

20. Run the Streamlit App Locally

.venv\Scripts\activate
streamlit run app.py

21. Google API Key Setup

Create the local file:

.streamlit/secrets.toml

and add:

FACTCHECK_API_KEY = "YOUR_GOOGLE_API_KEY"

Never hard-code the real key in app.py and never commit secrets.toml to GitHub.

22. GitHub + Streamlit Community Cloud Deployment

If the app is already deployed from your GitHub repository, changing the code normally does not require creating a new Streamlit app.

Local update workflow

git status
git add .
git commit -m "Update fake news detection and fact-check pipeline"
git push origin main

Use your actual branch if it is not main.

After the push, Streamlit Community Cloud detects repository changes and updates the existing deployment. If requirements.txt changed, dependency installation/rebuild can take longer.

Keep secrets out of GitHub

Your local file:

.streamlit/secrets.toml

should remain untracked. Configure the same secret in the deployed Streamlit application's secret settings:

FACTCHECK_API_KEY = "YOUR_GOOGLE_API_KEY"

Before pushing, verify

git status

Make sure you do not see:

.streamlit/secrets.toml

23. Recommended .gitignore

.venv/
venv/
env/

__pycache__/
*.py[cod]

.ipynb_checkpoints/

.streamlit/secrets.toml

.vscode/
.idea/

.DS_Store
Thumbs.db

Do not ignore the trained model artifacts if the deployed application loads them directly from the repository.

24. End-to-End Example

Input:

COVID-19 vaccines contain microchips that track people.

The application performs:

1. Claim extraction
2. Text cleaning
3. TF-IDF transformation
4. Best trained classifier
5. ML prediction
6. Google Fact Check search
7. SBERT semantic matching
8. DeBERTa MNLI relationship analysis
9. Fact-check rating normalization
10. Decision engine
11. Final verdict

Depending on the returned evidence and model output, a known fact-checked false claim can produce:

LIKELY FAKE

25. Example of Insufficient Evidence

Input:

A company announced a completely new technology yesterday.

Suppose the ML confidence is moderate and no relevant Google Fact Check exists.

The system can return:

NOT ENOUGH EVIDENCE

This is intentional because lack of a fact-check is not proof of falsity.

26. Limitations

Dataset bias

The classifier may learn source-specific or dataset-specific writing patterns instead of factuality itself.

Distribution shift

Historical training data can differ from future news and emerging topics.

Limited fact-check coverage

A new claim may not yet have a corresponding professional fact-check.

Text-only scope

The current system does not directly analyze images, videos, or audio.

Claim extraction limitations

Current claim extraction is lightweight and is not a full claim-detection or information-extraction model.

Adversarial wording

Deliberately rewritten misinformation can still challenge a traditional TF-IDF classifier.

27. Future Improvements

Potential improvements include:

Fine-tuning a transformer specifically for fake-news classification.

Training a dedicated claim-extraction model.

Using temporal train/test splits to measure future-news generalization.

Using source-based evaluation to reduce publisher-artifact bias.

Learning decision thresholds and fusion weights on a validation dataset rather than fixing them heuristically.

Improving probability calibration and uncertainty estimation.

Adding multimodal image/video analysis.

Improving fact-check retrieval and reranking.

Monitoring model drift after deployment.

28. Interview Explanation

A concise interview description:

I built a text-based Fake News Detection and claim verification system. I combine multiple labeled datasets, clean and deduplicate the text, and represent it using TF-IDF unigram and bigram features. Instead of manually selecting one classifier, I use RandomizedSearchCV with cross-validation across Logistic Regression, Linear SVM, Random Forest, and Multinomial Naive Bayes, selecting the best configuration using Macro-F1. The selected classifier is calibrated for probability estimates and saved along with the fitted TF-IDF vectorizer. During inference, the user's claim is also checked against the Google Fact Check Tools API. Retrieved fact-check claims are semantically matched with SBERT, then DeBERTa MNLI is used to determine entailment, neutrality, or contradiction. A decision engine combines ML evidence, fact-check rating, semantic similarity, NLI, and API availability to produce LIKELY REAL, LIKELY FAKE, NOT ENOUGH EVIDENCE, or CONFLICTING.

29. Common Interview Questions

Why TF-IDF?

It is efficient for sparse text features and provides a strong traditional baseline for text classification.

Why Logistic Regression?

It is a fast linear classifier that works well with high-dimensional sparse TF-IDF features and supports probability estimation.

Why SVM?

Linear SVM is well suited to high-dimensional sparse text and learns a maximum-margin decision boundary.

Why Random Forest?

It provides a nonlinear ensemble baseline so the system can compare it against linear text models instead of assuming a winner beforehand.

Why Naive Bayes?

It is very fast and is a standard baseline for text classification.

Why RandomizedSearchCV?

It automates hyperparameter selection while controlling the number of configurations evaluated.

Why Macro-F1?

It gives equal importance to Fake and Real classes instead of relying only on overall accuracy.

Why use a tuning subset?

It makes hyperparameter search computationally practical. The selected configuration is later refit on the full training set.

Why SBERT?

To identify semantically related claims even when the wording is different.

Why DeBERTa MNLI?

Because semantic similarity alone cannot distinguish agreement from contradiction. NLI provides an entailment/neutral/contradiction signal.

Why not just use Google Fact Check?

Because the API depends on existing fact-check coverage. New claims may have no matching review.

Why not NewsAPI?

The current architecture focuses on verification evidence rather than general news aggregation.

Why not GDELT?

GDELT is useful for news discovery and event monitoring, but the current system intentionally focuses on existing fact-check evidence.

Why not use an LLM for everything?

A fully LLM-based truth judge introduces additional issues around hallucination, cost, latency, reproducibility, and evaluation. This project uses specialized models for specialized tasks.

30. Security Notes

Never commit:

.streamlit/secrets.toml

or any file containing a real API key.

Use Streamlit's secret-management mechanism for deployment. If a credential is accidentally pushed to GitHub, revoke/rotate it immediately.

31. Project Status

TF-IDF
   +
Automatic Hyperparameter Tuning
   +
Classical ML
   +
Probability Calibration
   +
Google Fact Check Tools API
   +
SBERT
   +
DeBERTa MNLI
   +
Decision Engine
   =
Fake News Detection & Claim Verification System

No NewsAPI is required.

No GDELT is required.

32. Official Documentation

Streamlit Community Cloud: https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started

Streamlit app dependencies: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

Streamlit secrets management: https://docs.streamlit.io/deploy/concepts/secrets

Google Fact Check Tools API: https://developers.google.com/fact-check/tools/api

Google Claims Search API reference: https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims/search

33. License

Add the license you intend to use for the project. Before choosing a license, verify that all included datasets and model assets permit redistribution under that license.