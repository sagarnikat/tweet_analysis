"""
Configuration file for the Tweet Sentiment Analysis and Information Retrieval System.
Uses pathlib for robust cross-platform path resolution relative to repository root.
"""

from pathlib import Path

# Repository Root and Core Directories
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
SRC_DIR = REPO_ROOT / "src"
APP_DIR = REPO_ROOT / "app"

# Data File Paths
TWEETS_DATA_PATH = DATA_DIR / "tweets.csv"
FEEDBACK_DATA_PATH = DATA_DIR / "user_feedback.csv"
IR_QUERIES_PATH = DATA_DIR / "ir_queries.csv"

# Model Artifact Paths
LOGISTIC_REGRESSION_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
NAIVE_BAYES_MODEL_PATH = MODELS_DIR / "naive_bayes.joblib"
ENSEMBLE_MODEL_PATH = MODELS_DIR / "ensemble.joblib"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
TEST_DATA_SPLIT_PATH = MODELS_DIR / "test_data_split.joblib"
IR_INDEX_PATH = MODELS_DIR / "ir_index.joblib"

# Evaluation Results Paths
IR_METRICS_PATH = RESULTS_DIR / "ir_metrics.csv"
IR_PR_CURVE_PATH = RESULTS_DIR / "ir_pr_curve.png"
IR_METRICS_PLOT_PATH = RESULTS_DIR / "ir_metrics_comparison.png"
MODEL_COMPARISON_PLOT_PATH = RESULTS_DIR / "model_comparison.png"
SENTIMENT_METRICS_PATH = RESULTS_DIR / "sentiment_metrics.csv"

# Sentiment Classifier Parameters
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 3)
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

# Information Retrieval Parameters
BM25_K1 = 1.5
BM25_B = 0.75
ROCCHIO_ALPHA = 1.0
ROCCHIO_BETA = 0.75
ROCCHIO_TOP_K = 5
DEFAULT_TOP_K = 10
