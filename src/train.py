"""
Training script for sentiment analysis models.
Trains Logistic Regression, Naive Bayes, and a Soft-Voting Ensemble on combined TF-IDF + VADER features.
Includes deduplication on cleaned text and stratified train-test splitting.
"""

import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Path configuration
sys.path.append(str(Path(__file__).resolve().parent))
try:
    from config import (
        TWEETS_DATA_PATH,
        FEEDBACK_DATA_PATH,
        MODELS_DIR,
        LOGISTIC_REGRESSION_MODEL_PATH,
        NAIVE_BAYES_MODEL_PATH,
        ENSEMBLE_MODEL_PATH,
        TFIDF_VECTORIZER_PATH,
        TEST_DATA_SPLIT_PATH,
        TFIDF_MAX_FEATURES,
        TFIDF_NGRAM_RANGE,
        TEST_SPLIT_SIZE,
        RANDOM_STATE
    )
except ImportError:
    from src.config import (
        TWEETS_DATA_PATH,
        FEEDBACK_DATA_PATH,
        MODELS_DIR,
        LOGISTIC_REGRESSION_MODEL_PATH,
        NAIVE_BAYES_MODEL_PATH,
        ENSEMBLE_MODEL_PATH,
        TFIDF_VECTORIZER_PATH,
        TEST_DATA_SPLIT_PATH,
        TFIDF_MAX_FEATURES,
        TFIDF_NGRAM_RANGE,
        TEST_SPLIT_SIZE,
        RANDOM_STATE
    )

from preprocess import load_and_preprocess, clamp_negatives

def train_models():
    print("🚀 Self-Learning Model: Training process started...")
    
    if not os.path.exists(TWEETS_DATA_PATH):
        print(f"❌ ERROR: Dataset not found at: {TWEETS_DATA_PATH}")
        return
        
    print(f"📂 Loading and Preprocessing dataset from {TWEETS_DATA_PATH}...")
    df = load_and_preprocess(TWEETS_DATA_PATH)
    
    # Check for user feedback and merge for self-learning
    if os.path.exists(FEEDBACK_DATA_PATH):
        print(f"🔄 Self-Learning: Found user feedback at {FEEDBACK_DATA_PATH}. Incorporating into training...")
        fb_df = load_and_preprocess(FEEDBACK_DATA_PATH)
        if not fb_df.empty:
            df = pd.concat([df, fb_df], ignore_index=True)
            print(f"📊 Feedback assimilated. Total samples before deduplication: {len(df)}")

    # Deduplicate on cleaned text BEFORE splitting to prevent train-test data leakage
    n_before = len(df)
    df = df.drop_duplicates(subset=['clean_text']).reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"🧹 Deduplication on cleaned text: {n_before} -> {len(df)} samples ({n_removed} duplicates removed).")
    print(f"🚀 Training on {len(df)} total unique samples...")
    
    # Feature Engineering (TF-IDF with Bi-grams and Tri-grams up to max_features)
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)
    X_tfidf = tfidf.fit_transform(df['clean_text'])
    
    # Numerical features (VADER scores + punctuation count)
    X_extra = np.array(df['extra_features'].tolist())
    
    # Combine (Stack) Features
    X_combined = hstack([X_tfidf, X_extra])
    y = df['sentiment']
    
    # Stratified split to preserve class ratio across train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, 
        y, 
        test_size=TEST_SPLIT_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    print(f"📦 Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples (stratified)")
    
    # 1. Logistic Regression (Balanced weights)
    print("🧠 Training Logistic Regression model...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train, y_train)
    
    # 2. Naive Bayes Pipeline (Clamps negative VADER scores to 0 for MultinomialNB compatibility)
    print("🧠 Training Naive Bayes model...")
    nb_pipe = Pipeline([
        ('clamp', FunctionTransformer(clamp_negatives, accept_sparse=True)),
        ('nb', MultinomialNB())
    ])
    nb_pipe.fit(X_train, y_train)
    
    # 3. Soft-Voting Ensemble Model
    print("🧠 Building Ensemble Model (Soft-Voting: Logistic Regression + Naive Bayes)...")
    ensemble_model = VotingClassifier(
        estimators=[('lr', lr_model), ('nb', nb_pipe)],
        voting='soft'
    )
    ensemble_model.fit(X_train, y_train)
    
    # Create models folder
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save artifacts
    joblib.dump(lr_model, LOGISTIC_REGRESSION_MODEL_PATH)
    joblib.dump(nb_pipe, NAIVE_BAYES_MODEL_PATH)
    joblib.dump(ensemble_model, ENSEMBLE_MODEL_PATH)
    joblib.dump(tfidf, TFIDF_VECTORIZER_PATH)
    joblib.dump((X_test, y_test), TEST_DATA_SPLIT_PATH)
    
    print("✅ Self-Learning Model: Training completed successfully!")
    
    # Auto-run evaluation to update plots
    print("📊 Running evaluation to update performance charts...")
    try:
        from evaluate import evaluate_models
        evaluate_models()
    except Exception as e:
        print(f"⚠️ Warning: Evaluation failed: {e}")

if __name__ == "__main__":
    train_models()
