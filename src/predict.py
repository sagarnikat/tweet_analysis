"""
Inference module for sentiment analysis.
Provides the SentimentPredictor class for predicting sentiment of input texts.
Supports Logistic Regression, Naive Bayes, and Ensemble models.
"""

import os
import sys
from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import hstack

# Path configuration
sys.path.append(str(Path(__file__).resolve().parent))
try:
    from config import (
        MODELS_DIR,
        LOGISTIC_REGRESSION_MODEL_PATH,
        NAIVE_BAYES_MODEL_PATH,
        ENSEMBLE_MODEL_PATH,
        TFIDF_VECTORIZER_PATH
    )
except ImportError:
    from src.config import (
        MODELS_DIR,
        LOGISTIC_REGRESSION_MODEL_PATH,
        NAIVE_BAYES_MODEL_PATH,
        ENSEMBLE_MODEL_PATH,
        TFIDF_VECTORIZER_PATH
    )

from preprocess import clean_text, get_extra_features, clamp_negatives

class SentimentPredictor:
    """
    Predicts sentiment for input text using trained models.
    """
    def __init__(self, model_type="logistic_regression"):
        self.model_type = model_type.lower().replace(" ", "_")
        
        # Map model type to file path
        model_paths = {
            "logistic_regression": LOGISTIC_REGRESSION_MODEL_PATH,
            "naive_bayes": NAIVE_BAYES_MODEL_PATH,
            "ensemble": ENSEMBLE_MODEL_PATH
        }
        
        self.model_path = model_paths.get(self.model_type, LOGISTIC_REGRESSION_MODEL_PATH)
        self.tfidf_path = TFIDF_VECTORIZER_PATH
        
        # Fallback if selected model file does not exist
        if not os.path.exists(self.model_path):
            self.model_path = LOGISTIC_REGRESSION_MODEL_PATH
            self.model_type = "logistic_regression"
            
        if not os.path.exists(self.model_path) or not os.path.exists(self.tfidf_path):
            raise FileNotFoundError(f"❌ Models not found in {MODELS_DIR}. Please run training first.")
            
        self.model = joblib.load(self.model_path)
        self.tfidf = joblib.load(self.tfidf_path)

    def predict(self, text):
        """
        Predict sentiment and confidence score:
        1. Clean text (negation-aware)
        2. TF-IDF feature extraction
        3. VADER sentiment features & punctuation count
        4. Stack sparse TF-IDF and dense features
        5. Model inference
        """
        if not isinstance(text, str) or not text.strip():
            return {"sentiment": "Neutral", "confidence": 0.0}

        cleaned = clean_text(text)
        X_tfidf = self.tfidf.transform([cleaned])
        X_extra = np.array([get_extra_features(text)])
        X_combined = hstack([X_tfidf, X_extra])
        
        # If model is raw MultinomialNB (not pipeline), ensure non-negative inputs
        if self.model_type == "naive_bayes" and not hasattr(self.model, "named_steps"):
            X_combined = clamp_negatives(X_combined)
            
        prediction = self.model.predict(X_combined)[0]
        
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X_combined)[0]
            confidence = max(probabilities)
        else:
            confidence = 1.0

        return {
            "sentiment": str(prediction),
            "confidence": round(float(confidence), 4)
        }

if __name__ == "__main__":
    try:
        p = SentimentPredictor()
        test_case = "It is so sad that I am happy today"
        result = p.predict(test_case)
        print(f"--- Sarcasm Test ---\nText: {test_case}\nResult: {result}")
    except Exception as e:
        print(f"Please run train.py first. Error: {e}")
