"""
Evaluation script for trained sentiment analysis models.
Evaluates Logistic Regression, Naive Bayes, and Ensemble models on the test split.
Generates classification reports, confusion matrices, and accuracy comparison plots.
"""

import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Path configuration
sys.path.append(str(Path(__file__).resolve().parent))
try:
    from config import (
        MODELS_DIR,
        RESULTS_DIR,
        TEST_DATA_SPLIT_PATH,
        LOGISTIC_REGRESSION_MODEL_PATH,
        NAIVE_BAYES_MODEL_PATH,
        ENSEMBLE_MODEL_PATH,
        MODEL_COMPARISON_PLOT_PATH,
        SENTIMENT_METRICS_PATH
    )
except ImportError:
    from src.config import (
        MODELS_DIR,
        RESULTS_DIR,
        TEST_DATA_SPLIT_PATH,
        LOGISTIC_REGRESSION_MODEL_PATH,
        NAIVE_BAYES_MODEL_PATH,
        ENSEMBLE_MODEL_PATH,
        MODEL_COMPARISON_PLOT_PATH,
        SENTIMENT_METRICS_PATH
    )

def evaluate_models():
    # Load test data and models
    if not os.path.exists(TEST_DATA_SPLIT_PATH):
        print(f"❌ Test data split not found at {TEST_DATA_SPLIT_PATH}. Training first...")
        from train import train_models
        train_models()
        
    X_test, y_test = joblib.load(TEST_DATA_SPLIT_PATH)
    lr_model = joblib.load(LOGISTIC_REGRESSION_MODEL_PATH)
    nb_model = joblib.load(NAIVE_BAYES_MODEL_PATH)
    
    models = {
        "Logistic Regression": lr_model,
        "Naive Bayes": nb_model
    }
    
    if os.path.exists(ENSEMBLE_MODEL_PATH):
        ensemble_model = joblib.load(ENSEMBLE_MODEL_PATH)
        models["Ensemble"] = ensemble_model
    
    accuracies = {}
    metric_rows = []
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        macro = report.get("macro avg", {})
        weighted = report.get("weighted avg", {})
        metric_rows.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Macro_Precision": round(float(macro.get("precision", 0.0)), 4),
            "Macro_Recall": round(float(macro.get("recall", 0.0)), 4),
            "Macro_F1": round(float(macro.get("f1-score", 0.0)), 4),
            "Weighted_F1": round(float(weighted.get("f1-score", 0.0)), 4),
        })
        
        print(f"--- Model Evaluation: {name} ---")
        print(f"Accuracy: {acc:.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        classes = model.classes_ if hasattr(model, 'classes_') else ['Negative', 'Positive']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(RESULTS_DIR, f"cm_{name.lower().replace(' ', '_')}.png"))
        plt.close()
        
    # Model Comparison Chart
    plt.figure(figsize=(8, 5))
    model_names = list(accuracies.keys())
    model_scores = list(accuracies.values())
    sns.barplot(x=model_names, y=model_scores, hue=model_names, palette='viridis', legend=False)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy Score")
    plt.ylim(0, 1.05)
    for i, v in enumerate(model_scores):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
    plt.tight_layout()
    
    # Save comparison plot
    plt.savefig(MODEL_COMPARISON_PLOT_PATH)
    plt.close()

    # Save metrics table for the README and the Streamlit dashboard
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(SENTIMENT_METRICS_PATH, index=False)
    
    print(f"\n✅ Evaluation results and plots saved in: {RESULTS_DIR}")
    print(f"💾 Metrics table: {SENTIMENT_METRICS_PATH}")
    return accuracies

if __name__ == "__main__":
    evaluate_models()
