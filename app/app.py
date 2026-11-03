"""
Streamlit Web Application for Tweet Sentiment Analysis and Information Retrieval.
Provides two main interfaces:
1. Search Tweets (IR Engine): BM25 / TF-IDF Cosine ranking with sentiment-aware filtering and term highlighting.
2. Tweet Sentiment Analysis: Single tweet prediction with self-learning feedback flow.
3. System & IR Analytics: Metric tables, precision-recall curves, and confusion matrices.
"""

import os
import sys
import re
import html
from pathlib import Path
import pandas as pd
import streamlit as st

# Path configuration
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "src"))

from config import (
    IR_INDEX_PATH,
    IR_METRICS_PATH,
    IR_PR_CURVE_PATH,
    IR_METRICS_PLOT_PATH,
    MODEL_COMPARISON_PLOT_PATH,
    SENTIMENT_METRICS_PATH,
    LOGISTIC_REGRESSION_MODEL_PATH,
    ENSEMBLE_MODEL_PATH,
    TFIDF_VECTORIZER_PATH,
    RESULTS_DIR,
    FEEDBACK_DATA_PATH
)
from predict import SentimentPredictor
from ir.search import SearchEngine
from ir.query import parse_query_terms

# Streamlit UI Configuration
st.set_page_config(
    page_title="Tweet IR & Sentiment Search System",
    page_icon="🐦",
    layout="wide"
)

# Custom Styling (Dark-friendly, sleek cards and badges)
st.markdown("""
<style>
    .metric-card {
        background: #1e222d;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #2d3343;
        margin-bottom: 12px;
    }
    .result-card {
        background: #1e222d;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #2f3647;
        margin-bottom: 14px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .result-card:hover {
        border-color: #4a5568;
        transform: translateY(-2px);
    }
    .badge-rank {
        background: #3b82f6;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85em;
    }
    .badge-score {
        background: #374151;
        color: #9ca3af;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.85em;
        margin-left: 6px;
    }
    .badge-entity {
        background: #4b5563;
        color: #f3f4f6;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.85em;
        margin-left: 6px;
    }
    .badge-pos {
        background: #10b981;
        color: white;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85em;
        float: right;
    }
    .badge-neg {
        background: #ef4444;
        color: white;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85em;
        float: right;
    }
    .tweet-text {
        font-size: 1.05em;
        line-height: 1.5;
        margin-top: 10px;
        margin-bottom: 6px;
        color: #e5e7eb;
    }
    mark.highlight {
        background-color: #fbbf24;
        color: #111827;
        border-radius: 3px;
        padding: 1px 3px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Cached Resource Loaders
# ==========================================

@st.cache_resource(show_spinner="Loading Inverted Index and Search Engine...")
def load_search_engine():
    return SearchEngine()


@st.cache_resource(show_spinner="Loading Sentiment Classifier...")
def load_sentiment_predictor(model_type: str):
    return SentimentPredictor(model_type=model_type)


def highlight_terms(text: str, query: str) -> str:
    """Highlights query terms in text using HTML mark tags."""
    if not text or not query:
        return html.escape(text)
        
    escaped_text = html.escape(text)
    # Extract clean words for highlighting
    clean_words = set(re.findall(r'\b[a-zA-Z0-9]+\b', query.lower()))
    stopwords = {"and", "or", "not", "the", "a", "an", "is", "in", "to", "of", "for", "on", "with"}
    search_words = [w for w in clean_words if w not in stopwords and len(w) > 1]
    
    if not search_words:
        return escaped_text
        
    pattern = re.compile(r'\b(' + '|'.join(re.escape(w) for w in search_words) + r')\b', re.IGNORECASE)
    highlighted = pattern.sub(r'<mark class="highlight">\1</mark>', escaped_text)
    return highlighted


# ==========================================
# Main Header
# ==========================================

st.title("🐦 Tweet IR Search & Sentiment Analysis System")
st.caption("University Information Retrieval Mini Project: Inverted Index, BM25 / TF-IDF Ranking, & Sentiment-Aware Search")

tabs = st.tabs(["🔍 Search Tweets (IR Engine)", "🧠 Sentiment Analysis", "📊 IR & System Evaluation"])

# Fail fast with actionable instructions if generated artifacts are missing.
missing_artifacts = []
if not IR_INDEX_PATH.exists():
    missing_artifacts.append("inverted index (`models/ir_index.joblib`)")
if not TFIDF_VECTORIZER_PATH.exists() or not ENSEMBLE_MODEL_PATH.exists():
    missing_artifacts.append("trained sentiment models (`models/*.joblib`)")

if missing_artifacts:
    st.error("Required project artifacts are missing: " + ", ".join(missing_artifacts))
    st.markdown("Generate them once from the project root, then reload this page:")
    st.code(
        "python src/train.py        # train sentiment models\n"
        "python -m src.ir.index     # build the search index\n"
        "streamlit run app/app.py",
        language="bash",
    )
    st.stop()

# Initialize Search Engine instance
engine = load_search_engine()


# ==========================================
# TAB 1: Search Tweets (Information Retrieval)
# ==========================================
with tabs[0]:
    st.subheader("Search Tweet Collection")
    st.markdown("Search over **73,800+** indexed tweets using custom IR algorithms with sentiment-aware filtering.")
    
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        search_query = st.text_input(
            "Search Query",
            placeholder='Try: "call of duty", battlefield server, ps5 OR playstation, cyberpunk NOT bugs',
            label_visibility="collapsed"
        )
    with col_btn:
        search_clicked = st.button("Search", type="primary", width="stretch")

    # Search Configuration Row
    col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns(5)
    with col_c1:
        ranking_model = st.selectbox("Ranking Algorithm", ["BM25", "TF-IDF (Cosine)"], index=0)
        model_key = "bm25" if "BM25" in ranking_model else "tfidf"
    with col_c2:
        query_mode = st.selectbox("Query Mode", ["Auto-detect", "Free Text", 'Phrase ("...")', "Boolean (AND/OR/NOT)"], index=0)
        mode_mapping = {
            "Auto-detect": "auto",
            "Free Text": "free_text",
            'Phrase ("...")': "phrase",
            "Boolean (AND/OR/NOT)": "boolean"
        }
        query_type_val = mode_mapping[query_mode]
    with col_c3:
        top_k = st.slider("Results (k)", min_value=5, max_value=50, value=10, step=5)
    with col_c4:
        sentiment_filter = st.selectbox("Sentiment Filter", ["All", "Positive", "Negative"], index=0)
    with col_c5:
        # Collect distinct entities from index
        all_entities = ["All"] + sorted(list({d.get("entity") for d in engine.index.docs if d.get("entity")}))
        entity_filter = st.selectbox("Entity Filter", all_entities, index=0)

    # Optional Rocchio expansion checkbox
    use_rocchio = st.checkbox("⚡ Apply Rocchio Pseudo-Relevance Feedback (Query Expansion on top 5 docs)", value=False)

    if search_clicked or search_query:
        if not search_query.strip():
            st.info("ℹ️ Please enter a query string in the search box above.")
        else:
            with st.spinner("Executing Information Retrieval search..."):
                response = engine.search(
                    query=search_query,
                    k=top_k,
                    model=model_key,
                    sentiment_filter=sentiment_filter if sentiment_filter != "All" else None,
                    entity_filter=entity_filter if entity_filter != "All" else None,
                    use_query_expansion=use_rocchio,
                    query_type=query_type_val
                )
            
            # Metric Summary Cards
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Retrieved Documents", f"{len(response['results'])} / {response['total_matching']}")
            m_col2.metric("Latency", f"{response['latency_ms']:.2f} ms")
            m_col3.metric("Ranking Algorithm", model_key.upper())
            m_col4.metric("Query Mode", response["query_type"].capitalize())

            # Query expansion notice
            if response["expanded_terms"]:
                st.info(f"✨ **Rocchio Expansion Terms Added:** {', '.join(response['expanded_terms'])}")

            # Sentiment distribution
            s_dist = response["sentiment_distribution"]
            pos_cnt = s_dist.get("Positive", 0)
            neg_cnt = s_dist.get("Negative", 0)
            tot_cnt = pos_cnt + neg_cnt
            
            if tot_cnt > 0:
                pos_pct = (pos_cnt / tot_cnt) * 100
                neg_pct = (neg_cnt / tot_cnt) * 100
                st.markdown(
                    f"**Sentiment Distribution in Top-{len(response['results'])}:** "
                    f"🟢 **{pos_cnt} Positive** ({pos_pct:.1f}%) | "
                    f"🔴 **{neg_cnt} Negative** ({neg_pct:.1f}%)"
                )
                st.progress(pos_pct / 100.0)

            st.divider()

            # Results List
            if not response["results"]:
                st.warning("🔍 No matching documents found. Try broadening query terms or clearing active filters.")
            else:
                for item in response["results"]:
                    rank = item["rank"]
                    score = item["score"]
                    entity = item["entity"]
                    sent = item["predicted_sentiment"]
                    conf = item["sentiment_confidence"]
                    highlighted_snippet = highlight_terms(item["text"], search_query)
                    
                    sent_badge_class = "badge-pos" if sent == "Positive" else "badge-neg"
                    
                    card_html = f"""
                    <div class="result-card">
                        <div>
                            <span class="badge-rank">#{rank}</span>
                            <span class="badge-score">Score: {score:.4f}</span>
                            <span class="badge-entity">Entity: {html.escape(entity)}</span>
                            <span class="{sent_badge_class}">{sent} ({int(conf * 100)}% conf)</span>
                        </div>
                        <div class="tweet-text">{highlighted_snippet}</div>
                        <div style="font-size: 0.8em; color: #6b7280; margin-top: 4px;">
                            Doc ID: {item['doc_id']} | Tweet ID: {item['tweet_id']} | Ground Truth Label: {item['ground_truth_sentiment']}
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)


# ==========================================
# TAB 2: Sentiment Analysis (Classifier)
# ==========================================
with tabs[1]:
    st.subheader("Classify Single Tweet Sentiment")
    st.markdown("Predict tweet sentiment (**Positive** vs **Negative**) using trained models with sarcasm-aware negations.")

    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        sentiment_input = st.text_input(
            "Enter tweet text to analyze:",
            placeholder="Ex: Having an amazing day playing the new Battlefield update!",
            key="sentiment_text_input"
        )
    with col_m2:
        model_option = st.selectbox(
            "Classification Model",
            ["Ensemble (Soft Voting)", "Logistic Regression", "Naive Bayes"],
            index=0
        )
        model_key_name = (
            "ensemble" if "Ensemble" in model_option
            else "logistic_regression" if "Logistic" in model_option
            else "naive_bayes"
        )

    if st.button("Analyze Sentiment", type="primary"):
        if not sentiment_input.strip():
            st.warning("Please enter tweet text to analyze.")
        else:
            try:
                predictor = load_sentiment_predictor(model_key_name)
                prediction_res = predictor.predict(sentiment_input)
                # Store in session state for persistent feedback flow
                st.session_state["last_analysis"] = {
                    "text": sentiment_input,
                    "model": model_option,
                    "result": prediction_res
                }
            except Exception as e:
                st.error(f"Prediction error: {e}")

    # Persistent feedback block
    if "last_analysis" in st.session_state:
        analysis = st.session_state["last_analysis"]
        res = analysis["result"]
        sent = res["sentiment"]
        conf = res["confidence"]

        st.markdown("---")
        if sent == "Positive":
            st.success(f"### Predicted Sentiment: **{sent}** ✅ (Confidence: `{conf * 100:.1f}%`)")
        elif sent == "Negative":
            st.error(f"### Predicted Sentiment: **{sent}** ❌ (Confidence: `{conf * 100:.1f}%`)")
        else:
            st.info(f"### Predicted Sentiment: **{sent}** ℹ️ (Confidence: `{conf * 100:.1f}%`)")

        # Feedback Section
        st.subheader("Was this prediction accurate?")
        fb_c1, fb_c2 = st.columns(2)
        with fb_c1:
            if st.button("👍 Yes, Accurate", key="btn_fb_correct"):
                st.toast("Thank you! Feedback recorded.", icon="✅")
        with fb_c2:
            if st.button("👎 No, Incorrect", key="btn_fb_incorrect"):
                st.session_state["show_feedback_correction"] = True

        if st.session_state.get("show_feedback_correction"):
            with st.form("correction_form"):
                st.info("Help the model improve! Select the ground truth sentiment:")
                corrected_sentiment = st.radio("Correct Sentiment:", ["Positive", "Negative"])
                submitted = st.form_submit_button("Submit Correction")
                if submitted:
                    feedback_entry = pd.DataFrame([{
                        "text": analysis["text"],
                        "sentiment": corrected_sentiment,
                        "timestamp": pd.Timestamp.now()
                    }])
                    if not FEEDBACK_DATA_PATH.exists():
                        feedback_entry.to_csv(FEEDBACK_DATA_PATH, index=False)
                    else:
                        feedback_entry.to_csv(FEEDBACK_DATA_PATH, mode="a", header=False, index=False)
                    st.success("Correction saved! Will be assimilated into next training cycle.")
                    st.session_state["show_feedback_correction"] = False


# ==========================================
# TAB 3: System & IR Evaluation Analytics
# ==========================================
with tabs[2]:
    st.subheader("Information Retrieval Performance Benchmark")
    st.markdown("""
    **Evaluation Methodology:**
    Evaluated over **48 reproducible proxy queries** across 32 entities from `data/ir_queries.csv`.
    Relevance is defined via proxy ground truth: matching the query topic with document entity field.
    """)

    if IR_METRICS_PATH.exists():
        metrics_df = pd.read_csv(IR_METRICS_PATH)
        st.dataframe(metrics_df, width="stretch")
    else:
        st.info("Run `python -m src.ir.evaluate_ir` to generate IR metrics.")

    col_ir_p1, col_ir_p2 = st.columns(2)
    with col_ir_p1:
        if IR_PR_CURVE_PATH.exists():
            st.image(str(IR_PR_CURVE_PATH), caption="11-Point Interpolated Precision-Recall Curve", width="stretch")
    with col_ir_p2:
        if IR_METRICS_PLOT_PATH.exists():
            st.image(str(IR_METRICS_PLOT_PATH), caption="Information Retrieval Metrics Comparison", width="stretch")

    st.divider()
    st.subheader("Sentiment Classifier Evaluation")
    if SENTIMENT_METRICS_PATH.exists():
        st.dataframe(pd.read_csv(SENTIMENT_METRICS_PATH), width="stretch")
    else:
        st.info("Run `python src/evaluate.py` to generate sentiment metrics.")

    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        if MODEL_COMPARISON_PLOT_PATH.exists():
            st.image(str(MODEL_COMPARISON_PLOT_PATH), caption="Sentiment Model Accuracy Comparison", width="stretch")
    with col_sc2:
        cm_path = RESULTS_DIR / "cm_logistic_regression.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix (Logistic Regression)", width="stretch")

# Footer
st.markdown("---")
st.caption("University Information Retrieval Mini Project | Python • BM25 • Vector Space Model • Scikit-learn • Streamlit")
