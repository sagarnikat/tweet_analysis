"""
Information Retrieval Evaluation Module.
Evaluates ranking models (TF-IDF, BM25, BM25 + Rocchio) on a proxy test collection.

PROXY RELEVANCE NOTICE:
There are no human relevance judgments in this dataset.
Relevance is defined via proxy ground truth: a tweet is relevant to a query if and only if
its 'entity' field matches the target entity associated with the query.
This proxy judgment evaluates whether keyword and entity queries successfully retrieve
tweets belonging to the relevant entity topic.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Set, Dict, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path configuration
sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from config import (
        IR_INDEX_PATH,
        IR_QUERIES_PATH,
        IR_METRICS_PATH,
        IR_PR_CURVE_PATH,
        IR_METRICS_PLOT_PATH,
        RESULTS_DIR
    )
except ImportError:
    from src.config import (
        IR_INDEX_PATH,
        IR_QUERIES_PATH,
        IR_METRICS_PATH,
        IR_PR_CURVE_PATH,
        IR_METRICS_PLOT_PATH,
        RESULTS_DIR
    )

from ir.index import InvertedIndex
from ir.query import parse_query_terms
from ir.ranking import BM25Ranker, TFIDFRanker, RocchioExpander


# ==========================================
# Core IR Metric Functions
# ==========================================

def precision_at_k(retrieved_doc_ids: List[int], relevant_doc_ids: Set[int], k: int) -> float:
    """Calculates Precision@k: fraction of top-k retrieved docs that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    if not top_k:
        return 0.0
    relevant_retrieved = sum(1 for d in top_k if d in relevant_doc_ids)
    return relevant_retrieved / k


def recall_at_k(retrieved_doc_ids: List[int], relevant_doc_ids: Set[int], k: int) -> float:
    """Calculates Recall@k: fraction of relevant docs retrieved in top-k."""
    if not relevant_doc_ids or k <= 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    relevant_retrieved = sum(1 for d in top_k if d in relevant_doc_ids)
    return relevant_retrieved / len(relevant_doc_ids)


def average_precision(retrieved_doc_ids: List[int], relevant_doc_ids: Set[int], max_depth: int = 100) -> float:
    """
    Calculates Average Precision (AP) for a query up to max_depth.
    AP = sum_{k=1}^N (P@k * rel(k)) / |R|
    """
    if not relevant_doc_ids:
        return 0.0
    
    cutoff = min(len(retrieved_doc_ids), max_depth)
    hit_count = 0
    precisions = []
    
    for rank in range(1, cutoff + 1):
        doc_id = retrieved_doc_ids[rank - 1]
        if doc_id in relevant_doc_ids:
            hit_count += 1
            precisions.append(hit_count / rank)
            
    if not precisions:
        return 0.0
    return sum(precisions) / min(len(relevant_doc_ids), max_depth)


def reciprocal_rank(retrieved_doc_ids: List[int], relevant_doc_ids: Set[int]) -> float:
    """Calculates Reciprocal Rank (RR): 1 / rank of first relevant doc."""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_doc_ids: List[int], relevant_doc_ids: Set[int], k: int = 10) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain at k (nDCG@k).
    DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
    """
    if not relevant_doc_ids or k <= 0:
        return 0.0
        
    top_k = retrieved_doc_ids[:k]
    dcg = 0.0
    for i, doc_id in enumerate(top_k, start=1):
        rel = 1 if doc_id in relevant_doc_ids else 0
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / np.log2(i + 1.0)
            
    # Ideal DCG: perfect ranking with all relevant docs first
    idcg = 0.0
    ideal_hits = min(len(relevant_doc_ids), k)
    for i in range(1, ideal_hits + 1):
        idcg += (2.0 ** 1 - 1.0) / np.log2(i + 1.0)
        
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_interpolated_pr(
    retrieved_doc_ids: List[int],
    relevant_doc_ids: Set[int],
    recall_levels: np.ndarray = np.linspace(0.0, 1.0, 11)
) -> np.ndarray:
    """
    Computes 11-point interpolated precision values for a single query.
    P_interp(r) = max_{r' >= r} P(r')
    """
    if not relevant_doc_ids or not retrieved_doc_ids:
        return np.zeros_like(recall_levels)
        
    total_rel = len(relevant_doc_ids)
    hits = 0
    precisions = []
    recalls = []
    
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            hits += 1
            precisions.append(hits / rank)
            recalls.append(hits / total_rel)
            
    if not precisions:
        return np.zeros_like(recall_levels)
        
    interpolated = []
    for r_level in recall_levels:
        # Find maximum precision for all recall values >= r_level
        higher_precisions = [p for p, r in zip(precisions, recalls) if r >= r_level]
        if higher_precisions:
            interpolated.append(max(higher_precisions))
        else:
            interpolated.append(0.0)
            
    return np.array(interpolated)


# ==========================================
# Full Evaluation Runner
# ==========================================

def run_evaluation() -> pd.DataFrame:
    """
    Runs comprehensive IR evaluation comparing TF-IDF, BM25, and BM25+Rocchio.
    Produces ir_metrics.csv, ir_pr_curve.png, and ir_metrics_comparison.png.
    """
    print("\n" + "=" * 80)
    print("📊 INFORMATION RETRIEVAL EVALUATION")
    print("Proxy Relevance Ground Truth: Matches query topic with document entity field.")
    print("=" * 80)
    
    # Check index and file size
    if not os.path.exists(IR_INDEX_PATH):
        print(f"❌ Index not found at {IR_INDEX_PATH}. Building index first...")
        from ir.index import build_and_save_index
        build_and_save_index()
        
    index_size_mb = os.path.getsize(IR_INDEX_PATH) / (1024 * 1024)
    print(f"📦 Loaded Inverted Index from: {IR_INDEX_PATH} ({index_size_mb:.2f} MB)")
    
    t_load_start = time.time()
    index = InvertedIndex.load(IR_INDEX_PATH)
    t_load = time.time() - t_load_start
    print(f"⚡ Index load time: {t_load:.2f}s | Documents: {index.num_docs:,} | Vocab: {len(index.postings):,}")
    
    # Initialize rankers
    tfidf_ranker = TFIDFRanker(index)
    bm25_ranker = BM25Ranker(index)
    rocchio_expander = RocchioExpander(bm25_ranker, top_k_feedback=5)
    
    # Pre-index entity -> document IDs for ground truth lookups
    entity_docs: Dict[str, Set[int]] = {}
    for doc in index.docs:
        e = doc.get('entity')
        doc_id = doc.get('doc_id')
        if e not in entity_docs:
            entity_docs[e] = set()
        entity_docs[e].add(doc_id)
        
    # Load query collection
    queries_df = pd.read_csv(IR_QUERIES_PATH)
    print(f"📝 Loaded {len(queries_df)} proxy test queries from {IR_QUERIES_PATH}")
    
    models = {
        "TF-IDF": ("tfidf", tfidf_ranker, False),
        "BM25": ("bm25", bm25_ranker, False),
        "BM25 + Rocchio": ("bm25_rocchio", bm25_ranker, True)
    }
    
    results_summary = []
    pr_curves = {}
    recall_levels = np.linspace(0.0, 1.0, 11)
    
    for model_name, (m_type, ranker, use_expansion) in models.items():
        print(f"\n🏃 Evaluating model: {model_name}...")
        
        p5_list, p10_list, p20_list = [], [], []
        r10_list, r20_list = [], []
        ap_list, rr_list, ndcg10_list = [], [], []
        latencies = []
        all_query_pr = []
        
        for _, row in queries_df.iterrows():
            q_text = row['query_text']
            target_entity = row['target_entity']
            relevant_doc_ids = entity_docs.get(target_entity, set())
            
            q_terms = parse_query_terms(q_text)
            
            t0 = time.time()
            if use_expansion:
                expanded_terms = rocchio_expander.expand_query(q_terms, num_expansion_terms=3)
                scored_docs = ranker.score_documents(expanded_terms)
            else:
                scored_docs = ranker.score_documents(q_terms)
            latencies.append((time.time() - t0) * 1000.0)
            
            retrieved_doc_ids = [doc_id for doc_id, _ in scored_docs]
            
            p5_list.append(precision_at_k(retrieved_doc_ids, relevant_doc_ids, 5))
            p10_list.append(precision_at_k(retrieved_doc_ids, relevant_doc_ids, 10))
            p20_list.append(precision_at_k(retrieved_doc_ids, relevant_doc_ids, 20))
            
            r10_list.append(recall_at_k(retrieved_doc_ids, relevant_doc_ids, 10))
            r20_list.append(recall_at_k(retrieved_doc_ids, relevant_doc_ids, 20))
            
            ap_list.append(average_precision(retrieved_doc_ids, relevant_doc_ids, max_depth=100))
            rr_list.append(reciprocal_rank(retrieved_doc_ids, relevant_doc_ids))
            ndcg10_list.append(ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, 10))
            
            pr_curve = compute_interpolated_pr(retrieved_doc_ids, relevant_doc_ids, recall_levels)
            all_query_pr.append(pr_curve)
            
        mean_pr = np.mean(all_query_pr, axis=0)
        pr_curves[model_name] = mean_pr
        
        results_summary.append({
            "Model": model_name,
            "P@5": round(float(np.mean(p5_list)), 4),
            "P@10": round(float(np.mean(p10_list)), 4),
            "P@20": round(float(np.mean(p20_list)), 4),
            "Recall@10": round(float(np.mean(r10_list)), 4),
            "MAP": round(float(np.mean(ap_list)), 4),
            "MRR": round(float(np.mean(rr_list)), 4),
            "nDCG@10": round(float(np.mean(ndcg10_list)), 4),
            "Avg_Latency_ms": round(float(np.mean(latencies)), 2)
        })
        
    metrics_df = pd.DataFrame(results_summary)
    
    # Save metrics table
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_df.to_csv(IR_METRICS_PATH, index=False)
    print(f"\n💾 Saved IR Metrics table to: {IR_METRICS_PATH}")
    
    # Plot 1: 11-point Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    for (m_name, pr_vals), marker, color in zip(pr_curves.items(), markers, colors):
        plt.plot(recall_levels, pr_vals, marker=marker, label=m_name, color=color, linewidth=2)
        
    plt.title("11-Point Interpolated Precision-Recall Curve (Proxy Relevance)")
    plt.xlabel("Recall")
    plt.ylabel("Interpolated Precision")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(IR_PR_CURVE_PATH, dpi=300)
    plt.close()
    print(f"📈 Saved P-R Curve plot to: {IR_PR_CURVE_PATH}")
    
    # Plot 2: Metrics Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    plot_df = metrics_df.melt(
        id_vars=["Model"],
        value_vars=["P@5", "P@10", "P@20", "MAP", "MRR", "nDCG@10"],
        var_name="Metric",
        value_name="Score"
    )
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette="viridis")
    plt.title("Information Retrieval Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(IR_METRICS_PLOT_PATH, dpi=300)
    plt.close()
    print(f"📊 Saved Metrics Comparison plot to: {IR_METRICS_PLOT_PATH}")
    
    # Print formatted output table
    print("\n" + "=" * 95)
    print(f"{'Model':<16} | {'P@5':<7} | {'P@10':<7} | {'P@20':<7} | {'MAP':<7} | {'MRR':<7} | {'nDCG@10':<8} | {'Latency':<9}")
    print("-" * 95)
    for _, r in metrics_df.iterrows():
        print(f"{r['Model']:<16} | {r['P@5']:<7.4f} | {r['P@10']:<7.4f} | {r['P@20']:<7.4f} | {r['MAP']:<7.4f} | {r['MRR']:<7.4f} | {r['nDCG@10']:<8.4f} | {r['Avg_Latency_ms']:<7.2f} ms")
    print("=" * 95)
    
    return metrics_df


if __name__ == "__main__":
    run_evaluation()
