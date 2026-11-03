"""
Information Retrieval Ranking Module.
Implements:
1. BM25Ranker (Okapi BM25 with Robertson-Spärck Jones IDF)
2. TFIDFRanker (Vector Space Model with log-TF weighting and Cosine Similarity)
3. RocchioExpander (Pseudo-Relevance Feedback query expansion)
"""

import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class BM25Ranker:
    """
    Okapi BM25 ranking algorithm.
    Formula:
        Score(D, Q) = sum_{t in Q} IDF(t) * (tf(t, D) * (k1 + 1)) / (tf(t, D) + k1 * (1 - b + b * (|D| / avgdl)))
    Using Robertson-Spärck Jones non-negative IDF:
        IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
    """
    def __init__(self, index: Any, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b
        self.num_docs = index.num_docs
        self.avg_doc_length = index.avg_doc_length if index.avg_doc_length > 0 else 1.0
        self._idf_cache: Dict[str, float] = {}

    def compute_idf(self, term: str) -> float:
        """Computes Robertson-Spärck Jones IDF with non-negative smoothing."""
        if term in self._idf_cache:
            return self._idf_cache[term]
        
        df = self.index.get_doc_freq(term)
        if df == 0:
            idf = 0.0
        else:
            # Robertson-Spärck Jones IDF floored / non-negative smoothed
            idf = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))
            
        self._idf_cache[term] = idf
        return idf

    def score_documents(self, query_terms: List[str], candidate_doc_ids: Optional[set] = None) -> List[Tuple[int, float]]:
        """
        Calculates BM25 scores for all matching candidate documents.
        Returns list of (doc_id, score) sorted descending by score.
        """
        if not query_terms:
            return []
            
        scores = defaultdict(float)
        query_term_counts = Counter(query_terms)
        
        for term, q_tf in query_term_counts.items():
            postings = self.index.get_postings(term)
            if not postings:
                continue
                
            idf = self.compute_idf(term)
            if idf <= 0:
                continue
                
            for doc_id, tf, _ in postings:
                if candidate_doc_ids is not None and doc_id not in candidate_doc_ids:
                    continue
                    
                doc_len = self.index.doc_lengths[doc_id] if doc_id < len(self.index.doc_lengths) else self.avg_doc_length
                len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_length)
                tf_component = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
                
                scores[doc_id] += idf * tf_component * q_tf
                
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def rank(self, query_terms: List[str], k: int = 10, candidate_doc_ids: Optional[set] = None) -> List[Tuple[int, float]]:
        """Returns top-k documents with BM25 scores."""
        scores = self.score_documents(query_terms, candidate_doc_ids)
        return scores[:k]


class TFIDFRanker:
    """
    Vector Space Model with TF-IDF weighting and Cosine Similarity.
    Weights:
        tf_weight = 1 + ln(tf) if tf > 0 else 0 (sublinear tf)
        idf = ln((1 + N) / (1 + df)) + 1 (standard smooth IDF matching sklearn)
        Cosine similarity = (V_d . V_q) / (||V_d|| * ||V_q||)
    """
    def __init__(self, index: Any):
        self.index = index
        self.num_docs = index.num_docs
        self._idf_cache: Dict[str, float] = {}
        self.doc_norms: np.ndarray = self._precompute_doc_norms()

    def compute_idf(self, term: str) -> float:
        """Computes smooth IDF matching standard scikit-learn smooth_idf."""
        if term in self._idf_cache:
            return self._idf_cache[term]
        
        df = self.index.get_doc_freq(term)
        idf = math.log((1.0 + self.num_docs) / (1.0 + df)) + 1.0
        self._idf_cache[term] = idf
        return idf

    def _precompute_doc_norms(self) -> np.ndarray:
        """Precomputes Euclidean (L2) norms for all document vectors."""
        norms_sq = np.zeros(self.num_docs, dtype=np.float64)
        for term, postings in self.index.postings.items():
            idf = self.compute_idf(term)
            for doc_id, tf, _ in postings:
                if tf > 0 and doc_id < self.num_docs:
                    w = (1.0 + math.log(tf)) * idf
                    norms_sq[doc_id] += w * w
        return np.sqrt(norms_sq)

    def score_documents(self, query_terms: List[str], candidate_doc_ids: Optional[set] = None) -> List[Tuple[int, float]]:
        """
        Computes cosine similarity between query vector and documents.
        Returns list of (doc_id, score) sorted descending.
        """
        if not query_terms:
            return []
            
        query_counts = Counter(query_terms)
        query_weights: Dict[str, float] = {}
        query_norm_sq = 0.0
        
        for term, q_tf in query_counts.items():
            if q_tf > 0:
                idf = self.compute_idf(term)
                w_q = (1.0 + math.log(q_tf)) * idf
                query_weights[term] = w_q
                query_norm_sq += w_q * w_q
                
        query_norm = math.sqrt(query_norm_sq)
        if query_norm == 0.0:
            return []
            
        dot_products = defaultdict(float)
        
        for term, w_q in query_weights.items():
            postings = self.index.get_postings(term)
            idf = self.compute_idf(term)
            
            for doc_id, tf, _ in postings:
                if candidate_doc_ids is not None and doc_id not in candidate_doc_ids:
                    continue
                w_d = (1.0 + math.log(tf)) * idf
                dot_products[doc_id] += w_q * w_d
                
        scores = []
        for doc_id, dot in dot_products.items():
            doc_norm = self.doc_norms[doc_id]
            if doc_norm > 0:
                sim = dot / (doc_norm * query_norm)
                scores.append((doc_id, float(sim)))
                
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def rank(self, query_terms: List[str], k: int = 10, candidate_doc_ids: Optional[set] = None) -> List[Tuple[int, float]]:
        """Returns top-k documents with TF-IDF cosine similarity scores."""
        scores = self.score_documents(query_terms, candidate_doc_ids)
        return scores[:k]


class RocchioExpander:
    """
    Pseudo-Relevance Feedback using the Rocchio algorithm.
    Formula:
        q_new = alpha * q_0 + (beta / |D_R|) * sum_{d in D_R} d_vec
    Expands the query with top terms from top-ranked documents.
    """
    def __init__(self, ranker: Any, alpha: float = 1.0, beta: float = 0.75, top_k_feedback: int = 5):
        self.ranker = ranker
        self.alpha = alpha
        self.beta = beta
        self.top_k_feedback = top_k_feedback

    def expand_query(self, query_terms: List[str], num_expansion_terms: int = 3) -> List[str]:
        """
        Executes baseline retrieval to retrieve top_k_feedback documents,
        computes centroid term weights, and appends top new terms.
        """
        if not query_terms:
            return []
            
        initial_results = self.ranker.rank(query_terms, k=self.top_k_feedback)
        if not initial_results:
            return query_terms
            
        # Accumulate term weights from pseudo-relevant documents
        feedback_term_weights = defaultdict(float)
        num_feedback = len(initial_results)
        
        for doc_id, _ in initial_results:
            doc = self.ranker.index.get_doc(doc_id)
            if not doc:
                continue
            doc_terms = doc.get('clean_text', '').split()
            counts = Counter(doc_terms)
            for term, tf in counts.items():
                idf = self.ranker.compute_idf(term)
                w_d = (1.0 + math.log(tf)) * idf
                feedback_term_weights[term] += w_d / num_feedback
                
        # Candidate expansion terms (excluding terms already in original query)
        existing_terms = set(query_terms)
        candidate_terms = [
            (term, weight) for term, weight in feedback_term_weights.items()
            if term not in existing_terms and len(term) > 2
        ]
        candidate_terms.sort(key=lambda x: x[1], reverse=True)
        expansion_terms = [term for term, _ in candidate_terms[:num_expansion_terms]]
        
        expanded_query = list(query_terms) + expansion_terms
        return expanded_query
