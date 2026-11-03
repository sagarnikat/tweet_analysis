"""
Sentiment-Aware Search Engine.
Integrates InvertedIndex, BM25 / TF-IDF Rankers, Query Processing,
and SentimentPredictor for sentiment-aware tweet retrieval and filtering.
"""

import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Path configuration
sys.path.append(str(Path(__file__).resolve().parent.parent))
from predict import SentimentPredictor
from ir.index import InvertedIndex
from ir.query import (
    parse_query_terms,
    detect_query_type,
    evaluate_phrase_query,
    evaluate_boolean_query
)
from ir.ranking import BM25Ranker, TFIDFRanker, RocchioExpander


class SearchEngine:
    """
    SearchEngine orchestrates:
    - Inverted index retrieval
    - Ranking with BM25 or TF-IDF Cosine
    - Boolean and Phrase query execution
    - Optional Rocchio query expansion
    - Classification of retrieved tweets via SentimentPredictor
    - Filtering by predicted sentiment or entity
    - Sentiment distribution reporting
    """
    def __init__(
        self,
        index: Optional[InvertedIndex] = None,
        predictor: Optional[SentimentPredictor] = None,
        classifier_type: str = "ensemble"
    ):
        self.index = index if index is not None else InvertedIndex.load()
        try:
            self.predictor = predictor if predictor is not None else SentimentPredictor(model_type=classifier_type)
        except Exception as e:
            print(f"⚠️ Predictor load fallback: {e}")
            self.predictor = None

        self.bm25_ranker = BM25Ranker(self.index)
        self.tfidf_ranker = TFIDFRanker(self.index)
        self.rocchio = RocchioExpander(self.bm25_ranker)

    def search(
        self,
        query: str,
        k: int = 10,
        model: str = "bm25",
        sentiment_filter: Optional[str] = None,
        entity_filter: Optional[str] = None,
        use_query_expansion: bool = False,
        query_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Executes search with ranking, sentiment inference, and optional filters.
        
        Args:
            query: User search query string
            k: Number of top results to return
            model: 'bm25' or 'tfidf'
            sentiment_filter: Optional filter ('Positive' or 'Negative')
            entity_filter: Optional entity filter (e.g. 'CallOfDuty', 'Amazon')
            use_query_expansion: Whether to apply Rocchio pseudo-relevance feedback
            query_type: 'auto', 'free_text', 'phrase', or 'boolean'
            
        Returns:
            Dict containing results, sentiment distribution, latency, and query stats.
        """
        start_time = time.time()
        
        if not query or not query.strip():
            return {
                "query": query,
                "results": [],
                "total_matching": 0,
                "sentiment_distribution": {"Positive": 0, "Negative": 0},
                "latency_ms": 0.0,
                "model": model,
                "query_type": query_type,
                "expanded_terms": []
            }

        # Resolve query type
        actual_query_type = detect_query_type(query) if query_type == "auto" else query_type
        
        candidate_doc_ids = None
        expanded_terms: List[str] = []
        query_terms = parse_query_terms(query)
        
        if actual_query_type == "phrase":
            phrase_docs = evaluate_phrase_query(query, self.index)
            candidate_doc_ids = set(phrase_docs)
            if not candidate_doc_ids:
                return self._empty_response(query, model, actual_query_type, start_time)
        elif actual_query_type == "boolean":
            bool_docs = evaluate_boolean_query(query, self.index)
            candidate_doc_ids = bool_docs
            if not candidate_doc_ids:
                return self._empty_response(query, model, actual_query_type, start_time)
        else:
            # Free text with optional Rocchio expansion
            if use_query_expansion and query_terms:
                original_terms = list(query_terms)
                query_terms = self.rocchio.expand_query(original_terms, num_expansion_terms=3)
                expanded_terms = [t for t in query_terms if t not in original_terms]

        # Select ranker
        ranker = self.bm25_ranker if model.lower() == "bm25" else self.tfidf_ranker
        
        # Over-fetch candidates if filters are active to guarantee up to k filtered items
        fetch_k = k * 10 if (sentiment_filter or entity_filter) else k
        raw_scored = ranker.score_documents(query_terms, candidate_doc_ids=candidate_doc_ids)
        total_matching = len(raw_scored)
        
        results: List[Dict[str, Any]] = []
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        
        # Process and filter candidates
        for doc_id, score in raw_scored[:fetch_k]:
            doc = self.index.get_doc(doc_id)
            if not doc:
                continue
                
            doc_entity = doc.get("entity", "Unknown")
            if entity_filter and entity_filter.lower() != "all" and doc_entity != entity_filter:
                continue
                
            text = doc.get("text", "")
            
            # Predict sentiment using the classifier
            if self.predictor is not None:
                pred_result = self.predictor.predict(text)
                predicted_sentiment = pred_result.get("sentiment", "Neutral")
                confidence = pred_result.get("confidence", 0.0)
            else:
                predicted_sentiment = doc.get("sentiment", "Unknown")
                confidence = 1.0
                
            # Filter by predicted sentiment if requested
            if sentiment_filter and sentiment_filter.lower() != "all":
                if predicted_sentiment.title() != sentiment_filter.title():
                    continue
                    
            if len(results) < k:
                results.append({
                    "rank": len(results) + 1,
                    "doc_id": doc_id,
                    "tweet_id": doc.get("id", doc_id),
                    "score": round(float(score), 4),
                    "text": text,
                    "entity": doc_entity,
                    "ground_truth_sentiment": doc.get("sentiment", "Unknown"),
                    "predicted_sentiment": predicted_sentiment,
                    "sentiment_confidence": round(float(confidence), 2)
                })
                sentiment_counts[predicted_sentiment] = sentiment_counts.get(predicted_sentiment, 0) + 1
                
            if len(results) >= k:
                break
                
        latency_ms = (time.time() - start_time) * 1000.0
        
        return {
            "query": query,
            "results": results,
            "total_matching": total_matching,
            "sentiment_distribution": {
                "Positive": sentiment_counts.get("Positive", 0),
                "Negative": sentiment_counts.get("Negative", 0)
            },
            "latency_ms": round(latency_ms, 2),
            "model": model,
            "query_type": actual_query_type,
            "expanded_terms": expanded_terms
        }

    def _empty_response(self, query: str, model: str, query_type: str, start_time: float) -> Dict[str, Any]:
        return {
            "query": query,
            "results": [],
            "total_matching": 0,
            "sentiment_distribution": {"Positive": 0, "Negative": 0},
            "latency_ms": round((time.time() - start_time) * 1000.0, 2),
            "model": model,
            "query_type": query_type,
            "expanded_terms": []
        }
