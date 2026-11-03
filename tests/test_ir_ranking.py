"""
Unit tests for Ranking algorithms:
1. TF-IDF Cosine Similarity parity against scikit-learn
2. BM25 term frequency and length normalization ordering
3. Rocchio pseudo-relevance feedback query expansion
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from ir.index import InvertedIndex
from ir.ranking import TFIDFRanker, BM25Ranker, RocchioExpander


def test_tfidf_sklearn_parity():
    """
    Verifies that our custom TF-IDF Cosine implementation matches
    scikit-learn's TfidfVectorizer + cosine_similarity on identical cleaned text.
    """
    docs = [
        "information retrieval and data mining",
        "information retrieval engine search query",
        "python machine learning and deep learning",
        "search engine optimization for web data"
    ]
    df = pd.DataFrame([{"id": i, "entity": "test", "sentiment": "pos", "text": d} for i, d in enumerate(docs)])
    index = InvertedIndex().build_from_dataframe(df)
    ranker = TFIDFRanker(index)

    # Scikit-learn baseline
    clean_docs = [doc["clean_text"] for doc in index.docs]
    vec = TfidfVectorizer(sublinear_tf=True, norm="l2", smooth_idf=True)
    X_sk = vec.fit_transform(clean_docs)

    test_queries = ["information retrieval", "machine learning", "search engine"]
    for query in test_queries:
        clean_q = " ".join(query.split())
        q_sk = vec.transform([clean_q])
        sk_scores = cosine_similarity(q_sk, X_sk)[0]

        our_scored = ranker.score_documents(query.split())
        our_score_dict = {doc_id: score for doc_id, score in our_scored}

        for doc_id in range(len(docs)):
            our_s = our_score_dict.get(doc_id, 0.0)
            sk_s = float(sk_scores[doc_id])
            assert np.isclose(our_s, sk_s, atol=1e-4), f"Mismatch for '{query}' on doc {doc_id}: {our_s} vs {sk_s}"


def test_bm25_term_frequency_ordering():
    """
    Documents with higher term frequency of the query term should rank higher.
    """
    data = [
        {"id": 0, "entity": "game", "sentiment": "pos", "text": "overwatch is a game"},
        {"id": 1, "entity": "game", "sentiment": "pos", "text": "overwatch overwatch overwatch hero shooter"},
        {"id": 2, "entity": "game", "sentiment": "pos", "text": "unrelated gaming tweet"}
    ]
    df = pd.DataFrame(data)
    index = InvertedIndex().build_from_dataframe(df)
    ranker = BM25Ranker(index, k1=1.5, b=0.75)

    results = ranker.rank(["overwatch"], k=3)
    ranked_doc_ids = [d for d, s in results]

    # Doc 1 (3 mentions) must rank above Doc 0 (1 mention)
    assert ranked_doc_ids[0] == 1
    assert ranked_doc_ids[1] == 0


def test_bm25_length_normalization():
    """
    Between two documents with the same number of term occurrences,
    the shorter document should receive a higher BM25 score due to length normalization (b=0.75).
    """
    data = [
        {"id": 0, "entity": "tech", "sentiment": "pos", "text": "apple"},
        {"id": 1, "entity": "tech", "sentiment": "pos", "text": "apple banana orange cherry grape mango pear pineapple"}
    ]
    df = pd.DataFrame(data)
    index = InvertedIndex().build_from_dataframe(df)
    ranker = BM25Ranker(index, k1=1.5, b=0.75)

    results = ranker.rank(["apple"], k=2)
    # Doc 0 is shorter and has 1 mention -> ranks higher than Doc 1 which is long and has 1 mention
    assert results[0][0] == 0
    assert results[1][0] == 1
    assert results[0][1] > results[1][1]


def test_rocchio_query_expansion():
    """
    Verifies that Rocchio Pseudo-Relevance Feedback adds relevant expansion terms from top documents.
    """
    data = [
        {"id": 0, "entity": "COD", "sentiment": "pos", "text": "warzone modern warfare multiplayer"},
        {"id": 1, "entity": "COD", "sentiment": "pos", "text": "warzone battle royale verdansk"},
        {"id": 2, "entity": "other", "sentiment": "pos", "text": "completely unrelated cooking recipe"}
    ]
    df = pd.DataFrame(data)
    index = InvertedIndex().build_from_dataframe(df)
    ranker = BM25Ranker(index)
    expander = RocchioExpander(ranker, top_k_feedback=2)

    query = ["warzone"]
    expanded = expander.expand_query(query, num_expansion_terms=2)

    assert "warzone" in expanded
    assert len(expanded) > len(query)
