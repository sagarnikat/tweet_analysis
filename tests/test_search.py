"""
Unit tests for SearchEngine end-to-end integration:
Search execution, model selection, sentiment filter, entity filter, and latency measurement.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from ir.index import InvertedIndex
from ir.search import SearchEngine


def create_mock_engine():
    data = [
        {"id": 1, "entity": "Xbox", "sentiment": "Positive", "text": "I love playing on my Xbox series x, best console ever!"},
        {"id": 2, "entity": "Xbox", "sentiment": "Negative", "text": "My Xbox series x crashed again, terrible experience."},
        {"id": 3, "entity": "PlayStation", "sentiment": "Positive", "text": "PlayStation 5 dualsense controller feels amazing."},
        {"id": 4, "entity": "PlayStation", "sentiment": "Negative", "text": "PlayStation network is down and I hate it."}
    ]
    df = pd.DataFrame(data)
    index = InvertedIndex().build_from_dataframe(df)
    engine = SearchEngine(index=index)
    return engine


def test_search_engine_bm25_and_tfidf():
    engine = create_mock_engine()

    bm25_res = engine.search("xbox series", k=2, model="bm25")
    assert len(bm25_res["results"]) > 0
    assert bm25_res["results"][0]["entity"] == "Xbox"
    assert bm25_res["latency_ms"] >= 0.0

    tfidf_res = engine.search("xbox series", k=2, model="tfidf")
    assert len(tfidf_res["results"]) > 0
    assert tfidf_res["results"][0]["entity"] == "Xbox"


def test_search_engine_sentiment_filter():
    engine = create_mock_engine()

    # Search with sentiment filter Positive
    pos_res = engine.search("xbox", k=2, sentiment_filter="Positive")
    for r in pos_res["results"]:
        assert r["predicted_sentiment"] == "Positive"

    # Search with sentiment filter Negative
    neg_res = engine.search("xbox", k=2, sentiment_filter="Negative")
    for r in neg_res["results"]:
        assert r["predicted_sentiment"] == "Negative"


def test_search_engine_entity_filter():
    engine = create_mock_engine()

    # Query term might appear or match, but entity filter forces only PlayStation
    res = engine.search("console", k=2, entity_filter="PlayStation")
    for r in res["results"]:
        assert r["entity"] == "PlayStation"


def test_search_engine_empty_query():
    engine = create_mock_engine()
    res = engine.search("", k=5)
    assert res["results"] == []
    assert res["total_matching"] == 0
