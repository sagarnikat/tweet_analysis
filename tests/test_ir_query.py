"""
Unit tests for Query Processing: Free text, Boolean expressions, and Phrase queries.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from ir.index import InvertedIndex
from ir.query import (
    parse_query_terms,
    detect_query_type,
    evaluate_phrase_query,
    evaluate_boolean_query
)


def create_search_corpus():
    data = [
        {"id": 0, "entity": "AI", "sentiment": "Positive", "text": "deep learning information retrieval"},
        {"id": 1, "entity": "AI", "sentiment": "Positive", "text": "machine learning natural language processing"},
        {"id": 2, "entity": "AI", "sentiment": "Negative", "text": "deep reinforcement learning"},
        {"id": 3, "entity": "Web", "sentiment": "Positive", "text": "information retrieval web search engine"},
    ]
    df = pd.DataFrame(data)
    index = InvertedIndex()
    index.build_from_dataframe(df)
    return index


def test_detect_query_type():
    assert detect_query_type('"deep learning"') == "phrase"
    assert detect_query_type('xbox AND halo') == "boolean"
    assert detect_query_type('playstation OR ps5') == "boolean"
    assert detect_query_type('cyberpunk NOT bugs') == "boolean"
    assert detect_query_type('fortnite battle royale update') == "free_text"


def test_phrase_query_exact_match():
    index = create_search_corpus()

    # "information retrieval" appears consecutively in doc 0 and doc 3
    matches = evaluate_phrase_query('"information retrieval"', index)
    assert set(matches) == {0, 3}

    # "deep learning" appears in doc 0
    matches_dl = evaluate_phrase_query('"deep learning"', index)
    assert set(matches_dl) == {0}


def test_phrase_query_non_consecutive_rejected():
    index = create_search_corpus()

    # In doc 0: "deep learning information retrieval"
    # "deep retrieval" words are both in doc 0, but NOT consecutive!
    matches = evaluate_phrase_query('"deep retrieval"', index)
    assert matches == []


def test_boolean_and():
    index = create_search_corpus()

    # 'information AND retrieval' should match doc 0 and doc 3
    matches = evaluate_boolean_query('information AND retrieval', index)
    assert matches == {0, 3}


def test_boolean_or():
    index = create_search_corpus()

    # 'deep OR machine' should match doc 0, 1, 2
    matches = evaluate_boolean_query('deep OR machine', index)
    assert matches == {0, 1, 2}


def test_boolean_not():
    index = create_search_corpus()

    # 'learning NOT deep' should match doc 1 (doc 0 and doc 2 have 'deep')
    matches = evaluate_boolean_query('learning NOT deep', index)
    assert matches == {1}
