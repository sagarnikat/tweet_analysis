"""
Unit tests for the InvertedIndex on a tiny hand-made corpus.
"""

import sys
import tempfile
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from ir.index import InvertedIndex


def create_toy_corpus():
    data = [
        {"id": 101, "entity": "FruitCo", "sentiment": "Positive", "text": "fresh red apple"},
        {"id": 102, "entity": "FruitCo", "sentiment": "Negative", "text": "bad apple and yellow banana"},
        {"id": 103, "entity": "VeggieCo", "sentiment": "Positive", "text": "sweet red cherry and fresh apple"}
    ]
    return pd.DataFrame(data)


def test_index_build_and_vocab():
    df = create_toy_corpus()
    index = InvertedIndex()
    index.build_from_dataframe(df)

    assert index.num_docs == 3
    vocab = index.vocabulary()
    assert "apple" in vocab
    assert "banana" in vocab
    assert "cherry" in vocab


def test_postings_correctness():
    df = create_toy_corpus()
    index = InvertedIndex()
    index.build_from_dataframe(df)

    # 'apple' appears in doc 0, doc 1, and doc 2
    apple_postings = index.get_postings("apple")
    assert len(apple_postings) == 3
    doc_ids = [doc_id for doc_id, tf, pos in apple_postings]
    assert doc_ids == [0, 1, 2]

    # 'banana' appears only in doc 1
    banana_postings = index.get_postings("banana")
    assert len(banana_postings) == 1
    assert banana_postings[0][0] == 1  # doc_id 1
    assert banana_postings[0][1] == 1  # tf 1

    # Check positions for 'apple' in doc 0: "fresh red apple"
    # cleaned: 'fresh red apple' -> apple is pos 2
    assert 2 in apple_postings[0][2]


def test_doc_frequencies_and_lengths():
    df = create_toy_corpus()
    index = InvertedIndex()
    index.build_from_dataframe(df)

    assert index.get_doc_freq("apple") == 3
    assert index.get_doc_freq("banana") == 1
    assert index.get_doc_freq("nonexistent") == 0
    assert len(index.doc_lengths) == 3
    assert index.avg_doc_length > 0


def test_index_serialization_roundtrip():
    df = create_toy_corpus()
    index = InvertedIndex()
    index.build_from_dataframe(df)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        index.save(tmp_path)
        loaded = InvertedIndex.load(tmp_path)

        assert loaded.num_docs == index.num_docs
        assert loaded.avg_doc_length == index.avg_doc_length
        assert loaded.vocabulary() == index.vocabulary()
        assert loaded.get_doc(0)["text"] == index.get_doc(0)["text"]
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
