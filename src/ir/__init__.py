"""
Information Retrieval (IR) package.
Provides core indexing, query parsing, ranking, evaluation, and sentiment-aware search.

Exports are resolved lazily so that running a submodule as a script
(e.g. ``python -m src.ir.index``) does not trigger runpy's
"found in sys.modules after import of package" warning.
"""

from importlib import import_module

__all__ = [
    "InvertedIndex",
    "build_and_save_index",
    "parse_query_terms",
    "detect_query_type",
    "evaluate_phrase_query",
    "evaluate_boolean_query",
    "BM25Ranker",
    "TFIDFRanker",
    "RocchioExpander",
    "SearchEngine",
]

_LAZY_EXPORTS = {
    "InvertedIndex": ".index",
    "build_and_save_index": ".index",
    "parse_query_terms": ".query",
    "detect_query_type": ".query",
    "evaluate_phrase_query": ".query",
    "evaluate_boolean_query": ".query",
    "BM25Ranker": ".ranking",
    "TFIDFRanker": ".ranking",
    "RocchioExpander": ".ranking",
    "SearchEngine": ".search",
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
