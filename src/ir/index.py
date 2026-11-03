"""
Inverted Index implementation for Information Retrieval.
Builds an inverted index from tweet text with term frequencies, positional postings,
document frequencies, and document lengths.
"""

import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import joblib
import pandas as pd

# Path configuration
sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from config import IR_INDEX_PATH, TWEETS_DATA_PATH
except ImportError:
    from src.config import IR_INDEX_PATH, TWEETS_DATA_PATH

from preprocess import tokenize, clean_text, load_corpus


class InvertedIndex:
    """
    Inverted Index data structure for full-text search.
    Stores:
    - postings: term -> list of (doc_id, tf, positions)
    - doc_freqs: term -> document frequency (df)
    - doc_lengths: doc_id -> number of terms in document
    - docs: doc_id -> {id, entity, sentiment, text, clean_text}
    """
    def __init__(self):
        # term -> [(doc_id, tf, [pos0, pos1, ...])]
        self.postings: Dict[str, List[Tuple[int, int, List[int]]]] = {}
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: List[int] = []
        self.docs: List[Dict[str, Any]] = []
        self.num_docs: int = 0
        self.avg_doc_length: float = 0.0

    def build_from_dataframe(self, df: pd.DataFrame) -> "InvertedIndex":
        """
        Builds inverted index from a DataFrame containing:
        'id', 'entity', 'sentiment', 'text'.
        """
        print(f"🏗️ Building Inverted Index over {len(df)} tweets...")
        start_time = time.time()
        
        self.postings = defaultdict(list)
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = []
        self.docs = []
        
        for doc_id, row in enumerate(df.itertuples(index=False)):
            tweet_id = getattr(row, 'id', doc_id)
            entity = getattr(row, 'entity', 'Unknown')
            sentiment = getattr(row, 'sentiment', 'Unknown')
            raw_text = str(getattr(row, 'text', ''))
            
            cleaned = clean_text(raw_text)
            terms = cleaned.split() if cleaned else []
            doc_len = len(terms)
            self.doc_lengths.append(doc_len)
            
            # Record term positions for phrase queries and tf
            term_positions = defaultdict(list)
            for pos, term in enumerate(terms):
                term_positions[term].append(pos)
                
            for term, positions in term_positions.items():
                self.postings[term].append((doc_id, len(positions), positions))
                self.doc_freqs[term] += 1
                
            self.docs.append({
                'doc_id': doc_id,
                'id': tweet_id,
                'entity': entity,
                'sentiment': sentiment,
                'text': raw_text,
                'clean_text': cleaned
            })
            
        self.num_docs = len(self.docs)
        self.postings = dict(self.postings)
        self.doc_freqs = dict(self.doc_freqs)
        self.avg_doc_length = sum(self.doc_lengths) / self.num_docs if self.num_docs > 0 else 0.0
        
        elapsed = time.time() - start_time
        print(f"✅ Inverted Index built in {elapsed:.2f}s: {self.num_docs} documents, "
              f"{len(self.postings)} unique terms, avg doc length: {self.avg_doc_length:.1f} tokens.")
        return self

    def get_postings(self, term: str) -> List[Tuple[int, int, List[int]]]:
        """
        Returns the postings list for a given term:
        [(doc_id, tf, positions), ...]
        """
        return self.postings.get(term, [])

    def get_doc_freq(self, term: str) -> int:
        """Returns document frequency for a term."""
        return self.doc_freqs.get(term, 0)

    def get_doc(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Returns document metadata for a given doc_id."""
        if 0 <= doc_id < len(self.docs):
            return self.docs[doc_id]
        return None

    def vocabulary(self) -> List[str]:
        """Returns sorted list of terms in the vocabulary."""
        return sorted(self.postings.keys())

    def save(self, filepath: Optional[Path] = None) -> None:
        """Persists the index to disk using joblib."""
        save_path = filepath if filepath is not None else IR_INDEX_PATH
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'postings': self.postings,
            'doc_freqs': self.doc_freqs,
            'doc_lengths': self.doc_lengths,
            'docs': self.docs,
            'num_docs': self.num_docs,
            'avg_doc_length': self.avg_doc_length
        }
        joblib.dump(data, save_path, compress=3)
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"💾 Inverted Index saved to {save_path} ({file_size_mb:.2f} MB)")

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "InvertedIndex":
        """Loads a persisted InvertedIndex from disk."""
        load_path = filepath if filepath is not None else IR_INDEX_PATH
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"❌ Index file not found at {load_path}. Please build index first.")
            
        data = joblib.load(load_path)
        index = cls()
        index.postings = data['postings']
        index.doc_freqs = data['doc_freqs']
        index.doc_lengths = data['doc_lengths']
        index.docs = data['docs']
        index.num_docs = data['num_docs']
        index.avg_doc_length = data['avg_doc_length']
        return index


def build_and_save_index(data_path: Optional[Path] = None, save_path: Optional[Path] = None) -> InvertedIndex:
    """Helper function to load corpus, build index, and save to disk."""
    df = load_corpus(data_path if data_path else TWEETS_DATA_PATH)
    index = InvertedIndex()
    index.build_from_dataframe(df)
    index.save(save_path if save_path else IR_INDEX_PATH)
    return index


if __name__ == "__main__":
    build_and_save_index()
