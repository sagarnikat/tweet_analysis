import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER analyzer for emotional intensity
vader_analyzer = SentimentIntensityAnalyzer()

# Ensure corpora are downloaded once
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Compiled patterns for fast, consistent cleaning
URL_PATTERN = re.compile(
    r'https?://\S+|www\.\S+|\bpic\.twitter\.com/\S+|\bt\.co/\S+',
    re.IGNORECASE,
)
MENTION_PATTERN = re.compile(r'@[A-Za-z0-9_]+')
HASHTAG_PATTERN = re.compile(r'#([A-Za-z0-9_]+)')
# Keep letters, digits and apostrophes; replace all other punctuation with spaces.
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9'\s]")
WHITESPACE_PATTERN = re.compile(r'\s+')

# Negation words are CRITICAL for sentiment reversal (e.g. "not good").
NEGATION_WORDS = {'not', 'no', 'nor', 'neither', 'never', 'none', 'but', 'however'}
STOP_WORDS = set(stopwords.words('english')) - NEGATION_WORDS

LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    """
    Normalizes tweet text for both sentiment features and IR indexing.

    Steps:
    1. Lowercase.
    2. Strip URLs and @mentions.
    3. Keep the textual part of hashtags (#Battlefield -> battlefield).
    4. Remove punctuation while preserving apostrophes and negation words.
    5. Lemmatize tokens with WordNet.

    Punctuation is intentionally removed *before* stop-word filtering so that
    tokens like ``"game."`` and ``"game!"`` both become ``"game"``.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_PATTERN.sub(' ', text)
    text = MENTION_PATTERN.sub(' ', text)
    text = HASHTAG_PATTERN.sub(r'\1', text)
    text = NON_ALNUM_PATTERN.sub(' ', text)
    text = WHITESPACE_PATTERN.sub(' ', text).strip()

    words = [LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)

def tokenize(text):
    """
    Tokenizes text by applying clean_text and splitting into term tokens.
    Guarantees that indexer and query processor use the exact same preprocessing pipeline.
    """
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []

def clamp_negatives(X):
    """
    Clamps negative values in feature matrix to zero.
    Required for MultinomialNB compatibility when stacking with signed features like VADER compound scores.
    Handles both scipy sparse matrices (csr_matrix) and dense numpy arrays.
    """
    if hasattr(X, "tocsr") or (hasattr(X, "data") and not isinstance(X, np.ndarray)):
        X_copy = X.copy()
        X_copy.data[X_copy.data < 0] = 0
        return X_copy
    else:
        X_copy = np.array(X, copy=True)
        X_copy[X_copy < 0] = 0
        return X_copy

def get_extra_features(text):
    """
    Extracts numerical features beyond simple words:
    - VADER scores (Negative, Neutral, Positive, Compound)
    - Punctuation (Exclamations/Question marks often signal sarcasm)
    """
    scores = vader_analyzer.polarity_scores(text)
    
    exclam_count = text.count('!')
    quest_count = text.count('?')
    
    return [
        scores['compound'], 
        scores['neg'], 
        scores['neu'], 
        scores['pos'],
        exclam_count,
        quest_count
    ]

def load_and_preprocess(filepath):
    """
    Loads and applies deep cleaning + sentiment scoring.
    """
    try:
        # Load and let pandas infer headers first
        df = pd.read_csv(filepath)
        
        # If headers don't match or it's raw without headers
        if not all(col in df.columns for col in ['text', 'sentiment']):
            df_no_header = pd.read_csv(filepath, header=None)
            if df_no_header.shape[1] >= 4:
                df_no_header.columns = ['id', 'entity', 'sentiment', 'text'][:df_no_header.shape[1]]
                df = df_no_header
            elif df_no_header.shape[1] == 2:
                df_no_header.columns = ['text', 'sentiment']
                df = df_no_header
    except Exception as e:
        df = pd.read_csv(filepath)

    # Filter only Positive & Negative
    if 'sentiment' in df.columns:
        df = df[df['sentiment'].astype(str).str.title().isin(['Positive', 'Negative'])]

    # Drop nulls
    df = df.dropna(subset=['text', 'sentiment'])

    print("⏳ Deep Preprocessing & Sentiment Scoring... (this takes a moment)")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Calculate extra numerical features
    extra_feats = df['text'].apply(get_extra_features).tolist()
    df['extra_features'] = extra_feats

    # remove empty cleaning results
    df = df[df['clean_text'].str.strip() != ""]

    return df

def load_corpus(filepath=None):
    """
    Loads full tweet collection for Information Retrieval indexing.
    Preserves all 32 entities and sentiments (Positive, Negative, Neutral, Irrelevant).
    Returns DataFrame with columns ['id', 'entity', 'sentiment', 'text'].
    """
    if filepath is None:
        try:
            from config import TWEETS_DATA_PATH
            filepath = TWEETS_DATA_PATH
        except ImportError:
            from src.config import TWEETS_DATA_PATH
            filepath = TWEETS_DATA_PATH
            
    df = pd.read_csv(filepath, header=None)
    if df.shape[1] >= 4:
        df.columns = ['id', 'entity', 'sentiment', 'text'][:df.shape[1]]
    elif df.shape[1] == 2:
        df.columns = ['text', 'sentiment']
        df['id'] = range(len(df))
        df['entity'] = 'General'
        
    df = df.dropna(subset=['text']).copy()
    df['text'] = df['text'].astype(str)
    df = df[df['text'].str.strip() != ""].reset_index(drop=True)
    return df