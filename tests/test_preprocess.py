"""
Unit tests for text preprocessing, tokenization, and feature extraction.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from preprocess import clean_text, tokenize, get_extra_features, clamp_negatives


def test_clean_text_basic():
    text = "Check out https://t.co/xyz and visit www.test.com! @user #awesome"
    cleaned = clean_text(text)
    assert "http" not in cleaned
    assert "www" not in cleaned
    assert "@user" not in cleaned
    assert "#awesome" not in cleaned
    assert "check" in cleaned


def test_clean_text_preserves_negations():
    # Negation words must be preserved for sentiment detection
    text = "This movie is not good and neither is it fun, I never saw something so bad"
    cleaned = clean_text(text)
    assert "not" in cleaned.split()
    assert "neither" in cleaned.split()
    assert "never" in cleaned.split()


def test_clean_text_lemmatization():
    text = "Playing games with players and computers"
    cleaned = clean_text(text)
    # 'games' should lemmatize to 'game', 'players' to 'player', 'computers' to 'computer'
    words = cleaned.split()
    assert "game" in words
    assert "player" in words
    assert "computer" in words


def test_tokenize():
    text = "Amazing victory royale in Fortnite"
    tokens = tokenize(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert "victory" in tokens
    assert "fortnite" in tokens


def test_get_extra_features():
    text = "I love this so much!!! Really?"
    feats = get_extra_features(text)
    assert len(feats) == 6
    # Compound score
    assert -1.0 <= feats[0] <= 1.0
    # Punctuation counts
    assert feats[4] == 3  # three exclamation marks
    assert feats[5] == 1  # one question mark


def test_clamp_negatives():
    arr = np.array([-0.5, 0.0, 0.75, -1.0, 2.0])
    clamped = clamp_negatives(arr)
    assert np.all(clamped >= 0)
    assert clamped[0] == 0.0
    assert clamped[2] == 0.75
    assert clamped[4] == 2.0
