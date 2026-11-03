"""
Unit tests for Information Retrieval evaluation metrics using hand-computed ground truth examples.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from ir.evaluate_ir import (
    precision_at_k,
    recall_at_k,
    average_precision,
    reciprocal_rank,
    ndcg_at_k,
    compute_interpolated_pr
)


def test_hand_computed_precision_and_recall():
    """
    Retrieved: [10, 20, 30, 40, 50]
    Relevant in corpus: {10, 30, 60} (3 relevant docs)
    Hits: doc 10 (rank 1), doc 30 (rank 3)
    """
    retrieved = [10, 20, 30, 40, 50]
    relevant = {10, 30, 60}

    assert np.isclose(precision_at_k(retrieved, relevant, 1), 1.0)
    assert np.isclose(precision_at_k(retrieved, relevant, 2), 0.5)
    assert np.isclose(precision_at_k(retrieved, relevant, 3), 2.0 / 3.0)
    assert np.isclose(precision_at_k(retrieved, relevant, 5), 2.0 / 5.0)

    assert np.isclose(recall_at_k(retrieved, relevant, 1), 1.0 / 3.0)
    assert np.isclose(recall_at_k(retrieved, relevant, 3), 2.0 / 3.0)
    assert np.isclose(recall_at_k(retrieved, relevant, 5), 2.0 / 3.0)


def test_hand_computed_average_precision():
    """
    Retrieved: [10, 20, 30, 40, 50]
    Relevant: {10, 30, 60}
    P@1 = 1/1 = 1.0
    P@3 = 2/3 ≈ 0.6667
    AP = (1.0 + 2/3) / 3 = (5/3) / 3 = 5/9 ≈ 0.555556
    """
    retrieved = [10, 20, 30, 40, 50]
    relevant = {10, 30, 60}

    ap = average_precision(retrieved, relevant, max_depth=5)
    expected_ap = 5.0 / 9.0
    assert np.isclose(ap, expected_ap, atol=1e-5)


def test_hand_computed_reciprocal_rank():
    # First relevant at rank 1
    assert np.isclose(reciprocal_rank([10, 20, 30], {10}), 1.0)
    # First relevant at rank 2
    assert np.isclose(reciprocal_rank([20, 10, 30], {10}), 0.5)
    # First relevant at rank 3
    assert np.isclose(reciprocal_rank([20, 30, 10], {10}), 1.0 / 3.0)
    # No relevant found
    assert np.isclose(reciprocal_rank([20, 30, 40], {10}), 0.0)


def test_hand_computed_ndcg():
    """
    Retrieved: [10, 20, 30, 40, 50]
    Relevant: {10, 30, 60}
    DCG@5 = (2^1 - 1)/log2(2) + 0 + (2^1 - 1)/log2(4) + 0 + 0 = 1.0 + 0.5 = 1.5
    IDCG@5 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.63092975 + 0.5 = 2.13092975
    nDCG@5 = 1.5 / 2.13092975 ≈ 0.703918
    """
    retrieved = [10, 20, 30, 40, 50]
    relevant = {10, 30, 60}

    ndcg5 = ndcg_at_k(retrieved, relevant, k=5)
    expected_dcg = 1.0 + 0.5
    expected_idcg = 1.0 + (1.0 / np.log2(3.0)) + 0.5
    expected_ndcg = expected_dcg / expected_idcg
    assert np.isclose(ndcg5, expected_ndcg, atol=1e-4)


def test_interpolated_pr_monotonicity():
    """
    Interpolated precision must be monotonically non-increasing as recall increases.
    """
    retrieved = [10, 20, 30, 40, 50]
    relevant = {10, 30, 60}
    recall_levels = np.linspace(0.0, 1.0, 11)

    pr_values = compute_interpolated_pr(retrieved, relevant, recall_levels)
    assert len(pr_values) == 11
    # Verify non-increasing
    for i in range(len(pr_values) - 1):
        assert pr_values[i] >= pr_values[i + 1]
