from __future__ import annotations

import numpy as np
import torch

from scripts.collect_tsp100_pair_laplacian_descriptors import (
    PairContext,
    _matrix_and_drift,
    _retrieval_metrics,
)


def _two_node_context() -> PairContext:
    return PairContext(
        full_batch={},
        instance=torch.tensor([0]),
        winner=torch.tensor([0]),
        loser=torch.tensor([1]),
        winner_rank=torch.tensor([0]),
        loser_rank=torch.tensor([1]),
        batch_size=1,
        num_nodes=2,
    )


def test_laplacian_is_psd_and_translation_invariant() -> None:
    context = _two_node_context()
    matrix, drift = _matrix_and_drift(
        context,
        conductance=torch.tensor([4.0]),
        grad_w=torch.tensor([-2.0]),
        grad_l=torch.tensor([2.0]),
    )
    assert np.allclose(matrix @ np.ones(2), 0.0, atol=1e-12)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-12
    assert np.allclose(drift, np.array([-1.0, 1.0]) / np.sqrt(2.0))


def test_matrix_loses_direction_but_drift_retains_it() -> None:
    context = _two_node_context()
    matrix_a, drift_a = _matrix_and_drift(
        context,
        conductance=torch.tensor([1.0]),
        grad_w=torch.tensor([-1.0]),
        grad_l=torch.tensor([1.0]),
    )
    matrix_b, drift_b = _matrix_and_drift(
        context,
        conductance=torch.tensor([1.0]),
        grad_w=torch.tensor([1.0]),
        grad_l=torch.tensor([-1.0]),
    )
    assert np.allclose(matrix_a, matrix_b)
    assert np.allclose(drift_a, -drift_b)


def test_cross_seed_retrieval_prefers_repeated_descriptor() -> None:
    features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.99, 0.01],
            [0.01, 0.99],
        ]
    )
    metrics = _retrieval_metrics(
        features,
        candidate_ids=["a", "b", "a", "b"],
        seeds=[1, 1, 2, 2],
    )
    assert metrics["top1"] == 1.0
    assert metrics["median_rank"] == 1.0
