"""
tests/test_dsn_model.py: Unit tests for Deep Submodular Network (DSN) constraints,
output bounds, monotonicity, and unified SGCSummModel forward pass.
"""

import pytest
import torch
from src.dsn_model import DeepSubmodularNetwork, SGCSummModel


def test_dsn_architecture_and_output_range():
    input_dim = 64
    hidden_dim = 16
    dsn = DeepSubmodularNetwork(input_dim=input_dim, hidden_dim=hidden_dim)

    # Valid non-negative input
    Y = torch.rand(input_dim)
    score = dsn(Y)

    # Score must be scalar in [0, 1]
    assert 0.0 <= score.item() <= 1.0, "DSN score must lie in [0, 1]"

    # Batched forward pass
    Y_batch = torch.rand(8, input_dim)
    scores_batch = dsn(Y_batch)
    assert scores_batch.shape == (8,)
    assert (scores_batch >= 0.0).all() and (scores_batch <= 1.0).all()


def test_dsn_input_validation():
    dsn = DeepSubmodularNetwork(input_dim=32, hidden_dim=16)

    # Wrong shape
    invalid_dim = torch.rand(10)
    score = dsn(invalid_dim)
    assert score.item() == -1.0, "Invalid dimension must return -1.0 according to Algorithm 2"

    # Negative inputs
    invalid_neg = torch.tensor([-0.5] + [0.1] * 31)
    score = dsn(invalid_neg)
    assert score.item() == -1.0, "Negative inputs must fail validation"


def test_dsn_non_decreasing_monotonicity():
    # If Y1 <= Y2 elementwise, then DSN(Y1) <= DSN(Y2) (Remark 2)
    input_dim = 32
    dsn = DeepSubmodularNetwork(input_dim=input_dim, hidden_dim=16)
    dsn.project_non_negative()

    Y1 = torch.rand(input_dim)
    # Y2 is strictly greater than or equal to Y1
    delta = torch.rand(input_dim) * 0.5
    Y2 = Y1 + delta

    score1 = dsn(Y1).item()
    score2 = dsn(Y2).item()

    assert score2 >= score1 - 1e-6, f"DSN must be non-decreasing monotone: {score1} vs {score2}"


def test_unified_sgcsumm_model():
    model = SGCSummModel(
        num_channels=32,
        num_vertices=10,
        num_neighbors=12,
        hidden_dim=16
    )

    X = torch.rand(4, 32, 10, 12)
    score, Y = model(X)

    assert score.shape == (4,)
    assert Y.shape == (4, 32)
    assert (score >= 0.0).all() and (score <= 1.0).all()
    assert (Y >= 0.0).all()
