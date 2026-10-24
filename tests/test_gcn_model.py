"""
tests/test_gcn_model.py: Unit tests for GCN representation learning architecture,
tensor transformations, weight non-negativity, and output properties.
"""

import pytest
import torch
from src.gcn_model import GCNRepresentationLearner


def test_gcn_input_output_shapes():
    # Parameters: m - 1 = 64, n = 15, k = 20
    num_channels = 64
    num_vertices = 15
    num_neighbors = 20
    batch_size = 4

    gcn = GCNRepresentationLearner(
        num_channels=num_channels,
        num_vertices=num_vertices,
        num_neighbors=num_neighbors
    )

    # 4D Batched input tensor X
    x_batched = torch.rand(batch_size, num_channels, num_vertices, num_neighbors)
    y_batched = gcn(x_batched)
    assert y_batched.shape == (batch_size, num_channels)

    # 3D Unbatched input tensor X
    x_single = torch.rand(num_channels, num_vertices, num_neighbors)
    y_single = gcn(x_single)
    assert y_single.shape == (num_channels,)


def test_gcn_non_negative_weights_and_output():
    num_channels = 16
    num_vertices = 5
    num_neighbors = 6

    gcn = GCNRepresentationLearner(
        num_channels=num_channels,
        num_vertices=num_vertices,
        num_neighbors=num_neighbors
    )

    # Test weight projection
    with torch.no_grad():
        gcn.conv1.weight[0, 0, 0, 0] = -0.5
        gcn.conv2.weight[0, 0, 0, 0] = -0.8
    gcn.project_non_negative()

    assert (gcn.conv1.weight >= 0.0).all(), "Conv1 weights must be strictly non-negative"
    assert (gcn.conv2.weight >= 0.0).all(), "Conv2 weights must be strictly non-negative"

    # Forward pass on non-negative input
    x = torch.rand(num_channels, num_vertices, num_neighbors)
    y = gcn(x)

    # By Theorem 3: Y must be non-negative
    assert (y >= 0.0).all(), "Representation output Y must satisfy non-negativity"
    assert not torch.isnan(y).any()
