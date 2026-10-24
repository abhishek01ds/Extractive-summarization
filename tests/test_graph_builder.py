"""
tests/test_graph_builder.py: Unit tests for hyperspherical coordinate transformation,
modular mapping, sentence similarity, graph centrality labeling, and fixed-size tensor extraction.
"""

import numpy as np
import pytest

from src.bert_embeddings import (
    cartesian_to_hyperspherical,
    eliminate_first_component,
    modular_mapping_F
)
from src.graph_builder import (
    compute_pairwise_similarity,
    SentenceGraphBuilder
)


def test_hyperspherical_transformation_and_non_negativity():
    # 5 sentences with embedding dimension m = 10
    rng = np.random.RandomState(42)
    A = rng.randn(5, 10).astype(np.float32)

    # T1: Convert to hyperspherical coordinates
    B = cartesian_to_hyperspherical(A)
    assert B.shape == (5, 10)
    # Radial component is norm and all angular components must be >= 0
    assert (B >= 0.0).all(), "Hyperspherical coordinates must be strictly non-negative"

    # T2: Eliminate radial magnitude
    B_prime = eliminate_first_component(B)
    assert B_prime.shape == (5, 9)
    assert (B_prime >= 0.0).all()

    # Modular mapping F
    C = modular_mapping_F(B_prime)
    assert C.shape == (5, 9)
    assert (C >= 0.0).all()


def test_pairwise_similarity_matrix():
    # Construct 4 sample sentence vectors in R^8
    rng = np.random.RandomState(42)
    C = rng.uniform(0.1, 1.5, size=(4, 8)).astype(np.float32)

    S = compute_pairwise_similarity(C)
    assert S.shape == (4, 4)
    # Self-similarity must be exactly 1.0 (Eq. 4: cos(0) = 1)
    np.testing.assert_allclose(np.diag(S), 1.0, atol=1e-5)
    # Symmetry: S[i, j] == S[j, i]
    np.testing.assert_allclose(S, S.T, atol=1e-5)
    # Bounded in [-1, 1]
    assert (S >= -1.0 - 1e-5).all() and (S <= 1.0 + 1e-5).all()


def test_top_vertices_and_neighborhoods_selection():
    builder = SentenceGraphBuilder(top_n=3, top_k=4)
    
    # 6 sentences with similarity matrix
    rng = np.random.RandomState(123)
    C = rng.uniform(0.1, 2.0, size=(6, 16)).astype(np.float32)
    S = compute_pairwise_similarity(C)

    top_n_verts, neighborhoods = builder.select_top_vertices_and_neighbors(S)

    assert len(top_n_verts) == 3
    assert len(neighborhoods) == 3
    for neigh in neighborhoods:
        assert len(neigh) == 4
        # First neighbor should be the vertex itself (similarity 1.0)
        assert neigh[0] in range(6)


def test_graph_tensor_dimensions_and_padding_for_small_clusters():
    builder = SentenceGraphBuilder(top_n=15, top_k=20)
    
    # Tiny cluster with only 2 sentences
    C = np.array([
        [0.5, 0.8, 1.2, 0.3],
        [0.4, 0.7, 1.1, 0.2]
    ], dtype=np.float32)
    S = compute_pairwise_similarity(C)

    top_n_verts, neighborhoods = builder.select_top_vertices_and_neighbors(S)
    X = builder.construct_convpool_input(C, neighborhoods)

    # Tensor X must have shape (m - 1, n, k) = (4, 15, 20)
    assert X.shape == (4, 15, 20)
    assert (X >= 0.0).all(), "Input tensor X must be strictly non-negative"
