from typing import List, Tuple, Dict, Optional
import numpy as np
import networkx as nx

from src.preprocessing import Sentence
from src.utils import setup_logger

logger = setup_logger("SGCSumm.GraphBuilder")


def compute_pairwise_similarity(C: np.ndarray) -> np.ndarray:
    """
    Computes pairwise cosine similarity between sentence angle features.
    """
    N, feat_dim = C.shape
    if N == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if N == 1:
        return np.ones((1, 1), dtype=np.float32)

    # Average cosine difference across all feature dimensions
    diff = C[:, None, :] - C[None, :, :]
    S = np.mean(np.cos(diff), axis=-1).astype(np.float32)
    
    # Diagonal is always 1.0 (sentence with itself)
    np.fill_diagonal(S, 1.0)
    S = 0.5 * (S + S.T)
    return S


class SentenceGraphBuilder:
    """
    Builds a sentence graph and extracts top vertices and their neighbor tensors.
    """
    def __init__(
        self,
        top_n: int = 15,
        top_k: int = 20,
        similarity_threshold: float = 0.0
    ):
        self.top_n = top_n
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def build_graph(self, similarity_matrix: np.ndarray) -> nx.Graph:
        """
        Builds a NetworkX graph where edges connect similar sentences.
        """
        N = similarity_matrix.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(N))
        
        for i in range(N):
            for j in range(i + 1, N):
                sim = float(similarity_matrix[i, j])
                if sim > self.similarity_threshold:
                    dist = max(1e-5, 1.0 - sim)
                    G.add_edge(i, j, weight=sim, distance=dist)
                    
        return G

    def compute_labeling_scores(
        self,
        G: nx.Graph,
        similarity_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculates centrality score (degree + closeness) for each sentence.
        """
        N = len(G.nodes)
        if N <= 1:
            return np.ones(N, dtype=np.float32)

        # 1. Degree centrality
        degrees = np.array([G.degree(v) for v in range(N)], dtype=np.float32)
        max_deg = np.max(degrees)
        if max_deg > 0:
            deg_centrality = degrees / max_deg
        else:
            deg_centrality = np.zeros(N, dtype=np.float32)

        # 2. Closeness centrality
        closeness = np.zeros(N, dtype=np.float32)
        for v in range(N):
            lengths = nx.single_source_dijkstra_path_length(G, v, weight="distance")
            reachable = [dist for u, dist in lengths.items() if u != v]
            if reachable:
                total_dist = sum(reachable)
                num_reachable = len(reachable)
                if total_dist > 0:
                    closeness[v] = (num_reachable / total_dist) * (num_reachable / (N - 1))
            else:
                closeness[v] = 0.0

        scores = deg_centrality + closeness
        return scores.astype(np.float32)

    def select_top_vertices_and_neighbors(
        self,
        similarity_matrix: np.ndarray
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Selects top n central sentences and their top k closest neighbors.
        """
        N = similarity_matrix.shape[0]
        if N == 0:
            return [], []

        G = self.build_graph(similarity_matrix)
        l_scores = self.compute_labeling_scores(G, similarity_matrix)

        ranked_vertices = np.argsort(-l_scores).tolist()
        
        # Pick top n sentences (pad if fewer available)
        if len(ranked_vertices) >= self.top_n:
            top_n_vertices = ranked_vertices[:self.top_n]
        else:
            top_n_vertices = ranked_vertices.copy()
            while len(top_n_vertices) < self.top_n:
                top_n_vertices.append(ranked_vertices[len(top_n_vertices) % len(ranked_vertices)])

        # Pick top k neighbors for each selected sentence
        neighborhoods: List[List[int]] = []
        for v in top_n_vertices:
            sims = similarity_matrix[v].copy()
            neighbor_ranks = np.argsort(-sims).tolist()
            
            if v in neighbor_ranks:
                neighbor_ranks.remove(v)
            chosen_k = [v] + neighbor_ranks

            if len(chosen_k) >= self.top_k:
                chosen_k = chosen_k[:self.top_k]
            else:
                while len(chosen_k) < self.top_k:
                    chosen_k.append(chosen_k[len(chosen_k) % len(chosen_k)])
                    
            neighborhoods.append(chosen_k)

        return top_n_vertices, neighborhoods

    def construct_convpool_input(
        self,
        C: np.ndarray,
        neighborhoods: List[List[int]]
    ) -> np.ndarray:
        """
        Prepares the 3D tensor (features, n vertices, k neighbors) for GCN input.
        """
        m_minus_1 = C.shape[1]
        n = self.top_n
        k = self.top_k
        
        X = np.zeros((m_minus_1, n, k), dtype=np.float32)
        
        if len(neighborhoods) == 0:
            return X

        for y in range(n):
            neighbor_indices = neighborhoods[y]
            neighbor_features = C[neighbor_indices, :]
            X[:, y, :] = neighbor_features.T

        return X
