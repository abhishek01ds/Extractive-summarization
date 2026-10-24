"""
src/baselines.py: Baseline summarization models for comparison with SGCSumm.
Implements:
1. TF-IDF Summarizer: Classic statistical baseline scoring sentences by term frequency-inverse document frequency.
2. Deep Learning Summarizer: Neural extractive summarizer using deep BERT contextual sentence embeddings and semantic centroid similarity with redundancy penalty (MMR).
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import Sentence


class TfidfSummarizer:
    """
    Classic Statistical Baseline Summarizer using TF-IDF.
    Scores each sentence by aggregating TF-IDF weights and selecting top-ranked
    sentences subject to a word or byte budget constraint, preserving chronological order.
    """
    def __init__(self, budget_type: str = "word", budget_limit: int = 100):
        self.budget_type = budget_type.lower()
        self.budget_limit = budget_limit

    def summarize(
        self,
        sentences: List[Sentence],
        min_sentences: int = 1,
        target_ratio: Optional[float] = None
    ) -> Tuple[List[Sentence], str, List[Dict[str, Any]]]:
        if not sentences:
            return [], "", []

        # Target ratio rule
        total_cost = sum(s.byte_length if self.budget_type == "byte" else s.word_count for s in sentences)
        if target_ratio is not None and 0.0 < target_ratio <= 1.0:
            budget = max(25, int(total_cost * target_ratio))
        else:
            budget = self.budget_limit

        min_sents_target = max(1, min(min_sentences, len(sentences)))
        raw_texts = [s.raw_text for s in sentences]

        # Handle very small documents or single sentence
        if len(sentences) == 1:
            return sentences, sentences[0].raw_text, [{"sent_id": 0, "score": 1.0, "cost": len(sentences[0].raw_text.split())}]

        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(raw_texts)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            lengths = np.array([max(1, len(s.clean_tokens)) for s in sentences])
            normalized_scores = sentence_scores / lengths
        except Exception:
            normalized_scores = np.array([len(s.raw_text.split()) for s in sentences], dtype=float)

        ranked_indices = np.argsort(-normalized_scores)

        selected_indices = []
        current_cost = 0

        for idx in ranked_indices:
            sent = sentences[idx]
            cost = sent.byte_length if self.budget_type == "byte" else sent.word_count
            if current_cost + cost <= budget or len(selected_indices) < min_sents_target:
                selected_indices.append(idx)
                current_cost += cost
                if current_cost >= budget and len(selected_indices) >= min_sents_target:
                    break

        # Restore chronological order
        selected_indices.sort()
        selected_sentences = [sentences[i] for i in selected_indices]
        summary_text = " ".join(s.raw_text for s in selected_sentences)

        logs = [
            {
                "sent_id": sentences[i].sent_id,
                "score": float(normalized_scores[i]),
                "cost": sentences[i].byte_length if self.budget_type == "byte" else sentences[i].word_count,
                "text": sentences[i].raw_text
            }
            for i in selected_indices
        ]

        return selected_sentences, summary_text, logs


class DeepLearningSummarizer:
    """
    Deep Learning Neural Extractive Baseline Summarizer.
    Leverages deep contextual BERT sentence representations. Computes document-level
    semantic centroid and scores sentences using cosine similarity with Maximal Marginal
    Relevance (MMR) redundancy control under budget constraints.
    """
    def __init__(
        self,
        embedder,
        budget_type: str = "word",
        budget_limit: int = 100,
        diversity_lambda: float = 0.7
    ):
        self.embedder = embedder
        self.budget_type = budget_type.lower()
        self.budget_limit = budget_limit
        self.diversity_lambda = diversity_lambda

    def summarize(
        self,
        sentences: List[Sentence],
        min_sentences: int = 1,
        target_ratio: Optional[float] = None
    ) -> Tuple[List[Sentence], str, List[Dict[str, Any]]]:
        if not sentences:
            return [], "", []

        # Target ratio rule
        total_cost = sum(s.byte_length if self.budget_type == "byte" else s.word_count for s in sentences)
        if target_ratio is not None and 0.0 < target_ratio <= 1.0:
            budget = max(25, int(total_cost * target_ratio))
        else:
            budget = self.budget_limit

        min_sents_target = max(1, min(min_sentences, len(sentences)))

        if len(sentences) == 1:
            return sentences, sentences[0].raw_text, [{"sent_id": 0, "score": 1.0, "cost": len(sentences[0].raw_text.split())}]

        # 1. Obtain deep contextual BERT embeddings (N x D)
        embeddings = self.embedder.embed_sentences(sentences)  # (N, D)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        normalized_embeddings = embeddings / norms

        # 2. Compute document semantic centroid
        centroid = np.mean(normalized_embeddings, axis=0, keepdims=True)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        # 3. Compute semantic relevance to centroid
        relevance_scores = (normalized_embeddings @ centroid.T).flatten()  # (N,)

        # 4. Pairwise similarity matrix for redundancy control
        pairwise_sim = normalized_embeddings @ normalized_embeddings.T

        # 5. Greedy selection with Maximal Marginal Relevance (MMR)
        selected_indices = []
        remaining_indices = list(range(len(sentences)))
        current_cost = 0
        logs = []

        while remaining_indices and (current_cost < budget or len(selected_indices) < min_sents_target):
            best_mmr_score = -float("inf")
            best_idx = None

            for idx in remaining_indices:
                rel = relevance_scores[idx]
                if not selected_indices:
                    mmr_score = rel
                else:
                    max_sim_to_selected = max(pairwise_sim[idx, s_idx] for s_idx in selected_indices)
                    mmr_score = self.diversity_lambda * rel - (1.0 - self.diversity_lambda) * max_sim_to_selected

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

            if best_idx is None:
                break

            candidate = sentences[best_idx]
            cost = candidate.byte_length if self.budget_type == "byte" else candidate.word_count

            if current_cost + cost <= budget or len(selected_indices) < min_sents_target:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                current_cost += cost
                logs.append({
                    "sent_id": candidate.sent_id,
                    "score": float(best_mmr_score),
                    "relevance": float(relevance_scores[best_idx]),
                    "cost": cost,
                    "text": candidate.raw_text
                })
                if current_cost >= budget and len(selected_indices) >= min_sents_target:
                    break
            else:
                remaining_indices.remove(best_idx)

        # Restore chronological order
        selected_indices.sort()
        selected_sentences = [sentences[i] for i in selected_indices]
        summary_text = " ".join(s.raw_text for s in selected_sentences)

        return selected_sentences, summary_text, logs
