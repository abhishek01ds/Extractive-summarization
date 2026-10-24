from typing import List, Dict, Optional, Tuple
import numpy as np
import torch

from src.preprocessing import Sentence
from src.bert_embeddings import BERTSentenceEmbedder
from src.graph_builder import SentenceGraphBuilder, compute_pairwise_similarity
from src.dsn_model import SGCSummModel
from src.utils import setup_logger

logger = setup_logger("SGCSumm.SummaryGenerator")


class GreedySummaryGenerator:
    """
    Selects summary sentences step-by-step using greedy selection under a budget.
    """
    def __init__(
        self,
        model: SGCSummModel,
        embedder: BERTSentenceEmbedder,
        graph_builder: SentenceGraphBuilder,
        budget_type: str = "byte",
        budget_limit: int = 665,
        restore_order: bool = True,
        device: str = "cpu"
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.embedder = embedder
        self.graph_builder = graph_builder
        self.budget_type = budget_type.lower()
        self.budget_limit = budget_limit
        self.restore_order = restore_order

    def get_sentence_cost(self, sentence: Sentence) -> int:
        if self.budget_type == "byte":
            return sentence.byte_length
        return sentence.word_count

    def compute_subset_score(
        self,
        C_features: np.ndarray,
        S_sim: np.ndarray,
        subset_indices: List[int]
    ) -> float:
        if not subset_indices:
            return 0.0

        sub_C = C_features[subset_indices]
        sub_S = S_sim[np.ix_(subset_indices, subset_indices)]

        # Extract neighbor tensor for this subset
        _, neighborhoods = self.graph_builder.select_top_vertices_and_neighbors(sub_S)
        X_np = self.graph_builder.construct_convpool_input(sub_C, neighborhoods)
        
        X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            score, _ = self.model(X_tensor)
            return float(score.item())

    def generate_summary(
        self,
        sentences: List[Sentence],
        max_budget: Optional[int] = None,
        min_sentences: int = 1,
        target_ratio: Optional[float] = None
    ) -> Tuple[List[Sentence], str, List[Dict[str, float]]]:
        if not sentences:
            return [], "", []

        # Target length calculation (e.g. 30% of total words)
        total_cost = sum(self.get_sentence_cost(s) for s in sentences)
        if target_ratio is not None and 0.0 < target_ratio <= 1.0:
            budget = max(25, int(total_cost * target_ratio))
        else:
            budget = max_budget if max_budget is not None else self.budget_limit
        
        min_sents_target = max(1, min(min_sentences, len(sentences)))

        # Extract sentence features and similarity matrix
        C = self.embedder.extract_suitable_features(sentences)
        S = compute_pairwise_similarity(C)

        # Precompute centrality scores as fallback
        G_full = self.graph_builder.build_graph(S)
        labeling_scores = self.graph_builder.compute_labeling_scores(G_full, S)

        selected_indices: List[int] = []
        current_budget_used = 0
        current_score = 0.0
        selection_logs: List[Dict[str, float]] = []

        pool = list(range(len(sentences)))

        while pool and (current_budget_used < budget or len(selected_indices) < min_sents_target):
            best_gain = -float("inf")
            best_cand = None
            best_cand_score = current_score
            best_cand_cost = 0

            # Find candidate sentence that gives the highest score improvement
            for cand_idx in pool:
                cand_sentence = sentences[cand_idx]
                cand_cost = self.get_sentence_cost(cand_sentence)

                if current_budget_used + cand_cost > budget and len(selected_indices) >= min_sents_target:
                    continue

                candidate_subset = selected_indices + [cand_idx]
                new_score = self.compute_subset_score(C, S, candidate_subset)
                gain = new_score - current_score

                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand_idx
                    best_cand_score = new_score
                    best_cand_cost = cand_cost

            # If neural score has positive gain, pick it
            if best_cand is not None and best_gain > 1e-5:
                chosen_cand = best_cand
                chosen_gain = best_gain
                chosen_score = best_cand_score
                chosen_cost = best_cand_cost
            else:
                # Fallback to centrality ranking if score saturates
                affordable_cands = [
                    idx for idx in pool
                    if (current_budget_used + self.get_sentence_cost(sentences[idx]) <= budget)
                    or (len(selected_indices) < min_sents_target)
                ]
                if not affordable_cands:
                    break

                chosen_cand = max(affordable_cands, key=lambda idx: labeling_scores[idx])
                chosen_cost = self.get_sentence_cost(sentences[chosen_cand])
                chosen_gain = max(0.0, best_gain)
                chosen_score = max(current_score, best_cand_score)

            selected_indices.append(chosen_cand)
            pool.remove(chosen_cand)
            current_budget_used += chosen_cost
            current_score = chosen_score

            selection_logs.append({
                "selected_global_id": sentences[chosen_cand].global_id,
                "marginal_gain": chosen_gain,
                "cumulative_score": current_score,
                "cost": chosen_cost,
                "total_budget_used": current_budget_used
            })

        # Restore original document order
        selected_sentences = [sentences[idx] for idx in selected_indices]
        if self.restore_order:
            selected_sentences.sort(key=lambda s: (s.doc_id, s.sent_id, s.global_id))

        summary_text = " ".join([s.raw_text for s in selected_sentences])
        return selected_sentences, summary_text, selection_logs
