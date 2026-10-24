"""
src/metrics.py: Evaluation metrics for text summarization.
Implements:
1. Standard ROUGE-1, ROUGE-2, and ROUGE-L metrics via rouge-score.
2. Recall, Precision, and F1 computation.
3. Multi-reference summary evaluation.
4. Baseline comparison tables matching Paper Tables 6 and 7.
"""

from typing import List, Dict, Union, Optional
from rouge_score import rouge_scorer


# Published benchmark results from Paper Table 6 (DUC 2004) and Table 7 (DailyMail)
PUBLISHED_PAPER_RESULTS_DUC2004 = {
    "SGCSumm (Proposed)": {"rouge1": 40.66, "rouge2": 10.41},
    "DSNSum (Ghadimi & Beigy, 2020)": {"rouge1": 40.28, "rouge2": 10.23},
    "DPP (Kulesza & Taskar, 2012)": {"rouge1": 39.79, "rouge2": 9.62},
    "Submodular (Lin & Bilmes, 2011)": {"rouge1": 39.18, "rouge2": 9.35},
    "Centroid + G + new-tf-idf": {"rouge1": 39.11, "rouge2": 9.81},
    "R2N2-ILP (Cao et al., 2015)": {"rouge1": 38.78, "rouge2": 9.86},
    "Centroid + G": {"rouge1": 38.55, "rouge2": 9.73},
    "ICSISumm (Gillick et al., 2009)": {"rouge1": 38.41, "rouge2": 9.78},
    "GRU+GCN (Yasunaga et al., 2017)": {"rouge1": 38.23, "rouge2": 9.48},
    "LexRank (Erkan & Radev, 2004)": {"rouge1": 37.86, "rouge2": 9.67},
}

PUBLISHED_PAPER_RESULTS_DAILYMAIL = {
    "SGCSumm (Proposed)": {"rouge1": 38.893, "rouge2": 19.113, "rougeL": 21.351},
    "DSNSum (Ghadimi & Beigy, 2020)": {"rouge1": 38.655, "rouge2": 18.966, "rougeL": 21.127},
    "SummaRuNNer (Nallapati et al., 2017)": {"rouge1": 39.6, "rouge2": 16.2, "rougeL": 35.3},
    "REFRESH (Narayan et al., 2018)": {"rouge1": 40.0, "rouge2": 18.2, "rougeL": 36.6},
}


class SummarizationMetrics:
    """Computes ROUGE metrics for system vs reference summaries."""
    def __init__(self, metrics: Optional[List[str]] = None):
        if metrics is None:
            metrics = ["rouge1", "rouge2", "rougeL"]
        self.metrics = metrics
        self.scorer = rouge_scorer.RougeScorer(self.metrics, use_stemmer=True)

    def evaluate_pair(
        self,
        system_summary: str,
        reference_summaries: Union[str, List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a generated system summary against one or more reference summaries.
        If multiple references are provided, takes the maximum score across references (standard DUC).
        
        Returns:
            Dict: {metric_name: {'precision': p, 'recall': r, 'fmeasure': f}} (multiplied by 100)
        """
        if isinstance(reference_summaries, str):
            reference_summaries = [reference_summaries]

        if not reference_summaries or not system_summary.strip():
            return {
                m: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
                for m in self.metrics
            }

        best_scores = {
            m: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
            for m in self.metrics
        }

        for ref in reference_summaries:
            if not ref.strip():
                continue
            scores = self.scorer.score(ref, system_summary)
            for m in self.metrics:
                p = scores[m].precision * 100.0
                r = scores[m].recall * 100.0
                f = scores[m].fmeasure * 100.0
                if f > best_scores[m]["fmeasure"]:
                    best_scores[m] = {"precision": p, "recall": r, "fmeasure": f}

        return best_scores

    def evaluate_dataset(
        self,
        system_summaries: List[str],
        references_list: List[List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a complete corpus/dataset and compute micro/macro average ROUGE scores.
        """
        if len(system_summaries) != len(references_list):
            raise ValueError("Number of system summaries must match number of references lists.")

        n = len(system_summaries)
        if n == 0:
            return {m: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} for m in self.metrics}

        totals = {m: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} for m in self.metrics}

        for sys_sum, refs in zip(system_summaries, references_list):
            pair_score = self.evaluate_pair(sys_sum, refs)
            for m in self.metrics:
                totals[m]["precision"] += pair_score[m]["precision"]
                totals[m]["recall"] += pair_score[m]["recall"]
                totals[m]["fmeasure"] += pair_score[m]["fmeasure"]

        avg_scores = {
            m: {
                "precision": round(totals[m]["precision"] / n, 3),
                "recall": round(totals[m]["recall"] / n, 3),
                "fmeasure": round(totals[m]["fmeasure"] / n, 3),
            }
            for m in self.metrics
        }
        return avg_scores
