"""
src/dataset.py: Dataset loading, abstractive-to-extractive conversion, and data augmentation.
Implements:
1. DocumentCluster data structure representing a multi-document topic cluster.
2. Conversion of abstractive reference summaries to extractive targets using greedy ROUGE-1
   maximization (Nallapati et al., 2017; Algorithm 4).
3. Data augmentation via random sentence addition/removal to expand training patterns (Section 3.4 & 4.3).
4. Loading DUC 2004, DUC 2003, and CNN/DailyMail formatted directories or JSON files.
5. Built-in smoke-test dataset generator for instant offline CPU validation.
"""

from dataclasses import dataclass, field
import json
import os
import random
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from rouge_score import rouge_scorer

from src.preprocessing import Sentence, TextPreprocessor
from src.utils import setup_logger

logger = setup_logger("SGCSumm.Dataset")


@dataclass
class DocumentCluster:
    """
    Represents a collection of related documents D = {d_1, ..., d_p} on a single topic,
    along with gold reference summaries.
    """
    cluster_id: str
    documents: Dict[str, str]  # {doc_id: raw_text}
    reference_summaries: List[str]  # List of abstractive/extractive reference summary texts
    sentences: List[Sentence] = field(default_factory=list)
    extractive_target_indices: List[int] = field(default_factory=list)


def compute_rouge1_recall(
    candidate_tokens: List[str],
    reference_tokens: List[str]
) -> float:
    """
    Compute ROUGE-1 recall between candidate token list and reference token list.
    f_{ROUGE-1} is monotone non-decreasing and submodular (Section 3.4).
    """
    if not reference_tokens or not candidate_tokens:
        return 0.0
        
    cand_counts = {}
    for tok in candidate_tokens:
        cand_counts[tok] = cand_counts.get(tok, 0) + 1
        
    ref_counts = {}
    for tok in reference_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
        
    # Recall matching
    matched = 0
    for tok, ref_c in ref_counts.items():
        matched += min(ref_c, cand_counts.get(tok, 0))
        
    return float(matched / len(reference_tokens))


def convert_to_extractive(
    sentences: List[Sentence],
    reference_text: str,
    preprocessor: TextPreprocessor,
    max_sentences: int = 4
) -> List[int]:
    """
    Greedy conversion from abstractive gold summary to extractive target (Nallapati et al., 2017).
    Iteratively adds the sentence that yields the highest marginal gain in ROUGE-1 recall.
    
    Args:
        sentences: Pool of candidate sentences in the document cluster.
        reference_text: Gold human reference summary.
        preprocessor: TextPreprocessor for tokenizing reference.
        max_sentences: Max sentences in converted extractive summary.
        
    Returns:
        List of selected sentence indices (within sentences list).
    """
    if not sentences or not reference_text.strip():
        return []
        
    _, ref_tokens = preprocessor.process_sentence_tokens(reference_text)
    if not ref_tokens:
        return []

    selected_indices: List[int] = []
    current_tokens: List[str] = []
    current_score = 0.0

    for _ in range(min(max_sentences, len(sentences))):
        best_gain = 0.0
        best_idx = -1

        for idx, s in enumerate(sentences):
            if idx in selected_indices:
                continue
                
            candidate_tokens = current_tokens + s.stemmed_tokens
            score = compute_rouge1_recall(candidate_tokens, ref_tokens)
            gain = score - current_score

            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx != -1 and best_gain > 0.001:
            selected_indices.append(best_idx)
            current_tokens.extend(sentences[best_idx].stemmed_tokens)
            current_score += best_gain
        else:
            break

    return selected_indices


def augment_cluster_samples(
    cluster: DocumentCluster,
    preprocessor: TextPreprocessor,
    num_augmentations: int = 5,
    seed: int = 42
) -> List[Tuple[List[int], float]]:
    """
    Data augmentation procedure (Section 3.4 & 4.3).
    Expands base training patterns by adding and removing sentences to/from the extractive target.
    
    Returns:
        List of tuples: (subset_sentence_indices, target_rouge1_score)
    """
    rng = random.Random(seed)
    N = len(cluster.sentences)
    if N == 0:
        return []

    ref_text = cluster.reference_summaries[0] if cluster.reference_summaries else ""
    _, ref_tokens = preprocessor.process_sentence_tokens(ref_text)
    
    base_indices = cluster.extractive_target_indices
    if not base_indices:
        base_indices = [0] if N > 0 else []

    augmented_samples: List[Tuple[List[int], float]] = []

    # 1. Base extractive summary
    base_toks = [tok for idx in base_indices for tok in cluster.sentences[idx].stemmed_tokens]
    base_score = compute_rouge1_recall(base_toks, ref_tokens)
    augmented_samples.append((list(base_indices), base_score))

    all_indices = set(range(N))

    # 2. Add random sentence(s)
    available_to_add = list(all_indices - set(base_indices))
    for _ in range(num_augmentations):
        sample_indices = list(base_indices)
        
        # Decide action: add, remove, or swap
        action = rng.choice(["add", "remove", "random"])
        
        if action == "add" and available_to_add:
            add_idx = rng.choice(available_to_add)
            sample_indices.append(add_idx)
        elif action == "remove" and len(sample_indices) > 1:
            rem_idx = rng.choice(sample_indices)
            sample_indices.remove(rem_idx)
        else:
            # Random subset of 1 to 4 sentences
            k = rng.randint(1, min(4, N))
            sample_indices = rng.sample(range(N), k)
            
        sample_toks = [tok for idx in sample_indices for tok in cluster.sentences[idx].stemmed_tokens]
        score = compute_rouge1_recall(sample_toks, ref_tokens)
        augmented_samples.append((sample_indices, score))

    return augmented_samples


def create_smoke_test_dataset(output_dir: str = "./data/sample_duc2004") -> str:
    """
    Create a clean multi-document smoke-test dataset simulating DUC 2004 topic clusters.
    Enables immediate offline validation and unit testing.
    """
    os.makedirs(output_dir, exist_ok=True)
    sample_clusters = [
        {
            "cluster_id": "d30001t",
            "topic": "Cambodian Political Crisis and Elections",
            "documents": {
                "FBIS_1": "Cambodian political leaders gathered in Phnom Penh for a special summit. King Norodom Sihanouk urged rival parties to form a coalition government to stabilize the nation. Prime Minister Hun Sen stated his party is open to negotiations.",
                "FBIS_2": "The opposition party leader Prince Norodom Ranariddh agreed to participate in talks. International observers reported the election was generally peaceful despite minor voting irregularities in rural provinces.",
                "APW_1": "Talks on forming Cambodia's new government made preliminary progress on Monday. Both coalition partners pledged to respect the democratic constitution and restore economic stability."
            },
            "reference_summaries": [
                "Cambodian political rivals met in Phnom Penh under King Norodom Sihanouk to negotiate a coalition government following recent national elections. Leaders Hun Sen and Prince Ranariddh agreed to constitutional talks to restore political stability."
            ]
        },
        {
            "cluster_id": "d30002t",
            "topic": "Hurricane Mitch Central America Disaster",
            "documents": {
                "FBIS_3": "Hurricane Mitch battered Central America with torrential rains and destructive mudslides. Relief workers struggled to reach isolated mountain communities cut off by collapsed bridges.",
                "FBIS_4": "In Honduras and Nicaragua, emergency officials declared a state of catastrophe. The United Nations appealed for global humanitarian aid to provide clean water and medical supplies.",
                "APW_2": "International cargo planes delivered food rations and temporary shelters. Governments around the world pledged millions in emergency disaster relief funds."
            },
            "reference_summaries": [
                "Hurricane Mitch triggered catastrophic flooding and mudslides across Honduras and Nicaragua. The disaster prompted international emergency rescue operations and United Nations aid appeals."
            ]
        }
    ]

    cluster_paths = []
    for cluster_data in sample_clusters:
        filepath = os.path.join(output_dir, f"{cluster_data['cluster_id']}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cluster_data, f, indent=2)
        cluster_paths.append(filepath)

    logger.info(f"Created {len(sample_clusters)} sample smoke-test clusters in {output_dir}")
    return output_dir


def load_dataset_from_dir(
    data_dir: str,
    preprocessor: Optional[TextPreprocessor] = None
) -> List[DocumentCluster]:
    """
    Load document clusters from directory containing JSON cluster files.
    Preprocesses all documents and converts abstractive gold summaries to extractive targets.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    if preprocessor is None:
        preprocessor = TextPreprocessor()

    clusters: List[DocumentCluster] = []
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    
    if not json_files:
        logger.warning(f"No JSON files found in {data_dir}.")
        return []

    for fname in sorted(json_files):
        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        cluster_id = data.get("cluster_id", os.path.splitext(fname)[0])
        documents = data.get("documents", {})
        references = data.get("reference_summaries", [])

        # Preprocess documents into sentences
        sentences = preprocessor.preprocess_documents(documents)
        
        # Convert abstractive gold summary to extractive target
        ref_text = " ".join(references) if references else ""
        ext_indices = convert_to_extractive(sentences, ref_text, preprocessor)

        cluster = DocumentCluster(
            cluster_id=cluster_id,
            documents=documents,
            reference_summaries=references,
            sentences=sentences,
            extractive_target_indices=ext_indices
        )
        clusters.append(cluster)

    logger.info(f"Loaded {len(clusters)} document clusters from {data_dir}")
    return clusters
