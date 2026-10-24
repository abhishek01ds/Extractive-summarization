"""
evaluate.py: Evaluation script for SGCSumm.
Evaluates generated summaries against human reference summaries on ROUGE-1, ROUGE-2, and ROUGE-L.
Compares performance against published benchmarks from Table 6 and Table 7.

Usage:
    python evaluate.py --config config.yaml --data_dir data/sample_duc2004 --checkpoint models/sgcsumm_best.pt --use_mock
"""

import argparse
import os
import sys
import torch

from src.utils import load_config, set_seed, setup_logger, get_device
from src.preprocessing import TextPreprocessor
from src.bert_embeddings import BERTSentenceEmbedder
from src.graph_builder import SentenceGraphBuilder
from src.dsn_model import SGCSummModel
from src.summary_generator import GreedySummaryGenerator
from src.dataset import load_dataset_from_dir, create_smoke_test_dataset
from src.metrics import SummarizationMetrics, PUBLISHED_PAPER_RESULTS_DUC2004

logger = setup_logger("SGCSumm.EvaluateCLI")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SGCSumm Model on Benchmark Dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, default="./data/sample_duc2004", help="Path to evaluation dataset")
    parser.add_argument("--checkpoint", type=str, default="./models/sgcsumm_best.pt", help="Path to trained checkpoint")
    parser.add_argument("--budget", type=int, default=None, help="Override budget limit (e.g., 665 bytes)")
    parser.add_argument("--budget_type", type=str, default=None, help="Budget type: 'byte' or 'word'")
    parser.add_argument("--use_mock", action="store_true", help="Use mock embedder for offline fast evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cpu', 'cuda', or 'auto'")
    
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    set_seed(config["general"].get("seed", 42))

    device_name = args.device or config["general"].get("device", "cpu")
    device = get_device(device_name)

    data_dir = args.data_dir
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        logger.info(f"Dataset directory '{data_dir}' is empty or missing. Generating smoke-test dataset...")
        create_smoke_test_dataset(data_dir)

    # Preprocessor & Embedder
    preprocessor = TextPreprocessor()
    use_mock = args.use_mock or config["embedding"].get("use_mock", False)
    embedder = BERTSentenceEmbedder(
        model_name=config["embedding"]["model_name"],
        embedding_dim=config["embedding"]["embedding_dim"],
        layers_to_average=config["embedding"]["layers_to_average"],
        device=str(device),
        use_cache=config["embedding"].get("use_cache", True),
        cache_dir=config["embedding"].get("cache_dir", "./data/cache"),
        use_mock=use_mock
    )

    graph_builder = SentenceGraphBuilder(
        top_n=config["graph"]["top_n_vertices"],
        top_k=config["graph"]["top_k_neighbors"],
        similarity_threshold=config["graph"].get("similarity_threshold", 0.0)
    )

    num_channels = config["embedding"]["embedding_dim"] - 1
    model = SGCSummModel(
        num_channels=num_channels,
        num_vertices=config["graph"]["top_n_vertices"],
        num_neighbors=config["graph"]["top_k_neighbors"],
        hidden_dim=config["dsn"]["hidden_dim"]
    ).to(device)

    # Load checkpoint if available
    checkpoint_path = args.checkpoint
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.project_non_negative()
    else:
        logger.warning(f"Checkpoint '{checkpoint_path}' not found. Evaluating with initialized model.")

    # Generator setup
    budget_type = args.budget_type or config["summarization"].get("budget_type", "byte")
    budget_limit = args.budget or config["summarization"].get("budget_limit", 665)
    restore_order = config["summarization"].get("restore_order", True)

    generator = GreedySummaryGenerator(
        model=model,
        embedder=embedder,
        graph_builder=graph_builder,
        budget_type=budget_type,
        budget_limit=budget_limit,
        restore_order=restore_order,
        device=str(device)
    )

    # Load clusters
    clusters = load_dataset_from_dir(data_dir, preprocessor)
    if not clusters:
        logger.error(f"No document clusters loaded from {data_dir}.")
        sys.exit(1)

    metrics_calc = SummarizationMetrics(["rouge1", "rouge2", "rougeL"])
    generated_summaries = []
    gold_references_list = []

    print("\n" + "=" * 80)
    print("SGCSumm EVALUATION RESULTS PER DOCUMENT CLUSTER")
    print("=" * 80)

    for idx, cluster in enumerate(clusters, 1):
        selected_sents, summary_text, logs = generator.generate_summary(cluster.sentences)
        generated_summaries.append(summary_text)
        gold_references_list.append(cluster.reference_summaries)

        cluster_scores = metrics_calc.evaluate_pair(summary_text, cluster.reference_summaries)

        print(f"\n[Cluster {idx}/{len(clusters)}] ID: {cluster.cluster_id}")
        print(f"Total Sentences in Pool: {len(cluster.sentences)} | Selected: {len(selected_sents)}")
        print(f"Summary Length: {len(summary_text.encode('utf-8'))} bytes ({len(summary_text.split())} words)")
        print("-" * 80)
        print("Generated Summary:")
        print(f"  {summary_text}")
        print("-" * 80)
        print("Reference Summary:")
        ref_preview = cluster.reference_summaries[0] if cluster.reference_summaries else "N/A"
        print(f"  {ref_preview}")
        print("-" * 80)
        print(
            f"Scores -> ROUGE-1 Recall: {cluster_scores['rouge1']['recall']:.2f}% | "
            f"ROUGE-2 Recall: {cluster_scores['rouge2']['recall']:.2f}% | "
            f"ROUGE-L F1: {cluster_scores['rougeL']['fmeasure']:.2f}%"
        )

    # Average metrics over all clusters
    overall_scores = metrics_calc.evaluate_dataset(generated_summaries, gold_references_list)

    print("\n" + "=" * 80)
    print("OVERALL DATASET EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Number of Clusters Evaluated: {len(clusters)}")
    print(f"Budget Constraint: {budget_limit} {budget_type}s")
    print(f"ROUGE-1 -> Recall: {overall_scores['rouge1']['recall']:.2f}% | F1: {overall_scores['rouge1']['fmeasure']:.2f}%")
    print(f"ROUGE-2 -> Recall: {overall_scores['rouge2']['recall']:.2f}% | F1: {overall_scores['rouge2']['fmeasure']:.2f}%")
    print(f"ROUGE-L -> Recall: {overall_scores['rougeL']['recall']:.2f}% | F1: {overall_scores['rougeL']['fmeasure']:.2f}%")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON TABLE (Paper Table 6: DUC 2004 Test Set)")
    print("=" * 80)
    print(f"{'Method / System':<35} | {'ROUGE-1':<10} | {'ROUGE-2':<10}")
    print("-" * 80)
    for method, scores in PUBLISHED_PAPER_RESULTS_DUC2004.items():
        print(f"{method:<35} | {scores['rouge1']:<10.2f} | {scores['rouge2']:<10.2f}")
    print("-" * 80)
    print(f"{'Current Run (Evaluated)':<35} | {overall_scores['rouge1']['recall']:<10.2f} | {overall_scores['rouge2']['recall']:<10.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
