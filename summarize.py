"""
summarize.py: Command-line summarization tool for SGCSumm.
Summarizes a cluster of text documents into an extractive summary satisfying
submodular marginal gain maximization under a specified budget.

Usage:
    python summarize.py --input_dir path/to/docs/ --budget 665 --budget_type byte --use_mock
"""

import argparse
import os
import sys
import torch

from src.utils import load_config, setup_logger, get_device
from src.preprocessing import TextPreprocessor
from src.bert_embeddings import BERTSentenceEmbedder
from src.graph_builder import SentenceGraphBuilder
from src.dsn_model import SGCSummModel
from src.summary_generator import GreedySummaryGenerator

logger = setup_logger("SGCSumm.SummarizeCLI")


def main():
    parser = argparse.ArgumentParser(description="Summarize Multiple Related Documents using SGCSumm")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing .txt files to summarize")
    parser.add_argument("--files", nargs="+", default=None, help="Specific .txt files to summarize")
    parser.add_argument("--checkpoint", type=str, default="./models/sgcsumm_best.pt", help="Path to model checkpoint")
    parser.add_argument("--budget", type=int, default=665, help="Budget limit (default 665 bytes)")
    parser.add_argument("--budget_type", type=str, default="byte", choices=["byte", "word"], help="Budget type")
    parser.add_argument("--output_file", type=str, default=None, help="File path to save generated summary")
    parser.add_argument("--use_mock", action="store_true", help="Use mock embedder for offline fast CPU inference")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' or 'cuda'")

    args = parser.parse_args()

    # Read input documents
    doc_dict = {}
    if args.input_dir and os.path.exists(args.input_dir):
        for fname in sorted(os.listdir(args.input_dir)):
            if fname.endswith(".txt"):
                fpath = os.path.join(args.input_dir, fname)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    doc_dict[fname] = f.read()
    elif args.files:
        for fpath in args.files:
            if os.path.exists(fpath):
                fname = os.path.basename(fpath)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    doc_dict[fname] = f.read()
    else:
        # Provide sample documents if none provided
        logger.info("No input files provided. Using built-in demonstration multi-document cluster...")
        doc_dict = {
            "news_report_1.txt": (
                "Scientists have made a historic breakthrough in renewable solar energy efficiency. "
                "The research team developed a new perovskite crystal formulation that captures broader light spectrums. "
                "Early testing shows an efficiency increase of over twenty percent compared to conventional panels. "
                "Commercial deployment of the new technology is projected within the next two years."
            ),
            "news_report_2.txt": (
                "Leading energy analysts hailed the solar breakthrough as a turning point in climate action. "
                "The novel perovskite material can be manufactured at significantly lower production costs. "
                "Global solar manufacturers have already formed consortiums to license the fabrication patent. "
                "Experts caution that long-term durability in extreme weather conditions requires further validation."
            )
        }

    if not doc_dict:
        logger.error("No valid text documents found to summarize.")
        sys.exit(1)

    print(f"\nLoaded {len(doc_dict)} input document(s): {', '.join(doc_dict.keys())}")

    # Load configuration
    config = load_config(args.config)
    device = get_device(args.device)

    preprocessor = TextPreprocessor()
    sentences = preprocessor.preprocess_documents(doc_dict)
    print(f"Total Sentences Extracted: {len(sentences)}")

    use_mock = args.use_mock or config["embedding"].get("use_mock", False)
    embedder = BERTSentenceEmbedder(
        model_name=config["embedding"]["model_name"],
        embedding_dim=config["embedding"]["embedding_dim"],
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

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.project_non_negative()
        logger.info(f"Loaded trained checkpoint: {args.checkpoint}")

    generator = GreedySummaryGenerator(
        model=model,
        embedder=embedder,
        graph_builder=graph_builder,
        budget_type=args.budget_type,
        budget_limit=args.budget,
        restore_order=True,
        device=str(device)
    )

    selected_sents, summary_text, logs = generator.generate_summary(sentences)

    print("\n" + "=" * 80)
    print("SGCSumm EXTRACTIVE MULTI-DOCUMENT SUMMARY")
    print("=" * 80)
    print(summary_text)
    print("=" * 80)
    print(f"Summary Metrics: {len(summary_text.encode('utf-8'))} bytes | {len(summary_text.split())} words")
    print(f"Selected {len(selected_sents)} of {len(sentences)} sentences:")
    for i, s in enumerate(selected_sents, 1):
        print(f"  {i}. [{s.doc_id} - Sent #{s.sent_id}] (len={s.byte_length}b): {s.raw_text}")

    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"\nSaved summary to: {args.output_file}")


if __name__ == "__main__":
    main()
