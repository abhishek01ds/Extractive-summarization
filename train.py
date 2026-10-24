"""
train.py: Command-line entry point for training SGCSumm.
Usage:
    python train.py --config config.yaml --data_dir data/sample_duc2004 --epochs 100 --use_mock
"""

import argparse
import os
import sys

from src.utils import load_config, set_seed, setup_logger, get_device, ensure_dir
from src.preprocessing import TextPreprocessor
from src.bert_embeddings import BERTSentenceEmbedder
from src.graph_builder import SentenceGraphBuilder
from src.dsn_model import SGCSummModel
from src.dataset import load_dataset_from_dir, create_smoke_test_dataset
from src.training import SGCSummTrainer

logger = setup_logger("SGCSumm.TrainCLI")


def main():
    parser = argparse.ArgumentParser(description="Train SGCSumm Extractive Summarization Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, default="./data/sample_duc2004", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer: 'lbfgsb' or 'adam_projected'")
    parser.add_argument("--use_mock", action="store_true", help="Use deterministic mock embedder for offline fast CPU testing")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cpu', 'cuda', or 'auto'")
    
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Overrides
    seed = config["general"].get("seed", 42)
    set_seed(seed)
    
    device_name = args.device or config["general"].get("device", "cpu")
    device = get_device(device_name)
    logger.info(f"Using device: {device}")

    # Check dataset directory; create sample if missing
    data_dir = args.data_dir
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        logger.info(f"Dataset directory '{data_dir}' is empty or missing. Generating smoke-test dataset...")
        create_smoke_test_dataset(data_dir)

    # Initialize components
    logger.info("Initializing SGCSumm preprocessing and neural network components...")
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

    # Neural network: m - 1 channels
    num_channels = config["embedding"]["embedding_dim"] - 1
    model = SGCSummModel(
        num_channels=num_channels,
        num_vertices=config["graph"]["top_n_vertices"],
        num_neighbors=config["graph"]["top_k_neighbors"],
        hidden_dim=config["dsn"]["hidden_dim"]
    )

    # Load dataset
    clusters = load_dataset_from_dir(data_dir, preprocessor)
    if not clusters:
        logger.error(f"No document clusters loaded from {data_dir}. Exiting.")
        sys.exit(1)

    # 90% train, 10% test split (Paper Section 3.4)
    val_split = config["training"].get("val_split", 0.10)
    split_idx = max(1, int(len(clusters) * (1.0 - val_split)))
    train_clusters = clusters[:split_idx]
    val_clusters = clusters[split_idx:] if split_idx < len(clusters) else clusters[:1]

    logger.info(f"Dataset split: {len(train_clusters)} training clusters, {len(val_clusters)} validation clusters.")

    # Training configuration
    epochs = args.epochs or config["training"].get("epochs", 1000)
    lr = args.lr or config["training"].get("learning_rate", 1e-4)
    optimizer_type = args.optimizer or config["training"].get("optimizer", "lbfgsb")
    checkpoint_dir = config["training"].get("checkpoint_dir", "./models")

    trainer = SGCSummTrainer(
        model=model,
        embedder=embedder,
        graph_builder=graph_builder,
        preprocessor=preprocessor,
        learning_rate=lr,
        epochs=epochs,
        optimizer_type=optimizer_type,
        checkpoint_dir=checkpoint_dir,
        device=str(device)
    )

    history = trainer.train(
        train_clusters=train_clusters,
        val_clusters=val_clusters,
        num_augmentations=config["training"].get("augmentation_samples", 5)
    )

    logger.info("Training workflow successfully finished! Model checkpoint saved in: " + checkpoint_dir)


if __name__ == "__main__":
    main()
