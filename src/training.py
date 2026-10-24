"""
src/training.py: Training engine and optimization routines for SGCSumm.
Implements:
1. Complete training loop (Algorithm 4 & Section 3.4).
2. L-BFGS-B optimizer with bound constraints (w >= 0) via SciPy (Byrd et al., 1995).
3. Projected Adam optimizer with non-negative parameter clamping.
4. Checkpointing, validation, and training loss tracking.
5. Distinction between frozen pretrained BERT embeddings and trainable GCN/DSN parameters.
"""

import os
import time
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

from src.dataset import DocumentCluster, augment_cluster_samples
from src.preprocessing import TextPreprocessor
from src.bert_embeddings import BERTSentenceEmbedder
from src.graph_builder import SentenceGraphBuilder, compute_pairwise_similarity
from src.dsn_model import SGCSummModel
from src.utils import setup_logger, ensure_dir

logger = setup_logger("SGCSumm.Training")


class SGCSummTrainer:
    """
    Trainer for SGCSumm representation and scoring network.
    
    Trained parameters:
      - GCN convolution filter weights (conv1, conv2)
      - DSN fully-connected weights (fc1, fc2)
    Untrained / Frozen parameters:
      - Pretrained BERT weights (cached and kept fixed)
    """
    def __init__(
        self,
        model: SGCSummModel,
        embedder: BERTSentenceEmbedder,
        graph_builder: SentenceGraphBuilder,
        preprocessor: TextPreprocessor,
        learning_rate: float = 1e-4,
        epochs: int = 1000,
        optimizer_type: str = "lbfgsb",
        checkpoint_dir: str = "./models",
        device: str = "cpu"
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.embedder = embedder
        self.graph_builder = graph_builder
        self.preprocessor = preprocessor
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer_type = optimizer_type.lower()
        self.checkpoint_dir = checkpoint_dir
        ensure_dir(self.checkpoint_dir)
        
        self.loss_fn = nn.MSELoss()

    def prepare_cluster_tensors(
        self,
        cluster: DocumentCluster
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract suitable features C and pairwise similarity matrix S for a cluster.
        Cached on cluster object if available.
        """
        if not hasattr(cluster, "_cached_C") or cluster._cached_C is None:
            C = self.embedder.extract_suitable_features(cluster.sentences)
            S = compute_pairwise_similarity(C)
            cluster._cached_C = C
            cluster._cached_S = S
        return cluster._cached_C, cluster._cached_S

    def prepare_training_samples(
        self,
        clusters: List[DocumentCluster],
        num_augmentations: int = 5
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Build training dataset: (Tensor X, Target ROUGE-1 score) for each augmented sample.
        """
        dataset = []
        logger.info(f"Preparing augmented training samples for {len(clusters)} clusters...")
        
        for cluster in clusters:
            if not cluster.sentences:
                continue
                
            C, S = self.prepare_cluster_tensors(cluster)
            augmented = augment_cluster_samples(
                cluster,
                self.preprocessor,
                num_augmentations=num_augmentations
            )
            
            for subset_indices, target_score in augmented:
                if not subset_indices:
                    continue
                # For each subset of sentences, extract subgraph features
                subset_C = C[subset_indices]
                subset_S = S[np.ix_(subset_indices, subset_indices)]
                
                _, neighborhoods = self.graph_builder.select_top_vertices_and_neighbors(subset_S)
                X_np = self.graph_builder.construct_convpool_input(subset_C, neighborhoods)
                
                X_tensor = torch.tensor(X_np, dtype=torch.float32)
                y_tensor = torch.tensor(target_score, dtype=torch.float32)
                dataset.append((X_tensor, y_tensor))

        logger.info(f"Constructed {len(dataset)} total training samples.")
        return dataset

    def train_adam_projected(
        self,
        train_samples: List[Tuple[torch.Tensor, torch.Tensor]],
        val_samples: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Dict[str, Any]:
        """
        Train using Adam with projected gradient descent:
        Clamps weights W >= 0 after every parameter update to strictly satisfy submodularity.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": []}

        logger.info(f"Starting Adam (Projected) training for {self.epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_train_loss = 0.0

            # Batch through samples
            for X, y in train_samples:
                X = X.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                pred_score, _ = self.model(X)
                loss = self.loss_fn(pred_score, y)
                loss.backward()
                optimizer.step()

                # Project weights to non-negative orthant (Theorem 4)
                self.model.project_non_negative()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / max(1, len(train_samples))
            history["train_loss"].append(avg_train_loss)

            # Validation
            avg_val_loss = None
            if val_samples:
                self.model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for X_val, y_val in val_samples:
                        X_val = X_val.to(self.device)
                        y_val = y_val.to(self.device)
                        val_pred, _ = self.model(X_val)
                        val_loss = self.loss_fn(val_pred, y_val)
                        total_val_loss += val_loss.item()
                avg_val_loss = total_val_loss / max(1, len(val_samples))
                history["val_loss"].append(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_checkpoint("sgcsumm_best.pt", epoch, best_val_loss)

            if epoch % 50 == 0 or epoch == 1 or epoch == self.epochs:
                val_str = f" | Val Loss: {avg_val_loss:.6f}" if avg_val_loss is not None else ""
                logger.info(f"Epoch [{epoch}/{self.epochs}] | Train Loss: {avg_train_loss:.6f}{val_str}")

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f} seconds.")
        return history

    def train_lbfgsb(
        self,
        train_samples: List[Tuple[torch.Tensor, torch.Tensor]],
        val_samples: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        max_iter: int = 1000
    ) -> Dict[str, Any]:
        """
        Train using exact L-BFGS-B algorithm with bound constraints (w >= 0) via SciPy (Section 3.4).
        """
        logger.info("Starting L-BFGS-B optimization with non-negative bounds (W >= 0)...")
        
        # Flatten model parameters into 1D NumPy array
        params = [p for p in self.model.parameters() if p.requires_grad]
        param_shapes = [p.shape for p in params]
        param_numels = [p.numel() for p in params]
        
        def pack_params() -> np.ndarray:
            return np.concatenate([p.detach().cpu().numpy().ravel() for p in params])

        def unpack_params(theta: np.ndarray):
            offset = 0
            for p, shape, numel in zip(params, param_shapes, param_numels):
                chunk = theta[offset:offset + numel].reshape(shape)
                p.data.copy_(torch.from_numpy(chunk).to(p.device))
                offset += numel

        init_theta = pack_params()
        # Bound constraints: each weight in [0, None)
        bounds = [(0.0, None) for _ in range(len(init_theta))]

        # Stack batch inputs for fast forward pass
        stacked_X = torch.stack([s[0] for s in train_samples]).to(self.device)
        stacked_y = torch.stack([s[1] for s in train_samples]).to(self.device)

        iteration_count = [0]
        history = {"train_loss": []}

        def objective(theta: np.ndarray) -> Tuple[float, np.ndarray]:
            unpack_params(theta)
            self.model.zero_grad()
            
            pred, _ = self.model(stacked_X)
            loss = self.loss_fn(pred, stacked_y)
            loss.backward()
            
            loss_val = float(loss.item())
            grads = np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in params])
            
            iteration_count[0] += 1
            if iteration_count[0] % 50 == 0 or iteration_count[0] == 1:
                logger.info(f"L-BFGS-B Iteration [{iteration_count[0]}] | Loss: {loss_val:.6f}")
                history["train_loss"].append(loss_val)

            return loss_val, grads.astype(np.float64)

        res = minimize(
            fun=objective,
            x0=init_theta,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": max_iter, "disp": False}
        )

        unpack_params(res.x)
        self.model.project_non_negative()
        logger.info(f"L-BFGS-B optimization finished: {res.message} with Final Loss: {res.fun:.6f}")
        self.save_checkpoint("sgcsumm_best.pt", iteration_count[0], res.fun)
        return history

    def train(
        self,
        train_clusters: List[DocumentCluster],
        val_clusters: Optional[List[DocumentCluster]] = None,
        num_augmentations: int = 5
    ) -> Dict[str, Any]:
        """Main training entry point supporting both L-BFGS-B and Adam."""
        train_samples = self.prepare_training_samples(train_clusters, num_augmentations)
        val_samples = self.prepare_training_samples(val_clusters, 1) if val_clusters else None

        if self.optimizer_type == "lbfgsb":
            return self.train_lbfgsb(train_samples, val_samples, max_iter=self.epochs)
        else:
            return self.train_adam_projected(train_samples, val_samples)

    def save_checkpoint(self, filename: str, epoch: int, loss: float):
        """Save model checkpoint and metadata."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load trained model weights from checkpoint."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.project_non_negative()
        logger.info(f"Loaded checkpoint from {filepath} (Epoch {checkpoint.get('epoch')}, Loss: {checkpoint.get('loss'):.6f})")
