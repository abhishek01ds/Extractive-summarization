from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gcn_model import GCNRepresentationLearner


class DeepSubmodularNetwork(nn.Module):
    """
    3-layer neural network to score sentence subsets.
    Uses non-negative weights and ReLU/Sigmoid activations.
    """
    def __init__(
        self,
        input_dim: int = 767,
        hidden_dim: int = 16,
        hidden_activation: str = "relu",
        output_activation: str = "sigmoid"
    ):
        super(DeepSubmodularNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation.lower()
        self.output_activation = output_activation.lower()

        # Linear layers without bias
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)

        self._init_non_negative_weights()

    def _init_non_negative_weights(self):
        # Initialize with small positive weights
        with torch.no_grad():
            nn.init.uniform_(self.fc1.weight, a=0.01, b=0.1)
            nn.init.uniform_(self.fc2.weight, a=0.01, b=0.1)

    def project_non_negative(self):
        # Clamp weights so they stay non-negative (>= 0)
        with torch.no_grad():
            self.fc1.weight.clamp_(min=0.0)
            self.fc2.weight.clamp_(min=0.0)

    def validate_input(self, Y: torch.Tensor) -> bool:
        if Y is None:
            return False
        if Y.shape[-1] != self.input_dim:
            return False
        if torch.isnan(Y).any() or torch.isinf(Y).any():
            return False
        if (Y < -1e-6).any():
            return False
        return True

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        is_unbatched = (Y.ndim == 1)
        if is_unbatched:
            Y = Y.unsqueeze(0)

        if not self.validate_input(Y):
            device = Y.device if isinstance(Y, torch.Tensor) else "cpu"
            return torch.tensor([-1.0], device=device)

        # Keep weights positive
        w1 = torch.clamp(self.fc1.weight, min=0.0)
        w2 = torch.clamp(self.fc2.weight, min=0.0)

        # Hidden layer with ReLU
        h = F.linear(Y, w1)
        h = F.relu(h)

        # Output layer with Sigmoid to get a score between 0 and 1
        out = F.linear(h, w2)
        score = torch.sigmoid(out)

        if is_unbatched:
            score = score.squeeze(0).squeeze(-1)
        else:
            score = score.squeeze(-1)

        return score


class SGCSummModel(nn.Module):
    """
    Combined model connecting GCN feature learner and DSN scorer.
    """
    def __init__(
        self,
        num_channels: int = 767,
        num_vertices: int = 15,
        num_neighbors: int = 20,
        hidden_dim: int = 16
    ):
        super(SGCSummModel, self).__init__()
        self.representation_learner = GCNRepresentationLearner(
            num_channels=num_channels,
            num_vertices=num_vertices,
            num_neighbors=num_neighbors
        )
        self.dsn = DeepSubmodularNetwork(
            input_dim=num_channels,
            hidden_dim=hidden_dim
        )

    def project_non_negative(self):
        self.representation_learner.project_non_negative()
        self.dsn.project_non_negative()

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # X -> GCN -> Y -> DSN -> score
        Y = self.representation_learner(X)
        score = self.dsn(Y)
        return score, Y
