from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNRepresentationLearner(nn.Module):
    """
    Learns sentence representations from graph neighborhoods using 2D convolutions.
    """
    def __init__(
        self,
        num_channels: int = 767,
        num_vertices: int = 15,
        num_neighbors: int = 20,
        activation: str = "relu"
    ):
        super(GCNRepresentationLearner, self).__init__()
        self.num_channels = num_channels
        self.num_vertices = num_vertices
        self.num_neighbors = num_neighbors
        self.activation_name = activation.lower()

        # Conv over neighbors (1, k)
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(1, num_neighbors),
            stride=1,
            padding=0,
            groups=num_channels,
            bias=False
        )

        # Conv over vertices (n, 1)
        self.conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(num_vertices, 1),
            stride=1,
            padding=0,
            groups=num_channels,
            bias=False
        )

        self._init_non_negative_weights()

    def _init_non_negative_weights(self):
        # Initialize weights with small positive numbers
        with torch.no_grad():
            nn.init.uniform_(self.conv1.weight, a=0.01, b=0.1)
            nn.init.uniform_(self.conv2.weight, a=0.01, b=0.1)

    def project_non_negative(self):
        # Keep weights non-negative (>= 0)
        with torch.no_grad():
            self.conv1.weight.clamp_(min=0.0)
            self.conv2.weight.clamp_(min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_unbatched = (x.ndim == 3)
        if is_unbatched:
            x = x.unsqueeze(0)

        # Make sure inputs and weights stay positive
        x = torch.clamp(x, min=0.0)
        w1 = torch.clamp(self.conv1.weight, min=0.0)
        w2 = torch.clamp(self.conv2.weight, min=0.0)

        # Step 1: Convolve over neighbors
        out1 = F.conv2d(x, w1, bias=None, stride=1, padding=0, groups=self.num_channels)
        out1 = F.relu(out1)

        # Step 2: Convolve over top vertices
        out2 = F.conv2d(out1, w2, bias=None, stride=1, padding=0, groups=self.num_channels)
        out2 = F.relu(out2)

        # Flatten output to vector Y
        Y = out2.view(out2.size(0), self.num_channels)

        if is_unbatched:
            Y = Y.squeeze(0)

        return Y
