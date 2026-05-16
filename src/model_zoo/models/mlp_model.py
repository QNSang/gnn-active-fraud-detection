import torch
import torch.nn.functional as F
from torch import nn

from src.model_zoo.base_graph_model import BaseGraphModel, ModelConfig


class MLPModel(BaseGraphModel):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        second_hidden_channels: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.config = ModelConfig("mlp", in_channels, hidden_channels, 1, dropout)
        self.layers = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, second_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(second_hidden_channels, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.encode(x)).squeeze(-1)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        return self.encode(x)
