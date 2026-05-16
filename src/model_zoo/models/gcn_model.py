import torch
import torch.nn.functional as F
from torch import nn

from src.model_zoo.base_graph_model import BaseGraphModel, ModelConfig


try:
    from torch_geometric.nn import GCNConv
except ModuleNotFoundError:

    class GCNConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            return self.linear(x)


class GCNModel(BaseGraphModel):
    def __init__(self, in_channels: int, hidden_channels: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.config = ModelConfig("gcn", in_channels, hidden_channels, 1, dropout)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, 1)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        return self.conv2(x, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x, edge_index)).squeeze(-1)
