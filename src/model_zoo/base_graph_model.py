from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    name: str
    in_channels: int
    hidden_channels: int = 128
    out_channels: int = 1
    dropout: float = 0.2


class BaseGraphModel(nn.Module, ABC):
    config: ModelConfig

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        """Return raw logits."""

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(x, edge_index)

    def get_config(self) -> dict:
        return self.config.__dict__.copy()
