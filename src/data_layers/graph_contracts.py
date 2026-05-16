from dataclasses import dataclass
from typing import Literal

import torch


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class GraphSplit:
    """Boolean masks that define a leakage-safe temporal node split."""

    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    query_pool_mask: torch.Tensor

    def validate(self) -> None:
        masks = [self.train_mask, self.val_mask, self.test_mask]
        if any(mask.dtype != torch.bool for mask in masks):
            raise TypeError("Split masks must be boolean tensors.")
        if self.train_mask.shape != self.val_mask.shape or self.train_mask.shape != self.test_mask.shape:
            raise ValueError("Split masks must have identical shapes.")
        if torch.any(self.train_mask & self.val_mask) or torch.any(self.train_mask & self.test_mask):
            raise ValueError("Train split overlaps with validation/test.")
        if torch.any(self.val_mask & self.test_mask):
            raise ValueError("Validation split overlaps with test.")


@dataclass(frozen=True)
class GraphBuildMetadata:
    strategy: str
    build_split: str
    num_nodes: int
    num_edges: int
    avg_degree: float
    temporal_safe: bool = True


def ensure_edge_index(edge_index: torch.Tensor, num_nodes: int) -> None:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if edge_index.numel() == 0:
        return
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
        raise ValueError("edge_index contains node ids outside graph bounds.")
