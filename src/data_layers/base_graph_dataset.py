from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import torch

from src.data_layers.graph_data import Data


@dataclass(frozen=True)
class GraphDatasetInfo:
    name: str
    version: str
    num_nodes: int
    num_edges: int
    num_features: int
    num_classes: int
    time_steps: int
    label_names: list[str]
    unknown_label: int


class BaseGraphDataset(ABC):
    @abstractmethod
    def load(self, root: str | None = None) -> Data:
        """Load or generate a graph dataset."""

    @abstractmethod
    def preprocess(self, data: Data) -> Data:
        """Apply dataset-specific preprocessing."""

    @abstractmethod
    def build_masks(self, data: Data) -> Dict[str, torch.Tensor]:
        """Build train/validation/test and label-state masks."""

    @abstractmethod
    def build_graph_views(self, data: Data) -> Dict[str, Data]:
        """Build graph views used by the future training pipeline."""

    @abstractmethod
    def get_info(self) -> GraphDatasetInfo:
        """Return metadata for the loaded dataset."""
