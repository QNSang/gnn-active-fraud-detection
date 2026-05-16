from typing import Dict

import torch

from src.data_layers.base_graph_dataset import BaseGraphDataset, GraphDatasetInfo
from src.data_layers.graph_data import Data
from src.data_layers.graph_views import build_leakage_safe_views
from src.data_layers.splitters.temporal_splitter import build_temporal_masks


class MockGraphDataset(BaseGraphDataset):
    """Small synthetic graph dataset for testing the data pipeline shape."""

    def __init__(
        self,
        num_nodes: int = 300,
        num_edges: int = 900,
        num_features: int = 16,
        unknown_label: int = -1,
        seed: int = 42,
    ) -> None:
        if num_nodes < 1:
            raise ValueError("num_nodes must be positive.")
        if num_edges < 1:
            raise ValueError("num_edges must be positive.")

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_features = num_features
        self.unknown_label = unknown_label
        self.seed = seed
        self.data: Data | None = None
        self.masks: Dict[str, torch.Tensor] | None = None
        self.graph_views: Dict[str, Data] | None = None

    def load(self, root: str | None = None) -> Data:
        generator = torch.Generator().manual_seed(self.seed)

        x = torch.randn(self.num_nodes, self.num_features, generator=generator)
        edge_index = torch.randint(
            low=0,
            high=self.num_nodes,
            size=(2, self.num_edges),
            generator=generator,
        )
        y = torch.multinomial(
            torch.tensor([0.55, 0.15, 0.30]),
            num_samples=self.num_nodes,
            replacement=True,
            generator=generator,
        )
        y = torch.where(y == 2, torch.full_like(y, self.unknown_label), y)
        time_step = torch.randint(1, 50, (self.num_nodes,), generator=generator)

        self.data = Data(x=x, edge_index=edge_index, y=y, time_step=time_step)
        return self.data

    def preprocess(self, data: Data) -> Data:
        return data

    def build_masks(self, data: Data) -> Dict[str, torch.Tensor]:
        self.masks = build_temporal_masks(data, unknown_label=self.unknown_label)
        return self.masks

    def build_graph_views(self, data: Data) -> Dict[str, Data]:
        if self.masks is None:
            self.build_masks(data)
        self.graph_views = build_leakage_safe_views(data, self.masks)
        return self.graph_views

    def get_info(self) -> GraphDatasetInfo:
        if self.data is None:
            raise RuntimeError("Call load() before get_info().")

        return GraphDatasetInfo(
            name="mock",
            version="0.1.0",
            num_nodes=self.data.num_nodes,
            num_edges=self.data.num_edges,
            num_features=self.data.num_features,
            num_classes=2,
            time_steps=49,
            label_names=["licit", "illicit", "unknown"],
            unknown_label=self.unknown_label,
        )
