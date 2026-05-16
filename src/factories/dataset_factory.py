from typing import Any

from src.data_layers.datasets.bitcoinheist_dataset import BitcoinHeistDataset
from src.data_layers.datasets.mock_graph_dataset import MockGraphDataset


def create_dataset(name: str, **kwargs: Any):
    dataset_name = name.lower()
    if dataset_name == "mock":
        return MockGraphDataset(**kwargs)
    if dataset_name in {"bitcoinheist", "bitcoin_heist"}:
        return BitcoinHeistDataset(**kwargs)
    raise ValueError(f"Unknown dataset: {name}")
