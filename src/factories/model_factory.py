from typing import Any

from src.model_zoo.models.gcn_model import GCNModel
from src.model_zoo.models.graphsage_model import GraphSAGEModel
from src.model_zoo.models.mlp_model import MLPModel


def create_model(name: str, in_channels: int, **kwargs: Any):
    model_name = name.lower()
    if model_name in {"graphsage", "sage"}:
        return GraphSAGEModel(in_channels=in_channels, **kwargs)
    if model_name == "gcn":
        return GCNModel(in_channels=in_channels, **kwargs)
    if model_name == "mlp":
        return MLPModel(in_channels=in_channels, **kwargs)
    raise ValueError(f"Unknown model: {name}")
