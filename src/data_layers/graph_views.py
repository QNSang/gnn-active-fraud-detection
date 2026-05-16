from typing import Dict

import torch

from src.data_layers.graph_data import Data


def _induced_view(data: Data, node_mask: torch.Tensor) -> Data:
    if hasattr(data, "subgraph"):
        return data.subgraph(node_mask)

    node_indices = node_mask.nonzero(as_tuple=False).view(-1)
    old_to_new = torch.full((data.num_nodes,), -1, dtype=torch.long, device=data.edge_index.device)
    old_to_new[node_indices] = torch.arange(node_indices.numel(), device=data.edge_index.device)
    edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
    edge_index = old_to_new[data.edge_index[:, edge_mask]]
    attrs = {
        "x": data.x[node_mask],
        "edge_index": edge_index,
        "y": data.y[node_mask],
    }
    if hasattr(data, "time_step"):
        attrs["time_step"] = data.time_step[node_mask]
    return Data(**attrs)


def build_leakage_safe_views(data: Data, masks: Dict[str, torch.Tensor]) -> Dict[str, Data]:
    """Create cumulative graph views without exposing future nodes to earlier splits."""
    train_nodes = masks["train_mask"]
    val_nodes = train_nodes | masks["val_mask"]
    test_nodes = val_nodes | masks["test_mask"]
    return {
        "train": _induced_view(data, train_nodes),
        "val": _induced_view(data, val_nodes),
        "test": _induced_view(data, test_nodes),
    }
