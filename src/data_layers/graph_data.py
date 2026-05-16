import torch


try:
    from torch_geometric.data import Data
except ModuleNotFoundError:

    class Data:
        """Minimal fallback for local inspection when PyG is not installed."""

        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

        def to(self, device: str | torch.device) -> "Data":
            values = {}
            for key, value in self.__dict__.items():
                values[key] = value.to(device) if hasattr(value, "to") else value
            return Data(**values)

        @property
        def num_nodes(self) -> int:
            return int(self.x.shape[0])

        @property
        def num_edges(self) -> int:
            return int(self.edge_index.shape[1])

        @property
        def num_features(self) -> int:
            return int(self.x.shape[1])

        def subgraph(self, node_mask: torch.Tensor) -> "Data":
            node_indices = node_mask.nonzero(as_tuple=False).view(-1)
            old_to_new = torch.full((self.num_nodes,), -1, dtype=torch.long)
            old_to_new[node_indices] = torch.arange(node_indices.numel())

            edge_mask = node_mask[self.edge_index[0]] & node_mask[self.edge_index[1]]
            edge_index = old_to_new[self.edge_index[:, edge_mask]]

            return Data(
                x=self.x[node_mask],
                edge_index=edge_index,
                y=self.y[node_mask],
                time_step=self.time_step[node_mask],
            )
