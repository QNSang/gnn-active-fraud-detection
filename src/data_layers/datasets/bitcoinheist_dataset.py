from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from src.data_layers.bitcoinheist_schema import FEATURE_COLS, LABEL_COL, LOG1P_COLS, TARGET_COL, encode_binary_label
from src.data_layers.base_graph_dataset import BaseGraphDataset, GraphDatasetInfo
from src.data_layers.graph_data import Data
from src.data_layers.graph_views import build_leakage_safe_views
from src.data_layers.raw_loader import load_bitcoinheist_raw


class BitcoinHeistDataset(BaseGraphDataset):
    """BitcoinHeist row-level dataset with chronological preprocessing."""

    def __init__(
        self,
        csv_path: str | Path = "data/raw/BitcoinHeistData.csv",
        train_end_year: int = 2015,
        val_end_year: int = 2016,
        max_temporal_gap: int = 30,
        unknown_label: int = -1,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.train_end_year = train_end_year
        self.val_end_year = val_end_year
        self.max_temporal_gap = max_temporal_gap
        self.unknown_label = unknown_label
        self.frame: pd.DataFrame | None = None
        self.data: Data | None = None
        self.masks: Dict[str, torch.Tensor] | None = None

    def load(self, root: str | None = None) -> Data:
        path = Path(root) / self.csv_path if root is not None else self.csv_path
        df = load_bitcoinheist_raw(path)
        df["label_raw"] = df[LABEL_COL]
        df[TARGET_COL] = df[LABEL_COL].map(encode_binary_label).astype(int)
        self.frame = df
        self.data = self.preprocess_frame(df)
        return self.data

    def preprocess(self, data: Data) -> Data:
        return data

    def preprocess_frame(self, df: pd.DataFrame) -> Data:
        processed = df.copy()
        for col in LOG1P_COLS:
            processed[col] = np.log1p(processed[col].clip(lower=0))

        train_mask_np = processed["year"].to_numpy() <= self.train_end_year
        mean = processed.loc[train_mask_np, FEATURE_COLS].mean()
        std = processed.loc[train_mask_np, FEATURE_COLS].std().replace(0, 1.0).fillna(1.0)
        processed[FEATURE_COLS] = (processed[FEATURE_COLS] - mean) / std

        edge_index = self._build_temporal_edges(df)
        timestamp = (df["year"].to_numpy(dtype=np.int64) * 366) + df["day"].to_numpy(dtype=np.int64)
        return Data(
            x=torch.tensor(processed[FEATURE_COLS].to_numpy(dtype="float32")),
            edge_index=edge_index,
            y=torch.tensor(processed[TARGET_COL].to_numpy(dtype=np.int64)),
            time_step=torch.tensor(timestamp, dtype=torch.long),
        )

    def _build_temporal_edges(self, df: pd.DataFrame) -> torch.Tensor:
        edges: list[tuple[int, int]] = []
        last_seen: dict[str, tuple[int, int]] = {}
        timestamps = (df["year"].to_numpy(dtype=np.int64) * 366) + df["day"].to_numpy(dtype=np.int64)
        for row_id, (address, timestamp) in enumerate(zip(df["address"], timestamps)):
            if address in last_seen:
                prev_row, prev_time = last_seen[address]
                if timestamp >= prev_time and timestamp - prev_time <= self.max_temporal_gap:
                    edges.append((prev_row, row_id))
                    edges.append((row_id, prev_row))
            last_seen[address] = (row_id, int(timestamp))
        if not edges:
            self_edges = [(idx, idx) for idx in range(len(df))]
            edges = self_edges
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def build_masks(self, data: Data) -> Dict[str, torch.Tensor]:
        train_mask = data.time_step <= self.train_end_year * 366 + 366
        val_mask = (data.time_step > self.train_end_year * 366 + 366) & (
            data.time_step <= self.val_end_year * 366 + 366
        )
        test_mask = data.time_step > self.val_end_year * 366 + 366
        self.masks = {
            "known_mask": torch.ones(data.num_nodes, dtype=torch.bool),
            "unknown_mask": torch.zeros(data.num_nodes, dtype=torch.bool),
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "query_pool_mask": train_mask.clone(),
        }
        return self.masks

    def build_graph_views(self, data: Data) -> Dict[str, Data]:
        if self.masks is None:
            self.build_masks(data)
        return build_leakage_safe_views(data, self.masks)

    def get_info(self) -> GraphDatasetInfo:
        if self.data is None:
            raise RuntimeError("Call load() before get_info().")
        return GraphDatasetInfo(
            name="bitcoinheist",
            version="0.1.0",
            num_nodes=self.data.num_nodes,
            num_edges=self.data.num_edges,
            num_features=self.data.num_features,
            num_classes=2,
            time_steps=int(self.data.time_step.max().item()),
            label_names=["white", "fraud"],
            unknown_label=self.unknown_label,
        )
