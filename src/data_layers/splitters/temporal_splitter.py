from typing import Dict

import torch

from src.data_layers.graph_data import Data


def build_temporal_masks(
    data: Data,
    unknown_label: int = -1,
    train_end_step: int = 34,
    val_start_step: int = 35,
    val_end_step: int = 40,
    test_start_step: int = 41,
) -> Dict[str, torch.Tensor]:
    """Build leakage-safe temporal masks for node classification."""
    if not hasattr(data, "time_step"):
        raise ValueError("Data object must have a time_step attribute.")
    if not hasattr(data, "y"):
        raise ValueError("Data object must have a y attribute.")

    known_mask = data.y != unknown_label
    unknown_mask = data.y == unknown_label

    train_time = data.time_step <= train_end_step
    val_time = (data.time_step >= val_start_step) & (data.time_step <= val_end_step)
    test_time = data.time_step >= test_start_step

    train_mask = train_time & known_mask
    val_mask = val_time & known_mask
    test_mask = test_time & known_mask

    return {
        "known_mask": known_mask,
        "unknown_mask": unknown_mask,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "query_pool_mask": train_mask.clone(),
    }
