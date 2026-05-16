from collections import Counter
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data_layers.datasets.mock_graph_dataset import MockGraphDataset


LABEL_NAMES = {
    0: "normal",
    1: "fraud",
}


def format_counter(counter: Counter[int]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in sorted(counter.items()))


def main() -> None:
    dataset = MockGraphDataset()
    data = dataset.preprocess(dataset.load())
    masks = dataset.build_masks(data)
    info = dataset.get_info()

    label_counts = Counter(data.y.tolist())
    named_label_counts = {
        LABEL_NAMES.get(label, str(label)): count
        for label, count in sorted(label_counts.items())
    }
    time_counts = Counter(data.time_step.tolist())

    print(f"num_nodes: {info.num_nodes}")
    print(f"num_edges: {info.num_edges}")
    print(f"num_features: {info.num_features}")
    print(f"label_distribution: {named_label_counts}")
    print(f"train_nodes: {int(masks['train_mask'].sum())}")
    print(f"val_nodes: {int(masks['val_mask'].sum())}")
    print(f"test_nodes: {int(masks['test_mask'].sum())}")
    print(f"unknown_nodes: {int(masks['unknown_mask'].sum())}")
    print(f"query_pool_nodes: {int(masks['query_pool_mask'].sum())}")
    print(f"time_step_distribution: {format_counter(time_counts)}")

    overlap = (
        masks["train_mask"].to(torch.int)
        + masks["val_mask"].to(torch.int)
        + masks["test_mask"].to(torch.int)
    )
    if bool((overlap > 1).any()):
        raise RuntimeError("Temporal masks overlap.")


if __name__ == "__main__":
    main()
