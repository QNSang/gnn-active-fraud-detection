import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_layers.datasets.mock_graph_dataset import MockGraphDataset
from src.execution_engine.trainer import train_full_batch
from src.factories.model_factory import create_model
from src.utils.seed_manager import set_seed


def main() -> None:
    set_seed(42)
    dataset = MockGraphDataset(num_nodes=120, num_edges=360, num_features=8, seed=42)
    data = dataset.load()
    masks = dataset.build_masks(data)
    views = dataset.build_graph_views(data)
    model = create_model("graphsage", in_channels=data.num_features, hidden_channels=32)
    result = train_full_batch(
        model,
        data,
        masks["train_mask"],
        masks["val_mask"],
        epochs=3,
        patience=2,
        run_dir="outputs/runs/validate_pipeline",
    )
    print("Pipeline validation passed")
    print(f"nodes={data.num_nodes} edges={data.num_edges} train_view_edges={views['train'].num_edges}")
    print(f"best_val_pr_auc={result.best_val_pr_auc:.4f}")


if __name__ == "__main__":
    main()
