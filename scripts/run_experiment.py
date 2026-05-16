import argparse
import copy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.execution_engine.evaluator import Evaluator
from src.execution_engine.trainer import train_full_batch
from src.factories.dataset_factory import create_dataset
from src.factories.model_factory import create_model
from src.observability.run_manager import RunManager
from src.utils.config import deep_update, load_yaml
from src.utils.seed_manager import set_seed


DEFAULT_CONFIG = {
    "seed": 42,
    "output_dir": "outputs/runs/gnn",
    "dataset": {
        "name": "mock",
    },
    "model": {
        "name": "graphsage",
        "params": {},
    },
    "trainer": {
        "lr": 1e-3,
        "epochs": 20,
        "patience": 10,
        "device": "cpu",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a graph fraud-detection experiment.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", choices=["mock", "bitcoinheist"])
    parser.add_argument("--model", choices=["graphsage", "gcn", "mlp"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.config:
        config = deep_update(config, load_yaml(args.config))

    overrides = {}
    if args.dataset is not None:
        overrides.setdefault("dataset", {})["name"] = args.dataset
    if args.model is not None:
        overrides.setdefault("model", {})["name"] = args.model
    if args.epochs is not None:
        overrides.setdefault("trainer", {})["epochs"] = args.epochs
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    return deep_update(config, overrides)


def main() -> None:
    config = build_config(parse_args())
    set_seed(config["seed"])

    dataset_config = dict(config["dataset"])
    dataset_name = dataset_config.pop("name")
    if dataset_name == "mock":
        dataset_config.setdefault("seed", config["seed"])
    dataset = create_dataset(dataset_name, **dataset_config)
    data = dataset.load()
    masks = dataset.build_masks(data)

    model = create_model(
        config["model"]["name"],
        in_channels=data.num_features,
        **config["model"].get("params", {}),
    )

    run = RunManager(config["output_dir"], f"{dataset_name}_{config['model']['name']}")
    run.save_config(config)
    result = train_full_batch(
        model,
        data,
        masks["train_mask"],
        masks["val_mask"],
        lr=config["trainer"].get("lr", 1e-3),
        epochs=config["trainer"].get("epochs", 20),
        patience=config["trainer"].get("patience", 10),
        device=config["trainer"].get("device", "cpu"),
        run_dir=run.run_dir,
    )
    metrics = Evaluator(device=config["trainer"].get("device", "cpu")).evaluate(model, data, masks)
    run.save_metrics({"best_val_pr_auc": result.best_val_pr_auc, "splits": metrics})
    run.save_history(result.history)

    print(f"run_dir={run.run_dir}")
    print(f"best_val_pr_auc={result.best_val_pr_auc:.4f}")
    print(metrics)


if __name__ == "__main__":
    main()
