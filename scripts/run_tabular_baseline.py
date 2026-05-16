import argparse
import copy
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_layers.raw_loader import load_bitcoinheist_raw
from src.data_layers.tabular.preprocessing import prepare_tabular_dataset
from src.observability.run_manager import RunManager
from src.tabular_ml.models import create_tabular_model
from src.tabular_ml.trainer import train_tabular_model
from src.utils.config import deep_update, load_yaml


DEFAULT_CONFIG = {
    "seed": 42,
    "output_dir": "outputs/runs/tabular",
    "data": {
        "csv_path": "data/raw/BitcoinHeistData.csv",
        "train_end_year": 2015,
        "val_end_year": 2016,
    },
    "preprocessing": {
        "scale_features": True,
    },
    "model": {
        "name": "logreg",
        "params": {},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tabular BitcoinHeist baseline without graph construction.")
    parser.add_argument("--config", default="configs/pipeline/tabular_baseline.yaml")
    parser.add_argument("--csv-path")
    parser.add_argument("--model", choices=["logreg", "logistic_regression", "random_forest", "rf"])
    parser.add_argument("--train-end-year", type=int)
    parser.add_argument("--val-end-year", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.config:
        config = deep_update(config, load_yaml(args.config))

    overrides = {}
    if args.csv_path is not None:
        overrides.setdefault("data", {})["csv_path"] = args.csv_path
    if args.train_end_year is not None:
        overrides.setdefault("data", {})["train_end_year"] = args.train_end_year
    if args.val_end_year is not None:
        overrides.setdefault("data", {})["val_end_year"] = args.val_end_year
    if args.model is not None:
        overrides.setdefault("model", {})["name"] = args.model
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides["seed"] = args.seed
    return deep_update(config, overrides)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_model_bundle(path: Path, model, dataset) -> None:
    payload = {
        "model": model,
        "scaler": dataset.scaler,
        "feature_cols": dataset.feature_cols,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def main() -> None:
    config = build_config(parse_args())
    set_seed(config["seed"])
    run = RunManager(config["output_dir"], f"tabular_{config['model']['name']}")
    run.save_config(config)

    frame = load_bitcoinheist_raw(config["data"]["csv_path"])
    dataset = prepare_tabular_dataset(
        frame,
        train_end_year=config["data"]["train_end_year"],
        val_end_year=config["data"]["val_end_year"],
        scale_features=config["preprocessing"].get("scale_features", True),
    )
    model = create_tabular_model(
        config["model"]["name"],
        seed=config["seed"],
        **config["model"].get("params", {}),
    )
    result = train_tabular_model(model, dataset)

    run.save_metrics(result.metrics)
    save_model_bundle(run.path("model.pkl"), result.model, dataset)

    print(f"run_dir={run.run_dir}")
    print(result.metrics)


if __name__ == "__main__":
    main()
