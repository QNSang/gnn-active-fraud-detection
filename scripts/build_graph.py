import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_layers.datasets.bitcoinheist_dataset import BitcoinHeistDataset
from src.utils.config import load_yaml
from src.utils.io import write_json
from src.utils.seed_manager import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and save a BitcoinHeist graph artifact.")
    parser.add_argument("--config", default="configs/pipeline/graph_build.yaml")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(config.get("seed", 42))

    output_dir = Path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = BitcoinHeistDataset(**config["dataset"])
    data = dataset.load()
    masks = dataset.build_masks(data)
    info = dataset.get_info()

    torch.save(data, output_dir / "data.pt")
    torch.save(masks, output_dir / "masks.pt")
    write_json(asdict(info), output_dir / "info.json")
    write_json(config, output_dir / "config.json")

    print(f"graph_dir={output_dir}")
    print(f"nodes={info.num_nodes} edges={info.num_edges} features={info.num_features}")


if __name__ == "__main__":
    main()
