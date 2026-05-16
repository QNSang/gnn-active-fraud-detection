import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_layers.bitcoinheist_schema import LABEL_COL, WHITE_LABEL
from src.data_layers.raw_loader import load_bitcoinheist_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BitcoinHeist EDA tables and figures.")
    parser.add_argument("--csv-path", default="data/raw/BitcoinHeistData.csv")
    parser.add_argument("--output-dir", default="outputs/eda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frame = load_bitcoinheist_raw(args.csv_path)
    frame["target"] = (frame[LABEL_COL] != WHITE_LABEL).astype(int)

    year_summary = (
        frame.groupby("year")
        .agg(rows=("address", "size"), fraud_rows=("target", "sum"), fraud_rate=("target", "mean"))
        .reset_index()
    )
    label_summary = frame[LABEL_COL].value_counts().rename_axis("label").reset_index(name="rows")

    year_summary.to_parquet(tables_dir / "year_summary.parquet", index=False)
    label_summary.to_parquet(tables_dir / "label_summary.parquet", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(year_summary["year"], year_summary["fraud_rate"], marker="o")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraud rate")
    ax.set_title("BitcoinHeist fraud rate by year")
    fig.tight_layout()
    fig.savefig(figures_dir / "fraud_rate_by_year.png", dpi=160)
    plt.close(fig)

    top_labels = label_summary.head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_labels["label"][::-1], top_labels["rows"][::-1])
    ax.set_xlabel("Rows")
    ax.set_title("Top BitcoinHeist labels")
    fig.tight_layout()
    fig.savefig(figures_dir / "top_labels.png", dpi=160)
    plt.close(fig)

    print(f"eda_dir={output_dir}")


if __name__ == "__main__":
    main()
