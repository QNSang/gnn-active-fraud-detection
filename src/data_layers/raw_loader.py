from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_layers.bitcoinheist_schema import SCHEMA, TIME_COLS, validate_columns


def load_bitcoinheist_raw(csv_path: str | Path = "data/raw/BitcoinHeistData.csv") -> pd.DataFrame:
    """Load the BitcoinHeist CSV with schema validation and chronological ordering."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing BitcoinHeist CSV: {path}")

    frame = pd.read_csv(path, dtype=SCHEMA)
    validate_columns(set(frame.columns))
    return frame.sort_values(TIME_COLS).reset_index(drop=True)
