from __future__ import annotations

from dataclasses import dataclass


SCHEMA = {
    "address": str,
    "year": int,
    "day": int,
    "length": float,
    "weight": float,
    "count": float,
    "looped": float,
    "neighbors": float,
    "income": float,
    "label": str,
}

FEATURE_COLS = ["year", "day", "length", "weight", "count", "looped", "neighbors", "income"]
LOG1P_COLS = ["income", "count", "looped", "length", "weight"]
TIME_COLS = ["year", "day"]
LABEL_COL = "label"
ADDRESS_COL = "address"
TARGET_COL = "target"
WHITE_LABEL = "white"
FRAUD_LABEL = "fraud"


@dataclass(frozen=True)
class BitcoinHeistSplitConfig:
    train_end_year: int = 2015
    val_end_year: int = 2016


def validate_columns(columns: list[str] | set[str]) -> None:
    observed = set(columns)
    expected = set(SCHEMA)
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise ValueError(f"Schema mismatch. missing={sorted(missing)} extra={sorted(extra)}")


def encode_binary_label(label: str) -> int:
    return 0 if label == WHITE_LABEL else 1
