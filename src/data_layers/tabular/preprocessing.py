from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_layers.bitcoinheist_schema import (
    FEATURE_COLS,
    LABEL_COL,
    LOG1P_COLS,
    TARGET_COL,
    encode_binary_label,
)


@dataclass(frozen=True)
class TabularDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    masks: dict[str, np.ndarray]
    feature_cols: list[str]
    scaler: StandardScaler
    frame: pd.DataFrame


def add_binary_target(frame: pd.DataFrame) -> pd.DataFrame:
    processed = frame.copy()
    processed[TARGET_COL] = processed[LABEL_COL].map(encode_binary_label).astype(int)
    return processed


def apply_log1p_features(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    processed = frame.copy()
    for col in columns or LOG1P_COLS:
        processed[col] = np.log1p(processed[col].clip(lower=0))
    return processed


def build_year_splits(frame: pd.DataFrame, train_end_year: int, val_end_year: int) -> dict[str, np.ndarray]:
    years = frame["year"].to_numpy()
    return {
        "train": years <= train_end_year,
        "val": (years > train_end_year) & (years <= val_end_year),
        "test": years > val_end_year,
    }


def prepare_tabular_dataset(
    frame: pd.DataFrame,
    train_end_year: int = 2015,
    val_end_year: int = 2016,
    feature_cols: list[str] | None = None,
    scale_features: bool = True,
) -> TabularDataset:
    """Prepare leakage-aware tabular arrays from raw BitcoinHeist rows."""
    feature_cols = feature_cols or FEATURE_COLS
    processed = add_binary_target(frame)
    processed = apply_log1p_features(processed)
    masks = build_year_splits(processed, train_end_year, val_end_year)

    x = processed[feature_cols].to_numpy(dtype=np.float32)
    y = processed[TARGET_COL].to_numpy(dtype=np.int64)

    scaler = StandardScaler()
    if scale_features:
        x_train = scaler.fit_transform(x[masks["train"]])
        x = x.copy()
        x[masks["train"]] = x_train
        if int(masks["val"].sum()) > 0:
            x[masks["val"]] = scaler.transform(x[masks["val"]])
        if int(masks["test"].sum()) > 0:
            x[masks["test"]] = scaler.transform(x[masks["test"]])
    else:
        scaler.fit(x[masks["train"]])

    return TabularDataset(
        x_train=x[masks["train"]],
        y_train=y[masks["train"]],
        x_val=x[masks["val"]],
        y_val=y[masks["val"]],
        x_test=x[masks["test"]],
        y_test=y[masks["test"]],
        masks=masks,
        feature_cols=list(feature_cols),
        scaler=scaler,
        frame=processed,
    )
