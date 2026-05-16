from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def binary_tabular_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else 0.0,
        "recall_at_05": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision_at_05": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1_at_05": float(f1_score(y_true, y_pred, zero_division=0)),
        "nodes": int(len(y_true)),
        "fraud_rate": float(y_true.mean()) if len(y_true) else 0.0,
    }


def evaluate_tabular_splits(model, dataset) -> dict[str, dict[str, float]]:
    splits = {
        "train": (dataset.x_train, dataset.y_train),
        "val": (dataset.x_val, dataset.y_val),
        "test": (dataset.x_test, dataset.y_test),
    }
    metrics = {}
    for split, (x_values, y_values) in splits.items():
        if len(y_values) == 0:
            continue
        y_prob = model.predict_proba(x_values)[:, 1]
        metrics[split] = binary_tabular_metrics(y_values, y_prob)
    return metrics
