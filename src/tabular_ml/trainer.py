from __future__ import annotations

from dataclasses import dataclass

from src.tabular_ml.evaluator import evaluate_tabular_splits


@dataclass(frozen=True)
class TabularTrainResult:
    model: object
    metrics: dict[str, dict[str, float]]


def train_tabular_model(model, dataset) -> TabularTrainResult:
    model.fit(dataset.x_train, dataset.y_train)
    metrics = evaluate_tabular_splits(model, dataset)
    return TabularTrainResult(model=model, metrics=metrics)
