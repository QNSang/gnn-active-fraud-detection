from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def create_tabular_model(name: str, seed: int = 42, **kwargs):
    model_name = name.lower()
    if model_name in {"logreg", "logistic_regression"}:
        defaults = {
            "class_weight": "balanced",
            "max_iter": 1000,
            "n_jobs": -1,
            "random_state": seed,
        }
        defaults.update(kwargs)
        return LogisticRegression(**defaults)
    if model_name in {"random_forest", "rf"}:
        defaults = {
            "n_estimators": 300,
            "class_weight": "balanced_subsample",
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": seed,
        }
        defaults.update(kwargs)
        return RandomForestClassifier(**defaults)
    raise ValueError(f"Unknown tabular model: {name}")
