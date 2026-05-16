import numpy as np

from src.active_learning.base_strategy import QueryStrategy, clamp_budget


class MarginSampling(QueryStrategy):
    name = "margin"

    def select(self, unlabeled_idx: np.ndarray, budget: int, probs: np.ndarray | None = None, **kwargs) -> np.ndarray:
        if probs is None:
            raise ValueError("MarginSampling requires probabilities for unlabeled samples.")
        budget = clamp_budget(unlabeled_idx, budget)
        probs = np.asarray(probs)
        uncertainty = 1.0 - np.abs(2.0 * probs - 1.0)
        chosen_positions = np.argsort(uncertainty)[-budget:]
        return np.asarray(unlabeled_idx)[chosen_positions]
