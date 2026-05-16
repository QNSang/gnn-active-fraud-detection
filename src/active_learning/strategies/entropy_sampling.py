import numpy as np

from src.active_learning.base_strategy import QueryStrategy, clamp_budget


class EntropySampling(QueryStrategy):
    name = "entropy"

    def select(self, unlabeled_idx: np.ndarray, budget: int, probs: np.ndarray | None = None, **kwargs) -> np.ndarray:
        if probs is None:
            raise ValueError("EntropySampling requires probabilities for unlabeled samples.")
        budget = clamp_budget(unlabeled_idx, budget)
        probs = np.asarray(probs).clip(1e-10, 1.0 - 1e-10)
        entropy = -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs))
        chosen_positions = np.argsort(entropy)[-budget:]
        return np.asarray(unlabeled_idx)[chosen_positions]
