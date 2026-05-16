import numpy as np

from src.active_learning.base_strategy import QueryStrategy, clamp_budget


class RandomSampling(QueryStrategy):
    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, unlabeled_idx: np.ndarray, budget: int, **kwargs) -> np.ndarray:
        budget = clamp_budget(unlabeled_idx, budget)
        return self.rng.choice(unlabeled_idx, size=budget, replace=False)
