from abc import ABC, abstractmethod

import numpy as np


class QueryStrategy(ABC):
    name: str

    @abstractmethod
    def select(
        self,
        unlabeled_idx: np.ndarray,
        budget: int,
        probs: np.ndarray | None = None,
        embeddings: np.ndarray | None = None,
        labeled_idx: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return absolute indices to query from the unlabeled pool."""


def clamp_budget(unlabeled_idx: np.ndarray, budget: int) -> int:
    if budget < 1:
        raise ValueError("budget must be positive.")
    return min(int(budget), int(len(unlabeled_idx)))
