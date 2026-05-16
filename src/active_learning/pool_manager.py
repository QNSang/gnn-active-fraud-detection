import numpy as np


class PoolManager:
    """Pool-based oracle simulator over train indices only."""

    def __init__(self, all_train_indices: np.ndarray, labels: np.ndarray, initial_labeled: np.ndarray) -> None:
        self.all_indices = np.asarray(all_train_indices, dtype=int)
        self.labels = np.asarray(labels)
        initial = set(np.asarray(initial_labeled, dtype=int).tolist())
        all_indices = set(self.all_indices.tolist())
        if not initial.issubset(all_indices):
            raise ValueError("initial_labeled must be a subset of all_train_indices.")
        self.labeled_idx = initial
        self.unlabeled_idx = all_indices - initial

    @classmethod
    def bootstrap(
        cls,
        all_train_indices: np.ndarray,
        labels: np.ndarray,
        initial_size: int,
        fraud_floor: int = 0,
        seed: int = 42,
    ) -> "PoolManager":
        rng = np.random.default_rng(seed)
        train_idx = np.asarray(all_train_indices, dtype=int)
        y = np.asarray(labels)
        positives = train_idx[y[train_idx] == 1]
        negatives = train_idx[y[train_idx] == 0]
        n_pos = min(fraud_floor, len(positives), initial_size)
        chosen_pos = rng.choice(positives, size=n_pos, replace=False) if n_pos else np.array([], dtype=int)
        remaining = np.setdiff1d(train_idx, chosen_pos, assume_unique=False)
        n_remaining = min(max(initial_size - n_pos, 0), len(remaining))
        chosen_rest = rng.choice(remaining, size=n_remaining, replace=False) if n_remaining else np.array([], dtype=int)
        return cls(train_idx, y, np.concatenate([chosen_pos, chosen_rest]))

    def query(self, indices: list[int] | np.ndarray) -> np.ndarray:
        selected = set(np.asarray(indices, dtype=int).tolist())
        if not selected.issubset(self.unlabeled_idx):
            raise ValueError("Cannot query samples that are already labeled or outside the pool.")
        self.labeled_idx.update(selected)
        self.unlabeled_idx.difference_update(selected)
        selected_array = np.asarray(sorted(selected), dtype=int)
        return self.labels[selected_array]

    @property
    def labeled_array(self) -> np.ndarray:
        return np.asarray(sorted(self.labeled_idx), dtype=int)

    @property
    def unlabeled_array(self) -> np.ndarray:
        return np.asarray(sorted(self.unlabeled_idx), dtype=int)

    def budget_used(self) -> int:
        return len(self.labeled_idx)
