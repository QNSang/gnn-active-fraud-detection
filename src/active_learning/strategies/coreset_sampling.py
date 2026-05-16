import numpy as np

from src.active_learning.base_strategy import QueryStrategy, clamp_budget


class CoreSetSampling(QueryStrategy):
    name = "coreset"

    def select(
        self,
        unlabeled_idx: np.ndarray,
        budget: int,
        embeddings: np.ndarray | None = None,
        labeled_idx: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        if embeddings is None or labeled_idx is None:
            raise ValueError("CoreSetSampling requires embeddings and labeled_idx.")
        budget = clamp_budget(unlabeled_idx, budget)
        unlabeled_idx = np.asarray(unlabeled_idx, dtype=int)
        labeled_idx = np.asarray(labeled_idx, dtype=int)
        emb = np.asarray(embeddings)

        labeled_embeddings = emb[labeled_idx]
        unlabeled_embeddings = emb[unlabeled_idx]
        min_distances = ((unlabeled_embeddings[:, None, :] - labeled_embeddings[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        chosen_positions = np.argsort(min_distances)[-budget:]
        return unlabeled_idx[chosen_positions]
