import numpy as np
import torch

from src.active_learning.pool_manager import PoolManager
from src.evaluation.metric_manager import evaluate_mask
from src.execution_engine.trainer import train_full_batch


@torch.no_grad()
def _score_unlabeled(model, data, unlabeled_idx: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    data = data.to(device)
    logits = model(data.x, data.edge_index)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    embeddings = model.get_embeddings(data.x, data.edge_index).detach().cpu().numpy()
    return probs[unlabeled_idx], embeddings


def run_active_learning(
    model_factory,
    data,
    pool: PoolManager,
    strategy,
    val_mask: torch.Tensor,
    rounds: int = 5,
    query_budget: int = 32,
    train_kwargs: dict | None = None,
    device: str = "cpu",
) -> list[dict]:
    history: list[dict] = []
    train_kwargs = train_kwargs or {}

    for round_idx in range(rounds):
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[torch.as_tensor(pool.labeled_array, dtype=torch.long)] = True
        model = model_factory()
        train_full_batch(model, data, train_mask, val_mask, device=device, **train_kwargs)

        metrics = evaluate_mask(model, data, val_mask, device)
        row = {
            "round": round_idx,
            "budget_used": pool.budget_used(),
            "strategy": strategy.name,
            **metrics,
        }
        history.append(row)

        if round_idx == rounds - 1 or len(pool.unlabeled_array) == 0:
            break

        probs, embeddings = _score_unlabeled(model, data, pool.unlabeled_array, device)
        queried = strategy.select(
            pool.unlabeled_array,
            query_budget,
            probs=probs,
            embeddings=embeddings,
            labeled_idx=pool.labeled_array,
        )
        pool.query(queried)

    return history
