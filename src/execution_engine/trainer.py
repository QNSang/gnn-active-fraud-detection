from dataclasses import dataclass
from pathlib import Path

import torch

from src.evaluation.checkpoint_manager import save_checkpoint
from src.evaluation.metric_manager import evaluate_mask


@dataclass(frozen=True)
class TrainResult:
    best_val_pr_auc: float
    best_checkpoint: str
    history: list[dict]


def compute_pos_weight(y: torch.Tensor, mask: torch.Tensor) -> float:
    labels = y[mask]
    positives = float((labels == 1).sum().item())
    negatives = float((labels == 0).sum().item())
    return negatives / max(positives, 1.0)


def train_full_batch(
    model,
    data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 10,
    pos_weight: float | None = None,
    device: str | torch.device = "cpu",
    run_dir: str | Path = "outputs/runs/smoke",
) -> TrainResult:
    """Train a binary node classifier with full-batch updates."""
    model = model.to(device)
    data = data.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    if pos_weight is None:
        pos_weight = compute_pos_weight(data.y, train_mask)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_pr_auc = -1.0
    best_path = ""
    stale_epochs = 0
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_mask], data.y[train_mask].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_metrics = evaluate_mask(model, data, val_mask, device)
        row = {"epoch": epoch, "loss": float(loss.item()), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)

        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            best_path = str(save_checkpoint(model, optimizer, epoch, best_pr_auc, run_dir))
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    return TrainResult(best_pr_auc, best_path, history)
