from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class BinaryMetrics:
    pr_auc: float
    roc_auc: float
    recall_at_05: float
    precision_at_05: float
    f1_at_05: float


def _to_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return values


def binary_classification_metrics(
    y_true: torch.Tensor | np.ndarray,
    y_prob: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> BinaryMetrics:
    y_true_np = _to_numpy(y_true).astype(int)
    y_prob_np = _to_numpy(y_prob).astype(float)
    y_pred = (y_prob_np >= threshold).astype(int)

    try:
        from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

        pr_auc = float(average_precision_score(y_true_np, y_prob_np))
        roc_auc = float(roc_auc_score(y_true_np, y_prob_np)) if len(np.unique(y_true_np)) == 2 else 0.0
        recall = float(recall_score(y_true_np, y_pred, zero_division=0))
        precision = float(precision_score(y_true_np, y_pred, zero_division=0))
        f1 = float(f1_score(y_true_np, y_pred, zero_division=0))
    except ModuleNotFoundError:
        tp = float(((y_pred == 1) & (y_true_np == 1)).sum())
        fp = float(((y_pred == 1) & (y_true_np == 0)).sum())
        fn = float(((y_pred == 0) & (y_true_np == 1)).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        pr_auc = precision
        roc_auc = 0.0

    return BinaryMetrics(pr_auc, roc_auc, recall, precision, f1)


@torch.no_grad()
def evaluate_mask(model, data, mask: torch.Tensor, device: str | torch.device = "cpu") -> dict:
    model.eval()
    data = data.to(device)
    mask = mask.to(device)
    logits = model(data.x, data.edge_index)
    probs = torch.sigmoid(logits[mask])
    metrics = binary_classification_metrics(data.y[mask], probs)
    return metrics.__dict__
