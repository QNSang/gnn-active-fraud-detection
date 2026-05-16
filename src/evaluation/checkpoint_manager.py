from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    best_val_pr_auc: float,
    run_dir: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_val_pr_auc": best_val_pr_auc,
        "model_config": model.get_config() if hasattr(model, "get_config") else {},
    }
    if extra:
        payload.update(extra)
    path = run_path / "best_checkpoint.pt"
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, model=None, optimizer=None, map_location: str = "cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
