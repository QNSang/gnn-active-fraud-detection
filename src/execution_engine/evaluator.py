import torch

from src.evaluation.metric_manager import evaluate_mask


class Evaluator:
    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = device

    def evaluate(self, model, data, masks: dict[str, torch.Tensor]) -> dict[str, dict]:
        results = {}
        for split in ["train", "val", "test"]:
            key = f"{split}_mask"
            if key in masks and int(masks[key].sum()) > 0:
                results[split] = evaluate_mask(model, data, masks[key], self.device)
        return results
