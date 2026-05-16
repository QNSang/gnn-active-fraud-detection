from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.config import save_yaml
from src.utils.io import write_json


def make_run_id(name: str, timestamp: datetime | None = None) -> str:
    current = timestamp or datetime.now()
    safe_name = name.replace(" ", "_").replace("/", "_").lower()
    return f"{current:%Y%m%d_%H%M%S_%f}_{safe_name}"


class RunManager:
    def __init__(self, root_dir: str | Path, run_name: str) -> None:
        self.root_dir = Path(root_dir)
        self.run_name = run_name
        self.run_id = make_run_id(run_name)
        self.run_dir = self.root_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def path(self, *parts: str) -> Path:
        return self.run_dir.joinpath(*parts)

    def save_config(self, config: dict[str, Any]) -> Path:
        return save_yaml(config, self.path("config.yaml"))

    def save_metrics(self, metrics: dict[str, Any]) -> Path:
        return write_json(metrics, self.path("metrics.json"))

    def save_history(self, rows: list[dict[str, Any]], filename: str = "history.csv") -> Path | None:
        if not rows:
            return None
        target = self.path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return target
