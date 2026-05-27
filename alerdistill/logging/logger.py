from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from torch.utils.tensorboard import SummaryWriter

@dataclass
class WandbConfig:
    enabled: bool = True
    entity: Optional[str] = None
    tags: Optional[list[str]] = None
    notes: Optional[str] = None
    mode: str = "online"  # online/offline/disabled

class MetricLogger:
    def __init__(self, log_dir: str, project: str, run_name: Optional[str], wandb_cfg: WandbConfig, tensorboard_enabled: bool = True):
        self.tb = None
        if tensorboard_enabled:
            os.makedirs(log_dir, exist_ok=True)
            self.tb = SummaryWriter(log_dir=log_dir)

        self._wandb = None
        self._wandb_run = None
        if wandb_cfg.enabled and wandb_cfg.mode != "disabled":
            import wandb
            self._wandb = wandb
            self._wandb_run = wandb.init(
                project=project,
                name=run_name,
                entity=wandb_cfg.entity,
                tags=wandb_cfg.tags,
                notes=wandb_cfg.notes,
                mode=wandb_cfg.mode,
            )

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self.tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, step)
        if self._wandb_run is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
