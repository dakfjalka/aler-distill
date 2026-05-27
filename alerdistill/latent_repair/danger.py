from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DangerConfig:
    kl_reduction: str = "mean"
    ref_temperature: float = 1.0
    new_temperature: float = 1.0


def _kl_per_token(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    p_log = F.log_softmax(p_logits, dim=-1)
    q_log = F.log_softmax(q_logits, dim=-1)
    p_prob = p_log.exp()
    return (p_prob * (p_log - q_log)).sum(dim=-1)


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    kl = _kl_per_token(p_logits, q_logits)
    if reduction == "mean":
        return kl.mean()
    if reduction == "sum":
        return kl.sum()
    raise ValueError(f"Unknown reduction: {reduction}")


def danger_per_prompt(ref_last_logits: torch.Tensor, new_last_logits: torch.Tensor, cfg: DangerConfig) -> torch.Tensor:
    ref_logits = ref_last_logits / float(cfg.ref_temperature)
    new_logits = new_last_logits / float(cfg.new_temperature)
    return _kl_per_token(ref_logits, new_logits)
