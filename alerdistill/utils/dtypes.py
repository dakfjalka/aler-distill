from __future__ import annotations
import torch

def resolve_dtype(dtype: str) -> torch.dtype:
    d = dtype.lower()
    if d in ("fp16","float16"):
        return torch.float16
    if d in ("bf16","bfloat16"):
        return torch.bfloat16
    if d in ("fp32","float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")
