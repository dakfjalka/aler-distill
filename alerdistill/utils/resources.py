from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class GPUResources:
    """Resource split for this run."""

    train_gpus: int = 1
    infer_gpus: int = 1


def _list_physical_gpu_indices() -> List[str]:
    """Best-effort GPU index discovery without importing torch.

    Priority:
    1) CUDA_VISIBLE_DEVICES if set
    2) nvidia-smi index query
    3) fallback to ['0']
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() != "":
        # Could be comma separated list of ids.
        return [x.strip() for x in cvd.split(",") if x.strip() != ""]

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        idxs = [ln.strip() for ln in out.splitlines() if ln.strip()]
        return idxs or ["0"]
    except Exception:
        return ["0"]


def partition_gpus(resources: GPUResources) -> Tuple[str, str]:
    """Return (train_cvd, infer_cvd) CUDA_VISIBLE_DEVICES strings."""
    all_ids = _list_physical_gpu_indices()
    t = max(0, int(resources.train_gpus))
    i = max(0, int(resources.infer_gpus))

    if t + i > len(all_ids):
        # Be permissive: allocate as many as we can.
        t = min(t, len(all_ids))
        i = max(0, min(i, len(all_ids) - t))

    train_ids = all_ids[:t]
    infer_ids = all_ids[t : t + i]

    train_cvd = ",".join(train_ids) if train_ids else ""
    infer_cvd = ",".join(infer_ids) if infer_ids else ""
    return train_cvd, infer_cvd
