from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from datasets import load_dataset

from alerdistill.data.train_prep.dispatch import prepare_train_dataset


@dataclass
class ValSplitConfig:
    """Hold-out validation split derived from the training split."""

    enabled: bool = False

    # split mode:
    # - "fraction": split by ratio (test_size=fraction)
    # - "count": split by an absolute number of examples (test_size=num_examples)
    mode: str = "fraction"

    # used when mode="fraction"
    fraction: float = 0.0

    # used when mode="count"
    num_examples: int | None = None

    seed: int = 42

    # If set, cap the validation split to at most this many examples (applied after splitting).
    max_examples: int | None = None


@dataclass
class TrainDataConfig:
    dataset_name: str
    dataset_config_name: Optional[str]
    split: str
    streaming: bool
    format: str
    text_field: Optional[str]
    max_length: int
    shuffle: bool
    seed: int
    take_n: Optional[int]
    chat_template_enabled: bool
    chat_template_name: Optional[str]

    # Optional per-dataset filters (used by some loaders, e.g. SciKnowEval).
    filters: Optional[dict[str, Any]] = None


def load_train_dataset(cfg: TrainDataConfig):
    ds = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config_name,
        split=cfg.split,
        streaming=cfg.streaming,
    )

    if cfg.shuffle:
        ds = ds.shuffle(seed=cfg.seed, buffer_size=10_000) if cfg.streaming else ds.shuffle(seed=cfg.seed)

    if cfg.take_n is not None:
        ds = ds.take(cfg.take_n) if cfg.streaming else ds.select(range(min(cfg.take_n, len(ds))))

    # Prepare/format into {prompt, completion}.
    ds = prepare_train_dataset(cfg, ds)
    return ds


def split_train_val(ds, split_cfg: ValSplitConfig):
    """Split an in-memory HF dataset into train/val.

    This runs AFTER formatting so the val set can be evaluated with the same
    prompt/completion fields as training.

    Notes:
    - Streaming datasets are not supported (would require reservoir sampling).
    - If disabled, returns (ds, None).
    - If split_cfg.max_examples is set, val is truncated for quick checks.
    """
    if (split_cfg is None) or (not split_cfg.enabled):
        return ds, None

    # Streaming datasets can't be split reliably here.
    if not hasattr(ds, "train_test_split"):
        raise ValueError("Validation split is not supported for streaming datasets.")

    mode = str(split_cfg.mode or "fraction").lower()
    if mode not in ("fraction", "count"):
        raise ValueError(f"val_split.mode must be one of: fraction|count, got {split_cfg.mode!r}.")

    if mode == "fraction":
        frac = float(split_cfg.fraction)
        if not (0.0 < frac < 1.0):
            raise ValueError(f"val_split.fraction must be in (0,1), got {frac}.")
        split = ds.train_test_split(test_size=frac, seed=int(split_cfg.seed), shuffle=True)
    else:
        if split_cfg.num_examples is None:
            raise ValueError("val_split.num_examples must be set when mode='count'.")
        n = int(split_cfg.num_examples)
        if n <= 0:
            return ds, None
        split = ds.train_test_split(test_size=n, seed=int(split_cfg.seed), shuffle=True)

    train_ds = split["train"]
    val_ds = split["test"]

    # Optional: truncate validation for quick checks.
    if split_cfg.max_examples is not None:
        n = int(split_cfg.max_examples)
        if n >= 0:
            n = min(n, len(val_ds))
            val_ds = val_ds.select(range(n))

    return train_ds, val_ds
