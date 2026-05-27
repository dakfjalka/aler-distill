from __future__ import annotations

"""Train dataset preprocessing registry.

Goal: keep *each* training dataset's formatting / filtering in its own module,
while providing one stable dispatch function.

Each dataset module registers a callable under a unique `format` key.
"""

from typing import Any, Callable, Dict

from datasets import Dataset


TrainPrepFn = Callable[[Any, Dataset], Dataset]


TRAIN_PREPARERS: Dict[str, TrainPrepFn] = {}


def register_train_preparer(name: str):
    """Decorator to register a training preprocessor under `name`."""

    def _wrap(fn: TrainPrepFn) -> TrainPrepFn:
        if name in TRAIN_PREPARERS:
            raise KeyError(f"Duplicate train preprocessor registration: {name!r}")
        TRAIN_PREPARERS[name] = fn
        return fn

    return _wrap


def get_train_preparer(name: str) -> TrainPrepFn:
    if name not in TRAIN_PREPARERS:
        raise KeyError(
            f"Unknown train dataset format {name!r}. "
            f"Available: {sorted(TRAIN_PREPARERS.keys())}"
        )
    return TRAIN_PREPARERS[name]
