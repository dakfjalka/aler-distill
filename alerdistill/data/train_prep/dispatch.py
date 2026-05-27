from __future__ import annotations

from datasets import Dataset

from .registry import get_train_preparer

# Import all dataset modules for side-effect registration.
# This avoids maintaining a hard-coded list (no dataset-specific special casing).
import importlib
import pkgutil

from . import __path__ as _PKG_PATH

for m in pkgutil.iter_modules(_PKG_PATH):
    if m.name in {"registry", "dispatch"}:
        continue
    importlib.import_module(f"{__package__}.{m.name}")


def prepare_train_dataset(cfg, ds: Dataset) -> Dataset:
    """Apply dataset-specific preprocessing based on `cfg.format`.

    All dataset-specific logic should live in separate files under
    `alerdistill/data/train_prep/`.
    """

    fmt = str(getattr(cfg, "format", "sciknoweval_rlvr"))
    preparer = get_train_preparer(fmt)
    return preparer(cfg, ds)
