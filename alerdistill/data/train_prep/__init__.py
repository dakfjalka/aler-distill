"""Training dataset preparation (formatting + optional filtering).

Keep per-dataset logic in separate modules; `dispatch.prepare_train_dataset` is the
single entry point used by the training pipeline.
"""

from .dispatch import prepare_train_dataset

__all__ = ["prepare_train_dataset"]
