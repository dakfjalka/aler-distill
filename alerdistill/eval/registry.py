from __future__ import annotations

"""Eval registry.

Design goals:
- No central hard-coded routing tables.
- Adding a dataset/evaluator should only require creating a new module that
  registers itself.

Two independent registries:
- *preparer registry* keyed by eval config `kind` (how to build EvalExample list)
- *evaluator registry* keyed by `data_source` (how to score model outputs)

`PreparedEvalSource.name` is always the metric prefix (so multiple sources can
share the same data_source/evaluator).
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from alerdistill.eval.types import PreparedEvalSource


@dataclass(frozen=True)
class EvalContext:
    """Context passed to all eval preparers."""

    seed: int
    datasets: Dict[str, Any]
    default_data_source: Optional[str] = None
    run_dir: Optional[str] = None


EvalPreparer = Callable[[str, Dict[str, Any], EvalContext], PreparedEvalSource]
EvalEvaluator = Callable[[Any, Any, PreparedEvalSource], Dict[str, float]]


_PREPARERS: Dict[str, EvalPreparer] = {}
_EVALUATORS: Dict[str, EvalEvaluator] = {}


def register_eval_preparer(kind: str):
    kind = str(kind)

    def _wrap(fn: EvalPreparer) -> EvalPreparer:
        if kind in _PREPARERS:
            raise ValueError(f"Duplicate eval preparer kind={kind!r} registered by {fn!r}.")
        _PREPARERS[kind] = fn
        return fn

    return _wrap


def register_eval_evaluator(data_source: str):
    data_source = str(data_source)

    def _wrap(fn: EvalEvaluator) -> EvalEvaluator:
        if data_source in _EVALUATORS:
            raise ValueError(f"Duplicate eval evaluator data_source={data_source!r} registered by {fn!r}.")
        _EVALUATORS[data_source] = fn
        return fn

    return _wrap


def get_eval_preparer(kind: str) -> Optional[EvalPreparer]:
    return _PREPARERS.get(str(kind))


def get_eval_evaluator(data_source: str) -> Optional[EvalEvaluator]:
    return _EVALUATORS.get(str(data_source))


def list_eval_preparers() -> Dict[str, EvalPreparer]:
    return dict(_PREPARERS)


def list_eval_evaluators() -> Dict[str, EvalEvaluator]:
    return dict(_EVALUATORS)


# --- auto-import helpers ---

_IMPORTED = False


def ensure_eval_modules_imported() -> None:
    """Import all alerdistill.eval.* modules once so decorators run.

    We intentionally import the *whole* package subtree under alerdistill.eval
    because:
    - it's small
    - it avoids maintaining an allowlist
    """

    global _IMPORTED
    if _IMPORTED:
        return

    import importlib
    import pkgutil

    import alerdistill.eval as _pkg

    for m in pkgutil.walk_packages(_pkg.__path__, prefix=_pkg.__name__ + "."):
        # Avoid importing bytecode caches or local tooling if any.
        name = m.name
        if name.endswith(".__pycache__"):
            continue
        importlib.import_module(name)

    _IMPORTED = True
