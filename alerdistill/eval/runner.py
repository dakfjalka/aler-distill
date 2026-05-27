from __future__ import annotations

from typing import Any, Dict, Union

from alerdistill.eval.registry import ensure_eval_modules_imported, get_eval_evaluator
from alerdistill.eval.types import PreparedEvalSource, PreparedEvalSuite


def _run_prepared_source(
    engine,
    gen_cfg,
    src: PreparedEvalSource,
    *,
    seed: int,
    extra: Any = None,
) -> Dict[str, float]:
    """Route by `src.data_source` into different evaluator functions.

    Note: `src.name` is used as the metric prefix so a single routing key
    (e.g., multiple MCQA datasets) can still produce separate metrics.
    """

    ensure_eval_modules_imported()
    evaluator = get_eval_evaluator(src.data_source)
    if evaluator is None:
        raise KeyError(
            f"No evaluator registered for data_source={src.data_source!r} (source name={src.name!r}). "
            "Register it via register_eval_evaluator() in a module under alerdistill.eval."
        )
    return evaluator(engine, gen_cfg, src, extra=extra)


def run_all_evals(
    engine,
    gen_cfg,
    eval_suite: Union[Dict[str, Any], PreparedEvalSuite],
    seed: int,
    extra: Any = None,
) -> Dict[str, float]:
    """Run evaluations.

    - If given a PreparedEvalSuite, uses preprocessed datasets (recommended).
    - If given a config dict, falls back to legacy behavior that loads datasets at eval time.
    """

    out: Dict[str, float] = {}

    if isinstance(eval_suite, PreparedEvalSuite):
        for src in eval_suite.sources:
            out.update(_run_prepared_source(engine, gen_cfg, src, seed=seed, extra=extra))
            print("Eval metrics:", out)
        return out
    else:
        raise NotImplementedError
