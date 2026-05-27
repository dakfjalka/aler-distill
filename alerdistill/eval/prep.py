from __future__ import annotations

from typing import Any, Dict, Optional

from alerdistill.eval.registry import EvalContext, ensure_eval_modules_imported, get_eval_preparer
from alerdistill.eval.types import PreparedEvalSuite


def prepare_eval_suite(
    eval_cfg: Dict[str, Any],
    *,
    seed: int,
    datasets: Optional[Dict[str, Any]] = None,
    default_data_source: Optional[str] = None,
    run_dir: Optional[str] = None,
) -> PreparedEvalSuite:
    """Build an evaluation suite from config.

    Key design choices:
    - No dataset-specific special casing here.
    - Each dataset registers its own `kind` preparer in a separate module.
    - Evaluation routing is done by `data_source` via the evaluator registry.

    Args:
        eval_cfg: mapping from eval source name -> config dict.
        seed: random seed used by dataset preparers (if needed).
        datasets: optional extra in-memory datasets, keyed by `dataset_ref`.
                 This is used for `train_val` (prompt/completion datasets).
        default_data_source: if a preparer chooses to, it can default its
                 routing key to this value (e.g., train_val uses the training
                 data's format by default).
    """

    ensure_eval_modules_imported()

    ctx = EvalContext(seed=seed, datasets=datasets or {}, default_data_source=default_data_source, run_dir=run_dir)
    suite = PreparedEvalSuite(sources=[])

    for name, cfg_any in (eval_cfg or {}).items():
        # Hydra-style: allow a top-level `seed` key in the config section.
        if name in {"seed", "_seed"}:
            continue

        if not isinstance(cfg_any, dict):
            raise TypeError(f"eval_cfg[{name!r}] must be a dict, got {type(cfg_any)}")
        cfg: Dict[str, Any] = cfg_any

        if bool(cfg.get("enabled", True)) is False:
            continue

        kind = str(cfg.get("kind") or name)
        preparer = get_eval_preparer(kind)
        if preparer is None:
            raise KeyError(
                f"Unknown eval kind {kind!r} for source {name!r}. "
                "Make sure a module registered it with register_eval_preparer()."
            )

        src = preparer(name, cfg, ctx)
        suite.sources.append(src)

    return suite
