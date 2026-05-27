from __future__ import annotations

from typing import Any, Dict, List

from datasets import load_dataset

from alerdistill.eval.mcqa import CHOICE_LETTERS_10, _format_mc_question_box
from alerdistill.eval.registry import EvalContext, register_eval_preparer
from alerdistill.eval.types import EvalExample, PreparedEvalSource


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


@register_eval_preparer("mmlu_pro")
def prepare_mmlu_pro(name: str, cfg: Dict[str, Any], ctx: EvalContext) -> PreparedEvalSource:
    """Prepare MMLU-Pro (10-choice multiple-choice)."""

    data_source = str(cfg.get("data_source") or "mmlu_pro")
    dataset_name = str(cfg.get("dataset_name") or "TIGER-Lab/MMLU-Pro")
    split = str(cfg.get("split") or "test")
    max_examples = cfg.get("max_examples")
    batch_size = _as_int(cfg.get("batch_size"), 8)
    use_cot = bool(cfg.get("use_cot", False))

    ds = load_dataset(dataset_name, split=split)
    idxs = list(range(len(ds)))
    if max_examples is not None:
        idxs = idxs[: int(max_examples)]

    examples: List[EvalExample] = []
    for j in idxs:
        ex = ds[j]
        q = _format_mc_question_box(ex["question"], ex["options"], CHOICE_LETTERS_10)
        if use_cot:
            q = q + "\nYou may show reasoning, but the final answer MUST be in \\box{X}."
        a = CHOICE_LETTERS_10[int(ex["answer_index"]) ]
        examples.append(EvalExample(prompt=q, gold=a, data_source=data_source, extra={"idx": int(j)}))

    return PreparedEvalSource(name=name, data_source=data_source, batch_size=batch_size, examples=examples)
