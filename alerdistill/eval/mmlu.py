from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from datasets import get_dataset_config_names, load_dataset

from alerdistill.eval.mcqa import CHOICE_LETTERS_4, _format_mc_question_box
from alerdistill.eval.registry import EvalContext, register_eval_preparer
from alerdistill.eval.types import EvalExample, PreparedEvalSource


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _normalize_subject(subject: Any) -> Optional[Union[str, List[str]]]:
    if subject is None:
        return None
    if isinstance(subject, str) and subject.lower() in {"*", "null", "none"}:
        return None
    return subject


@register_eval_preparer("mmlu")
def prepare_mmlu(name: str, cfg: Dict[str, Any], ctx: EvalContext) -> PreparedEvalSource:
    """Prepare MMLU (multiple-choice).

    Config keys (same as conf/data/eval/default.yaml):
      - dataset_name
      - split
      - subject: null|all|<one subject>|[subjects...]
      - max_examples (per subject)
      - batch_size
      - data_source (routing key); defaults to "mmlu"
    """

    data_source = str(cfg.get("data_source") or "mmlu")
    dataset_name = str(cfg.get("dataset_name") or "cais/mmlu")
    split = str(cfg.get("split") or "test")
    subject = _normalize_subject(cfg.get("subject"))
    max_examples = cfg.get("max_examples")
    batch_size = _as_int(cfg.get("batch_size"), 8)

    subjects: List[str]
    if subject is None:
        subjects = list(get_dataset_config_names(dataset_name))
    elif isinstance(subject, (list, tuple)):
        subjects = [str(s) for s in subject]
    else:
        subjects = [str(subject)]

    examples: List[EvalExample] = []
    for subj in subjects:
        ds = load_dataset(dataset_name, subj, split=split)
        idxs = list(range(len(ds)))
        if max_examples is not None:
            idxs = idxs[: int(max_examples)]
        for j in idxs:
            ex = ds[j]
            # Some HF wrappers nest in {"train": ...}; keep robust.
            if isinstance(ex, dict) and "train" in ex and isinstance(ex["train"], dict):
                ex = ex["train"]

            q = _format_mc_question_box(ex["question"], ex["choices"], CHOICE_LETTERS_4)
            label = ex["answer"]
            if isinstance(label, int):
                label = "ABCD"[label]
            else:
                label = str(label).strip().upper()

            examples.append(
                EvalExample(
                    prompt=q,
                    gold=label,
                    data_source=data_source,
                    extra={"subject": subj, "idx": int(j)},
                )
            )

    return PreparedEvalSource(
        name=name,
        data_source=data_source,
        batch_size=batch_size,
        examples=examples,
    )
