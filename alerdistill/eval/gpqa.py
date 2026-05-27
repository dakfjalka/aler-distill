from __future__ import annotations

from typing import Any, Dict, List

from datasets import load_dataset

from alerdistill.eval.mcqa import CHOICE_LETTERS_4, _format_mc_question_box
from alerdistill.eval.registry import EvalContext, register_eval_preparer
from alerdistill.eval.types import EvalExample, PreparedEvalSource


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


@register_eval_preparer("gpqa")
def prepare_gpqa(name: str, cfg: Dict[str, Any], ctx: EvalContext) -> PreparedEvalSource:
    """Prepare GPQA (multiple-choice) evaluation set.

    We intentionally only support RLVR-friendly multiple-choice formats.
    """

    data_source = str(cfg.get("data_source") or "gpqa")
    dataset_name = str(cfg.get("dataset_name") or "casimiir/gpqa")
    config_name = cfg.get("config_name", None)
    split = str(cfg.get("split") or "test")
    max_examples = cfg.get("max_examples", None)
    batch_size = _as_int(cfg.get("batch_size"), 8)

    if config_name is None or str(config_name).lower() in {"", "none", "null"}:
        ds = load_dataset(dataset_name, split=split)
    else:
        ds = load_dataset(dataset_name, str(config_name), split=split)

    idxs = list(range(len(ds)))
    if max_examples is not None:
        idxs = idxs[: int(max_examples)]

    examples: List[EvalExample] = []
    for j in idxs:
        ex = ds[j]
        question = str(ex.get("question", ex.get("prompt", "")))

        options = ex.get("choices", None)
        if options is None and "incorrect_answers" in ex and "correct_answer" in ex:
            options = list(ex["incorrect_answers"]) + [ex["correct_answer"]]
        if options is None and "options" in ex:
            options = ex["options"]

        if isinstance(options, dict):
            if "text" in options:
                options = list(options["text"])
            else:
                options = [options[k] for k in sorted(options.keys())]
        if not isinstance(options, (list, tuple)):
            raise ValueError(f"GPQA: unsupported options format: {type(options)}")

        options = [str(o) for o in options]
        if len(options) < 4:
            raise ValueError(f"GPQA: expected >=4 options, got {len(options)}")

        # Only use first 4 to keep RLVR MCQA.
        options = options[:4]
        prompt = _format_mc_question_box(question, options, CHOICE_LETTERS_4)

        gold_letter = None
        if "correct_answer_idx" in ex:
            gold_letter = CHOICE_LETTERS_4[int(ex["correct_answer_idx"])]
        elif "answer_idx" in ex:
            gold_letter = CHOICE_LETTERS_4[int(ex["answer_idx"])]
        elif "answer" in ex:
            a = ex["answer"]
            if isinstance(a, int):
                gold_letter = CHOICE_LETTERS_4[int(a)]
            else:
                s = str(a).strip()
                if s.isdigit():
                    gold_letter = CHOICE_LETTERS_4[int(s)]
                else:
                    gold_letter = s.upper()[:1]
        elif "correct_answer" in ex:
            ca = str(ex["correct_answer"])
            try:
                gold_letter = CHOICE_LETTERS_4[options.index(ca)]
            except ValueError:
                gold_letter = None

        if gold_letter not in CHOICE_LETTERS_4:
            raise ValueError(f"GPQA: cannot infer gold label for example idx={j}")

        extra = {}
        for k in ("id", "subset", "split", "domain", "subdomain", "difficulty"):
            if k in ex:
                extra[k] = ex[k]
        extra["idx"] = int(j)

        examples.append(EvalExample(prompt=prompt, gold=gold_letter, data_source=data_source, extra=extra))

    return PreparedEvalSource(name=name, data_source=data_source, batch_size=batch_size, examples=examples)
