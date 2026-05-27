from __future__ import annotations

from typing import Any, Dict, List, Optional

from datasets import load_dataset

from alerdistill.eval.registry import EvalContext, register_eval_preparer
from alerdistill.eval.types import EvalExample, PreparedEvalSource


def _norm_list(x) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return [str(i) for i in x]


def _filter(ex: Dict[str, Any], domains, types, levels) -> bool:
    if domains is not None and str(ex.get("domain")) not in domains:
        return False
    if types is not None and str(ex.get("type")) not in types:
        return False
    if levels is not None:
        details = ex.get("details") or {}
        lvl = details.get("level") if isinstance(details, dict) else None
        if lvl is None:
            return False
        if str(lvl) not in levels:
            return False
    return True


def _extract_choices(ex: Dict[str, Any]) -> tuple[List[str], List[str]]:
    choices = ex.get("choices") or {}
    if isinstance(choices, dict) and "label" in choices and "text" in choices:
        labels = [str(x).strip() for x in choices["label"]]
        texts = [str(x) for x in choices["text"]]
        return labels, texts

    options = ex.get("options") or ex.get("choices") or []
    if isinstance(options, dict):
        if "text" in options:
            options = options["text"]
        else:
            options = [options[k] for k in sorted(options.keys())]
    if not isinstance(options, (list, tuple)):
        raise ValueError(f"SciKnowEval: unsupported options format: {type(options)}")

    texts = [str(o) for o in options]
    labels = [chr(ord("A") + i) for i in range(len(texts))]
    return labels, texts


def _extract_answer_key(ex: Dict[str, Any], labels: List[str]) -> str:
    ans_any = ex.get("answerKey")
    if ans_any is None or str(ans_any).strip() == "":
        ans_any = ex.get("answer")

    ans = str(ans_any or "").strip()
    if ans.isdigit():
        idx = int(ans)
        if 0 <= idx < len(labels):
            ans = labels[idx]
    ans = ans.strip().upper()
    return ans


def _format_prompt(ex: Dict[str, Any], labels: List[str], texts: List[str]) -> str:
    question = str(ex.get("question") or "")
    prompt_obj = ex.get("prompt") or {}
    prefix = str(prompt_obj.get("default") or "") if isinstance(prompt_obj, dict) else ""

    lines = [f"{lb}. {tx}" for lb, tx in zip(labels, texts)]

    parts: List[str] = []
    if prefix.strip():
        parts.append(prefix.strip())
    if question.strip():
        parts.append(question.strip())
    if lines:
        parts.append("\n".join(lines))
    parts.append("\nPlease give your final answer as \\box{X}.")
    return "\n\n".join([p for p in parts if p])


@register_eval_preparer("sciknoweval")
def prepare_sciknoweval(name: str, cfg: Dict[str, Any], ctx: EvalContext) -> PreparedEvalSource:
    """Prepare SciKnowEval as an eval source.

    Default filters match the training formatter: Chemistry + levels L-3/L3.
    """

    data_source = str(cfg.get("data_source") or "sciknoweval")
    dataset_name = str(cfg.get("dataset_name") or "hicai-zju/SciKnowEval")
    split = str(cfg.get("split") or "test")
    max_examples = cfg.get("max_examples")
    batch_size = int(cfg.get("batch_size") or 8)

    filters = cfg.get("filters") or {}
    domains = _norm_list(filters.get("domains"))
    types = _norm_list(filters.get("types"))
    levels = _norm_list(filters.get("levels"))

    if domains is None:
        domains = ["Chemistry"]
    if levels is None:
        levels = ["L-3", "L3"]

    ds = load_dataset(dataset_name, split=split)
    idxs = list(range(len(ds)))
    if max_examples is not None:
        idxs = idxs[: int(max_examples)]

    examples: List[EvalExample] = []
    for j in idxs:
        ex = ds[j]
        if not _filter(ex, domains, types, levels):
            continue

        labels, texts = _extract_choices(ex)
        # Only keep 4-choice MCQA for now.
        if len(labels) < 4 or len(texts) < 4:
            continue
        labels, texts = labels[:4], texts[:4]

        gold = _extract_answer_key(ex, labels)
        if gold not in {x.upper() for x in labels}:
            continue

        prompt = _format_prompt(ex, labels, texts)
        examples.append(EvalExample(prompt=prompt, gold=gold, data_source=data_source, extra={"idx": int(j)}))

    return PreparedEvalSource(name=name, data_source=data_source, batch_size=batch_size, examples=examples)
