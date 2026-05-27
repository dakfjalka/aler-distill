from __future__ import annotations

from typing import Any, Dict, List, Optional

from datasets import Dataset

from .registry import register_train_preparer


def _norm_list(x) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return [str(i) for i in x]


def _filter_sciknoweval(example: Dict[str, Any], domains, types, levels) -> bool:
    if domains is not None:
        if str(example.get("domain")) not in domains:
            return False
    if types is not None:
        if str(example.get("type")) not in types:
            return False
    if levels is not None:
        details = example.get("details") or {}
        level = details.get("level") if isinstance(details, dict) else None
        if level is None:
            return False
        if str(level) not in levels:
            return False
    return True


def _format_sciknoweval_rlvr(example: Dict[str, Any]) -> Dict[str, str]:
    # Schema (based on dataset card): question(str), choices(dict{label,text}), answer(str), prompt(dict), ...
    question = example.get("question") or ""
    prompt_obj = example.get("prompt") or {}
    prefix = ""
    if isinstance(prompt_obj, dict):
        prefix = str(prompt_obj.get("default") or "")
    # choices
    choices = example.get("choices") or {}
    labels = choices.get("label") if isinstance(choices, dict) else None
    texts = choices.get("text") if isinstance(choices, dict) else None
    if labels is None or texts is None:
        # Fallback: some variants may use 'options'
        options = example.get("options") or []
        labels = [chr(ord("A") + i) for i in range(len(options))]
        texts = options

    lines = []
    for lb, tx in zip(labels, texts):
        lines.append(f"{lb}. {tx}")

    prompt_parts = []
    if prefix:
        prompt_parts.append(prefix.strip())
    if question:
        prompt_parts.append(question.strip())
    if lines:
        prompt_parts.append("\n".join(lines))
    # Keep output format consistent with other MCQA datasets in this repo:
    # the model must output a boxed option letter.
    prompt_parts.append("\nPlease give your final answer as \\box{X}.")

    prompt = "\n\n".join([p for p in prompt_parts if p])

    # Ground truth: SciKnowEval uses `answerKey` in the official schema.
    # Some derived variants may instead use `answer`.
    ans_any = example.get("answerKey")
    if ans_any is None or str(ans_any).strip() == "":
        ans_any = example.get("answer")

    ans = str(ans_any or "").strip()

    # If `answer` is provided as an index (common in some conversions), map it
    # back to the corresponding choice label.
    if ans.isdigit() and labels is not None:
        idx = int(ans)
        if 0 <= idx < len(labels):
            ans = str(labels[idx]).strip()

    ans = ans.strip().upper()
    completion = f"\\box{{{ans}}}"

    return {"prompt": prompt, "completion": completion}


@register_train_preparer("sciknoweval_rlvr")
def prepare_sciknoweval_rlvr(cfg, ds: Dataset) -> Dataset:
    """Prepare SciKnowEval in RLVR-friendly multiple-choice format.

    Configurable filters via cfg.filters:
      filters:
        domains: [Chemistry]
        types: null
        levels: [L-3]

    Defaults (when not provided): Chemistry + L-3.
    """
    filters = getattr(cfg, "filters", None) or {}
    domains = _norm_list(filters.get("domains", None))
    types = _norm_list(filters.get("types", None))
    levels = _norm_list(filters.get("levels", None))

    # Default selection: Chemistry L-3 (dataset may store as 'L-3' or 'L3'; we allow both by default)
    if domains is None:
        domains = ["Chemistry"]
    if levels is None:
        levels = ["L-3", "L3"]

    def _has_answer(ex: Dict[str, Any]) -> bool:
        k = ex.get("answerKey")
        if k is not None and str(k).strip() != "":
            return True
        a = ex.get("answer")
        return a is not None and str(a).strip() != ""

    ds2 = ds.filter(lambda ex: _filter_sciknoweval(ex, domains, types, levels) and _has_answer(ex))
    remove_cols = None if bool(getattr(cfg, "streaming", False)) else list(ds2.features)
    ds2 = ds2.map(_format_sciknoweval_rlvr, remove_columns=remove_cols, load_from_cache_file=False)
    return ds2
