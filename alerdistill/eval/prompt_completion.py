from __future__ import annotations

import re
from typing import Any, Dict, List

from alerdistill.eval.generation import generate_batched
from alerdistill.eval.registry import EvalContext, register_eval_evaluator, register_eval_preparer
from alerdistill.eval.types import EvalExample, PreparedEvalSource


def _norm_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\r", "")
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _token_f1(pred: str, gold: str) -> float:
    p = _norm_text(pred).split()
    g = _norm_text(gold).split()
    if len(p) == 0 and len(g) == 0:
        return 1.0
    if len(p) == 0 or len(g) == 0:
        return 0.0
    # bag-of-words overlap
    from collections import Counter

    pc = Counter(p)
    gc = Counter(g)
    common = pc & gc
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(1, len(p))
    recall = num_same / max(1, len(g))
    return (2 * precision * recall) / max(1e-12, precision + recall)


@register_eval_preparer("prompt_completion")
def prepare_prompt_completion_source(name: str, cfg: Dict[str, Any], ctx: EvalContext) -> PreparedEvalSource:
    """Prepare any prompt/completion dataset as an eval source.

    This is used for `train_val`, and also works for any dataset that already
    contains columns:
      - prompt: str
      - completion: str

    Config keys:
      - dataset_ref: required; key into ctx.datasets
      - max_examples: optional
      - batch_size: optional
      - data_source: optional; if null, falls back to ctx.default_data_source
    """

    dataset_ref = cfg.get("dataset_ref")
    if dataset_ref is None:
        raise ValueError(f"{name}: prompt_completion requires cfg.dataset_ref")

    if dataset_ref not in ctx.datasets:
        raise KeyError(
            f"{name}: dataset_ref={dataset_ref!r} not provided. Available dataset refs: {sorted(ctx.datasets.keys())}"
        )

    ds = ctx.datasets[dataset_ref]
    data_source = cfg.get("data_source")
    if data_source is None:
        if ctx.default_data_source is None:
            data_source = "prompt_completion"
        else:
            data_source = ctx.default_data_source
    data_source = str(data_source)

    max_examples = cfg.get("max_examples")
    batch_size = int(cfg.get("batch_size") or 8)

    # datasets.Dataset supports __len__ and __getitem__. For robustness, we
    # also accept a plain list[dict].
    n = len(ds)
    idxs = list(range(n))
    if max_examples is not None:
        idxs = idxs[: int(max_examples)]

    examples: List[EvalExample] = []
    for i in idxs:
        row = ds[i]
        prompt = row.get("prompt")
        completion = row.get("completion")
        if prompt is None or completion is None:
            raise ValueError(
                f"{name}: dataset_ref={dataset_ref!r} must contain 'prompt' and 'completion' columns, got keys={list(row.keys())}"
            )
        examples.append(EvalExample(prompt=str(prompt), gold=completion, data_source=data_source, extra={"idx": int(i)}))

    return PreparedEvalSource(name=name, data_source=data_source, batch_size=batch_size, examples=examples)


def eval_prompt_completion_examples(
    engine,
    gen_cfg,
    examples: List[EvalExample],
    *,
    batch_size: int,
    metric_prefix: str = "prompt_completion",
    print_firsts: bool = True,
) -> Dict[str, float]:
    """Evaluate prompt->completion generation against gold completions.

    We evaluate by generation and compare to the gold completion using:
      - exact match (normalized)
      - token F1 (bag-of-words)

    For inspection, we also print the first exact-match-correct and first
    exact-match-wrong example (if any).
    """
    if len(examples) == 0:
        return {
            f"{metric_prefix}/exact_match": 0.0,
            f"{metric_prefix}/token_f1": 0.0,
            f"{metric_prefix}/n": 0.0,
        }

    prompts = [ex.prompt for ex in examples]
    outs = generate_batched(engine, prompts, gen_cfg, batch_size=batch_size)

    em = 0
    f1_sum = 0.0
    em_mask: List[bool] = []

    for out, ex in zip(outs, examples):
        pred = _norm_text(out)
        gold = _norm_text(str(ex.gold))
        ok = pred == gold
        em_mask.append(ok)
        if ok:
            em += 1
        f1_sum += _token_f1(pred, gold)

    if print_firsts:
        first_ok = None
        first_bad = None
        for i, ok in enumerate(em_mask):
            if ok and first_ok is None:
                first_ok = i
            if (not ok) and first_bad is None:
                first_bad = i
            if first_ok is not None and first_bad is not None:
                break

        def _dump(i: int, tag: str):
            ex = examples[i]
            print(f"\n[{metric_prefix}] {tag} example idx={i}")
            if ex.extra:
                print("extra:", ex.extra)
            print("PROMPT:\n", ex.prompt)
            print("RESPONSE:\n", outs[i])
            print("GOLD:\n", ex.gold)

        if first_ok is not None:
            _dump(first_ok, "FIRST_CORRECT")
        if first_bad is not None:
            _dump(first_bad, "FIRST_WRONG")

    n = len(examples)
    return {
        f"{metric_prefix}/exact_match": float(em) / float(max(1, n)),
        f"{metric_prefix}/token_f1": float(f1_sum) / float(max(1, n)),
        f"{metric_prefix}/n": float(n),
    }


# ---------------- registry-based routing ----------------


@register_eval_evaluator("prompt_completion")
def eval_prompt_completion_source(engine, gen_cfg, src, extra=None):
    return eval_prompt_completion_examples(
        engine,
        gen_cfg,
        src.examples,
        batch_size=src.batch_size,
        metric_prefix=src.name,
    )
