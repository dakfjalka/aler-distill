from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from alerdistill.eval.generation import generate_batched
from alerdistill.eval.types import EvalExample
from alerdistill.eval.registry import (
    EvalContext,
    register_eval_evaluator,
    register_eval_preparer,
)
from alerdistill.utils import ifeval


@register_eval_preparer("ifeval")
def prepare_ifeval_source(name: str, cfg: Dict[str, Any], ctx: EvalContext):
    """Prepare IFEval evaluation set (instruction-following).

    The dataset is RLVR-friendly: evaluation uses deterministic heuristics
    from the official `instruction_following_eval` repo (no judge model).
    """

    # Local imports to avoid extra dependency at module import time.
    from datasets import load_dataset

    from alerdistill.eval.types import PreparedEvalSource

    data_source = str(cfg.get("data_source") or "ifeval")
    dataset_name = str(cfg.get("dataset_name") or "google/IFEval")
    split = str(cfg.get("split") or "train")
    max_examples = cfg.get("max_examples")
    batch_size = int(cfg.get("batch_size") or 8)

    ds = load_dataset(dataset_name, split=split)
    idxs = list(range(len(ds)))
    if max_examples is not None:
        idxs = idxs[: int(max_examples)]

    eval_examples: List[EvalExample] = []
    for j in idxs:
        ex = ds[j]
        prompt = str(ex.get("prompt", ""))
        extra = {
            "key": ex.get("key", j),
            "instruction_id_list": ex.get("instruction_id_list", []),
            "kwargs": ex.get("kwargs", []),
            "idx": int(j),
        }
        eval_examples.append(
            EvalExample(prompt=prompt, gold=None, data_source=data_source, extra=extra)
        )

    return PreparedEvalSource(
        name=name,
        data_source=data_source,
        batch_size=batch_size,
        examples=eval_examples,
    )


def eval_ifeval_examples(
    engine,
    gen_cfg,
    examples: List[EvalExample],
    *,
    batch_size: int,
    metric_prefix: str = "ifeval",
    print_firsts: bool = True,
) -> Dict[str, float]:
    """
    strict
    """
    if len(examples) == 0:
        return {f"{metric_prefix}/n": 0.0}

    prompts = [ex.prompt for ex in examples]
    outs = generate_batched(engine, prompts, gen_cfg, batch_size=batch_size)

    correct = 0
    correct_mask: List[bool] = []

    for out, ex in zip(outs, examples):
        is_following_list = []
        kwargs_list = ex.extra.get("kwargs", [])
        for index, instruction_id in enumerate(ex.extra.get("instruction_id_list", [])):
            instruction_cls = ifeval.instructions_registry.INSTRUCTION_DICT[
                instruction_id
            ]
            instruction: ifeval.Instruction = instruction_cls(instruction_id)

            kwargs: dict = kwargs_list[index]
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
            instruction.build_description(**kwargs)
            args = instruction.get_instruction_args()

            if args and "prompt" in args:
                instruction.build_description(prompt=ex.prompt)
            if out.strip() and instruction.check_following(out):
                is_following_list.append(True)
            else:
                is_following_list.append(False)
        ok = all(is_following_list)
        correct_mask.append(ok)
        if ok:
            correct += 1

    if print_firsts:
        first_ok = None
        first_bad = None
        for i, ok in enumerate(correct_mask):
            if ok and first_ok is None:
                first_ok = i
            if (not ok) and first_bad is None:
                first_bad = i
            if first_ok is not None and first_bad is not None:
                break

        def _dump(i: int, tag: str):
            ex = examples[i]
            print(f"\n[{metric_prefix}] {tag} example idx={i}")
            print("Prompt:")
            print(ex.prompt)
            print("Response:")
            print(outs[i])
            print("Instruction IDs:")
            print(ex.extra.get("instruction_id_list", []))
            print("Kwargs:")
            print(ex.extra.get("kwargs", []))

        if first_ok is not None:
            _dump(first_ok, "first correct")
        if first_bad is not None:
            _dump(first_bad, "first incorrect")

    n = len(examples)
    return {
        f"{metric_prefix}/acc": float(correct) / float(max(1, n)),
        f"{metric_prefix}/n": float(n),
    }


# ---------------- registry-based routing ----------------


@register_eval_evaluator("ifeval")
def eval_ifeval_source(engine, gen_cfg, src, extra=None) -> Dict[str, float]:
    return eval_ifeval_examples(
        engine,
        gen_cfg,
        src.examples,
        batch_size=src.batch_size,
        metric_prefix=src.name,
    )
