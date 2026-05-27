from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import re

from datasets import load_dataset

from alerdistill.utils.asyncio import run_async
from alerdistill.eval.generation import generate_batched
from alerdistill.eval.types import EvalExample
from alerdistill.eval.registry import EvalContext, register_eval_evaluator, register_eval_preparer

# ---------------- parsing helpers ----------------

_BOX_RE = re.compile(r"\\box\{([^}]*)\}", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)

# Accept integers, integer-like decimals, and integer-valued fractions.
_NUM_TOKEN_RE = re.compile(r"[-+]?\d+(?:\.\d+)?|[-+]?\d+\s*/\s*\d+")


def extract_box_content(text: str) -> Optional[str]:
    """Return the LAST occurrence of \\box{...} or \\boxed{...} content."""
    if not text:
        return None
    for rx in (_BOX_RE, _BOXED_RE):
        ms = list(rx.finditer(text))
        if ms:
            return ms[-1].group(1).strip()
    return None


def _extract_final_number_from_gold(answer: str) -> Optional[int]:
    """GSM8K gold format typically contains '#### <number>'."""
    if not answer:
        return None
    m = re.search(r"####\s*([-+]?\d+)", answer)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    # fallback: last integer in string
    ms = re.findall(r"[-+]?\d+", answer)
    if not ms:
        return None
    try:
        return int(ms[-1])
    except Exception:
        return None


def _parse_number_to_int(s: str) -> Optional[int]:
    """Parse a numeric string into an int if it is exactly (or effectively) an integer.

    Supported:
    - integers: "57"
    - integer-like decimals: "57.00" -> 57
    - fractions that reduce to integer: "3/1" -> 3, "10/2" -> 5
    """
    if s is None:
        return None
    x = str(s).strip()
    if not x:
        return None

    # common cleanup
    x = x.replace(",", "")
    x = x.replace("$", "").strip()

    # fraction: a/b
    m = re.fullmatch(r"([-+]?\d+)\s*/\s*(\d+)", x)
    if m:
        try:
            num = int(m.group(1))
            den = int(m.group(2))
        except Exception:
            return None
        if den == 0:
            return None
        if num % den == 0:
            return num // den
        return None

    # integer
    if re.fullmatch(r"[-+]?\d+", x):
        try:
            return int(x)
        except Exception:
            return None

    # decimal: only accept if extremely close to integer (e.g. 57.00)
    if re.fullmatch(r"[-+]?\d+\.\d+", x):
        try:
            f = float(x)
        except Exception:
            return None
        r = round(f)
        if abs(f - r) < 1e-9:
            return int(r)
        return None

    return None


def _extract_pred_int_from_box(text: str) -> Optional[int]:
    """Extract the predicted final integer.

    - Prefer content inside \\box{...} / \\boxed{...}
    - Accept integer-like decimals such as 57.00 -> 57
    - Fallback: take last numeric token in the text (supports decimals and fractions)
    """
    s = "" if text is None else str(text)

    # 1) Prefer boxed content (key fix: parse as a whole number, not "last integer token")
    content = extract_box_content(s)
    if content is not None:
        v = _parse_number_to_int(content)
        if v is not None:
            return v

    # 2) Fallback: find numeric tokens (won't split 57.00 into 57 and 00)
    cand = (content if content is not None else s).replace(",", "")
    tokens = _NUM_TOKEN_RE.findall(cand)
    if not tokens:
        tokens = _NUM_TOKEN_RE.findall(s.replace(",", ""))

    # Reverse scan: last token that can be parsed as integer
    for tok in reversed(tokens):
        v = _parse_number_to_int(tok)
        if v is not None:
            return v

    return None


# ---------------- preparer ----------------

@register_eval_preparer("gsm8k")
def prepare_gsm8k_source(name: str, cfg: Dict[str, Any], ctx: EvalContext) -> "PreparedEvalSource": # type: ignore
    """Prepare GSM8K eval examples.

    Produces prompts that request the final answer inside \\box{...}.
    The evaluator extracts the last boxed content and compares to the gold integer.
    """
    # Local import to avoid import cycles.
    from alerdistill.eval.types import PreparedEvalSource

    data_source = str(cfg.get("data_source") or "gsm8k")
    dataset_name = str(cfg.get("dataset_name") or "openai/gsm8k")
    config_name = str(cfg.get("config_name") or "main")
    split = str(cfg.get("split") or "test")
    max_examples = cfg.get("max_examples")
    batch_size = int(cfg.get("batch_size") or 8)
    use_cot = bool(cfg.get("use_cot", True))

    ds = load_dataset(dataset_name, config_name, split=split)
    idxs = list(range(len(ds)))
    if max_examples is not None:
        idxs = idxs[: int(max_examples)]

    inst = "Solve the problem. Please give your final answer as \\box{...}."

    examples: List[EvalExample] = []
    for j in idxs:
        ex = ds[j]
        q = str(ex.get("question", ""))
        if use_cot:
            prompt = f"{inst}\n\nQ: {q}\nA: Let's think step by step. Final answer: "
        else:
            prompt = f"{inst}\n\nQ: {q}\nA: Final answer: "

        gold_int = _extract_final_number_from_gold(str(ex.get("answer", "")))
        examples.append(
            EvalExample(
                prompt=prompt,
                gold=gold_int,
                data_source=data_source,
                extra={"idx": int(j)},
            )
        )

    return PreparedEvalSource(
        name=name,
        data_source=data_source,
        batch_size=batch_size,
        examples=examples,
    )

# ---------------- registry-compatible eval over prepared examples ----------------

def eval_gsm8k_examples(
    engine,
    gen_cfg,
    examples: List[EvalExample],
    *,
    batch_size: int,
    metric_prefix: str = "gsm8k",
    print_firsts: bool = True,
) -> Dict[str, float]:
    if len(examples) == 0:
        return {f"{metric_prefix}/acc": 0.0, f"{metric_prefix}/n": 0.0}

    prompts = [ex.prompt for ex in examples]
    outs = generate_batched(engine, prompts, gen_cfg, batch_size=batch_size)

    correct = 0
    preds: List[Optional[int]] = []
    correct_mask: List[bool] = []

    for out, ex in zip(outs, examples):
        pred = _extract_pred_int_from_box(out)
        preds.append(pred)
        gold = ex.gold
        ok = pred is not None and gold is not None and int(pred) == int(gold)
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
            if ex.extra:
                print("extra:", ex.extra)
            print("PROMPT:\n", ex.prompt)
            print("RESPONSE:\n", outs[i])
            print("BOX:", extract_box_content(outs[i]))
            print("PRED:", preds[i])
            print("GOLD:", ex.gold)

        if first_ok is not None:
            _dump(first_ok, "FIRST_CORRECT")
        if first_bad is not None:
            _dump(first_bad, "FIRST_WRONG")

    n = len(examples)
    return {
        f"{metric_prefix}/acc": float(correct) / float(max(1, n)),
        f"{metric_prefix}/n": float(n),
    }


# ---------------- registry-based routing ----------------

@register_eval_evaluator("gsm8k")
def eval_gsm8k_source(engine, gen_cfg, src, extra=None) -> Dict[str, float]:
    return eval_gsm8k_examples(
        engine,
        gen_cfg,
        src.examples,
        batch_size=src.batch_size,
        metric_prefix=src.name,
    )
