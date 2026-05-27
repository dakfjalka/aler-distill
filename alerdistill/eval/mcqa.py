from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import re

from datasets import get_dataset_config_names, load_dataset

from alerdistill.eval.generation import generate_batched
from alerdistill.eval.types import EvalExample
from alerdistill.eval.registry import register_eval_evaluator
from alerdistill.utils.asyncio import run_async

CHOICE_LETTERS_4 = ["A", "B", "C", "D"]
CHOICE_LETTERS_10 = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

_BOX_RE = re.compile(r"\\box\{([^}]*)\}", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)


def extract_box_content(text: str) -> Optional[str]:
    if not text:
        return None
    m = None
    # prefer \box{...}; fall back to \boxed{...}
    for rx in (_BOX_RE, _BOXED_RE):
        ms = list(rx.finditer(text))
        if ms:
            m = ms[-1]  # use the last box occurrence
            break
    if m is None:
        return None
    return m.group(1).strip()


def extract_choice_from_box(text: str, letters: List[str]) -> Optional[str]:
    content = extract_box_content(text)
    if content is None:
        return None
    # allow content like "A" or "A." or "(A)" etc.
    m = re.search(r"[A-J]", content.upper())
    if not m:
        return None
    cand = m.group(0).upper()
    return cand if cand in set(letters) else None



def extract_choice_from_text(text: str, letters: List[str]) -> Optional[str]:
    """Extract a multiple-choice option letter from raw model output.

    Accepts:
      - plain letters: "A"
      - decorated: "A.", "(A)", "Answer: A"
      - boxed (backward compatible): "\box{A}"
    """
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    # 1) Backward-compatible: boxed answers.
    boxed = extract_choice_from_box(s, letters)
    if boxed is not None:
        return boxed

    up = s.upper().strip()

    # 2) If the whole output is just the letter (common case).
    if up in set(letters):
        return up

    # 3) Common patterns: start with letter, optionally followed by punctuation.
    m = re.match(r'^\s*\(?\s*([A-J])\s*\)?\s*[\.:\)]?\s*$', up)
    if m:
        cand = m.group(1)
        return cand if cand in set(letters) else None

    # 4) Search for a standalone allowed letter; use the last occurrence (often final answer).
    ms = list(re.finditer(r'\b([A-J])\b', up))
    for mm in reversed(ms):
        cand = mm.group(1)
        if cand in set(letters):
            return cand

    # 5) Search for patterns like "A." or "(A)" embedded in text.
    ms2 = list(re.finditer(r'\(?\s*([A-J])\s*\)?\s*[\.:\)]', up))
    for mm in reversed(ms2):
        cand = mm.group(1)
        if cand in set(letters):
            return cand

    return None

def _format_mc_question_box(question: str, options: List[str], letters: List[str]) -> str:
    """Format a multiple-choice question.

    Note: despite the function name, we *do not* require LaTeX boxing.
    We ask the model to output a single option letter only.
    """
    lines = [question.strip(), ""]
    for L, opt in zip(letters, options):
        lines.append(f"{L}. {str(opt).strip()}")
    lines.append("")
    letters_str = ", ".join(letters)
    lines.append(
        f'Your answer should be one of: {letters_str}. '
        f'Please give your final answer as \\\\box{{X}}. '
        # f'Do not provide any explanation.'
    )
    return "\n".join(lines)


def eval_mcqa_examples(
    engine,
    gen_cfg,
    examples: List[EvalExample],
    *,
    batch_size: int,
    metric_prefix: str,
    letters: List[str],
    print_firsts: bool = True,
) -> Dict[str, float]:
    """Evaluate a prepared multiple-choice QA dataset.

    Uses a unified batch generation interface and only relies on the pre-formatted
    prompt/gold in EvalExample.
    """
    if len(examples) == 0:
        return {f"{metric_prefix}/acc": 0.0, f"{metric_prefix}/n": 0.0}

    prompts = [ex.prompt for ex in examples]
    outs = generate_batched(engine, prompts, gen_cfg, batch_size=batch_size)

    correct = 0
    preds: List[Optional[str]] = []
    correct_mask: List[bool] = []

    for out, ex in zip(outs, examples):
        pred = extract_choice_from_text(out, letters)
        preds.append(pred)

        # Gold may be either a raw letter ("A") or a boxed answer ("\\box{A}")
        # depending on the dataset formatter. We normalize both.
        gold_text = "" if ex.gold is None else str(ex.gold)
        gold = extract_choice_from_text(gold_text, letters)
        ok = pred is not None and gold is not None and pred == gold
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
            gold_text = "" if ex.gold is None else str(ex.gold)
            gold_norm = extract_choice_from_text(gold_text, letters)
            print(f"\n[{metric_prefix}] {tag} example idx={i}")
            if ex.extra:
                print("extra:", ex.extra)
            print("PROMPT:\n", ex.prompt)
            print("RESPONSE:\n", outs[i])
            print("PRED:", preds[i])
            print("GOLD_RAW:", ex.gold)
            print("GOLD:", gold_norm)

        if first_ok is not None:
            _dump(first_ok, "FIRST_CORRECT")
        if first_bad is not None:
            _dump(first_bad, "FIRST_WRONG")

    n = len(examples)
    return {
        f"{metric_prefix}/acc": float(correct) / float(max(1, n)),
        f"{metric_prefix}/n": float(n),
    }


# ---------------- registry-based routing (no hard-coded lists in runner) ----------------


@register_eval_evaluator("mmlu")
@register_eval_evaluator("gpqa")
@register_eval_evaluator("sciknoweval")
@register_eval_evaluator("sciknoweval_rlvr")
def eval_mcqa_4_source(engine, gen_cfg, src, extra=None) -> Dict[str, float]:
    return eval_mcqa_examples(
        engine,
        gen_cfg,
        src.examples,
        batch_size=src.batch_size,
        metric_prefix=src.name,
        letters=CHOICE_LETTERS_4,
    )


@register_eval_evaluator("mmlu_pro")
def eval_mcqa_10_source(engine, gen_cfg, src, extra=None) -> Dict[str, float]:
    return eval_mcqa_examples(
        engine,
        gen_cfg,
        src.examples,
        batch_size=src.batch_size,
        metric_prefix=src.name,
        letters=CHOICE_LETTERS_10,
    )
