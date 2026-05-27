from __future__ import annotations

"""Shared eval datatypes.

Why this exists:
  prep.py builds prompts and may import formatting helpers from mcqa.py.
  mcqa.py / gsm8k.py / prompt_completion.py need the unified EvalExample type.
  If EvalExample lives in prep.py, importing it from evaluators creates a
  circular import (prep -> mcqa -> prep).

Putting the dataclasses here breaks that cycle.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EvalExample:
    """A unified eval unit.

    - prompt: user-side prompt fed into the generation engine
    - gold: task-specific gold label (str/int/...) used by evaluator
    - data_source: routing key used to pick evaluator ("mmlu", "gsm8k", "train_val", ...)
    - extra: any additional metadata needed for analysis.
    """

    prompt: str
    gold: Any
    data_source: str
    extra: Dict[str, Any]


@dataclass
class PreparedEvalSource:
    name: str
    data_source: str
    batch_size: int
    examples: List[EvalExample]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedEvalSuite:
    sources: List[PreparedEvalSource]

    def all_examples(self) -> List[EvalExample]:
        out: List[EvalExample] = []
        for s in self.sources:
            out.extend(s.examples)
        return out
