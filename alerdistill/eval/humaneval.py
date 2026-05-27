from datasets import load_dataset, Dataset
import re
from typing import Any, Dict, Optional, List
from hydra.core.hydra_config import HydraConfig
import json
import os
import subprocess

from alerdistill.eval.registry import (
    EvalContext,
    register_eval_evaluator,
    register_eval_preparer,
)
from alerdistill.eval.types import EvalExample, PreparedEvalSource
from alerdistill.eval.generation import generate_batched


def ensure_problem_file(
    problem_file_dir: str, ds: Dataset, clear_cache: bool = False
) -> str:
    if not os.path.exists(problem_file_dir):
        os.makedirs(problem_file_dir, exist_ok=True)

    problem_file_path = os.path.join(problem_file_dir, "humaneval_problems.jsonl")
    if not os.path.exists(problem_file_path) or clear_cache:
        with open(problem_file_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(ds):
                import json

                json.dump(
                    {
                        "task_id": item["task_id"],
                        "prompt": item["prompt"],
                        "canonical_solution": item["canonical_solution"],
                        "entry_point": item["entry_point"],
                        "test": item["test"],
                    },
                    f,
                    ensure_ascii=False,
                )
                f.write("\n")
    return problem_file_path


@register_eval_preparer("humaneval")
def prepare_humaneval_source(
    name: str, cfg: Dict[str, Any], ctx: EvalContext
) -> PreparedEvalSource:
    run_dir = ctx.run_dir
    if run_dir is None:
        raise ValueError("run_dir must be specified in EvalContext for humaneval preparer.")

    data_source = str(cfg.get("data_source") or "humaneval")
    dataset_name = str(cfg.get("dataset_name") or "openai/openai_humaneval")
    split = str(cfg.get("split") or "test")
    max_examples = cfg.get("max_examples")
    batch_size = int(cfg.get("batch_size") or 8)

    # special
    timeout_s = float(cfg.get("timeout_s", 3.0))
    pass_at_k = int(cfg.get("pass_at_k", 1))
    n_workers = int(cfg.get("n_workers", 16))
    eval_timeout_s = float(cfg.get("eval_timeout_s", 600.0))

    problem_file_dir = str(cfg.get("problem_file_dir") or f"humaneval_problems")
    problem_file_dir = os.path.join(run_dir, problem_file_dir)

    ds = load_dataset(dataset_name, split=split)
    print(
        f"[prepare_humaneval_source] loaded dataset {dataset_name} split={split} with {len(ds)} examples."
    )
    if max_examples is not None:
        ds = ds.select(range(int(max_examples)))
    print(
        f"[prepare_humaneval_source] using {len(ds)} examples after max_examples={max_examples}."
    )

    extra = {
        "timeout_s": timeout_s,
        "pass_at_k": pass_at_k,
        "n_workers": n_workers,
        "ds": ds,
        "problem_file_dir": problem_file_dir,
        "eval_timeout_s": eval_timeout_s,
    }

    idxs = list(range(len(ds)))
    examples: List[EvalExample] = []
    for j in idxs:
        ex = ds[j]
        prompt = str(ex.get("prompt", ""))
        task_id = ex.get("task_id", str(j))

        examples.append(
            EvalExample(
                prompt=prompt,
                gold=None,
                data_source=data_source,
                extra={"task_id": task_id},
            )
        )

    print("[prepare_humaneval_source] preparing problem file...")
    print(f"[prepare_humaneval_source] ds[0]: {ds[0]}")
    ensure_problem_file(problem_file_dir, ds, clear_cache=True)

    return PreparedEvalSource(
        name=name,
        data_source=data_source,
        batch_size=batch_size,
        extra=extra,
        examples=examples,
    )


_CODE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def extract_code_from_completion(completion: str, prompt: str) -> str:
    ms = _CODE_RE.findall(completion)
    code_str = ""
    if len(ms) > 0:
        code_str = ms[0]
    else:
        code_str = completion
    if code_str.startswith(prompt):
        code_str = code_str[len(prompt) :]
    return code_str


def eval_humaneval_examples(
    engine,
    gen_cfg,
    src: PreparedEvalSource,
    *,
    metric_prefix: str = "humaneval",
    print_firsts: bool = True,
    step: Optional[int] = None,
) -> Dict[str, float]:
    """
    $ git clone https://github.com/openai/human-eval
    $ pip install -e human-eval

    step 1: samples.jsonl
    step 2: python -m human_eval.evaluate_functional_correctness samples.jsonl
        FLAGS
            --problem_file=samples.jsonl
            --k=K
            --n_workers=N_WORKERS
            --timeout=TIMEOUT
        the output will be f"{sample_file}_results.jsonl"
    """
    examples = src.examples
    batch_size = src.batch_size
    if len(examples) == 0:
        return {f"{metric_prefix}/acc": 0.0, f"{metric_prefix}/n": 0.0}

    prompts = [ex.prompt for ex in examples]
    outs = generate_batched(engine, prompts, gen_cfg, batch_size=batch_size)
    # write to temp file
    jsonl_data = []
    for ex, out in zip(examples, outs):
        jsonl_data.append(
            {
                "task_id": ex.extra["task_id"],
                "completion": extract_code_from_completion(out, ex.prompt),
            }
        )
    problem_file_dir = src.extra["problem_file_dir"]
    ensure_problem_file(problem_file_dir, src.extra["ds"])
    problem_file_path = os.path.join(problem_file_dir, "humaneval_problems.jsonl")
    sample_file_path = os.path.join(
        problem_file_dir, f"samples_step{step if step is not None else 'final'}.jsonl"
    )

    with open(sample_file_path, "w", encoding="utf-8") as f:
        for item in jsonl_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"[eval_humaneval_examples] written samples to {sample_file_path}")

    timeout_s = src.extra.get("timeout_s", 3.0)
    pass_at_k = src.extra.get("pass_at_k", 1)
    n_workers = src.extra.get("n_workers", 16)
    eval_timeout_s = src.extra.get("eval_timeout_s", 600.0)

    cmd = [
        "python",
        "-m",
        "human_eval.evaluate_functional_correctness",
        sample_file_path,
        f"--problem_file={problem_file_path}",
        f"--k={pass_at_k}",
        f"--n_workers={n_workers}",
        f"--timeout={timeout_s}",
    ]
    print(f"[eval_humaneval_examples] running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=eval_timeout_s,
        )
        print("[eval_humaneval_examples] evaluation stdout:\n", result.stdout)
        print("[eval_humaneval_examples] evaluation stderr:\n", result.stderr)
    except subprocess.TimeoutExpired:
        print(
            f"[eval_humaneval_examples] Evaluation timed out after {eval_timeout_s} seconds."
        )
        return {f"{metric_prefix}/acc": 0.0, f"{metric_prefix}/n": 0.0}
    if result.returncode != 0:
        print("human-eval failed:", result.stderr)
        return {f"{metric_prefix}/acc": 0.0, f"{metric_prefix}/n": 0.0}

    result_file_path = f"{sample_file_path}_results.jsonl"
    if not os.path.exists(result_file_path):
        print(f"[eval_humaneval_examples] Result file {result_file_path} not found.")
        return {f"{metric_prefix}/acc": 0.0, f"{metric_prefix}/n": 0.0}
    # parse result file

    total = 0
    correct = 0
    correct_mask: List[bool] = []
    result_map = {}

    with open(result_file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            total += 1
            ok = item.get("passed", False)
            correct_mask.append(ok)
            if ok:
                correct += 1
            result_map[item["task_id"]] = item.get("result", None)

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
            print(
                f"\n[{metric_prefix}] {tag} example idx={i} task_id={ex.extra['task_id']}"
            )
            if ex.extra:
                print("extra:", ex.extra)
            print("PROMPT:\n", ex.prompt)
            print("RESPONSE:\n", outs[i])
            print("EVAL RESULT:\n", result_map.get(ex.extra["task_id"], None))

        if first_ok is not None:
            _dump(first_ok, "first correct")
        if first_bad is not None:
            _dump(first_bad, "first incorrect")
    acc = correct / total if total > 0 else 0.0
    return {f"{metric_prefix}/acc": acc, f"{metric_prefix}/n": float(total)}


@register_eval_evaluator("humaneval")
def eval_humaneval_source(
    engine, gen_cfg, src: PreparedEvalSource, extra=None
) -> Dict[str, float]:

    return eval_humaneval_examples(
        engine,
        gen_cfg,
        src=src,
        metric_prefix=src.name,
        step=extra.get("step", None) if extra else None,
    )
