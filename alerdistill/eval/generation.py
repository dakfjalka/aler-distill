from __future__ import annotations

from typing import List

from alerdistill.rollout.engines import RolloutEngine, GenConfig
from alerdistill.utils.asyncio import run_async


async def _generate_batched_async(
    engine: RolloutEngine,
    prompts: List[str],
    gen: GenConfig,
    batch_size: int,
) -> List[str]:
    """Generate in fixed-size batches.

    We keep batching outside of the engine so any OpenAI-compatible backend
    has consistent evaluation behavior.
    """
    if batch_size <= 0:
        batch_size = len(prompts) or 1
    outs: List[str] = []
    for i in range(0, len(prompts), batch_size):
        outs.extend(await engine.generate(prompts[i : i + batch_size], gen))
    return outs


def generate_batched(
    engine: RolloutEngine,
    prompts: List[str],
    gen: GenConfig,
    batch_size: int,
) -> List[str]:
    return run_async(_generate_batched_async(engine, prompts, gen, batch_size))
