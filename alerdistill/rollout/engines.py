from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List, Protocol, runtime_checkable


@dataclass
class GenConfig:
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0


@runtime_checkable
class RolloutEngine(Protocol):
    """Minimal interface used by evaluators.

    The evaluators call `await engine.generate([...prompts], gen)` and expect
    a list of strings with matching length.
    """

    async def generate(self, prompts: List[str], gen: GenConfig) -> List[str]:
        ...


class OpenAICompatibleEngine:
    """Async client for OpenAI-compatible chat completions endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 120.0,
        max_concurrent: int = 16,
    ):
        from openai import AsyncOpenAI

        self.base_url = str(base_url)
        self.api_key = str(api_key)
        self.model = str(model)
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout_s)
        self._sem = asyncio.Semaphore(int(max_concurrent))

    async def _one(self, prompt: str, gen: GenConfig) -> str:
        async with self._sem:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": str(prompt)}],
                max_tokens=int(gen.max_tokens),
                temperature=float(gen.temperature),
                top_p=float(gen.top_p),
            )
            return (resp.choices[0].message.content or "").strip()

    async def generate(self, prompts: List[str], gen: GenConfig) -> List[str]:
        tasks = [self._one(p, gen) for p in prompts]
        return list(await asyncio.gather(*tasks))
