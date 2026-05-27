from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .danger import DangerConfig, danger_per_prompt


@contextmanager
def _freeze_model_params(model: torch.nn.Module):
    flags = [p.requires_grad for p in model.parameters()]
    try:
        for p in model.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)


def _maybe_set_adapter(model: torch.nn.Module, adapter_name: str | None) -> None:
    if adapter_name is not None and hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)


@dataclass
class EntropyRegConfig:
    enabled: bool = True
    beta: float = 0.02


@dataclass
class L2RegConfig:
    enabled: bool = True
    gamma: float = 0.001
    target: str = "init"


@dataclass
class DiversityRegConfig:
    enabled: bool = True
    delta: float = 0.05


@dataclass
class CandidateTopKSimilar:
    anchor_tokens: List[str]
    topk: int


@dataclass
class RepairConfig:
    enabled: bool = True
    lambda_repair: float = 1.0
    kl_direction: str = "reverse"


@dataclass
class AlerDistillConfig:
    enabled: bool = True
    prompt_length: int = 8
    prompt_batch_size: int = 8
    search_steps: int = 10
    prompt_lr: float = 0.2
    temperature: float = 1.0
    candidate_k: int = 32
    candidate_mode: str = "random_vocab"
    candidate_seed: int = 123
    candidate_topk_similar: CandidateTopKSimilar = field(
        default_factory=lambda: CandidateTopKSimilar(
            anchor_tokens=["the", "a", "to", "of", "and", "in"],
            topk=64,
        )
    )
    entropy_reg: EntropyRegConfig = field(default_factory=EntropyRegConfig)
    l2_reg: L2RegConfig = field(default_factory=L2RegConfig)
    diversity_reg: DiversityRegConfig = field(default_factory=DiversityRegConfig)
    probe_suffix_text: str = "Answer the next question briefly and correctly: "
    danger: DangerConfig = field(default_factory=DangerConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    distributed_search_mode: str = "all_ranks"
    embedding_snapshot_enabled: bool = True
    embedding_snapshot_dtype: str = "float16"


class BatchedSoftPromptSearcher(nn.Module):
    """Search latent prompts that expose drift between reference and updated model."""

    def __init__(
        self,
        cfg: AlerDistillConfig,
        tokenizer,
        *,
        embedding_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self._embedding_weight = embedding_weight

        bsz = int(cfg.prompt_batch_size)
        prompt_len = int(cfg.prompt_length)
        candidates = int(cfg.candidate_k)
        self.w = nn.Parameter(torch.zeros((bsz, prompt_len, candidates)))
        self._w_optim = torch.optim.Adam([self.w], lr=float(cfg.prompt_lr))

    def _E(self, model) -> torch.Tensor:
        if self._embedding_weight is not None:
            return self._embedding_weight
        return model.get_input_embeddings().weight

    def _gather_cand_E(
        self,
        *,
        E: torch.Tensor,
        candidate_ids: torch.LongTensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if E.device.type != "cpu":
            return E[candidate_ids].to(device=device, dtype=dtype, non_blocking=True)

        ids_cpu = candidate_ids.detach().to("cpu")
        return E[ids_cpu].to(device=device, dtype=dtype, non_blocking=True)

    def _build_candidate_ids(self, model, device: torch.device) -> torch.LongTensor:
        cfg = self.cfg
        bsz = int(cfg.prompt_batch_size)
        prompt_len = int(cfg.prompt_length)
        candidates = int(cfg.candidate_k)
        vocab_size = self._E(model).shape[0]

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        gen = torch.Generator(device="cpu").manual_seed(int(cfg.candidate_seed) + max(rank, 0))

        if cfg.candidate_mode == "random_vocab":
            ids = torch.randint(0, vocab_size, (bsz, prompt_len, candidates), generator=gen, dtype=torch.long)
            return ids.to(device)

        if cfg.candidate_mode == "topk_similar":
            embeddings = self._E(model).detach()
            if embeddings.device.type == "cpu" and embeddings.dtype != torch.float32:
                embeddings = embeddings.float()
            embeddings = F.normalize(embeddings, dim=-1)

            pool: List[int] = []
            for token in cfg.candidate_topk_similar.anchor_tokens:
                token_ids = self.tokenizer(token, add_special_tokens=False)["input_ids"]
                if not token_ids:
                    continue
                anchor_id = int(token_ids[0])
                sims = embeddings @ embeddings[anchor_id]
                topk = torch.topk(
                    sims,
                    k=min(int(cfg.candidate_topk_similar.topk), vocab_size),
                    largest=True,
                ).indices
                pool.extend(topk.tolist())

            pool = list(dict.fromkeys(pool))
            if len(pool) < candidates:
                extra = torch.randint(0, vocab_size, (candidates - len(pool),), generator=gen, dtype=torch.long)
                pool.extend(extra.tolist())

            pool_t = torch.tensor(pool, dtype=torch.long)
            idx = torch.randint(0, len(pool_t), (bsz, prompt_len, candidates), generator=gen)
            return pool_t[idx].to(device)

        raise ValueError(f"Unknown candidate_mode: {cfg.candidate_mode}")

    def _soft(self, cand_E: torch.Tensor, w: torch.Tensor, temp: float) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.softmax(w / float(temp), dim=-1).to(dtype=cand_E.dtype)
        soft = torch.einsum("bmk,bmkd->bmd", alpha, cand_E)
        return soft, alpha

    @staticmethod
    def _entropy(alpha: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        return -(alpha * (alpha + eps).log()).sum(dim=-1).mean()

    @staticmethod
    def _l2(soft: torch.Tensor, target_soft: torch.Tensor) -> torch.Tensor:
        return ((soft - target_soft) ** 2).mean()

    @staticmethod
    def _diversity(soft: torch.Tensor) -> torch.Tensor:
        pooled = F.normalize(soft.mean(dim=1), dim=-1)
        sim = pooled @ pooled.T
        sim = sim - torch.eye(sim.shape[0], device=sim.device)
        return sim.abs().mean()

    def _suffix(self, model, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(self.cfg.probe_suffix_text, add_special_tokens=False, return_tensors="pt")
        ids = tok["input_ids"].to(device)
        attn = tok["attention_mask"].to(device)
        model_dtype = next(model.parameters()).dtype

        embeddings = self._E(model)
        if embeddings.device.type == "cpu":
            suffix_emb = embeddings[ids.detach().to("cpu")].to(device=device, dtype=model_dtype, non_blocking=True)
        else:
            suffix_emb = embeddings[ids].to(device=device, dtype=model_dtype, non_blocking=True)
        return suffix_emb, attn

    @torch.no_grad()
    def _ref_last_logits(
        self,
        model,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        ref_adapter: str | None = None,
        ref_model: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        if ref_model is not None:
            return ref_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False).logits[:, -1, :]
        _maybe_set_adapter(model, ref_adapter)
        return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False).logits[:, -1, :]

    def _new_last_logits(
        self,
        model,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        new_adapter: str | None = None,
    ) -> torch.Tensor:
        _maybe_set_adapter(model, new_adapter)
        return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False).logits[:, -1, :]

    def search(
        self,
        *,
        model,
        ref_adapter: str | None = None,
        new_adapter: str | None = None,
        ref_model: torch.nn.Module | None = None,
        accelerator=None,
        device: torch.device | None = None,
        prompt_batch_size: int | None = None,
    ) -> Dict[str, Any]:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        print(f"[rank{rank}][BatchedSoftPromptSearcher] starting search...", flush=True)

        cfg = self.cfg
        if device is None and accelerator is not None:
            device = accelerator.device
        if device is None:
            raise ValueError("BatchedSoftPromptSearcher.search requires a target device.")

        was_training = model.training
        model.eval()
        if ref_model is not None:
            ref_model.eval()

        try:
            stats: Dict[str, float] = {}
            with _freeze_model_params(model):
                embedding_weight = self._E(model).detach()
                bsz_max = int(cfg.prompt_batch_size)
                bsz = int(prompt_batch_size) if prompt_batch_size is not None else bsz_max
                if bsz < 1 or bsz > bsz_max:
                    raise ValueError(f"prompt_batch_size must be in [1, {bsz_max}], got {bsz}.")

                prompt_len = int(cfg.prompt_length)
                candidates = int(cfg.candidate_k)
                candidate_ids = self._build_candidate_ids(model, device)[:bsz]
                model_dtype = next(model.parameters()).dtype
                candidate_emb = self._gather_cand_E(
                    E=embedding_weight,
                    candidate_ids=candidate_ids,
                    device=device,
                    dtype=model_dtype,
                )

                if self.w.device != device:
                    self.to(device)
                with torch.no_grad():
                    self.w.zero_()
                    init_idx = torch.randint(0, candidates, (bsz, prompt_len), device=device)
                    self.w[:bsz].scatter_(2, init_idx.unsqueeze(-1), 3.0)

                w = self.w[:bsz]
                self._w_optim.state.clear()

                with torch.no_grad():
                    soft0, _ = self._soft(candidate_emb, w.detach(), cfg.temperature)
                target_soft = soft0 if cfg.l2_reg.target == "init" else torch.zeros_like(soft0)

                suffix_emb, suffix_attn = self._suffix(model, device)
                suffix_len = suffix_emb.shape[1]
                attention = torch.ones((bsz, prompt_len + suffix_len), device=device, dtype=torch.long)
                attention[:, prompt_len:] = suffix_attn.expand(bsz, -1)

                for step in range(int(cfg.search_steps)):
                    self._w_optim.zero_grad(set_to_none=True)
                    soft, alpha = self._soft(candidate_emb, w, cfg.temperature)
                    inputs_embeds = torch.cat([soft, suffix_emb.expand(bsz, -1, -1)], dim=1)

                    ref_last = self._ref_last_logits(
                        model,
                        inputs_embeds,
                        attention,
                        ref_adapter=ref_adapter,
                        ref_model=ref_model,
                    )
                    new_last = self._new_last_logits(model, inputs_embeds, attention, new_adapter=new_adapter)

                    danger_vec = danger_per_prompt(ref_last, new_last, cfg.danger)
                    danger_obj = torch.logsumexp(danger_vec, dim=0)
                    loss = -danger_obj

                    entropy = torch.tensor(0.0, device=device)
                    if cfg.entropy_reg.enabled:
                        entropy = self._entropy(alpha)
                        loss = loss + float(cfg.entropy_reg.beta) * entropy

                    l2 = torch.tensor(0.0, device=device)
                    if cfg.l2_reg.enabled:
                        l2 = self._l2(soft, target_soft)
                        loss = loss + float(cfg.l2_reg.gamma) * l2

                    diversity = torch.tensor(0.0, device=device)
                    if cfg.diversity_reg.enabled:
                        diversity = self._diversity(soft)
                        loss = loss + float(cfg.diversity_reg.delta) * diversity

                    if accelerator is not None:
                        accelerator.backward(loss)
                    else:
                        loss.backward()
                    self._w_optim.step()

                    stats = {
                        "alerdistill/search_iter": float(step),
                        "alerdistill/danger_obj": float(danger_obj.detach().cpu()),
                        "alerdistill/entropy": float(entropy.detach().cpu()),
                        "alerdistill/l2": float(l2.detach().cpu()),
                        "alerdistill/diversity": float(diversity.detach().cpu()),
                        "alerdistill/search_loss": float(loss.detach().cpu()),
                    }

                with torch.no_grad():
                    soft, alpha = self._soft(candidate_emb, w, cfg.temperature)
                    inputs_embeds = torch.cat([soft, suffix_emb.expand(bsz, -1, -1)], dim=1)
                    ref_last = self._ref_last_logits(
                        model,
                        inputs_embeds,
                        attention,
                        ref_adapter=ref_adapter,
                        ref_model=ref_model,
                    )
                    new_last = self._new_last_logits(model, inputs_embeds, attention, new_adapter=new_adapter)
                    danger_vec = danger_per_prompt(ref_last, new_last, cfg.danger)
                    best_idx = int(torch.argmax(danger_vec).item())
                    best_soft = soft[best_idx : best_idx + 1].detach()
                    best_alpha = alpha[best_idx : best_idx + 1].detach()
                    stats.update(
                        {
                            "alerdistill/danger_max": float(danger_vec.max().detach().cpu()),
                            "alerdistill/danger_mean": float(danger_vec.mean().detach().cpu()),
                        }
                    )

            _maybe_set_adapter(model, new_adapter)
            return {"best_soft_prompt_embeds": best_soft, "best_alpha": best_alpha, "stats": stats}
        finally:
            if was_training:
                model.train()
