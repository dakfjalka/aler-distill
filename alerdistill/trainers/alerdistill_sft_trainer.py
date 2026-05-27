from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Dict, Optional, Union

import torch
from trl import SFTTrainer

from alerdistill.latent_repair.danger import kl_divergence
from alerdistill.latent_repair.searcher import AlerDistillConfig, BatchedSoftPromptSearcher

try:
    from accelerate.utils import DistributedType
except Exception:  # pragma: no cover - accelerate is a runtime dependency for training.
    DistributedType = None


class AlerDistillSFTTrainer(SFTTrainer):
    """TRL SFT trainer with latent prompt search, RKL repair, and on-batch ref KL."""

    def __init__(
        self,
        *args,
        latent_repair_cfg: Optional[AlerDistillConfig] = None,
        ref_adapter: Optional[str] = None,
        new_adapter: Optional[str] = None,
        ref_model: Optional[torch.nn.Module] = None,
        ref_kl_enabled: bool = False,
        ref_kl_lambda: float = 0.0,
        metric_logger=None,
        **kwargs,
    ):
        rank = int(os.environ.get("RANK", -1))
        print(f"[rank{rank}][AlerDistillSFTTrainer] initializing", flush=True)
        super().__init__(*args, **kwargs)
        logging.getLogger().setLevel(logging.INFO)

        self.latent_repair_cfg = latent_repair_cfg
        self.ref_adapter = ref_adapter
        self.new_adapter = new_adapter
        self.ref_model = ref_model
        self._ref_ready = False
        self.ref_kl_enabled = bool(ref_kl_enabled)
        self.ref_kl_lambda = float(ref_kl_lambda)
        self.metric_logger = metric_logger

        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad_(False)

        tokenizer = self._get_tokenizer()
        self._embedding_weight_snapshot = None
        if latent_repair_cfg and latent_repair_cfg.enabled and latent_repair_cfg.embedding_snapshot_enabled:
            try:
                snap_dtype = getattr(torch, str(latent_repair_cfg.embedding_snapshot_dtype))
                self._embedding_weight_snapshot = (
                    self.model.get_input_embeddings().weight.detach().to("cpu", dtype=snap_dtype)
                )
            except Exception:
                self._embedding_weight_snapshot = None

        self._searcher = (
            BatchedSoftPromptSearcher(
                latent_repair_cfg,
                tokenizer=tokenizer,
                embedding_weight=self._embedding_weight_snapshot,
            )
            if latent_repair_cfg and latent_repair_cfg.enabled
            else None
        )
        if self._searcher is not None and getattr(self, "accelerator", None) is not None:
            self._searcher.to(self.accelerator.device)

    def _get_tokenizer(self):
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("AlerDistillSFTTrainer requires processing_class=tokenizer.")
        return tokenizer

    def _embed_from_snapshot(self, ids: torch.Tensor, device, dtype) -> torch.Tensor:
        snapshot = self._embedding_weight_snapshot
        if snapshot is None:
            raise RuntimeError("Embedding snapshot is unavailable; keep latent_repair.embedding_snapshot_enabled=true.")

        ids_cpu = ids.detach().to("cpu", non_blocking=True)
        flat = ids_cpu.reshape(-1)
        rows = snapshot.index_select(0, flat)
        emb = rows.view(*ids_cpu.shape, rows.shape[-1])
        return emb.to(device=device, dtype=dtype, non_blocking=True)

    @staticmethod
    def _has_adapter_support(model) -> bool:
        return hasattr(model, "set_adapter")

    def _ensure_ref_model(self, model: torch.nn.Module) -> None:
        if self._ref_ready:
            return
        if self.ref_model is None:
            return

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

        is_fsdp = False
        try:
            if getattr(self, "accelerator", None) is not None and DistributedType is not None:
                is_fsdp = getattr(self.accelerator, "distributed_type", None) == DistributedType.FSDP
        except Exception:
            is_fsdp = False

        if is_fsdp:
            try:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, device_placement=True)
                self.ref_model.eval()
                for param in self.ref_model.parameters():
                    param.requires_grad_(False)
            except Exception as exc:
                raise RuntimeError("Failed to shard the frozen reference model for FSDP training.") from exc
            self._ref_ready = True
            return

        try:
            if getattr(self.ref_model, "hf_device_map", None) is None:
                param = next(model.parameters())
                self.ref_model.to(device=param.device, dtype=param.dtype)
        except StopIteration:
            pass

        self._ref_ready = True

    def _require_ref_available(self, model, where: str) -> None:
        self._ensure_ref_model(model)
        if self.ref_model is not None:
            return
        if self.ref_adapter is not None and self.new_adapter is not None and self._has_adapter_support(model):
            return
        raise ValueError(f"{where} requires either ref_model or PEFT adapters named ref/new.")

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        if self.new_adapter is not None and self._has_adapter_support(model):
            model.set_adapter(self.new_adapter)

        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        if self.ref_kl_enabled:
            self._require_ref_available(model, where="on-batch ref KL")
            new_logits = outputs.logits if getattr(outputs, "logits", None) is not None else model(**inputs).logits

            with torch.no_grad():
                if self.ref_model is not None:
                    self._ensure_ref_model(model)
                    ref_logits = self.ref_model(**inputs).logits
                else:
                    model.set_adapter(self.ref_adapter)
                    ref_logits = model(**inputs).logits

            if self.new_adapter is not None and self._has_adapter_support(model):
                model.set_adapter(self.new_adapter)

            loss = loss + self.ref_kl_lambda * kl_divergence(ref_logits, new_logits, reduction="mean")

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._ensure_ref_model(model)

        activation_ctx_obj = getattr(self, "maybe_activation_offload_context", None)
        if activation_ctx_obj is None:
            activation_ctx = contextlib.nullcontext()
        elif callable(activation_ctx_obj):
            activation_ctx = activation_ctx_obj()
        else:
            activation_ctx = activation_ctx_obj

        with activation_ctx:
            prepare_cp = getattr(self, "_prepare_context_parallel_inputs", None)
            if prepare_cp is None:
                cp_context, inputs = contextlib.nullcontext, inputs
            else:
                cp_context, inputs = prepare_cp(model, inputs)

            with cp_context():
                if not hasattr(self, "_repair_micro_in_update"):
                    self._repair_micro_in_update = 0
                if not hasattr(self, "_repair_soft_prompt_cached"):
                    self._repair_soft_prompt_cached = None

                if self.new_adapter is not None and self._has_adapter_support(model):
                    model.set_adapter(self.new_adapter)

                cfg = self.latent_repair_cfg
                search_enabled = self._searcher is not None and cfg is not None and bool(cfg.enabled)
                repair_enabled = search_enabled and cfg.repair is not None and bool(cfg.repair.enabled)
                repair_weight = float(cfg.repair.lambda_repair) if repair_enabled else 0.0

                if search_enabled and self._repair_micro_in_update == 0:
                    dist_world = self.accelerator.num_processes if self.accelerator is not None else 1
                    dist_rank = self.accelerator.process_index if self.accelerator is not None else 0
                    mode = str(getattr(cfg, "distributed_search_mode", "all_ranks"))

                    if dist_world > 1:
                        if mode == "all_ranks":
                            local_batch = max(1, int(cfg.prompt_batch_size) // dist_world)
                        else:
                            local_batch = int(cfg.prompt_batch_size) if dist_rank == 0 else max(
                                1,
                                int(cfg.prompt_batch_size) // dist_world,
                            )
                    else:
                        local_batch = int(cfg.prompt_batch_size)

                    search_out = self._searcher.search(
                        model=model,
                        ref_adapter=self.ref_adapter,
                        new_adapter=self.new_adapter,
                        ref_model=self.ref_model,
                        accelerator=self.accelerator,
                        device=self.accelerator.device,
                        prompt_batch_size=local_batch,
                    )
                    soft_prompt = search_out["best_soft_prompt_embeds"]

                    if dist_world > 1 and mode == "rank0_broadcast" and torch.distributed.is_initialized():
                        torch.distributed.broadcast(soft_prompt, src=0)

                    self._repair_soft_prompt_cached = soft_prompt.detach()
                    model.zero_grad(set_to_none=True)
                    if self.optimizer is not None:
                        self.optimizer.zero_grad(set_to_none=True)

                model.train()
                if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                    self.optimizer.train()

                inputs = self._prepare_inputs(inputs)
                with self.compute_loss_context_manager():
                    sft_loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
                    loss = sft_loss
                    if (
                        repair_enabled
                        and repair_weight > 0.0
                        and self.accelerator.sync_gradients
                        and self._repair_soft_prompt_cached is not None
                    ):
                        repair_loss = self._repair_loss_on_prompts(model, self._repair_soft_prompt_cached)
                        loss = loss + repair_weight * repair_loss
                        rank = self.accelerator.process_index if self.accelerator is not None else -1
                        print(
                            f"[rank{rank}][AlerDistillSFTTrainer] sft={float(sft_loss.detach().cpu()):.6f}, "
                            f"rkl_repair={float(repair_loss.detach().cpu()):.6f}",
                            flush=True,
                        )

                del inputs

                if self.args.n_gpu > 1:
                    loss = loss.mean()

                accepts_loss_kwargs = bool(getattr(self, "model_accepts_loss_kwargs", False))
                compute_loss_func = getattr(self, "compute_loss_func", None)
                if (not accepts_loss_kwargs or num_items_in_batch is None) and compute_loss_func is None:
                    grad_accum = getattr(
                        self,
                        "current_gradient_accumulation_steps",
                        self.args.gradient_accumulation_steps,
                    )
                    loss = loss / grad_accum

                kwargs = {}
                if (
                    DistributedType is not None
                    and getattr(self.accelerator, "distributed_type", None) == DistributedType.DEEPSPEED
                ):
                    kwargs["scale_wrt_gas"] = False

                self.accelerator.backward(loss, **kwargs)

                if self.accelerator.sync_gradients:
                    self._repair_micro_in_update = 0
                    self._repair_soft_prompt_cached = None
                else:
                    self._repair_micro_in_update += 1

                return loss.detach()

    def _repair_loss_on_prompts(self, model, soft_prompts: torch.Tensor) -> torch.Tensor:
        self._require_ref_available(model, where="AlerDistill repair")

        cfg = self.latent_repair_cfg
        if str(cfg.repair.kl_direction).lower() != "reverse":
            raise ValueError("AlerDistill repair only supports kl_direction=reverse.")

        device = soft_prompts.device
        batch, prompt_len, _ = soft_prompts.shape
        tokenizer = self._get_tokenizer()
        suffix = tokenizer(cfg.probe_suffix_text, add_special_tokens=False, return_tensors="pt")
        suffix_ids = suffix["input_ids"].to(device)
        suffix_attn = suffix["attention_mask"].to(device)
        suffix_emb = self._embed_from_snapshot(suffix_ids, device=device, dtype=soft_prompts.dtype)
        suffix_len = suffix_emb.shape[1]

        inputs_embeds = torch.cat([soft_prompts, suffix_emb.expand(batch, -1, -1)], dim=1)
        attention = torch.ones((batch, prompt_len + suffix_len), device=device, dtype=torch.long)
        attention[:, prompt_len:] = suffix_attn.expand(batch, -1)

        with torch.no_grad():
            if self.ref_model is not None:
                self._ensure_ref_model(model)
                ref_last = self.ref_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention,
                    use_cache=False,
                ).logits[:, -1, :]
            else:
                model.set_adapter(self.ref_adapter)
                ref_last = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention,
                    use_cache=False,
                ).logits[:, -1, :]

        if self.new_adapter is not None and self._has_adapter_support(model):
            model.set_adapter(self.new_adapter)
        new_last = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention,
            use_cache=False,
        ).logits[:, -1, :]

        return kl_divergence(new_last, ref_last, reduction="mean")
