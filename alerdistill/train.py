# alerdistill/train.py
from __future__ import annotations

"""Training entrypoint.

Key requirements this file satisfies:
  - Launch via `python -m alerdistill.train` only (no accelerate/torchrun).
  - Optionally start an in-process SGLang server *before* any NCCL init.
  - Enable multi-GPU training via torch.multiprocessing.spawn + torch.distributed.
  - Keep code changes minimal by continuing to use TRL/Transformers Trainer.

Notes:
  - We partition GPUs into (train_gpus, infer_gpus). Training workers only see
    train GPUs (CUDA_VISIBLE_DEVICES=train_cvd). The SGLang server sees only
    infer GPUs (CUDA_VISIBLE_DEVICES=infer_cvd) in its own subprocess.
  - When distributed, we start SGLang once in the parent process (no NCCL) and
    attach from rank0 callback (no process management).
"""

import inspect
import os
import socket
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import timedelta
import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from alerdistill.utils.resources import GPUResources, partition_gpus


def _find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _model_cfg(cfg: DictConfig, *, distributed: bool) -> Any:
    """Build ModelConfig.

    When distributed, we must not use device_map='auto' because each worker will
    control placement via local_rank + DDP/FSDP.
    """
    from alerdistill.model.loading import ModelConfig, QuantConfig, PeftConfig

    device_map = None if distributed else (None if cfg.model.device_map in ("null", None) else str(cfg.model.device_map))

    return ModelConfig(
        name=str(cfg.model.name),
        revision=cfg.model.revision,
        trust_remote_code=bool(cfg.model.trust_remote_code),
        dtype=str(cfg.model.dtype),
        device_map=device_map,
        attn_implementation=str(cfg.model.attn_implementation),
        quantization=QuantConfig(**cfg.model.quantization),
        peft=PeftConfig(**cfg.model.peft),
        adapters=dict(cfg.model.adapters),
    )


def _train_data_cfg(cfg: DictConfig) -> Any:
    from alerdistill.data.loading import TrainDataConfig

    return TrainDataConfig(
        dataset_name=str(cfg.data.train.dataset_name),
        dataset_config_name=cfg.data.train.dataset_config_name,
        split=str(cfg.data.train.split),
        streaming=bool(cfg.data.train.streaming),
        format=str(cfg.data.train.format),
        text_field=cfg.data.train.text_field,
        max_length=int(cfg.data.train.max_length),
        shuffle=bool(cfg.data.train.shuffle),
        seed=int(cfg.data.train.seed),
        take_n=cfg.data.train.take_n,
        chat_template_enabled=bool(cfg.data.train.chat_template.enabled),
        chat_template_name=cfg.data.train.chat_template.template_name,
        filters=dict(cfg.data.train.filters) if getattr(cfg.data.train, "filters", None) is not None else None,
    )


def _val_split_cfg(cfg: DictConfig) -> Any:
    from alerdistill.data.loading import ValSplitConfig

    return ValSplitConfig(
        enabled=bool(cfg.data.val_split.enabled),
        mode=str(cfg.data.val_split.mode),
        fraction=float(cfg.data.val_split.fraction),
        num_examples=cfg.data.val_split.num_examples,
        seed=int(cfg.data.val_split.seed),
        max_examples=cfg.data.val_split.max_examples,
    )


def _latent_repair_cfg(cfg: DictConfig) -> Any:
    from alerdistill.latent_repair.searcher import (
        AlerDistillConfig,
        CandidateTopKSimilar,
        EntropyRegConfig,
        L2RegConfig,
        DiversityRegConfig,
        RepairConfig,
    )
    from alerdistill.latent_repair.danger import DangerConfig

    repair_cfg = cfg.latent_repair
    return AlerDistillConfig(
        enabled=bool(repair_cfg.enabled),
        distributed_search_mode=str(getattr(repair_cfg, "distributed_search_mode", "all_ranks")),
        prompt_length=int(repair_cfg.prompt_length),
        prompt_batch_size=int(repair_cfg.prompt_batch_size),
        search_steps=int(repair_cfg.search_steps),
        prompt_lr=float(repair_cfg.prompt_lr),
        temperature=float(repair_cfg.temperature),
        candidate_k=int(repair_cfg.candidate_k),
        candidate_mode=str(repair_cfg.candidate_mode),
        candidate_seed=int(repair_cfg.candidate_seed),
        candidate_topk_similar=CandidateTopKSimilar(**repair_cfg.candidate_topk_similar),
        entropy_reg=EntropyRegConfig(**repair_cfg.entropy_reg),
        l2_reg=L2RegConfig(**repair_cfg.l2_reg),
        diversity_reg=DiversityRegConfig(**repair_cfg.diversity_reg),
        probe_suffix_text=str(repair_cfg.probe_suffix.text),
        danger=DangerConfig(**repair_cfg.danger),
        repair=RepairConfig(**repair_cfg.repair),
        embedding_snapshot_enabled=bool(getattr(repair_cfg, "embedding_snapshot_enabled", True)),
        embedding_snapshot_dtype=str(getattr(repair_cfg, "embedding_snapshot_dtype", "float16")),
    )


def _build_sft_args(cfg: DictConfig, *, run_dir: str) -> Any:
    """Build SFTConfig with signature filtering (keeps compatibility across TRL versions)."""
    from trl import SFTConfig

    report_to = []
    if bool(cfg.logging.wandb.enabled):
        report_to.append("wandb")
    if bool(cfg.logging.tensorboard.enabled):
        report_to.append("tensorboard")
    
    sft_kwargs: Dict[str, Any] = dict(
        output_dir=os.path.join(run_dir, str(cfg.train.output_dir)),
        max_steps=cfg.train.max_steps,
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=str(cfg.train.lr_scheduler_type),
        warmup_ratio=float(cfg.train.warmup_ratio),
        weight_decay=float(cfg.train.weight_decay),
        logging_steps=int(cfg.train.logging_steps),
        save_steps=int(cfg.train.save_steps),
        bf16=bool(cfg.train.bf16),
        fp16=bool(cfg.train.fp16),
        max_grad_norm=float(cfg.train.max_grad_norm),
        packing=bool(cfg.train.sft.packing),
        remove_unused_columns=bool(cfg.train.sft.remove_unused_columns),
        dataset_num_proc=int(cfg.train.sft.dataset_num_proc),
        max_length=int(cfg.train.sft.max_seq_length),
        report_to=report_to,
    )

    extra_trainer_kwargs = dict(cfg.train.get("extra_trainer_kwargs", {}))
    if extra_trainer_kwargs:
        print("[SFTConfig] Adding extra_trainer_kwargs:", extra_trainer_kwargs, flush=True)
        sft_kwargs.update(extra_trainer_kwargs)

    # Optional: FSDP settings (supported by HF TrainingArguments).
    fsdp_cfg = getattr(cfg.train, "fsdp", None)
    if fsdp_cfg is not None and bool(getattr(fsdp_cfg, "enabled", False)):
        # Typical values: "full_shard auto_wrap" or "full_shard".
        sft_kwargs["fsdp"] = getattr(fsdp_cfg, "fsdp", "full_shard auto_wrap")
        # Pass through fsdp_config dict if present.
        if hasattr(fsdp_cfg, "config") and fsdp_cfg.config is not None:
            fsdp_config_dict = OmegaConf.to_container(fsdp_cfg.config, resolve=True)
            sft_kwargs["fsdp_config"] = fsdp_config_dict
            activation_ckpt = bool(fsdp_config_dict.get("activation_checkpointing", False))
            print("[SFTConfig] Using FSDP with config:", sft_kwargs["fsdp_config"], flush=True)

            # -------------------- Gradient checkpointing mapping --------------------
            # Your config uses: train.sft.gradient_checkpointing
            # Transformers/TrainingArguments expects: gradient_checkpointing (top-level arg)
            gc_from_cfg = bool(getattr(cfg.train.sft, "gradient_checkpointing", False))

            # Important invariant (Transformers constraint):
            # activation_checkpointing=True (FSDP) AND gradient_checkpointing=True (TrainingArgs) is NOT allowed.
            if activation_ckpt:
                if gc_from_cfg:
                    print(
                        "[SFTConfig] NOTE: train.sft.gradient_checkpointing=True is ignored because "
                        "FSDP activation_checkpointing=True. For FSDP full_shard, use activation_checkpointing only."
                    , flush=True)
                sft_kwargs["gradient_checkpointing"] = False
            else:
                sft_kwargs["gradient_checkpointing"] = gc_from_cfg

    # Filter unsupported args for the installed TRL.
    sig = inspect.signature(SFTConfig)
    valid = set(sig.parameters.keys())
    filtered = {k: v for k, v in sft_kwargs.items() if k in valid}
    dropped = sorted(set(sft_kwargs.keys()) - set(filtered.keys()))
    if dropped:
        print("[SFTConfig] Dropped unsupported args:", dropped, flush=True)
    return SFTConfig(**filtered)


def _train_worker(
    local_rank: int,
    world_size: int,
    cfg_dict: Dict[str, Any],
    run_dir: str,
    infer_cvd: str,
    master_addr: str,
    master_port: int,
    external_sglang: Optional[dict],
) -> None:
    """Single training worker (one GPU)."""
    if local_rank == 0:
        print(f"[rank {local_rank}][main] Starting training with world_size={world_size}", flush=True)
        print(f"[rank {local_rank}][main] run_dir={run_dir}", flush=True)
        print(f"[rank {local_rank}][main] master_addr={master_addr}, master_port={master_port}", flush=True)
        if external_sglang is not None:
            print(f"[rank {local_rank}][main] Using external SGLang service at {external_sglang['host']}:{external_sglang['port']}", flush=True)
    # NOTE: heavy imports happen here (after parent started SGLang).
    import torch
    from transformers import set_seed as hf_set_seed

    # Recreate cfg
    cfg = OmegaConf.create(cfg_dict)

    # Distributed env
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ.setdefault("LOCAL_WORLD_SIZE", str(world_size))
    os.environ.setdefault("NODE_RANK", "0")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1:
        backend = str(getattr(cfg.train.distributed, "backend", "nccl"))
        timeout_minutes = int(getattr(cfg.train.distributed, "timeout_minutes", 20))
        torch.distributed.init_process_group(backend=backend, rank=local_rank, world_size=world_size, timeout=timedelta(minutes=timeout_minutes))
        print(f"[rank {local_rank}][main] Initialized process group with backend={backend}", flush=True)

    # Make seeds rank-dependent to avoid identical shuffles.
    seed = int(cfg.train.seed) + int(local_rank)
    hf_set_seed(seed)

    # Rank0 only logging to avoid multiple wandb/tb writers.
    metric_logger = None
    if local_rank == 0:
        from alerdistill.logging.logger import MetricLogger, WandbConfig
        wandb_cfg = WandbConfig(**cfg.logging.wandb)
        metric_logger = MetricLogger(
            log_dir=os.path.join(run_dir, str(cfg.logging.log_dir)),
            project=str(cfg.logging.project),
            run_name=(None if cfg.logging.run_name in ("null", None) else str(cfg.logging.run_name)),
            wandb_cfg=wandb_cfg,
            tensorboard_enabled=bool(cfg.logging.tensorboard.enabled),
        )
        print(f"[rank {local_rank}][main] MetricLogger initialized at {run_dir}", flush=True)

    try:
        # Imports that depend on torch/transformers
        from alerdistill.data.loading import load_train_dataset, split_train_val
        from alerdistill.eval.prep import prepare_eval_suite
        from alerdistill.eval.callback import HotUpdateEvalCallback
        from alerdistill.model.loading import load_model_and_tokenizer, add_lora_adapters
        from alerdistill.trainers.alerdistill_sft_trainer import AlerDistillSFTTrainer

        # Model/tokenizer
        print(f"[rank {local_rank}][main] Loading model and tokenizer...", flush=True)
        distributed = world_size > 1
        mcfg = _model_cfg(cfg, distributed=distributed)
        model, tokenizer = load_model_and_tokenizer(mcfg)

        # Reference (frozen) model:
        #   - PEFT: use a frozen ref adapter on the same base model
        #   - no-PEFT: load an explicit ref_model (no deepcopy)
        ref_model = None
        if mcfg.peft.enabled:
            model = add_lora_adapters(model, mcfg.peft, adapter_new=mcfg.adapters["new"], adapter_ref=mcfg.adapters["ref"])
        else:
            # Only needed when the main path computes ref KL or latent repair.
            tr_cfg_tmp = _latent_repair_cfg(cfg)
            needs_ref_model = bool(cfg.method.ref_kl.enabled) or (tr_cfg_tmp is not None and bool(tr_cfg_tmp.enabled))
            if needs_ref_model:
                print(f"[rank {local_rank}][main] Loading frozen ref model (no-PEFT)...")
                ref_model, _tok2 = load_model_and_tokenizer(mcfg)

        # Data
        print(f"[rank {local_rank}][main] Loading training data...", flush=True)
        td_cfg = _train_data_cfg(cfg)
        train_ds = load_train_dataset(td_cfg)

        print(f"[rank {local_rank}][main] Preparing train/val split...", flush=True)
        train_ds, val_ds = split_train_val(train_ds, _val_split_cfg(cfg))
        if local_rank == 0:
            try:
                print(f"[rank {local_rank}][main] train ds size:", len(train_ds), flush=True)
            except Exception:
                pass

        # Args
        sft_args = _build_sft_args(cfg, run_dir=run_dir)

        # Eval suite (only rank0 will execute, but we can prepare on all ranks)
        prepared_eval_suite = prepare_eval_suite(
            OmegaConf.to_container(cfg.data.eval, resolve=True),
            seed=int(cfg.data.eval.seed),
            datasets={"train_val": val_ds} if val_ds is not None else {},
            default_data_source=str(td_cfg.format),
            run_dir=run_dir,
        )

        # Latent repair config
        tr_cfg = _latent_repair_cfg(cfg)

        trainer = AlerDistillSFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=train_ds,
            processing_class=tokenizer,
            latent_repair_cfg=tr_cfg,
            ref_adapter=(mcfg.adapters.get("ref") if mcfg.peft.enabled else None),
            new_adapter=(mcfg.adapters.get("new") if mcfg.peft.enabled else None),
            ref_model=ref_model,
            ref_kl_enabled=bool(cfg.method.ref_kl.enabled),
            ref_kl_lambda=float(cfg.method.ref_kl.lambda_kl),
            metric_logger=metric_logger,
        )

        # Eval callback: attach to external SGLang if provided.
        if bool(cfg.evaluation.enabled) and int(cfg.resources.infer_gpus) > 0:
            trainer.add_callback(
                HotUpdateEvalCallback(
                    cfg=cfg,
                    tokenizer=tokenizer,
                    metric_logger=metric_logger,
                    trainer=trainer,
                    model_cfg=cfg.model,
                    run_dir=run_dir,
                    prepared_eval_suite=prepared_eval_suite,
                    infer_cuda_visible_devices=infer_cvd,
                    external_service=external_sglang,
                )
            )
        print(f"[rank {local_rank}][main] Starting training...", flush=True)
        trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint)

    finally:
        if metric_logger is not None:
            metric_logger.close()
        if world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("[main] Training configuration:\n", OmegaConf.to_yaml(cfg))
    # -------------------- resources: partition GPUs --------------------
    res = GPUResources(train_gpus=int(cfg.resources.train_gpus), infer_gpus=int(cfg.resources.infer_gpus))
    train_cvd, infer_cvd = partition_gpus(res)
    print(f"[main] train CUDA_VISIBLE_DEVICES={train_cvd}, infer CUDA_VISIBLE_DEVICES={infer_cvd}", flush=True)

    # Training workers must only see train GPUs.
    if train_cvd:
        os.environ["CUDA_VISIBLE_DEVICES"] = train_cvd
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    # -------------------- run dir & config snapshot --------------------
    run_dir = HydraConfig.get().runtime.output_dir
    run_dir = str(Path(run_dir).resolve())
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # Save resolved config once (parent only).
    with open(Path(run_dir) / "resolved_config.yaml", "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))

    world_size = max(1, int(cfg.resources.train_gpus))

    # -------------------- optional: start SGLang before NCCL --------------------
    external_sglang: Optional[dict] = None
    sglang_service = None
    if bool(cfg.evaluation.enabled) and int(cfg.resources.infer_gpus) > 0:
        # Start service in parent when distributed. In single-process training,
        # the callback will manage the service itself.
        if world_size > 1:
            from alerdistill.rollout.sglang_service import SGLangService, SGLangServiceConfig

            rollout = cfg.rollout
            service_cfg = SGLangServiceConfig(
                host=str(rollout.host),
                port=int(rollout.port),
                work_dir=str(Path(run_dir) / "sglang_service"),
                startup_timeout_s=int(rollout.startup_timeout_s),
                shutdown_timeout_s=int(rollout.shutdown_timeout_s),
                extra_args=list(rollout.extra_args) if rollout.extra_args is not None else [],
                env=dict(rollout.env) if rollout.env is not None else {},
            )

            sglang_service = SGLangService(service_cfg, managed=True)
            # Important: start before any torch import / NCCL init.
            sglang_service.start(str(cfg.model.name), cuda_visible_devices=infer_cvd)
            external_sglang = {"host": sglang_service.host, "port": sglang_service.port}

    # -------------------- launch training --------------------
    try:
        if world_size == 1:
            # Single process: run worker directly.
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            print(f"[main] Starting single-process training...", flush=True)
            _train_worker(
                local_rank=0,
                world_size=1,
                cfg_dict=cfg_dict,
                run_dir=run_dir,
                infer_cvd=infer_cvd,
                master_addr="127.0.0.1",
                master_port=_find_free_port(),
                external_sglang=None,
            )
        else:
            # Multi-process: spawn nprocs=world_size.
            import torch.multiprocessing as mp

            master_addr = "127.0.0.1"
            master_port = _find_free_port(master_addr)
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            print(f"[main] Spawning {world_size} training workers...", flush=True)
            mp.spawn(
                _train_worker,
                args=(
                    world_size,
                    cfg_dict,
                    run_dir,
                    infer_cvd,
                    master_addr,
                    master_port,
                    external_sglang,
                ),
                nprocs=world_size,
                join=True,
            )
    finally:
        if sglang_service is not None:
            sglang_service.terminate()


if __name__ == "__main__":
    main()
