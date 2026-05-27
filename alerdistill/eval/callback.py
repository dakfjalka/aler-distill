# alerdistill/eval/callback.py
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback, TrainerState, TrainerControl
from omegaconf import DictConfig
from accelerate import Accelerator

from alerdistill.rollout.engines import OpenAICompatibleEngine, GenConfig
from alerdistill.rollout.sglang_service import SGLangService, SGLangServiceConfig
from alerdistill.eval.runner import run_all_evals


@dataclass
class HotUpdateEvalConfig:
    enabled: bool = True
    endpoint: str = "update_weights_from_disk"
    flush_cache: bool = True


class HotUpdateEvalCallback(TrainerCallback):
    """Evaluate periodically using a single persistent SGLang server.

    The callback:
      1) keeps a single SGLang service alive for the whole training run
      2) at each evaluation point, exports the current trainable weights to disk
      3) hot-updates the SGLang service (no restart)
      4) runs the eval suite through OpenAI-compatible API
    """

    def __init__(
        self,
        cfg: DictConfig,
        tokenizer,
        metric_logger,
        trainer,
        model_cfg,
        run_dir: str,
        prepared_eval_suite,
        infer_cuda_visible_devices: str = "",
        # If provided, the callback will attach to an already-running SGLang server
        # instead of starting/stopping it. Use this when you must start SGLang before
        # NCCL is initialized (e.g. when launching training via python -m + mp.spawn).
        external_service: Optional[dict] = None,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.metric_logger = metric_logger
        self.trainer = trainer
        self.model_cfg = model_cfg
        self.run_dir = run_dir
        self.prepared_eval_suite = prepared_eval_suite
        self.infer_cuda_visible_devices = infer_cuda_visible_devices

        self.enabled = bool(cfg.evaluation.enabled)
        self.schedule = str(cfg.evaluation.schedule)
        self.step_interval = int(cfg.evaluation.step_interval)
        self.epoch_interval = int(cfg.evaluation.epoch_interval)
        self.run_on_train_begin = bool(cfg.evaluation.run_on_train_begin)
        self.run_on_train_end = bool(cfg.evaluation.run_on_train_end)

        self.cleanup_tmp_dirs = bool(cfg.evaluation.cleanup_tmp_dirs)
        self.merge_lora_to_full = bool(cfg.evaluation.merge_lora_to_full)

        hu = getattr(cfg.evaluation, "hot_update", {})
        self.hot_update = HotUpdateEvalConfig(
            enabled=bool(getattr(hu, "enabled", True)),
            endpoint=str(getattr(hu, "endpoint", "update_weights_from_disk")),
            flush_cache=bool(getattr(hu, "flush_cache", True)),
        )

        rollout = cfg.rollout
        self.gen_cfg = GenConfig(**rollout.gen)
        model_name = rollout.model_name or cfg.model.name

        self.service_cfg = SGLangServiceConfig(
            host=str(rollout.host),
            port=int(rollout.port),
            work_dir=str(Path(self.run_dir) / "sglang_service"),
            startup_timeout_s=int(rollout.startup_timeout_s),
            shutdown_timeout_s=int(rollout.shutdown_timeout_s),
            extra_args=list(rollout.extra_args) if rollout.extra_args is not None else [],
            env=dict(rollout.env) if rollout.env is not None else {},
        )

        self._manage_service = external_service is None

        if external_service is not None:
            # Attach to already-running service.
            host = str(external_service.get("host", self.service_cfg.host))
            port = int(external_service.get("port", self.service_cfg.port))
            if port <= 0:
                raise ValueError("external_service must provide a valid 'port'")
            self.service_cfg.host = host
            self.service_cfg.port = port

        self._service = SGLangService(self.service_cfg, managed=self._manage_service)
        self._engine: Optional[OpenAICompatibleEngine] = None
        self._model_name = str(model_name)

    def _is_rank0(self) -> bool:
        # Prefer trainer's notion of rank0.
        try:
            return bool(self.trainer.is_world_process_zero())
        except Exception:
            return True
    
    def _wait_for_everyone(self):
        acc: Optional[Accelerator] = getattr(self.trainer, "accelerator", None)
        if acc is not None:
            acc.wait_for_everyone()
            return
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()

    # -------------------- lifecycle --------------------

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.enabled:
            return
        rank = int(os.environ.get("RANK", -1))
        # Run eval only on rank0 in distributed training.

        print(f"[rank{rank}][HotUpdateEvalCallback] on_train_begin called.", flush=True)
        # Start the persistent server on the *base* model. We'll hot-update weights at eval time.
        # If external_service was provided, this call only waits for readiness.

        if self._is_rank0():
            print(f"[rank{rank}][HotUpdateEvalCallback] Starting SGLang service at {self.service_cfg.host}:{self.service_cfg.port}...", flush=True)
            self._service.start(self.cfg.model.name, cuda_visible_devices=self.infer_cuda_visible_devices)
            self._engine = OpenAICompatibleEngine(
                base_url=self._service.openai_base_url,
                api_key="EMPTY",
                model=self._model_name,
            )
            print(f"[rank{rank}][HotUpdateEvalCallback] SGLang service started at {self._service.root_url}", flush=True)
        else:
            print(f"[rank{rank}][HotUpdateEvalCallback] Waiting for rank0 to start SGLang service...", flush=True)
            self._engine = None
        self._wait_for_everyone()


        if self.run_on_train_begin:
            print(f"[HotUpdateEvalCallback] Running evals at train begin (step {state.global_step})...", flush=True)
            self._execute_eval("train_begin", step=int(state.global_step))

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.enabled:
            return

        if self.run_on_train_end:
            self._execute_eval("train_end", step=int(state.global_step))

        if self._manage_service:
            rank = int(os.environ.get("RANK", -1))
            if self._is_rank0():
                print(f"[rank{rank}][HotUpdateEvalCallback] Terminating SGLang service...", flush=True)
                self._service.terminate()
            else:
                print(f"[rank{rank}][HotUpdateEvalCallback] Waiting for rank0 to terminate SGLang service...", flush=True)
            self._wait_for_everyone()

    # -------------------- scheduling hooks --------------------

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        rank = int(os.environ.get("RANK", -1))
        print(f"[rank{rank}][HotUpdateEvalCallback] on_step_end called at global step {state.global_step}.", flush=True)
        if not self.enabled or self.schedule != "step":
            return
        if self.step_interval <= 0:
            return
        step = int(state.global_step)
        if step == 0:
            return
        if step % self.step_interval == 0:
            print(f"[rank{rank}][HotUpdateEvalCallback] Running evals at step {step}...", flush=True)
            self._execute_eval(f"step_{step}", step=step)

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.enabled or self.schedule != "epoch":
            return
        if self.epoch_interval <= 0:
            return
        epoch = int(state.epoch or 0)
        if epoch % self.epoch_interval == 0:
            self._execute_eval(f"epoch_{epoch}", step=int(state.global_step))

    # -------------------- helpers --------------------

    def _export_model_for_hot_update(self, tmp_root: str) -> str:
        """Export current trainable weights to a directory usable by SGLang.

        Returns the directory path.
        """
        rank = int(os.environ.get("RANK", -1))
        tmp_root_p = Path(tmp_root)
        export_dir = tmp_root_p / "export"
        if self._is_rank0():
            print(f"[rank{rank}][HotUpdateEvalCallback] making temp dir {tmp_root}...", flush=True)
            export_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"[rank{rank}][HotUpdateEvalCallback] waiting for rank0 to make temp dir {tmp_root}...", flush=True)
        self._wait_for_everyone()

        # Full fine-tuning path: trainer.save_model exports a full model checkpoint.
        if not bool(self.model_cfg.peft.enabled):
            self.trainer.save_model(str(export_dir))
            return str(export_dir)

        # LoRA path: either export adapters only (not supported by SGLang), or merge.
        adapter_dir = tmp_root_p / "adapter"
        if self._is_rank0():
            print(f"[rank{rank}][HotUpdateEvalCallback] exporting LoRA adapters to {adapter_dir}...", flush=True)
            adapter_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"[rank{rank}][HotUpdateEvalCallback] waiting for rank0 to create adapter dir {adapter_dir}...", flush=True)
        self._wait_for_everyone()
        self.trainer.save_model(str(adapter_dir))

        if not self.merge_lora_to_full:
            raise RuntimeError(
                "LoRA is enabled, but evaluation.merge_lora_to_full=false. "
                "SGLang hot-update expects full weights. Set evaluation.merge_lora_to_full=true."
            )

        merged_dir = tmp_root_p / "merged"
        if self._is_rank0():
            print(f"[rank{rank}][HotUpdateEvalCallback] merging LoRA adapters into full model at {merged_dir}...", flush=True)
            merged_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"[rank{rank}][HotUpdateEvalCallback] waiting for rank0 to create merged dir {merged_dir}...", flush=True)
        self._wait_for_everyone()

        # Run merge in a separate process so we don't mutate the training model.
        env = os.environ.copy()
        if self.infer_cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.infer_cuda_visible_devices

        cmd = [
            "python",
            "-m",
            "alerdistill.eval.export_merge_lora",
            "--base_model",
            str(self.cfg.model.name),
            "--adapter_dir",
            str(adapter_dir),
            "--out_dir",
            str(merged_dir),
        ]
        if self._is_rank0():
            print(f"[rank{rank}][HotUpdateEvalCallback] running merge command: {' '.join(cmd)}", flush=True)
            subprocess.check_call(cmd, env=env)
        else:
            print(f"[rank{rank}][HotUpdateEvalCallback] waiting for rank0 to run merge command...", flush=True)
        self._wait_for_everyone()
        return str(merged_dir)

    def _execute_eval(self, tag: str, step: int) -> None:
        if self._is_rank0() and self._engine is None:
            raise RuntimeError("Eval engine not initialized")

        rank = int(os.environ.get("RANK", -1))
        print(f"[rank{rank}][HotUpdateEvalCallback] Running eval '{tag}' at step {step}...", flush=True)
        eval_tmp_dir = Path(self.run_dir) / "eval_tmp_" / f"{tag}"
        try:
            # 1) export model
            print(f"[rank{rank}][HotUpdateEvalCallback] 1/4) Exporting model for hot-update...", flush=True)
            model_dir = self._export_model_for_hot_update(str(eval_tmp_dir))

            # 2) hot-update server
            if self.hot_update.enabled:
                if self._is_rank0():
                    print(f"[rank{rank}][HotUpdateEvalCallback] 2/4) Hot-updating SGLang service from {model_dir}...", flush=True)
                    self._service.hot_update_from_disk(model_dir, endpoint=self.hot_update.endpoint)
                    if self.hot_update.flush_cache:
                        self._service.flush_cache()
                else:
                    print(f"[rank{rank}][HotUpdateEvalCallback] 2/4) Waiting for rank0 to hot-update SGLang service...", flush=True)
                self._wait_for_everyone()

            # 3) run evals
            if self._is_rank0():
                print(f"[rank{rank}][HotUpdateEvalCallback] 3/4) Running eval suite via OpenAI-compatible API...", flush=True)
                metrics = run_all_evals(
                    engine=self._engine,
                    gen_cfg=self.gen_cfg,
                    eval_suite=self.prepared_eval_suite,
                    seed=int(self.cfg.data.eval.seed),
                    extra={"step": step}
                )
            else:
                print(f"[rank{rank}][HotUpdateEvalCallback] 3/4) Waiting for rank0 to run eval suite...", flush=True)
                metrics = None
            self._wait_for_everyone()

            # 4) log metrics
            if metrics:
                if self._is_rank0():
                    print(f"[rank{rank}][HotUpdateEvalCallback] 4/4) Logging eval metrics...", flush=True)
                    for k, v in metrics.items():
                        self.metric_logger.log({f"eval/{k}": v}, step=step)
                else:
                    print(f"[rank{rank}][HotUpdateEvalCallback] 4/4) Waiting for rank0 to log eval metrics...", flush=True)
            self._wait_for_everyone()
        except Exception as e:
            print(f"[rank{rank}][HotUpdateEvalCallback] Error during eval '{tag}': {e}", flush=True)
            raise e
        finally:
            if self.cleanup_tmp_dirs:
                if self._is_rank0():
                    print(f"[rank{rank}][HotUpdateEvalCallback] Cleaning up eval temp dir {eval_tmp_dir}...", flush=True)
                    shutil.rmtree(eval_tmp_dir, ignore_errors=True)
                else:
                    print(f"[rank{rank}][HotUpdateEvalCallback] Waiting for rank0 to clean up eval temp dir {eval_tmp_dir}...", flush=True)
                self._wait_for_everyone()
