from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from alerdistill.utils.dtypes import resolve_dtype


def normalize_adapters(adapters: Dict[str, str] | None) -> Dict[str, str]:
    adapters = dict(adapters or {})
    return {key: str(adapters[key]) for key in ("new", "ref") if key in adapters}


@dataclass
class QuantConfig:
    enabled: bool = False
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class PeftConfig:
    enabled: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: Any = "auto"
    modules_to_save: Optional[List[str]] = None
    init_lora_weights: bool = True


@dataclass
class ModelConfig:
    name: str
    revision: Optional[str]
    trust_remote_code: bool
    dtype: str
    device_map: Optional[str]
    attn_implementation: str
    quantization: QuantConfig
    peft: PeftConfig
    adapters: Dict[str, str]

    def __post_init__(self):
        self.adapters = normalize_adapters(self.adapters)


def load_tokenizer(model_name: str, revision: Optional[str], trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_bnb_config(qcfg: QuantConfig):
    from transformers import BitsAndBytesConfig

    compute_dtype = resolve_dtype(qcfg.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=qcfg.load_in_4bit,
        bnb_4bit_quant_type=qcfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qcfg.bnb_4bit_use_double_quant,
    )


def load_model_and_tokenizer(cfg: ModelConfig):
    dtype = resolve_dtype(cfg.dtype)
    quant_cfg = _build_bnb_config(cfg.quantization) if cfg.quantization.enabled else None
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        revision=cfg.revision,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=dtype if not cfg.quantization.enabled else None,
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_implementation,
        quantization_config=quant_cfg,
    )
    tokenizer = load_tokenizer(cfg.name, cfg.revision, cfg.trust_remote_code)
    return model, tokenizer


def add_lora_adapters(
    model,
    peft_cfg: PeftConfig,
    adapter_new: str | None = None,
    adapter_ref: str | None = None,
):
    if not peft_cfg.enabled:
        return model
    if adapter_new is None or adapter_ref is None:
        raise ValueError("add_lora_adapters requires adapter_new and adapter_ref.")

    from peft import LoraConfig, TaskType

    target_modules = peft_cfg.target_modules
    if target_modules == "auto":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora = LoraConfig(
        r=peft_cfg.lora_r,
        lora_alpha=peft_cfg.lora_alpha,
        lora_dropout=peft_cfg.lora_dropout,
        bias=peft_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        modules_to_save=peft_cfg.modules_to_save,
        init_lora_weights=peft_cfg.init_lora_weights,
    )
    model.add_adapter(lora, adapter_name=adapter_new)
    model.add_adapter(lora, adapter_name=adapter_ref)
    copy_lora_adapter_weights(model, src=adapter_new, dst=adapter_ref)
    freeze_adapter(model, adapter_ref)
    model.set_adapter(adapter_new)
    return model


def freeze_adapter(model, adapter_name: str) -> None:
    for name, param in model.named_parameters():
        if f".{adapter_name}." in name:
            param.requires_grad_(False)


def copy_lora_adapter_weights(model, src: str, dst: str) -> None:
    for module in model.modules():
        for attr in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
            if not hasattr(module, attr):
                continue
            module_dict = getattr(module, attr)
            if isinstance(module_dict, (dict, torch.nn.ModuleDict)) and src in module_dict and dst in module_dict:
                for src_param, dst_param in zip(module_dict[src].parameters(), module_dict[dst].parameters()):
                    dst_param.data.copy_(src_param.data)

        if hasattr(module, "lora_alpha") and isinstance(getattr(module, "lora_alpha"), dict):
            if src in module.lora_alpha and dst in module.lora_alpha:
                module.lora_alpha[dst] = module.lora_alpha[src]
