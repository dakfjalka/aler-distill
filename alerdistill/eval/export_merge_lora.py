from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model = args.base_model
    adapter_dir = str(Path(args.adapter_dir).resolve())
    out_dir = str(Path(args.out_dir).resolve())

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)

    # Merge LoRA into the base weights and unload adapters.
    model = model.merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)

    # Tokenizer is required by many servers.
    try:
        tok = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tok.save_pretrained(out_dir)


if __name__ == "__main__":
    main()
