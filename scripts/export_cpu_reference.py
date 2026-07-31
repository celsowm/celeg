#!/usr/bin/env python3
"""Export official-model tokens and prefill logits for the C++ CPU comparator.

This script is validation-only. Python/Transformers are never runtime
requirements of libceleg_cpu.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system", default="")
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    if args.raw:
        text = args.prompt
    else:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=args.raw)
    with torch.inference_mode():
        logits = model(**encoded).logits[0, -1].float().cpu().numpy()
    tokens = encoded.input_ids[0].to(torch.int32).cpu().numpy()
    tokens.astype("<i4", copy=False).tofile(output / "tokens.i32")
    logits.astype("<f4", copy=False).tofile(output / "prefill_logits.f32")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "prompt": args.prompt,
                "system": args.system,
                "raw": args.raw,
                "token_count": int(tokens.size),
                "vocab_size": int(logits.size),
                "torch_dtype": "float32",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
