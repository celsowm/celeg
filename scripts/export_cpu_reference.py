#!/usr/bin/env python3
"""Export official-model tokens, logits, and greedy generations for C++ comparison.

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


def torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system", default="")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="float32"
    )
    parser.add_argument("--generate-tokens", type=int, default=0)
    args = parser.parse_args()
    if args.generate_tokens < 0:
        parser.error("--generate-tokens must be non-negative")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = torch_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
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

    generated_tokens = np.empty((0,), dtype=np.int32)
    if args.generate_tokens:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.generate_tokens,
                repetition_penalty=1.0,
                use_cache=True,
                pad_token_id=pad_token_id,
            )
        generated_tokens = (
            generated[0, encoded.input_ids.shape[1] :]
            .to(torch.int32)
            .cpu()
            .numpy()
        )
        generated_tokens.astype("<i4", copy=False).tofile(
            output / "generated_tokens.i32"
        )
        if generated_tokens.size and int(generated_tokens[0]) != int(np.argmax(logits)):
            raise RuntimeError(
                "Transformers greedy generation disagrees with exported prefill top-1"
            )

    (output / "metadata.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "prompt": args.prompt,
                "system": args.system,
                "raw": args.raw,
                "token_count": int(tokens.size),
                "vocab_size": int(logits.size),
                "torch_dtype": args.dtype,
                "generated_token_count": int(generated_tokens.size),
                "generated_tokens": [int(value) for value in generated_tokens],
                "prefill_top1": int(np.argmax(logits)),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
