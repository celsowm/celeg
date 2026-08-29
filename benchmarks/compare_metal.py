#!/usr/bin/env python3

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path.home() / ".cache/huggingface/hub/models--LiquidAI--LFM2.5-350M-GGUF",
    )
    parser.add_argument(
        "--celeg",
        type=Path,
        default=Path("out/darwin-metal-relwithdebinfo/celeg-metal-bench"),
    )
    parser.add_argument(
        "--llama-bench",
        type=Path,
        default=Path(".externals/llama.cpp/build-metal/bin/llama-bench"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/metal_llama_cpp_compare.json"),
    )
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    return parser.parse_args()


def resolve_models(model_dir):
    models = sorted(model_dir.glob("snapshots/*/*.gguf"))
    if not models:
        raise SystemExit(f"no GGUF files found under {model_dir}")
    return models


def run_json(command):
    process = subprocess.run(
        [str(value) for value in command],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip().splitlines()
        detail = "\n".join(detail[-12:])
        raise RuntimeError(f"exit {process.returncode}: {detail}")
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON: {error}") from error


def llama_rows(rows, prompt_tokens, decode_tokens):
    selected = {}
    for row in rows:
        key = (row.get("n_prompt"), row.get("n_gen"))
        selected[key] = row
    combined = selected[(prompt_tokens, decode_tokens)]
    prompt = selected[(prompt_tokens, 0)]
    return {
        "combined": {
            "tokens_per_second": combined["avg_ts"],
            "milliseconds": (prompt_tokens + decode_tokens) * 1000.0 / combined["avg_ts"],
            "samples_tokens_per_second": combined["samples_ts"],
        },
        "prefill_batched": {
            "tokens_per_second": prompt["avg_ts"],
            "milliseconds": prompt_tokens * 1000.0 / prompt["avg_ts"],
            "samples_tokens_per_second": prompt["samples_ts"],
        },
        "raw": rows,
    }


def celeg_result(result):
    total_ms = result["prefill_ms"] + result["decode_ms"]
    total_tokens = result["prompt_tokens"] + result["decode_tokens"]
    return {
        "combined": {
            "tokens_per_second": total_tokens * 1000.0 / total_ms,
            "milliseconds": total_ms,
        },
        "prefill_token_by_token": {
            "tokens_per_second": result["prefill_tokens_per_second"],
            "milliseconds": result["prefill_ms"],
        },
        "decode": {
            "tokens_per_second": result["decode_tokens_per_second"],
            "milliseconds": result["decode_ms"],
        },
        "raw": result,
    }


def main():
    args = parse_args()
    models = resolve_models(args.model_dir)
    results = []
    for model in models:
        entry = {"model": str(model), "size_bytes": model.stat().st_size}
        try:
            celeg = run_json(
                [
                    args.celeg,
                    "--model",
                    model,
                    "--context",
                    args.context,
                    "--prompt-tokens",
                    args.prompt_tokens,
                    "--decode-tokens",
                    args.decode_tokens,
                    "--warmup",
                    args.warmup,
                    "--repetitions",
                    args.repetitions,
                ]
            )
            llama = run_json(
                [
                    args.llama_bench,
                    "-m",
                    model,
                    "-p",
                    args.prompt_tokens,
                    "-n",
                    0,
                    "-pg",
                    f"{args.prompt_tokens},{args.decode_tokens}",
                    "-r",
                    args.repetitions,
                    "-b",
                    args.prompt_tokens,
                    "-ub",
                    args.prompt_tokens,
                    "-ctk",
                    "bf16",
                    "-ctv",
                    "bf16",
                    "-ngl",
                    99,
                    "-o",
                    "json",
                ]
            )
            entry["status"] = "ok"
            entry["celeg"] = celeg_result(celeg)
            entry["llama_cpp"] = llama_rows(llama, args.prompt_tokens, args.decode_tokens)
        except (RuntimeError, KeyError) as error:
            entry["status"] = "error"
            entry["error"] = str(error)
        results.append(entry)
        if entry["status"] == "ok":
            celeg_rate = entry["celeg"]["combined"]["tokens_per_second"]
            llama_rate = entry["llama_cpp"]["combined"]["tokens_per_second"]
            print(f"{model.name}: Celeg {celeg_rate:.2f} tok/s | llama.cpp {llama_rate:.2f} tok/s")
        else:
            print(f"{model.name}: ERROR {entry['error']}")

    document = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "configuration": {
            "context": args.context,
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": args.decode_tokens,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "llama_cpp": "-p 32 -n 0 -pg 32,8 -b 32 -ub 32 -ctk bf16 -ctv bf16 -ngl 99",
            "note": "Celeg prefill is token-by-token; llama.cpp prefill reference is batched.",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
