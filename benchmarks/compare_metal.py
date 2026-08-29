#!/usr/bin/env python3

import argparse
import json
import platform
import statistics
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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=15)
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


def distribution(milliseconds, token_count):
    rates = [token_count * 1000.0 / value for value in milliseconds]
    ordered = sorted(rates)
    return {
        "samples_milliseconds": milliseconds,
        "samples_tokens_per_second": rates,
        "median_milliseconds": statistics.median(milliseconds),
        "median_tokens_per_second": statistics.median(rates),
        "lower_quartile_tokens_per_second": ordered[(len(ordered) - 1) // 4],
        "upper_quartile_tokens_per_second": ordered[(len(ordered) - 1) * 3 // 4],
    }


def llama_rows(rows, prompt_tokens, decode_tokens):
    selected = {}
    for row in rows:
        key = (row.get("n_prompt"), row.get("n_gen"))
        selected[key] = row
    combined = selected[(prompt_tokens, decode_tokens)]
    prompt = selected[(prompt_tokens, 0)]
    combined_ms = [value / 1.0e6 for value in combined["samples_ns"]]
    prompt_ms = [value / 1.0e6 for value in prompt["samples_ns"]]
    return {
        "combined": distribution(combined_ms, prompt_tokens + decode_tokens),
        "prefill_batched": distribution(prompt_ms, prompt_tokens),
        "raw": rows,
    }


def celeg_result(result):
    total_tokens = result["prompt_tokens"] + result["decode_tokens"]
    combined_ms = [left + right for left, right in zip(
        result["prefill_samples_ms"], result["decode_samples_ms"])]
    return {
        "combined": distribution(combined_ms, total_tokens),
        "prefill": distribution(result["prefill_samples_ms"], result["prompt_tokens"]),
        "decode": distribution(result["decode_samples_ms"], result["decode_tokens"]),
        "execution": {
            key: result[key]
            for key in result
            if key.endswith("_ms") and key not in {"prefill_ms", "decode_ms",
                                                      "prefill_samples_ms", "decode_samples_ms"}
            or key.endswith("_buffers") or key.endswith("_dispatches")
            or key.startswith("resident_")
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
            celeg_command = [
                args.celeg,
                "--model", model,
                "--context", args.context,
                "--prompt-tokens", args.prompt_tokens,
                "--decode-tokens", args.decode_tokens,
                "--warmup", args.warmup,
                "--repetitions", args.repetitions,
            ]
            llama_command = [
                args.llama_bench,
                "-m", model,
                "-p", args.prompt_tokens,
                "-n", 0,
                "-pg", f"{args.prompt_tokens},{args.decode_tokens}",
                "-r", args.repetitions,
                "-b", args.prompt_tokens,
                "-ub", args.prompt_tokens,
                "-ctk", "bf16",
                "-ctv", "bf16",
                "-ngl", 99,
                "-o", "json",
            ]
            celeg = run_json(
                [
                    *celeg_command,
                ]
            )
            llama = run_json(llama_command)
            entry["status"] = "ok"
            entry["celeg"] = celeg_result(celeg)
            entry["llama_cpp"] = llama_rows(llama, args.prompt_tokens, args.decode_tokens)
        except (RuntimeError, KeyError) as error:
            entry["status"] = "error"
            entry["error"] = str(error)
        results.append(entry)
        if entry["status"] == "ok":
            celeg_rate = entry["celeg"]["combined"]["median_tokens_per_second"]
            llama_rate = entry["llama_cpp"]["combined"]["median_tokens_per_second"]
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
            "llama_cpp": (
                f"-p {args.prompt_tokens} -n 0 -pg {args.prompt_tokens},{args.decode_tokens} "
                f"-b {args.prompt_tokens} -ub {args.prompt_tokens} "
                "-ctk bf16 -ctv bf16 -ngl 99"
            ),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
