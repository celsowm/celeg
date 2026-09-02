#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


WORKLOADS = {
    "interactive": {
        "context": 128,
        "prompt_tokens": 32,
        "decode_tokens": 8,
    },
    "throughput": {
        "context": 640,
        "prompt_tokens": 512,
        "decode_tokens": 128,
    },
}

EXPECTED_MODELS = {
    "LFM2.5-350M-BF16.gguf",
    "LFM2.5-350M-F16.gguf",
    "LFM2.5-350M-Q4_0.gguf",
    "LFM2.5-350M-Q4_K_M.gguf",
    "LFM2.5-350M-Q5_K_M.gguf",
    "LFM2.5-350M-Q6_K.gguf",
    "LFM2.5-350M-Q8_0.gguf",
    "LFM2.5-350M-QAD-Q4_0.gguf",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_provenance(directory):
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(
            ["git", "-C", str(directory), "diff", "--binary", "HEAD"], text=True)
    except subprocess.CalledProcessError:
        return {"commit": "unknown", "dirty": "unknown", "diff_sha256": "unknown"}
    return {
        "commit": commit,
        "dirty": bool(diff),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


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
    parser.add_argument("--model", dest="models", action="append", metavar="FILENAME")
    parser.add_argument("--workload", choices=["interactive", "throughput", "all"], default="all")
    parser.add_argument("--context", type=int)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--decode-tokens", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--storage-mode", choices=["shared", "private"], default="shared")
    parser.add_argument(
        "--celeg-policy",
        choices=["strict", "relaxed"],
        default="relaxed",
        help=(
            "Celeg numerical policy. 'relaxed' enables the explicit Metal fast path "
            "used for apples-to-apples performance comparison with llama.cpp TensorOps; "
            "'strict' preserves Celeg's default numerical contract."
        ),
    )
    return parser.parse_args()


def resolve_models(model_dir, requested):
    models = sorted(model_dir.glob("snapshots/*/*.gguf"))
    if not models:
        raise SystemExit(f"no GGUF files found under {model_dir}")
    names = {model.name for model in models}
    selected = set(requested or EXPECTED_MODELS)
    unknown = sorted(selected - EXPECTED_MODELS)
    if unknown:
        raise SystemExit(f"unsupported model selection: {', '.join(unknown)}")
    missing = sorted(selected - names)
    if missing:
        raise SystemExit(f"missing expected GGUF files: {', '.join(missing)}")
    return [model for model in models if model.name in selected]


def celeg_env(policy):
    env = os.environ.copy()
    if policy == "relaxed":
        env["CELEG_METAL_TENSOR_RELAXED_PRECISION"] = "1"
    else:
        env.pop("CELEG_METAL_TENSOR_RELAXED_PRECISION", None)
    return env


def run_json(command, *, env=None):
    process = subprocess.run(
        [str(value) for value in command],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if process.returncode != 0:
        detail = process.stderr.strip().splitlines()
        detail = "\n".join(detail[-12:])
        raise RuntimeError(f"exit {process.returncode}: {detail}")
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON: {error}") from error


def run_warmups(celeg_command, llama_command, warmup, celeg_process_env):
    if warmup <= 0:
        return
    run_json(
        [*celeg_command, "--warmup", warmup, "--repetitions", 1],
        env=celeg_process_env,
    )
    run_json([*llama_command, "-r", warmup, "--no-warmup"])


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
    if len(combined_ms) != len(prompt_ms):
        raise RuntimeError("llama.cpp prompt and combined sample counts differ")
    decode_ms = [total - prefill for total, prefill in zip(combined_ms, prompt_ms)]
    if any(value <= 0.0 for value in decode_ms):
        raise RuntimeError("llama.cpp decode measurement is not positive")
    return {
        "combined": distribution(combined_ms, prompt_tokens + decode_tokens),
        "prefill_batched": distribution(prompt_ms, prompt_tokens),
        "decode": distribution(decode_ms, decode_tokens) if decode_tokens else None,
        "raw": rows,
    }


def workload_args(args):
    if args.context is not None or args.prompt_tokens is not None or args.decode_tokens is not None:
        if args.context is None or args.prompt_tokens is None or args.decode_tokens is None:
            raise SystemExit("--context, --prompt-tokens and --decode-tokens must be provided together")
        return [{
            "name": "custom",
            "context": args.context,
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": args.decode_tokens,
        }]
    names = list(WORKLOADS) if args.workload == "all" else [args.workload]
    return [{"name": name, **WORKLOADS[name]} for name in names]


def collect_workload(model, args, workload):
    context = workload["context"]
    prompt_tokens = workload["prompt_tokens"]
    decode_tokens = workload["decode_tokens"]
    celeg_process_env = celeg_env(args.celeg_policy)
    celeg_command = [
        args.celeg,
        "--model", model,
        "--context", context,
        "--prompt-tokens", prompt_tokens,
        "--decode-tokens", decode_tokens,
        "--storage-mode", args.storage_mode,
    ]
    llama_command = [
        args.llama_bench,
        "-m", model,
        "-p", prompt_tokens,
        "-n", 0,
        "-pg", f"{prompt_tokens},{decode_tokens}",
        "-b", prompt_tokens,
        "-ub", prompt_tokens,
        "-ctk", "bf16",
        "-ctv", "bf16",
        "-ngl", 99,
        "-o", "json",
    ]
    run_warmups(celeg_command, llama_command, args.warmup, celeg_process_env)
    celeg_samples = []
    llama_samples = []
    order = []
    for index in range(args.repetitions):
        commands = (("celeg", celeg_command), ("llama_cpp", llama_command))
        if index % 2:
            commands = commands[::-1]
        for engine, command in commands:
            order.append(engine)
            if engine == "celeg":
                result = run_json(
                    [*command, "--warmup", 0, "--repetitions", 1],
                    env=celeg_process_env,
                )
                celeg_samples.append(celeg_result(result))
            else:
                result = run_json([*command, "-r", 1, "--no-warmup"])
                llama_samples.append(llama_rows(result, prompt_tokens, decode_tokens))
    celeg_combined = [sample["combined"]["samples_milliseconds"][0] for sample in celeg_samples]
    llama_combined = [sample["combined"]["samples_milliseconds"][0] for sample in llama_samples]
    celeg_prefill = [sample["prefill"]["samples_milliseconds"][0] for sample in celeg_samples]
    celeg_decode = [sample["decode"]["samples_milliseconds"][0] for sample in celeg_samples]
    llama_prompt = [sample["prefill_batched"]["samples_milliseconds"][0] for sample in llama_samples]
    llama_decode = [sample["decode"]["samples_milliseconds"][0] for sample in llama_samples]
    celeg_distribution = distribution(celeg_combined, prompt_tokens + decode_tokens)
    llama_distribution = distribution(llama_combined, prompt_tokens + decode_tokens)
    return {
        "configuration": {**workload, "celeg_policy": args.celeg_policy},
        "order": order,
        "celeg": {
            "policy": args.celeg_policy,
            "combined": celeg_distribution,
            "prefill": distribution(celeg_prefill, prompt_tokens),
            "decode": distribution(celeg_decode, decode_tokens),
            "samples": celeg_samples,
        },
        "llama_cpp": {
            "combined": llama_distribution,
            "prefill_batched": distribution(llama_prompt, prompt_tokens),
            "decode": distribution(llama_decode, decode_tokens),
            "samples": llama_samples,
        },
        "promotion": {
            "median_speedup": (
                celeg_distribution["median_tokens_per_second"] /
                llama_distribution["median_tokens_per_second"]
            ),
            "median_at_least_1_10x": (
                celeg_distribution["median_tokens_per_second"] >=
                1.10 * llama_distribution["median_tokens_per_second"]
            ),
            "q1_celeg_at_least_llama": (
                celeg_distribution["lower_quartile_tokens_per_second"] >=
                llama_distribution["lower_quartile_tokens_per_second"]
            ),
            "prefill_median_at_least_llama": (
                distribution(celeg_prefill, prompt_tokens)["median_tokens_per_second"] >=
                distribution(llama_prompt, prompt_tokens)["median_tokens_per_second"]
            ),
            "decode_median_at_least_llama": (
                distribution(celeg_decode, decode_tokens)["median_tokens_per_second"] >=
                distribution(llama_decode, decode_tokens)["median_tokens_per_second"]
            ),
        },
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
    models = resolve_models(args.model_dir, args.models)
    results = []
    workloads = workload_args(args)
    celeg_provenance = git_provenance(Path("."))
    llama_cpp_provenance = git_provenance(Path(".externals/llama.cpp"))
    print(f"Celeg numerical policy: {args.celeg_policy}")
    for workload in workloads:
        if workload["context"] <= 0 or workload["prompt_tokens"] <= 0 or workload["decode_tokens"] < 0:
            raise SystemExit(f"invalid workload dimensions: {workload}")
        if workload["prompt_tokens"] + workload["decode_tokens"] > workload["context"]:
            raise SystemExit(f"workload exceeds context: {workload}")
    for model in models:
        for workload in workloads:
            entry = {
                "model": str(model),
                "size_bytes": model.stat().st_size,
                "sha256": file_sha256(model),
                "workload": workload["name"],
                "celeg_policy": args.celeg_policy,
                "celeg_commit": celeg_provenance["commit"],
                "llama_cpp_commit": llama_cpp_provenance["commit"],
            }
            try:
                entry.update(collect_workload(model, args, workload))
                entry["status"] = "ok"
            except (RuntimeError, KeyError, IndexError) as error:
                entry["status"] = "error"
                entry["error"] = str(error)
            results.append(entry)
            if entry["status"] == "ok":
                celeg_rate = entry["celeg"]["combined"]["median_tokens_per_second"]
                llama_rate = entry["llama_cpp"]["combined"]["median_tokens_per_second"]
                celeg_decode = entry["celeg"]["decode"]["median_tokens_per_second"]
                llama_decode = entry["llama_cpp"]["decode"]["median_tokens_per_second"]
                print(
                    f"{model.name} [{workload['name']}, {args.celeg_policy}]: "
                    f"Celeg {celeg_rate:.2f} tok/s ({celeg_decode:.2f} decode) | "
                    f"llama.cpp {llama_rate:.2f} tok/s ({llama_decode:.2f} decode)"
                )
            else:
                print(f"{model.name} [{workload['name']}]: ERROR {entry['error']}", file=sys.stderr)

    document = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "celeg": celeg_provenance,
        "llama_cpp": llama_cpp_provenance,
        "configuration": {
            "workloads": workloads,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "storage_mode": args.storage_mode,
            "celeg_policy": args.celeg_policy,
            "models": [model.name for model in models],
            "llama_cpp": (
                "-p <prompt> -n 0 -pg <prompt>,<decode> "
                "-b <prompt> -ub <prompt> "
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
