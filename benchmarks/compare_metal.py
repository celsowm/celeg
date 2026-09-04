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
        "decode_tokens": 8,
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


def command_output(command):
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    return process.stdout.strip() if process.returncode == 0 else "unknown"


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
        default=Path("benchmarks/results/metal_llama_cpp_compare_official.json"),
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
        choices=["strict", "fast"],
        default="fast",
        help=(
            "Celeg numerical policy. 'fast' enables the explicit Metal performance "
            "path used for comparison with llama.cpp; 'strict' preserves the default "
            "numerical contract."
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


def celeg_env():
    return os.environ.copy()


def run_json(command, *, env=None, include_stderr=False):
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
        result = json.loads(process.stdout)
        return (result, process.stderr) if include_stderr else result
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


def validate_fast_preflight(command, model):
    env = celeg_env()
    result = run_json(
        [*command, "--decode-tokens", 0, "--warmup", 0, "--repetitions", 1,
         "--profile-dispatches", "counts"],
        env=env,
    )
    backend = result.get("backend", "")
    if result.get("numerical_policy") != "fast":
        raise RuntimeError("Celeg preflight did not request the fast numerical policy")
    if "policy_requested=fast" not in backend or "policy_effective=fast" not in backend:
        raise RuntimeError(f"Celeg fast policy is unavailable for {model.name}: {backend}")
    profile = result.get("profile_dispatches") or {}
    if profile.get("mode") != "counts" or profile.get("schedule_distorted"):
        raise RuntimeError("Celeg preflight did not produce an undistorted count profile")
    histogram = {
        entry["kernel"]: int(entry["count"])
        for entry in profile.get("prefill", {}).get("kernels", [])
    }
    tensor_dispatches = sum(
        count for name, count in histogram.items()
        if name.startswith("celeg_matmul_tensor_")
    )
    fast_tensor_dispatches = sum(
        count for name, count in histogram.items()
        if name.startswith("celeg_matmul_tensor_") and (
            "_relaxed" in name or "_fast" in name
        )
    )
    scalar_dispatches = {
        name: count for name, count in histogram.items()
        if (name == "celeg_matmul" or name.startswith("celeg_matmul_"))
        and not name.startswith("celeg_matmul_tensor_")
    }
    if scalar_dispatches:
        raise RuntimeError(
            f"Celeg fast preflight used scalar GEMM for {model.name}: {scalar_dispatches}"
        )
    if tensor_dispatches == 0:
        raise RuntimeError(f"Celeg fast preflight used no TensorOps GEMM for {model.name}")
    if fast_tensor_dispatches == 0:
        raise RuntimeError(
            f"Celeg fast preflight fell back to Strict TensorOps for {model.name}: {backend}"
        )
    return {
        "backend": backend,
        "build_commit": result.get("build_commit", "unknown"),
        "build_dirty": result.get("build_dirty", "unknown"),
        "metal_source_sha256": result.get("metal_source_sha256", "unknown"),
        "dispatch_histogram": histogram,
    }


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
    decode = selected[(0, decode_tokens)] if decode_tokens else None
    combined_ms = [value / 1.0e6 for value in combined["samples_ns"]]
    prompt_ms = [value / 1.0e6 for value in prompt["samples_ns"]]
    decode_ms = [] if decode is None else [
        value / 1.0e6 for value in decode["samples_ns"]
    ]
    if any(value <= 0.0 for value in combined_ms + prompt_ms + decode_ms):
        raise RuntimeError("llama.cpp phase measurement is not positive")
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
    celeg_process_env = celeg_env()
    celeg_command = [
        args.celeg,
        "--model", model,
        "--context", context,
        "--prompt-tokens", prompt_tokens,
        "--decode-tokens", decode_tokens,
        "--storage-mode", args.storage_mode,
        "--numerical-policy", args.celeg_policy,
    ]
    llama_command = [
        args.llama_bench,
        "-m", model,
        "-p", prompt_tokens,
        "-n", decode_tokens,
        "-pg", f"{prompt_tokens},{decode_tokens}",
        "-b", prompt_tokens,
        "-ub", prompt_tokens,
        "-ctk", "bf16",
        "-ctv", "bf16",
        "-ngl", 99,
        "-o", "json",
    ]
    preflight = None
    if args.celeg_policy == "fast":
        preflight = validate_fast_preflight(celeg_command, model)
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
                    [*command, "--warmup", 1, "--repetitions", 1],
                    env=celeg_process_env,
                )
                celeg_samples.append(celeg_result(result))
            else:
                result = run_json([*command, "-r", 1])
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
        "preflight": preflight,
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
            "combined_median_at_least_llama": (
                celeg_distribution["median_tokens_per_second"] >=
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
    celeg_provenance = {
        "executable": str(args.celeg.resolve()),
        "sha256": file_sha256(args.celeg),
    }
    llama_cpp_provenance = git_provenance(Path(".externals/llama.cpp"))
    llama_cpp_provenance.update({
        "executable": str(args.llama_bench.resolve()),
        "sha256": file_sha256(args.llama_bench),
    })
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
                "llama_cpp_commit": llama_cpp_provenance["commit"],
            }
            try:
                entry.update(collect_workload(model, args, workload))
                entry["celeg_commit"] = entry["preflight"]["build_commit"] \
                    if entry["preflight"] else entry["celeg"]["samples"][0]["raw"].get(
                        "build_commit", "unknown")
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

    build_commits = sorted({
        entry.get("celeg_commit", "unknown") for entry in results
        if entry["status"] == "ok"
    })
    metal_source_hashes = sorted({
        entry["preflight"]["metal_source_sha256"]
        for entry in results if entry.get("preflight")
    })
    celeg_provenance["build_commits"] = build_commits
    celeg_provenance["metal_source_sha256"] = metal_source_hashes
    complete_matrix = len(results) == len(EXPECTED_MODELS) * len(WORKLOADS)
    all_entries_valid = all(entry["status"] == "ok" for entry in results)
    all_phase_medians = all(
        entry.get("promotion", {}).get("prefill_median_at_least_llama", False) and
        entry.get("promotion", {}).get("decode_median_at_least_llama", False)
        for entry in results
    )
    all_combined_q1 = all(
        entry.get("promotion", {}).get("q1_celeg_at_least_llama", False)
        for entry in results
    )
    all_combined_medians = all(
        entry.get("promotion", {}).get("combined_median_at_least_llama", False)
        for entry in results
    )
    geometric_mean = (
        statistics.geometric_mean(
            entry["promotion"]["median_speedup"] for entry in results
            if entry["status"] == "ok"
        ) if any(entry["status"] == "ok" for entry in results) else 0.0
    )
    promotion_passed = (
        complete_matrix and all_entries_valid and all_phase_medians and
        all_combined_medians and all_combined_q1 and geometric_mean >= 1.10
    )
    document = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "environment": {
            "macos_product_version": command_output(["sw_vers", "-productVersion"]),
            "macos_build_version": command_output(["sw_vers", "-buildVersion"]),
            "sdk_version": command_output(["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
        },
        "celeg": celeg_provenance,
        "llama_cpp": llama_cpp_provenance,
        "configuration": {
            "workloads": workloads,
            "warmup": args.warmup,
            "per_timed_process_warmup": 1,
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
        "promotion": {
            "complete_official_matrix": complete_matrix,
            "all_entries_valid": all_entries_valid,
            "all_phase_medians_at_least_llama": all_phase_medians,
            "all_combined_medians_at_least_llama": all_combined_medians,
            "all_combined_q1_at_least_llama": all_combined_q1,
            "combined_geometric_mean_speedup": geometric_mean,
            "passed": promotion_passed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"saved {args.output}")
    if not all_entries_valid:
        return 1
    return 2 if complete_matrix and not promotion_passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
