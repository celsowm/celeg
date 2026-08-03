#!/usr/bin/env python3
"""Reproducible native-GGUF CUDA comparison: Celeg versus llama.cpp.

The benchmark intentionally measures only the same on-device Q4_K_M file.
Neither engine is allowed to silently materialize the GGUF into BF16/INT8.
One complete run is discarded, then five measurements are reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "benchmarks" / "results" / "smollm3_cuda_q4_k_m_native.json"
LLAMA_REVISION = "6b36c2305644fd30db7cce3f4840c74574f31ce9"
SMOLLM3_REVISION = "4965cb60b150737b68a0408c36aeefb65078f894"
PREFILL_TOKENS = 512
DECODE_TOKENS = 128
BATCH = 512
CONTEXT = 1024
REPETITIONS = 5
MIN_WIN_RATIO = 1.05
MAX_NATIVE_WEIGHT_GIB = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_binary(root: Path, name: str) -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    for candidate in (root / f"{name}{suffix}", root / "bin" / f"{name}{suffix}",
                      root / "Release" / f"{name}{suffix}"):
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"cannot find {name} under {root}")


def find_smollm3_gguf() -> Path:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    snapshot = cache / "models--ggml-org--SmolLM3-3B-GGUF" / "snapshots" / SMOLLM3_REVISION
    model = snapshot / "SmolLM3-Q4_K_M.gguf"
    if not model.is_file():
        raise RuntimeError(f"SmolLM3 Q4_K_M is not cached at {model}")
    return model


def git_head(path: Path) -> str:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def gpu_info() -> dict[str, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return {"name": "unknown", "compute_capability": "unknown", "driver": "unknown"}
    name, capability, driver = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    return {"name": name, "compute_capability": capability, "driver": driver}


def parse_celeg(output: str) -> tuple[float, float, float]:
    values: dict[str, float] = {}
    for line in output.splitlines():
        if line.startswith("benchmark.") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = float(value)
    weight_match = re.search(r"memory\.weights=([0-9.]+) GiB", output)
    if not weight_match:
        raise RuntimeError("celeg-run did not emit memory.weights")
    return (values["benchmark.prefill_tokens_per_second"],
            values["benchmark.decode_tokens_per_second"], float(weight_match.group(1)))


def parse_llama(output: str) -> tuple[float, float]:
    prompt = re.search(
        rf'"n_prompt":\s*{PREFILL_TOKENS},.*?"avg_ts":\s*([0-9.]+)', output, re.S)
    decode = re.search(
        rf'"n_prompt":\s*0,\s*"n_gen":\s*{DECODE_TOKENS},.*?"avg_ts":\s*([0-9.]+)',
        output, re.S)
    if not prompt or not decode:
        raise RuntimeError("llama-bench JSON did not contain the requested rows")
    return float(prompt.group(1)), float(decode.group(1))


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--celeg-build", type=Path,
                        default=ROOT / "out" / "windows-cuda-relwithdebinfo")
    parser.add_argument("--llama-build", type=Path,
                        default=ROOT / "out" / "third_party" / "llama-cuda")
    args = parser.parse_args()

    model = (args.model or find_smollm3_gguf()).resolve()
    if model.name != "SmolLM3-Q4_K_M.gguf":
        raise SystemExit("the native comparison is pinned to SmolLM3-Q4_K_M.gguf")
    llama_source = args.llama_build.parent / "llama.cpp"
    if git_head(llama_source) != LLAMA_REVISION:
        raise SystemExit(f"llama.cpp must be {LLAMA_REVISION}; found {git_head(llama_source)}")
    celeg = find_binary(args.celeg_build, "celeg-run")
    llama = find_binary(args.llama_build, "llama-bench")
    env = dict(os.environ)
    cuda_bin = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64")
    env["PATH"] = os.pathsep.join([str(cuda_bin), str(llama.parent), env.get("PATH", "")])

    celeg_command = [str(celeg), "--model", str(model), "--weight-mode", "native",
                     "--prompt", "benchmark", "--raw", "--benchmark-prefill-tokens",
                     str(PREFILL_TOKENS), "--benchmark-decode", str(DECODE_TOKENS),
                     "--benchmark-warmup", "1", "--max-new-tokens", str(DECODE_TOKENS),
                     "--context", str(CONTEXT), "--prefill-chunk", str(BATCH),
                     "--kv-cache", "bf16", "--fused-residuals", "--fused-projections",
                     "--fast-attention", "--gemm-backend", "cublaslt", "--memory-report"]
    llama_command = [str(llama), "-m", str(model), "-p", str(PREFILL_TOKENS),
                     "-n", str(DECODE_TOKENS), "-b", str(BATCH), "-ub", str(BATCH),
                     "-ngl", "99", "-ctk", "bf16", "-ctv", "bf16", "-r", "1", "-o", "json"]

    celeg_prefill: list[float] = []
    celeg_decode: list[float] = []
    celeg_weights: list[float] = []
    llama_prefill: list[float] = []
    llama_decode: list[float] = []
    for iteration in range(REPETITIONS + 1):
        celeg_done = subprocess.run(celeg_command, text=True, capture_output=True, env=env, check=True)
        llama_done = subprocess.run(llama_command, text=True, capture_output=True, env=env, check=True)
        cp, cd, weights = parse_celeg(celeg_done.stdout + celeg_done.stderr)
        lp, ld = parse_llama(llama_done.stdout)
        if iteration:
            celeg_prefill.append(cp); celeg_decode.append(cd); celeg_weights.append(weights)
            llama_prefill.append(lp); llama_decode.append(ld)

    report = {
        "valid": False,
        "contract": {"model": "ggml-org/SmolLM3-3B-GGUF:SmolLM3-Q4_K_M.gguf",
                     "gguf_revision": SMOLLM3_REVISION, "weight_mode": "native",
                     "max_native_weight_gib": MAX_NATIVE_WEIGHT_GIB,
                     "prefill_tokens": PREFILL_TOKENS, "decode_tokens": DECODE_TOKENS,
                     "batch": BATCH, "ubatch": BATCH, "context": CONTEXT,
                     "kv_cache": "bf16", "gpu_layers": 99, "timed_repetitions": REPETITIONS,
                     "discarded_warmup_runs": 1, "minimum_win_ratio": MIN_WIN_RATIO},
        "gpu": gpu_info(), "llama_cpp_revision": LLAMA_REVISION,
        "celeg_commit": git_head(ROOT), "model_path": str(model),
        "model_sha256": sha256(model), "model_size_bytes": model.stat().st_size,
        "commands": {"celeg": celeg_command, "llama_cpp": llama_command},
        "celeg": {"prefill_tps": summary(celeg_prefill), "decode_tps": summary(celeg_decode),
                  "weight_gib": summary(celeg_weights)},
        "llama_cpp": {"prefill_tps": summary(llama_prefill), "decode_tps": summary(llama_decode)},
    }
    report["ratio"] = {
        "prefill": report["celeg"]["prefill_tps"]["mean"] / report["llama_cpp"]["prefill_tps"]["mean"],
        "decode": report["celeg"]["decode_tps"]["mean"] / report["llama_cpp"]["decode_tps"]["mean"],
    }
    report["valid"] = (report["celeg"]["weight_gib"]["mean"] <= MAX_NATIVE_WEIGHT_GIB and
                       report["ratio"]["prefill"] >= MIN_WIN_RATIO and
                       report["ratio"]["decode"] >= MIN_WIN_RATIO)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit("native GGUF win contract was not met")


if __name__ == "__main__":
    main()
