#!/usr/bin/env python3
"""Reproducible CUDA GGUF benchmark for lfm25 versus llama.cpp.

Both engines receive the exact same Q4_K_M file.  The first of six one-shot
runs is discarded; the remaining five are summarized in one JSON report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "benchmarks" / "results" / "cuda_gguf_q4_k_m.json"
LLAMA_REVISION = "0e4a0362239713ea95a6864a17a8de4b0ad90d62"
GGUF_REVISION = "fa224d4cb60cffe61eb58726712ef255bb64d0b7"
BUILD_ENV: dict[str, str] | None = None


def run(cmd: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=True, text=True,
                          capture_output=capture, env=BUILD_ENV)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_binary(root: Path, name: str) -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    for candidate in (root / "bin" / "Release" / (name + suffix),
                      root / "bin" / (name + suffix),
                      root / "Release" / (name + suffix),
                      root / (name + suffix)):
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"cannot find {name} under {root}")


def git_head(path: Path) -> str:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def gpu_info() -> dict[str, str]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
                             "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return {"name": "unknown", "compute_capability": "unknown", "driver": "unknown"}
    name, cap, driver = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
    return {"name": name, "compute_capability": cap, "driver": driver}


def setup_llama(build_dir: Path) -> None:
    global BUILD_ENV
    if BUILD_ENV is None:
        sys.path.insert(0, str(ROOT))
        from scripts.dev import discover_environment
        environment = discover_environment("cuda", "86")
        if not environment.ok:
            raise RuntimeError("CUDA toolchain is not ready for llama.cpp")
        BUILD_ENV = environment.values
    source = build_dir.parent
    if not (source / ".git").is_dir():
        raise RuntimeError(f"llama.cpp checkout not found at {source}")
    compiler = shutil.which("cl.exe", path=BUILD_ENV["PATH"])
    if compiler is None:
        raise RuntimeError("MSVC compiler is not available for llama.cpp")
    if git_head(source) != LLAMA_REVISION:
        raise RuntimeError(
            f"llama.cpp must be pinned to {LLAMA_REVISION}; found {git_head(source)}")
    run(["cmake", "-S", str(source), "-B", str(build_dir), "-G", "Ninja",
         "-DGGML_NATIVE=ON", "-DGGML_CUDA=ON", "-DGGML_CUDA_F16=ON",
         "-DCMAKE_CUDA_ARCHITECTURES=86", "-DLLAMA_BUILD_TESTS=OFF",
         "-DLLAMA_BUILD_EXAMPLES=ON", "-DCMAKE_BUILD_TYPE=Release",
         "-DCMAKE_CUDA_FLAGS_INIT=--use-local-env",
         f"-DCMAKE_C_COMPILER={compiler}", f"-DCMAKE_CXX_COMPILER={compiler}"])
    run(["cmake", "--build", str(build_dir), "--config", "Release",
         "--target", "llama-bench", "-j", "8"])


def parse_lfm(stderr: str) -> tuple[float, float]:
    values = {}
    for line in stderr.splitlines():
        if line.startswith("benchmark.") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = float(value)
    return values["benchmark.prefill_tokens_per_second"], values["benchmark.decode_tokens_per_second"]


def parse_llama(stdout: str, expected: Path) -> tuple[float, float]:
    prompt = re.search(r'"n_prompt":\s*512,.*?"avg_ts":\s*([0-9.]+)', stdout, re.S)
    decode = re.search(r'"n_prompt":\s*0,\s*"n_gen":\s*128,.*?"avg_ts":\s*([0-9.]+)', stdout, re.S)
    if not prompt or not decode:
        raise RuntimeError("llama-bench JSON did not contain prompt/decode rows")
    return float(prompt.group(1)), float(decode.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--lfm-build", type=Path, default=ROOT / "out" / "windows-cuda-release")
    parser.add_argument("--llama-build", type=Path, default=ROOT / ".externals" / "llama.cpp" / "build-cuda")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--setup-llama", action="store_true")
    args = parser.parse_args()
    if args.reps <= 0 or args.gguf.suffix.lower() != ".gguf":
        raise SystemExit("expected a .gguf file and positive --reps")
    if args.setup_llama:
        setup_llama(args.llama_build)
    model = args.gguf.resolve()
    digest = sha256(model)
    lfm = find_binary(args.lfm_build, "lfm25-run")
    llama = find_binary(args.llama_build, "llama-bench")
    env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    cuda_bin = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin")
    env["PATH"] = os.pathsep.join([str(llama.parent), str(cuda_bin),
                                    str(cuda_bin / "x64"), env.get("PATH", "")])

    lfm_prefill: list[float] = []
    lfm_decode: list[float] = []
    llama_prefill: list[float] = []
    llama_decode: list[float] = []
    for iteration in range(args.reps + 1):
        lfm_result = subprocess.run([
            str(lfm), "--model", str(model), "--prompt", "benchmark",
            "--raw", "--benchmark-prefill-tokens", "512",
            "--benchmark-decode", "128", "--benchmark-warmup", "1",
            "--max-new-tokens", "128", "--context", "1024",
            "--fused-residuals", "--fused-projections", "--fast-attention",
            "--gemm-backend", "cublaslt", "--kv-cache", "bf16",
            "--prefill-chunk", "256",
        ], text=True, capture_output=True, env=env, check=True)
        lp, ld = parse_lfm(lfm_result.stderr)
        llama_result = subprocess.run([
            str(llama), "-m", str(model), "-p", "512", "-n", "128", "-r", "1",
            "-o", "json", "-t", "8", "-b", "256", "-ub", "256",
            "-ngl", "99", "-ctk", "bf16", "-ctv", "bf16",
        ], text=True, capture_output=True, env=env, check=True)
        cp, cd = parse_llama(llama_result.stdout, model)
        if iteration == 0:
            continue
        lfm_prefill.append(lp); lfm_decode.append(ld)
        llama_prefill.append(cp); llama_decode.append(cd)

    report = {
        "valid": True,
        "backend": "cuda",
        "quant": "Q4_K_M",
        "gguf_revision": GGUF_REVISION,
        "llama_cpp_revision": LLAMA_REVISION,
        "lfm25_commit": git_head(ROOT),
        "gpu": gpu_info(),
        "model_path": str(model),
        "model_sha256": digest,
        "model_size_bytes": model.stat().st_size,
        "threads": 8, "prefill_tokens": 512, "decode_tokens": 128,
        "batch": 256, "ubatch": 256, "kv": "bf16", "gpu_layers": 99,
        "repetitions": args.reps,
        "lfm25": {"prefill_tps": statistics.mean(lfm_prefill),
                  "prefill_stddev": statistics.stdev(lfm_prefill) if len(lfm_prefill) > 1 else 0.0,
                  "decode_tps": statistics.mean(lfm_decode),
                  "decode_stddev": statistics.stdev(lfm_decode) if len(lfm_decode) > 1 else 0.0},
        "llama_cpp": {"prefill_tps": statistics.mean(llama_prefill),
                      "prefill_stddev": statistics.stdev(llama_prefill) if len(llama_prefill) > 1 else 0.0,
                      "decode_tps": statistics.mean(llama_decode),
                      "decode_stddev": statistics.stdev(llama_decode) if len(llama_decode) > 1 else 0.0},
    }
    report["ratio"] = {
        "prefill": report["lfm25"]["prefill_tps"] / report["llama_cpp"]["prefill_tps"],
        "decode": report["lfm25"]["decode_tps"] / report["llama_cpp"]["decode_tps"],
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
