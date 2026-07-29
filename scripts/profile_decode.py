#!/usr/bin/env python3
"""Per-phase GPU profile of a CUDA decode step, plus optional config A/B sweeps.

Nsight Compute needs elevated GPU-counter permissions (ERR_NVGPUCTRPERM) that
are frequently unavailable on developer Windows machines, and decode is captured
into one CUDA graph anyway, which flattens per-kernel attribution. This drives
the in-tree profiler (include/lfm/backend/cuda/phase_profile.hpp) instead, which
needs neither.

It answers "where is the time actually going" before anyone writes a kernel.
That matters: a plausible weight-bandwidth argument once pointed at the GEMV
path here, and this profile showed 66% of decode was really a single-threaded
top-k loop in the sampler.

Examples
--------
  # phase breakdown
  python scripts/profile_decode.py --model path/to/model.gguf

  # compare configurations end-to-end (graph on, so these are real numbers)
  python scripts/profile_decode.py --model model.gguf --sweep

  # profile the serving-style config
  python scripts/profile_decode.py --model model.gguf --top-k 50 --temperature 0.8

Note: the phase profile requires --no-cuda-graph (added automatically), so its
absolute ms/token runs higher than the graph-enabled steady state. Use the
percentages to locate the bottleneck, and --sweep for real throughput.
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE_FLAGS = [
    "--prompt", "benchmark", "--raw",
    "--fused-residuals", "--fused-projections", "--fast-attention",
    "--gemm-backend", "cublaslt", "--kv-cache", "bf16",
    "--prefill-chunk", "256",
]


def find_binary(build: Path, name: str) -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    for candidate in (build / f"{name}{suffix}", build / "bin" / f"{name}{suffix}",
                      build / "Release" / f"{name}{suffix}"):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"cannot find {name} under {build} -- build first:\n"
                     f"  python scripts/dev.py build --backend cuda --build-type Release")


def run(binary: Path, model: Path, extra: list[str], profile: bool,
        prefill: int, decode: int, context: int) -> str:
    cmd = [str(binary), "--model", str(model), *BASE_FLAGS,
           "--benchmark-prefill-tokens", str(prefill),
           "--benchmark-decode", str(decode), "--benchmark-warmup", "1",
           "--max-new-tokens", str(decode), "--context", str(context), *extra]
    if profile:
        cmd.append("--no-cuda-graph")
    env = dict(os.environ)
    if profile:
        env["LFM_PROFILE_DECODE"] = "1"
    else:
        env.pop("LFM_PROFILE_DECODE", None)
    done = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if done.returncode != 0:
        sys.stderr.write(done.stdout + done.stderr)
        raise SystemExit(f"lfm25-run failed ({done.returncode})")
    return done.stdout + done.stderr


def tps(output: str) -> float | None:
    m = re.search(r"benchmark\.decode_tokens_per_second=([0-9.]+)", output)
    return float(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="GGUF or checkpoint dir")
    ap.add_argument("--build", type=Path, default=ROOT / "out" / "windows-cuda-release")
    ap.add_argument("--prefill", type=int, default=512)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="A/B common configurations end-to-end (graph enabled)")
    ap.add_argument("--no-profile", action="store_true",
                    help="skip the phase profile, only report throughput")
    args = ap.parse_args()

    binary = find_binary(args.build, "lfm25-run")
    extra: list[str] = []
    if args.top_k is not None:
        extra += ["--top-k", str(args.top_k)]
    if args.temperature is not None:
        extra += ["--temperature", str(args.temperature)]

    if not args.no_profile:
        out = run(binary, args.model, extra, True, args.prefill, args.decode, args.context)
        started = False
        for line in out.splitlines():
            if line.startswith("=== decode phase profile"):
                started = True
            if started:
                print(line)
        if not started:
            print("no phase profile emitted -- is this a CUDA build with "
                  "phase_profile.hpp wired into execution.cu?")
        speed = tps(out)
        if speed:
            print(f"\n(--no-cuda-graph throughput: {speed:.1f} tok/s; "
                  f"graph-enabled steady state is faster -- use --sweep)")

    if args.sweep:
        configs = [
            ("baseline (graph on)", []),
            ("no cuda graph", ["--no-cuda-graph"]),
            ("attention single", ["--attention-mode", "single"]),
            ("attention chunk 256", ["--attention-chunk-tokens", "256"]),
            ("kv cache int8", ["--kv-cache", "int8"]),
            ("top_k=50 t=0.8", ["--top-k", "50", "--temperature", "0.8"]),
        ]
        print("\n=== configuration sweep (graph enabled unless noted) ===")
        print(f"{'config':<24} {'tok/s':>9} {'vs base':>9}")
        base = None
        for name, flags in configs:
            out = run(binary, args.model, extra + flags, False,
                      args.prefill, args.decode, args.context)
            speed = tps(out)
            if speed is None:
                print(f"{name:<24} {'n/a':>9}")
                continue
            if base is None:
                base = speed
            print(f"{name:<24} {speed:>9.1f} {speed / base:>8.2f}x")


if __name__ == "__main__":
    main()
