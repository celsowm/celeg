#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "manifests" / "metal_lfm25_350m_q4_k_m_pp512.json"
DEFAULT_BUILD_DIR = ROOT / "out" / "darwin-metal-release"
RESULT_DIR = ROOT / "benchmarks" / "results"
STRICT_RESULT = RESULT_DIR / "metal_lfm25_350m_q4_k_m_pp512_strict.json"
FAST_RESULT = RESULT_DIR / "metal_lfm25_350m_q4_k_m_pp512_fast.json"
EXPECTED_ATTENTION_LAYERS = 6
EXPECTED_FFN_LAYERS = 16


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def policy_env() -> dict[str, str]:
    env = os.environ.copy()
    return env


def benchmark(build_dir: pathlib.Path, output: pathlib.Path, *, fast: bool) -> dict[str, object]:
    run([
        sys.executable,
        str(ROOT / "benchmarks" / "run_metal_bench.py"),
        str(MANIFEST),
        "--build-dir", str(build_dir),
        "--output", str(output),
        "--numerical-policy", "fast" if fast else "strict",
    ], env=policy_env())
    return json.loads(output.read_text(encoding="utf-8"))


def prefill_dispatch_profile(
    build_dir: pathlib.Path, checkpoint: pathlib.Path, *, fast: bool
) -> list[tuple[str, int]]:
    binary = build_dir / "celeg-metal-bench"
    result = subprocess.run([
        str(binary),
        "--model", str(checkpoint),
        "--context", "640",
        "--prompt-tokens", "512",
        "--decode-tokens", "0",
        "--numerical-policy", "fast" if fast else "strict",
        "--warmup", "0",
        "--repetitions", "1",
        "--profile-dispatches", "counts",
    ], cwd=ROOT, env=policy_env(), text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)

    report = json.loads(result.stdout)
    histogram = {
        str(entry["kernel"]): int(entry["count"])
        for entry in report["profile_dispatches"]["prefill"]["kernels"]
    }
    return sorted(histogram.items(), key=lambda item: (-item[1], item[0]))


def metric(report: dict[str, object], name: str) -> float:
    return float(report[name])


def print_profile(label: str, entries: list[tuple[str, int]]) -> None:
    print(f"\n{label} prefill dispatch profile")
    for name, count in entries:
        print(f"  {name}={count}")


def verify_fast_policy(
    strict_entries: list[tuple[str, int]],
    fast_entries: list[tuple[str, int]],
) -> None:
    strict = dict(strict_entries)
    fast = dict(fast_entries)
    if strict.get("celeg_attention_tiled_simdgroup", 0) != 0:
        raise RuntimeError("strict Metal prefill unexpectedly used tiled relaxed attention")
    if strict.get("celeg_attention_batch", 0) != EXPECTED_ATTENTION_LAYERS:
        raise RuntimeError(
            "strict Metal prefill did not use the expected six causal attention dispatches"
        )
    if fast.get("celeg_attention_tiled_simdgroup", 0) != EXPECTED_ATTENTION_LAYERS:
        raise RuntimeError(
            "fast Metal prefill did not route all six LFM2.5 attention layers through tiled attention"
        )
    if fast.get("celeg_attention_batch", 0) != 0:
        raise RuntimeError("fast Metal pp512 silently fell back to one-exp attention")
    if strict.get("celeg_swiglu_batch_2d", 0) != EXPECTED_FFN_LAYERS:
        raise RuntimeError("strict Metal prefill did not use the expected sixteen SwiGLU dispatches")
    if strict.get("celeg_swiglu_batch_2d_relaxed", 0) != 0:
        raise RuntimeError("strict Metal prefill unexpectedly used relaxed SwiGLU")
    if fast.get("celeg_swiglu_batch_2d_relaxed", 0) != EXPECTED_FFN_LAYERS:
        raise RuntimeError("fast Metal prefill did not route all sixteen FFN layers through fast SwiGLU")
    if fast.get("celeg_swiglu_batch_2d", 0) != 0:
        raise RuntimeError("fast Metal pp512 silently fell back to strict SwiGLU")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build once and compare strict versus explicit fast Metal pp512 performance."
    )
    parser.add_argument("--build-dir", type=pathlib.Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 4)))
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    build_dir = args.build_dir.expanduser().resolve()

    if not args.no_build:
        run([
            sys.executable,
            str(ROOT / "scripts" / "dev.py"),
            "build",
            "--backend", "metal",
            "--celeg-tests", "off",
            "--build-type", "Release",
            "--build-dir", str(build_dir),
            "--jobs", str(args.jobs),
        ])

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    strict = benchmark(build_dir, STRICT_RESULT, fast=False)
    fast = benchmark(build_dir, FAST_RESULT, fast=True)

    strict_pp = metric(strict, "prefill_tokens_per_second")
    fast_pp = metric(fast, "prefill_tokens_per_second")
    strict_tg = metric(strict, "decode_tokens_per_second")
    fast_tg = metric(fast, "decode_tokens_per_second")

    print("\nMetal LFM2.5-350M Q4_K_M policy comparison")
    print(f"strict  prefill: {strict_pp:.1f} t/s")
    print(f"fast    prefill: {fast_pp:.1f} t/s  ({fast_pp / strict_pp:.3f}x)")
    print(f"strict  decode:  {strict_tg:.1f} t/s")
    print(f"fast    decode:  {fast_tg:.1f} t/s  ({fast_tg / strict_tg:.3f}x)")
    print(f"strict result:  {STRICT_RESULT}")
    print(f"fast result:    {FAST_RESULT}")

    checkpoint = pathlib.Path(str(strict["checkpoint"]))
    strict_profile = prefill_dispatch_profile(build_dir, checkpoint, fast=False)
    fast_profile = prefill_dispatch_profile(build_dir, checkpoint, fast=True)
    print_profile("strict", strict_profile)
    print_profile("fast", fast_profile)
    verify_fast_policy(strict_profile, fast_profile)
    print("\nfast policy gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
