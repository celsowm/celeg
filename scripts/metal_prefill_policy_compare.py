#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "manifests" / "metal_lfm25_350m_q4_k_m_pp512.json"
DEFAULT_BUILD_DIR = ROOT / "out" / "darwin-metal-release"
RESULT_DIR = ROOT / "benchmarks" / "results"
STRICT_RESULT = RESULT_DIR / "metal_lfm25_350m_q4_k_m_pp512_strict.json"
RELAXED_RESULT = RESULT_DIR / "metal_lfm25_350m_q4_k_m_pp512_relaxed.json"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def policy_env(*, relaxed: bool) -> dict[str, str]:
    env = os.environ.copy()
    if relaxed:
        env["CELEG_METAL_TENSOR_RELAXED_PRECISION"] = "1"
    else:
        env.pop("CELEG_METAL_TENSOR_RELAXED_PRECISION", None)
    return env


def benchmark(build_dir: pathlib.Path, output: pathlib.Path, *, relaxed: bool) -> dict[str, object]:
    run([
        sys.executable,
        str(ROOT / "benchmarks" / "run_metal_bench.py"),
        str(MANIFEST),
        "--build-dir", str(build_dir),
        "--output", str(output),
    ], env=policy_env(relaxed=relaxed))
    return json.loads(output.read_text(encoding="utf-8"))


def prefill_dispatch_profile(
    build_dir: pathlib.Path, checkpoint: pathlib.Path, *, relaxed: bool
) -> tuple[list[tuple[str, int]], list[tuple[str, float]]]:
    env = policy_env(relaxed=relaxed)
    env["CELEG_METAL_DISPATCH_PROFILE"] = "1"
    env["CELEG_METAL_GPU_PROFILE"] = "1"
    binary = build_dir / "celeg-metal-bench"
    result = subprocess.run([
        str(binary),
        "--model", str(checkpoint),
        "--context", "640",
        "--prompt-tokens", "512",
        "--decode-tokens", "0",
        "--warmup", "0",
        "--repetitions", "1",
    ], cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stdout + result.stderr)

    histogram: dict[str, int] = {}
    gpu_ms: dict[str, float] = {}
    section = ""
    for line in result.stderr.splitlines():
        if line == "metal gpu dispatch profile":
            section = "gpu"
            continue
        if line == "metal dispatch profile":
            section = "count"
            continue
        if section == "gpu":
            match = re.fullmatch(r"\s{2}(\S+)=([0-9.eE+-]+)ms", line)
            if match:
                gpu_ms[match.group(1)] = gpu_ms.get(match.group(1), 0.0) + float(match.group(2))
                continue
        if section == "count":
            match = re.fullmatch(r"\s{2}(\S+)=(\d+)", line)
            if match:
                histogram[match.group(1)] = histogram.get(match.group(1), 0) + int(match.group(2))

    if histogram and not gpu_ms:
        raise RuntimeError(
            "Metal dispatch counts were captured but no valid GPU timestamps were resolved:\n"
            + result.stderr
        )

    counts = sorted(histogram.items(), key=lambda item: (-item[1], item[0]))
    timings = sorted(gpu_ms.items(), key=lambda item: (-item[1], item[0]))
    return counts, timings


def metric(report: dict[str, object], name: str) -> float:
    return float(report[name])


def print_profile(
    label: str,
    counts: list[tuple[str, int]],
    timings: list[tuple[str, float]],
) -> None:
    gpu_by_name = dict(timings)
    names = [name for name, _ in timings]
    for name, _ in counts:
        if name not in gpu_by_name:
            names.append(name)
    count_by_name = dict(counts)

    print(f"\n{label} prefill GPU dispatch profile")
    print(f"  {'kernel':48} {'count':>7} {'gpu ms':>10}")
    for name in names:
        timing = gpu_by_name.get(name)
        timing_text = f"{timing:10.3f}" if timing is not None else f"{'n/a':>10}"
        print(f"  {name:48} {count_by_name.get(name, 0):7d} {timing_text}")
    if timings:
        print(f"  {'TOTAL SAMPLED':48} {'':7} {sum(value for _, value in timings):10.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build once and compare strict versus opt-in relaxed Metal pp512 performance."
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
    strict = benchmark(build_dir, STRICT_RESULT, relaxed=False)
    relaxed = benchmark(build_dir, RELAXED_RESULT, relaxed=True)

    strict_pp = metric(strict, "prefill_tokens_per_second")
    relaxed_pp = metric(relaxed, "prefill_tokens_per_second")
    strict_tg = metric(strict, "decode_tokens_per_second")
    relaxed_tg = metric(relaxed, "decode_tokens_per_second")

    print("\nMetal LFM2.5-350M Q4_K_M policy comparison")
    print(f"strict  prefill: {strict_pp:.1f} t/s")
    print(f"relaxed prefill: {relaxed_pp:.1f} t/s  ({relaxed_pp / strict_pp:.3f}x)")
    print(f"strict  decode:  {strict_tg:.1f} t/s")
    print(f"relaxed decode:  {relaxed_tg:.1f} t/s  ({relaxed_tg / strict_tg:.3f}x)")
    print(f"strict result:  {STRICT_RESULT}")
    print(f"relaxed result: {RELAXED_RESULT}")

    checkpoint = pathlib.Path(str(strict["checkpoint"]))
    strict_counts, strict_timings = prefill_dispatch_profile(
        build_dir, checkpoint, relaxed=False
    )
    relaxed_counts, relaxed_timings = prefill_dispatch_profile(
        build_dir, checkpoint, relaxed=True
    )
    print_profile("strict", strict_counts, strict_timings)
    print_profile("relaxed", relaxed_counts, relaxed_timings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
