#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-q4k-k32-llama-strict"
HOST_SOURCE = ROOT / "apps" / "benchmark" / "metal" / "q4k_k32.mm"
GENERATED_SOURCE = BUILD_DIR / "q4k_k32_llama_strict_lfm25.mm"
BINARY = BUILD_DIR / "celeg-metal-q4k-k32-llama-strict-benchmark"


def run(command: list[str], *, cwd: pathlib.Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    source = HOST_SOURCE.read_text(encoding="utf-8").replace("4608", "6656")
    source = source.replace(
        'apps/benchmark/metal/tensor_q4k_k32.metal',
        'apps/benchmark/metal/tensor_q4k_k32_llama_strict.metal',
    )
    source = source.replace(
        'celeg_matmul_tensor_q4k_k32"',
        'celeg_matmul_tensor_q4k_k32_llama_strict"',
    )
    source = source.replace(
        "Metal Q4_K K32 A/B on ",
        "Metal strict llama-style Q4_K K32 A/B on ",
    )
    source = source.replace(
        "candidate=64x128xK32/4KiB bit_exact=required",
        "candidate=llama-dequant 64x128xK32/4KiB relaxed=false bit_exact=required",
    )
    source = source.replace("K32 Q4_K is not bit-exact", "strict llama-style K32 Q4_K is not bit-exact")
    GENERATED_SOURCE.write_text(source, encoding="utf-8")
    run([
        "xcrun", "--sdk", "macosx", "clang++",
        "-std=c++20", "-fobjc-arc",
        str(GENERATED_SOURCE),
        "-framework", "Foundation",
        "-framework", "Metal",
        "-o", str(BINARY),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and run strict llama-style Q4_K K32 Metal A/B benchmark."
    )
    parser.add_argument(
        "--build-only", action="store_true",
        help="compile the Objective-C++ harness without running it",
    )
    args = parser.parse_args()
    build()
    if not args.build_only:
        run([str(BINARY)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
