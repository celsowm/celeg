#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-dense-stage16"
TEMPLATE_SOURCE = ROOT / "apps" / "benchmark" / "metal" / "dense_k32.mm"
GENERATED_SOURCE = BUILD_DIR / "dense_stage16.mm"
BINARY = BUILD_DIR / "celeg-metal-dense-stage16-benchmark"


def run(command: list[str], *, cwd: pathlib.Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def generate_host() -> None:
    source = TEMPLATE_SOURCE.read_text()
    replacements = {
        'apps/benchmark/metal/tensor_dense_k32.metal':
            'apps/benchmark/metal/tensor_dense_stage16.metal',
        'celeg_matmul_tensor_f16_fast_k32':
            'celeg_matmul_tensor_f16_fast_stage16',
        'celeg_matmul_tensor_bf16_fast_k32':
            'celeg_matmul_tensor_bf16_fast_stage16',
        'ffn_up_6656x1024': 'ffn_up_4608x1024',
        'ffn_down_1024x6656': 'ffn_down_1024x4608',
        '6656, 1024': '4608, 1024',
        '1024, 6656': '1024, 4608',
        'candidate=64x128xK32 relaxed':
            'candidate=64x128xK32 stage16 relaxed',
        'K32 ms': 'stage16 ms',
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"dense K32 harness template no longer contains: {old}")
        source = source.replace(old, new)
    GENERATED_SOURCE.write_text(source)


def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    generate_host()
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
        description=(
            "Build and run the benchmark-only dense Metal production K64 versus "
            "llama-style K32 16-wide staging experiment on LFM2.5 shapes."
        )
    )
    parser.add_argument(
        "--build-only", action="store_true",
        help="compile the generated Objective-C++ benchmark harness without running it",
    )
    args = parser.parse_args()
    build()
    if not args.build_only:
        run([str(BINARY)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
