#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-dense-stage16-sharedmem"
TEMPLATE_SOURCE = ROOT / "apps" / "benchmark" / "metal" / "dense_k32.mm"
GENERATED_SOURCE = BUILD_DIR / "dense_stage16_sharedmem.mm"
BINARY = BUILD_DIR / "celeg-metal-dense-stage16-sharedmem-benchmark"


def run(command: list[str], *, cwd: pathlib.Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise RuntimeError(f"dense K32 harness template no longer contains: {old}")
    return source.replace(old, new, 1)


def generate_host() -> None:
    source = TEMPLATE_SOURCE.read_text()
    source = replace_once(
        source,
        'apps/benchmark/metal/tensor_dense_k32.metal',
        'apps/benchmark/metal/tensor_dense_stage16.metal',
    )
    source = replace_once(
        source,
        '{"F16", "celeg_matmul_tensor_f16_fast", "celeg_matmul_tensor_f16_fast_k32",',
        '{"F16", "celeg_matmul_tensor_f16_fast_stage16", "celeg_matmul_tensor_f16_fast_stage16",',
    )
    source = replace_once(
        source,
        '{"BF16", "celeg_matmul_tensor_bf16_fast", "celeg_matmul_tensor_bf16_fast_k32",',
        '{"BF16", "celeg_matmul_tensor_bf16_fast_stage16", "celeg_matmul_tensor_bf16_fast_stage16",',
    )
    replacements = {
        'ffn_up_6656x1024': 'ffn_up_4608x1024',
        'ffn_down_1024x6656': 'ffn_down_1024x4608',
        '6656, 1024': '4608, 1024',
        '1024, 6656': '1024, 4608',
        'Metal dense TensorOps K-stage A/B': 'Metal dense stage16 shared-memory A/B',
        'baseline=64x128xK64 relaxed candidate=64x128xK32 relaxed numerics=reported':
            'same K32 stage16 kernel: baseline=8KiB dynamic TG memory candidate=4KiB numerics=reported',
        'K64 ms': '8KiB ms',
        'K32 ms': '4KiB ms',
        'K64 stage': '8KiB alloc',
        'K32 stage': '4KiB alloc',
        'staging_count(shape, tokens, kBaselineTileK)': 'kBaselineWeightTileBytes',
        'staging_count(shape, tokens, kCandidateTileK)': 'kCandidateWeightTileBytes',
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
            "Compare 8 KiB versus 4 KiB dynamic threadgroup allocation for the exact same "
            "dense K32 stage16 Metal kernel on the real LFM2.5 dense shapes."
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
