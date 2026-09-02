#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-q4k-static-stage128-strided"
HOST_SOURCE = ROOT / "apps" / "benchmark" / "metal" / "q4k_k32.mm"
GENERATED_SOURCE = BUILD_DIR / "q4k_static_stage128_strided_lfm25.mm"
BINARY = BUILD_DIR / "celeg-metal-q4k-static-stage128-strided-test"


def run(command: list[str], *, cwd: pathlib.Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    source = HOST_SOURCE.read_text(encoding="utf-8").replace("4608", "6656")
    source = source.replace(
        "block[2] = 0x00;\n    block[3] = 0x00;",
        "block[2] = 0x00;\n    block[3] = 0x34;  // half(0.25), exercise dmin",
    )
    source = source.replace(
        "constexpr NSUInteger kCandidateK = 32;",
        "constexpr NSUInteger kCandidateK = 128;",
    )
    source = source.replace(
        "apps/benchmark/metal/tensor_q4k_k32.metal",
        "src/backend/metal/kernels/tensor_q4k_static_stage128.metal",
    )
    source = source.replace(
        'celeg_matmul_tensor_q4k_k32"',
        'celeg_matmul_tensor_q4k_static_stage128"',
    )
    source = source.replace(
        "const uint32_t output_stride = shape.output_rows;",
        "const uint32_t output_stride = shape.output_rows * 2;",
    )
    source = source.replace(
        "[encoder setBuffer:output offset:0 atIndex:2];",
        "[encoder setBuffer:output "
        "offset:static_cast<NSUInteger>(shape.output_rows) * sizeof(float) atIndex:2];",
    )
    source = source.replace(
        "const size_t elements = static_cast<size_t>(tokens) * shape.output_rows;",
        "const size_t elements = static_cast<size_t>(tokens) * shape.output_rows * 2;",
    )
    source = source.replace(
        "static_cast<size_t>(tokens) * shape.output_rows * sizeof(float);",
        "static_cast<size_t>(tokens) * shape.output_rows * 2 * sizeof(float);",
    )
    source = source.replace(
        "Metal Q4_K K32 A/B on ",
        "Metal Q4_K static+stage128 strided-output A/B on ",
    )
    source = source.replace(
        "baseline=64x128xK64/8KiB candidate=64x128xK32/4KiB bit_exact=required",
        "layout=FFN-up offset=output_rows stride=2*output_rows bit_exact=required",
    )
    source = source.replace(
        "K32 Q4_K is not bit-exact",
        "static+stage128 strided Q4_K is not bit-exact",
    )
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
        description="Validate the production strict Q4_K fast path with FFN gate/up stride and offset."
    )
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    build()
    if not args.build_only:
        run([str(BINARY)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
