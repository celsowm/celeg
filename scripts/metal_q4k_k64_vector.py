#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-q4k-k64-vector"
BASE_HOST_SOURCE = ROOT / "apps" / "benchmark" / "metal" / "relaxed_precision.mm"
GENERATED_SOURCE = BUILD_DIR / "q4k_k64_vector_lfm25.mm"
BINARY = BUILD_DIR / "celeg-metal-q4k-k64-vector-benchmark"


def run(command: list[str], *, cwd: pathlib.Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    source = BASE_HOST_SOURCE.read_text(encoding="utf-8")
    source = source.replace("4608", "6656")
    # Exercise the Q4_K minimum/bias term as well as the scale term. The base
    # relaxed benchmark intentionally used dmin=0, which was too weak a gate
    # for promoting a decoder rewrite into production.
    source = source.replace(
        "block[2] = 0x00;\n    block[3] = 0x00;  // zero minimum term",
        "block[2] = 0x00;\n    block[3] = 0x34;  // half(0.25), non-zero minimum term",
    )
    source = source.replace(
        "tensor_relaxed_precision.metal", "tensor_q4k_k64_vector.metal"
    )
    source = source.replace(
        "celeg_matmul_tensor_q4k_relaxed", "celeg_matmul_tensor_q4k_k64_vector"
    )
    source = source.replace(
        "Metal TensorOps relaxed-precision A/B", "Metal Q4_K vector K64 A/B"
    )
    source = source.replace("relaxed precision", "vector K64")
    source = source.replace("relaxed ms", "vector ms")
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
        description="Build and run the vectorized production-K64 Q4_K Metal A/B benchmark."
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
