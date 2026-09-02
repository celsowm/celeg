#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-shortconv-batch-parallel"
SOURCE = ROOT / "apps" / "benchmark" / "metal" / "shortconv_batch_parallel.mm"
GENERATED_SOURCE = BUILD_DIR / "shortconv_batch_parallel_production.mm"
BINARY = BUILD_DIR / "celeg-metal-shortconv-batch-parallel-benchmark"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text(encoding="utf-8")
    source = source.replace(
        '            read_text("src/backend/metal/kernels/inference/convolution.metal") + "\\n" +\n'
        '            read_text("apps/benchmark/metal/shortconv_batch_parallel.metal");',
        '            read_text("src/backend/metal/kernels/inference/convolution.metal");',
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
        description="Benchmark production rows-parallel LFM2.5 Metal short convolution."
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
