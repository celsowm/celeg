#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-shortconv-batch-parallel"
SOURCE = ROOT / "apps" / "benchmark" / "metal" / "shortconv_batch_parallel.mm"
CURSORS = (0, 1, 2)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def generated_source(cursor: int) -> pathlib.Path:
    return BUILD_DIR / f"shortconv_batch_parallel_production_cursor{cursor}.mm"


def binary(cursor: int) -> pathlib.Path:
    return BUILD_DIR / f"celeg-metal-shortconv-batch-parallel-cursor{cursor}-benchmark"


def build(cursor: int) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text(encoding="utf-8")
    source = source.replace(
        '            read_text("src/backend/metal/kernels/inference/convolution.metal") + "\\n" +\n'
        '            read_text("apps/benchmark/metal/shortconv_batch_parallel.metal");',
        '            read_text("src/backend/metal/kernels/inference/convolution.metal");',
    )
    source = source.replace(
        "constexpr uint32_t kInitialCursor = 2;",
        f"constexpr uint32_t kInitialCursor = {cursor};",
    )
    source = source.replace(
        "constexpr uint32_t kRows[] = {128, 256, 512};",
        "constexpr uint32_t kRows[] = {1, 2, 3, 128, 256, 512};",
    )
    source = source.replace(
        'std::cout << "width=1024 cache_length=3 output_and_state_bit_exact=required\\n\\n";',
        f'std::cout << "width=1024 cache_length=3 initial_cursor={cursor} output_and_state_bit_exact=required\\n\\n";',
    )
    output_source = generated_source(cursor)
    output_source.write_text(source, encoding="utf-8")
    run([
        "xcrun", "--sdk", "macosx", "clang++",
        "-std=c++20", "-fobjc-arc",
        str(output_source),
        "-framework", "Foundation",
        "-framework", "Metal",
        "-o", str(binary(cursor)),
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
    if args.build_only:
        build(0)
        return 0
    for cursor in CURSORS:
        print(f"\n=== shortconv initial cursor {cursor} ===", flush=True)
        build(cursor)
        run([str(binary(cursor))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
