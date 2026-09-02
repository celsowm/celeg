#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-relaxed-precision"
HOST_SOURCE = ROOT / "apps" / "benchmark" / "metal" / "relaxed_precision.mm"
GENERATED_SOURCE = BUILD_DIR / "relaxed_precision_lfm25.mm"
BINARY = BUILD_DIR / "celeg-metal-relaxed-precision-benchmark"


def run(command: list[str], *, cwd: pathlib.Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    # The original Metal microbenchmark predates LFM2.5-350M's 6656-wide FFN
    # and still names the older 4608 control shape. Materialize the real LFM2.5
    # dimensions so A/B timings match the GGUF used by the end-to-end benchmark.
    GENERATED_SOURCE.write_text(
        HOST_SOURCE.read_text(encoding="utf-8").replace("4608", "6656"),
        encoding="utf-8",
    )
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
        description="Build and run the Metal TensorOps relaxed-precision A/B benchmark."
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
