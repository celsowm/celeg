#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-relaxed-precision"
HOST_SOURCE = ROOT / "apps" / "benchmark" / "metal" / "relaxed_precision.mm"
BASE_SHADER = ROOT / "src" / "backend" / "metal" / "kernels" / "tensor.metal"
EXPERIMENT_SHADER = ROOT / "apps" / "benchmark" / "metal" / "tensor_relaxed_precision.metal"
COMBINED_SHADER = BUILD_DIR / "relaxed_precision.metal"
AIR = BUILD_DIR / "relaxed_precision.air"
BINARY = BUILD_DIR / "celeg-metal-relaxed-precision-benchmark"


def run(command: list[str], *, cwd: pathlib.Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_SHADER.write_text(
        BASE_SHADER.read_text() + "\n" + EXPERIMENT_SHADER.read_text(),
        encoding="utf-8",
    )
    run([
        "xcrun", "--sdk", "macosx", "metal",
        "-c", str(COMBINED_SHADER), "-o", str(AIR),
    ])
    run([
        "xcrun", "--sdk", "macosx", "clang++",
        "-std=c++20", "-fobjc-arc",
        str(HOST_SOURCE),
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
        help="compile the experiment without executing it",
    )
    args = parser.parse_args()
    build()
    if not args.build_only:
        run([str(BINARY)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
