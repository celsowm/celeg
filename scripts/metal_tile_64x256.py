#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-tile-64x256"
HOST_SOURCE = ROOT / "apps" / "benchmark" / "metal" / "tile_64x256.mm"
PRODUCTION_SHADER = ROOT / "src" / "backend" / "metal" / "kernels" / "tensor.metal"
EXPERIMENT_SHADER = ROOT / "apps" / "benchmark" / "metal" / "tensor_tile_64x256.metal"
COMBINED_SHADER = BUILD_DIR / "tensor_tile_64x256_combined.metal"
AIR_FILE = BUILD_DIR / "tensor_tile_64x256.air"
BINARY = BUILD_DIR / "celeg-metal-tile-64x256-benchmark"


def run(command: list[str], *, cwd: pathlib.Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_SHADER.write_text(
        PRODUCTION_SHADER.read_text(encoding="utf-8")
        + "\n"
        + EXPERIMENT_SHADER.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    run([
        "xcrun", "--sdk", "macosx", "metal", "-c",
        str(COMBINED_SHADER), "-o", str(AIR_FILE),
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
        description="Build and run the benchmark-only Metal 64x256xK64 tensor-tile experiment."
    )
    parser.add_argument(
        "--build-only", action="store_true",
        help="compile the Metal experiment and Objective-C++ harness without running it",
    )
    args = parser.parse_args()
    build()
    if not args.build_only:
        run([str(BINARY)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
