#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-attention-one-exp"
SOURCE = ROOT / "apps" / "benchmark" / "metal" / "attention_one_exp.mm"
BINARY = BUILD_DIR / "celeg-metal-attention-one-exp-benchmark"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    run([
        "xcrun", "--sdk", "macosx", "clang++",
        "-std=c++20", "-fobjc-arc",
        str(SOURCE),
        "-framework", "Foundation",
        "-framework", "Metal",
        "-o", str(BINARY),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark bit-exact one-exp Metal causal attention."
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
