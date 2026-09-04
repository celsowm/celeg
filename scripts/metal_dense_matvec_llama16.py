#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-dense-matvec-llama16"
HOST_SOURCE = ROOT / "apps" / "benchmark" / "metal" / "dense_matvec_llama16.mm"
BINARY = BUILD_DIR / "celeg-metal-dense-matvec-llama16-benchmark"


def run(command: list[str], *, cwd: pathlib.Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
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
        description=(
            "Benchmark Celeg scalar dense decode matvec against a near-literal "
            "llama.cpp NB32/NF16 `_4` mapping on real LFM2.5-350M shapes."
        )
    )
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    build()
    if not args.build_only:
        run([str(BINARY)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
