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
    # The original microbenchmark predates LFM2.5-350M's 6656-wide FFN. It also
    # used to abort at the first relaxed-precision mismatch, which hid the one
    # number we still need for diagnosis: how much performance the non-bit-exact
    # descriptor buys on M5. The generated harness reports the mismatch and then
    # continues timing; production still keeps strict precision and its bit gate.
    source = HOST_SOURCE.read_text(encoding="utf-8").replace("4608", "6656")
    source = source.replace(
        "geometry=64x128xK64 threads=128 bit_exact=required\\n\\n",
        "geometry=64x128xK64 threads=128 bit_exact=reported_not_required_for_timing\\n\\n",
    )
    needle = (
        "                require_bit_exact(queue, baseline, relaxed, weights, input,\n"
        "                                  tokens, shape, row_bytes);"
    )
    replacement = (
        "                try {\n"
        "                    require_bit_exact(queue, baseline, relaxed, weights, input,\n"
        "                                      tokens, shape, row_bytes);\n"
        "                } catch (const std::exception& mismatch) {\n"
        "                    std::cout << \"bit_exact=no shape=\" << shape.label\n"
        "                              << \" pp\" << tokens << \" detail=\"\n"
        "                              << mismatch.what() << '\\n';\n"
        "                }"
    )
    if needle not in source:
        raise RuntimeError("relaxed-precision harness gate site changed")
    GENERATED_SOURCE.write_text(source.replace(needle, replacement), encoding="utf-8")
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
        description="Build and run the Metal TensorOps relaxed-precision diagnostic A/B benchmark."
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
