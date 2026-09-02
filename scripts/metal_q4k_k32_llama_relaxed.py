#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-q4k-k32-llama-relaxed"
HOST_SOURCE = ROOT / "apps" / "benchmark" / "metal" / "q4k_k32.mm"
GENERATED_SOURCE = BUILD_DIR / "q4k_k32_llama_relaxed_lfm25.mm"
BINARY = BUILD_DIR / "celeg-metal-q4k-k32-llama-relaxed-benchmark"


def run(command: list[str], *, cwd: pathlib.Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def build() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    source = HOST_SOURCE.read_text(encoding="utf-8").replace("4608", "6656")
    source = source.replace(
        'read_text("apps/benchmark/metal/tensor_q4k_k32.metal")',
        'read_text("src/backend/metal/kernels/tensor_q4k_relaxed.metal") + "\\n" +\n'
        '            read_text("apps/benchmark/metal/tensor_q4k_k32_llama_relaxed.metal")',
    )
    source = source.replace(
        'make_pipeline(device, library, "celeg_matmul_tensor_q4k");',
        'make_pipeline(device, library, "celeg_matmul_tensor_q4k_relaxed");',
    )
    source = source.replace(
        'make_pipeline(device, library, "celeg_matmul_tensor_q4k_k32");',
        'make_pipeline(device, library, "celeg_matmul_tensor_q4k_k32_llama_relaxed");',
    )
    source = source.replace(
        "Metal Q4_K K32 A/B on ",
        "Metal relaxed Q4_K K64 vs llama-style K32 A/B on ",
    )
    source = source.replace(
        "baseline=64x128xK64/8KiB candidate=64x128xK32/4KiB bit_exact=required",
        "baseline=relaxed K64/8KiB candidate=llama-dequant relaxed K32/4KiB bit_exact=reported_not_required",
    )
    source = source.replace(
        '<< std::setw(13) << "K64 ms"\n                  << std::setw(13) << "K32 ms"',
        '<< std::setw(13) << "K64 rel ms"\n                  << std::setw(13) << "K32 rel ms"',
    )
    source = source.replace(
        "block[2] = 0x00;\n    block[3] = 0x00;",
        "block[2] = 0x00;\n    block[3] = 0x34;  // half(0.25) minimum term",
    )
    source = source.replace(
        "K32 Q4_K is not bit-exact",
        "relaxed K32 Q4_K differs from relaxed K64",
    )
    required = """                require_bit_exact(device, queue, baseline, candidate, weights, input,
                                  tokens, shape, row_bytes);"""
    diagnostic = """                try {
                    require_bit_exact(device, queue, baseline, candidate, weights, input,
                                      tokens, shape, row_bytes);
                    std::cout << "bit_exact=yes shape=" << shape.label
                              << " pp" << tokens << '\\n';
                } catch (const std::exception& error) {
                    std::cout << "bit_exact=no shape=" << shape.label
                              << " pp" << tokens << " detail=" << error.what() << '\\n';
                }"""
    if required not in source:
        raise RuntimeError("q4k_k32.mm correctness call changed; update relaxed runner")
    source = source.replace(required, diagnostic)
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
        description=(
            "Compare Celeg relaxed Q4_K K64 against llama-style relaxed Q4_K K32 "
            "on the real LFM2.5-350M matrix shapes."
        )
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
