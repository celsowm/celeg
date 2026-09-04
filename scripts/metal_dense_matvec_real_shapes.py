#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build" / "metal-dense-matvec-real-shapes"
BASE_SOURCE = ROOT / "apps" / "benchmark" / "metal" / "dense_matvec_vector.mm"
MATERIALIZED_SOURCE = BUILD_DIR / "dense_matvec_real_shapes.mm"
BINARY = BUILD_DIR / "celeg-metal-dense-matvec-real-shapes-benchmark"

OLD_SHAPES = '''constexpr Shape kShapes[] = {
    {"proj_1024x1024", 1024, 1024},
    {"ffn_up_4608x1024", 4608, 1024},
    {"ffn_down_1024x4608", 1024, 4608},
};'''

REAL_SHAPES = '''constexpr Shape kShapes[] = {
    {"attn_kv_512x1024", 512, 1024},
    {"proj_1024x1024", 1024, 1024},
    {"conv_in_3072x1024", 3072, 1024},
    {"ffn_up_4608x1024", 4608, 1024},
    {"ffn_down_1024x4608", 1024, 4608},
    {"lm_head_65536x1024", 65536, 1024},
};'''

OLD_ITERATIONS = '''                constexpr int repetitions = 7;
                constexpr int iterations = 64;'''

REAL_ITERATIONS = '''                constexpr int repetitions = 7;
                const int iterations = shape.rows >= 65536u ? 8 :
                    (shape.rows >= 4096u ? 32 : 64);'''


def run(command: list[str], *, cwd: pathlib.Path = ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def materialize() -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    source = BASE_SOURCE.read_text()
    if OLD_SHAPES not in source:
        raise SystemExit("dense matvec benchmark shape block changed; update this runner")
    if OLD_ITERATIONS not in source:
        raise SystemExit("dense matvec benchmark iteration block changed; update this runner")
    source = source.replace(OLD_SHAPES, REAL_SHAPES, 1)
    source = source.replace(OLD_ITERATIONS, REAL_ITERATIONS, 1)
    MATERIALIZED_SOURCE.write_text(source)


def build() -> None:
    materialize()
    run([
        "xcrun", "--sdk", "macosx", "clang++",
        "-std=c++20", "-fobjc-arc",
        str(MATERIALIZED_SOURCE),
        "-framework", "Foundation",
        "-framework", "Metal",
        "-o", str(BINARY),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run scalar vs vec4 vs vec8 dense Metal matvec A/B on the real "
            "LFM2.5-350M decode shapes, including the LM head."
        )
    )
    parser.add_argument(
        "--build-only", action="store_true",
        help="materialize and compile the Objective-C++ harness without running it",
    )
    args = parser.parse_args()
    build()
    if not args.build_only:
        print("Real decode shape usage per token:")
        print("  attn_kv_512x1024:   12  (K+V across 6 attention layers)")
        print("  proj_1024x1024:     22  (10 conv out + 6 Q + 6 attention out)")
        print("  conv_in_3072x1024:  10")
        print("  ffn_up_4608x1024:   32  (gate+up across 16 layers)")
        print("  ffn_down_1024x4608: 16")
        print("  lm_head_65536x1024:  1")
        print("  total:               93 matvecs/token\n")
        run([str(BINARY)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
