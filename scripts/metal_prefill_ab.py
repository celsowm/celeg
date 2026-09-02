#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = (
    # Diagnostic only: reports numerical drift but still times the descriptor so
    # we can quantify how much of llama.cpp's advantage comes from relaxed MPP.
    ("relaxed precision diagnostic", ROOT / "scripts" / "metal_relaxed_precision.py"),
    ("modern Q4_K K32", ROOT / "scripts" / "metal_q4k_k32.py"),
    ("strict llama-style Q4_K K32", ROOT / "scripts" / "metal_q4k_k32_llama_strict.py"),
    ("vectorized Q4_K K32", ROOT / "scripts" / "metal_q4k_k32_vector.py"),
    ("vectorized production Q4_K K64", ROOT / "scripts" / "metal_q4k_k64_vector.py"),
    ("predecoded Q4_K F16 cache", ROOT / "scripts" / "metal_q4k_predecoded.py"),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run viable Metal prefill A/B candidates and diagnostics."
    )
    parser.add_argument(
        "--build-only", action="store_true",
        help="compile every Objective-C++ harness without executing GPU work",
    )
    args = parser.parse_args()

    failures: list[str] = []
    for label, script in EXPERIMENTS:
        print(f"\n=== {label} ===", flush=True)
        command = [sys.executable, str(script)]
        if args.build_only:
            command.append("--build-only")
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            failures.append(label)

    if failures:
        print("\nRejected/failed experiments: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("\nAll viable Metal prefill A/B candidates completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
