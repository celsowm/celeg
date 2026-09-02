#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENTS = (
    ("relaxed precision diagnostic", ROOT / "scripts" / "metal_relaxed_precision.py"),
    ("strict llama-style Q4_K K32", ROOT / "scripts" / "metal_q4k_k32_llama_strict.py"),
    ("Q4_K stage128 / strict K64x2", ROOT / "scripts" / "metal_q4k_stage128.py"),
    ("Q4_K static full tiles", ROOT / "scripts" / "metal_q4k_static_full.py"),
    ("Q4_K static + stage128", ROOT / "scripts" / "metal_q4k_static_stage128.py"),
    ("hardened vectorized Q4_K K64", ROOT / "scripts" / "metal_q4k_k64_vector.py"),
)


def main() -> int:
    failures: list[str] = []
    for label, script in EXPERIMENTS:
        print(f"\n=== {label} ===", flush=True)
        result = subprocess.run([sys.executable, str(script)], cwd=ROOT, check=False)
        if result.returncode != 0:
            failures.append(label)
    if failures:
        print("\nFailed/rejected: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("\nFocused M5 prefill diagnostics completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
