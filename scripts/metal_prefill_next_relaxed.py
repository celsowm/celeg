#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)


def main() -> int:
    run("metal_rope_table.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
