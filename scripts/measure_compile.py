#!/usr/bin/env python3
"""Compile-time baseline capture for Phase 0 of the multi-stage refactoring.

Captures (per docs/ARCHITECTURE_RULES.md section 0.4):
  - clean build time of the CUDA RelWithDebInfo build (configures + builds),
  - incremental rebuild after touching one attention kernel source,
  - incremental rebuild after touching one model orchestration source,
  - final binary size of celeg-run.

Writes benchmarks/compile_baseline.json. The harness reuses scripts/dev.py for
the actual build process so the discovered toolchain (VS, CUDA, arch, runtime
DLLs) matches what `dev.py verify` uses. dev.py does not expose a `--target`
filter, so the two incremental measurements rebuild the whole project; this is
acceptable for Phase 0 because the absolute time still records the cost of a
single-file change against a warm Ninja graph.

Usage:
    python scripts/measure_compile.py
    python scripts/measure_compile.py --skip-clean   # skip the long clean build
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "benchmarks" / "compile_baseline.json"
DEFAULT_BUILD_DIR = ROOT / "out" / "windows-cuda-relwithdebinfo"


class MeasureError(RuntimeError):
    pass


def discover_build_dir() -> Path:
    if DEFAULT_BUILD_DIR.is_dir():
        return DEFAULT_BUILD_DIR
    out = ROOT / "out"
    if not out.is_dir():
        raise MeasureError("no build dir; run `python scripts/dev.py verify --backend cuda` first")
    candidates = sorted(p for p in out.iterdir() if p.is_dir() and "cuda" in p.name.lower())
    if not candidates:
        raise MeasureError(f"no CUDA build dir under {out}")
    return candidates[0]


def run_dev(args: list[str]) -> tuple[float, str]:
    """Invoke scripts/dev.py with the given extra args; return (elapsed, stdout)."""
    cmd = [sys.executable, str(ROOT / "scripts" / "dev.py"), *args]
    print("+", " ".join(cmd))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", check=False)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise MeasureError(f"dev.py {' '.join(args)} failed:\n{proc.stdout}")
    return elapsed, proc.stdout


def find_run_exe(build_dir: Path) -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    for layout in ("bin/Release", "bin", "Release", "."):
        p = build_dir / layout / f"celeg-run{suffix}"
        if p.is_file():
            return p
    raise MeasureError(f"celeg-run not found under {build_dir}")


def touch(path: Path) -> None:
    t = time.time()
    os.utime(path, (t, t))


def first_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p.is_file():
            return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=None,
                        help=f"build dir (default: {DEFAULT_BUILD_DIR})")
    parser.add_argument("--skip-clean", action="store_true",
                        help="skip the long clean configure+build; only measure incremental")
    args = parser.parse_args()

    try:
        build_dir = args.build_dir or discover_build_dir()
        print(f"using build dir: {build_dir}")

        if not args.skip_clean:
            # `dev.py verify` always reconfigures fresh (configure_command uses
            # cmake --fresh) and runs the full build + tests + smoke. We only
            # want build time, so we use `dev.py build --backend cuda`, which
            # also reconfigures fresh when the dir already exists (Ninja regen
            # would be cheap, but our dev.py wraps it: configure() is invoked
            # unconditionally). For an honest "clean build", delete the dir.
            if build_dir.exists():
                print(f"removing existing build dir for clean measurement")
                shutil.rmtree(build_dir)
            clean_time, _ = run_dev(["build", "--backend", "cuda"])
        else:
            clean_time = -1.0

        exe = find_run_exe(build_dir)
        binary_size = exe.stat().st_size

        # Touch an attention kernel source and measure incremental rebuild.
        attention_src = first_existing(
            ROOT / "src" / "backend" / "cuda" / "kernels" / "attention.cu",
            *(p for p in (ROOT / "src" / "backend" / "cuda" / "kernels").glob("*.cu")),
        )
        if attention_src is not None:
            touch(attention_src)
            # `dev.py build` reconfigures (cheap when CMakeLists + cache match)
            # then builds only changed targets via Ninja.
            attn_incremental, _ = run_dev(["build", "--backend", "cuda"])
        else:
            attn_incremental = -1.0

        # Touch a model orchestration source and measure incremental rebuild.
        model_src = first_existing(
            ROOT / "src" / "backend" / "cuda" / "model" / "execution.cu",
            *(p for p in (ROOT / "src" / "backend" / "cuda" / "model").glob("*.cu")),
        )
        if model_src is not None:
            touch(model_src)
            model_incremental, _ = run_dev(["build", "--backend", "cuda"])
        else:
            model_incremental = -1.0

        result: dict[str, Any] = {
            "build_dir": str(build_dir),
            "clean_build_seconds": round(clean_time, 3),
            "incremental_after_attention_kernel_seconds": round(attn_incremental, 3),
            "attention_kernel_touched": str(attention_src) if attention_src else None,
            "incremental_after_model_orchestration_seconds": round(model_incremental, 3),
            "model_orchestration_touched": str(model_src) if model_src else None,
            "celeg_run_binary_bytes": binary_size,
            "platform": platform.platform(),
        }
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(result, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"wrote {BASELINE_PATH}")
        print(json.dumps(result, indent=2))
        return 0
    except MeasureError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
