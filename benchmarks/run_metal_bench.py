#!/usr/bin/env python3
"""Run the reproducible Metal benchmark described by a local manifest."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def cache_root() -> Path:
    value = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if value:
        root = Path(value).expanduser()
        return root if os.environ.get("HF_HUB_CACHE") else root / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def cached_model(manifest: dict[str, object]) -> Path:
    owner, repo = str(manifest["repo"]).split("/", 1)
    snapshots = cache_root() / f"models--{owner}--{repo}" / "snapshots"
    filename = str(manifest["gguf_filename"])
    candidates = sorted(
        (path / filename for path in snapshots.iterdir() if (path / filename).is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if snapshots.is_dir() else []
    if not candidates:
        raise RuntimeError(f"cached checkpoint not found: {manifest['repo']}:{filename}")
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--numerical-policy", choices=["strict", "fast"], default="strict")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    build = (args.build_dir or ROOT / "out" / "darwin-metal-relwithdebinfo").resolve()
    binary = build / "celeg-metal-bench"
    model = cached_model(manifest)
    command = [
        str(binary), "--model", str(model),
        "--context", str(manifest["context"]),
        "--prompt-tokens", str(manifest["prompt_tokens"]),
        "--decode-tokens", str(manifest["decode_tokens"]),
        "--warmup", str(manifest["warmup"]),
        "--repetitions", str(manifest["repetitions"]),
        "--numerical-policy", args.numerical_policy,
    ]
    if not binary.is_file():
        raise RuntimeError(f"Metal benchmark binary not found: {binary}")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(output.strip())
    report = json.loads(result.stdout)
    report["manifest"] = str(args.manifest.resolve())
    report["repo"] = manifest["repo"]
    report["checkpoint"] = str(model)
    report["platform"] = platform.platform()
    report["processor"] = platform.processor()
    report["celeg_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False).stdout.strip()
    destination = args.output or ROOT / "benchmarks" / "results" / f"{manifest['name']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
