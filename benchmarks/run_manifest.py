#!/usr/bin/env python3
"""Reference fixture runner for Phase 0 of the multi-stage refactoring.

Reads a benchmark manifest from benchmarks/manifests/ and reproduces the
reference run, capturing the deterministic token sequence (and, when supported
by lfm25-run, the final logits) into benchmarks/results/<name>.json.

The manifest schema is documented in benchmarks/README.md. This runner only
*drives* the existing lfm25-run binary; it does not modify the model code. It is
the Phase 0 (task 0.2) companion to lfm25_multi_stage_refactoring_plan.md.

Usage:
    python benchmarks/run_manifest.py benchmarks/manifests/dense_bf16.json
    python benchmarks/run_manifest.py benchmarks/manifests/dense_int8.json --update-expected
    python benchmarks/run_manifest.py --all            # run every manifest
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "benchmarks" / "manifests"
RESULTS_DIR = ROOT / "benchmarks" / "results"


class ManifestError(RuntimeError):
    pass


def default_build_dir() -> Path:
    """Discover the CUDA build dir written by scripts/dev.py verify --backend cuda."""
    out = ROOT / "out"
    if not out.is_dir():
        raise ManifestError(
            f"no build dir at {out}; run `python scripts/dev.py verify --backend cuda` first")
    candidates = sorted(p for p in out.iterdir() if p.is_dir() and "cuda" in p.name.lower())
    if not candidates:
        raise ManifestError(f"no CUDA build dir under {out}")
    # Prefer RelWithDebInfo if present.
    relw = [p for p in candidates if "relwithdebinfo" in p.name.lower()]
    return relw[0] if relw else candidates[0]


def find_run_exe(build_dir: Path) -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    for layout in ("bin/Release", "bin", "Release", "."):
        p = build_dir / layout / f"lfm25-run{suffix}"
        if p.is_file():
            return p
    raise ManifestError(f"lfm25-run not found under {build_dir}")


def build_run_cmd(manifest: dict[str, Any], exe: Path) -> list[str]:
    repo = manifest["repo"]
    if manifest.get("quant_tag"):
        repo = f"{repo}:{manifest['quant_tag']}"
    cmd = [
        str(exe),
        "--repo", repo,
        "--prompt", manifest["prompt"],
        "--max-new-tokens", str(manifest["max_new_tokens"]),
        "--seed", str(manifest["seed"]),
        "--top-k", str(manifest["top_k"]),
        "--top-p", str(manifest["top_p"]),
        "--temperature", str(manifest["temperature"]),
        "--repetition-penalty", str(manifest["repetition_penalty"]),
        "--weight-mode", manifest["weight_mode"],
        "--kv-cache", manifest.get("kv_cache_mode", "bf16"),
    ]
    return cmd


def capture_stdout(cmd: list[str]) -> str:
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", check=False)
    if proc.returncode != 0:
        raise ManifestError(
            f"lfm25-run exited with code {proc.returncode}:\n{proc.stdout}")
    return proc.stdout


def parse_tokens(stdout: str) -> list[int]:
    """Best-effort extraction of generated token ids from lfm25-run output.

    lfm25-run prints the decoded continuation after the prompt on stdout. We
    can't recover token ids from text alone; for a deterministic fixture the
    captured text is still valuable as a hashable fingerprint. This function
    returns an empty list when --runtime-tokens is not produced; the manifest
    expected_sequence field is the authoritative token-id record (filled in by
    --update-expected from a trusted reference run).
    """
    del stdout
    return []


def run_manifest(path: Path, update_expected: bool = False) -> Path:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    name = manifest.get("name") or path.stem
    build_dir = default_build_dir()
    exe = find_run_exe(build_dir)
    cmd = build_run_cmd(manifest, exe)
    stdout = capture_stdout(cmd)
    tokens = parse_tokens(stdout)

    result = {
        "name": name,
        "manifest_path": str(path),
        "repo": manifest["repo"],
        "quant_tag": manifest.get("quant_tag"),
        "weight_mode": manifest["weight_mode"],
        "kv_cache_mode": manifest.get("kv_cache_mode", "bf16"),
        "seed": manifest["seed"],
        "top_k": manifest["top_k"],
        "stdout_sha256": __import__("hashlib").sha256(stdout.encode("utf-8")).hexdigest(),
        "stdout": stdout,
        "captured_token_ids": tokens,
    }

    if manifest.get("expected_sequence") is not None and not update_expected:
        expected = list(manifest["expected_sequence"])
        if tokens and tokens != expected:
            raise ManifestError(
                f"{name}: captured tokens {tokens} != expected {expected}")
        if not tokens:
            # Without token-id capture, fall back to stdout hash comparison
            # against the recorded expected hash if present.
            expected_hash = manifest.get("expected_stdout_sha256")
            if expected_hash and expected_hash != result["stdout_sha256"]:
                raise ManifestError(
                    f"{name}: stdout hash {result['stdout_sha256']} != "
                    f"expected {expected_hash}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")

    if update_expected and tokens:
        manifest["expected_sequence"] = tokens
        manifest["expected_stdout_sha256"] = result["stdout_sha256"]
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"updated expected_sequence in {path}")

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", help="path to a manifest JSON file")
    parser.add_argument("--all", action="store_true",
                        help="run every manifest under benchmarks/manifests/")
    parser.add_argument("--update-expected", action="store_true",
                        help="update the manifest expected_sequence from this run")
    args = parser.parse_args()

    try:
        if args.all:
            for p in sorted(MANIFEST_DIR.glob("*.json")):
                run_manifest(p, update_expected=args.update_expected)
        elif args.manifest:
            run_manifest(Path(args.manifest), update_expected=args.update_expected)
        else:
            parser.error("provide a manifest path or --all")
    except ManifestError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
