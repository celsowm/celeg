#!/usr/bin/env python3
"""Reference fixture runner for deterministic CELEG benchmark manifests.

Reads a benchmark manifest from benchmarks/manifests/, runs celeg-run, and
captures the generated token ids into benchmarks/results/<name>.json.

A fixture with expected_sequence is only valid when celeg-run exposes the
actual generated token ids. Decoded text or a stdout hash is not an acceptable
substitute because different token sequences can decode to the same text.

Usage:
    python benchmarks/run_manifest.py benchmarks/manifests/dense_bf16.json
    python benchmarks/run_manifest.py benchmarks/manifests/dense_int8.json --update-expected
    python benchmarks/run_manifest.py --all
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "benchmarks" / "manifests"
RESULTS_DIR = ROOT / "benchmarks" / "results"
TOKEN_IDS_PREFIX = "CELEG_GENERATED_TOKEN_IDS="


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
    relw = [p for p in candidates if "relwithdebinfo" in p.name.lower()]
    return relw[0] if relw else candidates[0]


def find_run_exe(build_dir: Path) -> Path:
    suffix = ".exe" if platform.system() == "Windows" else ""
    for layout in ("bin/Release", "bin", "Release", "."):
        p = build_dir / layout / f"celeg-run{suffix}"
        if p.is_file():
            return p
    raise ManifestError(f"celeg-run not found under {build_dir}")


def build_run_cmd(manifest: dict[str, Any], exe: Path) -> list[str]:
    repo = manifest["repo"]
    if manifest.get("quant_tag"):
        repo = f"{repo}:{manifest['quant_tag']}"
    return [
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
        "--runtime-tokens",
    ]


def capture_stdout(cmd: list[str]) -> str:
    print("+", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise ManifestError(
            f"celeg-run exited with code {proc.returncode}:\n{proc.stdout}")
    return proc.stdout


def parse_tokens(stdout: str) -> list[int]:
    """Extract the machine-readable generated token-id record from celeg-run."""
    matches = [
        line[len(TOKEN_IDS_PREFIX):].strip()
        for line in stdout.splitlines()
        if line.startswith(TOKEN_IDS_PREFIX)
    ]
    if not matches:
        return []
    if len(matches) != 1:
        raise ManifestError("celeg-run emitted multiple generated-token records")
    try:
        value = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise ManifestError("invalid generated-token JSON emitted by celeg-run") from exc
    if not isinstance(value, list) or any(type(token) is not int for token in value):
        raise ManifestError("generated-token record must be a JSON array of integers")
    return value


def run_manifest(path: Path, update_expected: bool = False) -> Path:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    name = manifest.get("name") or path.stem
    build_dir = default_build_dir()
    exe = find_run_exe(build_dir)
    stdout = capture_stdout(build_run_cmd(manifest, exe))
    tokens = parse_tokens(stdout)

    if not tokens and manifest.get("max_new_tokens", 0) > 0:
        raise ManifestError(
            f"{name}: celeg-run did not expose generated token ids; refusing to "
            "treat decoded stdout or its hash as semantic validation")

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
        if tokens != expected:
            mismatch = next(
                (i for i, pair in enumerate(zip(tokens, expected)) if pair[0] != pair[1]),
                min(len(tokens), len(expected)),
            )
            actual = tokens[mismatch] if mismatch < len(tokens) else None
            wanted = expected[mismatch] if mismatch < len(expected) else None
            raise ManifestError(
                f"{name}: token mismatch at index {mismatch}: "
                f"actual={actual}, expected={wanted}; "
                f"actual_len={len(tokens)}, expected_len={len(expected)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{name}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")

    if update_expected:
        manifest["expected_sequence"] = tokens
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
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
