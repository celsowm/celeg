#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


CLASSIFICATIONS = {
    "PASS_GENERATED",
    "FAIL_EMPTY_GENERATION",
    "FAIL_NONFINITE",
    "UNSUPPORTED_CAPABILITY",
    "UNSUPPORTED_FORMAT",
    "UNSUPPORTED_DTYPE",
    "LOAD_ERROR",
    "RUNTIME_ERROR",
    "PROCESS_ERROR",
    "NOT_CONFIGURED",
}


def classify(returncode: int, stdout: str, stderr: str) -> str:
    combined = (stdout + "\n" + stderr).lower()
    if "generation produced no output tokens" in combined:
        return "FAIL_EMPTY_GENERATION"
    if "non-finite" in combined or "nonfinite" in combined or "nan logits" in combined:
        return "FAIL_NONFINITE"
    if returncode == 0:
        return "PASS_GENERATED" if stdout.strip() else "FAIL_EMPTY_GENERATION"
    if (
        "does not support rope scaling" in combined
        or "requires theta 10000" in combined
        or "does not support factorized latent attention" in combined
        or "unsupported capability" in combined
    ):
        return "UNSUPPORTED_CAPABILITY"
    if "no registered checkpoint format matches" in combined:
        return "UNSUPPORTED_FORMAT"
    if "unsupported dtype" in combined:
        return "UNSUPPORTED_DTYPE"
    if "failed to load" in combined or "load error" in combined:
        return "LOAD_ERROR"
    if "error:" in combined:
        return "RUNTIME_ERROR"
    return "PROCESS_ERROR"


def resolve_source(raw: str):
    if raw.startswith("repo:"):
        return "--repo", raw[5:]
    if raw.startswith("model:"):
        return "--model", raw[6:]
    if Path(raw).exists():
        return "--model", raw
    return "--repo", raw


def run_case(runner: Path, case: dict, defaults: dict, timeout: int) -> dict:
    source_env = case["source_env"]
    raw_source = os.environ.get(source_env, "").strip()
    base = {
        "name": case["name"],
        "family": case.get("family", ""),
        "format": case.get("format", ""),
        "storage": case.get("storage", ""),
        "tag": case.get("tag", ""),
        "source_env": source_env,
    }
    if not raw_source:
        return {**base, "classification": "NOT_CONFIGURED", "returncode": None,
                "elapsed_seconds": 0.0, "stdout": "", "stderr": ""}

    source_flag, source_value = resolve_source(raw_source)
    prompt = case.get("prompt", defaults.get("prompt", "Hello"))
    context = int(case.get("context", defaults.get("context", 4096)))
    max_new_tokens = int(case.get("max_new_tokens", defaults.get("max_new_tokens", 8)))
    command = [
        str(runner), source_flag, source_value,
        "--prompt", prompt,
        "--context", str(context),
        "--max-new-tokens", str(max_new_tokens),
        "--require-output",
    ]
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=timeout, check=False)
        elapsed = time.monotonic() - start
        result = {
            **base,
            "classification": classify(completed.returncode, completed.stdout, completed.stderr),
            "returncode": completed.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as error:
        elapsed = time.monotonic() - start
        result = {
            **base,
            "classification": "PROCESS_ERROR",
            "returncode": None,
            "elapsed_seconds": round(elapsed, 3),
            "stdout": error.stdout or "",
            "stderr": (error.stderr or "") + f"\ntimeout after {timeout}s",
        }
    assert result["classification"] in CLASSIFICATIONS
    return result


def write_markdown(path: Path, results: list[dict]) -> None:
    lines = [
        "# Metal all-model sweep",
        "",
        "| Model | Family | Format/storage | Result | Seconds |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for result in results:
        storage = "/".join(filter(None, [result["format"], result["storage"]]))
        lines.append(
            f"| `{result['name']}` | {result['family']} | {storage} | "
            f"**{result['classification']}** | {result['elapsed_seconds']:.3f} |"
        )
    lines.extend(["", "## Diagnostics", ""])
    for result in results:
        if result["classification"] in {"PASS_GENERATED", "NOT_CONFIGURED"}:
            continue
        diagnostic = (result["stderr"].strip() or result["stdout"].strip()).replace("\n", " ")
        if len(diagnostic) > 300:
            diagnostic = diagnostic[:297] + "..."
        lines.append(f"- `{result['name']}`: `{diagnostic}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Apple Metal all-model smoke matrix")
    parser.add_argument("--runner", type=Path,
                        default=Path("out/macos-metal-relwithdebinfo/celeg-metal-run"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("scripts/metal_model_matrix.json"))
    parser.add_argument("--json-out", type=Path,
                        default=Path("metal-model-sweep.json"))
    parser.add_argument("--markdown-out", type=Path,
                        default=Path("metal-model-sweep.md"))
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    if not args.runner.exists():
        parser.error(f"runner does not exist: {args.runner}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    defaults = manifest.get("defaults", {})
    selected = set(args.only)
    cases = [case for case in manifest["models"]
             if not selected or case["name"] in selected]
    unknown = selected - {case["name"] for case in manifest["models"]}
    if unknown:
        parser.error("unknown --only case(s): " + ", ".join(sorted(unknown)))

    results = []
    for case in cases:
        result = run_case(args.runner, case, defaults, args.timeout)
        results.append(result)
        print(f"{result['name']}: {result['classification']}", flush=True)

    payload = {
        "schema_version": 1,
        "runner": str(args.runner),
        "results": results,
    }
    args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    write_markdown(args.markdown_out, results)

    configured = [r for r in results if r["classification"] != "NOT_CONFIGURED"]
    return 0 if configured and all(r["classification"] == "PASS_GENERATED" for r in configured) else 1


if __name__ == "__main__":
    sys.exit(main())
