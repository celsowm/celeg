#!/usr/bin/env python3
"""Same-GGUF Celeg versus llama.cpp release gate.

The manifest is configuration only.  This harness records machine/build IDs,
the exact command lines, one discarded warmup, and five measured samples.  A
missing cached GGUF or binary is reported as ``not_run``; a completed run below
the manifest's median +5% threshold exits non-zero when ``--require-win`` is
used (the CI/release-gate mode).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def binary(build: Path, name: str) -> Path | None:
    suffix = ".exe" if os.name == "nt" else ""
    for base in (build, build / "bin", build / "Release", build / "bin" / "Release"):
        value = base / f"{name}{suffix}"
        if value.is_file():
            return value
    return None


def git_head(directory: Path) -> str:
    done = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                          text=True, capture_output=True, check=False)
    return done.stdout.strip() if done.returncode == 0 else "unknown"


def hub_cache() -> Path:
    configured = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    return Path(configured) if configured else Path.home() / ".cache" / "huggingface" / "hub"


def cached_gguf(manifest: dict[str, Any]) -> Path | None:
    owner, repo = manifest["repo"].split("/", 1)
    root = hub_cache() / f"models--{owner}--{repo}" / "snapshots"
    if not root.is_dir():
        return None
    candidates = sorted(root.glob(f"*/{manifest['gguf_filename']}"),
                        key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def gpu() -> dict[str, str]:
    done = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                           "--format=csv,noheader,nounits"],
                          text=True, capture_output=True, check=False)
    if done.returncode or not done.stdout.strip():
        return {"name": "unavailable"}
    name, driver, capability = (part.strip() for part in done.stdout.splitlines()[0].split(","))
    return {"name": name, "driver": driver, "compute_capability": capability}


def parse_celeg(output: str) -> tuple[float, float]:
    values = dict(re.findall(r"^(benchmark\.(?:prefill|decode)_tokens_per_second)=([0-9.]+)$",
                             output, re.MULTILINE))
    try:
        return float(values["benchmark.prefill_tokens_per_second"]), float(values["benchmark.decode_tokens_per_second"])
    except KeyError as error:
        raise RuntimeError("celeg-run did not emit both benchmark phases") from error


def parse_llama(output: str, prompt: int, decode: int, depth: int = 0) -> float:
    rows = re.findall(r'"n_prompt":\s*(\d+).*?"n_gen":\s*(\d+).*?"n_depth":\s*(\d+).*?"avg_ts":\s*([0-9.]+)',
                      output, re.S)
    values = {(int(p), int(n), int(d)): float(rate) for p, n, d, rate in rows}
    try:
        return values[(prompt, decode, depth)]
    except KeyError as error:
        raise RuntimeError("llama-bench JSON did not contain the requested context-qualified row") from error


def parse_cpu_bench(output: str, prompt: int, decode: int) -> tuple[float, float]:
    try:
        rows = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("celeg-bench did not emit JSON") from error
    values = {(int(row["n_prompt"]), int(row["n_gen"])): float(row["avg_ts"])
              for row in rows if isinstance(row, dict) and "n_prompt" in row}
    try:
        return values[(prompt, 0)], values[(0, decode)]
    except KeyError as error:
        raise RuntimeError("celeg-bench JSON did not contain requested prefill/decode rows") from error


def summary(samples: list[float]) -> dict[str, Any]:
    return {"median": statistics.median(samples), "mean": statistics.mean(samples),
            "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            "samples": samples}


def run(command: list[str], env: dict[str, str]) -> str:
    done = subprocess.run(command, text=True, capture_output=True, env=env, check=False)
    output = done.stdout + done.stderr
    if done.returncode:
        raise RuntimeError(f"command exited {done.returncode}:\n{output}")
    return output


def compare(manifest_path: Path, backend: str, celeg_build: Path, llama_build: Path,
            require_win: bool) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_path = ROOT / "benchmarks" / "results" / f"{manifest['name']}_{backend}_llama_gate.json"
    model = cached_gguf(manifest)
    celeg = binary(celeg_build, "celeg-bench" if backend == "cpu" else "celeg-run")
    llama = binary(llama_build, "llama-bench")
    report: dict[str, Any] = {
        "status": "not_run", "manifest": str(manifest_path), "backend": backend,
        "configuration": {
            "repo": manifest["repo"], "gguf_filename": manifest["gguf_filename"],
            "context": manifest["context"], "prefill_tokens": manifest["prefill_tokens"],
            "decode_tokens": manifest["decode_tokens"], "batch": manifest["batch"],
            "cpu_threads": manifest["cpu_threads"], "gpu_layers": 99 if backend == "cuda" else 0,
            "weight_mode": manifest.get("weight_mode", "auto"), "warmup_runs": 1,
            "measured_runs": 5,
        },
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "gpu": gpu()},
        "build_ids": {"celeg": git_head(ROOT), "llama_cpp": git_head(llama_build.parent)},
    }
    if not model or not celeg or not llama:
        report["reason"] = "missing cached GGUF or reference binary"
        report["missing"] = {"model": str(model) if model else None,
                             "celeg": str(celeg) if celeg else None,
                             "llama": str(llama) if llama else None}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 0

    p, n, batch = manifest["prefill_tokens"], manifest["decode_tokens"], manifest["batch"]
    threads = manifest["cpu_threads"]
    gpu_layers = "99" if backend == "cuda" else "0"
    if backend == "cpu":
        celeg_cmd = [str(celeg), "--model", str(model), "-p", str(p), "-n", str(n),
                     "-r", "1", "--warmup", "1", "-t", str(threads), "-b", str(batch),
                     "-ub", str(batch), "--context", str(manifest["context"]), "-o", "json"]
    else:
        celeg_cmd = [str(celeg), "--model", str(model), "--raw", "--prompt", "benchmark",
                     "--context", str(manifest["context"]), "--prefill-chunk", str(batch),
                     "--benchmark-prefill-tokens", str(p), "--benchmark-decode", str(n),
                     "--benchmark-warmup", "0", "--max-new-tokens", str(n),
                     "--weight-mode", manifest.get("weight_mode", "auto"),
                     # llama-bench measures deterministic decode.  Match it
                     # exactly rather than charging Celeg for stochastic
                     # top-k/top-p policy work absent from the reference.
                     "--temperature", "0", "--top-k", "1", "--top-p", "1",
                     "--repetition-penalty", "1"]
    # llama-bench no longer exposes an explicit context-size flag.  Its
    # default context covers every manifest; the equivalent live context is
    # fixed by -p for prefill and -d for decode below.
    llama_common = [str(llama), "-m", str(model),
                 "-b", str(batch), "-ub", str(batch), "-t", str(threads),
                 "-ngl", gpu_layers,
                 "-r", "1", "-o", "json"]
    llama_prefill_cmd = llama_common + ["-p", str(p), "-n", "0", "-d", "0"]
    # llama-bench's depth state is prefetched and restored before every
    # measured generation repetition. This makes its decode context exactly
    # the prompt length Celeg retains after benchmark-prefill.
    llama_decode_cmd = llama_common + ["-p", "0", "-n", str(n), "-d", str(p)]
    env = dict(os.environ)
    if backend == "cuda":
        cuda_root = Path(env.get("CUDA_PATH", ""))
        cuda_bins = [cuda_root / "bin", cuda_root / "bin" / "x64"] if cuda_root else []
        runtime_dirs = [str(path) for path in cuda_bins if path.is_dir()]
        runtime_dirs.append(str(llama.parent))
        env["PATH"] = os.pathsep.join(runtime_dirs + [env.get("PATH", "")])
    celeg_prefill: list[float] = []; celeg_decode: list[float] = []
    llama_prefill: list[float] = []; llama_decode: list[float] = []
    try:
        for iteration in range(6):
            celeg_output = run(celeg_cmd, env)
            cp, cd = (parse_cpu_bench(celeg_output, p, n) if backend == "cpu"
                      else parse_celeg(celeg_output))
            lp = parse_llama(run(llama_prefill_cmd, env), p, 0)
            ld = parse_llama(run(llama_decode_cmd, env), 0, n, p)
            if iteration:
                celeg_prefill.append(cp); celeg_decode.append(cd)
                llama_prefill.append(lp); llama_decode.append(ld)
    except RuntimeError as error:
        report.update(status="failed", reason=str(error), commands={"celeg": celeg_cmd, "llama_cpp_prefill": llama_prefill_cmd, "llama_cpp_decode": llama_decode_cmd})
    else:
        c_prefill, c_decode = summary(celeg_prefill), summary(celeg_decode)
        l_prefill, l_decode = summary(llama_prefill), summary(llama_decode)
        ratios = {"prefill": c_prefill["median"] / l_prefill["median"],
                  "decode": c_decode["median"] / l_decode["median"]}
        threshold = manifest.get("minimum_win_ratio", 1.05)
        report.update(status="passed" if min(ratios.values()) >= threshold else "failed",
                      commands={"celeg": celeg_cmd, "llama_cpp_prefill": llama_prefill_cmd, "llama_cpp_decode": llama_decode_cmd},
                      celeg={"prefill": c_prefill, "decode": c_decode},
                      llama_cpp={"prefill": l_prefill, "decode": l_decode},
                      median_ratio=ratios, minimum_win_ratio=threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if require_win and report["status"] != "passed" else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--celeg-build", type=Path)
    parser.add_argument("--llama-build", type=Path)
    parser.add_argument("--require-win", action="store_true")
    args = parser.parse_args()
    suffix = "cuda" if args.backend == "cuda" else "cpu"
    celeg_build = args.celeg_build or ROOT / "out" / f"windows-{suffix}-release"
    llama_build = args.llama_build or ROOT / ".externals" / "llama.cpp" / f"build-{suffix}"
    return compare(args.manifest, args.backend, celeg_build, llama_build, args.require_win)


if __name__ == "__main__":
    sys.exit(main())
