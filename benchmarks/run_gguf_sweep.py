#!/usr/bin/env python3
"""Sweep every cached GGUF file: celeg vs llama.cpp, CPU and GPU.

Unlike run_bench.py / run_cuda_bench.py (each pinned to one specific GGUF
file), this discovers every ``*.gguf`` file already sitting in the user's
default Hugging Face hub cache (``~/.cache/huggingface/hub``, or
``$HF_HOME``/``$HF_HUB_CACHE`` if set -- same resolution order as
``src/checkpoint/downloader.cpp`` and ``scripts/dev_environment.py:hf_cache_root``)
and benchmarks each one against llama.cpp on both backends.

Nothing is downloaded: only files already present locally are benchmarked.
``*imatrix*.gguf`` files are skipped -- they are quantization calibration
data, not inference checkpoints.

CPU side compares llama-bench against celeg-bench, which executes GGUF blocks
with native dot kernels for every quantization the type registry marks
``cpu_native_dot`` and repacks the rest into groupwise Q4.

GPU side compares llama-bench (-ngl 99) against celeg-run's built-in
benchmark mode. celeg's CUDA weight mode is recorded per row and printed in
the report, because it decides what is actually being measured: ``auto``
requantizes every GGUF quantization to int8 on load, so under ``auto`` the
GPU numbers do not vary with the source quantization. Use
``--gpu-weight-mode native`` to measure the packed-block MMQ path instead.

Each engine is recorded independently. A file that only one engine can load
is a result -- celeg runs architectures llama.cpp has no graph for, and the
reverse happens too -- so a failure is attributed to the engine that failed
rather than voiding the row.

Usage:
    python benchmarks/run_gguf_sweep.py                  # setup + sweep + report
    python benchmarks/run_gguf_sweep.py setup             # build llama.cpp (cpu+cuda) only
    python benchmarks/run_gguf_sweep.py sweep              # discover + benchmark + write report
    python benchmarks/run_gguf_sweep.py sweep --no-gpu     # CPU-only sweep
    python benchmarks/run_gguf_sweep.py sweep --gpu-weight-mode native
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "benchmarks" / "results" / "sweep"
EXT_DIR = ROOT / ".externals" / "llama.cpp"
LLAMA_CPU_BUILD = EXT_DIR / "build-cpu"
LLAMA_CUDA_BUILD = EXT_DIR / "build-cuda"
CELEG_CPU_BUILD = ROOT / "out" / "linux-cpu-release"
CELEG_CUDA_BUILD = ROOT / "out" / "linux-cuda-release"

PROMPT_TOKENS = 512
GEN_TOKENS = 128
REPS = 5
BATCH = 256
CUDA_ARCH = "120"  # RTX 5090 (sm_120); override via GENERATOR-style env if needed


class SweepError(RuntimeError):
    """A benchmark command failed.

    ``output`` holds the tail of what the command printed, kept separate from
    the command line itself: the invocations here run to several hundred
    characters, so folding both into one message means any truncation eats
    the diagnosis and leaves the flags.
    """

    def __init__(self, message: str, output: str = "") -> None:
        super().__init__(message)
        self.output = output


BENCH_TIMEOUT_SECONDS = 900  # a single quant/backend/engine measurement should not run longer than this

# Maps a distinctive fragment of an engine's error output onto a short tag, so
# the report can say *why* a cell is empty in a few characters instead of
# quoting a truncated stack of flags.
FAILURE_SIGNATURES = (
    ("unsupported GGUF linear quantization", "unsupported-quant"),
    ("unsupported tensor type", "unsupported-quant"),
    ("no CUDA host dequantizer", "unsupported-quant"),
    ("unsupported CPU GGUF", "unsupported-quant"),
    ("CPU linear tensor must be", "unsupported-dtype"),
    ("unknown model architecture", "unsupported-arch"),
    ("failed to load model", "load-failed"),
    ("out of memory", "oom"),
    ("no registered checkpoint format", "unrecognized-file"),
)


def classify_failure(error: SweepError) -> str:
    haystack = (error.output or "") + " " + str(error)
    if "timed out" in str(error):
        return "timeout"
    for needle, tag in FAILURE_SIGNATURES:
        if needle in haystack:
            return tag
    return "failed"


def failure_detail(error: SweepError) -> str:
    """The most informative line the failing command printed."""
    lines = [line.strip() for line in (error.output or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if "error" in line.lower() or "exception" in line.lower():
            return line[:300]
    return (lines[-1][:300] if lines else str(error)[:300])


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd))
    try:
        result = subprocess.run(cmd, **kw)
    except subprocess.TimeoutExpired as error:
        raise SweepError(f"command timed out after {BENCH_TIMEOUT_SECONDS}s") from error
    if result.returncode != 0:
        output = ((result.stdout or "") + (result.stderr or "")) if kw.get("capture_output") else ""
        raise SweepError(f"command failed ({result.returncode})", output[-4000:])
    return result


def find_exe(build_dir: Path, name: str) -> Path | None:
    suffix = ".exe" if platform.system() == "Windows" else ""
    for candidate in (build_dir / "bin" / name, build_dir / name,
                      build_dir / "bin" / "Release" / f"{name}{suffix}",
                      build_dir / "Release" / f"{name}{suffix}"):
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------
# HF cache discovery
# --------------------------------------------------------------------------

def hub_cache() -> Path:
    configured = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if configured:
        return Path(configured).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def discover_gguf_files() -> list[dict[str, Any]]:
    """One row per cached, non-imatrix GGUF file, grouped by repo."""
    root = hub_cache()
    entries: list[dict[str, Any]] = []
    if not root.is_dir():
        return entries
    for repo_dir in sorted(root.glob("models--*")):
        parts = repo_dir.name.removeprefix("models--").split("--", 1)
        if len(parts) != 2:
            continue
        owner, name = parts
        repo_id = f"{owner}/{name}"
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        snapshot_dirs = sorted((p for p in snapshots.iterdir() if p.is_dir()),
                               key=lambda p: p.stat().st_mtime, reverse=True)
        if not snapshot_dirs:
            continue
        snapshot = snapshot_dirs[0]
        for gguf in sorted(snapshot.glob("*.gguf")):
            if "imatrix" in gguf.name.lower():
                continue
            if not gguf.resolve().is_file():
                continue
            entries.append({
                "repo": repo_id,
                "filename": gguf.name,
                # Keep the snapshot symlink path (ends in .gguf), not the
                # resolved blob target: celeg's checkpoint format dispatch
                # matches by file extension, and HF cache blobs are named by
                # content hash with no extension.
                "path": gguf,
                "size_bytes": gguf.resolve().stat().st_size,
                "revision": snapshot.name,
            })
    return entries


# --------------------------------------------------------------------------
# llama.cpp setup
# --------------------------------------------------------------------------

def default_generator() -> list[str]:
    generator = os.environ.get("GENERATOR")
    if generator:
        return ["-G", generator]
    if shutil.which("ninja"):
        return ["-G", "Ninja"]
    return []


# This machine has two CUDA installs: Ubuntu's old nvidia-cuda-toolkit
# package (owns /usr/include and a standalone, unmanaged /usr/bin/cudafe++
# stuck at 12.0) and NVIDIA's official cuda-toolkit-13-2 at this path. `nvcc`
# resolves its sibling tools (cudafe++, cicc, ptxas) relative to how it was
# *invoked*: found bare via PATH, it resolves through the /usr/bin/nvcc
# symlink and reports its own location as /usr/bin, picking up the stale
# cudafe++ there (invalid '--static-host-stub' option -> compile failure).
# Invoked by this full path directly, it correctly finds its own real
# sibling tools. Always configure CUDA builds with this explicit compiler.
NVCC_PATH = "/usr/local/cuda-13.2/bin/nvcc"


def build_llama_bench(build_dir: Path, cuda: bool, jobs: int) -> None:
    if not (EXT_DIR / ".git").is_dir():
        raise SweepError(f"{EXT_DIR} is not a llama.cpp checkout; clone it first "
                          f"(see benchmarks/run_bench.py setup_llama_cpp)")
    configure_cmd = [
        "cmake", "-S", str(EXT_DIR), "-B", str(build_dir),
        *default_generator(),
        "-DGGML_NATIVE=ON",
        f"-DGGML_CUDA={'ON' if cuda else 'OFF'}",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=ON",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if cuda:
        configure_cmd.append(f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}")
        if Path(NVCC_PATH).is_file():
            configure_cmd.append(f"-DCMAKE_CUDA_COMPILER={NVCC_PATH}")
    run(configure_cmd)
    run(["cmake", "--build", str(build_dir), "--config", "Release",
         "--target", "llama-bench", "-j", str(jobs)])


def cmd_setup(args: argparse.Namespace) -> None:
    build_llama_bench(LLAMA_CPU_BUILD, cuda=False, jobs=args.jobs)
    if not args.no_gpu:
        build_llama_bench(LLAMA_CUDA_BUILD, cuda=True, jobs=args.jobs)


# --------------------------------------------------------------------------
# Per-file benchmarking
# --------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_info() -> dict[str, str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return {"name": "unavailable"}
    name, driver, capability = (part.strip() for part in result.stdout.splitlines()[0].split(","))
    return {"name": name, "driver": driver, "compute_capability": capability}


def git_commit(repo: Path) -> str:
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def parse_bench_json(raw: str) -> dict[tuple[int, int], dict[str, float]]:
    rows = json.loads(raw)
    out: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        if isinstance(row, dict) and "n_prompt" in row:
            out[(int(row["n_prompt"]), int(row["n_gen"]))] = {
                "avg_ts": float(row["avg_ts"]), "stddev_ts": float(row.get("stddev_ts", 0.0))}
    return out


def run_llama_bench(binary: Path, gguf: Path, threads: int, ngl: int) -> dict[str, Any]:
    cmd = [str(binary), "-m", str(gguf), "-p", str(PROMPT_TOKENS), "-n", str(GEN_TOKENS),
           "-r", str(REPS), "-o", "json", "-t", str(threads), "-b", str(BATCH),
           "-ub", str(BATCH), "-ngl", str(ngl), "-ctk", "bf16", "-ctv", "bf16"]
    result = run(cmd, capture_output=True, text=True, timeout=BENCH_TIMEOUT_SECONDS)
    rows = parse_bench_json(result.stdout)
    prefill = rows.get((PROMPT_TOKENS, 0))
    decode = rows.get((0, GEN_TOKENS))
    if not prefill or not decode:
        raise SweepError("llama-bench did not report both prefill and decode rows")
    return {"prefill_ts": prefill["avg_ts"], "prefill_stddev": prefill["stddev_ts"],
            "decode_ts": decode["avg_ts"], "decode_stddev": decode["stddev_ts"]}


def run_celeg_bench_cpu(binary: Path, gguf: Path, threads: int) -> dict[str, Any]:
    cmd = [str(binary), "--model", str(gguf), "-p", str(PROMPT_TOKENS), "-n", str(GEN_TOKENS),
           "-r", str(REPS), "-t", str(threads), "-b", str(BATCH), "-ub", str(BATCH), "-o", "json"]
    result = run(cmd, capture_output=True, text=True, timeout=BENCH_TIMEOUT_SECONDS)
    rows = parse_bench_json(result.stdout)
    prefill = rows.get((PROMPT_TOKENS, 0))
    decode = rows.get((0, GEN_TOKENS))
    if not prefill or not decode:
        raise SweepError("celeg-bench did not report both prefill and decode rows")
    return {"prefill_ts": prefill["avg_ts"], "prefill_stddev": prefill["stddev_ts"],
            "decode_ts": decode["avg_ts"], "decode_stddev": decode["stddev_ts"]}


def run_celeg_run_gpu(binary: Path, gguf: Path, env: dict[str, str],
                      weight_mode: str) -> dict[str, Any]:
    cmd = [str(binary), "--model", str(gguf), "--raw", "--prompt", "benchmark",
           "--benchmark-prefill-tokens", str(PROMPT_TOKENS), "--benchmark-decode", str(GEN_TOKENS),
           "--benchmark-warmup", "1", "--max-new-tokens", str(GEN_TOKENS),
           "--weight-mode", weight_mode, "--prefill-chunk", str(BATCH),
           "--temperature", "0", "--top-k", "1", "--top-p", "1", "--repetition-penalty", "1"]
    result = run(cmd, capture_output=True, text=True, env=env, timeout=BENCH_TIMEOUT_SECONDS)
    output = result.stdout + result.stderr
    values: dict[str, float] = {}
    for line in output.splitlines():
        if line.startswith("benchmark.") and "=" in line:
            key, value = line.split("=", 1)
            try:
                values[key] = float(value)
            except ValueError:
                continue
    try:
        return {"prefill_ts": values["benchmark.prefill_tokens_per_second"],
                "decode_ts": values["benchmark.decode_tokens_per_second"]}
    except KeyError as error:
        raise SweepError("celeg-run did not emit both benchmark phases") from error


# --------------------------------------------------------------------------
# Quality gate (P4): cross-engine top-1 agreement, greedy decode
#
# Nothing else in this repo checks output *quality* per quantization --
# benchmark_bench and the sweep's speed rows only prove a file loads and
# runs at a normal rate. cpu_gguf_kernels_test checks IQ dequant against
# ggml fixtures at the block level, over whichever blocks happen to be
# checked in. A decode-time bug (like the two fixed in commit 4aa90e1: an
# 8x-too-small attention score on every QK-norm model, and int-truncation
# in the AVX2 Q8_0 dot) passes all of that and still emits corrupted text.
# This section adds the missing check: run both engines greedy (temperature
# 0, top-k 1, top-p 1, repetition-penalty 1 -- deterministic and argmax on
# both sides) from the same prompt and diff the generated text.
# --------------------------------------------------------------------------

QUALITY_PROMPT = "The capital of France is"
QUALITY_TOKENS = 12

# Greedy/argmax decode flags shared by every engine invocation below. Any
# nonzero temperature or repetition penalty would make the two engines'
# outputs diverge from sampler noise instead of a real decode bug, which is
# exactly the confound this check exists to rule out.
_GREEDY_FLAGS = ["--temperature", "0", "--top-k", "1", "--top-p", "1",
                 "--repetition-penalty", "1"]


def run_celeg_completion_cpu(binary: Path, gguf: Path, threads: int) -> str:
    cmd = [str(binary), "--model", str(gguf), "--raw", "--prompt", QUALITY_PROMPT,
           "--max-new-tokens", str(QUALITY_TOKENS), "--threads", str(threads), *_GREEDY_FLAGS]
    result = run(cmd, capture_output=True, text=True, timeout=BENCH_TIMEOUT_SECONDS)
    return result.stdout.strip()


def run_celeg_completion_gpu(binary: Path, gguf: Path, env: dict[str, str],
                             weight_mode: str) -> str:
    cmd = [str(binary), "--model", str(gguf), "--raw", "--prompt", QUALITY_PROMPT,
           "--max-new-tokens", str(QUALITY_TOKENS), "--weight-mode", weight_mode, *_GREEDY_FLAGS]
    result = run(cmd, capture_output=True, text=True, env=env, timeout=BENCH_TIMEOUT_SECONDS)
    return result.stdout.strip()


def run_llama_completion(binary: Path, gguf: Path, threads: int, ngl: int) -> str:
    # --ignore-eos: llama.cpp treats every vocab token that *looks* like a
    # turn-end marker (<|im_end|>, <|endoftext|>, ...) as end-of-generation,
    # not only the single tokenizer.ggml.eos_token_id celeg's --raw mode
    # checks. Without this flag llama.cpp stops after the first such token
    # while celeg keeps generating the full QUALITY_TOKENS count, which
    # makes every row disagree on length alone before the text is even
    # compared. Forcing both engines to run the same fixed token count is
    # what makes the diff measure decode correctness instead of stopping
    # policy.
    cmd = [str(binary), "-m", str(gguf), "-p", QUALITY_PROMPT, "-n", str(QUALITY_TOKENS),
           "-no-cnv", "--no-display-prompt", "--simple-io", "--no-warmup", "-t", str(threads),
           "-ngl", str(ngl), "--temp", "0", "--top-k", "1", "--top-p", "1",
           "--repeat-penalty", "1", "--ignore-eos"]
    result = run(cmd, capture_output=True, text=True, timeout=BENCH_TIMEOUT_SECONDS)
    # With --no-display-prompt the generated text is all that's printed, but
    # it can itself contain newlines (reasoning models emit "</think>" and
    # blank lines mid-completion) -- keep every line, only drop a trailing
    # "> EOF by user"-style REPL status line if -no-cnv still emits one.
    lines = result.stdout.splitlines()
    while lines and (not lines[-1].strip() or lines[-1].lstrip().startswith(">")):
        lines.pop()
    return "\n".join(lines).strip()


def text_agreement(a: str, b: str) -> float:
    """Similarity ratio in [0, 1] between two greedy completions.

    Not exact-match, and deliberately not calibrated toward 1.0 even for a
    fully correct quant: two independent fp32/bf16 kernel implementations
    of the same model routinely pick different top-1 tokens a few tokens
    into open-ended generation once the logit gap between the best and
    second-best candidate gets small (this model closes its <think> block
    and then has many similarly-likely ways to continue) -- that is normal
    floating-point noise, not a bug, and no amount of correctness fixes
    removes it. What a real decode bug does that noise does not is corrupt
    the *early*, high-confidence tokens too, which is what collapses the
    ratio over the whole completion rather than just the tail. See
    docs/QUANTIZATION_SUPPORT_MATRIX.md "Quality gate" for the measured
    run the thresholds below come from, including two cases this exact
    check caught on its first real run.
    """
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# Quant-class thresholds, derived from one real measured run (Nanbeige-3B,
# all classes; see docs/QUANTIZATION_SUPPORT_MATRIX.md "Quality gate" for
# the raw numbers) rather than invented. A cell below its class threshold
# is not automatically a celeg bug -- it is evidence that needs to be
# looked at, the same way a speed regression is evidence and not an
# automatic revert. Two of the five classes are set to flag their observed
# value rather than clear it: legacy-q4-q5 (Q4_0 measured 0.18 against
# CUDA+llama.cpp agreeing with each other) and iq (IQ3_XXS measured 0.03)
# both landed on a real, investigated finding on the very first run -- see
# the writeup -- so the threshold stays above the number that finding
# produced instead of being loosened to hide it.
QUALITY_THRESHOLDS: dict[str, float] = {
    "dense-f16-bf16": 0.50,
    "q8_0": 0.50,
    "k-quant": 0.45,
    "legacy-q4-q5": 0.25,
    "iq": 0.20,
}


def quant_class(filename: str) -> str:
    name = filename.upper()
    if re.search(r"IQ\d", name):
        return "iq"
    if "Q8_0" in name:
        return "q8_0"
    if re.search(r"Q[2-6]_K", name):
        return "k-quant"
    if re.search(r"Q[45]_[01]", name):
        return "legacy-q4-q5"
    if "F16" in name or "BF16" in name or "F32" in name:
        return "dense-f16-bf16"
    return "iq"  # unrecognised naming: treat as the strictest-to-fail-loud class


def quality_cell(celeg_text: str, llama_text: str, filename: str) -> dict[str, Any]:
    qclass = quant_class(filename)
    agreement = text_agreement(celeg_text, llama_text)
    threshold = QUALITY_THRESHOLDS[qclass]
    return {
        "status": "ok" if agreement >= threshold else "below-threshold",
        "quant_class": qclass, "threshold": threshold, "agreement": round(agreement, 4),
        "celeg_text": celeg_text, "reference_text": llama_text,
    }


def cuda_env() -> dict[str, str]:
    env = dict(os.environ)
    cuda_root = Path(env.get("CUDA_PATH", "/usr/local/cuda"))
    cuda_bins = [cuda_root / "bin"]
    runtime_dirs = [str(p) for p in cuda_bins if p.is_dir()]
    env["PATH"] = os.pathsep.join(runtime_dirs + [env.get("PATH", "")])
    return env


def measure(label: str, thunk) -> dict[str, Any]:
    """Runs one engine on one file, recording its own success or failure.

    Each engine is measured independently so that a file only one of them can
    load still yields half a row instead of nothing. The previous shape --
    both engines inside one try block -- silently discarded a working celeg
    measurement whenever llama.cpp could not load the architecture.
    """
    try:
        result = thunk()
        result["status"] = "ok"
        return result
    except (SweepError, json.JSONDecodeError, KeyError) as error:
        if not isinstance(error, SweepError):
            error = SweepError(f"{label} produced unreadable output: {error}")
        return {"status": "failed", "category": classify_failure(error),
                "reason": failure_detail(error), "message": str(error)}


def backend_cell(llama: dict[str, Any], celeg: dict[str, Any]) -> dict[str, Any]:
    cell: dict[str, Any] = {"llama_cpp": llama, "celeg": celeg}
    if llama["status"] == "ok" and celeg["status"] == "ok":
        cell["status"] = "ok"
        cell["prefill_speedup"] = celeg["prefill_ts"] / llama["prefill_ts"]
        cell["decode_speedup"] = celeg["decode_ts"] / llama["decode_ts"]
    elif llama["status"] == "ok" or celeg["status"] == "ok":
        cell["status"] = "partial"
    else:
        cell["status"] = "failed"
    return cell


# No external reference exists for architectures llama.cpp cannot load at
# all (Lizzy-7B: `general.architecture = "lizzy"` has no llama.cpp graph).
# celeg CPU vs celeg CUDA on the same file is the only signal left, so it
# gets its own threshold rather than reusing a quant-class one calibrated
# against an independent engine -- two celeg backends running the same
# weight math are expected to agree closer than two different engines are.
CROSS_BACKEND_THRESHOLD = 0.90


def quality_cpu(celeg_cpu_run: Path | None, llama_completion_cpu: Path | None,
                gguf: Path, threads: int, filename: str) -> dict[str, Any]:
    if not celeg_cpu_run:
        return {"status": "skipped", "reason": "celeg-cpu-run not built"}
    try:
        celeg_text = run_celeg_completion_cpu(celeg_cpu_run, gguf, threads)
    except SweepError as error:
        return {"status": "failed", "category": classify_failure(error),
                "reason": failure_detail(error)}
    if not llama_completion_cpu:
        return {"status": "no-reference", "celeg_text": celeg_text,
                "reason": "llama-completion (cpu) not built"}
    try:
        llama_text = run_llama_completion(llama_completion_cpu, gguf, threads, ngl=0)
    except SweepError as error:
        return {"status": "no-reference", "celeg_text": celeg_text,
                "reason": f"llama.cpp reference unavailable: {failure_detail(error)}"}
    return quality_cell(celeg_text, llama_text, filename)


def quality_gpu(celeg_run_gpu: Path | None, llama_completion_cuda: Path | None,
                gguf: Path, env: dict[str, str], threads: int, weight_mode: str,
                filename: str) -> dict[str, Any]:
    if not celeg_run_gpu:
        return {"status": "skipped", "reason": "celeg-run (cuda) not built"}
    try:
        celeg_text = run_celeg_completion_gpu(celeg_run_gpu, gguf, env, weight_mode)
    except SweepError as error:
        return {"status": "failed", "category": classify_failure(error),
                "reason": failure_detail(error)}
    if not llama_completion_cuda:
        return {"status": "no-reference", "celeg_text": celeg_text,
                "reason": "llama-completion (cuda) not built"}
    try:
        llama_text = run_llama_completion(llama_completion_cuda, gguf, threads, ngl=99)
    except SweepError as error:
        return {"status": "no-reference", "celeg_text": celeg_text,
                "reason": f"llama.cpp reference unavailable: {failure_detail(error)}"}
    return quality_cell(celeg_text, llama_text, filename)


def cross_backend_quality(cpu_quality: dict[str, Any], gpu_quality: dict[str, Any]) -> dict[str, Any] | None:
    """celeg CPU vs celeg CUDA, only when neither has an external reference.

    This is the Lizzy-7B case: both backends load and run the model, but
    llama.cpp cannot, so cross-engine agreement can't be computed. Backend
    disagreement on the same weights is still a strong bug signal.
    """
    if cpu_quality.get("status") != "no-reference" or gpu_quality.get("status") != "no-reference":
        return None
    cpu_text, gpu_text = cpu_quality.get("celeg_text", ""), gpu_quality.get("celeg_text", "")
    agreement = text_agreement(cpu_text, gpu_text)
    return {"status": "ok" if agreement >= CROSS_BACKEND_THRESHOLD else "below-threshold",
            "threshold": CROSS_BACKEND_THRESHOLD, "agreement": round(agreement, 4),
            "cpu_text": cpu_text, "gpu_text": gpu_text}


def bench_one(entry: dict[str, Any], threads: int, llama_cpu: Path | None,
              llama_cuda: Path | None, celeg_cpu: Path | None, celeg_cuda: Path | None,
              want_gpu: bool, gpu_weight_mode: str, celeg_cpu_run: Path | None,
              llama_completion_cpu: Path | None, llama_completion_cuda: Path | None) -> dict[str, Any]:
    gguf = entry["path"]
    filename = entry["filename"]
    row: dict[str, Any] = {
        "repo": entry["repo"], "filename": filename,
        "size_bytes": entry["size_bytes"], "revision": entry["revision"],
        "cpu": {"status": "not_run"}, "gpu": {"status": "not_run"},
    }

    if llama_cpu and celeg_cpu:
        row["cpu"] = backend_cell(
            measure("llama-bench", lambda: run_llama_bench(llama_cpu, gguf, threads, ngl=0)),
            measure("celeg-bench", lambda: run_celeg_bench_cpu(celeg_cpu, gguf, threads)))
    else:
        row["cpu"] = {"status": "skipped", "category": "not-built",
                      "reason": "llama-bench (cpu) or celeg-bench not built"}
    row["cpu"]["quality"] = quality_cpu(celeg_cpu_run, llama_completion_cpu, gguf, threads, filename)

    if want_gpu:
        if llama_cuda and celeg_cuda:
            env = cuda_env()
            row["gpu"] = backend_cell(
                measure("llama-bench", lambda: run_llama_bench(llama_cuda, gguf, threads, ngl=99)),
                measure("celeg-run", lambda: run_celeg_run_gpu(celeg_cuda, gguf, env,
                                                               gpu_weight_mode)))
            row["gpu"]["celeg_weight_mode"] = gpu_weight_mode
        else:
            env = cuda_env()
            row["gpu"] = {"status": "skipped", "category": "not-built",
                          "reason": "llama-bench (cuda) or celeg-run not built"}
        row["gpu"]["quality"] = quality_gpu(celeg_cuda, llama_completion_cuda, gguf, env, threads,
                                            gpu_weight_mode, filename)
        cross = cross_backend_quality(row["cpu"]["quality"], row["gpu"]["quality"])
        if cross is not None:
            row["quality_cross_backend"] = cross

    return row


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def cpu_model_name() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def fmt_ts(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "n/a"


def fmt_speedup(value: float | None) -> str:
    return f"{value:.2f}x" if value is not None else "n/a"


def engine_cells(engine: dict[str, Any] | None) -> tuple[str, str]:
    """Prefill and decode columns for one engine, or its failure tag."""
    if not engine:
        return "--", "--"
    if engine.get("status") == "ok":
        return fmt_ts(engine["prefill_ts"]), fmt_ts(engine["decode_ts"])
    tag = engine.get("category", engine.get("status", "failed"))
    return tag, tag


def backend_table(rows: list[dict[str, Any]], backend: str) -> list[str]:
    lines = ["| quant file | size (GB) | llama.cpp prefill | celeg prefill | prefill | "
             "llama.cpp decode | celeg decode | decode | status |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        cell = row[backend]
        size_gb = row["size_bytes"] / (1024 ** 3)
        if cell["status"] in ("skipped", "not_run"):
            note = cell.get("category", cell["status"])
            lines.append(f"| `{row['filename']}` | {size_gb:.2f} | -- | -- | -- | -- | -- | -- "
                         f"| {note} |")
            continue
        # A row where only one engine loaded still carries that engine's
        # numbers; only the ratio columns go blank.
        llama_prefill, llama_decode = engine_cells(cell.get("llama_cpp"))
        celeg_prefill, celeg_decode = engine_cells(cell.get("celeg"))
        prefill_ratio = fmt_speedup(cell.get("prefill_speedup")) if "prefill_speedup" in cell else "--"
        decode_ratio = fmt_speedup(cell.get("decode_speedup")) if "decode_speedup" in cell else "--"
        status = cell["status"]
        if status != "ok":
            failed = [name for name, key in (("llama.cpp", "llama_cpp"), ("celeg", "celeg"))
                      if cell.get(key, {}).get("status") != "ok"]
            status = f"{status} ({', '.join(failed)})"
        lines.append(f"| `{row['filename']}` | {size_gb:.2f} | {llama_prefill} | {celeg_prefill} "
                     f"| {prefill_ratio} | {llama_decode} | {celeg_decode} "
                     f"| {decode_ratio} | {status} |")
    return lines


def failure_summary(all_rows: list[dict[str, Any]]) -> list[str]:
    """One line per (engine, failure category), with the files affected."""
    buckets: dict[tuple[str, str, str], list[str]] = {}
    for row in all_rows:
        for backend in ("cpu", "gpu"):
            cell = row.get(backend, {})
            for engine, key in (("llama.cpp", "llama_cpp"), ("celeg", "celeg")):
                result = cell.get(key)
                if isinstance(result, dict) and result.get("status") == "failed":
                    bucket = (backend, engine, result.get("category", "failed"))
                    buckets.setdefault(bucket, []).append(row["filename"])
    if not buckets:
        return ["Every cached file loaded in both engines on every backend."]
    lines = ["| backend | engine | reason | files |", "|---|---|---|---|"]
    for (backend, engine, category) in sorted(buckets):
        names = buckets[(backend, engine, category)]
        shown = ", ".join(f"`{n}`" for n in names[:4])
        if len(names) > 4:
            shown += f", +{len(names) - 4} more"
        lines.append(f"| {backend.upper()} | {engine} | {category} | {shown} |")
    return lines


def quality_summary(all_rows: list[dict[str, Any]]) -> list[str]:
    """Per-file, per-backend cross-engine agreement against the class threshold."""
    lines = ["| file | backend | quant class | agreement | threshold | status |",
             "|---|---|---|---|---|---|"]
    below: list[str] = []
    for row in all_rows:
        for backend in ("cpu", "gpu"):
            quality = row.get(backend, {}).get("quality")
            if not isinstance(quality, dict) or "agreement" not in quality:
                continue
            status = quality["status"]
            lines.append(f"| `{row['filename']}` | {backend.upper()} | {quality['quant_class']} "
                         f"| {quality['agreement']:.4f} | {quality['threshold']:.2f} | {status} |")
            if status == "below-threshold":
                below.append(f"{row['filename']} ({backend.upper()})")
        cross = row.get("quality_cross_backend")
        if isinstance(cross, dict):
            lines.append(f"| `{row['filename']}` | CPU-vs-CUDA | -- "
                         f"| {cross['agreement']:.4f} | {cross['threshold']:.2f} | {cross['status']} |")
            if cross["status"] == "below-threshold":
                below.append(f"{row['filename']} (CPU-vs-CUDA)")
    if len(lines) == 2:
        lines.append("| -- | -- | -- | -- | -- | no comparable rows |")
    header = ["**Below-threshold cells (investigate before trusting these quants' output): "
              + (", ".join(below) if below else "none") + "**", ""]
    return header + lines


def write_report(all_rows: list[dict[str, Any]], out_path: Path, want_gpu: bool,
                 threads: int, gpu_weight_mode: str) -> None:
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_repo.setdefault(row["repo"], []).append(row)

    lines = [
        "# GGUF benchmark sweep: celeg vs llama.cpp",
        "",
        "Every `.gguf` file present in the local Hugging Face hub cache "
        "(`*imatrix*` calibration files excluded) was benchmarked against "
        "upstream llama.cpp, on CPU and, where available, GPU. Tokens/sec "
        "are the `avg_ts` llama-bench/celeg-bench report over "
        f"{REPS} timed repetitions plus one discarded warmup, "
        f"{PROMPT_TOKENS} synthetic prefill tokens and {GEN_TOKENS} decode "
        "steps, batch/ubatch size "
        f"{BATCH}, BF16 KV cache on both engines. This is an exploratory "
        "survey across every cached quant, not the strict same-file release "
        "gate in `benchmarks/compare_llama.py`. Each engine is recorded "
        "independently: a file only one of them can load still contributes "
        "that engine's numbers, with the ratio columns left blank.",
        "",
        "## This machine",
        "",
        "```",
        f"CPU:    {cpu_model_name()}",
        f"Threads used: {threads}",
        f"GPU:    {gpu_info().get('name', 'unavailable')}",
        f"OS:     {platform.platform()}",
        "```",
        "",
        "## Methodology notes",
        "",
        "- On CPU, celeg executes GGUF blocks in place with native dot "
        "kernels for every quantization the type registry marks "
        "`cpu_native_dot`; anything else is dequantized on load and repacked "
        "into groupwise Q4. See `docs/QUANTIZATION_SUPPORT_MATRIX.md`.",
        f"- On GPU, celeg ran with `--weight-mode {gpu_weight_mode}`. "
        "**`auto` requantizes every GGUF quantization to int8 on load**, so "
        "under `auto` the GPU columns measure an int8 weight path and are "
        "largely insensitive to the source quantization -- only `native` "
        "keeps Q4_K/Q6_K blocks packed on the device.",
        "- llama.cpp runs `-ngl 99` on GPU rows (every layer offloaded) "
        "and `-ngl 0` on CPU rows.",
        f"- celeg commit: `{git_commit(ROOT)}`",
        f"- llama.cpp commit: `{git_commit(EXT_DIR)}`",
        "",
        "## Load failures",
        "",
    ]
    lines.extend(failure_summary(all_rows))
    lines.append("")
    lines.append("## Quality gate")
    lines.append("")
    lines.extend([
        f"Cross-engine agreement, not just liveness: both engines decode greedily "
        f"(temperature 0, top-k 1, top-p 1, repetition-penalty 1 -- deterministic "
        f"argmax on both sides) from the same {QUALITY_TOKENS}-token prompt, and the "
        "generated text is diffed (`difflib.SequenceMatcher` ratio). A speed row can "
        "be `ok` -- the file loads and benchmarks at a normal rate -- while this gate "
        "still catches a decode-time bug that corrupts every token past the first few; "
        "that is exactly how the two bugs fixed in commit `4aa90e1` (an 8x-too-small "
        "attention score on every QK-norm model, and int-truncation in the AVX2 Q8_0 "
        "dot) were found. Thresholds are per quant class -- low-bit IQ formats have a "
        "real accuracy cost from the codebook approximation itself, not a celeg bug, "
        "so their floor is lower. Lizzy-7B has no llama.cpp reference "
        "(`general.architecture = \"lizzy\"` has no llama.cpp graph) and is instead "
        "gated celeg-CPU-vs-celeg-CUDA.",
        "",
    ])
    lines.extend(quality_summary(all_rows))
    lines.append("")

    for repo in sorted(by_repo):
        rows = sorted(by_repo[repo], key=lambda r: r["filename"])
        lines.append(f"## {repo}")
        lines.append("")
        lines.append("### CPU")
        lines.append("")
        lines.extend(backend_table(rows, "cpu"))
        lines.append("")
        if want_gpu:
            lines.append("### GPU")
            lines.append("")
            lines.extend(backend_table(rows, "gpu"))
            lines.append("")

    lines.append("## Raw data")
    lines.append("")
    lines.append(f"Per-run JSON lives under `benchmarks/results/sweep/` (gitignored); "
                 f"the full machine-readable summary for this run is "
                 f"`benchmarks/results/sweep/summary.json`.")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_sweep(args: argparse.Namespace) -> None:
    entries = discover_gguf_files()
    if not entries:
        raise SweepError(f"no cached .gguf files found under {hub_cache()}")

    llama_cpu = find_exe(LLAMA_CPU_BUILD, "llama-bench")
    celeg_cpu = find_exe(CELEG_CPU_BUILD, "celeg-bench")
    celeg_cpu_run = find_exe(CELEG_CPU_BUILD, "celeg-cpu-run")
    llama_completion_cpu = find_exe(LLAMA_CPU_BUILD, "llama-completion")
    want_gpu = not args.no_gpu
    llama_cuda = find_exe(LLAMA_CUDA_BUILD, "llama-bench") if want_gpu else None
    celeg_cuda = find_exe(CELEG_CUDA_BUILD, "celeg-run") if want_gpu else None
    llama_completion_cuda = find_exe(LLAMA_CUDA_BUILD, "llama-completion") if want_gpu else None
    if want_gpu and not (llama_cuda and celeg_cuda):
        print("warning: CUDA llama-bench or celeg-run not found; GPU rows will be skipped "
              "(run `setup` first, or pass --no-gpu)", file=sys.stderr)
    if not (celeg_cpu_run and llama_completion_cpu):
        print("warning: celeg-cpu-run or llama-completion (cpu) not found; the quality gate "
              "will report no-reference/skipped instead of an agreement score", file=sys.stderr)

    threads = args.threads or (os.cpu_count() or 8)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, 1):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{entry['repo']}_{entry['filename']}")
        result_path = RESULTS_DIR / f"{safe_name}.json"
        if not args.no_resume and result_path.is_file():
            print(f"\n=== [{index}/{len(entries)}] {entry['repo']} :: {entry['filename']} "
                  f"-- reusing existing result ===")
            all_rows.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        print(f"\n=== [{index}/{len(entries)}] {entry['repo']} :: {entry['filename']} "
              f"({entry['size_bytes'] / (1024**3):.2f} GB) ===")
        row = bench_one(entry, threads, llama_cpu, llama_cuda, celeg_cpu, celeg_cuda,
                        want_gpu, args.gpu_weight_mode, celeg_cpu_run,
                        llama_completion_cpu, llama_completion_cuda)
        all_rows.append(row)
        result_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {summary_path}")

    write_report(all_rows, ROOT / "docs" / "GGUF_BENCHMARK_REPORT.md", want_gpu, threads,
                 args.gpu_weight_mode)


def cmd_all(args: argparse.Namespace) -> None:
    cmd_setup(args)
    cmd_sweep(args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p_setup = sub.add_parser("setup", help="build llama.cpp llama-bench (cpu, and cuda unless --no-gpu)")
    p_setup.add_argument("--jobs", type=int, default=12)
    p_setup.add_argument("--no-gpu", action="store_true")
    p_setup.set_defaults(func=cmd_setup)

    p_sweep = sub.add_parser("sweep", help="discover cached GGUFs, benchmark each, write the report")
    p_sweep.add_argument("--threads", type=int, default=0, help="0 = os.cpu_count()")
    p_sweep.add_argument("--no-gpu", action="store_true")
    p_sweep.add_argument("--no-resume", action="store_true",
                         help="re-benchmark files even if a result JSON already exists")
    p_sweep.add_argument("--gpu-weight-mode", choices=["auto", "native", "int8", "int4", "bf16"],
                         default="auto",
                         help="celeg CUDA weight mode; `auto` requantizes GGUF to int8, "
                              "so use `native` to measure the packed-block MMQ path")
    p_sweep.set_defaults(func=cmd_sweep)

    p_all = sub.add_parser("all", help="setup + sweep (default)")
    p_all.add_argument("--jobs", type=int, default=12)
    p_all.add_argument("--threads", type=int, default=0)
    p_all.add_argument("--no-gpu", action="store_true")
    p_all.add_argument("--no-resume", action="store_true")
    p_all.add_argument("--gpu-weight-mode", choices=["auto", "native", "int8", "int4", "bf16"],
                         default="auto",
                         help="celeg CUDA weight mode; `auto` requantizes GGUF to int8, "
                              "so use `native` to measure the packed-block MMQ path")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["all", *sys.argv[1:]])

    try:
        args.func(args)
    except SweepError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
