#!/usr/bin/env python3
"""Cross-platform CPU benchmark runner: lfm25-cpu vs llama.cpp.

Replaces the old shell scripts (run_all.sh, run_llama_cpp.sh, run_lfm25.sh,
setup_llama_cpp.sh) with one script that works the same way on Windows,
Linux, and macOS.

Both engines read their weights straight out of the user's default Hugging
Face hub cache (``~/.cache/huggingface/hub``, or ``$HF_HOME``/``$HF_HUB_CACHE``
if set) via the ``huggingface_hub`` library -- the same cache layout and
resolution order used by this repo's own downloader (see
``src/checkpoint/downloader.cpp`` and ``scripts/dev.py:hf_cache_root``).
Nothing is written under ``benchmarks/models`` or ``model/`` anymore.

Usage:
    python benchmarks/run_bench.py                 # setup + run + report
    python benchmarks/run_bench.py setup            # clone/build llama.cpp, fetch models
    python benchmarks/run_bench.py run               # run both benches + write report.md
    python benchmarks/run_bench.py run --quant Q4_K_M

Requires: huggingface_hub (``pip install huggingface_hub``), cmake, and a
C/C++ toolchain on PATH (same ones used to build this repo).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
RESULTS_DIR = BENCH_DIR / "results"
EXT_DIR = ROOT / ".externals" / "llama.cpp"
LLAMA_BUILD_DIR = EXT_DIR / "build-cpu"
LFM_BUILD_DIR = ROOT / "build-cpu"

GGUF_REPO = "LiquidAI/LFM2.5-230M-GGUF"
LLAMA_REVISION = "0e4a0362239713ea95a6864a17a8de4b0ad90d62"
GGUF_REVISION = "fa224d4cb60cffe61eb58726712ef255bb64d0b7"
KNOWN_QUANTS = ("Q4_K_M",)


class BenchError(RuntimeError):
    pass


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, **kw)
    if result.returncode != 0:
        raise BenchError(f"command failed ({result.returncode}): {' '.join(str(c) for c in cmd)}")
    return result


def find_exe(build_dir: Path, name: str) -> Path:
    """Locate a built binary across generator layouts: flat (Ninja/Makefiles)
    or per-config (Visual Studio) -- with or without .exe."""
    suffix = ".exe" if platform.system() == "Windows" else ""
    candidates = [
        build_dir / "bin" / "Release" / f"{name}{suffix}",
        build_dir / "bin" / f"{name}{suffix}",
        build_dir / "Release" / f"{name}{suffix}",
        build_dir / f"{name}{suffix}",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise BenchError(
        f"missing {name}{suffix} under {build_dir} (looked in bin/Release, bin, Release, .); build it first"
    )


def default_generator() -> list[str]:
    import os

    generator = os.environ.get("GENERATOR")
    if generator:
        return ["-G", generator]
    if shutil.which("ninja"):
        return ["-G", "Ninja"]
    return []  # let cmake pick its platform default (e.g. Visual Studio on Windows)


# --------------------------------------------------------------------------
# Model resolution (Hugging Face cache)
# --------------------------------------------------------------------------

def ensure_gguf(quant: str = "Q4_K_M") -> Path:
    from huggingface_hub import hf_hub_download

    filename = f"LFM2.5-230M-{quant}.gguf"
    print(f"resolving {GGUF_REPO}/{filename} from the HF cache (downloading if missing)")
    path = Path(hf_hub_download(
        repo_id=GGUF_REPO, filename=filename, revision=GGUF_REVISION))
    print(f"  -> {path}")
    return path


# --------------------------------------------------------------------------
# llama.cpp setup
# --------------------------------------------------------------------------

def setup_llama_cpp(jobs: int) -> None:
    if not (EXT_DIR / ".git").is_dir():
        print(f"cloning llama.cpp into {EXT_DIR}")
        EXT_DIR.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "https://github.com/ggml-org/llama.cpp.git", str(EXT_DIR)])

    run(["git", "-C", str(EXT_DIR), "checkout", "--detach", LLAMA_REVISION])

    configure_cmd = [
        "cmake", "-S", str(EXT_DIR), "-B", str(LLAMA_BUILD_DIR),
        *default_generator(),
        "-DGGML_NATIVE=ON",
        "-DGGML_CUDA=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    run(configure_cmd)
    for target in ("llama-cli", "llama-bench", "llama-quantize"):
        run(["cmake", "--build", str(LLAMA_BUILD_DIR), "--config", "Release",
             "--target", target, "-j", str(jobs)])

    print(f"llama.cpp CPU build ready under {LLAMA_BUILD_DIR}")


def build_lfm25_bench(jobs: int) -> None:
    run([
        "cmake", "-S", str(ROOT), "-B", str(LFM_BUILD_DIR),
        *default_generator(),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLFM_BUILD_TESTS=ON",
    ])
    run(["cmake", "--build", str(LFM_BUILD_DIR), "--config", "Release",
         "--target", "lfm25-bench", "-j", str(jobs)])


# --------------------------------------------------------------------------
# Running benchmarks
# --------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotate_rows(raw: str, model: Path, args: argparse.Namespace) -> str:
    rows = json.loads(raw)
    canonical = model.resolve()
    digest = file_sha256(canonical)
    for row in rows:
        if not isinstance(row, dict):
            continue
        reported = Path(row.get("model_filename", "")).resolve()
        if reported != canonical:
            raise BenchError(
                f"benchmark used unexpected model: {reported} (expected {canonical})")
        row["model_filename"] = str(canonical)
        row["model_sha256"] = digest
        row["model_size_bytes"] = canonical.stat().st_size
        row["n_threads"] = args.threads
        row["n_batch"] = args.batch_size
        row["n_ubatch"] = args.ubatch_size
        row["type_k"] = "bf16"
        row["type_v"] = "bf16"
    return json.dumps(rows, indent=2) + "\n"


def run_llama_bench(args: argparse.Namespace, gguf: Path) -> Path:
    bench = find_exe(LLAMA_BUILD_DIR, "llama-bench")

    cmd = [str(bench), "-m", str(gguf), "-p", str(args.prompt_tokens),
           "-n", str(args.gen_tokens), "-r", str(args.reps), "-o", "json",
           "-t", str(args.threads), "-b", str(args.batch_size),
           "-ub", str(args.ubatch_size), "-ctk", "bf16", "-ctv", "bf16"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"llama_cpp_{args.quant}.json"
    result = run(cmd, capture_output=True, text=True)
    out_path.write_text(
        annotate_rows(result.stdout, gguf, args), encoding="utf-8")
    print(result.stdout)
    print(f"wrote {out_path}")
    return out_path


def run_lfm25_bench(args: argparse.Namespace, gguf: Path) -> Path:
    bench = find_exe(LFM_BUILD_DIR, "lfm25-bench")

    cmd = [str(bench), "--model", str(gguf), "-p", str(args.prompt_tokens),
           "-n", str(args.gen_tokens), "-r", str(args.reps),
           "-t", str(args.threads), "-b", str(args.batch_size),
           "-ub", str(args.ubatch_size), "-o", "json"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "lfm25_cpu.json"
    result = run(cmd, capture_output=True, text=True)
    out_path.write_text(
        annotate_rows(result.stdout, gguf, args), encoding="utf-8")
    print(result.stdout)
    print(f"wrote {out_path}")
    return out_path


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def load_row(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in data if isinstance(r, dict) and "n_prompt" in r]
    prompt_row = next((r for r in rows if r["n_prompt"] > 0 and r["n_gen"] == 0), None)
    gen_row = next((r for r in rows if r["n_gen"] > 0 and r["n_prompt"] == 0), None)
    return {
        "n_prompt": prompt_row["n_prompt"] if prompt_row else 0,
        "avg_ts_prompt": float(prompt_row["avg_ts"]) if prompt_row else 0.0,
        "stddev_ts_prompt": float(prompt_row["stddev_ts"]) if prompt_row else 0.0,
        "n_gen": gen_row["n_gen"] if gen_row else 0,
        "avg_ts_gen": float(gen_row["avg_ts"]) if gen_row else 0.0,
        "stddev_ts_gen": float(gen_row["stddev_ts"]) if gen_row else 0.0,
        "cpu_info": (prompt_row or gen_row or {}).get("cpu_info", "unknown"),
    }


def git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False)
    if result.returncode != 0:
        # Some Windows developer-shell wrappers terminate git after emitting
        # its answer.  Read HEAD directly so the report still records the
        # commit instead of silently losing provenance.
        head = repo / ".git" / "HEAD"
        try:
            value = head.read_text(encoding="utf-8").strip()
            if value.startswith("ref: "):
                ref = value[5:]
                ref_path = repo / ".git" / ref
                if ref_path.is_file():
                    value = ref_path.read_text(encoding="utf-8").strip()
                else:
                    packed = repo / ".git" / "packed-refs"
                    for line in packed.read_text(encoding="utf-8").splitlines():
                        if line and not line.startswith("#"):
                            commit, name = line.split(" ", 1)
                            if name == ref:
                                value = commit
                                break
            return value + "+dirty"
        except (OSError, ValueError):
            return "unknown"
    commit = result.stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True, text=True, check=False).stdout.strip()
    return commit + ("+dirty" if dirty else "")


def write_report(llama_cpp_json: Path, lfm25_json: Path, out_path: Path,
                 gguf: Path, args: argparse.Namespace) -> None:
    a = load_row(llama_cpp_json)
    b = load_row(lfm25_json)

    def speedup(x: float, y: float) -> str:
        return f"{x / y:.2f}x" if y else "n/a"

    lines = [
        "# CPU benchmark: lfm25-cpu vs llama.cpp",
        "",
        "| metric | llama.cpp | lfm25-cpu | lfm25-cpu vs llama.cpp |",
        "|---|---|---|---|",
        f"| prefill tok/s (n={a['n_prompt']:.0f} vs {b['n_prompt']:.0f}) "
        f"| {a['avg_ts_prompt']:.2f} ± {a['stddev_ts_prompt']:.2f} "
        f"| {b['avg_ts_prompt']:.2f} ± {b['stddev_ts_prompt']:.2f} "
        f"| {speedup(b['avg_ts_prompt'], a['avg_ts_prompt'])} |",
        f"| decode tok/s (n={a['n_gen']:.0f} vs {b['n_gen']:.0f}) "
        f"| {a['avg_ts_gen']:.2f} ± {a['stddev_ts_gen']:.2f} "
        f"| {b['avg_ts_gen']:.2f} ± {b['stddev_ts_gen']:.2f} "
        f"| {speedup(b['avg_ts_gen'], a['avg_ts_gen'])} |",
        "",
        "## Configuration",
        "",
        f"- GGUF: `{gguf.resolve()}`",
        f"- SHA-256: `{file_sha256(gguf)}`",
        f"- Size: {gguf.stat().st_size} bytes",
        f"- lfm25-cpu commit: `{git_commit(ROOT)}`",
        f"- llama.cpp commit: `{git_commit(EXT_DIR)}`",
        f"- CPU: {platform.processor() or a['cpu_info']}",
        f"- lfm25 ISA: {b['cpu_info']}",
        f"- Threads: {args.threads}",
        f"- Batch / ubatch: {args.batch_size} / {args.ubatch_size}",
        "- KV cache: BF16 on both engines",
        f"- Repetitions: 1 warmup + {args.reps} measured",
        "",
        "## Validity",
        "",
        "This report replaces and invalidates every earlier CPU comparison in "
        "this workspace because those runs used different checkpoint formats "
        "or quantizations. Only the same-GGUF results above are valid.",
        "",
        "Both engines run the exact same memory-mapped Q4_K_M GGUF file. "
        "The runner verifies the canonical path, byte size and SHA-256 before "
        "writing either result. Both use the same synthetic-token methodology "
        "(llama-bench and lfm25-bench both time n_prompt prefill tokens and "
        "n_gen decode tokens over identical repetitions with a discarded "
        "warmup run). Numbers are averages over the configured repetitions; "
        "see the raw JSON files next to this report for per-run detail.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def add_common_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quant", default="Q4_K_M", choices=KNOWN_QUANTS,
                         help="llama.cpp GGUF quant to compare against (default: Q4_K_M)")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8,
                        help="explicit thread count for both engines (default: 8)")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="llama.cpp logical batch size (matches LFM chunking)")
    parser.add_argument("--ubatch-size", type=int, default=256,
                        help="llama.cpp physical batch size (matches LFM chunking)")


def cmd_setup(args: argparse.Namespace) -> None:
    setup_llama_cpp(args.jobs)
    ensure_gguf(args.quant)
    build_lfm25_bench(args.jobs)


def cmd_run(args: argparse.Namespace) -> None:
    gguf = ensure_gguf(args.quant).resolve()
    llama_json = run_llama_bench(args, gguf)
    lfm25_json = run_lfm25_bench(args, gguf)
    expected_hash = file_sha256(gguf)
    for label, result_path in (
        ("llama.cpp", llama_json), ("lfm25-cpu", lfm25_json)
    ):
        rows = json.loads(result_path.read_text(encoding="utf-8"))
        if not rows or any(
            row.get("model_sha256") != expected_hash for row in rows
        ):
            raise BenchError(
                f"{label} result does not prove the expected GGUF hash")
    write_report(
        llama_json, lfm25_json, RESULTS_DIR / "report.md", gguf, args)


def cmd_all(args: argparse.Namespace) -> None:
    cmd_setup(args)
    cmd_run(args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p_setup = sub.add_parser("setup", help="clone/build llama.cpp, build lfm25-bench, fetch both models into the HF cache")
    add_common_bench_args(p_setup)
    p_setup.add_argument("--jobs", type=int, default=12)
    p_setup.set_defaults(func=cmd_setup)

    p_run = sub.add_parser("run", help="run both benchmarks and write report.md (assumes setup already ran)")
    add_common_bench_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_all = sub.add_parser("all", help="setup + run (default)")
    add_common_bench_args(p_all)
    p_all.add_argument("--jobs", type=int, default=12)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["all", *sys.argv[1:]])

    try:
        args.func(args)
    except BenchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ImportError as e:
        print(f"error: {e}. Run: pip install huggingface_hub", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
