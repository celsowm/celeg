# CPU benchmarks: lfm25-cpu vs llama.cpp

Automated, reproducible comparison between this repo's native CPU engine and
upstream [llama.cpp](https://github.com/ggml-org/llama.cpp), both running the
same LFM2.5-230M weights.

Everything is driven by one cross-platform script, `run_bench.py` (Windows,
Linux, macOS -- no shell-specific scripts to keep in sync). llama.cpp is
cloned into `.externals/llama.cpp` (gitignored, never committed) and built
CPU-only in Release. Both engines read their weights straight out of the
user's **default Hugging Face hub cache** (`~/.cache/huggingface/hub`, or
`$HF_HOME`/`$HF_HUB_CACHE` if set) via `huggingface_hub` -- the same cache
layout and env-var resolution order used by this repo's own downloader
(`src/checkpoint/downloader.cpp`, `scripts/dev.py:hf_cache_root`). Nothing is
written under `benchmarks/models` or `model/`; run `huggingface-cli scan-cache`
to see what's there.

Both sides are timed with a purpose-built, in-process benchmark binary
(`llama-bench` for llama.cpp, this repo's own `lfm25-bench` for lfm25-cpu)
rather than shelling out to the interactive CLIs per run. Both use the same
methodology: synthetic/random token ids (not a real prompt) for a chosen
`n_prompt`/`n_gen`, model loaded once, KV/session reset between repetitions,
one discarded warmup pass before the timed repetitions. `lfm25-bench` emits
the exact same JSON row schema `llama-bench -o json` does (`n_prompt`,
`n_gen`, `avg_ns`, `stddev_ns`, `avg_ts`, `stddev_ts`), so `run_bench.py` uses
one loader for both.

## Requirements

- Python 3.9+, with `huggingface_hub` installed (`pip install huggingface_hub`).
- CMake and a C/C++ toolchain on `PATH` (the same ones you use to build this
  repo) -- needed to build llama.cpp and `lfm25-bench`.
- `git`, to clone llama.cpp.

## One-time setup + run

```bash
python benchmarks/run_bench.py
```

With no subcommand this clones/builds llama.cpp, builds `lfm25-bench`,
downloads both checkpoints into the HF cache if not already there, runs both
benchmarks, and writes `benchmarks/results/report.md`.

## Individual steps

```bash
python benchmarks/run_bench.py setup   # clone/build llama.cpp, build lfm25-bench, fetch both models
python benchmarks/run_bench.py run     # run llama-bench + lfm25-bench, write report.md
```

Writes `benchmarks/results/llama_cpp_<QUANT>.json`, `benchmarks/results/lfm25_cpu.json`
and `benchmarks/results/report.md` (a markdown table with prefill/decode
tokens/sec for both engines and the speedup ratio).

## Options

All available on `setup`, `run`, and the no-subcommand form:

| flag | default | meaning |
|---|---|---|
| `--quant` | `Q4_K_M` | llama.cpp GGUF quant to compare against |
| `--prompt-tokens` | 512 | synthetic prefill tokens |
| `--gen-tokens` | 128 | decode steps |
| `--threads` | 0 (auto) | thread count for both engines |
| `--reps` | 5 | timed repetitions |
| `--group` | 32 | lfm25-cpu weight quantization group size (32 or 64) |
| `--revision` | `main` | `LiquidAI/LFM2.5-230M` revision |
| `--jobs` | 12 | parallel build jobs (`setup` only) |

`GENERATOR` env var overrides the CMake generator used to build llama.cpp
(defaults to Ninja if found on `PATH`, otherwise CMake's platform default,
e.g. Visual Studio on Windows).

## Choosing a GGUF quant

Available upstream at `LiquidAI/LFM2.5-230M-GGUF`: `BF16`, `F16`, `Q4_0`,
`Q4_K_M` (default), `Q5_K_M`, `Q6_K`, `Q8_0`. Pass `--quant Q8_0` (etc.) to
compare against a different quant -- lfm25-cpu's own quantization is a fixed
groupwise-Q4 scheme (`--group`), so `Q4_K_M` or `Q4_0` are the closest
apples-to-apples points; `F16`/`BF16` compare against full precision instead.
