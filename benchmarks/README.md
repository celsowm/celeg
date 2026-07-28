# CPU benchmarks: lfm25-cpu vs llama.cpp

Automated, reproducible comparison between this repo's native CPU engine and
upstream [llama.cpp](https://github.com/ggml-org/llama.cpp), both running the
exact same pinned LFM2.5-230M Q4_K_M GGUF file.

Everything is driven by one cross-platform script, `run_bench.py` (Windows,
Linux, macOS -- no shell-specific scripts to keep in sync). llama.cpp is
cloned into `.externals/llama.cpp` (gitignored, never committed) and built
CPU-only in Release. Both engines read the same GGUF straight out of the
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

Decode timing uses direct evaluation of a predetermined token stream on the LFM
side; it does not include greedy vocabulary sampling. The runner passes an
explicit equal thread count to both engines, uses 256-token llama.cpp batch and
microbatch sizes to match the LFM chunk size, and pins the llama.cpp and LFM
GGUF revisions used by the comparison. Both KV caches use BF16.

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
downloads the pinned GGUF into the HF cache if it is not already there, runs both
benchmarks, and writes `benchmarks/results/report.md`.

## Individual steps

```bash
python benchmarks/run_bench.py setup   # clone/build both engines and fetch the GGUF
python benchmarks/run_bench.py run     # run llama-bench + lfm25-bench, write report.md
```

Writes `benchmarks/results/llama_cpp_<QUANT>.json`, `benchmarks/results/lfm25_cpu.json`
and `benchmarks/results/report.md` (a markdown table with prefill/decode
tokens/sec for both engines and the speedup ratio).

## Options

All available on `setup`, `run`, and the no-subcommand form:

| flag | default | meaning |
|---|---|---|
| `--quant` | `Q4_K_M` | fixed native GGUF quant supported by both engines |
| `--prompt-tokens` | 512 | synthetic prefill tokens |
| `--gen-tokens` | 128 | decode steps |
| `--threads` | 8 | explicit thread count for both engines |
| `--reps` | 5 | timed repetitions |
| `--jobs` | 12 | parallel build jobs (`setup` only) |

`GENERATOR` env var overrides the CMake generator used to build llama.cpp
(defaults to Ninja if found on `PATH`, otherwise CMake's platform default,
e.g. Visual Studio on Windows).

## Choosing a GGUF quant

The comparison intentionally accepts only `Q4_K_M`. That file contains native
Q4_K and Q6_K tensors, both executed directly by lfm25-cpu and llama.cpp.
Unsupported GGUF quantizations fail explicitly rather than being silently
dequantized or requantized.

Any comparison produced before native same-file GGUF support is invalid. Those
older runs compared different checkpoint formats or quantizations; only reports
whose JSON rows contain the same canonical path, size and SHA-256 are valid.
