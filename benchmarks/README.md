# CPU benchmarks: celeg-cpu vs llama.cpp

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
(`llama-bench` for llama.cpp, this repo's own `celeg-bench` for celeg-cpu)
rather than shelling out to the interactive CLIs per run. Both use the same
methodology: synthetic/random token ids (not a real prompt) for a chosen
`n_prompt`/`n_gen`, model loaded once, KV/session reset between repetitions,
one discarded warmup pass before the timed repetitions. `celeg-bench` emits
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
  repo) -- needed to build llama.cpp and `celeg-bench`.
- `git`, to clone llama.cpp.

## One-time setup + run

```bash
python benchmarks/run_bench.py
```

With no subcommand this clones/builds llama.cpp, builds `celeg-bench`,
downloads the pinned GGUF into the HF cache if it is not already there, runs both
benchmarks, and writes `benchmarks/results/report.md`.

## Individual steps

```bash
python benchmarks/run_bench.py setup   # clone/build both engines and fetch the GGUF
python benchmarks/run_bench.py run     # run llama-bench + celeg-bench, write report.md
```

Writes `benchmarks/results/llama_cpp_<QUANT>.json`, `benchmarks/results/celeg_cpu.json`
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
Q4_K and Q6_K tensors, both executed directly by celeg-cpu and llama.cpp.
Unsupported GGUF quantizations fail explicitly rather than being silently
dequantized or requantized.

Any comparison produced before native same-file GGUF support is invalid. Those
older runs compared different checkpoint formats or quantizations; only reports
whose JSON rows contain the same canonical path, size and SHA-256 are valid.

## CUDA native-GGUF race

`run_cuda_bench.py` is the separate, reproducible RTX 3060 comparison for
`ggml-org/SmolLM3-3B-GGUF:SmolLM3-Q4_K_M.gguf`. It requires the cached model,
the CUDA build under `out/windows-cuda-relwithdebinfo`, and the local pinned
llama.cpp checkout at commit `6b36c2305644fd30db7cce3f4840c74574f31ce9`.

```bash
python benchmarks/run_cuda_bench.py
```

It forces Celeg's `--weight-mode native`, uses BF16 KV with prefill 512 / decode
128 and batch/microbatch 512, discards one full run, then records five samples.
The JSON report includes the commands, GGUF SHA-256, GPU identity, mean,
median and standard deviation. It succeeds only when native weights stay at or
below 2.0 GiB and Celeg is at least 1.05x llama.cpp in both phases.

---

# Benchmark manifests (Phase 0 reference fixtures)

In addition to the CPU-vs-llama.cpp comparison above, this directory stores
**benchmark configuration** for reference fixtures, separated from
**benchmark results** (which live in `benchmarks/results/` and are gitignored).

A manifest describes how to reproduce a deterministic reference run: the
checkpoint, the runtime options, the prompt, and the sampling parameters. A
result is the captured output of executing a manifest (a token sequence, a
logits snapshot, or both). Storing them apart is Phase 0 task 0.2 of
`docs/ARCHITECTURE_RULES.md`.

## Manifest files

`benchmarks/manifests/*.json` — one file per reference scenario:

| file | checkpoint | mode |
|---|---|---|
| `dense_bf16.json` | `LiquidAI/LFM2.5-230M` | BF16 dense |
| `dense_int8.json` | `LiquidAI/LFM2.5-230M` | INT8 per-output-channel dense |
| `dense_int4.json` | `LiquidAI/LFM2.5-230M` | INT4 per-output-channel dense |
| `gguf_q4k.json` | `LiquidAI/LFM2.5-230M-GGUF:Q4_K_M` | native GGUF MMQ |
| `gguf_q6k.json` | `LiquidAI/LFM2.5-230M-GGUF:Q6_K` | native GGUF MMQ |
| `moe_bf16.json` | `LiquidAI/LFM2.5-8B-A1B` | BF16 MoE (use expert offload on 12 GB GPUs; larger GPU only for fully resident weights) |

Manifest schema and tolerance classes are documented inline in each manifest;
tolerances are model- and format-specific, never one global number (Phase 0
task 0.3). Each manifest declares its own threshold for cosine similarity,
RMSE, max absolute error, top-k agreement, and end-to-end token agreement.

## Reproducing a manifest

```bash
python benchmarks/run_manifest.py benchmarks/manifests/dense_bf16.json
python benchmarks/run_manifest.py --all                          # every manifest
python benchmarks/run_manifest.py <path> --update-expected       # refresh the recorded baseline
```

`run_manifest.py` resolves the checkpoint from the local HF cache (downloading
if missing, via `celeg-download`), invokes `celeg-run` against the build
discovered by `scripts/dev.py`, and writes the captured output to
`benchmarks/results/<name>.json`. When the manifest records an
`expected_sequence` (greedy `top_k == 1`), the runner compares the captured
sequence against it; when only `expected_stdout_sha256` is recorded (the
common case while token-id capture is not yet wired), the runner compares the
output hash.

## Compile-time baseline

```bash
python scripts/measure_compile.py          # clean + 2 incremental + binary size
python scripts/measure_compile.py --skip-clean   # incremental rebuilds only
```

Writes `benchmarks/compile_baseline.json` (gitignored) with clean build time,
two incremental rebuild times (one after touching an attention kernel, one
after touching a model orchestration file), and the final `celeg-run` binary
size.

## Metal benchmark

The Metal benchmark keeps model loading outside the timed samples and measures
deterministic direct token evaluation after one warm-up run. It resolves the
checkpoint from the local Hugging Face cache and records the platform, backend,
manifest, and CELEG revision in the result:

```bash
python benchmarks/run_metal_bench.py \
  benchmarks/manifests/metal_lfm25_350m_q4_k_m.json
```

The current baseline is recorded in `docs/METAL_BENCHMARK_REPORT.md`. It is a
correctness/performance reference, not a cross-machine performance threshold.
