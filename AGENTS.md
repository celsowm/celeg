# AGENTS.md

## Hard rules

- **NO backward compatibility.** When refactoring, fully replace old access
  patterns with the new interface. Do not keep legacy shortcuts, `impl_->`
  member access, or shim helpers "for compatibility". Delete the old path.
- When a regex substitution touches multiple overlapping patterns, apply the
  most specific pattern first (e.g. `position_device_` before `position_`),
  otherwise a partial match corrupts the output (seen: `position_device_` ->
  `position()device_`).
- Prefer interface accessors over reaching into internals:
  - `IPackedSession` exposes `position()` / `set_position(int)`,
    `sampled_host()` / `set_sampled_host_value(int32_t)`, `rope_cos()` /
    `rope_sin()` (raw `const __nv_bfloat16*`, NOT a class — do not call
    `.data()` on them), `metrics()`, `phase()` / `set_phase(...)`,
    `active_segmented_attention()` / `set_active_segmented_attention(...)`,
    `use_segmented_attention(int)`, `max_context()`, `packed_session()`.
  - `++x.member` on a value-returning accessor is illegal; use
    `x.set_member(x.member() + 1)`.

## Build

```text
python scripts/dev.py verify
```

The harness dynamically discovers Visual Studio, CUDA, the GPU architecture,
runtime DLLs and checkpoints. It always performs a fresh configure and writes
to `out/<platform>-<backend>-<build-type>`. Use `--backend cuda` when CUDA is
required or `--backend cpu` for a CPU-only check. All 45 configured CTest tests
must pass for a CUDA build.

## Model checkpoints (HuggingFace cache)

Checkpoints are normally resolved from the local HuggingFace cache, not from a
local `./model/...` directory. `lfm25-run` accepts `--repo <HF_REPO_ID>`, which
calls `lfm25::resolve_hf_model()` to locate the snapshot under the HF cache:

- Windows: `C:\Users\<user>\.cache\huggingface\hub\models--<owner>--<repo>\snapshots\<commit>\`
- Linux:   `~/.cache/huggingface/hub/models--<owner>--<repo>\snapshots\<commit>\`

The downloader (`scripts/download_model.sh` / `lfm25-download`) populates that
cache; `--repo` then auto-resolves without copying files. Sharded checkpoints
(`model.safetensors.index.json` + `model-0000N-of-0000M.safetensors`) are
resolved through `SafeTensorRepository`.

Prefer `--repo` over `--model` in smoke checks and tests so runs work against
the cached checkpoints already present on the machine.

## Performance investigation

**Measure before optimizing a kernel.** Nsight Compute usually cannot run here
(`ERR_NVGPUCTRPERM` needs elevated GPU counter permissions), and CUDA decode is
captured into one graph, so per-kernel attribution is not free. Use these:

```text
python scripts/profile_decode.py --model <gguf>            # per-phase breakdown
python scripts/profile_decode.py --model <gguf> --sweep    # config A/B, graph on
python scripts/gguf_census.py <gguf>                       # tensor types + traffic
./out/<build>/lfm25-decode-gemv-benchmark 200              # GEMV vs peak bandwidth
```

`profile_decode.py` drives the in-tree profiler in
`include/lfm/backend/cuda/phase_profile.hpp` (env `LFM_PROFILE_DECODE=1`, needs
`--no-cuda-graph`; it is a no-op branch otherwise, so it stays in the hot path).

Why this section exists: a plausible weight-bandwidth argument once concluded
the decode GEMV path ran at ~17% of peak and that native-quantized weights would
give a ~3x decode win. Measuring showed the GEMV path was already at 76% of peak
and only ~22% of a decode step, while 66% was a single-threaded top-k loop in
the sampler. The arithmetic was right; the attribution was wrong. Get a profile
first.

## Smoke checks

```text
python scripts/dev.py smoke --backend cuda
```

The smoke command checks CLI startup, the C API, expert residency and cached
230M inference. It never downloads a checkpoint implicitly.

Note: the LFM2.5-8B-A1B checkpoint is sharded across ~16 GB of BF16 weights and
does not fit in the 12 GB VRAM of the reference RTX 3060, so full BF16
inference validation of that model must run on a larger GPU.
