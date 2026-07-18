# End-to-end LFM2.5-1.2B support — implementation plan

Target checkpoint: `LiquidAI/LFM2.5-1.2B-Instruct` (hub cache:
`models--LiquidAI--LFM2.5-1.2B-Instruct`). The runtime must keep running
`LiquidAI/LFM2.5-230M` unchanged; the 1.2B is an additional supported
configuration, not a replacement.

## 0. Architectural delta (1.2B vs 230M)

| Field                  | 230M (current) | 1.2B (target)                |
| ---------------------- | -------------- | ---------------------------- |
| `hidden_size`          | 1024           | 2048                         |
| `intermediate_size`    | 2560           | 8192 (config says 12288 — see note) |
| `num_hidden_layers`    | 14             | 16                           |
| `num_attention_heads`  | 16             | 32                           |
| `num_key_value_heads`  | 8              | 8                            |
| `head_dim`             | 64             | 64                           |
| `vocab_size`           | 65536          | 65536                        |
| `conv_L_cache`         | 3              | 3                            |
| `conv_dim`             | 1024           | 2048                         |
| `norm_eps`             | 1e-5           | 1e-5                         |
| `rope_theta`           | 1,000,000      | 1,000,000                    |
| `layer_types` length   | 14             | 16                           |
| attn / conv count      | 6 / 8          | 6 / 10                       |
| tie field name         | `tie_word_embeddings` | `tie_embedding`       |
| rope_theta location    | `rope_parameters.rope_theta` | top-level `rope_theta` |

`layer_types` for the 1.2B (0-indexed):
`conv, conv, attn, conv, conv, attn, conv, conv, attn, conv, attn, conv, attn, conv, attn, conv`

**Note on `intermediate_size`**: the published 1.2B `config.json` reports
`intermediate_size: 12288`, but the stored `feed_forward.w1.weight` tensors are
`[8192, 2048]`. The runtime MUST take the FFN inner dimension from the actual
safetensors shape (as the existing loader already does via `expected` shape
validation), not from `config.intermediate_size`. The `ModelConfig` field
becomes informational only.

Tensor names are identical to the 230M layout (`model.embed_tokens.weight`,
`model.embedding_norm.weight`, `model.layers.N.{operator_norm,ffn_norm,
feed_forward.w1/w2/w3,conv.*,self_attn.*}`). The tokenizer is the same
`tokenizer.json` format. No new kernels are required — only generalization
of constants and per-layer scheduling.

## 1. Current hardcoding (what must change)

- `include/lfm/runtime_types.hpp:11` — `struct LfmConfig` is `static constexpr`
  with 230M dimensions. Referenced ~700 times across `src/model.cu`,
  `src/packed.cu`, `src/cpu_model.cpp`, `src/cpu_packed.cpp`, `src/paged_kv.cu`,
  `include/lfm/detail/model_impl.hpp`, `src/c_api.cpp`, `src/runtime_types.cpp`,
  and `tests/cuda_kernels_test.cu`.
- `src/config.cpp:105` — `ModelConfig::validate_compiled_backend()` rejects
  every dimension that differs from 230M.
- `src/model.cu:181` and `src/cpu_model.cpp:235` — `attention_map[LfmConfig::layers]`
  is a hardcoded 14-entry array. New layer counts/schedules are unreachable.
- `include/lfm/detail/model_impl.hpp:245-256` — workspace `DeviceBuffer<>`
  sizes use `LfmConfig::hidden`, `LfmConfig::vocab`, `LfmConfig::intermediate`,
  etc., so they cannot grow for the 1.2B.
- `src/config.cpp:46` — loader reads `rope_parameters.rope_theta` and
  `tie_word_embeddings` only; the 1.2B config uses top-level `rope_theta` and
  `tie_embedding`.
- `tests/config_test.cpp` and `tests/cuda_kernels_test.cu` hardcode 230M
  constants.

Kernel launch signatures in `include/lfm/kernels.cuh` already take
`q_heads`, `kv_heads`, `head_dim`, `kv_width` as runtime parameters — those
do not need to change. The CPU kernels in `include/lfm/cpu_kernels.hpp` are
similarly parameterized. The migration is therefore mostly about plumbing
runtime dimensions through state structs and removing the compile-time
constants, not rewriting kernels.

## 2. Phased migration

### Phase A — Config & shape plumbing (no behavior change for 230M)

Goal: load and validate the 1.2B config without crashing; 230M path stays
bit-identical.

1. **`src/config.cpp` / `include/lfm/config.hpp`**
   - Accept `rope_theta` either at root or inside `rope_parameters`.
     Prefer root; fall back to nested for the 230M config.
   - Accept `tie_embedding` as an alias for `tie_word_embeddings`.
   - Read `num_attention_heads` from `num_heads` if `num_attention_heads` is
     absent (the 1.2B publishes both with the same value; keep `num_attention_heads`
     authoritative when present).
   - Stop rejecting unknown top-level fields (already tolerant via `Json`).
   - Replace `validate_compiled_backend()`'s 230M-only equality check with a
     small **supported-architectures** table keyed by `(hidden, layers,
     q_heads, kv_heads, head_dim, vocab, conv_cache)`. Both 230M and 1.2B
     entries are listed. The `layer_types` schedule is validated against the
     table entry rather than a hardcoded array.
   - `ModelConfig::intermediate_size` stays as loaded for diagnostics, but
     callers stop using it for sizing (see step 3).

2. **`include/lfm/runtime_types.hpp`**
   - Convert `struct LfmConfig` from `static constexpr` to a plain runtime
     struct `LfmShape` with the same fields. Add a factory
     `LfmShape::from_config(const ModelConfig&)`.
   - Keep `LfmConfig` as a deprecated alias for the 230M defaults for one
     release, used only by tests that explicitly exercise the 230M path.
     New code uses `LfmShape`.

3. **`include/lfm/detail/model_impl.hpp` and `src/model.cu`**
   - Store `LfmShape shape_` on `Impl`. Build it in the constructor from
     `ModelConfig::load(...)`.
   - Replace every `LfmConfig::xxx` reference with `shape_.xxx` (or with the
     locally cached `int` in hot loops where the compiler already hoists it).
   - Replace `attention_map[14]` with iteration over `config.layer_types`:
     `for (int i = 0; i < shape_.layers; ++i) { if (layer_types[i] ==
     FullAttention) {...} else {...} }`.
   - Size all `DeviceBuffer<>` workspaces from `shape_` (hidden, intermediate,
     vocab, q_width, kv_width, qkv_width, conv_cache).
   - Take FFN inner dimension from the actual `feed_forward.w1.weight` shape
     (`shape[0]`) instead of `LfmConfig::intermediate`. Validate that
     `w2.weight` shape[1] matches.
   - `SessionHeader` (model.cu:41) already stores `layers/kv_width/kv_heads/
     vocab`; extend with `hidden` and `intermediate` so saved sessions are
     self-describing. Bump the magic to `LFMSESS2` to reject old sessions
     during the transition.

4. **`src/packed.cu`, `src/paged_kv.cu`, `src/c_api.cpp`,
   `src/runtime_types.cpp`**
   - Replace `LfmConfig::xxx` with `shape_.xxx` passed from `Impl` (or from a
     `CudaModelShape` context struct that wraps the per-session shape).

5. **CPU backend**
   - `src/cpu_model_internal.hpp:41` — `Shared` already takes `path`/`context`/
     `options`; add `LfmShape shape` constructed from `ModelConfig::load`.
   - `src/cpu_model.cpp` — replace 119 `LfmConfig::xxx` references with
     `shared->shape.xxx`; replace the 14-entry `attention_map` with
     `config.layer_types` iteration.
   - `src/cpu_packed.cpp`, `src/cpu_concurrent.cpp`, `src/cpu_compare_reference.cpp`
     — same mechanical replacement.

6. **Tests**
   - `tests/config_test.cpp` — add a second case that loads a synthesized
     1.2B config and verifies `validate_compiled_backend()` accepts it.
   - `tests/cuda_kernels_test.cu` — parameterize the 5 `LfmConfig::kv_width`
     references to a local `int kv_width` so the same tests can run against
     a 1.2B-shaped fixture.
   - Add `tests/shape_test.cpp` to lock in `LfmShape::from_config` for both
     checkpoints.

7. **CMake / CLI**
   - No new targets. `lfm25-cpu-run --model DIR` already discovers
     `config.json` next to `model.safetensors`; both checkpoints work with
     the same flow once the loader is generalized.

**Exit criteria for Phase A**: `lfm25-cpu-run --model
<LFM2.5-1.2B-Instruct> --prompt "hello" --max-new-tokens 1` runs to
completion without an assertion. Logits are not yet required to match the
official model. The 230M ctest suite stays green.

### Phase B — Numerical parity for the 1.2B

Goal: byte-for-byte (BF16) match against the official `transformers`
reference for the 1.2B, using the existing `PARITY.md` workflow.

1. **Reference exporter** — extend `scripts/export_cpu_reference.py` (currently
   230M-only) to accept any `Lfm2ForCausalLM` checkpoint. The script already
   uses `transformers.AutoModelForCausalLM`; only the path argument and the
   expected-vocab assertion need to be generalized.
2. **Comparator** — `src/compare_logits.cpp` and `src/cpu_compare_reference.cpp`
   are already dimension-generic (they read the vector length from the file
   header). Verify and add a 1.2B fixture to the comparator tests.
3. **Parity sweep** — run the recommended progression from `PARITY.md`
   against the 1.2B: tokenizer IDs, one-token BF16 logits, batched vs legacy
   prefill, official vs strict BF16, then quantization drift (W8A16, W4A16).
3. **Fix per-layer schedule mismatches** — the 1.2B has two consecutive conv
   layers at positions (3,4), (6,7). Verify the conv-state cursor advance and
   the layer-major prefill chunking handle two convs back-to-back without
   cross-contaminating state. Add a regression test in
   `tests/cpu_kv_cache_test.cpp` that runs a 16-layer synthetic schedule.
4. **Attention workspace sizing** — the 1.2B has 32 query heads per layer
   (vs 16). `attention_partial_max_/denom_/accum_` sizing in
   `model_impl.hpp:257` uses `LfmConfig::q_heads * attention_chunks_`; after
   Phase A this becomes `shape_.q_heads * attention_chunks_`. Re-check the
   segmented-attention chunk count and the per-block reduction bounds.

**Exit criteria for Phase B**: maximum/mean absolute logit error against the
official BF16 1.2B reference is at or below the 230M baseline; greedy and
seeded sequences match token-for-token over a prompt suite.

### Phase C — Performance, concurrency, and packaging

1. **Concurrent engine** — `src/concurrent.cpp`, `src/cpu_concurrent.cpp`,
   and `src/cpu_prefix_cache.cpp` must use the runtime shape for prefix
   snapshots, page counts and COW byte budgets. The 1.2B's larger
   `hidden=2048` doubles the per-page byte cost; revisit the default LRU
   byte budget in `CpuPrefixCacheManager` (currently sized for 230M pages).
2. **Prefix snapshot schema** — `PrefixState` (runtime_types.hpp:111) stores
   `conv_state_bf16` as a flat vector sized for 230M. Encode the conv-state
   length explicitly (`conv_cache * hidden * num_conv_layers`) and version
   the snapshot so a 1.2B session cannot be restored into a 230M model.
3. **Pack cache** — `CpuPackWriter`/`CpuPackReader` (cpu_pack.cpp,
   cpu_packed.cpp) write a fixed header. Add `hidden`/`intermediate`/`layers`
   to the pack header and reject mismatched packs at load time. Bump the pack
   format version.
4. **C API** — `include/lfm/c_api.h` and `cpu_c_api.h` expose
   `lfm25_model_create(safetensors_path, ...)`; no API change required, but
   the `MANIFEST.sha256` and example docs should add a 1.2B example.
5. **Benchmarks & docs** — extend `BENCHMARK.md`, `CPU.md`, `CPU_API.md`,
   `PARITY.md`, `STATUS.md`, and `README.md` with 1.2B usage. Add a
   `scripts/cpu_long_prefill_benchmark.sh` invocation example for the 1.2B
   path. Update `STATUS.md` "Not yet implemented" to remove the
   "physical official-checkpoint parity" caveat for the 1.2B once Phase B
   passes.

**Exit criteria for Phase C**: concurrent benchmark runs against the 1.2B
with prefix cache hits, NUMA placement, and paged KV; documented end-to-end
tokens/s on the packaging host; `MANIFEST.sha256` lists the 1.2B checkpoint.

## 3. Sequencing & risk

- Phase A is mechanical and high-volume (~700 `LfmConfig::` references) but
  low-risk: it is a refactor from compile-time to runtime constants plus a
  config-loader extension. Do it first, in one PR-sized commit per file
  group: (1) `runtime_types.hpp` + `config.cpp`, (2) `model.cu` +
  `model_impl.hpp`, (3) `packed.cu` + `paged_kv.cu`, (4) CPU model, (5) tests.
- Phase B is where real bugs surface. The biggest risk is the
  back-to-back conv layers at 1.2B positions (3,4) and (6,7); the existing
  conv-state ring buffer assumes one conv per attention boundary. Audit
  `cpu_qk_norm_rope`/`cpu_conv_decode`/`launch_conv_decode_*` cursor math
  before the first 1.2B prefill.
- Phase C is performance tuning and packaging; defer until parity is green.
- No GPU is available in the packaging environment, so the CUDA backend is
  validated by inspection and unit tests only. CPU is the primary target.

## 4. Validation commands (target state)

```bash
# 230M regression (must stay green)
cmake -S . -B build-cpu -DLFM_ENABLE_CUDA=OFF -DLFM_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure

# 1.2B end-to-end
./build-cpu/lfm25-cpu-run \
  --model ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/868df74dd56ff8a0c2ac5dbf281690c2dbebe4c9 \
  --prompt "Explique CUDA em uma frase." \
  --cpu-isa auto --cpu-kv-cache bf16 --threads 8 --max-new-tokens 32

# 1.2B parity vs official reference
python scripts/export_cpu_reference.py \
  --model LiquidAI/LFM2.5-1.2B-Instruct \
  --prompt "Explique CUDA." --out reference_1b.f32
./build-cpu/lfm25-cpu-run --model <1.2B dir> --prompt "Explique CUDA." \
  --max-new-tokens 0 --dump-logits batched_1b.f32
./build-cpu/lfm25-compare-logits reference_1b.f32 batched_1b.f32
```

## 5. Out of scope

- New SIMD kernels (AMX-INT8, ARM I8MM/SME2) — tracked separately in
  `STATUS.md`.
- INT8 CPU KV pages.
- CUDA end-to-end validation (no GPU in the packaging environment).
- Retraining or fine-tuning the 1.2B.
