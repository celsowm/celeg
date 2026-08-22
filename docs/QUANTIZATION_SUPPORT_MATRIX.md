# GGUF quantization support matrix

This matrix records which GGUF block types each backend can consume, and by
which path. Like `PRIMITIVE_CAPABILITY_MATRIX.md`, a cell is marked complete
only when the corresponding decoder or kernel exists *and* a test covers it.

The matrix is not documentation of a convention -- it is a transcription of
`kGgmlTypes` in `src/checkpoint/formats/gguf.cpp`, which is the single source
of truth the loaders query through `ggml_type_support()`. Backends must not
carry their own type lists: three divergent hardcoded chains in the CUDA
loader are why a Q4_0 file with Q4_1 tensors mixed in loaded on CPU and
failed on GPU (`docs/inference_report.md:124` records an earlier instance of
the same drift).

## Column meanings

| Column | Meaning |
| --- | --- |
| CPU dequant | `cpu_gguf_dequantize_row` decodes the block type. |
| CPU native dot | `cpu_gguf_dot_scalar` consumes packed blocks directly, so `CpuWeightCodec::matrix` keeps them packed instead of repacking to groupwise Q4. |
| CUDA dequant | `dequantize_gguf_to_bf16` decodes on the host at load time; the result is then held as BF16 or requantized to int8/int4 per `--weight-mode`. |
| CUDA native MMQ | `mmq.cu` has a kernel, so `--weight-mode native` can keep the packed blocks resident on the device. |

## Matrix

| Type | Ordinal | Block | Bytes | CPU dequant | CPU native dot | CUDA dequant | CUDA native MMQ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F32 | 0 | 1 | 4 | n/a (dense) | n/a | n/a (dense) | n/a |
| F16 | 1 | 1 | 2 | n/a (dense) | n/a | n/a (dense) | n/a |
| BF16 | 30 | 1 | 2 | n/a (dense) | n/a | n/a (dense) | n/a |
| Q4_0 | 2 | 32 | 18 | complete | complete | complete | pending |
| Q4_1 | 3 | 32 | 20 | complete | complete | complete | pending |
| Q5_0 | 6 | 32 | 22 | complete | complete | complete | pending |
| Q8_0 | 8 | 32 | 34 | complete | complete | complete | pending |
| Q2_K | 10 | 256 | 84 | complete | complete | complete | pending |
| Q3_K | 11 | 256 | 110 | complete | complete | complete | pending |
| Q4_K | 12 | 256 | 144 | complete | complete | complete | complete |
| Q5_K | 13 | 256 | 176 | complete | complete | complete | pending |
| Q6_K | 14 | 256 | 210 | complete | complete | complete | complete |
| IQ3_XXS | 18 | 256 | 98 | complete | reference only | complete | pending |
| IQ4_NL | 20 | 32 | 18 | complete | reference only | complete | pending |
| IQ3_S | 21 | 256 | 110 | complete | reference only | complete | pending |
| IQ2_S | 22 | 256 | 82 | complete | reference only | complete | pending |
| IQ4_XS | 23 | 256 | 136 | complete | reference only | complete | pending |

Dense types (F32/F16/BF16) are not block-quantized; they are handled by the
dtype paths in the weight loaders, not the GGUF quant kernels. All three are
accepted by both backends for linear weights.

Types absent from the table -- Q5_1 (7), Q8_1 (9), Q8_K (15), IQ2_XXS (16),
IQ2_XS (17), IQ1_S (19), IQ1_M (29), MXFP4 (39) -- are not recognised at all.
A file containing one fails at `GgufFile::tensor()` with the tensor name and
the raw ordinal, which is enough to add a row here without re-deriving what
went wrong.

## Evidence

| Claim | Test |
| --- | --- |
| Ordinals, block geometry and names round-trip | `tests/gguf_parse_test.cpp` |
| K-quant dequant and dot agree with each other and across ISAs | `tests/cpu_gguf_kernels_test.cpp` (unit blocks, then pseudo-random blocks) |
| IQ dequant matches upstream ggml element for element | `tests/cpu_gguf_kernels_test.cpp` against `tests/data/gguf_iq_reference.inc` |
| IQ scalar dot matches a float dot over the ggml reference | same test |
| The CUDA host dequantizer reproduces the ggml reference | `tests/cuda_gguf_kernels_test.cu` against the same fixture |
| Native Q4_K/Q6_K device kernels match a double-precision reference | `tests/cuda_gguf_kernels_test.cu` |

`tests/data/gguf_iq_reference.inc` holds real blocks lifted from cached GGUF
files together with the floats ggml's own `to_float` produces for them.
Regenerate it with `tests/tools/gen_gguf_iq_reference.cpp` (build instructions
are in that file's header); it links the vendored llama.cpp in `.externals/`
and is deliberately outside the CMake build.

## Why "reference only" for the IQ types

The IQ formats encode each group of 8 weights as an index into a fixed
codebook plus a sign mask. The scalar dot kernels are implemented and tested,
but the per-element table lookups do not vectorize the way the K-quants'
bit-slicing does. Measured on `Nanbeige_Nanbeige4.2-3B-IQ4_XS.gguf`, 32
threads, 128 prefill tokens:

| CPU path | prefill tok/s | decode tok/s |
| --- | --- | --- |
| native scalar IQ dot | 2.2 | 2.0 |
| dequantize + repack to groupwise Q4 | 17.3 | 8.4 |
| Q4_K_M on the same model, for scale | 26.8 | 8.1 |

So `cpu_native_dot` is off for the IQ types and the loader takes the repack
path, which costs memory (the packed IQ advantage is lost once resident) but
is roughly 8x faster to run. Flipping the flag in `kGgmlTypes` is the only
change needed once an AVX2 IQ kernel exists to beat the repack path; the
scalar kernels stay under test so that switch stays a one-line change.

## Adding a type

1. Add a row to `kGgmlTypes` in `src/checkpoint/formats/gguf.cpp` with all
   capability flags false. The type is now parseable and named.
2. Add the packed block struct and a decoder. Shared decoders belong in
   `include/celeg/checkpoint/gguf_iq.hpp` (or a sibling) so both backends use
   one implementation; the older K-quant decoders predate that rule and are
   still duplicated per backend.
3. Generate a reference fixture from ggml rather than hand-deriving expected
   values, and add the type to the test loops.
4. Flip the capability flags the new code actually earns, and add a row here.

No backend-side list needs editing: `linear_loader.cpp`, `gguf_dequant.cpp`,
`weight_upload.cpp`, `loader_experts.cu` and `weight_codec.cpp` all branch on
`ggml_type_support()`.

## Performance: two gaps investigated, not fixed

The full 40-file sweep (`docs/GGUF_BENCHMARK_REPORT.md`) reproduces two
performance gaps against llama.cpp. Both were investigated against the
originally-hypothesized root cause; neither hypothesis survived contact with
the complete data, so neither was "fixed" — the evidence doesn't point at a
specific change the way the IQ native-dot policy decision did.

**CPU prefill (0.18x-0.96x of llama.cpp).** The original partial sweep
suggested the ratio split cleanly along whether a type has a batched AVX2
`cpu_gguf_dot4_avx2` kernel (only Q4_K and Q6_K do; every other native-dot
type falls back to four individual `cpu_gguf_dot_avx2` calls). The full sweep
does not support this: Q5_K, which has no dot4 kernel, lands at 0.40x —
statistically indistinguishable from Q4_K's 0.44x and Q6_K's 0.41x — while
Q2_K, which also lacks dot4, lands at 0.27x-0.33x, and IQ types (dequantized
and repacked, not native-dot at all) span 0.22x-0.96x. There is no clean
correlation between "has a batched AVX2 kernel" and prefill ratio in this
data. Writing batched AVX2 kernels for the remaining seven native-dot types
is real SIMD work with real correctness risk; without evidence it closes a
specific gap, that risk isn't justified. The likely actual bottleneck —
untested here — is activation requantization happening per-GEMM-call rather
than once per prefill pass, which would explain a roughly uniform ratio
across types instead of a type-dependent one.

**GPU decode (0.42x-0.83x of llama.cpp, `ok`-status rows only).** Confirmed
flat across quantization types on both models in the full sweep, consistent
with the original hypothesis of per-step launch/sync overhead or bandwidth
ceiling rather than a kernel-specific inefficiency. This needs an `ncu`
kernel-level profile to separate `w8a16_gemv_kernel` efficiency from
per-token dispatch overhead; `ncu` is not installed on this machine (`nsys`
is, but it profiles at a coarser granularity than this question needs).
Deferred rather than guessed at.
