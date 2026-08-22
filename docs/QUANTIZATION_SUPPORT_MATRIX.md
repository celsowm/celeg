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

## Quality gate

Nothing else in this repo checks output *quality* per quantization --
`benchmarks/compare_llama.py` and `run_gguf_sweep.py`'s speed rows only prove
a file loads and runs at a normal rate, and the IQ dequant fixtures in
`tests/data/gguf_iq_reference.inc` check accuracy at the block level, over
whichever 4 blocks per type happen to be checked in. A decode-time bug can
pass all of that and still emit corrupted text, which is exactly what
happened with the two bugs fixed in commit `4aa90e1` (an 8x-too-small
attention score on every QK-norm model, and int-truncation in the AVX2 Q8_0
dot) -- both loaded cleanly, benchmarked at normal speed, and were invisible
to the full test suite.

`run_gguf_sweep.py` now runs both engines greedy (temperature 0, top-k 1,
top-p 1, repetition-penalty 1, `--ignore-eos` on the llama.cpp side so a
model that predicts an early turn-end token doesn't just make the row
shorter than celeg's fixed-length `--raw` output) from the same 12-token
prompt continuation and diffs the generated text with
`difflib.SequenceMatcher`. Lizzy-7B has no llama.cpp reference at all
(`general.architecture = "lizzy"` has no llama.cpp graph) and is gated
celeg-CPU-vs-celeg-CUDA instead.

**The ratio is not calibrated toward 1.0, even for a fully correct quant.**
Two independent kernel implementations of the same model routinely pick
different top-1 tokens a few tokens into open-ended generation, once the gap
between the best and second-best logit gets small -- this is ordinary
floating-point noise, not a bug, and every quant class exhibits it. What a
real decode bug does that noise does not is corrupt the *early*,
high-confidence tokens too, collapsing the ratio over the whole completion
rather than just its tail. Thresholds are therefore derived from one real
measured run (Nanbeige-3B, one file per class, `--max-new-tokens 12`) rather
than invented:

| quant class | file | agreement | threshold |
|---|---|---|---|
| dense-f16-bf16 | `Nanbeige_Nanbeige4.2-3B-bf16.gguf` | 0.60 | 0.50 |
| q8_0 | `Nanbeige_Nanbeige4.2-3B-Q8_0.gguf` | 0.60 | 0.50 |
| k-quant | `Nanbeige_Nanbeige4.2-3B-Q4_K_M.gguf` | 0.56 | 0.45 |
| legacy-q4-q5 | `Nanbeige_Nanbeige4.2-3B-Q4_0.gguf` | 0.18 | 0.25 |
| iq | `Nanbeige_Nanbeige4.2-3B-IQ4_XS.gguf` | 0.53 | 0.20 |
| iq | `Nanbeige_Nanbeige4.2-3B-IQ3_XXS.gguf` | 0.03 | 0.20 |

Two of those six rows are set to flag their own measured value rather than
clear it, because they are real, investigated findings rather than
calibration noise:

- **`Q4_0` on Nanbeige-3B (CPU only).** celeg CPU's native-dot decode of
  "The capital of France is" continues into unrelated Chinese text
  immediately after "Paris."; celeg CUDA (`--weight-mode auto`, which
  dequantizes GGUF blocks through the same shared `q4_0_decode` used by the
  host dequantizer, not the CPU's packed native-dot kernel) and llama.cpp
  both continue "Paris.\n\</think\>\n\n...". celeg CPU's scalar and AVX2 Q4_0
  dot kernels (`src/backend/cpu/kernels/gguf.cpp:467`,
  `gguf_avx2.cpp:444`) agree with each other byte-for-byte and their
  dot-product algebra (`dot - 8*bsum`, matching ggml's zero-point
  convention) checks out on inspection, which rules out an ISA-specific
  kernel bug the way the Q8_0 int-truncation bug was one. Not resolved: a
  packing-order bug in how `CpuWeightCodec` lays GGUF's raw Q4_0 bytes into
  `packed_row` has not been ruled out, and Q4_0 is also legacy's
  worst-precision format (single fp16 scale per 32 elements, no K-quant
  sub-block scale), so this is equally consistent with the flip being
  genuine quantization noise on a near-tied logit. Flagged rather than
  fixed or dismissed.
- **`IQ3_XXS` on Nanbeige-3B (CPU only).** celeg CPU produces unrelated,
  lower-quality text; celeg CUDA and llama.cpp agree with each other. Unlike
  Q4_0, this one has a specific, plausible mechanism: IQ3_XXS has no
  `cpu_native_dot` support (see the matrix above), so `CpuWeightCodec`
  dequantizes it through the *same* shared `gguf_iq.hpp` grid-table decode
  CUDA's host dequantizer uses -- ruling out a decode bug, since both
  backends call identical code -- and then re-quantizes that already-lossy
  3.06-bit-per-weight result into groupwise 4-bit blocks for the CPU's
  native Q4 dot kernel. That second, CPU-only quantization pass costs real
  precision on top of an already very lossy source, and IQ3_XXS is the
  lowest-bit IQ type in the cache. `IQ4_XS`, repacked through the identical
  path, did not show the same collapse, consistent with there being more
  headroom left after 4.25 bits/weight than after 3.06. This is the same
  double-quantization mechanism already flagged for CPU F16 support (W2):
  dense F16/BF16 weights get the identical groupwise-4-bit repack on CPU,
  and simulating that repack in fp64 was enough on its own to flip
  LFM2.5's top-1 prediction. Fixing this for real means giving IQ3_XXS (and
  the other native-dot-less IQ types) a CPU native-dot kernel instead of
  the repack path -- which is exactly the AVX2 IQ kernel work already
  sequenced after CPU prefill profiling below, not a quick patch here.

**Gate has teeth**, checked by mutation: reintroducing the query-scale bug
from `4aa90e1` into `apply_cpu_attention_qk`'s QK-norm branches and
rebuilding drops LFM2.5-350M-Q4_K_M's agreement from 1.00 to 0.33 against
its k-quant threshold of 0.45 -- caught. Reverting restores 1.00.

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
flat across quantization types on both models in the full sweep. It is flatter
than that: celeg's decode rate does not respond to resident weight bytes at
all. Every Nanbeige-3B GPU row sits at 130-135 tok/s from IQ3_XXS (1.76 GB)
through Q8_0 (4.13 GB) to bf16 (7.77 GB), while llama.cpp tracks size the way
a bandwidth-bound decoder should (318 -> 110 tok/s); Lizzy-7B q8_0 at 7.23 GB
decodes *faster* (141.6) than Nanbeige-3B at 2.50 GB. That rules out a
bandwidth ceiling and points at a per-step cost that is fixed with respect to
weight traffic.

Decode already runs under CUDA graphs (`decode_graphs.hpp`, captured in
`execution.cu`), so naive per-launch overhead should already be amortized,
which makes the flatness more suspicious rather than less. Resolving it needs
a kernel-level profile separating `w8a16_gemv_kernel` efficiency from
per-token dispatch. `ncu` **is** available on this machine at
`/usr/local/cuda-13.2/bin/ncu` -- it is simply not on `PATH`, the same
resolution trap that affects `nvcc` here. An earlier revision of this section
recorded it as not installed and deferred on that basis; that was wrong, and
the investigation is not blocked.
