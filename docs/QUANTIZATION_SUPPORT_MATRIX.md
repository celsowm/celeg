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
| Q4_0 | 2 | 32 | 18 | complete | complete | complete | not planned* |
| Q4_1 | 3 | 32 | 20 | complete | complete | complete | not planned* |
| Q5_0 | 6 | 32 | 22 | complete | complete | complete | not planned* |
| Q8_0 | 8 | 32 | 34 | complete | complete | complete | not planned* |
| Q2_K | 10 | 256 | 84 | complete | complete | complete | not planned* |
| Q3_K | 11 | 256 | 110 | complete | complete | complete | not planned* |
| Q4_K | 12 | 256 | 144 | complete | complete | complete | complete |
| Q5_K | 13 | 256 | 176 | complete | complete | complete | not planned* |
| Q6_K | 14 | 256 | 210 | complete | complete | complete | complete |
| IQ3_XXS | 18 | 256 | 98 | complete | reference only | complete | not planned* |
| IQ4_NL | 20 | 32 | 18 | complete | reference only | complete | not planned* |
| IQ3_S | 21 | 256 | 110 | complete | reference only | complete | not planned* |
| IQ2_S | 22 | 256 | 82 | complete | reference only | complete | not planned* |
| IQ4_XS | 23 | 256 | 136 | complete | reference only | complete | not planned* |

Dense types (F32/F16/BF16) are not block-quantized; they are handled by the
dtype paths in the weight loaders, not the GGUF quant kernels. All three are
accepted by both backends for linear weights.

\* **Not planned, not merely undone**: `--weight-mode native` (the mode these
kernels would serve) measures 4.85x slower prefill and ~27% slower decode
than the int8 (`auto`) path it would compete with, on the two types (Q4_K,
Q6_K) that already have MMQ kernels -- see "Native weight mode" under
Performance below. Adding kernels for the other 12 types to a path that
measures decisively worse than the alternative on every type it already
covers is negative-value work until that gap closes.

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
which makes the flatness more suspicious rather than less. `ncu` **is**
available on this machine at `/usr/local/cuda-13.2/bin/ncu` -- it is simply
not on `PATH`, the same resolution trap that affects `nvcc` here. An earlier
revision of this section recorded it as not installed and deferred on that
basis; that was wrong.

**Update: profiled with `nsys` (Nanbeige-3B-Q4_K_M, `--weight-mode auto`).**
CUDA graph capture does engage on the GGUF/int8 path: `celeg-run` now prints
`benchmark.cuda_graph_ready` after `--benchmark-decode`
(`src/app/cuda/main.cpp`, next to the other `benchmark.*` lines), and it
reads `1` for `auto`, `bf16`, and `native` alike. That rules out the
"graph capture silently isn't engaging" hypothesis from this doc's earlier
revision.

The profile itself needed one non-obvious flag: `nsys profile
--cuda-graph-trace=node` (the default is `graph`, which only records the one
real kernel execution from graph *capture* and shows every subsequent
*replay* as a single opaque node with no per-kernel breakdown -- with the
default flag, 132 decode steps across 44 layers looked like exactly 2 full
forward passes, which is what led to briefly suspecting a per-tensor
weight-mode resolver bug before the flag was found). With `node` tracing,
one 128-step decode run (`nsys stats --report cuda_gpu_kern_sum`) breaks
down as:

| kernel | instances | total time | share |
|---|---|---|---|
| `w8a16_gemv_kernel` | 34989 | 613 ms | 59.1% |
| `gqa_decode_segment_partial_kernel` | 5808 | 192 ms | 18.5% |
| `rmsnorm_kernel` | 12060 | 59 ms | 5.7% |
| `gqa_decode_segment_reduce_kernel` | 5808 | 37 ms | 3.6% |
| `argmax_bf16_kernel` | 132 | 33 ms | 3.2% |
| `paired_qk_norm_rope_kernel` | 11616 | 27 ms | 2.6% |

Summed GPU-kernel time across the trace is ~7.46 ms/decode step, matching
the ~7.7 ms/step `benchmark.decode_ms_per_token` from a clean (non-profiled)
run to within a few percent. **That resolves the plan's gate 2 on its own:
kernel time is not `<<` wall time, so this is not a host-side dispatch or
sync problem** -- the GPU is genuinely busy the whole step, spread thinly
across many 5-33 &micro;s launches rather than concentrated in one obviously
broken kernel. `w8a16_gemv_kernel` -- the per-layer int8-weight GEMV -- is
the largest single contributor by a wide margin, consistent with (but not
proof of) it being under-occupied for M=1: 34989 calls over 132 steps is
~265 calls/step, and 17.5 &micro;s average is small enough that per-launch
fixed cost (not sustained bandwidth) is a plausible explanation, which is
exactly what gate 3 exists to distinguish. `argmax_bf16_kernel` is a
secondary oddity worth a separate look: 253 &micro;s for one argmax over a
166144-entry vocabulary is high for what should be a straightforward
reduction, though at 3.2% of total time it is not the main story.

**Gate 3, resolved: `ncu --set full` against `w8a16_gemv_kernel`, two model
sizes.** `ERR_NVGPUCTRPERM` (the admin-only profiling-counter restriction
noted in an earlier revision of this section) is lifted on this machine —
`/etc/modprobe.d/99-nvidia-profiling.conf` sets
`NVreg_RestrictProfilingToAdminUsers=0`, confirmed by `RmProfilingAdminOnly:
0` in `/proc/driver/nvidia/params` after a reboot. Profiled Nanbeige-3B
Q4_K_M (2.68 GB) and Q8_0 (4.43 GB) with `ncu --set full --kernel-name
w8a16_gemv_kernel --launch-count 3 --target-processes all -- celeg-run
--model ... --raw --prompt "..." --benchmark-decode 8`, three launches each
(one per distinct output width in a decode step, since M=1 collapses
`grid.y` to 1):

| model n (output width) | grid | duration | achieved occupancy | memory (DRAM) throughput | compute (SM) throughput |
|---|---|---|---|---|---|
| ~6144 (wide, e.g. MLP gate/up) | (768,1,1) | 14.9-14.7 &micro;s | 71.8-72.9% | 72.0% | 19.1% |
| ~1024 (narrow, e.g. attn out-proj) | (128,1,1) | 5.2-5.7 &micro;s | 16.6-18.1% | 33.8-34.7% | 8.8-9.1% |

Both model sizes land on effectively the same numbers per grid shape — this
tracks with the "decode rate does not respond to resident weight bytes"
finding above, since `w8a16_gemv_kernel`'s per-launch cost is set by output
width and fixed overhead, not by which quant produced the int8 bytes it
reads. **Gate crossed**: the narrow-matrix launches sit at 17% achieved
occupancy, under the plan's 30% threshold, and `ncu` independently flags them
("grid too small to fill the available resources... only 0.13 full waves").
128 blocks of 8 warps each cannot occupy a 170-SM GPU (`Block Limit
Registers` allows 6 resident blocks/SM = ~1020 block slots per wave).

**Fix attempted and reverted: it made things worse.** The natural low-risk
change, per the gate's own framing ("a grid/tiling change"): drop
`warps_per_block` from 8 to 1, so `grid.x = n` instead of `n/8`, spreading
the same total warps across 8x more, smaller blocks. This is algorithmically
free — `w8a16_gemv_kernel` shares no state across the warps of a block (no
smem, no cross-warp reduction; each warp owns one output row start to
finish), unlike its neighbor `w4a16_linear_kernel`, which does use smem to
amortize the activation-vector load across a block's warps. Measured after
the change: achieved occupancy on the narrow-matrix launches **dropped** to
12.5-12.8% (worse, not better), the wide-matrix launches dropped from
71-73% to 39%, and end-to-end decode throughput did not move
(134.4 -> 134.9 tok/s on 128-token decode runs, within noise). At this
kernel's ~5 &micro;s duration, the fixed per-block dispatch cost of issuing
8x more, smaller blocks outweighs whatever SM-fill benefit more blocks would
otherwise provide — the naive "more blocks = better occupancy" reasoning
does not hold at this timescale. Reverted (`W8A16_WARPS_PER_BLOCK` stays 8,
now defined once in
`include/celeg/backend/cuda/kernels/gemv_kernels.cuh` and shared by the two
call sites that previously hardcoded it separately, closing a drift risk
even though the value itself is unchanged).

**Conclusion: per gate 3's own fallback ("otherwise write up and stop"),
this needs a redesign, not a tiling tweak.** The 5-15 &micro;s/launch,
20000+-launches-per-128-steps profile from the `nsys` breakdown above is
consistent with celeg paying a largely fixed per-kernel-node cost regardless
of occupancy — the real fix is reducing the *number* of GEMV launches per
decode step (e.g. fusing several of a layer's narrow linears — attention
out-proj, and whichever MLP projections share compatible shapes — into one
kernel call with a batched/strided weight layout), not reshaping the grid of
any one of them. That is a real kernel redesign with its own correctness
surface (weight layout, scale handling across fused matrices) and is out of
scope here per the plan's explicit "do not pre-commit to a kernel rewrite."

**Comparison against llama.cpp's actual decode-time kernel.** The revert
above was reasoned from first principles without reading the reference
implementation first — worth doing before any further attempt. llama.cpp's
equivalent of `w8a16_gemv_kernel` is `mul_mat_vec_q` in the vendored
`.externals/llama.cpp/ggml/src/ggml-cuda/mmvq.cu` (dispatch/tuning tables)
and `vecdotq.cuh` (per-type dot products). It differs from celeg's kernel in
three structural ways, not one:

1. **K-split, not row-split.** celeg: one warp computes an entire row's dot
   product alone; different warps in a block handle different, independent
   rows. llama.cpp: `nwarps` warps *cooperate* on the same row, each summing
   a disjoint slice of K (`blocks_per_iter = vdr*nwarps*warp_size/qi`), then
   reduce across warps via shared memory (`tmp_shared`) before one warp
   writes the row (`mmvq.cu:622-690`). This directly cuts a narrow row's
   *latency*, which is what the reverted fix here was trying (and failing)
   to do by changing block *count* instead.
2. **Integer SIMD dot products, not scalar bf16 multiply-add.** celeg reads
   bf16 activations and does per-element `float` multiply-accumulate against
   int8 weights. llama.cpp quantizes the activation vector to `block_q8_1`
   once per step and uses `__dp4a` (4-way int8 SIMD dot-product instruction)
   throughout `vecdotq.cuh` -- e.g. `ggml_cuda_dp4a(vi0, u[2*i], sumi)` at
   `vecdotq.cuh:129`. This is a compute-throughput difference independent of
   occupancy: more useful FLOPs per issued instruction.
3. **Kernel fusion.** `mul_mat_vec_q` takes an optional second weight
   (`vgate`) and GLU op (`has_fusion`, `active_glu`, `mmvq.cu:551-713`) so a
   SwiGLU MLP's gate and up projections are computed by *one* kernel launch
   sharing the same activation load, not two. That is exactly the
   "reducing the number of launches per step" direction named above, already
   shipping in production llama.cpp rather than hypothesized here.
4. `nwarps` itself is not a flat constant like celeg's -- `calc_nwarps`
   (`mmvq.cu:361-465`) is a hand-tuned lookup table keyed on
   architecture (RDNA4/RDNA3/Turing/Ampere+/...) *and* quant type, with
   comments recording specific regressions found by tuning (e.g. "Q3_K,
   IQ2_*, IQ3_* regress due to register pressure... on RDNA4").

None of this is a quick patch: it is a real GEMV kernel replacing an
activation-format decision (bf16 vs. quantized-and-`dp4a`), a reduction
strategy, and a fusion mechanism, tuned per architecture.

**Difference 1 (K-split) was then implemented and measured.**
`w8a16_gemv_ksplit_kernel` (`gemv_kernels.cuh`) is the celeg equivalent of
llama.cpp's cooperating-warps structure: `NWarps` warps share one output
row, each striding over a disjoint slice of K, reducing through shared
memory at the end. `launch_w8a16_linear` selects it for decode (`m == 1`)
whenever the matrix is too narrow to fill the GPU -- specifically when
`n * ksplit_warps` fits inside the device's resident warp capacity, queried
from `cudaDevAttrMultiProcessorCount` and
`cudaDevAttrMaxThreadsPerMultiProcessor` rather than hardcoded.

At the kernel level it does exactly what it was supposed to. `ncu` on a
1024x1024 int8 matrix:

| kernel | achieved occupancy | duration | DRAM throughput |
|---|---|---|---|
| `w8a16_gemv_kernel` | 16.6% | ~4.0 &micro;s | 14.6% |
| `w8a16_gemv_ksplit_kernel<4>` | **46.1%** | **~3.4 &micro;s** | 18.1% |

**End-to-end it is worth about 1%, and only on small models.** Decode
tok/s, `--benchmark-decode 256 --benchmark-warmup 32`, two runs each:

| model | baseline | k-split |
|---|---|---|
| LFM2.5-350M-Q4_K_M | 1022.7 / 1027.2 | 1034.5 / 1035.1 (+0.9%) |
| Nanbeige-3B-Q4_K_M | 132.3 / 135.0 | 132.4 / 133.1 (no change) |

Kept, because it is strictly better where it applies, guarded so it cannot
touch the shapes it would regress (`nwarps` of 8 and 16 measurably hurt
lm_head-sized matrices), and correctness-checked both by an A/B against the
plain kernel inside `celeg-decode-gemv-benchmark` and by the full 91-test
`ctest` run. But the honest summary is that it does not move the number
this investigation set out to move, and it explains why:
**occupancy was never the binding constraint.** Tripling occupancy bought
only 14.6% -> 18.1% of DRAM throughput. A 1 MB matrix read at the card's
1792 GB/s peak would take 0.58 &micro;s, which is the same order as DRAM
latency itself -- there is not enough work in a single narrow GEMV to keep
enough requests in flight, and no arrangement of warps over the same 1 MB
changes that. This is the second occupancy-shaped hypothesis this
investigation has falsified by measurement (the `warps_per_block` 8->1
attempt above being the first).

That leaves difference 3, **fusion**, as the one with real headroom: it is
the only one of the three that makes the matrices bigger rather than
rearranging the work over a matrix whose size is the actual problem. celeg
already does half of this: `CudaModelOptions::fused_projections`
(default on) concatenates a dense FFN's gate and up weights into one `w13`
tensor at load time (`weight_setup.cpp`, `load_concat_linear_weight`) and
computes both with one launch, `linear(normed, *w13, gate_up, 1,
2*intermediate, hidden)` -- so this session's benchmark numbers already
reflect that fusion, not a hypothetical.

**Attention's query+key+value projections were the obvious remaining
target -- tried, and it doesn't help.** Unlike gate/up, Q/K/V are read by
several execution paths (prefill, speculative decode, norm-width lookups),
so concatenating them at load time the way `w13` does would mean carrying
every attention weight twice in device memory. Instead built
`w8a16_gemv_fused_kernel` (mirroring llama.cpp's `has_fusion`/`vgate`
approach in `mmvq.cu` rather than celeg's own `w13` pattern): each matrix
keeps its own storage, and the kernel fuses only the launch, writing
directly into the qkv output buffer's existing q/k/v layout. It is
correct -- bit-identical greedy output with/without it -- and it does
exactly what it was built to do at the kernel level: `ncu` measures 92.8%
occupancy / ~21us for one fused Q+K+V launch (8192 total rows) vs. three
separate launches on Nanbeige-3B (Q already at ~72% occupancy/~14.85us
since 48 query heads make it wide on its own; K and V each ~4us at ~17%
occupancy) totaling ~22.85us. That's a real per-layer win, but only ~2us
out of a ~7.4ms decode step (<0.1%) -- unmeasurable against ~1-2% run-to-run
noise (131-133 tok/s either way). Reverted rather than kept: it adds a new
kernel, struct, option flag and guarded call site for a change this
session could not demonstrate moves the number. The finding that matters
is *why* it doesn't help here: on a GQA model with many query heads, Q
alone is already wide enough to be efficient, so only K/V -- a small
sliver of one layer's work -- were ever in the bad zone. QKV fusion would
matter more on an architecture where Q is *also* narrow (few query heads,
no GQA head multiplier); worth revisiting if one shows up in the sweep.

Difference 2 (`__dp4a` against `q8_1`-quantized activations) remains
unimplemented and is substantial, correctness-sensitive work (activation
requantization, arithmetic changes throughout the dot product) independent
of any fusion question -- it changes how each existing launch computes,
not how many launches there are.

**Benchmark methodology fix, found while doing the above.**
`celeg-decode-gemv-benchmark` was reporting 143% and 228% "of theoretical
peak" -- not suspicious numbers so much as impossible ones. Two causes,
both fixed in `src/app/benchmark/cuda/decode_gemv.cu`: every buffer was
filled with `zero_async` (uniform lines compress, so nominal DRAM traffic
never happens), and, dominating it, each shape was benchmarked by
re-running *one* matrix in a tight loop against a 100.7 MB L2 that every
shape except lm_head fits inside entirely -- so the benchmark measured
cache bandwidth, not the DRAM streaming real decode does. Buffers now hold
high-entropy data, and each shape rotates through enough copies of its
weight matrix (~300 MB) to overflow L2 the way a real decode step does.
Post-fix, no shape exceeds 100% of peak, and the size/efficiency curve
above becomes visible: 14% of peak at 1 MB, 49% at 5 MB, 87% at 67 MB.
Note also that the harness's *wall-clock* per-call figure bottoms out
around 4.1 &micro;s (its own host launch rate) and cannot resolve kernel
improvements below that -- the k-split win above is invisible in its output
and only shows up under `ncu`. Read that column accordingly.

**A genuine, quantified finding surfaced along the way, independent of the
decode-speed question**: `--weight-mode auto` (== `int8`) uses *more* GPU
memory than `--weight-mode bf16`, not less. `--memory-report` on
Nanbeige-3B-Q4_K_M:

| `--weight-mode` | `memory.weights` |
|---|---|
| `native` (packed GGUF) | 3.85 GiB |
| `bf16` | 10.47 GiB |
| `int8` / `auto` | 15.72 GiB |

The mechanism is deliberate, not a leak: `gemm_dispatcher.cpp:268,285`
falls back from the int8 GEMV path to a `cublas`/`cublasLt` **bf16** GEMM
whenever a call's batch size is `m > 1` (i.e. prefill, not decode), because
a real int8 GEMM implementation for the batched case doesn't exist here and
a dense tensor-core bf16 GEMM beats emulating one. To make that fallback
available, `linear_loader.cpp` keeps the full bf16-dequantized copy of
every GGUF linear tensor resident (`DeviceWeight::bf16_storage`, wired into
`Int8LinearStorage::bf16_fallback`) *in addition to* the int8-quantized
copy it actually decodes weights into for `m == 1` decode steps. So `auto`
mode is not "the model at int8" -- it is "the model at bf16, plus a second
int8 copy of the same model, so prefill can use one and decode the other."
This is a real, currently undocumented memory cost of choosing `auto`
(roughly 1.5x a pure-bf16 footprint on this model), separate from and not
explaining the decode-speed flatness above (decode's `w8a16_gemv_kernel`
reads only the int8 copy; the idle bf16 copy costs VRAM capacity, not
decode-step bandwidth, since it is never touched during decode). Whether
that trade is worth it -- versus, say, an int8 batched-GEMM path for
prefill that would let the bf16 copy be freed -- is a real design question
that needs its own profiling of prefill throughput under each option, not
a one-line fix here.

## CPU prefill/decode ratio: measured, partially explained

The plan's follow-up observation: celeg's prefill/decode ratio on
Nanbeige-3B-Q4_K_M is 2.3x (17.1 / 7.5 tok/s), while llama.cpp's is 7.5x
(60.9 / 8.1). If 512-token batching were paying off the way it should,
celeg's ratio should be well above 20x, not below decode's own speed.

**Step 1 (token-count scaling): linear, not sublinear.** Prefill time at
128/256/512 tokens on Q4_K_M: 4.78s / 9.34s / 19.31s, tok/s 26.8 / 27.4 /
26.5 -- flat throughput, perfectly linear time growth. This confirms the
plan's "each token costs a full weight sweep; batching provides no
structural benefit" branch, not "batching works, the floor is elsewhere."
Whatever `gemm_gguf`'s per-call batching buys (the dot4 4-row-at-a-time
path), it isn't compounding across the 512-token batch the way BLAS-style
GEMM batching would.

**Step 2 (`perf record` kernel/attention/quantize attribution): blocked.**
`perf record -g` fails with `perf_event_paranoid setting is 4` -- another
root-only restriction on this machine (`kernel.perf_event_paranoid`
sysctl), same class of blocker as `ncu`'s `ERR_NVGPUCTRPERM` above. Not
attempted to bypass.

**Step 3 (thread scaling, 1/4/8/16/32 threads, 64-token prefill,
Q4_K_M):**

| threads | tok/s | speedup vs 1 | efficiency |
|---|---|---|---|
| 1 | 2.21 | 1.0x | 100% |
| 4 | 8.27 | 3.75x | 94% |
| 8 | 15.39 | 6.97x | 87% |
| 16 | 19.73 | 8.93x | 56% |
| 32 | 25.67 | 11.62x | 36% |

No cliff, no collapse -- scaling is smooth and monotonically increasing
through 32 threads, just increasingly sublinear. That rules out the
sharpest version of the plan's `grain`-clamping hypothesis (`linear.cpp:407`,
`grain = tiles / (pool_size * 4)`, clamped to a floor of 1): if the whole
prefill were dominated by 32 threads thrashing over a `grain`-1 work queue,
this curve would flatten hard well before 32 threads, not still be gaining
17% from 16->32.

The mechanism is real but partial, not the whole floor. This machine
(`nproc`=32) is a hybrid 8P+16E-core part (i9-14900K): 8 P-cores x 2
threads (SMT) + 16 E-cores x 1 thread = 32 logical threads, but only 24
physical cores and two different per-core throughputs. Diminishing returns
starting around 16 threads is consistent with running out of P-core
capacity and spreading onto slower E-cores plus SMT contention -- a
hardware-topology effect independent of any celeg scheduling bug.

The grain formula does clamp to 1 for this model's k/v projections
specifically: `hidden_size=3072`, `intermediate_size=10752`,
`num_key_value_heads=8`, `head_dim=64`, so k_proj/v_proj output rows =
8*64=512, giving `tiles = 512/16 = 32`. At `pool_size` >= 8,
`pool_size*4` >= 32 >= tiles, so grain clamps to 1 for k/v projections at
every thread count this sweep used above 4. q/o/gate/up/down all have
>=3072 output rows (tiles >= 192), well clear of the clamp. Since k/v
projections are a small fraction of total per-layer FLOPs (512 vs.
3072+3072+10752+10752+3072 = 30720 for the other five matrices in this
model), this alone can't be the dominant floor, but it is a real,
verified-in-code contributor to the high-thread-count falloff and is worth
fixing opportunistically (e.g. a per-projection grain floor keyed to
absolute tile count, not just `pool_size`) whenever someone is next in
`linear.cpp`'s parallel_for.

**Step 2, revisited: `perf record` (root access granted mid-investigation).**
`kernel.perf_event_paranoid` was lowered from 4 to 1 (the user ran the
`sysctl -w` themselves after this doc's first pass recorded it as
root-blocked), which unblocked both `perf record` and `nsys` CPU sampling.
A 512-token prefill on Q4_K_M, 32 threads, `perf record -g`, resolved down
to a flat profile by self time:

| symbol | self time |
|---|---|
| `cpu_gguf_dot4_avx2` | 67.1% |
| `cpu_gguf_dot_avx2` (scalar, per-lane) | 25.8% |
| `update_online_avx2` (attention) | 3.6% |
| `__expf_fma` | 1.1% |
| `gemm_gguf`'s parallel_for dispatch lambda | 0.7% |
| `cpu_quantize_q8k_avx2` (activation requantization) | 0.07% |

This retires the "per-GEMM-call activation requantization" hypothesis
outright -- `cpu_quantize_q8k_avx2` is noise at 0.07%, not a hidden floor.
Threadpool dispatch overhead is likewise negligible (0.7%). **92.9% of
prefill time is the dot-product kernels themselves**, split 67.1% batched
(`dot4`) vs. 25.8% unbatched (`dot`, four separate scalar-ish calls per
4-row group). That 25.8% is disproportionate: `scripts/gguf_census.py` on
this exact file shows Q5_K is only 11.6% of weight elements (Q4_K 63.0%,
Q6_K 25.4%, Q5_K 11.6%). Cross-referencing against
`cpu_gguf_dot4_avx2`'s source (`gguf_avx2.cpp`) explains why: **the
function batches Q4_K and Q6_K but not Q5_K** -- Q5_K silently falls
through to four unbatched `cpu_gguf_dot_avx2` calls, the same fallback
used for every other native-dot type. 11.6% of elements consuming 25.8%
of cycles (a ~2.7x worse cycles-per-element ratio than the 88.4% of
elements running through real `dot4`) is exactly the dot4-coverage gap
the plan's original W5 hypothesis named -- the earlier rejection of that
hypothesis (full-sweep ratios showing Q5_K and Q4_K indistinguishable at
0.40x/0.44x) was confounded by comparing against llama.cpp's absolute
speed, which has its own advantages unrelated to this gap; it was never a
statement that dot4-vs-scalar doesn't matter for celeg's own throughput.

**Step 4, implemented.** Added a real `dot4` path for Q5_K in
`cpu_gguf_dot4_avx2` (`gguf_avx2.cpp`), mirroring the Q4_K/Q6_K structure:
unpack each 256-wide sub-block's 5-bit weights (4-bit nibble from `qs` +
1 high bit from `qh`, matching the existing scalar `Q5_K` branch bit for
bit) once per sub-block, then reuse the unpacked values across the 4
activation lanes via `_mm256_maddubs_epi16`/`_mm256_madd_epi16`,
accumulating per-lane `int32` totals exactly as the Q4_K branch does.
Covered by `tests/cpu_gguf_kernels_test.cpp`'s existing dot4
self-consistency check (`dot4(...)` output compared against four
independent `cpu_gguf_dot_avx2` calls, now run for Q4_K/Q6_K/Q5_K), with
a Q5_K test row that exercises a nonzero `qh` pattern -- the prior Q5_K
fixtures in this file all used an all-zero `qh`, which would have let a
high-bit-unpacking bug in the new code pass silently. `ctest` 81/81 still
passes.

**Measured effect** (Nanbeige-3B, 512-token prefill, 32 threads,
`celeg-bench -p 512 -n 0`):

| file | before | after | change |
|---|---|---|---|
| Q4_K_M (63% Q4_K / 25% Q6_K / 12% Q5_K by element) | 26.29 tok/s | 30.43 tok/s | +15.7% |
| Q5_K_M (pure Q5_K linear weights) | -- | 26.60 tok/s | new dot4 coverage |

The other six native-dot types without a `dot4` branch (Q4_0, Q4_1, Q5_0,
Q8_0, Q2_K, Q3_K) remain on the unbatched path. Per this same profile
their combined share of a typical GGUF file's weight elements is small
next to the K-quants (none of the 40 cached sweep files use them as the
majority format), so they were not extended in this pass; the pattern
established here (unpack once per sub-block, reuse across 4 lanes) is
mechanical to repeat for any of them if a file where they dominate shows
up.

## Native weight mode: measured, decision made

`--weight-mode native` keeps GGUF blocks packed on-device and runs them
through `mmq.cu`'s per-type MMQ kernels, instead of the host-dequant ->
int8/bf16 path every other mode uses. It only exists for Q4_K and Q6_K
today (the only two types with an MMQ kernel), which is why the matrix
above carries 12 `not planned*` cells in the CUDA-native-MMQ column.

**Measured on Nanbeige-3B-Q4_K_M** (`--benchmark-prefill-tokens 512
--benchmark-decode 8`):

| `--weight-mode` | prefill tok/s | decode tok/s |
|---|---|---|
| `native` | 637.1 | 96.7 |
| `auto` (int8) | 3089.6 | 132.1 |

Native is **4.85x slower at prefill** and **27% slower at decode** than the
int8 path, on the only two types it supports. `ncu` occupancy profiling of
`q4k_mmq_prefill_kernel` against the int8 GEMM was not run: the decision
below doesn't need it (4.85x is decisive on its own), and an `nsys`
kernel-breakdown attempt (the fallback used successfully for the GPU-decode
investigation) hung indefinitely under `--benchmark-prefill-tokens 512` in
native mode and was killed after two timeouts rather than debugged further.
`ERR_NVGPUCTRPERM` was lifted later in this session (see the GPU decode
section's gate 3 writeup) but that hang is unrelated to counter permissions
and remains unexplained -- worth a look with `ncu`/`nsys` now that counters
work, if native-mode MMQ expansion is ever revisited.

**Decision (plan's gate): native does not come anywhere close to the ~1.2x
threshold that would justify expanding MMQ coverage.** A 4.85x prefill gap
on the two best-supported types is not a shape this session's evidence
suggests closing with more kernels for more types -- if anything it argues
the existing two MMQ kernels have a shared structural problem. The 12
`not planned` cells in the matrix reflect this: MMQ kernel expansion is
not queued as future work unless someone first closes the native-vs-int8
gap on the two existing types, at which point it becomes a well-motivated,
separately-scoped follow-up rather than blind coverage expansion.

**The suspected structural problem is now confirmed, in a later session,
and it kills a separate idea too (see "Is `dp4a` worth building" below):**
`launch_quantize_q8_1` (`gemm_dispatcher.cpp:232`) fires far more than the
`NativeFanoutScope` cache's design intent suggests. `ncu` with
`--print-summary per-kernel` over one prefill+decode step of
Nanbeige-3B-Q4_K_M under `--weight-mode native`:

| kernel | invocations | avg duration |
|---|---|---|
| `quantize_q8_1_kernel` | 271 | 2.3-2.5 &micro;s |
| `q4k_mmq_kernel` | 162 | (decode-shaped launches) |
| `q6k_mmq_kernel` | 66 | (decode-shaped launches) |

Quantization fires nearly as often as the dot-product kernels combined
(271 vs. 228). This isn't a caching bug -- `native_fanout_scope` already
groups calls that share one activation (e.g. attention's Q/K/V), and 271
is roughly what a 44-layer model's count of *genuinely distinct*
intermediate activations (normed input, MLP-activated intermediate, etc.
per layer) would produce even with perfect caching. It's a structural cost
of quantizing activations at all in this architecture, not a fixable
inefficiency in how it's wired.

## Is `dp4a` worth building for GPU decode: investigated, stopped at the gate

llama.cpp's decode-time GEMV (`mmvq.cu`) differs from celeg's
`w8a16_gemv_kernel` in three ways (see the GPU decode section above for
the full comparison): K-split (tried, kept, small win), kernel fusion
(tried, reverted, no measurable win), and `__dp4a` int8 SIMD dot products
against quantized (`q8_1`) activations instead of celeg's scalar
bf16-float multiply-add against a bf16 activation. The third was untried
going into this investigation and is the one that changes arithmetic
rather than kernel shape or launch count.

**Not a green-field question**: celeg already has a full `dp4a` pipeline,
just applied to the native GGUF MMQ path (`src/backend/cuda/kernels/mmq.cu`)
-- `quantize_q8_1_kernel` plus `q4k_mmq_kernel`/`q6k_mmq_kernel`, using the
exact same `(n/8, 1)`-block, 8-warps-per-block GEMV grid as
`w8a16_gemv_kernel` when `m == 1`. And native mode is already measured 27%
*slower* at decode than the scalar path (previous section), despite using
`dp4a` throughout. That number alone doesn't isolate the variable, though
-- Q4_K/Q6_K need real per-superblock unpacking a plain int8 row wouldn't,
so a gated, two-step plan was written (full text in the plan file) rather
than porting `dp4a` to a new kernel blind:

1. **Profile the existing native-decode path with `ncu`** to see whether
   K-quant unpacking or the quantization pipeline itself explains the 27%,
   before writing any new kernel.
2. Only if step 1 doesn't already answer it: build an isolated
   plain-int8-weight `dp4a` GEMV in `celeg-decode-gemv-benchmark`,
   including the quantization kernel's cost, and gate on whether it beats
   the scalar path by a real margin.

**Step 1 answered it.** The invocation-count table in the previous section
is the result: `quantize_q8_1_kernel` fires 271 times against 228
dot-product-kernel calls for one prefill+decode step -- essentially one
new ~2.3-2.5&micro;s launch per distinct activation the model computes, not
a rare amortized cost. That figure alone is larger than the *entire*
measured saving from tripling a comparable GEMV's occupancy via k-split
earlier in this investigation (4.0 -> 3.4&micro;s, ~0.6&micro;s). On top of
that, `ncu`'s `SpeedOfLight` section shows these decode-shaped kernels
running at 0.7-2.3% Compute (SM) Throughput -- they were never
compute-bound, so a denser arithmetic format has little headroom to
exploit even before paying the quantization tax. Both facts hold
regardless of whether the weight is K-quant or plain int8, since
activation quantization is required input preparation either way -- they
are not confounds specific to Q4_K/Q6_K's unpacking cost.

**Stopped here rather than building step 2's isolated kernel.** The
evidence already indicates a plain-int8 `dp4a` GEMV would pay the same
~2.3&micro;s/shape quantization tax while chasing a compute-bound speedup
these particular kernels have no room to give, which is a structural
explanation for the already-observed 27% native-mode slowdown, not a
coincidence of K-quant complexity. Building it anyway to confirm would
very likely reproduce a third correct-but-unmeasurable-or-negative result
in this investigation (after k-split's neutral-on-large-models outcome and
QKV fusion's reversion) for a predictable reason rather than a new one.
**If this is ever revisited**, the lever with headroom is not the dot
product's arithmetic but the quantization *count* -- e.g. restructuring
decode to quantize far fewer, larger activation batches, which is a
different and larger redesign than swapping one kernel's math.
