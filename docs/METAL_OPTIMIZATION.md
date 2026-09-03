# Metal optimization on Apple M5

## Status and scope

The performance target is Apple M5. Other Apple Silicon devices use the same
capability checks and retain a correct fallback, but they do not inherit an M5
performance claim. This work does not introduce a checkpoint format or a
private repacking step: all kernels consume the eight existing GGUF variants
directly.

The original `benchmarks/results/metal_llama_cpp_compare.json` is preserved as
diagnostic evidence, but is **not eligible for promotion**. Its Celeg executable
was stale, the optional TensorOps source did not compile, and every prefill GEMM
silently used the scalar kernels. The repository HEAD written in that JSON was
not proof of the code embedded in the executable.

A rebuilt short A/B established the useful starting point. Q4_K_M reached
`1.09x` llama.cpp at pp32; the remaining ratios were approximately `0.73–0.77x`
for Q4 decode, `0.67–0.69x` for Q8 prefill, and `0.81–0.83x` for Q8 decode.
Only results produced by the hardened preflight described below are eligible.

## Numerical policies

`MetalNumericalPolicy::Strict` is the default in both C++ and the C API.
`MetalNumericalPolicy::Fast` explicitly enables specialized TensorOps, the
Fast SwiGLU path, and the tiled attention path. Reduced precision is enabled
only for the quantized families that satisfy their numerical gate; dense and
Q6 keep strict accumulation with their specialized prompt tiles. The former
environment-variable opt-in was removed rather than retained as a
compatibility alias.

Tensor source compilation is isolated into these libraries:

- Strict TensorOps core;
- Fast dense F16/BF16;
- Fast Q4_0;
- Fast Q4_K;
- Fast Q5_K;
- Fast Q6_K;
- Fast Q8_0.

The backend description reports the requested and effective policy, every
family capability, and family-specific compiler diagnostics. A failed Fast
family falls back to the corresponding correct Strict path for ordinary
inference. The official benchmark rejects that fallback for the model under
test.

## Benchmark preflight and provenance

`benchmarks/compare_metal.py` performs a profiled prefill before timing each
cell. A Fast cell is rejected when any of these conditions is true:

- the requested or effective policy is not Fast;
- the active model emits a scalar GEMM;
- no TensorOps GEMM is dispatched;
- the active model dispatches Strict rather than Fast TensorOps.

The JSON records the commit embedded at build time, dirty state, executable
SHA-256, aggregate Metal-source SHA-256, compiler, SDK, macOS product/build,
device description, model path/size/SHA-256, llama.cpp commit and executable
SHA-256, and the preflight dispatch histogram. Repository HEAD is never used as
a substitute for embedded Celeg provenance.

The test-only `CELEG_METAL_TENSOR_FAST_FAIL_FAMILY` injection makes one family
fail compilation. It verifies both sides of the contract: unrelated families
remain usable, while the official preflight rejects the affected model's
fallback.

## Kernel architecture

Prefill uses cooperative `matmul2d` tensors. Dense weights are read directly
from device memory. GGUF kernels decode complete 32-value sub-blocks into a
threadgroup tile, reuse each staged tile across the output cooperative tensor,
and accumulate over K. Full and edge extents share the same correctness path;
Fast has explicit N32 entry points so a short prompt does not carry an N128
cooperative destination.

The kernel benchmark covers the actual LFM2.5-350M projection, FFN-up,
FFN-down, and language-head shapes. It reports GPU time, effective weight
traffic, threadgroup geometry, and percentage of a measured copy roofline. Its
prefill sweep includes `64x32`, `64x64`, `64x128`, `128x128`, and `128x256`,
each with BK64 and BK128. It also includes a `128x256` control derived from the
structure of the pinned llama.cpp MPP kernel.

Decode retains format-native GGUF matvecs. Rows-per-thread candidates are
benchmarked independently before routing. Q4_K/Q5_K use the rows8 expansion
candidate where the isolated kernel wins; Q6 remains on rows4 because the
Fast generation-quality gate rejected its candidate. Q8 uses rows8 for the
measured FFN shapes. Fused Q4_0 and Q8 FFN-down kernels consume gate/up values,
apply SwiGLU, and perform the matvec in one dispatch. Q8 therefore drops from
165 to 149 dispatches per decoded token.

## Experiment log

| Candidate | Isolated result | End-to-end or quality result | Decision |
|---|---:|---:|---|
| Explicit N32 Fast TensorOps | Q8 pp32 about `+51%` in the storage profile | all Fast families compile and dispatch N32 | promoted |
| Q8 fused SwiGLU + matvec | 16 intermediate SwiGLU dispatches removed | 165 to 149 dispatches/token; about `+8%` in the adjacent decode sample | promoted |
| Q4_0 fused SwiGLU + matvec | max error `4.1e-8` | removes the separate decode SwiGLU dispatch | promoted |
| rows8 Q6 matvec with relaxed prefill | up to `2.17x` on isolated FFN-up | deterministic token agreement `0.625`, below `0.97` | rejected from routing; Q6 Fast uses strict accumulation and reaches `1.0` agreement |
| BK128 / 64-token quantized tile | up to about `9–14%` on selected Strict isolated shapes | Fast pp512 regressed about `3–6%` for Q4/Q5/Q8 | benchmark-only candidate |
| 128x256 control | structurally close to the pinned competitor geometry | about `3.6–4.8x` slower than Celeg's 64x128 control | rejected |

Promotion of another candidate requires at least `3%` improvement in its
target phase and no more than `1%` regression in the complete workload. The
runtime uses a static choice; it never autotunes during inference.

## Numerical gates

Strict results remain the reference. Fast is compared against Strict and CPU
at token counts `1, 2, 8, 15, 16, 31, 32, 33, 127, 128, 129, 512`, including
non-aligned output extents and strides.

| Family | Cosine | RMSE | Maximum error | Top-k | Generated-token agreement |
|---|---:|---:|---:|---:|---:|
| BF16/F16 | `>=0.999` | `<=0.005` | `<=0.02` | `1.0` | `>=90%` |
| Q4/QAD-Q4 | `>=0.998` | `<=0.1` | `<=0.5` | `1.0` | `>=90%` |
| Q5/Q6 | `>=0.9995` | `<=0.03` | `<=0.2` | `1.0` | `>=97%` |
| Q8 | `>=0.999` | `<=0.05` | `<=0.25` | `1.0` | `>=95%` |

No format is promoted if any relevant numerical check fails. Dense and Q6
initially passed the scalar similarity checks poorly or diverged in generated
tokens under reduced precision; their Fast kernels therefore retain strict
accumulation. The resulting eight-format run meets the listed Fast-vs-Strict
thresholds, including `1.0` generated-token agreement.

## Official M5 protocol and promotion gate

The official matrix is the eight GGUFs crossed with `32+8` and `512+8` token
workloads, shared storage, five warmups, 15 timed repetitions, and alternating
Celeg/llama.cpp order. Each timed subprocess performs one additional untimed
in-process preheat so lazy pipeline setup and first-dispatch residency are not
charged to either engine. The fixed llama.cpp revision is
`d7bd3bfcad3e29c7e49fd26f38c79ee3e9a3fd6b`.

Promotion requires all of the following:

- median Celeg prefill and decode at least llama.cpp in every one of 16 cells;
- median and first quartile of combined throughput at least llama.cpp in every
  cell;
- geometric mean of the 16 combined median speedups at least `1.10x`;
- no scalar fallback, TensorOps family failure, or numerical regression.

The output is `benchmarks/results/metal_llama_cpp_compare_official.json` so the
historical invalid diagnostic is never overwritten.

### Current official baseline

The corrected run on macOS `26.6.2` (build `25G83`), SDK `26.5`, Apple M5,
produced a combined geometric-mean speedup of `0.819x`. All 16 preflights were
valid, but the promotion gate is false.

| Format | Workload | Prefill speedup | Decode speedup | Combined speedup |
|---|---|---:|---:|---:|
| BF16 | 32+8 | `0.485x` | `0.942x` | `0.795x` |
| BF16 | 512+8 | `0.364x` | `0.895x` | `0.530x` |
| F16 | 32+8 | `0.476x` | `0.935x` | `0.788x` |
| F16 | 512+8 | `0.354x` | `0.880x` | `0.519x` |
| Q4_0 | 32+8 | `1.639x` | `0.971x` | `1.129x` |
| Q4_0 | 512+8 | `0.976x` | `0.879x` | `0.941x` |
| Q4_K_M | 32+8 | `1.519x` | `0.785x` | `0.945x` |
| Q4_K_M | 512+8 | `0.824x` | `0.732x` | `0.788x` |
| Q5_K_M | 32+8 | `1.421x` | `0.781x` | `0.924x` |
| Q5_K_M | 512+8 | `0.814x` | `0.730x` | `0.784x` |
| Q6_K | 32+8 | `0.944x` | `0.972x` | `0.959x` |
| Q6_K | 512+8 | `0.430x` | `0.887x` | `0.528x` |
| Q8_0 | 32+8 | `1.592x` | `0.787x` | `0.913x` |
| Q8_0 | 512+8 | `0.963x` | `0.729x` | `0.853x` |
| QAD-Q4_0 | 32+8 | `1.627x` | `0.952x` | `1.108x` |
| QAD-Q4_0 | 512+8 | `0.982x` | `0.867x` | `0.941x` |

The next work is therefore narrow and measurable: dense/Q6 prefill needs a
quality-preserving TensorOps improvement; Q4_K/Q5_K/Q8 decode needs higher
effective matvec bandwidth; and Q4_0/QAD decode needs the remaining dispatch
and kernel gap closed. No current result is labelled promoted.

## Primary references

- [Metal Performance Primitives Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
- [WWDC26: Metal tensors](https://developer.apple.com/videos/play/wwdc2026/330/)
- [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [Pinned llama.cpp `mul_mm.metal`](https://github.com/ggml-org/llama.cpp/blob/d7bd3bfcad3e29c7e49fd26f38c79ee3e9a3fd6b/ggml/src/ggml-metal/kernels/mul_mm.metal)

The pinned competitor source stages quantized A tiles in threadgroup memory,
keeps the other operand in device memory, uses function constants for layout
conditions, and accumulates through a cooperative MPP tensor. Those are the
comparison points used by the in-tree control; Celeg does not copy or vendor
the upstream source.
