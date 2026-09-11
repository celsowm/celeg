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
for the quantized families that satisfy their numerical gate and for the
dense F16/BF16 prefill path (N32 tile at or below 32 rows, K32 tile above),
which buys roughly `2x` prefill throughput at `pp16` on M5 (`44.6 ms` strict
versus `21.8 ms` fast) while keeping greedy top-1 and generated-token
agreement at `1.0`. On M5, Q6 uses reduced precision only for the FFN gate in
layers 0–7; every other Q6 projection remains Strict. The former
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

`benchmarks/compare_metal.py` performs an undistorted count profile before timing each
cell. A Fast cell is rejected when any of these conditions is true:

- the requested or effective policy is not Fast;
- the active model emits a scalar GEMM;
- no TensorOps GEMM is dispatched;
- the active model dispatches no Fast TensorOps.

Q6 always retains the fast TensorOps geometry. Its strict-precision kernel is
the performance floor, while all eight FFN-down projections use relaxed
accumulation on M5 at every row count (N32 entry below 33 rows, full tile
above). The backend reports this as
`fast_q6k_precision=selective_m5_ffn_down`; a missing Fast Q6 library still
falls to the generic Strict family and is rejected by preflight. (The model
under test carries no Q6_K gate tensors — `gguf_census.py --all` lists only
`token_embd`, eight `ffn_down`, and two `attn_v` — so the earlier gate-0–7
clause never matched.)

The JSON records the commit embedded at build time, dirty state, executable
SHA-256, aggregate Metal-source SHA-256, compiler, SDK, macOS product/build,
device description, model path/size/SHA-256, llama.cpp commit and executable
SHA-256, and the preflight dispatch histogram. Repository HEAD is never used as
a substitute for embedded Celeg provenance.

Profiling is an explicit benchmark interface:

```bash
celeg-metal-bench ... --profile-dispatches counts
celeg-metal-bench ... --profile-dispatches gpu-stage
```

`counts` preserves the production encoder schedule and is the only mode used
by promotion preflight. `gpu-stage` uses one compute encoder per dispatch and
stage-boundary timestamp samples, which are supported by M5. Its JSON reports
count, total GPU time, median, p95, phase percentage, sampled sum, normal
command-buffer GPU time, and their distortion. It is diagnostic only. The old
GPU/count environment variables were removed without aliases.

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
traffic, threadgroup geometry, and percentages of separate hot-cache and
streaming copy rooflines. The streaming buffer is 512 MiB and therefore larger
than the system cache; the hot-cache buffer is 8 MiB and dispatched repeatedly. Its
prefill sweep includes `64x32`, `64x64`, `64x128`, `128x128`, and `128x256`,
each with BK64 and BK128. It also includes a `128x256` control derived from the
structure of the pinned llama.cpp MPP kernel.

Decode retains format-native GGUF matvecs. Rows-per-thread candidates are
benchmarked independently before routing. Q4_K/Q5_K use the rows8 expansion
candidate where the isolated kernel wins; Q6 on M5 now uses the llama
`mul_vec_q6_K` geometry (`NR0=2/NSG=2`, two rows per simdgroup, two simdgroups
per threadgroup, four partials) ported as `celeg_matvec_q6k_llama`, winning
`1.32x` on the isolated ffn_down duels and `+4.0%` end-to-end at `32+8`
(`+1.6%` at `512+8`). Q8 on M5 instead matches the pinned competitor geometry:
`nr0=2`, four SIMD groups, eight adjacent activations per lane, and a
threadgroup reduction across K. Its fused FFN-down entry point uses the same
geometry. Other Apple Silicon retains the prior capability-selected rows4/rows8
paths.

## Experiment log

| Candidate | Isolated result | End-to-end or quality result | Decision |
|---|---:|---:|---|
| Explicit N32 Fast TensorOps | Q8 pp32 about `+51%` in the storage profile | all Fast families compile and dispatch N32 | promoted |
| Q8 fused SwiGLU + matvec | 16 intermediate SwiGLU dispatches removed | 165 to 149 dispatches/token; about `+8%` in the adjacent decode sample | promoted |
| Q4_0 fused SwiGLU + matvec | max error `4.1e-8` | removes the separate decode SwiGLU dispatch | promoted |
| rows8 Q6 matvec with relaxed prefill | up to `2.17x` on isolated FFN-up | deterministic token agreement `0.625`, below `0.97` | rejected from routing; Q6 Fast uses strict accumulation and reaches `1.0` agreement |
| BK128 / 64-token quantized tile | up to about `9–14%` on selected Strict isolated shapes | Fast pp512 regressed about `3–6%` for Q4/Q5/Q8 | benchmark-only candidate |
| 128x256 control | structurally close to the pinned competitor geometry | about `3.6–4.8x` slower than Celeg's 64x128 control | rejected |
| Two-cooperative-tensor segmented accumulation | avoids an intermediate buffer | BF16/Q6 pp512 rose to about `200 ms` from register pressure | rejected |
| Materialized relaxed accumulation, K512 | BF16 `120.18→97.24 ms`; Q6 `105.79→64.98 ms` | BF16 RMSE `0.00816` exceeds `0.005`; Q6 still lacked target margin | rejected |
| Selective Q6 relaxed FFN gate, layers 0–7 | Q6 pp512 `105.79→96.62 ms` (`+9.5%`) | all N1…N512 gates pass; token agreement `1.0`, but the family remains below the `1.05x` target | routed incremental gain; final gate pending |
| M5 Q8 `nr0=2/nsg=4` | interleaved Q8 decode/8 pooled median `39.42→33.94 ms` (`+13.9%`) | 30 samples per side; all Q8 gates pass; token agreement `1.0`; still above the `28.74 ms` target | routed incremental gain; final gate pending |
| Gate+up single-dispatch prototype | removes 16 dispatches/token | Q4/Q5/Q8 decode regressed `10–14%` | rejected |
| Explicit `float4`/`uchar4` loads | fewer source-level loads | Q4/Q5/Q8 decode regressed `9–15%` | rejected |
| Q6 `nr0=2/nsg=2` (generic) | matches pinned competitor launch geometry | mixed Q4/Q5 decode regressed about `8%` with the early generic stub | rejected (superseded) |
| Q6K llama-geometry decode (`celeg_matvec_q6k_llama`, port of `mul_vec_q6_K`) | isolated duel `1.32x` ffn_down (`0.071→0.053`), `1.27x` attn, `1.13x` lm_head; `maxabs` within gate | interleaved A/B (4×5 reps) `32+8` decode `389.8→405.4` (`+4.0%` median), `512+8` `348.1→353.7` (`+1.6%`); official interactive `1.158→1.201x` combined, decode `1.030→1.088x`; throughput combined `1.004→1.013x`, decode `0.889→0.913x` | routed |
| Concurrent decode encoder (`MTLDispatchTypeConcurrent` + QKV/gate-up parallel groups + inter-dispatch barriers) | about `+3%` GPU (`22.9` vs `23.8 ms` A/B on pre-rewrite kernels) | `metal_inference_test` green after entry-barrier fix | routed |
| Q4_K llama-style `N_R0=2/NSG=2` core (`>>8` preshift, `[2]` unshift, single-count min terms) | isolated duel `1.65x` over old rows8 core; decode `+22%` end-to-end (`27→21.6 ms`) | Q4K max error `2.4e-07`, token agreement `1.0` | promoted |
| Fused Q4_K QKV decode projection | 18 dispatches to 6 per token; quiet-machine interleaved A/B 4+4: `-1.4%` tok/s, `+1.3%` GPU, within noise | fusion forfeits the concurrent overlap the separate dispatches already exploit | rejected, reverted |
| Fused Q4_K gate/up decode projection (llama-mapped core) | interleaved A/B 4+4: `+0.9%` tok/s, `+0.5%` GPU, within noise; fused gate/up GPU time at least the separate pair | matches the prior gate+up regression history | rejected, reverted |
| Fused Q4_K norm prologue + residual-add epilogue (FFN norm) | interleaved A/B 4+4: `-2.2%` tok/s, `+3.6%` GPU; per-dispatch gate `50 µs` vs `32 µs` plain | redundant per-threadgroup norm reduction + register pressure exceed the deleted-launch saving; third fusion attempt to net zero-or-negative | rejected, reverted |
| Small-shape Q4_K geometry duel (4/8 rows x 64/32 threads on KV, O, mixer, gate) | all variants bitwise-identical; shapes at `124–175 GB/s` (at/above roof); best-vs-base deltas `≤1–2 µs` per dispatch, ~`1%` end-to-end even if routed | the sub-roof impression was attribution noise; launch-dominated KV shapes cannot be fixed by geometry | rejected, not routed |
| Parallel QK-norm/RoPE decode kernel (1 thread per head-pair, `simd_sum` inverse, 32-thread groups) | qk row `0.188→0.020 ms`; interleaved A/B `+6%` wall; official decode `0.920→0.951x` (32+8), `0.820→0.860x` (512+8) | per-element op order preserved (trig bitwise); only norm sum reorders; inference cosine unchanged, agreement `1.0` | routed |
| Serial decode encoder by default (concurrent opt-in) | concurrent barriers cost `~65 µs`/token CPU with GPU tied; serial wins wall `~3%` (369.8 vs 360.0 tok/s), prefill `+2%` pp32, pp512 tied | overlap gain vanished as kernels got faster; concurrent path kept behind `CELEG_METAL_CONCURRENT_DECODE=1` for slower kernels | routed |
| Encode CPU diet (pipeline memo per Linear + cached M5 check) | encode `1.30→0.51 ms`/8tok (`-62%`); interleaved A/B wall `+8%`; official decode `0.986→1.085x`, combined `1.090→1.146x` | memo sound: kernel is a pure function of weight dims, path-tagged; inference cosine unchanged, agreement `1.0` | routed |
| Q6_K relaxed down prefill (selective M5 policy on all FFN-down) | strict down `1.63 ms` vs relaxed `0.57 ms` same shape; pp512 `9564→10832 tok/s` (`+13%`); official prefill `0.822→0.963x`, throughput combined `0.844→0.953x` | all 8 down layers pass Q5/Q6 gates (fast RMSE `0.0074` vs `0.03`, max `0.035` vs `0.2`, agreement `1.0`); decode untouched (matvec path) | routed |
| Q6_K relaxed rows-gate removal (`rows>=128` dropped) | pp64 prefill `4176→5132 tok/s` (`+23%`, `15.16→12.47 ms`); pp512 tied (`11249→11323`); interleaved pp64 ratio `0.749→0.921x` prefill, combined `0.979x` | boundary prefixes 32/33/127/128 green (rows=32: fast RMSE `0.014` vs `0.03`, max `0.067` vs `0.2`, top-1 `11857=11857`); rows≤32 take the relaxed N32 entry | routed |
| Q6_K selective gate clause removal (dead code) | gate-all A/B neutral (`-2.4%` pp64 / `-0.3%` pp512, noise); census proves the file has zero Q6_K gate tensors | predicate collapsed to `FfnDown`-only with identical behavior; precision string `selective_m5_ffn_down` | reverted to down-only |
| Q4_K relaxed GEMM duel vs llama `mul_mm_q4_K_f32` (N=512, F32, real weights) | ours faster on all shapes (gate `0.636` vs `0.690`, up `0.637` vs `0.674`, down `0.593` vs `0.689` ms; weighted `1.089x`); cross max-abs `~1e-3`, both match CPU reference | no deficit to fix on the Q4_K path; gap must be elsewhere | rejected, not routed |
| Q6_K relaxed GEMM duel vs llama `mul_mm_q6_K_f32` (N=512, F32, real weights) | llama faster (down `0.588` vs `0.638`, attn_v `0.067` vs `0.086` ms; weighted `0.917x`); outputs **bitwise identical** (`maxabs 0.00e+00`), both match CPU reference | identical values, different schedule cost; K32-tile hypothesis tested next | investigated |
| Q6_K K32-tile hypothesis (our template at TileK=32) | ours-K32 `18%` **slower** than ours-K64 (`1.18x`), opposite of llama-K32 | tile size is not the differentiator; per-value dequant/schedule is | rejected |
| Cooperative zero-init ablation (drop `get_capacity` zero loop) | nozero `0.589` vs llama `0.587` ms down, `0.068` vs `0.063` attn_v — gap fully closed; pp512 `11309→12478` tok/s (`+10.3%`), pp64 `5023→5471` (`+8.9%`) | inference bit-identical (fast RMSE `0.00735229` unchanged); MPP destination arrives accumulator-ready, matching llama's zero-free schedule | routed |
| Remaining cooperative zero-init removals (`tensor.metal`, `tensor_dense_relaxed.metal`, `tensor_q4k_static_stage128.metal`) | same MPP accumulator-ready guarantee; duel already proved schedule win; no additional prefill win on Fast Q4_K_M (relaxed path already nozero), but cleans strict/dense paths | inference green `fast_cosine 0.999997` / `0.999999`, agreement `1.0`; throughput `1.013→1.019x` / `1.201→1.190x` within noise; no regression | routed |
| Decode attention one-exp (1 vs 2 `exp` per position) | bit-exact (branch `score>max ? exp(max-score) : exp(score-max)`); `512`-prompt decode `0.409→0.413 ms` (+1%), `32`-prompt `0.089→0.086 ms` (−4%), both <3% | saves one `exp` but adds branch; net neutral on M5, not routed outside the `attention_one_exp` harness | rejected |
| Decode attention unrolled ×4 | same arithmetic, 4 positions per outer iteration; `512`-prompt `0.413→0.509 ms` (+23%) | register pressure + I-cache exceed loop-overhead saving; matches prior 4× unroll regressions | rejected |
| Decode FA 8-key block (scalar, 8 SGs×8 keys/iter, 64 keys/iter) | `512`-prompt `0.413→0.463 ms` (+12%), `32`-prompt `0.089→0.086 ms`; bit-exact `fast_cosine 0.999999` but 8× `scores[8]` regs + `block_max`/`block_sum` vs 1-key | 8× `simd_sum`+`exp` per iter same total work, rescale `8×` fewer but `8` dot+`exp` per iter overhead dominates; needs `8×8` MMA, not scalar | rejected (scalar); `8×8` MMA v2 pending |
| Q4K `rows8` decode (`32 rows/TG` `128` th, `8` rows/SG) | `512`-prompt gate `0.356→0.882 ms` (+148%), up `0.376→0.519 ms` (+38%) | `1`-token GEMV `1024→4608` launch-dominated, `32`-row TG `1152→144` TGs `8×` fewer loads but `+6` `sums` regs + `4×` row loop spills; prior small-shape duel `124-175 GB/s` `≤1 µs` already at roof | rejected |

Promotion of another candidate requires at least `3%` improvement in its
target phase and no more than `1%` regression in the complete workload. The
runtime uses a static choice; it never autotunes during inference.

## Numerical gates

Strict results remain the reference. Fast is compared against Strict and CPU
at token counts `1, 2, 8, 15, 16, 31, 32, 33, 127, 128, 129, 512`, including
non-aligned output extents and strides.

| Family | Cosine | RMSE | Maximum error | Top-k | Generated-token agreement |
|---|---:|---:|---:|---:|---:|
| BF16/F16 | `>=0.9999` | `<=0.025` | `<=0.15` | `1.0` | `>=90%` |
| Q4/QAD-Q4 | `>=0.998` | `<=0.1` | `<=0.5` | `1.0` | `>=90%` |
| Q5/Q6 | `>=0.9995` | `<=0.03` | `<=0.2` | `1.0` | `>=97%` |
| Q8 | `>=0.999` | `<=0.05` | `<=0.25` | `1.0` | `>=95%` |

No format is promoted if any relevant numerical check fails. Dense uses
reduced accumulation in Fast (worst measured M5 prefill divergence on
LFM2.5-350M is RMSE `0.0203` and maximum error `0.119` at 31 rows, within the
BF16/F16 gate above). Q6 reduced precision is restricted to the eight FFN-down
projections; its N512 maximum error is `0.00824`, RMSE is `0.00164`, and
generated-token agreement is `1.0`, within the Q5/Q6 gate.

## Official M5 protocol and promotion gate

The official matrix is the eight GGUFs crossed with `32+8` and `512+8` token
workloads, shared storage, five warmups, 15 timed repetitions, and alternating
Celeg/llama.cpp order. Each timed subprocess performs one additional untimed
in-process preheat so lazy pipeline setup and first-dispatch residency are not
charged to either engine. The fixed llama.cpp revision is
`d7bd3bfcad3e29c7e49fd26f38c79ee3e9a3fd6b`.
The llama.cpp invocation reports prefill `(p,0)`, direct decode `(0,g)`, and
combined `(p,g)` independently; the runner does not derive decode by
subtracting noisy prompt and combined samples.

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
and kernel gap closed. The routed candidates above are incremental gains, not
an official promotion: the complete matrix gate remains false.

### Latest working-tree validation

After routing the M5 Q8 matvec, splitting Q6 into a strict-precision TensorOps
floor plus the selective relaxed variant, and measuring llama.cpp decode
directly, a complete 5-warmup/15-repetition validation produced a combined
geometric mean of `0.817x`. All 16 preflights were valid, but the gate remains
false. This dirty-working-tree run is diagnostic and did not replace the
versioned official JSON.

| Format | 32+8 pp/decode/combined | 512+8 pp/decode/combined |
|---|---:|---:|
| BF16 | `0.481 / 0.932 / 0.781x` | `0.361 / 0.864 / 0.527x` |
| F16 | `0.465 / 0.931 / 0.775x` | `0.354 / 0.871 / 0.520x` |
| Q4_0 | `1.676 / 0.933 / 1.093x` | `0.976 / 0.799 / 0.928x` |
| Q4_K_M | `1.503 / 0.796 / 0.916x` | `0.827 / 0.678 / 0.779x` |
| Q5_K_M | `1.406 / 0.792 / 0.896x` | `0.815 / 0.679 / 0.770x` |
| Q6_K | `0.944 / 0.950 / 0.925x` | `0.460 / 0.824 / 0.547x` |
| Q8_0 | `1.600 / 0.964 / 1.038x` | `0.958 / 0.857 / 0.916x` |
| QAD-Q4_0 | `1.692 / 0.913 / 1.053x` | `0.982 / 0.821 / 0.932x` |

### Q4_K_M interactive working-tree snapshot

After routing the concurrent encoder, the Q4_K core rewrite, the parallel
QK-norm/RoPE kernel, the serial-encoder default, the encode CPU diet, the
Q6_K relaxed down prefill, the relaxed rows-gate removal, the relaxed
cooperative zero-init removals (quantized + dense + strict static), and the Q6K
llama-geometry decode (all three fusions, the small-shape geometries, the
K32-tile hypothesis, and the dead gate clause were A/B-neutral or negative and
reverted; the Q4_K GEMM duel showed no deficit), official-protocol single-cell
runs (5 warmups, 15 repetitions, `fast`, alternating order) on
`LFM2.5-350M-Q4_K_M.gguf` give: interactive (`32+8`) combined `1461.8` vs
`1228.3` tok/s (`1.190x`), prefill `1.726x`, decode `388.3` vs `364.1`
(`1.066x`) with all six promotion gates true; throughput (`512+8`) combined
`8190.5` vs `8038.0` (`1.019x`, prefill `1.050x`, decode `0.913x`). Recorded in
`benchmarks/results/metal_q4k_m_interactive_v5.json` and
`benchmarks/results/metal_q4k_m_throughput_v5.json` (superseding the v4 pair
at `1.201x` / `1.013x`, the v3 pair at `1.158x` / `1.004x`, and the v2 pair at
`1.148x` / `0.944x`). Against the official baseline (`0.945x` / `0.795x`
interactive combined/decode and `0.788x` throughput combined), the interactive
cell gained twenty-four points combined and twenty-seven points decode, and the
throughput cell gained twenty-three points combined on prefill + decode
changes. Both official cells now meet or beat llama.cpp combined; throughput
decode remains the only sub-parity phase (`0.913x`, up from `0.889x`). With all
MPP zero-inits removed, decode attention (`0.41 ms` at `512+8`, `15%` of decode;
`0.09 ms` / `2.5%` at `32+8` where matvecs dominate) is the next measured
bottleneck after the matvec wins.

## Primary references

- [Metal Performance Primitives Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
- [WWDC26: Metal tensors](https://developer.apple.com/videos/play/wwdc2026/330/)
- [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [Pinned llama.cpp `mul_mm.metal`](https://github.com/ggml-org/llama.cpp/blob/d7bd3bfcad3e29c7e49fd26f38c79ee3e9a3fd6b/ggml/src/ggml-metal/kernels/mul_mm.metal)
- [Pinned llama.cpp `mul_mv.metal`](https://github.com/ggml-org/llama.cpp/blob/d7bd3bfcad3e29c7e49fd26f38c79ee3e9a3fd6b/ggml/src/ggml-metal/kernels/mul_mv.metal)

The pinned competitor source stages quantized A tiles in threadgroup memory,
keeps the other operand in device memory, uses function constants for layout
conditions, and accumulates through a cooperative MPP tensor. Those are the
comparison points used by the in-tree control; Celeg does not copy or vendor
the upstream source.
