# Attention IR Coverage

This document tracks the implementation status of the attention semantics expressible by `AttentionSpec` across the reference path and production backends.

It keeps four claims separate:

1. the IR can describe a feature;
2. the model/reference layer can lower or reason about it;
3. a backend can execute it;
4. every relevant execution mode of that backend is covered.

A feature must not be called implemented merely because its variant exists in `graph.hpp`.

## Status legend

| Symbol | Meaning |
|---|---|
| `✓` | implementation is present for the stated scope |
| `△` | implementation exists with explicit restrictions or incomplete mode/test coverage |
| `✗` | backend explicitly rejects the feature or the required runtime owner is absent |
| `?` | not yet audited strongly enough to claim support or absence |

## Current architectural conclusion

CUDA is close to complete for the modern decoder attention surface CELEG targets: ordinary MHA/GQA/MQA, causal and sliding-window attention, paged/contiguous KV, sparse patterns, shared KV, Q/K normalization, ordinary RoPE/M-RoPE, ALiBi, output gates, and projected/factorized latent attention are substantially represented.

That is not the same as saying the complete `AttentionSpec` IR is implemented end-to-end.

The largest remaining semantic gaps are:

1. a backend-neutral external-memory / cross-attention lifecycle beyond the scoped CPU preprojected-K/V baseline;
2. CUDA relative-position bias tables;
3. formal backend/mode coverage for bidirectional and Prefix-LM;
4. Metal sparse patterns, latent attention, and general layout/paging ownership.

Metal is no longer treated as unaudited. Its runtime has explicit full-causal and sliding-window paths over ordinary Q/K/V attention, ALiBi, relative-position bias, no-position attention, standard RoPE, ordinary three-axis interleaved M-RoPE in token/decode and batched prefill, all currently modeled Q/K normalization modes, current-value orthogonalization, ordinary sigmoid output gates, and shared-KV publisher/consumer execution. Unsupported pattern, latent, partial-width/scaled M-RoPE, and RoPE-scaling semantics are rejected before execution rather than silently approximated.

Packed HeadWise attention gates are not a Metal limitation: the IR now rejects that combination globally because the packed projection is head-dimension-wide while HeadWise semantics require one scalar per head. Packed gates therefore have a canonical representation only for OutputWise/ElementWise semantics.

## IR surface

### Attention patterns

- `FullCausalPattern`
- `SlidingWindowPattern`
- `BidirectionalPattern`
- `PrefixLmPattern`
- `BlockSparsePattern`
- `DynamicSparsePattern`

### KV source and sharing

- current sequence
- external-memory slot
- private KV
- shared KV publisher
- shared KV consumer

### Bias and position

- no bias
- ALiBi
- relative-position bias buckets
- RoPE
- multi-axis RoPE
- no position encoding

### State representation

- ordinary KV
- projected latent state
- factorized latent state
- paged or contiguous state storage
- state scalar types including BF16 and INT8

### Attention transforms

- optional Q/K normalization
- per-head or whole-vector normalization granularity
- weighted or weightless normalization
- output gates
- output transforms

## Backend coverage matrix

The table is deliberately conservative. `?` means prove it rather than probably absent.

| Capability | Reference/model | CPU | CUDA | Metal |
|---|---:|---:|---:|---:|
| Full causal | ✓ | ✓ | ✓ | ✓ |
| Sliding window | ✓ | ✓ | ✓ | ✓ |
| Bidirectional | ✓ | ✓ | △ | ✗ |
| Prefix-LM | ✓ | ✓ | △ | ✗ |
| BlockSparse | ✓ | ✓ | △ | ✗ |
| DynamicSparse | ✓ | ✓ | △ | ✗ |
| ALiBi | ✓ | ✓ | ✓ | ✓ |
| Relative-position bias | ✓ | ✓ | ✗ | ✓ |
| No position encoding | ✓ | ✓ | ✓ | ✓ |
| RoPE | ✓ | ✓ | ✓ | ✓ |
| M-RoPE, ordinary attention | ✓ | ✓ | ✓ | △ |
| M-RoPE, latent attention | ✓ | ✗ | ✗ | ✗ |
| Private KV | ✓ | ✓ | ✓ | ✓ |
| Shared KV publisher/consumer | ✓ | ✓ | ✓ | ✓ |
| Contiguous ordinary KV | ✓ | ✓ | ✓ | △ |
| Paged ordinary KV | ✓ | ✓ | ✓ | △ |
| BF16 ordinary KV | ✓ | ✓ | ✓ | ✓ |
| INT8 ordinary KV | ✓ | ✗ | ✓ | ✗ |
| Projected latent attention | ✓ | ✓ | ✓ | ✗ |
| Factorized latent attention | ✓ | ✓ | ✓ | ✗ |
| Q/K normalization | ✓ | ✓ | ✓ | ✓ |
| Output gate | ✓ | △ | ✓ | ✓ |
| Current-value orthogonalization | ✓ | ✓ | ✓ | ✓ |
| External-memory / cross-attention | IR only | △ | ✗ | ✗ |

Metal ordinary KV storage remains `△` for contiguous/paged because the runtime uses an internal page-sized physical layout without yet exposing the same general page-table/layout capability surface as CUDA/CPU.

Metal ordinary M-RoPE remains `△` because token/decode and batched prefill execute explicit three-axis positions, but the backend deliberately requires three interleaved axes, split-half pairing, full-width rotation, no RoPE scaling, and theta 10000. Unsupported forms are rejected before dispatch.

Metal shared KV is `✓` for the ordinary BF16 attention surface currently supported by Metal. Publisher/consumer execution works in token/decode and batched prefill, including current-value orthogonalization on consumers sourced directly from the publisher-owned value cache.

Metal output gates are `✓` for the representable gate surface of ordinary Metal attention: unpacked OutputWise, ElementWise, and HeadWise gates plus packed OutputWise/ElementWise gates execute in token/decode and batched prefill. Packed HeadWise is excluded by the IR representation contract before backend capability validation.

## CUDA restrictions that matter

CUDA capability support and semantic-combination validation are separated from kernel/layout capability selection.

Current semantic boundaries include:

- external-memory attention is rejected;
- relative-position bias tables are rejected; ALiBi is supported;
- bidirectional, Prefix-LM, BlockSparse, and DynamicSparse are standard-attention-only;
- those constrained patterns currently require no attention bias;
- Prefix-LM requires a positive prefix length;
- BlockSparse and DynamicSparse have validated shape/range limits;
- latent attention rejects M-RoPE and has additional gate/rank restrictions;
- malformed LongRoPE factor sets are rejected before dispatch.

Therefore CUDA bidirectional, Prefix-LM, BlockSparse, and DynamicSparse remain `△`, not unconditional `✓`.

## CPU evidence already present

The CPU attention policy has explicit semantics for all six pattern variants in `CpuAttentionPattern::allows()`, including future-read semantics for bidirectional and Prefix-LM.

CPU lowers and scores ALiBi and `RelativePositionBiasSpec`, including bidirectional relative buckets. Independent query-only, key-only, and mixed Q/K normalization are handled without assuming that both norms exist.

Ordinary M-RoPE has an end-to-end CPU fixture that compares scalar prefill with chunked prefill, explicit three-axis prompt positions, prefix snapshots, and decode behavior.

CPU shared-KV has explicit publisher/consumer topology and execution ownership. The common attention contract requires one earlier publisher per group and matching KV state geometry/storage before backend execution. A dedicated synthetic end-to-end fixture compares Shared-KV execution with a mathematically equivalent Private-KV reference across scalar prefill, chunk prefill, packed ragged prefill, scalar decode, and packed decode. Token, chunk, and packed paths all skip consumer K/V projection and address the publisher-owned state.

CPU ordinary state intentionally accepts FP32/BF16 storage semantics and rejects INT8 before execution, so INT8 is `✗`, not unaudited.

Projected latent attention has a dedicated end-to-end fixture covering scalar prefill, chunk prefill, packed ragged prefill, scalar decode, and packed decode. Those execution modes agree numerically, so projected latent execution is `✓` for the declared CPU capability surface.

Factorized latent attention also has a dedicated end-to-end fixture. It covers both ungated execution and an unpacked HeadWise sigmoid output gate through scalar prefill/decode, `CpuModel::prefill_chunk()`, `CpuModel::prefill_batch()`, and `CpuModel::decode_batch()`. Public chunk and packed APIs deliberately fall back to token-wise factorized execution rather than entering an incompatible specialized fast path. The absence of a factorized chunk/packed fast path is therefore a performance/dispatch limitation, not a semantic capability gap.

CPU output gates are implemented for ordinary attention and optional factorized-latent output. Direct projected-latent output gates are explicitly rejected, so the aggregate CPU gate cell remains `△`.

CPU now also has a scoped external-memory execution baseline. It binds a non-empty `CpuExternalAttentionMemory` by stable slot, validates projected KV width at execution, does not materialize local K/V weights, and executes full-memory bidirectional attention over the bound pages. The current compiler deliberately requires ordinary projected KV, private ownership, no position encoding, no bias, no Q/K norm, no current-value transform, and an unpacked gate when gating is requested. This is real execution, but it is not yet the general cross-attention lifecycle represented by the IR, so the cell is `△`.

## Metal evidence already present

Metal has an explicit backend capability contract consumed during model initialization.

The runtime supports:

- full causal and sliding-window attention;
- ALiBi and relative-position bias over both causal patterns;
- no-position Q/K preparation;
- full-width unscaled RoPE with `SplitHalf` and `AdjacentPairs` pairing;
- ordinary three-axis interleaved M-RoPE with split-half pairing in token/decode and batched prefill;
- ordinary private and shared BF16 KV state;
- standard attention execution;
- absent Q/K normalization;
- per-head Q/K normalization;
- whole-vector Q/K normalization;
- mixed Q/K normalization granularity/presence;
- weighted and weightless Q/K normalization;
- current-value orthogonalization before the attention output projection for private and shared KV attention;
- sigmoid attention output gates in token/decode and batched-prefill paths.

### Sliding window

Sliding-window execution reads `SlidingWindowPattern::window` directly from the compiled attention semantics. Decode and batched-prefill have normal and cooperative kernels, and each query starts at `max(0, sequence_length - window)`.

### Biases

ALiBi uses `-slope[head] * abs(query_position - key_position)` before softmax. Slopes are materialized once in a Metal-owned buffer.

Relative-position bias uses the same bucket contract as CPU: exact-distance buckets near zero, logarithmic buckets farther away, directional halves for `bidirectional=true`, and clamping at the last bucket. The resolved `[query_heads, bucket_count]` tensor is retained per layer.

### Position handling

`NoPositionEncodingSpec` has a position-free Q/K path rather than a synthetic zero-angle RoPE path.

Standard RoPE dispatches by `RopePairingKind`. Metal currently requires `rotary_fraction == 1.0` and `NoRopeScaling` and rejects other RoPE forms before execution.

Ordinary M-RoPE uses the same interleaved axis assignment as the CPU oracle (`axis = pair % 3`). `PromptEmbedding::rope_positions` supplies one position triplet per prefill token and `next_rope_position` survives decode and snapshots.

Batched M-RoPE prefill stages one shared `[rows, 3]` position buffer and uses `celeg_qk_mrope_position_batch` after standalone Q/K normalization. Explicit prompt positions and synthesized `{position, position, position}` rows share the same batch path. This removes the former token-by-token M-RoPE prefill fallback without multiplying fused norm/position kernel combinations.

The current Metal M-RoPE contract remains intentionally narrower than the IR: it requires full-width, unscaled, three-axis interleaved split-half M-RoPE with theta 10000. These restrictions are validated before device execution.

### Shared KV

Shared-KV topology is now governed by the common attention contract: one non-negative earlier publisher per group plus compatible KV state kind, geometry, paging/granularity, and scalar storage. Backend validators consume that contract instead of maintaining independent publisher/consumer truth tables.

At execution time the consumer aliases the publisher's Metal key/value cache. `CompiledAttentionExecution::has_key_value` is the ownership fact: consumers project, normalize, and position only Q, pass zero prepared key heads through Q/K preparation, skip K/V projection and KV store, then restore the semantic `key_value_heads` when scoring against the shared cache. Token/decode and batched-prefill therefore reuse the same attention kernels as private KV instead of maintaining a second shared-attention implementation.

Current-value orthogonalization also reuses the ordinary Metal transform kernel. KV-owning attention passes the local projected V workspace. A shared-KV consumer instead binds the publisher-owned `value_cache` at `position * key_value_width` for token/decode or `base_position * key_value_width` for batched prefill. The selected cache slice is contiguous in the current Metal page layout, so the transform observes the same `[rows, key_heads, head_dim]` current-value view without materializing a duplicate V projection.

### Q/K normalization

Q/K normalization no longer depends on a fake all-ones tensor being interpreted as an absent norm.

The runtime now distinguishes the semantic cases explicitly:

```text
both Q and K PerHead, ordinary KV-owning position
    -> fused norm + position fast path

shared-KV consumer / M-RoPE / mixed / WholeVector / one side absent / both absent
    -> normalize each locally owned side independently
    -> position-only Q/K preparation
```

Per-head normalization has standalone token and batch kernels for mixed, M-RoPE, and shared-consumer cases. Whole-vector normalization reuses the ordinary Metal RMSNorm path with the projection-wide weight shape emitted by the weight plan. Weightless norms synthesize an all-ones weight at the semantic norm width, while a truly absent norm skips RMS normalization entirely.

This preserves the fused hot path for the common ordinary per-head/per-head case without conflating absence, granularity, weightless semantics, M-RoPE position handling, or shared-KV ownership.

### Output gate

Unpacked gates use the resolved `TensorRole::AttentionGate` projection and support OutputWise, ElementWise, and HeadWise granularity. Packed gates preserve the checkpoint convention where each query head is stored as `[query_head, gate_head]`; Q is deinterleaved before Q/K normalization or position handling, while gate values stay in the packed staging buffer until they are applied.

The sigmoid gate is applied to the per-head attention result after any current-value orthogonalization and before `AttentionOutput` projection. Token/decode and batched-prefill use the same semantic ordering. Packed HeadWise is rejected by the backend-neutral IR representation contract rather than by Metal-specific capability policy.

### Output transform

`OrthogonalizeCurrentValueSpec` is applied to the per-head attention result before `AttentionOutput` projection, matching the CPU contract. Each query head removes its projection onto the current value head, with GQA/MQA query heads mapped to their corresponding value head. Token and batched-prefill paths share the same Metal kernel and validate a positive finite `minimum_norm_squared` floor before execution. Shared-KV consumers source the semantic current V directly from the publisher-owned value-cache slice instead of using a stale local workspace.

### Explicit Metal rejections

Metal still rejects before device/pipeline execution:

- bidirectional, Prefix-LM, BlockSparse, and DynamicSparse patterns;
- partial-width or scaled standard RoPE;
- M-RoPE forms outside full-width, unscaled, three-axis interleaved split-half theta-10000 execution;
- external-memory sources;
- non-BF16 KV state semantics;
- latent and factorized-latent execution, including latent M-RoPE.

Packed HeadWise gates are rejected earlier by the backend-neutral attention representation validator and therefore are not a Metal-specific rejection.

## What "AttentionSpec 100% implemented" must mean

The completion claim is allowed only when every IR dimension has an explicit capability outcome for every production backend and relevant execution mode.

At minimum, coverage must distinguish token/decode, graph decode, single/batched prefill, contiguous/paged state, ordinary/projected/factorized latent state, and dense/sparse attention where applicable.

Intentional rejection is valid. Unsupported combinations must fail at compile/bind time with a stable diagnostic rather than silently degrading to another semantic mode.

## Completion plan

### Phase 0 — Executable capability truth

Keep backend semantic support explicit and tested. New `AttentionSpec` variants must acquire backend capability decisions in the same change.

### Phase 1 — External-memory / cross-attention

Generalize the scoped CPU preprojected-K/V baseline into a backend-neutral binding lifecycle: stable slot identity, shape/type/lifetime metadata, explicit hidden-state versus preprojected-K/V representation, binding validation, reference execution, and then CUDA/Metal implementations. Preserve the existing CPU baseline as an executable subset rather than replacing it with a second source model.

### Phase 2 — CUDA relative-position bias

Reuse the existing bucket semantics, add CUDA-owned `[query_heads, bucket_count]` device storage, add the bias before softmax, cover paged KV, and differential-test CPU versus CUDA.

### Phase 3 — Bidirectional and Prefix-LM mode coverage

Prove the meaningful execution-mode Cartesian product instead of relying on compiler acceptance. Prefix-LM needs boundary tests around the prefix transition; bidirectional needs explicit future-key reads.

### Phase 4 — Extend Metal deliberately

Suggested order:

1. true layout/paging capability declaration;
2. generalize remaining M-RoPE theta/scaling/partial-width ownership;
3. sparse patterns;
4. latent attention;
5. external memory after the common lifecycle exists.

Every newly supported cell moves to `✓` only with a named implementation path and test.

### Phase 5 — Anti-regression

Keep capability tests backend-parameterized, require new IR variants to update them, keep semantic tests separate from performance tests, and add representative end-to-end fixtures.

## Design constraints

### One owner per semantic fact

Prefer:

```text
AttentionSpec
    ↓
representation/lifecycle contract
    ↓
backend capability policy
    ↓
compiler / execution validation
    ↓
execution dispatch
    ↓
table-driven tests
```

Documentation summarizes tested truth; it is not a second capability database.

### Capability is not dispatch

A backend can support a feature while choosing different kernels for token, prefill, paged, or sparse execution. Semantic support stays separate from kernel selection.

### Do not optimize cross-attention before ownership is correct

External memory needs lifecycle, slot binding, shape/type validation, and reference semantics before backend-specific optimization. The scoped CPU preprojected-K/V path is useful execution evidence, not a substitute for that lifecycle.

## Definition of done

The attention subsystem can be called complete with respect to the IR when:

- every `AttentionSpec` dimension has a backend capability outcome;
- external-memory attention has a common lifecycle and executes end-to-end on the chosen baseline backends;
- CUDA relative-position bias is implemented for its declared modes or deliberately scoped out;
- bidirectional and Prefix-LM have execution-mode tests rather than compiler-only acceptance;
- Metal has an audited matrix and rejects unsupported semantics before execution;
- unsupported combinations fail during compile/bind with stable diagnostics;
- capability tests prevent new IR variants from landing without backend decisions;
- the matrix has no unexplained `?` cells.