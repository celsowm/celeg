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
| `△` | implementation exists with explicit restrictions or incomplete mode coverage |
| `✗` | backend explicitly rejects the feature or the required runtime owner is absent |
| `?` | not yet audited strongly enough to claim support or absence |

## Current architectural conclusion

CUDA is close to complete for the modern decoder attention surface CELEG targets: ordinary MHA/GQA/MQA, causal and sliding-window attention, paged/contiguous KV, sparse patterns, shared KV, Q/K normalization, ordinary RoPE/M-RoPE, ALiBi, output gates, and projected/factorized latent attention are substantially represented.

That is not the same as saying the complete `AttentionSpec` IR is implemented end-to-end.

The largest remaining semantic gaps are:

1. external-memory / cross-attention lifecycle and execution;
2. CUDA relative-position bias tables;
3. formal backend/mode coverage for bidirectional and Prefix-LM;
4. Metal shared KV, sparse patterns, latent attention, general layout/paging ownership, and the unresolved packed-HeadWise gate representation.

Metal is no longer treated as unaudited. Its runtime has explicit full-causal and sliding-window paths over ordinary Q/K/V attention, ALiBi, relative-position bias, no-position attention, standard RoPE, ordinary three-axis interleaved M-RoPE, all currently modeled Q/K normalization modes, current-value orthogonalization, and ordinary sigmoid output gates. Unsupported pattern, sharing, latent, partial-width RoPE, RoPE-scaling, and packed-HeadWise gate semantics are rejected before execution rather than silently approximated.

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
| M-RoPE, ordinary attention | ✓ | ? | ✓ | △ |
| M-RoPE, latent attention | ✓ | ? | ✗ | ✗ |
| Private KV | ✓ | ✓ | ✓ | ✓ |
| Shared KV publisher/consumer | ✓ | ? | ✓ | ✗ |
| Contiguous ordinary KV | ✓ | ✓ | ✓ | △ |
| Paged ordinary KV | ✓ | ✓ | ✓ | △ |
| BF16 ordinary KV | ✓ | ✓ | ✓ | ✓ |
| INT8 ordinary KV | ✓ | ? | ✓ | ✗ |
| Projected latent attention | ✓ | ? | ✓ | ✗ |
| Factorized latent attention | ✓ | ? | ✓ | ✗ |
| Q/K normalization | ✓ | ✓ | ✓ | ✓ |
| Output gate | ✓ | ? | ✓ | △ |
| Current-value orthogonalization | ✓ | ✓ | ✓ | ✓ |
| External-memory / cross-attention | IR only | ✗ | ✗ | ✗ |

Metal ordinary KV storage remains `△` for contiguous/paged because the runtime uses an internal page-sized physical layout without yet exposing the same general page-table/layout capability surface as CUDA/CPU.

Metal ordinary M-RoPE remains `△` because token/decode and public prefill are semantically supported with explicit three-axis positions, but M-RoPE still disables the batched-prefill fast path and executes prefill token-by-token.

Metal output gates remain `△` because unpacked OutputWise, ElementWise, and HeadWise gates are supported, as are packed OutputWise/ElementWise gates, but packed HeadWise is rejected until that checkpoint representation has an unambiguous semantic contract.

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

CPU also lowers and scores ALiBi and `RelativePositionBiasSpec`, including bidirectional relative buckets. CPU and CUDA reject external-memory attention until a real external-memory lifecycle exists.

## Metal evidence already present

Metal has an explicit backend capability contract consumed during model initialization.

The runtime supports:

- full causal and sliding-window attention;
- ALiBi and relative-position bias over both causal patterns;
- no-position Q/K preparation;
- full-width unscaled RoPE with `SplitHalf` and `AdjacentPairs` pairing;
- ordinary three-axis interleaved M-RoPE with split-half pairing;
- ordinary private BF16 KV state;
- standard attention execution;
- absent Q/K normalization;
- per-head Q/K normalization;
- whole-vector Q/K normalization;
- mixed Q/K normalization granularity/presence;
- weighted and weightless Q/K normalization;
- current-value orthogonalization before the attention output projection;
- sigmoid attention output gates in token/decode and batched-prefill paths.

### Sliding window

Sliding-window execution reads `SlidingWindowPattern::window` directly from the compiled attention semantics. Decode and batched-prefill have normal and cooperative kernels, and each query starts at `max(0, sequence_length - window)`.

### Biases

ALiBi uses `-slope[head] * abs(query_position - key_position)` before softmax. Slopes are materialized once in a Metal-owned buffer.

Relative-position bias uses the same bucket contract as CPU: exact-distance buckets near zero, logarithmic buckets farther away, directional halves for `bidirectional=true`, and clamping at the last bucket. The resolved `[query_heads, bucket_count]` tensor is retained per layer.

### Position handling

`NoPositionEncodingSpec` has a position-free Q/K path rather than a synthetic zero-angle RoPE path.

Standard RoPE dispatches by `RopePairingKind`. Metal currently requires `rotary_fraction == 1.0` and `NoRopeScaling` and rejects other RoPE forms before execution.

Ordinary M-RoPE uses the same interleaved axis assignment as the CPU oracle (`axis = pair % 3`). `PromptEmbedding::rope_positions` supplies one position triplet per prefill token and `next_rope_position` survives decode and snapshots. M-RoPE prefill deliberately remains token-wise until a dedicated batched three-axis Q/K kernel exists.

### Q/K normalization

Q/K normalization no longer depends on a fake all-ones tensor being interpreted as an absent norm.

The runtime now distinguishes the semantic cases explicitly:

```text
both Q and K PerHead
    -> fused norm + position fast path

mixed / WholeVector / one side absent / both absent
    -> normalize each present side independently
    -> position-only Q/K preparation
```

Per-head normalization has standalone token and batch kernels for mixed cases. Whole-vector normalization reuses the ordinary Metal RMSNorm path with the projection-wide weight shape emitted by the weight plan. Weightless norms synthesize an all-ones weight at the semantic norm width, while a truly absent norm skips RMS normalization entirely.

This preserves the fused hot path for the common per-head/per-head case without conflating absence, granularity, or weightless semantics.

### Output gate

Unpacked gates use the resolved `TensorRole::AttentionGate` projection and support OutputWise, ElementWise, and HeadWise granularity. Packed gates preserve the checkpoint convention where each query head is stored as `[query_head, gate_head]`; Q is deinterleaved before Q/K normalization or position handling, while gate values stay in the packed staging buffer until they are applied.

The sigmoid gate is applied to the per-head attention result after any current-value orthogonalization and before `AttentionOutput` projection. Token/decode and batched-prefill use the same semantic ordering. Packed HeadWise is intentionally rejected because the packed projection is structurally head-dimension-wide and no canonical scalar-per-head packed representation is currently defined by the IR.

### Output transform

`OrthogonalizeCurrentValueSpec` is applied to the per-head attention result before `AttentionOutput` projection, matching the CPU contract. Each query head removes its projection onto the current value head, with GQA/MQA query heads mapped to their corresponding value head. Token and batched-prefill paths share the same Metal kernel and validate a positive finite `minimum_norm_squared` floor before execution.

### Explicit Metal rejections

Metal still rejects before device/pipeline execution:

- bidirectional, Prefix-LM, BlockSparse, and DynamicSparse patterns;
- partial-width or scaled RoPE;
- external-memory sources;
- shared KV publisher/consumer modes;
- non-BF16 KV state semantics;
- packed HeadWise attention gates;
- latent and factorized-latent execution, including latent M-RoPE.

## What "AttentionSpec 100% implemented" must mean

The completion claim is allowed only when every IR dimension has an explicit capability outcome for every production backend and relevant execution mode.

At minimum, coverage must distinguish token/decode, graph decode, single/batched prefill, contiguous/paged state, ordinary/projected/factorized latent state, and dense/sparse attention where applicable.

Intentional rejection is valid. Unsupported combinations must fail at compile/bind time with a stable diagnostic rather than silently degrading to another semantic mode.

## Completion plan

### Phase 0 — Executable capability truth

Keep backend semantic support explicit and tested. New `AttentionSpec` variants must acquire backend capability decisions in the same change.

### Phase 1 — External-memory / cross-attention

Build a backend-neutral external-memory binding lifecycle first: stable slot identity, shape/type/lifetime metadata, explicit hidden-state versus preprojected-K/V representation, binding validation, reference execution, then CPU and CUDA implementations.

### Phase 2 — CUDA relative-position bias

Reuse the existing bucket semantics, add CUDA-owned `[query_heads, bucket_count]` device storage, add the bias before softmax, cover paged KV, and differential-test CPU versus CUDA.

### Phase 3 — Bidirectional and Prefix-LM mode coverage

Prove the meaningful execution-mode Cartesian product instead of relying on compiler acceptance. Prefix-LM needs boundary tests around the prefix transition; bidirectional needs explicit future-key reads.

### Phase 4 — Extend Metal deliberately

Suggested order:

1. resolve packed HeadWise gate semantics;
2. true layout/paging capability declaration;
3. dedicated batched M-RoPE Q/K preparation;
4. shared KV;
5. sparse patterns;
6. latent attention;
7. external memory after the common lifecycle exists.

Every newly supported cell moves to `✓` only with a named implementation path and test.

### Phase 5 — Anti-regression

Keep capability tests backend-parameterized, require new IR variants to update them, keep semantic tests separate from performance tests, and add representative end-to-end fixtures.

## Design constraints

### One owner per semantic fact

Prefer:

```text
AttentionSpec
    ↓
backend capability policy
    ↓
compiler / lifecycle validation
    ↓
execution dispatch
    ↓
table-driven tests
```

Documentation summarizes tested truth; it is not a second capability database.

### Capability is not dispatch

A backend can support a feature while choosing different kernels for token, prefill, paged, or sparse execution. Semantic support stays separate from kernel selection.

### Do not optimize cross-attention before ownership is correct

External memory needs lifecycle, slot binding, shape/type validation, and reference semantics before backend-specific optimization.

## Definition of done

The attention subsystem can be called complete with respect to the IR when:

- every `AttentionSpec` dimension has a backend capability outcome;
- external-memory attention executes end-to-end on the chosen baseline backends;
- CUDA relative-position bias is implemented for its declared modes or deliberately scoped out;
- bidirectional and Prefix-LM have execution-mode tests rather than compiler-only acceptance;
- Metal has an audited matrix and rejects unsupported semantics before execution;
- unsupported combinations fail during compile/bind with stable diagnostics;
- capability tests prevent new IR variants from landing without backend decisions;
- the matrix has no unexplained `?` cells.
