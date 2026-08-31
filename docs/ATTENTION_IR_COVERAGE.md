# Attention IR Coverage

This document tracks the implementation status of the attention semantics expressible by `AttentionSpec` across the reference path and production backends.

It exists to keep four different claims separate:

1. the IR can describe a feature;
2. the model/reference layer can lower or reason about it;
3. a backend can execute it;
4. every relevant execution mode of that backend is covered.

A feature must not be called "implemented" merely because its variant exists in `graph.hpp`.

## Status legend

| Symbol | Meaning |
|---|---|
| `✓` | implementation is present for the stated scope |
| `△` | implementation/lowering exists, but with explicit restrictions or incomplete mode coverage |
| `✗` | backend explicitly rejects the feature or the required runtime owner is absent |
| `?` | not yet audited strongly enough to claim support or absence |

## Current architectural conclusion

The CUDA decoder-attention subsystem is close to complete for the modern decoder models CELEG targets: ordinary MHA/GQA/MQA, causal and sliding-window attention, paged/contiguous KV, sparse patterns, shared KV, Q/K normalization, RoPE/M-RoPE on the ordinary path, ALiBi, output gates, and projected/factorized latent attention are all substantially represented in the current implementation.

That is not the same as saying the complete `AttentionSpec` IR is implemented end-to-end.

The largest remaining semantic gaps are:

1. external-memory / cross-attention lifecycle and execution;
2. CUDA relative-position bias tables;
3. formal backend/mode coverage for non-decoder-oriented patterns, especially bidirectional and Prefix-LM;
4. extending Metal beyond its now-explicit causal/sliding + RoPE + standard-attention contract.

Metal is no longer treated as unaudited by default. Its runtime now has explicit full-causal and sliding-window paths over ordinary Q/K/V attention, while unsupported pattern, bias, multi-axis, sharing, transform, and latent semantics are rejected during model initialization rather than being silently executed as causal attention.

## IR surface

`AttentionSpec` currently describes the following independent dimensions.

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

- Q/K normalization
- output gates
- output transforms

## Backend coverage matrix

The table below is deliberately conservative. `?` means "prove it" rather than "probably absent".

| Capability | Reference/model | CPU | CUDA | Metal |
|---|---:|---:|---:|---:|
| Full causal | ✓ | ✓ | ✓ | ✓ |
| Sliding window | ✓ | ✓ | ✓ | ✓ |
| Bidirectional | ✓ | ✓ | △ | ✗ |
| Prefix-LM | ✓ | ✓ | △ | ✗ |
| BlockSparse | ✓ | ✓ | △ | ✗ |
| DynamicSparse | ✓ | ✓ | △ | ✗ |
| ALiBi | ✓ | ✓ | ✓ | ✗ |
| Relative-position bias | ✓ | ✓ | ✗ | ✗ |
| RoPE | ✓ | ✓ | ✓ | ✓ |
| M-RoPE, ordinary attention | ✓ | ? | ✓ | ✗ |
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
| Output gate | ✓ | ? | ✓ | ✗ |
| External-memory / cross-attention | IR only | ✗ | ✗ | ✗ |

Metal ordinary KV storage is marked `△` rather than unconditional `✓` for contiguous/paged because its current runtime uses an internal page-sized physical layout without yet exposing the same general page-table/layout capability surface as CUDA/CPU.

### CUDA restrictions that matter

CUDA backend capability support and semantic-combination validation are now separated from kernel/layout capability selection.

The semantic policy makes these boundaries explicit:

- external-memory attention is rejected;
- relative-position bias tables are rejected; ALiBi is supported instead;
- bidirectional, Prefix-LM, BlockSparse, and DynamicSparse are accepted only as standard attention;
- those constrained patterns currently require no attention bias;
- Prefix-LM requires a positive prefix length;
- BlockSparse and DynamicSparse have validated shape/range limits;
- latent attention rejects M-RoPE and has additional gate/rank restrictions;
- malformed LongRoPE factor sets are rejected before dispatch.

Therefore the CUDA entries for bidirectional, Prefix-LM, BlockSparse, and DynamicSparse are `△`, not unconditional `✓`.

### CPU evidence already present

The CPU attention policy has explicit semantics for all six attention-pattern variants in `CpuAttentionPattern::allows()`, including future-read semantics for bidirectional and Prefix-LM.

The same policy lowers and scores both ALiBi and `RelativePositionBiasSpec`, including bidirectional relative buckets.

CPU and CUDA now both reject external-memory attention at compile time until a real external-memory lifecycle exists.

This is stronger evidence than merely finding the variants in the IR, but execution-mode-specific tests are still required before broadening every CPU cell above.

### Metal evidence already present

Metal has an explicit backend capability contract consumed during model initialization.

The runtime now supports:

- full causal attention;
- sliding-window attention in token/decode and batched prefill paths;
- ordinary private KV state;
- RoPE;
- standard attention execution;
- Q/K normalization;
- BF16 attention state semantics.

Sliding-window execution uses the compiled `SlidingWindowPattern::window` directly rather than copying it into `MetalModel::Impl::Layer`. Separate normal and cooperative kernels exist for both token/decode and batched prefill; each query reads keys starting at `max(0, sequence_length - window)`.

The following semantics are explicitly rejected before device/pipeline setup rather than silently degrading to another attention mode:

- bidirectional, Prefix-LM, BlockSparse, and DynamicSparse patterns;
- ALiBi and relative-position bias;
- no-position and M-RoPE modes;
- external-memory sources;
- shared KV publisher/consumer modes;
- non-BF16 KV state semantics;
- output gates and output transforms;
- latent and factorized-latent execution.

This turns the Metal coverage into explicit capability outcomes without claiming kernels that do not exist.

## What "AttentionSpec 100% implemented" must mean

The completion claim is allowed only when every IR dimension has an explicit capability outcome for every production backend and relevant execution mode.

At minimum, coverage must distinguish:

- token/decode;
- graph decode;
- single-sequence prefill;
- batched prefill;
- contiguous state;
- paged state;
- ordinary KV;
- projected latent;
- factorized latent;
- dense vs sparse attention where applicable.

A backend may intentionally not support a semantic feature. In that case the matrix must record an explicit, tested capability rejection rather than leaving behavior accidental.

## Completion plan

### Phase 0 — Turn the matrix into executable capability truth

Goal: stop learning backend support from scattered `if`, `std::holds_alternative`, and dispatch failures.

1. Define an attention backend capability descriptor owned by the backend/compiler boundary.
2. Express support/restrictions for:
   - attention pattern;
   - bias kind;
   - position kind;
   - KV source;
   - state kind;
   - paged/contiguous storage;
   - execution kind/mode.
3. Make compiler/lifecycle validation consume that descriptor where practical.
4. Add table-driven tests that enumerate `AttentionSpec` variants and verify accepted/rejected combinations.
5. Generate or validate this document's matrix from the same source of truth if feasible; do not create a second manually maintained capability database.

Acceptance criteria:

- no backend capability is inferred solely from lack of a crash;
- every `AttentionPatternSpec`, `AttentionBiasSpec`, and `AttentionKeyValueSource` variant has an explicit backend result;
- rejection messages identify the unsupported dimension rather than failing later during execution.

### Phase 1 — External-memory / cross-attention end-to-end

This is the largest true IR/runtime gap.

The existing `ExternalMemorySource { slot }` is only a semantic selector. Full support requires an owner for the memory behind that slot.

Implement a backend-neutral lifecycle before adding backend-specific kernels:

1. Introduce an external-memory binding/session abstraction with stable slot identity.
2. Define slot metadata:
   - sequence length;
   - hidden/KV dimensions;
   - scalar/storage type;
   - ownership/lifetime;
   - device/backend placement.
3. Define whether external memory stores hidden states or preprojected K/V. Prefer one explicit representation in the first implementation rather than an ambiguous union.
4. Add binding validation during model/session setup.
5. Add reference execution for cross-attention.
6. Implement CPU first as the semantic oracle.
7. Implement CUDA with reusable K/V projection + attention dispatch owners rather than duplicating self-attention paths.
8. Add token and batched-prefill tests using deterministic external-memory fixtures.
9. Only then consider persistent/paged external-memory caching if real models need it.

Acceptance criteria:

- a model using `ExternalMemorySource{slot}` can bind memory and execute end-to-end;
- missing, duplicate, incompatible, or stale slots fail at binding/validation time;
- self-attention KV cache lifecycle is not reused accidentally as external-memory ownership;
- CPU/reference/CUDA agree numerically on a small cross-attention fixture.

### Phase 2 — Complete relative-position bias on CUDA

The IR and CPU already define the semantic contract. CUDA currently rejects it explicitly.

1. Reuse the existing bucket semantics as the canonical definition.
2. Add CUDA-owned device representation for `[query_heads, bucket_count]` bias tables.
3. Centralize bucket computation so token, graph, and prefill paths do not each reinvent it.
4. Integrate bias addition before softmax in standard dense attention.
5. Extend to paged KV without changing bucket semantics.
6. Decide separately whether sparse and constrained patterns support relative bias; do not silently broaden them.
7. Differential-test CPU vs CUDA for:
   - unidirectional buckets;
   - bidirectional buckets;
   - exact-distance region;
   - logarithmic-distance region;
   - clamped maximum distance;
   - multiple query heads.

Acceptance criteria:

- CUDA no longer rejects `RelativePositionBiasSpec` for the supported standard paths;
- token/graph/prefill coverage is explicit;
- unsupported combinations remain compile-time capability errors.

### Phase 3 — Close Bidirectional and Prefix-LM mode coverage

The semantics exist, and CUDA lowering accepts them under restrictions. The remaining job is to prove execution completeness rather than assume it.

For each pattern, test the Cartesian product that is actually meaningful:

- CPU / CUDA / Metal;
- token/decode where semantically meaningful;
- graph decode;
- prefill;
- batched prefill;
- contiguous/paged KV;
- bias combinations explicitly allowed by capability policy.

Prefix-LM deserves dedicated boundary cases:

- query inside prefix;
- key inside/outside prefix;
- first query after prefix;
- prefix equal to sequence length;
- invalid zero/negative prefix rejected.

Bidirectional deserves explicit future-key tests so a causal implementation cannot accidentally pass a weak fixture.

Acceptance criteria:

- no `?` remains for these two patterns;
- every unsupported execution mode is intentionally rejected;
- reference/CPU/CUDA tests prove future-key behavior.

### Phase 4 — Extend Metal deliberately

Metal now has an audited baseline rather than an unknown matrix.

Extend it only by carrying each semantic fact through execution explicitly and testing the corresponding kernel behavior.

Suggested order:

1. true layout/paging capability declaration;
2. ALiBi;
3. relative bias;
4. M-RoPE;
5. shared KV;
6. sparse patterns;
7. latent attention;
8. external memory after the backend-neutral lifecycle exists.

Every newly supported cell must move from `✗`/`△` to `✓` only with a named implementation path and test.

### Phase 5 — Keep the matrix from regressing

1. Keep the attention capability test suite backend-parameterized.
2. Require new `AttentionSpec` variants to update capability tests in the same change.
3. Require a backend to reject unsupported variants at compile/bind time.
4. Add representative end-to-end model fixtures for decoder attention and cross-attention.
5. Keep performance tests separate from semantic capability tests.

## Design constraints

### One owner per semantic fact

Do not create independent copies of the same capability matrix in docs, CPU, CUDA, and Metal.

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

The documentation should summarize the tested truth, not become another source of truth.

### Capability is not dispatch

A backend can support a feature while choosing different kernels for token, prefill, paged, or sparse execution. Keep semantic support separate from kernel selection.

### Intentional rejection is valid

"100% of the IR implemented" does not require every backend to execute every feature. It requires the system to have a complete, explicit contract: supported combinations execute correctly; unsupported combinations are rejected early and deliberately.

### Do not optimize cross-attention before ownership is correct

External memory needs lifecycle, slot binding, shape/type validation, and reference semantics before CUDA-specific optimization. Reusing self-attention KV cache state merely because the tensors look similar would erase an important ownership boundary.

## Definition of done

The attention subsystem can be called complete with respect to the IR when:

- every `AttentionSpec` variant/dimension has a backend capability outcome;
- external-memory attention executes end-to-end on the chosen baseline backends;
- CUDA relative-position bias is either implemented for its declared modes or deliberately scoped out by capability policy;
- Bidirectional and Prefix-LM have execution-mode tests rather than compiler-only acceptance;
- Metal has an audited matrix and rejects unsupported semantics before execution;
- unsupported combinations fail during compile/bind with stable diagnostics;
- capability tests prevent a new IR variant from landing without backend decisions;
- this matrix has no unexplained `?` cells.
