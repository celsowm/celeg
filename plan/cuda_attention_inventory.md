# CUDA Attention Algorithm Inventory

This inventory records the semantic boundaries used by Phase 3 of `refactoring_plan.md`.

The canonical public capability taxonomy already lives in
`src/backend/cuda/attention_capability.hpp`. Do not introduce a parallel enum or
selection table for refactoring purposes. Shared kernel primitives must stay below
that capability layer.

## Canonical classification dimensions

`AttentionCapability` already resolves the following dimensions before execution:

- KV format: BF16 or INT8;
- operation: prefill or decode;
- KV layout: contiguous, paged, or batch pointers;
- position source: host scalar or device counter;
- bias family: none, ALiBi, or relative-position bias;
- algorithm: strict, online, segmented, flash, GEMM, ALiBi, or relative bias.

These dimensions are the source of truth for launcher selection.

## Algorithm families

### Strict multi-pass softmax

Strict kernels preserve an explicit three-pass contract:

1. maximum score;
2. denominator;
3. rounded probability and weighted value accumulation.

The un-biased strict path rounds the score as:

```text
BF16(BF16(dot) * scale)
```

and rounds the probability before value accumulation.

Current strict families include:

- dense contiguous BF16 and INT8;
- batch-pointer BF16 and INT8;
- paged BF16 and INT8;
- block-sparse prefill/decode, contiguous and paged, BF16 and INT8;
- the post-selection softmax stage of dynamic-sparse attention.

The three-pass structure, probability rounding, mask/access policy, and launch
geometry are part of the numerical/performance contract and must remain explicit.

### Online/rescaled softmax

Online kernels update a running maximum and rescale both denominator and value
accumulators with `alpha`/`beta` factors in one token pass.

Current online families include:

- dense contiguous BF16 and INT8;
- batch-pointer BF16 and INT8;
- paged BF16 and INT8;
- ALiBi variants;
- relative-position-bias variants.

Online softmax is algorithmically different from strict softmax. It must not be
folded into a shared strict/online mega-template merely because the surrounding KV
loads look similar.

### Segmented attention

Segmented kernels perform online/rescaled softmax over partial token ranges and
materialize partial maximum, denominator, and accumulator state. Reducers then
merge segments against a global maximum using exponential reweighting.

Contiguous and paged segmented paths share the merge semantics, but their token
access and launch planning remain distinct. A small segment-merge primitive is a
valid future candidate only if it preserves register pressure and generated code.

### Sparse policies

Block-sparse attention keeps strict softmax semantics while filtering visible
blocks/tokens. The visibility policy belongs to the sparse owner and is not part of
strict softmax itself.

Dynamic-sparse attention first selects blocks using its own full-precision block
score policy and only then runs strict rounded softmax over selected blocks. The
selection stage must remain separate from strict score post-processing.

### Bias families

ALiBi and relative-position bias use algorithm-specific score functions and online
rescaling. Their score policies are not interchangeable with the un-biased strict
BF16-rounded score primitive.

### Latent, Flash, and GEMM families

Latent attention, Flash attention, and GEMM-backed attention have separate
execution/storage contracts and are not DRY targets for the strict/online cleanup.

## Orthogonal dimensions safe to share below launchers

The following dimensions can share small primitives when the generated code and
numerical behavior remain equivalent:

- BF16 vs INT8 after the dot product has produced the same logical score input;
- contiguous, paged, and batch-pointer layouts after KV addressing is complete;
- prefill vs decode when the kernel already shares the same strict algorithm body;
- visibility/mask policies only at the boundary where they decide whether a token
  participates, not by merging their traversal loops;
- strict score post-processing when there is no additional score bias.

## Algorithm-defining dimensions that remain explicit

Do not abstract away:

- strict three-pass vs online one-pass/rescaled softmax;
- segmented partial-state materialization and global merge;
- dynamic sparse block selection;
- ALiBi and relative-position score bias;
- KV addressing loops when paged/contiguous/batch-pointer memory behavior differs;
- launcher thread/block geometry;
- synchronization placement;
- strict probability BF16 rounding.

## Current extraction order

1. `strict_attention_score(dot, scale)` in `attention_common.cuh`.
   - `__device__ __forceinline__`;
   - exact `BF16(BF16(dot) * scale)` semantics;
   - first migrated only in dense strict BF16/INT8 for an isolated compile/codegen gate.
2. Expand the same primitive to other un-biased strict families only after the
   dense gate is clean.
3. Evaluate a segmented merge helper separately.
4. Do not extract a generic online-state object until benchmark evidence shows it
   does not increase register pressure or obscure per-layout behavior.

## Validation requirements

For every extraction slice:

- preserve strict/online algorithm selection from `attention_capability.hpp`;
- preserve launcher geometry and synchronization;
- preserve BF16 score/probability rounding exactly;
- compile Linux and Windows CUDA backends;
- run available attention parity coverage;
- compare decode/prefill performance before broadening an abstraction whose
  inlining or register usage could affect hot kernels.
