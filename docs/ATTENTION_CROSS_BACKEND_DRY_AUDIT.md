# Cross-backend attention DRY audit

This document records the final CPU/CUDA/Metal attention DRY pass and distinguishes shared model semantics from backend-specific execution details.

## Canonical shared semantics

The following model-level rules now have canonical C++/CUDA definitions under `src/celeg/attention/`:

- `bias_semantics.hpp`
  - ALiBi distance/bias
  - relative-position bucket selection
- `pattern_semantics.hpp`
  - causal visibility
  - sliding-window visibility/start
  - Prefix-LM visibility/future-read behavior
  - block-sparse visibility
- `dynamic_sparse_semantics.hpp`
  - content-ranked dynamic sparse block selection
  - causal block geometry
  - deterministic top-K insertion
  - selected-block membership
  - score ties prefer the lower block index
- `online_semantics.hpp`
  - online-softmax state transition
  - scalar accumulator update
- `merge_semantics.hpp`
  - partial-state rescaling
  - pairwise partial-state merge
  - scalar merged accumulator update
- `micro_semantics.hpp`
  - GQA query-head to KV-head mapping
  - query-position / sequence-length conversion
  - attention scale

CPU and CUDA consume these helpers directly where their execution model permits it. Metal keeps small MSL mirrors because Metal shaders cannot include the C++/CUDA headers directly; conformance probes compare those mirrors against the canonical host semantics.

## Backend-specific code that should remain backend-specific

The following duplication is intentional and should not be removed merely to reduce line count:

- CPU AVX/AVX2/FMA loops
- CUDA warp reductions, shared memory, launch geometry and synchronization
- Metal simdgroup/threadgroup reduction and dispatch details
- KV-cache address calculation when layouts differ by backend
- quantized load/dequantization arithmetic whose association is required for parity
- block-sparse three-pass accumulation versus online-softmax recurrence
- dynamic sparse block-score reduction and accelerator storage for the selected set
- paged versus contiguous storage traversal

These are execution mechanisms, not independent definitions of model semantics.

## Metal mirrors

Metal mirrors are acceptable only when they are covered by cross-backend conformance tests. Current mirrors/probes cover:

- bias semantics
- supported pattern semantics
- online-softmax transition
- partial-state merge semantics
- attention micro-semantics

The production `celeg_attention_span()` path now consumes the same `celeg_online_transition()` and `celeg_partial_rescale()` helpers exercised by the conformance probes. The helpers live in `common.metal`; probe fragments no longer carry independent copies of those formulas.

Do not introduce a new Metal semantic formula without adding it to the corresponding conformance contract.

## Resolved DynamicSparsePattern contract

`DynamicSparsePattern` means content-ranked causal block selection, not a position-only visibility mask.

For each query position:

1. consider every causal block from block zero through the query block;
2. score each candidate block by the maximum scaled Q.K score among the causally visible tokens in that block;
3. retain at most `max_selected_blocks` candidates with the highest scores;
4. on an exact score tie, prefer the lower block index;
5. run attention only over tokens belonging to the selected blocks.

The old position-only `dynamic_sparse_visible(query_position, key_position, ...)` helper was removed because visibility cannot be decided from positions alone under this contract.

The canonical selection mechanics are defined in `dynamic_sparse_semantics.hpp`. CUDA consumes those mechanics directly while retaining its own warp reduction, shared-memory layout, synchronization and BF16-rounded three-pass attention implementation.

### Backend support

| Backend | DynamicSparsePattern |
| --- | --- |
| CPU | Unsupported. The compiler capability is false and the low-level `CpuAttentionPattern` path rejects the pattern explicitly. |
| CUDA | Supported as content-ranked top-K. The CUDA capability layer retains its backend limit on the maximum selected-block count. |
| Metal | Unsupported. `metal_attention_capabilities()` advertises `dynamic_sparse = false`. |

Host and CUDA-device semantic tests cover ranking, K selection, negative scores and deterministic tie behavior. The existing CUDA end-to-end dynamic-sparse test remains the output-level check that the content-ranked block wins.

## Final rule

Share **what attention means**; specialize **how each backend executes it**.

A duplicated expression is a DRY violation only when it independently defines model behavior. Hardware scheduling, memory layout, vectorization and numerically intentional arithmetic grouping remain backend responsibilities.
