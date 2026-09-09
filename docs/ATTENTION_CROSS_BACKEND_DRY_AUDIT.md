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
  - dynamic-sparse reference visibility
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
- dynamic sparse top-K block scoring/selection implementation
- paged versus contiguous storage traversal

These are execution mechanisms, not independent definitions of model semantics.

## Metal mirrors

Metal mirrors are acceptable only when they are covered by cross-backend conformance tests. Current mirrors/probes cover:

- bias semantics
- supported pattern semantics
- online-softmax transition
- partial-state merge semantics
- attention micro-semantics

Do not introduce a new Metal semantic formula without adding it to the corresponding conformance contract.

## Remaining semantic decision: DynamicSparsePattern

`DynamicSparsePattern` still needs an explicit architectural decision before further DRY refactoring.

The canonical host/reference visibility rule currently models a deterministic block policy, while the CUDA prefill implementation performs score-based top-K block selection at runtime. These are not merely two optimized implementations of the same visible-token predicate; they can select different tokens.

Therefore this must not be "fixed" by mechanically routing CUDA through `dynamic_sparse_visible()` or by changing the host reference to imitate CUDA without deciding the intended model contract first.

Before changing this area:

1. define whether `DynamicSparsePattern` means a deterministic sparse mask or score-based top-K block selection;
2. encode that meaning in the model-level spec;
3. determine which backends are expected to support it;
4. add parity/golden tests for selected blocks and final attention output;
5. only then unify backend implementations around the chosen contract.

## Final rule

Share **what attention means**; specialize **how each backend executes it**.

A duplicated expression is a DRY violation only when it independently defines model behavior. Hardware scheduling, memory layout, vectorization and numerically intentional arithmetic grouping remain backend responsibilities.
