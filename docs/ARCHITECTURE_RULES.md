# Architecture Rules

These rules govern refactoring of `celeg-cuda-cpp` toward a multi-model runtime.
They are the binding contract for every change landing after the Phase 0
baseline (see `docs/ARCHITECTURE_RULES.md` section 0.5). A pull
request that violates a rule must either fix the violation or document an
explicit, time-bounded exception in its description.

The rules are enforced by review today and by CI static checks in later phases
(see `docs/ARCHITECTURE_RULES.md` section 17.5).

## R1 — No new LFM-specific type in generic runtime directories

Generic runtime code lives under `include/celeg/runtime/`, `src/runtime/`, and
any future `models/`-neutral boundary. A type name prefixed `Lfm` (for example
`CelegModel`, `CelegModelShape`, `CelegWeights`) must not be introduced there. New
LFM-specific types belong under `src/models/lfm2/` or a private include tree.

## R2 — No CUDA type in backend-neutral model headers

Headers under `include/celeg/model/`, `include/celeg/runtime/`,
`include/celeg/checkpoint/`, `include/celeg/text/`, and `include/celeg/serve/` must
not name CUDA types (`cudaStream_t`, `cudaEvent_t`, `__nv_bfloat16`,
`cublasHandle_t`, `CUstream`, ...). Backend-neutral interfaces use opaque
handles, forward declarations, or backend-neutral value types. CUDA headers
are restricted to `include/celeg/backend/cuda/`.

## R3 — No new `.inl` implementation aggregation

The `.inl` extension is reserved for unavoidable template implementation or
generated code only. A new `.inl` file must not be introduced as a textual
unit-assembly device (the pattern `namespace celeg { #include "x.inl" }` is
forbidden). Existing `.inl` aggregations are being removed; do not add more.

## R4 — No optional interface method that throws "not supported" by default

An interface must not declare an operation that a valid implementation is
allowed to fail with `std::runtime_error("not supported")`. Split capability
into a separate interface (capability discovery by `dynamic_cast` or
composition) so that consumers can require only the capability they use and a
valid implementation never fails merely because an inherited optional is
inapplicable.

## R5 — No architecture switch in backend operator code

Backend operator code (`src/backend/**`, `include/celeg/backend/**`) must not
contain architecture dispatch (`if (architecture == Lfm) ...`,
`switch (model_type) ...`). Architecture-specific scaling, naming, and
topology belong in architecture-owned model programs (Phase 10) or in input
arguments to the operator. The backend learns the model is LFM or Granite only
indirectly through resolved argument values.

## R6 — No new quantized format without quality and performance tests

A new `WeightMode` / `RuntimeWeightEncoding` / `GgufEncoding` value must land
with at least one quality test (synthetic + real checkpoint matrix, RMSE,
cosine similarity, max error) and one performance test (decode and prefill
measured separately) registered in `tests/`. The dispatcher must report the
new encoding truthfully in execution-plan diagnostics.

## R7 — Performance preservation

Refactoring must not silently replace a specialized CUDA path with a generic
abstraction. Architecture resolution, format dispatch, and policy selection
occur during model construction or plan compilation; decode and prefill hot
paths use resolved concrete data structures and direct calls. No per-operator
virtual dispatch inside a layer loop.

## R8 — No backward-compatibility shims

The repository policy is clean replacement, not compatibility. When an
abstraction is structurally wrong, replace it and delete the old path in the
same pull request. A temporary adapter is allowed only when it (a) keeps the
pull request small, (b) has an explicit deletion milestone in this plan, and
(c) is not exposed as public API.

## R9 — Text profiles are data, not central switches

Text and tokenizer code must not branch on architecture or chat-profile enums.
Profiles provide tokenizer configuration and declared chat capabilities through
composition-root data.

## R10 — Neutral contracts do not expose checkpoint formats

Backend-neutral contracts may depend only on repository capabilities and
neutral tensor descriptors, never on a concrete GGUF or Safetensors type.

## R11 — Serving performs inference only

The HTTP serving layer does not execute tools and may not access the filesystem,
spawn subprocesses, or make outbound network calls.
