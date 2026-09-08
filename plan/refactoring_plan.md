# Kernel Refactoring Plan: DRY, SOLID, and Performance-Safe Boundaries

This plan restructures the CUDA, CPU, and Metal kernel layers around clear ownership, explicit capabilities, small interfaces, and proven semantic reuse without introducing abstraction for its own sake.

The objective is not to maximize the number of classes, templates, policies, or interfaces. The objective is to make the kernel architecture easier to extend and reason about while preserving or improving correctness, compile time, binary size, generated code quality, and runtime performance.

## 1. Architectural Principles

### 1.1 Hard repository constraints

All work must preserve the current `AGENTS.md` rules:

- New or updated source documentation uses Doxygen comments only.
- Refactors fully replace old access paths; no backward-compatibility shims.
- No LFM-specific types in generic runtime directories.
- No CUDA types in backend-neutral model/checkpoint headers.
- No new `.inl` aggregation files.
- No optional interface method that defaults to throwing `not supported`.
- No architecture switch statements inside backend operator code.
- No new quantized format without quality and performance coverage.

### 1.2 Performance-safe SOLID interpretation

SOLID is applied as an architectural constraint, not as an object-oriented checklist.

#### SRP: one reason to change

Prefer clear ownership of implementation domains over large files or objects that mix routing, capability discovery, format-specific math, temporary-buffer management, and execution.

SRP does not require one class per responsibility. A small facade plus implementation units and stateless helpers is preferred when no independent object lifetime or state exists.

#### DIP: resolve environment outside execution objects

Hardware detection, ISA selection, backend capability resolution, and architecture selection belong at bootstrap/compiler/runtime construction boundaries.

Execution objects consume already-resolved capabilities. They should not rediscover the host environment while dispatching an operation.

#### ISP: expose narrow backend APIs

Public backend headers must be split by domain so a consumer of normalization, RoPE, attention, convolution, or linear algebra does not need the entire CPU or GPU kernel surface.

Do not replace one umbrella header with another umbrella abstraction.

#### OCP: policies only for genuinely orthogonal variation

Use templates/policies when a dimension varies independently and is naturally substitutable, such as attention bias policy or a stable decode primitive.

Do not create mega-templates that combine unrelated dimensions such as paging, sparsity, KV layout, softmax algorithm, block format, query format, and launch geometry merely to avoid repeated source text.

Specialized kernels remain specialized when specialization is important for code generation or performance.

#### LSP: capabilities must be truthful

An interface must not claim support for an operation that an implementation can only reject at runtime.

A missing kernel is a capability fact, not a Null Object opportunity. Nullable function pointers or explicit capability flags may represent unsupported native operations as long as construction/dispatch validates them before use.

Do not replace `nullptr` capability state with fake kernels that silently change semantics or hide unsupported operations.

### 1.3 DRY rule for kernels

Deduplicate semantics, not merely similar-looking code.

Before extracting shared kernel code, verify that the candidate implementations have the same:

- numerical contract;
- precision and rounding behavior;
- synchronization assumptions;
- memory layout;
- launch geometry requirements;
- error/validation semantics;
- performance-sensitive specialization requirements.

If two kernels only look similar but intentionally differ in one of these dimensions, share smaller primitives instead of forcing them through one abstraction.

---

## Phase 0: Baseline, Inventory, and Refactor Gates

Before moving code, capture the current baseline for the backend being changed.

### 0.1 Required baseline

Run the current verification commands appropriate to the touched backend:

```text
python scripts/dev.py verify --backend cpu
python scripts/dev.py verify --backend cuda
python scripts/dev.py smoke --backend cuda
```

Capture the deterministic reference run used by the repository:

```text
python benchmarks/run_manifest.py benchmarks/manifests/reference_230m.json
```

Where compile-time or binary-size changes are expected, also capture:

```text
python scripts/measure_compile.py
```

For performance-sensitive CUDA work, measure the affected path before changing it using the profiling tools documented in `AGENTS.md`.

### 0.2 Per-slice validation rule

Verification is not deferred to the end of the full refactor.

Every implementation slice must include, in the same slice:

1. source/CMake changes;
2. focused unit or parity tests;
3. backend verification;
4. numerical comparison where math changed;
5. smoke coverage where runtime paths changed;
6. focused benchmark/profile comparison where a hot kernel changed.

A structural cleanup is not complete if it preserves tests but produces a measurable unexplained performance regression.

---

## Phase 1: Canonical Low-Level Primitives

This phase removes duplication that is both clearly semantic and broadly reused before changing higher-level ownership.

### 1.1 Canonical GGUF block layouts

Target:

```text
src/celeg/checkpoint/gguf_blocks.hpp
src/backend/cuda/kernels/gguf.cu
src/backend/cuda/kernels/mmq.cu
```

Current CUDA paths duplicate Q4_K/Q6_K block layouts and related decoding knowledge.

Refactor the checkpoint layer into the canonical definition of serialized GGUF block layout and backend-neutral scale/min decoding primitives.

Requirements:

- Backend-neutral definitions use fixed-width storage types only; do not place `__half`, CUDA intrinsics, or CUDA runtime types in checkpoint headers.
- Represent serialized FP16 fields using their raw fixed-width storage representation where necessary and convert at backend boundaries.
- Add compile-time size/layout assertions for canonical blocks where practical.
- Remove duplicated `BlockQ4K`/`BlockQ6K` declarations from CUDA implementation files.
- Keep backend-specific value conversion and optimized decode operations in backend code when they rely on backend intrinsics.

Acceptance criteria:

- one canonical serialized layout per GGUF block type;
- no backend-specific types leak into checkpoint headers;
- CUDA GGUF and MMQ paths preserve numerical output and performance.

### 1.2 Canonical CUDA reduction primitives

Introduce a small reduction primitive header under the CUDA kernel layer and consolidate genuinely identical warp/block sum/max implementations.

Candidates include existing variants such as:

```text
warp_sum
warp_reduce_sum
block_sum
warp_max
block_max
```

Requirements:

- keep the API minimal and device-only;
- preserve specialized reductions when their synchronization or launch assumptions differ;
- do not route all reductions through a highly generic abstraction;
- validate generated behavior in attention, MMQ/GGUF, normalization, and any other migrated path.

### 1.3 RoPE semantic consolidation

Audit the CUDA RoPE implementations before extracting a common core.

Consolidate only frequency/scaling/indexing logic that is numerically identical across ordinary RoPE, QKV RoPE, and pairing paths.

Requirements:

- resolve any existing boundary-condition drift before sharing the implementation;
- preserve specialized launchers and memory access patterns;
- keep architecture/model policy outside generic backend kernels.

---

## Phase 2: CPU Ownership, DIP, and ISP

### 2.1 Finish `CpuKernelBackend` dependency inversion

`CpuKernelBackend` already centralizes important ISA/kernel capability information. Complete that direction rather than introducing another capability system.

Target architecture:

```text
bootstrap / compiler / runtime construction
                |
                v
      resolve CpuKernelBackend
                |
                v
        CpuLinearEngine
```

Refactor `CpuLinearEngine` so it receives already-resolved kernel capabilities instead of performing host capability detection and backend resolution internally.

Requirements:

- hardware detection occurs at the construction/bootstrap boundary;
- backend resolution has one owner;
- `CpuLinearEngine` consumes a resolved backend/kernel table;
- unsupported native operations remain explicit capability state;
- do not introduce Null Object kernels to mask missing capabilities;
- validate required operations before entering hot dispatch paths.

### 2.2 Decompose `CpuLinearEngine` by implementation responsibility

Target:

```text
src/backend/cpu/kernels/linear.cpp
```

The current implementation mixes storage classification, routing, Q4, INT8, BF16, GGUF, GEMV/GEMM, embeddings, temporary activation quantization, transpose, and sliced operations.

Decompose implementation ownership without creating a class hierarchy unless independent state/lifetime justifies one.

Preferred shape:

```text
linear.cpp                 small facade/orchestration
linear_dispatch.cpp        storage/segment routing
linear_q4.cpp              Q4 operations
linear_int8.cpp            INT8 operations
linear_bf16.cpp            BF16 operations
linear_gguf.cpp            GGUF operations
```

Exact filenames may adapt to the existing build structure, but responsibilities must remain explicit.

Requirements:

- keep `CpuLinearEngine` as a small public facade if that remains the cleanest API;
- use stateless helpers/free functions for format implementations unless object state is real;
- move validation toward construction/preparation boundaries where possible;
- avoid repeated storage classification in hot paths when the storage kind can be established earlier;
- preserve optimized compiler-specific paths.

### 2.3 Split the CPU public kernel surface

Target:

```text
src/celeg/backend/cpu/kernels.hpp
```

Replace the fat umbrella surface with narrow domain headers, for example:

```text
src/celeg/backend/cpu/linear.hpp
src/celeg/backend/cpu/normalization.hpp
src/celeg/backend/cpu/rope.hpp
src/celeg/backend/cpu/attention.hpp
src/celeg/backend/cpu/convolution.hpp
src/celeg/backend/cpu/gated_delta.hpp
src/celeg/backend/cpu/quantization.hpp
```

Requirements:

- migrate all includes in the same refactor; do not leave a compatibility umbrella shim;
- avoid circular ownership between public headers;
- keep implementation-only helper declarations out of the public surface;
- use forward declarations where they genuinely reduce coupling without obscuring ownership.

### 2.4 Compiler-specific Q4/Q8 cleanup

Audit the overlap between:

```text
src/backend/cpu/kernels/quantized_dot.cpp
src/backend/cpu/kernels/quantized_dot_avx2_msvc.cpp
```

Share representation-level primitives and scalar semantics where they are truly common, but do not force GCC/Clang and MSVC SIMD implementations through one abstraction if doing so harms compiler code generation.

Preferred reusable pieces:

- Q4 nibble decode semantics;
- layout constants;
- validation contracts;
- portable reference operations used by tests.

Keep compiler-specific intrinsics isolated when necessary.

---

## Phase 3: CUDA Attention Refactoring

CUDA attention is the highest-risk DRY target because similar source code may encode different numerical or launch contracts.

### 3.1 Classify attention implementations before extracting

Inventory current attention implementations by algorithm rather than filename:

- strict multi-pass softmax;
- online/rescaled softmax;
- paged KV access;
- contiguous KV access;
- block sparse access;
- dynamic sparse access;
- segmented/reduction paths;
- bias variants.

Document which dimensions are genuinely orthogonal and which are algorithm-defining.

### 3.2 Extract only stable shared primitives

Good candidates include:

- shared reductions from Phase 1;
- score post-processing/bias policies;
- small query/head geometry helpers;
- KV addressing helpers when layout semantics are identical;
- numerically identical strict-softmax stages;
- numerically identical online-rescale update primitives.

Avoid a single mega-template such as a universal attention kernel parameterized by every storage, sparsity, softmax, block, paging, and query dimension.

### 3.3 Preserve specialized launchers

Dense, paged, sparse, and segmented launchers may remain distinct even when they share low-level primitives.

Any proposed context object such as `GqaThreadCtx` is acceptable only if it:

- removes repeated invariant calculations;
- does not increase register pressure materially;
- does not obscure memory ownership;
- remains trivially inlinable;
- produces equivalent or better performance.

Acceptance criteria:

- reduced semantic duplication;
- no loss of specialized launch geometry;
- numerical parity for all migrated attention paths;
- no unexplained decode/prefill regression.

---

## Phase 4: Metal Refactoring from the Existing Common Core

The Metal backend already has a meaningful shared base in:

```text
src/backend/metal/kernels/inference/common.metal
```

Do not create a parallel `decode_common.metal` abstraction that duplicates this ownership.

### 4.1 Rationalize `common.metal`

Treat the existing file as the canonical starting point for shared decode primitives and attention infrastructure.

If it becomes too broad, split it by stable responsibility rather than creating overlapping common layers. Possible domains include:

```text
quant_decode.metal
attention_common.metal
math_common.metal
```

Only split when the dependency direction remains obvious and the resulting files are independently coherent.

### 4.2 Extend the existing attention policy model conservatively

The existing `CelegAttentionSpan` and bias-policy approach already represents a good use of OCP because bias is an orthogonal dimension.

Extend this model only for additional dimensions that are demonstrably orthogonal.

Do not replace existing tuned attention variants with a universal template merely to make their source text look uniform.

### 4.3 Matvec consolidation

Audit:

```text
matvec_kernels.metal
matvec_q4k.metal
matvec_q6k.metal
matvec_q80.metal
matvec_dense.metal
```

Separate three concerns:

1. public Metal kernel entrypoints;
2. format-specific decode/math cores;
3. common launch-independent row/tile mechanics.

A `DEFINE_MATVEC_KERNEL` macro is acceptable only when wrappers differ solely by names/types and preserve debuggability. Prefer ordinary shared functions/templates when they remain readable and generate identical code.

Do not merge format-specific cores if their optimal memory access or accumulation strategy differs.

### 4.4 Tensor matmul variants

Do not automatically collapse:

```text
tensor.metal
tensor_q4k_relaxed.metal
tensor_dense_relaxed.metal
tensor_q4k_static_stage128.metal
```

into one policy mega-kernel.

First identify what is truly shared:

- tile indexing;
- staging mechanics;
- epilogue logic;
- common decode primitives;
- synchronization patterns.

Extract those pieces while keeping tuned variants as explicit entrypoints when variant-specific scheduling, resource usage, or specialization is performance-relevant.

### 4.5 Metal build integration

Any shader source split/rename must update `src/backend/metal/CMakeLists.txt` in the same slice.

Validate both shader compilation/runtime pipeline creation and the benchmark path after every topology change.

---

## Phase 5: Final Boundary Cleanup

After the backend-specific phases, perform a repository-wide boundary audit.

### 5.1 Remove dead compatibility and duplicate paths

Verify that each completed refactor deleted the old path rather than leaving:

- duplicate helpers;
- compatibility includes;
- duplicate capability resolution;
- stale launchers;
- dead format-specific wrappers;
- obsolete CMake entries.

### 5.2 Verify SOLID boundaries

The final architecture should satisfy these concrete checks:

#### SRP

- capability resolution has one owner;
- each format implementation has clear ownership;
- execution facades do not own unrelated platform discovery;
- common kernel files contain genuinely common primitives rather than unrelated utilities.

#### DIP

- CPU execution consumes resolved capabilities;
- backend-neutral headers contain no backend runtime types;
- architecture policy does not leak into generic backend operators.

#### ISP

- consumers include only the backend domains they use;
- no new fat kernel umbrella replaces the old one.

#### OCP

- policy abstractions exist only for stable orthogonal variation;
- adding a new bias/decode policy does not require editing unrelated algorithms;
- specialized kernels are not forced through universal templates.

#### LSP

- no implementation advertises an operation it can only reject as unsupported;
- missing native kernels remain explicit capabilities;
- no Null Object hides unsupported execution.

### 5.3 Final verification

Run the full repository verification required by `AGENTS.md`, including all applicable CPU/CUDA tests and CUDA smoke coverage.

Re-run the deterministic manifest and compare against the Phase 0 baseline.

For every performance-sensitive slice, retain the before/after measurement with enough context to explain any material change.

---

## Recommended Implementation Order

Execute the plan in this order to minimize risk and dependency churn:

1. Phase 0 baseline and measurement.
2. Canonical GGUF block layouts.
3. CUDA reduction primitives.
4. RoPE semantic consolidation.
5. `CpuKernelBackend` injection cleanup.
6. `CpuLinearEngine` implementation decomposition.
7. CPU public-header ISP split.
8. CPU compiler-specific Q4/Q8 cleanup.
9. CUDA attention classification and incremental shared-primitives extraction.
10. Metal `common.metal` rationalization.
11. Metal matvec consolidation.
12. Metal tensor shared-primitives extraction.
13. Final boundary/dead-code audit.
14. Full verification and benchmark comparison.

Each numbered item should be independently reviewable and should leave the tree buildable and validated before the next item begins.

## Definition of Done

The refactor is complete when:

- duplicated semantics have one canonical owner;
- backend-neutral headers contain no backend-specific runtime types;
- CPU capability discovery is outside hot execution objects;
- `CpuLinearEngine` is a small facade rather than a multi-format implementation hub;
- CPU consumers no longer depend on a fat kernel umbrella;
- CUDA attention shares primitives without losing specialized algorithms;
- Metal builds on its existing common infrastructure instead of parallel common layers;
- unsupported operations remain explicit capabilities rather than fake implementations;
- old paths and compatibility shims are removed;
- numerical behavior remains within established tolerances;
- verification and smoke tests pass;
- no material performance regression is introduced without an explicit, measured justification.
