# Shared Kernel Architecture Refactoring Plan

## Status

Proposed implementation roadmap.

Audit baseline: `master` at `3b63eab0bf1807f9a70dc4c1973fd25644bc9ae7`.

This plan follows the architectural rules already established by
`docs/EXTENSIBILITY_REFACTORING_PLAN.md`. It is intentionally narrower: it
focuses on useful reuse between CPU, CUDA, and Metal without erasing the real
execution-model differences between those backends.

The goal is not to make all kernels look object-oriented. The goal is to share
semantic contracts, format knowledge, validation, planning, capability
selection, and non-device-specific codecs while keeping CPU, CUDA, and Metal
hot paths concrete and independently optimizable.

---

# 1. Problem statement

CELEG now has three materially different execution targets:

- CPU, with ISA-selected native functions and function tables;
- CUDA, with kernels launched through grids, blocks, streams, and device-specific
  shared-memory behavior;
- Metal, with pipeline states, command buffers, encoders, threadgroups, and
  Apple GPU-specific execution paths.

These backends should not share a fake universal execution API. Their low-level
execution semantics are legitimately different.

However, they currently duplicate or leak knowledge that is not
backend-specific. The most visible examples are:

1. weight-format concepts such as F32, F16, BF16, Q4_0, Q4_K, Q5_K, Q6_K, and
   Q8_0 are represented independently in backend code;
2. quantization block size, bytes per block, row-byte calculation, and validation
   are re-derived in backend-specific code;
3. Metal currently includes `celeg/backend/cpu/gguf.hpp` and uses CPU GGUF
   dequantization as a convenience path;
4. Metal storage format selection is repeated across loading, embedding,
   matvec, matvec-pair, matmul, matmul-pair, and batch dispatch;
5. Metal shader files are physically split but implicitly depend on helpers
   declared earlier in a CMake-generated concatenated source;
6. `MetalModel::Impl` owns unrelated responsibilities ranging from weight
   loading and pipeline caching to attention, recurrent execution, session
   state, metrics, buffers, and snapshot persistence;
7. the same semantic question — "can this backend execute this operation for
   this weight representation?" — is answered differently in each backend.

The result is a growing edit surface. Adding one new storage format or device
path can require coordinated edits across unrelated backend files even when the
new feature has a small semantic footprint.

---

# 2. Architectural objective

The target layering is:

```text
                 model / compiled program
                          |
                          v
             +-------------------------+
             | shared kernel contracts |
             |-------------------------|
             | weight representation   |
             | operation descriptors   |
             | quantization traits     |
             | capability descriptors  |
             | validation / planning   |
             +------------+------------+
                          |
              +-----------+-----------+
              |           |           |
              v           v           v
             CPU         CUDA        Metal
          function      device      pipelines
           tables       kernels     + encoders
```

The shared layer describes **what** must happen and **what representation** is
being operated on.

Each backend remains responsible for **how** that operation executes.

The architecture should maximize reuse in these areas:

- weight/storage vocabulary;
- quantization block metadata;
- shape and row-layout validation;
- host-side decoding where applicable;
- operation descriptors;
- capability queries;
- lowering/planning decisions;
- backend registration metadata;
- regression tests for format semantics.

The architecture should deliberately avoid forced reuse in these areas:

- CPU SIMD implementation details;
- CUDA grid/block/stream launch behavior;
- Metal encoder/pipeline/threadgroup behavior;
- backend-specific fused kernels;
- backend-specific tuning heuristics;
- per-token virtual dispatch.

---

# 3. Governing rules

## 3.1 No universal `IKernel::run()` abstraction

Do not introduce an abstraction such as:

```cpp
class IKernel {
public:
    virtual void run(...) = 0;
};
```

with CPU, CUDA, and Metal implementations forced behind it.

Such an interface would either become too weak to be useful or leak every
backend's execution details through optional parameters and downcasts.

## 3.2 Interfaces belong at cold boundaries

Runtime polymorphism is acceptable at:

- format/codec registration;
- planning;
- capability discovery;
- backend construction;
- repository/checkpoint boundaries;
- testing seams.

The hot execution path should prefer:

- concrete backend types;
- function tables;
- enums/variants where the domain is intentionally closed;
- pre-bound kernel/pipeline handles;
- static dispatch where practical.

## 3.3 One semantic fact has one owner

Examples:

- Q4_K block size must have one canonical definition;
- a weight's runtime representation must be resolved once;
- a backend must not rediscover storage layout by re-reading checkpoint syntax;
- operation support should have one backend-owned capability source, not
  separate switches in every caller.

## 3.4 Backend isolation is mandatory

A backend must not depend on another backend merely to reuse a codec or utility.

In particular:

```text
Metal -> CPU -> GGUF
```

must become:

```text
             +-> CPU
shared GGUF -+
             +-> Metal
```

The same rule applies to future CUDA/CPU/Metal utility reuse.

## 3.5 Closed-world enums and variants are allowed

A switch is not automatically an OCP violation.

If CELEG intentionally wants a compile-time exhaustive semantic set, an enum or
`std::variant` is appropriate. The problem is duplicated ownership and repeated
switches across unrelated modules, not the existence of every switch.

---

# 4. Target shared vocabulary

Names below are illustrative. Exact naming can be adjusted during
implementation to fit existing CELEG conventions.

## 4.1 Weight representation

Introduce or consolidate a backend-neutral representation equivalent to:

```cpp
enum class WeightFormat {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_0,
};
```

This is a runtime execution/storage representation, not a checkpoint filename
or architecture identifier.

It should replace backend-local enums whose alternatives mean the same thing,
including Metal's current `LinearStorage` where semantically equivalent.

If an existing neutral CELEG type already provides the correct semantics, reuse
and strengthen it rather than creating a competing enum.

## 4.2 Quantization traits

Provide one canonical descriptor per block-quantized format, conceptually:

```cpp
struct QuantizationTraits {
    WeightFormat format;
    uint32_t block_size;
    uint32_t block_bytes;
};
```

The shared API should own:

- block size;
- bytes per block;
- valid row-width constraints;
- row-byte calculation;
- basic storage validation.

Backend-specific kernel availability must not live in this generic trait.

## 4.3 Operation descriptors

Represent the semantic operation independently of launch mechanics.

Example:

```cpp
enum class LinearOperationKind {
    MatVec,
    MatVecPair,
    MatMul,
    MatMulPair,
    Embedding,
    EmbeddingBatch,
};

struct LinearOperation {
    LinearOperationKind kind;
    WeightFormat format;
    uint32_t input_features;
    uint32_t output_features;
};
```

This does not execute anything. It is a value used by planning, validation,
capability discovery, and tests.

## 4.4 Capability descriptors

Each backend should expose a compact backend-owned capability view, for example:

```cpp
struct LinearKernelCapabilities {
    bool matvec = false;
    bool matvec_pair = false;
    bool matmul = false;
    bool matmul_pair = false;
    bool embedding = false;
    bool embedding_batch = false;
};
```

or an equivalent operation-query API.

Capabilities should be resolved before execution whenever possible.

Unsupported paths should fail during planning/setup rather than deep inside a
per-token dispatch when the limitation was statically knowable.

---

# 5. Phase 0 — Lock the architecture with tests

Before moving implementation, add tests that describe the intended dependency
and extension properties.

## 5.1 Shared-format trait tests

For every supported shared quantized representation verify:

- canonical block size;
- canonical bytes per block;
- valid row-width constraints;
- row-byte calculation;
- rejection of malformed storage.

## 5.2 Cross-backend representation test

Verify that CPU, CUDA, and Metal map the same physical weight representation to
the same neutral `WeightFormat` value where support exists.

The test should not require equal backend capability; only equal representation
semantics.

## 5.3 Backend isolation guard

Add or extend an architecture-boundary test that rejects includes such as:

```text
src/backend/metal/** -> include/celeg/backend/cpu/**
src/backend/cuda/**  -> include/celeg/backend/cpu/**
src/backend/cpu/**   -> include/celeg/backend/metal/**
```

except for explicitly documented neutral compatibility headers if any remain.

The preferred end state is zero cross-backend dependency.

## 5.4 Capability consistency tests

For each backend, verify that declared support corresponds to an actual bound
implementation/pipeline.

A capability must not report true while the runtime later discovers that the
kernel name or function pointer is missing.

### Acceptance criteria

- tests capture format semantics before relocation;
- no execution behavior changes;
- CPU/CUDA/Metal suites remain green on their supported hosts.

---

# 6. Phase 1 — Extract neutral GGUF/block-format primitives

This is the highest-priority structural change.

Metal currently reaches into CPU GGUF code to dequantize host-side tensors.
That must be removed.

## 6.1 Move neutral block traits out of CPU

Identify the parts of `celeg/backend/cpu/gguf.hpp` and related implementation
that describe GGUF/GGML block representation rather than CPU execution.

Move them into a neutral module, for example one of:

```text
include/celeg/quantization/ggml.hpp
src/quantization/ggml.cpp
```

or the closest existing neutral checkpoint/weight module.

The neutral module should own only representation/codec concerns.

## 6.2 Extract host-side row decoding

If host-side dequantization is useful to more than CPU, expose it as a neutral
codec operation.

Possible shape:

```cpp
using DecodeRowFunction = void (*)(
    std::span<const std::byte> source,
    std::span<float> destination);

struct WeightCodec {
    WeightFormat format;
    DecodeRowFunction decode_row = nullptr;
};
```

A virtual `IWeightDecoder` is acceptable if it fits an existing codec/catalog
boundary better, but a function table is preferred for this small closed
operation set.

## 6.3 Migrate CPU and Metal consumers

CPU and Metal should both depend on the neutral codec/traits layer.

Delete the Metal include of `celeg/backend/cpu/gguf.hpp`.

### Acceptance criteria

- Metal has zero dependency on CPU backend headers;
- host dequantization numerical output is bitwise/equivalently unchanged within
  existing tolerances;
- CPU GGUF behavior is unchanged;
- shared codec tests cover all formats used by Metal fallback/loading paths.

---

# 7. Phase 2 — Introduce one neutral weight-format representation

Eliminate semantically duplicated storage enums where possible.

## 7.1 Audit existing types

Before adding a new type, inspect existing:

- `TensorDType`;
- block-encoding types;
- GGML/GGUF type enums;
- CPU packed/native representations;
- CUDA storage variants;
- Metal `LinearStorage`.

Choose the smallest representation that means exactly:

> the physical representation presented to a backend linear/embedding kernel.

Do not overload checkpoint source format, semantic tensor role, and execution
storage into one enum.

## 7.2 Migrate Metal `Linear`

Replace:

```cpp
LinearStorage storage;
```

with the neutral representation or a backend-specific variant that contains the
neutral representation as its storage fact.

A Metal weight object may still contain Metal-only state such as:

- `id<MTLBuffer>`;
- rows/cols;
- row bytes;
- pre-bound pipeline information.

## 7.3 Centralize row-byte validation

Remove duplicated literals such as block sizes/type sizes from Metal loading
logic when they are semantic format facts.

### Acceptance criteria

- adding a new neutral block format does not require defining a second Metal-only
  enum alternative with the same meaning;
- format validation is shared;
- backend-specific device state remains backend-specific.

---

# 8. Phase 3 — Add backend kernel registries/function tables

CPU already demonstrates the desired direction with `CpuKernelTable` and
`CpuKernelBackend`.

Apply the same architectural idea, not necessarily the exact type shape, to
CUDA and Metal.

## 8.1 Metal linear kernel registry

Replace repeated storage-to-kernel switches with a table conceptually similar
to:

```cpp
struct MetalLinearKernelSet {
    std::string_view matvec;
    std::string_view matvec_tuned;
    std::string_view matvec_pair;
    std::string_view matmul;
    std::string_view matmul_pair;
    std::string_view embedding;
    std::string_view embedding_batch;
};

struct MetalWeightBackend {
    WeightFormat format;
    LinearKernelCapabilities capabilities;
    MetalLinearKernelSet kernels;
};
```

Do not require strings in the final representation if pre-bound
`MTLComputePipelineState` handles are more appropriate after initialization.

A strong end state is:

```text
format
  -> registry row
  -> capability validation
  -> pipeline binding
  -> execution uses already selected handle
```

rather than repeated runtime switches.

## 8.2 CUDA registry audit

Do not mechanically refactor CUDA merely for symmetry.

Audit whether CUDA currently repeats the same representation/operation dispatch
knowledge across unrelated modules. Where it does, introduce a backend-local
registry or table. Where direct static calls are already clear and localized,
leave them alone.

## 8.3 CPU registry preservation

Keep the current CPU table approach as the reference architecture. Extend it
only where a shared descriptor meaningfully reduces duplication.

### Acceptance criteria

- Metal storage-to-kernel mapping has one owner;
- a missing Metal implementation is detected when binding/planning, not by an
  arbitrary late switch;
- adding one new format normally adds one registry row plus the actual kernels,
  not edits across all dispatch methods;
- no virtual dispatch is added to the per-token path.

---

# 9. Phase 4 — Separate capability planning from execution

Backend code should know whether an operation is supported before entering the
hot execution path.

## 9.1 Define operation support queries

Use the shared `LinearOperation`/operation kind plus backend-local capability
information.

Example conceptual flow:

```text
resolved weight representation
        +
compiled operation
        |
        v
backend capability check
        |
        +-- unsupported -> explicit setup/planning error
        |
        +-- supported -> bind concrete execution path
```

## 9.2 Prefer pre-bound execution descriptors

For Metal, consider a concrete backend-local object such as:

```cpp
struct MetalLinearExecution {
    id<MTLBuffer> weight;
    id<MTLComputePipelineState> matvec_pipeline;
    id<MTLComputePipelineState> matmul_pipeline;
    // geometry/tuning metadata as needed
};
```

The exact representation can remain a variant if different formats need
materially different fields.

The important property is that the hot path should not repeatedly rediscover
which kernel belongs to the storage format.

## 9.3 Keep dynamic tuning backend-local

Environment flags, device-family checks, tensor-core/Metal-tensor capability,
shared-memory thresholds, and tuned-vs-generic decisions remain backend-owned.

### Acceptance criteria

- planning/setup owns support rejection;
- execution paths consume concrete/pre-resolved backend state;
- capability logic is testable independently from device execution where
  practical.

---

# 10. Phase 5 — Reorganize Metal shaders by semantic responsibility

The current `vector.metal`, `state.metal`, `batch.metal`, `pair.metal`, and
`projection.metal` split is useful as an initial extraction but still groups
substantially by execution mode.

Move toward semantic modules such as:

```text
src/backend/metal/kernels/
  common/
    numeric.metal
    quantization.metal
    reduction.metal
  embedding.metal
  linear.metal
  norm.metal
  activation.metal
  rope.metal
  attention.metal
  recurrent.metal
  shortconv.metal
```

Exact file count is not a goal. Cohesion is.

## 10.1 Explicit common shader helpers

Quantization helpers such as:

- BF16/F16 conversion;
- Q4_0 decode;
- Q4_K scale/min and value decode;
- Q5_K scale/min and value decode;
- Q6_K value decode;
- common reductions;

should have an explicit common home.

`pair.metal` and batch kernels must not depend on an accidental prior source
file merely because CMake concatenates files in a particular order.

## 10.2 Organize variants under the operation family

For example, embedding variants should live together conceptually:

```text
embedding
  F32 single
  F16 single
  BF16 single
  quantized single
  batch variants
```

Likewise linear operations can own:

```text
matvec
matvec tuned
matvec pair
matmul
matmul pair
```

This keeps changes to one semantic operation localized.

## 10.3 Make source composition explicit

If runtime MSL compilation still requires generating one large source string,
make dependency composition explicit rather than relying on source-order
visibility accidentally.

Options include:

1. generated explicit common prelude + operation sources;
2. supported MSL include structure with a known include root;
3. build-generated source that documents/imports common declarations once.

Do not duplicate helpers to make files compile independently if that creates a
maintenance fork.

### Acceptance criteria

- helper dependencies are explicit;
- file ordering is not an undocumented semantic requirement;
- one quantization decoding fix has one shader implementation site;
- single/batch/pair variants share the same helper definitions.

---

# 11. Phase 6 — Decompose `MetalModel::Impl`

Do this only after the representation and kernel boundaries are stabilized.
Splitting the large class first would otherwise spread current duplication into
more classes.

Candidate components:

```text
MetalResourceStore
MetalPipelineCache
MetalLinearExecutor
MetalAttentionExecutor
MetalRecurrentExecutor
MetalFeedForwardExecutor
MetalSessionState
MetalWorkspace
MetalExecutionMetrics/Recorder
```

Names are illustrative.

## 11.1 `MetalPipelineCache`

Own:

- library references needed for pipeline creation;
- inference/tensor pipeline lookup;
- pipeline caching;
- pipeline creation errors.

It should not own model semantics.

## 11.2 `MetalResourceStore` / weight loader boundary

Own:

- immutable/private/shared buffer creation;
- raw weight upload;
- device-resident weight accounting;
- zero/state buffer creation if appropriate;
- conversion from resolved weight representation to concrete Metal weight
  objects.

Checkpoint tensor naming must already be resolved before this layer.

## 11.3 `MetalLinearExecutor`

Own:

- matvec;
- matvec pair;
- matmul;
- matmul pair;
- embedding if embedding uses the same weight-kernel registry;
- tuning choice and launch geometry for those operations.

It consumes concrete Metal weight objects and pre-resolved kernel metadata.

## 11.4 Semantic executors

Attention, recurrent, and feed-forward executors should orchestrate Metal
linear/norm/activation primitives without owning unrelated storage or session
transport concerns.

## 11.5 Session/state owner

Generation position, RNG, seen-token state, session snapshot import/export, and
other session-lifetime state should not remain mixed with kernel registry and
pipeline creation.

### Acceptance criteria

- no new virtual dispatch in hot kernels;
- each extracted component has one clear reason to change;
- `MetalModel::Impl` becomes composition/orchestration rather than the owner of
  every implementation detail;
- tests remain behavior-preserving after each extraction.

---

# 12. Phase 7 — Align CPU/CUDA/Metal planning without forcing parity

Once the common vocabulary is stable, align shared planning where that reduces
real duplication.

Examples:

- all backends consume the same neutral weight representation;
- all backends validate block layouts from the same trait source;
- all backends can answer operation capability using the same operation
  descriptor vocabulary;
- shared tests can enumerate representation/operation pairs and compare declared
  capability surfaces.

Do **not** require every backend to support every operation.

A valid matrix may look like:

```text
                 CPU     CUDA    Metal
Q4_K matvec       yes      yes      yes
Q4_K matmul       maybe    yes      yes
FP8 matmul        no       yes      no
AMX int8          yes      n/a      n/a
Metal tensor      n/a      n/a      yes
```

Capability differences are features of the concrete backend, not LSP defects.

### Acceptance criteria

- shared vocabulary does not imply fake capability parity;
- adding a backend-specific optimization does not require changing unrelated
  backend implementations;
- generic planning can reason about support without knowing launch mechanics.

---

# 13. Phase 8 — Extension-cost tests

Add tests that measure architecture, not only numerical correctness.

## 13.1 Synthetic new format test

Where practical, create a lightweight test-only representation/registry entry
that proves adding a new format requires changes only in:

1. the shared format/trait definition;
2. the target backend registry;
3. the target implementation.

It should not require editing unrelated attention, recurrent, or session code.

## 13.2 Synthetic operation capability test

Prove a backend can declare a format unsupported for one operation but supported
for another without introducing special-case branching in generic model code.

## 13.3 Dependency graph guard

Keep automated checks preventing cross-backend includes from reappearing.

## 13.4 Shader helper ownership test/build guard

Ensure Metal shader composition contains common helpers exactly once and that
all required operation modules build with the declared composition.

---

# 14. Suggested implementation sequence

Implement in small, independently verifiable commits.

Recommended order:

1. add format trait and backend-isolation regression tests;
2. extract neutral GGUF/GGML block traits;
3. extract neutral host row codec/dequantization;
4. migrate CPU to neutral codec without behavior change;
5. migrate Metal off `backend/cpu/gguf.hpp`;
6. consolidate neutral runtime `WeightFormat` semantics;
7. migrate Metal `LinearStorage` to the new representation;
8. add Metal format/kernel registry;
9. move `encode_matvec` dispatch to registry/pre-bound descriptors;
10. move matvec-pair dispatch;
11. move matmul/matmul-pair dispatch;
12. move embedding/embedding-batch dispatch;
13. add explicit Metal common shader helpers;
14. reorganize Metal shader families;
15. extract `MetalPipelineCache`;
16. extract Metal resource/weight ownership;
17. extract `MetalLinearExecutor`;
18. extract semantic executors where cohesion clearly improves;
19. add cross-backend capability/extension-cost tests;
20. update architecture documentation and remove obsolete internal paths.

Each commit should leave the tree green. Do not combine a large shader
optimization with a structural move unless the optimization is required to
preserve behavior.

---

# 15. Validation strategy

Structural refactoring is complete only if behavior and performance contracts
remain explicit.

## 15.1 Correctness

Run all relevant existing suites:

- CPU tests;
- CUDA tests;
- Metal tests;
- architecture-boundary tests;
- quantization tests;
- real-model smoke tests already used by the project.

For quantized paths, include numerical comparisons for every migrated format.

## 15.2 Performance

The refactor must not introduce per-token runtime polymorphism or repeated
string-based lookup that did not exist before.

Benchmark at minimum:

- Metal F16/BF16 matvec;
- tuned Metal matvec;
- Q4_0 and K-quant matvec;
- Metal matmul/tensor path;
- embedding single/batch;
- representative decode and prefill workloads.

Any material regression should be investigated before proceeding to the next
phase.

## 15.3 Architecture

Run automated dependency checks verifying:

- no backend includes another backend for reusable semantics;
- checkpoint naming does not leak into kernel dispatch;
- shared code does not include Metal/CUDA/CPU implementation headers;
- neutral format metadata has one owner.

---

# 16. Definition of done

This refactoring is complete when all of the following are true:

1. Metal no longer depends on CPU backend code for GGUF or quantization utility
   behavior;
2. CPU, CUDA, and Metal consume a common backend-neutral vocabulary for physical
   weight representation where semantics are the same;
3. quantization block metadata and row-layout validation have one neutral owner;
4. Metal storage-to-kernel mapping has one backend-owned registry/binding point;
5. adding a new Metal-supported weight format does not require editing separate
   switches in loader, matvec, matmul, embedding, and batch paths;
6. Metal shader helper dependencies are explicit and do not rely on accidental
   concatenation order;
7. `MetalModel::Impl` is reduced to coherent composition/orchestration rather
   than owning every kernel/resource/session concern;
8. capability discovery occurs before execution where support is statically
   knowable;
9. no generic `IKernel::run()` abstraction or hot-path virtual dispatch was
   introduced merely for SOLID compliance;
10. CPU/CUDA/Metal behavior remains correct and representative performance does
    not materially regress;
11. architecture tests prevent cross-backend dependency regressions;
12. future formats such as additional K-quants, IQ formats, FP8, or NVFP4 can be
    added with an edit surface proportional to the actual feature.

---

# 17. Expected result

The desired end state is not "three backends sharing kernel code".

It is:

```text
shared semantics / traits / codecs / planning
                    |
        +-----------+-----------+
        |           |           |
       CPU         CUDA        Metal
        |           |           |
    optimized    optimized    optimized
      native       CUDA        Metal
     kernels      kernels      kernels
```

CELEG should reuse everything that is genuinely the same and specialize
everything that is genuinely hardware-specific.

That boundary gives the project stronger SRP, OCP, ISP, and DIP properties while
preserving the low-level control required for a high-performance inference
runtime.
