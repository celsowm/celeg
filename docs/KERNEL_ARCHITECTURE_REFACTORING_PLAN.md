# Shared Kernel Architecture Refactoring Plan

## Status

Revised implementation roadmap.

Audit baseline: `master` at `724bedc0f5f85facae559793d8fb7f6e2a1b8e1c`.

This plan follows the architectural rules already established by
`docs/EXTENSIBILITY_REFACTORING_PLAN.md`. It is intentionally narrower: it
focuses on useful reuse between CPU, CUDA, and Metal without erasing the real
execution-model differences between those backends.

The goal is not to make all kernels look object-oriented. The goal is to share
semantic contracts, portable encoding knowledge, validation, planning, and
non-device-specific codecs while keeping CPU, CUDA, and Metal resident storage,
kernel binding, tuning, and hot execution paths concrete and independently
optimizable.

This revision incorporates an explicit SOLID + DRY review. In particular, the
architecture must avoid replacing duplicated source code with duplicated
semantic state. A capability, encoding fact, block-layout fact, or kernel
binding must have one authoritative owner; any other view must be derived from
that owner.

---

# 1. Problem statement

CELEG has three materially different execution targets:

- CPU, with ISA-selected native functions and function tables;
- CUDA, with device-resident storage, grids, blocks, streams, cuBLAS/cuBLASLt,
  and custom kernels;
- Metal, with `MTLBuffer`, pipeline states, command buffers, encoders,
  threadgroups, and Apple-GPU-specific execution paths.

These backends should not share a fake universal execution API. Their low-level
execution semantics are legitimately different.

However, the current implementation duplicates or misplaces knowledge in ways
that increase extension cost:

1. weight-encoding concepts such as F32, F16, BF16, Q4_0, Q4_K, Q5_K, Q6_K,
   Q8_0, FP8, and NVFP4 are represented at several layers with overlapping but
   not identical meanings;
2. quantization block size, bytes per block, row-byte calculation, and
   validation are partly re-derived in backend-specific code;
3. Metal currently includes `celeg/backend/cpu/gguf.hpp` and constructs CPU GGUF
   types to perform host-side dequantization;
4. apparently neutral model headers currently contain CUDA types such as
   `__nv_bfloat16`, `__nv_fp8_e4m3`, CUDA execution-plan types, and CUDA-specific
   resident-storage alternatives;
5. `checkpoint/formats/gguf.hpp` currently contains backend capability flags
   (`cpu_dequantize`, `cpu_native_dot`, `cuda_dequantize`, `cuda_native_mmq`),
   making the checkpoint layer an owner of CPU/CUDA execution facts;
6. Metal storage-to-kernel selection is repeated across loading, embedding,
   matvec, matvec-pair, matmul, matmul-pair, and batch dispatch;
7. Metal shader files are physically split but implicitly depend on helpers
   declared earlier in a CMake-generated concatenated source;
8. `MetalModel::Impl` owns unrelated responsibilities ranging from weight
   loading and pipeline caching to attention, recurrent execution, session
   state, metrics, buffers, and snapshot persistence;
9. the semantic question "can this backend execute operation X for encoding Y?"
   is represented in multiple places instead of being derived from one concrete
   binding/catalog owner.

The result is a growing edit surface. Adding one new storage format, operation,
or device path can require coordinated edits across unrelated files even when
the feature itself is small.

---

# 2. Architectural objective

The target dependency direction is:

```text
checkpoint syntax / repository
            |
            v
   resolved tensor semantics
            |
            v
+-------------------------------+
| neutral encoding + traits     |
|-------------------------------|
| portable WeightEncoding       |
| quantization/block traits     |
| row-layout validation         |
| host codecs where applicable  |
| semantic operation descriptors|
+---------------+---------------+
                |
                v
       compiled/planned operation
                |
       +--------+--------+
       |        |        |
       v        v        v
      CPU      CUDA     Metal
       |        |        |
 resident    resident   resident
 storage     storage    storage
 + bound     + bound    + bound
 kernels     kernels    pipelines
```

The shared layer describes **portable facts**: what encoding a weight uses, what
an operation means, how a block format is laid out, and what constraints are
intrinsic to that format.

Each backend owns **backend facts**: how the weight is resident on that device,
which implementation is bound, whether a tuned path exists, launch geometry,
resource requirements, and device capability.

The architecture should maximize reuse in these areas:

- portable weight-encoding vocabulary;
- quantization block metadata;
- shape and row-layout validation;
- host-side decoding where the decoder is genuinely portable;
- semantic operation descriptors;
- planning inputs and validation;
- regression tests for portable format semantics.

The architecture should deliberately avoid forced reuse in these areas:

- CPU SIMD/AMX implementation details;
- CUDA resident storage and CUDA scalar types;
- CUDA grid/block/stream/cuBLAS launch behavior;
- Metal buffers, encoders, pipelines, threadgroups, and tensor paths;
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
backend's execution details through optional parameters, giant context objects,
or downcasts.

## 3.2 Interfaces belong at cold boundaries

Runtime polymorphism is acceptable at:

- format/codec registration;
- planning;
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

- Q4_K block size has one canonical definition;
- a portable encoding is resolved once;
- a backend does not rediscover storage layout by re-reading checkpoint syntax;
- kernel availability is owned by the backend binding/catalog, not copied into a
  separate boolean capability table;
- a pipeline name/handle and a `supports_*` flag must not independently encode
  the same truth.

## 3.4 DRY applies to knowledge, not text

Removing repeated code is useful only when it removes repeated knowledge.

Do not introduce parallel representations such as:

```text
LinearOperationKind list
+ capability bool list
+ kernel-name field list
```

when all three encode the same operation set.

Prefer one authoritative representation and derive secondary views from it.

Likewise, do not introduce a new `WeightEncoding`/`WeightFormat` type if an
existing neutral type already carries exactly the same semantic fact. A new
abstraction must eliminate ambiguity or duplicated ownership, not add another
conversion layer.

## 3.5 Dependency direction is mandatory

Backends may depend on neutral contracts. Neutral code must not depend on
backend implementation types.

Forbidden dependency shapes include:

```text
Metal -> CPU
CUDA  -> Metal
CPU   -> CUDA
neutral model -> CUDA
neutral model -> Metal
checkpoint format -> CPU/CUDA/Metal capability
```

The desired shape is:

```text
                 +-> CPU
neutral contracts+-> CUDA
                 +-> Metal
```

Portable checkpoint/quantization modules must not include CUDA, Metal, or CPU
backend implementation headers.

## 3.6 Impossible states should be unrepresentable

Do not store capability separately from the implementation whose existence
proves the capability.

Avoid representations such as:

```cpp
struct WeightCodec {
    WeightEncoding encoding;
    DecodeRowFunction decode_row = nullptr;
};
```

if a `WeightCodec` semantically means that decoding exists. Prefer a valid codec
value returned through `std::optional`, a catalog lookup, or another sum type.

The same rule applies to backend kernel bindings.

## 3.7 Closed-world enums and variants are allowed

A switch is not automatically an OCP violation.

If CELEG intentionally wants a compile-time exhaustive semantic set, an enum or
`std::variant` is appropriate. The problem is duplicated ownership and repeated
switches across unrelated modules, not the existence of every switch.

## 3.8 Compose independent axes

Operation kind and weight encoding are independent facts.

Do not flatten them unnecessarily into a single object that every consumer must
rewrite. Prefer:

```cpp
struct LinearOperation {
    LinearOperationKind kind;
    uint32_t input_features;
    uint32_t output_features;
};

struct LinearKernelRequest {
    LinearOperation operation;
    WeightEncoding encoding;
};
```

or an equivalent representation.

---

# 4. Target shared vocabulary

Names below are illustrative. Exact naming should follow existing CELEG
conventions and existing neutral types should be reused when their semantics are
already correct.

## 4.1 Portable weight encoding

Introduce or consolidate a backend-neutral value representing the portable
encoding/layout presented to planning and backend lowering. Call it
`WeightEncoding` here for clarity.

Conceptually it may contain values such as:

```cpp
enum class WeightEncoding {
    F32,
    F16,
    BF16,
    GGML_Q4_0,
    GGML_Q4_K,
    GGML_Q5_K,
    GGML_Q6_K,
    GGML_Q8_0,
    FP8_E4M3,
    NVFP4,
};
```

The exact enum is not prescribed. In particular, if `GgmlType`,
`TensorBlockEncoding`, or another existing neutral value is sufficient for a
subset of the domain, reuse it rather than creating a second isomorphic enum.

The important semantic distinction is:

```text
portable encoding != backend resident storage
```

A BF16 checkpoint may be lowered by CUDA into rowwise INT8 storage, kept as BF16
on another CUDA path, converted for CPU, or uploaded as BF16/F32 on Metal. Those
backend-resident representations must not be forced into a universal storage
sum type.

## 4.2 Backend-resident storage stays backend-owned

Examples:

```text
CpuLinearStorage
CudaLinearStorage
MetalLinearStorage
```

may legitimately contain different alternatives and fields.

CUDA types such as `__nv_bfloat16` and `__nv_fp8_e4m3` belong in CUDA-owned
headers. `id<MTLBuffer>` belongs in Metal-owned types. CPU ISA-specific packed
structures belong in CPU-owned types.

The neutral layer may carry portable encoding/layout metadata, but it must not
carry device pointers or backend kernel kinds.

## 4.3 Quantization traits

Provide one canonical descriptor per portable block encoding, conceptually:

```cpp
struct QuantizationTraits {
    WeightEncoding encoding;
    uint32_t block_size;
    uint32_t block_bytes;
};
```

The neutral API should own only intrinsic representation facts:

- block size;
- bytes per block;
- valid row-width constraints;
- row-byte calculation;
- basic packed-storage validation.

Backend-specific kernel availability must not live in this trait.

## 4.4 Semantic operations

Represent operation semantics independently from launch mechanics and encoding.

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
    uint32_t input_features;
    uint32_t output_features;
};
```

If later operations such as grouped matmul or fused projection are added, the
operation algebra can remain intentionally closed and exhaustive.

## 4.5 Capability is derived from backend bindings

Do not keep a parallel struct of capability booleans when the backend binding
already proves whether an implementation exists.

Prefer a shape conceptually similar to:

```cpp
struct MetalKernelBinding {
    id<MTLComputePipelineState> pipeline;
    MetalDispatchGeometry geometry;
};

std::optional<MetalKernelBinding>
bind_linear_kernel(WeightEncoding encoding, LinearOperationKind operation);
```

or a backend-local registry/table indexed by `(encoding, operation)`.

Then:

```cpp
supports(encoding, operation)
```

is derived from whether a valid binding exists.

Tuned/generic alternatives may be represented inside the backend binding as a
backend-local choice. They must not create a second global capability truth.

---

# 5. Phase 0 — Lock dependency and ownership invariants with tests

Before relocating code, make the intended architecture executable.

## 5.1 Portable trait tests

For every shared block encoding verify:

- canonical block size;
- canonical bytes per block;
- valid row-width constraints;
- row-byte calculation;
- rejection of malformed storage.

## 5.2 Encoding semantics tests

Verify that equivalent physical checkpoint encodings map to the same portable
semantic value regardless of the backend that will later consume them.

Do **not** require backend resident storage to be identical.

## 5.3 Cross-backend isolation guard

Reject cross-backend includes such as:

```text
src/backend/metal/** -> include/celeg/backend/cpu/**
src/backend/metal/** -> include/celeg/backend/cuda/**
src/backend/cuda/**  -> include/celeg/backend/cpu/**
src/backend/cuda/**  -> include/celeg/backend/metal/**
src/backend/cpu/**   -> include/celeg/backend/cuda/**
src/backend/cpu/**   -> include/celeg/backend/metal/**
```

unless a narrowly documented platform bridge is explicitly justified.

## 5.4 Neutral-to-backend dependency guard

Also reject the more important inversion:

```text
include/celeg/model/**        -> include/celeg/backend/**
include/celeg/detail/model/** -> include/celeg/backend/**
include/celeg/checkpoint/**   -> include/celeg/backend/**
include/celeg/quantization/** -> include/celeg/backend/**
src/model/**                  -> src/backend/**
src/quantization/**           -> src/backend/**
```

Portable headers must not include CUDA headers or expose CUDA/Metal types in
their public representation.

## 5.5 Capability ownership test

For each backend, verify that operation support is derived from the actual bound
implementation/catalog entry. Do not create a second boolean table merely so it
can be consistency-tested against the binding table.

### Acceptance criteria

- dependency guards fail on the known current inversions;
- tests capture portable encoding semantics before relocation;
- no execution behavior changes;
- CPU/CUDA/Metal suites remain green on their supported hosts.

---

# 6. Phase 1 — Extract neutral GGML/GGUF representation and codec primitives

This is the highest-priority structural change.

## 6.1 Separate GGUF parsing from GGML block semantics

`checkpoint/formats/gguf.hpp` may own GGUF file syntax, metadata parsing, tensor
ordinal decoding, and mapping into a neutral block-encoding value.

Intrinsic GGML block traits and reusable host codecs should live in a neutral
quantization/encoding module, for example:

```text
include/celeg/quantization/ggml.hpp
src/quantization/ggml.cpp
```

or the closest existing neutral weight-encoding module.

## 6.2 Remove backend capability from the checkpoint layer

Delete the current `GgufTypeSupport` ownership pattern from
`checkpoint/formats/gguf.hpp`.

Facts such as:

```text
cpu_dequantize
cpu_native_dot
cuda_dequantize
cuda_native_mmq
```

belong to CPU/CUDA registries or codec catalogs, not to GGUF parsing.

A GGUF type knows its block layout. It does not know which current CELEG backend
has a kernel for it.

## 6.3 Extract portable host row decoding

If a host-side decoder is implementation-independent, expose it as a valid
codec entry, conceptually:

```cpp
using DecodeRowFunction = void (*)(
    std::span<const std::byte> source,
    std::span<float> destination);

struct WeightCodec {
    WeightEncoding encoding;
    DecodeRowFunction decode_row; // always valid for an existing codec
};

std::optional<WeightCodec> weight_codec(WeightEncoding encoding);
```

A virtual decoder interface is acceptable at this cold catalog boundary if it
fits an existing CELEG registry better, but a function table/value is preferred
for a small closed operation set.

## 6.4 Migrate CPU and Metal consumers

CPU and Metal should both depend on the neutral trait/codec layer where the
behavior is genuinely shared.

Delete the Metal include of `celeg/backend/cpu/gguf.hpp` and stop constructing
`CpuGgufMatrix` solely to decode host data.

### Acceptance criteria

- Metal has zero dependency on CPU backend headers;
- `checkpoint/formats/gguf.hpp` contains no CPU/CUDA/Metal execution capability
  flags;
- host dequantization numerical output is unchanged within existing tolerances;
- CPU GGUF behavior is unchanged;
- codec tests cover every portable decoder moved out of a backend.

---

# 7. Phase 2 — Remove backend storage leakage from neutral model types

The current model/detail weight representation must be audited before any new
universal weight enum is introduced.

## 7.1 Fix CUDA leakage in neutral-looking headers

Types containing any of the following are CUDA-owned and must not live in a
portable model header:

- `__nv_bfloat16`;
- `__nv_fp8_e4m3`;
- CUDA execution-plan/kernel-kind types;
- CUDA device pointers;
- CUDA-specific BF16 fallbacks;
- NVFP4/cuBLASLt-specific resident metadata.

Move those representations under the CUDA backend or split the type into:

```text
portable weight description
        +
CUDA resident weight
```

Do not solve the problem by teaching CPU and Metal about CUDA storage variants.

## 7.2 Preserve backend sum types

CUDA may continue to use a variant such as BF16/INT8/INT4/GGUF/FP8/NVFP4 if that
accurately models CUDA-resident alternatives.

CPU and Metal may use different sum types.

The shared architecture requires a common **portable encoding vocabulary**, not
one universal device-storage union.

## 7.3 Keep checkpoint syntax out of resident storage

After planning/lowering, backend resident weight objects should not need tensor
names, GGUF key spellings, architecture names, or repository probing.

### Acceptance criteria

- neutral model/checkpoint headers compile without CUDA headers;
- no neutral type exposes CUDA/Metal/CPU implementation types;
- backend resident storage remains free to model backend-specific optimizations;
- checkpoint naming is resolved before backend storage construction.

---

# 8. Phase 3 — Consolidate portable encoding and row-layout ownership

Only after Phases 1 and 2 should CELEG decide whether a new `WeightEncoding`
type is necessary.

## 8.1 Audit existing neutral representations

Inspect and classify:

- `TensorDType`;
- `TensorBlockEncoding`;
- `GgmlType`;
- packed-checkpoint encoding metadata;
- model weight descriptions;
- backend resident-storage tags.

For each type, document exactly which semantic axis it owns.

## 8.2 Do not create isomorphic enums

If the proposed `WeightEncoding` would merely mirror `GgmlType` for all
quantized cases and `TensorDType` for dense cases, prefer composition or a small
sum/value type over another duplicated enum plus conversion switches.

Any new type must reduce ambiguity and conversion count.

## 8.3 Centralize row-byte validation

Remove duplicated block-size/type-size literals from Metal and other backend
loaders when those values are intrinsic encoding facts.

### Acceptance criteria

- each portable encoding/layout fact has one owner;
- adding a new GGML block format does not require copying block metadata into a
  Metal- or CUDA-only enum;
- format validation is shared where semantics are genuinely identical;
- backend-specific resident state remains backend-specific.

---

# 9. Phase 4 — Add backend kernel registries/bindings without duplicated capability state

CPU already demonstrates the desired direction with `CpuKernelTable` and
`CpuKernelBackend`: implementation presence can itself represent capability.

Apply the same architectural idea, not necessarily the exact type shape, to
Metal and only where useful to CUDA.

## 9.1 Metal linear binding registry

Replace repeated storage-to-kernel switches with one backend-owned mapping from:

```text
(portable encoding, operation)
```

to a concrete binding.

Conceptually:

```cpp
struct MetalKernelBinding {
    id<MTLComputePipelineState> pipeline;
    MetalDispatchGeometry geometry;
    MetalKernelTuning tuning;
};

const MetalKernelBinding* metal_linear_binding(
    WeightEncoding encoding,
    LinearOperationKind operation);
```

The final implementation may use arrays, tables, variants, generated static
metadata, or pre-bound handles. Strings are acceptable during initialization but
should not be repeatedly looked up in the hot path.

Do **not** pair this with a parallel struct such as:

```cpp
struct LinearKernelCapabilities {
    bool matvec;
    bool matmul;
    ...
};
```

when the existence of the binding already answers the same question.

## 9.2 Model tuned/generic alternatives inside the backend binding

F16/BF16 tensor paths, Q4_0 tuned matvec, device-family choices, and environment
flags are Metal-owned choices.

The registry may resolve them at setup time into the final execution binding.
The generic planner only needs to know whether a valid binding exists.

## 9.3 CUDA registry audit

Do not mechanically refactor CUDA merely for symmetry.

Audit whether CUDA repeats the same `(encoding, operation) -> implementation`
knowledge across unrelated modules. Introduce a backend-local registry only
where it reduces real duplicated knowledge.

## 9.4 CPU registry preservation

Keep the current CPU table approach as the reference architecture. Extend it
only where a shared semantic descriptor meaningfully reduces duplication.

### Acceptance criteria

- Metal storage/encoding-to-kernel mapping has one owner;
- capability is derived from actual bound implementations;
- no parallel capability boolean table can drift from the registry;
- adding one new Metal-supported encoding normally changes one registry/binding
  location plus the actual kernels;
- adding one new operation does not require adding parallel fields to multiple
  structs;
- no virtual dispatch is added to the per-token path.

---

# 10. Phase 5 — Separate planning from execution

Backend code should know whether an operation is supported before entering the
hot execution path whenever support is statically knowable.

## 10.1 Keep operation and encoding as independent axes

Conceptual flow:

```text
portable encoding
        +
compiled semantic operation
        |
        v
backend binding lookup
        |
        +-- no binding -> explicit setup/planning error
        |
        +-- binding -> concrete execution descriptor
```

Generic code should not know Metal pipeline names, CUDA kernel kinds, CPU ISA
selectors, or backend launch geometry.

## 10.2 Prefer pre-bound execution descriptors

For Metal, a backend-local weight/execution object may contain:

```text
MTLBuffer
rows / cols / row bytes
selected operation bindings or binding-table row
backend-specific geometry/tuning metadata
```

The exact representation may remain a variant if different resident storage
families need materially different fields.

The important property is that the hot path does not repeatedly rediscover
which kernel belongs to the encoding.

## 10.3 Keep dynamic tuning backend-local

Environment flags, device-family checks, tensor-core/Metal-tensor capability,
shared-memory thresholds, and tuned-vs-generic decisions remain backend-owned.

### Acceptance criteria

- setup/planning owns statically knowable support rejection;
- execution consumes concrete/pre-resolved backend state;
- capability queries are derived from backend binding state;
- generic planning is independent of launch mechanics.

---

# 11. Phase 6 — Reorganize Metal shaders by semantic responsibility

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

Exact file count is not a goal. Cohesion and single ownership are.

## 11.1 Explicit common shader helpers

Quantization helpers such as:

- BF16/F16 conversion;
- Q4_0 decode;
- Q4_K scale/min and value decode;
- Q5_K scale/min and value decode;
- Q6_K value decode;
- common reductions;

should have an explicit common home.

Single/batch/pair kernels must not carry separate copies of the same decoder.

## 11.2 Remove accidental source-order dependencies

`pair.metal` and batch kernels must not depend on a helper merely because another
source file happened to be concatenated first by CMake.

If runtime MSL compilation still requires one large source string, make the
composition explicit through a generated prelude/include structure or another
clear dependency mechanism.

### Acceptance criteria

- helper dependencies are explicit;
- file ordering is not an undocumented semantic requirement;
- one quantization decoding fix has one shader implementation site;
- single/batch/pair variants share the same helper definitions.

---

# 12. Phase 7 — Decompose `MetalModel::Impl`

Do this only after encoding, resident-storage, and kernel-binding boundaries are
stable. Splitting the large class first would spread current duplication into
more classes.

Target responsibilities may include:

```text
MetalBufferFactory
MetalWeightLoader
MetalPipelineCache
MetalLinearExecutor
MetalAttentionExecutor
MetalRecurrentExecutor
MetalFeedForwardExecutor
MetalSessionState
MetalWorkspace
MetalExecutionMetrics/Recorder
```

Names and exact class count are illustrative. The goal is responsibility
ownership, not class proliferation.

## 12.1 `MetalPipelineCache`

Own:

- library references needed for pipeline creation;
- inference/tensor pipeline lookup;
- pipeline caching;
- pipeline creation errors.

It must not own model semantics.

## 12.2 Buffer allocation vs weight loading

Treat these as separate conceptual responsibilities even if initially colocated:

```text
MetalBufferFactory
  -> allocation, storage mode, upload/blit mechanics

MetalWeightLoader
  -> resolved HostTensorView/encoding -> Metal resident weight

MetalWorkspace
  -> mutable execution/state buffers

MetalExecutionMetrics/Recorder
  -> accounting/measurement
```

Avoid creating a new `MetalResourceStore` god object that simultaneously owns
allocation policy, tensor decoding, checkpoint semantics, workspace lifecycle,
and metrics.

## 12.3 `MetalLinearExecutor`

Own:

- matvec;
- matvec pair;
- matmul;
- matmul pair;
- embedding when it shares the same binding system;
- backend tuning choice and launch geometry.

It consumes concrete Metal resident weights and pre-resolved bindings.

## 12.4 Semantic executors

Attention, recurrent, and feed-forward executors should orchestrate Metal
linear/norm/activation primitives without owning unrelated storage, repository,
pipeline-cache, or session-transport concerns.

## 12.5 Session/state owner

Generation position, RNG, seen-token state, snapshot import/export, and other
session-lifetime state should not remain mixed with kernel registry and pipeline
creation.

## 12.6 Explicit dependency DAG

Extracted components must depend toward primitives, not back toward the original
god object. A target dependency shape is:

```text
MetalModel::Impl orchestration
        |
        +-> SessionState
        +-> semantic executors
                |
                +-> LinearExecutor
                +-> PipelineCache
                +-> Workspace
        +-> WeightLoader -> BufferFactory
        +-> Metrics/Recorder
```

`MetalAttentionExecutor` must not gain a pointer to `MetalModel::Impl` merely to
recover convenient access to unrelated state.

### Acceptance criteria

- no new virtual dispatch in hot kernels;
- each extracted component has one clear reason to change;
- dependencies follow the explicit DAG;
- `MetalModel::Impl` becomes composition/orchestration rather than the owner of
  every implementation detail;
- tests remain behavior-preserving after each extraction.

---

# 13. Phase 8 — Align CPU/CUDA/Metal planning without forcing parity

Once the portable vocabulary is stable, align shared planning only where it
reduces real duplicated semantics.

Examples:

- all backends consume the same portable encoding/layout facts where semantics
  are actually the same;
- all backends validate shared block layouts from the same trait source;
- all backends receive the same semantic operation descriptor vocabulary;
- each backend independently lowers `(encoding, operation)` to its concrete
  resident/binding representation.

Do **not** require every backend to support every operation or resident storage.

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
- backend-specific optimizations do not require changing unrelated backends;
- generic planning can reason about support through a narrow backend query
  without knowing launch mechanics;
- backend resident storage remains independently evolvable.

---

# 14. Phase 9 — Extension-cost and DRY architecture tests

Add tests that measure architecture, not only numerical correctness.

## 14.1 Synthetic new encoding test

Where practical, prove that adding a new portable block encoding requires only:

1. the portable encoding/trait definition if the encoding is truly new;
2. the target backend binding/registry entry;
3. the target implementation.

It should not require editing unrelated attention, recurrent, session, or
checkpoint-name code.

## 14.2 Synthetic operation test

Prove a backend can support an encoding for one operation but not another
without introducing special-case branching in generic model code.

Adding an operation should not require adding a bool field to one struct and a
parallel kernel-name field to another.

## 14.3 Dependency graph guard

Keep automated checks preventing both:

- backend-to-backend includes;
- neutral-to-backend includes.

## 14.4 Single-owner assertions

Add focused architecture tests or static checks where useful to prevent:

- backend capability flags from reappearing in checkpoint format modules;
- CUDA types from reappearing in neutral model headers;
- duplicate Metal format-to-kernel switches from reappearing outside the binding
  owner.

## 14.5 Shader helper ownership guard

Ensure Metal shader composition contains common helpers exactly once and all
required operation modules build with the declared composition.

---

# 15. Suggested implementation sequence

Implement in small, independently verifiable commits.

Recommended order:

1. add cross-backend and neutral-to-backend dependency guards;
2. add portable GGML trait/row-layout regression tests;
3. split intrinsic GGML block traits from backend support facts;
4. remove `GgufTypeSupport` backend capability ownership from the checkpoint
   layer;
5. extract portable host row codecs/dequantization;
6. migrate CPU to the neutral codec/trait owner without behavior change;
7. migrate Metal off `backend/cpu/gguf.hpp`;
8. audit and remove CUDA implementation types from neutral model/detail headers;
9. relocate/split CUDA resident `LinearWeight` storage into CUDA ownership;
10. audit `TensorDType`, `TensorBlockEncoding`, `GgmlType`, and proposed
    `WeightEncoding` semantics;
11. introduce a new portable encoding value only if the audit proves it removes
    ambiguity rather than duplicating an existing type;
12. centralize row-byte/block validation;
13. add the Metal `(encoding, operation) -> binding` registry;
14. migrate `encode_matvec` to pre-resolved binding;
15. migrate matvec-pair;
16. migrate matmul/matmul-pair;
17. migrate embedding/embedding-batch;
18. delete parallel Metal capability booleans/switches made redundant by the
    binding registry;
19. add explicit Metal common shader helpers;
20. reorganize Metal shader families without duplicating helpers;
21. extract `MetalPipelineCache`;
22. extract Metal buffer-allocation and weight-loading responsibilities;
23. extract `MetalLinearExecutor`;
24. extract semantic executors and session/workspace ownership where cohesion
    clearly improves;
25. add cross-backend extension-cost/DRY tests;
26. update architecture documentation and delete obsolete compatibility paths.

Each commit should leave the tree green. Do not combine a large kernel
optimization with a structural move unless the optimization is required to
preserve behavior.

---

# 16. Validation strategy

Structural refactoring is complete only if behavior, architecture, and
performance contracts remain explicit.

## 16.1 Correctness

Run all relevant existing suites:

- CPU tests;
- CUDA tests;
- Metal tests;
- architecture-boundary tests;
- quantization tests;
- real-model smoke tests already used by the project.

For quantized paths, include numerical comparisons for every migrated codec or
binding.

## 16.2 Performance

The refactor must not introduce per-token runtime polymorphism, repeated string
lookup, or repeated `(encoding, operation)` resolution that did not exist before.

Benchmark at minimum:

- Metal F16/BF16 matvec;
- tuned Metal matvec;
- Q4_0 and K-quant matvec;
- Metal matmul/tensor path;
- embedding single/batch;
- representative decode and prefill workloads.

Any material regression should be investigated before proceeding to the next
phase.

## 16.3 Architecture

Run automated dependency checks verifying:

- no backend includes another backend for reusable semantics;
- neutral model/checkpoint/quantization code does not include backend
  implementation headers;
- checkpoint naming does not leak into kernel dispatch;
- checkpoint format code does not own backend execution capability;
- portable format metadata has one owner;
- operation capability is derived from backend binding state;
- CUDA/Metal implementation scalar or device types do not leak into neutral
  model contracts.

## 16.4 DRY

During review, distinguish textual duplication from semantic duplication.

The refactor should reduce the number of independent places that answer:

```text
What is this encoding?
What is its block layout?
Can this backend execute this operation?
Which implementation will execute it?
```

Each question should have one authoritative owner at the correct layer.

---

# 17. Definition of done

This refactoring is complete when all of the following are true:

1. Metal no longer depends on CPU backend code for GGUF or quantization utility
   behavior;
2. neutral model/checkpoint/quantization headers do not expose CUDA, Metal, or
   CPU implementation types;
3. GGUF/checkpoint format code contains only format/representation facts, not
   CPU/CUDA/Metal execution capability flags;
4. portable encoding/layout semantics have one authoritative neutral owner;
5. backend-resident storage remains backend-owned rather than being forced into
   one universal device-storage representation;
6. quantization block metadata and row-layout validation have one neutral owner;
7. Metal `(encoding, operation) -> kernel/pipeline` mapping has one backend-owned
   binding point;
8. operation capability is derived from real bound implementations rather than a
   parallel boolean capability table;
9. adding a new Metal-supported encoding does not require editing separate
   switches in loader, matvec, matmul, embedding, and batch paths;
10. adding a new linear operation does not require adding parallel fields to
    multiple capability/kernel structs;
11. Metal shader helper dependencies are explicit and do not rely on accidental
    concatenation order;
12. `MetalModel::Impl` is reduced to coherent composition/orchestration rather
    than owning every kernel/resource/session concern;
13. extracted Metal components follow a one-way dependency DAG and do not depend
    back on `MetalModel::Impl` for unrelated state;
14. statically knowable unsupported paths fail during setup/planning;
15. no generic `IKernel::run()` abstraction or hot-path virtual dispatch was
    introduced merely for SOLID compliance;
16. CPU/CUDA/Metal behavior remains correct and representative performance does
    not materially regress;
17. architecture tests prevent both cross-backend and neutral-to-backend
    dependency regressions;
18. future formats such as additional K-quants, IQ formats, FP8, or NVFP4 can be
    added with an edit surface proportional to the actual feature rather than
    the number of layers that copied its metadata.

---

# 18. Expected result

The desired end state is not "three backends sharing kernel code".

It is:

```text
portable semantics / traits / codecs / planning
                     |
          +----------+----------+
          |          |          |
         CPU        CUDA       Metal
          |          |          |
      CPU storage CUDA storage Metal storage
          |          |          |
      bound native bound CUDA bound Metal
        kernels     kernels    pipelines
```

CELEG should reuse everything that is genuinely the same and specialize
everything that is genuinely hardware-specific.

The central architecture rule is:

> **Share portable knowledge once; derive capabilities from real bindings; keep
> backend storage and execution concrete.**

That boundary gives CELEG stronger SRP, OCP, ISP, DIP, impossible-state safety,
and DRY properties while preserving the low-level control required for a
high-performance inference runtime.
