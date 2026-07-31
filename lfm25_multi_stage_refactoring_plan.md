# Multi-Stage Refactoring Plan for `lfm25-cuda-cpp`

## 1. Purpose

This document defines a staged refactoring plan for transforming `lfm25-cuda-cpp` from an LFM-specific CUDA inference implementation into a maintainable, testable, high-performance, multi-model inference runtime.

The plan consolidates the main findings from the architecture review:

- excessive responsibility concentrated in `LfmModel::Impl`;
- broad, state-leaking interfaces such as `IPackedSession`;
- CUDA types leaking into model and checkpoint layers;
- `.cu`, `.cuh`, and `.inl` files used inconsistently;
- large translation units assembled through textual inclusion;
- host-only code compiled by NVCC;
- source-format-dependent quantization behavior;
- incomplete CUDA INT4/INT8 support for MoE;
- execution plans that do not always describe the kernels actually executed;
- central switches that make new architectures and weight formats expensive to add;
- insufficient test coverage for numerical quality and backend parity;
- architecture-specific configuration and tensor naming embedded in common model structures;
- limited readiness for supporting additional LLM families such as Granite 4.1.

The target is not a generic graph interpreter. The target is a runtime with:

1. reusable backend operators;
2. architecture-specific compiled execution programs;
3. explicit checkpoint, materialization, and quantization boundaries;
4. narrow runtime contracts;
5. no unnecessary virtual dispatch inside hot layer loops;
6. stable performance throughout the migration.

---

## 2. Guiding Principles

### 2.1 Preserve performance by default

Refactoring must not silently replace specialized CUDA paths with generic abstractions.

Architecture resolution, format dispatch, and policy selection should occur during model construction or plan compilation. Decode and prefill hot paths should use resolved concrete data structures and direct calls.

### 2.2 Prefer clean replacement over compatibility shims

The repository policy does not require backward compatibility. When an abstraction is structurally wrong, replace it rather than layering adapters indefinitely.

Temporary adapters are acceptable only when they:

- keep pull requests small;
- have an explicit deletion milestone;
- do not become public API.

### 2.3 Separate architecture from backend

A model architecture defines:

- layer composition;
- tensor roles and naming;
- numerical modifiers;
- topology;
- supported execution patterns.

A backend defines:

- memory allocation;
- kernels;
- GEMM dispatch;
- KV storage;
- synchronization;
- device-specific execution.

The CUDA backend must not know whether an operator is used by LFM, Granite, or another model family.

### 2.4 Separate source encoding from runtime encoding

The following concepts must not be represented by one global enum:

- checkpoint source format;
- source tensor encoding;
- requested loading policy;
- materialized runtime encoding;
- quantization layout;
- kernel implementation;
- KV-cache encoding.

### 2.5 Use interfaces for behavior, not for exposing object internals

An interface containing dozens of getters for buffers, layer vectors, flags, and mutable state is not meaningful decoupling.

Prefer operation-specific input/output contexts and cohesive state owners.

### 2.6 Refactor through verified vertical slices

Each stage must end with:

- passing tests;
- benchmark comparison;
- no unresolved duplicated path;
- a clear deletion of replaced code when feasible.

---

## 3. Target Architecture

A possible final repository structure is:

```text
include/
└── runtime/
    ├── model.hpp
    ├── session.hpp
    ├── generation.hpp
    ├── diagnostics.hpp
    └── architecture.hpp

src/
├── core/
│   ├── checkpoint/
│   │   ├── repository.hpp
│   │   ├── capabilities.hpp
│   │   ├── safetensors/
│   │   └── gguf/
│   ├── model/
│   │   ├── definition.hpp
│   │   ├── dimensions.hpp
│   │   ├── numerics.hpp
│   │   └── tensor_roles.hpp
│   ├── registry/
│   ├── sampling/
│   ├── text/
│   └── runtime/
│
├── models/
│   ├── lfm2/
│   │   ├── architecture.cpp
│   │   ├── config.cpp
│   │   ├── tensors.cpp
│   │   ├── weights.cpp
│   │   ├── cuda_program.cpp
│   │   └── cpu_program.cpp
│   └── granite/
│       ├── architecture.cpp
│       ├── config.cpp
│       ├── tensors.cpp
│       ├── weights.cpp
│       ├── cuda_program.cpp
│       └── cpu_program.cpp
│
├── backend/
│   ├── cuda/
│   │   ├── core/
│   │   │   ├── error.hpp
│   │   │   ├── stream.hpp
│   │   │   ├── event.hpp
│   │   │   ├── device_buffer.hpp
│   │   │   └── cublas_handle.hpp
│   │   ├── operators/
│   │   │   ├── embedding/
│   │   │   ├── linear/
│   │   │   ├── attention/
│   │   │   ├── norm/
│   │   │   ├── rope/
│   │   │   ├── swiglu/
│   │   │   ├── residual/
│   │   │   ├── convolution/
│   │   │   ├── sampling/
│   │   │   └── kv/
│   │   ├── quantization/
│   │   ├── memory/
│   │   ├── runtime/
│   │   └── model/
│   └── cpu/
│       ├── operators/
│       ├── quantization/
│       ├── memory/
│       └── runtime/
│
├── serve/
├── api/
└── app/
```

This structure is illustrative. The important boundaries are:

```text
model architecture
    depends on
backend-neutral model definitions and checkpoint abstractions

backend implementation
    depends on
backend-neutral definitions and architecture-produced execution data

public API
    depends on
generic runtime concepts, not LFM-specific implementation types
```

---

## 4. Refactoring Strategy Overview

The recommended sequence is:

1. establish baselines and safety rails;
2. fix immediate quantization correctness hazards;
3. neutralize public names and package boundaries;
4. clean CUDA translation-unit organization;
5. decompose model state and execution responsibilities;
6. split checkpoint access, materialization, quantization, and cache;
7. introduce backend-neutral model definitions;
8. introduce architecture providers and compiled model programs;
9. migrate LFM onto the new architecture layer;
10. add Granite 4.1 as the second architecture;
11. unify scheduler, serving, and public API surfaces;
12. harden performance, testing, diagnostics, and documentation.

The phases below are designed to be implemented as separate pull requests or small groups of pull requests.

---

# Phase 0 — Baseline, Guardrails, and Refactoring Policy

## Objective

Create a trustworthy baseline so that architectural changes can be evaluated against correctness, quality, memory use, compile time, and throughput.

## Tasks

### 0.1 Capture reference behavior

Record reference outputs for representative models and modes:

- LFM dense BF16;
- LFM dense INT8;
- LFM dense INT4;
- native GGUF Q4_K;
- native GGUF Q6_K;
- LFM MoE BF16;
- MoE with expert residency/offload where supported.

For each case capture:

- selected token sequence under deterministic sampling;
- final logits for a small prompt;
- intermediate layer outputs for one or two selected layers;
- peak GPU memory;
- model load time;
- prefill tokens per second;
- decode tokens per second.

### 0.2 Add benchmark manifests

Store benchmark configuration separately from benchmark results.

Example:

```text
benchmarks/
├── manifests/
│   ├── dense_bf16.json
│   ├── dense_int4.json
│   ├── gguf_q4k.json
│   └── moe_bf16.json
└── README.md
```

### 0.3 Establish numerical thresholds

Define explicit tolerance classes:

| Comparison | Suggested metric |
|---|---|
| Exact deterministic path | exact token sequence |
| BF16 kernel parity | max absolute error and RMSE |
| Quantized linear | cosine similarity, RMSE, max error |
| Full logits | cosine similarity and top-k agreement |
| End-to-end generation | token agreement over fixed prompts |

Thresholds should be model- and format-specific rather than one global tolerance.

### 0.4 Measure compile behavior

Capture:

- clean build time;
- incremental rebuild after modifying one attention kernel;
- incremental rebuild after modifying one model orchestration file;
- object size and final binary size;
- NVCC time per translation unit where available.

### 0.5 Define refactoring rules

Add a short architecture decision document stating:

- no new LFM-specific type in generic runtime directories;
- no CUDA type in backend-neutral model headers;
- no new `.inl` implementation aggregation;
- no optional interface method that throws “not supported” by default;
- no architecture switch in backend operator code;
- no new quantized format without quality and performance tests.

## Deliverables

- baseline report;
- deterministic reference fixtures;
- benchmark manifests;
- numerical acceptance thresholds;
- architecture rules document.

## Exit Criteria

- current supported paths can be compared automatically before and after a refactor;
- at least one CI job runs host tests;
- CUDA CI or a reproducible local CUDA validation procedure is documented;
- benchmark commands are stable and versioned.

---

# Phase 1 — Immediate Quantization Correctness Fixes

## Objective

Remove known correctness hazards before reorganizing the quantization architecture.

## Tasks

### 1.1 Fix MoE router loading

The MoE router must not depend on the global linear quantization mode when the routing kernel expects BF16 or FP32 data.

Implement a dedicated router materialization path:

```cpp
RouterWeight materialize_router_weight(...);
```

Recommended behavior:

- load router as BF16 or FP32;
- explicitly convert once to the runtime representation consumed by routing;
- do not access `LinearWeight::bf16` unless the encoding contract guarantees it exists.

### 1.2 Make MoE quantization support explicit

Until CUDA INT4/INT8 experts are implemented, choose one of these policies:

1. reject INT4/INT8 for MoE models with a clear error; or
2. expose an explicit mixed policy where dense weights are quantized and experts remain BF16.

Do not silently report the whole model as INT4 when the dominant expert weights remain BF16.

### 1.3 Add targeted regression tests

Add tests for:

- safetensors + MoE + INT8;
- safetensors + MoE + INT4;
- router materialization;
- expert encoding reporting;
- unsupported-policy diagnostics.

### 1.4 Correct execution-plan reporting

Ensure diagnostics report the actual runtime encoding and actual selected kernel path.

A plan must not report BF16/cuBLASLt while native GGUF MMQ is executed.

## Deliverables

- router bug fix;
- explicit MoE quantization policy;
- regression tests;
- corrected runtime diagnostics.

## Exit Criteria

- no null BF16 router access under INT4/INT8;
- unsupported combinations fail at model construction, not during decode;
- runtime memory and encoding descriptions match actual storage.

---

# Phase 2 — Public API and Naming Neutralization

## Objective

Stop expanding the LFM-specific naming surface before introducing another model family.

## Tasks

### 2.1 Introduce generic public concepts

Add generic public names:

```text
Model
InferenceSession
ModelDiagnostics
GenerationConfig
ModelOptions
RuntimeMetrics
```

Existing LFM-specific names may temporarily forward to the generic API internally, but new code must use generic names.

### 2.2 Isolate LFM-specific APIs

Move LFM-specific configuration and implementation contracts under:

```text
src/models/lfm2/
```

or a private include tree.

### 2.3 Stop installing internal headers

Do not install:

- `include/lfm/detail/**`;
- internal kernel headers;
- internal `.cuh` files;
- implementation `.inl` files;
- model implementation headers.

Create explicit install lists instead of installing the entire include directory blindly.

### 2.4 Separate public API from backend availability

The generic public API should not require CUDA headers for CPU-only consumers.

Use opaque handles, PImpl, or backend-neutral value types.

## Deliverables

- generic runtime façade;
- reduced install surface;
- internal/private include targets;
- API compilation tests for CPU-only consumers.

## Exit Criteria

- a program can include the public API without CUDA Toolkit headers;
- no public header exposes `LfmModel::Impl`;
- internal CUDA kernel headers are not installed;
- new architecture work does not require adding more `Lfm*` public types.

---

# Phase 3 — CUDA File and Translation-Unit Cleanup

## Objective

Make `.cpp`, `.hpp`, `.cu`, and `.cuh` usage predictable, reduce NVCC compilation scope, and eliminate implementation assembly through `.inl`.

## File Rules

| Extension | Intended use |
|---|---|
| `.cpp` | host orchestration, loaders, cache, model setup, runtime control |
| `.hpp` | host declarations and launcher interfaces |
| `.cu` | CUDA kernels, device functions requiring compilation, launcher definitions |
| `.cuh` | private device helpers and device templates |
| `.inl` | only unavoidable template implementation or generated code |

## Tasks

### 3.1 Convert host-only `.cu` files to `.cpp`

Review every CUDA source file.

Files without any of the following should normally become `.cpp`:

- `__global__`;
- `__device__`;
- `__constant__`;
- kernel launch syntax;
- device template instantiation requiring NVCC.

Likely candidates include:

```text
model/facade.cu
model/loader_cache.cu
model/persistence.cu
model/weight_layout.cu
model/experts.cu
model/setup.cpp
model/configuration.cpp
model/resources.cpp
model/weight_setup.cpp
model/weight_validation.cpp
model/rope.cpp
model/warmup.cpp
model/session_resources.cpp
model/lifecycle.cpp
model/execution.cu
model/decode.cpp
model/prefill.cpp
model/prefill_batched.cpp
model/paged_prefill.cpp
model/prefill_profile.cpp
model/gguf_dequant.cpp
model/weight_upload.cpp
model/linear_loader.cpp
model/expert_loader.cpp
model/packed_execution.cu
moe/host_expert_store.cu
memory/paged_kv.cu
```

Each file must be checked rather than renamed mechanically.

### 3.2 Split CUDA utility headers

Replace broad `utils.cuh` usage with focused host headers:

```text
cuda/core/error.hpp
cuda/core/stream.hpp
cuda/core/event.hpp
cuda/core/device_buffer.hpp
cuda/core/pinned_buffer.hpp
cuda/core/cublas_handle.hpp
```

These are host-side RAII wrappers even though they manage CUDA objects.

### 3.3 Remove namespace-wrapped includes

Eliminate patterns such as:

```cpp
namespace lfm {
#include "attention.inl"
}
```

Every source and header must declare its own namespace explicitly.

### 3.4 Decompose `attention.cu`

Replace the aggregate attention translation unit with focused units:

```text
attention/decode_dense.cu
attention/decode_segmented.cu
attention/decode_paged.cu
attention/decode_batch_ptrs.cu
attention/prefill_online.cu
attention/prefill_segmented.cu
attention/prefill_gemm.cu
attention/prefill_flash.cu
attention/detail/reductions.cuh
attention/detail/online_softmax.cuh
attention/detail/kv_addressing.cuh
```

Shared device helpers should use internal linkage or `__forceinline__` definitions in private `.cuh` files.

### 3.5 Decompose `transform.cu`

Separate unrelated concerns:

```text
linear/
norm/
rope/
convolution/
```

### 3.6 Remove duplicate compiled kernel definitions

Benchmarks should call production launchers rather than include kernel definitions again.

### 3.7 Avoid global separable compilation

Do not enable relocatable device code globally only to support decomposition.

Prefer:

- header-defined `__device__ __forceinline__` helpers;
- kernel and launcher in the same focused `.cu`;
- explicit shared device libraries only when justified.

## Deliverables

- no implementation aggregation through `.inl`;
- focused CUDA translation units;
- host-only files compiled by the host compiler;
- private device helper headers;
- benchmark use of production launchers.

## Exit Criteria

- modifying one attention strategy does not rebuild every unrelated kernel;
- compile-time measurements improve or remain stable;
- binary contains no unintended duplicate kernel variants;
- all kernel source files can be understood without include-order assumptions.

---

# Phase 4 — Model State and Execution Decomposition

## Objective

Replace the distributed god object with cohesive owners and operation-specific execution contexts.

## Tasks

### 4.1 Split immutable model resources from mutable session state

Create:

```cpp
class ModelResources;
class SessionState;
```

`ModelResources` should own immutable or shared data:

- materialized weights;
- topology;
- model numerics;
- tokenizer metadata;
- RoPE tables where precomputed;
- backend execution program;
- shared caches.

`SessionState` should own request/session-local data:

- position;
- phase;
- KV state;
- convolution state;
- sampling state;
- request metrics;
- pending asynchronous decode state.

### 4.2 Create focused executors

Recommended responsibilities:

```text
DecodeExecutor
PrefillExecutor
AttentionExecutor
LinearExecutor or GemmDispatcher
Sampler
CudaGraphManager
ExpertResidencyManager
```

Executors should receive compact context objects rather than a model god object.

### 4.3 Replace `IPackedSession`

Do not reproduce the same problem as several smaller getter interfaces.

Use operation-specific structures:

```cpp
struct PackedSessionLane {
    SessionState* session;
    DecodeBuffers* buffers;
    KvState* kv;
    SamplingState* sampling;
};

struct PackedDecodeBatch {
    std::span<PackedSessionLane> lanes;
    const ModelResources* resources;
    PackedDecodeWorkspace* workspace;
};
```

The packed executor should not mutate arbitrary model internals.

### 4.4 Encapsulate layer state

Do not expose:

```cpp
std::vector<Layer>& layers();
```

Introduce explicit layer-state abstractions:

```cpp
LayerRuntimeState&
SessionState::layer_state(int layer);
```

or typed spans passed to the relevant executor.

### 4.5 Separate metrics collection

Runtime metrics should be updated through a small recorder or event interface rather than exposing a mutable metrics object broadly.

## Deliverables

- separate `ModelResources` and `SessionState`;
- focused decode and prefill executors;
- removal of broad `IPackedSession`;
- explicit packed decode batch structures;
- reduced friend access and mutable getters.

## Exit Criteria

- decode and prefill can be unit-tested with synthetic resources and state;
- no executor needs access to an entire model implementation object;
- session state can be created and destroyed independently from shared weights;
- adding a new executor does not require expanding a god interface.

---

# Phase 5 — Checkpoint Repository Capability Refactor

## Objective

Make checkpoint access contracts true, narrow, and format-neutral.

## Tasks

### 5.1 Split repository capabilities

Replace one interface with optional methods that throw with capability-specific interfaces:

```cpp
class ITensorRepository {
public:
    virtual bool contains(TensorName) const = 0;
    virtual HostTensorView tensor(TensorName) const = 0;
    virtual std::vector<TensorName> names() const = 0;
};

class ILocatableTensorRepository {
public:
    virtual TensorLocator locate(TensorName) const = 0;
};

class IRandomAccessTensorReader {
public:
    virtual void read(
        const TensorLocator&,
        std::span<std::byte>) const = 0;
};
```

Use composition or capability discovery where a repository implements multiple behaviors.

### 5.2 Separate source metadata from runtime types

Checkpoint headers must not expose CUDA types.

Use backend-neutral host representations:

```cpp
struct HostTensorView {
    SourceTensorEncoding encoding;
    TensorShape shape;
    std::span<const std::byte> bytes;
    std::optional<GgufEncoding> gguf_encoding;
};
```

### 5.3 Separate tensor naming from repository implementation

Repositories should expose source tensors.

Architecture-specific naming policies should translate semantic tensor roles into source names.

### 5.4 Isolate format implementations

Keep:

```text
checkpoint/safetensors/
checkpoint/gguf/
```

behind repository interfaces and metadata parsers.

## Deliverables

- truthful repository contracts;
- no default “not supported” methods;
- backend-neutral tensor views;
- architecture-owned tensor naming policies.

## Exit Criteria

- consumers can require only the capability they use;
- no valid interface implementation fails because an inherited optional operation is unsupported;
- GGUF and safetensors repositories can be tested independently of CUDA.

---

# Phase 6 — Weight Loading and Materialization Decomposition

## Objective

Replace the multi-responsibility `WeightLoader` with explicit stages.

## Target Pipeline

```text
checkpoint repository
    ↓
tensor resolution
    ↓
source tensor view
    ↓
materialization policy
    ↓
runtime weight encoding
    ↓
backend allocation
    ↓
model weight set
```

## Tasks

### 6.1 Introduce semantic tensor roles

Define backend-neutral roles:

```cpp
enum class TensorRole {
    TokenEmbedding,
    LanguageModelHead,
    FinalNorm,

    AttentionInputNorm,
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,

    FfnInputNorm,
    FfnGate,
    FfnUp,
    FfnDown,

    ShortConvInput,
    ShortConvKernel,
    ShortConvOutput,

    MoeRouter,
    MoeExpertGate,
    MoeExpertUp,
    MoeExpertDown
};
```

Architecture policies map roles to source names.

### 6.2 Introduce `TensorResolver`

Responsibilities:

- resolve canonical and alternative source names;
- validate shape;
- report missing required tensors;
- support tied tensor aliases.

It must not allocate GPU memory or quantize.

### 6.3 Introduce `WeightMaterializer`

Responsibilities:

- choose runtime encoding from source encoding and policy;
- convert or preserve source representation;
- allocate backend storage;
- return a typed materialized weight.

### 6.4 Introduce `DeviceWeightCache`

Responsibilities:

- cache by device, checkpoint identity, architecture, policy, and residency configuration;
- own cache lifetime explicitly;
- expose metrics and invalidation;
- avoid hidden process-global static state.

### 6.5 Introduce `ExpertWeightLoader`

Responsibilities:

- expert tensor discovery;
- packed expert materialization;
- host residency catalog;
- offload storage preparation.

### 6.6 Replace pointer-soup weight structures

Prefer typed variants:

```cpp
using LinearWeight = std::variant<
    Bf16Weight,
    Int8PerChannelWeight,
    Int4PerChannelWeight,
    Int4GroupwiseWeight,
    GgufSegmentedWeight>;
```

Each variant owns or references only fields valid for that encoding.

## Deliverables

- semantic tensor roles;
- tensor resolver;
- weight materializer;
- explicit weight cache;
- specialized expert loading;
- typed runtime weight variants.

## Exit Criteria

- adding a source format does not require modifying CUDA execution code;
- adding a runtime encoding does not require modifying checkpoint parsers;
- invalid pointer combinations are unrepresentable;
- source naming is not hardcoded inside the materializer.

---

# Phase 7 — Quantization Architecture Redesign

## Objective

Make quantization explicit, consistent across source formats, testable, and extensible.

## Tasks

### 7.1 Define distinct concepts

Recommended definitions:

```cpp
enum class SourceEncoding {
    Bf16,
    F16,
    F32,
    Int8,
    GgufQ4K,
    GgufQ6K
};

enum class RuntimeWeightEncoding {
    Bf16,
    Int8PerOutputChannel,
    Int4PerOutputChannel,
    Int4Groupwise,
    GgufQ4K,
    GgufQ6K
};

enum class WeightLoadPolicy {
    PreserveSource,
    DequantizeToBf16,
    RequantizeInt8,
    RequantizeInt4PerChannel,
    RequantizeInt4Groupwise
};

enum class KvEncoding {
    Bf16,
    Int8PerHead
};

struct QuantizationLayout {
    int bits;
    int group_size;
    ScaleEncoding scale_encoding;
    bool symmetric;
};
```

### 7.2 Make policy component-specific

Replace one global mode with a policy:

```cpp
struct ModelQuantizationPolicy {
    WeightLoadPolicy embedding;
    WeightLoadPolicy attention;
    WeightLoadPolicy dense_ffn;
    WeightLoadPolicy moe_router;
    WeightLoadPolicy moe_experts;
    WeightLoadPolicy lm_head;
    KvEncoding kv_cache;
};
```

Presets may still provide a simple CLI experience:

```text
bf16
int8
int4
native-gguf
balanced
```

But presets must resolve into explicit component policies.

### 7.3 Remove source-format-dependent execution semantics

The same requested policy should produce the same runtime encoding regardless of whether the source is safetensors or GGUF, unless the policy explicitly says `PreserveSource`.

Avoid pipelines such as:

```text
GGUF GPU upload
→ GPU dequantization
→ full CPU copy
→ CPU requantization
→ GPU upload
```

Materialization should happen once in the most appropriate location.

### 7.4 Separate decode and prefill kernels

Use different strategies:

- decode `m = 1`: GEMV-optimized kernels;
- small batch decode: batched GEMV or grouped GEMM;
- prefill `m > 1`: proper tiled quantized GEMM.

Do not use a scalar GEMV-style INT4/INT8 kernel as the default prefill implementation.

### 7.5 Implement groupwise CUDA INT4

The current per-output-row scale format should be explicitly named and retained only if it has a justified use case.

Add groupwise INT4 with:

- group size 32 or 64;
- FP16 or BF16 scales;
- documented packing;
- CPU/CUDA format compatibility where practical.

### 7.6 Make kernel selection encoding-driven

The dispatcher should use the actual materialized weight variant.

The execution plan may select policy and preferred strategy, but it must not pretend that a GGUF segmented weight is BF16.

### 7.7 Quantized MoE roadmap

Implement in this order:

1. explicit mixed policy;
2. quantized expert storage representation;
3. quantized expert decode kernel;
4. quantized expert prefill kernel;
5. offload support for quantized experts;
6. end-to-end quality and residency benchmarks.

## Deliverables

- explicit quantization concepts;
- component-specific policy;
- consistent safetensors/GGUF behavior;
- separate decode and prefill quantized kernels;
- groupwise INT4;
- encoding-driven dispatcher;
- documented MoE quantization status.

## Exit Criteria

- the same runtime encoding behaves consistently across source formats;
- plan diagnostics match actual storage and kernels;
- no unsupported combination silently falls back to a different precision;
- quantized prefill is benchmarked separately from decode;
- quantization quality tests cover real matrices and model logits.

---

# Phase 8 — Backend Operator Library

## Objective

Create reusable backend operators that model architectures can compose without embedding architecture knowledge in the backend.

## CUDA Operators

At minimum:

```text
embedding lookup
scaled embedding
RMSNorm
head RMSNorm if still required
RoPE
GQA decode attention
GQA prefill attention
paged attention
segmented attention
BF16 linear
INT8 linear
INT4 linear
native GGUF linear
SwiGLU
scaled residual add
short convolution
KV store and load
sampling
```

## Tasks

### 8.1 Define narrow argument structures

Instead of launchers with many positional arguments:

```cpp
struct DecodeAttentionArgs {
    TensorView q;
    KvView kv;
    TensorView output;
    AttentionDimensions dimensions;
    AttentionScale scale;
    PositionView position;
    cudaStream_t stream;
};
```

Keep structs trivially readable and avoid owning resources.

### 8.2 Separate operator policy from implementation

Example:

```cpp
enum class AttentionStrategy {
    Online,
    Segmented,
    Paged,
    FlashPrefill,
    GemmPrefill
};
```

The architecture requests semantic attention behavior. The CUDA runtime chooses a compatible implementation based on shape and runtime conditions.

### 8.3 Avoid architecture branches

Backend operator code must not contain:

```cpp
if (architecture == Granite) ...
if (architecture == Lfm) ...
```

Architecture-specific scaling should be provided as input or implemented in the architecture program through reusable scaled operators.

### 8.4 Unify benchmark entry points

All kernel benchmarks should call production operator entry points.

## Deliverables

- reusable CUDA operator modules;
- operation-specific argument structures;
- resolved operator strategies;
- production-backed benchmarks.

## Exit Criteria

- LFM and Granite can call the same GQA, RMSNorm, RoPE, and SwiGLU operators;
- no backend operator includes model-family headers;
- operator tests can run without constructing a full model.

---

# Phase 9 — Backend-Neutral Model Definition

## Objective

Replace the LFM-specific universal config with common model definitions plus architecture-specific specifications.

## Tasks

### 9.1 Extract common dimensions

```cpp
struct TransformerDimensions {
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    int vocab_size;
    int max_context;
};
```

### 9.2 Extract positional encoding specification

```cpp
struct RopeSpec {
    RopeType type;
    double theta;
    std::optional<RopeScalingSpec> scaling;
};
```

### 9.3 Extract numerical behavior

```cpp
struct ModelNumerics {
    float norm_epsilon;
    float embedding_multiplier = 1.0f;
    float attention_multiplier = 0.0f;
    float attention_output_multiplier = 1.0f;
    float residual_multiplier = 1.0f;
    float logits_divisor = 1.0f;
};
```

Use names matching runtime behavior rather than one model’s config terminology.

### 9.4 Extract token metadata

```cpp
struct TokenIds {
    int bos = -1;
    int eos = -1;
    int pad = -1;
};
```

### 9.5 Move architecture-specific fields out of `ModelShape`

LFM-specific fields:

- convolution cache and dimensions;
- layer-type schedule;
- LFM MoE topology.

These belong in:

```cpp
struct LfmArchitectureSpec;
```

Granite-specific fields belong in:

```cpp
struct GraniteArchitectureSpec;
```

### 9.6 Create immutable validated definitions

Model definitions should be validated once during construction and treated as immutable afterward.

## Deliverables

- common model dimensions;
- positional encoding spec;
- numerical behavior spec;
- token metadata;
- architecture-specific spec types;
- immutable validated model definition.

## Exit Criteria

- common definitions contain no `DenseLfm2` or `MoeLfm2` logic;
- Granite configuration does not require adding Granite-only fields to a universal superset struct;
- hot paths read precomputed derived dimensions.

---

# Phase 10 — Architecture Provider and Compiled Program Layer

## Objective

Make model families pluggable without turning the runtime into an interpreted graph engine.

## Interfaces

A possible design:

```cpp
class IArchitectureProvider {
public:
    virtual ~IArchitectureProvider() = default;

    virtual bool supports(
        const CheckpointMetadata&) const = 0;

    virtual ModelDefinition inspect(
        const CheckpointMetadata&) const = 0;

    virtual std::unique_ptr<ITensorNamingPolicy>
    create_tensor_naming(
        const ModelDefinition&) const = 0;

    virtual ArchitectureWeightPlan create_weight_plan(
        const ModelDefinition&) const = 0;

    virtual std::unique_ptr<ICudaModelProgram>
    create_cuda_program(
        const ModelDefinition&,
        const MaterializedModelWeights&,
        CudaRuntimeContext&) const = 0;

    virtual std::unique_ptr<ICpuModelProgram>
    create_cpu_program(
        const ModelDefinition&,
        const MaterializedModelWeights&,
        CpuRuntimeContext&) const = 0;
};
```

### 10.1 Add architecture registry

```cpp
ArchitectureRegistry registry;
registry.register_provider("lfm2", ...);
registry.register_provider("granite", ...);
```

The registry should detect architecture from checkpoint metadata and `model_type`.

### 10.2 Compile architecture-specific programs once

Use concrete program classes:

```text
LfmCudaProgram
GraniteCudaProgram
```

The virtual call may occur at the entry to `decode()` or `prefill()`. Inside the layer loop, use concrete resolved structures.

### 10.3 Keep execution programs thin

Programs should compose backend operators and own architecture-specific loop structure.

They should not reimplement:

- CUDA allocation;
- attention kernels;
- GEMM kernels;
- quantization;
- KV paging;
- sampling.

### 10.4 Define architecture-owned weight sets

Example:

```cpp
struct GraniteLayerWeights {
    NormWeight input_norm;
    AttentionWeights attention;
    NormWeight post_attention_norm;
    SwiGluWeights mlp;
};

struct GraniteWeights {
    EmbeddingWeight embedding;
    std::vector<GraniteLayerWeights> layers;
    NormWeight final_norm;
    LinearWeight lm_head;
};
```

LFM receives its own typed weight set.

## Deliverables

- architecture registry;
- provider contract;
- architecture-specific tensor naming;
- typed architecture weight sets;
- compiled CPU/CUDA model programs.

## Exit Criteria

- adding an architecture does not require editing backend operator switches;
- no per-operator virtual dispatch occurs inside the layer loop;
- architecture-specific state is not stored in a universal optional-field structure.

---

# Phase 11 — Migrate LFM to the New Architecture Layer

## Objective

Prove the new architecture with the existing model family before adding Granite.

## Tasks

### 11.1 Implement `LfmArchitectureProvider`

Responsibilities:

- detect LFM checkpoints;
- parse dense and MoE configurations;
- produce common definition plus `LfmArchitectureSpec`;
- provide LFM tensor naming;
- provide LFM weight plan;
- construct CPU and CUDA programs.

### 11.2 Implement typed LFM weights

Separate:

- dense LFM layer weights;
- convolution layer weights;
- MoE layer weights;
- shared model weights.

### 11.3 Port decode and prefill

Migrate one path at a time:

1. dense BF16 decode;
2. dense BF16 prefill;
3. native GGUF decode;
4. quantized dense decode;
5. quantized dense prefill;
6. MoE BF16;
7. expert offload;
8. packed/concurrent decode.

### 11.4 Delete old paths after parity

Do not keep old and new complete implementations indefinitely.

Each migrated path should remove or clearly deprecate its predecessor.

## Deliverables

- LFM provider;
- LFM typed weight sets;
- LFM CPU/CUDA programs;
- parity report against Phase 0 baselines;
- deletion of obsolete implementation paths.

## Exit Criteria

- all previously supported LFM paths run through the architecture layer;
- no core runtime branch checks for `DenseLfm2` versus `MoeLfm2`;
- performance remains within agreed thresholds;
- quality and deterministic outputs remain within baseline tolerance.

---

# Phase 12 — Add Granite 4.1 as the Second Model Family

## Objective

Validate that the new architecture supports a genuinely different dense transformer without contaminating the backend with model-specific branches.

## Recommended Scope

Start with a smaller Granite 4.1 checkpoint for faster iteration, then validate larger variants.

Initial support target:

- safetensors;
- BF16;
- CUDA decode;
- CUDA prefill;
- tied embeddings where defined;
- standard generation;
- deterministic reference comparison.

Add quantized and CPU paths after BF16 parity.

## Tasks

### 12.1 Implement `GraniteArchitectureProvider`

Responsibilities:

- detect Granite metadata;
- parse Granite configuration;
- produce common dimensions and numerical modifiers;
- supply Granite tensor naming;
- produce Granite weight plan;
- construct Granite execution programs.

### 12.2 Add Granite numerical behavior

Represent, as configuration-derived model numerics:

- embedding scaling;
- attention scaling;
- residual scaling;
- logits scaling;
- normalization epsilon;
- RoPE behavior.

Do not hardcode Granite checks inside generic kernels.

### 12.3 Implement Granite typed weights

```cpp
struct GraniteLayerWeights {
    NormWeight input_norm;
    AttentionWeights attention;
    NormWeight post_attention_norm;
    SwiGluWeights mlp;
};
```

### 12.4 Implement Granite CUDA program

The program should compose:

```text
scaled embedding
RMSNorm
GQA attention
scaled residual add
RMSNorm
SwiGLU
scaled residual add
final RMSNorm
LM head
logits scaling
```

### 12.5 Add reference validation

Compare against a trusted implementation using:

- embedding output;
- first layer norm output;
- first attention output;
- first MLP output;
- final hidden state;
- final logits;
- greedy token sequence.

### 12.6 Add quantization after BF16 parity

Recommended order:

1. BF16;
2. INT8 weights;
3. groupwise INT4;
4. native source-preserving formats if added later;
5. KV INT8;
6. FP8 only after policy and kernel support are explicit.

## Deliverables

- Granite provider;
- config parser;
- tensor naming policy;
- typed weights;
- CUDA program;
- end-to-end tests;
- documented supported formats and limitations.

## Exit Criteria

- Granite support requires no Granite-specific branch in generic CUDA operators;
- serving and generation APIs work without Granite-specific public endpoints;
- LFM performance and correctness remain unchanged;
- adding Granite demonstrates that architecture registration is sufficient.

---

# Phase 13 — Scheduler, Packed Decode, and Session Runtime Unification

## Objective

Make concurrency and batching backend-aware but architecture-neutral.

## Tasks

### 13.1 Define generic request/session lifecycle

Common states:

```text
Created
Prefilling
ReadyForDecode
Decoding
Completed
Cancelled
Failed
```

Architecture programs should not own scheduling policy.

### 13.2 Create backend batch adapters

A scheduler produces logical batches.

Backend adapters transform them into:

```text
CudaDecodeBatch
CudaPrefillBatch
CpuDecodeBatch
CpuPrefillBatch
```

### 13.3 Make packed decode operate on generic model programs

Packed decode should require compatible:

- backend;
- architecture program type;
- model resources;
- runtime encoding;
- shape constraints.

Compatibility must be explicit and validated.

### 13.4 Isolate expert residency

Expert residency is a backend/runtime service used only by architectures with MoE.

The scheduler should not depend on LFM-specific expert types.

## Deliverables

- generic session lifecycle;
- backend batch adapters;
- architecture-neutral packed decode;
- isolated expert residency service.

## Exit Criteria

- scheduler code has no LFM-specific types;
- Granite dense sessions and LFM dense sessions use the same scheduling framework;
- incompatible packed sessions fail before launch with clear diagnostics.

---

# Phase 14 — Serving and C API Neutralization

## Objective

Expose models through one runtime surface without backend- or architecture-specific duplication.

## Tasks

### 14.1 Define one inference service contract

Use one service abstraction for:

- CPU;
- CUDA;
- future backends.

Backend selection should happen at model load or service construction.

### 14.2 Remove duplicate serve logic

Keep shared:

- OpenAI-compatible request mapping;
- tokenization;
- generation lifecycle;
- streaming;
- cancellation;
- usage metrics;
- error mapping.

Backend-specific services should only adapt runtime execution.

### 14.3 Generalize model metadata

Model listing should report:

- architecture;
- backend;
- source format;
- runtime weight encodings;
- context limit;
- capabilities;
- unsupported features.

### 14.4 Evolve the C API

Use generic handles:

```c
runtime_model_t*
runtime_session_t*
runtime_generation_t*
```

Avoid adding more `lfm25_*` symbols for new architectures.

Because backward compatibility is not required, a clean API version replacement is preferable to a permanent compatibility layer.

## Deliverables

- unified inference service;
- shared serving logic;
- generic model metadata;
- generic C API.

## Exit Criteria

- serving Granite requires no new route implementation;
- CPU and CUDA services share protocol code;
- public API names no longer imply one model family.

---

# Phase 15 — Testing Matrix Expansion

## Objective

Make architectural and quantization regressions visible before release.

## Required Test Categories

### 15.1 Operator unit tests

For each backend operator:

- shape validation;
- zero and boundary dimensions;
- residual `beta` behavior;
- deterministic reference comparison;
- multiple head dimensions;
- multiple batch sizes;
- multiple context lengths.

### 15.2 CPU/CUDA parity

Where both implementations exist:

- RMSNorm;
- RoPE;
- attention;
- SwiGLU;
- quantized dot products;
- embedding;
- logits projection.

### 15.3 Quantization quality

For each format:

- synthetic random matrices;
- real checkpoint matrices;
- RMSE;
- cosine similarity;
- max error;
- layer-output error;
- logits error;
- perplexity or evaluation-set degradation where feasible.

### 15.4 Source-format consistency

Compare equivalent safetensors and GGUF paths where representations permit.

### 15.5 Architecture reference tests

For both LFM and Granite:

- config parsing;
- tensor-role resolution;
- weight-shape validation;
- first-layer intermediates;
- final logits;
- deterministic generation.

### 15.6 Concurrency and lifecycle

- session reset;
- cancellation;
- prefix cache reuse;
- packed decode;
- paged KV;
- asynchronous decode;
- memory cleanup;
- cache sharing;
- multi-model process behavior.

### 15.7 Compile-surface tests

- public headers compile without CUDA for CPU-only use;
- internal headers are not required by consumers;
- architecture modules can be compiled independently.

## Exit Criteria

- every supported runtime encoding has quality tests;
- every architecture has reference logits tests;
- every backend has lifecycle tests;
- CI clearly separates host, CPU, CUDA, and integration failures.

---

# Phase 16 — Performance Hardening

## Objective

Recover or exceed baseline performance after structural migration.

## Tasks

### 16.1 Profile by execution phase

Measure separately:

- checkpoint parsing;
- weight materialization;
- model setup;
- prefill;
- decode;
- sampling;
- KV operations;
- expert residency;
- host/device transfers.

### 16.2 Tune dispatch thresholds

Resolve thresholds for:

- GEMV versus GEMM;
- online versus segmented attention;
- paged versus dense KV;
- flash versus GEMM prefill;
- quantized versus BF16 fallback;
- small batch versus grouped execution.

Thresholds should be data-driven and visible in diagnostics.

### 16.3 Reduce model-load traffic

Eliminate unnecessary:

- GPU-to-CPU round trips;
- duplicate BF16 materialization;
- repeated tensor-name resolution;
- repeated scale conversion;
- redundant host copies.

### 16.4 Re-evaluate CUDA Graph capture

After state decomposition, ensure graph capture owns stable buffers and does not require broad mutable model access.

### 16.5 Track compile-time performance

Architectural cleanup should also improve developer iteration time.

## Exit Criteria

- no migrated path exceeds agreed regression limits;
- dispatch diagnostics explain chosen strategies;
- load-time transfers match the intended materialization pipeline;
- incremental CUDA rebuilds are materially smaller than the baseline.

---

# Phase 17 — Documentation, Tooling, and Final Cleanup

## Objective

Make the new architecture understandable and prevent regression into model-specific coupling.

## Tasks

### 17.1 Add architecture documentation

Document:

- core/backend/model dependency direction;
- architecture provider lifecycle;
- tensor-role resolution;
- weight materialization pipeline;
- runtime encoding model;
- execution program model;
- session lifecycle.

### 17.2 Add “How to Add a Model Architecture”

The guide should require:

1. architecture detection;
2. config parser;
3. tensor naming policy;
4. typed weights;
5. CPU/CUDA program as supported;
6. reference tests;
7. supported-format declaration.

### 17.3 Add “How to Add a Quantization Format”

Require:

1. runtime encoding definition;
2. materializer;
3. storage variant;
4. decode kernel;
5. prefill kernel;
6. dispatcher registration;
7. quality tests;
8. performance benchmarks;
9. diagnostics support.

### 17.4 Remove temporary adapters

Search for:

- deprecated LFM public aliases;
- compatibility wrappers;
- old `WeightMode` assumptions;
- architecture switches in generic layers;
- obsolete `.inl` files;
- internal installed headers;
- old implementation paths.

### 17.5 Add static architecture checks

Possible checks:

- forbid CUDA includes in selected core directories;
- forbid model-family includes in backend operator directories;
- forbid `.inl` source aggregation;
- detect public inclusion of `detail/`;
- verify source manifests.

## Exit Criteria

- no temporary adapter remains without an explicit issue;
- documentation matches actual directory and dependency structure;
- contributors can add a model or format without editing unrelated central switches;
- architecture boundary checks run in CI.

---

## 5. Recommended Pull Request Breakdown

Large all-at-once refactors should be avoided. A practical PR sequence is:

### PR 1 — Baseline and regression harness

- benchmark manifests;
- deterministic fixtures;
- numerical comparison utilities;
- compile-time baseline.

### PR 2 — Quantization P0 correctness

- MoE router fix;
- explicit unsupported policy;
- diagnostics;
- tests.

### PR 3 — Public/internal header boundary

- stop installing detail headers;
- generic public façade;
- CPU-only header test.

### PR 4 — CUDA utility split

- split `utils.cuh`;
- convert selected host-only `.cu` files to `.cpp`.

### PR 5 — Attention translation-unit split

- remove attention `.inl` aggregation;
- add private device helper headers.

### PR 6 — Transform translation-unit split

- separate linear, norm, RoPE, and convolution.

### PR 7 — Model resources and session state

- introduce immutable resources and session-local state.

### PR 8 — Replace `IPackedSession`

- explicit packed decode contexts;
- remove state-leaking interface.

### PR 9 — Repository capability split

- truthful checkpoint interfaces;
- no optional throwing operations.

### PR 10 — Loader decomposition

- resolver;
- materializer;
- cache;
- expert loader.

### PR 11 — Runtime weight variants

- typed storage;
- actual-encoding dispatch;
- corrected plans.

### PR 12 — Quantization policy redesign

- component policies;
- source/runtime separation;
- consistent materialization.

### PR 13 — Groupwise INT4 and prefill kernels

- CUDA groupwise format;
- proper prefill GEMM;
- quality and performance tests.

### PR 14 — Backend-neutral model definition

- common dimensions;
- numerics;
- architecture specs.

### PR 15 — Architecture registry and programs

- provider contract;
- tensor naming;
- compiled programs.

### PR 16–18 — LFM migration

- dense;
- quantized/GGUF;
- MoE/offload/packed execution.

### PR 19–21 — Granite 4.1

- config and weights;
- BF16 CUDA execution;
- reference tests and serving.

### PR 22 — Runtime and serve unification

- generic service;
- generic C API;
- metadata.

### PR 23 — Cleanup

- delete adapters;
- rename remaining LFM-specific generic concepts;
- finalize docs and CI checks.

---

## 6. Dependency Map

```text
Phase 0 baseline
    ├── Phase 1 quantization correctness
    ├── Phase 2 API boundary
    └── Phase 3 CUDA file cleanup

Phase 2 API boundary
    └── Phase 4 model/session decomposition

Phase 4 model/session decomposition
    ├── Phase 8 operator library
    └── Phase 13 scheduler unification

Phase 5 repository capabilities
    └── Phase 6 loader/materializer split
            └── Phase 7 quantization redesign

Phase 8 operator library
Phase 9 model definition
Phase 10 architecture provider
    └── Phase 11 LFM migration
            └── Phase 12 Granite support

LFM + Granite architecture layer
    ├── Phase 13 scheduler unification
    ├── Phase 14 serving/API neutralization
    ├── Phase 15 testing expansion
    └── Phase 16 performance hardening

All phases
    └── Phase 17 final cleanup and documentation
```

---

## 7. Risk Register

### Risk 1 — Performance regression from abstraction

**Mitigation**

- resolve architecture and formats during construction;
- avoid per-operator virtual dispatch;
- preserve specialized kernels;
- benchmark each vertical slice.

### Risk 2 — Long-lived duplicate implementations

**Mitigation**

- define deletion criteria in every migration PR;
- migrate one complete path at a time;
- prevent new features from being added to obsolete paths.

### Risk 3 — Quantization quality degradation

**Mitigation**

- introduce real-matrix and logits tests before changing formats;
- report runtime encoding explicitly;
- keep source-preserving native formats separate from requantized formats.

### Risk 4 — Over-generalized model graph

**Mitigation**

- use architecture-specific compiled programs;
- keep generic abstractions at operator and resource boundaries;
- do not build an interpreted graph VM unless a demonstrated need appears.

### Risk 5 — Excessive repository churn

**Mitigation**

- stage directory moves separately from behavior changes where practical;
- use mechanical rename PRs with no logic changes;
- keep commits focused and reviewable.

### Risk 6 — CPU backend becoming second-class

**Mitigation**

- keep model definitions backend-neutral;
- use the same architecture provider to build CPU and CUDA programs;
- add CPU/CUDA parity tests for common operators.

### Risk 7 — Hidden source-format differences

**Mitigation**

- materialization tests must assert actual runtime encodings;
- source format must be visible in diagnostics but must not silently change requested semantics.

### Risk 8 — Granite support contaminates generic layers

**Mitigation**

- require architecture-owned numerical modifiers and tensor naming;
- reject any PR adding `if Granite` to generic operator code without strong justification.

---

## 8. Completion Criteria for the Whole Refactor

The refactoring is complete when all of the following are true.

### Architecture

- LFM and Granite are registered architecture providers.
- Generic runtime code contains no architecture-specific execution branches.
- Model-specific layer composition lives in compiled model programs.
- Common definitions contain no LFM-only optional field collection.

### SOLID

- no god object owns loading, execution, session state, metrics, graph capture, and offload together;
- interfaces expose behavior rather than raw mutable internals;
- optional repository capabilities are represented by separate contracts;
- core abstractions do not depend on CUDA;
- new weight formats and architectures do not require editing many unrelated switches.

### CUDA organization

- host orchestration is compiled as C++;
- `.cu` contains actual CUDA implementation;
- `.cuh` is private device code;
- `.inl` aggregation has been removed;
- attention and transform kernels are split by responsibility;
- benchmarks use production launchers.

### Quantization

- source encoding, runtime encoding, and policy are distinct;
- safetensors and GGUF obey explicit materialization semantics;
- runtime plans describe actual encodings and kernels;
- decode and prefill use appropriate quantized kernels;
- MoE support is explicit and tested;
- quality and memory metrics are available.

### Multi-model runtime

- Granite can be loaded through the same public runtime and serving API;
- adding Granite did not require duplicating scheduler, KV cache, sampling, or serving;
- LFM and Granite share backend operators;
- architecture-specific tensor naming and numerical behavior are isolated.

### Quality

- deterministic reference tests pass;
- numerical thresholds pass;
- no unacceptable throughput regression remains;
- peak memory is understood and reported;
- public headers compile in CPU-only environments;
- architecture boundary checks run in CI.

---

## 9. Recommended Immediate Starting Point

The first implementation cycle should contain three tightly scoped efforts:

### Workstream A — Safety

1. add baseline fixtures and benchmark manifests;
2. fix the MoE router quantization bug;
3. make unsupported MoE quantization policies explicit.

### Workstream B — Build and boundary cleanup

1. stop installing internal headers;
2. split CUDA host utilities;
3. convert obvious host-only `.cu` files to `.cpp`;
4. begin removing `.inl` aggregation.

### Workstream C — Architectural seam

1. introduce `ModelResources` and `SessionState`;
2. replace `IPackedSession` with explicit packed decode contexts;
3. define backend-neutral common model dimensions and numerics.

These workstreams create the minimum safe foundation for the later loader, quantization, and multi-model phases without requiring an immediate rewrite of the entire runtime.

---

## 10. Final Design Test

The strongest design test is the cost of adding a third architecture after Granite.

A well-refactored runtime should require:

```text
src/models/<new-architecture>/
    config parser
    tensor naming
    typed weights
    CPU/CUDA execution program
    reference tests
```

It may require a genuinely new backend operator.

It should not require:

```text
editing every weight loader branch
editing generic scheduler logic
editing serve routes
editing KV cache implementation
editing sampling
adding architecture checks to CUDA kernels
adding more optional fields to a universal ModelShape
expanding a session god interface
```

When this condition is met, the runtime will be structurally prepared for additional families without sacrificing the specialized performance required by modern LLM inference.
