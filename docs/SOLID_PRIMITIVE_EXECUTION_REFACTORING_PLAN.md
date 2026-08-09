# CELEG S.O.L.I.D. and Primitive Execution Refactoring Plan

## Purpose

This plan captures the next architectural refactoring cycle for CELEG after the recent move to backend-neutral model resolution, runtime catalogs, descriptor-driven architectures, PIMPL facades, packed execution stages, prefix-cache isolation, and explicit runtime composition.

The current architecture is no longer primarily constrained by model-family coupling. The remaining structural pressure is concentrated in large execution and descriptor translation units that repeatedly interpret the same model primitives across token, chunked-prefill, packed CPU, ordinary CUDA, and packed CUDA paths.

The goal of this plan is therefore not to add another abstraction layer for its own sake. The goal is to make **model primitives the unit of extension**, while preserving CELEG's performance characteristics and keeping hot paths statically composable where appropriate.

---

## Current baseline

### S.O.L.I.D. assessment

Approximate current baseline after the recent architectural refactors:

| Principle | Baseline | Main remaining concern |
|---|---:|---|
| SRP | 6.8/10 | Large orchestrators still own several independent responsibilities |
| OCP | 8.0/10 | New model families are extensible, but new execution primitives still require editing central dispatchers |
| LSP | 8.5/10 | Contracts are generally substitutable; ABI backend option decoding is still an implicit secondary capability |
| ISP | 8.5/10 | Interfaces are focused; tokenizer provider still returns a concrete tokenizer type |
| DIP | 9.0/10 | Neutral model/runtime layers are well isolated from architecture- and backend-specific implementation details |

The macro-architecture is healthy. The next work is mostly about making the implementation structure match that macro-architecture.

### Largest C++/CUDA translation units observed during review

The line counts are diagnostic signals, not pass/fail criteria. A large kernel file can be completely reasonable; a smaller orchestration file can still violate SRP.

| File | Approx. lines | Architectural interpretation |
|---|---:|---|
| `tests/cuda_kernels_test.cu` | 1376 | Test aggregation problem, not production architecture |
| `src/model/descriptor.cpp` | 1206 | High-priority SRP/OCP concern |
| `src/backend/cpu/memory/paged_kv.cpp` | 760 | High-priority SRP concern: storage + NUMA + attention kernels |
| `src/backend/cuda/kernels/mmq.cu` | 722 | Large but potentially cohesive performance kernel; do not split by LOC alone |
| `src/backend/cuda/model/packed_execution.cu` | 672 | Partially decomposed, but still owns too much orchestration |
| `src/backend/cpu/concurrent.cpp` | 644 | Scheduler/lifecycle/metrics/cache/NUMA responsibilities remain concentrated |
| `src/backend/cuda/model/execution.cu` | 635 | Central primitive interpreter; major OCP target |
| `src/checkpoint/downloader.cpp` | 620 | Lower-priority SRP review after the execution work |

---

## Architectural invariants that must not regress

The refactor must preserve the strongest properties already achieved:

1. Backends must not dispatch on `architecture_id`, `architecture_kind`, `model_type`, or equivalent family identity.
2. Backend-neutral model/runtime/checkpoint/text/serve code must not depend on CUDA implementation types.
3. Neutral model/runtime/backend code must not include `celeg/models/<family>/...`.
4. Family-specific checkpoint interpretation belongs at the architecture/import boundary, before backend compilation.
5. `ResolvedModel` remains backend-neutral.
6. Model families that are expressible using the descriptor vocabulary must not reintroduce family-owned execution implementations.
7. Generic caches must not know checkpoint formats, CUDA allocation details, or semantic tensor names such as expert gate/up/down regions.
8. The public C/C++ API must not construct concrete backend services directly when a factory boundary exists.
9. No version-stacked compatibility architecture is to be introduced merely to ease this refactor.
10. No virtual-dispatch requirement is imposed on inner token/layer hot paths. SOLID boundaries may be implemented with variants, templates, compile-time tables, function objects, or compiled programs when they produce better locality and performance.

Extend `scripts/check_architecture_boundaries.py` whenever a structural rule from this plan can be enforced cheaply and deterministically.

---

# Target architecture

## Model resolution remains family-oriented

```text
checkpoint
    |
    v
IArchitecture / descriptor architecture
    |
    v
ResolvedModel
    |- ModelGraph
    |- model policies
    |- WeightPlan
    |- capabilities
    `- provenance
```

A new Qwen/Gemma/LFM/etc. variant that can be expressed with existing primitives should normally require descriptor/import work only, not backend execution changes.

## Backend compilation becomes primitive-oriented

```text
ResolvedModel
    |
    v
BackendCompiler
    |
    +----------------------+----------------------+
    |                                             |
    v                                             v
CompiledCpuProgram                           CompiledCudaProgram
    |                                             |
    +--> primitive programs                      +--> primitive programs
    +--> state plan                               +--> state plan
    +--> workspace plan                           +--> workspace plan
    `--> scheduling capabilities                  `--> scheduling capabilities
```

Execution should consume a compiled program instead of repeatedly rediscovering graph semantics in each hot path.

## Primitive ownership

A primitive should own or contribute the following concepts:

```text
model primitive
    |- semantic spec
    |- validation
    |- state requirements
    `- weight requirements

CPU lowering
    |- compiled primitive representation
    |- token execution
    |- chunk/prefill execution where supported
    `- packed execution where supported

CUDA lowering
    |- compiled primitive representation
    |- token execution
    |- packed execution where supported
    `- workspace/state requirements
```

Candidate directory shape:

```text
include/celeg/model/primitives/
src/model/primitives/

src/backend/cpu/primitives/
src/backend/cuda/primitives/
```

The exact directory names can change during implementation; the dependency direction must not.

---

# Phase 0 - Lock the baseline and make architectural drift observable

## Objectives

Before moving responsibilities, make regression detection strong enough that refactoring cannot silently alter semantics or performance.

## Work

- Record current CPU and CUDA benchmark baselines using the existing benchmark manifests.
- Record representative cases for dense BF16, quantized paths, GGUF Q4_K/Q6_K, MoE, long-context paged attention, packed decode, ragged/chunked prefill, prefix-cache reuse, and recurrent primitives.
- Ensure parity/reference tests exist for the primitives that will be moved.
- Add or retain architecture-boundary checks before source movement begins.

## Performance policy

Refactoring is not an excuse for a persistent performance regression. Use repeated benchmark samples rather than one noisy run. A practical review budget is:

- no unexplained median throughput regression greater than roughly 3%;
- no unexplained p95 TTFT/ITL regression greater than roughly 5%;
- no increase in hot-path allocations;
- no new device allocation or execution-plan compilation inside packed/token hot paths.

A deliberate tradeoff may exceed these numbers only when documented with a benchmark rationale.

## Exit criteria

- Current behavior is reproducibly benchmarked.
- Existing parity tests pass before the first structural change.
- Architecture checks pass from a clean checkout.

---

# Phase 1 - Introduce a tokenizer abstraction

## Problem

`ITokenizerProvider` is an abstraction, but its `create()` contract returns `std::unique_ptr<BpeTokenizer>`. That leaks one concrete tokenizer implementation through the provider boundary and weakens DIP/OCP.

## Target

Introduce a minimal tokenizer contract, for example:

```cpp
class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    virtual std::vector<int32_t> encode(std::string_view text,
                                        bool add_bos = true) const = 0;
    virtual std::string decode(const std::vector<int32_t>& ids,
                               bool skip_special = true) const = 0;
    virtual std::string decode_token(int32_t id,
                                     bool skip_special = true) const = 0;
    virtual std::optional<int32_t> token_id(std::string_view text) const = 0;
    virtual int32_t bos_id() const = 0;
    virtual int32_t eos_id() const = 0;
    virtual int32_t pad_id() const = 0;
};
```

`BpeTokenizer` becomes one implementation and `ITokenizerProvider::create()` returns `std::unique_ptr<ITokenizer>`.

## Constraints

- Do not force virtual calls into per-symbol BPE internals; the interface is at the tokenizer object boundary.
- Preserve current tokenizer behavior byte-for-byte/token-for-token.
- Do not introduce speculative tokenizer implementations merely to prove extensibility; a fake tokenizer test is enough.

## Exit criteria

No provider interface or neutral consumer needs to name `BpeTokenizer`.

---

# Phase 2 - Split `descriptor.cpp` by responsibility

## Problem

`src/model/descriptor.cpp` currently mixes JSON parsing helpers, field/default resolution, probe parsing/matching, textual enum/role codecs, position/attention/layer-schedule decoding, descriptor validation, metadata interpretation, topology construction, graph construction, and architecture integration.

## Proposed decomposition

```text
src/model/descriptor/
    parser.cpp
    field_resolver.cpp
    probe.cpp
    role_codec.cpp
    topology_builder.cpp
    graph_builder.cpp
    architecture.cpp
```

Possible internal contracts include `DescriptorParser`, `DescriptorFieldResolver`, `DescriptorProbe`, `DescriptorTopologyBuilder`, `DescriptorGraphBuilder`, and `DescriptorArchitecture`. Prefer small free-function modules where a stateful class provides no value.

## Rules

- Parsing JSON must not build backend data.
- Probe logic must not construct graph/weights.
- Graph construction should consume resolved semantic data rather than repeatedly reading raw JSON keys.
- Role string decoding has one semantic owner.
- Descriptor loading has one validation boundary with actionable errors.
- Preserve the current descriptor format unless a semantic change is genuinely needed.

## Exit criteria

- `src/model/descriptor.cpp` is removed or reduced to a small façade/composition unit.
- Adding a descriptor field affects the smallest relevant parser/resolver/builder module.
- Descriptor and architecture-resolution tests remain green.

---

# Phase 3 - Make primitive weight requirements composable

## Problem

`descriptor_weight_plan.cpp` centrally enumerates tensor requirements for Attention, latent/external attention, ShortConv, GatedDeltaNet, Mamba2, MoE, dense FFN, MLP-only and per-layer inputs.

## Target

Move primitive-specific weight planning behind primitive-owned functions/builders, conceptually:

```cpp
append_attention_weight_requirements(...);
append_short_conv_weight_requirements(...);
append_gated_delta_net_weight_requirements(...);
append_mamba2_weight_requirements(...);
append_dense_ffn_weight_requirements(...);
append_moe_weight_requirements(...);
```

The central planner orchestrates common layer concerns and delegates primitive details.

Do **not** make checkpoint tensor names part of execution primitives. A primitive describes semantic roles and expected shapes; naming remains the naming-policy/binding responsibility.

## Exit criteria

- A new primitive does not enlarge one central weight-plan conditional.
- Tensor naming remains decoupled from execution semantics.
- Weight-plan tests cover primitive requirements independently.

---

# Phase 4 - Define a canonical primitive and compiled-layer vocabulary

## Problem

The same semantic decisions are rediscovered in CPU token, CPU chunk, CPU packed, CUDA ordinary and CUDA packed execution.

## Target

Establish one canonical model primitive vocabulary and backend-specific lowering. The representation may remain variant-based rather than virtual, for example:

```cpp
using MixerSpec = std::variant<
    AttentionSpec,
    ShortConvolutionSpec,
    GatedDeltaNetSpec,
    Mamba2Spec,
    MlpOnlySpec
>;
```

A compiled backend layer should contain already-lowered decisions needed at runtime, never architecture-family identity.

Compile once where feasible:

- mixer kind;
- chunk capability;
- state ownership/sharing;
- attention layout and KV owner mapping;
- FFN kind;
- operator widths;
- packed support;
- state/workspace footprints;
- backend policy choices stable for a session/execution plan.

## Avoid

- virtual dispatch for every layer/token unless benchmarked and justified;
- family-name-based variants;
- a single giant `PrimitiveKind` switch merely moved elsewhere;
- raw checkpoint metadata in compiled programs.

## Exit criteria

The code and documentation clearly distinguish model semantic primitive, backend lowering/compiled primitive, and runtime primitive execution.

---

# Phase 5 - Refactor CPU primitive execution

## Problem

CPU token, chunk and packed paths contain parallel interpretations of the same model semantics.

## Target structure

```text
CpuCompiledProgram
    |
    v
CpuLayerExecutor / static primitive dispatch
    |
    +-- attention
    +-- short convolution
    +-- gated delta net
    +-- mamba2
    +-- mlp-only
    `-- feed-forward
         +-- dense
         `-- moe
```

Token, chunk and packed pipelines share primitive semantic ownership even when numerical implementations remain optimized separately.

## Work

- Extract token attention execution from `model_forward_token.cpp`.
- Extract recurrent/mixer execution into primitive modules.
- Extract common residual/norm/FFN sequencing into a small layer pipeline.
- Make chunk capability explicit in the compiled layer.
- Route `forward_chunk()` through compiled semantics.
- Route packed CPU execution through the same primitive ownership model.
- Remove duplicated query-gate, latent/external-attention, MoE-vs-dense and recurrent decision trees where they can be lowered once.

## Performance constraints

- Preserve GEMM batching.
- Preserve native GGUF activation reuse.
- No per-token heap allocation.
- No virtual dispatch inside numerical inner loops.
- Keep specialized token/chunk/packed implementations when forced sharing would reduce throughput.

## Exit criteria

A new CPU primitive normally requires its semantic definition, CPU lowering/state/workspace, optimized execution implementations and focused tests; it should not require editing every CPU execution path.

---

# Phase 6 - Separate CPU paged storage from paged attention

## Problem

`src/backend/cpu/memory/paged_kv.cpp` currently combines aligned allocation, free-page management, refcounts, NUMA placement, cloning/COW, FP32/BF16 access, AVX2/FMA attention code, online softmax and attention pattern/bias handling.

## Target

```text
src/backend/cpu/memory/
    kv_page_layout.cpp
    kv_page_pool.cpp
    kv_page_copy.cpp

src/backend/cpu/primitives/attention/
    paged_attention.cpp
    paged_attention_avx2.cpp
```

The page pool exposes storage access primitives; attention consumes them.

## Rules

- NUMA placement stays with allocation/storage.
- AVX2/FMA online attention stays with attention kernels.
- COW/page ownership stays with storage/cache.
- Do not introduce abstraction calls in the innermost dot-product loop.

## Exit criteria

`CpuKvPagePool` implementation contains no online softmax, attention scoring, attention bias or SIMD attention-kernel implementation.

---

# Phase 7 - Refactor ordinary CUDA execution into primitive executors

## Problem

`src/backend/cuda/model/execution.cu` remains a central primitive interpreter. `enqueue_decode_forward()` branches through attention subtypes, KV formats, attention algorithms, gating, recurrent mixers and FFN choices.

## Target

```text
CudaDecodePipeline
    |
    v
CudaLayerExecutor
    |
    +-- CudaAttentionExecutor
    +-- CudaGatedDeltaNetExecutor
    +-- CudaShortConvExecutor
    +-- CudaMamba2Executor
    +-- CudaMlpOnlyExecutor
    +-- CudaDenseFfnExecutor
    `-- CudaMoeExecutor
```

`execution.cu` becomes orchestration.

Attention itself may decompose internally into projection, position/RoPE, KV/state store, attention algorithm policy, query gate and output projection. Stable policy choices should be lowered once where possible rather than repeatedly expressed through nested conditionals.

## Exit criteria

- `execution.cu` is primarily pipeline orchestration.
- A new CUDA primitive does not add a large branch body to the central decode loop.
- CUDA phase profiling and graph-capture assumptions remain valid.

---

# Phase 8 - Finish packed CUDA decomposition

## Current positive state

Packed CUDA already delegates important work to dedicated attention, convolution, GatedDeltaNet, MoE and dense-FFN executors plus pipeline, metadata and workspace components.

## Remaining issue

`PackedDecodeExecutorImpl` still owns session/batch validation, compatibility policy, metadata cache lifecycle, metadata staging, attention batch planning, segmented workspace preparation, embedding dispatch, layer/FFN dispatch, pipeline ownership and metrics.

## Target

```text
PackedExecutor
    |- PackedBatchValidator
    |- PackedCompatibilityPolicy
    |- PackedMetadataManager
    |- PackedAttentionBatchPlanner
    |- PackedWorkspace
    |- PackedLayerExecutor
    |- PackedDecodePipeline
    `- PackedPrefillPipeline
```

Extract only responsibilities with independent invariants/state/tests; do not create forwarding-only classes.

## Exit criteria

- `PackedDecodeExecutorImpl` becomes a composition/orchestration object.
- Validation, metadata lifecycle and layer execution can be tested separately.
- Packed decode and ragged-prefill throughput do not regress materially.

---

# Phase 9 - Decompose `CpuSchedulerDriver`

## Problem

`src/backend/cpu/concurrent.cpp` still combines engine/worker lifecycle, request transitions, admission, priority ordering, prefill/decode planning, NUMA placement, prefix-cache coordination, batch execution, cancellation/failure handling and metrics.

## Target

Potential internal decomposition:

```text
CpuSchedulerDriver
    |- CpuAdmissionController
    |- CpuBatchScheduler
    |- CpuRequestLifecycle
    |- CpuPrefixCoordinator
    |- CpuNumaPlacement
    `- CpuMetricsCollector
```

## Required boundaries

- planning does not mutate request state;
- request lifecycle owns legal state transitions;
- metrics consume events/outcomes instead of being scattered through branches;
- NUMA placement is a focused policy;
- prefix coordination does not own priority semantics;
- batch execution returns structured outcomes consumed by lifecycle/metrics.

## Exit criteria

The scheduler loop reads as admission -> plan -> execute -> apply outcomes rather than implementing all concerns inline.

---

# Phase 10 - Split the CUDA kernel test mega-file

`tests/cuda_kernels_test.cu` is not a production SOLID violation, but it harms test discoverability and primitive ownership.

Suggested split:

```text
tests/cuda/
    attention_test.cu
    rope_test.cu
    quantization_test.cu
    sampling_test.cu
    moe_test.cu
    recurrent_test.cu
    mmq_test.cu
    embedding_test.cu
```

Keep shared CUDA test assertions/helpers in support headers.

## Exit criteria

A developer adding a primitive can find its tests without navigating a 1300+ line omnibus file.

---

# Phase 11 - Slim `RuntimeTopology`

## Problem

`RuntimeTopology` has accumulated attention, recurrent, MoE, MLP-only, per-layer-input, shared-KV, MTP and allocation-oriented fields. `ModelGraph` is already intended to be the canonical semantic owner.

## Direction

Move toward:

```text
ResolvedModel
    |- ModelGraph
    |- ModelPolicies
    |- WeightPlan
    |- capabilities
    `- provenance
```

Backend compilation derives:

```text
CompiledBackendProgram
    |- layer programs
    |- workspace plan
    |- state plan
    |- allocation maxima
    `- scheduling capabilities
```

## Migration

Classify current topology fields as canonical model policy, graph-derived semantics, backend allocation convenience data, or obsolete/redundant data. Move one category at a time with parity tests rather than deleting `RuntimeTopology` wholesale.

## Exit criteria

- `ModelGraph` is the single owner of per-layer semantic schedule.
- Backend allocation maxima live in backend compilation products when appropriate.
- New primitives do not require unrelated global fields merely to expose data to executors.

---

# Phase 12 - Clarify backend ABI option capabilities

## Problem

`IBackendFactory` is enough for runtime registration while the C API additionally expects `IBackendOptionsDecoder`, discovered with `dynamic_cast`.

## Options

### Explicit ABI backend interface

```cpp
class IAbiBackendFactory : public IBackendFactory {
public:
    virtual std::shared_ptr<const IBackendOptions>
    decode_options(std::span<const std::byte>) const = 0;
};
```

### Composition

Register independent `BackendFactory` and `BackendOptionsCodec` capabilities. Prefer composition if multiple front-end configuration formats are expected later.

## Exit criteria

The C API no longer relies on an undocumented assumption that a selected factory also happens to implement an additional interface.

---

# Phase 13 - Review lower-priority large translation units

## `src/backend/cuda/kernels/mmq.cu`

Do **not** split solely because it is large. Split only when independent kernel families or duplicated dispatch policy are confirmed. CUDA locality, template specialization, compilation behavior and generated-code quality take precedence over arbitrary LOC limits.

## `src/checkpoint/downloader.cpp`

Review independently for transport, retry/resume, model/repository resolution, local cache/path policy, progress reporting and validation/checksum responsibilities. Extract only when distinct change reasons are confirmed.

---

# Cross-cutting DRY rules

## One semantic owner

Each of these should have one semantic owner:

- primitive kind/spec;
- primitive tensor requirements;
- state requirements;
- chunk/packed capability declaration;
- query-gate semantics;
- KV-sharing ownership semantics;
- position semantics;
- residual/norm ordering;
- request lifecycle transitions;
- descriptor role-string mapping.

## Deliberate numerical duplication is allowed

Implementations may intentionally differ between token/batched, CPU scalar/SIMD, CPU/CUDA, ordinary/packed CUDA and FP32/BF16/INT8 specialized paths.

Do not force code reuse that destroys vectorization, batching, memory locality, graph capture or kernel specialization. Share **semantics and compiled policy** even when numerical implementations remain specialized.

---

# Tests and CI additions

## Primitive contract tests

For each primitive, cover as applicable:

- spec validation;
- graph construction;
- tensor roles/shapes;
- CPU/CUDA compiled-layer properties;
- state/workspace sizing;
- token parity;
- chunk parity;
- packed parity.

## Extension tests

Maintain explicit proof that:

1. a new architecture using existing primitives can be registered without backend edits;
2. a fake backend can be registered without runtime-core edits;
3. a fake tokenizer implementation can be supplied without BPE coupling;
4. backend source files cannot dispatch on architecture identity;
5. model-family includes cannot cross neutral boundaries.

## Structural checks

Add incrementally to `check_architecture_boundaries.py` where robust:

- forbid concrete `BpeTokenizer` references in provider interfaces/neutral consumers after Phase 1;
- forbid attention-kernel implementation in CPU memory/page-pool files after Phase 6;
- forbid primitive execution bodies from migrating back into descriptor/import files;
- retain architecture-identity and hot-path allocation/plan-compilation prohibitions.

Avoid fragile LOC-based CI failures. File size is a signal, not an architectural invariant.

---

# Definition of extensibility success

## New model family using existing primitives

Expected work: descriptor/import metadata, tensor naming/bindings, optional chat/tokenizer/vision contribution, architecture tests and parity fixtures.

Expected backend changes: **none**.

## New primitive

Expected work: primitive semantic spec/validation, weight requirements, CPU lowering/execution, CUDA lowering/execution, packed/chunk implementation only where supported, tests, and primitive-vocabulary registration.

Unexpected work: editing every global token/chunk/packed interpreter, adding family checks to a backend, adding checkpoint-format checks to a backend, or enlarging a monolithic descriptor weight-plan conditional.

## New backend

Expected work: backend factory, optional ABI/options codec, lowering from backend-neutral `ResolvedModel`, implementation and tests. Core changes should be minimal unless a genuinely backend-neutral capability is missing.

---

# Suggested implementation order

1. Phase 0 - benchmark/parity baseline.
2. Phase 1 - `ITokenizer` boundary.
3. Phase 2 - split `descriptor.cpp`.
4. Phase 3 - primitive-owned weight requirements.
5. Phase 6 - split CPU paged storage from attention kernels.
6. Phase 4 - canonical primitive/compiled-layer vocabulary.
7. Phase 5 - CPU primitive execution.
8. Phase 7 - ordinary CUDA primitive execution.
9. Phase 8 - finish packed CUDA decomposition.
10. Phase 9 - scheduler decomposition.
11. Phase 10 - split CUDA kernel tests.
12. Phase 11 - slim `RuntimeTopology`.
13. Phase 12 - explicit ABI backend option capability.
14. Phase 13 - review remaining large TUs individually.

The early phases deliberately target safer SRP/DIP improvements before altering the most performance-sensitive execution paths.

---

# Per-phase completion checklist

A phase is complete only when all applicable items hold:

- [ ] production CPU build succeeds;
- [ ] CUDA build succeeds on supported configurations;
- [ ] affected unit tests pass;
- [ ] reference/parity tests pass;
- [ ] `check_architecture_boundaries.py` passes;
- [ ] CMake source lists and `MANIFEST.sha256` contain no obsolete path;
- [ ] no architecture-family dispatch was introduced;
- [ ] no hot-path allocation regression was introduced;
- [ ] benchmark delta was measured and accepted;
- [ ] extension-point documentation reflects the new boundary;
- [ ] temporary migration shims are removed rather than becoming permanent compatibility layers.

---

# Final desired state

The recent refactors already moved CELEG from family-driven backend conditionals to backend-neutral resolution:

```text
checkpoint
    -> architecture/descriptor
    -> backend-neutral ResolvedModel
    -> backend
```

This plan finishes the next step:

```text
checkpoint
    -> architecture/descriptor
    -> ModelGraph + policies + semantic WeightPlan
    -> backend compiler
    -> compiled primitive program
    -> token/chunk/packed pipelines
         -> primitive executors
         -> specialized numerical kernels
```

The primary extension axes then become independent:

```text
new family       -> describe/resolve the model
new primitive    -> add semantic primitive + backend lowering/execution
new backend      -> lower existing semantic primitives
new kernel       -> optimize one backend primitive without changing model semantics
```

That is the intended S.O.L.I.D. endpoint for CELEG: **family identity disappears before execution, primitive semantics have one owner, backends depend on neutral contracts, and performance-specialized code remains free to specialize without duplicating architectural decisions.**
