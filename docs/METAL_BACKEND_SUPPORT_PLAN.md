# Apple Metal Backend Support Plan

## Status

Initial native Metal vertical slice implemented; broader backend coverage remains pending.

The current implementation provides the backend build/configuration surface,
Apple Silicon capability discovery, a runtime-compiled Metal command-buffer
probe, serving/API registration, and cached-checkpoint smoke coverage. The
macOS arm64 CPU baseline is verified with the complete 80-test CPU suite.
The native path currently covers the cached LFM2.5-350M convolution/attention/
SwiGLU graph and runs its linear, convolutional, attention, and logits operations
through Metal. Q4_K and Q6_K GGUF tensors remain in native block layout and
use Metal dequantizing GEMV and embedding kernels; other quantized formats
still use the explicit host fallback. The cached LFM2.5-8B-A1B MoE checkpoint
now runs a one-token Metal smoke using demand-loaded expert matrices. The
service also has page-addressed KV storage, longest-prefix session reuse, and
independent-request scheduling. Full recurrent graph coverage, a shared
physical page allocator, batched GPU dispatch, and serving scale-out remain
later roadmap phases.

Audit baseline: `master` at `cdd716677364e5f52d1875f9b5e68e11e89afa81`.

Primary physical validation target: Apple Silicon MacBook Air with M5 on macOS.

The goal is to make Metal a first-class CELEG execution backend with backend id
`metal`, alongside `cpu` and `cuda`, while preserving the current architecture
boundaries and keeping all model semantics, checkpoint resolution, serving
contracts, and backend selection independent from Apple-specific types.

---

# 1. Executive decision

CELEG should implement Metal as a **new native backend**, not as a CUDA shim and
not as a model-specific path.

The target shape is:

```text
checkpoint / GGUF / Safetensors
            |
            v
canonical model semantics + WeightPlan + CompiledModelProgram
            |
            +-----------------+-----------------+
            |                 |                 |
            v                 v                 v
      CPU backend        CUDA backend       Metal backend
            |                 |                 |
            v                 v                 v
      CPU kernels        CUDA kernels        MSL kernels
                                               |
                                               v
                                      Apple Silicon GPU
```

Metal must enter through the existing runtime/backend factory boundary. The
backend-neutral runtime must not learn Metal API types, Objective-C types,
`MTL*` handles, Apple GPU-family checks, MSL source names, or M5-specific feature
flags.

The first implementation must use ordinary Metal compute kernels and the normal
Metal resource/command-buffer model. M5-specific tensor acceleration is a later,
feature-detected fast path. It must not be required for correctness because it
is newer, more hardware-specific, and more sensitive to SDK/runtime behavior
than baseline Metal compute.

The host-side implementation should use a **thin Objective-C++ bridge** around
the system Metal framework rather than exposing Objective-C objects across the
codebase. Public and internal CELEG interfaces remain C++20. Objective-C++ is
confined to the Metal backend implementation files.

Metal Performance Shaders / MPSGraph may be evaluated as optional internal
acceleration providers after the direct backend is correct, but CELEG should not
make MPSGraph its semantic execution engine. Quantized GGUF kernels, recurrent
mixers, MoE, paged KV, and CELEG-specific scheduling need explicit control over
layout, residency, synchronization, and dispatch.

---

# 2. Why the current architecture is ready for a third backend

The repository already contains the major seams required for Metal:

1. `celeg_base` owns backend-neutral runtime/model/checkpoint code.
2. `celeg_cpu_backend` and `celeg_cuda_backend` are separate libraries.
3. Backend sources already have separate CMake manifests.
4. `IRequestService`, `ISchedulerDriver`, and `IServiceDiagnostics` define the
   serving substitution boundary.
5. `ServiceBundle` already wraps any concrete backend service that implements
   those roles.
6. `celeg_engine_create` selects a backend by string `backend_id` through the
   runtime backend catalog.
7. The C API explicitly documents `celeg_model_*` as a CPU-only convenience API
   and directs new backend-aware code to `celeg_engine_*`.
8. The extensibility roadmap already states that a new backend should register
   through the runtime without forcing backend-neutral API enums to grow.

Therefore Metal does **not** justify a new generic graph runtime, a new model
semantic layer, or a CPU/CUDA merge. The implementation should consume the same
resolved semantic program and weight plan that the existing backends consume.

---

# 3. Goals

The Metal backend is complete when CELEG can build and run native Apple Silicon
inference with the following properties:

- `python scripts/dev.py verify --backend metal` works on a supported Mac.
- CMake exposes `CELEG_ENABLE_METAL` independently from CUDA.
- `celeg_engine_create(..., backend_id="metal", ...)` creates a Metal service.
- `celeg-serve --backend metal` uses the same OpenAI-compatible protocol path as
  CPU/CUDA.
- GGUF and Safetensors use the existing canonical checkpoint and weight-plan
  machinery.
- At least one small real model runs end-to-end before optimization work begins.
- Kernel correctness is checked against CELEG's CPU reference implementation.
- Quantized weights execute natively on the GPU instead of being universally
  expanded to BF16/FP16 on the host.
- KV cache, prefix cache, batching, cancellation, and serving can be added
  without changing model semantics.
- The M5 Air can produce reproducible prefill/decode benchmarks and Metal GPU
  captures.
- M5-only acceleration is optional and runtime-feature-detected.

---

# 4. Explicit non-goals

## 4.1 Do not port CUDA source mechanically

CUDA and Metal have different command submission, synchronization, subgroup,
memory, pipeline, and compilation models. Reproducing every CUDA class with a
Metal spelling would preserve CUDA-specific design decisions rather than the
semantic architecture.

Reuse algorithms and numerical contracts, not CUDA plumbing.

## 4.2 Do not expose Apple SDK types outside the Metal backend

No `id<MTLDevice>`, `MTL::Device`, `MTLBuffer`, Foundation type, Objective-C
header, or MSL-specific structure belongs in backend-neutral model/runtime
headers.

## 4.3 Do not make M5 tensor features a minimum requirement

The baseline backend must remain ordinary Metal compute and should run on a
reasonable Apple Silicon floor chosen during implementation. M5-specific tensor
features may accelerate GEMM/attention only after the generic path is passing.

## 4.4 Do not target the Apple Neural Engine in this roadmap

Metal targets the Apple GPU. ANE/Core ML is a different backend problem and
would require a separate capability, compilation, and execution design.

## 4.5 Do not optimize before establishing numerical parity

Every optimization phase must have a measured baseline and a correctness oracle.
The current CELEG rule remains authoritative: measure before optimizing.

---

# 5. Phase 0 — make macOS arm64 a supported host before adding Metal

The new M5 machine first needs to become a reliable CELEG CPU reference host.
This phase is intentionally separate from Metal so GPU failures are not confused
with unrelated portability failures.

## Work

- Add macOS/arm64 detection to `scripts/dev_environment.py`.
- Extend the developer CLI backend choices only after environment discovery can
  describe Apple Silicon correctly.
- Make `python scripts/dev.py doctor --backend cpu` report:
  - macOS version;
  - Apple Silicon architecture;
  - compiler path/version;
  - Xcode/SDK path;
  - CMake/Ninja availability;
  - CPU backend availability.
- Verify the x86-specific CPU source files compile safely on arm64 or are
  conditionally excluded when their ISA is impossible.
- Run the complete CPU CTest suite on the M5.
- Run one cached small-model CPU smoke test on the M5.
- Add a `macos-cpu-release` or equivalent CMake preset only if a platform-
  specific preset is materially clearer than the existing portable CPU preset.

## Exit gate

```text
python scripts/dev.py verify --backend cpu
```

must pass on the M5 before Phase 1 Metal work is considered valid.

This gives Metal development a same-machine, same-checkpoint numerical oracle.

---

# 6. Phase 1 — backend skeleton, build system, and device probe

## New CMake surface

Add:

```text
CELEG_ENABLE_METAL
CELEG_RUN_METAL_TESTS
```

Recommended policy:

- default `CELEG_ENABLE_METAL=ON` on Apple hosts and `OFF` elsewhere;
- configuring Metal on a non-Apple host should fail clearly only when explicitly
  requested;
- CUDA and Metal options remain independent;
- CPU remains always available as the portable reference backend.

Add:

```text
cmake/sources/metal_backend.cmake
```

and a target:

```text
celeg_metal_backend
```

linked to:

```text
celeg_runtime
Metal.framework
Foundation.framework
Threads::Threads
```

Enable Objective-C++ only inside the Apple/Metal configuration path rather than
making it a global project-language requirement on Windows/Linux.

## Initial source layout

```text
include/celeg/backend/metal/
  runtime_types.hpp
  device.hpp
  model.hpp
  concurrency.hpp

src/backend/metal/
  runtime/
    device.mm
    command_context.mm
    pipeline_cache.mm
  memory/
    buffer.mm
  model/
    compiler.cpp
    resources.cpp
    lifecycle.cpp
  kernels/
    probe.metal

src/serve/
  metal_inference_service.cpp

include/celeg/serve/
  metal_inference_service.hpp

src/app/metal/
  main.cpp
```

The exact file split can evolve, but Apple API ownership must stay under the
Metal backend.

## Device context

Create a Metal-owned RAII context responsible for:

- selecting the default Metal device;
- owning the command queue;
- exposing immutable device capabilities through CELEG-owned value types;
- compiling/loading the shader library;
- caching compute pipeline states;
- creating command buffers/encoders;
- reporting device name and resource limits;
- optionally exposing capture hooks for diagnostics.

The first executable milestone is a device probe that launches a trivial compute
kernel and validates a host-visible result.

## Exit gate

- `celeg_metal_backend` builds on the M5.
- one MSL kernel is compiled and dispatched successfully;
- device diagnostics print the Apple GPU and supported Metal feature families;
- no Metal/Objective-C symbol appears in backend-neutral public headers.

---

# 7. Metal shader compilation strategy

Prefer **offline shader compilation** through the active Xcode SDK:

```text
*.metal -> *.air -> *.metallib
```

CMake should invoke the SDK tools selected by `xcrun` and make shader compilation
a normal dependency of `celeg_metal_backend`.

The release executable should not depend on arbitrary source-tree paths. Choose
one of these packaging strategies during Phase 1 and keep it deterministic:

1. generate an embedded byte array from the compiled `.metallib`; or
2. install the `.metallib` beside the CELEG binaries and resolve it through a
   well-defined runtime asset path.

Embedding is preferred for the first backend because it makes CLI, C API, and
server packaging self-contained.

Runtime source compilation may exist as a developer-only escape hatch, but it
must not be the production default.

Add shader compilation diagnostics to `scripts/dev.py doctor --backend metal` so
missing Xcode/SDK tools fail before the C++ build starts.

---

# 8. Host API design

Use a pure C++ host interface with implementation hidden behind Objective-C++.
For example, the rest of the Metal backend should see concepts such as:

```text
MetalDevice
MetalBuffer
MetalPipeline
MetalCommandContext
MetalCapabilities
```

not raw SDK objects.

The wrapper must remain thin. It is an ownership/boundary layer, not a second
GPU framework.

## Command submission rule

Avoid one command buffer per tiny operation. The default execution shape should
encode a complete logical unit of work into as few command buffers/compute
encoders as correctness allows.

For decode, the target is normally one command buffer per decode step, with
pipeline states cached and synchronization performed on-device between dependent
dispatches.

Do not add an indirect-command-buffer or command-replay abstraction until a
profile proves CPU encoding overhead is significant.

---

# 9. Unified-memory strategy

Apple Silicon uses unified physical memory, but CELEG must not treat every Metal
resource mode as equivalent.

Start with a simple, observable policy:

## Host-visible state

Use shared resources for data the CPU must read/write frequently:

- input token ids;
- sampled token/result slots;
- compact metrics/status structures;
- test buffers;
- small control structures.

## GPU-dominant state

Benchmark shared versus private GPU resources for:

- model weights;
- KV cache;
- intermediate workspaces;
- large attention/recurrent state.

A private resource may still be preferable on unified-memory hardware because
it can use GPU-optimal placement/layout. If private storage wins, upload weights
through a temporary staging buffer and release the staging copy immediately.

The steady-state design should avoid keeping duplicate complete CPU and GPU
weight copies merely because the memory is unified.

## Allocation ownership

Metal memory planning must derive from `CompiledModelProgram` / backend planning,
mirroring the direction already taken by CPU/CUDA. Do not reintroduce model-
architecture switches into the allocator.

---

# 10. Kernel architecture

Metal kernels should be organized by semantic primitive, not model name.

Recommended groups:

```text
src/backend/metal/kernels/
  elementwise.metal
  reduction.metal
  norm.metal
  embedding.metal
  linear.metal
  quantized_linear.metal
  rope.metal
  attention.metal
  sampling.metal
  short_convolution.metal
  mamba2.metal
  gated_delta.metal
  moe.metal
```

No `lfm`, `granite`, `minicpm`, `smollm`, or `nemotron` switch belongs in a
kernel file. Model semantics are already resolved before execution reaches a
backend.

## Numerical policy

For the initial implementation:

- accumulate numerically sensitive reductions in FP32;
- support FP16/BF16 storage only when the device/compiler path is verified;
- compare outputs against CPU with explicit tolerance policies;
- preserve deterministic tie-breaking for greedy sampling;
- test edge dimensions, non-multiple threadgroup tails, and odd head widths.

---

# 11. Primitive delivery order

Implement kernels in an order that creates end-to-end vertical slices instead
of a large pile of unintegrated primitives.

## Stage A — infrastructure primitives

- buffer fill/copy;
- embedding lookup;
- elementwise add/multiply;
- activation functions required by the first model;
- RMSNorm;
- RoPE;
- FP16/BF16 linear GEMV/GEMM baseline;
- logits projection;
- greedy argmax.

## Stage B — standard transformer path

- GQA/MHA decode attention;
- prefill attention;
- dense feed-forward;
- paged KV layout compatible with CELEG's runtime cache model.

## Stage C — CELEG hybrid/recurrent primitives

- short convolution;
- Mamba-2;
- gated delta / recurrent state update;
- hybrid layer scheduling from `CompiledModelProgram`.

## Stage D — quantized linear path

Prioritize formats based on CELEG's real checkpoint usage, not theoretical
coverage. The first useful set should include the types needed by a small local
GGUF smoke model, then expand from measured usage.

Likely priority:

1. F16/BF16 baseline;
2. Q4_K_M path;
3. Q8_0 / Q8_K activation or reference support as required;
4. additional K-quants;
5. I-quants/other formats only with dedicated quality and performance tests.

Quantized weights should remain quantized in Metal memory whenever a native
kernel exists.

## Stage E — MoE

- router scores;
- top-k expert selection;
- expert grouping/batching;
- expert GEMV/GEMM;
- shared expert path;
- residency policy for large models.

Do not copy the CUDA host-offload design blindly. Apple unified memory changes
the cost model, so expert residency needs M5 measurements before policy is
fixed.

---

# 12. First real-model milestones

Use progressively harder real models so each milestone proves a new capability.
The exact model can change if local checkpoint availability makes another model
more reproducible, but the progression should remain.

## Milestone 1 — smallest deterministic vertical slice

Run one small cached model with greedy generation and compare:

```text
CPU logits/tokens == Metal logits/tokens within defined tolerance
```

Prefer the smallest CELEG-supported checkpoint that exercises the least number
of primitives while still using the real checkpoint/semantic pipeline.

## Milestone 2 — standard attention model

Use a small standard/GQA transformer model to validate:

- prefill;
- decode;
- RoPE;
- KV cache;
- dense FFN;
- GGUF quantized linear execution.

## Milestone 3 — hybrid/recurrent model

Use an LFM/Nemotron-style hybrid checkpoint to validate the recurrent primitive
families and mixed layer schedule.

## Milestone 4 — MoE

Use the smallest practical CELEG-supported MoE model. Correctness comes before
expert-residency tuning.

---

# 13. Backend factory and C API integration

The C API already has the correct third-backend boundary: `celeg_engine_options`
contains a string `backend_id` plus an opaque backend-options block.

Add:

```text
celeg_metal_model_options
celeg_metal_engine_options
celeg_metal_backend_options
celeg_metal_backend_options_init(...)
```

and backend id:

```text
metal
```

Add `MetalBackendOptions` and `MetalBackendFactory` beside the CPU/CUDA factories
and register it only when Metal was built.

Compile definition:

```text
CELEG_API_WITH_METAL=1
```

The old `celeg_backend` enum in `api.h` is not the selection mechanism for the
engine factory and must not become a new architectural dependency. If it is
unused/dead, remove it under the repository's no-backward-compatibility policy
rather than extending it mechanically.

The CPU-only `celeg_model_*` convenience family remains CPU-only.

---

# 14. Serving and concurrency

Create `MetalInferenceService` implementing the same three roles used by the
other backends:

```text
IRequestService
ISchedulerDriver
IServiceDiagnostics
```

Do not add Metal behavior to protocol routes. The server route layer should only
select/configure the service bundle.

## Initial serving implementation

Start with a correctness-first scheduler:

- one active decode batch;
- explicit request lifecycle;
- cancellation between steps;
- metrics compatible with the existing service surface.

Then add:

- batched prefill;
- batched decode;
- prefix cache;
- paged KV page reuse;
- admission control;
- concurrent request scheduling.

The scheduler should be driven by backend-neutral request semantics, with Metal-
specific batching limits derived from measured device capabilities.

---

# 15. Developer harness

Extend:

```text
python scripts/dev.py doctor --backend metal
python scripts/dev.py build --backend metal
python scripts/dev.py test --backend metal
python scripts/dev.py smoke --backend metal
python scripts/dev.py verify --backend metal
```

`--backend auto` should select in this order:

1. CUDA when a working CUDA toolchain/device is available;
2. Metal on supported Apple Silicon;
3. CPU otherwise.

On a Mac, CUDA will normally be absent and `auto` should resolve to Metal once
the backend is mature enough for general use.

`doctor --backend metal` should report at minimum:

- Apple Silicon CPU architecture;
- macOS version;
- Xcode version;
- macOS SDK path/version;
- Metal compiler availability;
- default Metal device name;
- supported GPU/Metal families;
- maximum buffer/threadgroup limits used by CELEG;
- whether optional M5 fast paths are available.

---

# 16. CMake presets

Add:

```text
metal-release
metal-relwithdebinfo
```

with build directories analogous to the existing CPU/CUDA presets.

The preset should set:

```text
CELEG_ENABLE_CUDA=OFF
CELEG_ENABLE_METAL=ON
CELEG_RUN_METAL_TESTS=ON
```

where runtime Metal tests are known to have an actual device.

A compile-only CI preset may set runtime tests off while still compiling all MSL
and Metal C++/Objective-C++ sources.

---

# 17. Test strategy

## 17.1 Shader/kernel unit tests

For each primitive:

1. create deterministic host inputs;
2. compute the reference through CPU/reference code;
3. dispatch the Metal kernel;
4. copy/read the result;
5. compare using CELEG numerical utilities;
6. include malformed/tail/boundary dimensions.

Tests must cover both prefill-shaped and decode-shaped workloads because a
kernel that is fast/correct for large matrices may behave differently at batch
1.

## 17.2 End-to-end logits

Add a Metal equivalent of CELEG's reference-compare workflow:

```text
celeg-metal-compare-reference
```

or generalize the existing comparison tool if that can be done without mixing
backend internals.

Compare:

- first-token logits;
- top-k agreement;
- cosine similarity;
- RMSE/max error;
- deterministic greedy token sequence.

## 17.3 Cache tests

Explicitly test:

- KV append/read;
- page-boundary transitions;
- prefix reuse;
- request cancellation/release;
- context-length boundaries;
- concurrent page ownership.

## 17.4 Architecture-boundary tests

Extend `scripts/check_architecture_boundaries.py` so:

- backend-neutral directories cannot include Metal headers;
- Metal backend code cannot depend on CUDA implementation headers;
- CUDA backend code cannot depend on Metal implementation headers;
- model-name checks remain absent from backend operator code.

---

# 18. Benchmark plan for the M5 Air

The physical M5 Air is the primary performance lab because hosted CI is not a
substitute for a stable, known GPU.

Create a reproducible Metal benchmark script using the existing benchmark
manifest philosophy.

Capture at least:

```text
model
checkpoint format / quantization
context capacity
resident context length
prompt tokens
new tokens
prefill tokens/s
prefill latency
steady-state decode tokens/s
first-token latency
peak CELEG resident memory
Metal allocated memory if available
backend options
macOS / Xcode versions
CELEG commit
```

## Required A/B dimensions

- CPU vs Metal on the same M5;
- shared vs private weight storage;
- BF16/FP16 vs native quantized weights;
- short vs long resident context;
- single request vs concurrent requests;
- command-buffer/encoder strategies when profiling proves dispatch overhead.

## Profiling tools

Add developer documentation for:

- Xcode GPU Capture / Metal System Trace;
- Metal validation layer in debug builds;
- pipeline statistics where supported;
- optional `powermetrics` runs for longer thermal/performance studies.

Because the MacBook Air is fanless, sustained benchmarks must record warm-up and
run duration. Short burst numbers and thermally stabilized numbers should not be
mixed.

---

# 19. M5-specific optimization track

Only begin this track after the generic Metal backend passes real-model
correctness.

## 19.1 Feature detection, never chip-name branching

Do not write:

```text
if device_name contains "M5"
```

Use Metal capability/family/API availability checks and keep the generic path as
fallback.

## 19.2 Tensor acceleration

Evaluate the newer Metal tensor/matrix acceleration APIs for:

- dense GEMM;
- attention projections;
- large prefill matmuls;
- MoE expert matrices where shapes are suitable.

Keep custom MSL paths for quantized formats and shapes not handled efficiently
by the tensor path.

The tensor path must be optional because SDK/runtime support can lag hardware or
have feature-specific compiler/runtime limitations.

## 19.3 Decode optimization

Measure before choosing a strategy. Candidate bottlenecks include:

- batch-1 quantized GEMV bandwidth;
- tiny reduction occupancy;
- command encoding/commit overhead;
- synchronization between many small dispatches;
- attention over short live context versus allocated context;
- sampler reductions over large vocabularies.

The recent CUDA work has already shown that plausible bottleneck theories can be
wrong until measured. Metal should begin with instrumentation, not assumptions.

## 19.4 Pipeline cache

If shader/pipeline startup becomes material, evaluate Metal binary archives or
another SDK-supported pipeline-cache mechanism. This is a startup optimization,
not a Phase 1 requirement.

---

# 20. CI strategy

Add a macOS workflow that always validates:

- CMake configure for CPU + Metal;
- Objective-C++ compilation;
- MSL -> metallib compilation;
- Metal backend link;
- backend-neutral tests;
- CPU tests;
- Metal tests that do not require reliable GPU execution.

Do **not** make hosted GitHub macOS GPU runtime behavior the sole correctness
gate. GPU availability/capabilities on hosted runners are not a stable physical
M5 test target.

For real GPU runtime coverage, use one of:

1. the physical M5 Air as the normal local acceptance machine; or
2. later, a self-hosted M5 runner for scheduled/nightly Metal smoke and
   benchmarks.

If a self-hosted runner is introduced, keep benchmark history separate from
pass/fail correctness tests so thermal/background noise does not make CI flaky.

---

# 21. Recommended implementation sequence

## Sprint M0 — macOS CPU baseline

- macOS/arm64 developer-harness support;
- CPU build fixes;
- full CPU CTest on M5;
- small CPU model smoke.

**Gate:** CPU reference is trustworthy on the same machine.

## Sprint M1 — Metal bootstrapping

- `CELEG_ENABLE_METAL`;
- Metal source manifest;
- SDK/shader compiler discovery;
- Objective-C++ device wrapper;
- embedded metallib;
- trivial dispatch test;
- CMake presets.

**Gate:** Metal device + shader dispatch works from CELEG.

## Sprint M2 — dense vertical slice

- embedding;
- norm;
- FP16/BF16 linear baseline;
- activation;
- RoPE;
- attention;
- dense FFN;
- greedy sampling;
- one small model end-to-end.

**Gate:** deterministic small-model generation with CPU numerical parity.

## Sprint M3 — native GGUF quantization

- Q4_K_M-first native quantized GEMV/GEMM;
- quantization correctness tests;
- GGUF small-model smoke;
- CPU-vs-Metal benchmark report.

**Gate:** quantized model runs without whole-model BF16 expansion.

Current status: Q4_K and Q6_K native block residency, Metal GEMV/embedding
dispatch, CPU reference comparison, cached LFM2.5-350M-Q4_K_M smoke, and the
reproducible `celeg-metal-bench` manifest pass on the physical M5. Additional
K-quants remain on the host fallback path.

## Sprint M4 — hybrid/recurrent execution

- short convolution;
- Mamba-2;
- gated delta/recurrent primitives;
- hybrid program dispatch.

**Gate:** at least one CELEG-supported hybrid checkpoint generates correctly.

Current status: short-convolution hybrid inference is validated with cached
LFM2.5-350M, and native single-token Mamba-2 and Gated Delta state kernels have
deterministic GPU primitive coverage. No Mamba-2 or Gated Delta checkpoint is
currently present in the local Hugging Face cache, so real-checkpoint numerical
parity remains an acceptance task when such a fixture is available.

## Sprint M5 — runtime cache and serving

- paged KV;
- prefix cache;
- `MetalInferenceService`;
- backend factory/C API options;
- `celeg-serve --backend metal`;
- cancellation and metrics.

**Gate:** OpenAI-compatible streamed generation on Metal.

## Sprint M6 — batching and concurrency

- batched prefill;
- batched decode;
- admission control;
- page reuse under concurrent requests;
- concurrency benchmark.

**Gate:** multi-request service correctness and reproducible throughput metrics.

## Sprint M7 — MoE

- routing;
- expert execution;
- shared expert;
- unified-memory-aware expert residency policy.

**Gate:** smallest practical CELEG MoE checkpoint runs correctly.

## Sprint M8 — M5 optimization

- profile-guided kernel tuning;
- storage-mode A/B;
- tensor API fast paths where stable/useful;
- dispatch reduction only if measured;
- thermal-stabilized M5 benchmark report.

**Gate:** optimization report contains measured before/after numbers and no
correctness regression.

## Sprint M9 — CI/productization

- macOS compile workflow;
- optional self-hosted M5 runtime workflow;
- README support matrix update;
- install/package validation;
- `AGENTS.md` Metal build/performance guidance.

**Gate:** Metal is documented as a supported CELEG backend rather than an
experimental branch.

---

# 22. Definition of done

Metal support is considered first-class only when all of the following are true:

- [x] macOS arm64 CPU verify passes.
- [x] Metal device discovery and shader compilation are part of the standard
      developer harness.
- [x] `celeg_metal_backend` is isolated from backend-neutral code.
- [x] backend id `metal` works through `celeg_engine_create`.
- [x] Metal serving uses the existing service interfaces.
- [x] Safetensors model smoke passes.
- [x] GGUF native quantized model smoke passes.
- [x] standard attention path passes CPU numerical comparison.
- [x] at least one hybrid/recurrent model passes.
- [x] page-addressed KV and prefix-cache tests pass.
- [x] concurrent independent-request tests pass; batched GPU dispatch remains
      a later optimization.
- [x] the cached LFM2.5-8B-A1B MoE checkpoint passes a one-token smoke.
- [x] M5 benchmark results are reproducible and committed as benchmark output,
      not hard-coded claims in implementation comments.
- [x] hosted macOS CI compiles the Metal backend and shaders.
- [x] real Metal runtime is validated on the physical M5 or a self-hosted Apple
      Silicon runner.
- [ ] optional M5 tensor fast paths are feature-detected and retain the generic
      Metal fallback.
- [x] README/support matrix and developer documentation are updated.

---

# 23. First commands to run on the M5

Before implementing Metal, establish the host baseline:

```text
xcode-select -p
xcrun --sdk macosx --show-sdk-path
clang++ --version
cmake --version
ninja --version
python3 scripts/dev.py doctor --backend cpu
python3 scripts/dev.py verify --backend cpu
```

Then the first Metal-specific probe introduced by Sprint M1 should make these
possible:

```text
python3 scripts/dev.py doctor --backend metal
cmake --preset metal-relwithdebinfo
cmake --build --preset metal-relwithdebinfo
ctest --preset metal-relwithdebinfo
python3 scripts/dev.py smoke --backend metal
```

This sequence deliberately makes the new M5 Air useful from day one: first as a
macOS/arm64 CPU oracle, then as the real Metal correctness machine, and finally
as the dedicated performance lab for M5-specific optimization.

---

# 24. External technical references to track during implementation

- Apple Metal documentation and Metal feature-set tables.
- Apple Metal shader-language documentation.
- Apple Metal GPU capture/profiling documentation.
- Apple Metal-cpp documentation as a reference for C++ API mapping, even though
  the initial host boundary proposed here is a thin Objective-C++ wrapper.
- `ggml` / `llama.cpp` Metal backend implementation for comparative ideas about
  quantized kernels, residency, and Apple-Silicon-specific behavior. Treat it as
  reference evidence, not as CELEG architecture.

Any M5-specific API adopted from these references must be isolated behind a
capability check and benchmarked against CELEG's generic Metal path.
