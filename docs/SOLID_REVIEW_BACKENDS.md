# SOLID & Anti-Pattern Review — `src/backend/*`

Scope: `src/backend/cpu/**`, `src/backend/cuda/**`, their public/backend headers,
and the implementation boundaries used by backend execution.

Original audit snapshot: `cf6315c`.

Historical revalidation snapshot: `22589152e208d750f37e589a5b8844d88b119702`.

Code-remediation closure baseline: `febf7c8a0485d91fe8a32fcb24f5742c7e6f82ed`.

## Status

**Architecture remediation: CLOSED.**

All 15 prioritized code-remediation items from the review have been implemented.
The remaining performance gate is deliberately tracked separately because it needs
measurements on target CUDA hardware; lack of a benchmark result is not treated as
an unresolved SOLID refactor.

The architectural invariants established by this review are now regression rules:

- backend execution consumes semantic model/runtime structures, not model-family
  names;
- mixer extension points are exhaustive;
- serving exposes backend-neutral request/metrics/scheduler semantics;
- execution policy is explicit rather than hidden in environment reads or branch
  accidents;
- packed execution uses typed state/services rather than opaque ownership;
- paged and contiguous execution share semantic structure;
- CUDA collaborators own cohesive behavior rather than serving only as data bags;
- format/math/storage semantics that are backend-neutral are shared.

---

## 1. Closure summary

| # | Remediation | Status | Closure commit |
|---|---|---|---|
| 1 | Make `Layer` handling exhaustive | ✅ closed | `4a861340` |
| 2 | Remove duplicated `CpuConcurrentMetrics` contract | ✅ closed | `5790761c` |
| 3 | Resolve execution-relevant environment state at configuration boundary | ✅ closed | `adf17cf1` |
| 4 | Make CUDA attention capability/launch policy explicit | ✅ closed | `043723e4` |
| 5 | Unify paged and contiguous token-forward structure | ✅ closed | `2f4fb0cd` |
| 6 | Split decode/prefill mega-functions along semantic seams | ✅ closed | `2f4fb0cd` |
| 7 | Move behavior out of `CudaCompiledModel` into cohesive collaborators | ✅ closed | `febf7c8a` |
| 8 | Split `detail/model/types.hpp` by dependency direction | ✅ closed | `03d8ed5c` |
| 9 | Replace `PackedSessionContext` pointer bag with typed views/services | ✅ closed | `16fb581b` |
| 10 | Finish narrow CPU execution/state contexts | ✅ closed | `85c4b833` |
| 11 | Share scheduler option semantics and use neutral serving boundaries | ✅ closed | `8b3d7c9e` |
| 12 | Replace long attention launcher argument lists with typed aggregates | ✅ closed | `571eece5` |
| 13 | Route short-convolution weights through semantic `TensorRole`s | ✅ closed | `845226a6` |
| 14 | Share sampling determinism/top-p and CUDA quantize/store semantics | ✅ closed | `4499b740` |
| 15 | Share backend-neutral checked math/format utility semantics | ✅ closed | `53b77aa3` |

The table is the authoritative status of the original remediation plan. Historical
findings below are retained only where they define an architectural invariant or
explain the resulting boundary.

---

## 2. Open/Closed Principle — closed state

### 2.1 Exhaustive mixer dispatch

`Layer` remains a closed semantic variant, but backend execution no longer relies on
unchecked “everything else is convolution” branches. `visit_exhaustive` /
`visit_layer` require every alternative to be named, and a compile-oriented test
covers the extension mechanism.

**Invariant:** adding a new `Layer` alternative must create compiler-visible work at
required execution/state extension points. Do not restore a generic catch-all
visitor.

### 2.2 Attention launch surface

The CUDA kernel specialization matrix remains intentionally specialized for
performance. Host-side calls now use typed aggregates from
`attention_args.hpp` instead of long positional argument lists.

**Invariant:** kernel specialization may remain closed and explicit; host-side
marshalling must remain type-safe. Do not merge device kernels merely for DRY if it
adds divergent runtime branches or register pressure.

### 2.3 Semantic tensor roles

Short-convolution input/kernel/output weights now pass through semantic tensor-role
mapping like other backend weights.

**Invariant:** checkpoint naming belongs upstream of `src/backend/*`.

---

## 3. Interface Segregation & Dependency Inversion — closed state

### 3.1 Packed execution boundary

`PackedSessionContext` is no longer an opaque `void*` ownership protocol. Packed
execution is split into typed concerns:

```text
PackedSessionContext
├── PackedExecutionServices
│   ├── typed SessionState identity
│   ├── execution plan / compiled program
│   ├── shared weights
│   └── expert residency workspace
├── PackedImmutableView
│   ├── options / topology / program
│   ├── shared weights / layout
│   └── immutable weight bindings
└── PackedSessionStateView
    ├── phase / position / generation
    ├── sampling/logit buffers
    ├── layer state
    └── runtime metrics
```

Metadata staging keys on typed session identity plus storage generation. Segmented
attention and expert-residency services are typed; no `IPackedLane` virtual hierarchy
was introduced.

**Invariant:** no opaque `void*` model owner may return to packed execution. Storage
replacement must continue to advance the generation identity used by metadata
staging.

### 3.2 CPU operator contexts

Recurrent, attention and MoE CPU operators use narrow execution/state views rather
than receiving `CpuCompiledModel` by default.

**Invariant:** an operator receives only the state/services it is entitled to read or
mutate.

### 3.3 CUDA implementation headers

The former `detail/model/types.hpp` aggregate was deleted and split by dependency
direction, including separate linear/expert/feed-forward/device/shared/layer/GEMM
headers.

**Invariant:** a component needing `LinearWeight` must not depend on unrelated expert
cache, session and layer domains through an umbrella implementation header.

---

## 4. Backend-neutral serving — closed state

CPU and CUDA now populate the same backend-neutral `ConcurrentMetrics` contract;
CPU-specific counters live in a narrow extras structure instead of cloning the full
metrics surface.

Generic serving already depends on role-specific contracts:

```text
IRequestService
ISchedulerDriver
IServiceDiagnostics
```

Scheduler concepts common to CPU and CUDA now share `ConcurrentSchedulerOptions`:

```text
max_active_requests
max_batched_tokens
worker_thread
prefix_cache
```

Each backend keeps only execution-specific scheduler knobs and its own defaults.

**Invariant:** do not introduce a second generic serving contract inside a backend.
The neutral service roles are the runtime substitution boundary; a new virtual
`IConcurrentEngine` hierarchy is unnecessary unless a new independent substitution
point appears.

---

## 5. Single Responsibility and ownership — closed state

### 5.1 Execution structure

Standalone and paged host token execution share one layer program through a typed KV
policy. Decode/prefill execution is separated by semantic seams rather than arbitrary
line-count helper extraction:

```text
layer iteration
mixer execution
attention capability selection
KV access policy
feed-forward execution
state transition
```

### 5.2 `CudaCompiledModel` as coordinator

`CudaCompiledModel` remains the model-level coordinator; it is not required to become
a graph of virtual services. The review required cohesive collaborators to own real
behavior, not merely data.

That boundary is now materially stronger:

- `CudaModelResources` owns model/resource state;
- `SessionState` owns mutable session state;
- `GemmDispatcher` owns GEMM dispatch/plans;
- `CudaWorkspace` owns execution workspace;
- `CudaSamplingState` owns sampling buffers **and sampling execution policy**,
  including forced-prefix handling;
- `CudaDecodeGraphs` owns graph instances **and capture begin/end/abort lifecycle**;
- `SessionStore` owns persistence serialization/restoration mechanics;
- packed execution consumes typed views/services rather than reaching through the
  model owner.

`CudaCompiledModel` still orchestrates these collaborators, which is its intended
responsibility.

**Invariant:** new cohesive behavior should first be placed with the collaborator
that owns its state. Do not create a forwarding-only service whose implementation
still lives entirely in `CudaCompiledModel`.

---

## 6. Configuration and capability policy — closed state

Execution-relevant environment values such as flash-attention, MMQ tensor-core and
managed-weight choices are resolved at the configuration boundary. Execution paths
consume resolved options rather than process-global `getenv()` state.

CUDA attention support is expressed through an explicit capability resolver/matrix,
including supported and deliberately unsupported combinations of KV format, layout,
position source, bias, operation and algorithm.

**Invariant:** an unsupported combination must fail deliberately with a reason; it
must never exist merely as an accidental fall-through in nested control flow.

---

## 7. DRY and semantic duplication — closed state

### 7.1 Paged/contiguous execution

Non-KV token execution is shared; KV addressing is the policy variation. This removes
the former drift risk where recurrent-mixer/residual fixes could land in one complete
forward function but not the other.

### 7.2 Sampling determinism

CUDA stochastic paths share one `sample_sorted_topk` primitive for top-p cutoff, RNG
advancement and stochastic choice over an already deterministically ordered top-k
list. Standalone, fused, partial-merge and packed sampling no longer carry independent
copies of this contract.

The top-k candidate ordering remains performance-specific and deterministic: higher
score first, then lower vocabulary index on exact ties.

### 7.3 Linear storage semantics

CUDA single and concatenated linear loaders share the Int8/Int4 allocation, device
copy, fallback binding and `LinearWeight` finalization helpers. Checkpoint naming is
not part of those helpers.

### 7.4 Backend-neutral math/format helpers

Overflow-checked size arithmetic is shared under `celeg/runtime/checked_math.hpp` and
used by neutral page-layout math plus CPU/CUDA paged KV allocation. Existing neutral
BF16/GGUF/quantization utilities remain the source of format semantics.

**Invariant:** backend-specific code may specialize implementation for ISA/device
performance, but a file-format or checked-size rule must not be independently
redefined in CPU and CUDA code.

---

## 8. Acceptance gates

The original review required mechanical gates. Their closure state is:

| Gate | State | Evidence |
|---|---|---|
| Mixer exhaustiveness | ✅ code-closed | exhaustive visitor + compile-oriented regression test |
| Backend-neutral metrics | ✅ code-closed | CPU/CUDA use `ConcurrentMetrics`; CPU extras are explicit |
| Configuration boundary | ✅ code-closed | execution-relevant env resolution moved to options/bootstrap |
| Attention capability | ✅ code-closed | explicit capability resolver + capability tests |
| Paged/contiguous parity | ✅ code-closed | shared token-layer structure/KV policy; recurrent paths exercised by existing CUDA test work |
| Packed lifetime/type boundary | ✅ code-closed | typed session identity/views/services + generation-aware metadata cache |
| Dependency boundary | ✅ code-closed | `detail/model/types.hpp` deleted and split |
| Architecture neutrality | ✅ code-closed | `check_architecture_boundaries.py` rejects backend architecture/model-family leakage |
| Performance | ⚠️ measurement required | run before/after decode/prefill benchmarks on target CUDA hardware |

### Performance gate

This is intentionally not marked green from static review alone. Structural refactors
in hot decode/prefill paths must be measured on the hardware used for performance
claims.

Minimum closure evidence should record, for the same model/configuration/hardware:

- decode tokens/s and per-token latency;
- prefill throughput/latency;
- CUDA-graph on/off where affected;
- contiguous and paged serving where affected;
- BF16 and relevant quantized modes;
- no statistically meaningful regression beyond the project's accepted tolerance.

The visitor/policy work already preserved specialized kernels rather than merging
inner loops. The benchmark is therefore a verification gate, not another architecture
refactor.

---

## 9. Regression rules

The review should be reopened only if one of these properties regresses:

1. a new mixer can be added without compiler-visible handling work;
2. backend execution branches on a concrete model family/architecture identity;
3. CPU/CUDA clone generic serving metrics or scheduler semantics;
4. execution paths read hidden process environment policy directly;
5. attention support becomes implicit branch fall-through again;
6. packed execution introduces opaque owner/lifetime state;
7. standalone and paged paths regain duplicate complete layer programs;
8. a broad implementation umbrella header recreates cross-domain coupling;
9. sampling paths implement independent stochastic/tie behavior;
10. backend-neutral format/math/storage semantics are copied across backends;
11. a cohesive CUDA collaborator is reduced to a data bag while its behavior moves
    back into the model coordinator.

---

## 10. Final assessment

```text
semantic program                         ✅ strong
model-family neutrality                  ✅ strong
kernel specialization                    ✅ explicit/performance-aware
file-level decomposition                 ✅ strong
exhaustive mixer extension points        ✅ closed
backend-neutral serving contracts        ✅ closed
attention capability policy              ✅ explicit
implementation-state boundaries          ✅ materially narrowed
packed-session type/lifetime boundary     ✅ typed + generation-aware
paged/contiguous execution reuse         ✅ shared semantic structure
sampling determinism contract            ✅ shared
backend-neutral utility semantics        ✅ shared
performance revalidation                 ⚠️ requires target-hardware measurement
```

The original SOLID/anti-pattern remediation plan is therefore **closed at the code and
architecture level** as of `febf7c8a0485d91fe8a32fcb24f5742c7e6f82ed`.

Future backend work should preserve semantic extension points, explicit capabilities,
typed ownership and performance-specific closures. Architecture growth should make
the compiler/tests identify required implementation work without reintroducing model-
family knowledge into execution.
