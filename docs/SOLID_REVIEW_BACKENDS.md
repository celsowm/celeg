# SOLID & Anti-Pattern Review — `src/backend/*`

Scope: `src/backend/cpu/**`, `src/backend/cuda/**`, the headers they publish under
`include/celeg/backend/**`, and the implementation headers under
`include/celeg/detail/model/**` / `src/backend/cpu/detail/**`.

Original audit snapshot: `cf6315c`.

Revalidated against: `22589152e208d750f37e589a5b8844d88b119702`.

---

## 0. How to read this review

Celeg is a high-performance inference engine. SOLID is useful here only when it
improves one of these things:

- change velocity;
- correctness confidence;
- backend portability;
- testability;
- explicit capability boundaries;
- performance without hidden coupling.

A hand-unrolled kernel, a template specialization, a large launch signature, or a
closed dispatch domain is not automatically a design defect. Conversely, splitting
a large translation unit into many files is not sufficient decomposition if the
files are still methods of one state-owning type and still require the same broad
implementation headers.

This document therefore distinguishes:

- **structural findings** — architecture that increases the cost or risk of change;
- **performance closures** — deliberate specialization that should remain closed;
- **capability gaps** — combinations that are not implemented or not explicit;
- **duplication** — textual or semantic behavior that can drift independently;
- **historical measurements** — counts measured at `cf6315c`, retained as evidence
  but not presented as if they had been recomputed after every commit.

Severity legend:

- 🔴 high — actively taxes ordinary backend work or creates a strong correctness
  hazard;
- 🟠 medium — meaningful structural debt, but not evidence by itself of an active
  correctness failure;
- 🟡 low — consistency, clarity, or cheap maintenance cleanup.

---

## 1. Executive summary

The central diagnosis from the original review still holds:

> **The backend is substantially better decomposed at file level than at type and
> capability-boundary level.**

The most valuable property of the backend is also still intact: model-family and
architecture knowledge is compiled upstream into semantic runtime structures such
as `CompiledModelProgram`; backend execution does not branch on concrete model
families. That property must not regress.

The current high-value issues are:

1. **Mixer dispatch is still open-coded through `as_attention()` /
   `as_gated_delta_net()` / `as_mamba2()` / `as_mlp_only()` /
   `as_convolution()` ladders.** The original scan found the pattern across roughly
   20 backend files. The current `Layer` is still a `std::variant`, yet important
   paths such as CUDA decode still finish with an unchecked catch-all
   `else -> *as_convolution(layer)`. Adding a mixer can therefore compile while
   leaving execution paths incomplete. (§2.1)
2. **CPU and CUDA serving expose duplicated backend-specific metrics/options and
   parallel engine surfaces even though backend-neutral runtime contracts already
   exist.** The sharpest concrete instance is `CpuConcurrentMetrics` versus
   `ConcurrentMetrics`. (§4)
3. **`CudaCompiledModel` remains a broad state-owning implementation type.** Its
   responsibilities span loading, execution, sampling, graphs, speculative state,
   persistence, metrics and packed execution. The newer `resources_` / `session_`
   decomposition is useful, but responsibility boundaries are still weakly
   enforced. (§5.2)
4. **`PackedSessionContext` remains a large mutable alias view with raw pointers and
   `void*` callbacks.** The original review overstated this as a demonstrated
   silent-corruption class. The current generation-aware metadata cache mitigates
   the known storage-replacement path. The real problem is a manually maintained
   alias/lifetime protocol with weak type safety. (§3.1)
5. **CUDA attention capability selection is encoded as nested control flow rather
   than an explicit capability policy.** BF16 and Int8 have different available
   kernels; that asymmetry may be legitimate, but the code does not make the reason
   or capability matrix explicit. (§6.2)
6. **`forward_token_host` and `forward_token_paged_host` duplicate large amounts of
   non-KV execution logic.** This is a real drift hazard because paged serving and
   standalone decode can receive fixes independently. (§7.1)
7. **`include/celeg/detail/model/types.hpp` has become a broad CUDA backend domain
   aggregate.** Hoisting nested implementation types improved friendship and local
   reuse, but did not by itself create narrow dependency boundaries. (§3.4)

The review should be used as an engineering plan, not as a demand to maximize
virtual interfaces or class counts. Prefer typed value/view boundaries, exhaustive
compile-time dispatch, capability policies and narrow state ownership. Introduce
runtime polymorphism only where a runtime substitution boundary actually requires
it and benchmark it when it sits near hot execution.

---

## 2. Open/Closed Principle

### 2.1 🔴 Mixer dispatch is not exhaustive

The layer representation already contains the right foundation:

```cpp
using Layer = std::variant<
    AttentionLayer,
    ConvolutionLayer,
    GatedDeltaNetLayer,
    Mamba2Layer,
    MlpOnlyLayer>;
```

But backend call sites commonly inspect it through helpers:

```cpp
as_attention(layer)
as_gated_delta_net(layer)
as_mamba2(layer)
as_mlp_only(layer)
as_convolution(layer)
```

The original audit measured this ladder across 14 CUDA files and 6 CPU files. That
count is a snapshot from `cf6315c`; the important point after revalidation is that
the pattern itself remains present in current execution/state paths.

The dangerous form is a catch-all branch such as:

```cpp
} else if (MlpOnlyLayer* mlp = as_mlp_only(layer)) {
    ...
} else {
    ConvolutionLayer& convolution = *as_convolution(layer);
    ...
}
```

That `else` does not mean "ConvolutionLayer" to the compiler. It means "anything
not matched above". If a sixth variant alternative is added, this code still
compiles and can dereference null.

**Required remediation:**

1. Replace catch-all mixer branches with exhaustive variant dispatch or another
   compile-time exhaustive mechanism.
2. Do not add a generic `auto` fallback that restores non-exhaustiveness.
3. Where a hot path uses `std::visit`, inspect generated code and benchmark the
   affected path. The design benefit is compile-time exhaustiveness; this review
   does **not** assume that every compiler necessarily lowers every visit to an
   identical jump table or that the change has provably zero cost.
4. After exhaustiveness is established, group cross-cutting mixer concerns
   (`allocate_state`, `reset`, `snapshot`, `restore`, `warmup`, `decode`,
   `prefill`, packed execution) so adding a mixer produces compiler failures at a
   small number of semantic extension points rather than a search across backend
   directories.

The success criterion is not "use `std::visit` everywhere". It is:

> adding a new `Layer` alternative must make every required backend concern fail to
> compile until that alternative is handled.

### 2.2 🟠 Attention launcher API is combinatorial

CUDA attention has a real specialization matrix across axes such as:

```text
{decode,prefill}
× {bf16,int8}
× {alibi,none}
× {paged,contiguous}
× {segmented,whole}
× {online,strict,...}
× {single,batch,device-ptrs,...}
```

Specialized kernels are legitimate. The problem is that the specialization matrix
leaks into many flat launcher names and long positional argument lists.

Two separate concerns should not be conflated:

- **kernel specialization:** often performance-justified;
- **launcher interface shape:** can be made safer without changing generated kernel
  code.

Prefer typed argument aggregates for coherent groups:

```cpp
struct GqaDecodeArgs {
    QueryView query;
    KvView kv;
    OutputView output;
    AttentionGeometry geometry;
    AttentionLaunchConfig launch;
};
```

Then specialize the implementation by compile-time capability axes where that is
measurably useful.

Do not merge distinct kernels merely to satisfy DRY if doing so adds divergent
branches or register pressure. The first target is the host-side API and duplicated
preambles, not necessarily the kernel inner loops.

### 2.3 🟠 Tensor naming must remain behind semantic roles

Backend weight loading should not know checkpoint-family spelling. Where short
convolution weights are loaded through literal names rather than the same semantic
role mechanism used elsewhere, the loader becomes an extension point for checkpoint
naming conventions.

The target remains:

```text
checkpoint-specific names
        ↓
TensorRole / weight request mapping
        ↓
backend weight loader
```

not:

```text
backend loader
  └── "conv.in_proj.weight"
  └── "conv.conv.weight"
  └── "conv.out_proj.weight"
```

Add/complete semantic roles for short-convolution input, convolution kernel and
output projection, and keep checkpoint naming out of `src/backend/*`.

---

## 3. Interface Segregation & Dependency Inversion

### 3.1 🟠 `PackedSessionContext` is a weakly typed alias protocol

`PackedSessionContext` is intended to expose only what packed execution needs. That
is a good goal. The current representation, however, contains a large collection of
non-owning pointers into model/session state plus callbacks carrying `void* owner`.
It also exposes mutators through `const PackedSessionContext&` because constness of
the view does not imply constness of the pointed-to state.

This weakens encapsulation and makes the contract difficult to audit.

The original review described the generation counter as if "nothing enforces" it
and concluded that the structure was already a demonstrated silent-memory-
corruption class. That conclusion was too strong.

Current code includes an explicit mitigation:

- `CudaCompiledModel::reset()` increments `storage_generation_` before replacing
  relevant session-owned storage;
- `PackedMetadataCache` keys staged metadata by both owner identity and storage
  generation;
- a generation change forces packed metadata to be considered changed and staged
  again.

Also, several members are pointers to stable owner objects such as `DeviceBuffer`
or `std::vector<Layer>` objects; resetting the allocation held by a `DeviceBuffer`
does not necessarily invalidate the address of the `DeviceBuffer` object itself.

So the accurate finding is:

> **Packed execution correctness depends on a manually maintained alias/lifetime
> protocol. The known reset/reallocation path is generation-aware, but every future
> operation that changes the identity or meaning of aliased storage must participate
> in that protocol correctly.**

That is meaningful architectural debt, but it is not evidence by itself of an
active use-after-free.

**Preferred remediation:** first replace the monolithic pointer bag with small typed
views/services, for example conceptually:

```text
PackedLane
├── PackedImmutableView
│   ├── compatibility key
│   ├── program/topology views
│   └── immutable weights/layout
│
├── PackedSessionStateView
│   ├── phase / position
│   ├── logits / sampling state
│   └── layer/session metrics
│
└── PackedExecutionServices
    ├── segmented-attention policy
    └── expert-residency service
```

Prefer typed references/pointers and explicit lifetime rules. Avoid `void*` owner
callbacks when the type system can express the dependency.

A virtual `IPackedLane` is one possible implementation, **not a required conclusion
of this review**. Use runtime polymorphism only if packed execution truly needs
runtime substitution across independent implementations. If a typed non-owning view
or static policy provides the boundary, prefer that simpler mechanism.

### 3.2 🟠 CPU operator contexts are inconsistently narrow

The CPU backend already has the right direction with `CpuExecutionContext`: operators
that need execution workspace and session profiling should not automatically receive
the full `CpuCompiledModel`.

The problem is inconsistent adoption. Some recurrent, MoE and attention operators
still receive the owner model, and attention paths may receive both a narrow context
and the broad owner.

Finish the separation using contexts based on actual responsibilities, e.g.:

```text
CpuExecutionContext
CpuAttentionStateView
CpuRecurrentStateView
CpuExpertExecutionView
```

The names are less important than the rule:

> an operator should receive only the state/services it is entitled to mutate or
> observe.

### 3.3 🟠 Broad implementation headers keep conceptual coupling high

Narrow `.cpp` files do not create narrow dependencies if their signatures require a
large implementation header.

The CPU operator headers still depend heavily on the model-internal aggregate. The
CUDA model implementation similarly pulls broad model state through
`compiled_model.hpp` and related detail headers.

Forward declarations alone will not solve this while APIs name nested/broad state
types. The fix is to establish narrow state/view types first and then narrow include
boundaries around them.

### 3.4 🟠 `detail/model/types.hpp` is now a CUDA backend domain aggregate

A useful refactor moved several implementation types out of nested
`CudaCompiledModel` scope. This removed some friendship pressure and made components
such as `WeightLoader`, `GemmDispatcher` and packed execution able to name those
types directly.

That was progress, but it did not fully create dependency inversion.

`include/celeg/detail/model/types.hpp` now contains or depends on concerns including:

- CUDA buffers and CUDA runtime types;
- cuBLAS/cuBLASLt plan state;
- linear and expert weight representations;
- checkpoint/GGUF-related representation details;
- expert offload and residency state;
- host expert cache state;
- per-layer execution objects;
- recurrent mixer state;
- MoE helpers;
- `Layer` / feed-forward variants.

This is better than one nested god class, but it can become a **god implementation
header** if every backend component includes it for one or two types.

Split by dependency direction, not merely file size. A plausible target is:

```text
include/celeg/detail/model/
├── linear_weights.hpp
├── expert_weights.hpp
├── layer_state.hpp
├── shared_weights.hpp
├── session_state.hpp
└── resources.hpp
```

Exact filenames are not important. The acceptance criterion is that a component
needing `LinearWeight` should not have to depend on expert caches, session state and
all layer alternatives merely because they were hoisted into one common header.

### 3.5 🟡 `GemmDispatcher` is a good extraction; finish its encapsulation

`GemmDispatcher` is one of the stronger backend abstractions: handle lifetime,
cuBLAS/cuBLASLt selection and plan caching belong together.

Keep that extraction. Small cleanup remains appropriate where internal helper
operations are unnecessarily public or where lifetime depends on a reference member
to caller-owned options. These are not reasons to redesign the dispatcher.

---

## 4. Backend-neutral serving boundaries

### 4.1 🔴 CPU and CUDA duplicate serving contracts

The original review placed this under Single Responsibility. The evidence remains
valid, but the better classification is:

- duplicated abstraction;
- incomplete backend-neutral boundary;
- DRY at the semantic/type level;
- Dependency Inversion.

CPU and CUDA expose parallel engine concepts with strongly overlapping operations:

```text
submit
poll
status
cancel
release
step
start
stop
metrics
```

The clearest concrete issue remains metrics. `ConcurrentMetrics` is intended as a
backend-neutral runtime snapshot, yet CPU still exposes a separate
`CpuConcurrentMetrics` containing many of the same counters and the same derived
TTFT/ITL concepts.

**First remediation:** make CPU populate the neutral metrics representation. If
CPU-only counters are genuinely useful, either promote them to backend-neutral
metrics where their semantics are universal or place them in an explicit extension
rather than cloning the entire aggregate.

### 4.2 🟠 Engine options need a shared semantic core

CPU and CUDA scheduler options share concepts such as active-request limits, batch
budgets, worker-thread behavior and prefix-cache policy. Their types and names should
not drift independently when the semantics are the same.

Create a backend-neutral scheduling/options core and keep only genuinely
backend-specific execution knobs in backend extensions.

### 4.3 🟠 A common engine contract is needed; a virtual base class is optional

The original remediation prescribed a common `IConcurrentEngine`. The architectural
requirement is slightly broader and less prescriptive:

> generic serving code must depend on a backend-neutral engine/request contract.

That contract can be implemented with:

- an abstract interface;
- type erasure;
- a template/concept boundary;
- existing service/driver interfaces composed at a higher level.

Do not introduce a virtual base class merely because two concrete classes have
similar APIs. Introduce the narrowest mechanism required by the actual runtime
substitution point.

Also remove the naming asymmetry where CUDA owns unprefixed generic-looking names
while CPU is explicitly prefixed if the type is in fact CUDA-specific.

---

## 5. Single Responsibility and ownership

### 5.1 🔴 Mega-functions remain high-leverage refactoring targets

The original audit identified several functions hundreds of lines long in weight
loading, decode and prefill. The exact line counts belong to the `cf6315c` snapshot,
but the structural issue remains visible in current CUDA decode/prefill and CPU
loading/forward paths.

Do not split these functions merely by extracting arbitrary 30-line helpers. Split
along stable semantic seams already implied by the program:

```text
layer iteration
mixer execution
attention capability selection
KV access policy
feed-forward execution
state transition
weight-role loading
```

This simultaneously addresses OCP, duplication and testability.

### 5.2 🔴/🟠 `CudaCompiledModel` remains a broad implementation owner

`CudaCompiledModel` currently owns or orchestrates, directly or through broad member
state:

- model configuration and checkpoint weight loading;
- resource allocation;
- prefill/decode/paged decode;
- CUDA graph capture;
- sampling;
- speculative/MTP state;
- session persistence;
- expert residency integration;
- metrics and diagnostics;
- packed-session binding;
- execution workspace and streams.

The newer `CudaModelResources`, `SessionState`, `CudaWorkspace`, `CudaSamplingState`,
`CudaDecodeGraphs` and `GemmDispatcher` members show that good seams already exist.
The next step is to make those seams own behavior as well as data where appropriate,
rather than leaving `CudaCompiledModel` as the coordinator for every concern.

Do not turn every member into a virtual service. Prefer cohesive concrete
collaborators with clear ownership.

### 5.3 🟠 Facade forwarding is boilerplate, but do not pierce the Pimpl boundary

`CpuModel` / `CudaModel` facade and session/diagnostics views contain forwarding
boilerplate. The original review proposed giving those public-facing views direct
`CpuCompiledModel*` / `CudaCompiledModel*` pointers.

That is not the preferred remediation.

The facade/Pimpl boundary has architectural value: public API views should not need
to know the deepest implementation type merely to save forwarding lines.

Reduce the boilerplate by moving the narrow operations into internal state/service
objects or by letting a view reference a narrow internal interface/state concept.
Do **not** solve a middle-man smell by coupling higher-level API types directly to
the backend god object.

### 5.4 🟡 Platform file I/O should be RAII-separated from expert indexing

Where expert sidecar storage combines POSIX descriptors, type-erased Windows handles
and sidecar parsing/indexing in one class, a small platform-file RAII layer remains
a sensible cleanup. This is a low-risk ownership improvement, not a priority over
execution boundaries.

---

## 6. Configuration and capability policy

### 6.1 🟠 Execution-relevant `getenv` state is invisible configuration

The original audit found execution/configuration decisions read directly from
environment variables, including:

```text
CELEG_FLASH_ATTN
CELEG_MMQ_TENSOR_CORES
CELEG_CUDA_MANAGED_WEIGHTS
```

`CELEG_FLASH_ATTN` is especially notable because it is cached in a function-local
`static const bool` in prefill execution.

The original review claimed this could allow two packed sessions with different
effective attention kernels to be judged compatible. For a process-global cached
environment value, that specific argument is not sound: two models in the same
process cannot independently choose different values through that mechanism.

The real problems are sufficient on their own:

- execution policy is not represented in model options;
- it is absent from normal diagnostics/config dumps;
- it is harder to reproduce a run from explicit configuration;
- tests cannot configure independent models cleanly in one process;
- a process-global `static` prevents programmatic per-model choice;
- hidden configuration can bypass the same validation/fingerprinting path used by
  explicit options.

Resolve environment values once at a configuration boundary. Environment variables
may remain convenient defaults/overrides, but backend execution should consume
explicit resolved options rather than call `getenv()`.

Only execution choices that can genuinely differ between simultaneously packed
lanes need to participate in `PackedCompatibilityKey`. Do not add fields to that key
merely because a value originated in the environment.

### 6.2 🟠 Int8/BF16 attention selection is an implicit capability matrix

Current prefill code exposes an asymmetry:

```text
Int8:
  alibi
  online
  strict

BF16:
  alibi
  flash
  gemm
  segmented
  strict
```

The original review called this "silent drift". That is stronger than the evidence.
The missing Int8 variants may be intentional because the corresponding kernels do
not exist, are not beneficial, or have different numerical/performance constraints.

The actual architectural finding is:

> **attention capabilities are encoded implicitly by branch structure, so a reader
> cannot distinguish an intentional unsupported combination from an implementation
> that simply never received a newer kernel path.**

Make the capability decision explicit, conceptually:

```cpp
struct AttentionCapability {
    KvFormat kv_format;
    PositionBias bias;
    AttentionOperation operation;
    AttentionAlgorithm algorithm;
    bool supported;
    UnsupportedReason reason;
};
```

This does not require a literal runtime table in a hot path. It can be compile-time
policy or a resolver that selects a launch plan before entering the deepest loop.
What matters is that supported/unsupported combinations become testable data or
exhaustive policy rather than accidental control-flow shape.

Add tests for at least:

- BF16 and Int8 strict paths;
- fast path selection by head dimension / row count;
- ALiBi combinations;
- paged vs contiguous where applicable;
- explicitly unsupported combinations with a reason;
- long-context fallback selection.

### 6.3 🟡 `CELEG_FLASH_ATTN` condition should state its real policy

The condition:

```cpp
if ((use_flash && flash_supported) ||
    (head_dim > 64 && flash_supported))
```

is clearer as:

```cpp
if (flash_supported && (use_flash || head_dim > 64))
```

That makes it obvious that the environment flag changes behavior only in part of the
supported head-dimension range. Whether that policy is desired should be tested and
documented rather than inferred from nested boolean structure.

---

## 7. DRY and semantic duplication

The original clone scan at `cf6315c` measured 2,166 redundant lines out of 26,520
backend lines (8.2%) across exact normalized clones. Keep those numbers as a
historical measurement, not as a continuously current metric.

Its most useful conclusion remains valid:

> CPU/CUDA duplication is mostly **structural**, while the largest textual
> duplication clusters occur **within** CUDA execution/kernel files.

### 7.1 🔴 Standalone and paged token-forward paths duplicate mixer execution

`forward_token_host` and `forward_token_paged_host` primarily differ in how
attention accesses KV state, yet substantial non-attention execution is duplicated.
The original scan found 156 identical non-comment lines and a 47-line verbatim run.

This is dangerous because a recurrent-mixer fix can land in standalone decode but
not paged serving, or vice versa.

Refactor around KV access / attention execution policy, not around two complete
forward functions:

```text
shared layer loop
  ├── shared non-attention mixers
  ├── shared residual / FFN semantics
  └── attention execution
       ├── contiguous KV accessor
       └── paged KV accessor
```

This work should be coordinated with exhaustive mixer dispatch (§2.1); they are the
same seam from two directions.

### 7.2 🟠 Kernel-family duplication needs performance-aware treatment

Duplicated launcher signatures, argument marshalling and identical preambles are
safe candidates for extraction. Duplicated inner kernel loops are not automatically
safe candidates.

Rule:

> deduplicate host-side shape and declaration boilerplate aggressively; deduplicate
> device inner loops only after codegen/performance validation.

Do not convert explicit specialized kernels into a single runtime-branch kernel for
architectural aesthetics.

### 7.3 🟠 Sampling should have one determinism contract

Where argmax/top-k reductions duplicate the same tie-break rule, that rule is a
user-visible determinism contract. Put the rule in one shared implementation or one
well-tested primitive so different sampling paths cannot evolve distinct tie
behavior.

Likewise, duplicate top-p filtering over sorted top-k candidates should be reduced
where doing so does not change kernel synchronization or memory behavior.

### 7.4 🟠 Linear-weight quantization tails should be shared

`load_linear_weight` and concat-linear loading historically duplicated Int8/Int4
quantize-and-store logic with renamed dimensions. Extract the backend storage step
once the dense buffer and final `(rows, cols)` are known.

A helper in this area should encode storage semantics, not checkpoint naming or
model-family rules.

### 7.5 🟡 Backend-neutral format utilities should be shared

Small utilities such as overflow-checked multiplication, IEEE half conversion and
GGUF block/nibble decoding should not be copied across CPU/CUDA implementations when
the operation is a property of the format rather than the backend.

Keep ISA-specific implementations separate where compiler flags or code generation
require it.

---

## 8. What is done well — preserve these properties

### 8.1 Architecture/model-family neutrality in the backend

This remains the strongest structural property in Celeg. Backend code should consume
semantic execution structures rather than compare model-family names.

Do not "simplify" backend code by reintroducing:

```text
if gemma...
if qwen...
if lfm...
if minicpm...
```

or `ModelArchitecture::X` execution branches.

When a new model fails, first ask whether it exposes a missing semantic primitive,
metadata shape, weight-role mapping or capability — not whether a model-specific
backend path should be added.

### 8.2 `CompiledModelProgram` as the backend contract

The compiled semantic program is the right direction: inference/import logic resolves
checkpoint observations upstream and backend code executes a neutral program.

Keep pushing topology and numerical decisions into that explicit program rather than
rediscovering them from tensor names or architecture IDs inside CPU/CUDA code.

### 8.3 Free-function kernel layer

CUDA/CPU kernels are naturally transformations over explicit buffers. Wrapping every
kernel family in an object hierarchy would add indirection and ownership semantics
where none are needed.

Keep free-function launch APIs; improve their typed argument boundaries and
capability selection instead.

### 8.4 Existing collaborators are real seams

Keep and strengthen abstractions such as:

- `GemmDispatcher`;
- `CudaWorkspace`;
- `CudaSamplingState`;
- `CudaDecodeGraphs`;
- `CudaModelResources`;
- `SessionState`;
- CPU execution contexts;
- backend-neutral runtime request/service contracts.

The next refactor should move behavior toward these cohesive seams rather than create
new naming layers that still forward everything to `CudaCompiledModel`.

### 8.5 Performance-specific closures can remain closed

ISA-specific files, kernel template specialization and intentionally closed GEMM or
weight-format dispatch domains are legitimate. The requirement is that the closure
is explicit and that extension failure is detectable, preferably at compile time or
through a capability test.

---

## 9. Revised prioritized remediation

The order below supersedes the original priority table.

| # | Work item | Sev | Effort | Primary payoff |
|---|---|---|---|---|
| 1 | Make all `Layer` handling exhaustive; remove unchecked catch-all mixer branches | 🔴 | M | New mixer => compile-time worklist instead of latent gaps |
| 2 | Delete `CpuConcurrentMetrics` duplication and populate backend-neutral metrics | 🔴 | S | One serving metrics contract |
| 3 | Resolve execution-relevant environment variables into explicit configuration | 🟠 | S | Reproducible/testable execution policy |
| 4 | Introduce explicit attention capability/launch policy for KV format × bias × algorithm | 🟠 | M | Unsupported combinations become visible/testable |
| 5 | Unify paged and contiguous token-forward structure around KV/attention policy | 🔴 | M/L | Removes high-risk decode duplication |
| 6 | Split mega-functions along mixer/KV/weight-role semantic seams | 🔴 | L | Testability + reduced change surface |
| 7 | Move more behavior out of `CudaCompiledModel` into existing cohesive collaborators | 🟠 | L | Real SRP instead of file-only decomposition |
| 8 | Split `detail/model/types.hpp` by dependency direction | 🟠 | M | Lower compile/conceptual coupling |
| 9 | Replace `PackedSessionContext` pointer bag with typed views/services | 🟠 | M/L | Stronger lifetime/type contract |
| 10 | Finish narrow CPU execution/state contexts | 🟠 | M | Operator-level ISP |
| 11 | Share scheduler option semantics across CPU/CUDA; define a neutral engine boundary | 🟠 | M | Backend-agnostic serving |
| 12 | Struct-ify long attention launcher argument lists | 🟠 | M | Eliminates positional argument mistakes |
| 13 | Complete semantic tensor roles for short convolution | 🟠 | S | Keeps checkpoint naming out of backend |
| 14 | Dedupe sampling determinism/filter primitives and quantize/store tails | 🟠 | S/M | Prevents semantic drift |
| 15 | Share backend-neutral format/math utilities; clean umbrella-header contradictions | 🟡 | S | Cheap maintenance cleanup |

### Sprint A — correctness and explicit policy

Do #1, #2, #3, #4 and #13 first.

This creates the guardrails needed for future architectures/backends without making
a large ownership refactor first.

### Sprint B — execution structure

Do #5 and #6 together, then #10 and #12 where they naturally touch the same code.

The paged/non-paged duplication, mixer ladder and deep attention dispatch are not
independent problems; they are different symptoms of the same missing execution
policies.

### Sprint C — ownership/dependency boundaries

Do #7, #8 and #9 after the execution policies are explicit.

Do **not** begin by introducing `IPackedLane` or a large virtual hierarchy. First
identify the actual immutable view, mutable session state and services packed
execution needs. If a runtime-polymorphic lane interface remains useful after that,
introduce it deliberately and benchmark the call boundary.

---

## 10. Acceptance criteria

This review is complete only when the architecture changes are mechanically
verifiable. Suggested gates:

1. **Mixer exhaustiveness gate**
   - no production layer-dispatch path ends with an unchecked "everything else is
     convolution" branch;
   - adding a dummy sixth `Layer` alternative in a compile-only test causes required
     backend visitors/policies to fail compilation.
2. **Backend-neutral metrics gate**
   - one neutral concurrent metrics contract is populated by CPU and CUDA;
   - backend-specific diagnostics, if any, are explicit extensions rather than full
     copies.
3. **Configuration gate**
   - execution kernels/model paths do not call `getenv()` directly;
   - environment overrides are resolved at configuration/bootstrap boundaries.
4. **Attention capability gate**
   - supported and unsupported combinations of KV format, position bias, layout and
     algorithm are covered by policy tests;
   - an unsupported combination fails deliberately, not by falling through to an
     unrelated kernel.
5. **Paged/contiguous parity gate**
   - non-attention mixer execution is implemented once or tested through a shared
     policy;
   - recurrent mixer parity is exercised under paged serving.
6. **Packed lifetime gate**
   - packed execution has no untyped `void*` owner callback unless justified by a
     measured boundary;
   - storage invalidation rules are encoded in typed ownership/generation contracts
     and tested across reset/reallocation.
7. **Dependency gate**
   - components needing only linear weights do not require the full CUDA model domain
     aggregate;
   - `detail/model/types.hpp` is reduced or split so unrelated concerns do not travel
     together.
8. **Architecture-neutrality gate**
   - backend CI rejects reintroduction of concrete model-family execution branches or
     architecture-name dispatch.
9. **Performance gate**
   - visitor/policy refactors on decode/prefill include before/after benchmarks;
   - kernel-inner-loop deduplication is accepted only with equivalent or improved
     codegen/performance.

---

## 11. Comments and design claims

Design comments should describe guarantees the type system or tests actually
support.

Avoid comments that claim an ISP/OCP boundary merely because code was moved to a new
file or because callbacks replaced a direct call. Prefer statements that name the
real contract, for example:

```text
This view is non-owning and valid until session storage generation changes.
PackedMetadataCache includes storage generation in its staging identity.
```

or:

```text
This dispatch domain is intentionally closed because each alternative maps to a
separately benchmarked kernel specialization. Adding an alternative requires the
capability matrix and benchmark suite to be updated.
```

Aspirational comments are useful only when marked as TODOs. Otherwise they make
structural debt harder to see.

---

## 12. Final assessment

Celeg's backend architecture is in a good intermediate state: the project has already
won the hard battle of moving concrete model-family knowledge out of execution and
representing model semantics upstream. The remaining debt is mostly inside backend
execution composition:

```text
semantic program                         ✅ strong
model-family neutrality                  ✅ strong
kernel specialization                    ✅ appropriate
file-level decomposition                 ✅ substantially improved
exhaustive mixer extension points        ❌ incomplete
backend-neutral serving contracts        ⚠️ partial
attention capability policy              ⚠️ implicit
implementation-state boundaries          ⚠️ broad
packed-session type/lifetime boundary     ⚠️ manual
paged/contiguous execution reuse         ❌ duplicated
```

The next phase should therefore optimize for **semantic extension points and explicit
capabilities**, not for more files or more interfaces. A new mixer, KV format, GPU
backend or attention algorithm should make the compiler/tests identify the exact
places that require implementation, while model-family names remain absent from the
backend entirely.
