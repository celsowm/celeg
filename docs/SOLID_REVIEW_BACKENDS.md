# SOLID & Anti-Pattern Review — `src/backend/*`

Scope: `src/backend/cpu/**` (65 files) and `src/backend/cuda/**` (94 files), plus the
headers they publish under `include/celeg/backend/**` and the private headers they
depend on (`include/celeg/detail/model/**`, `src/backend/cpu/detail/**`,
`src/backend/cuda/runtime/engine_internal.hpp`).

Reviewed at commit `cf6315c`.

---

## 0. Reading this document

Celeg is a high-performance inference engine. That fact changes what counts as a
defect: a hand-unrolled AVX2 kernel with a 20-argument signature is *not* a design
failure, and a `switch` in a hot loop is not automatically an OCP violation. This
review therefore separates:

- **Structural findings** — design decisions that cost you *change velocity* and
  *correctness confidence*, and would keep costing you as more architectures land.
- **Deliberate closures** — places where the code chose coupling for speed, said so
  in a comment, and is right to.
- **False economies** — places where the code *claims* a SOLID benefit in a comment
  but the actual structure does not deliver it. These are the most expensive
  findings, because they suppress the instinct to fix the real problem.

Findings are ranked by cost, not by principle.

Severity legend: 🔴 high (actively taxing every change) · 🟠 medium (taxes some
changes, will worsen) · 🟡 low (cosmetic / consistency).

---

## 1. Executive summary

The backend layer is **well-decomposed at the file level and poorly decomposed at
the type level**. Somebody has clearly done the work of splitting 5,000-line
translation units into 40 focused ones — `src/backend/cuda/model/` alone has 34
files with names like [`attention_execution.cu`](../src/backend/cuda/model/attention_execution.cu),
[`expert_setup.cpp`](../src/backend/cuda/model/expert_setup.cpp),
[`weight_upload.cpp`](../src/backend/cuda/model/weight_upload.cpp). That is real
progress and it shows.

But those 34 files are all **members of the same class**. `CudaCompiledModel`
([compiled_model.hpp:39-250](../include/celeg/detail/model/compiled_model.hpp#L39-L250))
declares ~60 public methods and ~20 public data members across weight loading,
graph capture, sampling, speculative decoding, paged KV, MoE residency and metrics.
Splitting its *definition* across 34 files did not split its *responsibility*; it
distributed one god object over a directory. 29 of the 94 CUDA backend files
`#include` that single header, so the compile-time and conceptual coupling is
unchanged from the monolith it replaced.

The CPU backend has the same shape (`CpuCompiledModel`,
[model_internal.hpp:181-455](../src/backend/cpu/detail/model_internal.hpp#L181-L455))
but is meaningfully further along: it extracted real collaborators
(`CpuExecutionContext`, `operators/*.hpp`, `detail/admission_controller.hpp`) and
those extractions have teeth. The CUDA backend's equivalent extraction,
`PackedSessionContext`, does not — see §3.1.

**The three findings that actually cost you:**

1. **Layer-kind dispatch is open-coded in 14 CUDA files and 6 CPU files.** Adding a
   mixer means finding and editing all 20. (§2.1)
2. **`PackedSessionContext` is a 24-pointer window into `CudaCompiledModel`'s
   privates** that markets itself as an ISP boundary. It is the single most
   dangerous structure in the backend. (§3.1)
3. **CPU and CUDA duplicate the entire engine/scheduler/metrics stack** with no
   shared abstraction, two incompatible options structs and two incompatible
   metrics structs — while a backend-neutral `ConcurrentMetrics` already exists and
   is used by only one of them. (§4.1)
4. **`forward_token_paged_host` is 58% a verbatim copy of `forward_token_host`**
   (156 identical lines, one 47-line block byte-for-byte) — the mixer ladder from
   finding 1, duplicated wholesale rather than shared. A fix applied to one and not
   the other is correct in single-session tests and wrong under concurrent paged
   serving. (§7.1)

**What is genuinely good** (do not "fix" these): the kernel layer's free-function
`launch_*` API; `CompiledModelProgram` as the architecture-neutral IR — there is
*not a single* `ModelArchitecture::` comparison anywhere in `src/backend/`, which
is a remarkable and hard-won property; `GemmDispatcher`'s extraction of cuBLAS
lifetime; and the CPU `operators/` split.

---

## 2. Open/Closed Principle

### 2.1 🔴 Layer-kind dispatch is open-coded in 20 files

The mixer taxonomy (Attention / GatedDeltaNet / Mamba2 / MlpOnly / Convolution) is
resolved by an if-else ladder over `as_attention()` / `as_gated_delta_net()` /
`as_mamba2()` / `as_mlp_only()` / `as_convolution()`. That ladder appears in:

| File | Ladder sites |
|---|---|
| [`cuda/model/decode.cpp`](../src/backend/cuda/model/decode.cpp) | 13 |
| [`cuda/packed/operators.cu`](../src/backend/cuda/packed/operators.cu) | 6 |
| [`cuda/model/speculative_state.cu`](../src/backend/cuda/model/speculative_state.cu) | 6 |
| [`cuda/model/session_resources.cpp`](../src/backend/cuda/model/session_resources.cpp) | 6 |
| [`cuda/model/execution.cu`](../src/backend/cuda/model/execution.cu) | 5 |
| [`cuda/packed/metadata.cu`](../src/backend/cuda/packed/metadata.cu) | 4 |
| [`cuda/packed/layer_executor.cu`](../src/backend/cuda/packed/layer_executor.cu) | 4 |
| [`cuda/model/prefill_batched.cpp`](../src/backend/cuda/model/prefill_batched.cpp) | 4 |
| [`cuda/model/persistence.cu`](../src/backend/cuda/model/persistence.cu) | 4 |
| [`cuda/model/non_attention_execution.cu`](../src/backend/cuda/model/non_attention_execution.cu) | 4 |
| [`cuda/model/warmup.cpp`](../src/backend/cuda/model/warmup.cpp) | 3 |
| [`cuda/model/attention_execution.cu`](../src/backend/cuda/model/attention_execution.cu) | 2 |
| [`cuda/packed/validation.cu`](../src/backend/cuda/packed/validation.cu) | 1 |
| [`cuda/model/mtp_execution.cu`](../src/backend/cuda/model/mtp_execution.cu) | 1 |

Plus the CPU mirror in [`model_forward_token.cpp:83-96`](../src/backend/cpu/model_forward_token.cpp#L83-L96),
[`model_forward_chunk.cpp`](../src/backend/cpu/model_forward_chunk.cpp),
[`packed/execution.cpp`](../src/backend/cpu/packed/execution.cpp),
[`model_state.cpp`](../src/backend/cpu/model_state.cpp) (26 variant-dispatch sites),
[`weights_loader.cpp`](../src/backend/cpu/weights_loader.cpp) and
[`weights.cpp`](../src/backend/cpu/weights.cpp).

Adding a mixer therefore touches **~20 files across 6 concerns** — execution, packed
execution, weight setup, state allocation, persistence, warmup — and the compiler
helps you with **none** of them. Every ladder ends in a silent `else` fallthrough
rather than an exhaustive match:

```cpp
// cuda/model/decode.cpp:232-244
} else {
    ConvolutionLayer& convolution = *as_convolution(layer);   // ← unchecked deref
```

A new mixer that reaches this line dereferences null. Compare the CPU path at
[`model_forward_token.cpp:92-95`](../src/backend/cpu/model_forward_token.cpp#L92-L95),
which *does* check and throws `std::logic_error` — the CPU backend is safer here,
and the inconsistency between the two is itself a finding.

**Why this is real and not principle-worship:** you already have the right
abstraction. `CompiledMixer` is a closed enum in the compiled program, and
`std::variant<...>` holds the layer. `std::visit` with an exhaustive overload set
turns all 20 of these into compile errors when a mixer is added. The variant is
*already there* — [`model_internal.hpp:270-272`](../src/backend/cpu/detail/model_internal.hpp#L270-L272)
uses it. The CUDA side just doesn't visit it.

**Remediation (ordered by payoff):**

1. Replace the `else` fallthroughs with exhaustive `std::visit` over an overload
   set that has no generic `auto` case. This is mechanical, has zero runtime cost
   (`std::visit` over a 5-alternative variant compiles to a jump table), and
   converts 20 silent-corruption sites into 20 compile errors.
2. Collapse the six *concerns* into a per-mixer policy type
   (`allocate_state`, `snapshot`, `restore`, `warmup`, `execute_decode`,
   `execute_prefill`) so a new mixer is one new file, not 20 edits. Keep the
   execute methods non-virtual and dispatch via the variant — this preserves
   inlining in the hot loop.

Do (1) now regardless of whether you ever do (2).

### 2.2 🟠 Combinatorial kernel-launcher explosion

[`include/celeg/backend/cuda/kernels/attention.hpp`](../include/celeg/backend/cuda/kernels/attention.hpp)
declares **38 launchers**, 25 of which are `launch_gqa_{decode,prefill}_*`. They are
a cross-product of orthogonal axes:

```
{decode, prefill} × {bf16, int8} × {alibi, none} × {paged, contiguous}
                  × {segmented, whole} × {online, strict} × {device, host, batch_ptrs}
```

Yielding names like `launch_gqa_decode_int8_paged_segmented_batch` (26 parameters)
and `launch_gqa_decode_alibi_int8_paged_batch` (22 parameters). Adding a KV format
(say FP8) is not "one kernel" — it is a new column in a 25-row table, and every
caller ladder in §2.1 grows a branch.

This is the classic combinatorial-API anti-pattern, and here it is **partly
justified**: these are `__global__` launch configurations where a runtime branch
inside the kernel costs occupancy and register pressure. Template specialization
over the axes is the standard resolution and preserves every bit of the codegen:

```cpp
template <KvFormat Kv, PositionBias Bias, KvLayout Layout, AttentionMode Mode>
void launch_gqa_decode(const GqaDecodeArgs& args);
```

The 26-parameter signature is a separate, unambiguous problem — those parameters
are not orthogonal, they are five coherent groups (query view, KV view, output
view, geometry, launch config). A `struct GqaDecodeArgs` with designated
initializers costs nothing at runtime and eliminates the argument-transposition
bug class entirely, which at 26 positional parameters is not hypothetical.

**Verdict:** the axis explosion is defensible; the flat naming and the 26-parameter
positional signatures are not. Fix the signatures first — it is lower risk and
higher value.

### 2.3 🟠 Hardcoded tensor names bypass the role system

The weight system resolves names through `TensorRole` so checkpoint naming
conventions stay out of the loader. `ConvolutionLayer` opts out:

```cpp
// cuda/model/weight_setup.cpp:687-697
convolution_layer.conv_in = ...load_linear_weight(
    repo, layer_name(i, "conv.in_proj.weight"), ...);
convolution_layer.conv_weight = ...load_weight(
    repo, layer_name(i, "conv.conv.weight"), ...);
convolution_layer.conv_out = ...load_linear_weight(
    repo, layer_name(i, "conv.out_proj.weight"), ...);
```

Every other layer kind in the same function uses
`tensor_name(requests, TensorRole::X, i)`. A checkpoint with a differently-named
short-conv block cannot be supported without editing this function, whereas any
other layer kind is supported by adding a role mapping. Add
`TensorRole::ShortConv{Input,Weight,Output}` and delete the literals.

---

## 3. Interface Segregation & Dependency Inversion

### 3.1 🔴 `PackedSessionContext` — a false ISP boundary

[`include/celeg/backend/cuda/packed/session.hpp:56-134`](../include/celeg/backend/cuda/packed/session.hpp#L56-L134)

The header's own comment states the intent clearly:

> *"Operation-specific, non-owning context for one lane... It contains only the
> resources that the packed executor is allowed to touch; model loading, graph
> capture, and unrelated session operations are deliberately absent."*

What the struct actually contains: **24 raw non-owning pointers into
`CudaCompiledModel`'s private state**, one `shared_ptr`, and **two C function
pointers with a `void* owner`**:

```cpp
void* owner = nullptr;
uint64_t storage_generation_value = 0;
SessionPhase*  phase_state = nullptr;
int*           position_state = nullptr;
bool*          active_segmented_attention_state = nullptr;
DeviceBuffer<__nv_bfloat16>* logits_state = nullptr;
std::vector<Layer>*          layers_state = nullptr;
RuntimeMetrics*              metrics_state = nullptr;
// ...18 more...
SegmentedAttentionFn segmented_attention = nullptr;   // bool(*)(const void*, int)
ExpertResidencyFn    ensure_expert_residency = nullptr;
```

This is not interface segregation. It is **`CudaCompiledModel`'s private section,
re-exported as a public struct of mutable aliases**, with the encapsulation removed
rather than narrowed. Concretely:

- **It is strictly *more* dangerous than passing `CudaCompiledModel&`.** A reference
  can only be misused through the class's own invariants. This struct hands out
  `int* position_state` and `bool* active_segmented_attention_state` — raw writable
  pointers to fields the owner believes it controls. `set_phase()` is `const` and
  mutates through the pointer, so `const PackedSessionContext&` provides no
  protection whatsoever.
- **Lifetime is managed by a hand-rolled generation counter.**
  `storage_generation_value` exists (per its own comment) because "pointer identity
  alone is insufficient — a reset may replace a buffer while retaining the same
  model owner." That is an admission that these 24 pointers dangle under a
  documented, reachable code path, and the mitigation is a manually-incremented
  `uint64_t` that nothing enforces. Every future `reset` path must remember to bump
  `storage_generation_` ([compiled_model.hpp:238](../include/celeg/detail/model/compiled_model.hpp#L238))
  or the executor silently reads freed device memory.
- **The callbacks defeat type safety to avoid an abstraction the code needed.** The
  comment says they *"avoid a virtual getter interface and keep architecture and
  dispatch decisions out of the packed executor's layer loop."* They do neither: the
  layer loop in [`packed/operators.cu`](../src/backend/cuda/packed/operators.cu)
  still runs the §2.1 mixer ladder. What was actually purchased is
  `bool(*)(const void*, int)` plus a `static_cast<CudaCompiledModel*>` on the far
  side ([compiled_model.hpp:228-232](../include/celeg/detail/model/compiled_model.hpp#L228-L232)),
  which is a devirtualized call at the cost of the compiler's ability to check the
  `owner` type. These two callbacks fire **once per layer per step**, not per
  element — the virtual-call saving is unmeasurable against a `cudaLaunchKernel`.

**Remediation.** The genuine requirement — "the packed executor must see N sessions
uniformly without owning them" — is served by a narrow abstract interface:

```cpp
class IPackedLane {   // implemented by CudaCompiledModel
public:
    virtual ~IPackedLane() = default;
    virtual PackedCompatibilityKey compatibility() const = 0;
    virtual LaneExecutionView view() = 0;     // small by-value struct of the
                                              // ~6 buffers actually touched
    virtual bool use_segmented_attention(int position) const = 0;
    virtual void ensure_expert_residency(int layer, const int* sel, int rows,
                                         cudaStream_t, const float* scores) = 0;
};
```

`LaneExecutionView` is rebuilt per step from live accessors, so the generation
counter disappears with the dangling-pointer class it was invented to paper over.
The two virtuals cost ~2 indirect calls per layer per step. Measure it — I expect
it to be unmeasurable, and if it is not, `ensure_expert_residency` can stay a
callback while everything else moves.

**This is the highest-value fix in the review.** It is the only finding here where
the current design can produce silent memory corruption rather than merely slowing
you down.

### 3.2 🟠 Inconsistent operator context in the CPU backend

The CPU backend built exactly the right abstraction — and then used it for half the
operators. From [`model_internal.hpp:449-453`](../src/backend/cpu/detail/model_internal.hpp#L449-L453):

```cpp
struct CpuExecutionContext {
    CpuCompiledModel::Shared& shared;
    CpuWorkspace& workspace;
    CpuCompiledModel::CpuSessionState& session;
};
```

Its comment: *"Operators that only need linear execution, scratch storage, and
session profiling no longer receive the owning model object and cannot accidentally
reach unrelated state."* Correct, and well-judged. But:

| Operator | Takes |
|---|---|
| `execute_cpu_mlp_only_token` | `CpuExecutionContext&` ✅ |
| `execute_cpu_dense_feed_forward_token/chunk` | `CpuExecutionContext&` ✅ |
| `execute_cpu_gated_delta_token/chunk` | `CpuCompiledModel&` ❌ |
| `execute_cpu_mamba2_token` | `CpuCompiledModel&` ❌ |
| `execute_cpu_short_convolution_token/chunk` | `CpuCompiledModel&` ❌ |
| `execute_cpu_moe_token/chunk` | `CpuCompiledModel&` ❌ |
| `execute_cpu_attention_token` | **both** ❌❌ |

(Sources: [`operators/feed_forward.hpp`](../src/backend/cpu/operators/feed_forward.hpp),
[`operators/recurrent.hpp`](../src/backend/cpu/operators/recurrent.hpp),
[`operators/moe.hpp`](../src/backend/cpu/operators/moe.hpp),
[`operators/attention.hpp`](../src/backend/cpu/operators/attention.hpp).)

`execute_cpu_attention_token(CpuExecutionContext&, CpuCompiledModel&, ...)` takes
both, which is the worst option available: the narrow context provides no
protection while the god object is also in scope, and a reader cannot tell which
state the function is entitled to touch.

The recurrent/attention operators need `run_attention`, `store_kv`,
`attention_state(i)` and `release_attention_pages` — these belong on a second
narrow view (`CpuAttentionContext`) that carries the KV pool and layer state, not
on the whole model.

### 3.3 🟠 Every CPU operator header includes the god header

All four `src/backend/cpu/operators/*.hpp` begin with
`#include "../detail/model_internal.hpp"` — a 466-line header that transitively
pulls in the pack format, prefix cache, thread pool, NUMA, quantization, expert
backing and `CompiledModelProgram`. So every operator TU depends on every backend
concern, and touching the pack format recompiles the MoE kernels. The same holds on
the CUDA side: 29 of 94 files include `detail/model/compiled_model.hpp`, which
itself includes 20 headers including `<cublasLt.h>` transitively.

This is a direct consequence of §1 (types not decomposed). The narrow contexts from
§3.1/§3.2 are the fix; forward declarations alone will not help while the operator
signatures name nested types of the god class.

### 3.4 🟡 `GemmDispatcher` leaks its internals and holds a reference member

[`gemm_dispatcher.hpp`](../include/celeg/backend/cuda/gemm_dispatcher.hpp) is one of
the better-factored types here — the extraction of cuBLAS/cuBLASLt handle lifetime
and plan caching out of the model is correct and the header says why. Two blemishes:

- Lines 96-102 make `get_or_create_lt_plan` and `begin_native_fanout` /
  `end_native_fanout` public, with the comment *"Internal scope operations used by
  NativeFanoutScope"*. `NativeFanoutScope` is a nested class and already has access;
  these belong in `private:`.
- Line 155: `const CudaModelOptions& options_;` — a reference member to
  caller-owned storage. `CudaCompiledModel` owns `resources_.options_` and outlives
  the dispatcher today, so it is correct *by construction order*, silently. Since
  `CudaModelOptions` is a plain aggregate, store it by value and remove the
  invariant.

### 3.5 🟡 Weight-source triple-threading in the CPU loader

Every CPU weight load threads three mutually-constrained sources:

```cpp
// cpu/weights_loader.cpp — repeated ~40 times
load_matrix(IWeightRepository* repository,   // null iff reading from pack
            CpuPackReader*     reader,       // null iff writing/repository
            CpuPackWriter*     writer,       // null iff reading from pack
            const std::string& name, const std::vector<int64_t>& expected);
```

`source = reader ? nullptr : repository.get()`
([weights_loader.cpp:134](../src/backend/cpu/weights_loader.cpp#L134)) establishes an
invariant that all ~40 call sites must preserve, and no type enforces. This is the
control-coupling / flag-argument anti-pattern at scale. One `ICpuWeightSource`
implemented by `PackSource` and `RepositorySource` (with the writer as an optional
observing sink) reduces every call to `source.matrix(name, expected)`.

---

## 4. Single Responsibility

### 4.1 🔴 CPU and CUDA duplicate the entire serving stack

Two engines, structurally identical, sharing nothing:

| Concern | CPU | CUDA |
|---|---|---|
| Engine facade | `CpuConcurrentEngine` ([concurrent.hpp:70](../include/celeg/backend/cpu/concurrent.hpp#L70)) | `ConcurrentEngine` ([concurrency.hpp:41](../include/celeg/backend/cuda/concurrency.hpp#L41)) |
| Options | `CpuConcurrentEngineOptions` (11 fields) | `ConcurrentEngineOptions` (14 fields) |
| Metrics | `CpuConcurrentMetrics` (**35 fields**) | `ConcurrentMetrics` (**45 fields**) |
| Driver | `CpuSchedulerDriver` | `CudaSchedulerDriver` |
| Admission | `CpuAdmissionController` | `admit_requests_locked()` |
| Service | `CpuInferenceService` | `CudaInferenceService` |

The two engines expose a **byte-identical public API** — `submit`, `poll`, `status`,
`cancel`, `release`, `step`, `start`, `stop`, `metrics` — with identical semantics
down to the `release` contract comment, which is copy-pasted verbatim into both
headers. Yet they share no base class, and `CpuConcurrentEngine` cannot be passed
anywhere `ConcurrentEngine` is accepted.

The metrics duplication is the sharpest instance. `ConcurrentMetrics` in
[`include/celeg/runtime/concurrency/metrics.hpp`](../include/celeg/runtime/concurrency/metrics.hpp)
is *explicitly documented* as *"Backend-neutral aggregate snapshot... Backend-specific
engines populate this value at their boundary."* The CUDA engine populates it. The
CPU engine defines its own 35-field struct with ~24 identically-named fields
(`prefill_tokens`, `prefix_cache_hits`, `cumulative_ttft_ms`, `ttft_samples`,
`prefix_cow_bytes`…) and re-implements `average_ttft_ms()` and `average_itl_ms()`
identically. The backend-neutral type exists, is named as such, and is used by
exactly one backend.

There is also an implicit hierarchy embedded in the naming: CUDA types are
unprefixed (`ConcurrentEngine`, `ConcurrentEngineOptions`), CPU types are prefixed.
The unprefixed names read as "the" engine, which will keep pulling generic code
toward CUDA-specific assumptions.

**Remediation:**
1. Delete `CpuConcurrentMetrics`; have the CPU engine populate `ConcurrentMetrics`.
   Add the two CPU-only fields (`chunked_prefill_*`, `attention_parallel_calls`) to
   the neutral struct or a small extension. This is a mostly-mechanical change with
   an immediate payoff: one metrics path in the serve layer.
2. Unify the options structs on a shared core plus per-backend extensions —
   `max_active_requests`, `max_batched_tokens`, `prefill_chunk_tokens`,
   `worker_thread` and the prefix-cache knobs are already common (and gratuitously
   differ in type: `size_t` vs `int`).
3. Give both engines a common `IConcurrentEngine` — `serve::IRequestService` +
   `ISchedulerDriver` already define exactly this shape and both services already
   implement them. The engines are one level below and should mirror it.

### 4.2 🔴 Functions that are entire files

| Function | File | Lines |
|---|---|---|
| `load_checkpoint_weights` (one lambda body) | [`cuda/model/weight_setup.cpp:36-705`](../src/backend/cuda/model/weight_setup.cpp#L36-L705) | **~665** |
| `forward_token_host` | [`cuda/model/decode.cpp:15-286`](../src/backend/cuda/model/decode.cpp#L15-L286) | 271 |
| `forward_token_paged_host` | [`cuda/model/decode.cpp:288-682`](../src/backend/cuda/model/decode.cpp#L288-L682) | 394 |
| `prefill_batched` | [`cuda/model/prefill_batched.cpp:20-569`](../src/backend/cuda/model/prefill_batched.cpp) | ~549 |
| `Shared::load_weights` | [`cpu/weights_loader.cpp:102-685`](../src/backend/cpu/weights_loader.cpp#L102-L685) | ~583 |
| `forward_chunk` | [`cpu/model_forward_chunk.cpp:31-582`](../src/backend/cpu/model_forward_chunk.cpp) | ~550 |

`load_checkpoint_weights` deserves specific mention: the function body is a single
`CudaWeightSetup::load(*this, path, bootstrap, [this](const IWeightRepository& repo) {
... 660 lines ... })`. The lambda-as-callback shape gives the *appearance* of
inversion — a `LayerLoader` strategy is being injected! — while the strategy is one
non-reusable 660-line closure over `this`. Neither is unit-testable below
full-model granularity. See §6.6 for what the length does to nesting depth.

These are also the files where §2.1's ladder lives, so the two findings compound:
the ladder is long *because* the function is long, and the function is long *because*
each ladder arm inlines the whole per-mixer body.

### 4.3 🟠 `CudaCompiledModel` is a god object

[compiled_model.hpp:39-250](../include/celeg/detail/model/compiled_model.hpp#L39-L250) —
one struct owning:

- checkpoint loading (`load_checkpoint_weights`, `configure_model`)
- resource allocation (`allocate_celeg_resources`, `allocate_prefill_workspace`)
- the forward pass, decode, prefill, paged prefill, batched prefill
- CUDA graph capture (`capture_decode_graph`, `graph_for_attention`)
- sampling (`enqueue_sampling`, RNG state, `sampled_host/device`)
- speculative decoding + MTP (`snapshot_speculative_state`, `run_mtp_forward`, …)
- session persistence (`save_session`, `load_session`, prefix export/restore)
- MoE expert residency callbacks
- warmup, benchmarking, metrics, memory stats

with **~20 public data members**, including `stream_`, `workspace_`, `sampling_` and
a 14-field `SpeculativeStateSnapshot`. There is no `private:` section.

The `resources_` / `session_` split (`CudaModelResources` for immutable topology vs
`SessionState` for per-request state) is the right seam and is already drawn — it is
just not enforced, since everything is public and `CudaCompiledModel` reaches
through both freely. Making those members private and routing through the existing
accessors would be a cheap first step that immediately reveals which of the 29
including files actually need what.

### 4.4 🟠 `CpuModel` is a pure middle man

[`include/celeg/backend/cpu/model.hpp:150-196`](../include/celeg/backend/cpu/model.hpp#L150-L196)
declares 21 private `session_*` methods (`session_isa()`, `session_vocab_size()`,
`session_backend_description()`, …) that exist solely so the three friend views
(`CpuInferenceSession`, `CpuDiagnostics`, `CpuPersistence`) can forward to
`state_->…`. [`cpu/model.cpp`](../src/backend/cpu/model.cpp) is 306 lines containing
47 such forwarders and essentially no logic.

The view/facade split itself is good — `CpuDiagnostics` genuinely is a narrower
surface than `CpuModel`. The waste is the double hop: view → `CpuModel` private
method → `state_`. Give each view a `CpuCompiledModel*` at construction and delete
the middle layer (~250 lines and 16 `friend` declarations across the two backends
go with it). The same pattern exists on `CudaModel`.

### 4.5 🟡 `ExpertSidecar` hand-rolls cross-platform file I/O

[`moe/offload.hpp:69-75`](../include/celeg/backend/cuda/moe/offload.hpp#L69-L75):

```cpp
int    fd_ = -1;             // POSIX
void*  file_handle_ = nullptr;  // Windows HANDLE, type-erased
```

One class holds both platforms' handles, with `valid()` returning
`fd_ >= 0 || file_handle_ != nullptr` and a manual destructor. Two failure modes:
`void*` erases `HANDLE`, so the Windows path is unchecked by the type system; and
the class mixes platform resource management with sidecar indexing/parsing. A
small RAII `PlatformFile` (or the existing `ExpertIoBackend` enum realized as an
interface — `ThreadPool` / `IoUring` / `WindowsOverlapped` are already enumerated
there, suggesting this was intended) separates the two.

---

## 5. Liskov Substitution

Little to report, largely because there is little polymorphism — which is itself the
observation. The abstractions that exist (`IWeightLayout`, `IWeightRepository`,
`IPagedKvCache`, `IExpertSource`, `serve::IRequestService`) are clean, narrow and
substitutable. `PhysicalPagedKvCache` overrides its base faithfully.

The one substitutability defect is at §4.1: `CpuConcurrentEngine` and
`ConcurrentEngine` are behaviourally substitutable and structurally unrelated — the
inverse of an LSP violation, and equally costly.

Minor: `CudaCompiledModel::forward_token_paged_host` throws on MTP
([decode.cpp:297-299](../src/backend/cuda/model/decode.cpp#L297-L299)) while
`forward_token_host` supports it. Two methods with the same conceptual contract and
divergent capability sets is a partial-implementation smell; the constraint is real
(MTP + paged KV genuinely conflict), so the fix is to surface it as a capability
query rather than a runtime throw deep in the forward pass.

---

## 6. Cross-cutting anti-patterns

### 6.1 🟠 Hidden global configuration via `getenv`

| Variable | Site |
|---|---|
| `CELEG_FLASH_ATTN` | [`prefill_batched.cpp:431`](../src/backend/cuda/model/prefill_batched.cpp#L431) — inside a `static const bool` lambda, at nesting depth 6, in a 549-line function |
| `CELEG_MMQ_TENSOR_CORES` | [`mmq.cu:78`](../src/backend/cuda/kernels/mmq.cu#L78) |
| `CELEG_CUDA_MANAGED_WEIGHTS` | [`global_weight_setup.cpp:31`](../src/backend/cuda/model/global_weight_setup.cpp#L31) |

`CudaModelOptions` exists and is threaded everywhere. These three settings bypass
it, so two `CudaModel`s in one process cannot disagree, the settings do not appear
in `execution_plan_description()`, and they are invisible to
`PackedCompatibilityKey` — meaning **two sessions with different effective attention
kernels can be judged packed-compatible**. That is a correctness consequence, not a
style one.

`CELEG_FLASH_ATTN` is additionally cached in a function-local `static const bool`,
so it is read once per process and cannot be changed between models even
programmatically. Resolve all three into `CudaModelOptions` at construction (env var
as the *default*, if you want the override), and fold them into the compatibility
key. `mmq.cu`'s `thread_local` device cache is well-reasoned and can stay — it just
needs its input to come from options rather than the environment.

### 6.2 🟠 Silent-swallow error handling

Six sites catch and continue with a `std::clog` message:

- [`cpu/weights_loader.cpp:114,118`](../src/backend/cpu/weights_loader.cpp#L114) — pack cache rejected
- [`cpu/model_state.cpp:167,169`](../src/backend/cpu/model_state.cpp#L167) — **KV page cleanup failed**
- [`cpu/model_state.cpp:403,406`](../src/backend/cpu/model_state.cpp#L403) — **prefix snapshot rollback failed**

The pack-cache case is a legitimate degrade-and-rebuild. The other two are not: a
failed KV page release leaks pool pages and a failed snapshot rollback leaves
session state torn — both silently, in a library, to a stream the embedding
application does not control. There is no logging abstraction in the backend, so
these cannot be routed or suppressed. Route them through `RuntimeContext` (already
threaded through every backend constructor) and, for the two state-corruption
cases, mark the session unusable rather than continuing.

### 6.3 🟡 Exception-type discipline

635 throw sites across 6 standard exception types: `invalid_argument` (309),
`runtime_error` (242), `logic_error` (55), `out_of_range` (17), `overflow_error`
(11), `bad_alloc` (1). The split is used consistently and sensibly — `logic_error`
for internal invariants, `invalid_argument` for caller errors. The gap is that a
caller cannot distinguish *recoverable* failures (context limit reached, OOM on a
lane) from *fatal* ones without matching on message strings. `"context limit
reached"` ([decode.cpp:19](../src/backend/cuda/model/decode.cpp#L19)) is exactly the
condition a scheduler wants to catch and handle by evicting. A small
`celeg::ContextLimitError : std::runtime_error` for the handful of
scheduler-actionable conditions would pay for itself.

### 6.4 🟡 Duplicated post-FFN residual block

[`cpu/model_forward_token.cpp:113-134`](../src/backend/cpu/model_forward_token.cpp#L113-L134):
the MoE and dense branches differ in one call (`execute_cpu_moe_token` vs
`execute_cpu_dense_feed_forward_token`) and then repeat the *identical* seven lines
of residual-multiplier / post-FFN-norm / residual-add. Hoist the tail out of the
branch.

Same file, lines 83-88: a subtle asymmetry — `gated_delta` falls through to the
shared tail while `mamba2` `continue`s past it, and `mlp_only` `continue`s from a
different guard at line 77. Three different control-flow exits from one ladder is
how post-attention-norm bugs get introduced on the next mixer.

### 6.5 🟡 Umbrella header defeats its own stated purpose

[`kernels/kernels.cuh`](../include/celeg/backend/cuda/kernels/kernels.cuh) opens with:

> *"Umbrella header... New callers should depend on the narrow per-concern header
> they need rather than this aggregate... (Interface Segregation Principle)."*

**23 of 23** files that need kernel declarations include the umbrella. Zero include a
narrow header. The advice is correct and universally ignored, which means it is not
advice — it is an unenforced convention. Either delete the umbrella (forcing the
narrow includes, a mechanical one-time change) or delete the comment. Keeping both
trains readers to skip design comments in this codebase, which devalues the many
good ones (`GemmDispatcher`, `PackedCompatibilityKey`, `CpuExecutionContext`).

### 6.6 🟠 Nesting depth: concentrated, not endemic — and it hides drift

I measured this rather than eyeballing it. Control-flow nesting counts only braces
opened by `if`/`else`/`for`/`while`/`switch` — namespace, class, function and lambda
braces are excluded, so this is the depth a reader actually has to hold in their
head.

**Distribution across all 3,640 control statements in `src/backend/`:**

| Nesting depth | Sites | Share |
|---|---|---|
| 1 | 1,976 | 54.3% |
| 2 | 1,047 | 28.8% |
| 3 | 409 | 11.2% |
| 4 | 157 | 4.3% |
| 5 | 45 | 1.2% |
| 6 | 6 | 0.16% |
| 7+ | **0** | — |

**The honest verdict: this is not an arrow-anti-pattern codebase.** 83% of control
flow sits at depth ≤ 2, nothing anywhere exceeds depth 6, and the median function
is flat. If you came here expecting the review to confirm a general nesting problem,
the data does not support it, and I would rather tell you that than pad the finding
list.

What the data *does* show is **concentration**. The 51 sites at depth ≥ 5:

| File | Sites ≥5 | Character |
|---|---|---|
| [`cuda/model/prefill_batched.cpp`](../src/backend/cuda/model/prefill_batched.cpp) | **17** | ❌ dispatch ladder |
| [`cuda/model/decode.cpp`](../src/backend/cuda/model/decode.cpp) | **8** | ❌ dispatch ladder |
| [`cuda/moe/route.cu`](../src/backend/cuda/moe/route.cu) | 6 | ✅ in-kernel top-k insertion sort |
| [`cpu/kernels/linear.cpp`](../src/backend/cpu/kernels/linear.cpp) | 5 | ✅ tiled GEMM loop nest |
| [`cpu/model_forward_chunk.cpp`](../src/backend/cpu/model_forward_chunk.cpp) | 4 | ⚠️ mixed |
| [`cpu/packed/execution.cpp`](../src/backend/cpu/packed/execution.cpp) | 3 | ⚠️ mixed |
| [`cuda/kernels/sampling.cu`](../src/backend/cuda/kernels/sampling.cu), [`attention_gemm.cuh`](../src/backend/cuda/kernels/attention_gemm.cuh), [`linear.cuh`](../src/backend/cuda/kernels/linear.cuh), [`cpu/attention/paged_attention.cpp`](../src/backend/cpu/attention/paged_attention.cpp) | 7 | ✅ kernel loop nests |
| [`cuda/model/weight_setup.cpp`](../src/backend/cuda/model/weight_setup.cpp) | 1 | ⚠️ |

**Half of all deep nesting lives in two files** — and they are precisely the two
files §2.1 and §2.2 already implicate. Deep loop nests in `linear.cpp` (tiling:
tile → row → block → lane) and `route.cu` (top-k insertion sort) are what
performance kernels correctly look like; flattening them would be a real
pessimization. Leave them alone.

#### The actual shape of the problem

The deep sites are not complex *logic*. They are **pure dispatch, five levels of
it**, with no computation at any level:

```
for (Layer& layer : layers)                    ← 1  layer loop
  if (AttentionLayer* attention = ...)         ← 2  §2.1 mixer ladder
    if (kv_cache_mode == KvCacheMode::Int8)    ← 3  KV format
      if (attention->alibi_slopes.data())      ← 4  position bias
        else if (options.fast_attention)       ← 5  kernel mode
          if (use_flash && flash_supported)    ← 6  kernel capability
```

Six levels of nesting to select one of 25 `launch_gqa_*` symbols — this *is* the
§2.2 cross-product, written out as an indentation staircase. It is why fixing §2.2
(template axes) and §2.1 (mixer policy) collapses these 25 sites automatically. There
is no separate "reduce nesting" work item; it falls out of the other two.

#### Consequence: the Int8 and BF16 paths have silently drifted

This is what deep dispatch nesting actually costs you, and it is concrete.
[`prefill_batched.cpp:386-478`](../src/backend/cuda/model/prefill_batched.cpp#L386-L478)
has two sibling branches at depth 3 that are supposed to be format variants of one
algorithm:

| | Int8 branch (L387-416) | BF16 branch (L417-478) |
|---|---|---|
| alibi | ✅ `launch_gqa_prefill_alibi_int8` | ✅ `launch_gqa_prefill_alibi` |
| fast → flash | ❌ **absent** | ✅ `launch_gqa_prefill_flash` |
| fast → gemm | ❌ **absent** | ✅ `launch_gqa_prefill_gemm` |
| fast → segmented | ❌ **absent** | ✅ `launch_gqa_prefill_segmented` |
| fast → online | ✅ `launch_gqa_prefill_online_int8` | ❌ **absent** |
| strict | ✅ | ✅ |

The BF16 path gained three optimized kernels and a row-count fallback; the Int8 path
never did, and instead kept an `online` variant BF16 no longer uses. Running
`kv_cache_mode=Int8` silently forfeits flash attention and the segmented long-context
path. That may well be intentional — but nothing in the code says so, and the two
branches are 40 lines apart at depth 4, which is exactly far enough that nobody
diffs them. A dispatch **table** keyed on `(format, bias, mode)` makes a missing cell
visible; a 90-line staircase does not.

#### The `CELEG_FLASH_ATTN` override is very nearly a no-op

[`prefill_batched.cpp:441-442`](../src/backend/cuda/model/prefill_batched.cpp#L441-L442):

```cpp
if ((use_flash && flash_supported) ||
    (owner_layout.head_dim > 64 && flash_supported)) {
```

Factor it: `flash_supported && (use_flash || head_dim > 64)`, where
`flash_supported == (head_dim <= 128)`. So the env var `use_flash` changes the
outcome **only when `head_dim` is in [1, 64]**. For `head_dim` 65-128 flash is
already unconditional, and above 128 the flag cannot enable it. The comment at
[mmq.cu:70](../src/backend/cuda/kernels/mmq.cu#L70) explains these overrides exist
"to bisect a regression between kernels" — this one cannot bisect the range where
regressions are most likely. Writing the condition in factored form would have made
that obvious at a glance; the duplicated-`flash_supported` form hides it. (See also
§6.1 — this is the same `getenv` that escapes `PackedCompatibilityKey`.)

#### One loop-invariant hoisted *into* the loop

[`cpu/model_forward_chunk.cpp:221-238`](../src/backend/cpu/model_forward_chunk.cpp#L221-L238) —
the depth-6 site:

```cpp
for (size_t row = 0; row < rows; ++row) {
    if (layout.output_gate.packed_with_query) { ... }
    else {
        if (row == 0) {                    // ← whole-matrix GEMM, once
            layer_gemm(attention->gate, ...);
        }
        gate = workspace_.chunk_attention_gate.data() + row * q_width;
    }
```

The gate GEMM is loop-invariant and computed once, but it is *placed* inside the row
loop behind a `row == 0` guard. That is the sixth level of nesting, and it costs a
predictable branch per row plus a reader's double-take. Hoist the GEMM above the
loop and the depth-6 site disappears.

#### Boolean-condition complexity: mostly fine

47 conditions have ≥4 boolean terms. I checked them: **roughly 40 are kernel
precondition guards** of the form
`if (!q || !key_cache || !value_cache || !output || sequence_length <= 0 || ...) return;`
([`cpu/kernels/attention.cpp:16`](../src/backend/cpu/kernels/attention.cpp#L16),
[`convolution.cpp:10`](../src/backend/cpu/kernels/convolution.cpp#L10),
[`quantization.cpp:221`](../src/backend/cpu/kernels/quantization.cpp#L221), …). These
are good — a flat null-and-range guard at the top of a kernel is exactly right, and
7 terms in one is not a complexity problem. **Do not "simplify" these.**

Two genuine exceptions:

- [`cuda/model/experts.cu:111`](../src/backend/cuda/model/experts.cu#L111) —
  `if (!gu_shape && !gu_flat_shape || !down_shape && !down_flat_shape || ...)`.
  Unparenthesized mixed `&&`/`||`. The behaviour is *correct* (`&&` binds tighter,
  which is what was meant), but this is the exact pattern GCC and Clang emit
  `-Wparentheses` for, and a reader must recall the precedence table to verify a
  validation predicate. Add the parentheses.
- [`cpu/kernels/quantized_dot.cpp:252`](../src/backend/cpu/kernels/quantized_dot.cpp#L252) —
  `if (isa == CpuIsa::Neon || isa == CpuIsa::DotProd || isa == CpuIsa::I8mm || isa == CpuIsa::Sve2 || ...)`.
  An enum-membership test spelled as an equality chain; a `switch` with explicit
  cases gets `-Wswitch` coverage when a new ISA is added, which this does not.

#### Out of scope but surfaced by the scan

[`cuda/moe/route.cu:75`](../src/backend/cuda/moe/route.cu#L75) declares a per-thread
`float best[128]` and then fills it with `for (int i = 0; i < group_score_top_k; ++i)`
where `group_score_top_k` is a runtime kernel argument. If it ever exceeds 128 this
writes past a thread-local stack array inside a kernel. Not a SOLID finding — flagging
it because the nesting scan put me in the function and it would be irresponsible to
walk past it. Worth an `assert` or a compile-time bound.

**Bottom line on nesting:** no standalone remediation item. Depth ≥ 5 is 1.4% of
control flow, the deepest sites in kernels are correct, and the ~25 that are not
correct are the §2.1/§2.2 cross-product wearing a different hat. Two things *are*
independently worth doing now: the Int8/BF16 drift table (which is a capability gap,
possibly a bug) and the factored `use_flash` condition.

---

## 7. DRY / duplication

Measured with a token-normalized clone detector over all 159 backend files (strings
and char literals collapsed, comments and `#include` lines dropped, brace-only lines
dropped, minimum block 6-8 normalized lines, blocks greedily extended to maximal
length).

**Headline: 2,166 redundant lines out of 26,520 — 8.2%, across 132 distinct
duplicated blocks.**

Important caveat, stated up front because it changes how you should read that
number: this detector finds **exact clones only**. Copy-paste that was subsequently
renamed (`rows`/`cols` → `total_rows`/`common_width`) is invisible to it. I
confirmed at least one large instance by hand (§7.4), so **8.2% is a floor, not an
estimate.** Real Type-2 duplication is meaningfully higher.

### 7.0 Where the duplication is — and is not

| Category | Blocks | Redundant lines |
|---|---|---|
| Self-duplication (same file) | 160 | **1,672** |
| Across files, same backend | 58 | 619 |
| **Cross-backend (cpu ↔ cuda)** | **3** | **28** |

The distribution is the opposite of what I expected, and it corrects an impression
§4.1 could leave you with.

**There is essentially no copy-paste between the CPU and CUDA backends** — 28 lines
total, and the largest is a legitimate shared concern (§7.5). The CPU/CUDA
duplication in §4.1 is **structural, not textual**: two parallel type hierarchies
with the same *shape*, the same method names and the same semantics, written in
entirely different code. No clone detector will ever flag it, which is precisely
why it survived. Keep §4.1 and §7 separate in your head — they are different
problems needing different fixes, and fixing one does nothing for the other.

The real duplication is **within single files**, and it is concentrated in exactly
the places §2.1, §2.2 and §4.2 already point at.

### 7.1 🔴 `forward_token_paged_host` is 58% a copy of `forward_token_host`

[`cuda/model/decode.cpp`](../src/backend/cuda/model/decode.cpp) — the single worst
instance in the backend, and the top hit at every window size I tried.

```
forward_token_host        (L15-286):  271 non-comment lines
forward_token_paged_host  (L288-682): 392 non-comment lines
identical lines shared:               156  → 58% of the non-paged variant
longest verbatim run:                  47 lines (L160-206 ≡ L562-608)
runs of >= 8 identical lines:          5, covering 89 lines
```

The 47-line verbatim block is the **entire GatedDeltaNet arm**, byte-for-byte
identical in both functions. The Mamba2, MlpOnly and Convolution arms are near-copies
too. This is structurally inevitable: the two functions differ *only* in how attention
reads and writes KV (contiguous cache vs paged), and every non-attention mixer is by
definition unaffected by that choice — yet the code duplicates all of them anyway.

**The maintenance hazard is immediate and asymmetric.** A GatedDeltaNet fix applied
at L160 and not at L562 produces a model that is correct in normal decode and wrong
under paged KV — i.e. correct in single-session tests and wrong under concurrent
serving, which is the hardest failure mode to catch. §6.6 documents an instance where
exactly this already happened between the Int8 and BF16 prefill branches.

**Fix:** hoist the shared layer loop and parameterize the KV access. A small
`KvAccessor` (contiguous vs paged) passed to the attention arm, with every other
arm written once, removes ~150 lines and the entire class of divergence. This is the
same seam as §2.1's mixer policy — do them together.

### 7.2 🟠 Kernel-family copy-paste — the DRY face of §2.2

The largest cluster of self-duplication is the CUDA attention kernels, and it is the
combinatorial explosion of §2.2 seen from the other side:

| File | Duplicated block | Lines |
|---|---|---|
| [`attention_paged.cuh`](../src/backend/cuda/kernels/attention_paged.cuh) | L305 ≡ L390 | 41 |
| [`attention_segmented.cuh`](../src/backend/cuda/kernels/attention_segmented.cuh) | L19 ≡ L88 | 29 |
| [`attention_paged.cuh`](../src/backend/cuda/kernels/attention_paged.cuh) | L22 ≡ L189 | 21 |
| [`attention_dense.cuh`](../src/backend/cuda/kernels/attention_dense.cuh) | L14 ≡ L97 | 21 |
| [`attention_alibi.cuh`](../src/backend/cuda/kernels/attention_alibi.cuh) | L17 ≡ L67 | 20 |
| [`attention_batch_ptrs.cuh`](../src/backend/cuda/kernels/attention_batch_ptrs.cuh) | L14 ≡ L73 ≡ L150 ≡ L214 | 16 (×4) |
| [`mmq.cu`](../src/backend/cuda/kernels/mmq.cu) | L444 ≡ L558, L407 ≡ L517, L120 ≡ L215 | 20/19/18 |
| [`sampling.cu`](../src/backend/cuda/kernels/sampling.cu) | L147 ≡ L472, L20 ≡ L101 | 25/16 |
| [`rope.cuh`](../src/backend/cuda/kernels/rope.cuh) | L317 ≡ L439 | 19 |

`gqa_decode_strict_kernel` and `gqa_decode_online_kernel`
([attention_dense.cuh:14, :97](../src/backend/cuda/kernels/attention_dense.cuh#L14))
share an identical 21-line signature and preamble and differ only in the softmax
strategy. The int8 variants at L158/L233 repeat the same pair again with `int8_t`
substituted. That is four kernels where two templates on
`<SoftmaxMode, KvFormat>` would do.

**This one deserves nuance.** Some kernel duplication buys real performance —
divergent code paths in one kernel cost occupancy, and a `__device__` helper that
does not inline costs more than the duplication saves. But a **21-line parameter
list and preamble** buys nothing: it is pure declaration, identical in both, and
templating it changes zero generated instructions. Split the finding:

- Duplicated *signatures and preambles* (the majority of the lines above): extract
  to a params struct + template. Zero codegen risk.
- Duplicated *inner loops*: leave alone unless you have a benchmark showing the
  templated version is equal. `attention_batch_ptrs.cuh`'s 16-line block appearing
  **4×** is worth one careful attempt.

### 7.3 🟠 `sampling.cu` duplicates its own top-k/top-p filter

[`sampling.cu:147`](../src/backend/cuda/kernels/sampling.cu#L147) ≡
[`sampling.cu:472`](../src/backend/cuda/kernels/sampling.cu#L472) — 25 identical
lines implementing the top-p cutoff scan over an already-sorted top-k array. A second
16-line clone at [L20](../src/backend/cuda/kernels/sampling.cu#L20) ≡
[L101](../src/backend/cuda/kernels/sampling.cu#L101) is the argmax reduction, including
its tie-break rule:

```cpp
if (value > best || (value == best && (index < 0 || i < index))) {
```

That tie-break is a **determinism guarantee** — lowest index wins on ties. Having it
written twice means a future change to sampling determinism has two places to land
and no test that will notice if only one is updated. For a serving engine where
reproducible sampling is a user-visible contract, this is the duplicate I would fix
first in this file.

### 7.4 🟠 `linear_loader.cpp` — the clone the detector *undercounts*

[`load_linear_weight`](../src/backend/cuda/model/linear_loader.cpp#L25) (342 lines)
and [`load_concat_linear_weight`](../src/backend/cuda/model/linear_loader.cpp#L388)
(352 lines) share 91 exactly-identical lines (27%). Hand inspection of L250-300 vs
L617-667 shows the **real** overlap is far higher: the Int8 and Int4 quantization
tails are the same algorithm with variables renamed —

```cpp
// L259                                  // L626
weight.int8_storage.reset(count);        weight.int8_storage.reset(count);
    dense_data, (size_t)rows,                dense_data, (size_t)total_rows,
    (size_t)cols);                           (size_t)common_width);
```

— which is exactly the Type-2 clone my detector cannot see. The concat variant even
carries the comment *"Keep BF16 device buffer as prefill fallback (see
load_linear_weight)"*, i.e. the author knew it was a copy and left a pointer instead
of a shared function.

**Fix:** extract `quantize_and_store(LinearWeight&, const __nv_bfloat16* dense,
size_t rows, size_t cols, WeightMode)`. Both callers already have the dense buffer
and the dimensions; the only reason they differ is the variable names.

### 7.5 🟡 Small utilities copy-pasted rather than shared

Genuine, cheap, unambiguous fixes:

| Utility | Copies | Locations |
|---|---|---|
| `checked_multiply` + `round_up` | **2** (22 lines, verbatim) | [`cpu/attention/paged_attention.cpp:48`](../src/backend/cpu/attention/paged_attention.cpp#L48), [`cpu/memory/paged_kv.cpp:16`](../src/backend/cpu/memory/paged_kv.cpp#L16) |
| `fp16_to_float` | **3** (13 lines each) | [`cpu/weight_codec.cpp:42`](../src/backend/cpu/weight_codec.cpp#L42), [`cpu/kernels/gguf.cpp:83`](../src/backend/cpu/kernels/gguf.cpp#L83), [`cpu/kernels/gguf_avx2.cpp:35`](../src/backend/cpu/kernels/gguf_avx2.cpp#L35) |
| GGUF nibble unpack | **2**, *cross-backend* | [`cpu/kernels/gguf.cpp:156`](../src/backend/cpu/kernels/gguf.cpp#L156), [`cuda/model/gguf_dequant.cpp:88`](../src/backend/cuda/model/gguf_dequant.cpp#L88) |
| RoPE frequency scaling | **2** | [`cuda/kernels/rope_pairing.cu:10`](../src/backend/cuda/kernels/rope_pairing.cu#L10), [`cuda/kernels/rope.cuh:260`](../src/backend/cuda/kernels/rope.cuh#L260) (31 lines) |
| RMSNorm preamble | **3** | [`kernel_common.cuh:110`](../src/backend/cuda/kernels/kernel_common.cuh#L110), [`qkv_rope.cuh:81`](../src/backend/cuda/kernels/qkv_rope.cuh#L81), [`rope.cuh:46`](../src/backend/cuda/kernels/rope.cuh#L46) |

`checked_multiply` is the sharpest: two identical overflow-guard helpers in two
anonymous namespaces, both throwing `std::overflow_error("CPU KV aligned allocation
overflow")` — the *same message string*. An overflow-check policy that exists in two
copies is one edit away from disagreeing about what overflows.

The GGUF nibble-unpack clone is the one genuinely interesting cross-backend
duplicate: `constexpr uint32_t low_nibble = 0x0f0f0f0f;` plus 13 lines of identical
bit manipulation. Quantization *format* decoding is backend-neutral by definition —
Q4_K's layout does not depend on where it is unpacked. That belongs in a shared
`celeg/model/weights/` header, not once per backend.

`fp16_to_float` × 3 is the same story in miniature: an IEEE-754 half-float decoder,
which is about as backend-neutral as code gets, written three times in one backend.

### 7.6 🟡 Duplicated post-FFN residual tail

Already covered in §6.4 — [`cpu/model_forward_token.cpp:113-134`](../src/backend/cpu/model_forward_token.cpp#L113-L134),
seven identical lines in both arms of the MoE/dense branch. Listed here for
completeness because it is the smallest member of the same family as §7.1: a branch
that duplicates its own tail rather than joining it.

### 7.7 What the duplication is *not*

To keep the finding honest:

- **The `_avx2` / `_avx2_msvc` files are not DRY violations.** The detector flags
  ~12 lines shared between [`quantized_dot.cpp:48`](../src/backend/cpu/kernels/quantized_dot.cpp#L48)
  and [`quantized_dot_avx2_msvc.cpp:78`](../src/backend/cpu/kernels/quantized_dot_avx2_msvc.cpp#L78).
  That is a deliberate ISA-specialized copy, split at the file level so MSVC compiles
  it with different flags. Correct as-is; do not merge.
- **Repeated `#include` blocks** (I excluded them from the numbers above) are not
  duplication in any meaningful sense.
- **Repeated null-guard preambles** in kernels (§6.6) are duplication a detector sees
  but that reads better inline than behind a macro.

### 7.8 Priority within DRY

1. **§7.1** — 156 lines, and the divergence risk is a correctness risk in the
   concurrent-serving path specifically. Fix with §2.1.
2. **§7.3 argmax tie-break** — small, but it duplicates a user-visible determinism
   contract.
3. **§7.4** — extract `quantize_and_store`; the author already documented the copy.
4. **§7.5** — an afternoon, no risk, removes 5 multi-copy utilities.
5. **§7.2 signatures/preambles** — mechanical, zero codegen risk. Leave the inner
   loops alone.

---

## 8. What is done well

Worth stating explicitly, both to avoid regression and because several of these are
harder than the problems above.

- **Architecture neutrality is complete.** `grep -rn "ModelArchitecture::" src/backend`
  returns **nothing**. All architecture knowledge is compiled into
  `CompiledModelProgram` upstream and the backends consume a neutral IR. This is
  the single best structural property in the codebase and is what makes §2.1
  tractable — the taxonomy is already closed and named, it just isn't dispatched on
  exhaustively.
- **`PackedCompatibilityKey`** ([session.hpp:22-45](../include/celeg/backend/cuda/packed/session.hpp#L22-L45))
  is exemplary: an immutable value capturing every execution-relevant decision, with
  `operator==` defaulted, so compatibility is compared rather than re-derived from
  mutable options. (Its one gap is §6.1 — the env-var settings escape it.)
- **`CpuExecutionContext`** is the right abstraction, correctly motivated. It just
  needs to be applied to the remaining operators (§3.2).
- **The kernel layer's free-function design** is appropriate. Kernels are pure
  transformations over device pointers; wrapping them in classes would add
  indirection for nothing. `elementwise_avx2_msvc` / `quantized_dot_avx2_msvc` /
  `gguf_avx2` split by ISA at the file level, which is the correct granularity.
- **Deliberate closures are documented as such.** `GemmDispatcher` ("intentionally
  closed performance dispatch domain") and `WeightLoader` ("closed domain requiring
  coordinated loader, plan, and test changes") both state the trade-off. That is
  the right way to close a hierarchy.
- **Test coverage is real** — 86 test files, with the backend's testable seams
  (`cpu_batch_scheduler`, `cpu_moe_route`, `cpu_kv_topology`, `batch_planner`,
  `concurrent_policy`, `cuda_gguf_kernels`) actually covered. Note that the
  *untested* seams correlate almost exactly with the god objects: there is no
  `cuda_compiled_model_test` because there cannot be one.

---

## 9. Prioritized remediation

| # | Finding | § | Sev | Effort | Payoff |
|---|---|---|---|---|---|
| 1 | Replace `PackedSessionContext` pointer-bag with `IPackedLane` + per-step view | 3.1 | 🔴 | L | Removes a silent-corruption class |
| 2 | Exhaustive `std::visit` at all 20 mixer-dispatch sites | 2.1 | 🔴 | M | 20 silent bugs → 20 compile errors |
| 3 | Delete `CpuConcurrentMetrics`; populate neutral `ConcurrentMetrics` | 4.1 | 🔴 | S | One metrics path; unblocks 4 & 5 |
| 4 | Resolve the 3 `getenv` settings into `CudaModelOptions` + compatibility key | 6.1 | 🟠 | S | Fixes packed-compatibility correctness gap |
| 5 | Common `IConcurrentEngine` for both engines; unify options | 4.1 | 🔴 | M | Backend-agnostic serve layer |
| 6 | Split the 6 mega-functions along the mixer-policy seam | 4.2 | 🔴 | L | Do with #2 — same seam |
| 7 | `CpuExecutionContext` for all CPU operators; add `CpuAttentionContext` | 3.2 | 🟠 | M | Finishes work already started |
| 8 | Struct-ify the 18-26 parameter kernel signatures | 2.2 | 🟠 | M | Kills argument-transposition bugs |
| 9 | Make `CudaCompiledModel` members private; route via accessors | 4.3 | 🟠 | M | Reveals true coupling of 29 files |
| 10 | Delete `CpuModel`/`CudaModel` forwarder layers | 4.4 | 🟠 | S | −250 lines, −16 `friend`s |
| 11 | Route the 6 `std::clog` sites through `RuntimeContext`; fail hard on the 2 state-corruption cases | 6.2 | 🟠 | S | Library stops writing to stderr |
| 12 | `TensorRole` for short-conv weights | 2.3 | 🟠 | S | Consistency |
| 13 | Delete `kernels.cuh` umbrella or its comment | 6.5 | 🟡 | S | Restores trust in design comments |
| 14 | Audit Int8-vs-BF16 prefill kernel drift; document or close the gap | 6.6 | 🟠 | S | Int8 silently loses flash + segmented |
| 15 | Factor the `use_flash` condition; parenthesize `experts.cu:111` | 6.6 | 🟡 | S | Override currently near-inert |
| 16 | Hoist the loop-invariant gate GEMM out of the row loop | 6.6 | 🟡 | S | Removes the deepest CPU nest |
| 17 | Merge `forward_token_paged_host` into `forward_token_host` via a `KvAccessor` | 7.1 | 🔴 | M | Removes 156 duplicated lines; same seam as #2/#6 |
| 18 | Dedupe `sampling.cu`'s argmax tie-break and top-p scan | 7.3 | 🟠 | S | One copy of a determinism contract, not two |
| 19 | Extract `quantize_and_store` from `linear_loader.cpp`'s two loaders | 7.4 | 🟠 | S | Author already flagged the copy in a comment |
| 20 | Share `checked_multiply`/`round_up`, `fp16_to_float`, GGUF nibble unpack | 7.5 | 🟡 | S | 5 multi-copy utilities, one afternoon |

**Suggested first sprint:** #3, #4, #11, #12, #13, #14, #15, #16, #18, #19, #20 —
all small, independent, no architectural commitment. #4 and #14 each close a real
correctness gap; #18-20 are pure DRY cleanup with no design risk.

Note that there is **no "reduce nesting" item**: §6.6 measured it, and the deep sites
are either legitimate kernel loop nests or the §2.1/§2.2 dispatch cross-product. Doing
#2 and #6 removes them as a side effect.

**Suggested second sprint:** #2 + #6 + #17 together. All three land on the same
mixer-policy seam — the CUDA decode/paged-decode split *is* the mixer ladder
duplicated wholesale (§7.1), so unifying the ladder and unifying the two functions
are one change, not two.

**#1 is the one to schedule deliberately.** It is the largest change here and the
only one where the current design can corrupt memory rather than merely slow you
down.

---

## 10. A note on the comments

Several headers in this backend describe SOLID benefits the structure does not
deliver: `PackedSessionContext`'s "deliberately absent" (§3.1), `kernels.cuh`'s ISP
note (§6.5), `load_checkpoint_weights`' injected-strategy shape (§4.2), and
`CudaCompiledModel`'s "New GEMM backends are added by extending GemmDispatcher
(OCP)" — true for GEMM backends, but the surrounding class is the god object that
comment sits inside.

This is worth naming as its own finding. Aspirational comments are more expensive
than no comments: a reader who trusts `PackedSessionContext`'s header will pass it
around believing it is a safe narrow view. The fix is not to soften the comments
but to make them true — and in the interim, to mark the gap explicitly
(`// TODO(solid): this is currently a raw-pointer view, not an interface`) so the
documentation and the code disagree *visibly* rather than silently.
