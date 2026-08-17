# Extensibility Refactoring Plan

## Status

Active implementation roadmap.

Audit baseline: `master` at `32872bc19d992cf0c9bfb8ac9fdd5fece4931f18`.

### Progress

- **Sprint A — impossible-state cleanup: done.**
  - 1.1 CUDA `LinearWeight` storage variant — `bb7ae24`.
  - 1.2 CUDA `FeedForwardWeights` gains `std::monostate` — `3215871`.
  - 1.3 CUDA MoE resident/offloaded expert storage variant — `578dd34`.
  - 1.4 CUDA `AttentionLayer` runtime state split by semantic family — `ea6d143`.
  - Verified against a real model (LiquidAI/LFM2.5-350M) on CUDA: logit-level parity across attention/KV-cache mode configs, plus a real session-persistence bug found and fixed along the way (`85579be`, hash-based model-identity check replacing a truncating fixed-size buffer; session format bumped v4→v5) and positional `--model`/`--repo` CLI auto-resolution added to `celeg-run`/`celeg-cpu-run`/`celeg-serve`.
- **Sprint B — CPU composition: done.**
  - 5–7 CPU `WeightLayer` mixer×MoE Cartesian variant replaced with `CpuLayerWeights{common, CpuMixerWeights mixer, CpuFeedForwardWeights feed_forward}` composition; nested duplicated memory-accounting visitor simplified to two flat visitors — `c3ded51`.
  - Verified: CPU ctest 77/77 passing.
- **Sprint C — checkpoint boundary: done except the MTP sub-gap.**
  - 8/9 `resolve_weight_plan`/`build_weight_plan_from_graph` made existence-aware (checks `IWeightRepository::contains()` per naming candidate instead of trusting the first template); `append_moe()` and the automatic-inference `bind_moe()` now decide packed-vs-individual expert layout once from real checkpoint evidence and record it in the resolved plan (this also fixed a latent bug where `individual_expert_model` was unconditionally true and CUDA's packed-expert loading path was dead code). CUDA `weight_setup.cpp`'s duplicated `repo.contains(literal)` boolean blocks deleted in favor of reading role presence from the resolved plan plus the existing `INativeBlockStorageRepository` capability. Dead narrower `celeg::weights::TensorRole` enum in `weights_topology.hpp` removed — `facb607`.
  - 9/9 inner expert loaders rewritten onto resolved names: `WeightLoader::load_moe_gate_up/load_moe_down/load_moe_experts_host/build_expert_catalog` now take a `MoeExpertTensorNames` bundle extracted from the resolved plan (`moe_expert_tensor_names()` in `roles.hpp`); the `_named` variants, `expert_name()`/`named_expert_name()` builders, the inner packed `repo.contains()` probe, and the three-spelling `load_router_weight()` prober are deleted. Storage family (GGUF block-quantized vs checkpoint-packed int4 vs BF16) is now decided by tensor dtype/capability inside the loaders, so `weight_setup.cpp` no longer needs the `named_expert_model` distinction or the `INativeBlockStorageRepository` dynamic_cast for it. `append_moe()` plans `MoeRouterBias` whenever the MoE semantics say a bias exists, and `bind_moe()` resolves its spelling once (mlp-gate and feed_forward conventions), so backends read the bias name from the plan instead of their own literal fallback chains.
  - Plan resolution learned the checkpoint-packed int4 convention (`name_packed`/`_scale`/`_shape` virtual base names): `repository_has_tensor()` uses the same repository-level predicate the codecs use. This fixed real-model breakage — Ling-3.0-tiny-int4 (128 experts, int4-packed) failed on master with "tensor request has no resolved source name: moe_expert_gate" on CPU; it now runs end-to-end on CPU.
  - CPU packed-MoE branch no longer unconditionally dereferences `moe_semantics.shared` or demands an (never-planned) `MoeSharedGateWeight` request — both latent crashes for packed models made reachable by `facb607`'s layout fix.
  - 11 poisoned-tensor-name regression test added (`tests/fake_repository_backend_boundary_test.cpp`) proving MoE layout/role resolution is plan-driven, not name-matching; extended to cover `moe_expert_tensor_names()` and `MoeRouterBias` planning.
  - Verified: CPU ctest 77/77, CUDA ctest 87/87; real-model checks — CPU Ling-3.0-tiny-int4 decode (fixed by this sprint), CUDA LFM2.5-350M deterministic with cross-KV-mode parity, CUDA Nemotron3-Nano-4B-GGUF identical to baseline. Known pre-existing issue (not caused by this sprint, unchanged): CUDA decode of Ling-3.0-tiny-int4 segfaults on the 12GB reference GPU (master behaves identically; the int4 experts dequantize to ~22GB of BF16 device storage, which exceeds VRAM).
- **Sprint D — inference extensibility: done.**
  - 12–15 Layer inference-rule catalog `ILayerInferenceRule` implemented in `src/model/inference/rules.hpp`, `rules.cpp`, `rules_attention.cpp`, and `rules_recurrent.cpp`. All 7 mixer grammar rules (`StandardAttentionRule`, `LatentAttentionRule`, `FusedGatedDeltaRule`, `FactorizedGatedDeltaRule`, `Mamba2Rule`, `ShortConvolutionRule`, `MlpOnlyRule`) emit both semantic mixer specs and physical tensor-role bindings directly. The central `has_mamba/has_mla/has_kda/...` ordered cascade and duplicate binding dispatch deleted — `afd1c27`.
  - 16 `NormalizedModelMetadata` decomposed into typed fact groups (`CoreModelFacts`, `AttentionFacts`, `LatentAttentionFacts`, `ShortConvolutionFacts`, `GatedDeltaFacts`, `Mamba2Facts`, `MoeFacts`). Stringly-typed MoE fields replaced with `MoeRouterScoreFunction` and `MoeRouterSelectionMethod` enums.
  - 17 Generic `input.is_gguf()` semantic branches replaced with `DecayParameterEncoding` on `GatedDeltaFacts` and `Mamba2Facts`, resolved at format boundary.
  - Verified: CPU ctest 78/78, CUDA ctest 88/88; custom grammar extension test passing (`tests/layer_inference_rule_test.cpp`).
- **Sprint E (backend planning): items 18–22 done.**
- **Sprint F (extension ABI and orchestration): items 23–25 done; 26–27 pending.**
  - 18 Mixer/FFN maxima (`maximum_attention_*`, `maximum_mamba_*`, `maximum_gated_delta_*`,
    `max_feed_forward_intermediate`, `mamba2_intermediate`, `conv_cache`, `conv_dim`) removed
    from neutral `ExecutionTopology`; `ExecutionTopology::derive` no longer computes them.
  - 19 CPU workspace now derives from the compiled program via `CpuWorkspacePlan::from_program`
    (replacing `from_topology`); CUDA gained an equivalent `CudaWorkspacePlan::from_program`
    consumed by `allocate_celeg_resources`/`allocate_prefill_workspace`, and
    `PackedWorkspaceRequirements::derive` now derives all maxima from `CompiledModelProgram`
    instead of `ExecutionTopology`.
  - 20 `CpuWorkspace` is already decomposed into common/attention/recurrent/feed-forward
    sub-workspaces; `conv_cache` is now part of `CpuWorkspacePlan` and every short-convolution
    consumer sources `cache_length` from its own `ShortConvolutionSpec` (CPU `ConvolutionWeights`
    and CUDA `ConvolutionLayer` both gained a `spec` member) rather than `ExecutionTopology`.
  - Verified: CUDA build green (492 targets) and the full 88-test ctest suite passes.

This plan consolidates the extensibility findings from the backend SOLID review,
the follow-up source audit, the semantic-state cleanup work, and the discussion
around GGUF native/repacked CPU paths. It is intentionally broader than a SOLID
checklist: the target is to reduce the number of unrelated places that must be
edited when CELEG gains a new checkpoint convention, mixer, feed-forward form,
weight storage, CPU ISA, expert-residency mode, or backend.

This plan has **no backward-compatibility constraint for internal C++ APIs**.
When a representation is replaced, migrate all consumers and delete the old
path. Do not retain adapters, aliases, proxy views, deprecated accessors, or
fallback shims merely to preserve an internal interface.

The existing public boundaries and architecture guidance remain authoritative,
in particular `AGENTS.md`, `docs/EXTENDING_ARCHITECTURES.md`,
`docs/EXTENDING_MOE.md`, `docs/PRIMITIVE_CAPABILITY_MATRIX.md`, and
`docs/API.md`. Where the current implementation violates those boundaries, this
plan treats the documentation as the intended architecture and removes the
violation.

---

# 1. Goals

The primary goal is not "more abstractions". It is to make extension cost track
the semantic size of the feature being added.

A compatible checkpoint naming convention should normally require work only at
the format/inference boundary. A new backend should register through the runtime
without forcing edits to backend-neutral API enums. A new weight storage should
add one storage alternative and its kernels without requiring pointer/tag
reconciliation throughout the backend. A new mixer should add its semantic
alternative and backend implementation without multiplying through unrelated
MoE, workspace, topology, and weight-storage representations.

The governing rules are:

1. **Impossible runtime states must be unrepresentable.** If a validator mainly
   proves that a tag, nullable pointer set, or boolean agrees with another
   payload, replace the representation with a sum type or a dedicated value
   type.
2. **One semantic fact has one owner.** Do not re-infer a checkpoint layout in a
   backend after canonical resolution has already produced a `WeightPlan`.
3. **Checkpoint syntax stops at the checkpoint/inference boundary.** Tensor
   spellings, architecture names, source-format quirks, and repository naming
   conventions must not leak into backend execution/setup code.
4. **Compose independent axes instead of flattening their Cartesian product.**
   Mixer choice and feed-forward choice are independent and should be stored as
   independent values.
5. **Closed-world semantic algebras may deliberately use `std::variant`.** A
   semantic `std::variant` is not "OCP-compliant" in the classical open-world
   sense, and that is acceptable. Compile-time exhaustiveness is a feature when
   CELEG wants every semantic extension to force all required interpreters to
   acknowledge it.
6. **Open-world extension belongs at plugin/catalog boundaries.** Backends,
   checkpoint formats, tokenizer providers, vision providers, and inference
   rule registration are the places where runtime registration or factories are
   appropriate.
7. **Do not replace a switch with `std::visit` merely to claim OCP.** The
   important question is whether the set is intentionally closed and whether
   every required consumer is exhaustive.
8. **Do not add virtual dispatch to hot execution paths without a measured need.**
   Load-time/planning catalogs may be polymorphic; kernels and per-token paths
   should remain statically bound where practical.
9. **Capability differences are not automatically LSP violations.** CPU and CUDA
   sessions are separate concrete APIs today. Do not force identical async
   methods unless a real common substitutable session abstraction is required.
10. **Measure before turning cleanup into kernel work.** A structural fallback
    that exists because a kernel does not support a layout is not itself an
    architectural anti-pattern.

---

# 2. Explicit non-goals

The following are intentionally outside this refactoring plan unless a later
measurement or feature request independently justifies them.

## 2.1 Do not "open" the semantic model by replacing all variants with virtual classes

`MixerSpec`, attention-state variants, RoPE-scaling variants, MoE-selection
variants, and similar model semantics are intentionally closed-world values.
Adding a new alternative should continue to fail compilation in semantic
fingerprinting, validation, lowering, topology/planning, and backend capability
analysis until that alternative is handled deliberately.

## 2.2 Do not eliminate switches as a goal

A switch over an intentionally closed enum is not a defect by itself. Replace a
switch only when the representation or ownership is wrong, not because a switch
exists.

## 2.3 Do not create a generic `IFileSystem` solely to satisfy DIP

Direct filesystem use in cache/session persistence can reduce testability, but a
large filesystem abstraction is not justified without a concrete alternate
transport or testing need. Prefer separating serialization/codec logic from file
transport where that seam is useful.

## 2.4 Do not invent another weight-format abstraction where one already exists

CELEG already has checkpoint format catalogs, `IWeightRepository`, capability
interfaces such as `ILocatableTensorRepository` and
`IRandomAccessTensorReader`, and CPU/CUDA weight codecs/loaders. The problem is
not the absence of all weight-format abstraction; the remaining problems are
specific duplicated storage states, name/layout re-inference, and dispatch
sprawl.

## 2.5 Do not force CPU/CUDA session API parity

The serving substitution boundary is already represented by
`IRequestService`, `ISchedulerDriver`, and `IServiceDiagnostics`. The absence of
`decode_async_begin/finish` on `CpuInferenceSession` is an API/capability
difference, not an inherent LSP violation. Revisit only if a real shared session
contract is introduced.

## 2.6 Do not treat the Q4_0/Q5_0 CPU repack fallback as cleanup

The current CPU GGUF path quantizes activations into 256-element Q8_K
superblocks. Layouts that cannot use the current native dot path are repacked.
Removing that fallback is kernel/performance work and requires numerical tests
and benchmarks; it is not part of the structural refactor phases below.

See Section 12 for the precise future kernel work.

---

# 3. Cross-cutting acceptance criteria

Every phase below is complete only when the relevant criteria hold:

1. old internal representations are deleted rather than wrapped for compatibility;
2. no new architecture-name switch is introduced in backend operator code;
3. no backend setup path re-inspects checkpoint tensor spellings after canonical
   binding/weight planning has resolved them;
4. semantic and runtime variants used for lowering, validation, fingerprinting,
   allocation, or capability analysis are compile-time exhaustive;
5. no alternative is represented by "all pointers null", a magic zero, an empty
   payload, or another invalid object if absence is a real semantic state;
6. a new alternative cannot leave unrelated stale payload live beside it;
7. CPU and CUDA test suites remain green after each independently mergeable cut;
8. numerical behavior is unchanged for structural refactors unless the phase
   explicitly changes semantics;
9. a new quantized execution path includes quality and performance validation;
10. extension tests demonstrate the intended edit surface, not only local unit
    correctness.

---

# Phase 0 — Lock extension boundaries with tests

Before changing representations, add focused tests that make the desired
extension properties executable.

## 0.1 Backend extension test

Strengthen the existing runtime backend extension coverage so a synthetic
backend can be registered and selected without modifying a core CPU/CUDA enum or
switch.

Target property:

```text
third backend
  -> backend factory/module registration
  -> backend-owned options decoding
  -> common serving interfaces
  -> no edit to backend-neutral runtime selection logic
```

## 0.2 Checkpoint-format extension test

Register a synthetic `ICheckpointFormat` and verify that format selection/opening
requires no changes to the built-in format selector beyond registration.

The existing `CheckpointFormatCatalog` is already the desired shape; preserve it.

## 0.3 Semantic exhaustiveness tests

Keep compile-time guardrails around all closed semantic variants. New mixer,
feed-forward, attention-state, RoPE, MoE-selection, and storage alternatives
must fail at every interpreter that must understand them.

## 0.4 "No backend tensor-name inference" regression test

Add a test fixture whose physical tensor spellings differ while its resolved
`TensorRole`/weight plan is equivalent. Backend setup must behave identically.
This test should fail if CPU or CUDA begins selecting behavior by raw checkpoint
names.

---

# Phase 1 — Finish runtime sum types in CUDA

This phase removes the clearest remaining tag/pointer-sentinel representations.
It is the highest-value low-risk cleanup.

## 1.1 Replace CUDA `LinearWeight` tag + parallel payloads with a real storage variant

### Current problem

`include/celeg/detail/model/linear_weights.hpp` stores:

- `LinearStorageKind kind`;
- `bf16` pointer;
- `int8` pointer;
- `int4` pointer;
- `scales` pointer;
- `gguf_segments`;
- dimensions.

`validate_storage()` then proves that only the pointers appropriate to `kind`
are populated. This permits contradictory states to be constructed and makes
new storage formats multiply through conditionals and validators.

### Target

Use dedicated alternatives, for example:

```cpp
struct Bf16LinearStorage {
    const __nv_bfloat16* data = nullptr;
};

struct Int8LinearStorage {
    const int8_t* data = nullptr;
    const float* scales = nullptr;
};

struct Int4LinearStorage {
    const uint8_t* data = nullptr;
    const float* scales = nullptr;
};

struct GgufLinearStorage {
    std::vector<GgufLinearSegment> segments;
};

using LinearStorage = std::variant<
    Bf16LinearStorage,
    Int8LinearStorage,
    Int4LinearStorage,
    GgufLinearStorage>;

struct LinearWeight {
    int rows = 0;
    int cols = 0;
    LinearStorage storage;
};
```

Do not keep `LinearStorageKind` as a parallel authoritative tag. If a cheap kind
query is useful for diagnostics or dispatch, derive it from the active
alternative.

GGUF segment type (`GgmlType`) already identifies Q4_K/Q6_K and future native
GGUF layouts, so avoid creating one top-level alternative per GGUF quantization
unless execution semantics truly differ at that level.

### Migrate

- `validate_storage()`;
- row slicing;
- linear loaders;
- GEMM/GEMV dispatch;
- expert views;
- memory accounting;
- all `kind == ...` checks.

### Tests

- each storage alternative validates independently;
- contradictory pointer/tag states are impossible to construct;
- row slicing preserves only the active storage payload;
- adding a synthetic storage alternative makes dispatch visitors fail
  exhaustively where handling is required.

## 1.2 Represent absence in `FeedForwardWeights`

### Current problem

The semantic program represents no FFN with `std::monostate`, but CUDA runtime
weights use:

```cpp
using FeedForwardWeights = std::variant<DenseFfnWeights, MoeFfnWeights>;
```

and mixer-only layers are represented as an empty `DenseFfnWeights{}` with null
pointers.

### Target

Mirror the semantic state directly:

```cpp
using FeedForwardWeights = std::variant<
    std::monostate,
    DenseFfnWeights,
    MoeFfnWeights>;
```

Delete null dense payload as an absence sentinel. A present `DenseFfnWeights`
must mean that a dense FFN exists and its required weights are valid.

## 1.3 Replace resident/offloaded expert pointer bags with a storage variant

### Current problem

`MoeFfnWeights::offloaded()` is derived from `gate_up_ptrs != nullptr`, while the
same object also carries resident `ExpertLinearWeight*` payloads. Multiple
residency modes can coexist structurally.

### Target

Model expert storage explicitly, for example:

```cpp
struct ResidentExpertWeights {
    const ExpertLinearWeight* gate_up;
    const ExpertLinearWeight* down;
};

struct OffloadedExpertWeights {
    const __nv_bfloat16* const* gate_up;
    const __nv_bfloat16* const* down;
};

using ExpertWeightStorage = std::variant<
    ResidentExpertWeights,
    OffloadedExpertWeights>;
```

The type should be able to grow naturally to host-resident, disk-cached, managed
memory, or other future residency modes without adding another nullable pointer
family to `MoeFfnWeights`.

Do not preserve `offloaded()` as an authoritative boolean. A convenience query
may be derived from the active alternative if callers genuinely need it.

## 1.4 Split CUDA attention runtime state by semantic/storage alternative

### Current problem

`AttentionLayer` currently aggregates ordinary attention, latent attention,
BF16 KV cache, INT8 KV cache, latent cache, optional gate/norm payloads, and
shared-KV ownership into one structure. Many fields are meaningful only for a
subset of attention semantics.

This is another manual union, even though the semantic layer already has typed
attention-state alternatives.

### Target

Separate immutable/common attention weights from mutually exclusive runtime
state, for example:

```text
AttentionWeights
AttentionRuntimeState = variant<
    OrdinaryBf16KvState,
    OrdinaryInt8KvState,
    LatentAttentionRuntimeState,
    ...>
```

Do not require every attention layer to own buffers for every possible state
family. Keep shared immutable geometry where it is genuinely common.

This phase should follow the current semantic `AttentionStateSpec` rather than
inventing a second independent state taxonomy.

---

# Phase 2 — Remove the CPU mixer × FFN Cartesian variant

## 2.1 Replace `WeightLayer` flattening with composition

### Current problem

`CpuCompiledModel` currently has:

```text
WeightLayer = variant<
    AttentionWeights,
    ConvolutionWeights,
    GatedDeltaNetWeights,
    Mamba2Weights,
    MlpOnlyWeights,
    MoeWeights>
```

while `MoeWeights` itself contains another variant of the mixer-specific weight
structs. This encodes the product of two independent axes by duplicating mixer
alternatives inside the MoE alternative.

The duplication is already visible in memory accounting, where a visitor over
`WeightLayer` contains another visitor over `MoeWeights::operator_layer` and
repeats mixer-specific accounting.

### Target

Mirror the semantic decomposition:

```cpp
using CpuMixerWeights = std::variant<
    AttentionWeights,
    ConvolutionWeights,
    GatedDeltaNetWeights,
    Mamba2Weights,
    MlpOnlyWeights>;

using CpuFeedForwardWeights = std::variant<
    std::monostate,
    DenseFeedForwardWeights,
    MoeWeights>;

struct CpuLayerWeights {
    CommonWeights common;
    CpuMixerWeights mixer;
    CpuFeedForwardWeights feed_forward;
};
```

Move common weights to one owner rather than embedding a duplicate `CommonWeights`
inside every mixer and again through `MoeWeights`.

### Completion condition

Adding a new mixer must require adding it once to `CpuMixerWeights`; it must not
also require adding it as an alternative nested inside the MoE representation.

## 2.2 Align CPU layer state with the same independent axes

Audit `LayerState` and any feed-forward/session state for the same flattening
pattern. Mixer recurrence state should remain a mixer concern; MoE routing/cache
scratch should remain a feed-forward concern.

---

# Phase 3 — Make `WeightPlan` the authoritative physical binding contract

This phase removes one of the most important current boundary violations.

## 3.1 Stop CUDA expert setup from sniffing checkpoint tensor names

### Current problem

CUDA weight setup currently checks spellings such as:

```text
mlp.experts.0.gate_proj.weight
mlp.experts.0.gate_proj.weight_packed
feed_forward.experts.0.w1.weight
...
```

and derives categories such as "individual expert model", "packed expert
model", and "named expert model" during backend setup.

This violates the intended extension path documented in
`docs/EXTENDING_ARCHITECTURES.md` and `docs/EXTENDING_MOE.md`: checkpoint names
should already have been resolved to semantic roles and payload locations before
backend execution/setup.

### Target

Extend the canonical weight plan only as much as necessary to carry irreducible
physical information, for example:

```cpp
struct ExpertTensorLocation {
    TensorRole role;
    int expert;
    TensorLocator locator;
};

struct ExpertWeightLayout {
    ExpertPayloadLayout layout;
    std::vector<ExpertTensorLocation> tensors;
};
```

The exact type may differ, but backend code must receive answers to questions
such as:

- individual vs packed payload;
- expert count;
- role/shape for gate/up/down;
- source locator/region;
- shared-expert payload;
- whether native block storage is available;

without re-deriving them from names.

### Delete

- raw `repo.contains("...expert...")` layout selection in CUDA backend setup;
- suffix-based `_packed` probing used to decide semantic/physical layout;
- backend fallback naming trees for already-resolved tensor roles.

## 3.2 Apply the same audit to CPU

CPU weight loading should likewise consume canonical requests/locators rather
than infer checkpoint family/layout from names when the information already
exists upstream.

## 3.3 Preserve repository capabilities, do not duplicate them

Continue using existing repository capability interfaces for concerns that are
truly repository capabilities (`ILocatableTensorRepository`,
`IRandomAccessTensorReader`, `INativeBlockStorageRepository`). Do not copy
those facts into a parallel backend-specific format enum.

---

# Phase 4 — Make automatic layer inference rule-driven

This is the largest architectural extensibility phase.

## 4.1 Replace the central tensor-grammar cascade

### Current problem

`src/model/inference/layer_semantics.cpp` currently recognizes layer families by
hard-coded tensor spelling predicates and an ordered cascade covering standard
attention, latent attention, fused/factorized gated-delta, Mamba-2,
short-convolution, MLP-only, FFN, and MoE.

Adding another reusable primitive or another distinct tensor grammar expands the
same central function and risks precedence conflicts between detectors.

### Target

Introduce a load-time inference-rule boundary. The implementation may use a
small interface, value-erased callable, or another non-hot-path registry. The
important contract is conceptually:

```cpp
struct LayerInferenceResult {
    MixerSpec mixer;
    FeedForwardSpec feed_forward;
    std::vector<TensorRoleBinding> bindings;
    std::vector<EvidenceItem> evidence;
};

class ILayerInferenceRule {
public:
    virtual ~ILayerInferenceRule() = default;
    virtual std::string_view id() const = 0;
    virtual LayerProbe probe(const InferenceInput&, int layer) const = 0;
    virtual LayerInferenceResult resolve(const InferenceInput&, int layer) const = 0;
};
```

The interface shape is illustrative, not mandatory. Avoid virtual dispatch in
execution; this catalog is load-time inference only.

Possible built-in rules:

```text
StandardAttentionInferenceRule
LatentAttentionInferenceRule
FusedGatedDeltaInferenceRule
FactorizedGatedDeltaInferenceRule
Mamba2InferenceRule
ShortConvolutionInferenceRule
MlpOnlyInferenceRule
```

Rules must report specificity/conflict explicitly rather than relying on an
accidental `if/continue` order.

## 4.2 Produce semantic result and bindings from the same rule

### Current problem

`layer_semantics.cpp` recognizes a grammar and constructs a semantic mixer;
`layer_bindings.cpp` then dispatches over the mixer and repeats family-specific
tensor naming/shape knowledge to bind the physical tensors.

This creates two coordinated registries that must evolve together.

### Target

A rule that recognizes a physical grammar should emit both:

- the canonical semantic alternative;
- the role bindings/evidence that justified it.

The generic pipeline should solve/validate the returned bindings, but should not
rediscover the grammar independently.

## 4.3 Keep semantic variants closed and exhaustive

Rule registration makes **checkpoint grammar recognition** open to extension. It
does not make `MixerSpec` open-world. A genuinely new executable primitive still
adds a semantic alternative and must be handled by every relevant backend.

This distinction is intentional:

```text
new spelling for existing attention
    -> new/extended inference rule only

new semantic mixer primitive
    -> semantic variant + compiler/backend support
```

---

# Phase 5 — Replace the universal metadata property bag with typed fact groups

## 5.1 Decompose `NormalizedModelMetadata`

### Current problem

`NormalizedModelMetadata` accumulates optional fields for every supported
family: core dimensions, attention, Mamba-2, recurrent/gated-delta, latent
attention, MoE, XSA, and more. New architectures naturally add another set of
`std::optional<T>` fields to the same universal struct.

This scales toward a "schema of every model ever supported" and makes it easy to
construct combinations that do not belong to any coherent inference profile.

### Target

Group facts by semantic concern, for example:

```cpp
struct CoreModelFacts;
struct AttentionFacts;
struct MoeFacts;
struct Mamba2Facts;
struct GatedDeltaFacts;
struct LatentAttentionFacts;
```

Then represent mutually exclusive recurrent families with a variant or other
explicit sum type when the facts are genuinely exclusive.

Do not over-normalize shared facts: hidden size, layer count, vocabulary, token
policy, and other truly global values should remain globally owned.

## 5.2 Eliminate stringly typed canonical facts

Fields such as MoE score/selection methods should become typed values before
semantic synthesis. Strings are acceptable at import boundaries; they should
not remain canonical model facts when the domain is known.

## 5.3 Preserve layer-scoped facts explicitly

`LayerScopedValue<T>` is a useful representation where the checkpoint can
provide either a global default or per-layer override. Preserve that idea rather
than flattening it into repeated arrays or backend-specific defaults.

---

# Phase 6 — Remove source-format identity from generic semantic inference

## 6.1 Replace `input.is_gguf()` semantic decisions with normalized facts

### Current problem

Recent cleanup correctly replaced tensor-name sniffing such as `"blk."` with
`InferenceInput::source_format`, but generic semantic code still uses
`input.is_gguf()` to decide facts such as whether recurrent parameters require a
transform.

That is better than name sniffing, but generic inference still knows the source
format identity.

### Target

Make each importer/normalizer produce the actual semantic encoding fact, for
example:

```cpp
enum class DecayParameterEncoding {
    LogA,
    Pretransformed,
};
```

or a more precise typed representation owned by the relevant recurrent fact
set.

Then generic synthesis asks "what does this value mean?", not "did this come
from GGUF?".

## 6.2 Keep unavoidable format conventions at the format boundary

The recent GGUF no-RoPE profile is the right pattern when the file format itself
cannot express a distinction and format-specific knowledge is unavoidable.
Keep such knowledge in a GGUF-specific importer/profile module and emit a
canonical semantic result.

---

# Phase 7 — Move mixer-specific allocation requirements out of neutral topology

## 7.1 Shrink `ExecutionTopology`

### Current problem

`ExecutionTopology` currently contains mixer-specific counters and maxima such
as:

- attention layer count and many `maximum_attention_*` values;
- convolution counts/dimensions;
- gated-delta maxima;
- Mamba-2 maxima;
- MLP-only counts.

Every new mixer therefore expands a backend-neutral topology type even when the
extra fields exist only to size one backend's workspace.

### Target

Keep only genuinely backend-neutral derived topology in the neutral layer.
Move allocation/execution sizing into backend plans:

```text
ModelGraph / CompiledModelProgram
    -> CpuExecutionPlan / CpuWorkspacePlan
    -> CudaExecutionPlan / CUDA resource planning
```

A backend plan may exhaustively visit every mixer because it genuinely needs to
know what that backend allocates. The neutral topology should not become a
catalog of every backend scratch requirement.

## 7.2 Remove "derive neutral maxima, then re-derive backend workspace" chains

`CpuWorkspacePlan::from_topology()` currently copies/combines many neutral
maximum fields into another backend-owned object. Prefer deriving the CPU plan
directly from the compiled semantic program plus truly neutral dimensions.

## 7.3 Preserve reusable neutral topology only where multiple backends consume it

Examples may include layer count, checkpoint-layer mapping, and other facts with
independent semantic/runtime meaning. Do not move values merely to make the
neutral type smaller if they are genuinely shared concepts.

---

# Phase 8 — Replace monolithic CPU scratch growth with workspace composition/planning

## 8.1 Decompose `CpuWorkspace`

### Current problem

`CpuWorkspacePlan` and `CpuWorkspace` contain dedicated members for every known
operator and repeat them for token/chunk paths. Adding a mixer often means adding
fields to:

```text
ExecutionTopology
CpuWorkspacePlan
CpuWorkspace::ensure
CpuWorkspace::ensure_chunk
CpuWorkspace members
forward execution
```

### Target A — typed workspace composition

At minimum, separate concerns:

```cpp
struct CpuCommonWorkspace;
struct CpuAttentionWorkspace;
struct CpuRecurrentWorkspace;
struct CpuFeedForwardWorkspace;
```

and make each operator family own its scratch requirements.

### Target B — scratch-region planner

If profiling shows memory reuse is worthwhile, evolve toward a backend scratch
planner that allocates/aliases regions according to non-overlapping lifetimes.
Do not build a generic allocator framework before measuring the benefit.

## 8.2 Remove duplicated token/chunk buffer taxonomies where lifetimes permit

Audit whether `qkv` vs `chunk_qkv`, latent vs chunk-latent, and recurrent vs
chunk-recurrent buffers must be physically independent. Where execution phases
never overlap, plan aliases rather than maintaining two ever-growing parallel
sets.

---

# Phase 9 — Centralize CPU linear-storage and ISA dispatch

Implemented (Sprint E items 21–22):

- **9.1 (storage dispatch):** `CpuLinearEngine` no longer re-classifies the
  `CpuLinearWeight` variant in four independent `std::all_of(holds_alternative<…>)`
  blocks. A single exhaustive `classify_weight` (`LinearStorageKind::{Q4,Int8,Gguf}`)
  is the one decision point; `gemv`/`gemm` route each segment through per-storage
  helpers (`gemv_int8`, `gemv_gguf`, `gemm_int8`) or the existing `Q4GroupMatrix` /
  `gemm_gguf` paths. Mixed storage still throws at one obvious site.
- **9.2 (ISA metadata):** `CpuCompiledModel::Shared::resolve_isa` and the four
  per-kernel selectors' divergent resolution knowledge are replaced by one registry
  (`include/celeg/backend/cpu/kernel_backend.hpp`, `CpuKernelBackend`/`CpuKernelTable`)
  built in `kernel_backend.cpp`. `cpu_resolve_kernel_backend` is the single home for
  ISA selection: it unifies known / compiled (`cpu_isa_compiled`) / host-supported
  (`CpuKernelBackend::supports_hw`) / kernel-available into one table and resolves
  `Auto` by priority. The four `select_*` functions remain as the per-ISA kernel
  registry consumed by the table. `CpuLinearEngine` and the model weights path consult
  only `cpu_resolve_kernel_backend`.

## 9.1 Give each storage alternative one execution dispatch home

### Current problem

`CpuLinearEngine::gemv`, `gemm`, `gemv_transpose`, and `gemm_grouped` each
contain independent storage classification logic (`Q4GroupMatrix`, INT8,
GGUF-native, mixed-storage fallback/rejection). Every new format risks edits in
all of these functions.

### Target

Centralize storage dispatch through exhaustive visitors or a per-storage kernel
set, for example:

```cpp
struct CpuLinearKernelSet {
    GemvFn gemv;
    GemmFn gemm;
    GemvTransposeFn transpose;
    GroupedGemmFn grouped;
};
```

The exact shape must reflect real capability differences; do not add optional
function pointers that silently mean "unsupported" without a load/plan-time
capability check.

A new storage should have one obvious implementation registration point and
explicitly declared unsupported operations.

## 9.2 Replace duplicated ISA knowledge with one implementation table

### Current problem

`CpuIsa` names more ISAs than every kernel family currently implements, while
`resolve_isa()` carries a separate manual subset and individual kernel selectors
have their own dispatch knowledge.

### Target

Centralize ISA implementation metadata, conceptually:

```cpp
struct CpuKernelBackend {
    CpuIsa isa;
    int priority;
    CpuKernelTable kernels;
    CpuCapabilityPredicate supported;
};
```

The actual design may remain compile-time/static. The important property is that
"known ISA", "compiled ISA", "host-supported ISA", and "kernel implementation
available" are not four divergent switch trees.

## 9.3 Keep kernel selection out of semantic model objects

ISA and storage-kernel capability remain backend planning concerns, not fields in
`ModelGraph` or `CompiledModelProgram` semantic identity.

---

# Phase 10 — Make backend extension ABI-consistent

## 10.1 Replace address-based C++ option type tagging at the extension boundary

### Current problem

`IBackendOptions::as<T>()` uses an address of a function-local static object as a
manual type tag and returns data through `const void*`. That is adequate inside
one process image when all participants share the same C++ instantiation, but it
is a fragile identity mechanism for a real ABI/plugin boundary across separate
DSOs/DLLs.

This deserves review because `IAbiBackendFactory` explicitly exists as an ABI
extension concept.

### Target

Choose an ABI-stable option identity/decoding contract. Viable directions
include:

- backend ID + backend-owned byte payload/version decoded only by its factory;
- stable numeric/string schema IDs plus versioned payload;
- another explicit C ABI representation.

Do not depend on C++ RTTI/type identity crossing a plugin ABI unless that is a
conscious, documented same-toolchain constraint.

## 10.2 Make the public C API consistently N-backend capable

### Current problem

`celeg_engine_options` already selects a backend by `backend_id` and passes an
opaque backend-owned payload, but `include/celeg/api.h` simultaneously exposes:

```text
celeg_backend { CPU, CUDA }
celeg_backend_capabilities(celeg_backend)
CPU-specific single-model creation
CPU/CUDA concrete backend option structs in the core header
```

The lower layer is extensible while the public surface still models the backend
universe as CPU/CUDA.

### Target

Make the core engine C ABI backend-neutral, for example:

```c
celeg_status celeg_backend_capabilities(
    const char* backend_id,
    ...);
```

Backend-specific convenience initializers/config structs may live in backend
headers or remain optional convenience APIs, but adding a third backend must not
require extending a central public enum.

Evaluate whether the CPU-only `celeg_model_*` direct API should:

1. remain explicitly documented as a legacy/specialized CPU convenience API;
2. move to a CPU-specific header; or
3. become a backend-neutral direct-session API.

Do not keep two competing generic creation models.

## 10.3 Preserve the existing serving substitution boundary

`IRequestService`, `ISchedulerDriver`, and `IServiceDiagnostics` are already the
right common serving contracts. New backends should implement those roles rather
than forcing a new monolithic engine base class.

---

# Phase 11 — Reduce internal orchestration aggregates without facade churn

The earlier backend SOLID review correctly identified oversized internal
orchestration/state aggregates, but method count alone is not the criterion.
The public facades are already reasonably small and implementation is physically
split across translation units. Refactor only where state ownership/reasons to
change remain mixed.

## 11.1 Decompose `CudaCompiledModel` by ownership, not by forwarding wrappers

`CudaCompiledModel` still coordinates model resources, session state,
execution, attention/MoE paths, sampling, graph capture, diagnostics,
persistence, speculative state, and resource lifecycle.

Continue moving cohesive ownership into existing or new collaborators where
that collaborator can own both state and behavior. Good examples are the
already-extracted sampling and decode-graph lifecycle components.

Do **not** "fix" this by creating many interfaces whose only purpose is to
forward calls back into `CudaCompiledModel`.

Target seams include, where current state ownership supports them:

```text
session lifecycle/state
weight/bootstrap ownership
persistence codec/state snapshot
execution orchestration
speculative state
backend diagnostics
```

## 11.2 Apply the same standard to `CpuCompiledModel::Shared` and `CpuCompiledModel`

The CPU side also aggregates checkpoint/bootstrap state, repository, pack cache,
linear engine, workspace planning, topology/program, weight store, expert
backing/cache, KV pools, external attention memory, session state, and execution.
Do not treat this as only a CUDA SRP problem.

Candidate ownership cuts include:

```text
CpuModelResources / immutable compiled resources
CpuWeightStore / weight loading + memory accounting
CpuPackCacheContext
CpuKvResources
CpuExpertResources
CpuSessionState
CpuExecutionWorkspace
```

Names are illustrative. Prefer ownership clarity over arbitrary file/class count.

## 11.3 Split persistence serialization from file transport where useful

CUDA session persistence currently couples serialization logic to direct file
I/O and device/host transfer. Extract a byte/blob codec or snapshot value if it
improves deterministic tests and enables alternate transport. Do not introduce a
repository-wide filesystem abstraction unless another caller needs it.

---

# Phase 12 — Future CPU GGUF native-kernel work (feature/performance, not cleanup)

This section is deliberately separated from the refactor phases so a structural
cleanup does not accidentally turn into a kernel project.

## 12.1 Current behavior

The CPU native GGUF dot path uses Q8_K activations in 256-element superblocks.
`cpu_quantize_q8k*` and the native dot entry points require column counts that are
multiples of 256.

`weight_codec.cpp` repacks layouts that cannot use the current native path. This
is a valid fallback, not an architectural smell by itself.

## 12.2 Q4_0/Q5_0 nuance

Q4_0 and Q5_0 use 32-element weight blocks, but that does **not** make them
mathematically incompatible with a 256-element Q8_K activation block when the
matrix width is a multiple of 256. A native kernel can consume eight 32-element
weight blocks against the corresponding eight 32-element slices of one Q8_K
activation superblock.

Therefore the future work for aligned Q4_0/Q5_0 is better described as:

> implement and benchmark native Q4_0/Q5_0 × Q8_K scalar/SIMD dot kernels

rather than "a mandatory new 32-element activation representation".

## 12.3 Non-256-aligned widths

Weights whose column count is not a multiple of 256 genuinely do not fit the
current activation contract. Supporting them natively requires a deliberate
execution design, such as:

- a Q8_K tail kernel;
- padded Q8_K execution with proven correctness/performance;
- a 32-element Q8 activation path;
- another storage-specific tail strategy.

## 12.4 Q2_K/Q3_K audit

The scalar GGUF implementation contains Q2_K/Q3_K logic, while the current
weight codec also repacks those formats in paths where the optimized native
execution is not appropriate. Audit this separately from Q4_0/Q5_0 and decide
whether the desired feature is:

- optimized AVX2/AVX-VNNI/AVX-512 native dots;
- keeping repack as the preferred performance path;
- or selecting based on measured matrix shape/ISA.

Do not change the fallback based on aesthetics.

## 12.5 Required validation for every new native quantized path

Before making a native path default:

1. compare against dequantized/reference output using the numerical comparison
   utilities;
2. test representative real GGUF models;
3. benchmark decode and prefill separately;
4. measure end-to-end impact, not only the microkernel;
5. include memory-bandwidth/working-set effects;
6. retain the old repack path only if it is still a legitimate execution
   strategy, not as compatibility scaffolding.

---

# Phase 13 — Extension-cost acceptance scenarios

The refactor is complete when the following scenarios have a small, predictable
edit surface.

## 13.1 New checkpoint spelling for existing semantics

Example: another SafeTensors family expresses ordinary GQA + dense MLP with new
tensor names.

Expected edits:

```text
format/inference rule or naming grammar
tests
```

Forbidden edits:

```text
CPU execution
CUDA execution
backend operator switches
expert cache/residency
C API
semantic ModelGraph types
```

## 13.2 New source format for existing semantics

Expected edits:

```text
ICheckpointFormat implementation
repository/import normalization
tests/registration
```

Backends must not gain `if (source_format == NewFormat)` branches.

## 13.3 New mixer semantic primitive

Expected edits:

```text
new MixerSpec alternative
semantic validation/fingerprint
inference rule(s) that can produce it
weight roles/plan if it needs new tensors
CPU lowering/execution if supported
CUDA lowering/execution if supported
backend workspace/resource planning
capability matrix/tests
```

It is acceptable and desirable that exhaustive semantic visitors fail to compile
until these required consumers are updated.

Forbidden secondary explosion:

```text
new duplicate mixer alternative inside MoE weight wrappers
new neutral-topology fields solely for one backend's scratch
checkpoint-name checks inside backend execution
unrelated C API enum edits
```

## 13.4 New CPU weight storage

Expected edits:

```text
one CpuLinearMatrix/LinearStorage alternative
codec/loading
central storage kernel dispatch
quality/perf tests
```

It should not require a new set of ad-hoc `if (format)` branches across every
GEMM/GEMV caller.

## 13.5 New CPU ISA implementation

Expected edits:

```text
ISA implementation descriptor/kernel table
kernel implementations
capability tests/benchmarks
```

It should not require manually synchronizing independent "known",
"compiled", "supported", and "selectable" switch lists.

## 13.6 Third backend

Expected edits:

```text
backend implementation
factory/module registration
backend-owned option schema/decoder
serving interfaces
backend tests
```

Forbidden edits:

```text
central public backend enum
CPU/CUDA-specific selection switch in generic runtime
model semantic types only to advertise backend capability
```

---

# 14. Recommended implementation order

Use small commits that each remove one old representation completely.

### Sprint A — impossible-state cleanup — DONE

1. [x] CUDA `FeedForwardWeights` gains `std::monostate`; delete empty-dense sentinel. (`3215871`)
2. [x] CUDA `LinearWeight` becomes a storage variant; migrate all consumers. (`bb7ae24`)
3. [x] CUDA MoE resident/offloaded expert storage becomes a variant. (`578dd34`)
4. [x] CUDA attention runtime storage is split by active attention-state/storage family. (`ea6d143`)

### Sprint B — CPU composition — DONE

5. [x] Replace CPU `WeightLayer` Cartesian flattening with `mixer + feed_forward` composition. (`c3ded51`)
6. [x] Consolidate common CPU layer weights under one owner. (`c3ded51`)
7. [x] Simplify memory accounting/visitors after the new composition. (`c3ded51`)

### Sprint C — checkpoint boundary — DONE (one disclosed MTP sub-gap)

8. [x] Enrich canonical weight planning with the irreducible expert physical layout. (`facb607`)
9. [x] Delete CUDA raw expert-name/layout probing. (`facb607` for outer `weight_setup.cpp`; inner `experts.cu`/`loader_experts.cu`/`weight_upload.cpp` loaders rewritten onto `MoeExpertTensorNames` resolved-plan bundles — `_named` variants, `expert_name()`/`named_expert_name()` builders, inner packed probes, and the `load_router_weight()` spelling prober deleted; `mtp_weight_setup.cpp` now feeds the same API from one local `mtp_expert_names()` helper because MTP layers are not planned yet — that residue is the disclosed sub-gap)
10. [x] Delete equivalent CPU re-inference where found. (CPU was already plan-driven; picked up the existence-aware fix for free via `facb607`, the int4 virtual-name resolution fix, and the packed-branch shared-expert guards)
11. [x] Add poisoning/alternate-spelling regression tests proving backends consume roles/locators. (`facb607`, extended for `moe_expert_tensor_names()`/`MoeRouterBias`)

### Sprint D — inference extensibility — DONE

12. [x] Introduce a layer inference-rule catalog. (`afd1c27`)
13. [x] Move existing mixer grammar detection into rules without semantic change. (`afd1c27`)
14. [x] Make each rule emit both semantics and tensor-role bindings. (`afd1c27`)
15. [x] Delete the central ordered `has_mamba/has_mla/has_kda/...` cascade and duplicate binding dispatch. (`afd1c27`)
16. [x] Decompose `NormalizedModelMetadata` into typed fact groups.
17. [x] Replace remaining generic `is_gguf()` semantic branches with normalized encoding facts (`DecayParameterEncoding`).

### Sprint E — backend planning

18. [x] Move mixer-specific maxima out of neutral `ExecutionTopology` where they exist only for backend allocation.
19. [x] Derive CPU workspace requirements directly from compiled program semantics (`CpuWorkspacePlan::from_program`).
20. [x] Decompose/plan `CpuWorkspace`; remove avoidable token/chunk duplicate scratch families; source `conv_cache` from per-layer `ShortConvolutionSpec`.
21. [x] Centralize CPU linear storage dispatch.
22. [x] Centralize CPU ISA implementation metadata/selection.

### Sprint F — extension ABI and orchestration

23. [x] Replace address-based `IBackendOptions` type tagging with an ABI-stable decoding contract.
24. [x] Make the core C engine API consistently backend-ID based; remove the requirement for a central CPU/CUDA backend enum to add a backend.
25. [x] Decide the explicit future of the CPU-only direct model C API (kept as a documented CPU-only convenience API; `celeg_engine_*` is the backend-neutral path).
26. Continue ownership-based decomposition of `CudaCompiledModel` and `CpuCompiledModel::Shared`.
27. Split persistence codec/snapshot from file transport where tests justify it.

### Separate performance project

28. Benchmark/design native Q4_0/Q5_0 × Q8_K CPU dots.
29. Evaluate Q2_K/Q3_K optimized native execution vs repack.
30. Design/test non-256-width activation tails only if real models/performance justify it.

Items 28-30 must not block completion of the structural extensibility plan.

---

# 15. Architectural end state

The desired dependency flow is:

```text
checkpoint bytes/files
        |
        v
ICheckpointFormat / repository / format-specific normalization
        |
        v
canonical evidence + typed normalized facts
        |
        v
layer inference rules
        |
        +------> semantic ModelGraph / CompiledModelProgram
        |
        +------> TensorRole bindings / authoritative WeightPlan
                         |
             +-----------+-----------+
             |                       |
             v                       v
        CPU compiler             CUDA compiler
             |                       |
             v                       v
      CPU execution plan       CUDA execution plan
      storage/ISA kernels      storage/GPU kernels
             |                       |
             +-----------+-----------+
                         |
                         v
       IRequestService / ISchedulerDriver / IServiceDiagnostics
                         |
                         v
              backend-neutral engine API
```

The architecture should make the following statements true:

- **Models describe semantics, not backend capabilities.**
- **Weight plans describe resolved physical roles/locations, not architecture names.**
- **Backends never identify a checkpoint family by tensor spelling.**
- **Runtime storage alternatives cannot coexist in contradictory states.**
- **MoE residency is a storage policy, not a nullable-pointer convention.**
- **Mixer and FFN are independent axes in both semantic and backend weight state.**
- **Checkpoint grammar recognition is extensible at load time.**
- **Semantic alternatives remain deliberately exhaustive at compile time.**
- **Backend workspace/capability planning belongs to the backend.**
- **Adding a backend does not expand a central backend enum.**
- **Kernel limitations remain explicit performance work rather than being disguised as cleanup.**
