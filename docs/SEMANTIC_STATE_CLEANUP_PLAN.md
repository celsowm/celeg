# Semantic State Cleanup Plan

## Status

Re-audited against the current `master` tree.

This document is the active roadmap for the remaining semantic-state cleanup. It
**supersedes `docs/MODEL_TOPOLOGY_OWNERSHIP_REFACTORING_PLAN.md` as the active
implementation plan**. The topology plan remains useful historical context until
Phase 4 below is completed, but new implementation decisions should be recorded
here instead of creating a second competing roadmap. Once the remaining topology
work has landed, delete the superseded plan rather than maintaining two sources
of architectural truth.

This plan has no backward-compatibility constraint for internal C++ APIs.
Struct layouts, function signatures and internal headers may change as clean
breaks. Migrate every consumer and delete obsolete names rather than introducing
deprecated wrappers, aliases, views, adapters or fallback compatibility paths.

## Goal

Continue the model/runtime cleanup started by the explicit-sum-types refactor
and remove the remaining cases where CELEG stores multiple representations of
the same semantic choice.

The governing rule is:

> If `validate()` mainly exists to prove that a tag, boolean, sentinel,
> optional, derived cache, or parallel payload describes the same alternative
> as another field, the representation is wrong. Prefer `std::variant`,
> `std::optional`, a dedicated value type, or a backend-owned execution plan so
> invalid states are unrepresentable.

A second rule applies to variants:

> Every visitor that performs semantic lowering, validation, allocation,
> capability analysis or fingerprinting must be compile-time exhaustive. A new
> semantic alternative must fail compilation everywhere that needs to
> understand it instead of silently inheriting a default interpretation.

## Non-goals

- No public compatibility layer for internal C++ implementation types.
- No architecture-name switches in backends.
- No backend capability flags added to semantic model objects.
- No duplicated semantic fingerprints whose inputs can diverge from the actual
  executable state.
- No broad rewrite unrelated to the state-modeling problems listed below.
- No simultaneous redesign of the external descriptor JSON schema unless a
  later change is independently justified. First type the internal C++ IR.

## Invariants for every phase

Each phase is complete only when:

1. impossible combinations are no longer constructible through the normal
   validated type surface;
2. dead/inactive payload no longer participates in semantic fingerprints;
3. derived values have one source of truth;
4. backend-specific capability decisions live in backend compilation/planning;
5. all consumers use the new representation directly;
6. old enums, sentinels, boolean enable flags, proxy views, adapters and
   fallback compatibility paths are deleted;
7. semantic `std::variant` visitors used for lowering, validation, allocation,
   capability analysis or fingerprinting are compile-time exhaustive, using an
   explicit final `static_assert(always_false_v<T>)` or equivalent;
8. runtime code does not fall back from missing per-layer/program semantics to
   checkpoint-wide defaults;
9. tests cover the valid alternatives, removed ambiguity and semantic identity
   before the old representation is deleted;
10. CPU and CUDA build/tests remain green after each commit-sized cut.

---

# Phase 0 — Lock semantic identity and exhaustiveness guardrails

Do this before the structural refactors. Phase 9 remains the final identity
audit, but the protection must exist from the first migration rather than being
added after fifteen semantic changes have already landed.

## 0.1 Establish fingerprint regression baselines

Add focused tests for representative semantic alternatives before changing the
representations. At minimum cover:

- unscaled, linear, Dynamic NTK, YaRN, LongRoPE and Llama-3 frequency scaling;
- optional norms;
- ordinary and latent attention state;
- attention output gates;
- top-k and grouped MoE selection;
- shared and private KV;
- current-sequence and external-memory attention sources;
- per-layer input enabled/absent behavior as it exists before migration.

The tests should assert semantic relationships, not preserve the textual format
of the current fingerprint. A refactor may legitimately change fingerprint
encoding while preserving semantic equality/inequality.

## 0.2 Enforce exhaustive semantic visitors

Audit every visitor that converts or interprets semantic variants. Remove
fallthrough/default interpretations such as "anything not dense is MoE" or
"anything not one of the known mixers is MLP-only".

Use the same pattern everywhere:

```cpp
std::visit([](const auto& value) {
    using T = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<T, AlternativeA>) {
        // ...
    } else if constexpr (std::is_same_v<T, AlternativeB>) {
        // ...
    } else {
        static_assert(always_false_v<T>, "unhandled semantic alternative");
    }
}, value);
```

This applies especially to:

- semantic fingerprints;
- graph/program lowering;
- backend capability checks;
- execution-topology derivation;
- weight-plan derivation;
- descriptor-to-semantic construction.

## 0.3 Add descriptor fixtures before rewriting descriptor IR

The descriptor directory may not contain every runtime architecture used in
practice. Add in-tree fixtures that exercise the real descriptor registration
and resolution path before Phase 5.

Cover at least:

- dense attention;
- one scaled-RoPE alternative;
- MoE with shared expert and softmax routing;
- grouped MoE routing;
- split/optional norms;
- external-memory or latent-attention configuration where supported by the
  current descriptor path.

---

# Phase 1 — Finish semantic sum types in `model/`

This is the highest-priority structural phase because these types participate
directly in `ModelGraph::fingerprint()` and therefore in compiled semantic
identity.

## 1.1 Replace `RopeScalingSpec` tagged union

### Current problem

`RopeScalingSpec` stores a `RopeScalingKind` plus payload for every scaling
family at the same time. Inactive fields can be populated and currently
participate in the semantic fingerprint.

The replacement must not merely distribute the old fields among several
structs. Each alternative must contain **only values consumed by that
alternative's executable semantics**.

### Target

Following the execution semantics currently implemented by
`rope_frequency()`/`rope_attention_scale()`, the target should be close to:

```cpp
struct NoRopeScaling {};

struct LinearRopeScaling {
    double factor;
};

struct DynamicNtkRopeScaling {
    double factor;
    int original_context;
};

struct YarnRopeScaling {
    double factor;
    double attention_factor;
    double beta_fast;
    double beta_slow;
};

struct LongRopeScaling {
    int original_context;
    std::vector<float> short_factors;
    std::vector<float> long_factors;
};

struct Llama3FrequencyScaling {
    double factor;
    int original_context;
    double low_frequency_factor;
    double high_frequency_factor;
};

using RopeScalingSpec = std::variant<
    NoRopeScaling,
    LinearRopeScaling,
    DynamicNtkRopeScaling,
    YarnRopeScaling,
    LongRopeScaling,
    Llama3FrequencyScaling>;
```

If execution semantics change while implementing this phase, update the
alternative payload deliberately and test the new semantics. Do not retain an
unused field merely because the old tagged union contained it.

### Delete

- `RopeScalingKind` if it becomes fully derivable from the variant;
- switch/tag reconciliation logic;
- validation requirements for fields unused by an alternative;
- serialization of inactive scaling fields.

### Tests

- equal executable scaling semantics produce equal fingerprints;
- changing an active scaling field changes the fingerprint;
- a field that has no meaning for an alternative does not exist on that type;
- each scaling alternative validates independently;
- all scaling visitors are compile-time exhaustive.

## 1.2 Make optional norms actually optional

### Current problem

`NormSpec::enabled()` treats `epsilon <= 0` as absence while
`NormSpec::validate()` treats the same value as invalid. This uses an invalid
object as an optional/sentinel value.

### Target

A `NormSpec` represents an existing norm and is valid after construction and
validation. Required norm positions remain plain `NormSpec`; semantically
optional positions become `std::optional<NormSpec>`.

Do not conflate:

- norm absent;
- norm present but weightless (`NormWeightKind::None`).

### Delete

- `NormSpec::enabled()`;
- zero-epsilon sentinel construction;
- guards of the form `if (norm.enabled()) norm.validate()`;
- default-constructed invalid norms used to mean "not present".

### Tests

- absence is represented only by `std::nullopt` where the norm is optional;
- required norms cannot be silently absent;
- weightless norms remain representable without pretending the norm is absent.

## 1.3 Give per-layer input one semantic owner

### Current problem

`PerLayerInputSpec` stores payload plus `enabled`, so disabled payload can still
alter `ModelGraph::fingerprint()`.

There is a deeper ownership problem: the current compiled derivation finds an
enabled layer and then requires every layer to carry the same input size,
activation and normalization epsilon. That means the current execution contract
is model-wide even though the same configuration is duplicated in every
`LayerSpec`.

### Target

First decide the actual semantic contract instead of only replacing `enabled`
with `optional`.

For the current uniform contract, prefer one graph-owned value such as:

```cpp
struct PerLayerInputPolicy {
    int input_size;
    ActivationKind activation;
    NormSpec norm;
};

struct ModelGraph {
    // ...
    std::optional<PerLayerInputPolicy> per_layer_input;
};
```

Then delete duplicated `LayerSpec::per_layer_input` and
`LayerSpec::per_layer_input_norm` state if they have no independent per-layer
meaning.

If investigation shows that different layers genuinely need different
per-layer-input semantics, model that variation directly instead and make the
compiled plan per-layer as well. Do not keep a per-layer representation while
the compiler requires all values to be identical.

### Delete

- `PerLayerInputSpec::enabled`;
- zero-width absence conventions;
- duplicated per-layer copies of a model-wide policy;
- fingerprint serialization of inactive or redundant per-layer input fields.

## 1.4 Remove derived state from `PerLayerInputPlan`

### Current problem

The compiled plan stores values derivable from canonical dimensions/policy,
including packed width, token scale, context scale and a constant residual
scale. The validator then has to prove that cached values still match their
inputs.

### Target

Represent absence with `std::optional<PerLayerInputPlan>` and store only
irreducible planning inputs.

Cheap deterministic values should be:

- computed by accessors from canonical fields; or
- materialized inside an immutable backend allocation/execution object whose
  constructor is the only derivation path.

### Delete

- `enabled` in the plan;
- mutable cached values that are cheap deterministic functions of canonical
  fields;
- validation whose only purpose is rechecking those derivations.

## 1.5 Replace `AttentionOutputGateSpec` tag + payload

### Current problem

`AttentionGateKind::None` can coexist with gate packing/granularity fields, and
those inactive fields participate in fingerprints.

### Target

If sigmoid remains the only actual gate algorithm:

```cpp
struct SigmoidAttentionGateSpec {
    bool packed_with_query;
    AttentionGateGranularity granularity;
};

using AttentionOutputGateSpec = std::optional<SigmoidAttentionGateSpec>;
```

If multiple real gate algorithms exist when this phase is implemented, use a
variant of those algorithms instead. Absence must still be represented once.

### Delete

- `AttentionGateKind` if it only distinguishes none vs sigmoid;
- `enabled()` helpers based on the tag;
- inactive gate fields in fingerprints.

## 1.6 Split latent projection alternatives

### Current problem

`LatentAttentionStateSpec` contains `factorized` plus fields only meaningful for
factorized projections.

### Target

Introduce an explicit projection alternative:

```cpp
struct DirectLatentProjection {};

struct FactorizedLatentProjection {
    int query_rank;
    int value_head_dim;
    NormSpec query_latent_norm;
    NormSpec key_latent_norm;
};

using LatentProjectionSpec = std::variant<
    DirectLatentProjection,
    FactorizedLatentProjection>;
```

Embed it in the latent state spec and keep common latent-state dimensions only
where both projection alternatives actually consume them.

### Delete

- `factorized`;
- factorized-only fields from the common latent-state object;
- validators whose purpose is to ensure the boolean agrees with the payload.

## 1.7 Couple attention state storage to the state alternative

### Current problem

`AttentionStateStorageSpec` carries key/value/latent/rotary/recurrent storage
simultaneously even though ordinary KV and latent attention use different
regions. The same parallel bag is then copied into compiled state layouts.

### Target

Use storage types that match the semantic state alternative:

```cpp
struct OrdinaryKvStorageSpec {
    StateScalarType key;
    StateScalarType value;
    StateQuantizationGranularity granularity;
    bool paged;
};

struct LatentStorageSpec {
    StateScalarType latent;
    StateScalarType rotary;
    StateQuantizationGranularity granularity;
    bool paged;
};
```

Prefer coupling storage directly to the corresponding state alternative so it
is impossible to pair latent semantics with ordinary-only storage fields, for
example conceptually:

```cpp
struct OrdinaryKvStateSpec {
    bool quantizable;
    OrdinaryKvStorageSpec storage;
};

struct LatentAttentionStateSpec {
    // latent semantics...
    LatentStorageSpec storage;
};
```

The compiled layout should preserve the same alternative rather than copying a
larger generic storage object into both variants.

### Investigation required

`state_storage.recurrent` currently appears to be configured/fingerprinted
without a clear attention execution consumer. Verify every consumer. If it is
not executable attention state, delete it. If recurrent mixers require an
equivalent storage policy, model that in the recurrent primitive/backend plan
where it belongs.

## 1.8 Replace KV-sharing sentinels

### Current problem

`KvSharingSpec` uses `group < 0` plus a separate publishing boolean.

### Target

Model only valid protocol states, for example:

```cpp
struct PrivateKv {};
struct SharedKvPublisher { int group; };
struct SharedKvConsumer { int group; };

using KvSharingSpec = std::variant<
    PrivateKv,
    SharedKvPublisher,
    SharedKvConsumer>;
```

Choose the exact alternatives from the actual sharing protocol; do not preserve
`group/publishes` merely because consumers currently inspect them separately.

### Delete

- negative-group sentinels;
- contradictory `group/publishes` combinations;
- helpers that rediscover the active alternative from those fields.

## 1.9 Replace attention-source sentinels

### Current problem

Attention source policy stores source tags plus `memory_slot = -1`, requiring
validation to reconcile external-memory source with the slot.

### Target

Represent the actual alternatives directly, for example:

```cpp
struct CurrentSequenceSource {};
struct ExternalMemorySource { int slot; };

using AttentionKeyValueSource = std::variant<
    CurrentSequenceSource,
    ExternalMemorySource>;
```

If query source has independently meaningful alternatives, model it separately
with its own sum type. Do not keep an unused query-source enum simply for layout
symmetry.

---

# Phase 2 — Remove backend capability leakage from resolved semantics

## 2.1 Delete backend support flags from `ModelCapabilities`

### Current problem

The resolved model currently carries CPU/CUDA support flags while backend
compilers independently inspect the semantic program and can reject unsupported
primitives. The CPU path, for example, can first trust `supports_cpu` and then
perform its own capability validation. This creates two authorities for one
decision.

### Target

The resolved model describes the model. A backend compiler/planner decides
whether and how it can execute that model by inspecting the semantic graph or
compiled program.

Do not require a specific class shape; the invariant is that capability is a
backend compilation result, not imported semantic state.

### Delete

- `supports_cpu`;
- `supports_cuda`;
- `supports_expert_offload` if it has no independent model meaning;
- compiler prechecks that trust importer/descriptor support flags before
  inspecting the program.

Keep `tied_embeddings` only if it remains a genuine checkpoint/model fact, and
move it to a semantically named location instead of leaving it mixed with
backend support flags.

### Tests

- a model is accepted/rejected based only on backend compiler capabilities;
- no importer/descriptor can falsely declare CUDA/CPU compatibility;
- adding a new semantic primitive forces relevant backend capability visitors
  to fail compilation until handled.

---

# Phase 3 — Preserve sum types through CPU physical lowering

## 3.1 Replace `CpuStatePageLayout` manual union and type its access surface

### Current problem

CPU state pages store ordinary-KV and latent widths in one struct, reintroducing
impossible combinations after the model layer has already separated them.

Changing only the layout to a variant is insufficient because `CpuKvPagePool`
also exposes ordinary and latent operations simultaneously (`write`,
`write_latent`, ordinary accessors, latent accessors). A caller can therefore
still request an operation that is impossible for the active layout.

### Target

Use explicit physical layout alternatives:

```cpp
struct CpuOrdinaryKvPageLayout { ... };
struct CpuLatentPageLayout { ... };
using CpuStatePageLayout = std::variant<
    CpuOrdinaryKvPageLayout,
    CpuLatentPageLayout>;
```

Then carry the alternative through the access API. Acceptable directions
include:

- typed ordinary/latent pool views returned from a common owner;
- a variant of typed physical pool implementations;
- internal exhaustive dispatch that exposes only operations valid for the
  selected state alternative.

The end state must not depend on "this accessor throws because the other layout
is active" as the normal discriminator.

### Delete

- zero-width fields used to identify the active physical layout;
- generic accessors whose validity depends on inactive width fields;
- validation that only proves ordinary and latent fields are not active
  simultaneously.

## 3.2 Replace `CpuAttentionPattern` manual tagged union

### Current problem

`AttentionPatternSpec` is correctly modeled as a variant, then
`CpuAttentionPattern::lower()` converts it into `kind + every payload field`.

### Target

Keep the sum type through backend lowering. Either reuse the semantic pattern
when no backend-specific lowering is necessary or create backend-specific
alternatives containing only data used by each kernel path.

### Delete

- `CpuAttentionPatternKind` if derivable from the variant;
- zero-filled inactive payload fields;
- switch-based reconciliation over a manual tagged union.

## 3.3 Replace `CpuAttentionBias` nullable pointer bag

### Current problem

ALiBi and relative-position payloads coexist in the same nullable pointer bag.
Pointer presence becomes the discriminator.

### Target

Use a backend view variant such as:

```cpp
struct CpuNoAttentionBias {};
struct CpuAlibiBiasView { std::span<const float> slopes; };
struct CpuRelativeBiasView { ... };
using CpuAttentionBias = std::variant<
    CpuNoAttentionBias,
    CpuAlibiBiasView,
    CpuRelativeBiasView>;
```

Avoid nullable pointer combinations as the semantic or physical discriminator.

---

# Phase 4 — Finish topology ownership and retire the superseded topology plan

This phase absorbs the remaining active work from
`MODEL_TOPOLOGY_OWNERSHIP_REFACTORING_PLAN.md`.

## 4.1 Make `ModelGraph` the final executable semantic source

### Invariant

Once synthesis/resolution returns a `ModelGraph`, no assembler, compiled-program
builder or backend may repair or reinterpret model semantics from checkpoint or
binding evidence.

Physical tensor evidence may be used to resolve semantics before graph
finalization, but after that boundary the flow is one-way:

```text
checkpoint / descriptor / tensor evidence
        |
        v
canonical facts
        |
        v
final ModelGraph
        |
        v
CompiledModelProgram
        |
        v
backend execution plan
```

No execution path may bypass the graph/program and reread an import-owned copy
of the same policy.

## 4.2 Move graph-derived topology to execution/planning ownership

### Current problem

`ExecutionTopology` is explicitly derived from `ModelGraph`, yet
`ResolvedModel` stores both. The topology contains layer counts, index mappings,
maxima and scratch-sizing information that can theoretically diverge from the
graph despite having a private construction path.

### Target

Preferred direction:

```text
ResolvedModel
  ├── checkpoint facts still needed after resolution
  ├── ModelGraph
  ├── weight plan / source bindings
  └── provenance

Backend compilation
  └── BackendExecutionPlan
        ├── derived execution topology
        ├── scratch maxima
        ├── backend capability decisions
        └── backend-specific state layout
```

If some topology remains genuinely backend-neutral and expensive to derive,
keep it as an immutable value constructed exclusively from `ModelGraph`. Do not
expose public mutable fields that duplicate graph semantics, and do not make
`ResolvedModel::validate()` responsible for reconciling two independently
mutable representations.

### Delete/move

- mutable graph-derived caches from `ResolvedModel`;
- validation whose only purpose is checking graph/topology agreement;
- backend reads of checkpoint/import state when graph/program semantics already
  exist;
- runtime fallbacks from missing compiled per-layer values to global checkpoint
  dimensions/policies.

### Completion

When the remaining ownership work is implemented and tests are green, delete
`docs/MODEL_TOPOLOGY_OWNERSHIP_REFACTORING_PLAN.md`. Its still-relevant
invariants belong in this document and in architecture tests/`AGENTS.md`, not in
a second roadmap.

---

# Phase 5 — Type the declarative descriptor intermediate representation

This is a large but important cleanup. The descriptor parser currently contains
many strings, booleans, sentinels and optional parallel fields that reconstruct
old invalid-state representations before eventually producing a clean
`ModelGraph`.

Do not rewrite the JSON descriptor format and the C++ IR in one step. Parse the
existing JSON into a typed internal descriptor representation first. Simplify
the external schema only in a separate change if useful.

## 5.1 Replace `Field` fallback dual representation

### Current problem

A field may contain both numeric fallback storage and fallback expression
storage, with external presence checks deciding which is meaningful.

### Target

Use:

```cpp
using FieldFallback = std::variant<
    NoFallback,
    NumericFallback,
    ExpressionFallback>;
```

or an optional variant if absence is better represented separately.

## 5.2 Replace `ProbeCondition` manual alternatives

### Current problem

Probe conditions keep `equals`, `contains`, `integer_equals`,
`has_integer_equals` and case-sensitivity side-by-side.

### Target

Use explicit predicates:

```cpp
using ProbePredicate = std::variant<
    StringEquals,
    StringContains,
    IntegerEquals,
    KeyPresent,
    ...>;
```

Place case-sensitivity only on predicates where it is meaningful.

## 5.3 Type position and RoPE descriptors

### Current problem

The descriptor IR contains `position_kind`, optional kind fields,
`rope_scaling_kind`, optional scaling-kind fields and payload fields for every
scaling family at the same time. This would recreate the exact manual union
removed in Phase 1.1.

### Target

Create descriptor-level alternatives that mirror the semantic alternatives but
store unresolved `Field` values where metadata evaluation is still required.
For example:

```cpp
struct DescriptorLinearRopeScaling { Field factor; };
struct DescriptorDynamicNtkRopeScaling { Field factor; Field original_context; };
struct DescriptorYarnRopeScaling { ... };
struct DescriptorLongRopeScaling { ... };
struct DescriptorLlama3RopeScaling { ... };

using DescriptorRopeScaling = std::variant<...>;
```

Likewise type the position alternative itself rather than keeping
`position_kind` plus M-RoPE-only payload fields in one bag.

The descriptor alternative must not contain fields that do not exist in the
corresponding semantic alternative.

## 5.4 Type norm configuration

Replace pairs such as:

```text
query_norm_enabled + query_norm_kind
key_norm_enabled + key_norm_kind
```

with optional typed norm descriptions. Preserve the distinction between absent
and weightless norms.

## 5.5 Type attention gate configuration

Replace:

```text
attention_gate_kind + attention_gate_packed_with_query
```

with the same optional/variant gate shape used by semantic construction.

## 5.6 Type attention-pattern descriptors

### Current problem

Pattern selection is spread across generic fields such as
`attention_pattern`, `sliding_window`, `sliding_pattern_value` and attention
variants/schedules.

### Target

Represent closed pattern alternatives explicitly. Metadata-driven schedules may
remain descriptor logic, but once one layer's pattern is resolved it should be
constructed directly as an `AttentionPatternSpec` alternative rather than via
string tag plus parallel payload.

## 5.7 Type attention state and storage descriptors together

Replace:

```text
attention_state_kind
+ state_key_storage
+ state_value_storage
+ state_latent_storage
+ state_rotary_storage
+ state_recurrent_storage
+ state_storage_granularity
+ state_paged
+ latent-only fields
```

with descriptor alternatives matching the semantic state/storage coupling from
Phase 1.7.

Ordinary-KV descriptor state must not be able to carry latent-only storage or
latent projection fields. Latent state must not carry ordinary key/value
storage. Recurrent storage must be removed from attention descriptors if the
Phase 1.7 investigation finds no attention consumer.

## 5.8 Type attention source descriptors

Replace:

```text
attention_key_value_source
+ attention_memory_slot optional
```

with explicit current-sequence/external-memory alternatives.

## 5.9 Type repeated-layer schedules

Replace:

```text
repeated_layers bool
+ repeat_count Field
```

with a schedule type where repeat count only exists for a repeated schedule.
Apply the same rule to other schedule/tag pairs where payload validity depends
on a separate boolean or string discriminator.

## 5.10 Construct MoE selection directly

### Current problem

Descriptor resolution extracts grouped-routing fields into zeros and later uses
`routing_group_count > 0` to choose `MoeGroupedTopKSelectionSpec`.

### Target

Construct `MoeSelectionSpec` directly from a typed descriptor selection
alternative. No intermediate "zero means ordinary top-k" state.

Use designated initialization or typed constructors for large semantic structs
so field-order changes cannot silently shift convertible values into the wrong
members.

---

# Phase 6 — Type automatic inference working state

## 6.1 Replace `CanonicalInferenceContext` MoE field cluster

### Current problem

The context stores `has_moe`, expert counts, routed/shared widths,
grouped-routing dimensions, routing booleans and scales as independent fields
with sentinel zero values.

### Target

Use one optional canonical MoE value:

```cpp
struct CanonicalMoeFacts {
    int num_experts;
    int experts_per_token;
    int intermediate_size;
    MoeSelectionSpec selection;
    std::optional<CanonicalSharedExpertFacts> shared;
    bool normalize_topk;
    bool use_expert_bias;
    float routed_scaling_factor;
    MoeRouterScoreKind score;
};

std::optional<CanonicalMoeFacts> moe;
```

Exact types may be inference-specific, but they should mirror semantic
alternatives rather than reconstructing them from bool/sentinel fields later.

## 6.2 Audit other inference metadata clusters

Apply the same rule to recurrent, Mamba and latent-attention inference facts:

- group fields that are only meaningful together;
- use optional structured facts for optional primitives;
- avoid `.value_or(0)` as an implicit semantic discriminator;
- distinguish "metadata absent, can infer" from "primitive absent";
- construct final semantic variants directly rather than passing through
  another tag/payload bag.

---

# Phase 7 — Remove stored self-referential facade views

This phase is lower risk and can be performed independently after the semantic
cuts, but lifetime safety is part of the target.

## 7.1 CPU model views

### Current problem

`CpuModel` stores `CpuInferenceSession`, `CpuDiagnostics` and `CpuPersistence`,
each containing `CpuModel* owner_`. This forces manual move logic to rebind the
self-referential views.

### Target

Do not store non-owning self views as model state.

Returning lightweight values on demand is acceptable only if their lifetime is
safe across model moves. A view obtained before moving a `CpuModel` must not
remain bound to a moved-from facade.

Preferred options include:

- make the ephemeral view explicitly non-storable/non-surviving across owner
  moves by API design; or
- anchor the view to stable heap-owned compiled/session state whose address does
  not change when the facade's `unique_ptr` is moved.

Conceptually:

```cpp
CpuInferenceSession session();
CpuDiagnostics diagnostics();
CpuPersistence persistence();
```

The implementation should not require a member whose pointer targets `this`.

### Delete

- stored `*_view_` members;
- manual move logic whose only purpose is rebinding those views;
- lifetime assumptions that depend on callers never retaining a view across a
  legal owner move.

## 7.2 CUDA model views

Apply the same ownership rule to CUDA. Reassess whether `CudaModel` can become
movable after stored self views disappear, but do not add move support merely
for symmetry. CUDA resource ownership may independently require a stable facade
address.

---

# Phase 8 — Collapse packed execution compatibility state

## 8.1 Reduce `PackedCompatibilityKey`

### Current problem

The key stores `execution_plan_fingerprint` and then repeats many of the same
plan inputs individually: modes, booleans, context size, tuning parameters and
device choices.

### Target

A compatibility key should contain independent identities exactly once.

Candidate shape:

```cpp
struct PackedCompatibilityKey {
    const SharedModelWeights* weights_identity;
    uint64_t execution_plan_fingerprint;
    uint64_t compiled_program_id;
    // only independent identities not already covered above
};
```

Do not delete fields until tests document exactly what the plan/program
fingerprints guarantee. `expert_residency_fingerprint`, device identity or any
other field should remain only if it is demonstrably independent of the
existing identities.

### Tests

Mutate every execution-relevant input and prove that the appropriate
plan/program/compatibility identity changes. Then remove duplicated fields.

## 8.2 Make packed-session requirements valid by construction

### Current problem

`PackedSessionContext` has been split into sub-objects, but those sub-objects are
still largely default-constructible pointer bags. Methods dereference required
fields directly, while some services use one pointer as a proxy for overall
validity.

### Target

Split required and optional dependencies and remove default-invalid
construction for operation contexts.

Use references, `std::reference_wrapper`, a non-null wrapper,
constructors/factories or tightly scoped operation contexts so required
resources cannot be absent after construction.

Do not replace the current pointer bag with another generic service locator.

Separate contexts by operation only when requirements materially differ, for
example:

```text
PackedDecodeContext
PackedPrefillContext
PackedExpertResidencyContext
```

Optional facilities should be optional because the operation can genuinely
proceed without them, not because construction is incomplete.

---

# Phase 9 — Final semantic fingerprint hardening

Phase 0 establishes protection before refactoring. This phase is the final
repository-wide identity audit after all structural changes have landed.

## Objective

A semantic fingerprint changes if and only if execution semantics relevant to
that identity change.

## Audit targets

- RoPE scaling alternatives;
- optional norms;
- per-layer input policy;
- attention gates;
- latent projection mode;
- state storage actually used by each state alternative;
- KV sharing;
- attention source;
- MoE selection/shared expert;
- any remaining default/sentinel fields.

## Required tests

For each sum type:

1. changing an active semantic field changes the fingerprint;
2. no inactive payload exists to mutate;
3. reconstructing the same semantics through different import paths yields the
   same semantic fingerprint where provenance is intentionally excluded;
4. provenance/source identity remains separate from semantic identity where the
   distinction matters;
5. fingerprint visitors are compile-time exhaustive.

Also audit the relation among:

- `ModelGraph::fingerprint()`;
- `program_semantic_fingerprint()` or its current equivalent;
- `CompiledModelProgram::semantic_fingerprint`;
- CUDA execution-plan fingerprint;
- packed compatibility identity.

Each identity must have a documented responsibility. Do not keep a second
fingerprint merely because an older layer already exposes one.

---

# Phase 10 — Cleanup and architecture guardrails

## Delete obsolete artifacts

After all consumers migrate, search and delete:

- removed enum/tag names;
- `enabled` methods used as optional sentinels;
- `*_kind()` helpers that merely rediscover a variant alternative;
- `.value_or(0)` uses that discriminate semantic alternatives;
- zero/negative sentinel comments;
- compatibility aliases/views introduced for old internal APIs;
- validators that only reconcile duplicate state;
- fingerprint fields that are no longer semantic state;
- the superseded topology roadmap after Phase 4 is complete.

## Add architecture tests/checks

Extend architecture-boundary checks where practical to catch regressions such
as:

- semantic model headers depending on CPU/CUDA backend headers;
- backend support booleans added to `ResolvedModel`/`ModelGraph`;
- new internal proxy/view compatibility layers around variants;
- manual `enum kind + parallel payload` structures for closed alternatives
  where a sum type is appropriate;
- semantic visitors with default/fallthrough handling instead of exhaustive
  alternatives;
- runtime fallbacks from compiled semantic state to checkpoint-wide defaults;
- fingerprints serializing fields that are inactive for the selected semantic
  alternative.

Static scripts cannot prove all of these mechanically. Encode the most
important rules in focused compile/unit tests and keep the architectural rules
in `AGENTS.md` once the refactor is stable.

---

# Recommended implementation order

Use the following commit-sized sequence:

1. `test: lock semantic identity and exhaustive visitor guardrails`
2. `refactor: model rope scaling as minimal explicit alternatives`
3. `refactor: make optional norms explicit`
4. `refactor: give per-layer input one semantic owner`
5. `refactor: simplify compiled per-layer input planning`
6. `refactor: model attention gate and latent projection alternatives`
7. `refactor: couple attention state and storage alternatives`
8. `refactor: model kv sharing and attention sources explicitly`
9. `refactor: move backend capability decisions out of resolved model`
10. `refactor: preserve sum types through cpu physical attention state`
11. `refactor: preserve sum types in cpu pattern and bias lowering`
12. `refactor: finish execution topology ownership`
13. `refactor: type descriptor field and probe alternatives`
14. `refactor: type descriptor position rope and attention alternatives`
15. `refactor: type descriptor schedules norms and moe semantics`
16. `refactor: type automatic inference working state`
17. `refactor: remove stored self-referential model views`
18. `refactor: collapse packed compatibility duplication`
19. `refactor: make packed contexts valid by construction`
20. `test: complete semantic fingerprint identity audit`
21. `chore: remove obsolete semantic compatibility code and superseded plan`

The order is deliberate: protect identity first; make the semantic graph
truthful; preserve that truth through backend lowering; finish ownership of
derived planning state; then type construction intermediates and simplify
runtime facades/packed infrastructure. The final fingerprint step is an audit,
not the first line of defense.

---

# Definition of done

This cleanup is complete when all of the following are true:

- `ModelGraph` uses explicit alternatives/optionals instead of booleans or
  numeric sentinels for presence and closed semantic choices.
- Each semantic alternative stores only fields meaningful to that alternative.
- Per-layer input has one semantic owner rather than N identical layer copies
  plus a model-wide compiled reconstruction.
- CPU lowering does not convert semantic `std::variant` values back into manual
  tagged unions or expose ordinary/latent physical operations through one
  invalid-state-dependent API.
- `ResolvedModel` does not declare whether CPU/CUDA can execute it.
- `ModelGraph`/compiled program are the final source of executable semantics;
  backends do not reread import-owned duplicates.
- Graph-derived runtime maxima/counts are owned by execution/planning rather
  than duplicated mutable semantic state.
- Descriptor and automatic-inference intermediates cannot represent
  contradictory RoPE, norm, MoE, attention-state, storage, source or schedule
  alternatives through independent flags/strings/zero values.
- Every semantic variant visitor involved in lowering, validation, allocation,
  capability analysis or fingerprinting is compile-time exhaustive.
- Stored facade views do not force rule-of-five code or create dangling
  self-pointers across legal owner moves.
- Packed execution compatibility information has one source of truth for each
  independent plan/program/resource identity.
- Required packed-session dependencies are non-null by construction.
- Semantic fingerprints include only active executable semantics.
- Runtime code contains no fallback from missing compiled per-layer semantics
  to a checkpoint-wide default.
- `MODEL_TOPOLOGY_OWNERSHIP_REFACTORING_PLAN.md` has been retired after its
  remaining invariants are implemented here.
- A repository-wide search finds no internal compatibility shim whose only
  purpose is preserving one of the representations removed by this plan.
