# Semantic State Cleanup Plan

## Goal

Continue the model/runtime cleanup started by the explicit-sum-types refactor and remove the remaining cases where CELEG stores multiple representations of the same semantic choice.

The governing rule is:

> If `validate()` mainly exists to prove that a tag, boolean, sentinel, optional, derived cache, or parallel payload describes the same alternative as another field, the representation is wrong. Prefer `std::variant`, `std::optional`, a dedicated value type, or a backend-owned execution plan so invalid states are unrepresentable.

This plan is intentionally internal. CELEG must not preserve obsolete internal APIs for compatibility. When a type changes, migrate its consumers and delete the old representation rather than adding adapters, views, aliases, fallback fields, or compatibility shims.

## Non-goals

- No public compatibility layer for internal C++ implementation types.
- No architecture-name switches in backends.
- No backend capability flags added to semantic model objects.
- No duplicated semantic fingerprints whose inputs can diverge from the actual executable state.
- No broad rewrite unrelated to the state-modeling problems listed below.

## Invariants for every phase

Each phase is complete only when:

1. impossible combinations are no longer constructible through the normal type surface;
2. dead/inactive payload no longer participates in semantic fingerprints;
3. derived values have one source of truth;
4. backend-specific capability decisions live in backend compilation/planning;
5. all consumers use the new representation directly;
6. old enums, sentinels, boolean enable flags, proxy views, adapters, and fallback compatibility paths are deleted;
7. tests cover the valid alternatives and the removed ambiguity;
8. CPU and CUDA build/tests remain green after each commit-sized cut.

---

# Phase 1 — Finish semantic sum types in `model/`

This is the highest-priority phase because these types participate directly in `ModelGraph::fingerprint()` and therefore in the compiled semantic identity.

## 1.1 Replace `RopeScalingSpec` tagged union

### Current problem

`RopeScalingSpec` stores a `RopeScalingKind` plus the payloads for every scaling family at the same time:

- generic factor/original context;
- YaRN beta fields;
- Llama-3 frequency factors;
- LongRoPE short/long factor vectors.

Inactive fields can be populated and currently participate in the semantic fingerprint.

### Target

Introduce one type per actual scaling alternative, for example:

```cpp
struct NoRopeScaling {};

struct LinearRopeScaling {
    double factor;
    int original_context;
};

struct DynamicNtkRopeScaling {
    double factor;
    int original_context;
};

struct YarnRopeScaling {
    double factor;
    int original_context;
    double attention_factor;
    double beta_fast;
    double beta_slow;
};

struct LongRopeScaling {
    double factor;
    int original_context;
    double attention_factor;
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

Exact field placement should follow execution semantics rather than compatibility with the old layout.

### Delete

- `RopeScalingKind` if it becomes fully derivable from the variant;
- all switch/tag reconciliation logic;
- serialization of inactive scaling fields.

### Tests

- equal executable scaling semantics produce equal fingerprints;
- changing a field that does not exist in an alternative is impossible;
- each scaling alternative validates independently.

## 1.2 Make optional norms actually optional

### Current problem

`NormSpec::enabled()` treats `epsilon <= 0` as absence while `NormSpec::validate()` treats the same value as invalid. This uses an invalid object as an optional/sentinel value.

### Target

A `NormSpec` represents an existing norm and is always valid after construction/validation.

Use `std::optional<NormSpec>` for semantically optional norms, including the appropriate layer normalization positions.

Do not conflate:

- norm absent;
- norm present but weightless (`NormWeightKind::None`).

### Delete

- `NormSpec::enabled()`;
- zero-epsilon sentinel construction;
- guards of the form `if (norm.enabled()) norm.validate()`.

### Tests

- absence is represented only by `std::nullopt`;
- weightless norms remain representable without pretending the norm is absent.

## 1.3 Replace `PerLayerInputSpec::enabled`

### Current problem

`PerLayerInputSpec` stores payload plus `enabled`. Disabled payload can still alter `ModelGraph::fingerprint()`.

### Target

Use:

```cpp
std::optional<PerLayerInputSpec> per_layer_input;
```

where the contained value has only meaningful fields such as input width and activation.

### Delete

- `enabled`;
- zero-width absence convention;
- fingerprint serialization of inactive per-layer input fields.

## 1.4 Remove derived state from `PerLayerInputPlan`

### Current problem

The plan stores values derivable from canonical dimensions/policy, including candidates such as:

- packed width;
- token scale;
- context scale;
- constant residual scale.

The validator cannot guarantee all cached values still equal the derivation.

### Target

Store only irreducible backend/runtime planning inputs. Make straightforward derivations functions/accessors or compute them once into an immutable backend allocation object that cannot be partially modified.

Prefer:

```cpp
std::optional<PerLayerInputPlan>
```

rather than an `enabled` bit on the compiled plan.

### Delete

- `enabled` in the plan;
- cached values that are cheap deterministic functions of canonical fields.

## 1.5 Replace `AttentionOutputGateSpec` tag + payload

### Current problem

`AttentionGateKind::None` can coexist with gate packing/granularity fields, and those inactive fields participate in fingerprints.

### Target

If sigmoid remains the only real gate:

```cpp
struct SigmoidAttentionGateSpec {
    bool packed_with_query;
    AttentionGateGranularity granularity;
};

using AttentionOutputGateSpec = std::optional<SigmoidAttentionGateSpec>;
```

If more gate algorithms are expected imminently, use a `std::variant` instead.

### Delete

- `AttentionGateKind` if it only distinguishes none vs sigmoid;
- `enabled()` helpers based on the tag;
- inactive gate fields in fingerprints.

## 1.6 Split latent projection alternatives

### Current problem

`LatentAttentionStateSpec` contains `factorized` plus fields only meaningful for factorized projections.

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

and embed it in the latent state spec.

### Delete

- `factorized`;
- factorized-only fields from the common latent-state object;
- validators whose purpose is to ensure the boolean agrees with the payload.

## 1.7 Split attention state storage by state alternative

### Current problem

`AttentionStateStorageSpec` carries key/value/latent/rotary/recurrent storage simultaneously, even though ordinary-KV and latent attention need different regions.

### Target

Use storage types that match the state alternative, for example:

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

Attach the correct storage to the corresponding state/layout representation.

### Investigation required

`state_storage.recurrent` currently appears to be configured/fingerprinted without a clear execution consumer. Verify every consumer. If it is not executable attention state, delete it rather than preserving it for compatibility. If recurrent mixers require equivalent storage policy, model that in the recurrent primitive/backend plan where it belongs.

## 1.8 Replace KV-sharing sentinels

### Current problem

`KvSharingSpec` uses values such as `group < 0` plus a separate publishing boolean.

### Target

Model only valid states, for example:

```cpp
struct PrivateKv {};
struct SharedKvPublisher { int group; };
struct SharedKvConsumer { int group; };

using KvSharingSpec = std::variant<
    PrivateKv,
    SharedKvPublisher,
    SharedKvConsumer>;
```

Choose exact alternatives according to the actual sharing protocol.

### Delete

- negative-group sentinels;
- contradictory `group/publishes` combinations.

## 1.9 Replace attention-source sentinels

### Current problem

Attention source policy stores source tags plus `memory_slot = -1`, requiring validation to reconcile external-memory source with the slot.

### Target

Represent the actual alternatives directly, for example:

```cpp
struct CurrentSequenceSource {};
struct ExternalMemorySource { int slot; };

using AttentionKeyValueSource = std::variant<
    CurrentSequenceSource,
    ExternalMemorySource>;
```

If query source has independently meaningful alternatives, model it separately with its own sum type.

---

# Phase 2 — Remove backend capability leakage from resolved semantics

## 2.1 Delete backend support flags from `ModelCapabilities`

### Current problem

The resolved model currently carries flags such as CPU/CUDA support, while each backend compiler independently inspects the semantic program and can reject unsupported primitives. This creates two authorities for backend support.

### Target

The resolved model describes the model. A backend compiler/planner decides whether and how it can execute it.

Move capability decisions to objects/functions such as:

```cpp
CpuExecutionPlan CpuModelCompiler::compile(...);
CudaExecutionPlan CudaModelCompiler::compile(...);
```

or typed backend capability validation during compilation.

### Delete

- `supports_cpu`;
- `supports_cuda`;
- `supports_expert_offload` if it has no independent semantic meaning;
- compiler prechecks that trust those flags before inspecting the program.

Keep `tied_embeddings` only if it remains a genuine checkpoint/model fact, and move it to a semantically named location rather than leaving it mixed with backend support flags.

### Tests

- a model is accepted/rejected based only on backend compiler capabilities;
- no importer/descriptor can falsely declare CUDA/CPU compatibility.

---

# Phase 3 — Stop CPU lowering from recreating manual tagged unions

## 3.1 `CpuStatePageLayout`

### Current problem

CPU state pages store ordinary-KV and latent widths in one struct, reintroducing impossible combinations after the model layer has already separated them.

### Target

Use explicit physical layout alternatives:

```cpp
struct CpuOrdinaryKvPageLayout { ... };
struct CpuLatentPageLayout { ... };
using CpuStatePageLayout = std::variant<...>;
```

Page allocation/access should visit the physical layout rather than inspect zero fields.

## 3.2 `CpuAttentionPattern`

### Current problem

`AttentionPatternSpec` is correctly modeled as a variant, then `CpuAttentionPattern::lower()` converts it into `kind + every payload field`.

### Target

Keep the sum type through backend lowering. Either reuse the semantic pattern when no backend-specific lowering is necessary or create backend-specific alternatives with only the data used by each kernel path.

### Delete

- `CpuAttentionPatternKind` if derivable from the variant;
- zero-filled inactive payload fields;
- switch-based reconciliation over a manual tagged union.

## 3.3 `CpuAttentionBias`

### Current problem

ALiBi and relative-position payloads coexist in the same nullable pointer bag.

### Target

Use a backend view variant such as:

```cpp
struct CpuNoAttentionBias {};
struct CpuAlibiBiasView { std::span<const float> slopes; };
struct CpuRelativeBiasView { ... };
using CpuAttentionBias = std::variant<...>;
```

Avoid nullable pointer combinations as the discriminator.

---

# Phase 4 — Fix derived topology ownership

## 4.1 Re-evaluate `ExecutionTopology`

### Current problem

`ExecutionTopology` is explicitly derived from `ModelGraph`, yet `ResolvedModel` stores both. The derived topology contains layer counts, index mappings, maxima and scratch-sizing information that can theoretically diverge from the graph.

### Target

Separate semantic resolution from execution/allocation planning.

Preferred direction:

```text
ResolvedModel
  ├── checkpoint facts
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

If some topology is genuinely backend-neutral and expensive to derive, encapsulate it as an immutable value constructed exclusively from the graph, not public mutable duplicated fields.

### Delete/move

- mutable graph-derived caches from `ResolvedModel`;
- validation whose only purpose is checking graph/topology agreement.

### Migration note

Do this after Phases 1–3 so the derivation consumes stable semantic sum types.

---

# Phase 5 — Type the declarative descriptor intermediate representation

This is a large but important cleanup. The descriptor parser currently contains many strings, booleans, sentinels and optional parallel fields that reconstruct old invalid-state representations before eventually producing a clean `ModelGraph`.

Do not rewrite the JSON descriptor format and the C++ IR in one step. First parse existing JSON into a typed internal descriptor representation, then simplify the external schema separately if useful.

## 5.1 Replace `Field` fallback dual representation

### Current problem

A field may contain both numeric fallback storage and fallback expression storage, with external presence checks deciding which is meaningful.

### Target

Use:

```cpp
using FieldFallback = std::variant<
    NoFallback,
    NumericFallback,
    ExpressionFallback>;
```

or `std::optional<std::variant<...>>`.

## 5.2 Replace `ProbeCondition` manual alternatives

### Current problem

Probe conditions keep `equals`, `contains`, `integer_equals`, and `has_integer_equals` side-by-side.

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

## 5.3 Type norm configuration

Replace pairs such as:

```text
query_norm_enabled + query_norm_kind
key_norm_enabled + key_norm_kind
```

with optional typed norm descriptions.

## 5.4 Type attention gate configuration

Replace:

```text
attention_gate_kind + attention_gate_packed_with_query
```

with the same semantic gate alternative used by the model builder.

## 5.5 Type attention state descriptors

Replace:

```text
attention_state_kind string
+ latent_rank optional
+ latent_rope_head_dim optional
+ latent_nope_head_dim optional
+ latent_decoupled_rope bool
```

with a descriptor-level variant that cannot express latent-only fields for ordinary KV.

## 5.6 Type attention source descriptors

Replace:

```text
attention_key_value_source string
+ attention_memory_slot optional
```

with explicit current-sequence/external-memory alternatives.

## 5.7 Type repeated-layer schedule

Replace:

```text
repeated_layers bool
+ repeat_count Field
```

with a schedule type where the repeat count only exists for a repeated schedule.

## 5.8 Stop reconstructing MoE selection with numeric sentinels

### Current problem

Descriptor resolution currently extracts grouped-routing fields into zeros and later uses `routing_group_count > 0` to choose `MoeGroupedTopKSelectionSpec`.

### Target

Construct `MoeSelectionSpec` directly from the descriptor representation. No intermediate "zero means top-k" state.

---

# Phase 6 — Type automatic inference working state

## 6.1 Replace `CanonicalInferenceContext` MoE field cluster

### Current problem

The context stores:

- `has_moe`;
- expert counts;
- routed intermediate width;
- shared expert width;
- grouped-routing dimensions;
- routing booleans/scales;

as independent fields with sentinel zero values.

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

Exact types may be inference-specific, but they should mirror semantic alternatives rather than reconstructing them from bool/sentinel fields later.

## 6.2 Audit other metadata clusters

Apply the same rule to recurrent, Mamba and latent-attention inference facts:

- group fields that are only meaningful together;
- use optional structured facts for optional primitives;
- avoid `.value_or(0)` as an implicit semantic discriminator;
- distinguish "metadata absent, can infer" from "primitive absent".

---

# Phase 7 — Simplify stored facade views

This phase is lower risk and can be performed independently after the semantic cuts.

## 7.1 CPU model views

### Current problem

`CpuModel` stores `CpuInferenceSession`, `CpuDiagnostics` and `CpuPersistence`, each containing `CpuModel* owner_`. This forces manual move construction to recreate the self-referential views.

### Target

Do not store non-owning self views as model state.

Prefer lightweight values returned on demand:

```cpp
CpuInferenceSession session() { return CpuInferenceSession{*this}; }
CpuDiagnostics diagnostics() { return CpuDiagnostics{*this}; }
CpuPersistence persistence() { return CpuPersistence{*this}; }
```

or another non-self-referential interface that preserves focused APIs.

### Delete

- stored `*_view_` members;
- manual move logic whose only purpose is rebinding those views.

## 7.2 CUDA model views

Apply the same change to CUDA. Reassess whether `CudaModel` can become movable after the stored self views disappear; do not add move support merely for symmetry if CUDA resource ownership has a separate reason to prohibit it.

---

# Phase 8 — Collapse packed execution compatibility state

## 8.1 Reduce `PackedCompatibilityKey`

### Current problem

The key stores `execution_plan_fingerprint` and then repeats many of the same plan inputs individually: modes, booleans, context size, tuning parameters and device choices.

### Target

A compatibility key should contain independent identities exactly once.

Candidate shape:

```cpp
struct PackedCompatibilityKey {
    const SharedModelWeights* weights_identity;
    uint64_t execution_plan_fingerprint;
    uint64_t compiled_program_id;
    // only identities not already covered above
};
```

Before deleting fields, document and test exactly what each fingerprint guarantees.

### Tests

Create compatibility tests that mutate every execution-relevant input and prove that the plan/program identity changes appropriately. Then remove the duplicated fields.

## 8.2 Make `PackedSessionContext` requirements non-null by construction

### Current problem

The context is a large default-constructible pointer bag. Methods dereference required fields without a single construction invariant, while some services test only one pointer as an approximation of validity.

### Target

Split required and optional dependencies.

Use references, `std::reference_wrapper`, a non-null wrapper, constructors/factories, or tightly scoped operation contexts so required resources cannot be absent after construction.

Do not replace the current pointer bag with another generic service locator.

### Suggested direction

Separate contexts by operation if their requirements materially differ:

```text
PackedDecodeContext
PackedPrefillContext
PackedExpertResidencyContext
```

Only do this where it reduces the dependency surface; avoid gratuitous class proliferation.

---

# Phase 9 — Semantic fingerprint hardening

Run this after the structural changes above, but add regression tests as each type is migrated.

## Objective

A semantic fingerprint changes if and only if execution semantics relevant to compiled model identity change.

## Audit targets

- RoPE scaling alternatives;
- optional norms;
- per-layer input;
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
3. reconstructing the same semantics through different import paths yields the same semantic fingerprint where provenance is intentionally excluded;
4. provenance/source identity remains separate from semantic identity where the distinction matters.

Also audit the relation among:

- `ModelGraph::fingerprint()`;
- `program_semantic_fingerprint()`;
- `CompiledModelProgram::semantic_fingerprint`;
- CUDA execution-plan fingerprint;
- packed compatibility identity.

Each should have a documented responsibility and should not duplicate another identity without a clear reason.

---

# Phase 10 — Cleanup and architecture guardrails

## Delete obsolete artifacts

After all consumers migrate, search and delete:

- removed enum/tag names;
- `enabled` methods used as optional sentinels;
- `*_kind()` helpers for variant alternatives;
- `value_or(0)` uses that discriminate semantic alternatives;
- zero/negative sentinel comments;
- compatibility aliases/views introduced for old internal APIs;
- validators that only reconcile duplicate state;
- fingerprint fields that are no longer semantic state.

## Add architecture tests/checks

Extend architecture-boundary checks where practical to catch regressions such as:

- semantic model headers depending on CPU/CUDA backend headers;
- backend support booleans added to `ResolvedModel`/`ModelGraph`;
- new internal proxy/view compatibility layers around variants;
- manual `enum kind + parallel payload` structures for closed alternatives where a sum type is appropriate.

Static scripts cannot prove all of these mechanically, so encode the most important rules in focused compile/unit tests and document the rest in `AGENTS.md` if not already covered.

---

# Recommended implementation order

Use the following commit-sized sequence:

1. `refactor: model rope scaling as explicit alternatives`
2. `refactor: make optional norms explicit`
3. `refactor: make per-layer input optional`
4. `refactor: model attention gate and latent projection alternatives`
5. `refactor: specialize attention state storage`
6. `refactor: model kv sharing and attention sources explicitly`
7. `refactor: move backend capability decisions out of resolved model`
8. `refactor: preserve sum types in cpu attention lowering`
9. `refactor: move derived execution topology into planning`
10. `refactor: type descriptor condition and field alternatives`
11. `refactor: type descriptor attention and moe semantics`
12. `refactor: type automatic inference moe state`
13. `refactor: remove stored self facade views`
14. `refactor: collapse packed compatibility duplication`
15. `refactor: make packed contexts valid by construction`
16. `test: harden semantic fingerprint invariants`
17. `chore: remove obsolete semantic compatibility code`

The order is deliberate: first make the semantic graph truthful, then keep that truth through backend lowering, then remove derived/cache duplication, and only afterward simplify construction/facade infrastructure.

---

# Definition of done

This cleanup is complete when the following statements are true:

- `ModelGraph` uses explicit alternatives/optionals instead of booleans or numeric sentinels for presence and closed semantic choices.
- CPU lowering does not convert semantic `std::variant` values back into manual tagged unions.
- `ResolvedModel` does not declare whether CPU/CUDA can execute it.
- Graph-derived runtime maxima/counts are owned by an execution/planning layer rather than duplicated mutable semantic state.
- Descriptor and automatic-inference intermediates cannot represent contradictory MoE/attention alternatives through independent flags and zero values.
- Stored facade views do not force rule-of-five code merely to rebind pointers to their owner.
- Packed execution compatibility information has one source of truth for plan/program identity.
- Required packed-session dependencies are non-null by construction.
- Semantic fingerprints include only active executable semantics.
- A repository-wide search finds no internal compatibility shim whose only purpose is preserving one of the representations removed by this plan.
