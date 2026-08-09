# Descriptorless Checkpoint Inference Refactoring Plan

## Status

Planned architectural refactor.

Baseline: `master` after the GPT-X2.5 semantic work, including generic attention-output transforms, adjacent-pair RoPE, CPU/CUDA coverage, declarative GPT-X2.5 resolution tests, and the official-reference parity harness.

## Objective

Allow CELEG to load a previously unknown Hugging Face-style checkpoint **without a model-specific descriptor** when the checkpoint can be mapped completely and unambiguously onto CELEG's existing backend-neutral semantic vocabulary.

A descriptor remains an explicit fallback/override for checkpoints whose semantics cannot be inferred safely. It must no longer be mandatory for otherwise conventional architectures.

The first acceptance model is `AxiomicLabs/GPT-X2.5-135M`: after this refactor, deleting `descriptors/gptx25.json` must not change its resolved semantics or numerical behavior.

The final architecture must preserve the current execution boundary: inference occurs once at checkpoint load time and produces the same immutable backend-neutral `ResolvedModel`/compiled program consumed by CPU and CUDA today.

---

# 1. Non-negotiable design rules

## 1.1 Evidence, never guessing

Auto-discovery must not mean heuristic execution.

CELEG may execute an automatically resolved checkpoint only when every required structural fact, semantic policy, and tensor binding is supported by explicit evidence and passes the validation boundary responsible for it.

Examples:

- metadata says `hidden_size = 576`;
- token-embedding and projection shapes independently agree with width 576;
- exactly 30 layer indices exist and metadata also reports 30 layers;
- Q/K/V geometry agrees with 9 query heads, 3 KV heads, and head dimension 64;
- `xsa_projection = true` maps through a registered semantic rule to `OrthogonalizeCurrentValueSpec`;
- RoPE pairing is resolved as a semantic fact, never from a family-name branch;
- tied embeddings are accepted only when binding evidence is consistent.

Missing, contradictory, unsupported, or ambiguous evidence is a load-time failure. There is no `best effort` runtime mode.

## 1.2 Rules propose; solvers decide

Inference rules must be **pure proposal producers**. They must not mutate shared resolution state and must not depend on rule execution order.

Forbidden shape:

```cpp
rule.apply(mutable_context);
```

Required direction:

```cpp
InferenceProposal rule.evaluate(const InferenceInput& input) const;
```

Conceptually:

```text
rule A --- proposal + evidence ---+
rule B --- proposal + evidence ---+--> FactSolver --> CanonicalModelFacts
rule C --- proposal + evidence ---+
```

A rule may state what it observed and what fact/binding it proposes. Only a solver/validator may accept, reject, combine, or report conflict between proposals.

This makes rule order irrelevant and prevents a procedural `GenericHfResolver` from reappearing behind interfaces.

## 1.3 One canonical boundary before semantic synthesis

Both descriptor resolution and automatic inference must converge on one immutable, backend-neutral representation:

```text
CanonicalModelFacts
```

No descriptor-only graph builder and no auto-only graph builder may survive long term.

## 1.4 Backend support is decided by the backend

Checkpoint inference must describe **required semantics**, not claim that a checkpoint “supports CPU” or “supports CUDA”.

The inference side may derive requirements such as:

```text
OrdinaryKv
GQA
AdjacentPairRoPE
OrthogonalizeCurrentValue
SwiGLU
```

Each backend compiler compares those requirements with its own capabilities and accepts or rejects the model.

The checkpoint layer must not know CUDA kernels, CPU ISA details, offload implementation, or backend availability.

## 1.5 Explainability observes; it does not control resolution

`ResolutionReport` is generated from evidence, proposals, solver decisions, and failures. The report must not participate in deciding them.

The same inference result must be produced with reporting disabled.

---

# 2. Non-goals

This refactor must **not**:

- put architecture IDs or `model_type` branches in CPU/CUDA execution code;
- infer new mathematics from arbitrary metadata names;
- silently approximate an unsupported primitive;
- introduce decode-time or token-time model probing;
- create one giant `GenericHfResolver` full of family logic;
- make rule order semantically significant;
- let rules mutate a shared inference context;
- use model/repository names as confidence signals;
- make `GraphSynthesizer` responsible for graph, weight planning, backend capabilities, and provenance at once;
- create one generic `IInferenceValidator` that gradually owns every validation concern;
- remove descriptors before their replacement path has golden and numerical coverage;
- preserve obsolete descriptor-only access paths merely for backward compatibility.

---

# 3. Desired architecture

```text
                            CheckpointView
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
           MetadataNormalizer          TensorInventoryBuilder
                    |                           |
                    +-------------+-------------+
                                  |
                                  v
                           InferenceInput
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
        Metadata/Semantic Rules          Tensor Binding Rules
                    |                           |
                    +------- proposals ---------+
                                  |
                                  v
                              FactSolver
                                  |
                                  v
                        CanonicalModelFacts
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
          GraphSynthesizer                WeightPlanSynthesizer
                 |                                 |
                 +----------------+----------------+
                                  |
                                  v
                         ResolutionAssembler
                                  |
                                  v
                            ResolvedModel
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
              CPU compiler                 CUDA compiler
```

Evidence flows alongside this pipeline into `ResolutionReport`.

No inference or synthesis stage may depend on a backend.

---

# 4. Core neutral data model

## 4.1 `NormalizedModelMetadata`

A canonical view of common checkpoint configuration facts. It contains normalized observations, not execution choices.

Example:

```cpp
struct NormalizedModelMetadata {
    std::optional<int> hidden_size;
    std::optional<int> intermediate_size;
    std::optional<int> layer_count;
    std::optional<int> query_heads;
    std::optional<int> key_value_heads;
    std::optional<int> head_dim;
    std::optional<int> vocab_size;
    std::optional<int> context_length;
    std::optional<float> norm_epsilon;
    std::optional<double> rope_theta;
    std::optional<bool> tied_embeddings;
    TokenIds tokens;
};
```

Aliases such as `hidden_size`, `n_embd`, and `d_model` may feed this view, but conflicts must remain visible to validation rather than being silently overwritten.

## 4.2 `TensorInventory`

An immutable index over tensor names, dimensions, dtype, shard location, aliases, and stable tensor identity.

It supports queries such as:

```text
all tensors matching a layer-indexed naming grammar
all rank-2 tensors with shape [576,576]
all candidate embeddings
all candidate Q/K/V projections for logical layer 7
```

It does **not** assign `TensorRole` by itself.

## 4.3 `EvidenceItem`

Every proposed fact/binding records why it exists.

```cpp
struct EvidenceItem {
    EvidenceKind kind;
    std::string source;
    std::string fact;
};
```

Prefer structured fields internally where useful; human-readable text is a presentation concern.

## 4.4 `InferenceProposal<T>`

Rules return immutable proposals rather than modifying shared state.

Conceptually:

```cpp
template <typename T>
struct InferenceProposal {
    T value;
    std::vector<EvidenceItem> evidence;
    ProposalStrength strength;
    std::string rule_id;
};
```

`ProposalStrength` must not become a fuzzy “confidence score”. Prefer deterministic categories such as explicit metadata, shape-derived, naming-derived, or format-guaranteed. Equal valid contradictory proposals fail closed unless format semantics define deterministic precedence.

## 4.5 `CanonicalModelFacts`

This is the architectural boundary missing from the previous plan. It is the **only accepted input to semantic synthesis**.

Conceptually:

```cpp
struct CanonicalModelFacts {
    TransformerDimensions dimensions;
    TokenIds tokens;
    CanonicalNumerics numerics;
    std::vector<CanonicalLayerFacts> layers;
    TensorRoleBindings bindings;
    SemanticFeatureSet features;
    ProvenanceFacts provenance;
};
```

Properties:

- immutable after construction;
- fully validated;
- no raw checkpoint key spelling required by downstream synthesis;
- no architecture/family identity required for generic semantics;
- no CPU/CUDA types;
- no report/UX objects;
- enough information to deterministically synthesize graph and weight plan.

Descriptor resolution must eventually produce this same type.

## 4.6 `SemanticRequirements`

Derived from canonical semantics and/or the synthesized graph. It describes what an executor must implement, for example:

```text
attention: ordinary KV + GQA
position: RoPE adjacent-pairs
attention-output-transform: orthogonalize-current-value
ffn: SwiGLU
state storage: BF16-capable ordinary KV
```

It must not contain booleans such as `supports_cuda` or `supports_cpu`.

## 4.7 `ResolutionReport`

An observer-facing result containing:

- resolution mode;
- normalized facts and supporting evidence;
- accepted/rejected proposals;
- selected tensor grammars;
- every `TensorRole -> tensor` binding;
- defaults used and their rule source;
- ambiguities/conflicts;
- unsupported semantic features;
- final canonical facts fingerprint;
- graph/weight-plan fingerprint when synthesis succeeds.

---

# 5. Responsibility boundaries

The following responsibilities must remain distinct even if some start as free functions rather than classes.

## `MetadataNormalizer`

Input: raw `CheckpointMetadata`.

Output: normalized metadata observations + evidence.

Does not inspect tensor shapes, build a graph, or decide backend support.

## `TensorInventoryBuilder`

Input: checkpoint tensor repository/index.

Output: immutable `TensorInventory`.

Does not assign semantic roles.

## Rule catalogs

Input: immutable `InferenceInput`.

Output: immutable proposals.

Do not mutate accepted facts.

## `FactSolver`

Input: proposals.

Output: accepted canonical structural/semantic facts or typed conflicts.

This is the only component that resolves competing fact proposals.

## `BindingSolver`

Input: tensor-binding proposals + accepted structural facts + inventory.

Output: complete `TensorRoleBindings` or a typed incomplete/ambiguous result.

It must solve the required role set globally; it must not greedily accept the first matching tensor.

## `GraphSynthesizer`

Input: `CanonicalModelFacts`.

Output: `ModelGraph` only.

It must not create `WeightPlan`, decide backend support, inspect raw tensor paths, or build user-facing diagnostics.

## `WeightPlanSynthesizer`

Input: `CanonicalModelFacts` and synthesized graph where needed.

Output: `WeightPlan` only.

It owns semantic tensor-role requests/bindings, not CPU/CUDA physical layouts.

## `ResolutionAssembler`

Input: validated graph, weight plan, canonical provenance, and neutral model properties.

Output: `ResolvedModel`.

It composes already-resolved results and must contain no inference heuristics.

---

# 6. Rule catalogs, not a monolithic resolver

## 6.1 Metadata alias rules

Examples:

```text
hidden_size       -> hidden_size
n_embd             -> hidden_size
d_model            -> hidden_size
num_hidden_layers  -> layer_count
n_layer            -> layer_count
num_attention_heads -> query_heads
n_head             -> query_heads
num_key_value_heads -> key_value_heads
```

A rule whose only justification is `model_type == X` is not generic inference. It belongs in an explicit import/descriptor provider unless that fact can be reformulated as a reusable checkpoint convention.

## 6.2 Semantic feature rules

Known metadata mathematics map to neutral primitives:

```text
qk_norm=true
    -> query-key normalization semantic fact

xsa_projection=true
    -> OrthogonalizeCurrentValueSpec{1e-6}

rope_scaling.type=...
    -> RopeScalingSpec

sliding_window=N
    -> SlidingWindowPattern{N}
```

A key with unknown mathematical meaning must produce `UnsupportedSemanticFeature`, never be ignored merely because the rest of the checkpoint looks familiar.

## 6.3 Tensor naming grammars

Represent reusable naming conventions, not model families:

```text
model.layers.{layer}.self_attn.q_proj.weight
transformer.h.{layer}.attn.q_proj.weight
layers.{layer}.attention.wq.weight
```

A grammar proposes role candidates. It does not select them.

Fused representations are explicit grammar/layout concepts:

```text
QKV fused
Gate+Up fused
stacked experts
packed experts
```

Do not infer fused semantics from vague substrings.

## 6.4 Interfaces

Virtual dispatch is acceptable here because resolution is load-time, but do not create interfaces without an extension requirement.

Likely useful extension points:

```cpp
class IMetadataInferenceRule;
class ISemanticInferenceRule;
class ITensorBindingRule;
```

Do **not** introduce a universal `IInferenceValidator`. Validation is intentionally split by boundary.

Adding a reusable convention should mean registering one rule/module, not editing a central family switch.

---

# 7. Structure and tensor-role constraint solving

Infer structural facts before semantics that depend on them:

```text
logical layer set
hidden width
attention projection geometry
FFN geometry
repeated/heterogeneous layer schedule
tied vs untied output head
dense vs MoE
stateful/recurrent structures
optional per-layer features
```

Tensor-role matching must validate constraints, for example:

```text
AttentionQuery:
    rows == query_heads * head_dim
    cols == hidden_size

AttentionKey / AttentionValue:
    rows == key_value_heads * head_dim
    cols == hidden_size

AttentionOutput:
    rows == hidden_size
    cols == query_heads * head_dim

FfnGate / FfnUp:
    rows == intermediate_size
    cols == hidden_size

FfnDown:
    rows == hidden_size
    cols == intermediate_size
```

Checkpoint orientation/layout conventions must flow through checkpoint/weight-layout abstractions, never backend-specific assumptions.

Acceptance is **global**:

```text
naming candidates
      +
shape constraints
      +
structural facts
      +
required role set
      |
      v
complete unique assignment
```

If two complete assignments remain equally valid, automatic resolution is ambiguous and fails unless an explicit descriptor/provider supplies the missing evidence.

---

# 8. Validation is layered, not a god interface

Each validation boundary has a different reason to change.

## `MetadataValidator`

- dimensions/ranges;
- token IDs;
- contradictory aliases;
- head geometry that can already be checked from metadata.

## `TensorInventoryValidator`

- duplicate tensor identities;
- invalid rank/dtype metadata;
- invalid shard aliases;
- layer-index continuity where the grammar promises it.

## `BindingValidator`

- every required role resolved exactly once unless aliases are semantically intentional;
- role shape constraints;
- tied aliases deliberate;
- fused-layout constraints complete.

## `CanonicalFactsValidator`

- all required facts present;
- structural and semantic proposals internally consistent;
- no unresolved conflicts or meaningful unknown feature remains.

## `GraphValidator`

The existing graph-level invariants remain responsible for primitive correctness and layer consistency.

## Backend capability validation

Remains inside backend compilation. It compares `SemanticRequirements`/compiled semantics against what that backend implements.

No load-time auto-inference validator may weaken backend capability checks.

---

# 9. Shared synthesis for descriptors and auto inference

The long-term target is:

```text
Descriptor importer ----+
                        +--> CanonicalModelFacts
Auto inference ---------+
                                  |
                                  +--> GraphSynthesizer --> ModelGraph
                                  |
                                  +--> WeightPlanSynthesizer --> WeightPlan
                                                     |
                                                     v
                                             ResolutionAssembler
                                                     |
                                                     v
                                               ResolvedModel
```

Descriptor code may still be responsible for obtaining unusual/import-specific facts, but once those facts exist it must enter the same canonical boundary.

There must not be two implementations of “how these facts become an `AttentionSpec`” or two implementations of “how these bindings become a `WeightPlan`”.

This convergence is a prerequisite for broad descriptor deletion.

---

# 10. Backend capability inversion

`ModelCapabilities` currently contains backend-oriented properties. During this refactor, review whether those fields belong in inference at all.

Preferred direction:

```text
Resolved semantic requirements
            x
CPU backend capability set
            |
            v
      accept/reject CPU

Resolved semantic requirements
            x
CUDA backend capability set
            |
            v
      accept/reject CUDA
```

For example, XSA does not mean “the model supports CUDA”. It means the model requires `OrthogonalizeCurrentValue`; CUDA support exists only if the CUDA compiler can lower that primitive.

Do not move backend feature matrices upward into checkpoint inference.

---

# 11. Resolution order

Recommended deterministic order:

```text
1. explicitly user-selected importer/descriptor
2. explicit special provider whose semantics genuinely cannot be inferred generically
3. automatic evidence-driven inference
4. failure with ResolutionReport
```

Development/debug modes may force a path:

```text
--resolution auto
--resolution descriptor
```

The normal runtime must not require an architecture name when automatic resolution is complete.

Descriptors that exist only because generic inference did not previously exist should be migrated and deleted after parity is proven.

---

# 12. GPT-X2.5 acceptance migration

GPT-X2.5 is the first proof of this architecture.

Automatic inference must resolve at least:

```text
hidden                 576
intermediate           1728
layers                 30
query heads            9
KV heads               3
head dim               64
vocab                   32770
context                 8192
RMSNorm epsilon         1e-6
RoPE theta              100000
QK norm                 false
RoPE pairing            adjacent pairs
XSA                     OrthogonalizeCurrentValueSpec{1e-6}
FFN                     SwiGLU
tied embeddings         true
```

Its `transformer.h.{layer}.*` bindings must resolve through reusable naming grammars, never a `gptx2` branch.

## Acceptance sequence

1. Freeze the descriptor-driven `CanonicalModelFacts`, `ModelGraph`, `WeightPlan`, semantic fingerprint, tokenizer outputs, and official-reference logits as golden evidence.
2. Resolve the same checkpoint through auto inference.
3. Diff canonical facts, graph, weight plan, and fingerprints.
4. Require numerical parity through the existing official-reference harness.
5. Only then delete `descriptors/gptx25.json`.
6. Re-run CPU and CUDA smoke/parity coverage after deletion.

No backend source may contain `gptx`, `GPT-X`, `model_type`, or architecture-specific branching as a consequence of this migration.

---

# 13. Descriptor role after the refactor

Descriptors remain valid for:

- explicit semantic ambiguity that the checkpoint does not disambiguate;
- non-standard tensor naming for which no genuinely reusable grammar exists;
- import transformations whose semantics cannot be represented by a reusable checkpoint convention;
- truly special checkpoint layouts requiring explicit evidence or transformation.

Descriptors should become declarations of **missing evidence/import semantics**, not copies of conventional transformer structure.

Never add a generic rule merely to delete a descriptor. A rule enters the generic catalog only if its meaning is reusable independently of one family.

---

# 14. Failure taxonomy

Use typed failures such as:

```text
MissingRequiredMetadata
ConflictingMetadata
ConflictingInferenceFacts
AmbiguousTensorBinding
MissingTensorRole
ShapeConstraintViolation
UnsupportedSemanticFeature
UnsupportedTensorLayout
UnsupportedGraphPrimitive
IncompleteLayerSchedule
BackendCapabilityMismatch
```

Diagnostics must show evidence and rejected alternatives.

Bad:

```text
unsupported model
```

Good:

```text
automatic resolution failed:
  layer 0 AttentionValue has two complete valid bindings:
    transformer.h.0.attn.v_proj.weight [192,576]
    transformer.h.0.attn.value.weight  [192,576]
  no evidence rule distinguishes them
```

---

# 15. Explainability

Add an inspection API/CLI equivalent to:

```text
celeg-inspect --repo ... --explain-resolution
```

Example:

```text
resolution: automatic

hidden_size: 576
  accepted:
    config.hidden_size = 576
    token_embedding.shape[1] = 576

attention[0]: GQA 9Q/3KV x 64
  accepted:
    num_attention_heads = 9
    num_key_value_heads = 3
    q_proj = [576,576]
    k_proj = [192,576]

output transform: orthogonalize-current-value
  accepted:
    xsa_projection = true

RoPE pairing: adjacent-pairs
  accepted:
    semantic rule <rule-id>
```

The report is built from recorded decisions; it must not call back into rules or influence the solver.

---

# 16. Architecture-boundary enforcement

Extend `scripts/check_architecture_boundaries.py` so regressions become mechanically difficult.

Rules should include:

- backend roots cannot reference `model_type`, repository/model IDs, or inference-rule types;
- generic inference cannot include `src/models/<family>`;
- generic rules cannot branch on family IDs;
- rules cannot mutate accepted-resolution state;
- graph synthesis cannot inspect raw tensor path strings;
- `GraphSynthesizer` cannot produce `WeightPlan`;
- `WeightPlanSynthesizer` cannot depend on CPU/CUDA physical weight types;
- `CanonicalModelFacts` cannot depend on report/UI types or backend types;
- descriptor and auto paths must converge before semantic synthesis/backend compilation;
- no catch-all rule may silently accept unknown semantic metadata;
- backend support booleans must not be inferred from model-family identity.

Add architectural tests where static boundary checks cannot express an invariant cleanly.

---

# 17. Performance constraints

All inference work is checkpoint-load time.

Requirements:

- zero decode-time model-name checks;
- zero token-time rule evaluation;
- immutable inventory built once;
- rule catalogs evaluated once per resolution;
- solvers run only at load time;
- final execution uses the existing immutable compiled program;
- no virtual dispatch introduced into hot primitive execution for this feature.

Only add a checkpoint-fingerprint inference cache after measuring load-time resolution cost.

---

# 18. Tests before broad migration

## Unit coverage

Cover independently:

- metadata alias proposals;
- conflicting aliases;
- order independence of rule catalogs;
- immutable proposal behavior;
- fact solver conflicts;
- tensor grammar matching;
- shape-based candidate rejection;
- globally ambiguous bindings;
- fused QKV/gate-up layouts;
- tied embeddings;
- XSA feature mapping;
- RoPE pairing mapping;
- unknown meaningful semantic metadata;
- canonical-facts validation;
- graph synthesis independent of raw tensor names;
- weight-plan synthesis independent of backend types;
- report generation independent of inference decisions.

## Synthetic fixtures

Basic inference tests use in-memory metadata and tensor inventories. They must not require model downloads.

## Negative/mutation fixtures

Mutate valid inputs with:

- wrong head count;
- missing K/V projection;
- duplicate equally-valid role candidate;
- conflicting hidden aliases;
- unknown XSA-like feature;
- mismatched layer count;
- invalid tied LM head;
- incomplete fused QKV layout;
- valid graph requiring an unsupported backend primitive.

Every unsafe mutation must fail before execution.

## Golden real-checkpoint coverage

GPT-X2.5 is the first golden case. Later fixtures should be chosen for distinct reusable checkpoint conventions rather than simply accumulating model families.

---

# 19. Implementation phases

## Phase 0 — Freeze the current golden result

- keep `descriptors/gptx25.json` temporarily;
- capture descriptor-driven canonical semantics, graph, weight plan, semantic fingerprint, tokenizer fixture, and official-reference logits;
- capture the actual GPT-X2.5 tensor inventory.

**Exit:** the current correct result is reproducible without relying on backend behavior to describe its semantics.

## Phase 1 — Evidence, proposals, and typed failures

- introduce `EvidenceItem`;
- introduce immutable `InferenceProposal<T>`;
- introduce typed failure categories;
- add rule-order-independence tests.

**Exit:** rules can propose facts without mutating a shared context.

## Phase 2 — Metadata normalization

- add canonical metadata observations;
- add generic alias rules;
- add `MetadataValidator`;
- reject contradictions deterministically.

**Exit:** GPT-X2.5 dimensions/numerics can be proposed without a family-specific rule.

## Phase 3 — Tensor inventory

- build immutable tensor index;
- support reusable layer-index naming grammars;
- expose shape/dtype/shard evidence;
- add `TensorInventoryValidator`.

**Exit:** GPT-X2.5 structural tensor inventory is understood without `gptx2` knowledge.

## Phase 4 — Fact solver

- centralize proposal acceptance/conflict handling;
- derive structural facts from metadata + tensor evidence;
- ensure rule ordering cannot alter the accepted result.

**Exit:** structural facts are unique or resolution fails with evidence.

## Phase 5 — Tensor binding solver

- generate role proposals;
- apply shape/structure constraints;
- solve complete assignments globally;
- add `BindingValidator` and ambiguity diagnostics.

**Exit:** GPT-X2.5 gets a complete unique neutral `TensorRoleBindings` set.

## Phase 6 — Semantic rule catalog

- map known semantics: QK norm, RoPE, pairing, scaling, sliding window, XSA, activation, tied embeddings;
- reject unknown meaningful features;
- keep family identity out.

**Exit:** GPT-X2.5 semantic proposals equal today's descriptor semantics.

## Phase 7 — `CanonicalModelFacts`

- introduce the immutable canonical boundary;
- add `CanonicalFactsValidator`;
- assemble accepted structural facts, semantic facts, and bindings;
- make auto inference terminate at this type before synthesis.

**Exit:** automatic GPT-X2.5 inference produces validated canonical facts.

## Phase 8 — Split semantic synthesis

- implement/refactor `GraphSynthesizer` to produce only `ModelGraph`;
- implement/refactor `WeightPlanSynthesizer` to produce only `WeightPlan`;
- introduce a thin `ResolutionAssembler`;
- eliminate duplicated graph/weight-plan decisions.

**Exit:** canonical facts deterministically reproduce the descriptor-driven graph and weight plan.

## Phase 9 — Converge descriptor resolution

- adapt descriptor import to produce the same `CanonicalModelFacts`;
- remove descriptor-only synthesis paths after parity;
- no compatibility shim remains once callers migrate.

**Exit:** descriptor and automatic paths are indistinguishable downstream of canonical facts.

## Phase 10 — Backend capability inversion

- derive neutral semantic requirements;
- ensure CPU/CUDA compilers validate those requirements themselves;
- remove any newly-unnecessary inference-side backend-support decision.

**Exit:** adding a backend never requires changes to checkpoint inference.

## Phase 11 — Automatic provider/composition

- register one generic auto-inference provider at the composition boundary;
- it succeeds only with a complete validated result;
- explicit special importers can still outrank it when genuinely necessary.

**Exit:** `--repo AxiomicLabs/GPT-X2.5-135M` resolves automatically.

## Phase 12 — Remove GPT-X2.5 descriptor

- verify canonical/graph/weight-plan equality;
- run tokenizer and official-reference logits parity;
- run CPU/CUDA smoke/parity where available;
- delete `descriptors/gptx25.json`;
- update tests that assume one descriptor per architecture.

**Exit:** full GPT-X2.5 support remains with no GPT-X2.5-specific descriptor/backend branch.

## Phase 13 — Harden false-positive resistance

- mutation fixtures;
- ambiguity tests;
- fuzz naming/metadata combinations;
- rule-order randomized tests;
- fail-closed regression suite.

**Exit:** plausible-but-wrong checkpoints cannot silently execute.

## Phase 14 — Migrate conventional descriptors selectively

For each descriptor:

1. run automatic inference;
2. diff `CanonicalModelFacts`, `ModelGraph`, `WeightPlan`, and semantic fingerprint;
3. identify the missing convention;
4. add it generically only if reusable;
5. otherwise keep the descriptor;
6. delete only after golden/numerical parity.

Descriptor deletion is not a goal by itself.

## Phase 15 — Diagnostics and resolution cache

Only after measurement:

- stable `ResolutionReport` serialization;
- explain CLI/API;
- checkpoint-fingerprint cache if worthwhile;
- resolution-time benchmarks.

---

# 20. SOLID mapping

## SRP

Responsibilities are explicitly separated:

```text
metadata normalization
inventory construction
proposal generation
fact solving
binding solving
canonical-fact assembly
graph synthesis
weight-plan synthesis
resolution assembly
backend capability validation
reporting
```

Each has a distinct reason to change. In particular, `GraphSynthesizer` no longer owns weight planning/capabilities and validation is not collapsed into a god `IInferenceValidator`.

## OCP

New reusable metadata/naming/semantic conventions are added through rule catalogs. New mathematics are added through neutral semantic primitives. New backends advertise/validate their own lowering capabilities.

No central family switch should require editing for any of those extensions.

## LSP

Rules obey the same side-effect-free proposal contract and can be substituted without changing solver semantics or depending on invocation order.

A `ResolvedModel` produced through automatic inference is indistinguishable to downstream compilation from one produced through an explicit descriptor/importer.

## ISP

Interfaces remain narrow:

- metadata rules need metadata/evidence views, not tensor solvers;
- tensor rules need inventory/structural facts, not graph builders;
- semantic rules do not see CPU/CUDA types;
- backends do not see inference catalogs;
- reporting consumes recorded decisions but does not control them.

Do not create an interface merely because a class exists; prefer concrete/free-function components until there is a real substitution/extension requirement.

## DIP

High-level inference depends on backend-neutral abstractions and immutable facts. Descriptor and auto importers depend inward on the canonical boundary. CPU/CUDA depend on semantic graph/program requirements, never the other way around.

Most importantly:

```text
checkpoint does not say "I support CUDA"
CUDA says "I can lower every semantic requirement of this checkpoint"
```

---

# 21. Anti-pattern checklist

Reject a change if it introduces any of these:

- `if (model_type == ...)` in generic inference or backend execution;
- one resolver object that normalizes metadata, binds tensors, builds graph, and emits diagnostics;
- mutable shared inference context modified by rules;
- rule priority/confidence magic needed to make order-dependent inference work;
- `GraphSynthesizer` returning `WeightPlan` or backend flags;
- universal `IInferenceValidator` accumulating unrelated validation methods;
- raw tensor names visible below canonical binding/synthesis boundaries;
- report formatting required for correctness;
- descriptor and auto code constructing semantically equivalent graphs independently;
- inferred `supports_cpu`/`supports_cuda` based on model identity;
- silent fallback for an unknown semantic key;
- model-specific generic rule added only to eliminate one JSON file.

---

# 22. Definition of done

This refactor is complete when all statements below are true:

1. `descriptors/gptx25.json` no longer exists.
2. GPT-X2.5 resolves from ordinary HF files without a GPT-X2.5-specific architecture implementation.
3. Rules are side-effect-free proposal producers and rule ordering cannot change results.
4. Automatic resolution produces a validated `CanonicalModelFacts` before graph/weight synthesis.
5. Descriptor importers and automatic inference converge at that same canonical boundary.
6. `GraphSynthesizer` produces only graph semantics; `WeightPlanSynthesizer` owns weight-plan synthesis.
7. GPT-X2.5 resolves 9Q/3KV GQA, head dim 64, SwiGLU, adjacent-pair RoPE, no QK norm, and `OrthogonalizeCurrentValueSpec{1e-6}`.
8. Every required tensor role is inferred and shape-validated as a complete unique assignment.
9. CPU and CUDA see only semantic graph/program requirements and never `gptx2`.
10. Backend support is validated by backend capability/lowering logic, not inferred from model identity.
11. Tokenizer and official-reference logits parity remain green.
12. Ambiguous, contradictory, or unsupported checkpoints fail with typed evidence-backed diagnostics before execution.
13. `ResolutionReport` is auditable but has no influence on inference decisions.
14. Architecture-boundary checks mechanically prevent family logic, backend types, and raw checkpoint names from leaking across the new boundaries.
15. Automatic inference adds no decode-time overhead and no virtual dispatch to hot primitive execution.

The desired end state is not “CELEG knows every model.” It is:

> CELEG knows a vocabulary of model semantics and checkpoint conventions, can prove when an unseen checkpoint is completely expressible using that vocabulary, and hands the resulting immutable semantics to whichever backend can actually lower them.
