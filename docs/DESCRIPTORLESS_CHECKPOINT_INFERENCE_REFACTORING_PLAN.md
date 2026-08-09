# Descriptorless Checkpoint Inference Refactoring Plan

## Status

Planned architectural refactor.

Baseline: `master` after the GPT-X2.5 semantic work, including generic attention-output transforms, adjacent-pair RoPE, CPU/CUDA coverage, declarative GPT-X2.5 resolution tests, and the official-reference parity harness.

## Objective

Allow CELEG to load a previously unknown Hugging Face-style checkpoint **without a model-specific descriptor** when the checkpoint can be mapped completely and unambiguously onto CELEG's existing backend-neutral semantic vocabulary.

The target flow is:

```text
checkpoint
    |
    v
metadata normalization + tensor inventory
    |
    v
evidence-producing semantic rules
    |
    v
tensor-role binding solver
    |
    v
ModelGraph + WeightPlan + policies
    |
    v
strict semantic validation
    |
    v
backend compiler
```

A descriptor remains an explicit fallback/override for checkpoints whose semantics cannot be inferred safely. It must no longer be the mandatory mechanism for otherwise conventional architectures.

The first acceptance model is `AxiomicLabs/GPT-X2.5-135M`: after this refactor, deleting `descriptors/gptx25.json` must not change its resolved semantics or numerical behavior.

---

## Core rule: evidence, never guessing

Auto-discovery must not mean heuristic execution.

CELEG may execute an automatically resolved checkpoint only if all required semantics and tensor bindings are supported by explicit rules and validated against independent evidence.

Examples of independent evidence:

- metadata says `hidden_size = 576`;
- Q/K/V tensor shapes agree with hidden/head geometry;
- exactly 30 layer indices exist and metadata also says 30 layers;
- `num_attention_heads = 9`, `num_key_value_heads = 3`, and projection shapes agree;
- `xsa_projection = true` maps through a registered semantic feature rule to `OrthogonalizeCurrentValueSpec`;
- RoPE pairing is resolved explicitly rather than inferred from a family name;
- tied embeddings are accepted only when metadata and/or tensor identity make the binding consistent.

If required evidence is missing, contradictory, or ambiguous, automatic resolution must fail before backend compilation.

There is no `best effort` execution mode in the runtime path.

---

## Non-goals

This refactor must **not**:

- put architecture IDs or `model_type` branches in CPU/CUDA execution code;
- infer new mathematics from arbitrary metadata names;
- silently approximate an unsupported primitive;
- introduce runtime/decode-time architecture probing;
- create a single giant `GenericHfResolver` full of `if/else` family logic;
- use model names as confidence signals;
- remove descriptors before their replacement inference path has golden coverage;
- preserve obsolete descriptor-only access paths merely for backward compatibility.

Inference happens once during checkpoint resolution. The execution program remains immutable and architecture-neutral.

---

# 1. Desired architecture

## 1.1 Resolution pipeline

Introduce a backend-neutral inference pipeline with narrowly scoped stages:

```text
CheckpointView
    |
    +--> MetadataNormalizer
    |
    +--> TensorInventoryBuilder
             |
             v
        InferenceEvidence
             |
    +--------+---------+
    |                  |
    v                  v
SemanticRuleCatalog   TensorBindingRuleCatalog
    |                  |
    +--------+---------+
             v
       BindingSolver
             |
             v
       GraphSynthesizer
             |
             v
      InferenceValidator
             |
             v
       ResolvedModel
             +--> ResolutionReport
```

No stage may depend on a backend.

## 1.2 Proposed neutral types

Names may change during implementation, but responsibilities should remain separate.

### `NormalizedModelMetadata`

Canonical values that can be derived from common checkpoint configuration spellings without encoding a model family.

Examples:

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

This type contains canonical facts, not derived execution choices.

### `TensorInventory`

An immutable index over tensor names, dimensions, dtype, shard location, and aliases.

It must support queries such as:

```text
all tensors matching a layer-indexed naming grammar
all rank-2 tensors with shape [576, 576]
all candidate token embeddings
all candidate Q/K/V projections for logical layer 7
```

It must not assign `TensorRole` by itself.

### `InferenceEvidence`

Every inference must record why it was made.

Example:

```cpp
struct EvidenceItem {
    EvidenceKind kind;
    std::string source;
    std::string fact;
};
```

A semantic fact should carry its evidence rather than returning a naked boolean.

### `ResolutionReport`

A user/debug-facing result containing:

- resolved dimensions;
- resolved primitive semantics;
- selected tensor naming grammar(s);
- every `TensorRole -> tensor` binding;
- fallback/default values used;
- rejected alternatives and why;
- ambiguities;
- unsupported semantics;
- final resolution mode: `auto` or `descriptor`.

This report is important both for debugging and to prevent inference logic from becoming opaque.

---

# 2. Rule catalogs, not a monolithic resolver

## 2.1 Metadata normalization rules

Create small deterministic rules for canonical aliases.

Examples:

```text
hidden_size              -> hidden_size
n_embd                    -> hidden_size
d_model                   -> hidden_size

num_hidden_layers        -> layer_count
n_layer                   -> layer_count

num_attention_heads      -> query_heads
n_head                    -> query_heads

num_key_value_heads      -> key_value_heads
```

Rules must be generic conventions. A rule whose only justification is `model_type == X` belongs in a descriptor/provider, not in generic inference.

Conflicting aliases must be an error unless a rule explicitly defines a precedence backed by format semantics.

## 2.2 Semantic feature rules

Metadata keys that describe known mathematics are mapped to CELEG semantic primitives by dedicated rules.

Examples:

```text
qk_norm=true
    -> AttentionSpec::query_key_norm = true

xsa_projection=true
    -> OrthogonalizeCurrentValueSpec{1e-6}

rope_scaling.type=...
    -> RopeScalingSpec

sliding_window=N
    -> SlidingWindowPattern{N}
```

The important distinction is:

```text
metadata spelling -> known CELEG primitive
```

not:

```text
model family -> hard-coded backend behavior
```

A feature key whose mathematical semantics are unknown must yield `unsupported semantic feature`, never be ignored silently.

## 2.3 Tensor naming grammars

Represent common tensor naming conventions as reusable grammars.

Examples include families of spelling, not model families:

```text
model.layers.{layer}.self_attn.q_proj.weight
transformer.h.{layer}.attn.q_proj.weight
layers.{layer}.attention.wq.weight
```

Each grammar can propose candidates for roles such as:

```text
TokenEmbedding
AttentionInputNorm
AttentionQuery
AttentionKey
AttentionValue
AttentionOutput
FfnInputNorm
FfnGate
FfnUp
FfnDown
FinalNorm
LanguageModelHead
```

A grammar only proposes bindings. Shape and graph validation decide whether the proposal is valid.

Fused representations must be modeled explicitly rather than guessed from substrings:

```text
QKV fused
Gate+Up fused
packed expert tensors
```

## 2.4 Rule extension

Use registries/catalogs following the same composition style already used by CELEG runtime modules.

Potential interfaces:

```cpp
class IMetadataInferenceRule;
class ISemanticInferenceRule;
class ITensorBindingRule;
class IInferenceValidator;
```

These do not belong in hot paths and may use virtual dispatch safely during load-time resolution.

Adding a new generic convention should mean registering one rule, not editing a central switch.

---

# 3. Tensor-role binding must be a constraint solver

Name matching alone is not enough.

For every proposed binding, validate constraints such as:

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

The solver must consider checkpoint orientation/layout conventions through existing weight-layout abstractions rather than baking backend layout into inference.

A candidate is accepted only if the complete required role set for the graph is satisfiable.

If two complete candidate assignments remain valid, resolution is ambiguous and must fail unless an explicit descriptor resolves it.

---

# 4. Infer structure before semantics that depend on structure

Use tensor inventory and metadata together to establish structural facts first:

1. logical layer index set;
2. hidden width;
3. attention projection geometry;
4. FFN geometry;
5. repeated vs heterogeneous layer schedules;
6. tied/untied output head;
7. dense vs MoE tensors;
8. stateful/recurrent primitives;
9. optional per-layer features.

Only then synthesize `LayerSpec` values.

This prevents a metadata key from forcing a graph that the tensor shapes cannot represent.

---

# 5. Graph synthesis

Introduce a `GraphSynthesizer` that consumes only validated canonical facts and bindings.

It should produce:

```text
ModelGraph
WeightPlan
ModelCapabilities
ModelProvenance
```

The synthesizer must not inspect raw checkpoint key spelling. By this stage:

```text
raw names -> TensorRole bindings
raw metadata -> canonical/semantic facts
```

must already be complete.

The existing descriptor graph/weight-plan builders should converge on the same downstream builder APIs so descriptor resolution and automatic resolution cannot drift semantically.

Target:

```text
DescriptorResolution ----+
                         +--> canonical resolved facts --> GraphSynthesizer
AutoInference -----------+
```

Do not maintain two independent graph-construction implementations long term.

---

# 6. Resolution order

Recommended deterministic order:

```text
1. explicit user-selected descriptor/provider
2. descriptor that explicitly probes a truly special checkpoint
3. automatic evidence-driven inference
4. failure with ResolutionReport
```

However, for checkpoints whose descriptor exists only because generic inference did not previously exist, migrate them to auto inference and delete the redundant descriptor.

An alternative CLI/debug mode should allow forcing auto resolution while developing:

```text
--resolution auto
--resolution descriptor
```

The normal runtime should not require users to know an architecture name when auto resolution is complete.

---

# 7. GPT-X2.5 acceptance migration

GPT-X2.5 is the first proof that the architecture works.

## Required automatically inferred facts

From metadata/tensors, CELEG must resolve at least:

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

The tensor grammar must bind its `transformer.h.{layer}.*` names without a `gptx2` rule.

## Exit criterion

Delete:

```text
descriptors/gptx25.json
```

Then all of the following must still pass:

- architecture/resolution semantics test;
- tensor binding/weight-plan test against the actual checkpoint inventory;
- tokenizer parity fixture;
- XSA primitive tests;
- adjacent-pair RoPE tests;
- official-reference logits parity harness;
- CPU smoke inference;
- CUDA smoke inference where available.

No backend source may contain `gptx`, `GPT-X`, `model_type`, or architecture-specific branching as a result.

---

# 8. Descriptor role after the refactor

Descriptors remain useful for three cases.

## 8.1 Explicit semantic ambiguity

Example: identical tensor shapes/names can represent two different mathematical operations and metadata does not disambiguate them.

## 8.2 Non-standard checkpoint naming

The semantics are known, but no generic reusable naming grammar applies.

## 8.3 Truly family-specific import semantics

A checkpoint requires transformation/import logic that cannot be described as a general semantic primitive or reusable format convention.

Descriptors should become small declarations of missing evidence, not copies of conventional transformer structure.

A useful metric is:

```text
model-specific descriptor lines / supported model families
```

which should trend downward over time.

---

# 9. Failure taxonomy

Automatic inference needs precise failures.

Introduce typed failure categories such as:

```text
MissingRequiredMetadata
ConflictingMetadata
AmbiguousTensorBinding
MissingTensorRole
ShapeConstraintViolation
UnsupportedSemanticFeature
UnsupportedTensorLayout
UnsupportedGraphPrimitive
IncompleteLayerSchedule
```

CLI/API diagnostics should report the exact evidence and candidates involved.

Bad:

```text
unsupported model
```

Good:

```text
automatic resolution failed:
  layer 0 AttentionValue has two valid candidates:
    transformer.h.0.attn.v_proj.weight [192,576]
    transformer.h.0.attn.value.weight  [192,576]
  no rule provides evidence to choose between them
```

---

# 10. Validation layers

Use multiple validation boundaries rather than one late validation call.

## Metadata validation

- dimensions positive;
- head geometry integral;
- context valid;
- token IDs in range when vocab is known.

## Tensor inventory validation

- layer indices contiguous or explicitly mapped;
- tensor ranks/dtypes supported;
- shard duplicates rejected.

## Binding validation

- every required role exactly once unless role semantics allow aliases;
- tensor shape satisfies semantic role;
- tied aliases are intentional.

## Graph validation

- all primitives internally valid;
- every layer has valid mixer/FFN semantics;
- semantic policies are complete.

## Backend validation

Existing backend compiler capability validation remains the final boundary. Auto inference must not weaken it.

---

# 11. Tests before broad migration

## Unit tests

Create focused tests for:

- metadata alias normalization;
- conflicting alias rejection;
- tensor grammar matching;
- shape-based candidate rejection;
- ambiguous complete bindings;
- fused QKV/gate-up layouts;
- tied embeddings;
- XSA feature mapping;
- RoPE pairing mapping;
- unknown semantic metadata;
- resolution report contents.

## Synthetic checkpoint inventories

Tests should not require downloading a model for basic inference behavior. Construct metadata + tensor inventories in memory.

## Golden real-checkpoint tests

Maintain a small set of real checkpoint fixtures/manifests representing distinct conventions.

GPT-X2.5 must be the first golden case.

## Negative fixtures

Deliberately mutate valid configurations:

- wrong head count;
- one missing K projection;
- duplicate output projection;
- conflicting hidden aliases;
- unknown XSA-like feature;
- mismatched layer count;
- invalid tied LM head.

Every mutation must fail before execution.

---

# 12. Explainability tooling

Add a command or inspection API equivalent to:

```text
celeg-inspect --repo ... --explain-resolution
```

Example output:

```text
resolution: automatic
hidden_size: 576
  evidence: config.hidden_size=576
  evidence: token_embedding.shape[1]=576

attention[0]: GQA 9Q/3KV x 64
  evidence: num_attention_heads=9
  evidence: num_key_value_heads=3
  evidence: q_proj=[576,576]
  evidence: k_proj=[192,576]

output transform: orthogonalize-current-value
  evidence: xsa_projection=true

RoPE pairing: adjacent-pairs
  evidence: explicit semantic rule ...
```

This is not merely UX. It is a design constraint forcing inference decisions to be auditable.

---

# 13. Architecture-boundary enforcement

Extend `scripts/check_architecture_boundaries.py` with rules such as:

- backend roots may not reference `model_type` or auto-inference classes;
- generic inference roots may not include `src/models/<family>`;
- tensor binding rules may depend on `TensorRole`, never CPU/CUDA weight types;
- graph synthesis may not inspect raw tensor path strings;
- descriptor code and auto-inference code must converge before backend compilation;
- no model-family IDs inside generic semantic rules;
- no catch-all rule that accepts unknown semantics silently.

The boundary checker should make architectural regression mechanically difficult.

---

# 14. Performance constraints

All auto-discovery work is checkpoint-load time.

Requirements:

- zero decode-time model-name checks;
- zero token-time rule evaluation;
- tensor inventory built once;
- rules run once per checkpoint resolution;
- final output is the existing immutable compiled program;
- inference result may be cached by checkpoint fingerprint if resolution cost becomes measurable.

Do not optimize this prematurely; first measure model-load overhead.

---

# 15. Implementation phases

## Phase 0 — Freeze acceptance evidence

- keep current GPT-X2.5 descriptor and parity harness;
- capture the exact current `ResolvedModel`/semantic fingerprint as a golden baseline;
- add a tensor inventory fixture for the real checkpoint naming and shapes.

**Exit:** current descriptor-driven result is reproducible without backend execution.

## Phase 1 — Evidence and reporting model

- add `EvidenceItem`/equivalent;
- add `ResolutionReport`;
- make inference failures typed;
- no automatic resolution yet.

**Exit:** unit tests can explain a synthetic resolution/failure.

## Phase 2 — Metadata normalization

- introduce canonical metadata view;
- implement alias rules;
- reject contradictions;
- keep family names out.

**Exit:** GPT-X2.5 dimensions/numerics resolve without its descriptor.

## Phase 3 — Tensor inventory

- build immutable tensor index;
- parse layer indices through reusable naming grammars;
- expose shape/dtype evidence.

**Exit:** GPT-X2.5 tensor structure is discovered without `gptx2` knowledge.

## Phase 4 — Tensor-role binding solver

- generic role proposals;
- shape constraints;
- complete-assignment validation;
- ambiguity diagnostics.

**Exit:** GPT-X2.5 complete WeightPlan bindings are inferred.

## Phase 5 — Semantic feature catalog

- map known metadata semantics to neutral primitives;
- start with QK norm, RoPE, RoPE pairing, sliding-window semantics, XSA, activation and tied embeddings;
- unknown meaningful features fail closed.

**Exit:** GPT-X2.5 produces the same `AttentionSpec` as today.

## Phase 6 — Shared graph synthesis

- introduce canonical graph-builder inputs;
- route auto inference through them;
- migrate descriptor path to the same graph/weight-plan synthesis APIs where practical;
- remove duplicated graph-building semantics.

**Exit:** descriptor and auto paths produce equal semantic fingerprints for GPT-X2.5.

## Phase 7 — Automatic architecture provider

- add one generic auto-inference architecture/provider at composition level;
- run it only when it can produce a complete validated result;
- prevent it from winning against an explicit higher-specificity special provider when semantics are genuinely non-generic.

**Exit:** `--repo AxiomicLabs/GPT-X2.5-135M` resolves automatically.

## Phase 8 — Remove GPT-X2.5 descriptor

- delete `descriptors/gptx25.json`;
- update architecture catalog count tests so they do not assume one descriptor per supported family;
- preserve the official parity harness.

**Exit:** full GPT-X2.5 acceptance suite passes without any GPT-X2.5-specific descriptor or backend code.

## Phase 9 — Harden false-positive resistance

- negative/mutation fixtures;
- ambiguous naming tests;
- contradiction tests;
- fuzz tensor names/metadata combinations;
- verify fail-closed behavior.

**Exit:** unknown checkpoints cannot accidentally resolve to a plausible but incorrect graph.

## Phase 10 — Migrate conventional descriptors

For each existing descriptor:

1. run auto inference;
2. diff `ResolvedModel`, WeightPlan and semantic fingerprint;
3. identify missing reusable convention;
4. add a generic rule only if it applies beyond that family;
5. otherwise keep the descriptor;
6. delete descriptor only after numerical/golden parity.

Never make descriptor deletion a goal by itself.

## Phase 11 — Resolution cache and diagnostics polish

Only after measuring load-time cost:

- checkpoint-fingerprint keyed inference cache;
- stable report serialization;
- CLI explain command;
- benchmark resolution time.

---

# 16. SOLID mapping

This design must preserve the recent refactor rather than undo it.

## SRP

Separate metadata normalization, tensor indexing, binding, semantic inference, graph synthesis and validation.

## OCP

New reusable checkpoint conventions are added as rules; new execution mathematics are added as semantic primitives. Neither requires editing a central model-family switch.

## LSP

An automatically resolved `ResolvedModel` is indistinguishable from a descriptor-resolved one to backend compilers.

## ISP

Rules consume only the evidence they need. Tensor naming rules do not need graph synthesis APIs; semantic rules do not need CUDA/CPU types.

## DIP

Inference depends on backend-neutral `CheckpointMetadata`, tensor inventory, `TensorRole`, `ModelGraph` and semantic primitive types. Backends remain downstream consumers.

---

# 17. Definition of done

This refactor is complete when all statements below are true:

1. `descriptors/gptx25.json` no longer exists.
2. GPT-X2.5 resolves from its ordinary HF checkpoint files without a family-specific architecture implementation.
3. Its automatically resolved graph contains 9Q/3KV GQA, head dim 64, SwiGLU, adjacent-pair RoPE, no QK norm and `OrthogonalizeCurrentValueSpec{1e-6}`.
4. Every required tensor role is inferred and shape-validated.
5. CPU and CUDA see only the semantic graph/program, never `gptx2`.
6. Tokenizer and official-reference logits parity tests remain green.
7. An ambiguous or unsupported synthetic checkpoint fails with a precise `ResolutionReport` instead of executing an approximation.
8. Existing descriptor-backed models continue through the same downstream semantic compilation boundary while descriptors are migrated selectively.
9. Architecture-boundary checks prevent family logic from leaking into generic inference or backends.
10. Automatic inference adds no measurable decode-time overhead.

The desired end state is not “CELEG knows every model.” It is:

> CELEG knows a vocabulary of model semantics and checkpoint conventions, and can prove when an unseen checkpoint is completely expressible using that vocabulary.
