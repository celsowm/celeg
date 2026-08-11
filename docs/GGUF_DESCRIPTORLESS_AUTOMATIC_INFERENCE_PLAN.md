# GGUF Descriptorless Automatic Inference Plan

## Status

Planned architectural refactor.

Target repository: `celsowm/celeg`.

This document is a focused companion to:

- `ZERO_MODEL_FAMILIES_REFACTORING_PLAN.md`;
- `DESCRIPTORLESS_CHECKPOINT_INFERENCE_REFACTORING_PLAN.md`.

It addresses the remaining gap exposed by native GGUF checkpoints: CELEG's final architecture must be able to resolve GGUF models from checkpoint evidence without reintroducing model-family production code or disguising `dense_transformers.json` as C++.

---

# 1. Goal

GGUF is **not** incompatible with descriptorless inference.

The target is:

> Any GGUF checkpoint whose mathematics, tensor layouts, tokenizer behavior, interaction protocol, and multimodal primitives are already expressible by CELEG's semantic vocabulary must load without a model-family source change.

A previously unknown GGUF repository must not require:

```text
src/models/<family>/
include/celeg/models/<family>/
make_<family>_architecture()
make_<family>_vision_provider()
if (architecture == "...")
switch (model_type)
```

Adding support for a genuinely new mathematical primitive remains legitimate, but the implementation must represent the reusable primitive rather than the first model family that introduced it.

Examples:

```text
Good:  Mamba2
Good:  GatedDeltaNet
Good:  SlidingWindowAttention
Good:  OrthogonalizeCurrentValue
Good:  GroupedTopK
Bad:   Lfm2Layer
Bad:   QwenAttention
Bad:   NemotronRouter
```

---

# 2. Important limit

Descriptorless inference cannot make an unknown operation executable merely because GGUF describes its tensors.

The correct distinction is:

```text
new model composed of known CELEG semantics
    -> no model-family code

new reusable mathematical primitive
    -> add the primitive once to CELEG
    -> teach semantic inference how to recognize evidence for it
    -> implement/lower it in the relevant backends
```

Therefore the acceptance criterion is not:

> Every future GGUF must run without CELEG source changes.

It is:

> Every future GGUF composed entirely of already-supported CELEG semantics must be resolvable without family-specific source changes.

---

# 3. Why the current automatic path fails on native GGUF

The current automatic inference path is biased toward Hugging Face/Safetensors-style canonical names.

Typical Hugging Face metadata/tensor spellings include:

```text
hidden_size
num_hidden_layers
num_attention_heads
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight
```

Native GGUF commonly exposes equivalent facts using a format namespace and canonical GGUF tensor grammar, for example:

```text
<architecture>.embedding_length
<architecture>.block_count
<architecture>.attention.head_count
<architecture>.attention.head_count_kv
<architecture>.context_length

token_embd.weight
blk.0.attn_q.weight
blk.0.attn_k.weight
blk.0.attn_v.weight
blk.0.attn_output.weight
```

The architecture prefix is checkpoint metadata namespace, not permission to reintroduce family dispatch.

The automatic path must normalize these conventions into canonical facts before semantic synthesis.

---

# 4. Non-negotiable design rule: namespace is evidence, not behavior

A GGUF key such as:

```text
lfm2.embedding_length
```

may be normalized because GGUF defines the structural convention:

```text
*.embedding_length -> hidden_size observation
```

The prefix may be retained as provenance:

```text
source = "lfm2.embedding_length"
```

but must not select runtime behavior:

```cpp
// forbidden
if (architecture == "lfm2") {
    use_short_convolution_schedule();
}
```

Desired direction:

```text
GGUF key/value evidence
        +
GGUF tensor inventory
        +
shape constraints
        +
registered semantic rules
        |
        v
CanonicalModelFacts
```

A family/architecture string may identify the source namespace, but downstream behavior must be justified by reusable semantic evidence.

---

# 5. Desired GGUF resolution pipeline

```text
GGUF file
   |
   +--> GGUF metadata reader
   |
   +--> GGUF tensor inventory
   |
   +--> GGUF tokenizer metadata
   |
   v
CheckpointView
   |
   +---------------------------+
   |                           |
   v                           v
Metadata normalization     Tensor grammar matching
   |                           |
   +------------+--------------+
                |
                v
          InferenceInput
                |
      +---------+---------+
      |                   |
      v                   v
Fact proposal rules   Binding proposal rules
      |                   |
      +---------+---------+
                |
                v
        deterministic solvers
                |
                v
        CanonicalModelFacts
                |
      +---------+---------+
      |                   |
      v                   v
 GraphSynthesizer   WeightPlanSynthesizer
      |                   |
      +---------+---------+
                |
                v
           ResolvedModel
                |
          +-----+-----+
          |           |
          v           v
        CPU         CUDA
```

No stage after checkpoint normalization may require a model-family identity.

---

# 6. Phase 1 — Generic GGUF metadata normalization

Introduce or extend GGUF-aware normalization rules for structural metadata.

The rules must operate on semantic suffix/convention, not a hardcoded family list.

Examples:

```text
*.embedding_length              -> hidden_size
*.feed_forward_length           -> intermediate_size candidate
*.block_count                   -> physical/logical layer-count evidence
*.context_length                -> context_length
*.attention.head_count          -> query_heads
*.attention.head_count_kv       -> key_value_heads
*.attention.key_length          -> key width/head-dimension evidence
*.attention.value_length        -> value width/head-dimension evidence
*.rope.freq_base                -> rope_theta
*.attention.layer_norm_rms_epsilon
                                -> normalization epsilon
```

Exact GGUF field semantics must be implemented according to the format contract and validated against tensor evidence.

Rules must return observations/proposals with provenance, for example:

```cpp
InferenceProposal<int>{
    .value = 4096,
    .evidence = {{
        EvidenceKind::ExplicitMetadata,
        "<namespace>.embedding_length",
        "hidden_size"
    }},
    .rule_id = "gguf.embedding-length"
};
```

Do not silently let one alias overwrite another. Contradictory explicit metadata must fail closed.

---

# 7. Phase 2 — Native GGUF tensor grammars

CELEG must understand reusable GGUF tensor naming grammar directly.

Examples of generic binding proposals include:

```text
token_embd.weight
    -> TensorRole::TokenEmbedding

output.weight
    -> TensorRole::LanguageModelHead

output_norm.weight
    -> TensorRole::FinalNorm

blk.{layer}.attn_norm.weight
    -> TensorRole::AttentionInputNorm

blk.{layer}.attn_q.weight
    -> TensorRole::AttentionQuery

blk.{layer}.attn_k.weight
    -> TensorRole::AttentionKey

blk.{layer}.attn_v.weight
    -> TensorRole::AttentionValue

blk.{layer}.attn_output.weight
    -> TensorRole::AttentionOutput
```

Dense FFN, MoE, recurrent, convolutional, fused, stacked, and shared-expert patterns must each be represented by reusable grammar/layout rules.

A naming match is only a proposal.

Final binding requires agreement with:

```text
role grammar
+
shape constraints
+
layer schedule
+
metadata geometry
+
required role set
```

If more than one complete assignment remains equally valid, automatic inference must return `AmbiguousTensorBinding`.

---

# 8. Phase 3 — Infer layer schedules instead of assuming dense Transformer layers

The current descriptorless implementation must stop assuming that every logical layer is:

```text
Attention
+
Dense SwiGLU FFN
```

The tensor inventory must be used to build a per-layer evidence set.

Conceptually:

```text
layer 0:
  attention evidence
  dense FFN evidence

layer 1:
  convolution evidence
  dense FFN evidence

layer 2:
  Mamba2 evidence
  no generic attention tensors

layer 3:
  attention evidence
  MoE evidence
```

Inference then produces a semantic schedule:

```cpp
CanonicalLayerFacts {
    mixer = ...;
    feed_forward = ...;
    norms = ...;
    position = ...;
    state = ...;
};
```

The solver must distinguish:

```text
ordinary attention
sliding-window attention
short convolution
GatedDeltaNet
Mamba2
MLP-only blocks
Dense FFN
MixtureOfExperts
shared experts
```

using evidence, not architecture identity.

---

# 9. Phase 4 — Primitive inference rules

Automatic inference should be decomposed into independent rules rather than expanding `infer_canonical_model_facts()` into a larger monolith.

Desired extension shape:

```text
MetadataFactRule[]
LayerScheduleRule[]
AttentionSemanticsRule[]
PositionSemanticsRule[]
FeedForwardSemanticsRule[]
TensorBindingRule[]
ValidationRule[]
```

Each rule:

- observes immutable input;
- emits immutable proposals + evidence;
- does not mutate accepted state;
- does not depend on registration order;
- does not know CPU/CUDA;
- does not know repository names;
- does not branch on a model family.

`FactSolver` and `BindingSolver` remain the only components that accept/reject competing proposals.

This is both a GGUF requirement and an SRP/OCP correction.

---

# 10. Phase 5 — Position and normalization semantics

GGUF descriptorless resolution must infer semantic position policies independently from family identity.

Supported facts may include:

```text
RoPE
adjacent-pair RoPE
split-half RoPE
M-RoPE
scaled RoPE
sliding-window attention
relative position/bias
NoPE
```

Likewise normalization must resolve explicit reusable semantics such as:

```text
RMSNorm
LayerNorm
query/key normalization
pre/post attention norms
pre/post FFN norms
embedding post-norm
```

Unknown or contradictory semantics are load-time errors.

Do not default silently to a familiar architecture just because the tensor shapes resemble one.

---

# 11. Phase 6 — MoE and packed/fused layouts

GGUF support is incomplete unless automatic inference can recognize already-supported MoE semantics.

The canonical facts must distinguish at least:

```text
router score semantics
TopK vs GroupedTopK
normalization of selected expert scores
expert bias
routed scaling
number of experts
experts per token
shared expert presence
individual vs stacked vs fused expert payloads
expert tensor role layout
```

These are properties of the semantic program, not model-family names.

Backend capability validation remains inverted:

```text
semantic MoE requirements
        x
backend capabilities
        |
        v
accept/reject backend
```

Automatic inference must not claim `supports_cpu` or `supports_cuda` merely from checkpoint identity.

---

# 12. Phase 7 — Tokenizer and interaction remain separate concerns

Resolving the numerical `ModelGraph` is not sufficient to claim complete GGUF support.

GGUF tokenizer metadata must continue through the neutral tokenizer path:

```text
GGUF tokenizer metadata
      |
      v
TokenizerBehaviorFacts
      |
      v
TokenizerDefinition
      |
      v
ITokenizer
```

Chat/tool/vision behavior must likewise resolve through reusable interaction semantics.

The numerical architecture resolver must not become a god object that also decides:

```text
chat template
tool grammar
image preprocessing
video protocol
```

Those concerns converge at runtime composition only after each has been independently resolved.

---

# 13. `dense_transformers.json` migration rule

`descriptors/dense_transformers.json` must **not** be deleted merely to reach the target directory structure early.

Until descriptorless inference has full parity, treat the descriptor as a temporary migration oracle.

It is not the final architecture.

It must not be copied into:

```text
src/model/automatic_dense_transformers.cpp
src/model/gguf_known_models.cpp
src/model/family_rules.cpp
```

or disguised behind a renamed JSON/catalog.

The correct migration is:

```text
existing descriptor semantics
        |
        +--> golden/parity expectations
        |
        v
generic reusable inference rules
        |
        v
same canonical semantics
```

Deletion gate:

> Delete `dense_transformers.json` only after every currently descriptor-backed fixture that is expected to be auto-resolvable produces descriptorless semantic parity.

After deletion, add/retain a structural guardrail preventing a monolithic family descriptor from returning as the production resolution mechanism.

---

# 14. Descriptor vs descriptorless parity harness

Before deleting a descriptor-backed case, resolve the same checkpoint fixture by both paths:

```text
Path A: descriptor importer
Path B: automatic inference
```

Compare canonical results rather than textual implementation details.

Required comparison surface:

```text
RuntimeTopology / canonical dimensions
per-layer mixer schedule
per-layer FFN schedule
attention geometry
position semantics
normalization semantics
MoE semantics
TensorRoleBindings
WeightPlan
ModelGraph
semantic requirements
capabilities/provenance where source-independent
canonical semantic fingerprint
```

Expected source-specific differences such as resolution provenance may differ but must be explicit.

The test must fail on semantic drift.

---

# 15. First GGUF acceptance case

Use the currently failing native `Q4_K_M` GGUF case as the first end-to-end target.

The implementation sequence must be incremental:

1. read the failing checkpoint without requiring `dense_transformers.json` at bootstrap;
2. normalize its architecture-prefixed GGUF structural metadata generically;
3. resolve native GGUF tensor names into semantic role proposals;
4. infer its actual layer schedule;
5. solve all required tensor bindings;
6. synthesize the same backend-neutral model semantics produced by the descriptor path;
7. run existing repository tests;
8. run the model itself;
9. add the case to the permanent descriptor-vs-auto parity suite;
10. generalize every rule so it is named for the convention/semantic primitive, never for the model.

Do not proceed by inserting the model's architecture name into a generic switch.

---

# 16. Regression matrix

The descriptorless GGUF work must exercise progressively harder semantic compositions.

Minimum categories:

```text
A. ordinary dense attention + SwiGLU
B. GQA / MQA variants
C. alternate RoPE pairing/scaling
D. sliding-window/full-attention hybrid schedules
E. short-convolution hybrids
F. GatedDeltaNet/recurrent hybrids
G. Mamba2 hybrids
H. ordinary MoE
I. grouped routing/shared experts
J. fused/stacked expert layouts
K. repeated/physical-vs-logical layer schedules
L. multimodal text component with independent vision pipeline
```

The suite should prefer existing CELEG fixtures/checkpoints before inventing synthetic cases.

Each category must answer two different questions:

```text
Can checkpoint evidence resolve the semantic program?
Can the selected backend execute that semantic program?
```

Do not merge these into one capability decision.

---

# 17. SOLID constraints

## SRP

Keep these responsibilities separate:

```text
GGUF parsing
metadata normalization
tensor inventory
proposal production
fact solving
binding solving
graph synthesis
weight-plan synthesis
backend lowering
execution
```

`infer_canonical_model_facts()` must shrink as rule catalogs/solvers take ownership of independent concerns.

## OCP

Adding a new reusable GGUF naming convention or semantic rule should normally mean registering a rule, not editing a model-family switch.

Adding a genuinely new primitive may require new semantic and backend implementation, but existing model families must not be reopened merely because a new repository uses known semantics.

## LSP

All inference rules must obey the same deterministic proposal contract. A rule must not secretly mutate global resolution state or depend on execution order.

## ISP

Do not create a universal `IGgufModelResolver` or `IInferenceEverything`. Prefer narrow contracts for metadata facts, semantic facts, tensor bindings, and synthesis stages.

## DIP

Backend code depends on compiled semantic programs.

It must never depend on:

```text
GGUF architecture strings
model_type
repository name
descriptor ID
family identity
```

---

# 18. Static guardrails

Extend `scripts/check_architecture_boundaries.py` where useful to prevent regressions such as:

```text
family-name dispatch in automatic inference
family-prefixed factories
family production directories
architecture/model_type dispatch in backends
GGUF format checks leaking into execution code
monolithic known-model tables in runtime composition
```

Format-specific knowledge is allowed at the checkpoint/import boundary.

Semantic execution must remain format-neutral.

---

# 19. Completion criteria

This plan is complete when all of the following are true:

- native GGUF metadata aliases are normalized generically;
- native GGUF tensor grammars participate in tensor-role solving;
- automatic inference can represent heterogeneous layer schedules already supported by CELEG;
- dense, recurrent, convolutional, Mamba, MoE, fused/stacked layouts and position policies are inferred from semantic evidence where representable;
- no model-family production directory or runtime dispatch is needed;
- `dense_transformers.json` is no longer required for checkpoints whose semantics are inferable;
- descriptor-vs-auto parity coverage exists before descriptor deletion;
- the currently failing Q4_K_M acceptance model resolves and runs through the descriptorless path;
- CPU/CUDA continue to consume only backend-neutral compiled semantics;
- a new GGUF repository composed of known CELEG semantics requires zero family-specific source changes.

The architectural invariant is:

> GGUF describes evidence. CELEG resolves evidence into semantics. Backends execute semantics. Model families never become runtime architecture.
