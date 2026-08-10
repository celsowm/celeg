# Muse Glimmer Native Support Plan

## Status

Planned native-model support.

Target checkpoint family: `meta-models/Muse-Glimmer-30B`.

This plan is intentionally architectural rather than family-hardcoded. Muse Glimmer is the first acceptance model for several reusable semantic capabilities that CELEG does not yet represent completely. The implementation must add those capabilities to the backend-neutral model vocabulary and then describe Muse Glimmer using them.

This plan should be read together with `DESCRIPTORLESS_CHECKPOINT_INFERENCE_REFACTORING_PLAN.md`: any new primitive introduced here must remain discoverable by the future generic checkpoint-inference path and must not depend on a Muse-specific execution branch.

---

# 1. Objective

Add first-class, numerically faithful support for Muse Glimmer while preserving the current CELEG architecture:

```text
checkpoint metadata + tensors
          |
          v
architecture/descriptor resolution
          |
          v
backend-neutral semantic graph + weight plan
          |
     +----+----+
     |         |
     v         v
    CPU       CUDA
```

The final implementation must support the Muse Glimmer text model without introducing a model-specific forward pass.

Multimodal vision/video support and DFlash speculative decoding are deliberately separate milestones because they are separate responsibilities with different change reasons.

The desired result is:

```text
Muse-specific knowledge
    = metadata/tensor spelling + chat/multimodal adapter details

Reusable CELEG semantics
    = norms + gated attention + per-layer positional policy
      + attention schedule + embedding transforms + ordinary transformer graph

Backend implementation
    = only reusable semantic operations
```

---

# 2. Non-negotiable SOLID rules

## 2.1 Single Responsibility Principle

Each new component must have one reason to change.

Examples:

- norm semantics belong in the graph/model vocabulary;
- checkpoint tensor spelling belongs in bindings/descriptors/inference rules;
- CPU execution belongs in CPU operators/lowering;
- CUDA execution belongs in CUDA operators/lowering;
- image preprocessing belongs in a visual input/provider layer;
- Muse chat formatting belongs in a chat profile;
- speculative decoding policy belongs in a speculative-decoding subsystem.

Forbidden:

```cpp
if (architecture == "muse_glimmer") {
    // special norm
    // special gate
    // special RoPE
    // special tensor lookup
}
```

Required direction:

```text
Muse checkpoint
   |
   v
reusable semantic specs
   |
   v
generic compiler/lowering
```

## 2.2 Open/Closed Principle

After this work, adding another model that uses any of these features must require registration/configuration rather than modification of central model-family switches:

- centered RMSNorm;
- weightless RMSNorm;
- a separate attention-output gate projection;
- layer-specific NoPE/RoPE;
- mixed sliding/full attention schedules.

Adding a new architecture should preferentially add a descriptor/provider/rule and tests.

## 2.3 Liskov Substitution Principle

A semantic primitive must mean the same thing regardless of who requested it.

For example, an `AttentionOutputGateSpec` must not mean one formula for Muse and another formula for a future model. If different formulas exist, represent different specs.

CPU and CUDA lowerings must accept the same validated graph contract and produce equivalent semantics.

## 2.4 Interface Segregation Principle

Do not create a giant `IMuseGlimmerProvider` or a generic interface that forces unrelated capabilities onto implementers.

Prefer narrow extension points such as:

```text
architecture resolution
chat profile
visual embedding provider
speculative decoding strategy
```

A text-only runtime must not depend on vision or DFlash interfaces.

## 2.5 Dependency Inversion Principle

High-level model semantics must not depend on CPU/CUDA implementation details.

The graph describes what must happen. Backends decide how to lower it.

No descriptor, architecture provider, chat profile, tokenizer module, or vision provider may include CUDA kernel assumptions.

No CPU/CUDA executor may inspect `model_type`, repository name, architecture ID, or Hugging Face tensor names to select Muse behavior.

---

# 3. What Muse Glimmer exercises

Muse Glimmer is useful as an architecture test because most of its text backbone is conventional while a small number of semantic details are easy to model incorrectly.

The relevant text semantics are:

1. dense decoder-only Transformer;
2. GQA attention;
3. a mixed per-layer attention schedule with sliding-window and full-context layers;
4. Q/K normalization;
5. a learned attention-output gate computed from the pre-attention hidden state and applied as a sigmoid multiplier before `o_proj`;
6. per-layer positional policy where some layers use RoPE and others use NoPE;
7. centered RMSNorm on decoder normalization boundaries;
8. a weightless RMSNorm immediately after token embedding;
9. SwiGLU feed-forward blocks;
10. final RMSNorm and normal LM-head projection.

The implementation must derive checkpoint-specific dimensions, layer counts, token IDs, schedule values, context size, window size, scaling values, and tensor shapes from checkpoint evidence. They must not be duplicated as CELEG source constants unless a constant is genuinely part of a stable mathematical convention.

---

# 4. Current CELEG fit

CELEG already has most of the necessary semantic vocabulary:

- `AttentionSpec`;
- GQA geometry;
- `SlidingWindowPattern` and `FullCausalPattern`;
- `NoPositionEncodingSpec` and `RopePositionSpec`;
- per-layer graph construction;
- Q/K normalization support;
- SwiGLU;
- four normalization boundaries in `LayerSpec`;
- CPU/CUDA compilation from a backend-neutral graph;
- declarative descriptors and tensor-role bindings;
- multimodal prompt embedding abstractions;
- separate visual provider registration;
- chat-profile registration.

The gaps are therefore semantic deltas, not justification for a new model-specific forward implementation.

---

# 5. Architectural gaps to close

## 5.1 Norm semantics must become explicit

Current `NormSpec` mainly carries epsilon. Muse requires multiple RMSNorm weight conventions:

```text
ordinary RMSNorm:
    y = rms(x) * weight

centered RMSNorm:
    y = rms(x) * (1 + weight)

weightless RMSNorm:
    y = rms(x)
```

Represent the mathematical distinction explicitly.

Suggested direction:

```cpp
enum class NormWeightKind : uint8_t {
    Scale,
    OnePlusScale,
    None,
};

struct NormSpec {
    float epsilon = 0.0f;
    NormWeightKind weight_kind = NormWeightKind::Scale;
};
```

Naming is flexible; semantics are not.

Do not introduce `bool muse_centered` or infer the norm formula from a tensor name.

### Validation

- `None` must not require a norm weight tensor;
- `Scale` and `OnePlusScale` require a correctly shaped weight;
- CPU and CUDA must implement identical math;
- existing models must preserve their current norm semantics explicitly or via one validated default during migration.

## 5.2 Embedding post-transform must be graph-owned

Muse normalizes token embeddings with a weightless RMSNorm before entering layer 0.

This is not a tokenizer concern and should not be hidden inside a Muse loader.

Introduce a backend-neutral embedding transform/pipeline, for example:

```cpp
struct EmbeddingTransformSpec {
    std::optional<NormSpec> post_norm;
    float multiplier = 1.0f;
};
```

or an equivalent extensible representation.

The important boundary is:

```text
TokenEmbedding -> semantic embedding transform -> first decoder layer
```

The compiled backends consume this semantic operation. The architecture layer only describes it.

## 5.3 Separate attention-output gate projection

Muse stores its gate as a projection independent of `q_proj` and computes conceptually:

```text
gate = sigmoid(W_gate * hidden_before_attention)
attended = Attention(Q, K, V)
gated = attended * gate
output = W_o * gated
```

CELEG already has query-gating machinery, but a fused query/gate projection and an independently stored gate tensor are checkpoint-layout concerns and must not be conflated.

Add a reusable semantic representation for the gate source.

Suggested direction:

```cpp
enum class AttentionGateKind : uint8_t {
    None,
    Sigmoid,
};

struct AttentionOutputGateSpec {
    AttentionGateKind kind = AttentionGateKind::None;
    // source semantics indicate that the gate is projected from the
    // layer input, not derived from attention output.
};
```

Add a separate tensor role:

```text
TensorRole::AttentionGate
```

The descriptor/binding layer maps Muse's `gate_proj.weight` to this role.

### Important

If another model packs Q and gate in one checkpoint tensor, that packing belongs in weight-layout/binding resolution. The semantic graph should still describe the same gate operation.

Do not force one physical checkpoint layout into the semantic API.

## 5.4 Per-layer position policy

Muse uses RoPE in selected layers and NoPE in others.

CELEG already has `NoPositionEncodingSpec`, so this work should extend descriptor/inference synthesis to populate a `PositionSpec` per attention layer from checkpoint evidence.

Desired graph result:

```text
layer N AttentionSpec.position = RopePositionSpec{...}

or

layer N AttentionSpec.position = NoPositionEncodingSpec{}
```

Do not encode NoPE as `rope_theta = 0` inside backend kernels. Translate that checkpoint representation once at the architecture boundary into the semantic NoPE variant.

## 5.5 Attention schedule remains declarative

The mixed local/global schedule must remain data-driven.

The descriptor should translate the checkpoint layer schedule into:

```text
SlidingWindowPattern{window}
FullCausalPattern{}
```

per layer.

CPU/CUDA must never contain layer-index modulo rules such as `layer % 4` for Muse.

If the checkpoint supplies an explicit schedule, consume it. A repeated pattern may be used as an import rule only when it is an official, validated checkpoint convention and the resolved graph still contains the complete explicit per-layer result.

## 5.6 Q/K normalization must preserve Muse numerics

Muse applies weightless RMS normalization to Q and K and then applies its configured Q scale.

Do not reuse a weighted-QK-norm path by passing dummy weight tensors.

Represent whether Q/K norm carries learned scale independently from the general decoder norm policy if necessary.

If the existing `query_key_norm` boolean is not expressive enough, evolve it into a semantic Q/K norm spec rather than adding a family flag.

---

# 6. Phase 0 - Freeze the reference

Before changing execution code, create a reproducible parity fixture against the official Hugging Face Transformers implementation.

## Deliverables

- capture the exact model revision used for acceptance;
- record relevant `config.json` fields;
- record the resolved layer attention/position schedule;
- enumerate tensor names/shapes for a representative layer and special tensors;
- add a small deterministic prompt fixture;
- capture intermediate reference values where practical:
  - post-embedding vector;
  - first-layer norm output;
  - Q/K sample values;
  - attention gate sample values;
  - first-layer output;
  - final hidden state slice;
  - logits slice/top tokens.

For multimodal work later, add a separate fixed image fixture and reference embedding/logit outputs.

## Acceptance

No implementation phase is considered numerically complete using text-generation appearance alone. It must have measurable parity against the reference.

---

# 7. Phase 1 - Generalize normalization semantics

## Work

1. evolve `NormSpec` into an explicit mathematical contract;
2. update graph validation;
3. update weight planning so weightless norms do not request tensors;
4. update CPU norm lowering;
5. update CUDA norm lowering;
6. migrate current models without changing their outputs;
7. add unit tests for all supported norm weight conventions.

## Tests

```text
ordinary RMSNorm golden
centered RMSNorm golden
weightless RMSNorm golden
CPU == scalar reference
CUDA == scalar reference
existing architecture regression suite unchanged
```

## SOLID gate

This phase fails review if the executor must know which model requested the norm.

---

# 8. Phase 2 - Add semantic embedding transforms

## Work

Introduce an explicit post-embedding operation capable of representing Muse's weightless normalization.

Compilation should lower:

```text
embedding lookup
    -> optional post norm
    -> existing embedding multiplier/other declared transforms
    -> decoder graph
```

Ordering must be explicit and validated.

## Tests

- no-transform path remains bit/numerically equivalent where applicable;
- weightless RMSNorm embedding fixture;
- CPU/CUDA parity;
- invalid transform combinations fail during validation/compilation, not during token execution.

## SOLID gate

Embedding lookup must not absorb arbitrary model-family preprocessing. The transform remains a separate semantic responsibility.

---

# 9. Phase 3 - Generalize attention-output gating

## Work

1. introduce an explicit attention gate semantic spec;
2. add `TensorRole::AttentionGate`;
3. extend descriptor tensor bindings;
4. extend weight plan validation and compiled weight structures;
5. lower the separate gate projection in CPU;
6. lower it in CUDA;
7. reuse/factor the existing sigmoid elementwise gate implementation;
8. keep fused physical layouts possible through binding/layout adapters rather than semantic branching.

Correct operator ordering:

```text
normalized layer input
   |\
   | +--> gate projection --> sigmoid -------+
   |                                        |
   +--> Q/K/V --> attention --> concat ------*--> o_proj
```

The gate projection consumes the same pre-attention hidden representation expected by the model.

## Tests

- gate-disabled attention regression;
- separate-gate synthetic test;
- fused-vs-separate physical layout equivalence if both are supported;
- CPU/CUDA parity;
- compiler rejects missing `AttentionGate` tensor when gate semantics require it.

## SOLID gate

A future model with the same sigmoid gate but different tensor names must require only a different binding rule/descriptor entry.

---

# 10. Phase 4 - Per-layer NoPE/RoPE and schedule synthesis

## Work

1. ensure descriptor parsing can obtain the layer-specific position schedule;
2. translate checkpoint sentinel values such as a zero theta into `NoPositionEncodingSpec` at resolution time;
3. synthesize each layer's `AttentionSpec.position` independently;
4. preserve sliding/full attention selection as a separate dimension from position encoding;
5. validate schedule length against physical/logical layer count;
6. reject incomplete, contradictory, or unsupported schedules.

Important orthogonality:

```text
attention pattern != position encoding

Sliding + RoPE     valid
Sliding + NoPE     semantically representable
Full + RoPE        valid
Full + NoPE        valid
```

Do not introduce assumptions that global attention implies NoPE or that sliding attention implies RoPE unless the descriptor explicitly supplies both facts for a checkpoint.

## Tests

Build synthetic schedules covering every pair above, then add the real Muse resolved schedule fixture.

## SOLID gate

Pattern and positional policy remain independent semantic axes with independent reasons to change.

---

# 11. Phase 5 - Muse Glimmer text descriptor/provider

Only after the reusable primitives exist should Muse itself be registered.

## Descriptor responsibilities

The Muse descriptor/import path may know:

- `model_type` / architecture probe spelling;
- where text configuration fields live;
- checkpoint tensor names;
- token IDs;
- layer schedule source fields;
- RoPE/NoPE source representation;
- gate tensor spelling;
- norm epsilon fields;
- attention scaling fields;
- tied/untied embedding metadata;
- chat profile ID.

It must not know CPU or CUDA execution mechanics.

## Expected tensor bindings

Add the exact official names after Phase 0 inventory, conceptually including:

```text
TokenEmbedding
LanguageModelHead
FinalNorm
AttentionInputNorm
AttentionQuery
AttentionKey
AttentionValue
AttentionGate
AttentionOutput
AttentionPostNorm
FfnInputNorm
FfnGate
FfnUp
FfnDown
FfnOutputNorm
```

The embedding post-norm is weightless and therefore intentionally has no tensor role.

## Descriptor test

The resolved `ModelGraph` is the golden artifact.

Assert semantic facts rather than implementation details:

- correct layer count;
- correct attention geometry;
- correct local/global pattern per layer;
- correct RoPE/NoPE position variant per layer;
- correct norm weight convention at every boundary;
- embedding post-norm enabled and weightless;
- sigmoid attention gate enabled;
- correct FFN activation;
- correct tensor roles and shapes;
- no architecture-family branch required by CPU/CUDA.

---

# 12. Phase 6 - Text numerical parity

## Layer-by-layer parity

Use the frozen Transformers reference and compare progressively:

```text
embedding lookup
post-embedding norm
layer 0 input norm
Q/K/V projections
Q/K norm + scaling
RoPE or NoPE result
attention probabilities/output
gate projection + sigmoid
post-attention norm/residual
FFN input norm
SwiGLU output
post-FFN norm/residual
selected later local layer
selected global/NoPE layer
final norm
LM logits
```

Comparing only final generated text makes localization of a subtle semantic mismatch unnecessarily expensive.

## Precision policy

Define tolerances per dtype/backend in the test harness. Do not loosen a single global tolerance until the test passes.

A mismatch between CPU and CUDA must be treated as a backend bug unless reference evidence proves a supported precision difference.

## Completion criteria

Text support is native when:

1. official Safetensors checkpoint resolves without model-specific runtime branches;
2. CPU compiles and executes the semantic graph;
3. CUDA compiles and executes the same graph;
4. reference parity tests pass within documented tolerances;
5. deterministic generation agrees with the reference at the configured acceptance precision;
6. no vision dependency is required for text-only load/run when the checkpoint packaging allows text weights to be resolved independently.

---

# 13. Phase 7 - Chat/tokenizer integration

Keep chat handling out of the model graph.

## Work

- register a Muse chat profile matching the official template;
- confirm special-token behavior;
- ensure text-only completion remains usable independently of chat formatting;
- add template golden tests;
- add tool/reasoning conventions only if the official model defines them and only through chat/protocol abstractions.

## SOLID gate

Changing a prompt template must never require touching attention, graph compilation, or backend code.

---

# 14. Phase 8 - Native vision/image support

Multimodal support is a separate subsystem and should follow the existing `IVisualEmbeddingProvider` style rather than entering the text model implementation.

Suggested module:

```text
src/models/muse_glimmer/vision.*
```

or its future architecture-neutral equivalent if common ViT pieces are extracted first.

## Responsibilities

### Image preprocessing

Own:

- decode/normalization;
- aspect-ratio-aware resizing;
- patchification;
- spatial/temporal patch layout;
- position interpolation inputs;
- merge/grid bookkeeping.

### Vision encoder

Own only visual-network semantics:

- patch projection;
- positional representation/interpolation;
- vision attention;
- vision MLP;
- visual normalization;
- pixel shuffle/merge/projector stages as required by the official architecture.

### Multimodal bridge

Produce the existing neutral visual embedding representation consumed by prompt prefill.

The language backend should continue seeing prompt embedding replacements rather than Muse image objects.

## Reuse before duplication

Compare Muse vision operations with the current Gemma 4 and Qwen 3.5 providers. Extract reusable image/ViT primitives when two or more providers truly share semantics.

Do not prematurely create a universal vision abstraction that merely hides three family-specific implementations behind switches.

## Tests

- preprocessing golden fixture;
- patch/grid count fixture;
- vision-encoder intermediate reference;
- projected embedding parity;
- end-to-end image prompt logits;
- multiple image placements;
- malformed input validation.

---

# 15. Phase 9 - Video support

Video should build on reusable image/vision primitives without making the image provider responsible for temporal sampling policy.

Separate responsibilities:

```text
video decode/sample
      |
      v
temporal/spatial patchification
      |
      v
shared Muse vision encoder
      |
      v
neutral prompt embeddings
```

Test temporal padding, sampling cadence, grid bookkeeping, placeholder count, and mixed text/image/video prompts independently.

Video support is not a prerequisite for declaring the text model native.

---

# 16. Phase 10 - DFlash / speculative decoding

Do not overload CELEG's MTP implementation just because both features accelerate decoding.

First model DFlash at the algorithmic contract level:

```text
draft/proposal producer
verification against target model
accept/reject policy
state/cache synchronization
metrics
```

Then determine which portions can reuse CELEG's existing speculative/MTP infrastructure.

Preferred design:

```cpp
class ISpeculativeStrategy {
    ... narrow strategy contract ...
};
```

only if the existing code demonstrates that multiple strategies need a stable abstraction. Do not introduce the interface speculatively before auditing the current MTP boundaries.

DFlash support must be optional. The base Muse model must decode correctly without it.

## SOLID gate

A performance feature must never become required for semantic correctness.

---

# 17. Descriptorless-inference compatibility

Every primitive added for Muse must be representable in the future `CanonicalModelFacts` path.

Candidate reusable inference facts include:

```text
norm weight convention
embedding post norm
attention gate kind
attention gate source/binding
per-layer attention pattern
per-layer position encoding
Q/K norm convention
```

The Muse descriptor can provide these facts initially. Later, generic inference rules may infer them from metadata/tensor evidence without changing graph synthesis or backend execution.

This is an explicit acceptance requirement: no Muse implementation should create a dead-end representation that the descriptorless architecture would later need to bypass.

---

# 18. Backend implementation rules

## CPU

CPU lowering may branch on semantic variants such as:

```text
NormWeightKind
AttentionGateKind
PositionSpec alternative
AttentionPatternSpec alternative
```

It may not branch on Muse identity.

## CUDA

Same rule.

Kernel specialization is allowed when driven by semantics/shape/dtype and selected by compiler capability, not architecture name.

Potential fusion examples:

```text
RMSNorm + projection
attention output + sigmoid gate
post-attention norm + residual
```

Fusion is a later optimization. First establish simple, readable semantic parity.

Do not encode a fused kernel requirement into the graph.

---

# 19. Error model

Unsupported or malformed Muse checkpoints must fail closed during load/compile with actionable errors.

Examples:

```text
Muse descriptor requires attention gate tensor for layer 17
layer position schedule has 51 entries but resolved graph has 52 layers
centered RMSNorm requested without required weight tensor
sliding attention selected but sliding window is invalid
unknown Muse attention schedule value: ...
vision checkpoint tensors are incomplete
```

Avoid errors such as:

```text
unsupported Muse model
bad config
kernel failed
```

The error should identify the semantic invariant, not merely the family.

---

# 20. Testing strategy

## Unit tests

Cover every reusable primitive independently:

- norm variants;
- embedding transforms;
- separate attention gate;
- position variants;
- pattern schedule parsing;
- tensor-role validation.

## Graph-resolution tests

Muse-specific tests verify checkpoint evidence is translated into the correct neutral graph.

## Backend compiler tests

Feed synthetic semantic graphs directly into CPU/CUDA compilers. These tests should not need Muse metadata.

## Numerical integration tests

Use the official reference fixture.

## Regression tests

Existing Gemma, Qwen, Granite, LFM, MoE, recurrent/hybrid, M-RoPE, packed execution, and other architecture tests must remain green.

A new generic primitive is not complete if it fixes Muse by silently changing the meaning of an existing graph.

---

# 21. Suggested implementation order

```text
0. freeze official reference + tensor inventory
1. explicit norm semantics
2. embedding post-transform
3. separate attention-output gate role/spec
4. per-layer position policy import
5. Muse text descriptor
6. text CPU parity
7. text CUDA parity
8. chat/tokenizer integration
9. image vision provider
10. video path
11. DFlash/speculative decoding
12. performance/fusion work
13. descriptorless inference rules for reusable Muse semantics
```

Do not begin with the vision encoder or speculative decoding before the text backbone reaches deterministic parity.

---

# 22. Candidate file impact

Exact paths may evolve during implementation; responsibilities should not.

Likely core files:

```text
include/celeg/model/graph.hpp
include/celeg/model/weights/roles.hpp
src/model/descriptor/architecture.cpp
src/model/descriptor/parser.cpp
src/model/descriptor/graph_builder.cpp
src/model/graph_builder.cpp
src/backend/cpu/operators/*
src/backend/cpu/model_forward_*.cpp
src/backend/cuda/compiler.cpp
src/backend/cuda/model/*
src/backend/cuda/kernels/*
descriptors/*.json
src/composition/builtin_runtime.cpp
```

Likely new family adapters:

```text
src/models/muse_glimmer/chat_template.cpp
include/celeg/models/muse_glimmer/chat_template.hpp
src/models/muse_glimmer/vision.cpp
include/celeg/models/muse_glimmer/vision.hpp
```

A family adapter should contain only unavoidable family-specific integration. If text execution code starts accumulating there, revisit the semantic model before continuing.

---

# 23. Definition of native support

## Tier A - Native text

Complete when:

- official text weights load directly;
- complete semantic graph resolves;
- CPU works;
- CUDA works;
- reference numerical tests pass;
- generation works;
- chat profile works;
- no Muse-specific backend branch exists.

## Tier B - Native image multimodal

Tier A plus:

- official image preprocessing;
- native vision encoder/projector;
- prompt embedding injection;
- reference image parity.

## Tier C - Native video multimodal

Tier B plus official video preprocessing/temporal path and parity.

## Tier D - Native accelerated Muse

Tier C plus optional DFlash/speculative decoding with correctness and performance tests.

The README/support matrix should distinguish these tiers rather than treating a single `Muse Glimmer supported` boolean as sufficient.

---

# 24. SOLID review checklist for every Muse commit

Before accepting each implementation commit, answer all of these:

### SRP

- Does each changed class/module have one clear reason to change?
- Did a loader/provider accidentally start executing model math?
- Did a backend start parsing checkpoint metadata?

### OCP

- Could another architecture request the new semantic primitive without editing a family switch?
- Is model-specific behavior registered/described rather than centrally hardcoded?

### LSP

- Does the primitive mean the same thing on CPU and CUDA?
- Can any valid instance of the semantic spec be substituted without architecture-specific assumptions?

### ISP

- Did we add a narrow capability or a broad interface with irrelevant methods?
- Can text-only builds/use remain independent of vision/video/DFlash concerns?

### DIP

- Does the model graph depend only on backend-neutral contracts?
- Do high-level model modules avoid CPU/CUDA dependencies?
- Do backends avoid architecture and repository identities?

### DRY companion rule

- Is mathematical behavior implemented once per backend primitive rather than once per model family?
- Are tensor-name differences data/binding rules instead of duplicated lookup code?
- Are shared vision operations extracted only when they are semantically identical?

Any `if (muse_glimmer)` outside an import/registration/chat/vision integration boundary is a design smell requiring explicit justification.

---

# 25. Final acceptance checklist

The plan is complete only when all relevant items are true:

- [ ] official revision/reference frozen;
- [ ] tensor inventory captured;
- [ ] centered RMSNorm is a generic semantic primitive;
- [ ] weightless RMSNorm is a generic semantic primitive;
- [ ] post-embedding norm is graph-owned;
- [ ] separate sigmoid attention gate is a generic semantic primitive;
- [ ] `AttentionGate` has a real tensor role/binding;
- [ ] per-layer NoPE/RoPE resolves semantically;
- [ ] sliding/full pattern resolves independently of position policy;
- [ ] Muse text descriptor/provider exists;
- [ ] graph golden tests pass;
- [ ] CPU primitive tests pass;
- [ ] CUDA primitive tests pass;
- [ ] text reference parity passes;
- [ ] chat/tokenizer integration passes;
- [ ] no Muse branch exists in CPU/CUDA execution;
- [ ] existing architecture regression suite remains green;
- [ ] image support has its own provider and parity tests;
- [ ] video support is isolated from text semantics;
- [ ] DFlash remains optional and strategy-separated from base correctness;
- [ ] new primitives are compatible with descriptorless checkpoint inference;
- [ ] final SOLID review finds no family-specific execution leakage.

---

# 26. Architectural success criterion

Muse Glimmer support is successful not merely when the model produces good text.

It is successful when the implementation demonstrates that CELEG can absorb a newly released architecture by extending its **semantic vocabulary** in small reusable pieces, while the core execution architecture stays closed to model-family modifications.

The strongest success signal is that the next model using centered norms, NoPE layers, separate gated attention, or the same multimodal building blocks becomes materially cheaper to support than Muse Glimmer itself.
