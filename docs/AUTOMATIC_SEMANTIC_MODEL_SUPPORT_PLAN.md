# Automatic Semantic Model Support Plan

## Status

Proposed implementation plan.

## Motivation

CELEG should support checkpoints because their executable semantics can be
represented by the canonical model graph, not because the project recognizes a
checkpoint family name.

`flwrlabs/Lizzy-7B` is a useful motivating checkpoint because it combines
several semantics that CELEG already represents, or is close to representing:

- dense decoder layers;
- RMSNorm;
- gated/SwiGLU feed-forward blocks;
- query/key normalization;
- RoPE with YaRN scaling;
- a per-layer attention schedule that alternates sliding-window and full causal
  attention;
- post-attention and post-MLP normalization.

The desired result is **not** "add Lizzy support". The desired result is:

> Any checkpoint that exposes the same structural and numerical facts should
> resolve to the same canonical `ModelGraph`, regardless of `model_type`,
> `general.architecture`, repository name, or model-family identity.

This plan extends the policy already documented in
`docs/EXTENDING_ARCHITECTURES.md`: identity strings are provenance and
namespace information only; they must not select runtime mathematics.

---

## Primary architectural rule

The following must remain true throughout the implementation:

```text
checkpoint
    -> normalized metadata + tensor inventory
    -> canonical semantic facts
    -> ModelGraph + resolved tensor roles
    -> backend-neutral compiled program
    -> CPU / CUDA / future backends
```

No execution path may become:

```text
checkpoint
    -> model_type == "lizzy"
    -> Lizzy-specific graph/runtime path
```

The checkpoint identity may be retained for diagnostics, provenance, logging,
or as a metadata namespace. It must not change the resulting mathematics.

### Forbidden implementation patterns

Do not introduce any of the following:

```cpp
if (model_type == "lizzy") { ... }
if (architecture == "lizzy") { ... }
switch (model_family) { case Lizzy: ... }
```

Do not introduce:

- `lizzy.json`;
- `LizzyArchitecture`;
- `LizzyDescriptor`;
- a Lizzy-specific tensor map;
- a Lizzy-specific backend branch;
- a family registry whose entries select semantics;
- compatibility fallbacks keyed by repository/model names.

A conditional is acceptable when it tests a **semantic property**, for example:

```cpp
if (layer_pattern == AttentionPatternKind::SlidingWindow) { ... }
if (norm_topology.after.has_value()) { ... }
```

---

## What Lizzy exposes semantically

The current Lizzy configuration can be understood as a composition of generic
facts rather than as a new architecture family.

At a high level, the relevant facts are:

```text
hidden size:          4096
layer count:          32
attention heads:      32
KV heads:             32
head dimension:       128
normalization:        RMSNorm
feed-forward:         gated/SwiGLU
attention Q/K norm:   enabled
position encoding:    RoPE + YaRN
attention schedule:   sliding, sliding, sliding, full, repeated
sliding window:       4096
block norm topology:  post-attention norm + post-MLP norm
```

The important part is not any exact value above. The important part is that
each value or topology choice should map to a reusable semantic fact already in
or added to CELEG's canonical vocabulary.

The same implementation must therefore support a hypothetical checkpoint with:

```text
model_type = "completely_unknown_name"
```

when all structural evidence is otherwise equivalent.

---

## Current CELEG state

CELEG is already close to the target design.

### Automatic architecture resolution is already identity-independent

`AutomaticArchitecture` probes structural metadata and then calls the canonical
inference pipeline. For GGUF it builds metadata keys from
`metadata.architecture_type()` plus semantic suffixes. This is the correct use
of an architecture string: it identifies the metadata namespace, not the
runtime behavior.

Conceptually:

```cpp
metadata.architecture_type() + ".embedding_length"
metadata.architecture_type() + ".block_count"
metadata.architecture_type() + ".attention.head_count"
```

A GGUF declaring `general.architecture = "lizzy"` should therefore not require
CELEG to contain the word `lizzy` anywhere.

### The graph already represents the attention pattern

`AttentionPatternSpec` already includes:

```cpp
FullCausalPattern
SlidingWindowPattern
```

and `AttentionSpec` already represents query/key normalization and position
semantics. Therefore an alternating local/full schedule is not a new backend
architecture. It is a per-layer composition of existing attention primitives.

### Tensor roles already distinguish norm responsibilities

The weight-role vocabulary already includes roles such as:

```text
AttentionInputNorm
AttentionQueryNorm
AttentionKeyNorm
AttentionPostNorm
FfnInputNorm
FfnOutputNorm
```

This is the correct abstraction boundary. Source tensor names should be bound
to these roles during resolution so that backends never inspect model-family
spellings.

---

## Main semantic gap: normalization placement

The most important blocker is not Lizzy-specific metadata. It is the fact that
CELEG's current layer representation and execution still assume a pre-mixer
normalization strongly enough that a pure post-norm block cannot be represented
cleanly.

The current graph contains several partially overlapping fields such as an
unconditional `operator_norm` plus optional pre/post normalization fields. CPU
execution also applies the operator norm before the mixer unconditionally.

This makes a topology such as:

```text
x
 -> attention(x)
 -> norm
 -> residual add
 -> MLP
 -> norm
 -> residual add
```

impossible to express without pretending that a pre-norm exists.

The fix should generalize normalization placement itself.

---

# Target semantic model

## 1. Represent norm placement symmetrically

Replace mandatory pre-normalization assumptions with an explicit before/after
representation for each sublayer.

A target shape is:

```cpp
struct SublayerNormSpec {
    std::optional<NormSpec> before;
    std::optional<NormSpec> after;
};

struct LayerSpec {
    SublayerNormSpec mixer_norm;
    MixerSpec mixer;

    SublayerNormSpec feed_forward_norm;
    FeedForwardSpec feed_forward;

    ResidualSpec residual;
    float layer_scalar = 1.0f;
};
```

The exact naming may change during implementation, but the semantic invariant
must remain: **absence is represented by absence**, not by a dummy norm, a
boolean plus payload, or a family-specific exception.

This allows CELEG to represent all of the following without an architecture
name.

### Conventional pre-norm

```cpp
mixer_norm.before = RMSNorm(...);
mixer_norm.after  = std::nullopt;

feed_forward_norm.before = RMSNorm(...);
feed_forward_norm.after  = std::nullopt;
```

### Pure post-norm

```cpp
mixer_norm.before = std::nullopt;
mixer_norm.after  = RMSNorm(...);

feed_forward_norm.before = std::nullopt;
feed_forward_norm.after  = RMSNorm(...);
```

### Sandwich norm

```cpp
mixer_norm.before = RMSNorm(...);
mixer_norm.after  = RMSNorm(...);
```

No enum such as `DecoderPostNorm`, `LizzyBlock`, or `OlmoBlock` should survive
into the canonical graph if the same behavior can be expressed by composition.
Source vocabulary belongs in normalization/import logic; canonical semantics
should contain only executable facts.

---

## 2. Make backend execution follow the graph

CPU, CUDA, and future backends should execute the same generic sequence:

```cpp
copy_residual(hidden);

if (layer.mixer_norm.before)
    apply_norm(hidden, *layer.mixer_norm.before);

execute_mixer(...);

if (layer.mixer_norm.after)
    apply_norm(mixer_output, *layer.mixer_norm.after);

residual_add(...);

if (!std::holds_alternative<std::monostate>(layer.feed_forward)) {
    copy_residual(hidden);

    if (layer.feed_forward_norm.before)
        apply_norm(hidden, *layer.feed_forward_norm.before);

    execute_feed_forward(...);

    if (layer.feed_forward_norm.after)
        apply_norm(ffn_output, *layer.feed_forward_norm.after);

    residual_add(...);
}
```

The precise buffers depend on the backend, but the ordering must come entirely
from canonical semantics.

Backend code must never ask which architecture produced the layer.

---

## 3. Normalize layer-scoped categorical metadata

CELEG already supports layer-scoped numeric metadata. Extend the same concept
to categorical schedules rather than creating architecture-family resolvers.

The normalized metadata layer should be able to represent facts equivalent to:

```text
attention pattern per layer
normalization topology per layer
```

For example:

```cpp
enum class AttentionPatternKind {
    FullCausal,
    SlidingWindow,
};

LayerScopedValue<AttentionPatternKind> attention_pattern;
```

The enum above is illustrative; directly normalizing into canonical variants is
also acceptable and may be preferable.

Source fields that can provide evidence include generic configuration concepts
such as:

```text
layer_types
layer_layouts
sliding_window
use_pre_attn_norm
use_post_attn_norm
use_pre_mlp_norm
use_post_mlp_norm
```

These names are **source aliases**, not canonical semantics. They should be
consumed in metadata normalization and converted into the canonical graph.

### Example normalization

A source schedule equivalent to:

```json
{
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention"
  ],
  "sliding_window": 4096
}
```

should resolve to:

```cpp
layers[0].attention.pattern = SlidingWindowPattern{4096};
layers[1].attention.pattern = SlidingWindowPattern{4096};
layers[2].attention.pattern = SlidingWindowPattern{4096};
layers[3].attention.pattern = FullCausalPattern{};
```

and the same rule repeats for any schedule length supplied by the checkpoint.

The resolver must validate that a per-layer schedule has exactly
`layer_count` entries. Incomplete or contradictory schedules must fail during
resolution rather than silently falling back to a family default.

---

## 4. Infer normalization topology from evidence

The normalized metadata should resolve the four independent facts:

```text
mixer norm before?
mixer norm after?
FFN norm before?
FFN norm after?
```

Possible evidence may come from explicit booleans, layout strings, or tensor
inventory.

The canonical result should not retain the source vocabulary. For example,
`decoder_postnorm` should be translated into structural facts and then
forgotten:

```cpp
layer.mixer_norm.before = std::nullopt;
layer.mixer_norm.after = RMSNorm(...);
layer.feed_forward_norm.before = std::nullopt;
layer.feed_forward_norm.after = RMSNorm(...);
```

### Evidence precedence

Prefer evidence in this order:

1. explicit semantic metadata;
2. unambiguous layer-scoped metadata;
3. tensor inventory evidence;
4. safe derivation from already normalized facts.

If two strong sources conflict, resolution must fail with a
`ConflictingMetadata`-style error and explain both sources.

Do not resolve conflicts using the model-family name.

---

## 5. Bind norm tensors by semantic role

Extend tensor candidate grammars only where necessary to map source spellings to
existing semantic roles.

Examples of physical spellings that may map to the same roles include:

```text
AttentionPostNorm:
  model.layers.N.post_attn_norm.weight
  model.layers.N.post_attention_norm.weight
  blk.N.post_attention_norm.weight

FfnOutputNorm:
  model.layers.N.post_mlp_norm.weight
  model.layers.N.post_feed_forward_norm.weight
  blk.N.post_ffw_norm.weight
```

These are examples of naming grammar, not Lizzy mappings.

The binding solver should bind a norm tensor only when the canonical graph
requires that norm. Conversely, if the graph requires a post norm and no unique
tensor can satisfy the corresponding role, resolution must fail before backend
execution.

### Required cleanup

The current automatic layer binding must stop assuming that every attention
layer has an `AttentionInputNorm` and that every FFN has an `FfnInputNorm`.

Required roles become conditional on the canonical topology:

```text
mixer_norm.before  -> AttentionInputNorm
mixer_norm.after   -> AttentionPostNorm
FFN norm before    -> FfnInputNorm
FFN norm after     -> FfnOutputNorm
```

This mapping must be represented once in model resolution/weight planning.
CPU/CUDA weight loaders should consume the resolved plan rather than trying
alternate tensor names themselves.

---

## 6. Keep Q/K norm generic

Query/key normalization already belongs in `AttentionSpec` as semantic state.
The automatic resolver should infer it from explicit metadata and/or the
presence of uniquely bindable query/key norm tensors.

The result should remain:

```cpp
attention.query_norm = ...;
attention.key_norm = ...;
```

There must be no family-specific "Lizzy QK norm" behavior.

If the source declares Q/K norm but the required tensors cannot be bound, fail
at resolution time. If tensors prove Q/K norm but metadata says it is disabled,
treat that as conflicting evidence rather than ignoring one side.

---

## 7. Keep sliding-window attention generic

`SlidingWindowPattern` is already the canonical primitive. The missing work is
primarily schedule inference and validation.

For every attention layer:

```cpp
if (normalized_pattern[layer] == FullCausal)
    attention.pattern = FullCausalPattern{};
else if (normalized_pattern[layer] == SlidingWindow)
    attention.pattern = SlidingWindowPattern{window};
```

This conditional is intentionally based on the semantic property.

Validation requirements:

- window must be positive;
- schedule length must equal layer count;
- a sliding layer requires a known window;
- a full-causal layer must not accidentally inherit a local window because of
  a global backend setting;
- CPU and CUDA KV/attention policies must consume the per-layer pattern from the
  compiled program.

A model may mix local and full attention in any schedule; no periodic pattern
should be hard-coded.

---

## 8. Treat YaRN as position semantics, not family semantics

The Lizzy motivating checkpoint uses RoPE with YaRN scaling. If all required
YaRN parameters are already representable by `RopePositionSpec`, the work here
is only normalization of source aliases.

If a required YaRN parameter is not yet represented, extend the RoPE scaling
variant itself. Do not carry a family identity into position execution.

The semantic fingerprint must include only parameters that affect executable
RoPE mathematics.

---

# Implementation phases

## Phase 0 — Lock the architecture rule with tests

Before changing execution, add tests that make family-specific support
impossible to justify later.

### 0.1 Unknown-identity clone

Construct two synthetic checkpoints with identical structural metadata and
tensor inventory but different identity strings:

```text
model_type = "lizzy"
model_type = "unknown_test_model"
```

or the GGUF equivalent.

Require:

```cpp
EXPECT_EQ(a.graph.fingerprint(), b.graph.fingerprint());
EXPECT_EQ(a.weight_plan_semantics(), b.weight_plan_semantics());
```

Use whatever stable comparison primitives are available rather than adding a
new test-only API solely for this assertion.

### 0.2 Poisoned identity

Provide a deliberately misleading known-looking family name while preserving
the structural evidence. The resulting graph must still be derived from the
evidence.

### 0.3 Failure behavior

Add negative tests for:

- truncated layer schedule;
- unknown layer pattern token;
- sliding pattern without a window;
- contradictory pre/post norm metadata;
- topology requiring a norm whose tensor cannot be bound;
- metadata disabling a semantic feature that tensor evidence proves exists.

---

## Phase 1 — Normalize `LayerSpec` norm topology

1. Replace mandatory/overlapping norm fields with a structural before/after
   representation.
2. Update `ModelGraph::validate()`.
3. Update graph fingerprinting.
4. Update compiled program construction.
5. Update weight planning so norm roles are emitted only when semantically
   required.
6. Migrate every existing consumer in the same change.
7. Remove obsolete fields rather than keeping adapters or compatibility views.

### Invariant

There must be exactly one canonical way to represent each norm placement.

Do not keep both:

```text
operator_norm + mixer_norm.before
```

or:

```text
feed_forward_norm + feed_forward_norm.before
```

as parallel APIs.

---

## Phase 2 — Update CPU and CUDA execution

Make both backends consume the structural norm topology.

Required parity cases:

```text
pre / none
none / post
pre / post
none / none (only where semantically valid)
```

for both mixer and FFN sublayers.

Tests should compare the backend against the model/reference path on small
synthetic tensors where possible.

Delete any assumption that a normalization buffer is always populated simply
because the layer is attention or FFN.

---

## Phase 3 — Add layer-scoped semantic schedules

Extend metadata normalization to read categorical/string arrays and normalize
known source vocabulary into canonical attention patterns and norm topology.

Requirements:

- source aliases remain localized to inference/import code;
- the canonical graph contains no source strings;
- vector lengths are validated against `layer_count`;
- global and per-layer evidence conflicts are rejected;
- evidence records identify which metadata key produced each resolved fact.

Do not special-case a periodic 3-local/1-full schedule. The resolver must accept
arbitrary valid schedules.

---

## Phase 4 — Make tensor binding topology-aware

1. Bind pre/post norm roles according to `LayerSpec`.
2. Add generic candidate spellings needed by the motivating checkpoint.
3. Ensure Q/K norms are bound by `AttentionQueryNorm` and `AttentionKeyNorm`.
4. Remove unconditional input-norm binding assumptions.
5. Keep physical names out of CPU/CUDA execution.

A source tensor spelling should be interpreted exactly once.

---

## Phase 5 — Verify position and attention semantics

Validate that the existing canonical position model expresses every active YaRN
parameter required by the checkpoint.

Validate that sliding-window behavior is truly per layer through:

- decode;
- single-sequence prefill;
- batched prefill;
- KV-cache sizing/policy;
- CPU;
- CUDA.

If a backend cannot execute a represented primitive, capability negotiation
must reject that backend/model combination explicitly. Do not degrade to full
attention or another approximation.

---

## Phase 6 — End-to-end Lizzy-as-evidence validation

Only after the generic changes above are complete, use the real
`flwrlabs/Lizzy-7B` checkpoint as an integration fixture/manual validation
case.

The source tree must still contain no Lizzy execution specialization.

The resolver should derive a graph equivalent in shape to:

```text
32 dense decoder layers
RMSNorm
Q/K norm
SwiGLU FFN
RoPE + YaRN
per-layer local/full causal attention schedule
4096-token window on local layers
post-attention normalization
post-FFN normalization
```

The exact graph is determined from the checkpoint's current evidence, not from
this document.

### Identity replacement acceptance test

Take the same synthetic representation used for the integration semantics and
replace only its provenance/identity string with an arbitrary value. Resolution
must produce the same semantic fingerprint.

This test is the key acceptance criterion for "support without `if lizzy`".

---

# Acceptance criteria

The work is complete when all of the following are true:

- [ ] No source-code conditional tests for `lizzy`, `Lizzy`, the Hugging Face
      repository name, or an equivalent family identifier.
- [ ] No `lizzy.json`, descriptor, registry entry, or backend specialization.
- [ ] `LayerSpec` can represent pre-norm, post-norm, and sandwich-norm without
      dummy values or contradictory state.
- [ ] CPU execution no longer applies a pre-mixer norm unconditionally.
- [ ] CUDA follows the same canonical ordering.
- [ ] Automatic inference accepts layer-scoped categorical schedules.
- [ ] Alternating full/sliding attention is derived per layer.
- [ ] Sliding-window size is canonical semantic state and validated.
- [ ] Query/key normalization is inferred and bound through generic roles.
- [ ] Post-attention and post-FFN norm tensors are bound through generic roles.
- [ ] YaRN parameters are resolved as position semantics only.
- [ ] Unknown-identity clone produces the same graph fingerprint.
- [ ] Poisoned identity cannot override structural evidence.
- [ ] Ambiguous or contradictory evidence fails during resolution.
- [ ] Backend code contains no checkpoint-family tensor-name fallback chains.
- [ ] Existing supported models remain semantically unchanged unless a previous
      graph representation was demonstrably incorrect.

---

# Design constraints for future model support

This change should establish the pattern used for later checkpoints.

When a future model fails to load, ask these questions in order:

1. **Can the existing canonical graph already express the mathematics?**
   - If yes, improve evidence normalization or tensor binding.

2. **Is the missing concept a reusable semantic primitive?**
   - If yes, add that primitive to the graph and implement it generically.

3. **Is the difference only a physical tensor spelling/layout?**
   - If yes, extend binding/import normalization, not runtime semantics.

4. **Is the difference only a metadata vocabulary?**
   - If yes, add a source alias and normalize it away.

5. **Does the proposed change need the model's name to decide mathematics?**
   - If yes, the abstraction is probably wrong and should be redesigned before
     implementation.

The goal is that CELEG grows a vocabulary of **model semantics**, not a catalog
of **model brands**.

---

# Expected long-term result

After this work, support should be described in terms such as:

```text
CELEG supports:
- pre/post/sandwich RMSNorm layer topology;
- Q/K normalized attention;
- arbitrary per-layer full/sliding causal schedules;
- YaRN-scaled RoPE;
- dense gated feed-forward blocks;
- generic tensor-role binding for those primitives.
```

not:

```text
CELEG supports Lizzy.
```

If another model publishes the same executable semantics under a completely
unrelated identity, it should load without a source change. That is the
architectural definition of success for this plan.
