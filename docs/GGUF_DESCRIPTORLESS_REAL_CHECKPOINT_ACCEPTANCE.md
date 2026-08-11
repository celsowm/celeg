# GGUF Descriptorless Real-Checkpoint Acceptance Ledger

## Status

Active implementation companion to:

- `GGUF_DESCRIPTORLESS_AUTOMATIC_INFERENCE_PLAN.md`;
- `DESCRIPTORLESS_CHECKPOINT_INFERENCE_REFACTORING_PLAN.md`;
- `ZERO_MODEL_FAMILIES_REFACTORING_PLAN.md`.

This document records failures discovered with real GGUF checkpoints and turns each failure into a generic architectural requirement.

It is intentionally not a model-family support table. Model names are used only to identify the real regression fixture that exposed a reusable gap.

---

# 1. Non-negotiable triage rule

A real checkpoint failure must be classified at the narrowest responsible boundary before CELEG declares a semantic feature unsupported.

Use this order:

```text
1. checkpoint / GGUF reader gap
2. GGUF naming / loading grammar gap
3. metadata normalization representation gap
4. tensor-role binding gap
5. semantic inference / solver gap
6. graph synthesis gap
7. backend lowering/capability gap
8. genuinely unsupported mathematical primitive
```

Only item 8 justifies `UnsupportedSemanticFeature` merely because the required mathematics does not exist in CELEG yet.

The following are **not** sufficient reasons to classify a model as unsupported:

```text
metadata is an array instead of a scalar
metadata varies by layer
GGUF uses a native canonical tensor name
one layer has different attention geometry
one layer uses a different already-supported mixer
one known tensor role has a different format spelling
```

Those are representation/import/inference problems and must be generalized at their owning boundary.

Do not move to another model merely to avoid the current generic gap. Fix the gap and rerun the same checkpoint first.

---

# 2. Current real-checkpoint findings

## 2.1 Conventional descriptorless GGUF baseline

Observed progress:

- CPU compilation passes for the new descriptorless GGUF inference coverage;
- the generic GGUF metadata path is now reached without requiring model-family code;
- real checkpoints reach progressively deeper neutral boundaries.

This establishes that GGUF itself is not the blocker. The remaining failures expose missing generic conventions and semantic representations.

---

# 3. LFM2 real GGUF — per-layer metadata is not an unsupported primitive

## Observed failure

The real LFM2 GGUF reaches the generic metadata resolver on both CPU and CUDA.

Resolution then encounters a per-layer vector for:

```text
attention.head_count_kv
```

The current automatic path is still biased toward one globally homogeneous attention geometry and therefore cannot safely flatten the vector into one global GQA configuration.

## Classification

```text
metadata normalization representation gap
+
per-layer semantic inference gap
```

This is **not** evidence that the model is semantically unsupported.

A vector-valued `attention.head_count_kv` is concrete evidence that CELEG must preserve layer-dependent facts instead of forcing every checkpoint through a scalar global topology.

## Required architectural correction

Introduce a neutral representation for metadata observations that can be global or layer-scoped.

Conceptually:

```cpp
template <typename T>
struct LayerScopedValue {
    std::optional<T> global;
    std::vector<std::optional<T>> per_layer;
};
```

The exact type may differ, but it must support at least:

```text
one global value
one complete per-layer vector
global default + explicit per-layer overrides, when the format semantics allow it
missing value at a layer
conflicting observations
```

Do not represent a per-layer vector by choosing its first element, maximum, minimum, or most common value.

## Canonical ownership

Layer-varying semantics belong in canonical per-layer facts / `ModelGraph`.

`RuntimeTopology` may derive allocation maxima, counts, indexing tables, and other runtime summaries from those facts, but must not be the semantic source of truth that forces heterogeneity back into one global value.

Desired direction:

```text
GGUF metadata array
       |
       v
layer-scoped observations
       |
       +---- tensor presence / shapes
       |
       v
per-layer fact solving
       |
       v
CanonicalLayerFacts[]
       |
       v
ModelGraph
       |
       v
derived RuntimeTopology
```

## Required attention inference

Attention geometry must be able to vary by logical layer where checkpoint evidence permits it:

```text
layer 0
  query_heads = ...
  kv_heads    = ...
  head_dim    = ...

layer 1
  mixer       = another already-supported primitive

layer 2
  query_heads = ...
  kv_heads    = ...
  head_dim    = ...
```

No `Lfm2Schedule`, `if (architecture == "lfm2")`, repository-name branch, or family-prefixed rule is allowed.

## Required schedule inference

A heterogeneous schedule must be inferred from reusable evidence such as:

- layer-scoped metadata;
- presence/absence of attention tensor roles;
- recurrent/convolution/Mamba tensor grammars;
- tensor shapes;
- state-related metadata;
- already-supported semantic primitive rules.

A zero or missing value must not be assigned an invented meaning without a format/semantic rule that justifies that interpretation.

## Required tests

Add coverage for:

1. scalar metadata broadcast to all applicable layers;
2. valid complete per-layer vector;
3. valid global value plus per-layer overrides when supported by the source convention;
4. vector length different from logical layer count;
5. conflicting global and per-layer evidence;
6. heterogeneous KV-head geometry;
7. heterogeneous query-head geometry where valid;
8. layers where attention is absent and another known mixer is inferred;
9. failure when a layer remains genuinely ambiguous;
10. descriptor-vs-auto semantic parity for the real LFM2 fixture once the path resolves.

## Acceptance gate

Do not classify this real LFM2 GGUF as unsupported and do not use a different model to bypass the failure.

The checkpoint must be rerun after the generic layer-scoped representation is implemented and should advance until either:

- it runs descriptorlessly; or
- it exposes a genuinely new mathematical primitive that CELEG does not implement.

---

# 4. MiniCPM real GGUF — `output.weight` is a format grammar gap

## Observed failure

CPU and CUDA attempts for the real MiniCPM GGUF reach the GGUF reader but stop before semantic resolution because the standard tensor:

```text
output.weight
```

has no canonical loading mapping in the current path.

## Classification

```text
GGUF naming / loading grammar gap
```

This is not a model-ID issue and is not evidence of unsupported mathematics.

The primary plan already recognizes the reusable GGUF semantic mapping:

```text
output.weight
    -> TensorRole::LanguageModelHead
```

The real checkpoint now proves that recognizing the spelling in the later inference grammar is insufficient if an earlier loading/mapping boundary rejects the tensor first.

## Required architectural correction

The GGUF checkpoint boundary must preserve native canonical GGUF tensor names long enough for neutral tensor-role binding to consume them.

The pipeline must not require a family descriptor merely to translate a standardized GGUF source tensor into an internal loadable tensor.

Desired direction:

```text
GGUF tensor directory
       |
       v
source tensor identity + dtype + shape + offset
       |
       v
TensorInventory
       |
       v
GGUF tensor grammar proposals
       |
       v
BindingSolver
       |
       v
TensorRole::LanguageModelHead -> output.weight
       |
       v
WeightPlan / loader
```

If physical loading currently requires a canonicalized source-name table before `TensorInventory`/binding, refactor that boundary so semantic role assignment owns the mapping rather than a model-family descriptor.

## Source name vs semantic role

Keep these concepts distinct:

```text
source_name = "output.weight"
semantic_role = TensorRole::LanguageModelHead
```

The loader may use the source identity/offset to retrieve bytes, while graph/backend code consumes the semantic role.

Do not rename the raw checkpoint tensor globally just to make an existing family-oriented loader happy.

## Tied embeddings

`output.weight` handling must preserve the normal semantic distinction:

```text
untied checkpoint
    -> explicit language-model-head tensor required

tied checkpoint
    -> language-model head may deliberately alias token embedding
```

Absence of `output.weight` is therefore not always an error; it depends on validated tied-embedding evidence.

Conversely, when `output.weight` exists, its presence must not be ignored merely because tied embeddings are possible.

## Required tests

Add generic GGUF coverage for:

1. `output.weight` mapping to `TensorRole::LanguageModelHead`;
2. untied embeddings with explicit `output.weight`;
3. tied embeddings with no independent output tensor;
4. explicit output tensor whose shape disagrees with vocabulary/hidden dimensions;
5. duplicate/ambiguous language-model-head candidates;
6. quantized `output.weight` retaining correct source identity and dtype/layout metadata;
7. loading the mapped tensor through the weight plan without a descriptor/family table;
8. CPU and CUDA reaching semantic resolution through the same neutral binding result;
9. descriptor-vs-auto semantic parity for the MiniCPM fixture once resolution succeeds.

## Acceptance gate

Do not mark MiniCPM as unsupported and do not add a MiniCPM-specific tensor map.

Fix the generic GGUF `output.weight` path and rerun the same cached real checkpoint until it advances beyond this boundary.

---

# 5. Failure taxonomy must become observable

Real-model acceptance runs should report a typed stage/failure classification rather than only a generic load exception.

The diagnostics should make it possible to distinguish, for example:

```text
CheckpointFormatFailure
MetadataNormalizationFailure
TensorGrammarFailure
TensorBindingFailure
CanonicalFactConflict
UnsupportedSemanticFeature
BackendCapabilityMismatch
```

The exact public types can reuse the existing `ResolutionFailureKind` hierarchy where appropriate.

The important requirement is that a format/import gap cannot be accidentally recorded as an unsupported model.

---

# 6. Real-checkpoint workflow

For every real GGUF checkpoint used during this migration:

```text
run checkpoint
      |
      v
capture narrowest failure boundary
      |
      v
classify generic gap
      |
      v
write synthetic regression for that convention
      |
      v
implement reusable rule/representation
      |
      v
run complete repository tests
      |
      v
rerun SAME real checkpoint
      |
      +--> advances -> repeat at next boundary
      |
      +--> runs -> add permanent parity/acceptance coverage
```

Do not immediately rotate through many models while leaving known generic gaps open. That produces a list of failures rather than a descriptorless architecture.

A second model is useful when it proves that the just-added convention is reusable, not as an escape from the first model's unresolved boundary.

---

# 7. Updated implementation order

The current priority order is now:

1. preserve native GGUF tensors through the checkpoint boundary;
2. close standard GGUF tensor grammar/loading gaps, including `output.weight`;
3. introduce scalar-or-layer-scoped metadata observations;
4. make canonical attention geometry layer-scoped;
5. infer heterogeneous mixer schedules from metadata + tensor evidence;
6. shrink the dense-only logic still concentrated in `infer_canonical_model_facts()` into reusable rules/solvers;
7. rerun the LFM2 real Q4_K_M fixture;
8. rerun the MiniCPM real GGUF fixture;
9. continue deeper only after these previously observed generic gaps are closed;
10. delete `dense_transformers.json` only after the descriptorless path reaches the parity gate defined by the primary plan.

The order may be adjusted when one correction is a prerequisite for another, but neither observed failure may be bypassed with family-specific code.

---

# 8. Acceptance matrix

| Fixture | Boundary reached | Current gap | Correct owner | May be called unsupported now? |
|---|---|---|---|---|
| Conventional synthetic/native GGUF coverage | automatic inference / CPU compile | baseline passes | generic path | no failure |
| Real LFM2 GGUF | generic metadata resolver on CPU + CUDA | per-layer `attention.head_count_kv`; heterogeneous schedule | metadata normalization + per-layer semantic solver | **Resolved**: layer-scoped metadata reaches the neutral runtime |
| Real MiniCPM GGUF | GGUF reader on CPU + CUDA | `output.weight` has no canonical loading mapping | GGUF tensor grammar / checkpoint loading boundary | **Resolved**: native head spelling is consumed by descriptorless binding |
| Real LFM2.5-230M Q4_0 GGUF | CPU inference; CUDA OpenAI-compatible server chat | Q4_0 loading initially rejected in fused feed-forward concatenation | generic GGUF quantization loading | **Resolved**: CPU repacks decoded Q4_0; CUDA loads it as BF16 when native blocks have no kernel |
| Real Nemotron-3-Nano-4B GGUF | automatic CPU/CUDA inference + CUDA OpenAI-compatible server | heterogeneous 21 Mamba-2, 17 MLP-only, and 4 attention blocks; direct serving path is required because packed recurrent metadata is not supported | generic mixer/FFN schedule inference, GGUF tensor-shape adaptation, and server capability selection | **Resolved**: CPU runner, CUDA runner, `/v1/models`, and tool-schema chat protocol all passed |
| Real GPT-X2.5-135M Safetensors | automatic CPU/CUDA raw inference | F32 source tensors and `transformer.h.*.mlp.w_{up,gate,down}` convention; official checkpoint has no chat template | generic Safetensors CPU F32/repack path and reusable tensor binding | **Resolved**: CPU and CUDA raw runners passed; chat/tool protocol is not applicable to this base checkpoint |
| Real LFM2.5-230M Safetensors | automatic CPU/CUDA inference + CPU OpenAI-compatible server | `embedding_norm`, alternating conv/attention blocks, `feed_forward.w1/w2/w3`, explicit tokenizer IDs, and sidecar `chat_template.jinja` | generic tensor naming, tokenizer precedence, and Safetensors interaction metadata boundary | **Resolved**: CPU/CUDA runners, `/v1/models`, and tool-schema chat request passed; base model generated no tool call |
| Real LFM2.5-1.2B-Instruct Safetensors | automatic CPU/CUDA inference + CPU OpenAI-compatible server | tied head spelling `tie_embedding` and declared FFN size differs from auto-adjusted tensor size | generic metadata alias and tensor-derived FFN geometry when auto-adjust is declared | **Resolved**: validated Safetensors header, CPU/CUDA runners, `/v1/models`, and tool-schema chat request passed; generated no tool call |

Update this table as each real checkpoint advances. Replace the current gap with the next narrow failure rather than accumulating stale failures.

---

# 9. Additional SOLID constraints exposed by these failures

## SRP

The LFM2 failure demonstrates why metadata normalization, layer-schedule inference, graph synthesis, and topology derivation must remain separate.

The MiniCPM failure demonstrates why physical tensor discovery/loading and semantic tensor-role assignment must remain separate.

## OCP

Adding support for:

```text
per-layer GGUF value convention
output.weight language-model-head convention
```

must extend reusable format/semantic rules, not reopen a model-family implementation.

## LSP

Scalar and vector metadata providers must produce observations that obey the same fact-solving contract. A caller must not depend on one implementation always returning a scalar.

## ISP

Do not respond to these failures by creating one large `IGgufArchitectureMapper`. Keep metadata value normalization, tensor grammar matching, binding, and layer semantics independently testable.

## DIP

CPU and CUDA must receive the same resolved semantic program and weight-role plan. Neither backend may learn that the source checkpoint was LFM2 or MiniCPM in order to work around the import gap.

---

# 10. Updated invariant

The real-checkpoint failures refine the architectural invariant to:

> GGUF source spelling and value shape belong to the checkpoint boundary. Layer-dependent facts belong to canonical semantics. Tensor roles belong to binding. Backend execution consumes only resolved semantics. A model is unsupported only when its required mathematics is actually unsupported — never because CELEG has not yet generalized a GGUF representation convention.
