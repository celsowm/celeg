# CELEG Model Topology Ownership Refactoring Plan

## Status

Revised architectural refactor plan, re-audited against the current `master` tree.

Target repository: `celsowm/celeg`.

This plan has **no backward-compatibility constraint**. Struct layouts, function
signatures, and public headers may change as clean breaks. No deprecated
wrappers, transitional shims, or re-exported legacy names.

The refactor is guided by one stronger ownership rule than earlier revisions:

> **`ModelGraph` is the single source of executable model semantics.
> `ExecutionTopology` is a derived execution/allocation cache only, and
> checkpoint/import state is not observable by execution backends.**

The refactor is complete only when all of the following hold:

1. A checkpoint resolver or descriptor front-end is physically unable to write a
   graph-derived execution field.
2. `GraphSynthesizer` returns a semantically final `ModelGraph`; no assembler,
   compiler, or backend may repair or reinterpret its semantics later.
3. A backend does not read semantic execution policy from checkpoint/import
   state when the graph or compiled program can express that policy.
4. Every semantic variant-to-enum projection and every variant dispatch is
   compile-time exhaustive. Adding a new mixer/FFN alternative must break the
   compilation at every concern that needs to understand it.
5. The compiled model has no runtime fallback from per-layer semantic data to a
   checkpoint-wide default when the per-layer value is required for execution.

---

# 1. Current problem

CELEG has the right high-level layers but semantic ownership is still split
between four representations:

```text
checkpoint / descriptor / inferred facts
        |
        v
RuntimeTopology              <-- currently both import facts and execution semantics
        |
        v
ModelGraph                   <-- canonical intent, but not yet final in all paths
        |
        v
CompiledModelProgram         <-- still re-derives/repairs some semantics
        |
        v
CPU / CUDA backends          <-- still read semantic flags from RuntimeTopology
```

That means values can be written in one representation, consumed by a second,
recomputed in a third, and then bypassed by a backend reading the original
copy. The type system does not make incorrect ownership impossible.

Two resolution paths currently exist:

- descriptorless automatic inference;
- descriptor-driven architecture extensions loaded from the configured
  descriptor directory when present.

The descriptor path is an optional extension point, not a guaranteed in-tree
instance. The built-in automatic architecture is registered directly, while
external/installed descriptor JSON files can register additional architectures
at runtime.

## 1.1 `RuntimeTopology` is a mixed-ownership god-struct

`RuntimeTopology` currently mixes:

### Import/checkpoint facts

Examples:

```text
hidden
intermediate
vocab_size
max_position_embeddings
checkpoint_layer_for_layer
token_policy
mtp_num_hidden_layers
```

### Graph-derived execution caches

Examples:

```text
num_hidden_layers
mixer_kinds
feed_forward_kinds
execute_feed_forward
feed_forward_intermediates
attention_layouts
gated_delta_net_layouts
mamba2_layouts
mlp_only_layouts
attention_slot_for_layer
layer_for_attention_slot
attention_layer_count
conv_layer_count
gated_delta_net_layer_count
mamba2_layer_count
mlp_only_layer_count
num_dense_layers
mamba2_intermediate
max_feed_forward_intermediate
dense_intermediate
moe_intermediate
shared_expert_intermediate
num_experts
experts_per_token
normalize_topk
moe_router_softmax
use_expert_bias
routed_scaling_factor
moe_routing_*
has_per_layer_input
per_layer_input_size
conv_cache
conv_dim
```

### Semantic policies that should not remain import-owned

The previous revision incorrectly treated `NumericalPolicy` as a harmless
checkpoint-owned category. It is not. Several of its values already have a
semantic representation in the graph or compiled program:

```text
embedding_multiplier  -> ModelGraph::embedding_transform.multiplier
logits_divisor        -> ModelGraph::logits_divisor
norm_eps              -> NormSpec values
residual_multiplier   -> LayerSpec::residual.multiplier
attention scaling     -> AttentionSpec / compiled attention semantics
```

Keeping those values readable from `RuntimeTopology` leaves two semantic owners
and lets backends bypass graph/program semantics.

The target design therefore does **not** move the whole current
`NumericalPolicy` into `CheckpointDimensions`. Values needed only while building
semantic objects may exist as import facts temporarily, but they must disappear
from the execution-facing boundary once the graph/program is built.

## 1.2 The graph is not yet semantically final

The automatic path currently synthesizes a `ModelGraph`, then
`ResolutionAssembler` mutates attention output-gate granularity by inspecting the
physical `AttentionGate` tensor binding.

Later, `build_model_program()` performs the same interpretation again from the
`WeightPlan`, changing the compiled copy of `AttentionSpec` to `HeadWise` or
`ElementWise`.

This violates the intended boundary twice:

```text
GraphSynthesizer -> graph
Assembler        -> repairs graph semantics
Compiler         -> repairs graph semantics again
```

`AttentionGateGranularity` is semantic. Tensor shape is evidence used to resolve
that semantic; it must be consumed before `GraphSynthesizer::synthesize()`
returns.

The correct flow is:

```text
physical tensor evidence
        |
        v
CanonicalModelFacts
        |
        v
final LayerSpec / ModelGraph
        |
        v
WeightPlan
        |
        v
CompiledModelProgram
```

`ResolutionAssembler` composes results. It does not infer semantics.
`build_model_program()` lowers semantics. It does not infer semantics.

## 1.3 Semantic reads still bypass the graph/program

The current split-norm flag is one example, but not the only one.

### Split norms

Backends still branch on model-wide `shape.has_split_attention_norms` despite
`LayerSpec` / `CompiledLayerProgram` already carrying per-layer
`post_attention_norm` and `post_feed_forward_norm`.

This is a direct agnosticism and ownership leak. The backend should ask the
per-layer semantic object whether a norm is enabled.

### Residual multiplier

CPU execution still reads `shape.numerical_policy.residual_multiplier` in the
layer loop even though `LayerSpec::residual.multiplier` is already semantic
per-layer data. That read must move into the compiled layer program.

### Attention scaling

Attention execution still combines graph-owned `AttentionSpec::query_scale`
with topology-owned `numerical_policy.attention_multiplier` depending on the
path. The final attention scaling contract must be represented once in
`AttentionSpec`/compiled attention semantics.

### Backend configuration

CUDA configuration still makes execution decisions from topology-level
numerical flags. If such a property affects execution behavior, it belongs in a
semantic/compiled policy visible to the backend compiler, not in checkpoint
state carried into runtime.

### Intermediate-width fallbacks

Execution/weight-loading code still contains shapes equivalent to:

```cpp
layer_program.feed_forward_intermediate > 0
    ? layer_program.feed_forward_intermediate
    : shape.intermediate;
```

and MoE backing can fall back from `moe_intermediate` to `intermediate`.

After this refactor, an executable FFN layer must have a valid per-layer width.
A compiled model that lacks it is invalid. Runtime fallback to a checkpoint-wide
import default is forbidden.

## 1.4 Variant exhaustiveness is weaker than the plan previously claimed

Fixing only the `std::visit` bodies is insufficient.

`LayerSpec::mixer_kind()` currently checks known alternatives and falls through
to `MlpOnly`. `feed_forward_kind()` checks dense and treats the remaining
alternative as MoE. A future variant alternative can therefore silently acquire
the wrong enum tag.

The same risk exists anywhere that projects a semantic variant to an enum or
performs variant-specific lowering.

New rule:

> **No semantic variant -> enum projection may contain a default/fallthrough
> interpretation.**

All of these must be exhaustive:

- `LayerSpec::mixer_kind()`;
- `LayerSpec::feed_forward_kind()`;
- `ExecutionTopology::derive` visitors;
- compiled-program lowering visitors;
- weight-requirement derivation visitors;
- any future semantic capability projection.

Use explicit `if constexpr` alternatives plus an unreachable
`static_assert(always_false_v<T>)`, or an equivalent compiler-checked visitor.

## 1.5 Known correctness defects

### Defect A — grouped MoE routing is lost during topology derivation

`derive_runtime_topology_from_graph()` clears the `moe_routing_*` fields but does
not restore them from `MixtureOfExpertsSpec` in its MoE branch.

The graph/program can therefore carry grouped routing while topology consumers
see ordinary top-K. This is a correctness bug, not cleanup.

### Defect B — descriptor MoE aggregate initialization is field-shifted

The descriptor graph builder initializes `MixtureOfExpertsSpec` positionally
with fewer arguments than the struct fields and omits grouped-routing slots. The
following values can bind to the wrong fields while still compiling because the
types are convertible.

All semantic aggregate construction of `MixtureOfExpertsSpec` must use
designated initializers.

The same rule should be used for other large semantic structs where field order
is not itself part of the contract.

### Defect C — `double_wide_shared_suffix` is applied before graph derivation

The descriptor path mutates topology widths that are later re-derived from the
graph, making the feature ineffective unless the graph builder itself applies
the semantic width.

The flag's effect belongs when the relevant `LayerSpec` is constructed.

## 1.6 Descriptor coverage is a prerequisite

The descriptor directory may be absent in the source tree and descriptor
architectures may come from installed/external JSON files. Structural changes to
`src/model/descriptor/**` therefore need explicit test fixtures rather than
assuming the repository ships live descriptors.

Before rewriting that front-end, add in-tree test fixtures for at least:

- dense attention;
- MoE with shared expert and softmax routing;
- grouped MoE routing;
- `double_wide_shared_suffix`;
- `split_attention_norms: true`.

The tests must resolve through the real descriptor registration/resolution path,
not only call helper functions directly.

---

# 2. Target ownership model

## 2.1 Import facts are one-way inputs

Checkpoint/descriptor/inference state exists to build semantic objects. Once a
`ResolvedModel` is executable, backend code must not depend on import state.

Conceptually:

```cpp
struct CheckpointDimensions {
    int vocab_size = 0;
    int max_position_embeddings = 0;
    std::vector<int> checkpoint_layer_for_layer;
    TokenPolicy token_policy;
    int mtp_num_hidden_layers = 0;

    void validate() const;
};
```

This type should contain only facts that truly remain import-owned and are not
execution semantics duplicated elsewhere.

`hidden` is deliberately not retained here as the execution source of truth.
See section 2.3.

`intermediate` is not a permitted runtime fallback. If retained as an import
fact during migration, it must be consumed while constructing the graph and not
used by execution after compilation.

Numerical import values may be carried in an importer-local structure while
building semantic policies, but they are not part of the backend-visible
checkpoint dimensions contract.

## 2.2 `ModelGraph` owns executable semantics

`ModelGraph` must contain every semantic fact required to compile the model,
including:

- residual-stream width;
- per-layer mixer semantics;
- per-layer FFN semantics and widths;
- per-layer norms;
- attention scaling;
- attention gate granularity;
- state/storage semantics;
- MoE routing semantics;
- shared-expert semantics;
- embedding/final-output transforms;
- intermediate normalization boundaries;
- per-layer execution enablement.

The graph is immutable after synthesis/validation.

## 2.3 `hidden` belongs to the semantic graph

The residual-stream width defines the meaning and shapes of projections,
normalization, MoE payloads, embeddings, and output heads. It is graph semantics
that happens to originate from checkpoint evidence.

Add it to `ModelGraph` (name may be `hidden`, `hidden_size`, or
`residual_width`, chosen consistently).

Backends and compiled-program construction must consume the graph/compiled value,
not the original checkpoint copy.

## 2.4 `ExecutionTopology` is derived cache only

```cpp
class ExecutionTopology {
public:
    static ExecutionTopology derive(const ModelGraph& graph);

    // allocation/index/cache queries only

private:
    ExecutionTopology() = default;
};
```

It may contain cached values such as:

```text
num_hidden_layers
attention_slot_for_layer
layer_for_attention_slot
attention_layer_count
conv_layer_count
gated_delta_net_layer_count
mamba2_layer_count
mlp_only_layer_count
maximum projection/workspace widths
maximum recurrent cache dimensions
allocation-oriented MoE maxima
```

It must **not** be used as a second semantic representation of values that the
compiled layer already owns.

A useful review question for every proposed field is:

> If this field disagreed with `ModelGraph`, would execution semantics change or
> only allocation/indexing fail?

If semantics would change, the field does not belong in `ExecutionTopology` as
an independently consumable value.

## 2.5 Runtime pair

A transitional or final wrapper may remain:

```cpp
struct RuntimeTopology {
    CheckpointDimensions dims;
    ExecutionTopology exec;

    void validate() const;
    std::string summary() const;
};
```

But execution-facing code should progressively prefer
`CompiledModelProgram`/backend-compiled state. `RuntimeTopology` is not a semantic
API.

## 2.6 Compiled program is complete

`CompiledModelProgram` must be sufficient for token/chunk execution without
semantic fallbacks into checkpoint dimensions.

A `CompiledLayerProgram` that executes an FFN must have a valid
`feed_forward_intermediate`.

Residual multipliers, norm policies, attention scaling, MoE routing, shared
experts, and per-layer optional features must be explicit compiled semantics.

If a backend needs an execution policy that is not represented in the compiled
program, fix the compiled semantic model rather than reading it from import
state.

---

# 3. Resolution contract

## 3.1 Canonical facts before graph synthesis

Any semantic fact inferred from tensor shape/name/metadata must be resolved in
`CanonicalModelFacts` (or the equivalent canonical input) before graph synthesis.

In particular, move attention output-gate granularity resolution out of:

- `ResolutionAssembler`;
- `build_model_program()`.

The binding resolver already knows the physical shape. It should publish the
resolved semantic granularity into canonical layer facts.

## 3.2 Graph synthesizer returns the final graph

The contract becomes:

```cpp
class GraphSynthesizer {
public:
    ModelGraph synthesize(const CanonicalModelFacts& facts) const;
};
```

After return:

- no semantic field is patched;
- no tensor shape is interpreted to change graph semantics;
- graph validation can run immediately and permanently establish the semantic
  model.

## 3.3 Resolution assembler only composes

`ResolutionAssembler` should:

- accept validated graph;
- accept validated weight plan;
- derive execution topology;
- attach capabilities/provenance;
- validate the composed resolved model.

It must contain no inference heuristic and no semantic repair loop.

## 3.4 One resolution pipeline

Descriptorless and descriptor-driven front-ends must converge on the same
semantic synthesis boundary. The long-term direction is:

```text
CheckpointView
    |
    +--> automatic evidence/facts -----+
    |                                  |
    +--> descriptor evidence/facts ----+
                                       |
                                       v
                              CanonicalModelFacts
                                       |
                                       v
                                  ModelGraph
                                       |
                              +--------+--------+
                              |                 |
                              v                 v
                         WeightPlan      ExecutionTopology
                              |                 |
                              +--------+--------+
                                       |
                                       v
                                  ResolvedModel
                                       |
                                       v
                              CompiledModelProgram
```

A descriptor may provide explicit facts that automatic inference cannot prove,
but it should not own a separate semantic graph model forever.

---

# 4. Phases

Each phase must leave the tree building and all applicable tests green.

## Phase 0 — Correctness first

**Goal:** stop shipping known wrong semantics before structural churn.

### 0a — grouped-MoE derivation

- Restore all four grouped-routing fields from `MixtureOfExpertsSpec` during
  topology derivation.
- Add a regression test for grouped-routing resolution.
- Add a regression test at the consumer boundary that proves grouped selection
  remains enabled after resolution/compilation.
- Land separately so the correctness fix can be backported independently.

### 0b — descriptor test harness

Add real descriptor fixtures and route them through descriptor registration and
resolution.

### 0c — descriptor MoE initialization

- Replace positional `MixtureOfExpertsSpec{...}` construction with designated
  initializers.
- Supply all grouped-routing/shared/router fields explicitly.
- Apply the same designated-initializer rule to the automatic graph path.
- Add tests for shared expert, softmax score, grouped routing, and combine order.

### 0d — `double_wide_shared_suffix`

- Move the doubling to `LayerSpec` construction.
- Delete the topology-only mutation.
- Add a resolved-graph assertion.

**Exit criteria:** all known correctness defects have failing-then-passing tests;
descriptor fixtures exist; no structural refactor is required for the fixes.

## Phase 1 — Make graph synthesis semantically final

**Goal:** establish a hard semantic boundary before topology splitting.

### 1a — attention gate granularity

- Move shape-derived gate granularity into canonical facts/binding resolution.
- `GraphSynthesizer` emits the final granularity.
- Delete the post-synthesis graph mutation in `ResolutionAssembler`.
- Delete the second granularity reinterpretation in `build_model_program()`.
- Add a test for both head-wise and element-wise gates proving the graph and
  compiled program agree without repair logic.

### 1b — eliminate other semantic post-processing

Audit for any other pattern equivalent to:

```text
build graph -> inspect weights/checkpoint -> change semantic field
```

Move each one before graph synthesis.

**Exit criteria:** after `GraphSynthesizer::synthesize`, `ModelGraph` is validated
and never semantically mutated.

## Phase 2 — Move semantic policy out of execution-facing topology

**Goal:** prevent backends from bypassing graph/program ownership.

### 2a — split norms

- Replace every `shape.has_split_attention_norms` read with the corresponding
  per-layer `NormSpec::enabled()` in graph/compiled semantics.
- Remove the flag from execution-facing topology.
- Keep the descriptor key only as an input that causes per-layer norms to be
  emitted.

### 2b — residual multiplier

- Ensure per-layer residual multiplier is present in `CompiledLayerProgram`.
- Replace CPU/CUDA reads of topology numerical residual multiplier.
- Remove the execution-facing duplicate.

### 2c — attention scaling

- Define one final semantic attention scaling contract in `AttentionSpec` (or a
  dedicated semantic field if needed).
- Remove backend dependence on `topology.numerical_policy.attention_multiplier`.
- Preserve current numerical behavior with targeted tests.

### 2d — remaining numerical policy

Audit every current `shape.numerical_policy.*` backend read.

For each field:

- move it to graph/program semantics if it changes model mathematics;
- move it to backend options if it is an executor policy rather than model
  semantics;
- delete it from backend-visible import state when no longer needed.

Do not mechanically nest the old `NumericalPolicy` under
`CheckpointDimensions`.

**Exit criteria:** no backend execution branch uses an import-owned numerical or
model-wide semantic flag when the compiled model contains the answer.

## Phase 3 — Front-ends emit semantic layers directly

**Goal:** remove the topology -> graph transcription cycle.

Both current graph builders consume `RuntimeTopology` tables produced by their
front-end. That cycle must disappear before topology ownership can be enforced.

### 3a — automatic front-end

Change automatic synthesis so canonical facts/layer facts construct
`LayerSpec` directly instead of first populating `mixer_kinds`, attention layout
tables, FFN tables, and MoE scalars.

### 3b — descriptor front-end

Make descriptor resolution produce the same canonical layer facts / `LayerSpec`
semantics, using explicit descriptor values as evidence/overrides rather than
maintaining a separate topology-transcription architecture.

### 3c — remove graph builders that only transcribe topology

Delete or radically shrink `build_dense_transformer_graph` and
`build_descriptor_graph` once no per-layer graph semantics come from
`RuntimeTopology`.

**Exit criteria:** front-ends do not write graph-derived execution cache fields.

## Phase 4 — Split import dimensions from `ExecutionTopology`

**Goal:** make illegal topology ownership unrepresentable.

### 4a — introduce `CheckpointDimensions`

Keep only real import/checkpoint facts that remain necessary after graph
construction.

Do not include semantic numerical policy wholesale.

### 4b — move `hidden` to `ModelGraph`

- Add residual width to `ModelGraph`.
- Update graph validation and weight requirement shapes.
- Update compiled program to carry what execution needs.
- Rename backend reads to the compiled/semantic location, not merely
  `shape.exec.hidden` if the backend already has a compiled-program value.

### 4c — remove runtime `intermediate` fallback

- Make per-layer FFN intermediate mandatory for executable FFN layers.
- Make MoE per-layer intermediate mandatory where MoE executes.
- Remove `?: shape.intermediate` / `?: shape.moe_intermediate` runtime fallbacks.
- If a checkpoint-wide intermediate remains useful during import, keep it out of
  execution.

### 4d — derive execution topology by value

Replace mutating:

```cpp
derive_runtime_topology_from_graph(RuntimeTopology&, const ModelGraph&)
```

with:

```cpp
ExecutionTopology ExecutionTopology::derive(const ModelGraph&)
```

and make the constructor private.

### 4e — delete dead/leaky fields

Delete fields such as `shared_kv_group_count` that have no valid execution
consumer, rather than deriving them for completeness.

**Exit criteria:** a front-end cannot construct or mutate `ExecutionTopology`;
`ExecutionTopology` is produced only from `ModelGraph`; runtime code cannot fall
back to import widths for required per-layer semantics.

## Phase 5 — Compile-time exhaustive semantic variants

**Goal:** make new semantics fail loudly until every concern supports them.

### 5a — tag projection

Rewrite `LayerSpec::mixer_kind()` and `feed_forward_kind()` with exhaustive
visitors. No default/fallthrough interpretation.

### 5b — topology derivation

Every mixer/FFN visitor in `ExecutionTopology::derive` must enumerate each
variant and finish with compiler-failing unreachable handling.

### 5c — compiled-program lowering

The lowering to `CompiledMixer`, `CompiledFeedForward`, state layout, MoE
program, and any related enum must be exhaustive.

### 5d — weight requirements

Weight requirement derivation must use the same exhaustiveness property.

**Exit criteria:** adding a new alternative to `LayerSpec::mixer` or
`feed_forward` produces compile errors in every concern that must be taught the
new semantics.

## Phase 6 — One weight-requirement derivation

**Goal:** eliminate drift between automatic and descriptor plans.

Create one graph-driven implementation that consumes:

```cpp
const ModelGraph&
const CheckpointDimensions&   // only for real import mapping such as physical layer
const ITensorNamingPolicy&
```

or a narrower mapping object if possible.

Per-layer semantic shapes must come from the graph:

- hidden/residual width;
- attention layout;
- query/key norm;
- attention gate;
- split norms;
- FFN intermediate;
- MoE/shared expert;
- KV sharing;
- recurrent semantics.

Checkpoint-layer mapping and source naming are binding concerns, not graph
semantics.

Delete duplicated automatic/descriptor mappings once parity tests pass.

## Phase 7 — One resolution pipeline

**Goal:** eliminate duplicated ordering and special assembler behavior.

Both automatic and descriptor-driven paths should call one composition pipeline
that accepts already-resolved canonical semantics.

`ExecutionTopology::derive` must have exactly one production call site in the
resolution pipeline.

## Phase 8 — Validation and error model

**Goal:** make the graph trustworthy now that it is the single semantic owner.

### 8a — semantic validators

Add or strengthen validation for:

- `MixtureOfExpertsSpec`;
- grouped-routing consistency;
- shared-expert dimensions;
- required FFN widths;
- attention output-gate granularity;
- per-layer norm consistency;
- residual width/projection compatibility where practical.

### 8b — remove tautological graph-vs-derived-topology checks

Once `ExecutionTopology` is a pure function of `ModelGraph`, checks comparing
those two representations are tautologies. Move invariant validation into the
graph and derived-cache validation into topology.

### 8c — one typed resolution error model

Unify model-resolution invariant failures under typed `ResolutionError` kinds
instead of mixing `invalid_argument`, `runtime_error`, and resolution-specific
errors for equivalent categories.

## Phase 9 — Residual DRY cleanup

After the ownership boundary is stable:

- unify descriptor key selection / placeholder expansion;
- replace duplicated candidate scans with shared helpers;
- consolidate quantization row scans;
- remove dead topology adapters/helpers exposed by the old ownership model.

---

# 5. Backend rule after the refactor

Backends may consume three classes of information:

### Compiled semantics

Examples:

```text
mixer kind and detailed spec
attention scaling
norms
residual multiplier
FFN/MoE widths
MoE router program
state layout
per-layer execution enablement
```

### Derived resource topology

Examples:

```text
number of attention slots
maximum workspace widths
cache slot/index mapping
maximum recurrent cache dimensions
allocation maxima
```

### Explicit backend options

Examples:

```text
weight mode
fused kernel policy
offload budget
cache budget
NUMA/device placement
```

A checkpoint/import field is not a fourth execution input category.

---

# 6. Validation strategy

The refactor is too broad for fingerprints alone.

Use layered tests:

## Semantic graph tests

Assert exact `LayerSpec` semantics for:

- dense attention;
- query/key norm;
- head-wise and element-wise attention gates;
- hybrid recurrent layers;
- Mamba2;
- MLP-only layers;
- dense/MoE mixed schedules;
- grouped routing;
- shared experts;
- split norms;
- per-layer inputs.

## Weight-plan tests

Assert expected roles and shapes derive from the graph without topology-only
fallbacks.

## Compiled-program tests

Assert the compiler preserves graph semantics exactly and performs no hidden
shape-based reinterpretation.

## Backend parity tests

For representative real/minimal checkpoints, compare logits or another stable
numerical oracle across the refactor for all behavior-preserving phases.

## Negative ownership tests

The private `ExecutionTopology` constructor and mandatory compiled per-layer
widths should turn illegal test fixtures into compile errors. Rewrite fixtures to
build valid graphs instead of hand-populating derived topology fields.

---

# 7. Non-goals

- New model families or architecture-specific execution branches.
- Backward compatibility for old topology layouts.
- Preserving descriptor-only semantic pipelines after the unified canonical
  boundary exists.
- Changing numerical behavior except for explicit Phase 0 correctness fixes.
- Moving executor policy into model semantics merely to delete a topology field.
  Backend policy stays backend policy.
- Treating `ExecutionTopology` as a second canonical model description.

---

# 8. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| The ownership refactor touches many backend reads | Move semantic reads to `CompiledModelProgram` first, then split topology. This converts ambiguous runtime dependencies into explicit compiler errors in smaller steps |
| Descriptor path has no guaranteed live in-tree descriptor set | Build real descriptor fixtures before structural changes |
| Graph becomes canonical while validation is incomplete | Strengthen semantic spec validators before deleting graph-vs-topology checks |
| Moving numerical policy changes behavior accidentally | Convert one semantic field at a time with targeted numerical tests |
| `hidden` move creates large mechanical churn | Prefer compiled-model access in backend code rather than blindly renaming every site to another topology nesting |
| Removing `intermediate` fallbacks exposes incomplete synthesized layers | Treat that as a desired invariant failure; fix the synthesizer rather than restoring runtime fallback |
| New variant kinds still silently map to old enums | Make tag projection itself exhaustive, not only downstream visitors |
| Weight-plan and compiler currently infer semantics from physical shapes | Resolve all such evidence into canonical facts before graph synthesis |

---

# 9. Sequencing summary

```text
Phase 0  correctness bugs + descriptor harness
   |
   v
Phase 1  graph synthesis becomes semantically final
   |
   v
Phase 2  remove backend semantic reads from RuntimeTopology
   |
   v
Phase 3  front-ends emit semantic layers directly
   |
   v
Phase 4  split CheckpointDimensions / ExecutionTopology
         move hidden to graph
         remove intermediate fallbacks
   |
   v
Phase 5  compile-time exhaustive semantic variants
   |
   +----------+
   |          |
   v          v
Phase 6    Phase 7
one weight one resolution
plan       pipeline
   |          |
   +----+-----+
        |
        v
Phase 8  validation + typed errors
        |
        v
Phase 9  residual DRY cleanup
```

Natural stopping points:

- **After Phase 0:** known correctness bugs are fixed and descriptor coverage
  exists.
- **After Phase 2:** backend semantic ownership is substantially cleaner even
  before the large topology split.
- **After Phase 4:** the main architectural ownership guarantee is enforced by
  the type system.
- **After Phase 5:** adding new semantic variants becomes safely compiler-driven.

The architectural argument is complete only when `ModelGraph` is final,
`ExecutionTopology` is derivation-only, and execution backends cannot observe
checkpoint/import state as a semantic source.
