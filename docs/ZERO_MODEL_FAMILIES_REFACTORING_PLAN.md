# CELEG Zero Model Families Refactoring Plan

## Status

Proposed architectural refactor.

Target repository: `celsowm/celeg`.

This document is the umbrella plan for making CELEG genuinely model-agnostic. The existing `DESCRIPTORLESS_CHECKPOINT_INFERENCE_REFACTORING_PLAN.md` remains useful, but becomes one subproblem of this larger refactor: descriptorless structural/semantic inference is necessary, but it is not sufficient while tokenizer, chat, tool calling, vision, multimodal support, and runtime composition still know concrete model families.

## Primary goal

> CELEG must support semantics, not model families.

The refactor is complete only when the following production directories no longer exist:

```text
src/models/
include/celeg/models/
```

Renaming them to `architectures/`, `families/`, `plugins/`, `profiles/`, or another family-oriented directory does not satisfy this plan.

---

# 1. Problem statement

CELEG has already moved most numerical execution toward backend-neutral semantic resolution:

```text
CheckpointView
      |
      v
InferenceInput
      |
      v
CanonicalModelFacts
      |
      v
ResolutionAssembler
      |
      v
ResolvedModel
      |
      +--> CPU compiler
      |
      +--> CUDA compiler
```

That is the correct direction. CPU and CUDA should execute semantic programs and capability requirements, never branch on model identity.

However, the repository still contains family-owned production code under paths such as:

```text
src/models/lfm2/
src/models/gemma4/
src/models/granite/
src/models/minicpm5/
src/models/nanbeige/
src/models/smollm3/
src/models/qwen35/
src/models/nemotron_h/
src/models/muse_glimmer/
```

Those files still own responsibilities such as:

- chat templates;
- tool-call encoding/decoding;
- tokenizer behavior;
- vision preprocessing;
- multimodal conventions;
- family-specific runtime registration.

The composition root also explicitly knows model families.

Therefore the current architecture is still effectively:

```text
model-agnostic numerical execution
        +
model-aware integration layer
        =
not fully model-agnostic
```

The final architecture must remove that contradiction.

---

# 2. Architectural principle

Production CELEG code may know reusable semantics such as:

```text
GroupedQueryAttention
SlidingWindowAttention
RoPE
M-RoPE
AdjacentPairRoPE
SwiGLU
MoE
LatentAttention
RecurrentMixer
VisionPatchEmbedding
ImageProjection
ToolCallGrammar
ChatTemplateProgram
BPE
SentencePieceNormalization
```

Production runtime code must not need concepts such as:

```text
Gemma
Qwen
LFM
Granite
MiniCPM
Nemotron
Muse
GPT-X
```

A model repository name, `model_type`, or architecture string may exist as:

- provenance;
- diagnostics;
- importer evidence;
- source metadata.

It must not directly select runtime behavior.

Forbidden:

```cpp
if (model_type == "qwen3_5") {
    use_mrope();
}
```

Forbidden:

```cpp
switch (architecture) {
    case Architecture::Gemma:
        ...
}
```

Forbidden:

```cpp
make_qwen35_vision_provider();
```

Desired:

```cpp
VisionPipelineSpec spec = infer_vision_pipeline(checkpoint);
vision_runtime.compile(spec);
```

The implementation must be selected by semantic facts, not family identity.

---

# 3. Non-negotiable acceptance criterion

A previously unknown Hugging Face-compatible checkpoint must load without CELEG source changes when its behavior can already be represented by CELEG's semantic vocabulary.

For an unknown repository such as:

```text
SomeOrganization/FooBar-17B
```

if all required semantics are already supported, adding support must require:

```text
0 new .cpp files
0 new .cu files
0 new family classes
0 new family registrations
0 new model-name branches
0 edits to builtin runtime composition
```

Only genuinely new mathematics, tokenizer behavior, interaction protocol semantics, or multimodal primitives may require new CELEG code.

Even then, the code must name and implement the reusable semantic concept rather than the first model that introduced it.

Bad:

```cpp
class FooBarAttention;
```

Good:

```cpp
class OrthogonalizedValueAttention;
```

Bad:

```cpp
FooBarToolCodec;
```

Good:

```cpp
TaggedJsonToolCallGrammar;
```

---

# 4. Desired end-state architecture

```text
                         Checkpoint
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
      Metadata            Tensors          Assets/config
          |                  |                  |
          +------------------+------------------+
                             |
                             v
                  Checkpoint Normalization
                             |
                             v
                       Evidence Set
                             |
                             v
                    Semantic Inference
                             |
                             v
                    CanonicalModelFacts
                             |
         +-------------------+-------------------+
         |                   |                   |
         v                   v                   v
     ModelGraph       TokenizerDefinition   InteractionSpec
         |                                       |
         |                          +------------+------------+
         |                          |                         |
         |                          v                         v
         |                  ChatProtocolSpec          VisionPipelineSpec
         |                          |
         |                          v
         |                  ToolProtocolSpec
         |
         v
 SemanticRequirements
         |
   +-----+-----+
   |           |
   v           v
 CPU         CUDA
compiler    compiler
```

No stage after normalization may need a model-family identity.

---

# 5. Relationship with descriptorless inference

`DESCRIPTORLESS_CHECKPOINT_INFERENCE_REFACTORING_PLAN.md` remains an important foundation.

Its core direction is correct:

```text
checkpoint evidence
      |
      v
inference proposals
      |
      v
fact solving
      |
      v
CanonicalModelFacts
```

This plan extends the same philosophy to all remaining family-owned concerns:

```text
model structure   -> semantic facts
model weights     -> semantic tensor roles
chat              -> ChatTemplateProgram
工具/tool calling  -> ToolProtocolSpec / grammar
vision            -> VisionPipelineSpec
multimodal        -> component semantics
runtime assembly  -> generic capabilities only
```

The objective is not only descriptorless model execution. It is zero family-specific production behavior.

---

# 6. Phase 0 — Freeze the architectural boundary

Strengthen `scripts/check_architecture_boundaries.py` before broad migration.

## 6.1 Ban family production directories

The checker must fail if either path exists after the final migration gate:

```text
src/models/
include/celeg/models/
```

During migration, an explicit temporary allow-list may exist, but it must shrink monotonically to zero.

## 6.2 Ban family includes

Production code must not include:

```text
celeg/models/
```

The rule must explicitly cover:

```text
src/model
src/runtime
src/checkpoint
src/backend
src/composition
src/text
src/serve
include/celeg/**
```

This is important because deleting `src/models` is meaningless if family awareness merely migrates into `src/composition` or `src/runtime`.

## 6.3 Ban family dispatch

Detect obvious production behavior switches based on concepts such as:

```text
model_type
architecture_id
repository_id
model_family
family
```

Narrow exceptions may exist for:

```text
checkpoint import
metadata normalization
provenance
resolution reports
```

Even there, the value may only produce evidence or diagnostics. It must not directly choose an execution implementation.

## 6.4 Ban family-prefixed factories

Symbols such as:

```text
make_gemma*
make_qwen*
make_lfm*
make_granite*
```

must disappear from production runtime composition.

---

# 7. Phase 1 — Unified interaction semantics

Chat, tools, and multimodal behavior must become resolved semantic data rather than scattered family implementations.

Introduce a neutral representation along the lines of:

```cpp
struct InteractionSpec {
    ChatProtocolSpec chat;
    ToolProtocolSpec tools;
    std::optional<VisionPipelineSpec> vision;
    std::optional<AudioPipelineSpec> audio;
};
```

The exact decomposition may differ, but the resulting types must be:

- backend-neutral;
- family-neutral;
- immutable after resolution;
- derived from checkpoint/config evidence;
- inspectable for diagnostics;
- narrow enough to preserve SRP.

Do not turn `ResolvedModel` into a god object. Composition is acceptable; ownership of parsing, solving, compilation, and execution must remain separated.

---

# 8. Phase 2 — Remove family-specific tokenizer rules

The tokenizer layer must stop receiving hardcoded model/family identifier tables from runtime composition.

## 8.1 Parse tokenizer semantics directly

For `tokenizer.json`, derive behavior from the actual tokenizer graph:

```text
normalizer
pre_tokenizer
decoder
added_tokens
post_processor
regex patterns
byte fallback
special tokens
```

Example:

```text
regex containing \p{N}{1,3}
```

may produce a neutral behavior such as:

```text
NumericTriplets
```

without knowing which model uses that tokenizer.

## 8.2 Normalize GGUF tokenizer metadata

GGUF fields such as:

```text
tokenizer.ggml.pre
```

must be interpreted inside the GGUF tokenizer adapter and normalized into semantic tokenizer facts.

Desired:

```text
GGUF metadata
      |
      v
GGUF tokenizer adapter
      |
      v
TokenizerBehaviorFacts
      |
      v
TokenizerDefinition
```

Generic tokenizer runtime must not branch on family identifiers.

## 8.3 Acceptance

The built-in tokenizer provider must be constructible without a model-name table.

Desired composition:

```cpp
builder.add_tokenizer_provider(
    make_builtin_tokenizer_provider());
```

Not:

```cpp
make_builtin_tokenizer_provider({
    {"lfm2", ...},
    {"gemma4", ...},
    {"granite", ...},
});
```

---

# 9. Phase 3 — Chat templates become programs, not family classes

Current family-owned chat classes must disappear.

Introduce a neutral compiled representation such as:

```cpp
struct ChatTemplateProgram {
    std::vector<ChatInstruction> instructions;
};
```

Possible semantic instructions include:

```text
EmitLiteral
EmitMessageContent
EmitRole
ForEachMessage
ForEachTool
IfRole
IfToolsPresent
EmitToolSchema
EmitGenerationPrompt
```

The exact internal representation is open, but execution must use a neutral program instead of dispatching to model-specific C++.

## 9.1 Hugging Face / Jinja import

Where available, import:

```text
tokenizer_config.json -> chat_template
```

through a deterministic frontend:

```text
HF chat_template
      |
      v
Template parser
      |
      v
ChatTemplate AST
      |
      v
ChatTemplateProgram
      |
      v
generic formatter
```

Support the subset actually required by real checkpoints. Do not embed unrestricted scripting unless necessary.

Unsupported constructs must fail explicitly at load time, for example:

```text
UnsupportedChatTemplateConstruct
```

There must be no silent fallback to family-name logic.

## 9.2 Descriptor/native import

Descriptors or native CELEG metadata may provide an equivalent declarative representation, but every source must converge to the same `ChatTemplateProgram`.

---

# 10. Phase 4 — Tool calling becomes declarative protocol grammar

Family-owned `tool_call_codec.cpp` implementations must disappear.

Tool calling should be decomposed into neutral responsibilities:

```text
tool definition serialization
assistant tool-call serialization
assistant tool-call parsing
tool response formatting
parallel call representation
```

Introduce a semantic representation such as:

```cpp
struct ToolProtocolSpec {
    ToolDefinitionEncoding definition_encoding;
    ToolCallEncoding call_encoding;
    ToolResponseEncoding response_encoding;
};
```

For more complex cases, use a neutral grammar:

```text
DelimitedBlock
JsonObject
JsonArray
FunctionName
ArgumentsJson
RepeatedCalls
ParallelCallContainer
TextPrefix
TextSuffix
```

Then runtime parsing becomes conceptually:

```cpp
ToolCallParser parser(spec.call_grammar);
```

Never:

```cpp
if (profile == "lfm2") ...
else if (profile == "qwen") ...
```

Capabilities such as developer role, tool role, assistant tool calls, native codecs, and parallel tool calls must be properties of the resolved interaction protocol, not manually attached to model-family registrations.

---

# 11. Phase 5 — Generic vision pipeline

Family modules such as `Gemma4VisionModule` and `Qwen35VisionModule` must disappear.

Vision should follow the same architecture as text execution:

```text
configuration
      |
      v
semantic facts
      |
      v
VisionPipelineSpec
      |
      v
compiled generic pipeline
      |
      v
runtime/backend primitives
```

Introduce reusable semantics, for example:

```cpp
struct VisionPipelineSpec {
    ImageResizeSpec resize;
    ImageNormalizationSpec normalization;
    PatchExtractionSpec patches;
    VisionEncoderSpec encoder;
    VisionProjectionSpec projection;
    ImageTokenPlacementSpec token_placement;
};
```

The exact public API should preserve SRP and avoid a god struct, but the fundamental rule is fixed: configuration resolves into reusable semantics.

Reusable primitives may include:

```text
Resize
Crop
Normalize
Patchify
ConvPatchEmbed
VisionAttention
VisionMLP
PositionEmbedding
PositionInterpolation
FeatureSelection
FeatureProjection
TokenInsertion
ImagePlaceholderExpansion
```

New models compose these primitives. They do not receive new family implementations.

## 11.1 Processor/config import

Infer vision behavior from assets such as:

```text
processor_config.json
preprocessor_config.json
config.json
tokenizer_config.json
```

Format/config-specific spelling is normalized into semantic facts before generic execution.

---

# 12. Phase 6 — Composite checkpoint inference

Automatic inference must stop rejecting composite configurations merely because they contain structures such as:

```text
text_config
vision_config
```

Configuration normalization should recurse into components and produce semantic component facts.

Conceptually:

```text
Root config
   |
   +-- text_config
   +-- vision_config
   +-- projector config
```

becomes:

```text
CompositeModelFacts
   |
   +-- LanguageModelFacts
   +-- VisionEncoderFacts
   +-- ProjectionFacts
```

No family identity is required.

When useful, introduce a neutral semantic component graph:

```text
image
  |
VisionEncoder
  |
Projector
  |
TextDecoder
```

The graph may contain reusable component types such as:

```text
TokenEmbedding
TextDecoder
VisionEncoder
AudioEncoder
Projector
Connector
OutputHead
```

It must never become `QwenVisionModel`, `GemmaVisionModel`, or equivalent family composition.

---

# 13. Phase 7 — Purify `src/composition/builtin_runtime.cpp`

`src/composition/builtin_runtime.cpp` is one of the main remaining model-family dependency hubs and therefore deserves an explicit migration phase.

The file itself may remain as a composition root. The problem is what it currently composes.

Its final purpose must be:

> compose generic CELEG capabilities.

It must not compose known model families.

## 13.1 Remove model-family includes

All includes of the form:

```cpp
#include "celeg/models/..."
```

must disappear from `src/composition/builtin_runtime.cpp`.

The final dependency direction should be:

```text
builtin_runtime.cpp
        |
        +--> checkpoint inference
        +--> descriptor/import infrastructure
        +--> tokenizer engine
        +--> chat-template interpreter
        +--> tool-protocol interpreter
        +--> vision-pipeline compiler
        +--> generic runtime services
```

It must not depend on concrete model names.

## 13.2 Delete `kDeclarativeChats`

A compiled table such as:

```cpp
constexpr BuiltinChatRegistration kDeclarativeChats[] = {
    {"lfm2-chat", ...},
    {"granite-chat", ...},
    {"gemma4-chat", ...},
    {"minicpm5-chat", ...},
    {"smollm3-chat", ...},
    {"nanbeige42-chat", ...},
    {"qwen35-chat", ...},
    {"muse-glimmer-chat", ...},
    {"nemotron-h-chat", ...},
};
```

must not exist in the final architecture.

The runtime registers the generic chat-template engine, not every known template.

Desired:

```cpp
builder.add_chat_template_engine(
    make_chat_template_engine());
```

The actual protocol comes from checkpoint/config resolution.

## 13.3 Remove family-specific vision modules

Delete concepts such as:

```cpp
class Gemma4VisionModule;
class Qwen35VisionModule;
```

The composition root registers one reusable vision pipeline facility.

Desired:

```cpp
builder.add_vision_pipeline_compiler(
    make_builtin_vision_pipeline_compiler());
```

Actual behavior comes from `VisionPipelineSpec`.

## 13.4 Remove tokenizer family tables

Construction like:

```cpp
make_builtin_tokenizer_provider({
    {"lfm2", ...},
    {"smaug-bpe", ...},
    {"gemma4", ...},
    {"gpt2", ...},
    {"granite", ...},
});
```

must disappear.

`builtin_runtime.cpp` should not know tokenizer family identifiers.

## 13.5 Reconsider architecture registration semantics

This line is conceptually compatible with the current migration:

```cpp
builder.add_architecture(make_automatic_architecture());
```

However, after semantic checkpoint resolution becomes canonical, review whether `IArchitecture` remains the right abstraction.

A possible later direction is:

```cpp
builder.add_checkpoint_resolver(
    make_semantic_checkpoint_resolver());
```

Do not perform a rename-only refactor. Change the abstraction only if responsibility genuinely changes.

## 13.6 Descriptor registration

Descriptor loading may remain temporarily, but descriptors must converge into the same canonical fact pipeline.

Desired:

```text
descriptor
    |
    v
DescriptorImporter
    |
    v
CanonicalModelFacts
```

The composition root may register a descriptor fact source/importer, but it must not register descriptor-defined executable model families.

## 13.7 Desired final shape

Conceptually, `builtin_runtime.cpp` should become small:

```cpp
void register_builtin_runtime(RuntimeBuilder& builder) {
    register_checkpoint_resolution(builder);
    register_tokenizer_runtime(builder);
    register_chat_runtime(builder);
    register_tool_protocol_runtime(builder);
    register_vision_runtime(builder);
}
```

Or, if module polymorphism remains useful:

```cpp
std::vector<std::unique_ptr<IRuntimeModule>>
make_builtin_runtime_modules() {
    std::vector<std::unique_ptr<IRuntimeModule>> modules;
    modules.push_back(make_checkpoint_resolution_module());
    modules.push_back(make_tokenizer_runtime_module());
    modules.push_back(make_chat_template_runtime_module());
    modules.push_back(make_tool_protocol_runtime_module());
    modules.push_back(make_vision_pipeline_runtime_module());
    return modules;
}
```

Do not preserve `IRuntimeModule` merely for ceremony. If these capabilities are mandatory and not meaningfully substitutable, direct composition may be more SOLID than interface inflation.

## 13.8 Definition of done for `builtin_runtime.cpp`

This phase is complete when:

- it includes no `celeg/models/...` header;
- it contains no model-family identifier;
- `kDeclarativeChats` no longer exists;
- no model-specific chat implementation is registered;
- no model-specific vision provider is registered;
- no tokenizer rule table contains model/family identifiers;
- it registers only reusable runtime capabilities;
- descriptors, if still present, feed canonical resolution rather than executable family registration;
- adding an unknown compatible model requires a zero-line diff in this file.

The architecture checker must explicitly add `src/composition/**` to the model-family-neutral dependency boundary so family awareness cannot simply move there.

---

# 14. Phase 8 — Reusable import rules, never family plugins

Not every checkpoint will be inferable from perfectly standardized metadata. Import-specific knowledge may remain necessary.

It must be expressed as reusable conventions.

Good:

```text
MetadataAliasRule
TensorNamingGrammar
FusedQkvLayoutRule
ExpertTensorLayoutRule
ChatTemplateImportRule
VisionConfigAliasRule
TokenizerRegexRule
HuggingFaceSplitQkvProjectionGrammar
```

Bad:

```text
GemmaRule
QwenRule
MuseRule
LfmRule
```

Rule names must describe why the rule is valid, not which family first required it.

Do not create a generic `IModelPlugin` with dozens of family-owned callbacks. That would recreate `src/models` behind dependency inversion theater.

---

# 15. Phase 9 — Descriptor convergence

Descriptors may remain as explicit import hints for ambiguous or unusual checkpoints, but they must produce the same canonical representations as automatic inference.

Model semantics:

```text
automatic inference --------+
                            |
descriptor importer --------+--> CanonicalModelFacts
                            |
format-specific importer ---+
```

Interaction semantics:

```text
HF chat template -----------+
                            |
descriptor -----------------+--> ChatTemplateProgram / InteractionSpec
                            |
native metadata ------------+
```

There must not be two implementations of semantic synthesis.

Descriptor-only graph builders, descriptor-only chat execution, or descriptor-only multimodal runtime paths must disappear once convergence is complete.

---

# 16. Phase 10 — Delete `src/models` and public family headers

Migration should happen incrementally, but deletion is mandatory.

Suggested order:

```text
1. tokenizer family rules
2. simple chat templates
3. tool codecs
4. remaining complex chat templates
5. first vision family
6. second vision family
7. composite automatic inference
8. family headers
9. family composition registrations
10. src/models
11. include/celeg/models
```

At the end:

```bash
test ! -d src/models
test ! -d include/celeg/models
```

must succeed.

---

# 17. Phase 11 — Remove model names from production symbol space

Search the production tree for known family names, including current supported families.

Every remaining occurrence must be classified as one of:

```text
test fixture
documentation
checkpoint metadata string
provenance
diagnostics
```

There must be no production behavior implementation named after model families.

This is acceptable:

```text
resolution report: model_type=qwen3_5
```

This is not:

```cpp
if (model_type == "qwen3_5") {
    use_mrope();
}
```

---

# 18. Phase 12 — Architectural and parity tests

## 18.1 Unknown-family clone test

Take a supported checkpoint fixture and mutate only identity metadata:

```text
model_type
architectures
repository name
```

to an invented family such as:

```text
model_type = "celeg_test_unknown_model"
architectures = ["CelegTestUnknownModelForCausalLM"]
```

If those strings are not semantically required, resolution must remain identical.

Compare fingerprints or canonical serialized forms for:

```text
CanonicalModelFacts
ModelGraph
WeightPlan
TokenizerDefinition
ChatTemplateProgram
ToolProtocolSpec
VisionPipelineSpec
```

## 18.2 Family-name poisoning test

Provide misleading family identity while preserving the actual structural evidence.

Inference must follow evidence, not the family label.

## 18.3 Descriptor deletion tests

For checkpoints supported by automatic inference, delete/disable the descriptor and prove that canonical resolution and numerical behavior remain identical.

## 18.4 No-family-linkage test

Build production targets and verify no family-specific runtime symbols remain linked.

## 18.5 Numerical and protocol parity

Existing supported checkpoints must preserve:

```text
CPU numerical parity
CUDA numerical parity
chat formatting golden output
tool-call parse/serialize parity
vision preprocessing parity
vision embedding/projector parity
```

---

# 19. Phase 13 — Backend independence remains mandatory

CPU and CUDA must continue consuming only semantic requirements.

Backend code must not receive or inspect:

```text
model_type
architecture_id
chat_profile
vision_family
repository identity
```

Correct:

```text
model requires M-RoPE
        x
CUDA supports M-RoPE
        |
        v
compile
```

Incorrect:

```text
model is Qwen
        |
        v
use Qwen CUDA path
```

Backend capability validation remains owned by each backend.

---

# 20. Phase 14 — Explainability

Generic inference must remain debuggable.

Extend resolution reporting to surface:

```text
structural facts
semantic facts
tokenizer facts
chat protocol source
tool protocol source
vision pipeline source
tensor bindings
evidence
conflicts
defaults
unsupported features
canonical fingerprints
```

Example:

```text
chat template:
  source: tokenizer_config.json
  parser: hf-jinja-subset
  program fingerprint: ...

vision resize:
  source: preprocessor_config.json
  width: 896
  height: 896

rope:
  kind: mrope
  evidence:
    - normalized rope_scaling metadata
    - positional section configuration
```

Explainability observes resolution. It must never control it.

---

# 21. SOLID requirements

## 21.1 Single Responsibility Principle

Keep these concerns distinct:

```text
metadata normalization
tensor inventory
fact inference
fact solving
tensor binding
chat-template parsing
chat-program execution
tool-call parsing
vision preprocessing
component graph synthesis
backend compilation
composition
```

Do not create a `UniversalModelResolver` that owns all of them.

## 21.2 Open/Closed Principle

A new reusable semantic convention should normally require adding/registering one reusable rule or primitive without editing a central family dispatch table.

## 21.3 Liskov Substitution Principle

Use polymorphic contracts only for genuinely interchangeable strategies.

Do not build interfaces purely to wrap existing family implementations.

## 21.4 Interface Segregation Principle

Prefer narrow extension contracts where useful, such as:

```text
IMetadataInferenceRule
ISemanticInferenceRule
ITensorBindingRule
IChatTemplateSource
IVisionConfigSource
```

Avoid a broad `IModelPlugin` interface.

## 21.5 Dependency Inversion Principle

High-level resolution depends on semantic contracts.

Checkpoint formats and concrete backends depend toward those boundaries.

Semantic layers must not depend on:

```text
HF repository identity
GGUF implementation details
CUDA
CPU kernels
model-family code
```

---

# 22. DRY requirements

There must be one canonical representation for each semantic concern.

One:

```text
CanonicalModelFacts
```

One:

```text
ChatTemplateProgram
```

One:

```text
ToolProtocolSpec
```

One:

```text
VisionPipelineSpec
```

One canonical synthesis path from facts to model graph/weight plan.

One backend capability mechanism.

Do not keep family-specific implementations beside generic implementations indefinitely “for safety”. Once parity exists, delete the old path.

No compatibility layer is required merely to preserve obsolete internal architecture.

---

# 23. Explicitly forbidden anti-patterns

## 23.1 Family plugin architecture

Forbidden end state:

```text
plugins/
  gemma/
  qwen/
  lfm/
```

This only renames the problem.

## 23.2 Giant generic resolver

Forbidden:

```cpp
GenericModelResolver::resolve() {
    if (...) ...
    else if (...) ...
    else if (...) ...
}
```

## 23.3 Model-type confidence

Forbidden reasoning:

```text
model_type says X
therefore assume semantic Y
```

unless a source-format/import specification explicitly defines the mapping and it is converted into normal evidence before solving.

## 23.4 Silent fallback

Unsupported or ambiguous semantics must fail closed.

## 23.5 Runtime probing

Semantic inference occurs at checkpoint load time. Do not probe token-by-token behavior to guess architecture.

## 23.6 Backend-driven inference

CUDA/CPU availability must never affect what the checkpoint means.

## 23.7 Compatibility duplication

Once generic semantics cover a family implementation, delete the family implementation.

---

# 24. Migration strategy: vertical slices

Do not build an enormous generic framework first and migrate models only at the end.

Each vertical slice should:

```text
1. introduce the smallest neutral semantic representation needed;
2. migrate one real existing behavior onto it;
3. prove parity;
4. migrate a second behavior/family where possible;
5. generalize only when the comparison justifies it;
6. delete the old family implementation;
7. strengthen the architecture checker.
```

The abstraction should be proven continuously against real checkpoints.

---

# 25. Suggested implementation slices

## Slice A — tokenizer family registry removal

Goal:

```text
BuiltinTokenizerModule contains zero model identifiers.
```

Deliverables:

- semantic tokenizer behavior inference;
- GGUF normalization;
- tokenizer.json parity;
- tests;
- removal of family-based pre-tokenizer registration.

## Slice B — first chat-template migration

Choose the simplest existing family chat template.

Deliverables:

- `ChatTemplateProgram`;
- generic formatter;
- importer;
- golden parity tests;
- deletion of that family chat implementation.

## Slice C — complex/tool-enabled chat

Choose a profile with:

```text
tools
tool responses
parallel calls
developer/system mapping
```

Deliverables:

- declarative `ToolProtocolSpec`;
- generic parser/serializer;
- parity tests;
- deletion of the corresponding family codec.

## Slice D — remaining chat implementations

Acceptance:

```text
src/models/*/chat_template.cpp == 0
src/models/*/tool_call_codec.cpp == 0
```

## Slice E — first generic vision migration

Deliverables:

- `VisionPipelineSpec`;
- processor/config normalization;
- generic preprocessing;
- generic projection semantics;
- numerical parity.

## Slice F — second vision migration

This is the proof that the abstraction is genuinely semantic.

If the generic pipeline requires family branches to support both existing vision implementations, redesign it before proceeding.

## Slice G — composite automatic inference

Deliverables:

- recursive `text_config` / `vision_config` normalization;
- component semantic graph if required;
- automatic text + vision resolution;
- removal of family-identity rejection paths.

## Slice H — purify runtime composition

Remove:

```text
kDeclarativeChats
Gemma4VisionModule
Qwen35VisionModule
family-specific tokenizer rules
all celeg/models includes from src/composition
```

## Slice I — delete family directories

Delete:

```text
src/models
include/celeg/models
```

Then make their return a CI failure.

---

# 26. CI gates

The final CI should include hard gates for:

1. `src/models` does not exist;
2. `include/celeg/models` does not exist;
3. no model-family includes in production;
4. no model-family dispatch in backend code;
5. no model-family dispatch in composition code;
6. no family registration table in `builtin_runtime.cpp`;
7. unknown-family clone tests pass;
8. family-name poisoning tests pass;
9. descriptorless inference tests pass;
10. CPU numerical parity passes;
11. CUDA numerical parity passes;
12. chat/tool golden tests pass;
13. vision preprocessing/projector parity passes.

---

# 27. Documentation changes

Update `docs/EXTENDING_ARCHITECTURES.md` so it no longer primarily teaches:

```text
How to add support for model X
```

It should instead teach:

```text
How to add a reusable semantic primitive
How to add a metadata inference rule
How to add a tensor naming grammar
How to add a tokenizer behavior
How to add a chat-template construct
How to add a tool-call grammar primitive
How to add a vision operation
How to add a checkpoint/config importer convention
```

The ideal answer to:

> How do I add model FooBar?

should usually be:

> You do not. If FooBar uses semantics CELEG already supports, CELEG should infer it automatically.

---

# 28. Definition of done

This refactor is complete only when all of the following are true:

- `src/models/` does not exist;
- `include/celeg/models/` does not exist;
- production code contains no model-family implementation registry;
- `src/composition/builtin_runtime.cpp` contains no model-specific include or registration;
- tokenizer behavior is inferred from tokenizer semantics rather than family tables;
- chat formatting is represented by a neutral program;
- tool calling is represented by neutral protocol/grammar semantics;
- vision processing is represented by a neutral pipeline;
- composite text/vision checkpoints can be normalized without family identity;
- descriptors and automatic inference converge on the same canonical semantics;
- CPU contains no architecture-family dispatch;
- CUDA contains no architecture-family dispatch;
- checkpoint names do not influence runtime semantics;
- unknown-family clone tests pass;
- family-name poisoning tests pass;
- existing checkpoints preserve numerical and protocol parity;
- architecture CI prevents model-family production code from returning.

---

# 29. Strong final architectural test

Take a supported checkpoint and rewrite only identity metadata:

```json
{
  "model_type": "foobar_9000",
  "architectures": [
    "FooBar9000ForCausalLM"
  ]
}
```

Preserve all actual structural, tokenizer, interaction, and multimodal configuration.

If CELEG still resolves the same:

```text
CanonicalModelFacts
ModelGraph
WeightPlan
TokenizerDefinition
ChatTemplateProgram
ToolProtocolSpec
VisionPipelineSpec
```

then family identity is no longer controlling behavior.

If changing the family name changes execution where no semantic evidence changed, a family dependency still exists.

---

# 30. Architectural north star

The final mental model should be:

> CELEG does not support models. CELEG supports semantics. Checkpoints describe compositions of those semantics.

Therefore:

```text
new model + known semantics
        =
zero CELEG source changes
```

and:

```text
new mathematical/protocol primitive
        =
one reusable semantic extension
```

That is the standard required before CELEG can be described as genuinely model-agnostic without qualification.
