# Celeg SOLID Refactoring and OpenAI-Compatible Tool Calling Implementation

You are working directly on the GitHub repository `celsowm/celeg`.

Celeg is a native C++20 LLM inference runtime supporting multiple model architectures, CPU and CUDA backends, GGUF and Safetensors checkpoints, an OpenAI-compatible HTTP server, a tokenizer/chat-template layer, and a C API.

Your task is to perform a careful architectural refactoring and implement first-class OpenAI-compatible tool calling without degrading performance, introducing architecture-specific checks into generic code, or turning the inference server into an agent/tool-execution runtime.

Do not blindly rewrite the project. First inspect the current repository, understand the existing architecture, tests, conventions, CMake source manifests, public APIs, model-resolution pipeline, serving protocol, chat templates, schedulers, CPU/CUDA boundaries, and current support for LFM2/LFM2.5, Granite, and Gemma4.

The implementation must preserve existing functionality and proceed incrementally with tests after each meaningful phase.

# Primary goals

1. Improve the actual SOLID compliance of the codebase, not merely its documentation.
2. Remove architecture-specific and checkpoint-format-specific knowledge from generic runtime and backend code.
3. Make adding future architectures possible without editing multiple unrelated central switches.
4. Implement OpenAI-compatible tool calling for `/v1/chat/completions`.
5. Keep CPU and CUDA inference backends completely unaware of chat roles, tools, JSON Schema, OpenAI DTOs, and tool-call syntax.
6. Keep tool execution and agent loops outside the core Celeg inference server.
7. Preserve low-overhead, closed representations where they are beneficial in performance-critical execution paths.
8. Add explicit capability modeling instead of relying on exceptions from supposedly substitutable chat-template implementations.
9. Avoid creating another oversized generic “superset” structure that grows every time a model architecture is added.
10. Deliver production-quality code, tests, documentation, and OpenAPI updates.

# Existing architectural concerns to verify

Inspect the current implementation and confirm or correct the following observations before modifying code.

## Generic and architecture boundaries

The repository already defines an `IArchitecture` abstraction and an `ArchitectureCatalog`, but built-in architecture creation is centrally hardcoded.

The bootstrap process currently obtains built-in checkpoint-format and architecture catalogs internally rather than receiving them through composition or dependency injection.

Generic or supposedly neutral code still contains concrete checkpoint-format knowledge, including direct references to GGUF-related concrete types or fields.

Neutral tensor concepts and repository interfaces appear to be declared inside a Safetensors-specific header. These abstractions must not belong to a concrete checkpoint format.

Generic model weight-role headers currently include concrete format headers and also contain architecture-specific naming policies such as Celeg/LFM, Granite, and Gemma policies.

The base runtime source manifest appears to include CPU- and CUDA-specific compiler or execution-plan source files. Verify whether the target graph truly respects its declared dependency direction.

The architecture-boundary checker should be improved so that it catches semantic leaks such as checking concrete GGUF pointers instead of only searching for strings like `is_gguf`.

## Model representation

The current model-resolution pipeline appears to maintain several overlapping representations:

- model definition;
- runtime topology;
- model graph;
- weight plan;
- compiled model program;
- backend-specific compiled state.

Determine which representation is authoritative at each phase.

The current `RuntimeTopology` appears to behave like a manual tagged union or superset DTO containing attention, convolution, MoE, shared-KV, and per-layer arrays simultaneously.

Verify whether it allows meaningless or invalid states, such as requiring attention layouts for convolution-only layers or depending on implicit conventions.

Verify assumptions such as:

- all MoE layers forming a suffix after dense layers;
- a fixed number of shared-KV groups;
- backend code mentioning Gemma directly;
- architecture-specific failure messages inside CPU or CUDA model loading;
- backend decisions based on concrete checkpoint-format types.

Do not preserve these assumptions unless they are explicitly encoded as validated generic graph semantics.

## CPU and serving responsibilities

Inspect whether the CPU compiled-model shared state currently performs too many unrelated responsibilities, such as:

- CPU feature detection;
- thread-pool construction;
- checkpoint bootstrap;
- model resolution;
- model compilation;
- weight loading;
- packed-weight cache handling;
- KV-cache planning;
- architecture-specific shared-KV behavior.

Inspect whether the CPU scheduler driver combines:

- admission;
- request registry;
- prioritization;
- NUMA placement;
- prefix caching;
- worker lifecycle;
- prefill/decode execution;
- metrics;
- cancellation;
- failure handling.

Refactor only where it improves ownership and testability. Do not split classes mechanically into meaningless one-method wrappers.

## Chat templates and substitutability

The current chat-template interface supports roles that some concrete implementations reject at runtime.

This is an LSP and capability-modeling problem. A caller should not need to discover supported roles only by attempting rendering and catching an exception.

The current closed `ChatTemplateKind` enum and centralized factory logic should not become the growing integration point for every new model or tool-call format.

# Architectural principles to enforce

## Dependency direction

The intended direction should be approximately:

```text
protocol adapters / HTTP routes / C API
                ↓
application services / chat orchestration
                ↓
model-neutral runtime contracts
                ↓
compiled model graph and execution interfaces
                ↓
CPU or CUDA backend implementations
```

Model architecture plugins may produce model-neutral graph, weight, tokenizer, and chat-profile descriptions.

Checkpoint-format plugins may expose metadata and weight repositories through neutral interfaces.

CPU and CUDA backends must depend on neutral compiled-model contracts, not on LFM, Granite, Gemma, GGUF, Safetensors, OpenAI, or chat-template types.

## Switches and variants

Do not treat every switch or `std::variant` as an anti-pattern.

Closed variants are acceptable when they represent the intentionally closed set of operators understood by a compiled backend, particularly in performance-sensitive code.

For example, a backend-level variant for:

- attention versus short convolution;
- dense feed-forward versus mixture-of-experts;

may be appropriate if these are explicit graph operator kinds.

The actual problem is architecture-oriented branching such as:

```cpp
if (architecture_id == "gemma4") { ... }
```

inside generic code or backends.

Likewise, checkpoint-format branching such as:

```cpp
if (checkpoint.gguf != nullptr) { ... }
```

must not be used to select runtime behavior.

Use semantic graph capabilities, neutral interfaces, or compile-time operator variants instead.

## Invalid states

Prefer representations that make invalid states difficult to construct.

Do not replace one large `RuntimeTopology` superset with another equally large collection of optional fields.

Prefer explicit structures such as:

```cpp
struct AttentionNodeSpec;
struct ShortConvolutionNodeSpec;
struct DenseFeedForwardNodeSpec;
struct MixtureOfExpertsNodeSpec;
struct SharedKvBinding;
struct LayerGraph;
struct ModelGraph;
```

A layer should contain only the mixer and feed-forward information applicable to that layer.

Architecture-specific metadata may exist while resolving a model, but it must not leak into the compiled backend contract.

## Composition roots and catalogs

Built-in catalogs may still exist as convenience factories, but core bootstrap functions should be able to receive catalogs or registries explicitly.

Target a structure similar to:

```cpp
struct ModelBootstrapDependencies {
    const ICheckpointFormatCatalog& checkpoint_formats;
    const IArchitectureCatalog& architectures;
};
```

Provide a default composition root for CLI/server/C API usage, but make the core model-loading workflow independently testable with custom catalogs.

# Required SOLID refactoring

Implement the following improvements unless repository inspection reveals a better equivalent.

## 1. Extract neutral checkpoint and tensor contracts

Create neutral headers for concepts such as:

```text
include/celeg/checkpoint/tensor.hpp
include/celeg/checkpoint/weight_repository.hpp
include/celeg/checkpoint/checkpoint_view.hpp
```

Move neutral types out of Safetensors- or GGUF-specific headers, including as applicable:

- tensor dtype;
- tensor locator;
- host tensor view;
- weight repository;
- locatable tensor repository;
- random-access tensor reader;
- checkpoint metadata/view.

A neutral `CheckpointView` must not expose a concrete `GgufFile`, `SafeTensorFile`, or format-specific pointer.

If a format has optional behavior, model it as a neutral capability interface or repository capability.

Keep concrete implementations under format-specific paths.

## 2. Move architecture-specific naming policies

Move LFM/Celeg, Granite, and Gemma tensor-naming policies into their respective architecture modules.

The generic weight-planning layer may depend on `ITensorNamingPolicy`, but it must not know every concrete policy.

Architecture registration should provide the naming policy or weight-plan builder as part of the resolved architecture result.

Adding a future architecture must not require modifying a generic central switch containing all tensor names.

## 3. Remove backend architecture leaks

Remove all architecture names, architecture IDs, model-type checks, and checkpoint-format checks from CPU and CUDA backends.

In particular:

- remove fixed-size shared-KV assumptions;
- replace hardcoded two-group vectors with graph-derived ownership/binding data;
- remove errors mentioning Gemma from backend-generic code;
- remove suffix-based MoE assumptions unless represented explicitly in the model graph;
- derive behavior from compiled semantic graph nodes and bindings.

Add tests that construct synthetic graphs so shared-KV ownership and MoE placement can be validated independently of any named architecture.

## 4. Clarify model-resolution phases

Define and document the authoritative representation in each phase.

A desired pipeline is:

```text
checkpoint metadata
    ↓
architecture resolution
    ↓
validated ModelGraph + WeightPlan + model metadata
    ↓
backend compilation
    ↓
backend-specific CompiledModel
```

Minimize duplicated information between `RuntimeTopology`, `ModelGraph`, `WeightPlan`, and `CompiledModelProgram`.

If `RuntimeTopology` remains, restrict it to architecture-resolution internals and avoid exposing it as a second long-lived source of truth.

Prefer building the validated graph directly from architecture metadata.

Do not make backends repeatedly infer graph semantics from flattened global counters.

## 5. Split oversized responsibilities

Refactor CPU model initialization into cohesive components, for example:

```text
CpuRuntimeEnvironment
CpuModelCompiler
CpuWeightLoader
CpuKvCachePlanner
CpuPackedWeightCache
CpuCompiledModelState
```

Names may differ, but responsibilities must be clear.

Refactor scheduling only where cohesive collaborators can be identified, for example:

```text
RequestRegistry
AdmissionController
SchedulingPolicy
PrefixCacheCoordinator
WorkerCoordinator
GenerationExecutor
SchedulerMetrics
```

Do not over-fragment the code. Each extraction should have meaningful state, tests, or replaceable policy.

## 6. Improve architecture-boundary enforcement

Extend the boundary-checking script and CI tests to detect:

- includes of model-specific headers from backend directories;
- includes of format-specific headers from neutral directories;
- architecture-name literals in backend code;
- concrete checkpoint-format fields or type names in runtime decisions;
- CUDA includes in neutral public headers;
- accidental reverse dependencies in CMake manifests.

Prefer structural include/path checks in addition to string heuristics.

# Tool calling architecture

Implement OpenAI-compatible tool calling as a protocol and chat-orchestration feature.

Do not implement tool execution inside the core `/v1/chat/completions` route.

The expected lifecycle is:

```text
Client sends messages and tools
        ↓
Celeg renders a model-specific tool-aware prompt
        ↓
The model generates text and/or structured tool calls
        ↓
Celeg returns assistant.tool_calls
        ↓
The client executes those tools
        ↓
The client sends assistant tool_calls and role=tool results
        ↓
Celeg renders the continued conversation
        ↓
The model generates the final assistant response
```

# Domain model for tool calling

Create neutral domain types, separate from OpenAI wire DTOs.

A possible location is:

```text
include/celeg/text/tool_call.hpp
include/celeg/text/conversation.hpp
include/celeg/text/chat_profile.hpp
include/celeg/text/tool_call_codec.hpp
```

Use equivalent naming if the project already has a better module structure.

The domain should represent at least:

```cpp
struct JsonSchema {
    std::string serialized;
};

struct ToolFunction {
    std::string name;
    std::string description;
    JsonSchema parameters;
    bool strict = false;
};

struct ToolDefinition {
    std::string type;
    ToolFunction function;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;
};

enum class ToolChoiceMode {
    None,
    Auto,
    Required,
    Specific,
};

struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
    std::string function_name;
};
```

Do not initially build a full C++ object model for every possible JSON Schema keyword.

The inference runtime only needs to preserve and render the schema. Keeping the schema as validated or serialized JSON avoids unnecessary coupling and accidental loss of unsupported properties.

# Structured chat messages

The current `role + string content` representation is insufficient.

Support messages with:

- optional textual content;
- assistant tool calls;
- tool-call IDs;
- optional names where required by the protocol;
- `role=tool` results.

Use a semantic equivalent of:

```cpp
struct ChatMessage {
    ChatRole role;
    std::optional<std::string> content;
    std::vector<ToolCall> tool_calls;
    std::optional<std::string> tool_call_id;
    std::optional<std::string> name;
};
```

Enforce invariants:

- a `tool` message requires `tool_call_id`;
- a `tool` message requires a result content value;
- only assistant messages may contain `tool_calls`;
- an assistant message may contain content, tool calls, or both;
- messages that contain neither content nor tool calls are invalid;
- tool-call IDs referenced by results should correspond to prior assistant calls when conversation validation is enabled;
- duplicate tool-call IDs in a single assistant turn are invalid.

Keep protocol validation errors explicit and suitable for a `400 Bad Request`.

# Chat capabilities and LSP

Do not force every `IChatTemplate` implementation to support every role or tool-call syntax.

Introduce explicit capabilities.

A possible design is:

```cpp
struct ChatCapabilities {
    bool system_role = true;
    bool developer_role = false;
    bool tool_role = false;
    bool assistant_tool_calls = false;
    bool parallel_tool_calls = false;
    bool strict_tool_arguments = false;
};

struct ChatProfile {
    std::string id;
    std::shared_ptr<const IChatTemplate> chat_template;
    std::shared_ptr<const IToolCallCodec> tool_call_codec;
    ChatCapabilities capabilities;
};
```

The actual ownership type may be references, unique pointers, shared pointers, or immutable catalog entries according to existing project conventions.

The important requirement is that callers can inspect capabilities without invoking unsupported methods.

Profiles without tool support must return a clear protocol error when the request includes tools or tool messages.

Do not silently inject a generic JSON prompt for every model and claim native tool-call compatibility.

# Tool-call codec

Tool-call syntax is model/chat-profile-specific.

Introduce an interface equivalent to:

```cpp
class IToolCallCodec {
public:
    virtual ~IToolCallCodec() = default;

    virtual bool supports_parallel_calls() const noexcept = 0;

    virtual std::string render_tool_definitions(
        std::span<const ToolDefinition> tools,
        const ToolChoice& choice) const = 0;

    virtual std::string render_assistant_tool_calls(
        std::span<const ToolCall> calls) const = 0;

    virtual std::string render_tool_result(
        const ChatMessage& message) const = 0;

    virtual ToolParseResult parse_generation(
        std::string_view generated) const = 0;
};
```

The parser result should distinguish:

```cpp
enum class ToolParseStatus {
    NotToolCall,
    Complete,
    Incomplete,
    Invalid,
};
```

A possible result:

```cpp
struct ToolParseResult {
    ToolParseStatus status = ToolParseStatus::NotToolCall;
    std::string assistant_text;
    std::vector<ToolCall> calls;
    std::string error;
    std::size_t consumed_bytes = 0;
};
```

The `Incomplete` state is mandatory for future streaming support.

Do not assume that each generated token chunk contains a complete JSON object.

# Prompt rendering

Separate semantic chat rendering from tokenization.

The current protocol mapping should not directly own every part of:

- OpenAI DTO validation;
- domain conversion;
- chat capability validation;
- tool rendering;
- chat-template rendering;
- tokenization;
- generation-option construction.

Introduce cohesive stages such as:

```cpp
struct ChatRenderRequest {
    std::vector<ChatMessage> messages;
    std::vector<ToolDefinition> tools;
    ToolChoice tool_choice;
    bool add_generation_prompt = true;
};

class ChatPromptRenderer {
public:
    std::string render(const ChatRenderRequest& request) const;
};
```

The resulting flow should be approximately:

```text
OpenAI request DTO
    ↓
protocol validation and domain mapping
    ↓
ChatRenderRequest
    ↓
ChatPromptRenderer + ChatProfile
    ↓
rendered prompt text
    ↓
tokenizer
    ↓
GenerateRequest
```

The tokenizer should tokenize text and expose chat-related special tokens where needed, but it should not become the owner of OpenAI tool semantics.

# OpenAI DTO support

Extend `/v1/chat/completions` request DTOs to support:

- `tools`;
- `tool_choice`;
- `parallel_tool_calls`;
- assistant messages with `tool_calls`;
- `tool` messages with `tool_call_id`;
- optional or null assistant `content`.

Support function tools with:

- `type = "function"`;
- function `name`;
- optional `description`;
- `parameters` JSON Schema;
- optional `strict`.

Support `tool_choice` forms:

```json
"none"
```

```json
"auto"
```

```json
"required"
```

and:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather"
  }
}
```

Reject unsupported or malformed combinations with useful errors.

Preserve `function.arguments` as a JSON string in responses.

Do not return it as a nested JSON object.

# OpenAI response support

A completed tool-call response should follow the expected shape:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 0,
  "model": "...",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_...",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Rio de Janeiro\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

Support assistant text and tool calls coexisting when the model format allows it.

Add `FinishReason::ToolCalls` to the application/protocol-facing completion semantics.

Do not make CPU or CUDA request lifecycle objects parse tool calls.

# Generation interpretation

CPU and CUDA should continue producing tokens.

Add a layer above backend generation that interprets generated assistant output.

A possible abstraction is:

```cpp
class ChatGenerationInterpreter {
public:
    ChatGenerationDelta consume(
        std::span<const std::int32_t> tokens,
        bool finished);
};
```

It should:

- incrementally decode generated tokens safely;
- preserve UTF-8 boundaries;
- distinguish assistant text from tool-call syntax;
- use the selected `IToolCallCodec`;
- expose completed calls;
- expose incomplete parser state;
- produce protocol-neutral deltas;
- determine whether the final semantic finish reason is `tool_calls`.

A backend event ending with EOS may therefore become either:

- `stop`, for normal assistant text;
- `tool_calls`, for successfully parsed tool calls;
- `error`, if the model emitted a clearly intended but malformed tool-call structure and the selected policy treats it as fatal.

Document the selected malformed-output policy.

# Initial stopping behavior

For the first implementation, allow the model to generate through its normal EOS or configured stop token.

Parse the complete generated output afterward.

Do not initially integrate JSON grammar constraints or token-level stopping into every sampler.

After non-streaming tool calling is stable, optionally introduce a generic stop-detector abstraction above the backend execution loop:

```cpp
class IGenerationStopDetector {
public:
    virtual StopDecision consume(
        std::span<const std::int32_t> tokens) = 0;
};
```

This may recognize a model-specific completed tool-call marker and request a clean stop.

The stop detector must not contain architecture-name checks.

# Streaming tool calls

Implement non-streaming tool calling first.

Then add streaming support with OpenAI-compatible deltas.

The first tool-call chunk may include:

```json
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "tool_calls": [
          {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\":\"Rio"
            }
          }
        ]
      },
      "finish_reason": null
    }
  ]
}
```

Subsequent chunks should append argument fragments without repeating fields unnecessarily.

The final chunk should contain:

```json
{
  "choices": [
    {
      "index": 0,
      "delta": {},
      "finish_reason": "tool_calls"
    }
  ]
}
```

Ensure:

- stable tool-call indexes;
- stable IDs;
- IDs are emitted once;
- function names are emitted predictably;
- argument fragments concatenate exactly to the final string;
- partial UTF-8 is not emitted incorrectly;
- incomplete JSON is not treated as an error before generation finishes;
- multiple parallel calls preserve ordering.

# Tool execution is out of scope for the inference server

Do not add filesystem access, subprocess execution, HTTP callbacks, plugin loading, credentials, approval handling, or function execution to `chat_completions.cpp`.

The core architecture should remain:

```text
celeg runtime
    model loading, compilation, scheduling, token generation

celeg OpenAI server
    HTTP protocol, request validation, prompt rendering,
    tool-call encoding and decoding

optional external celeg-agent component
    tool registry, approvals, execution, retries, agent loop
```

An optional future agent component may define:

```cpp
class IToolExecutor {
public:
    virtual ToolExecutionResult execute(
        std::string_view name,
        std::string_view arguments_json,
        const ToolExecutionContext& context) = 0;
};
```

Do not implement this executor in the main inference runtime as part of this task.

# Model support strategy

Tool calling is a capability of a model/chat profile, not of CPU or CUDA.

Implement the infrastructure independently of named architectures.

Then select exactly one model/profile whose actual chat format supports tool calling and implement one real codec for that profile.

Before implementing a codec, verify the expected prompt and response syntax from the model’s tokenizer/chat-template metadata or authoritative model documentation.

Do not invent a supposed “native” format based solely on intuition.

For current profiles without a verified tool-call format:

- report `assistant_tool_calls = false`;
- reject tool requests clearly;
- continue supporting normal chat unchanged.

A generic prompted-JSON fallback may be added only as an explicitly experimental profile, for example:

```text
prompted-json-tools-experimental
```

It must not be advertised as equivalent to native tool calling.

# Catalog design

Avoid expanding a central enum and switch for every profile.

Prefer a catalog such as:

```cpp
class IChatProfileCatalog {
public:
    virtual const ChatProfile& find(
        std::string_view profile_id) const = 0;
};
```

Architecture resolution may produce or reference a `chat_profile_id`.

The server composition root then obtains the profile from the catalog.

Do not write this in the route:

```cpp
if (architecture_id == "granite") {
    ...
} else if (architecture_id == "gemma4") {
    ...
}
```

The route must depend on a resolved chat profile.

# Compatibility requirements

Preserve existing behavior for requests that do not use tools.

Existing plain-text chat completions must produce the same prompt and response semantics unless an existing bug is corrected and documented.

Preserve existing:

- CPU generation;
- CUDA generation;
- streaming text;
- non-streaming text;
- cancellation;
- usage accounting;
- tokenization endpoint;
- C API behavior;
- checkpoint loading;
- architecture-resolution tests.

Avoid public ABI breaks where practical.

If a public API break is necessary, document it and update all call sites and tests in the same change.

# Validation requirements

Validate at least:

- request contains messages;
- role names are recognized;
- content requirements by role;
- tools use supported types;
- function names are non-empty;
- function schemas are valid JSON values;
- selected function in `tool_choice` exists in `tools`;
- `parallel_tool_calls=true` is rejected for profiles without support;
- `required` is rejected when tools are empty;
- tool messages reference tool-call IDs;
- assistant tool calls have unique IDs;
- function arguments remain strings;
- profile capabilities permit every requested feature.

Return structured OpenAI-style error objects rather than constructing JSON by string concatenation.

Introduce a reusable error DTO, for example:

```cpp
struct ErrorDetailDto {
    std::string message;
    std::string type;
    std::optional<std::string> param;
    std::optional<std::string> code;
};

struct ErrorResponseDto {
    ErrorDetailDto error;
};
```

Ensure messages are JSON-escaped through the serializer.

# Tests

Add focused unit and integration tests.

At minimum, cover:

```text
plain_chat_request_remains_compatible
tools_request_without_supported_profile_returns_400
developer_role_without_capability_returns_400
tool_role_without_capability_returns_400
assistant_message_accepts_tool_calls_without_content
assistant_message_accepts_text_and_tool_calls
tool_message_requires_tool_call_id
tool_message_requires_content
tool_choice_required_requires_tools
specific_tool_choice_must_reference_declared_tool
duplicate_tool_call_ids_are_rejected
single_tool_call_round_trip
tool_result_round_trip
multiple_tool_calls_preserve_order
plain_text_is_not_misclassified_as_tool_call
malformed_tool_arguments_follow_documented_policy
finish_reason_is_tool_calls
function_arguments_are_returned_as_json_string
usage_counts_generated_tool_tokens
streamed_arguments_reassemble_exactly
streamed_tool_ids_are_stable
streaming_final_reason_is_tool_calls
unsupported_parallel_calls_return_400
custom_architecture_catalog_can_be_injected
custom_checkpoint_format_catalog_can_be_injected
backend_sources_do_not_include_model_specific_headers
neutral_sources_do_not_include_format_specific_headers
shared_kv_groups_are_graph_driven
moe_layers_need_not_be_a_suffix
```

Add parser tests that feed generated text in every possible split position, including splitting:

- inside a function name;
- inside escaped JSON strings;
- between UTF-8 bytes;
- between tool calls;
- immediately before the closing marker.

For each complete sample, test every possible two-chunk split.

# OpenAPI and documentation

Update the bundled OpenAPI document to describe:

- tools;
- function definitions;
- JSON Schema parameters;
- tool choice;
- parallel tool calls;
- assistant tool calls;
- tool messages;
- streaming tool-call deltas;
- `finish_reason = "tool_calls"`;
- capability-related errors.

Update architecture documentation with:

```text
checkpoint format plugin
architecture resolver
validated model graph
backend compiler
chat profile
chat template
tool-call codec
protocol adapter
```

Update the “how to add a model architecture” documentation to explain that an architecture may register or select:

- metadata resolution;
- graph construction;
- weight naming/planning;
- tokenizer configuration;
- chat profile;
- optional tool-call codec.

Make it clear that a backend must not be edited merely because a new architecture was added, unless that architecture introduces a genuinely new graph operator unsupported by the backend.

# Implementation phases

Proceed in small, reviewable phases.

## Phase 0: repository analysis

Before editing:

1. Inspect the current default branch.
2. Identify all relevant files.
3. Run or inspect existing tests.
4. Produce a concise implementation map.
5. Identify public APIs and likely compatibility risks.
6. Confirm which current model profile can truthfully support native tool calling.

Do not begin with broad file moves before understanding the dependency graph.

## Phase 1: architectural boundary cleanup

Implement:

- neutral tensor/checkpoint repository headers;
- removal of concrete GGUF/Safetensors dependencies from neutral code;
- injectable checkpoint and architecture catalogs;
- improved architecture-boundary checks;
- CMake target/source-manifest corrections.

Keep runtime behavior unchanged.

## Phase 2: graph and backend cleanup

Implement:

- graph-driven shared-KV ownership;
- removal of fixed group counts;
- removal of architecture names from backends;
- removal of MoE suffix assumptions;
- consolidation of duplicated model representations where safe;
- focused CPU model-loading responsibility extraction.

Keep generated outputs unchanged.

## Phase 3: structured conversation domain

Implement:

- structured `ChatMessage`;
- tool definitions and calls;
- tool choice;
- chat capabilities;
- chat-profile catalog;
- protocol-neutral validation.

Existing chat remains compatible.

## Phase 4: OpenAI wire support without active codecs

Implement:

- request and response DTOs;
- OpenAI error DTOs;
- mapping and validation;
- OpenAPI updates;
- explicit rejection for profiles without tool support.

At the end of this phase, the protocol understands tools but does not falsely claim model support.

## Phase 5: first verified tool-call codec

Implement one real, verified model-specific codec.

Support:

- rendering tool definitions;
- rendering prior assistant tool calls;
- rendering tool results;
- parsing one or more generated tool calls;
- non-streaming responses;
- `finish_reason = "tool_calls"`.

## Phase 6: streaming

Implement:

- incremental parser;
- tool-call deltas;
- stable IDs and indexes;
- exact argument-fragment reconstruction;
- final streaming reason.

## Phase 7: optional stopping and constraints

Only after all prior phases are stable, evaluate:

- incremental tool-call completion stopping;
- grammar-constrained arguments;
- reusable token constraints;
- CPU/CUDA sampler integration.

Do not block the core implementation on constrained decoding.

# Deliverables

Produce:

1. The code changes.
2. New and updated tests.
3. Updated architecture documentation.
4. Updated OpenAPI schema.
5. A migration note for any public API changes.
6. A summary of files changed by phase.
7. A list of remaining limitations.
8. Exact commands used to build and test.
9. Evidence that CPU and CUDA backends contain no tool-call or OpenAI logic.
10. Evidence that current non-tool chat behavior remains compatible.

# Quality bar

The final implementation must not:

- put architecture-name branches in CPU or CUDA;
- expose concrete checkpoint formats through neutral contracts;
- add tool execution to the HTTP inference route;
- use string concatenation to build JSON errors;
- claim tool support for models without a verified codec;
- parse each streaming chunk as an independent complete JSON document;
- require every chat template subtype to support every role;
- add another global switch containing every architecture and chat profile;
- encode a fixed number of shared-KV groups;
- assume all MoE layers are contiguous or form a suffix;
- duplicate the same semantic truth across multiple mutable representations without explicit ownership;
- regress existing text chat, tokenization, CPU, CUDA, C API, or checkpoint behavior.

# Decision-making rules

When multiple designs are possible:

1. Prefer semantic graph information over architecture IDs.
2. Prefer capabilities over exception-based discovery.
3. Prefer composition-root defaults plus injectable core dependencies.
4. Prefer neutral interfaces over concrete checkpoint-format pointers.
5. Prefer validated structured domain objects over protocol DTOs flowing into the runtime.
6. Prefer incremental refactoring over a complete rewrite.
7. Prefer measurable simplification over creating abstractions with no meaningful ownership.
8. Preserve hot-path efficiency.
9. Keep model-specific serialization syntax inside the model/chat-profile module.
10. Keep OpenAI-specific shapes inside the serving protocol module.

# Working method

Work directly from the repository rather than relying only on this prompt.

For every phase:

1. Inspect the relevant current code.
2. Explain the concrete issue found.
3. State the intended boundary after the change.
4. Implement the smallest coherent change.
5. Update or add tests.
6. Build and run the relevant test suite.
7. Fix failures before continuing.
8. Keep commits or logical change groups reviewable.

When you discover that one of the assumptions in this prompt is inaccurate, do not force the proposed class name or structure. Preserve the architectural objective and adapt the implementation to the actual codebase.

At the end, provide a final report with:

- SOLID assessment before and after;
- architectural boundary improvements;
- tool-calling architecture;
- supported chat profiles;
- unsupported profiles and reasons;
- test results;
- compatibility risks;
- follow-up work, especially constrained decoding and optional agent-runtime integration.
