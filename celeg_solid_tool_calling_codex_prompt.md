# Celeg SOLID Refactoring and OpenAI-Compatible Tool Calling Implementation

You are working directly on the GitHub repository `celsowm/celeg`.

Celeg is a native C++20 LLM inference runtime supporting multiple model architectures, CPU and CUDA backends, GGUF and Safetensors checkpoints, an OpenAI-compatible HTTP server, a tokenizer/chat-template layer, and a C API.

Your task is to perform a careful architectural refactoring and implement first-class OpenAI-compatible tool calling without degrading performance, introducing architecture-specific checks into generic code, or turning the inference server into an agent/tool-execution runtime.

The implementation must preserve existing functionality and proceed incrementally with tests after each meaningful phase.

> **Baseline.** The findings in this document were verified against commit `74ee464` on `master` (2026-08-02). Every claim under “Verified repository state” carries a `file:line` citation. Re-check the citations before acting on them: if the repository has moved on, trust the code and preserve the architectural objective rather than the exact line number, class name, or structure proposed here.

# Primary goals

1. Improve the actual SOLID compliance of the codebase, not merely its documentation.
2. Remove architecture-specific and checkpoint-format-specific knowledge from generic runtime, text, and backend code.
3. Make adding future architectures possible without editing multiple unrelated central switches.
4. Implement OpenAI-compatible tool calling for `/v1/chat/completions`.
5. Keep CPU and CUDA inference backends completely unaware of chat roles, tools, JSON Schema, OpenAI DTOs, and tool-call syntax.
6. Keep tool execution and agent loops outside the core Celeg inference server.
7. Preserve low-overhead, closed representations where they are beneficial in performance-critical execution paths.
8. Add explicit capability modeling instead of relying on exceptions from supposedly substitutable chat-template implementations.
9. Avoid creating another oversized generic “superset” structure that grows every time a model architecture is added.
10. Deliver production-quality code, tests, documentation, and OpenAPI updates.

# Project constraints you must honor

These come from the repository itself, not from this prompt. They override any conflicting suggestion below.

## `AGENTS.md` — hard rules

- **No backward compatibility for internal C++ interfaces.** When refactoring, fully replace the old access pattern. Do not keep legacy shortcuts, `impl_->` member access, or compatibility shims. Delete the old path and update every call site in the same change.
- Prefer interface accessors over reaching into internals.

This directly overrides the older “avoid public ABI breaks where practical” guidance. The correct reading is:

- **Internal C++ headers (`include/celeg/model`, `runtime`, `checkpoint`, `text`, `serve`, `backend`, `detail`): break freely, migrate all call sites, delete the old path.**
- **The C API (`include/celeg/api.h`) is the one surface to keep additive.** It is covered by `tests/api_header_test.c` and `tests/api_smoke_test.c`. Prefer new entry points over changed signatures; if a break is unavoidable, update both tests and document it in `CHANGELOG.md`.

## `docs/ARCHITECTURE_RULES.md` — binding rules R1–R6

Already in force; do not invent a parallel rulebook. Relevant here:

- **R1** — no LFM-specific type in generic runtime directories.
- **R2** — no CUDA type in backend-neutral headers (`model/`, `runtime/`, `checkpoint/`, `text/`, `serve/`).
- **R4** — no optional interface method that a valid implementation may fail with “not supported”. Split capability into a separate interface. *This rule already forbids the current `IChatTemplate` design; the chat-capability work in this task is R4 enforcement, not a new idea.*
- **R5** — no architecture switch in backend operator code.

**Extend this file with new rules** as part of the deliverable, numbered continuing from the existing sequence, covering at minimum:

- no architecture or chat-profile enum branching in `include/celeg/text` or `src/text`;
- no concrete checkpoint-format type reachable through a backend-neutral contract;
- no tool execution, filesystem, subprocess, or outbound network call in `src/app/serve`.

## Repository conventions that will break your tests if ignored

`scripts/check_architecture_boundaries.py` runs in CI as the `architecture_boundary_test` CTest target. It will fail your change for:

- a raw `assert(` anywhere in `include/celeg`, `src`, or `tests` — tests must use `tests/support/assertions.hpp` (`check_architecture_boundaries.py:39`);
- `using namespace` in any test translation unit (`:46-49`);
- a `MANIFEST.sha256` entry pointing at a path that no longer exists (`:22-29`) — **every file you move, rename, or delete requires a `MANIFEST.sha256` update**;
- CUDA tokens or `celeg/detail` includes in the neutral header roots (`:51-70`).

Other conventions:

- Tests are one binary per `tests/*.cpp` file, registered individually in `CMakeLists.txt` (~line 454 onward) and listed in `cmake/sources/tests.cmake`. The test *names* given later in this document are behaviors to cover, not a mandated file layout — group them into a small number of new test binaries (for example `chat_conversation_test`, `tool_call_codec_test`, `chat_capabilities_test`, `model_graph_semantics_test`).
- JSON is Glaze (`include/celeg/serve/protocol/json.hpp`). DTO field names must match the wire format exactly so Glaze reflection works without `glz::meta`. `std::optional` drives null/absent semantics — this matters for `content: null`.
- Build and test: `python scripts/dev.py verify --backend cpu` and `--backend cuda`. Per `AGENTS.md`, all configured CTest tests must pass for a CUDA build.

# Verified repository state

The following is what the code actually does today. Do **not** re-derive it from scratch; do verify the citations still hold.

## Already done — do not redo

These are the parts of the original plan that are already implemented. Building them again would be wasted work.

- **`IArchitecture` + `ArchitectureCatalog` exist** (`include/celeg/model/architecture.hpp:23-42`), with one architecture per directory: `src/models/lfm2/`, `src/models/granite/`, `src/models/gemma4/`. Each produces a `ResolvedModel` and sets its own `chat_profile_id`.
- **Neutral repository interfaces exist**: `IWeightRepository`, `ILocatableTensorRepository`, `IRandomAccessTensorReader`, `HostTensorView`, `TensorLocator`, `TensorDType`, plus the `require_locatable_tensor_repository` / `require_random_access_tensor_reader` capability helpers (`include/celeg/checkpoint/formats/safetensors.hpp:19-92`). The R4 capability-split pattern is already applied here — reuse it, do not reinvent it.
- **A per-layer `ModelGraph` with variant mixers already exists** (`include/celeg/model/graph.hpp:97-133`): `LayerSpec` holds `std::variant<AttentionSpec, ShortConvolutionSpec>` and `std::variant<DenseFeedForwardSpec, MixtureOfExpertsSpec>`, and `KvSharingSpec` (`:36-41`) already models shared KV as an open `group` index, not a fixed pair. The “prefer explicit node specs over a superset DTO” goal is **already satisfied at the type level**. The real problem is who owns it (see below).
- **`ITensorNamingPolicy` exists** (`include/celeg/model/weights/roles.hpp:58-62`) and architectures already attach their policy to the resolved model (`ResolvedModel::tensor_naming`).
- **A boundary checker exists and already enforces** `is_gguf` / `CheckpointSourceFormat` and `architecture_id` / `model_type` bans inside `src/backend` and `include/celeg/backend` (`check_architecture_boundaries.py:97-105`), CUDA-in-neutral-headers, and MANIFEST staleness.
- **Neutral-header and capability tests exist**: `tests/neutral_headers_compile_test.cpp`, `tests/checkpoint_capabilities_test.cpp`, `tests/checkpoint_format_catalog_test.cpp`, `tests/architecture_resolution_test.cpp`, `tests/tensor_resolver_test.cpp`. Extend these rather than creating parallel ones.

## Confirmed defects — the actual work

### D1. `ModelGraph` is derived from `RuntimeTopology`, not the other way around

This is the most important correction to the original plan, which assumed the graph was authoritative.

- `RuntimeTopology` (`include/celeg/model/resolved.hpp:18-95`) is the flattened superset DTO: parallel arrays, global counters, and MoE/attention/conv fields all at once.
- `build_dense_transformer_graph()` reads `model.topology` and *writes* `model.graph` (`src/model/graph_builder.cpp:9-31`). The graph is a downstream shadow.
- Backends consume `RuntimeTopology`, not the graph: `CompiledModel::shape()` (`include/celeg/detail/model/compiled_model.hpp:115`), `PackedSession::shape()` (`include/celeg/backend/cuda/packed_session.hpp:72`), `engine_internal.hpp:60`, `src/backend/cpu/detail/model_internal.hpp:162`, CUDA sampler/paged-KV/MoE offload signatures.
- The **only** backend read of `ModelGraph` is a single scalar: `src/backend/cpu/weights.cpp:160` (`graph.final_logit_softcap`). `src/model/program.cpp:30-46` is the other consumer.

So `ModelGraph` is nearly dead code and `RuntimeTopology` is the de-facto contract. **Invert this.** Architectures must build the validated `ModelGraph` directly; `RuntimeTopology` must either disappear or shrink to an architecture-resolution internal that never crosses into a backend.

### D2. `RuntimeTopology` hardcodes the MoE-suffix assumption

```cpp
// include/celeg/model/resolved.hpp:81-84
bool layer_uses_moe(int layer) const {
    return layer >= 0 && layer < static_cast<int>(mixer_kinds.size()) &&
           num_experts > 0 && layer >= num_dense_layers;
}
```

MoE membership is inferred from a global `num_dense_layers` counter. `LayerSpec::feed_forward_kind()` already answers this per layer, correctly, for any placement.

### D3. Fixed two-group shared-KV vectors and a Gemma error string in generic CPU code

```cpp
// src/backend/cpu/weights.cpp:169
std::vector<int> shared_owner(2, -1);
// src/backend/cpu/weights.cpp:178
std::vector<int> shared_pool(2, -1);
// src/backend/cpu/weights.cpp:186
throw std::runtime_error("Gemma shared KV consumer has no owner pool");
```

The group count must come from the graph. The message must not name an architecture.

**The root cause is upstream, in the resolver, not in the backend** (see D14.3). The Gemma4 resolver only ever emits groups `0` and `1` (`src/models/gemma4/architecture.cpp:54-56`) and counts them with `for (int group = 0; group < 2; ++group)` (`:177`). The backend merely copied that constant. Fixing only `weights.cpp` would leave the system just as unable to express a third group — fix both, and add the three-group synthetic-graph test to prove it.

Same class of leak in CUDA: `src/backend/cuda/model/residency.cu:13`, `:40`, `:66` all throw `"Gemma per-layer input weights are not BF16-resident"`.

Note that the boundary checker does **not** catch these — its backend regex only matches identifiers (`architecture_id`, `model_type`, `is_gguf`), and its architecture-name regex (`:130`) covers only `Lfm2|LFM2|Granite` and only under `runtime/`. Extend both.

### D4. `CheckpointView` and `ResolvedModel` leak the concrete GGUF format

```cpp
// include/celeg/checkpoint/view.hpp:16-21
struct CheckpointView {
    CheckpointMetadata metadata;
    std::shared_ptr<IWeightRepository> repository;
    std::shared_ptr<GgufFile> gguf;      // concrete format in a neutral contract
    std::filesystem::path path;
};

// include/celeg/model/resolved.hpp:110
bool is_gguf = false;                    // format flag on the neutral resolved model
```

Model these as neutral capability interfaces on the repository, following the `require_*` pattern already present in `formats/safetensors.hpp`.

### D5. Neutral tensor contracts live in a Safetensors header that includes a GGUF header

`include/celeg/checkpoint/formats/safetensors.hpp:3` includes `formats/gguf.hpp`, because `HostTensorView::ggml_type` (`:36`) is a GGUF enum. Every consumer of the neutral `IWeightRepository` therefore transitively depends on both concrete formats. `include/celeg/model/weights/roles.hpp:3` includes this Safetensors header purely to obtain `HostTensorView`.

Move the neutral types to `include/celeg/checkpoint/tensor.hpp` and `include/celeg/checkpoint/weight_repository.hpp`. The quantized-block identity currently expressed as `ggml_type` must become a neutral block-encoding descriptor, with the GGUF mapping owned by the GGUF module.

### D6. Architecture-specific naming policies sit in the generic weight-role header

`CelegTensorNamingPolicy`, `GraniteTensorNamingPolicy`, `Gemma4TensorNamingPolicy` are all declared in `include/celeg/model/weights/roles.hpp:66-79` and implemented in `src/model/weights/roles.cpp`. Move each into its owning `src/models/<arch>/` module; the generic layer keeps only `TensorRole`, `TensorRequest`, `ITensorNamingPolicy`, and `TensorResolver`.

### D7. The tokenizer is the chat renderer and branches on a chat-template enum

Not mentioned in the original plan, and it will silently defeat the `ChatProfile` work if left in place.

- `BpeTokenizer` owns `std::unique_ptr<IChatTemplate> chat_template_` and exposes `format_chat()` (`include/celeg/text/tokenizer.hpp:34-35`, `:77`).
- `IChatTemplate::kind()` returns the closed enum `ChatTemplateKind` (`include/celeg/text/chat_template.hpp:35`; enum at `include/celeg/text/chat_profile.hpp:7-11`), and the tokenizer branches on it:

```cpp
// src/text/tokenizer.cpp:486-499
void reject_gemma4_unsupported_input(const IChatTemplate& chat_template, std::string_view text) {
    if (chat_template.kind() != ChatTemplateKind::Gemma4Instruct) return;
    ...
    throw std::invalid_argument("Gemma 4 text-only mode rejects multimodal and tool inputs");
}
```

- `BpeTokenizer` carries a **second** closed architecture enum: `BpeProfile { Generic, ByteLevelLfm2, ByteLevelGranite, RawUtf8Gemma }` plus a `gemma_normalization_` flag (`include/celeg/text/tokenizer.hpp:42-47`, `:74-75`).
- `BpeTokenizer` also takes a concrete `const GgufFile&` constructor (`:29`), leaking a checkpoint format into `include/celeg/text/`.

Required outcome: **`kind()` is deleted.** Anything currently decided by it becomes a declared capability or a policy object supplied by the profile. Tokenization behavior currently selected by `BpeProfile` becomes injected tokenizer configuration produced during architecture resolution, not an enum the tokenizer switches on. Chat rendering moves out of the tokenizer; the tokenizer tokenizes text and exposes special tokens.

### D8. `make_chat_template` is the growing central switch

`src/text/chat_template.cpp:123-146` maps both `ChatTemplateKind` and `chat_profile_id` strings to concrete templates, throwing `"unknown chat profile"`. Six call sites hardcode this bootstrap path (`src/app/serve/main.cpp:78`, `src/app/cpu/main.cpp:177,180`, `src/app/cuda/main.cpp:351,353`, `src/app/benchmark/cpu/concurrent.cpp:38`). Replace with a `ChatProfile` catalog resolved once at the composition root.

### D9. `IChatTemplate::format` rejects roles at runtime — the LSP defect

`src/text/chat_template.cpp` throws `std::invalid_argument` for `ChatRole::Tool` at `:15-17`, `:28-30`, `:108-110`, and for `ChatRole::Developer` at `:105-107`. `ChatRole::Tool` and `Developer` exist in the enum (`chat_template.hpp:12-18`) but no template supports Tool. This is the R4 violation to fix with declared capabilities.

### D10. The chat route builds JSON errors by string concatenation

```cpp
// src/app/serve/routes/chat_completions.cpp:80-83
res->writeStatus("400 Bad Request")
    ->writeHeader("Content-Type", "application/json")
    ->end(std::string("{\"error\":\"") + error.what() + "\"}");
```

Unescaped, and not the OpenAI error shape. Also note the route’s signature takes `const BpeTokenizer&` and a bare `eos_token_id` (`:48-54`) — it must take a resolved chat profile instead.

### D11. `ChatMessage` is role + string, on both sides

`celeg::ChatMessage` (`include/celeg/text/chat_template.hpp:20-23`) and `ChatMessageDto` (`include/celeg/serve/protocol/chat.hpp:14-17`) are both `{role, content}` with non-optional content. `FinishReason` (`include/celeg/serve/types.hpp:20-26`) has no `ToolCalls`. The request DTO has no `tools` / `tool_choice` / `parallel_tool_calls`.

### D12. `base_runtime` compiles backend sources

`cmake/sources/base_runtime.cmake` lists `src/backend/cuda/execution_plan.cpp`, `src/backend/cpu/compiler.cpp`, and `src/backend/cuda/compiler.cpp` in the backend-independent target. The declared dependency direction is not what the build graph does.

### D13. Duplicated spellings in the graph vocabulary

`MixerKind` declares alias enumerators `FullAttention = Attention` and `Convolution = ShortConvolution`, plus `using LayerType = MixerKind` (`include/celeg/model/graph.hpp:14-20`); `RuntimeTopology` then carries both `mixer_kinds` and `layer_types` (`resolved.hpp:37-38`). Collapse to one spelling — this is cheap and prevents the two vectors drifting.

### D14. The architecture resolvers are the weakest layer in the repository

`ArchitectureCatalog` itself is sound — `add`/`freeze`/`find`/`select` with a specificity score and explicit ambiguity detection (`src/model/architecture.cpp:26-44`) is a reasonable design. The problem is entirely in the three `IArchitecture::resolve` implementations, which are copy-paste variants of each other carrying real defects. Treat this as a first-class work item, not cleanup.

**D14.1 — Fake shared ownership, copy-pasted three times.**

```cpp
// src/models/gemma4/architecture.cpp:40-44 (identically at granite:12-15, lfm2:45-48)
std::shared_ptr<const ITensorNamingPolicy> naming_policy() {
    static const Gemma4TensorNamingPolicy policy;
    return std::shared_ptr<const ITensorNamingPolicy>(&policy,
        [](const ITensorNamingPolicy*) {});
}
```

`ResolvedModel::tensor_naming` advertises shared ownership it does not have. Either make the policy genuinely shared, or change the field to a non-owning `const ITensorNamingPolicy*` / reference and say so. Do not keep a `shared_ptr` that owns nothing.

**D14.2 — `add_request` desynchronizes two parallel arrays.**

```cpp
// src/models/gemma4/architecture.cpp:46-52 (duplicated at lfm2:50)
void add_request(ResolvedModel& model, TensorRequest request) {
    if (model.tensor_naming) {
        const auto names = model.tensor_naming->candidates(request);
        if (!names.empty()) model.tensor_bindings.source_names.push_back(names.front());
    }
    model.weight_plan.requests.push_back(std::move(request));   // always
}
```

`weight_plan.requests` grows unconditionally; `tensor_bindings.source_names` grows only when a name exists. Downstream code that assumes index alignment between the two is silently wrong whenever a policy returns no candidate. Either make the binding a field of `TensorRequest` (removing the parallel array entirely — preferred) or make the invariant explicit and enforced.

**D14.3 — Two shared-KV groups hardcoded at the source.**

```cpp
// src/models/gemma4/architecture.cpp:54-56
int group_for_type(std::string_view layer_type) {
    return layer_type == "sliding_attention" ? 0 : 1;
}
// src/models/gemma4/architecture.cpp:177
for (int group = 0; group < 2; ++group) { ... }
```

Groups must be allocated from the layer schedule, not mapped from a two-valued string predicate.

**D14.4 — Copy, mutate the copy, re-copy.**

In `Gemma4Architecture::resolve`: `model.topology = topology` (`:261`), then the layer loop mutates the **local** `topology` (`:305-309`, writing `max_feed_forward_intermediate` and `feed_forward_intermediates`), then `model.topology = topology` again (`:315`). Between those two lines `model.topology` is stale, and any future code inserted there reads wrong values. `build_gemma_weight_plan` (`:316`) then reads `model.topology` *and* `model.graph` inside the same loop (`:206` vs `:209`, `:225`) — D1's two sources of truth colliding inside one function.

**D14.5 — Dead write.**

`topology.shared_kv_group_count` is incremented inside the layer loop (`gemma4:173`), then reset to `0` (`:176`) and recomputed. The first computation is unreachable in effect. Delete it.

**D14.6 — The same predicate validated three times.**

`ArchitectureCatalog::select` calls `probe` (`architecture.cpp:31`); `resolve` calls `probe` again (`gemma4:257`, `lfm2:~272`); `decode_topology` re-checks the same metadata condition a third time (`gemma4:62-65`). Define the contract once: either `resolve` may assume a successful probe, or `probe` is the only validator and `resolve` trusts it.

**D14.7 — Metadata re-parsed instead of reading resolved values.**

`hidden_size_per_layer_input` is read three times (`gemma4:93`, `:94`, `:290`), `num_kv_shared_layers` twice (`:117`, `:301`), `final_logit_softcapping` twice (`:90`, `:289`). Each read re-does a string concatenation and a map lookup, and each is an opportunity for the three sites to disagree.

**D14.8 — A fabricated layer schedule when metadata is absent.**

```cpp
// src/models/gemma4/architecture.cpp:100-107
} else {
    layer_types.resize(...);
    for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
        layer_types[layer] = ((layer + 1) % 6 == 0 || layer == topology.num_hidden_layers - 1)
            ? "full_attention" : "sliding_attention";
    }
}
```

When `layer_types` is missing, the resolver invents an attention schedule. A checkpoint that does not follow this pattern loads successfully and produces wrong output. Require the field, or gate the fallback behind an explicit, logged, opt-in.

**D14.9 — Model identification by magic dimensions and repository-name substring.**

```cpp
// src/models/lfm2/architecture.cpp:197-205
} else if (t.intermediate == 12288 && t.hidden == 2048 && t.num_hidden_layers == 16) {
    if (checkpoint.repository && checkpoint.repository->contains("model.layers.0.feed_forward.w1.weight")) {
        ... t.intermediate = shape.front();
    } else if (contains_ci(m.repository_hint, "1.2b")) {
        t.intermediate = 8192;
    }
}
```

This is the worst construct in the resolver layer: a specific model recognized by a dimension triple, with a fallback that pattern-matches the HuggingFace repository string. Any future checkpoint that happens to share those three dimensions is silently reconfigured. Replace with an explicit, declared checkpoint quirk keyed on something authoritative, and fail loudly when the true value cannot be determined.

**D14.10 — The resolver probes the weight repository with hardcoded tensor names.**

```cpp
// src/models/lfm2/architecture.cpp:181-189
while (checkpoint.repository->contains(
    "model.layers." + std::to_string(t.num_dense_layers) + ".feed_forward.w1.weight")) {
    ++t.num_dense_layers;
}
```

The architecture counts dense layers by trial lookup, spelling tensor names inline and bypassing `ITensorNamingPolicy` — the abstraction that exists for exactly this. If a checkpoint spells the tensor differently, this loop silently returns `0`. Route every name through the policy, and make "derive layer count from the repository" an explicit, named capability rather than an inline `while` loop.

**D14.11 — Useless diagnostics from the tensor resolver.**

```cpp
// src/model/weights/roles.cpp:172-175
message << "required tensor role is missing: " << static_cast<int>(request.role);
```

The user sees `required tensor role is missing: 27` — a raw enum ordinal, no role name, no layer, no expert index, no list of the candidate names that were tried. Same at `roles.cpp:60-61`. Add a `to_string(TensorRole)` and include layer/expert plus the attempted candidates.

**D14.12 — Three parallel switches over one enum, with R4-violating defaults.**

`CelegTensorNamingPolicy`, `GraniteTensorNamingPolicy`, and `Gemma4TensorNamingPolicy` (`src/model/weights/roles.cpp:35-158`) are three switches over the same `TensorRole`, each ending in `default: throw std::invalid_argument("tensor role has no Granite spelling")`. Adding a role means editing three switches, and the throwing default is precisely the pattern R4 forbids. A policy that legitimately has no spelling for a role should return "no candidates" and let the caller decide, or the role set should be scoped per architecture. Also note `role_suffix` uses an empty string as a sentinel (`:29`, checked at `:59`) — use `std::optional<std::string_view>`.

**Required outcome for D14.** After this work, a resolver should read as: validate metadata once → build a validated `ModelGraph` → build a `WeightPlan` through the naming policy → return. No repository sniffing by hardcoded name, no magic-number model detection, no invented schedules, no mutate-after-assign, and no shared ownership that isn't. Factor the genuinely shared helpers (`add_request`, the policy accessor, the metadata readers) into one place instead of three copies — but only the parts that are actually identical.

### D15. The tokenizer has a sound core and a defective periphery

Unlike the resolvers, `BpeTokenizer` is not uniformly weak. Judge the two halves separately and **do not rewrite the good half.**

**Keep as-is.** `bpe_symbols` (`src/text/tokenizer.cpp:502-573`) is a correct doubly-linked-list + priority-queue BPE merge with lazy invalidation of stale queue entries (`:546-553`) — the proper O(n log n) formulation, not the naive repeated-minimum scan. The GPT-2 byte↔unicode table (`:177-196`) is correct, and special tokens are sorted by descending length for longest-match priority (`:235-238`, `:314-316`). Preserve this behavior exactly; it is covered by `tests/tokenizer_test.cpp` and the `tests/fixtures/tokenizer_unicode.txt` fixture.

**D15.1 — `encode()` is quadratic in special tokens.**

```cpp
// src/text/tokenizer.cpp:647-656
while (cursor < text.size()) {
    for (const auto& special : specials_) {
        const size_t pos = text.find(special.text, cursor);   // full scan, per special
        if (pos < best_pos) { best_pos = pos; best = &special; }
    }
    ...
```

For `S` special tokens and `K` occurrences in the text, this is `O(K × S × n)`. A multi-turn chat prompt is exactly the worst case: dozens of markers, a vocabulary with ~100 specials, and a long prompt — all on the prefill path. Replace with a single forward scan (Aho–Corasick, or a first-byte dispatch table plus one ordered pass).

**This defect is invisible to the existing benchmark.** `tests/tokenizer_benchmark.cpp:12-15` builds its corpus from plain prose, code, and Unicode text containing **no special tokens at all**. Before optimizing, extend the benchmark with a realistic multi-turn chat prompt full of the profile's markers, so the improvement is measurable. Tool-call prompts will make this worse, not better — tool definitions and results add markers to every turn.

**D15.2 — A heap allocation per merge-rank lookup, in the hot loop.**

```cpp
// src/text/tokenizer.cpp:144-151
std::string pair_key(const std::string& a, const std::string& b) {
    std::string out;
    out.reserve(a.size() + b.size() + 1);
    out += a; out.push_back('\0'); out += b;
    return out;
}
```

Called from `add_candidate` (`:536`) and from the merge loop (`:552`) — so every candidate push and every pop allocates and hashes a freshly concatenated string. The algorithm is asymptotically right but each operation carries a `malloc`. Intern symbols to integer ids and key `merge_rank_` on a packed pair, or use a transparent heterogeneous hash over a `string_view` pair. Verify the win with the extended benchmark from D15.1.

**D15.3 — Model family detected by substring-sniffing a config field.**

```cpp
// src/text/tokenizer.cpp:325-335
if (const Json* regex = find_regex_node(root["pre_tokenizer"])) {
    if (regex->as_string().find("\\p{N}{1,3}") != std::string::npos) {
        profile_ = BpeProfile::ByteLevelLfm2;
    } else if (profile_ != BpeProfile::RawUtf8Gemma) {
        profile_ = BpeProfile::ByteLevelGranite;      // silent catch-all
    }
}
```

Same class of defect as D14.9: identity inferred from a substring rather than from an authoritative field, and an `else` that silently applies Granite's pre-tokenization rules to **any** unrecognized tokenizer. `gemma_normalization_ = byte_fallback_ && vocab_.contains("▁")` (`:323`) and the GGUF path's `tokenizer.ggml.pre` string matching (`:252-258`) are the same pattern. Whatever replaces `BpeProfile` (see D7) must be selected explicitly during architecture resolution and must fail loudly on an unrecognized tokenizer rather than defaulting.

**D15.4 — Hand-rolled pre-tokenizer with per-model branches.**

`pretokenize` (`:339-455`) is a hand-written approximation of the HF regex pre-tokenizer with `profile_ == BpeProfile::ByteLevelLfm2` conditionals at `:348`, `:375-376`, `:378`, `:402`, `:417`, `:433`, plus another at `:626` in `encode_ordinary`. `category()` (`:122-129`) approximates Unicode classes without ICU and treats every unassigned code point as a letter — honestly documented at `:126-127`, but it does diverge from `\p{L}` for combining marks and emoji modifiers.

Do not attempt a full regex engine as part of this task. Do: make the per-profile rules a data-driven pre-tokenization policy supplied by the profile, so the branches leave the function body, and add round-trip tests against the existing Unicode fixture for each profile before touching it.

**D15.5 — `<|end_of_text|>` sets both BOS and EOS.**

```cpp
// src/text/tokenizer.cpp:304-311
if (token.text == "<|startoftext|>" || ... || token.text == "<|end_of_text|>") bos_id_ = token.id;
if (token.text == "<|im_end|>"      || ... || token.text == "<|end_of_text|>") eos_id_ = token.id;
```

`<|end_of_text|>` appears in both lists. A checkpoint whose `added_tokens` contains it but not `<|startoftext|>` gets `bos_id_` set to the end-of-text id. Verify against the real checkpoints before changing — if no shipped checkpoint hits it, it is latent; if one does, it is an active bug and its correction must be documented as an intentional behavior change. In either case, prefer the authoritative `bos_token`/`eos_token` fields over a hardcoded name list.

**D15.6 — Silent data loss on malformed input.**

- `next_cp` (`:32-48`) never validates continuation bytes (`0x80-0xBF`), overlong encodings, or surrogate ranges; it masks blindly and only returns `U+FFFD` when the sequence length does not fit the buffer. Malformed input yields plausible-looking wrong code points.
- `byte_decode` (`:463-473`) silently drops any code point absent from `byte_decoder_`.
- `decode` (`:676`, `:695`) silently skips out-of-range token ids.
- `std::stoul` on a byte-fallback token (`:680`) is unguarded; a malformed `<0xZZ>` throws from deep inside `decode`.

**This block is a direct prerequisite for the streaming tool-call work.** The `ChatGenerationInterpreter` requirement to "preserve UTF-8 boundaries" is built on `next_cp`; harden it and add the split-position tests there, not only at the codec layer.

**D15.7 — Diagnostics without context.** `throw std::runtime_error("BPE produced token absent from vocabulary")` (`:635`) names neither the token nor the source piece; the Gemma path at `:616` at least includes the token. Same fix as D14.11.

**D15.8 — Per-encode marker scan.** `reject_gemma4_unsupported_input` runs on every `encode()` call (`:643`) and, for the Gemma profile, scans the text for nine forbidden markers each time. Once D7 removes the `kind()` comparison, make this a declared input policy evaluated where the input enters the system, not on every tokenization.

### D16. Algorithmic complexity defects in hot paths

**Scope note.** This section is performance work, orthogonal to tool calling. It must not block Phases 3–6. Each item is independently landable and independently reviewable — treat them as separate changes, not one commit. If time is limited, D16.1 and D16.3 are the two that scale with load; the rest can wait.

**Measure first — this is a repository rule, not a suggestion.** `AGENTS.md` documents a case where a plausible bandwidth argument concluded the decode GEMV path was at ~17% of peak and that native-quantized weights would give ~3x; measurement showed GEMV was already at 76% of peak and only ~22% of a decode step, while 66% was a single-threaded top-k loop in the sampler. *The arithmetic was right and the attribution was wrong.* Every item below is a hypothesis with a citation, not a measured result. Profile before changing anything, and report before/after numbers.

Available harnesses: `scripts/profile_decode.py` (per-phase breakdown, needs `--no-cuda-graph`), `scripts/cpu_benchmark.sh`, `scripts/cpu_concurrency_benchmark.sh`, `scripts/concurrency_benchmark.sh`, the benchmark binaries under `src/app/benchmark/`, and the reproducible manifests in `benchmarks/manifests/`. Capture a baseline manifest run before each change and after.

**D16.1 — The CPU sampler allocates a vocabulary-sized buffer per token and sorts with an expensive comparator.**

```cpp
// src/backend/cpu/sampler.cpp:54-62
std::vector<std::int32_t> indices(static_cast<size_t>(shape.vocab_size));
std::iota(indices.begin(), indices.end(), 0);
auto adjusted = [&](std::int32_t token) { return penalized(token) / temperature; };
std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
    [&](std::int32_t a, std::int32_t b) {
        const float av = adjusted(a);
        const float bv = adjusted(b);
        return av == bv ? a < b : av > bv;
    });
```

Two defects in one place. The `indices` vector is a fresh 256 KB allocation and fill for a 65536-token vocabulary, on every sampled token. And the comparator evaluates `adjusted()` twice per comparison — each one a load, a repetition-penalty branch, and a division — so `partial_sort`'s roughly `V·log k` comparisons become on the order of `2·V·log k` divisions to select `k` elements.

Replace with a single bounded-heap pass: compute the adjusted value once per token (`V` evaluations, not `2·V·log k`), keep a `k`-sized min-heap, and hold the scratch buffer as persistent sampler state rather than a local.

This is the same defect shape that measurement already found and fixed on the CUDA side (see the comment at `src/backend/cuda/kernels/sampling.cu:188`, which records the 65536-vocabulary measurement). The greedy path (`sampler.cpp:41-51`) is already a correct single `O(V)` pass — leave it alone.

Sampling must remain bit-identical for a fixed seed. `tests/cpu_sampler_test.cpp` and `tests/reference_test.cpp` are the guards; the tie-break rule (`av == bv ? a < b`) is part of the observable contract and must be preserved exactly.

**D16.2 — `PinnedExpertCache` looks up slots by linear scan.**

```cpp
// src/runtime/cache/pinned_expert_cache.cpp:121-128
int PinnedExpertCache::find_slot(int layer, int expert) {
    for (int i = 0; i < capacity_; ++i) {
        if (slots_[i].layer == layer && slots_[i].expert == expert) return i;
    }
    return -1;
}
```

`choose_victim_slot` (`:130-148`) adds two more full passes on a miss. `capacity_ = budget_bytes / bytes_per_expert` (`:86`), so a large pinned budget yields hundreds to thousands of slots, and `acquire` is called per `(layer, expert)` on the MoE offload decode path (`src/backend/cuda/model/residency.cu:368`, `:411`, `:636`).

The CUDA sibling already solves this correctly with a direct index — `expert_slot_[expert]` (`src/backend/cuda/moe/expert_layer_cache.cu:324`). Bring the host cache to the same standard: a hash map keyed on `(layer, expert)` plus an intrusive LRU list for `O(1)` victim selection. `tests/pinned_expert_cache_test.cpp` and `tests/expert_offload_test.cpp` cover the semantics; eviction order must not change observably.

**D16.3 — The CPU scheduler rebuilds and re-sorts its plan every decode step.**

```cpp
// src/backend/cpu/concurrent.cpp:226-244 (duplicated at :246-266 and :183-197)
for (const auto& [id, request] : requests) { ... result.push_back(request); }
...
const auto order = planner.order_priority(views);
for (const size_t index : order) ordered.push_back(result[index]);
result = std::move(ordered);
if (result.size() > limit) result.resize(limit);      // sorts, then discards
```

Per scheduling pass: three vector allocations, an `O(N log N)` sort, and a truncation that throws away most of what was just sorted. Each `push_back` of a `shared_ptr` is an atomic refcount increment, and the copy-then-move sequence costs roughly three atomic operations per request per pass — all under the scheduler mutex. The same block is copy-pasted three times.

`RequestRegistry` already maintains a priority-ordered `std::set` keyed on `(-priority, id)` with `best_queued()` (`src/runtime/concurrency/request_registry.cpp:21`, `:47-50`) — exactly this problem, already solved, and unused by the CPU scheduler because it keeps its own request map. Fixing this and the D-series responsibility split (see "Responsibilities worth splitting") are the same work: move the CPU scheduler onto the existing collaborator instead of reimplementing it worse.

Use `scripts/cpu_concurrency_benchmark.sh` with a high concurrent-request count; the effect is invisible at low concurrency.

**D16.4 — `PrefixRadixIndex` is a token-per-node trie, not a radix tree.**

`insert` (`src/runtime/cache/prefix_radix.cpp:14-22`) allocates one heap `Node`, each owning an `unordered_map`, for every token of every cached prefix. Lookup complexity is fine (`O(L)`); the cost is memory and allocation churn — `N` cached prompts of `L` tokens produce `N×L` nodes. The name promises path compression that the implementation does not do.

Either implement actual path compression (store a token span per node, split on divergence) or rename the type to match what it is. Do not do both halfway. `tests/prefix_radix_test.cpp` and `tests/prefix_cache_test.cpp` define the behavior to preserve.

**D16.5 — `CpuKvPagePool::allocate` scans all free pages for a NUMA match.**

```cpp
// src/backend/cpu/memory/paged_kv.cpp:261-264
selected = std::find_if(free_pages_.begin(), free_pages_.end(),
    [&](CpuKvPageId candidate) {
        return pages_[candidate] && pages_[candidate]->numa_node == requested_node;
    });
```

One linear scan over the free list per page allocation. Replace with per-node free lists. `tests/cpu_paged_kv_test.cpp` and `tests/cpu_numa_test.cpp` cover this.

**D16.6 — Minor: string allocation per GGUF metadata lookup.** `GgufFile::has` and `GgufFile::value` (`src/checkpoint/formats/gguf.cpp:355-365`) construct a `std::string` from the `string_view` key because `kv_` has no transparent hash. Load-time only, so low priority — but it compounds with D14.7's repeated metadata reads, and adding heterogeneous lookup is nearly free.

**Not defects — do not "fix" these.** `RequestRegistry` (hash map plus ordered set) is correct. The CPU forward pass reuses a persistent workspace, so the `.resize()` calls at `src/backend/cpu/model_forward.cpp:43-45` and `:146-150` are no-ops on an already-sized vector, not per-token allocations. GGUF and JSON metadata containers are genuine hash maps. Attention is inherently quadratic in sequence length; that is the algorithm, not a defect.

## Responsibilities worth splitting

Confirmed sizes: `src/backend/cpu/weights.cpp` (663 lines) performs checkpoint bootstrap, compilation, weight loading, pack-path/cache handling, and KV-pool planning in one initializer. `src/backend/cpu/concurrent.cpp` (675 lines) combines admission, registry, placement, prefix caching, worker lifecycle, execution, metrics, and cancellation.

Refactor **only** where it improves ownership and testability. Each extraction must own meaningful state, a replaceable policy, or a test. Do not split classes mechanically into one-method wrappers. Note that `src/runtime/concurrency/` already contains `request_registry.cpp`, `policy.cpp`, `batch_planner.cpp`, `worker.cpp`, and `metrics.cpp` — prefer moving CPU logic onto those existing collaborators over inventing new names.

# Architectural principles to enforce

## Dependency direction

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

Model architecture plugins may produce model-neutral graph, weight, tokenizer, and chat-profile descriptions. Checkpoint-format plugins may expose metadata and weight repositories through neutral interfaces.

CPU and CUDA backends must depend on neutral compiled-model contracts, not on LFM, Granite, Gemma, GGUF, Safetensors, OpenAI, or chat-template types.

## Switches and variants

Do not treat every switch or `std::variant` as an anti-pattern. Closed variants are correct when they represent the intentionally closed set of operators a compiled backend understands — `std::variant<AttentionSpec, ShortConvolutionSpec>` in `LayerSpec` is a good use and must stay.

The problems are architecture-oriented branching (`if (architecture_id == "gemma4")`, or its disguised form `if (chat_template.kind() != ChatTemplateKind::Gemma4Instruct)`) and checkpoint-format branching (`if (checkpoint.gguf != nullptr)`, `if (model.is_gguf)`) used to select runtime behavior.

Use semantic graph capabilities, neutral interfaces, or compile-time operator variants instead.

## Invalid states

Prefer representations that make invalid states hard to construct. Do not replace `RuntimeTopology` with another equally large collection of optional fields — the per-layer specs in `graph.hpp` already show the target shape.

A layer contains only the mixer and feed-forward information applicable to that layer. Architecture-specific metadata may exist while resolving a model, but must not leak into the compiled backend contract.

## Composition roots and catalogs

`load_model_bootstrap()` (`include/celeg/detail/checkpoint/bootstrap.hpp:16`) currently takes only a path and obtains catalogs internally. Target:

```cpp
struct ModelBootstrapDependencies {
    const ICheckpointFormatCatalog& checkpoint_formats;
    const IArchitectureCatalog& architectures;
};
```

Keep a default composition root for CLI/server/C API usage, but make the core model-loading workflow independently testable with custom catalogs.

# Required SOLID refactoring

## 1. Extract neutral checkpoint and tensor contracts (D4, D5)

Create:

```text
include/celeg/checkpoint/tensor.hpp
include/celeg/checkpoint/weight_repository.hpp
```

Move out of `formats/safetensors.hpp`: `TensorDType`, `TensorLocator`, `HostTensorView`, `IWeightRepository`, `ILocatableTensorRepository`, `IRandomAccessTensorReader`, and the two `require_*` helpers. Leave `SafeTensorFile` behind.

Replace `HostTensorView::ggml_type` with a neutral block-encoding descriptor; the GGUF module owns the mapping between that descriptor and `GgmlType`.

Remove `CheckpointView::gguf` and `ResolvedModel::is_gguf`. Where behavior genuinely differs, express it as a capability query on the repository, not a format flag. Update `include/celeg/model/weights/roles.hpp:3` to include the new neutral header.

Update `MANIFEST.sha256` for every moved file.

## 2. Move architecture-specific naming policies (D6)

Relocate the three naming policies to their architecture modules. The generic weight-planning layer depends on `ITensorNamingPolicy` only. Adding an architecture must not require touching a central file listing every tensor name.

## 3. Remove backend architecture leaks (D2, D3)

- Replace `shared_owner(2, -1)` / `shared_pool(2, -1)` with containers sized from graph-derived group ownership.
- Remove the Gemma strings from `src/backend/cpu/weights.cpp:186` and `src/backend/cuda/model/residency.cu:13,40,66`; state the missing semantic condition instead.
- Delete `RuntimeTopology::layer_uses_moe`’s suffix arithmetic in favor of per-layer `feed_forward_kind()`.
- Add tests that build synthetic graphs — three shared-KV groups, and MoE layers interleaved with dense layers rather than forming a suffix — so both behaviors are validated with no named architecture involved.

## 4. Clarify model-resolution phases (D1)

Target pipeline:

```text
checkpoint metadata
    ↓
architecture resolution
    ↓
validated ModelGraph + WeightPlan + model metadata      ← authoritative
    ↓
backend compilation
    ↓
backend-specific CompiledModel
```

Architectures build `ModelGraph` directly. `graph_builder.cpp`’s topology→graph direction is reversed or removed. Anything backends still need from `RuntimeTopology` (workspace sizing maxima, vocab, context) is either derived from the graph or carried on an explicit, small, backend-facing value type — not a second mutable source of truth.

Do not make backends re-infer graph semantics from flattened global counters.

## 5. Split oversized responsibilities

CPU model initialization: separate runtime environment, compilation, weight loading, KV-cache planning, and packed-weight caching into cohesive components with clear names. Scheduling: move logic onto the existing `src/runtime/concurrency/` collaborators where it fits.

## 6. Improve architecture-boundary enforcement (D3)

Extend `scripts/check_architecture_boundaries.py` to detect:

- architecture-name **string literals** (`Gemma`, `Granite`, `LFM`, `Lfm2`) in `src/backend`, `include/celeg/backend`, `src/text`, and `include/celeg/text` — with an explicit allowlist for benchmark/measurement comments, which are legitimate and numerous under `src/backend/cuda/moe/`;
- concrete checkpoint-format **type names** (`GgufFile`, `SafeTensorFile`, `GgmlType`) reachable from neutral header roots, not just the `is_gguf` identifier;
- includes of `celeg/checkpoint/formats/*` from `include/celeg/model`, `include/celeg/text`, `include/celeg/runtime`;
- includes of `celeg/models/*` or `src/models` headers from `src/backend`;
- `ChatTemplateKind` used anywhere outside its own definition (once D7 lands);
- reverse dependencies in the CMake manifests — specifically, `src/backend/**` entries inside `CELEG_BASE_RUNTIME_SOURCES` (D12).

Prefer structural include/path checks over string heuristics wherever both are possible.

# Tool calling architecture

Implement OpenAI-compatible tool calling as a protocol and chat-orchestration feature. Do not implement tool execution inside the core `/v1/chat/completions` route.

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

Create neutral domain types, separate from OpenAI wire DTOs, for example under `include/celeg/text/`:

```text
tool_call.hpp
conversation.hpp
chat_profile.hpp        (replaces the current ChatTemplateKind enum file)
tool_call_codec.hpp
```

```cpp
struct JsonSchema {
    std::string serialized;      // validated as JSON, stored verbatim
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
    std::string arguments;       // JSON string, never a parsed object
};

enum class ToolChoiceMode { None, Auto, Required, Specific };

struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
    std::string function_name;
};
```

Do not build a C++ object model for JSON Schema keywords. The runtime only preserves and renders the schema.

To satisfy the “schemas are valid JSON” validation requirement without that object model: parse the schema with Glaze into a generic JSON value purely to reject malformed input, then **store and re-emit the original serialized text**. Round-tripping through a typed model would silently drop keywords Celeg does not know about.

# Structured chat messages

Replace `celeg::ChatMessage` (D11). Both `IChatTemplate::format` and `BpeTokenizer::format_chat` take `std::span<const ChatMessage>`, so this change reaches `src/text/chat_template.cpp`, `src/text/tokenizer.cpp`, `src/serve/protocol/mapping.cpp`, `tests/chat_template_test.cpp`, and `tests/chat_protocol_mapping_test.cpp`. Per `AGENTS.md`, migrate them all; do not add an overload for the old shape.

```cpp
struct ChatMessage {
    ChatRole role;
    std::optional<std::string> content;
    std::vector<ToolCall> tool_calls;
    std::optional<std::string> tool_call_id;
    std::optional<std::string> name;
};
```

Invariants:

- a `tool` message requires `tool_call_id`;
- a `tool` message requires a result content value;
- only assistant messages may contain `tool_calls`;
- an assistant message may contain content, tool calls, or both;
- a message with neither content nor tool calls is invalid;
- tool-call IDs in results must correspond to prior assistant calls when conversation validation is enabled;
- duplicate tool-call IDs in a single assistant turn are invalid.

Validation errors must be explicit and mappable to `400 Bad Request`.

# Chat capabilities and LSP (D7, D9)

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
    std::shared_ptr<const IToolCallCodec> tool_call_codec;   // null when unsupported
    ChatCapabilities capabilities;
};
```

Ownership may be references, unique pointers, shared pointers, or immutable catalog entries per project convention. The requirement is that callers inspect capabilities without invoking unsupported methods.

`ChatCapabilities` must also absorb what `IChatTemplate::kind()` is used for today. When D7 is complete, `kind()` and `ChatTemplateKind` are both deleted, and `reject_gemma4_unsupported_input` becomes either a profile-supplied input policy or a declared capability — not a comparison against an enum value.

Profiles without tool support must return a clear protocol error when the request includes tools or tool messages. Do not silently inject a generic JSON prompt and claim native tool-call compatibility.

# Tool-call codec

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

enum class ToolParseStatus { NotToolCall, Complete, Incomplete, Invalid };

struct ToolParseResult {
    ToolParseStatus status = ToolParseStatus::NotToolCall;
    std::string assistant_text;
    std::vector<ToolCall> calls;
    std::string error;
    std::size_t consumed_bytes = 0;
};
```

`Incomplete` is mandatory for streaming. Never assume a generated chunk contains a complete JSON object.

# Prompt rendering

Separate semantic chat rendering from tokenization. Today `to_generate_request()` (`include/celeg/serve/protocol/mapping.hpp:23-25`) takes the tokenizer and calls `format_chat` through it, so DTO validation, domain conversion, rendering, tokenization, and generation-option construction all collapse into one function.

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

The tokenizer tokenizes text and exposes chat-related special tokens. It does not own chat rendering, chat profiles, or OpenAI tool semantics.

# OpenAI DTO support

Extend the request DTOs in `include/celeg/serve/protocol/chat.hpp` with `tools`, `tool_choice`, `parallel_tool_calls`, assistant `tool_calls`, `tool` messages with `tool_call_id`, and optional/null assistant `content`.

`ChatMessageDto::content` becomes `std::optional<std::string>`; verify Glaze’s null-vs-absent handling for it explicitly in a test, since the response shape requires emitting `"content": null` rather than omitting the field.

`tool_choice` is polymorphic on the wire — a string (`"none"`, `"auto"`, `"required"`) or an object:

```json
{ "type": "function", "function": { "name": "get_weather" } }
```

Glaze reflection will not deduce this from a plain struct. Use a variant with an explicit `glz::meta`, or read `tool_choice` as a raw JSON value and map it in the protocol layer. Cover both wire forms with tests.

Function tools support `type = "function"`, `name`, optional `description`, `parameters` JSON Schema, and optional `strict`. Reject unsupported or malformed combinations with useful errors.

**Preserve `function.arguments` as a JSON string in responses. Never emit it as a nested JSON object.**

# OpenAI response support

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
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```

Support assistant text and tool calls coexisting when the model format allows it.

Add `FinishReason::ToolCalls` to `include/celeg/serve/types.hpp:20-26` and map it in `finish_reason_to_string`. Note that `FinishReason` crosses into `GenerateEvent`, which backends populate — the backend must never *produce* `ToolCalls`; the interpreter layer upgrades `Stop` to `ToolCalls` above the backend. If keeping the enum backend-visible makes that ambiguous, introduce a separate application-facing finish reason and map at the boundary.

Do not make CPU or CUDA request lifecycle objects parse tool calls.

# Generation interpretation

CPU and CUDA continue producing tokens. Add a layer above backend generation:

```cpp
class ChatGenerationInterpreter {
public:
    ChatGenerationDelta consume(std::span<const std::int32_t> tokens, bool finished);
};
```

It must incrementally decode tokens safely, preserve UTF-8 boundaries, distinguish assistant text from tool-call syntax via the selected `IToolCallCodec`, expose completed calls and incomplete parser state, produce protocol-neutral deltas, and decide the final semantic finish reason.

Note that `BpeTokenizer::decode()` takes a full `std::vector<int32_t>` and has no incremental/streaming mode, so the interpreter must either buffer and re-decode, or a streaming decode entry point must be added. Pick one and state which.

A backend event ending with EOS becomes:

- `stop`, for normal assistant text;
- `tool_calls`, for successfully parsed tool calls;
- `error`, if the model emitted a clearly intended but malformed tool-call structure and the selected policy treats it as fatal.

**Document the malformed-output policy explicitly**, including what happens to any assistant text that preceded the malformed call.

# Initial stopping behavior

For the first implementation, let the model generate through its normal EOS or configured stop token, then parse the complete output. Do not integrate JSON grammar constraints or token-level stopping into the samplers initially.

After non-streaming tool calling is stable, optionally add a stop detector above the backend execution loop:

```cpp
class IGenerationStopDetector {
public:
    virtual StopDecision consume(std::span<const std::int32_t> tokens) = 0;
};
```

It must not contain architecture-name checks.

# Streaming tool calls

Implement non-streaming first. Then add OpenAI-compatible deltas.

First tool-call chunk:

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
            "function": { "name": "get_weather", "arguments": "{\"city\":\"Rio" }
          }
        ]
      },
      "finish_reason": null
    }
  ]
}
```

Final chunk:

```json
{ "choices": [ { "index": 0, "delta": {}, "finish_reason": "tool_calls" } ] }
```

Ensure stable tool-call indexes and IDs, IDs emitted once, predictable function-name emission, argument fragments that concatenate exactly to the final string, no incorrectly split UTF-8, incomplete JSON not treated as an error before generation finishes, and preserved ordering for parallel calls.

The current streaming path emits one chunk per event and a separate final chunk (`chat_completions.cpp:109-134`); the tool-call delta state (which IDs and names have already been sent) must live in the interpreter, not in the route lambda captures.

# Tool execution is out of scope for the inference server

Do not add filesystem access, subprocess execution, HTTP callbacks, plugin loading, credential handling, approval flows, or function execution to `chat_completions.cpp`.

```text
celeg runtime
    model loading, compilation, scheduling, token generation

celeg OpenAI server
    HTTP protocol, request validation, prompt rendering,
    tool-call encoding and decoding

optional external celeg-agent component
    tool registry, approvals, execution, retries, agent loop
```

An optional future agent component may define an `IToolExecutor`. **Do not implement it in this task.**

# Model support strategy

Tool calling is a capability of a model/chat profile, not of CPU or CUDA. Implement the infrastructure independently of named architectures, then implement exactly one real codec for one profile whose chat format genuinely supports tool calling.

## Required verification before writing a codec

Do not infer a format from intuition or from this document. For the candidate profile:

1. Locate the checkpoint in the HF cache (see `AGENTS.md`; `celeg-run --repo <HF_REPO_ID>` resolves it).
2. For Safetensors checkpoints, read the `chat_template` field of `tokenizer_config.json` in the snapshot directory. For GGUF, read the `tokenizer.chat_template` metadata key — `scripts/gguf_census.py` and the existing GGUF metadata reader can dump it.
3. Extract, verbatim: the tool-definition block markers, the assistant tool-call markers, the tool-result markers, and whether arguments are JSON or another syntax.
4. Confirm each marker token actually exists in the checkpoint’s vocabulary (`tests/tokenizer_probe.cpp` and `tests/tokenizer_benchmark.cpp` already exercise the tokenizer directly). A codec that emits a special token the tokenizer splits into fragments is broken.
5. Record the extracted template and the marker list in the codec’s header comment and in the docs, so the next reader can re-verify without the checkpoint.

## Current profiles

Three profiles exist: `lfm2-instruct`, `granite-instruct`, `gemma4-instruct` (set in `src/models/*/architecture.cpp`).

- **Gemma4 must report `assistant_tool_calls = false`.** The current implementation actively rejects tool markers as unsupported input (`src/text/tokenizer.cpp:486-499`) — that is the honest capability, and after D7 it becomes a declared `false` rather than a runtime string scan.
- **LFM2 and Granite are the plausible candidates.** Both model families document native tool-calling chat templates in their upstream tokenizer configs. Verify per the procedure above and pick whichever checkpoint is actually present in the local HF cache. State which you chose and paste the evidence.

For profiles without a verified format: report `assistant_tool_calls = false`, reject tool requests clearly, and leave normal chat unchanged.

A generic prompted-JSON fallback may be added only as an explicitly experimental profile (`prompted-json-tools-experimental`) and must not be advertised as equivalent to native tool calling.

# Catalog design (D8)

```cpp
class IChatProfileCatalog {
public:
    virtual const ChatProfile& find(std::string_view profile_id) const = 0;
};
```

Architecture resolution already produces `chat_profile_id`; the server composition root obtains the profile from the catalog. Mirror the existing `ArchitectureCatalog` add/freeze/find shape (`include/celeg/model/architecture.hpp:31-42`) for consistency.

The route must depend on a resolved chat profile — never on `architecture_id` branching, and never on `ChatTemplateKind`.

# Compatibility requirements

Preserve existing behavior for requests that do not use tools. Plain-text chat completions must produce the same prompt and response semantics unless an existing bug is corrected and documented.

Preserve: CPU generation, CUDA generation, streaming text, non-streaming text, cancellation, usage accounting, the tokenize endpoint, C API behavior, checkpoint loading, and architecture-resolution tests.

Prove prompt-rendering compatibility mechanically: capture the exact rendered prompt string for a fixed message list per profile **before** the refactor, commit it as a golden fixture, and assert byte equality after. `tests/chat_template_test.cpp` and `tests/chat_protocol_mapping_test.cpp` are the natural homes.

Internal C++ interface breaks are expected and required (see the `AGENTS.md` rule). Keep the C API additive; document any C API change in `CHANGELOG.md`.

# Validation requirements

Validate at least:

- request contains messages;
- role names are recognized;
- content requirements by role;
- tools use supported types;
- function names are non-empty;
- function schemas are valid JSON values;
- the function selected by `tool_choice` exists in `tools`;
- `parallel_tool_calls=true` rejected for profiles without support;
- `required` rejected when tools are empty;
- tool messages reference tool-call IDs;
- assistant tool calls have unique IDs;
- function arguments remain strings;
- profile capabilities permit every requested feature.

Return structured OpenAI-style error objects (D10), never string-concatenated JSON:

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

Serialize through `protocol::to_json` so messages are escaped. Apply this to every route that currently hand-writes an error body, not only the chat route.

# Tests

Behaviors to cover (group into a few new binaries; register each in `CMakeLists.txt` and `cmake/sources/tests.cmake`):

```text
plain_chat_request_remains_compatible
rendered_prompt_matches_golden_fixture_per_profile
tools_request_without_supported_profile_returns_400
developer_role_without_capability_returns_400
tool_role_without_capability_returns_400
assistant_message_accepts_tool_calls_without_content
assistant_message_accepts_text_and_tool_calls
tool_message_requires_tool_call_id
tool_message_requires_content
tool_choice_required_requires_tools
tool_choice_accepts_string_and_object_forms
specific_tool_choice_must_reference_declared_tool
duplicate_tool_call_ids_are_rejected
assistant_content_null_is_emitted_not_omitted
single_tool_call_round_trip
tool_result_round_trip
multiple_tool_calls_preserve_order
plain_text_is_not_misclassified_as_tool_call
malformed_tool_arguments_follow_documented_policy
finish_reason_is_tool_calls
function_arguments_are_returned_as_json_string
usage_counts_generated_tool_tokens
error_responses_escape_json_special_characters
streamed_arguments_reassemble_exactly
streamed_tool_ids_are_stable
streaming_final_reason_is_tool_calls
unsupported_parallel_calls_return_400
custom_architecture_catalog_can_be_injected
custom_checkpoint_format_catalog_can_be_injected
backend_sources_do_not_include_model_specific_headers
neutral_sources_do_not_include_format_specific_headers
base_runtime_manifest_excludes_backend_sources
shared_kv_groups_are_graph_driven
three_shared_kv_groups_are_supported
moe_layers_need_not_be_a_suffix
interleaved_moe_and_dense_layers_compile
resolver_emits_more_than_two_kv_groups
weight_plan_requests_and_bindings_stay_aligned
missing_layer_types_metadata_is_rejected_not_fabricated
resolver_does_not_identify_models_by_dimension_triple
resolver_resolves_tensor_names_only_through_the_naming_policy
missing_tensor_error_names_role_layer_and_candidates
resolve_does_not_reprobe_metadata
resolved_topology_is_not_mutated_after_assignment
chat_template_kind_enum_is_gone
tokenizer_does_not_depend_on_chat_profile
unrecognized_tokenizer_is_rejected_not_defaulted
bos_and_eos_ids_come_from_authoritative_fields
malformed_utf8_decodes_to_replacement_character
overlong_and_surrogate_sequences_are_rejected
byte_decode_does_not_silently_drop_bytes
special_token_scan_is_linear_in_prompt_length
pretokenize_round_trips_unicode_fixture_per_profile
tokenizer_output_is_unchanged_for_all_shipped_profiles
sampler_output_is_bit_identical_for_a_fixed_seed
sampler_allocates_no_per_token_scratch
expert_cache_lookup_is_constant_time
expert_cache_eviction_order_is_unchanged
scheduler_plan_does_not_sort_discarded_requests
kv_page_allocation_prefers_requested_numa_node
```

Parser tests must feed generated text at every split position — inside a function name, inside escaped JSON strings, between UTF-8 continuation bytes, between tool calls, and immediately before the closing marker. For each complete sample, test every possible two-chunk split.

Remember: no raw `assert(`, no `using namespace` — use `tests/support/assertions.hpp`.

# OpenAPI and documentation

Update `src/serve/protocol/openapi.json` (239 lines today) to describe tools, function definitions, JSON Schema parameters, tool choice, parallel tool calls, assistant tool calls, tool messages, streaming tool-call deltas, `finish_reason = "tool_calls"`, and capability-related errors. Verify it still renders in the bundled `/docs` route (`src/app/serve/routes/docs.cpp`).

Update `docs/ARCHITECTURE.md` with the layer chain: checkpoint format plugin → architecture resolver → validated model graph → backend compiler → chat profile → chat template → tool-call codec → protocol adapter.

Update `docs/ARCHITECTURE_RULES.md` with the new numbered rules described earlier.

Update `docs/HOW_TO_ADD_A_MODEL_ARCHITECTURE.md` — it is currently 8 steps and misnumbered (it jumps 4 → 6). Fix the numbering and state that an architecture may register or select: metadata resolution, graph construction, weight naming/planning, tokenizer configuration, chat profile, and an optional tool-call codec. Make clear that a backend is edited only when an architecture introduces a genuinely new graph operator.

Update `CHANGELOG.md` and `MANIFEST.sha256`.

# Implementation phases

## Phase 0: repository analysis

1. Re-verify the `file:line` citations in this document against the current tree.
2. Run the existing test suite and record the baseline: `python scripts/dev.py verify --backend cpu`, then `--backend cuda` if a GPU is available.
3. Capture golden prompt-rendering fixtures for all three profiles **before** any change.
4. Complete the codec verification procedure and state which profile you chose, with evidence.
5. Produce a concise implementation map and list the public API and compatibility risks.

Do not begin with broad file moves before understanding the dependency graph.

## Phase 1: architectural boundary cleanup (D4, D5, D6, D12)

Neutral tensor/checkpoint headers; removal of concrete GGUF/Safetensors dependencies from neutral code; naming policies moved to architecture modules; injectable checkpoint and architecture catalogs; extended boundary checks; CMake manifest corrections; `MANIFEST.sha256` updated.

Runtime behavior unchanged.

## Phase 2: resolver, graph and backend cleanup (D1, D2, D3, D13, D14)

Do the resolver work (D14) **before or together with** the graph-authority work (D1) — they touch the same functions, and fixing the backend's fixed group count without fixing the resolver that produces it (D14.3) achieves nothing.

Graph-driven shared-KV ownership; no fixed group counts anywhere in the chain; no architecture names in backends; no MoE suffix assumption; `ModelGraph` becomes authoritative and `RuntimeTopology` is reduced or removed; duplicate `MixerKind`/`LayerType` spellings collapsed; resolvers cleaned per D14.1–D14.12; focused CPU model-loading extraction.

Generated outputs unchanged — verify against `tests/reference_test.cpp` and `tests/numerical_compare_test.cpp`.

D14.8 and D14.9 change behavior for checkpoints that currently rely on the fabricated schedule or the magic-dimension override. Before removing either, load every checkpoint available in the local HF cache and record which ones took those paths. If a real checkpoint depends on one, keep the behavior but make it an explicit, named, logged quirk rather than an inline heuristic — and say so in the final report.

## Phase 3: text-layer decoupling and structured conversation domain (D7, D8, D9, D11, D15)

Delete `ChatTemplateKind` and `IChatTemplate::kind()`; move chat rendering out of `BpeTokenizer`; replace `BpeProfile` branching with injected tokenizer configuration; remove the `GgufFile` constructor dependency from `include/celeg/text/`; introduce structured `ChatMessage`, tool definitions/calls/choice, `ChatCapabilities`, the chat-profile catalog, and protocol-neutral validation.

Tokenizer work (D15) lands here because D7 and D15.3 touch the same profile-selection logic. Order within the phase:

1. Extend `tests/tokenizer_benchmark.cpp` with a special-token-dense chat prompt and record the baseline — the current corpus has no special tokens and cannot see D15.1 (`tokenizer_benchmark.cpp:12-15`).
2. Capture per-profile tokenization golden outputs against `tests/fixtures/tokenizer_unicode.txt`.
3. Harden `next_cp` and the decode paths (D15.6) — the streaming interpreter in Phase 6 depends on this.
4. Then the structural changes (D15.3, D15.4, D15.8) and the performance fixes (D15.1, D15.2).

Do not rewrite `bpe_symbols`. Existing chat remains byte-compatible against the Phase 0 golden fixtures, and tokenization remains byte-identical for every shipped profile.

## Phase 4: OpenAI wire support without active codecs (D10)

Request and response DTOs; OpenAI error DTOs replacing every hand-written error body; mapping and validation; OpenAPI updates; explicit rejection for profiles without tool support.

At the end of this phase the protocol understands tools but claims no model support.

## Phase 5: first verified tool-call codec

One real, verified model-specific codec: rendering tool definitions, prior assistant tool calls, and tool results; parsing one or more generated tool calls; non-streaming responses; `finish_reason = "tool_calls"`.

## Phase 6: streaming

Incremental parser; tool-call deltas; stable IDs and indexes; exact argument-fragment reconstruction; final streaming reason.

## Phase 7: optional stopping and constraints

Only after everything above is stable: incremental tool-call completion stopping, grammar-constrained arguments, reusable token constraints, sampler integration. Do not block the core implementation on constrained decoding.

## Independent track: hot-path complexity (D16)

Not a phase — an orthogonal track. Each D16 item lands as its own change, at any point, without blocking the tool-calling phases. Sequence each one as: profile → record baseline → change → re-measure → report both numbers. A D16 change that ships without before/after measurements is not done.

Two ordering constraints:

- **D16.1 before Phase 7.** The sampler is where constrained decoding would integrate; do not build token constraints on top of the per-token allocation and the double-evaluating comparator.
- **D16.3 together with the CPU scheduler responsibility split.** They are the same edit — moving onto `RequestRegistry` fixes the ownership problem and the per-step sort at once. Doing them separately means writing the same code twice.

D16.4 is a design decision, not a fix: implement real path compression or rename the type. Ask before doing either if the memory cost has not been measured.

# Deliverables

1. The code changes.
2. New and updated tests.
3. Updated architecture documentation and architecture rules.
4. Updated OpenAPI schema.
5. A migration note for every changed internal interface, plus any C API change.
6. A summary of files changed by phase.
7. A list of remaining limitations.
8. Exact commands used to build and test, with output.
9. Evidence that CPU and CUDA backends contain no tool-call or OpenAI logic.
10. Evidence that non-tool chat behavior is unchanged — the golden-fixture diff.
11. The extracted upstream chat template and marker list justifying the implemented codec.
12. The checkpoint-quirk audit from Phase 2: which locally available checkpoints relied on the fabricated Gemma4 layer schedule (D14.8) or the LFM2 magic-dimension override (D14.9), and what replaced each.
13. For every D16 item attempted: the profile that justified it, before/after numbers from a named harness, and proof that sampled output and eviction order are unchanged. For every D16 item skipped: say so explicitly rather than leaving it unmentioned.

# Quality bar

The final implementation must not:

- put architecture-name branches or string literals in CPU or CUDA code;
- branch on a chat-template or tokenizer-profile enum anywhere in `src/text`;
- expose concrete checkpoint formats through neutral contracts;
- add tool execution to the HTTP inference route;
- build JSON errors by string concatenation;
- claim tool support for a model without a verified codec;
- parse each streaming chunk as an independent complete JSON document;
- require every chat template subtype to support every role;
- add another global switch containing every architecture or chat profile;
- encode a fixed number of shared-KV groups, in a backend **or** in a resolver;
- assume MoE layers are contiguous or form a suffix;
- identify a model by a magic dimension tuple, a substring of the repository name, or a substring of a pre-tokenizer regex;
- silently default an unrecognized tokenizer or checkpoint to another model's rules;
- silently drop bytes, code points, or token ids instead of reporting malformed input;
- claim a tokenizer performance improvement measured on a corpus containing no special tokens;
- land a performance change without a before/after measurement from an existing harness;
- change sampled output for a fixed seed, or change expert-eviction order observably, in the name of performance;
- reimplement a collaborator that already exists in `src/runtime/concurrency/` instead of using it;
- fabricate architecture metadata (layer schedules, layer counts) when the checkpoint omits it, without an explicit and logged opt-in;
- spell a tensor name inline in a resolver instead of going through `ITensorNamingPolicy`;
- return a `shared_ptr` that owns nothing, or keep two parallel arrays whose index alignment is assumed but not enforced;
- report a missing tensor as a bare enum ordinal;
- keep the same semantic truth in both `RuntimeTopology` and `ModelGraph` without one clearly owning it;
- leave a stale `MANIFEST.sha256` entry;
- regress existing text chat, tokenization, CPU, CUDA, C API, or checkpoint behavior.

# Decision-making rules

1. Prefer semantic graph information over architecture IDs.
2. Prefer capabilities over exception-based discovery.
3. Prefer composition-root defaults plus injectable core dependencies.
4. Prefer neutral interfaces over concrete checkpoint-format pointers.
5. Prefer validated structured domain objects over protocol DTOs flowing into the runtime.
6. Prefer incremental refactoring over a complete rewrite.
7. Prefer measurable simplification over abstractions with no meaningful ownership.
8. Preserve hot-path efficiency.
9. Keep model-specific serialization syntax inside the model/chat-profile module.
10. Keep OpenAI-specific shapes inside the serving protocol module.
11. When this document and the repository disagree, the repository wins — preserve the architectural objective, adapt the implementation.

# Working method

For every phase:

1. Inspect the relevant current code.
2. Explain the concrete issue found, with `file:line`.
3. State the intended boundary after the change.
4. Implement the smallest coherent change.
5. Update or add tests.
6. Build and run the relevant suite.
7. Fix failures before continuing.
8. Keep commits or logical change groups reviewable.

When you find an assumption in this prompt to be inaccurate, do not force the proposed class name or structure — say so, preserve the objective, and adapt.

At the end, provide a final report with:

- SOLID assessment before and after, referencing the D-numbered defects;
- architectural boundary improvements;
- tool-calling architecture;
- supported chat profiles, with the evidence for each;
- unsupported profiles and reasons;
- test results;
- compatibility risks;
- follow-up work, especially constrained decoding and optional agent-runtime integration.
