#include "celeg/model/inference.hpp"

#include "celeg/model/graph_builder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace celeg {
namespace {

[[noreturn]] void fail(ResolutionFailureKind kind, std::string message,
                       std::vector<EvidenceItem> evidence = {}) {
    throw ResolutionError(kind, std::move(message), std::move(evidence));
}

template <typename T>
std::optional<T> scalar(const CheckpointMetadata& metadata, std::string_view key) {
    if (!metadata.contains(key)) return std::nullopt;
    const MetadataValue& value = metadata.value(key);
    if constexpr (std::is_same_v<T, int>) {
        if (const auto* integer = std::get_if<int64_t>(&value)) {
            if (*integer < std::numeric_limits<int>::min() ||
                *integer > std::numeric_limits<int>::max()) {
                fail(ResolutionFailureKind::ConflictingMetadata,
                     "metadata value is outside the supported integer range: " +
                         std::string(key));
            }
            return static_cast<int>(*integer);
        }
        if (const auto* number = std::get_if<double>(&value)) return static_cast<int>(*number);
        if (const auto* boolean = std::get_if<bool>(&value)) return *boolean ? 1 : 0;
    } else if constexpr (std::is_same_v<T, float>) {
        if (const auto* number = std::get_if<double>(&value)) return static_cast<float>(*number);
        if (const auto* integer = std::get_if<int64_t>(&value)) {
            return static_cast<float>(*integer);
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (const auto* number = std::get_if<double>(&value)) return *number;
        if (const auto* integer = std::get_if<int64_t>(&value)) {
            return static_cast<double>(*integer);
        }
    } else if constexpr (std::is_same_v<T, bool>) {
        if (const auto* boolean = std::get_if<bool>(&value)) return *boolean;
        if (const auto* integer = std::get_if<int64_t>(&value)) return *integer != 0;
    }
    fail(ResolutionFailureKind::ConflictingMetadata,
         "metadata key has an incompatible type: " + std::string(key));
}

template <typename T>
std::optional<T> aliases(const CheckpointMetadata& metadata,
                         std::initializer_list<std::string_view> keys,
                         std::vector<EvidenceItem>& evidence,
                         std::string_view fact) {
    std::optional<T> result;
    std::string source;
    for (const std::string_view key : keys) {
        const auto value = scalar<T>(metadata, key);
        if (!value.has_value()) continue;
        if (result.has_value() && *result != *value) {
            fail(ResolutionFailureKind::ConflictingMetadata,
                 "conflicting metadata aliases for " + std::string(fact));
        }
        result = value;
        source = std::string(key);
    }
    if (result.has_value()) {
        evidence.push_back({EvidenceKind::AliasMetadata, source,
                            std::string(fact) + " = " + std::to_string(*result)});
    }
    return result;
}

template <typename T>
void require_positive(const std::optional<T>& value, std::string_view name) {
    if (!value.has_value()) {
        fail(ResolutionFailureKind::MissingRequiredMetadata,
             "automatic resolution requires metadata: " + std::string(name));
    }
    if (*value <= 0) {
        fail(ResolutionFailureKind::ConflictingMetadata,
             "automatic resolution received a non-positive " + std::string(name));
    }
}

std::vector<int> token_list(const CheckpointMetadata& metadata, std::string_view key) {
    if (!metadata.contains(key)) return {};
    const MetadataValue& value = metadata.value(key);
    if (const auto* integer = std::get_if<int64_t>(&value)) return {static_cast<int>(*integer)};
    if (const auto* values = std::get_if<std::vector<int64_t>>(&value)) {
        std::vector<int> result;
        result.reserve(values->size());
        for (int64_t item : *values) result.push_back(static_cast<int>(item));
        return result;
    }
    fail(ResolutionFailureKind::ConflictingMetadata,
         "token metadata has an incompatible type: " + std::string(key));
}

void reject_unknown_semantic_metadata(const CheckpointMetadata& metadata) {
    static const std::unordered_set<std::string> known = {
        "qk_norm", "query_key_norm", "xsa_projection",
        "xsa_projection_minimum_norm_squared", "rope_pairing", "rope_interleaved",
        "rope_theta", "rotary_fraction", "rope_scaling", "rope_parameters",
    };
    for (const auto& [key, value] : metadata.values) {
        (void)value;
        const bool semantic_name = key.find("xsa") != std::string::npos ||
            key.find("qk_norm") != std::string::npos ||
            key.find("rope_pair") != std::string::npos;
        if (semantic_name && !known.contains(key)) {
            fail(ResolutionFailureKind::UnsupportedSemanticFeature,
                 "automatic resolution does not know the mathematics of metadata key: " + key);
        }
    }
}

bool shape_is(const TensorInventoryEntry& entry, std::initializer_list<int64_t> expected) {
    return entry.shape == std::vector<int64_t>(expected);
}

const TensorInventoryEntry* find_unique(const TensorInventory& inventory,
                                        const std::vector<std::string>& candidates,
                                        TensorRole role, int layer,
                                        std::initializer_list<int64_t> shape,
                                        std::vector<EvidenceItem> evidence) {
    std::vector<const TensorInventoryEntry*> matches;
    for (const std::string& candidate : candidates) {
        if (const auto* entry = inventory.find(candidate)) matches.push_back(entry);
    }
    if (matches.empty()) {
        fail(ResolutionFailureKind::MissingTensorRole,
             "automatic resolution could not bind " + std::string(tensor_role_name(role)) +
                 (layer >= 0 ? " for layer " + std::to_string(layer) : ""),
             std::move(evidence));
    }
    if (matches.size() != 1) {
        std::string message = "automatic resolution found multiple bindings for " +
            std::string(tensor_role_name(role));
        if (layer >= 0) message += " for layer " + std::to_string(layer);
        for (const auto* entry : matches) message += "\n  " + entry->name;
        fail(ResolutionFailureKind::AmbiguousTensorBinding, std::move(message),
             std::move(evidence));
    }
    if (!shape_is(*matches.front(), shape)) {
        fail(ResolutionFailureKind::ShapeConstraintViolation,
             "tensor " + matches.front()->name + " has a shape inconsistent with " +
                 std::string(tensor_role_name(role)));
    }
    evidence.push_back({EvidenceKind::TensorName, matches.front()->name,
                        std::string(tensor_role_name(role))});
    return matches.front();
}

std::vector<std::string> layer_candidates(int layer, std::string_view suffix) {
    const std::string index = std::to_string(layer);
    return {
        "transformer.h." + index + ".attn." + std::string(suffix),
        "model.layers." + index + ".self_attn." + std::string(suffix),
        "layers." + index + ".attention." + std::string(suffix),
    };
}

std::vector<std::string> layer_mlp_candidates(int layer, std::string_view suffix) {
    const std::string index = std::to_string(layer);
    return {
        "transformer.h." + index + ".mlp." + std::string(suffix),
        "model.layers." + index + ".mlp." + std::string(suffix),
        "layers." + index + ".feed_forward." + std::string(suffix),
    };
}

void add_binding(TensorRoleBindings& bindings, TensorRole role, int layer,
                 const TensorInventoryEntry& tensor, std::vector<EvidenceItem> evidence,
                 int physical_layer = -1) {
    bindings.values.push_back({role, layer, -1, physical_layer, tensor.name, tensor.shape,
                               std::move(evidence)});
}

} // namespace

ResolutionError::ResolutionError(ResolutionFailureKind kind, std::string message,
                                 std::vector<EvidenceItem> evidence)
    : std::runtime_error(std::move(message)), kind_(kind), evidence_(std::move(evidence)) {}

TensorInventory::TensorInventory(std::vector<TensorInventoryEntry> entries)
    : entries_(std::move(entries)) {
    std::sort(entries_.begin(), entries_.end(),
              [](const auto& left, const auto& right) { return left.name < right.name; });
    for (size_t index = 1; index < entries_.size(); ++index) {
        if (entries_[index - 1].name == entries_[index].name) {
            fail(ResolutionFailureKind::UnsupportedTensorLayout,
                 "checkpoint contains duplicate tensor identity: " + entries_[index].name);
        }
    }
}

const TensorInventoryEntry* TensorInventory::find(std::string_view name) const noexcept {
    const auto it = std::lower_bound(entries_.begin(), entries_.end(), name,
        [](const auto& entry, std::string_view value) { return entry.name < value; });
    return it != entries_.end() && it->name == name ? &*it : nullptr;
}

std::vector<const TensorInventoryEntry*> TensorInventory::with_prefix(
    std::string_view prefix) const {
    std::vector<const TensorInventoryEntry*> result;
    for (const auto& entry : entries_) {
        if (entry.name.starts_with(prefix)) result.push_back(&entry);
    }
    return result;
}

const TensorRoleBinding* TensorRoleBindings::find(TensorRole role, int layer,
                                                  int expert) const noexcept {
    for (const auto& binding : values) {
        if (binding.role == role && binding.layer == layer && binding.expert == expert) {
            return &binding;
        }
    }
    return nullptr;
}

void TensorRoleBindings::validate() const {
    std::unordered_set<std::string> identities;
    for (const auto& binding : values) {
        if (binding.source_name.empty() || binding.shape.empty()) {
            fail(ResolutionFailureKind::MissingTensorRole, "tensor role binding is incomplete");
        }
        const std::string identity = std::to_string(static_cast<int>(binding.role)) + ":" +
            std::to_string(binding.layer) + ":" + std::to_string(binding.expert);
        if (!identities.insert(identity).second) {
            fail(ResolutionFailureKind::AmbiguousTensorBinding,
                 "tensor role is bound more than once: " + identity);
        }
    }
}

TensorRoleBindings BindingSolver::solve(
    const std::vector<TensorRoleBinding>& candidates) const {
    TensorRoleBindings result;
    std::unordered_map<std::string, size_t> assignment;
    for (const auto& candidate : candidates) {
        const std::string identity = std::to_string(static_cast<int>(candidate.role)) + ":" +
            std::to_string(candidate.layer) + ":" + std::to_string(candidate.expert);
        const auto [it, inserted] = assignment.emplace(identity, result.values.size());
        if (!inserted) {
            fail(ResolutionFailureKind::AmbiguousTensorBinding,
                 "more than one complete tensor assignment remains for " + identity);
        }
        (void)it;
        result.values.push_back(candidate);
    }
    result.validate();
    return result;
}

NormalizedModelMetadata normalize_model_metadata(const CheckpointMetadata& metadata) {
    reject_unknown_semantic_metadata(metadata);
    NormalizedModelMetadata result;
    result.hidden_size = aliases<int>(metadata, {"hidden_size", "n_embd", "d_model"},
                                      result.evidence, "hidden_size");
    result.intermediate_size = aliases<int>(
        metadata, {"intermediate_size", "n_inner", "ffn_dim"}, result.evidence,
        "intermediate_size");
    result.layer_count = aliases<int>(
        metadata, {"num_hidden_layers", "n_layer", "num_layers"}, result.evidence,
        "layer_count");
    result.query_heads = aliases<int>(
        metadata, {"num_attention_heads", "n_head"}, result.evidence, "query_heads");
    result.key_value_heads = aliases<int>(
        metadata, {"num_key_value_heads", "n_kv_heads"}, result.evidence,
        "key_value_heads");
    result.head_dim = aliases<int>(metadata, {"head_dim"}, result.evidence, "head_dim");
    result.vocab_size = aliases<int>(metadata, {"vocab_size", "n_vocab"}, result.evidence,
                                     "vocab_size");
    result.context_length = aliases<int>(metadata,
        {"max_position_embeddings", "max_seq_len", "context_length"}, result.evidence,
        "context_length");
    result.norm_epsilon = aliases<float>(
        metadata, {"rms_norm_eps", "rms_norm_epsilon", "layer_norm_epsilon"},
        result.evidence, "norm_epsilon");
    result.rope_theta = aliases<double>(metadata, {"rope_theta"}, result.evidence,
                                         "rope_theta");
    result.rotary_fraction = aliases<float>(metadata, {"rotary_fraction"}, result.evidence,
                                            "rotary_fraction");
    result.bos_token_id = aliases<int>(metadata, {"bos_token_id"}, result.evidence,
                                       "bos_token_id");
    result.pad_token_id = aliases<int>(metadata, {"pad_token_id"}, result.evidence,
                                       "pad_token_id");
    result.query_key_norm = aliases<bool>(
        metadata, {"qk_norm", "query_key_norm"}, result.evidence, "query_key_norm");
    result.xsa_projection = aliases<bool>(metadata, {"xsa_projection"}, result.evidence,
                                          "xsa_projection");
    result.xsa_minimum_norm_squared = aliases<float>(
        metadata, {"xsa_projection_minimum_norm_squared"}, result.evidence,
        "xsa_minimum_norm_squared");
    result.tied_embeddings = aliases<bool>(
        metadata, {"tie_word_embeddings", "tied_embeddings"}, result.evidence,
        "tied_embeddings");

    const std::vector<int> eos = token_list(metadata, "eos_token_id");
    result.eos_token_ids = eos.empty() ? token_list(metadata, "eos_token_ids") : eos;
    if (!metadata.contains("bos_token_id")) result.bos_token_id = 0;
    if (result.eos_token_ids.empty()) result.eos_token_ids = {0};
    if (!metadata.contains("pad_token_id")) result.pad_token_id = 1;
    if (!metadata.contains("rms_norm_eps") && !metadata.contains("rms_norm_epsilon") &&
        !metadata.contains("layer_norm_epsilon")) result.norm_epsilon = 1.0e-6f;
    if (!metadata.contains("rope_theta")) result.rope_theta = 100000.0;
    if (!result.rotary_fraction.has_value()) result.rotary_fraction = 1.0f;
    if (!result.query_key_norm.has_value()) result.query_key_norm = false;
    if (!result.xsa_projection.has_value()) result.xsa_projection = false;
    if (!result.xsa_minimum_norm_squared.has_value()) result.xsa_minimum_norm_squared = 1.0e-6f;
    if (!result.tied_embeddings.has_value()) result.tied_embeddings = false;

    if (metadata.contains("rope_pairing")) {
        const std::string pairing = metadata.string("rope_pairing");
        if (pairing == "adjacent_pairs" || pairing == "interleaved") {
            result.rope_pairing = RopePairingKind::AdjacentPairs;
        } else if (pairing == "split_half") {
            result.rope_pairing = RopePairingKind::SplitHalf;
        } else {
            fail(ResolutionFailureKind::UnsupportedSemanticFeature,
                 "unsupported RoPE pairing: " + pairing);
        }
    } else if (*result.xsa_projection) {
        // XSA's published checkpoint convention uses interleaved adjacent
        // rotary pairs. This is a reusable semantic convention, not a family
        // identity test.
        result.rope_pairing = RopePairingKind::AdjacentPairs;
        result.evidence.push_back({EvidenceKind::FormatGuarantee,
                                   "xsa_projection", "RoPE pairing = adjacent_pairs"});
    } else {
        result.rope_pairing = RopePairingKind::SplitHalf;
    }
    return result;
}

TensorInventory build_tensor_inventory(const IWeightRepository& repository) {
    std::vector<TensorInventoryEntry> entries;
    for (const std::string& name : repository.names()) {
        const HostTensorView tensor = repository.tensor(name);
        if (name.empty() || tensor.shape.empty() || tensor.shape.size() > 4) {
            fail(ResolutionFailureKind::UnsupportedTensorLayout,
                 "tensor inventory contains invalid tensor metadata: " + name);
        }
        entries.push_back({name, tensor.shape, tensor.dtype});
    }
    return TensorInventory(std::move(entries));
}

InferenceInput build_inference_input(const CheckpointView& checkpoint) {
    if (!checkpoint.repository) {
        fail(ResolutionFailureKind::MissingTensorRole,
             "automatic checkpoint resolution requires a tensor repository");
    }
    return {normalize_model_metadata(checkpoint.metadata),
            build_tensor_inventory(*checkpoint.repository)};
}

CanonicalModelFacts infer_canonical_model_facts(const InferenceInput& input) {
    const auto& m = input.metadata;
    require_positive(m.hidden_size, "hidden_size");
    require_positive(m.intermediate_size, "intermediate_size");
    require_positive(m.layer_count, "layer_count");
    require_positive(m.query_heads, "query_heads");
    require_positive(m.key_value_heads, "key_value_heads");
    require_positive(m.vocab_size, "vocab_size");
    require_positive(m.context_length, "context_length");
    if (!m.head_dim.has_value()) {
        if (*m.hidden_size % *m.query_heads != 0) {
            fail(ResolutionFailureKind::ConflictingInferenceFacts,
                 "hidden_size is not divisible by query_heads; head_dim is missing");
        }
    }
    const int head_dim = m.head_dim.value_or(*m.hidden_size / *m.query_heads);
    if (*m.query_heads % *m.key_value_heads != 0 || head_dim <= 0 || head_dim % 2 != 0) {
        fail(ResolutionFailureKind::ConflictingInferenceFacts,
             "attention head geometry is not a valid GQA layout");
    }
    if (*m.key_value_heads * head_dim > *m.hidden_size) {
        fail(ResolutionFailureKind::ConflictingInferenceFacts,
             "key/value projection width exceeds hidden_size");
    }
    if (*m.bos_token_id < 0 || *m.pad_token_id < 0 || m.eos_token_ids.empty()) {
        fail(ResolutionFailureKind::ConflictingMetadata, "token IDs are incomplete");
    }

    const TensorInventoryEntry* embedding = nullptr;
    for (const std::string& name : {std::string("transformer.wte.weight"),
                                    std::string("model.embed_tokens.weight"),
                                    std::string("tok_embeddings.weight")}) {
        if (const auto* candidate = input.inventory.find(name)) {
            if (embedding != nullptr) {
                fail(ResolutionFailureKind::AmbiguousTensorBinding,
                     "multiple token-embedding candidates are present");
            }
            embedding = candidate;
        }
    }
    if (embedding == nullptr) {
        fail(ResolutionFailureKind::MissingTensorRole, "automatic resolution could not find token embedding");
    }
    if (!shape_is(*embedding, {*m.vocab_size, *m.hidden_size})) {
        fail(ResolutionFailureKind::ShapeConstraintViolation,
             "token embedding shape does not agree with normalized metadata");
    }

    std::unordered_set<int> layers;
    const std::regex layer_pattern(R"((?:transformer\.h|model\.layers|layers)\.(\d+)\.)");
    for (const auto& entry : input.inventory.entries()) {
        std::smatch match;
        if (std::regex_search(entry.name, match, layer_pattern)) {
            layers.insert(std::stoi(match[1].str()));
        }
    }
    if (layers.size() != static_cast<size_t>(*m.layer_count)) {
        fail(ResolutionFailureKind::IncompleteLayerSchedule,
             "tensor inventory layer count does not agree with metadata");
    }
    for (int layer = 0; layer < *m.layer_count; ++layer) {
        if (!layers.contains(layer)) {
            fail(ResolutionFailureKind::IncompleteLayerSchedule,
                 "tensor inventory has a gap at layer " + std::to_string(layer));
        }
    }

    CanonicalModelFacts facts;
    facts.resolution_mode = "automatic";
    facts.tied_embeddings = *m.tied_embeddings;
    facts.evidence = m.evidence;
    RuntimeTopology& topology = facts.topology;
    topology.hidden = *m.hidden_size;
    topology.intermediate = *m.intermediate_size;
    topology.dense_intermediate = topology.intermediate;
    topology.max_feed_forward_intermediate = topology.intermediate;
    topology.num_hidden_layers = *m.layer_count;
    topology.vocab_size = *m.vocab_size;
    topology.max_position_embeddings = *m.context_length;
    topology.mixer_kinds.assign(static_cast<size_t>(*m.layer_count), MixerKind::Attention);
    topology.feed_forward_kinds.assign(static_cast<size_t>(*m.layer_count), FeedForwardKind::Dense);
    topology.feed_forward_intermediates.assign(static_cast<size_t>(*m.layer_count), topology.intermediate);
    topology.feed_forward_activations.assign(static_cast<size_t>(*m.layer_count), ActivationKind::SwiGLU);
    topology.attention_layouts.resize(static_cast<size_t>(*m.layer_count));
    topology.attention_slot_for_layer.resize(static_cast<size_t>(*m.layer_count));
    topology.layer_for_attention_slot.reserve(static_cast<size_t>(*m.layer_count));
    topology.attention_layer_count = *m.layer_count;
    topology.num_dense_layers = *m.layer_count;
    topology.dense_intermediate = topology.intermediate;
    topology.shared_kv_group_count = 0;
    topology.token_policy = {*m.bos_token_id, m.eos_token_ids, *m.pad_token_id};
    topology.numerical_policy.norm_eps = *m.norm_epsilon;
    topology.numerical_policy.attention_multiplier = 0.125f;

    const auto make_attention = [&]() {
        AttentionSpec attention{*m.query_heads, *m.key_value_heads, head_dim,
                                *m.query_key_norm, FullCausalPattern{}, {}, 1.0f, false};
        attention.position = RopePositionSpec{*m.rope_theta, *m.rotary_fraction,
                                              RopeScalingSpec{}};
        std::get<RopePositionSpec>(attention.position).pairing = *m.rope_pairing;
        attention.state_storage = AttentionStateStorageSpec{};
        if (*m.xsa_projection) {
            attention.output_transform = OrthogonalizeCurrentValueSpec{
                *m.xsa_minimum_norm_squared};
        }
        return attention;
    };
    for (int layer = 0; layer < *m.layer_count; ++layer) {
        topology.attention_slot_for_layer[static_cast<size_t>(layer)] = layer;
        topology.layer_for_attention_slot.push_back(layer);
        topology.attention_layouts[static_cast<size_t>(layer)] = make_attention();
    }
    topology.validate();

    const auto add_global = [&](TensorRole role, const TensorInventoryEntry& tensor,
                                int physical_layer = -1) {
        add_binding(facts.bindings, role, -1, tensor, {
            {EvidenceKind::TensorName, tensor.name, std::string(tensor_role_name(role))},
        });
    };
    add_global(TensorRole::TokenEmbedding, *embedding);
    const std::vector<std::string> head_names = {"lm_head.weight", "transformer.lm_head.weight"};
    const TensorInventoryEntry* head = nullptr;
    for (const std::string& name : head_names) {
        if (const auto* candidate = input.inventory.find(name)) {
            if (head != nullptr) fail(ResolutionFailureKind::AmbiguousTensorBinding,
                                      "multiple language-model heads are present");
            head = candidate;
        }
    }
    if (head == nullptr) {
        if (!facts.tied_embeddings) {
            fail(ResolutionFailureKind::MissingTensorRole,
                 "untied checkpoint has no language-model head");
        }
        head = embedding;
        facts.evidence.push_back({EvidenceKind::Derived, embedding->name,
                                  "language-model head is tied to token embedding"});
    } else if (!shape_is(*head, {*m.vocab_size, *m.hidden_size})) {
        fail(ResolutionFailureKind::ShapeConstraintViolation,
             "language-model head shape does not agree with normalized metadata");
    }
    add_global(TensorRole::LanguageModelHead, *head);

    const std::vector<std::string> final_norm_names = {
        "transformer.ln_f.weight", "model.norm.weight", "norm.weight"};
    const TensorInventoryEntry* final_norm = nullptr;
    for (const auto& name : final_norm_names) {
        if (const auto* candidate = input.inventory.find(name)) {
            if (final_norm != nullptr) fail(ResolutionFailureKind::AmbiguousTensorBinding,
                                             "multiple final norms are present");
            final_norm = candidate;
        }
    }
    if (final_norm == nullptr) fail(ResolutionFailureKind::MissingTensorRole,
                                     "automatic resolution could not find final norm");
    if (!shape_is(*final_norm, {*m.hidden_size})) {
        fail(ResolutionFailureKind::ShapeConstraintViolation, "final norm shape mismatch");
    }
    add_global(TensorRole::FinalNorm, *final_norm);

    for (int layer = 0; layer < *m.layer_count; ++layer) {
        const auto norm_candidates = [&] {
            const std::string index = std::to_string(layer);
            return std::vector<std::string>{
                "transformer.h." + index + ".ln_1.weight",
                "model.layers." + index + ".input_layernorm.weight",
                "model.layers." + index + ".self_attn_layer_norm.weight",
            };
        }();
        const auto ffn_norm_candidates = [&] {
            const std::string index = std::to_string(layer);
            return std::vector<std::string>{
                "transformer.h." + index + ".ln_2.weight",
                "model.layers." + index + ".post_attention_layernorm.weight",
            };
        }();
        std::vector<EvidenceItem> evidence;
        const auto* attention_norm = find_unique(input.inventory, norm_candidates,
            TensorRole::AttentionInputNorm, layer, {*m.hidden_size}, evidence);
        add_binding(facts.bindings, TensorRole::AttentionInputNorm, layer,
                    *attention_norm, std::move(evidence));
        const auto* q = find_unique(input.inventory,
            layer_candidates(layer, "q_proj.weight"), TensorRole::AttentionQuery, layer,
            {*m.query_heads * head_dim, *m.hidden_size}, {});
        add_binding(facts.bindings, TensorRole::AttentionQuery, layer, *q, {});
        const auto* k = find_unique(input.inventory,
            layer_candidates(layer, "k_proj.weight"), TensorRole::AttentionKey, layer,
            {*m.key_value_heads * head_dim, *m.hidden_size}, {});
        add_binding(facts.bindings, TensorRole::AttentionKey, layer, *k, {});
        const auto* v = find_unique(input.inventory,
            layer_candidates(layer, "v_proj.weight"), TensorRole::AttentionValue, layer,
            {*m.key_value_heads * head_dim, *m.hidden_size}, {});
        add_binding(facts.bindings, TensorRole::AttentionValue, layer, *v, {});
        const auto* o = find_unique(input.inventory,
            layer_candidates(layer, "o_proj.weight"), TensorRole::AttentionOutput, layer,
            {*m.hidden_size, *m.query_heads * head_dim}, {});
        add_binding(facts.bindings, TensorRole::AttentionOutput, layer, *o, {});
        const auto* ffn_norm = find_unique(input.inventory, ffn_norm_candidates,
            TensorRole::FfnInputNorm, layer, {*m.hidden_size}, {});
        add_binding(facts.bindings, TensorRole::FfnInputNorm, layer, *ffn_norm, {});
        const auto* gate = find_unique(input.inventory,
            layer_mlp_candidates(layer, "w_gate.weight"), TensorRole::FfnGate, layer,
            {*m.intermediate_size, *m.hidden_size}, {});
        add_binding(facts.bindings, TensorRole::FfnGate, layer, *gate, {});
        const auto* up = find_unique(input.inventory,
            layer_mlp_candidates(layer, "w_up.weight"), TensorRole::FfnUp, layer,
            {*m.intermediate_size, *m.hidden_size}, {});
        add_binding(facts.bindings, TensorRole::FfnUp, layer, *up, {});
        const auto* down = find_unique(input.inventory,
            layer_mlp_candidates(layer, "w_down.weight"), TensorRole::FfnDown, layer,
            {*m.hidden_size, *m.intermediate_size}, {});
        add_binding(facts.bindings, TensorRole::FfnDown, layer, *down, {});
    }
    facts.bindings = BindingSolver{}.solve(facts.bindings.values);
    facts.validate();
    return facts;
}

std::string CanonicalModelFacts::fingerprint() const {
    std::ostringstream out;
    out << resolution_mode << ':' << source_format << ':' << topology.fingerprint()
        << ":tied=" << tied_embeddings;
    for (const auto& binding : bindings.values) {
        out << ':' << static_cast<int>(binding.role) << ':' << binding.layer << ':'
            << binding.expert << ':' << binding.source_name;
        for (const auto dimension : binding.shape) out << ':' << dimension;
    }
    return out.str();
}

void CanonicalModelFacts::validate() const {
    topology.validate();
    bindings.validate();
    const auto require = [&](TensorRole role, int layer) {
        if (bindings.find(role, layer) == nullptr) {
            fail(ResolutionFailureKind::MissingTensorRole,
                 "canonical facts omit required tensor role " +
                     std::string(tensor_role_name(role)));
        }
    };
    require(TensorRole::TokenEmbedding, -1);
    require(TensorRole::LanguageModelHead, -1);
    require(TensorRole::FinalNorm, -1);
    for (int layer = 0; layer < topology.num_hidden_layers; ++layer) {
        require(TensorRole::AttentionInputNorm, layer);
        require(TensorRole::AttentionQuery, layer);
        require(TensorRole::AttentionKey, layer);
        require(TensorRole::AttentionValue, layer);
        require(TensorRole::AttentionOutput, layer);
        require(TensorRole::FfnInputNorm, layer);
        require(TensorRole::FfnGate, layer);
        require(TensorRole::FfnUp, layer);
        require(TensorRole::FfnDown, layer);
    }
}

ModelGraph GraphSynthesizer::synthesize(const CanonicalModelFacts& facts) const {
    facts.validate();
    ResolvedModel intermediate;
    intermediate.topology = facts.topology;
    build_dense_transformer_graph(intermediate);
    return std::move(intermediate.graph);
}

WeightPlan WeightPlanSynthesizer::synthesize(const CanonicalModelFacts& facts) const {
    facts.validate();
    WeightPlan result;
    result.requests.reserve(facts.bindings.values.size());
    for (const auto& binding : facts.bindings.values) {
        result.requests.push_back({binding.role, binding.layer, binding.expert,
                                   binding.shape, binding.source_name, binding.physical_layer});
    }
    return result;
}

ResolvedModel ResolutionAssembler::assemble(const CanonicalModelFacts& facts) const {
    facts.validate();
    ResolvedModel result;
    result.topology = facts.topology;
    result.graph = GraphSynthesizer{}.synthesize(facts);
    result.weight_plan = WeightPlanSynthesizer{}.synthesize(facts);
    result.capabilities = {true, true, false, facts.tied_embeddings};
    result.provenance.architecture_id = facts.resolution_mode;
    result.provenance.source_format = facts.source_format;
    result.provenance.checkpoint_profile_id = facts.resolution_mode;
    derive_runtime_topology_from_graph(result.topology, result.graph);
    result.topology.validate();
    result.provenance.identity = facts.fingerprint();
    result.validate();
    return result;
}

ResolutionReport explain_resolution(const CheckpointView& checkpoint) {
    ResolutionReport report;
    report.resolution_mode = "automatic";
    try {
        const InferenceInput input = build_inference_input(checkpoint);
        report.metadata = input.metadata;
        const CanonicalModelFacts facts = infer_canonical_model_facts(input);
        report.accepted = facts.evidence;
        for (const auto& binding : facts.bindings.values) {
            report.accepted.insert(report.accepted.end(), binding.evidence.begin(),
                                   binding.evidence.end());
        }
        report.canonical_fingerprint = facts.fingerprint();
    } catch (const ResolutionError& error) {
        report.failures.push_back(error.what());
        report.rejected = error.evidence();
    }
    return report;
}

} // namespace celeg
