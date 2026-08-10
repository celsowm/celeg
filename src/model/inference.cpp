#include "celeg/model/inference.hpp"

#include "inference/support.hpp"

#include <regex>
#include <sstream>
#include <unordered_set>

namespace celeg {
namespace {

void require_positive(const std::optional<int>& value, std::string_view name) {
    if (!value.has_value()) {
        inference_detail::fail(ResolutionFailureKind::MissingRequiredMetadata,
                               "automatic resolution requires metadata: " +
                                   std::string(name));
    }
    if (*value <= 0) {
        inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                               "automatic resolution received a non-positive " +
                                   std::string(name));
    }
}

} // namespace

ResolutionError::ResolutionError(ResolutionFailureKind kind, std::string message,
                                 std::vector<EvidenceItem> evidence)
    : std::runtime_error(std::move(message)), kind_(kind), evidence_(std::move(evidence)) {}

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
            inference_detail::fail(
                ResolutionFailureKind::ConflictingInferenceFacts,
                "hidden_size is not divisible by query_heads; head_dim is missing");
        }
    }
    const int head_dim = m.head_dim.value_or(*m.hidden_size / *m.query_heads);
    if (*m.query_heads % *m.key_value_heads != 0 || head_dim <= 0 || head_dim % 2 != 0) {
        inference_detail::fail(ResolutionFailureKind::ConflictingInferenceFacts,
                               "attention head geometry is not a valid GQA layout");
    }
    if (*m.key_value_heads * head_dim > *m.hidden_size) {
        inference_detail::fail(ResolutionFailureKind::ConflictingInferenceFacts,
                               "key/value projection width exceeds hidden_size");
    }
    if (!m.bos_token_id.has_value() || !m.pad_token_id.has_value() ||
        *m.bos_token_id < 0 || *m.pad_token_id < 0 || m.eos_token_ids.empty()) {
        inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                               "token IDs are incomplete");
    }

    const TensorInventoryEntry* embedding = nullptr;
    for (const std::string& name : {std::string("transformer.wte.weight"),
                                    std::string("model.embed_tokens.weight"),
                                    std::string("tok_embeddings.weight")}) {
        if (const auto* candidate = input.inventory.find(name)) {
            if (embedding != nullptr) {
                inference_detail::fail(ResolutionFailureKind::AmbiguousTensorBinding,
                                       "multiple token-embedding candidates are present");
            }
            embedding = candidate;
        }
    }
    if (embedding == nullptr) {
        inference_detail::fail(ResolutionFailureKind::MissingTensorRole,
                               "automatic resolution could not find token embedding");
    }
    if (!inference_detail::shape_is(*embedding, {*m.vocab_size, *m.hidden_size})) {
        inference_detail::fail(ResolutionFailureKind::ShapeConstraintViolation,
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
        inference_detail::fail(ResolutionFailureKind::IncompleteLayerSchedule,
                               "tensor inventory layer count does not agree with metadata");
    }
    for (int layer = 0; layer < *m.layer_count; ++layer) {
        if (!layers.contains(layer)) {
            inference_detail::fail(
                ResolutionFailureKind::IncompleteLayerSchedule,
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
    topology.feed_forward_kinds.assign(static_cast<size_t>(*m.layer_count),
                                        FeedForwardKind::Dense);
    topology.feed_forward_intermediates.assign(static_cast<size_t>(*m.layer_count),
                                               topology.intermediate);
    topology.feed_forward_activations.assign(static_cast<size_t>(*m.layer_count),
                                             ActivationKind::SwiGLU);
    topology.attention_layouts.resize(static_cast<size_t>(*m.layer_count));
    topology.attention_slot_for_layer.resize(static_cast<size_t>(*m.layer_count));
    topology.layer_for_attention_slot.reserve(static_cast<size_t>(*m.layer_count));
    topology.attention_layer_count = *m.layer_count;
    topology.num_dense_layers = *m.layer_count;
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

    const auto add_global = [&](TensorRole role, const TensorInventoryEntry& tensor) {
        inference_detail::add_binding(
            facts.bindings, role, -1, tensor,
            {{EvidenceKind::TensorName, tensor.name, std::string(tensor_role_name(role))}});
    };
    add_global(TensorRole::TokenEmbedding, *embedding);

    const std::vector<std::string> head_names = {
        "lm_head.weight", "transformer.lm_head.weight"};
    const TensorInventoryEntry* head = nullptr;
    for (const std::string& name : head_names) {
        if (const auto* candidate = input.inventory.find(name)) {
            if (head != nullptr) {
                inference_detail::fail(ResolutionFailureKind::AmbiguousTensorBinding,
                                       "multiple language-model heads are present");
            }
            head = candidate;
        }
    }
    if (head == nullptr) {
        if (!facts.tied_embeddings) {
            inference_detail::fail(ResolutionFailureKind::MissingTensorRole,
                                   "untied checkpoint has no language-model head");
        }
        head = embedding;
        facts.evidence.push_back({EvidenceKind::Derived, embedding->name,
                                  "language-model head is tied to token embedding"});
    } else if (!inference_detail::shape_is(*head, {*m.vocab_size, *m.hidden_size})) {
        inference_detail::fail(ResolutionFailureKind::ShapeConstraintViolation,
                               "language-model head shape does not agree with normalized metadata");
    }
    add_global(TensorRole::LanguageModelHead, *head);

    const std::vector<std::string> final_norm_names = {
        "transformer.ln_f.weight", "model.norm.weight", "norm.weight"};
    const TensorInventoryEntry* final_norm = nullptr;
    for (const auto& name : final_norm_names) {
        if (const auto* candidate = input.inventory.find(name)) {
            if (final_norm != nullptr) {
                inference_detail::fail(ResolutionFailureKind::AmbiguousTensorBinding,
                                       "multiple final norms are present");
            }
            final_norm = candidate;
        }
    }
    if (final_norm == nullptr) {
        inference_detail::fail(ResolutionFailureKind::MissingTensorRole,
                               "automatic resolution could not find final norm");
    }
    if (!inference_detail::shape_is(*final_norm, {*m.hidden_size})) {
        inference_detail::fail(ResolutionFailureKind::ShapeConstraintViolation,
                               "final norm shape mismatch");
    }
    add_global(TensorRole::FinalNorm, *final_norm);

    for (int layer = 0; layer < *m.layer_count; ++layer) {
        const std::string index = std::to_string(layer);
        const std::vector<std::string> norm_candidates = {
            "transformer.h." + index + ".ln_1.weight",
            "model.layers." + index + ".input_layernorm.weight",
            "model.layers." + index + ".self_attn_layer_norm.weight",
        };
        const std::vector<std::string> ffn_norm_candidates = {
            "transformer.h." + index + ".ln_2.weight",
            "model.layers." + index + ".post_attention_layernorm.weight",
        };
        const auto* attention_norm = inference_detail::find_unique(
            input.inventory, norm_candidates, TensorRole::AttentionInputNorm, layer,
            {*m.hidden_size}, {});
        inference_detail::add_binding(facts.bindings, TensorRole::AttentionInputNorm, layer,
                                      *attention_norm, {});
        const auto* q = inference_detail::find_unique(
            input.inventory, inference_detail::attention_tensor_candidates(layer, "q_proj.weight"),
            TensorRole::AttentionQuery, layer, {*m.query_heads * head_dim, *m.hidden_size}, {});
        inference_detail::add_binding(facts.bindings, TensorRole::AttentionQuery, layer, *q, {});
        const auto* k = inference_detail::find_unique(
            input.inventory, inference_detail::attention_tensor_candidates(layer, "k_proj.weight"),
            TensorRole::AttentionKey, layer, {*m.key_value_heads * head_dim, *m.hidden_size}, {});
        inference_detail::add_binding(facts.bindings, TensorRole::AttentionKey, layer, *k, {});
        const auto* v = inference_detail::find_unique(
            input.inventory, inference_detail::attention_tensor_candidates(layer, "v_proj.weight"),
            TensorRole::AttentionValue, layer, {*m.key_value_heads * head_dim, *m.hidden_size}, {});
        inference_detail::add_binding(facts.bindings, TensorRole::AttentionValue, layer, *v, {});
        const auto* o = inference_detail::find_unique(
            input.inventory, inference_detail::attention_tensor_candidates(layer, "o_proj.weight"),
            TensorRole::AttentionOutput, layer, {*m.hidden_size, *m.query_heads * head_dim}, {});
        inference_detail::add_binding(facts.bindings, TensorRole::AttentionOutput, layer, *o, {});
        const auto* ffn_norm = inference_detail::find_unique(
            input.inventory, ffn_norm_candidates, TensorRole::FfnInputNorm, layer,
            {*m.hidden_size}, {});
        inference_detail::add_binding(facts.bindings, TensorRole::FfnInputNorm, layer,
                                      *ffn_norm, {});
        const auto* gate = inference_detail::find_unique(
            input.inventory, inference_detail::feed_forward_tensor_candidates(layer,
                "w_gate.weight"), TensorRole::FfnGate, layer,
            {*m.intermediate_size, *m.hidden_size}, {});
        inference_detail::add_binding(facts.bindings, TensorRole::FfnGate, layer, *gate, {});
        const auto* up = inference_detail::find_unique(
            input.inventory, inference_detail::feed_forward_tensor_candidates(layer,
                "w_up.weight"), TensorRole::FfnUp, layer,
            {*m.intermediate_size, *m.hidden_size}, {});
        inference_detail::add_binding(facts.bindings, TensorRole::FfnUp, layer, *up, {});
        const auto* down = inference_detail::find_unique(
            input.inventory, inference_detail::feed_forward_tensor_candidates(layer,
                "w_down.weight"), TensorRole::FfnDown, layer,
            {*m.hidden_size, *m.intermediate_size}, {});
        inference_detail::add_binding(facts.bindings, TensorRole::FfnDown, layer, *down, {});
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
            inference_detail::fail(
                ResolutionFailureKind::MissingTensorRole,
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

} // namespace celeg
