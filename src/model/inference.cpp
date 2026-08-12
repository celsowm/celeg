#include "celeg/model/inference.hpp"

#include "inference/support.hpp"

#include <cmath>
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
    require_positive(m.layer_count, "layer_count");
    require_positive(m.vocab_size, "vocab_size");
    require_positive(m.context_length, "context_length");
    if (!m.bos_token_id.has_value() || !m.pad_token_id.has_value() ||
        *m.bos_token_id < 0 || *m.pad_token_id < 0 || m.eos_token_ids.empty()) {
        inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                               "token IDs are incomplete");
    }

    const TensorInventoryEntry* embedding = nullptr;
    for (const std::string& name : {std::string("transformer.wte.weight"),
                                    std::string("model.embed_tokens.weight"),
                                    std::string("model.language_model.embed_tokens.weight"),
                                    std::string("tok_embeddings.weight"),
                                    std::string("token_embd.weight"),
                                    std::string("backbone.embeddings.weight"),
                                    std::string("model.word_embeddings.weight")}) {
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
    const std::regex layer_pattern(R"((?:transformer\.h|model\.language_model\.layers|model\.layers|backbone\.layers|layers|blk)\.(\d+)\.)");
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
    facts.tied_embeddings = m.tied_embeddings.value_or(false);
    facts.evidence = m.evidence;
    RuntimeTopology& topology = facts.topology;
    NumericalPolicy& numerical_policy = facts.numerical_policy;
    topology.hidden = *m.hidden_size;
    std::vector<int> intermediate_sizes;
    intermediate_sizes.reserve(static_cast<size_t>(*m.layer_count));
    for (int layer = 0; layer < *m.layer_count; ++layer) {
        const std::optional<int> value = m.intermediate_size.value_for(layer);
        const std::string index = std::to_string(layer);
        const TensorInventoryEntry* ffn_up = input.inventory.find(
            "blk." + index + ".ffn_up.weight");
        if (!ffn_up) {
            ffn_up = input.inventory.find(
                "model.layers." + index + ".mlp.up_proj.weight");
        }
        if (!ffn_up) {
            ffn_up = input.inventory.find(
                "model.language_model.layers." + index + ".mlp.up_proj.weight");
        }
        if (!ffn_up) {
            ffn_up = input.inventory.find(
                "transformer.h." + index + ".mlp.w_up.weight");
        }
        if (!ffn_up) {
            ffn_up = input.inventory.find(
                "model.layers." + index + ".feed_forward.w1.weight");
        }
        if (!ffn_up && topology.num_experts > 0 &&
            layer >= topology.num_dense_layers) {
            ffn_up = input.inventory.find(
                "model.layers." + index + ".mlp.experts.0.gate_proj.weight");
        }
        if (!ffn_up && topology.num_experts > 0 &&
            layer >= topology.num_dense_layers) {
            ffn_up = input.inventory.find(
                "model.layers." + index + ".mlp.experts.0.gate_proj.weight_packed");
        }
        const bool tensor_defines_intermediate =
            ffn_up && ffn_up->shape.size() == 2 && ffn_up->shape[0] > 0 &&
            m.feed_forward_auto_adjust.value_or(false);
        if (!value.has_value() || *value <= 0 || tensor_defines_intermediate) {
            if (ffn_up && ffn_up->shape.size() == 2 && ffn_up->shape[0] > 0) {
                intermediate_sizes.push_back(static_cast<int>(ffn_up->shape[0]));
            } else {
                // Mixer-only layers deliberately carry a positive compiled
                // placeholder; their FFN execution and weight requests are
                // disabled by the per-layer schedule below.
                intermediate_sizes.push_back(1);
            }
        } else {
            intermediate_sizes.push_back(*value);
        }
    }
    topology.intermediate = *std::max_element(intermediate_sizes.begin(), intermediate_sizes.end());
    topology.dense_intermediate = topology.intermediate;
    topology.max_feed_forward_intermediate = topology.intermediate;
    topology.num_hidden_layers = *m.layer_count;
    topology.checkpoint.vocab_size = *m.vocab_size;
    topology.checkpoint.max_position_embeddings = *m.context_length;
    topology.mixer_kinds.assign(static_cast<size_t>(*m.layer_count), MixerKind::Attention);
    topology.feed_forward_kinds.assign(static_cast<size_t>(*m.layer_count),
                                        FeedForwardKind::Dense);
    topology.execute_feed_forward.assign(static_cast<size_t>(*m.layer_count), true);
    topology.feed_forward_intermediates = intermediate_sizes;
    topology.feed_forward_activations.assign(static_cast<size_t>(*m.layer_count),
                                             ActivationKind::SwiGLU);
    if (topology.num_experts > 0 && topology.experts_per_token > 0) {
        for (int layer = topology.num_dense_layers; layer < *m.layer_count; ++layer) {
            topology.feed_forward_kinds[static_cast<size_t>(layer)] =
                FeedForwardKind::MixtureOfExperts;
        }
    }
    topology.attention_layouts.resize(static_cast<size_t>(*m.layer_count));
    topology.gated_delta_net_layouts.resize(static_cast<size_t>(*m.layer_count));
    topology.mamba2_layouts.resize(static_cast<size_t>(*m.layer_count));
    topology.mlp_only_layouts.resize(static_cast<size_t>(*m.layer_count));
    topology.attention_slot_for_layer.assign(static_cast<size_t>(*m.layer_count), -1);
    topology.layer_for_attention_slot.reserve(static_cast<size_t>(*m.layer_count));
    topology.attention_layer_count = 0;
    topology.conv_cache = m.shortconv_cache.value_or(0);
    topology.conv_dim = *m.hidden_size;
    topology.num_dense_layers = m.first_dense_layer.value_or(*m.layer_count);
    if (topology.num_dense_layers < 0 || topology.num_dense_layers > *m.layer_count) {
        inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                               "first dense layer is outside the layer schedule");
    }
    topology.num_experts = m.moe_experts.value_or(0);
    topology.experts_per_token = m.moe_experts_per_token.value_or(0);
    if (topology.num_experts > 0 && topology.experts_per_token > 0) {
        for (int layer = 0; layer < *m.layer_count; ++layer) {
            topology.feed_forward_kinds[static_cast<size_t>(layer)] =
                layer >= topology.num_dense_layers
                    ? FeedForwardKind::MixtureOfExperts : FeedForwardKind::Dense;
        }
    }
    topology.moe_intermediate = m.moe_intermediate.value_or(0);
    topology.shared_expert_intermediate = m.moe_shared_intermediate.value_or(0);
    topology.normalize_topk = m.moe_normalize_topk.value_or(false);
    topology.use_expert_bias = m.moe_expert_bias.value_or(false);
    topology.routed_scaling_factor = m.moe_routed_scaling.value_or(1.0f);
    topology.moe_router_softmax = m.moe_score_function.value_or("sigmoid") == "softmax";
    topology.moe_routing_group_count = m.moe_routing_groups.value_or(0);
    topology.moe_routing_groups_per_token = m.moe_routing_groups.value_or(0);
    topology.moe_routing_group_count = m.moe_total_routing_groups.value_or(
        topology.moe_routing_group_count);
    topology.moe_routing_group_score_top_k = m.moe_group_score_top_k.value_or(0);
    if (topology.num_experts > 0 && topology.moe_routing_group_count > 0) {
        if (topology.num_experts % topology.moe_routing_group_count != 0) {
            inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                                   "MoE expert count is not divisible by routing group count");
        }
        topology.moe_routing_experts_per_group =
            topology.num_experts / topology.moe_routing_group_count;
        if (topology.moe_routing_groups_per_token <= 0 ||
            topology.moe_routing_groups_per_token > topology.moe_routing_group_count) {
            inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                                   "invalid number of selected MoE routing groups");
        }
        if (topology.moe_routing_group_score_top_k <= 0 ||
            topology.moe_routing_group_score_top_k > topology.moe_routing_experts_per_group) {
            inference_detail::fail(ResolutionFailureKind::ConflictingMetadata,
                                   "invalid MoE routing group score width");
        }
    }
    topology.checkpoint.token_policy = {*m.bos_token_id, m.eos_token_ids, *m.pad_token_id};
    numerical_policy.norm_eps = *m.norm_epsilon;
    numerical_policy.embedding_multiplier = m.embedding_multiplier.value_or(1.0f);
    numerical_policy.residual_multiplier = m.residual_multiplier.value_or(1.0f);
    numerical_policy.logits_multiplier = m.logits_multiplier.value_or(1.0f);
    numerical_policy.logits_divisor = m.logits_divisor.value_or(1.0f);

    ModelGraph& graph = facts.graph;
    graph.hidden = *m.hidden_size;
    graph.final_norm = {numerical_policy.norm_eps, NormWeightKind::Scale};
    graph.embedding_transform.multiplier = numerical_policy.embedding_multiplier;
    graph.logits_multiplier = numerical_policy.logits_multiplier;
    graph.logits_divisor = numerical_policy.logits_divisor;
    graph.layers.resize(static_cast<size_t>(*m.layer_count));
    for (int layer = 0; layer < *m.layer_count; ++layer) {
        LayerSpec& semantic_layer = graph.layers[static_cast<size_t>(layer)];
        semantic_layer.operator_norm = {numerical_policy.norm_eps, NormWeightKind::Scale};
        semantic_layer.feed_forward_norm = {numerical_policy.norm_eps, NormWeightKind::Scale};
        semantic_layer.residual.multiplier = numerical_policy.residual_multiplier;
        semantic_layer.execute_feed_forward = topology.execute_feed_forward.at(
            static_cast<size_t>(layer));
        if (topology.feed_forward_kinds.at(static_cast<size_t>(layer)) ==
            FeedForwardKind::MixtureOfExperts) {
            semantic_layer.feed_forward = MixtureOfExpertsSpec{
                .intermediate_size = topology.moe_intermediate,
                .num_experts = topology.num_experts,
                .experts_per_token = topology.experts_per_token,
                .normalize_topk = topology.normalize_topk,
                .use_expert_bias = topology.use_expert_bias,
                .routed_scaling_factor = topology.routed_scaling_factor,
                .routing_group_count = topology.moe_routing_group_count,
                .routing_experts_per_group = topology.moe_routing_experts_per_group,
                .routing_groups_per_token = topology.moe_routing_groups_per_token,
                .routing_group_score_top_k = topology.moe_routing_group_score_top_k,
                .has_shared_expert = topology.shared_expert_intermediate > 0,
                .shared_intermediate_size = topology.shared_expert_intermediate,
                .router_softmax = topology.moe_router_softmax};
        } else {
            semantic_layer.feed_forward = DenseFeedForwardSpec{
                intermediate_sizes.at(static_cast<size_t>(layer)), ActivationKind::SwiGLU};
        }
    }

    const auto make_attention = [&](int query_heads, int key_value_heads, int head_dim,
                                    bool query_key_norm) {
        AttentionSpec attention;
        attention.query_heads = query_heads;
        attention.key_value_heads = key_value_heads;
        attention.head_dim = head_dim;
        attention.query_norm = {query_key_norm ? *m.norm_epsilon : 0.0f,
                                query_key_norm ? NormWeightKind::Scale
                                                : NormWeightKind::None};
        attention.key_norm = attention.query_norm;
        attention.pattern = FullCausalPattern{};
        attention.query_scale = 1.0f;
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
        LayerSpec& semantic_layer = graph.layers[static_cast<size_t>(layer)];
        const auto has_tensor = [&](std::string_view name) {
            return input.inventory.find(name) != nullptr;
        };
        const auto find_mamba_tensor = [&](std::string_view suffix) {
            const auto candidates = inference_detail::mamba2_tensor_candidates(layer, suffix);
            const TensorInventoryEntry* found = nullptr;
            for (const auto& candidate : candidates) {
                if (const auto* tensor = input.inventory.find(candidate)) {
                    if (found != nullptr) {
                        inference_detail::fail(
                            ResolutionFailureKind::AmbiguousTensorBinding,
                            "multiple Mamba-2 tensor spellings are present for layer " +
                                std::to_string(layer));
                    }
                    found = tensor;
                }
            }
            return found;
        };
        const std::string layer_prefix = "blk." + std::to_string(layer) + ".";
        const std::string model_layer_prefix =
            "model.layers." + std::to_string(layer) + ".";
        const bool has_kda =
            has_tensor(model_layer_prefix + "attention.q_proj.weight") &&
            has_tensor(model_layer_prefix + "attention.f_proj.weight") &&
            has_tensor(model_layer_prefix + "attention.q_conv1d.weight");
        const bool has_mla = has_tensor(model_layer_prefix + "attention.q_a_proj.weight");
        const bool has_attention =
            has_tensor(layer_prefix + "attn_q.weight") ||
            has_tensor("model.layers." + std::to_string(layer) + ".self_attn.q_proj.weight") ||
            has_tensor("model.language_model.layers." + std::to_string(layer) + ".self_attn.q_proj.weight") ||
            has_tensor("transformer.h." + std::to_string(layer) + ".attn.q_proj.weight") ||
            has_mla;
        const bool has_mamba = find_mamba_tensor("in_proj.weight") != nullptr;
        const bool has_shortconv = has_tensor(layer_prefix + "shortconv.in_proj.weight") ||
            has_tensor("model.layers." + std::to_string(layer) + ".conv.in_proj.weight") ||
            has_tensor("model.language_model.layers." + std::to_string(layer) + ".conv.in_proj.weight");
        const bool has_ffn =
            find_mamba_tensor("in_proj.weight") == nullptr &&
            (has_tensor(layer_prefix + "ffn_up.weight") ||
             has_tensor("model.layers." + std::to_string(layer) + ".mlp.up_proj.weight") ||
             has_tensor("model.language_model.layers." + std::to_string(layer) + ".mlp.up_proj.weight") ||
             has_tensor("transformer.h." + std::to_string(layer) + ".mlp.w_up.weight") ||
             has_tensor("model.layers." + std::to_string(layer) + ".feed_forward.w1.weight") ||
             (m.moe_experts.value_or(0) > 0 &&
              has_tensor(model_layer_prefix + "mlp.experts.0.gate_proj.weight")));
        const auto query_heads = m.query_heads.value_for(layer);
        const auto key_value_heads = m.key_value_heads.value_for(layer);
        const auto explicit_head_dim = m.head_dim.value_for(layer);
        if (has_mamba && has_attention) {
            inference_detail::fail(ResolutionFailureKind::ConflictingInferenceFacts,
                                   "layer has both attention and Mamba-2 tensor grammars: " +
                                       std::to_string(layer));
        }
        if (has_kda) {
            const auto* q = input.inventory.find(model_layer_prefix + "attention.q_proj.weight");
            const auto* k = input.inventory.find(model_layer_prefix + "attention.k_proj.weight");
            const auto* v = input.inventory.find(model_layer_prefix + "attention.v_proj.weight");
            const auto* f = input.inventory.find(model_layer_prefix + "attention.f_proj.weight");
            const auto* b = input.inventory.find(model_layer_prefix + "attention.b_proj.weight");
            const auto* g = input.inventory.find(model_layer_prefix + "attention.g_proj.weight");
            const auto* qc = input.inventory.find(model_layer_prefix + "attention.q_conv1d.weight");
            const auto* kc = input.inventory.find(model_layer_prefix + "attention.k_conv1d.weight");
            const auto* vc = input.inventory.find(model_layer_prefix + "attention.v_conv1d.weight");
            const auto* dt = input.inventory.find(model_layer_prefix + "attention.dt_bias");
            const auto* a_log = input.inventory.find(model_layer_prefix + "attention.A_log");
            const auto* norm = input.inventory.find(model_layer_prefix + "attention.o_norm.weight");
            const auto* out = input.inventory.find(model_layer_prefix + "attention.o_proj.weight");
            const int heads = m.recurrent_key_heads.value_or(*m.query_heads.value_for(layer));
            const int key_dim = m.recurrent_key_dim.value_or(*m.head_dim.value_for(layer));
            const int value_dim = m.recurrent_value_dim.value_or(key_dim);
            const int width = heads * key_dim;
            const int conv_kernel = m.recurrent_conv_kernel.value_or(
                qc && qc->shape.size() == 3 ? static_cast<int>(qc->shape[2]) : 0);
            if (!q || !k || !v || !f || !b || !g || !qc || !kc || !vc || !dt || !a_log ||
                !norm || !out || heads <= 0 || key_dim <= 0 || value_dim <= 0 ||
                q->shape != std::vector<std::int64_t>{width, *m.hidden_size} ||
                k->shape != std::vector<std::int64_t>{width, *m.hidden_size} ||
                v->shape != std::vector<std::int64_t>{width, *m.hidden_size} ||
                f->shape != std::vector<std::int64_t>{width, *m.hidden_size} ||
                b->shape != std::vector<std::int64_t>{heads, *m.hidden_size} ||
                g->shape != std::vector<std::int64_t>{width, *m.hidden_size} ||
                qc->shape != std::vector<std::int64_t>{width, 1, conv_kernel} ||
                kc->shape != std::vector<std::int64_t>{width, 1, conv_kernel} ||
                vc->shape != std::vector<std::int64_t>{width, 1, conv_kernel} ||
                dt->shape != std::vector<std::int64_t>{width} ||
                a_log->shape != std::vector<std::int64_t>{heads} ||
                norm->shape != std::vector<std::int64_t>{value_dim} ||
                out->shape != std::vector<std::int64_t>{*m.hidden_size, width}) {
                inference_detail::fail(ResolutionFailureKind::ShapeConstraintViolation,
                    "recurrent linear-attention tensor shapes do not agree with geometry for layer " +
                    std::to_string(layer));
            }
            topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::GatedDeltaNet;
            topology.gated_delta_net_layouts[static_cast<size_t>(layer)] = GatedDeltaNetSpec{
                conv_kernel, key_dim, value_dim, heads, heads, true,
                m.recurrent_safe_decay.value_or(false),
                m.recurrent_decay_lower_bound.value_or(-5.0f), true, true};
            semantic_layer.mixer = topology.gated_delta_net_layouts[static_cast<size_t>(layer)];
            ++topology.gated_delta_net_layer_count;
            topology.execute_feed_forward[static_cast<size_t>(layer)] = has_ffn;
            continue;
        }
        if (has_mla) {
            const auto* q_a = input.inventory.find(model_layer_prefix + "attention.q_a_proj.weight");
            const auto* q_a_norm = input.inventory.find(model_layer_prefix + "attention.q_a_layernorm.weight");
            const auto* q_b = input.inventory.find(model_layer_prefix + "attention.q_b_proj.weight");
            const auto* kv_a = input.inventory.find(model_layer_prefix + "attention.kv_a_proj_with_mqa.weight");
            const auto* kv_a_norm = input.inventory.find(model_layer_prefix + "attention.kv_a_layernorm.weight");
            const auto* kv_b = input.inventory.find(model_layer_prefix + "attention.kv_b_proj.weight");
            const auto* out = input.inventory.find(model_layer_prefix + "attention.dense.weight");
            if (!out) {
                out = input.inventory.find(model_layer_prefix + "attention.o_proj.weight");
            }
            const auto* gate = input.inventory.find(model_layer_prefix + "attention.g_proj.weight");
            const int q_rank = m.latent_query_rank.value_or(0);
            const int kv_rank = m.latent_kv_rank.value_or(0);
            const int nope = m.latent_query_nope_dim.value_or(0);
            const int rope = m.latent_query_rope_dim.value_or(0);
            const int value_dim = m.latent_value_head_dim.value_or(0);
            const int heads = *m.query_heads.value_for(layer);
            const std::vector<std::int64_t> head_gate_shape = {heads, *m.hidden_size};
            const std::vector<std::int64_t> element_gate_shape = {
                heads * value_dim, *m.hidden_size};
            const bool head_wise_gate = gate && gate->shape == head_gate_shape;
            const bool element_wise_gate = gate && gate->shape == element_gate_shape;
            if (!q_a || !q_a_norm || !q_b || !kv_a || !kv_a_norm || !kv_b || !out || !gate ||
                q_rank <= 0 || kv_rank <= 0 || nope <= 0 || rope <= 0 || value_dim <= 0 ||
                q_a->shape != std::vector<std::int64_t>{q_rank, *m.hidden_size} ||
                q_a_norm->shape != std::vector<std::int64_t>{q_rank} ||
                q_b->shape != std::vector<std::int64_t>{heads * (nope + rope), q_rank} ||
                kv_a->shape != std::vector<std::int64_t>{kv_rank + rope, *m.hidden_size} ||
                kv_a_norm->shape != std::vector<std::int64_t>{kv_rank} ||
                kv_b->shape != std::vector<std::int64_t>{heads * (nope + value_dim), kv_rank} ||
                out->shape != std::vector<std::int64_t>{*m.hidden_size, heads * value_dim} ||
                (!head_wise_gate && !element_wise_gate)) {
                inference_detail::fail(ResolutionFailureKind::ShapeConstraintViolation,
                    "factorized latent-attention tensor shapes do not agree with geometry for layer " +
                    std::to_string(layer));
            }
            AttentionSpec& attention = topology.attention_layouts[static_cast<size_t>(layer)];
            attention = make_attention(heads, 1, value_dim, false);
            attention.query_heads = heads;
            attention.key_value_heads = 1;
            attention.head_dim = value_dim;
            attention.query_norm = {0.0f, NormWeightKind::None};
            attention.key_norm = {0.0f, NormWeightKind::None};
            attention.output_gate = {AttentionGateKind::Sigmoid, false,
                head_wise_gate ? AttentionGateGranularity::HeadWise
                               : AttentionGateGranularity::ElementWise};
            attention.state = LatentAttentionStateSpec{
                kv_rank, rope, nope, true, true, q_rank, value_dim,
                NormSpec{*m.norm_epsilon, NormWeightKind::Scale},
                NormSpec{*m.norm_epsilon, NormWeightKind::Scale}};
            // The latent attention score combines the absorbed content part
            // with the decoupled rotary part, so its softmax scale is based on
            // qk_head_dim rather than the value head width used by the
            // storage/output layout.
            attention.query_scale = std::sqrt(static_cast<float>(value_dim) /
                                              static_cast<float>(nope + rope));
            semantic_layer.mixer = attention;
            topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::Attention;
            topology.attention_slot_for_layer[static_cast<size_t>(layer)] =
                topology.attention_layer_count++;
            topology.layer_for_attention_slot.push_back(layer);
            topology.execute_feed_forward[static_cast<size_t>(layer)] = has_ffn;
            continue;
        }
        if (has_mamba) {
            const auto* input_projection = find_mamba_tensor("in_proj.weight");
            const auto* convolution = find_mamba_tensor("conv1d.weight");
            const auto* convolution_bias = find_mamba_tensor("conv1d.bias");
            const auto* dt_bias = find_mamba_tensor("dt_bias");
            const auto* a_log = find_mamba_tensor("A_log");
            const auto* d = find_mamba_tensor("D");
            const auto* norm = find_mamba_tensor("norm.weight");
            const auto* output_projection = find_mamba_tensor("out_proj.weight");
            if (!input_projection || !convolution || !convolution_bias || !dt_bias ||
                !a_log || !d || !norm || !output_projection) {
                inference_detail::fail(ResolutionFailureKind::MissingTensorRole,
                                       "Mamba-2 tensor grammar is incomplete for layer " +
                                           std::to_string(layer));
            }
            const int inner = m.mamba_intermediate.value_or(
                static_cast<int>(output_projection->shape.at(1)));
            const int heads = m.mamba_num_heads.value_or(
                static_cast<int>(dt_bias->shape.at(0)));
            const int head_dim = m.mamba_head_dim.value_or(
                heads > 0 && inner % heads == 0 ? inner / heads : 0);
            const int state_size = m.mamba_state_size.value_or(0);
            const int group_count = m.mamba_group_count.value_or(0);
            const int conv_kernel = m.mamba_conv_kernel.value_or(
                convolution->shape.size() == 3 ? static_cast<int>(convolution->shape.at(2)) : 0);
            const int time_step_rank = m.mamba_time_step_rank.value_or(heads);
            const int chunk_size = m.mamba_chunk_size.value_or(0);
            const int conv_dim = inner + 2 * group_count * state_size;
            if (inner <= 0 || heads <= 0 || head_dim <= 0 || state_size <= 0 ||
                group_count <= 0 || heads % group_count != 0 || conv_kernel <= 0 ||
                conv_dim <= 0 || input_projection->shape !=
                    std::vector<std::int64_t>{2 * inner + 2 * group_count * state_size + heads,
                                               *m.hidden_size} ||
                convolution->shape != std::vector<std::int64_t>{conv_dim, 1, conv_kernel} ||
                convolution_bias->shape != std::vector<std::int64_t>{conv_dim} ||
                dt_bias->shape != std::vector<std::int64_t>{heads} ||
                a_log->shape != std::vector<std::int64_t>{heads} ||
                d->shape != std::vector<std::int64_t>{heads} ||
                norm->shape != std::vector<std::int64_t>{inner} ||
                output_projection->shape != std::vector<std::int64_t>{*m.hidden_size, inner}) {
                inference_detail::fail(ResolutionFailureKind::ShapeConstraintViolation,
                                       "Mamba-2 tensor shapes do not agree with recurrent geometry for layer " +
                                           std::to_string(layer));
            }
            topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::Mamba2;
            topology.mamba2_layouts[static_cast<size_t>(layer)] =
                Mamba2Spec{conv_kernel, inner, state_size, time_step_rank, heads, head_dim,
                           group_count, chunk_size, true, false};
            semantic_layer.mixer = topology.mamba2_layouts[static_cast<size_t>(layer)];
            ++topology.mamba2_layer_count;
            topology.mamba2_intermediate = std::max(topology.mamba2_intermediate, inner);
            topology.execute_feed_forward[static_cast<size_t>(layer)] = false;
            continue;
        }
        if (!has_attention) {
            if (has_shortconv) {
                if (!m.shortconv_cache.has_value() || *m.shortconv_cache <= 0) {
                    inference_detail::fail(
                        ResolutionFailureKind::MissingRequiredMetadata,
                        "short-convolution layer has no positive cache length");
                }
                topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::ShortConvolution;
                semantic_layer.mixer = ShortConvolutionSpec{*m.shortconv_cache, *m.hidden_size, false};
                ++topology.conv_layer_count;
                topology.execute_feed_forward[static_cast<size_t>(layer)] = has_ffn;
                continue;
            }
            if (!has_ffn) {
                inference_detail::fail(ResolutionFailureKind::UnsupportedGraphPrimitive,
                                       "layer has no recognized mixer or FFN tensor grammar: " +
                                           std::to_string(layer));
            }
            topology.mixer_kinds[static_cast<size_t>(layer)] = MixerKind::MlpOnly;
            topology.mlp_only_layouts[static_cast<size_t>(layer)] =
                MlpBlockSpec{intermediate_sizes.at(static_cast<size_t>(layer)),
                             ActivationKind::Relu2};
            semantic_layer.mixer = topology.mlp_only_layouts[static_cast<size_t>(layer)];
            ++topology.mlp_only_layer_count;
            topology.execute_feed_forward[static_cast<size_t>(layer)] = false;
            continue;
        }
        if (!query_heads.has_value() || !key_value_heads.has_value()) {
            inference_detail::fail(
                ResolutionFailureKind::MissingRequiredMetadata,
                "automatic resolution requires attention geometry for layer " +
                    std::to_string(layer));
        }
        if (*query_heads <= 0) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingInferenceFacts,
                "query_heads is non-positive for layer " + std::to_string(layer));
        }
        const int head_dim = explicit_head_dim.value_or(*m.hidden_size / *query_heads);
        if (*key_value_heads <= 0 || *query_heads % *key_value_heads != 0 ||
            head_dim <= 0 || head_dim % 2 != 0 ||
            *key_value_heads * head_dim > *m.hidden_size) {
            inference_detail::fail(
                ResolutionFailureKind::ConflictingInferenceFacts,
                "attention head geometry is not a valid GQA layout for layer " +
                    std::to_string(layer));
        }
        const bool has_query_norm =
            *m.query_key_norm || has_tensor(layer_prefix + "attn_q_norm.weight") ||
            has_tensor("model.layers." + std::to_string(layer) +
                       ".self_attn.q_layernorm.weight");
        const bool has_key_norm =
            *m.query_key_norm || has_tensor(layer_prefix + "attn_k_norm.weight") ||
            has_tensor("model.layers." + std::to_string(layer) +
                       ".self_attn.k_layernorm.weight");
        if (has_query_norm != has_key_norm) {
            inference_detail::fail(ResolutionFailureKind::ConflictingInferenceFacts,
                                   "query/key normalization evidence is incomplete for layer " +
                                       std::to_string(layer));
        }
        topology.attention_slot_for_layer[static_cast<size_t>(layer)] =
            topology.attention_layer_count++;
        topology.layer_for_attention_slot.push_back(layer);
        topology.attention_layouts[static_cast<size_t>(layer)] =
            make_attention(*query_heads, *key_value_heads, head_dim, has_query_norm);
        semantic_layer.mixer = topology.attention_layouts[static_cast<size_t>(layer)];
        topology.execute_feed_forward[static_cast<size_t>(layer)] = has_ffn;
    }
    if (m.attention_multiplier.has_value()) {
        numerical_policy.attention_multiplier = *m.attention_multiplier;
    } else {
        int attention_head_dim = 0;
        for (int layer = 0; layer < *m.layer_count; ++layer) {
            if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Attention) {
                attention_head_dim = topology.attention_layouts[static_cast<size_t>(layer)].head_dim;
                break;
            }
        }
    numerical_policy.attention_multiplier = attention_head_dim > 0
            ? 1.0f / std::sqrt(static_cast<float>(attention_head_dim)) : 1.0f;
    }
    for (LayerSpec& semantic_layer : graph.layers) {
        if (auto* attention = std::get_if<AttentionSpec>(&semantic_layer.mixer);
            attention != nullptr && attention->query_heads > 0) {
            attention->query_scale *= numerical_policy.attention_multiplier;
        }
    }
    topology.validate();

    const auto add_global = [&](TensorRole role, const TensorInventoryEntry& tensor) {
        inference_detail::add_binding(
            facts.bindings, role, -1, tensor,
            {{EvidenceKind::TensorName, tensor.name, std::string(tensor_role_name(role))}});
    };
    add_global(TensorRole::TokenEmbedding, *embedding);

    const std::vector<std::string> head_names = {
        "lm_head.weight", "transformer.lm_head.weight", "output.weight"};
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
        if (input.source_format == CheckpointSourceFormat::Gguf &&
            !m.tied_embeddings.has_value()) {
            facts.tied_embeddings = true;
            facts.evidence.push_back({EvidenceKind::FormatGuarantee, "output.weight",
                                      "GGUF omits an independent language-model head"});
        } else if (!facts.tied_embeddings) {
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
        "transformer.ln_f.weight", "model.norm.weight", "norm.weight", "output_norm.weight",
        "model.language_model.norm.weight", "backbone.norm_f.weight",
        "model.embedding_norm.weight",
        "token_embd_norm.weight"};
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
        const std::string layer_prefix = "blk." + index + ".";
        const auto has_tensor = [&](std::string_view name) {
            return input.inventory.find(name) != nullptr;
        };
        const int query_head_count = *m.query_heads.value_for(layer);
        const int key_value_head_count = *m.key_value_heads.value_for(layer);
        const int layer_head_dim = m.head_dim.value_for(layer).value_or(
            *m.hidden_size / query_head_count);
        const int layer_intermediate = intermediate_sizes.at(static_cast<size_t>(layer));
        const std::vector<std::string> norm_candidates = {
            "transformer.h." + index + ".ln_1.weight",
            "model.layers." + index + ".input_layernorm.weight",
            "model.layers." + index + ".self_attn_layer_norm.weight",
            "model.language_model.layers." + index + ".input_layernorm.weight",
            "model.layers." + index + ".operator_norm.weight",
            "backbone.layers." + index + ".norm.weight",
            "blk." + index + ".attn_norm.weight",
        };
        const std::vector<std::string> ffn_norm_candidates = {
            "transformer.h." + index + ".ln_2.weight",
            "model.layers." + index + ".post_attention_layernorm.weight",
            "model.language_model.layers." + index + ".post_attention_layernorm.weight",
            "model.layers." + index + ".ffn_norm.weight",
            "blk." + index + ".ffn_norm.weight",
        };
        const auto* attention_norm = inference_detail::find_unique(
            input.inventory, norm_candidates, TensorRole::AttentionInputNorm, layer,
            {*m.hidden_size}, {});
        inference_detail::add_binding(facts.bindings, TensorRole::AttentionInputNorm, layer,
                                      *attention_norm, {});
        if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Mamba2) {
            const auto bind_mamba = [&](TensorRole role, std::string_view suffix,
                                        std::initializer_list<std::int64_t> shape) {
                const auto* tensor = inference_detail::find_unique(
                    input.inventory,
                    inference_detail::mamba2_tensor_candidates(layer, suffix),
                    role, layer, shape, {});
                inference_detail::add_binding(facts.bindings, role, layer, *tensor, {});
            };
            const Mamba2Spec& spec = topology.mamba2_layouts.at(static_cast<size_t>(layer));
            const int conv_dim = spec.intermediate_size +
                2 * spec.group_count * spec.state_size;
            bind_mamba(TensorRole::Mamba2Input, "in_proj.weight",
                       {2 * spec.intermediate_size + 2 * spec.group_count * spec.state_size +
                        spec.num_heads, *m.hidden_size});
            bind_mamba(TensorRole::Mamba2Conv, "conv1d.weight",
                       {conv_dim, 1, spec.conv_kernel});
            bind_mamba(TensorRole::Mamba2ConvBias, "conv1d.bias", {conv_dim});
            bind_mamba(TensorRole::Mamba2DtBias, "dt_bias", {spec.num_heads});
            bind_mamba(TensorRole::Mamba2ALog, "A_log", {spec.num_heads});
            bind_mamba(TensorRole::Mamba2D, "D", {spec.num_heads});
            bind_mamba(TensorRole::Mamba2Norm, "norm.weight", {spec.intermediate_size});
            bind_mamba(TensorRole::Mamba2Output, "out_proj.weight",
                       {*m.hidden_size, spec.intermediate_size});
        } else if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::GatedDeltaNet &&
                   topology.gated_delta_net_layouts.at(static_cast<size_t>(layer)).factorized_projections) {
            const GatedDeltaNetSpec& spec = topology.gated_delta_net_layouts.at(
                static_cast<size_t>(layer));
            const std::string prefix = "model.layers." + index + ".attention.";
            const auto bind = [&](TensorRole role, std::string_view suffix,
                                  std::initializer_list<std::int64_t> shape) {
                const auto* tensor = inference_detail::find_unique(
                    input.inventory, {prefix + std::string(suffix)}, role, layer, shape, {});
                inference_detail::add_binding(facts.bindings, role, layer, *tensor, {});
            };
            bind(TensorRole::GatedDeltaNetQuery, "q_proj.weight",
                 {spec.key_heads * spec.key_head_dim, *m.hidden_size});
            bind(TensorRole::GatedDeltaNetKey, "k_proj.weight",
                 {spec.key_heads * spec.key_head_dim, *m.hidden_size});
            bind(TensorRole::GatedDeltaNetValue, "v_proj.weight",
                 {spec.value_width(), *m.hidden_size});
            bind(TensorRole::GatedDeltaNetDecay, "f_proj.weight",
                 {spec.decay_width(), *m.hidden_size});
            bind(TensorRole::GatedDeltaNetOutputGate, "g_proj.weight",
                 {spec.value_width(), *m.hidden_size});
            bind(TensorRole::GatedDeltaNetQueryConv, "q_conv1d.weight",
                 {spec.key_heads * spec.key_head_dim, 1, spec.conv_kernel});
            bind(TensorRole::GatedDeltaNetKeyConv, "k_conv1d.weight",
                 {spec.key_heads * spec.key_head_dim, 1, spec.conv_kernel});
            bind(TensorRole::GatedDeltaNetValueConv, "v_conv1d.weight",
                 {spec.value_width(), 1, spec.conv_kernel});
            bind(TensorRole::GatedDeltaNetBeta, "b_proj.weight",
                 {spec.value_heads, *m.hidden_size});
            bind(TensorRole::GatedDeltaNetDtBias, "dt_bias", {spec.decay_width()});
            bind(TensorRole::GatedDeltaNetALog, "A_log", {spec.value_heads});
            bind(TensorRole::GatedDeltaNetNorm, "o_norm.weight", {spec.value_head_dim});
            bind(TensorRole::GatedDeltaNetOutput, "o_proj.weight",
                 {*m.hidden_size, spec.value_width()});
        } else if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::ShortConvolution) {
            const auto* input_projection = inference_detail::find_unique(
                input.inventory,
                inference_detail::shortconv_tensor_candidates(layer, "in_proj.weight"),
                TensorRole::ShortConvInput, layer,
                {3 * *m.hidden_size, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::ShortConvInput,
                                          layer, *input_projection, {});
            const auto* kernel = inference_detail::find_unique(
                input.inventory,
                inference_detail::shortconv_tensor_candidates(layer, "conv.weight"),
                TensorRole::ShortConvKernel, layer,
                {*m.hidden_size, 1, topology.conv_cache}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::ShortConvKernel,
                                          layer, *kernel, {});
            const auto* output_projection = inference_detail::find_unique(
                input.inventory,
                inference_detail::shortconv_tensor_candidates(layer, "out_proj.weight"),
                TensorRole::ShortConvOutput, layer,
                {*m.hidden_size, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::ShortConvOutput,
                                          layer, *output_projection, {});
        } else if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Attention) {
            const AttentionSpec& attention = topology.attention_layouts.at(
                static_cast<size_t>(layer));
            if (attention.uses_latent_state() && attention.latent_state()->factorized) {
                const auto& latent = *attention.latent_state();
                const std::string prefix = "model.layers." + index + ".attention.";
                const auto bind = [&](TensorRole role, std::string suffix,
                                      std::initializer_list<std::int64_t> shape) {
                    const auto* tensor = inference_detail::find_unique(
                        input.inventory, {prefix + std::move(suffix)}, role, layer, shape, {});
                    inference_detail::add_binding(facts.bindings, role, layer, *tensor, {});
                };
                bind(TensorRole::AttentionLatentQueryProjection, "q_a_proj.weight",
                     {latent.query_rank, *m.hidden_size});
                bind(TensorRole::AttentionLatentQueryNorm, "q_a_layernorm.weight",
                     {latent.query_rank});
                bind(TensorRole::AttentionLatentQueryExpansion, "q_b_proj.weight",
                     {query_head_count * (latent.nope_head_dim + latent.rope_head_dim),
                      latent.query_rank});
                bind(TensorRole::AttentionLatentKeyProjection, "kv_a_proj_with_mqa.weight",
                     {latent.latent_rank + latent.rope_head_dim, *m.hidden_size});
                bind(TensorRole::AttentionLatentKeyNorm, "kv_a_layernorm.weight",
                     {latent.latent_rank});
                bind(TensorRole::AttentionLatentExpansion, "kv_b_proj.weight",
                     {query_head_count * (latent.nope_head_dim + latent.value_head_dim),
                      latent.latent_rank});
                const auto* output = inference_detail::find_unique(
                    input.inventory,
                    {prefix + "dense.weight", prefix + "o_proj.weight"},
                    TensorRole::AttentionLatentOutput, layer,
                    {*m.hidden_size, attention.latent_output_width()}, {});
                inference_detail::add_binding(facts.bindings,
                                              TensorRole::AttentionLatentOutput,
                                              layer, *output, {});
                bind(TensorRole::AttentionGate, "g_proj.weight",
                     {attention.output_gate_width(), *m.hidden_size});
            } else {
            const auto* q = inference_detail::find_unique(
                input.inventory,
                inference_detail::attention_tensor_candidates(layer, "q_proj.weight"),
                TensorRole::AttentionQuery, layer,
                {query_head_count * layer_head_dim, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::AttentionQuery,
                                          layer, *q, {});
            const auto* k = inference_detail::find_unique(
                input.inventory,
                inference_detail::attention_tensor_candidates(layer, "k_proj.weight"),
                TensorRole::AttentionKey, layer,
                {key_value_head_count * layer_head_dim, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::AttentionKey,
                                          layer, *k, {});
            const auto* v = inference_detail::find_unique(
                input.inventory,
                inference_detail::attention_tensor_candidates(layer, "v_proj.weight"),
                TensorRole::AttentionValue, layer,
                {key_value_head_count * layer_head_dim, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::AttentionValue,
                                          layer, *v, {});
            const auto* o = inference_detail::find_unique(
                input.inventory,
                inference_detail::attention_tensor_candidates(layer, "o_proj.weight"),
                TensorRole::AttentionOutput, layer,
                {*m.hidden_size, query_head_count * layer_head_dim}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::AttentionOutput,
                                          layer, *o, {});
            if (has_tensor(layer_prefix + "attn_q_norm.weight")) {
                const auto* q_norm = inference_detail::find_unique(
                    input.inventory,
                    {layer_prefix + "attn_q_norm.weight"},
                    TensorRole::AttentionQueryNorm, layer,
                    {layer_head_dim}, {});
                inference_detail::add_binding(facts.bindings, TensorRole::AttentionQueryNorm,
                                              layer, *q_norm, {});
            } else if (has_tensor("model.layers." + index + ".self_attn.q_layernorm.weight")) {
                const auto* q_norm = inference_detail::find_unique(
                    input.inventory,
                    {"model.layers." + index + ".self_attn.q_layernorm.weight"},
                    TensorRole::AttentionQueryNorm, layer,
                    {layer_head_dim}, {});
                inference_detail::add_binding(facts.bindings, TensorRole::AttentionQueryNorm,
                                              layer, *q_norm, {});
            }
            if (has_tensor(layer_prefix + "attn_k_norm.weight")) {
                const auto* k_norm = inference_detail::find_unique(
                    input.inventory,
                    {layer_prefix + "attn_k_norm.weight"},
                    TensorRole::AttentionKeyNorm, layer,
                    {layer_head_dim}, {});
                inference_detail::add_binding(facts.bindings, TensorRole::AttentionKeyNorm,
                                              layer, *k_norm, {});
            } else if (has_tensor("model.layers." + index + ".self_attn.k_layernorm.weight")) {
                const auto* k_norm = inference_detail::find_unique(
                    input.inventory,
                    {"model.layers." + index + ".self_attn.k_layernorm.weight"},
                    TensorRole::AttentionKeyNorm, layer,
                    {layer_head_dim}, {});
                inference_detail::add_binding(facts.bindings, TensorRole::AttentionKeyNorm,
                                              layer, *k_norm, {});
            }
            }
        }
        const MixerKind mixer = topology.mixer_kinds[static_cast<size_t>(layer)];
        if (mixer == MixerKind::MlpOnly) {
            const auto* up = inference_detail::find_unique(
                input.inventory, inference_detail::feed_forward_tensor_candidates(layer,
                    "w_up.weight"), TensorRole::FfnUp, layer,
                {layer_intermediate, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::FfnUp, layer, *up, {});
            const auto* down = inference_detail::find_unique(
                input.inventory, inference_detail::feed_forward_tensor_candidates(layer,
                    "w_down.weight"), TensorRole::FfnDown, layer,
                {*m.hidden_size, layer_intermediate}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::FfnDown, layer, *down, {});
        } else if (topology.execute_feed_forward.at(static_cast<size_t>(layer))) {
            const auto* ffn_norm = inference_detail::find_unique(
                input.inventory, ffn_norm_candidates, TensorRole::FfnInputNorm, layer,
                {*m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::FfnInputNorm, layer,
                                          *ffn_norm, {});
            if (topology.feed_forward_kinds[static_cast<size_t>(layer)] ==
                FeedForwardKind::MixtureOfExperts) {
                const std::string prefix = "model.layers." + index + ".mlp.";
                const auto bind = [&](TensorRole role, std::string name,
                                      std::initializer_list<std::int64_t> shape,
                                      int expert = -1) {
                    const auto* tensor = inference_detail::find_unique(
                        input.inventory, {std::move(name)}, role, layer, shape, {});
                    facts.bindings.values.push_back({role, layer, expert, -1,
                                                     tensor->name, tensor->shape, {}});
                };
                bind(TensorRole::MoeRouter, prefix + "gate.weight",
                     {*m.moe_experts, *m.hidden_size});
                const std::string bias_name = prefix + "gate.expert_bias";
                if (has_tensor(bias_name)) {
                    bind(TensorRole::MoeRouterBias, bias_name, {*m.moe_experts});
                }
                const int expert_intermediate = m.moe_intermediate.value_or(layer_intermediate);
                for (int expert = 0; expert < *m.moe_experts; ++expert) {
                    const std::string expert_prefix = prefix + "experts." +
                        std::to_string(expert) + ".";
                    bind(TensorRole::MoeExpertGate, expert_prefix + "gate_proj.weight",
                         {expert_intermediate, *m.hidden_size}, expert);
                    bind(TensorRole::MoeExpertUp, expert_prefix + "up_proj.weight",
                         {expert_intermediate, *m.hidden_size}, expert);
                    bind(TensorRole::MoeExpertDown, expert_prefix + "down_proj.weight",
                         {*m.hidden_size, expert_intermediate}, expert);
                }
                if (topology.shared_expert_intermediate > 0) {
                    const std::string shared = prefix + "shared_experts.";
                    bind(TensorRole::MoeSharedGate, shared + "gate_proj.weight",
                         {topology.shared_expert_intermediate, *m.hidden_size});
                    bind(TensorRole::MoeSharedUp, shared + "up_proj.weight",
                         {topology.shared_expert_intermediate, *m.hidden_size});
                    bind(TensorRole::MoeSharedDown, shared + "down_proj.weight",
                         {*m.hidden_size, topology.shared_expert_intermediate});
                }
            } else {
            const auto* gate = inference_detail::find_unique(
                input.inventory, inference_detail::feed_forward_tensor_candidates(layer,
                    "w_gate.weight"), TensorRole::FfnGate, layer,
                {layer_intermediate, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::FfnGate, layer, *gate, {});
            const auto* up = inference_detail::find_unique(
                input.inventory, inference_detail::feed_forward_tensor_candidates(layer,
                    "w_up.weight"), TensorRole::FfnUp, layer,
                {layer_intermediate, *m.hidden_size}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::FfnUp, layer, *up, {});
            const auto* down = inference_detail::find_unique(
                input.inventory, inference_detail::feed_forward_tensor_candidates(layer,
                    "w_down.weight"), TensorRole::FfnDown, layer,
                {*m.hidden_size, layer_intermediate}, {});
            inference_detail::add_binding(facts.bindings, TensorRole::FfnDown, layer, *down, {});
            }
        }
    }
    facts.bindings = BindingSolver{}.solve(facts.bindings.values);
    facts.validate();
    return facts;
}

std::string CanonicalModelFacts::fingerprint() const {
    std::ostringstream out;
    out << resolution_mode << ':' << source_format << ':' << topology.fingerprint()
        << ":num=" << numerical_policy.norm_eps << ':'
        << numerical_policy.post_norm_eps << ':'
        << numerical_policy.embedding_multiplier << ':'
        << numerical_policy.attention_multiplier << ':'
        << numerical_policy.residual_multiplier << ':'
        << numerical_policy.logits_multiplier << ':'
        << numerical_policy.logits_divisor << ':'
        << numerical_policy.final_logit_softcap
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
    numerical_policy.validate();
    graph.validate();
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
        if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Attention) {
            const AttentionSpec& attention = topology.attention_layouts.at(
                static_cast<size_t>(layer));
            if (attention.uses_latent_state()) {
                if (attention.latent_state()->factorized) {
                    require(TensorRole::AttentionLatentQueryProjection, layer);
                    require(TensorRole::AttentionLatentQueryExpansion, layer);
                    require(TensorRole::AttentionLatentQueryNorm, layer);
                    require(TensorRole::AttentionLatentKeyProjection, layer);
                    require(TensorRole::AttentionLatentKeyNorm, layer);
                    require(TensorRole::AttentionLatentExpansion, layer);
                } else {
                    require(TensorRole::AttentionLatentQuery, layer);
                    require(TensorRole::AttentionLatentKey, layer);
                    require(TensorRole::AttentionLatentValue, layer);
                }
            } else {
                require(TensorRole::AttentionQuery, layer);
                require(TensorRole::AttentionKey, layer);
                require(TensorRole::AttentionValue, layer);
            }
            require(attention.uses_latent_state()
                        ? TensorRole::AttentionLatentOutput
                        : TensorRole::AttentionOutput,
                    layer);
        } else if (topology.mixer_kinds[static_cast<size_t>(layer)] ==
                   MixerKind::ShortConvolution) {
            require(TensorRole::ShortConvInput, layer);
            require(TensorRole::ShortConvKernel, layer);
            require(TensorRole::ShortConvOutput, layer);
        } else {
            if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::Mamba2) {
                require(TensorRole::Mamba2Input, layer);
                require(TensorRole::Mamba2Conv, layer);
                require(TensorRole::Mamba2ConvBias, layer);
                require(TensorRole::Mamba2DtBias, layer);
                require(TensorRole::Mamba2ALog, layer);
                require(TensorRole::Mamba2D, layer);
                require(TensorRole::Mamba2Norm, layer);
                require(TensorRole::Mamba2Output, layer);
            } else if (topology.mixer_kinds[static_cast<size_t>(layer)] ==
                       MixerKind::GatedDeltaNet) {
                const GatedDeltaNetSpec& spec = topology.gated_delta_net_layouts.at(
                    static_cast<size_t>(layer));
                if (spec.factorized_projections) {
                    require(TensorRole::GatedDeltaNetQuery, layer);
                    require(TensorRole::GatedDeltaNetKey, layer);
                    require(TensorRole::GatedDeltaNetValue, layer);
                    require(TensorRole::GatedDeltaNetDecay, layer);
                    require(TensorRole::GatedDeltaNetOutputGate, layer);
                    require(TensorRole::GatedDeltaNetQueryConv, layer);
                    require(TensorRole::GatedDeltaNetKeyConv, layer);
                    require(TensorRole::GatedDeltaNetValueConv, layer);
                } else {
                    require(TensorRole::GatedDeltaNetQkv, layer);
                    require(TensorRole::GatedDeltaNetZ, layer);
                    require(TensorRole::GatedDeltaNetAlpha, layer);
                }
                require(TensorRole::GatedDeltaNetBeta, layer);
                require(TensorRole::GatedDeltaNetDtBias, layer);
                require(TensorRole::GatedDeltaNetALog, layer);
                require(TensorRole::GatedDeltaNetNorm, layer);
                require(TensorRole::GatedDeltaNetOutput, layer);
            } else if (topology.mixer_kinds[static_cast<size_t>(layer)] == MixerKind::MlpOnly) {
                require(TensorRole::FfnUp, layer);
                require(TensorRole::FfnDown, layer);
            } else {
                inference_detail::fail(ResolutionFailureKind::UnsupportedGraphPrimitive,
                                       "automatic resolution has no binding contract for mixer");
            }
        }
        if (topology.mixer_kinds[static_cast<size_t>(layer)] != MixerKind::MlpOnly &&
            (topology.execute_feed_forward.empty() ||
             topology.execute_feed_forward.at(static_cast<size_t>(layer)))) {
            require(TensorRole::FfnInputNorm, layer);
            if (topology.feed_forward_kinds[static_cast<size_t>(layer)] ==
                FeedForwardKind::MixtureOfExperts) {
                require(TensorRole::MoeRouter, layer);
                for (int expert = 0; expert < topology.num_experts; ++expert) {
                    if (bindings.find(TensorRole::MoeExpertGate, layer, expert) == nullptr ||
                        bindings.find(TensorRole::MoeExpertUp, layer, expert) == nullptr ||
                        bindings.find(TensorRole::MoeExpertDown, layer, expert) == nullptr) {
                        inference_detail::fail(ResolutionFailureKind::MissingTensorRole,
                            "canonical facts omit an MoE expert tensor role");
                    }
                }
            } else {
                require(TensorRole::FfnGate, layer);
                require(TensorRole::FfnUp, layer);
                require(TensorRole::FfnDown, layer);
            }
        }
    }
}

} // namespace celeg
