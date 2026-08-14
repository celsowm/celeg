#include "canonical_internal.hpp"
#include "support.hpp"

#include <regex>
#include <unordered_set>

namespace celeg::inference_detail {
namespace {

void require_positive(const std::optional<int>& value, std::string_view name) {
    if (!value.has_value()) {
        fail(
            ResolutionFailureKind::MissingRequiredMetadata,
            "automatic resolution requires metadata: " + std::string(name));
    }
    if (*value <= 0) {
        fail(
            ResolutionFailureKind::ConflictingMetadata,
            "automatic resolution received a non-positive " + std::string(name));
    }
}

const TensorInventoryEntry* find_embedding(const InferenceInput& input) {
    const auto& m = input.metadata;
    const TensorInventoryEntry* embedding = nullptr;
    for (const std::string& name : {
             std::string("transformer.wte.weight"),
             std::string("model.embed_tokens.weight"),
             std::string("model.language_model.embed_tokens.weight"),
             std::string("tok_embeddings.weight"),
             std::string("token_embd.weight"),
             std::string("backbone.embeddings.weight"),
             std::string("model.word_embeddings.weight")}) {
        if (const auto* candidate = input.inventory.find(name)) {
            if (embedding != nullptr) {
                fail(
                    ResolutionFailureKind::AmbiguousTensorBinding,
                    "multiple token-embedding candidates are present");
            }
            embedding = candidate;
        }
    }

    if (embedding == nullptr) {
        fail(
            ResolutionFailureKind::MissingTensorRole,
            "automatic resolution could not find token embedding");
    }
    if (!shape_is(*embedding, {*m.vocab_size, *m.hidden_size})) {
        fail(
            ResolutionFailureKind::ShapeConstraintViolation,
            "token embedding shape does not agree with normalized metadata");
    }
    return embedding;
}

void validate_layer_inventory(const InferenceInput& input) {
    const auto& m = input.metadata;
    std::unordered_set<int> layers;
    const std::regex layer_pattern(
        R"((?:transformer\.h|model\.language_model\.layers|model\.layers|backbone\.layers|layers|blk)\.(\d+)\.)");
    for (const auto& entry : input.inventory.entries()) {
        std::smatch match;
        if (std::regex_search(entry.name, match, layer_pattern)) {
            layers.insert(std::stoi(match[1].str()));
        }
    }

    if (layers.size() != static_cast<size_t>(*m.layer_count)) {
        fail(
            ResolutionFailureKind::IncompleteLayerSchedule,
            "tensor inventory layer count does not agree with metadata");
    }
    for (int layer = 0; layer < *m.layer_count; ++layer) {
        if (!layers.contains(layer)) {
            fail(
                ResolutionFailureKind::IncompleteLayerSchedule,
                "tensor inventory has a gap at layer " + std::to_string(layer));
        }
    }
}

std::vector<int> infer_intermediate_sizes(
    const CanonicalInferenceContext& context) {
    const auto& input = context.input;
    const auto& m = input.metadata;
    std::vector<int> result;
    result.reserve(static_cast<size_t>(context.layer_count));

    for (int layer = 0; layer < context.layer_count; ++layer) {
        const std::optional<int> value = m.intermediate_size.value_for(layer);
        const std::string index = std::to_string(layer);

        const TensorInventoryEntry* ffn_up =
            input.inventory.find("blk." + index + ".ffn_up.weight");
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
        if (!ffn_up && context.has_moe && layer >= context.dense_start) {
            ffn_up = input.inventory.find(
                "model.layers." + index + ".mlp.experts.0.gate_proj.weight");
        }
        if (!ffn_up && context.has_moe && layer >= context.dense_start) {
            ffn_up = input.inventory.find(
                "model.layers." + index +
                ".mlp.experts.0.gate_proj.weight_packed");
        }

        const bool tensor_defines_intermediate =
            ffn_up && ffn_up->shape.size() == 2 && ffn_up->shape[0] > 0 &&
            m.feed_forward_auto_adjust.value_or(false);
        if (!value.has_value() || *value <= 0 || tensor_defines_intermediate) {
            if (ffn_up && ffn_up->shape.size() == 2 && ffn_up->shape[0] > 0) {
                result.push_back(static_cast<int>(ffn_up->shape[0]));
            } else {
                result.push_back(1);
            }
        } else {
            result.push_back(*value);
        }
    }
    return result;
}

void initialize_graph(CanonicalInferenceContext& context) {
    const auto& m = context.input.metadata;
    auto& graph = context.facts.graph;
    const auto& numerical_policy = context.facts.numerical_policy;

    graph.hidden = *m.hidden_size;
    graph.final_norm = {
        numerical_policy.norm_eps,
        NormWeightKind::Scale};
    graph.embedding_transform.multiplier =
        numerical_policy.embedding_multiplier;
    graph.logits_multiplier = numerical_policy.logits_multiplier;
    graph.logits_divisor = numerical_policy.logits_divisor;
    graph.layers.resize(static_cast<size_t>(context.layer_count));

    for (int layer = 0; layer < context.layer_count; ++layer) {
        LayerSpec& semantic_layer =
            graph.layers[static_cast<size_t>(layer)];
        semantic_layer.operator_norm = {
            numerical_policy.norm_eps,
            NormWeightKind::Scale};
        semantic_layer.feed_forward_norm = {
            numerical_policy.norm_eps,
            NormWeightKind::Scale};
        semantic_layer.residual.multiplier =
            numerical_policy.residual_multiplier;
        semantic_layer.mixer = AttentionSpec{};
        semantic_layer.execute_feed_forward = true;

        if (context.has_moe && layer >= context.dense_start) {
            semantic_layer.feed_forward = MixtureOfExpertsSpec{
                .intermediate_size = context.moe_intermediate,
                .num_experts = context.num_experts,
                .experts_per_token = context.experts_per_token,
                .normalize_topk = context.normalize_topk,
                .use_expert_bias = context.use_expert_bias,
                .routed_scaling_factor = context.routed_scaling_factor,
                .routing_group_count = context.total_routing_groups,
                .routing_experts_per_group = context.experts_per_group,
                .routing_groups_per_token = context.groups_per_token,
                .routing_group_score_top_k = context.group_score_top_k,
                .has_shared_expert =
                    context.shared_expert_intermediate > 0,
                .shared_intermediate_size =
                    context.shared_expert_intermediate,
                .router_softmax = context.moe_router_softmax};
        } else {
            semantic_layer.feed_forward = DenseFeedForwardSpec{
                context.intermediate_sizes.at(
                    static_cast<size_t>(layer)),
                ActivationKind::SwiGLU};
        }
    }
}

} // namespace

CanonicalInferenceContext initialize_canonical_facts(
    const InferenceInput& input) {
    const auto& m = input.metadata;
    require_positive(m.hidden_size, "hidden_size");
    require_positive(m.layer_count, "layer_count");
    require_positive(m.vocab_size, "vocab_size");
    require_positive(m.context_length, "context_length");

    if (!m.bos_token_id.has_value() || !m.pad_token_id.has_value() ||
        *m.bos_token_id < 0 || *m.pad_token_id < 0 ||
        m.eos_token_ids.empty()) {
        fail(
            ResolutionFailureKind::ConflictingMetadata,
            "token IDs are incomplete");
    }

    CanonicalInferenceContext context(input);
    context.embedding = find_embedding(input);
    validate_layer_inventory(input);

    context.facts.resolution_mode = "automatic";
    context.facts.tied_embeddings = m.tied_embeddings.value_or(false);
    context.facts.evidence = m.evidence;

    context.layer_count = *m.layer_count;
    context.dense_start =
        m.first_dense_layer.value_or(context.layer_count);
    if (context.dense_start < 0 ||
        context.dense_start > context.layer_count) {
        fail(
            ResolutionFailureKind::ConflictingMetadata,
            "first dense layer is outside the layer schedule");
    }

    context.num_experts = m.moe_experts.value_or(0);
    context.experts_per_token =
        m.moe_experts_per_token.value_or(0);
    context.has_moe =
        context.num_experts > 0 &&
        context.experts_per_token > 0 &&
        context.dense_start < context.layer_count;
    context.moe_intermediate = m.moe_intermediate.value_or(0);
    context.shared_expert_intermediate =
        m.moe_shared_intermediate.value_or(0);
    context.normalize_topk =
        m.moe_normalize_topk.value_or(false);
    context.use_expert_bias =
        m.moe_expert_bias.value_or(false);
    context.routed_scaling_factor =
        m.moe_routed_scaling.value_or(1.0f);
    context.moe_router_softmax =
        m.moe_score_function.value_or("sigmoid") == "softmax";
    context.routing_groups = m.moe_routing_groups.value_or(0);
    context.total_routing_groups =
        m.moe_total_routing_groups.value_or(context.routing_groups);
    context.groups_per_token = context.routing_groups;
    context.group_score_top_k =
        m.moe_group_score_top_k.value_or(0);

    if (context.has_moe && context.total_routing_groups > 0) {
        if (context.num_experts % context.total_routing_groups != 0) {
            fail(
                ResolutionFailureKind::ConflictingMetadata,
                "MoE expert count is not divisible by routing group count");
        }
        context.experts_per_group =
            context.num_experts / context.total_routing_groups;
        if (context.groups_per_token <= 0 ||
            context.groups_per_token > context.total_routing_groups) {
            fail(
                ResolutionFailureKind::ConflictingMetadata,
                "invalid number of selected MoE routing groups");
        }
        if (context.group_score_top_k <= 0 ||
            context.group_score_top_k > context.experts_per_group) {
            fail(
                ResolutionFailureKind::ConflictingMetadata,
                "invalid MoE routing group score width");
        }
    }

    auto& checkpoint = context.facts.checkpoint;
    auto& numerical_policy = context.facts.numerical_policy;
    checkpoint.vocab_size = *m.vocab_size;
    checkpoint.max_position_embeddings = *m.context_length;
    checkpoint.token_policy = {
        *m.bos_token_id,
        m.eos_token_ids,
        *m.pad_token_id};

    numerical_policy.norm_eps = *m.norm_epsilon;
    numerical_policy.embedding_multiplier =
        m.embedding_multiplier.value_or(1.0f);
    numerical_policy.residual_multiplier =
        m.residual_multiplier.value_or(1.0f);
    numerical_policy.logits_multiplier =
        m.logits_multiplier.value_or(1.0f);
    numerical_policy.logits_divisor =
        m.logits_divisor.value_or(1.0f);

    context.intermediate_sizes =
        infer_intermediate_sizes(context);
    initialize_graph(context);
    return context;
}

} // namespace celeg::inference_detail
