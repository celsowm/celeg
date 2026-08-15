#include "celeg/model/program.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace celeg {
namespace {

constexpr float kPerLayerResidualScale = 0.7071067811865475f;

std::size_t checked_product(std::size_t left, std::size_t right,
                            const char* message) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::invalid_argument(message);
    }
    return left * right;
}

template <typename T>
void append_field(std::ostringstream& out, T value) {
    out << static_cast<int>(value) << ';';
}

void append_field(std::ostringstream& out, int value) { out << value << ';'; }
void append_field(std::ostringstream& out, std::size_t value) { out << value << ';'; }
void append_field(std::ostringstream& out, float value) { out << value << ';'; }
void append_field(std::ostringstream& out, bool value) { out << (value ? 1 : 0) << ';'; }

std::string fingerprint_text(const std::string& text) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : text) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
}

void validate_feed_forward(const CompiledFeedForwardProgram& program) {
    std::visit([](const auto& value) {
        using FeedForward = std::decay_t<decltype(value)>;
        if constexpr (!std::is_same_v<FeedForward, std::monostate>) {
            value.validate();
        }
    }, program);
}

void validate_attention_state_layout(const CompiledAttentionStateLayout& layout) {
    std::visit([](const auto& state) { state.validate(); }, layout);
}

} // namespace

void ExpertPayloadSchema::validate() const {
    for (const ExpertPayloadRegion& region : regions) {
        if (region.elements == 0) {
            throw std::invalid_argument("MoE payload region has no elements");
        }
    }
}

std::size_t ExpertPayloadSchema::elements() const {
    std::size_t total = 0;
    for (const ExpertPayloadRegion& region : regions) {
        if (region.elements > std::numeric_limits<std::size_t>::max() - total) {
            throw std::overflow_error("MoE payload element count overflows size_t");
        }
        total += region.elements;
    }
    return total;
}

std::string ExpertPayloadSchema::fingerprint() const {
    std::ostringstream out;
    append_field(out, layout);
    for (const auto& region : regions) {
        append_field(out, region.role);
        append_field(out, region.elements);
    }
    return fingerprint_text(out.str());
}

void RouterProgram::validate() const {
    if (expert_count <= 0 || experts_per_token <= 0 ||
        experts_per_token > expert_count || !std::isfinite(routed_scaling) ||
        routed_scaling == 0.0f) {
        throw std::invalid_argument("invalid compiled MoE router program");
    }
    std::visit([&](const auto& selection_program) {
        using Selection = std::decay_t<decltype(selection_program)>;
        if constexpr (std::is_same_v<Selection, MoeTopKSelectionSpec>) {
            return;
        } else if constexpr (std::is_same_v<Selection, MoeGroupedTopKSelectionSpec>) {
            if (selection_program.group_count <= 0 ||
                selection_program.experts_per_group <= 0 ||
                selection_program.group_count * selection_program.experts_per_group !=
                    expert_count ||
                selection_program.groups_per_token <= 0 ||
                selection_program.groups_per_token > selection_program.group_count ||
                selection_program.group_score_top_k <= 0 ||
                selection_program.group_score_top_k >
                    selection_program.experts_per_group) {
                throw std::invalid_argument("invalid grouped MoE selection program");
            }
        } else {
            static_assert(always_false_v<Selection>,
                          "unhandled MoE selection program");
        }
    }, selection);
}

std::string RouterProgram::fingerprint() const {
    std::ostringstream out;
    append_field(out, score);
    append_field(out, normalization);
    append_field(out, expert_count);
    append_field(out, experts_per_token);
    append_field(out, has_expert_bias);
    append_field(out, routed_scaling);
    std::visit([&](const auto& selection_program) {
        using Selection = std::decay_t<decltype(selection_program)>;
        if constexpr (std::is_same_v<Selection, MoeTopKSelectionSpec>) {
            out << "top-k;";
        } else if constexpr (std::is_same_v<Selection, MoeGroupedTopKSelectionSpec>) {
            out << "grouped-top-k;";
            append_field(out, selection_program.group_count);
            append_field(out, selection_program.experts_per_group);
            append_field(out, selection_program.groups_per_token);
            append_field(out, selection_program.group_score_top_k);
        }
    }, selection);
    return fingerprint_text(out.str());
}

void ExpertMlpProgram::validate() const {
    if (hidden_size <= 0 || intermediate_size <= 0) {
        throw std::invalid_argument("invalid compiled MoE expert MLP program");
    }
}

std::string ExpertMlpProgram::fingerprint() const {
    std::ostringstream out;
    append_field(out, activation);
    append_field(out, hidden_size);
    append_field(out, intermediate_size);
    return fingerprint_text(out.str());
}

void MoeLayerProgram::validate() const {
    router.validate();
    routed.mlp.validate();
    routed.payload.validate();
    if (routed.payload.regions.empty()) {
        throw std::invalid_argument("MoE program has no expert payload regions");
    }
    if (shared) shared->mlp.validate();
}

std::string MoeLayerProgram::fingerprint() const {
    std::ostringstream out;
    out << router.fingerprint() << ';' << routed.mlp.fingerprint() << ';'
        << routed.payload.fingerprint() << ';';
    if (shared) {
        out << shared->mlp.fingerprint() << ';';
        append_field(out, shared->combine_order);
    } else {
        out << "no-shared;";
    }
    return fingerprint_text(out.str());
}

void CompiledDenseFeedForwardProgram::validate() const {
    if (intermediate_size <= 0) {
        throw std::invalid_argument("compiled dense feed-forward has no width");
    }
}

PerLayerInputPlan PerLayerInputPlan::derive(const ResolvedModel& model) {
    PerLayerInputPlan result;
    const ModelGraph& graph = model.graph;
    const auto enabled_layer = std::find_if(
        graph.layers.begin(), graph.layers.end(),
        [](const LayerSpec& layer) { return layer.per_layer_input.enabled; });
    if (enabled_layer == graph.layers.end()) {
        return result;
    }
    if (graph.layers.empty() || graph.hidden <= 0 ||
        enabled_layer->per_layer_input.input_size <= 0) {
        throw std::invalid_argument("invalid per-layer input dimensions");
    }
    result.enabled = true;
    result.layer_count = static_cast<int>(graph.layers.size());
    result.input_size = enabled_layer->per_layer_input.input_size;
    result.packed_width = checked_product(
        static_cast<std::size_t>(result.layer_count),
        static_cast<std::size_t>(result.input_size),
        "per-layer input width overflows size_t");
    if (result.packed_width > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "per-layer input width exceeds the compiled execution index range");
    }
    result.token_scale = std::sqrt(static_cast<float>(result.input_size));
    result.context_scale = 1.0f / std::sqrt(static_cast<float>(graph.hidden));
    result.residual_scale = kPerLayerResidualScale;
    result.activation = graph.layers.front().per_layer_input.activation;
    result.norm_epsilon = graph.layers.front().per_layer_input_norm.epsilon;
    for (const LayerSpec& layer : graph.layers) {
        if (!layer.per_layer_input.enabled ||
            layer.per_layer_input.input_size != result.input_size ||
            layer.per_layer_input.activation != result.activation ||
            layer.per_layer_input_norm.epsilon != result.norm_epsilon) {
            throw std::invalid_argument("inconsistent per-layer input specification");
        }
    }
    result.validate();
    return result;
}

std::size_t PerLayerInputPlan::checked_elements(std::size_t rows) const {
    if (!enabled) return 0;
    return checked_product(rows, packed_width,
                           "per-layer input row product overflows size_t");
}

void PerLayerInputPlan::validate() const {
    if (!enabled) return;
    if (layer_count <= 0 || input_size <= 0 || packed_width == 0 ||
        packed_width != checked_product(static_cast<std::size_t>(layer_count),
                                        static_cast<std::size_t>(input_size),
                                        "per-layer input width overflows size_t") ||
        !std::isfinite(token_scale) || !std::isfinite(context_scale) ||
        !std::isfinite(residual_scale) || !(norm_epsilon > 0.0f) ||
        !std::isfinite(norm_epsilon)) {
        throw std::invalid_argument("invalid compiled per-layer input plan");
    }
}

void CompiledOrdinaryKvStateLayout::validate() const {
    if (key_width <= 0 || value_width <= 0 || persistent_elements() == 0) {
        throw std::invalid_argument("ordinary KV state layout has invalid widths");
    }
    (void)compiled_state_scalar_bytes(storage.key);
    (void)compiled_state_scalar_bytes(storage.value);
    if (persistent_bytes() == 0) {
        throw std::invalid_argument("compiled ordinary KV state has zero byte size");
    }
}

void CompiledLatentStateLayout::validate() const {
    if (latent_width <= 0 || rotary_width < 0 || persistent_elements() == 0) {
        throw std::invalid_argument("latent state layout has invalid widths");
    }
    (void)compiled_state_scalar_bytes(storage.latent);
    (void)compiled_state_scalar_bytes(storage.rotary);
    if (persistent_bytes() == 0) {
        throw std::invalid_argument("compiled latent state has zero byte size");
    }
}

bool CompiledModelProgram::has_moe() const {
    return std::any_of(layers.begin(), layers.end(), [](const auto& layer) {
        return std::holds_alternative<MoeLayerProgram>(layer.feed_forward);
    });
}

void CompiledModelProgram::validate() const {
    if (hidden <= 0 || !std::isfinite(logits_multiplier) ||
        !std::isfinite(logits_divisor) || logits_divisor <= 0.0f ||
        !std::isfinite(final_logit_softcap) || final_logit_softcap < 0.0f) {
        throw std::invalid_argument("compiled model has invalid residual/output policy");
    }
    per_layer_input.validate();
    for (size_t index = 0; index < norm_after_layers.size(); ++index) {
        const int layer = norm_after_layers[index];
        if (layer < 0 || layer >= static_cast<int>(layers.size()) - 1 ||
            (index > 0 && norm_after_layers[index - 1] >= layer)) {
            throw std::invalid_argument("invalid compiled norm boundary");
        }
    }
    for (const auto& layer : layers) {
        if (layer.weight_request_indices.empty()) {
            throw std::invalid_argument("compiled layer has no weight plan");
        }
        for (const std::size_t index : layer.weight_request_indices) {
            if (index >= weight_request_count) {
                throw std::invalid_argument("compiled layer has invalid weight index");
            }
        }
        if (const auto* attention = std::get_if<CompiledAttentionProgram>(&layer.mixer)) {
            validate_attention_state_layout(attention->state_layout);
        }
        validate_feed_forward(layer.feed_forward);
        if (!std::isfinite(layer.residual.multiplier)) {
            throw std::invalid_argument("compiled layer has invalid residual multiplier");
        }
    }
}

} // namespace celeg
