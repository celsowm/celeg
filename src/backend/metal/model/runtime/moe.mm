#include "detail.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace celeg {

namespace {

struct MetalMoeRoute {
    std::vector<int> experts;
    std::vector<float> weights;
};

float metal_moe_sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

MetalMoeRoute route_metal_moe(const RouterProgram& program,
                              std::span<const float> logits,
                              std::span<const float> expert_bias) {
    if (logits.size() != static_cast<size_t>(program.expert_count)) {
        throw std::invalid_argument("Metal MoE router width does not match program");
    }
    std::vector<float> probabilities(logits.size());
    const float maximum = program.score == MoeRouterScoreKind::SoftmaxLogits
        ? *std::max_element(logits.begin(), logits.end()) : 0.0f;
    float probability_sum = 0.0f;
    for (size_t index = 0; index < logits.size(); ++index) {
        probabilities[index] = program.score == MoeRouterScoreKind::SoftmaxLogits
            ? std::exp(logits[index] - maximum) : metal_moe_sigmoid(logits[index]);
        probability_sum += probabilities[index];
    }
    if (program.score == MoeRouterScoreKind::SoftmaxLogits) {
        for (float& probability : probabilities) probability /= probability_sum;
    }

    std::vector<std::pair<float, int>> scored(logits.size());
    for (size_t index = 0; index < logits.size(); ++index) {
        const float bias = program.has_expert_bias && index < expert_bias.size()
            ? expert_bias[index] : 0.0f;
        scored[index] = {probabilities[index] + bias, static_cast<int>(index)};
    }
    if (const auto* grouped =
            std::get_if<MoeGroupedTopKSelectionSpec>(&program.selection)) {
        std::vector<std::pair<float, int>> groups;
        groups.reserve(static_cast<size_t>(grouped->group_count));
        for (int group = 0; group < grouped->group_count; ++group) {
            std::vector<float> group_scores;
            group_scores.reserve(static_cast<size_t>(grouped->experts_per_group));
            const int first = group * grouped->experts_per_group;
            for (int offset = 0; offset < grouped->experts_per_group; ++offset) {
                group_scores.push_back(probabilities[static_cast<size_t>(first + offset)]);
            }
            const int score_count = std::min(grouped->group_score_top_k,
                                             grouped->experts_per_group);
            std::partial_sort(group_scores.begin(), group_scores.begin() + score_count,
                              group_scores.end(), std::greater<float>());
            float score = 0.0f;
            for (int index = 0; index < score_count; ++index) score += group_scores[index];
            groups.emplace_back(score, group);
        }
        std::partial_sort(groups.begin(),
                          groups.begin() + grouped->groups_per_token,
                          groups.end(), [](const auto& left, const auto& right) {
            return left.first == right.first ? left.second < right.second
                                             : left.first > right.first;
        });
        std::vector<bool> selected(static_cast<size_t>(grouped->group_count), false);
        for (int index = 0; index < grouped->groups_per_token; ++index) {
            selected[static_cast<size_t>(groups[static_cast<size_t>(index)].second)] = true;
        }
        for (auto& entry : scored) {
            if (!selected[static_cast<size_t>(
                    entry.second / grouped->experts_per_group)]) {
                entry.first = -std::numeric_limits<float>::infinity();
            }
        }
    }
    std::partial_sort(scored.begin(), scored.begin() + program.experts_per_token,
                      scored.end(), [](const auto& left, const auto& right) {
        return left.first == right.first ? left.second < right.second
                                         : left.first > right.first;
    });

    MetalMoeRoute result;
    result.experts.resize(static_cast<size_t>(program.experts_per_token));
    result.weights.resize(result.experts.size());
    float selected_sum = 0.0f;
    for (int route = 0; route < program.experts_per_token; ++route) {
        const int expert = scored[static_cast<size_t>(route)].second;
        result.experts[static_cast<size_t>(route)] = expert;
        result.weights[static_cast<size_t>(route)] = probabilities[static_cast<size_t>(expert)];
        selected_sum += result.weights[static_cast<size_t>(route)];
    }
    if (program.normalization == MoeNormalizationKind::SumSelected) {
        const float inverse = 1.0f / (selected_sum + 1.0e-6f);
        for (float& weight : result.weights) weight *= inverse;
    }
    for (float& weight : result.weights) weight *= program.routed_scaling;
    return result;

}

}

void MetalModel::Impl::encode_moe(
    id<MTLCommandBuffer>& command_buffer,
    id<MTLComputeCommandEncoder>& encoder, Layer& layer) {
    const uint32_t hidden_width = static_cast<uint32_t>(model.graph.hidden);
    finish_commands(command_buffer, encoder);
    std::vector<float> router_logits(
        static_cast<size_t>(layer.moe->router.expert_count));
    for (int expert = 0; expert < layer.moe->router.expert_count; ++expert) {
        float sum = 0.0f;
        const size_t base = static_cast<size_t>(expert) * hidden_width;
        for (uint32_t index = 0; index < hidden_width; ++index) {
            sum += layer.moe->router_weight[base + index] *
                   static_cast<const float*>(normed.contents)[index];
        }
        router_logits[static_cast<size_t>(expert)] = sum;
    }
    const MetalMoeRoute route = route_metal_moe(
        layer.moe->router, router_logits, layer.moe->router_bias);
    std::memset(moe_output.contents, 0,
                static_cast<size_t>(hidden_width) * sizeof(float));
    begin_commands(command_buffer, encoder);
    const uint32_t intermediate = static_cast<uint32_t>(layer.intermediate);
    for (size_t route_index = 0; route_index < route.experts.size(); ++route_index) {
        const int expert = route.experts[route_index];
        const Layer::Expert& names =
            layer.moe->experts[static_cast<size_t>(expert)];
        const Linear gate = load_linear_source(
            names.gate_name, layer.intermediate, hidden_width);
        const Linear up = load_linear_source(
            names.up_name, layer.intermediate, hidden_width);
        const Linear down = load_linear_source(
            names.down_name, hidden_width, layer.intermediate);
        encode_matvec(encoder, gate, normed, gate_up, 0);
        encode_matvec(encoder, up, normed, gate_up,
                      static_cast<NSUInteger>(layer.intermediate) * sizeof(float));
        set_buffer(encoder, gate_up, 0);
        set_buffer(encoder, activated, 1);
        set_bytes(encoder, &intermediate, sizeof(intermediate), 2);
        dispatch(encoder, "celeg_swiglu", intermediate);
        encode_matvec(encoder, down, activated, operation);
        encode_weighted_add(encoder, operation, moe_output, hidden_width,
                            route.weights[route_index]);
    }
}

}
