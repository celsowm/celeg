#include "moe_router.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace celeg {

namespace {

float moe_sigmoid(float v) noexcept {
    return 1.0f / (1.0f + std::exp(-v));
}

}

MetalMoeRoute route_metal_moe(
    const RouterProgram& program,
    std::span<const float> logits,
    std::span<const float> expert_bias) {
    if (logits.size() != static_cast<size_t>(program.expert_count)) {
        throw std::invalid_argument("Metal MoE router width does not match program");
    }
    std::vector<float> probs(logits.size());
    const float maxv = program.score == MoeRouterScoreKind::SoftmaxLogits
        ? *std::max_element(logits.begin(), logits.end()) : 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = program.score == MoeRouterScoreKind::SoftmaxLogits
            ? std::exp(logits[i] - maxv) : moe_sigmoid(logits[i]);
        sum += probs[i];
    }
    if (program.score == MoeRouterScoreKind::SoftmaxLogits) {
        for (float& p : probs) p /= sum;
    }
    std::vector<std::pair<float, int>> scored(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        const float bias = program.has_expert_bias && i < expert_bias.size() ? expert_bias[i] : 0.0f;
        scored[i] = {probs[i] + bias, static_cast<int>(i)};
    }
    if (const auto* grouped = std::get_if<MoeGroupedTopKSelectionSpec>(&program.selection)) {
        std::vector<std::pair<float, int>> groups;
        groups.reserve(static_cast<size_t>(grouped->group_count));
        for (int g = 0; g < grouped->group_count; ++g) {
            std::vector<float> gs;
            gs.reserve(static_cast<size_t>(grouped->experts_per_group));
            const int first = g * grouped->experts_per_group;
            for (int o = 0; o < grouped->experts_per_group; ++o) gs.push_back(probs[static_cast<size_t>(first + o)]);
            const int k = std::min(grouped->group_score_top_k, grouped->experts_per_group);
            std::partial_sort(gs.begin(), gs.begin() + k, gs.end(), std::greater<float>());
            float s = 0.0f;
            for (int i = 0; i < k; ++i) s += gs[static_cast<size_t>(i)];
            groups.emplace_back(s, g);
        }
        std::partial_sort(
            groups.begin(), groups.begin() + grouped->groups_per_token, groups.end(),
            [](const auto& l, const auto& r) { return l.first == r.first ? l.second < r.second : l.first > r.first; });
        std::vector<bool> sel(static_cast<size_t>(grouped->group_count), false);
        for (int i = 0; i < grouped->groups_per_token; ++i) sel[static_cast<size_t>(groups[static_cast<size_t>(i)].second)] = true;
        for (auto& e : scored) {
            if (!sel[static_cast<size_t>(e.second / grouped->experts_per_group)]) e.first = -std::numeric_limits<float>::infinity();
        }
    }
    std::partial_sort(
        scored.begin(), scored.begin() + program.experts_per_token, scored.end(),
        [](const auto& l, const auto& r) { return l.first == r.first ? l.second < r.second : l.first > r.first; });
    MetalMoeRoute out;
    out.experts.resize(static_cast<size_t>(program.experts_per_token));
    out.weights.resize(out.experts.size());
    float sel_sum = 0.0f;
    for (int r = 0; r < program.experts_per_token; ++r) {
        const int e = scored[static_cast<size_t>(r)].second;
        out.experts[static_cast<size_t>(r)] = e;
        out.weights[static_cast<size_t>(r)] = probs[static_cast<size_t>(e)];
        sel_sum += out.weights[static_cast<size_t>(r)];
    }
    if (program.normalization == MoeNormalizationKind::SumSelected) {
        const float inv = 1.0f / (sel_sum + 1.0e-6f);
        for (float& w : out.weights) w *= inv;
    }
    for (float& w : out.weights) w *= program.routed_scaling;
    return out;
}

}
