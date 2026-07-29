#include "detail/model_internal.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace lfm {
namespace {
uint64_t xorshift64star(uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 2685821657736338717ULL;
}

float random_unit(uint64_t& state) {
    const uint64_t bits = xorshift64star(state);
    const double unit = (static_cast<double>(bits >> 11) + 0.5) *
                        (1.0 / 9007199254740992.0);
    return static_cast<float>(unit);
}
} // namespace

int32_t CpuModel::Impl::sample() {
    const float temperature = generation.temperature;
    auto penalized = [&](int32_t token) {
        float value = logits[static_cast<size_t>(token)];
        if (seen[static_cast<size_t>(token)] &&
            generation.repetition_penalty > 1.0f) {
            value = value >= 0.0f ? value / generation.repetition_penalty
                                  : value * generation.repetition_penalty;
        }
        return value;
    };
    if (temperature <= 0.0f || generation.top_k == 1) {
        int32_t best = 0;
        float best_value = penalized(0);
        for (int32_t token = 1; token < shared->shape.vocab_size; ++token) {
            const float value = penalized(token);
            if (value > best_value) {
                best = token;
                best_value = value;
            }
        }
        return best;
    }
    const int top_k = std::min(generation.top_k, shared->shape.vocab_size);
    std::vector<int32_t> indices(static_cast<size_t>(shared->shape.vocab_size));
    std::iota(indices.begin(), indices.end(), 0);
    auto adjusted = [&](int32_t token) { return penalized(token) / temperature; };
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
        [&](int32_t a, int32_t b) {
            const float av = adjusted(a);
            const float bv = adjusted(b);
            return av == bv ? a < b : av > bv;
        });
    indices.resize(static_cast<size_t>(top_k));
    const float maximum = adjusted(indices.front());
    std::vector<float> probabilities(indices.size());
    float total = 0.0f;
    for (size_t i = 0; i < indices.size(); ++i) {
        probabilities[i] = std::exp(adjusted(indices[i]) - maximum);
        total += probabilities[i];
    }
    for (float& probability : probabilities) probability /= total;
    size_t active = probabilities.size();
    if (generation.top_p < 1.0f) {
        float cumulative = 0.0f;
        active = 0;
        do {
            cumulative += probabilities[active++];
        } while (active < probabilities.size() && cumulative < generation.top_p);
        total = std::accumulate(
            probabilities.begin(),
            probabilities.begin() + static_cast<ptrdiff_t>(active), 0.0f);
    } else {
        total = 1.0f;
    }
    float target = random_unit(rng_state) * total;
    for (size_t i = 0; i < active; ++i) {
        target -= probabilities[i];
        if (target <= 0.0f) return indices[i];
    }
    return indices[active - 1];
}

void CpuModel::Impl::set_generation(GenerationConfig config) {
    config.validate();
    generation = config;
    rng_state = generation.seed;
}

} // namespace lfm
