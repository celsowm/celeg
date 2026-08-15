#include "celeg/backend/cpu/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace celeg {
namespace {

std::uint64_t xorshift64star(std::uint64_t& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 2685821657736338717ULL;
}

float random_unit(std::uint64_t& state) {
    const std::uint64_t bits = xorshift64star(state);
    const double unit = (static_cast<double>(bits >> 11) + 0.5) *
                        (1.0 / 9007199254740992.0);
    return static_cast<float>(unit);
}

}

std::int32_t CpuSampler::sample(std::span<const float> logits,
                                int vocab_size,
                                const GenerationConfig& generation,
                                std::span<const std::uint8_t> seen,
                                std::uint64_t& rng_state) {
    if (generation.forced_prefix &&
        generation.forced_prefix->position < generation.forced_prefix->tokens.size()) {
        return generation.forced_prefix->tokens[generation.forced_prefix->position++];
    }
    const float temperature = generation.temperature;
    auto penalized = [&](std::int32_t token) {
        float value = logits[static_cast<size_t>(token)];
        if (seen[static_cast<size_t>(token)] && generation.repetition_penalty > 1.0f) {
            value = value >= 0.0f ? value / generation.repetition_penalty
                                  : value * generation.repetition_penalty;
        }
        return value;
    };
    if (temperature <= 0.0f || generation.top_k == 1) {
        std::int32_t best = 0;
        float best_value = penalized(0);
        for (std::int32_t token = 1; token < vocab_size; ++token) {
            const float value = penalized(token);
            if (value > best_value) {
                best = token;
                best_value = value;
            }
        }
        return best;
    }
    const int top_k = std::min(generation.top_k, vocab_size);
    struct Candidate {
        std::int32_t token;
        float score;
    };
    const auto better = [](const Candidate& a, const Candidate& b) {
        return a.score == b.score ? a.token < b.token : a.score > b.score;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(vocab_size));
    for (std::int32_t token = 0; token < vocab_size; ++token) {
        candidates.push_back({token, penalized(token) / temperature});
    }
    if (static_cast<int>(candidates.size()) > top_k) {
        std::nth_element(candidates.begin(), candidates.begin() + top_k,
                         candidates.end(), better);
        candidates.resize(static_cast<size_t>(top_k));
    }
    std::sort(candidates.begin(), candidates.end(), better);
    const float maximum = candidates.front().score;
    std::vector<float> probabilities(candidates.size());
    float total = 0.0f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probabilities[i] = std::exp(candidates[i].score - maximum);
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
        total = 0.0f;
        for (size_t i = 0; i < active; ++i) total += probabilities[i];
    } else {
        total = 1.0f;
    }
    float target = random_unit(rng_state) * total;
    for (size_t i = 0; i < active; ++i) {
        target -= probabilities[i];
        if (target <= 0.0f) return candidates[i].token;
    }
    return candidates[active - 1].token;
}

}
