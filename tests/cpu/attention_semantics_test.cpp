#include "operators/attention.hpp"
#include "kernels/math.hpp"
#include "celeg/backend/cpu/rope.hpp"
#include "celeg/backend/cpu/paged_kv.hpp"
#include "support/assertions.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

bool close(float actual, float expected, float tolerance = 1.0e-5f) {
    return std::abs(actual - expected) <= tolerance;
}

const celeg::CpuMathEngine& scalar_math() {
    return celeg::cpu_math_engine(celeg::CpuIsa::Scalar);
}

void test_current_value_orthogonalization() {
    celeg::AttentionSpec layout;
    layout.query_heads = 4;
    layout.key_value_heads = 2;
    layout.head_dim = 2;
    layout.output_transform = celeg::OrthogonalizeCurrentValueSpec{1.0e-6f};

    float output[] = {
        1.0f, 1.0f,
        2.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 2.0f,
    };
    const float current_value[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };

    celeg::apply_cpu_attention_output_transform(layout, output, current_value);

    CELEG_TEST_CHECK(close(output[0], 0.0f));
    CELEG_TEST_CHECK(close(output[1], 1.0f));
    CELEG_TEST_CHECK(close(output[2], 0.0f));
    CELEG_TEST_CHECK(close(output[3], 0.0f));
    CELEG_TEST_CHECK(close(output[4], 1.0f));
    CELEG_TEST_CHECK(close(output[5], 0.0f));
    CELEG_TEST_CHECK(close(output[6], 0.0f));
    CELEG_TEST_CHECK(close(output[7], 0.0f));

    for (int query_head = 0; query_head < layout.query_heads; ++query_head) {
        const int value_head = query_head / (layout.query_heads / layout.key_value_heads);
        const float* value = current_value + value_head * layout.head_dim;
        const float* transformed = output + query_head * layout.head_dim;
        const float dot = transformed[0] * value[0] + transformed[1] * value[1];
        CELEG_TEST_CHECK(close(dot, 0.0f));
    }
}

void test_adjacent_pair_rope() {
    celeg::RopePositionSpec rope;
    rope.theta = 10000.0;
    rope.rotary_fraction = 1.0;
    rope.pairing = celeg::RopePairingKind::AdjacentPairs;

    float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    celeg::cpu_rope(values, 1, 4, 1, rope);

    const float c0 = std::cos(1.0f);
    const float s0 = std::sin(1.0f);
    const float second_frequency = std::pow(10000.0f, -0.5f);
    const float c1 = std::cos(second_frequency);
    const float s1 = std::sin(second_frequency);

    CELEG_TEST_CHECK(close(values[0], 1.0f * c0 - 2.0f * s0));
    CELEG_TEST_CHECK(close(values[1], 2.0f * c0 + 1.0f * s0));
    CELEG_TEST_CHECK(close(values[2], 3.0f * c1 - 4.0f * s1));
    CELEG_TEST_CHECK(close(values[3], 4.0f * c1 + 3.0f * s1));
}

/// HF-style partial-rotary reference: the whole head is RMS-normalized,
/// then the first `rotary_pairs` pairs rotate. Proportional RoPE pads its
/// frequency table to the full head width, so pairs are full-dimension
/// pairs (`i` mixes `(i, i + head_dim / 2)`); every other scaling builds
/// prefix-width tables, so pairs stay prefix-local.
std::vector<float> reference_partial_qk_norm_rope(
    const std::vector<float>& input, const std::vector<float>& norm_weight,
    int heads, int head_dim, int rotary_dim, bool full_dim_pairs,
    double theta, double fraction, bool proportional, int position,
    float eps) {
    const int rotated_pairs = rotary_dim / 2;
    const int pair_count = full_dim_pairs ? head_dim / 2 : rotated_pairs;
    std::vector<float> output = input;
    for (int head = 0; head < heads; ++head) {
        float* vector = output.data() + static_cast<size_t>(head) * head_dim;
        double sum = 0.0;
        for (int d = 0; d < head_dim; ++d) {
            sum += static_cast<double>(vector[d]) * vector[d];
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(sum / head_dim) + eps);
        for (int d = 0; d < head_dim; ++d) {
            vector[d] *= inv * norm_weight[static_cast<size_t>(d)];
        }
        for (int pair = 0; pair < rotated_pairs; ++pair) {
            int first = pair;
            int second = pair_count + pair;
            double base_frequency = std::pow(
                theta, -2.0 * static_cast<double>(pair) / static_cast<double>(rotary_dim));
            if (proportional) {
                base_frequency = std::pow(base_frequency, fraction);
            }
            const float angle = static_cast<float>(position) * static_cast<float>(base_frequency);
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            const float a = vector[first];
            const float b = vector[second];
            vector[first] = a * c - b * s;
            vector[second] = b * c + a * s;
        }
    }
    return output;
}

void test_proportional_partial_rope_uses_full_dim_pairs() {
    /// Gemma-4 full-attention geometry, scaled down: head_dim 8, rotary
    /// fraction 0.25 (one rotated pair), theta 1e6. The rotated pair must be
    /// (0, 4) with a fully normalized tail; the old code rotated (0, 1) and
    /// left dims 2..7 as raw GEMM output.
    constexpr int kHeadDim = 8;
    constexpr int kHeads = 2;
    constexpr int kPosition = 7;
    constexpr float kEps = 1.0e-6f;

    celeg::RopePositionSpec rope;
    rope.theta = 1000000.0;
    rope.rotary_fraction = 0.25;
    rope.scaling = celeg::ProportionalRopeScaling{1.0};
    rope.pairing = celeg::RopePairingKind::SplitHalf;

    std::vector<float> norm_weight(kHeadDim);
    for (int d = 0; d < kHeadDim; ++d) {
        norm_weight[d] = 0.5f + 0.125f * static_cast<float>(d);
    }
    std::vector<float> input(static_cast<size_t>(kHeads) * kHeadDim);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) * (i % kHeadDim == 6 ? 50.0f : 1.0f);
    }
    const std::vector<float> expected = reference_partial_qk_norm_rope(
        input, norm_weight, kHeads, kHeadDim, 2, true,
        rope.theta, rope.rotary_fraction, true, kPosition, kEps);

    std::vector<float> fused = input;
    scalar_math().qk_norm_rope(fused.data(), norm_weight.data(), kHeads,
                               kHeadDim, kPosition, rope, kEps);
    CELEG_TEST_CHECK(fused.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        CELEG_TEST_CHECK(close(fused[i], expected[i], 1.0e-4f));
    }

    std::vector<float> rope_only = input;
    celeg::cpu_rope(rope_only.data(), kHeads, kHeadDim, kPosition, rope);
    const float angle = static_cast<float>(kPosition);
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    for (int head = 0; head < kHeads; ++head) {
        const float* row_in = input.data() + static_cast<size_t>(head) * kHeadDim;
        const float* row_out = rope_only.data() + static_cast<size_t>(head) * kHeadDim;
        CELEG_TEST_CHECK(close(row_out[0], row_in[0] * c - row_in[4] * s, 1.0e-5f));
        CELEG_TEST_CHECK(close(row_out[4], row_in[4] * c + row_in[0] * s, 1.0e-5f));
        for (int d = 1; d < kHeadDim; ++d) {
            if (d == 4) continue;
            CELEG_TEST_CHECK(close(row_out[d], row_in[d], 1.0e-5f));
        }
    }
}

void test_default_partial_rope_keeps_prefix_pairs_and_norms_tail() {
    /// Ling-style legacy convention: head_dim 8, rotary fraction 0.5, no
    /// scaling. Pairs stay prefix-local ((0, 2), (1, 3)) but the whole head
    /// -- including the unrotated tail -- is normalized; the old code left
    /// dims 4..7 raw.
    constexpr int kHeadDim = 8;
    constexpr int kHeads = 2;
    constexpr int kPosition = 3;
    constexpr float kEps = 1.0e-6f;

    celeg::RopePositionSpec rope;
    rope.theta = 10000.0;
    rope.rotary_fraction = 0.5;
    rope.scaling = celeg::NoRopeScaling{};
    rope.pairing = celeg::RopePairingKind::SplitHalf;

    std::vector<float> norm_weight(kHeadDim);
    for (int d = 0; d < kHeadDim; ++d) {
        norm_weight[d] = 1.5f - 0.125f * static_cast<float>(d);
    }
    std::vector<float> input(static_cast<size_t>(kHeads) * kHeadDim);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + 1) * (i % kHeadDim == 7 ? 40.0f : 1.0f);
    }
    const std::vector<float> expected = reference_partial_qk_norm_rope(
        input, norm_weight, kHeads, kHeadDim, 4, false,
        rope.theta, rope.rotary_fraction, false, kPosition, kEps);

    std::vector<float> fused = input;
    scalar_math().qk_norm_rope(fused.data(), norm_weight.data(), kHeads,
                               kHeadDim, kPosition, rope, kEps);
    CELEG_TEST_CHECK(fused.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        CELEG_TEST_CHECK(close(fused[i], expected[i], 1.0e-4f));
    }
}

void test_bidirectional_pattern_reads_future_keys() {
    celeg::CpuAttentionPattern pattern;
    pattern.storage = celeg::BidirectionalPattern{};

    CELEG_TEST_CHECK(pattern.allows(0, 0));
    CELEG_TEST_CHECK(pattern.allows(0, 1));
    CELEG_TEST_CHECK(pattern.allows(1, 0));
    CELEG_TEST_CHECK(pattern.allows(2, 7));
    CELEG_TEST_CHECK(pattern.first_candidate(3) == 0);
    CELEG_TEST_CHECK(pattern.may_read_future(0, 8));
    CELEG_TEST_CHECK(pattern.may_read_future(7, 8));
}

void test_prefix_lm_pattern_boundaries() {
    celeg::CpuAttentionPattern pattern;
    pattern.storage = celeg::PrefixLmPattern{4};

    CELEG_TEST_CHECK(pattern.allows(0, 0));
    CELEG_TEST_CHECK(pattern.allows(0, 3));
    CELEG_TEST_CHECK(!pattern.allows(0, 4));
    CELEG_TEST_CHECK(pattern.allows(2, 3));
    CELEG_TEST_CHECK(!pattern.allows(2, 5));
    CELEG_TEST_CHECK(pattern.allows(4, 0));
    CELEG_TEST_CHECK(pattern.allows(4, 4));
    CELEG_TEST_CHECK(!pattern.allows(4, 5));
    CELEG_TEST_CHECK(pattern.allows(7, 6));
    CELEG_TEST_CHECK(!pattern.allows(7, 8));
    CELEG_TEST_CHECK(pattern.first_candidate(0) == 0);
    CELEG_TEST_CHECK(pattern.first_candidate(7) == 0);
    CELEG_TEST_CHECK(pattern.may_read_future(0, 8));
    CELEG_TEST_CHECK(pattern.may_read_future(3, 8));
    CELEG_TEST_CHECK(!pattern.may_read_future(4, 8));

    celeg::CpuAttentionPattern full_prefix;
    full_prefix.storage = celeg::PrefixLmPattern{8};
    CELEG_TEST_CHECK(!full_prefix.may_read_future(0, 8));
    CELEG_TEST_CHECK(full_prefix.allows(0, 7));
}

void test_query_key_norm_uses_same_query_scale() {
    constexpr int kHeadDim = 64;
    constexpr int kHeads = 2;
    constexpr int kWidth = kHeads * kHeadDim;

    celeg::RopePositionSpec rope;
    rope.theta = 10000.0;
    rope.rotary_fraction = 1.0;
    rope.pairing = celeg::RopePairingKind::SplitHalf;

    celeg::AttentionSpec plain;
    plain.query_heads = kHeads;
    plain.key_value_heads = kHeads;
    plain.head_dim = kHeadDim;
    plain.query_scale = 0.125f;
    plain.position = rope;

    celeg::AttentionSpec normed = plain;
    celeg::NormSpec norm;
    norm.epsilon = 1.0e-6f;
    norm.granularity = celeg::NormGranularity::PerHead;
    normed.query_norm = norm;

    celeg::CpuCompiledModel::AttentionWeights weights;
    weights.q_norm.assign(kHeadDim, 1.0f);

    std::vector<float> plain_query(kWidth);
    for (int i = 0; i < kWidth; ++i) plain_query[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    std::vector<float> normed_query = plain_query;

    const std::array<int32_t, 3> position{0, 0, 0};
    celeg::apply_cpu_attention_qk(plain, weights, scalar_math(),
                                  plain_query.data(), nullptr, nullptr, 0, position);
    celeg::apply_cpu_attention_qk(normed, weights, scalar_math(),
                                  normed_query.data(), nullptr, nullptr, 0, position);

    for (int i = 0; i < kWidth; ++i) {
        CELEG_TEST_CHECK(close(plain_query[i], (i % 2 == 0) ? 1.0f : -1.0f));
        CELEG_TEST_CHECK(close(normed_query[i], plain_query[i]));
    }
}

void test_query_only_norm_with_key_projection() {
    celeg::AttentionSpec layout;
    layout.query_heads = 1;
    layout.key_value_heads = 1;
    layout.head_dim = 4;
    layout.query_scale = 0.5f;
    layout.position = celeg::NoPositionEncodingSpec{};
    celeg::NormSpec norm;
    norm.epsilon = 1.0e-6f;
    norm.granularity = celeg::NormGranularity::PerHead;
    layout.query_norm = norm;

    celeg::CpuCompiledModel::AttentionWeights weights;
    weights.q_norm.assign(4, 1.0f);
    weights.k.segments.emplace_back(celeg::CpuInt8Matrix{});

    float query[] = {2.0f, 2.0f, 2.0f, 2.0f};
    float key[] = {3.0f, 3.0f, 3.0f, 3.0f};
    const std::array<int32_t, 3> position{0, 0, 0};
    celeg::apply_cpu_attention_qk(layout, weights, scalar_math(),
                                  query, key, nullptr, 0, position);

    for (float value : query) CELEG_TEST_CHECK(close(value, 1.0f));
    for (float value : key) CELEG_TEST_CHECK(close(value, 3.0f));
}

void test_key_only_norm_without_query_norm() {
    celeg::AttentionSpec layout;
    layout.query_heads = 1;
    layout.key_value_heads = 1;
    layout.head_dim = 4;
    layout.query_scale = 0.5f;
    layout.position = celeg::NoPositionEncodingSpec{};
    celeg::NormSpec norm;
    norm.epsilon = 1.0e-6f;
    norm.granularity = celeg::NormGranularity::PerHead;
    layout.key_norm = norm;

    celeg::CpuCompiledModel::AttentionWeights weights;
    weights.k_norm.assign(4, 1.0f);
    weights.k.segments.emplace_back(celeg::CpuInt8Matrix{});

    float query[] = {3.0f, 3.0f, 3.0f, 3.0f};
    float key[] = {2.0f, 2.0f, 2.0f, 2.0f};
    const std::array<int32_t, 3> position{0, 0, 0};
    celeg::apply_cpu_attention_qk(layout, weights, scalar_math(),
                                  query, key, nullptr, 0, position);

    for (float value : query) CELEG_TEST_CHECK(close(value, 3.0f));
    for (float value : key) CELEG_TEST_CHECK(close(value, 1.0f));
}

}

int main() {
    test_current_value_orthogonalization();
    test_adjacent_pair_rope();
    test_proportional_partial_rope_uses_full_dim_pairs();
    test_default_partial_rope_keeps_prefix_pairs_and_norms_tail();
    test_bidirectional_pattern_reads_future_keys();
    test_prefix_lm_pattern_boundaries();
    test_query_key_norm_uses_same_query_scale();
    test_query_only_norm_with_key_projection();
    test_key_only_norm_without_query_norm();
    return 0;
}
